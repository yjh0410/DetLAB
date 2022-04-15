from matplotlib.pyplot import sca
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ...backbone import build_backbone
from ...neck import build_fpn
from ...head.decoupled_head import DecoupledHead
from .loss import Criterion


DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


class YOLO(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(YOLO, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk

        # backbone
        self.backbone, bk_dim = build_backbone(model_name=cfg['backbone'], 
                                               pretrained=trainable,
                                               norm_type=cfg['norm_type'])

        # neck
        self.fpn = build_fpn(cfg=cfg, 
                             in_dims=bk_dim, 
                             out_dim=cfg['head_dim'],
                             from_c5=False,
                             p6_feat=False,
                             p7_feat=False)
                                     
        # head
        self.head = DecoupledHead(head_dim=cfg['head_dim'],
                                  num_cls_head=cfg['num_cls_head'],
                                  num_reg_head=cfg['num_reg_head'],
                                  act_type=cfg['act_type'],
                                  norm_type=cfg['head_norm'])

        # pred
        self.cls_pred = nn.Conv2d(cfg['head_dim'], 
                                  self.num_classes, 
                                  kernel_size=3, 
                                  padding=1)
        self.reg_pred = nn.Conv2d(cfg['head_dim'], 
                                  4, 
                                  kernel_size=3, 
                                  padding=1)

        # scale
        self.scales = nn.ModuleList([Scale() for _ in range(len(self.stride))])

        if trainable:
            # init bias
            self._init_pred_layers()

        # criterion
        if self.trainable:
            self.criterion = Criterion(cfg=cfg,
                                       device=device,
                                       alpha=cfg['alpha'],
                                       gamma=cfg['gamma'],
                                       loss_cls_weight=cfg['loss_cls_weight'],
                                       loss_reg_weight=cfg['loss_reg_weight'],
                                       num_classes=num_classes)


    def _init_pred_layers(self):  
        # init cls pred
        nn.init.normal_(self.cls_pred.weight, mean=0, std=0.01)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
        # init reg pred
        nn.init.normal_(self.reg_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)


    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors
        

    def decode_boxes(self, level, anchors, pred_ctr_offset, pred_size):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [M, 4] (l, t, r, b)
        """
        ctr_offset = pred_ctr_offset.sigmoid() * 3.0 - 1.5
        pred_ctr = anchors + ctr_offset

        pred_box_wh = pred_size.exp()
        pred_x1y1 = pred_ctr - pred_box_wh * 0.5
        pred_x2y2 = pred_ctr + pred_box_wh * 0.5

        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1) * self.stride[level]

        return pred_box


    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    @torch.no_grad()
    def inference_single_image(self, x):
        img_h, img_w = x.shape[2:]
        # backbone
        feats = self.backbone(x)
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]

        # neck
        pyramid_feats = self.fpn(pyramid_feats)

        # shared head
        all_scores = []
        all_labels = []
        all_bboxes = []
        for level, (feat, scale) in enumerate(zip(pyramid_feats, self.scales)):
            cls_feat, reg_feat = self.head(feat)

            # [1, C, H, W]
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)

            # decode box
            _, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [1, C, H, W] -> [H, W, C] -> [M, C]
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)
            pred_ctr_offset = reg_pred[..., :2]
            pred_size = scale(reg_pred[..., 2:])

            # scores
            scores, labels = torch.max(cls_pred.sigmoid(), dim=-1)

            # [M, 4]
            anchors = self.generate_anchors(level, fmp_size)
            # topk
            if scores.shape[0] > self.topk:
                scores, indices = torch.topk(scores, self.topk)
                labels = labels[indices]
                pred_ctr_offset = pred_ctr_offset[indices]
                pred_size = pred_size[indices]
                anchors = anchors[indices]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(level, anchors, pred_ctr_offset, pred_size)

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # normalize bbox
        bboxes[..., [0, 2]] /= img_w
        bboxes[..., [1, 3]] /= img_h
        bboxes = bboxes.clip(0., 1.)

        return bboxes, scores, labels


    def forward(self, x, mask=None, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            feats = self.backbone(x)
            pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]

            # neck
            pyramid_feats = self.fpn(pyramid_feats) # [P3, P4, P5]

            # shared head
            all_anchors = []
            all_cls_preds = []
            all_box_preds = []
            all_masks = []
            for level, (feat, scale) in enumerate(zip(pyramid_feats, self.scales)):
                cls_feat, reg_feat = self.head(feat)
                # [B, C, H, W]
                cls_pred = self.cls_pred(cls_feat)
                reg_pred = self.reg_pred(reg_feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                pred_ctr_offset = reg_pred[..., :2]
                pred_size = scale(reg_pred[..., 2:])
                box_pred = self.decode_boxes(level, anchors, pred_ctr_offset, pred_size)

                all_anchors.append(anchors)
            
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)

                if mask is not None:
                    # [B, H, W]
                    mask_i = torch.nn.functional.interpolate(mask[None], size=[H, W]).bool()[0]
                    # [B, H, W] -> [B, M]
                    mask_i = mask_i.flatten(1)
                    
                    all_masks.append(mask_i)

            # output dict
            outputs = {"pred_cls": all_cls_preds,  # List [B, M, C]
                       "pred_box": all_box_preds,  # List [B, M, 4]
                       'strides': self.stride,
                       "mask": all_masks}          # List [B, M,]

            # loss
            loss_dict = self.criterion(outputs = outputs, 
                                       targets = targets, 
                                       anchors = all_anchors)

            return loss_dict 
    