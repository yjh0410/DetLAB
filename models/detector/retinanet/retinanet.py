import enum
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn as nn
from ...backbone import build_backbone
from ...neck.fpn import build_fpn
from ...head.decoupled_head import DecoupledHead
from .loss import Criterion

DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)



class RetinaNet(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(RetinaNet, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.anchor_size = self.generate_anchor_sizes(cfg)  # [S, KA, 2]
        self.num_anchors = self.anchor_size.shape[1]

        # backbone
        self.backbone, bk_dim = build_backbone(model_name=cfg['backbone'], 
                                               pretrained=trainable,
                                               norm_type=cfg['norm_type'])

        # neck
        self.fpn = build_fpn(cfg=cfg, 
                             in_dims=bk_dim, 
                             out_dim=cfg['head_dim'],
                             from_c5=cfg['from_c5'],
                             p6_feat=cfg['p6_feat'],
                             p7_feat=cfg['p7_feat'])
                                     
        # head
        self.head = DecoupledHead(head_dim=cfg['head_dim'],
                                  num_cls_head=cfg['num_cls_head'],
                                  num_reg_head=cfg['num_reg_head'],
                                  act_type=cfg['act_type'],
                                  norm_type=cfg['head_norm'])

        # pred
        self.cls_pred = nn.Conv2d(cfg['head_dim'], 
                                  self.num_anchors * self.num_classes, 
                                  kernel_size=3,
                                  padding=1)
        self.reg_pred = nn.Conv2d(cfg['head_dim'], 
                                  self.num_anchors * 4, 
                                  kernel_size=3, 
                                  padding=1)

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


    def generate_anchor_sizes(self, cfg):
        basic_anchor_size = cfg['anchor_config']['basic_size']
        anchor_aspect_ratio = cfg['anchor_config']['aspect_ratio']
        anchor_area_scale = cfg['anchor_config']['area_scale']

        num_scales = len(basic_anchor_size)
        num_anchors = len(anchor_aspect_ratio) * len(anchor_area_scale)
        anchor_sizes = []
        for size in basic_anchor_size:
            for ar in anchor_aspect_ratio:
                for s in anchor_area_scale:
                    ah, aw = size
                    area = ah * aw * s
                    anchor_sizes.append([math.sqrt(ar * area), math.sqrt(area / ar)])
        # [S * KA, 2] -> [S, KA, 2]
        anchor_sizes = torch.as_tensor(anchor_sizes).view(num_scales, num_anchors, 2)

        return anchor_sizes


    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        # [KA, 2]
        anchor_size = self.anchor_size[level]

        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
        anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
        anchor_xy *= self.stride[level]

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

        # [HW, KA, 4] -> [M, 4], M = HW x KA
        anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

        return anchor_boxes
        

    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[Tensor]) [1, M, 4] or [M, 4]
            pred_reg:     (List[Tensor]) [B, M, 4] or [M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
        if self.cfg['ctr_clamp'] is not None:
            pred_ctr_offset = torch.clamp(pred_ctr_offset,
                                        max=self.cfg['ctr_clamp'],
                                        min=-self.cfg['ctr_clamp'])
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset

        # w = w_anchor * exp(tw)
        # h = h_anchor * exp(th)
        pred_dwdh = pred_reg[..., 2:]
        pred_dwdh = torch.clamp(pred_dwdh, 
                                max=DEFAULT_SCALE_CLAMP)
        pred_wh = anchor_boxes[..., 2:] * pred_dwdh.exp()

        # convert [x, y, w, h] -> [x1, y1, x2, y2]
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

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

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
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
        for level, feat in enumerate(pyramid_feats):
            cls_feat, reg_feat = self.head(feat)

            # [1, KAxC, H, W]
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)

            # decode box
            _, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [1, KAxC, H, W] -> [H, W, KAxC] -> [H, W, KA, C] -> [M, C]
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(H, W, self.num_anchors, -1).view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(H, W, self.num_anchors, -1).view(-1, 4)

            # scores
            scores, labels = torch.max(cls_pred.sigmoid(), dim=-1)

            # topk
            anchor_boxes = self.generate_anchors(level, fmp_size) # [M, 4]
            if scores.shape[0] > self.topk:
                scores, indices = torch.topk(scores, self.topk)
                labels = labels[indices]
                reg_pred = reg_pred[indices]
                anchor_boxes = anchor_boxes[indices]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchor_boxes, reg_pred)


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
            pyramid_feats = self.fpn(pyramid_feats) # [P3, P4, P5, P6, P7]

            # shared head
            all_anchor_boxes = []
            all_cls_preds = []
            all_reg_preds = []
            all_masks = []
            for level, feat in enumerate(pyramid_feats):
                cls_feat, reg_feat = self.head(feat)
                # [B, KAxC, H, W]
                cls_pred = self.cls_pred(cls_feat)
                reg_pred = self.reg_pred(reg_feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # [B, KAxC, H, W] -> [B, H, W, KAxC] -> [B, H, W, KA, C] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, self.num_anchors, -1).view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, H, W, self.num_anchors, -1).view(B, -1, 4)

                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)

                if mask is not None:
                    # [B, H, W]
                    mask_i = torch.nn.functional.interpolate(mask[None], size=[H, W]).bool()[0]
                    # [B, H, W] -> [B, HW]
                    mask_i = mask_i.flatten(1)
                    # [B, HW] -> [B, HW, KA] -> [B, M], M= HW x KA
                    mask_i = mask_i[..., None].repeat(1, 1, self.num_anchors).flatten(1)
                    
                    all_masks.append(mask_i)

                # generate anchor boxes: [M, 4]
                anchor_boxes = self.generate_anchors(level, fmp_size)
                all_anchor_boxes.append(anchor_boxes)
            
            all_cls_preds = torch.cat(all_cls_preds, dim=1)
            all_reg_preds = torch.cat(all_reg_preds, dim=1)
            all_masks = torch.cat(all_masks, dim=1)

            # decode box: [M, 4]
            all_anchor_boxes = torch.cat(all_anchor_boxes)
            all_box_preds = self.decode_boxes(all_anchor_boxes[None], all_reg_preds)

            outputs = {"pred_cls": all_cls_preds,
                       "pred_box": all_box_preds,
                       'strides': self.stride,
                       "mask": all_masks}

            # loss
            loss_dict = self.criterion(outputs = outputs, 
                                       targets = targets, 
                                       anchor_boxes = all_anchor_boxes)

            return loss_dict 
    