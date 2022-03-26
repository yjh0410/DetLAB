import torch
import torch.nn as nn
from .matcher import UniformMatcher
from utils.box_ops import *
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(nn.Module):
    def __init__(self, 
                 cfg, 
                 device, 
                 alpha=0.25,
                 gamma=2.0,
                 loss_cls_weight=1.0, 
                 loss_reg_weight=1.0, 
                 num_classes=80):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        if cfg['matcher'] == 'uniform_matcher':
            self.matcher = UniformMatcher(match_times=cfg['topk'])


    def loss_labels(self, pred_cls, tgt_cls, num_boxes):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_focal_loss(pred_cls, tgt_cls, self.alpha, self.gamma, reduction='none')

        return loss_cls.sum() / num_boxes


    def loss_bboxes(self, pred_box, tgt_box, num_boxes):
        """
            pred_box: (Tensor) [N, 4]
            tgt_box:  (Tensor) [N, 4]
        """
        # giou
        pred_giou = generalized_box_iou(pred_box, tgt_box)  # [N, M]
        # giou loss
        loss_reg = 1. - torch.diag(pred_giou)

        return loss_reg.sum() / num_boxes


    def forward(self,
                outputs, 
                targets, 
                anchor_boxes=None):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_box']: (Tensor) [B, M, 4]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
            anchor_boxes: (Tensor) [M, 4]
        """
        pred_box = outputs['pred_box']
        pred_cls = outputs['pred_cls'].reshape(-1, self.num_classes)
        pred_box_copy = pred_box.detach().clone().cpu()
        anchor_boxes_copy = anchor_boxes.clone().cpu()
        # rescale tgt boxes
        B = len(targets)
        indices = self.matcher(pred_box_copy, anchor_boxes_copy, targets)
        anchor_boxes_copy = box_cxcywh_to_xyxy(anchor_boxes_copy)
        # [M, 4] -> [1, M, 4] -> [B, M, 4]
        anchor_boxes_copy = anchor_boxes_copy[None].repeat(B, 1, 1)

        ious = []
        pos_ious = []
        for i in range(B):
            src_idx, tgt_idx = indices[i]
            # iou between predbox and tgt box
            iou, _ = box_iou(pred_box_copy[i, ...], (targets[i]['boxes']).clone())
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]
            # iou between anchorbox and tgt box
            a_iou, _ = box_iou(anchor_boxes_copy[i], (targets[i]['boxes']).clone())
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)

        ious = torch.cat(ious)
        ignore_idx = ious > self.cfg['igt']
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.cfg['iou_t']

        src_idx = torch.cat(
            [src + idx * anchor_boxes_copy[0].shape[0] for idx, (src, _) in
             enumerate(indices)])
        # [BM,]
        gt_cls = torch.full(pred_cls.shape[:1],
                                self.num_classes,
                                dtype=torch.int64,
                                device=self.device)
        gt_cls[ignore_idx] = -1
        tgt_cls_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        tgt_cls_o[pos_ignore_idx] = -1

        gt_cls[src_idx] = tgt_cls_o.to(self.device)

        foreground_idxs = (gt_cls >= 0) & (gt_cls != self.num_classes)
        num_foreground = foreground_idxs.sum()

        gt_cls_target = torch.zeros_like(pred_cls)
        gt_cls_target[foreground_idxs, gt_cls[foreground_idxs]] = 1

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        # cls loss
        masks = outputs['mask']
        valid_idxs = (gt_cls >= 0) & masks
        loss_labels = self.loss_labels(pred_cls[valid_idxs], 
                                       gt_cls_target[valid_idxs], 
                                       num_foreground)

        # box loss
        tgt_boxes = torch.cat([t['boxes'][i]
                                    for t, (_, i) in zip(targets, indices)], dim=0).to(self.device)
        tgt_boxes = tgt_boxes[~pos_ignore_idx]
        matched_pred_box = pred_box.reshape(-1, 4)[src_idx[~pos_ignore_idx]]
        loss_bboxes = self.loss_bboxes(matched_pred_box, 
                                       tgt_boxes, 
                                       num_foreground)

        # total loss
        losses = self.loss_cls_weight * loss_labels + self.loss_reg_weight * loss_bboxes

        return loss_labels, loss_bboxes, losses

    
if __name__ == "__main__":
    pass
