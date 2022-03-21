import torch
import torch.nn as nn
from utils.box_ops import *
from utils.matcher import build_matcher
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
        self.matcher = build_matcher(cfg, num_classes)
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight


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
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
            anchor_boxes: (Tensor) [M, 4]
        """
        tgt_classes, matched_tgt_boxes = self.matcher(anchor_boxes, targets)

        # [B, M, C] -> [BM, C]
        pred_cls = outputs['pred_cls'].view(-1, self.num_classes)
        pred_box = outputs['pred_box'].view(-1, 4)

        tgt_classes = tgt_classes.flatten()
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

        foreground_idxs = (tgt_classes >= 0) & (tgt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()


        gt_cls_target = torch.zeros_like(pred_cls)
        gt_cls_target[foreground_idxs, tgt_classes[foreground_idxs]] = 1

        # cls loss
        masks = outputs['mask'].view(-1)
        valid_idxs = (tgt_classes >= 0) & masks
        loss_labels = self.loss_labels(pred_cls[valid_idxs], 
                                       gt_cls_target[valid_idxs], 
                                       num_foreground)

        # box loss
        matched_pred_box = pred_box[foreground_idxs]
        loss_bboxes = self.loss_bboxes(matched_pred_box,
                                        matched_tgt_boxes,
                                        num_foreground)

        # total loss
        losses = self.loss_cls_weight * loss_labels + self.loss_reg_weight * loss_bboxes

        return loss_labels, loss_bboxes, losses

    
if __name__ == "__main__":
    pass
