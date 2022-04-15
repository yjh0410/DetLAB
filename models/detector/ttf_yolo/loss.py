import torch
from .matcher import OTA_Matcher
from utils.box_ops import *
from utils.misc import sigmoid_varifocal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 alpha=0.75,
                 gamma=2.0,
                 loss_cls_weight=1.0, 
                 loss_reg_weight=1.0,
                 num_classes=80):
        self.cfg = cfg
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        self.matcher = OTA_Matcher(cfg, 
                                    num_classes, 
                                    box_weights=[1., 1., 1., 1.])


    def loss_labels(self, pred_cls, tgt_cls, num_boxes=1.0):
        """
            pred_cls: (Tensor) [N, C]
            tgt_cls:  (Tensor) [N, C]
        """
        # cls loss: [V, C]
        loss_cls = sigmoid_varifocal_loss(pred_cls, tgt_cls, self.alpha, self.gamma, reduction='none')

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


    def __call__(self,
                 outputs, 
                 targets, 
                 anchors=None):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_box']: (Tensor) [B, M, 4]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
            anchors: (List of Tensor) List[Tensor[M, 4]], len(anchors) == num_fpn_levels
        """
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        gt_classes, gt_bboxes, gt_ious = self.matcher(fpn_strides = fpn_strides, 
                                                      anchors = anchors, 
                                                      pred_cls_logits = outputs['pred_cls'], 
                                                      pred_boxes = outputs['pred_box'], 
                                                      targets = targets)

        # [B, M, C] -> [BM, C]
        pred_cls =  torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_box =  torch.cat(outputs['pred_box'], dim=1).view(-1, 4)
        masks = torch.cat(outputs['mask'], dim=1).view(-1)

        gt_classes = gt_classes.flatten().to(device)
        gt_bboxes = gt_bboxes.view(-1, 4).to(device)
        gt_ious = gt_ious.view(-1, 1).to(device)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1
        gt_classes_target = gt_classes_target * gt_ious  # classification with iou-awareness

        # cls loss
        valid_idxs = (gt_classes >= 0) & masks
        loss_labels = self.loss_labels(
            pred_cls[valid_idxs],
            gt_classes_target[valid_idxs],
            num_boxes=num_foreground)

        # box loss
        loss_bboxes = self.loss_bboxes(
            pred_box[foreground_idxs],
            gt_bboxes[foreground_idxs],
            num_boxes=num_foreground)

        # total loss
        losses = self.loss_cls_weight * loss_labels + \
                 self.loss_reg_weight * loss_bboxes

        loss_dict = dict(
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                losses = losses
        )

        return loss_dict


if __name__ == "__main__":
    pass
