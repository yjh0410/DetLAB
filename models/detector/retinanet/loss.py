import torch
import torch.nn as nn
from .matcher import Matcher
from utils.box_ops import *
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 alpha=0.25,
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
        self.matcher = Matcher(num_classes=num_classes,
                               iou_threshold=cfg['iou_t'],
                               iou_labels=cfg['iou_labels'],
                               allow_low_quality_matches=cfg['allow_low_quality_matches'])


    def _ema_update(self, name: str, value: float, initial_value: float, momentum: float = 0.9):
            """
            Apply EMA update to `self.name` using `value`.
            This is mainly used for loss normalizer. In Detectron1, loss is normalized by number
            of foreground samples in the batch. When batch size is 1 per GPU, #foreground has a
            large variance and using it lead to lower performance. Therefore we maintain an EMA of
            #foreground to stabilize the normalizer.
            Args:
                name: name of the normalizer
                value: the new value to update
                initial_value: the initial value to start with
                momentum: momentum of EMA
            Returns:
                float: the updated EMA value
            """
            if hasattr(self, name):
                old = getattr(self, name)
            else:
                old = initial_value
            new = old * momentum + value * (1 - momentum)
            setattr(self, name, new)
            return new


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


    def __call__(self,
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
        bs = outputs['pred_cls'].size(0)
        # [M, 4] -> [B, M, 4]
        anchors = anchor_boxes[None].repeat(bs, 1, 1)
        # convert [x, y, w, h] -> [x1, y1, x2, y2]
        anchors = box_cxcywh_to_xyxy(anchors)
        # label assignment
        tgt_classes, tgt_boxes = self.matcher(anchors, targets)

        # [B, M, C] -> [BM, C]
        pred_cls = outputs['pred_cls'].view(-1, self.num_classes)
        pred_box = outputs['pred_box'].view(-1, 4)

        tgt_classes = tgt_classes.flatten()
        tgt_boxes = tgt_boxes.view(-1, 4)

        foreground_idxs = (tgt_classes >= 0) & (tgt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        print(num_foreground)

        gt_cls_target = torch.zeros_like(pred_cls)
        gt_cls_target[foreground_idxs, tgt_classes[foreground_idxs]] = 1

        # cls loss
        masks = outputs['mask'].view(-1)
        valid_idxs = (tgt_classes >= 0) & masks
        loss_labels = self.loss_labels(pred_cls[valid_idxs], 
                                       gt_cls_target[valid_idxs], 
                                       num_foreground)

        # box loss
        loss_bboxes = self.loss_bboxes(pred_box[foreground_idxs],
                                        tgt_boxes[foreground_idxs].to(pred_box.device),
                                        num_foreground)

        # total loss
        losses = self.loss_cls_weight * loss_labels + self.loss_reg_weight * loss_bboxes

        loss_dict = dict(
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                losses = losses
        )

        return loss_dict

    
if __name__ == "__main__":
    pass
