import math
import torch
from utils.box_ops import *


class Matcher(object):
    """
    Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors.

    Args:
        match_times(int): Number of positive anchors for each gt box.
    """

    def __init__(self, 
                 num_classes,
                 fpn_strides, 
                 center_sampling_radius, 
                 object_sizes_of_interest,
                 weights=[1, 1, 1, 1]):
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.center_sampling_radius = center_sampling_radius
        self.object_sizes_of_interest = object_sizes_of_interest
        self.weights = weights


    def get_deltas(self, shifts, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `shifts` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, shifts)`` is true.

        Args:
            shifts (Tensor): shifts, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(shifts, torch.Tensor), type(shifts)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        deltas = torch.cat((shifts - boxes[..., :2], boxes[..., 2:] - shifts),
                           dim=-1) * shifts.new_tensor(self.weights)
        return deltas


    @torch.no_grad()
    def __call__(self, shifts, targets):
        gt_classes = []
        gt_shifts_deltas = []
        gt_centerness = []
        # shifts 是个List，第一层List对应batch维度
        # 第二层List对应fpn level的维度，shifts[bi][fpn_i].shape = [M, 2]
        for shifts_per_image, targets_per_image in zip(shifts, targets):
            # generate object_sizes_of_interest: List[[M, 2]]
            object_sizes_of_interest = [shifts_i.new_tensor(size).unsqueeze(0).expand(shifts_i.size(0), -1) 
                                        for shifts_i, size in zip(shifts_per_image, self.object_sizes_of_interest)]
            # List[[M, 2]] -> [FxM,2], F是FPN的尺度数量
            object_sizes_of_interest = torch.cat([object_sizes_of_interest], dim=0)
            # List[[M, 2]] -> [FxM,2], F是FPN的尺度数量
            shifts_over_all_feature_maps = torch.cat(shifts_per_image, dim=0)
            # [N, 4]
            tgt_box = targets_per_image['boxes']
            tgt_cls = targets_per_image['labels']

            deltas = self.get_deltas(shifts_over_all_feature_maps, tgt_box.unsqueeze(1))

            if self.center_sampling_radius > 0:
                # bbox centers: [N, 2]
                centers = torch.stack([(tgt_box[..., 0] + tgt_box[..., 1]) * 0.5, 
                                       (tgt_box[..., 1] + tgt_box[..., 3]) * 0.5], 
                                       dim=1)
                is_in_boxes = []
                for stride, shifts_i in zip(self.fpn_strides, shifts_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, tgt_box[:, :2]),
                        torch.min(centers + radius, tgt_box[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.get_deltas(shifts_i, center_boxes.unsqueeze(1))
                    is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                is_in_boxes = torch.cat(is_in_boxes, dim=1)
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = deltas.min(dim=-1).values > 0

            max_deltas = deltas.max(dim=-1).values
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                (max_deltas <= object_sizes_of_interest[None, :, 1])

            tgt_box_area = (tgt_box[:, 2] - tgt_box[:, 0]) * (tgt_box[:, 3] - tgt_box[:, 1])
            gt_positions_area = tgt_box_area.unsqueeze(1).repeat(
                1, shifts_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = math.inf
            gt_positions_area[~is_cared_in_the_level] = math.inf

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

            # ground truth box regression
            gt_shifts_reg_deltas_i = self.get_deltas(
                shifts_over_all_feature_maps, tgt_box[gt_matched_idxs])

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                tgt_cls_i = tgt_cls[gt_matched_idxs]
                # Shifts with area inf are treated as background.
                tgt_cls_i[positions_min_area == math.inf] = self.num_classes
            else:
                tgt_cls_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes

            # ground truth centerness
            left_right = gt_shifts_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_shifts_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            gt_classes.append(tgt_cls_i)
            gt_shifts_deltas.append(gt_shifts_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

        return torch.stack(gt_classes), torch.stack(gt_shifts_deltas), torch.stack(gt_centerness)
