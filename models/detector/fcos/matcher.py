import math
import torch
import torch.nn.functional as F
from utils.box_ops import *
from utils.misc import sigmoid_focal_loss, SinkhornDistance


@torch.no_grad()
def get_ious_and_iou_loss(inputs,
                          targets,
                          weight=None,
                          box_mode="xyxy",
                          loss_type="iou",
                          reduction="none"):
    """
    Compute iou loss of type ['iou', 'giou', 'linear_iou']

    Args:
        inputs (tensor): pred values
        targets (tensor): target values
        weight (tensor): loss weight
        box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
        loss_type (str): 'giou' or 'iou' or 'linear_iou'
        reduction (str): reduction manner

    Returns:
        loss (tensor): computed iou loss.
    """
    if box_mode == "ltrb":
        inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
        targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
    elif box_mode != "xyxy":
        raise NotImplementedError

    eps = torch.finfo(torch.float32).eps

    inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
        * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
    targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
        * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

    w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                   - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
    h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                   - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

    area_intersect = w_intersect * h_intersect
    area_union = targets_area + inputs_area - area_intersect
    ious = area_intersect / area_union.clamp(min=eps)

    if loss_type == "iou":
        loss = -ious.clamp(min=eps).log()
    elif loss_type == "linear_iou":
        loss = 1 - ious
    elif loss_type == "giou":
        g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
            - torch.min(inputs[..., 0], targets[..., 0])
        g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
            - torch.min(inputs[..., 1], targets[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1 - gious
    else:
        raise NotImplementedError
    if weight is not None:
        loss = loss * weight.view(loss.size())
        if reduction == "mean":
            loss = loss.sum() / max(weight.sum().item(), eps)
    else:
        if reduction == "mean":
            loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()

    return ious, loss


class Matcher(object):
    def __init__(self, 
                 cfg,
                 num_classes,
                 box_weights=[1, 1, 1, 1]):
        self.num_classes = num_classes
        self.center_sampling_radius = cfg['center_sampling_radius']
        self.object_sizes_of_interest = cfg['object_sizes_of_interest']
        self.box_weights = box_weights


    def get_deltas(self, anchors, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `anchors` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, anchors)`` is true.

        Args:
            anchors (Tensor): anchors, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(anchors, torch.Tensor), type(anchors)
        assert isinstance(boxes, torch.Tensor), type(boxes)

        deltas = torch.cat((anchors - boxes[..., :2], boxes[..., 2:] - anchors),
                           dim=-1) * anchors.new_tensor(self.box_weights)
        return deltas


    @torch.no_grad()
    def __call__(self, fpn_strides, anchors, targets):
        gt_classes = []
        gt_anchors_deltas = []
        gt_centerness = []
        # anchors 是个List，第一层List对应batch维度
        # 第二层List对应fpn level的维度，anchors[bi][fpn_i].shape = [M, 2]
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # generate object_sizes_of_interest: List[[M, 2]]
            object_sizes_of_interest = [anchors_i.new_tensor(size).unsqueeze(0).expand(anchors_i.size(0), -1) 
                                        for anchors_i, size in zip(anchors_per_image, self.object_sizes_of_interest)]
            # List[[M, 2]] -> [FxM,2], F是FPN的尺度数量
            object_sizes_of_interest = torch.cat([object_sizes_of_interest], dim=0)
            # List[[M, 2]] -> [FxM,2], F是FPN的尺度数量
            anchors_over_all_feature_maps = torch.cat(anchors_per_image, dim=0)
            # [N, 4]
            tgt_box = targets_per_image['boxes']
            tgt_cls = targets_per_image['labels']

            deltas = self.get_deltas(anchors_over_all_feature_maps, tgt_box.unsqueeze(1))

            if self.center_sampling_radius > 0:
                # bbox centers: [N, 2]
                centers = torch.stack([(tgt_box[..., 0] + tgt_box[..., 1]) * 0.5, 
                                       (tgt_box[..., 1] + tgt_box[..., 3]) * 0.5], 
                                       dim=1)
                is_in_boxes = []
                for stride, anchors_i in zip(fpn_strides, anchors_per_image):
                    radius = stride * self.center_sampling_radius
                    center_boxes = torch.cat((
                        torch.max(centers - radius, tgt_box[:, :2]),
                        torch.min(centers + radius, tgt_box[:, 2:]),
                    ), dim=-1)
                    center_deltas = self.get_deltas(anchors_i, center_boxes.unsqueeze(1))
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
                1, anchors_over_all_feature_maps.size(0))
            gt_positions_area[~is_in_boxes] = math.inf
            gt_positions_area[~is_cared_in_the_level] = math.inf

            # if there are still more than one objects for a position,
            # we choose the one with minimal area
            positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

            # ground truth box regression
            gt_anchors_reg_deltas_i = self.get_deltas(
                anchors_over_all_feature_maps, tgt_box[gt_matched_idxs])

            # ground truth classes
            has_gt = len(targets_per_image) > 0
            if has_gt:
                tgt_cls_i = tgt_cls[gt_matched_idxs]
                # anchors with area inf are treated as background.
                tgt_cls_i[positions_min_area == math.inf] = self.num_classes
            else:
                tgt_cls_i = torch.zeros_like(
                    gt_matched_idxs) + self.num_classes

            # ground truth centerness
            left_right = gt_anchors_reg_deltas_i[:, [0, 2]]
            top_bottom = gt_anchors_reg_deltas_i[:, [1, 3]]
            gt_centerness_i = torch.sqrt(
                (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
            )

            gt_classes.append(tgt_cls_i)
            gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
            gt_centerness.append(gt_centerness_i)

        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas), torch.stack(gt_centerness)


class OTA_Matcher(object):
    def __init__(self, 
                 cfg,
                 num_classes,
                 box_weight=[1.0, 1.0, 1.0, 1.0]) -> None:
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.center_sampling_radius = cfg['center_sampling_radius']
        self.sinkhorn = SinkhornDistance(eps=cfg['eps'], max_iter=cfg['max_iter'])


    def get_deltas(self, anchors, bboxes):
        """
        Get box regression transformation deltas (dl, dr) that can be used
        to transform the `anchors` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, anchors)`` is true.

        Args:
            anchors (Tensor): anchors, e.g., feature map coordinates
            bboxes (Tensor): target of the transformation, e.g., ground-truth bboxes.
        """
        assert isinstance(anchors, torch.Tensor), type(anchors)
        assert isinstance(anchors, torch.Tensor), type(anchors)

        deltas = torch.cat((anchors - bboxes[..., :2], bboxes[..., 2:] - anchors),
                           dim=-1) * anchors.new_tensor(self.box_weight)
        return deltas


    @torch.no_grad()
    def __call__(self, fpn_strides, anchors, pred_cls_logits, pred_deltas, targets):
        gt_cls_labels = []
        gt_box_deltas = []
        assigned_units = []

        # [M, 2], share among all images, M = M1 + M2 + ...
        anchors_over_all = torch.cat(anchors, dim=0)

        # [B, M, C]
        pred_cls_logits = torch.cat(pred_cls_logits, dim=1)
        pred_deltas = torch.cat(pred_deltas, dim=1)

        for tgt_per_image, pred_cls_per_image, pred_deltas_per_image in zip(targets, pred_cls_logits, pred_deltas):
            tgt_labels_per_images = tgt_per_image["labels"]
            tgt_bboxes_per_images = tgt_per_image["boxes"]

            # In gt box and center. [N, M]
            deltas = self.get_deltas(anchors_over_all, tgt_bboxes_per_images.unsqueeze(1))
            # [N, M], N is the number of targets, M is the number of all anchors
            is_in_bboxes = deltas.min(dim=-1).values > 0.01

            # targets bbox centers: [N, 2]
            centers = (tgt_bboxes_per_images[:, :2] + tgt_bboxes_per_images[:, 2:]) * 0.5
            is_in_centers = []
            for stride, anchors_i in zip(fpn_strides, anchors):
                radius = stride * self.center_sampling_radius
                center_bboxes = torch.cat((
                    torch.max(centers - radius, tgt_bboxes_per_images[:, :2]),
                    torch.min(centers + radius, tgt_bboxes_per_images[:, 2:]),
                ), dim=-1)
                center_deltas = self.get_deltas(anchors_i, center_bboxes.unsqueeze(1))
                is_in_centers.append(center_deltas.min(dim=-1).values > 0)
            # [N, M], N is the number of targets, M is the number of all anchors
            is_in_centers = torch.cat(is_in_centers, dim=1)

            del centers, center_bboxes, deltas, center_deltas
            # [N, M], N is the number of targets, M is the number of all anchors
            is_in_bboxes = (is_in_bboxes & is_in_centers)

            num_gt = len(tgt_labels_per_images)  # N
            num_anchor = len(anchors_over_all)   # M
            shape = (num_gt, num_anchor, -1)     # [N, M, -1]

            gt_cls_per_image = F.one_hot(tgt_labels_per_images, self.num_classes).float()

            with torch.no_grad():
                loss_cls = sigmoid_focal_loss(
                    pred_cls_per_image.unsqueeze(0).expand(shape), # [M, C] -> [1, M, C] -> [N, M, C]
                    gt_cls_per_image.unsqueeze(1).expand(shape),   # [N, C] -> [N, 1, C] -> [N, M, C]
                ).sum(dim=-1) # [N, M, C] -> [N, M]

                loss_cls_bg = sigmoid_focal_loss(
                    pred_cls_per_image,
                    torch.zeros_like(pred_cls_per_image),
                ).sum(dim=-1) # [M, C] -> [M]

                # [N, M, 2]
                gt_delta_per_image = self.get_deltas(anchors_over_all, tgt_bboxes_per_images.unsqueeze(1))

                ious, loss_delta = get_ious_and_iou_loss(
                    pred_deltas_per_image.unsqueeze(0).expand(shape), # [M, 4] -> [1, M, 4] -> [N, M, 4]
                    gt_delta_per_image,
                    box_mode="ltrb",
                    loss_type='iou'
                ) # [N, M]

                loss = loss_cls + 3.0 * loss_delta + 1e6 * (1 - is_in_bboxes.float())

                # Performing Dynamic k Estimation, top_candidates = 20
                topk_ious, _ = torch.topk(ious * is_in_bboxes.float(), 20, dim=1)
                mu = ious.new_ones(num_gt + 1)
                mu[:-1] = torch.clamp(topk_ious.sum(1).int(), min=1).float()
                mu[-1] = num_anchor - mu[:-1].sum()
                nu = ious.new_ones(num_anchor)
                loss = torch.cat([loss, loss_cls_bg.unsqueeze(0)], dim=0)

                # Solving Optimal-Transportation-Plan pi via Sinkhorn-Iteration.
                _, pi = self.sinkhorn(mu, nu, loss)

                # Rescale pi so that the max pi for each gt equals to 1.
                rescale_factor, _ = pi.max(dim=1)
                pi = pi / rescale_factor.unsqueeze(1)

                # matched_gt_inds: [M,]
                max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)
                gt_cls_labels_i = tgt_labels_per_images.new_ones(num_anchor) * self.num_classes
                # fg_mask: [M,]
                fg_mask = matched_gt_inds != num_gt
                gt_cls_labels_i[fg_mask] = tgt_labels_per_images[matched_gt_inds[fg_mask]]
                gt_cls_labels.append(gt_cls_labels_i)
                assigned_units.append(max_assigned_units)

                gt_box_deltas_per_image = gt_delta_per_image.new_zeros((num_anchor, 4))
                gt_box_deltas_per_image[fg_mask] = \
                    gt_delta_per_image[matched_gt_inds[fg_mask], torch.arange(num_anchor)[fg_mask]]
                gt_box_deltas.append(gt_box_deltas_per_image)

        # [B, M, C]
        gt_cls_labels = torch.stack(gt_cls_labels)
        gt_box_deltas = torch.stack(gt_box_deltas)

        return gt_cls_labels, gt_box_deltas
