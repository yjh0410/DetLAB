import math
import torch
import torch.nn.functional as F
from utils.box_ops import *
from utils.misc import sigmoid_focal_loss, SinkhornDistance


@torch.no_grad()
def get_ious_and_iou_loss(bboxes1, bboxes2):
    """
        pred_box: (Tensor) [N, 4]
        tgt_box:  (Tensor) [N, 4]
    """
    eps = torch.finfo(torch.float32).eps
    # iou
    ious, _ = box_iou(bboxes1, bboxes2)  # [N, M]
    # giou loss
    iou_loss = -ious.clamp(min=eps).log()

    return ious, iou_loss


class OTA_Matcher(object):
    def __init__(self, 
                 cfg,
                 num_classes,
                 box_weights=[1.0, 1.0, 1.0, 1.0]) -> None:
        self.num_classes = num_classes
        self.box_weights = box_weights
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
                           dim=-1) * anchors.new_tensor(self.box_weights)
        return deltas


    @torch.no_grad()
    def __call__(self, fpn_strides, anchors, pred_cls_logits, pred_boxes, targets):
        gt_classes = []
        gt_bboxes = []
        gt_ious = []
        assigned_units = []
        device = anchors[0].device

        # List[F, M, 2] -> [M, 2]
        anchors_over_all_feature_maps = torch.cat(anchors, dim=0)

        # [B, M, C]
        pred_cls_logits = torch.cat(pred_cls_logits, dim=1)
        pred_boxes = torch.cat(pred_boxes, dim=1)

        for tgt_per_image, pred_cls_per_image, pred_bboxes_per_image in zip(targets, pred_cls_logits, pred_boxes):
            tgt_labels_per_images = tgt_per_image["labels"].to(device)
            tgt_bboxes_per_images = tgt_per_image["boxes"].to(device)

            # [N, M, 4], N is the number of targets, M is the number of all anchors
            deltas = self.get_deltas(anchors_over_all_feature_maps, tgt_bboxes_per_images.unsqueeze(1))
            # [N, M]
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
                # [N, Mi, 2]
                center_deltas = self.get_deltas(anchors_i, center_bboxes.unsqueeze(1))
                is_in_centers.append(center_deltas.min(dim=-1).values > 0)
            # [N, M], M = M1 + M2 + ... + MF
            is_in_centers = torch.cat(is_in_centers, dim=1)

            del centers, center_bboxes, deltas, center_deltas

            # [N, M]
            is_in_bboxes = (is_in_bboxes & is_in_centers)

            num_gt = len(tgt_labels_per_images)               # N
            num_anchor = len(anchors_over_all_feature_maps)   # M
            shape = (num_gt, num_anchor, -1)                  # [N, M, -1]

            gt_classes_per_image = F.one_hot(tgt_labels_per_images, self.num_classes).float()

            with torch.no_grad():
                loss_cls = sigmoid_focal_loss(
                    pred_cls_per_image.unsqueeze(0).expand(shape),     # [M, C] -> [1, M, C] -> [N, M, C]
                    gt_classes_per_image.unsqueeze(1).expand(shape),   # [N, C] -> [N, 1, C] -> [N, M, C]
                ).sum(dim=-1) # [N, M, C] -> [N, M]

                loss_cls_bg = sigmoid_focal_loss(
                    pred_cls_per_image,
                    torch.zeros_like(pred_cls_per_image),
                ).sum(dim=-1) # [M, C] -> [M]

                # iou: [N, M]; iou_loss: [N, M]
                ious, loss_bboxes = get_ious_and_iou_loss(tgt_bboxes_per_images, pred_bboxes_per_image)

                loss = loss_cls + 3.0 * loss_bboxes + 1e6 * (1 - is_in_bboxes.float())

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
                # [M,]
                gt_classes_i = tgt_labels_per_images.new_ones(num_anchor) * self.num_classes
                # fg_mask: [M,]
                fg_mask = matched_gt_inds != num_gt
                gt_classes_i[fg_mask] = tgt_labels_per_images[matched_gt_inds[fg_mask]]
                gt_classes.append(gt_classes_i)
                assigned_units.append(max_assigned_units)

                tgt_bboxes_per_images_ = tgt_bboxes_per_images.unsqueeze(1).expand(shape)  # [N, 4] -> [N, 1, 4] -> [N, M, 4]
                # [M, 4]
                gt_bboxes_per_image = tgt_bboxes_per_images.new_zeros((num_anchor, 4))
                gt_bboxes_per_image[fg_mask] = tgt_bboxes_per_images_[matched_gt_inds[fg_mask], torch.arange(num_anchor)[fg_mask]]
                gt_bboxes.append(gt_bboxes_per_image)

                # [M,]
                gt_ious_per_image = ious.new_zeros((num_anchor, 1))
                gt_ious_per_image[fg_mask] = ious[matched_gt_inds[fg_mask],
                                                  torch.arange(num_anchor)[fg_mask]].unsqueeze(1)
                gt_ious.append(gt_ious_per_image)


        # [B, M, C]
        gt_classes = torch.stack(gt_classes)
        # [B, M, 4]
        gt_bboxes = torch.stack(gt_bboxes)
        # [B, M,]
        gt_ious = torch.stack(gt_ious)

        return gt_classes, gt_bboxes, gt_ious
