from turtle import forward
import torch
import torch.nn as nn


class BasicMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        pass


    def forward(self, pred_boxes, anchor_boxes, targets):
        """
            pred_boxes: (Tensor)   [B, num_queries, 4]
            anchor_boxes: (Tensor) [num_queries, 4]
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """
        return
        