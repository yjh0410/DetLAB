from .basic_matcher import BasicMatcher
from .uniform_matcher import UniformMatcher
from .hungarian_matcher import HungarianMatcher


def build_matcher(cfg, num_classes=80):
    matcher = cfg['matcher']
    print('==============================')
    print('Matcher: {}'.format(matcher.upper()))

    if matcher == 'basic_matcher':
        matcher = BasicMatcher(num_classes=num_classes,
                               iou_t=cfg['iou_t'],
                               iou_labels=cfg['iou_labels'],
                               allow_low_quality_matches=cfg['allow_low_quality_matches'])

    elif matcher == 'uniform_matcher':
        matcher = UniformMatcher(cfg['topk'])

    elif matcher == 'atss':
        matcher = None

    elif matcher == 'hungarian_matcher':
        matcher = HungarianMatcher(cost_class=cfg['cost_class_weight'], 
                                   cost_bbox=cfg['cost_bbox_weight'], 
                                   cost_giou=cfg['cost_giou_weight'])

    elif matcher == 'ota':
        matcher = None

    elif matcher == 'sim_ota':
        matcher = None


    return matcher

