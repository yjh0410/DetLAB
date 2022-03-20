from .basic_matcher import BasicMatcher
from .uniform_matcher import UniformMatcher
from .hungarian_matcher import HungarianMatcher


def build_matcher(cfg):
    matcher = cfg['matcher']
    print('==============================')
    print('Matcher: {}'.format(matcher.upper()))

    if matcher == 'basic_matcher':
        matcher = BasicMatcher()

    elif matcher == 'uniform_matcher':
        matcher = UniformMatcher(cfg['topk'])

    elif matcher == 'atss_matcher':
        matcher = None

    elif matcher == 'hungarian_matcher':
        matcher = HungarianMatcher(cost_class=cfg['cost_class_weight'], 
                                   cost_bbox=cfg['cost_bbox_weight'], 
                                   cost_giou=cfg['cost_giou_weight'])

    elif matcher == 'ota_matcher':
        matcher = None

    elif matcher == 'sim_ota_matcher':
        matcher = None


    return matcher

