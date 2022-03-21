from .dilated_encoder import DilatedEncoder
from .fpn import BasicFPN


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'dilated_encoder':
        neck = DilatedEncoder(in_dim, 
                              out_dim, 
                              expand_ratio=cfg['expand_ratio'], 
                              dilation_list=cfg['dilation_list'],
                              act_type=cfg['act_type'])

    return neck


def build_fpn(cfg, in_dims, out_dim, from_c5=False, p6_feat=False, p7_feat=False):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'basic_fpn':
        fpn_net = BasicFPN(in_dims=in_dims, 
                           out_dim=out_dim,
                           from_c5=from_c5, 
                           p6_feat=p6_feat,
                           p7_feat=p7_feat)

    return fpn_net

