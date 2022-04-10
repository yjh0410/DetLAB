from .dilated_encoder import DilatedEncoder
from .spp import SPP, SPPBlock, SPPBlockCSP
from .fpn import BasicFPN
from .bifpn import BiFPN
from .pafpn import PaFPN
from .yolofpn import YoloFPN
from .yolopafpn import YoloPaFPN


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
    elif model == 'spp':
        neck = SPP(in_dim, 
                   out_dim, 
                   e=cfg['expand_ratio'], 
                   kernel_sizes=cfg['kernel_sizes'],
                   norm_type=cfg['neck_norm'],
                   act_type=cfg['neck_act'])

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

    elif model == 'bifpn':
        fpn_net = BiFPN(in_dims=in_dims,
                        out_dim=out_dim, 
                        num_bifpn_layers=cfg['num_bifpn_layers'],
                        fuse_type=cfg['fuse_type'], 
                        norm_type=cfg['neck_norm'], 
                        memory_efficient=cfg['memory_efficient'],
                        p6_feat=p6_feat,
                        p7_feat=p7_feat)

    elif model == 'pafpn':
        fpn_net = PaFPN(in_dims=in_dims,
                        out_dim=out_dim, 
                        norm_type=cfg['fpn_norm'], 
                        p6_feat=p6_feat,
                        p7_feat=p7_feat)

                            
    elif model == 'yolo_fpn':
        fpn_net = YoloFPN(in_dims=in_dims,
                          norm_type=cfg['fpn_norm'],
                          act_type=cfg['fpn_act'],
                          spp=cfg['use_spp'],
                          out_dim=cfg['head_dim'])

    elif model == 'yolo_pafpn':
        fpn_net = None

    return fpn_net

