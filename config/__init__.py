from .yolof_config import yolof_config
from .retinanet_config import retinanet_config
from .fcos_config import fcos_config
from .ssdx_config import ssdx_config


def build_config(args):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    if 'yolof' in args.version:
        cfg = yolof_config[args.version]

    elif 'retinanet' in args.version:
        cfg = retinanet_config[args.version]

    elif 'fcos' in args.version:
        cfg = fcos_config[args.version]

    elif 'ssd' in args.version:
        cfg = ssdx_config[args.version]

    return cfg
