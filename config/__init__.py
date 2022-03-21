from .yolof_config import yolof_config
from .retinanet_config import retinanet_config


def build_config(args):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    if 'yolof' in args.version:
        cfg = yolof_config[args.version]

    elif 'retinanet' in args.version:
        cfg = retinanet_config[args.version]

    return cfg
