# retinanet config


retinanet_config = {
    'retinanet18': {
        # input
        'min_size': 800,
        'max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': [128, 64, 32, 16, 8],  # P7, P6, P5, P4, P3
        'act_type': 'relu',
        # neck
        'fpn': 'basic_fpn',
        # head
        'head_dim': 256,
        'head': 'decoupled_head',
        'num_cls_head': 4,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_config': {'basic_size': [[512, 512], [256, 256], [128, 128], [64, 64], [32, 32]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'iou_t': [0.4, 0.5],
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [400, 500, 600, 700, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [400, 500, 600, 700, 800]},
        },
    },

    'retinanet50': {
        # input
        'min_size': 800,
        'max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet50',
        'norm_type': 'FrozeBN',
        'stride': [128, 64, 32, 16, 8],
        'act_type': 'relu',
        # neck
        'fpn': 'basic_fpn',
        # head
        'head_dim': 256,
        'head': 'decoupled_head',
        'num_cls_head': 4,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_config': {'basic_size': [[512, 512], [256, 256], [128, 128], [64, 64], [32, 32]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'iou_t': [0.4, 0.5],
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [400, 500, 600, 700, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [400, 500, 600, 700, 800]},
        },
    },

    'retinanet101': {
        # input
        'min_size': 800,
        'max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet101',
        'norm_type': 'FrozeBN',
        'stride': [128, 64, 32, 16, 8],
        'act_type': 'relu',
        # neck
        'fpn': 'basic_fpn',
        # head
        'head_dim': 256,
        'head': 'decoupled_head',
        'num_cls_head': 4,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_config': {'basic_size': [[512, 512], [256, 256], [128, 128], [64, 64], [32, 32]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'iou_t': [0.4, 0.5],
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [400, 500, 600, 700, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [400, 500, 600, 700, 800]},
        },
    },

    'retinanet-rt': { # Real Time RetinaNet
        # input
        'min_size': 640,
        'max_size': 640,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'RandomShift', 'max_shift': 32},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet50',
        'norm_type': 'FrozeBN',
        'stride': [32, 16, 8],
        'act_type': 'relu',
        # neck
        'fpn': 'basic_fpn',
        # head
        'head_dim': 256,
        'head': 'decoupled_head',
        'num_cls_head': 4,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        # anchor box
        'anchor_config': {'basic_size': [[256, 256], [64, 64], [16, 16]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'iou_t': [0.4, 0.5],
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1500,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [448, 480, 512, 544, 576, 608, 640]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [448, 480, 512, 544, 576, 608, 640]},
        },
    },

}