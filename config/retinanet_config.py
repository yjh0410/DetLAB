# retinanet config


retinanet_config = {
    'retinanet18': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet18',
        'norm_type': 'FrozeBN',
        'stride': [8, 16, 32, 64, 128],  # P3, P4, P5, P6, P7
        # neck
        'fpn': 'basic_fpn',
        'from_c5': True,
        'p6_feat': True,
        'p7_feat': True,
        # head
        'head_dim': 256,
        'head_norm': '',
        'act_type': 'relu',
        'head': 'decoupled_head',
        'num_cls_head': 4,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        'test_score_thresh': 0.35,
        # anchor box
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'matcher': 'matcher',
        'iou_t': [0.4, 0.5],
        'iou_labels': [0, -1, 1], # [negative sample, ignored sample, positive sample]
        'allow_low_quality_matches': True,
        'ctr_clamp': None,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
        },
    },

    'retinanet50': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet50',
        'norm_type': 'FrozeBN',
        'stride': [8, 16, 32, 64, 128],
        # neck
        'fpn': 'basic_fpn',
        'from_c5': True,
        'p6_feat': True,
        'p7_feat': True,
        # head
        'head_dim': 256,
        'head_norm': '',
        'act_type': 'relu',
        'head': 'decoupled_head',
        'num_cls_head': 4,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        'test_score_thresh': 0.35,
        # anchor box
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'matcher': 'matcher',
        'iou_t': [0.4, 0.5],
        'iou_labels': [0, -1, 1],
        'allow_low_quality_matches': True,
        'ctr_clamp': None,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
        },
    },

    'retinanet101': {
        # input
        'train_min_size': 800,
        'train_max_size': 1333,
        'test_min_size': 800,
        'test_max_size': 1333,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '1x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '2x':[{'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}],

            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        # model
        'backbone': 'resnet101',
        'norm_type': 'FrozeBN',
        'stride': [8, 16, 32, 64, 128],
        # neck
        'fpn': 'basic_fpn',
        'from_c5': True,
        'p6_feat': True,
        'p7_feat': True,
        # head
        'head_dim': 256,
        'head_norm': '',
        'act_type': 'relu',
        'head': 'decoupled_head',
        'num_cls_head': 4,
        'num_reg_head': 4,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        'test_score_thresh': 0.35,
        # anchor box
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
                          'aspect_ratio': [0.5, 1.0, 2.0],
                          'area_scale': [2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)]},
        # matcher
        'matcher': 'matcher',
        'iou_t': [0.4, 0.5],
        'iou_labels': [0, -1, 1],
        'allow_low_quality_matches': True,
        'ctr_clamp': None,
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup': 'linear',
        'wp_iter': 1000,
        'warmup_factor': 0.00066667,
        'epoch': {
            '1x': {'max_epoch': 12, 
                    'lr_epoch': [8, 11], 
                    'multi_scale': None},
            '2x': {'max_epoch': 24, 
                    'lr_epoch': [16, 22], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [640, 672, 704, 736, 768, 800]},
        },
    },

}