# retinanet config


retinanet_config = {
    'ssdv2_vgg16_320': {
        # input
        'train_min_size': 384,
        'train_max_size': 384,
        'test_min_size': 320,
        'test_max_size': 320,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        'val_padding': False,
        # model
        'backbone': 'vgg16',
        'norm_type': '',
        'stride': [8, 16, 32, 64],  # P3, P4, P5, P6
        # neck
        'neck': 'spp',
        'fpn': 'pafpn',
        'fpn_norm': 'GN',
        'from_c5': True,
        'p6_feat': True,
        'p7_feat': False,
        # head
        'head_dim': 256,
        'head_norm': 'GN',
        'act_type': 'relu',
        'head': 'decoupled_head',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.05,
        'train_nms_thresh': 0.6,
        'test_nms_thresh': 0.45,
        'test_score_thresh': 0.35,
        # matcher
        'matcher': 'ota_matcher',
        'topk_candidate': 10,
        'eps': 0.1, 
        'max_iter': 50,
        'ctr_clamp': None,
        'center_sampling_radius': 2.5,
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
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [192, 256, 320, 384]},
        },
    },

    'ssdv2_vgg16_512': {
        # input
        'train_min_size': 640,
        'train_max_size': 640,
        'test_min_size': 512,
        'test_max_size': 512,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        'val_padding': False,
        # model
        'backbone': 'vgg16',
        'norm_type': '',
        'stride': [8, 16, 32, 64],  # P3, P4, P5, P6
        # neck
        'neck': 'spp',
        'fpn': 'basic_fpn',
        'from_c5': True,
        'p6_feat': True,
        'p7_feat': False,
        # head
        'head_dim': 256,
        'head_norm': '',
        'act_type': 'relu',
        'head': 'decoupled_head',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.05,
        'nms_thresh': 0.6,
        'test_score_thresh': 0.35,
        # anchor box
        'anchor_config': {'basic_size': [[32, 32], [64, 64], [128, 128]],
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
            '3x': {'max_epoch': 36, 
                    'lr_epoch': [24, 33], 
                    'multi_scale': [256, 320, 384, 448, 512, 576, 640]},
        },
    },

}