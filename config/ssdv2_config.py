# ssdv2 config


ssdv2_config = {
    'ssdv2_vgg16': {
        # input
        'train_min_size': 640,
        'train_max_size': 640,
        'test_min_size': 640,
        'test_max_size': 640,
        'format': 'RGB',
        'pixel_mean': [0.485, 0.456, 0.406],
        'pixel_std': [0.229, 0.224, 0.225],
        'transforms': {
            '3x':[{'name': 'DistortTransform',
                   'hue': 0.1,
                   'saturation': 1.5,
                   'exposure': 1.5},
                  {'name': 'RandomHorizontalFlip'},
                  {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                  {'name': 'ToTensor'},
                  {'name': 'Resize'},
                  {'name': 'Normalize'},
                  {'name': 'PadImage'}]},
        'val_padding': False,
        # model
        'backbone': 'vgg16',
        'norm_type': '',
        'stride': [8, 16, 32],  # P3, P4, P5, P6
        # neck
        'neck': 'spp',
        'expand_ratio': 0.5,
        'kernel_sizes': [5, 9, 13],
        'neck_norm': 'GN',
        'neck_act': 'relu',
        # fpn neck
        'fpn': 'pafpn',
        'fpn_norm': 'GN',
        'from_c5': False,
        'p6_feat': False,
        'p7_feat': False,
        # head
        'head_dim': 160,
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
        'loss_reg_weight': 2.0,
        'loss_iou_weight': 0.5,
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
                    'multi_scale': [320, 3352, 384, 416, 448, 480, 512, 544, 576, 608, 640]},
        },
    },

}