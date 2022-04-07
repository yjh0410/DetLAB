from .resnet import build_resnet
from .convnext import build_convnext
from .vggnet import vgg16
from .darknet19 import darknet19
from .darknet53 import darknet53


def build_backbone(model_name='resnet50-d', 
                   pretrained=False, 
                   norm_type='BN',
                   in_22k=False):
    print('==============================')
    print('Backbone: {}'.format(model_name.upper()))
    print('--pretrained: {}'.format(pretrained))

    if 'resnet' in model_name:
        model, feat_dim = build_resnet(model_name=model_name, 
                                       pretrained=pretrained,
                                       norm_type=norm_type)

    elif 'convnext' in model_name:
        if model_name[-1] == 'd':
            model, feat_dim = build_convnext(model_name=model_name,
                                            pretrained=pretrained,
                                            res_dilation=True,
                                            in_22k=in_22k)
        else:
            model, feat_dim = build_convnext(model_name=model_name,
                                            pretrained=pretrained,
                                            res_dilation=False,
                                            in_22k=in_22k)
    
    elif model_name == 'vgg16':
        model, feat_dim = vgg16(pretrained=pretrained)

    elif model_name == 'darknet19':
        model, feat_dim = darknet19(pretrained=pretrained, 
                                    norm_type=norm_type)

    elif model_name == 'darknet53':
        model, feat_dim = darknet53(pretrained=pretrained, 
                                    norm_type=norm_type)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
