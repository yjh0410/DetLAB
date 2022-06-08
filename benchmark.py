import argparse
import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn

from config.yolof_config import yolof_config
from dataset.transforms import ValTransforms
from dataset.coco import COCODataset, coco_class_index, coco_class_labels
from utils.com_flops_params import FLOPs_and_Params
from utils import fuse_conv_bn
from utils.misc import load_weight

from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Benchmark')
    # Model
    parser.add_argument('-v', '--version', default='yolof50', type=str,
                        help='build yolof')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse conv and bn')
    parser.add_argument('--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('--nms_thresh', default=0.45, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='NMS threshold')
    # data root
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    # basic
    parser.add_argument('--min_size', default=800, type=int,
                        help='the min size of input image')
    parser.add_argument('--max_size', default=1333, type=int,
                        help='the min size of input image')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    # cuda
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')

    return parser.parse_args()


def test(args, net, device, testset, transform):
    # Step-1: Compute FLOPs and Params
    FLOPs_and_Params(model=net, 
                     min_size=args.min_size,
                     max_size=args.max_size, 
                     device=device)

    # Step-2: Compute FPS
    num_images = 2002
    total_time = 0
    count = 0
    with torch.no_grad():
        for index in range(num_images):
            if index % 500 == 0:
                print('Testing image {:d}/{:d}....'.format(index+1, num_images))
            image, _ = testset.pull_image(index)

            orig_h, orig_w, _ = image.shape
            orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])

            # prepare
            x = transform(image)[0]
            x = x.unsqueeze(0).to(device)

            # star time
            torch.cuda.synchronize()
            start_time = time.perf_counter()    

            # inference
            bboxes, scores, cls_inds = net(x)
            
            # rescale
            if transform.padding:
                # The input image is padded with 0 on the short side, aligning with the long side.
                bboxes *= max(orig_h, orig_w)
            else:
                # the input image is not padded.
                bboxes *= orig_size

            # end time
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            # print("detection time used ", elapsed, "s")
            if index > 1:
                total_time += elapsed
                count += 1
            
        print('- FPS :', 1.0 / (total_time / count))



if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    print('test on coco-val ...')
    data_dir = os.path.join(args.root, 'COCO')
    class_names = coco_class_labels
    class_indexs = coco_class_index
    num_classes = 80
    dataset = COCODataset(
                data_dir=data_dir,
                image_set='val2017')

    # YOLOF Config
    cfg = yolof_config[args.version]
    # build model
    model = build_model(args=args, 
                        cfg=cfg,
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False)

    # load trained weight
    model = load_weight(device=device, 
                        model=model, 
                        path_to_ckpt=args.weight)


    # fuse conv bn
    if args.fuse_conv_bn:
        print('fuse conv and bn ...')
        model = fuse_conv_bn(model)

    # transform
    transform = ValTransforms(min_size=args.min_size, 
                              max_size=args.max_size,
                              pixel_mean=cfg['pixel_mean'],
                              pixel_std=cfg['pixel_std'],
                              format=cfg['format'],
                              padding=cfg['val_padding'])

    # run
    test(args=args,
        net=model,
        device=device, 
        testset=dataset,
        transform=transform
        )
