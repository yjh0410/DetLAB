# Object Detection Laboratory
This project is my library of **One-stage Object Detection**.
My code is clean and concise, without too deep encapsulation, 
so it is easy to understand the function of each module.

For example, 

- If you want to know how to pull data from `COCO`, you just open `dataset/coco.py`.
- If you want to know how to build `RetinaNet`, you just open `models/detector/retinanet/` and `config/retinanet_config.py`.
- If you want to know how to build `FPN`, you just open `models/neck/fpn.py/`.
- If you want to know the whole pipeline of training, you just open `train.py`.
- If you want to know the whole pipeline of evaluation, you just open `eval.py` and `evaluator/coco_evaluator.py`.
- If you want to know how to visualize the detection results on detection benchmark like `COCO` or `VOC`, you just open `test.py`.
- If you want to know how to run a demo with images or videos on your own device, you just open `demo.py`.
- If you want to know how to run a demo with your own camero, you just also open `demo.py`.

So, it is very friendly, right?

I am sure you will soon become familiar with this benchmark and add your own modules to it.

***However, limited by my computing resources, I cannot use larger backbone networks like `ResNet-101` 
and `ResNeXt-101` to train more powerful detectors. If you have sufficient computing resources and are already using these larger backbone 
networks to train the detectors in this benchmark, I look forward to your open source 
weight files to complement this project. Thanks a lot.***

# Coming soon
- [x] YOLOF
- [x] RetinaNet
- [] FCOS
- [] SSDX
- [] TTF-YOLO
- [] Anchor DeTR
- [] Dynamic Head

More advanced detectors are coming soon ...


# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n detection python=3.6
```

- Then, activate the environment:
```Shell
conda activate detection
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

We suggest that PyTorch should be higher than 1.9.0 and Torchvision should be higher than 0.10.3. 
At least, please make sure your torch is version 1.x.

# Main results on COCO-val
## YOLOF

| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| YOLOF_R_18_C5_1x               |  800,1333  |   31.6  | [github](https://github.com/yjh0410/ObjectDetectionBenchmark/releases/download/object-detection-benchmark-weight/yolof_r18_C5_1x_31.6.pth) |
| YOLOF_R_50_C5_1x               |  800,1333  |   37.6  | [github](https://github.com/yjh0410/ObjectDetectionBenchmark/releases/download/object-detection-benchmark-weight/yolof_r50_C5_1x_37.6.pth) |
| YOLOF-RT_R_50_DC5_3x           |  640,640   |   38.1  | [github](https://github.com/yjh0410/ObjectDetectionBenchmark/releases/download/object-detection-benchmark-weight/yolof-rt_r50_DC5_1x_38.1.pth) |

## RetinaNet
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| RetinaNet_R_18_1x              |  800,1333  |   29.3  | [github](https://github.com/yjh0410/ObjectDetectionBenchmark/releases/download/object-detection-benchmark-weight/retinanet_r18_1x_29.3.pth) |
| RetinaNet_R_50_1x              |  800,1333  |   35.8  | [github](https://github.com/yjh0410/ObjectDetectionBenchmark/releases/download/object-detection-benchmark-weight/retinanet_r50_1x_35.8.pth) |

In my RetinaNet:
- For regression head, `GIoU Loss` is deployed rather than `SmoothL1Loss`

## FCOS
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| FCOS_R_18_1x                   |  800,1333  |  30.4   | [github](https://github.com/yjh0410/ObjectDetectionBenchmark/releases/download/object-detection-benchmark-weight/fcos_r18_1x_30.4.pth) |
| FCOS_R_50_1x                   |  800,1333  |         |       |
| FCOS_R_50_OTA_1x               |  800,1333  |         |       |
| FCOS-RT_R_50_OTA_3x            |  640,640   |         |       |

In my FCOS:
- For regression head, `GIoU loss` is deployed rather than `IoU loss`

## SSDX
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| SSDX_VGG_16_320_3x             |  320,320   |         |       |
| SSDX_VGG_16_512_3x             |  512,512   |         |       |
| SSDX_VGG_16_512_3x             |  640,640   |         |       |

Plan to do for SSDX:
- [] `C3`, `C4`, `C5` backbone features
- [] `SPP` block for neck network
- [] `FPN` for feature pyramid
- [] `Anchor free` for bounding box regression
- [] `OTA` for dynamic label assignment
- [] `Focal loss` for classification head
- [] `GIoU loss` for regression head
- [] `IoU-Aware` branch for regression head


## TTF-YOLO
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| TTF-YOLO_D_19_3x               |  640,640   |         |       |
| TTF-YOLO_D_53_3x               |  640,640   |         |       |

Plan to do for TTF-YOLO:
- [] Based on YOLOv3 structure with `SPP` module
- [] Remove `objectness` head
- [] `Anchor free` for bounding box regression
- [] `OTA` for dynamic label assignment
- [] `VariFocal loss` for classification head with `IoU-awareness`
- [] `GIoU loss` for regression head
- [] `3x` training schedule (~37 epochs) rather than 300 epochs
- [] `Mosaic` augmentation

## Anchor-DeTR
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| Anchor-DeTR_R_50_C5_4x         |  800,1333  |         |       |
| Anchor-DeTR-RT_R_50_DC5_4x     |  512,736   |         |       |


# Train
## Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

## Multi GPUs
```Shell
sh train_ddp.sh
```

You can change the configurations of `train_ddp.sh`, according to your own situation.

# Test
```Shell
python test.py -d coco \
               --cuda \
               -v yolof50 \
               --weight path/to/weight \
               --root path/to/dataset/ \
               --show
```

# Demo
I have provide some images in `data/demo/images/`, so you can run following command to run a demo:

```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               -v yolof50 \
               --cuda \
               --weight path/to/weight \
               --show
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolof50 \
               --cuda \
               --weight path/to/weight \
               --show
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v yolof50 \
               --cuda \
               --weight path/to/weight \
               --show
```
