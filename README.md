# Object Detection Benchmark
This benchmark is my library of object detection.
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


# Coming soon
- [] YOLOF
- [] RetinaNet
- [] FCOS
- [] SSD
- [] YOLOv3
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
| YOLOF_RT                       |  640,640   |         |       |

## RetinaNet
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| RetinaNet_R_18_1x              |  800,1333  |         |       |
| RetinaNet_R_50_1x              |  800,1333  |         |       |
| RetinaNet_RT                   |  512,736   |         |       |

## FCOS
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| FCOS_R_18_1x                   |  800,1333  |         |       |
| FCOS_R_50_1x                   |  800,1333  |         |       |
| FCOS_RT_R_50                   |  640,640   |         |       |
| FCOS_RT_R_50_OTA               |  512,736   |         |       |

As for now, limited by my computation source, only `FCOS_RT` supports `OTA`.

## SSD
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| SSD_VGG_16_320_3x              |  320,320   |         |       |
| SSD_VGG_16_512_3x              |  512,512   |         |       |

## YOLOv3
| Model                          |  scale     |   mAP   | Weight|
|--------------------------------|------------|---------|-------|
| YOLOv3_D_53_608_3x             |  608,608   |         |       |
| YOLOv3_D_53_608_9x             |  608,608   |         |       |


Limited by my computing resources, I cannot use larger backbone networks like `ResNet-101` 
and `ResNeXt-101` to train more powerful detectors.

If you have sufficient computing resources and are already using these larger backbone 
networks to train the detectors in this benchmark, I look forward to your open source 
weight files to complement this project. Thanks a lot.

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
