# Object Detection Benchmark

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

We suggest that PyTorch should be higher than 1.9.0 and Torchvision should be higher than 0.10.3. At least, please make sure your torch is version 1.x.

# Main results on COCO-val
## YOLOF

| Model                                       |  scale     |   mAP   | Weight|
|---------------------------------------------|------------|---------|-------|
| YOLOF_R_18_C5_1x                            |  800,1333  |         | [github](https://github.com/yjh0410/PyTorch_YOLOF/releases/download/YOLOF-weight/yolof_R_18_C5_1x_31.9.pth) |
| YOLOF_R_50_C5_1x                            |  800,1333  |         | [github](https://github.com/yjh0410/PyTorch_YOLOF/releases/download/YOLOF-weight/yolof_R_50_C5_1x_37.3.pth) |
| YOLOF_R_50_DC5_640_3x                       |  640,640   |         |       |

## RetinaNet
| Model                                       |  scale     |   mAP   | Weight|
|---------------------------------------------|------------|---------|-------|
| RetinaNet_R_18_1x                           |  800,1333  |         |       |
| RetinaNet_R_50_1x                           |  800,1333  |         |       |
| RetinaNet_R_50_640_3x                       |  640,640   |         |       |

## FCOS
| Model                                       |  scale     |   mAP   | Weight|
|---------------------------------------------|------------|---------|-------|
| FCOS_R_18_1x                                |  800,1333  |         |       |
| FCOS_R_50_1x                                |  800,1333  |         |       |
| FCOS_R_50_640_3x                            |  640,640   |         |       |


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
               --min_size 800 \
               --max_size 1333 \
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
               --min_size 800 \
               --max_size 1333 \
               --show
```

If you want run a demo of streaming video detection, you need to set `--mode` to `video`, and give the path to video `--path_to_vid`。

```Shell
python demo.py --mode video \
               --path_to_img data/demo/videos/your_video \
               -v yolof50 \
               --cuda \
               --weight path/to/weight \
               --min_size 800 \
               --max_size 1333 \
               --show
```

If you want run video detection with your camera, you need to set `--mode` to `camera`。

```Shell
python demo.py --mode camera \
               -v yolof50 \
               --cuda \
               --weight path/to/weight \
               --min_size 800 \
               --max_size 1333 \
               --show
```
