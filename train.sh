python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v retinanet18 \
        --lr_scheduler step \
        --schedule 1x \
        --grad_clip_norm 4.0
