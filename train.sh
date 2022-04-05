python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v retinanet-rt \
        -lr 0.01 \
        -lr_bk 0.01 \
        --batch_size 8 \
        --schedule 1x \
        --grad_clip_norm 4.0 \
