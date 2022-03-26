python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset/ \
        -v yolof-rt \
        -lr 0.03 \
        -lr_bk 0.01 \
        --batch_size 16 \
        --schedule 3x \
        --grad_clip_norm 4.0 \
