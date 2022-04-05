# 2 GPUs
# Attention, the following batch size is on single GPU, not all GPUs.
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v retinanet-rt \
                                                    -lr 0.01 \
                                                    -lr_bk 0.01 \
                                                    --batch_size 1 \
                                                    --grad_clip_norm 4.0 \
                                                    --num_workers 4 \
                                                    --schedule 4x \
                                                    --sybn
                                                    # --mosaic
