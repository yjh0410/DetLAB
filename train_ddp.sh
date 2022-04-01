# 2 GPUs
python -m torch.distributed.run --nproc_per_node=2 train.py \
                                                    --cuda \
                                                    -dist \
                                                    --num_gpu 2 \
                                                    -d coco \
                                                    --root /mnt/share/ssd2/dataset/ \
                                                    -v retinanet18 \
                                                    -lr 0.005 \
                                                    -lr_bk 0.01 \
                                                    --batch_size 8 \
                                                    --grad_clip_norm 4.0 \
                                                    --num_workers 4 \
                                                    --schedule 1x \
                                                    # --mosaic
