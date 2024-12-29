# RUN

```bash
torchrun --standalone --max-restarts=1 --nnodes=1 --nproc_per_node=1 train.py --amp --train \
--date_dir $pwd/data --data-workers 6 --model resnet50 --batch-sz 128 --lr 0.5 \
--lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear \
--criterion ce-with-label-smoothing --optim sgd --lr-scheduler cosineannealinglr
--auto-augment ta_wide --epochs 100 --random-erase 0.1 --weight-decay 0.00002 \
--norm-weight-decay 0.0 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 \
--train-crop-size 176 --val-resize-size 232 --ra-sampler --ra-reps 4
```