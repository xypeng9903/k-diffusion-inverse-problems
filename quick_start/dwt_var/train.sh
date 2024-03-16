accelerate launch train.py \
--config configs/train_ffhq_dwt_large.json \
--batch-size 8 \
--grad-accum-steps 4 \
--sample-n 1 \
--name runs/train/ffhq_dwt_large