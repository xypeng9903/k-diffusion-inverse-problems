THRES=$1
GUIDANCE=II-$THRES
CONFIG="configs/test_ffhq_dwt.json"
CHECKPOINT="../model_zoo/ffhq_dwt.ckpt"
GLOBAL_ARGS="--config $CONFIG --checkpoint $CHECKPOINT --guidance $GUIDANCE"


python sample_condition.py \
    $GLOBAL_ARGS \
    --operator-config configs/inpainting_config.yaml \
    --logdir runs/sample_condition/ffhq/inpaint/$GUIDANCE

python sample_condition.py \
    $GLOBAL_ARGS \
    --operator-config configs/gaussian_deblur_config.yaml \
    --logdir runs/sample_condition/ffhq/gaussian_deblur/$GUIDANCE

python sample_condition.py \
    $GLOBAL_ARGS \
    --operator-config configs/motion_deblur_config.yaml \
    --logdir runs/sample_condition/ffhq/motion_deblur/$GUIDANCE

python sample_condition.py \
    $GLOBAL_ARGS \
    --operator-config configs/super_resolution_4x_config.yaml \
    --logdir runs/sample_condition/ffhq/super_resolution/$GUIDANCE