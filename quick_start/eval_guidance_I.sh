COV=$1
GUIDANCE=I-$COV
CONFIG="configs/test_ffhq.json"
CHECKPOINT="../model_zoo/diffusion_ffhq_10m.pt"
GLOBAL_ARGS="--config $CONFIG --checkpoint $CHECKPOINT --guidance $GUIDANCE --ode"


python sample_condition_openai.py \
    $GLOBAL_ARGS \
    --operator-config configs/inpainting_config.yaml \
    --logdir runs/sample_condition_openai/ffhq/inpaint/$GUIDANCE

python sample_condition_openai.py \
    $GLOBAL_ARGS \
    --operator-config configs/gaussian_deblur_config.yaml \
    --logdir runs/sample_condition_openai/ffhq/gaussian_deblur/$GUIDANCE

python sample_condition_openai.py \
    $GLOBAL_ARGS \
    --operator-config configs/motion_deblur_config.yaml \
    --logdir runs/sample_condition_openai/ffhq/motion_deblur/$GUIDANCE

python sample_condition_openai.py \
    $GLOBAL_ARGS \
    --operator-config configs/super_resolution_4x_config.yaml \
    --logdir runs/sample_condition_openai/ffhq/super_resolution/$GUIDANCE