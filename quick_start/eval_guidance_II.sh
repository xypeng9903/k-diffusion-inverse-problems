COV=$1
GUIDANCE=II-$COV
CONFIG="configs/test_ffhq.json"
CHECKPOINT="../model_zoo/diffusion_ffhq_10m.pt"
GLOBAL_ARGS="--config $CONFIG --checkpoint $CHECKPOINT --guidance $GUIDANCE"


for LAM in 1e-2 1e-1 1e0 1e1 1e2
do
    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --operator-config configs/inpainting_config.yaml \
        --logdir runs/sample_condition_openai/ffhq/inpaint/$GUIDANCE/lambda_$LAM

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --operator-config configs/gaussian_deblur_config.yaml \
        --logdir runs/sample_condition_openai/ffhq/gaussian_deblur/$GUIDANCE/lambda_$LAM

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --operator-config configs/motion_deblur_config.yaml \
        --logdir runs/sample_condition_openai/ffhq/motion_deblur/$GUIDANCE/lambda_$LAM

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --operator-config configs/super_resolution_4x_config.yaml \
        --logdir runs/sample_condition/ffhq/super_resolution/$GUIDANCE/lambda_$LAM
done