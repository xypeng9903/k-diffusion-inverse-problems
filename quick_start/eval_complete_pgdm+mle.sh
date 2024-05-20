DATASET="$1"
COV="$2"

if [ "${DATASET}" = "ffhq" ]; then
    CONFIG="configs/test_ffhq.json"
    CHECKPOINT="../model_zoo/diffusion_ffhq_10m.pt"
elif [ "${DATASET}" = "imagenet" ]; then
    CONFIG="configs/test_imagenet.json"
    CHECKPOINT="../model_zoo/256x256_diffusion_uncond.pt"
else
    echo "Invalid dataset."
fi

GLOBAL_ARGS="--save-img --ode --guidance pgdm+mle --xstart-cov-type ${COV} --config ${CONFIG} --checkpoint ${CHECKPOINT}"


python sample_condition_openai.py \
    $GLOBAL_ARGS \
    --operator-config configs/gaussian_deblur_config.yaml \
    --logdir runs/sample_condition_openai/pgdm+mle/${DATASET}/gaussian_deblur/${COV}

python sample_condition_openai.py \
    $GLOBAL_ARGS \
    --operator-config configs/motion_deblur_config.yaml \
    --logdir runs/sample_condition_openai/pgdm+mle/${DATASET}/motion_deblur/${COV} 

python sample_condition_openai.py \
    $GLOBAL_ARGS \
    --operator-config configs/inpainting_config.yaml \
    --logdir runs/sample_condition_openai/pgdm+mle/${DATASET}/inpaint/${COV}

python sample_condition_openai.py \
    $GLOBAL_ARGS \
    --operator-config configs/super_resolution_4x_config.yaml \
    --logdir runs/sample_condition_openai/pgdm+mle/${DATASET}/super_resolution/${COV}
