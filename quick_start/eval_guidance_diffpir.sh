DATASET="$1"

if [ "${DATASET}" = "ffhq" ]; then
    CONFIG="configs/test_ffhq.json"
    CHECKPOINT="../model_zoo/diffusion_ffhq_10m.pt"
elif [ "${DATASET}" = "imagenet" ]; then
    CONFIG="configs/test_imagenet.json"
    CHECKPOINT="../model_zoo/256x256_diffusion_uncond.pt"
else
    echo "Invalid dataset."
fi

GLOBAL_ARGS="--save-img --guidance diffpir --config ${CONFIG} --checkpoint ${CHECKPOINT}"


for LAM in 1e-2 1e-1 1e1 1e2
do
    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --lam ${LAM} \
        --operator-config configs/inpainting_config.yaml \
        --logdir runs/sample_condition_openai/guidance_diffpir/${DATASET}/inpaint/lam_${LAM}

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --lam ${LAM} \
        --operator-config configs/gaussian_deblur_config.yaml \
        --logdir runs/sample_condition_openai/guidance_diffpir/${DATASET}/gaussian_deblur/lam_${LAM}

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --lam ${LAM} \
        --operator-config configs/motion_deblur_config.yaml \
        --logdir runs/sample_condition_openai/guidance_diffpir/${DATASET}/motion_deblur/lam_${LAM}

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --lam ${LAM} \
        --operator-config configs/super_resolution_4x_config.yaml \
        --logdir runs/sample_condition_openai/guidance_diffpir/${DATASET}/super_resolution/lam_${LAM}
done