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

GLOBAL_ARGS="--save-img --guidance II --config ${CONFIG} --checkpoint ${CHECKPOINT}"


for COV in analytic convert
do
    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --xstart-cov-type $COV \
        --operator-config configs/inpainting_config.yaml \
        --logdir runs/sample_condition_openai/guidance_II/${DATASET}/inpaint/$COV

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --xstart-cov-type $COV \
        --operator-config configs/gaussian_deblur_config.yaml \
        --logdir runs/sample_condition_openai/guidance_II/${DATASET}/gaussian_deblur/$COV

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --xstart-cov-type $COV \
        --operator-config configs/motion_deblur_config.yaml \
        --logdir runs/sample_condition_openai/guidance_II/${DATASET}/motion_deblur/$COV

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --xstart-cov-type $COV \
        --operator-config configs/super_resolution_4x_config.yaml \
        --logdir runs/sample_condition_openai/guidance_II/${DATASET}/super_resolution/$COV
done