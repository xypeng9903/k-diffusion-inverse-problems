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

GLOBAL_ARGS="--save-img --ode --guidance I --config ${CONFIG} --checkpoint ${CHECKPOINT}"


for COV in convert analytic tmpd dps pgdm
do
    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --xstart-cov-type ${COV} \
        --operator-config configs/inpainting_config.yaml \
        --logdir runs/sample_condition_openai/guidance_I/${DATASET}/inpaint/${COV}
    
    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --xstart-cov-type ${COV} \
        --operator-config configs/gaussian_deblur_config.yaml \
        --logdir runs/sample_condition_openai/guidance_I/${DATASET}/gaussian_deblur/${COV} 

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --xstart-cov-type ${COV} \
        --operator-config configs/motion_deblur_config.yaml \
        --logdir runs/sample_condition_openai/guidance_I/${DATASET}/motion_deblur/${COV} 

    python sample_condition_openai.py \
        $GLOBAL_ARGS \
        --xstart-cov-type ${COV} \
        --operator-config configs/super_resolution_4x_config.yaml \
        --logdir runs/sample_condition_openai/guidance_I/${DATASET}/super_resolution/${COV}
done