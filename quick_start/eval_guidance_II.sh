DATASET="$1"

if [ "${DATASET}" = "ffhq" ]; then
    CONFIG="configs/config_256x256_ffhq.json"
    CHECKPOINT="../model_zoo/diffusion_ffhq_10m.pt"
elif [ "${DATASET}" = "imagenet" ]; then
    CONFIG="configs/config_256x256_imagenet.json"
    CHECKPOINT="../model_zoo/256x256_diffusion_uncond.pt"
else
    echo "Invalid dataset."
fi

for COV in diffpir analytic convert
do
    for LAM in 1e-2 1e-1 1e0 1e1 1e2
    do
        # python sample_condition_openai.py \
        # --guidance II \
        # --xstart-cov-type ${COV} \
        # --save-img \
        # --config ${CONFIG} \
        # --checkpoint ${CHECKPOINT} \
        # --operator-config configs/gaussian_deblur_config.yaml \
        # --logdir runs/sample_condition_openai/guidance_II/${DATASET}/gaussian_deblur/${COV}/lam_${LAM} \
        # --lam ${LAM} \

        # python sample_condition_openai.py \
        # --guidance II \
        # --xstart-cov-type ${COV} \
        # --save-img \
        # --config ${CONFIG} \
        # --checkpoint ${CHECKPOINT} \
        # --operator-config configs/motion_deblur_config.yaml \
        # --logdir runs/sample_condition_openai/guidance_II/${DATASET}/motion_deblur/${COV}/lam_${LAM} \
        # --lam ${LAM} \

        python sample_condition_openai.py \
        --guidance II \
        --xstart-cov-type ${COV} \
        --save-img \
        --config ${CONFIG} \
        --checkpoint ${CHECKPOINT} \
        --operator-config configs/inpainting_config.yaml \
        --logdir runs/sample_condition_openai/guidance_II/${DATASET}/inpaint/${COV}/lam_${LAM} \
        --lam ${LAM}

        python sample_condition_openai.py \
        --guidance II \
        --xstart-cov-type ${COV} \
        --save-img \
        --config ${CONFIG} \
        --checkpoint ${CHECKPOINT} \
        --operator-config configs/super_resolution_4x_config.yaml \
        --logdir runs/sample_condition_openai/guidance_II/${DATASET}/super_resolution/${COV}/lam_${LAM} \
        --lam ${LAM}
    done
done