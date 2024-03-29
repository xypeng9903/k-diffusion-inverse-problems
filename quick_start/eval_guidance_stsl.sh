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

for GAMMA in 0 1e-2 1e-1 1
do
    python sample_condition_openai.py \
    --guidance stsl \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/gaussian_deblur_config.yaml \
    --logdir runs/sample_condition_openai/guidance_stsl/${DATASET}/gaussian_deblur/${COV}/zeta_1e3_gamma_${GAMMA} \
    --zeta 1e3 \
    --gamma ${GAMMA} \
    --num-hutchinson-samples 1 \
    --ode

    # python sample_condition_openai.py \
    # --guidance stsl \
    # --save-img \
    # --config ${CONFIG} \
    # --checkpoint ${CHECKPOINT} \
    # --operator-config configs/motion_deblur_config.yaml \
    # --logdir runs/sample_condition_openai/guidance_stsl/${DATASET}/motion_deblur/${COV}/zeta_1e2_gamma_${GAMMA} \
    # --zeta 1e2 \
    # --num-hutchinson-samples 1 \
    # --gamma ${GAMMA} \
    # --ode

    # python sample_condition_openai.py \
    # --guidance stsl \
    # --save-img \
    # --config ${CONFIG} \
    # --checkpoint ${CHECKPOINT} \
    # --operator-config configs/inpainting_config.yaml \
    # --logdir runs/sample_condition_openai/guidance_stsl/${DATASET}/inpaint/${COV}/zeta_1e3_gamma_${GAMMA} \
    # --zeta 1e3 \
    # --gamma ${GAMMA} \
    # --num-hutchinson-samples 1 \
    # --ode

    # python sample_condition_openai.py \
    # --guidance stsl \
    # --save-img \
    # --config ${CONFIG} \
    # --checkpoint ${CHECKPOINT} \
    # --operator-config configs/super_resolution_4x_config.yaml \
    # --logdir runs/sample_condition_openai/guidance_stsl/${DATASET}/super_resolution/${COV}/zeta_1e3_gamma_${GAMMA} \
    # --zeta 1e3 \
    # --gamma ${GAMMA} \
    # --num-hutchinson-samples 1 \
    # --ode
done