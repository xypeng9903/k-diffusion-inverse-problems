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

for GAMMA in 1e-2 1e-1 1e0 1e1
do
    python sample_condition_openai.py \
    --guidance stsl+mle \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/motion_deblur_config.yaml \
    --logdir runs/sample_condition_openai/guidance_stsl+mle/${DATASET}/motion_deblur/${COV}/zeta_1e2_eta_${GAMMA} \
    --zeta 1e2 \
    --eta ${GAMMA} \
    --num-hutchinson-samples 1 \
    --ode

    python sample_condition_openai.py \
    --guidance stsl+mle \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/inpainting_config.yaml \
    --logdir runs/sample_condition_openai/guidance_stsl+mle/${DATASET}/inpaint/${COV}/zeta_1e3_eta_${GAMMA} \
    --zeta 1e3 \
    --eta ${GAMMA} \
    --num-hutchinson-samples 1 \
    --ode

    python sample_condition_openai.py \
    --guidance stsl+mle \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/gaussian_deblur_config.yaml \
    --logdir runs/sample_condition_openai/guidance_stsl+mle/${DATASET}/gaussian_deblur/${COV}/zeta_1e3_eta_${GAMMA} \
    --zeta 1e3 \
    --eta ${GAMMA} \
    --num-hutchinson-samples 1 \
    --ode

    python sample_condition_openai.py \
    --guidance stsl+mle \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/super_resolution_4x_config.yaml \
    --logdir runs/sample_condition_openai/guidance_stsl+mle/${DATASET}/super_resolution/${COV}/zeta_1e3_eta_${GAMMA} \
    --zeta 1e3 \
    --eta ${GAMMA} \
    --num-hutchinson-samples 1 \
    --ode
done