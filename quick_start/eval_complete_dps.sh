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

for ZETA in 1e0 1e1 1e2 1e3 1e4 1e5
do 
    python sample_condition_openai.py \
    --save-img \
    --ode \
    --guidance dps \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/gaussian_deblur_config.yaml \
    --logdir runs/sample_condition_openai/complete_dps/${DATASET}/gaussian_deblur/zeta_${ZETA} \
    --zeta ${ZETA}  

    python sample_condition_openai.py \
    --save-img \
    --ode \
    --guidance dps \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/motion_deblur_config.yaml \
    --logdir runs/sample_condition_openai/complete_dps/${DATASET}/motion_deblur/zeta_${ZETA} \
    --zeta ${ZETA} 

    python sample_condition_openai.py \
    --save-img \
    --ode \
    --guidance dps \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/inpainting_config.yaml \
    --logdir runs/sample_condition_openai/complete_dps/${DATASET}/inpaint/zeta_${ZETA} \
    --zeta ${ZETA} 

    python sample_condition_openai.py \
    --save-img \
    --ode \
    --guidance dps \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/super_resolution_4x_config.yaml \
    --logdir runs/sample_condition_openai/complete_dps/${DATASET}/super_resolution/zeta_${ZETA} \
    --zeta ${ZETA} 
done
