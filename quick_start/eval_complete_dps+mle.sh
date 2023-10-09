DATASET="$1"
COV="$2"

if [ "${DATASET}" = "ffhq" ]; then
    CONFIG="configs/config_256x256_ffhq.json"
    CHECKPOINT="../model_zoo/diffusion_ffhq_10m.pt"
elif [ "${DATASET}" = "imagenet" ]; then
    CONFIG="configs/config_256x256_imagenet.json"
    CHECKPOINT="../model_zoo/256x256_diffusion_uncond.pt"
else
    echo "Invalid dataset."
fi

for ZETA in 1e0 1e1 1e2 1e3 1e4 1e5
do 
    python sample_condition_openai.py \
    --save-img \
    --ode \
    --guidance dps+mle \
    --xstart-cov-type ${COV} \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/gaussian_deblur_config.yaml \
    --logdir runs/sample_condition_openai/dps+mle/${DATASET}/gaussian_deblur/${COV}/zeta_${ZETA} \
    --zeta ${ZETA}  

    python sample_condition_openai.py \
    --save-img \
    --ode \
    --guidance dps+mle \
    --xstart-cov-type ${COV} \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/motion_deblur_config.yaml \
    --logdir runs/sample_condition_openai/dps+mle/${DATASET}/motion_deblur/${COV}/zeta_${ZETA} \
    --zeta ${ZETA} 

    python sample_condition_openai.py \
    --save-img \
    --ode \
    --guidance dps+mle \
    --xstart-cov-type ${COV} \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/inpainting_config.yaml \
    --logdir runs/sample_condition_openai/dps+mle/${DATASET}/inpaint/${COV}/zeta_${ZETA} \
    --zeta ${ZETA} 

    python sample_condition_openai.py \
    --save-img \
    --ode \
    --guidance dps+mle \
    --xstart-cov-type ${COV} \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/super_resolution_4x_config.yaml \
    --logdir runs/sample_condition_openai/dps+mle/${DATASET}/super_resolution/${COV}/zeta_${ZETA} \
    --zeta ${ZETA} 
done
