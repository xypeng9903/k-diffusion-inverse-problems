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

python sample_condition_openai.py \
--save-img \
--ode \
--guidance pgdm \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/gaussian_deblur_config.yaml \
--logdir runs/sample_condition_openai/complete_pgdm/${DATASET}/gaussian_deblur

python sample_condition_openai.py \
--save-img \
--ode \
--guidance pgdm \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/motion_deblur_config.yaml \
--logdir runs/sample_condition_openai/complete_pgdm/${DATASET}/motion_deblur

python sample_condition_openai.py \
--save-img \
--ode \
--guidance pgdm \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/inpainting_config.yaml \
--logdir runs/sample_condition_openai/complete_pgdm/${DATASET}/inpaint

python sample_condition_openai.py \
--save-img \
--ode \
--guidance pgdm \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/super_resolution_4x_config.yaml \
--logdir runs/sample_condition_openai/complete_pgdm/${DATASET}/super_resolution

