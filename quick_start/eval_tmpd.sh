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


python sample_condition_openai.py \
--guidance I \
--xstart-cov-type tmpd \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/gaussian_deblur_config.yaml \
--logdir runs/sample_condition_openai/guidance_I/${DATASET}/gaussian_deblur/tmpd \
--mle-sigma-thres 80

python sample_condition_openai.py \
--guidance I \
--xstart-cov-type tmpd \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/motion_deblur_config.yaml \
--logdir runs/sample_condition_openai/guidance_I/${DATASET}/motion_deblur/tmpd \
--mle-sigma-thres 80

python sample_condition_openai.py \
--guidance I \
--xstart-cov-type tmpd \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/inpainting_config.yaml \
--logdir runs/sample_condition_openai/guidance_I/${DATASET}/inpaint/tmpd \
--mle-sigma-thres 80

python sample_condition_openai.py \
--guidance I \
--xstart-cov-type tmpd \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/super_resolution_4x_config.yaml \
--logdir runs/sample_condition_openai/guidance_I/${DATASET}/super_resolution/tmpd \
--mle-sigma-thres 80