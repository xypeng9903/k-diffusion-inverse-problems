DATASET="$1"
MLE_SIGMA_THRES="$2"
COV=pgdm

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
--guidance I-vjp \
--xstart-cov-type ${COV} \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/gaussian_deblur_config.yaml \
--logdir runs/sample_condition_openai/guidance_I_vjp/${DATASET}/gaussian_deblur/${COV} \
--mle-sigma-thres ${MLE_SIGMA_THRES}

python sample_condition_openai.py \
--guidance I-vjp \
--xstart-cov-type ${COV} \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/motion_deblur_config.yaml \
--logdir runs/sample_condition_openai/guidance_I_vjp/${DATASET}/motion_deblur/${COV} \
--mle-sigma-thres ${MLE_SIGMA_THRES}

python sample_condition_openai.py \
--guidance I-vjp \
--xstart-cov-type ${COV} \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/inpainting_config.yaml \
--logdir runs/sample_condition_openai/guidance_I_vjp/${DATASET}/inpaint/${COV} \
--mle-sigma-thres ${MLE_SIGMA_THRES}

python sample_condition_openai.py \
--guidance I-vjp \
--xstart-cov-type ${COV} \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/super_resolution_4x_config.yaml \
--logdir runs/sample_condition_openai/guidance_I_vjp/${DATASET}/super_resolution/${COV} \
--mle-sigma-thres ${MLE_SIGMA_THRES}
