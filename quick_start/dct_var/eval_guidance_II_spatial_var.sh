MLE_SIGMA_THRES="$1"
DATASET="ffhq"
CONFIG="configs/test_ffhq_dct.json"
CHECKPOINT="../model_zoo/ffhq_dct.pth"


python sample_condition.py \
--guidance II \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/motion_deblur_config.yaml \
--logdir runs/sample_condition/guidance_II/${DATASET}/motion_deblur/spvar_mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES} \
--spatial-var

python sample_condition.py \
--guidance II \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/inpainting_config.yaml \
--logdir runs/sample_condition/guidance_II/${DATASET}/inpaint/spvar_mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES} \
--spatial-var

python sample_condition.py \
--guidance II \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/super_resolution_4x_config.yaml \
--logdir runs/sample_condition/guidance_II/${DATASET}/super_resolution/spvar_mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES} \
--spatial-var

python sample_condition.py \
--guidance II \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/gaussian_deblur_config.yaml \
--logdir runs/sample_condition/guidance_II/${DATASET}/gaussian_deblur/spvar_mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES} \
--spatial-var



