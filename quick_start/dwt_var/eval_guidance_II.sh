MLE_SIGMA_THRES="$1"
DATASET="ffhq"
CONFIG="configs/test_ffhq_dwt.json"
CHECKPOINT="../model_zoo/ffhq_dwt.ckpt"


python sample_condition_openai_v2.py \
--guidance II \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/inpainting_config.yaml \
--logdir runs/sample_condition_openai_v2/guidance_II/${DATASET}/inpaint/mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES}

python sample_condition_openai_v2.py \
--guidance II \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/gaussian_deblur_config.yaml \
--logdir runs/sample_condition_openai_v2/guidance_II/${DATASET}/gaussian_deblur/mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES} \

python sample_condition_openai_v2.py \
--guidance II \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/motion_deblur_config.yaml \
--logdir runs/sample_condition_openai_v2/guidance_II/${DATASET}/motion_deblur/mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES} 

python sample_condition_openai_v2.py \
--guidance II \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/super_resolution_4x_config.yaml \
--logdir runs/sample_condition_openai_v2/guidance_II/${DATASET}/super_resolution/mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES}






