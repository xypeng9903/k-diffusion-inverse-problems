MLE_SIGMA_THRES="$1"
DATASET="ffhq"
CONFIG="configs/test_ffhq_dct.json"
CHECKPOINT="../model_zoo/ffhq_dct.pth"


python sample_condition_v2.py \
--guidance I \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/gaussian_deblur_config.yaml \
--logdir runs/sample_condition_v2/guidance_I/${DATASET}/gaussian_deblur/mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES} 

python sample_condition_v2.py \
--guidance I \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/super_resolution_4x_config.yaml \
--logdir runs/sample_condition_v2/guidance_I/${DATASET}/super_resolution/mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES}

python sample_condition_v2.py \
--guidance I \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/inpainting_config.yaml \
--logdir runs/sample_condition_v2/guidance_I/${DATASET}/inpaint/mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES}

python sample_condition_v2.py \
--guidance I \
--ode \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/motion_deblur_config.yaml \
--logdir runs/sample_condition_v2/guidance_I/${DATASET}/motion_deblur/mle_sigma_thres_${MLE_SIGMA_THRES} \
--mle-sigma-thres ${MLE_SIGMA_THRES} 



