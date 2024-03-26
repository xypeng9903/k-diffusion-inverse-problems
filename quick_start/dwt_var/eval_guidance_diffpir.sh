DATASET="ffhq"
CONFIG="configs/test_ffhq_dwt.json"
CHECKPOINT="../model_zoo/ffhq_dwt.ckpt"


python sample_condition_openai_v2.py \
--guidance diffpir \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/inpainting_config.yaml \
--logdir runs/sample_condition_openai_v2/guidance_diffpir/${DATASET}/inpaint/lam_1 \
--lam 1

python sample_condition_openai_v2.py \
--guidance diffpir \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/gaussian_deblur_config.yaml \
--logdir runs/sample_condition_openai_v2/guidance_diffpir/${DATASET}/gaussian_deblur/lam_1 \
--lam 1

python sample_condition_openai_v2.py \
--guidance diffpir \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/motion_deblur_config.yaml \
--logdir runs/sample_condition_openai_v2/guidance_diffpir/${DATASET}/motion_deblur/lam_10 \
--lam 10 

python sample_condition_openai_v2.py \
--guidance diffpir \
--save-img \
--config ${CONFIG} \
--checkpoint ${CHECKPOINT} \
--operator-config configs/super_resolution_4x_config.yaml \
--logdir runs/sample_condition_openai_v2/guidance_diffpir/${DATASET}/super_resolution/lam_10 \
--lam 10







