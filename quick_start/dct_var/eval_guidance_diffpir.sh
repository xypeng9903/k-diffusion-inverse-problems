DATASET="ffhq"
CONFIG="configs/test_ffhq_dct.json"
CHECKPOINT="../model_zoo/ffhq_dct.pth"


for LAM in 1e-2 1e-1 1e0 1e1 1e2
do
    python sample_condition_v2.py \
    --guidance diffpir \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/inpainting_config.yaml \
    --logdir runs/sample_condition_v2/guidance_diffpir/${DATASET}/inpaint/lam_${LAM} \
    --lam ${LAM}

    python sample_condition_v2.py \
    --guidance diffpir \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/gaussian_deblur_config.yaml \
    --logdir runs/sample_condition_v2/guidance_diffpir/${DATASET}/gaussian_deblur/lam_${LAM} \
    --lam ${LAM}

    python sample_condition_v2.py \
    --guidance diffpir \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/motion_deblur_config.yaml \
    --logdir runs/sample_condition_v2/guidance_diffpir/${DATASET}/motion_deblur/lam_${LAM} \
    --lam ${LAM} 

    python sample_condition_v2.py \
    --guidance diffpir \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/super_resolution_4x_config.yaml \
    --logdir runs/sample_condition_v2/guidance_diffpir/${DATASET}/super_resolution/lam_${LAM} \
    --lam ${LAM}
done






