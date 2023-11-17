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

for COV in diffpir analytic
do
    for LAM in 1e-2 1e-1 1e0 1e1 1e2
    do
        python sample_condition_openai.py \
        --guidance II \
        --xstart-cov-type ${COV} \
        --save-img \
        --config ${CONFIG} \
        --checkpoint ${CHECKPOINT} \
        --operator-config configs/colorization_config.yaml \
        --logdir runs/sample_condition_openai/rebuttal/guidance_II/${DATASET}/colorization/${COV}/lam_${LAM} \
        --lam ${LAM} 
    done
done