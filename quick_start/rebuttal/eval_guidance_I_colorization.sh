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

for COV in dps pgdm convert analytic 
do
    python sample_condition_openai.py \
    --guidance I \
    --xstart-cov-type ${COV} \
    --ode \
    --save-img \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT} \
    --operator-config configs/colorization_config.yaml \
    --logdir runs/sample_condition_openai/rebuttal/guidance_I/${DATASET}/colorization/${COV}
done