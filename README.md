## Getting started 

### 1) Download pretrained checkpoint
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the FFHQ checkpoint "ffhq_10m.pt", rename to "diffusion_ffhq_10m.pt", and paste it to ../model_zoo

### 2) Setup conda envrionment
For creating the conda environment and installing dependencies run
```
conda env create -f environment.yml
```
Then activate the environment by
```
conda activate k-diffusion
```

### 3) Reproducing results

For reproducing results on FFHQ dataset in Table 3, run
```
bash quick_start/eval_guidance_I.sh ffhq
```

For reproducing results on FFHQ dataset in Figure 3, run
```
bash quick_start/eval_guidance_II.sh ffhq
```

For reproducing results on FFHQ dataset in Table 4, run
```
bash quick_start/eval_complete_dps.sh ffhq
bash quick_start/eval_complete_dps+mle.sh ffhq analytic
bash quick_start/eval_complete_dps+mle.sh ffhq convert
```

For reproducing results on FFHQ dataset in Figure 4, run
```
bash quick_start/eval_complete_pgdm.sh ffhq
bash quick_start/eval_complete_pgdm+mle.sh ffhq analytic
bash quick_start/eval_complete_pgdm+mle.sh ffhq convert
```