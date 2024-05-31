## Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance <br><sub>Official PyTorch implementation of the ICML 2024 paper</sub>

**Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance** <br>
Xinyu Peng, Ziyang Zheng, Wenrui Dai, Nuoqian Xiao, Chenglin Li, Junni Zou, Hongkai Xiong <br>
https://arxiv.org/abs/2402.02149

**Abstract:** *Recent diffusion models provide a promising zero-shot solution to noisy linear inverse problems without retraining for specific inverse problems. In this paper, we reveal that recent methods can be uniformly interpreted as employing a Gaussian approximation with hand-crafted isotropic covariance for the intractable denoising posterior to approximate the conditional posterior mean. Inspired by this finding, we propose to improve recent methods by using more principled covariance determined by maximum likelihood estimation. To achieve posterior covariance optimization without retraining, we provide general plug-and-play solutions based on two approaches specifically designed for leveraging pre-trained models with and without reverse covariance. In addition, we propose a scalable method for learning posterior covariance prediction by leveraging widely-used orthonormal basis for image processing. Experimental results demonstrate that the proposed methods significantly enhance the overall performance and eliminate the need for hyperparameter tuning.*





This code is based on: 

- [K diffusion](https://github.com/crowsonkb/k-diffusion): Provide the code structure.

- [DPS](https://github.com/DPS2022/diffusion-posterior-sampling): Provide the code for degradation opterators.

- [DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR): Provide tools for implementing closed-form solutions.

- [GPyTorch](https://github.com/cornellius-gp/gpytorch): Provide tools for implementing differentiable Gaussian likelihoods, enabling auto-computed Type I guidance (Use `--guidance autoI`).



## Getting Started
### Setup Conda Environment
For creating the conda environment and installing dependencies run
```
conda env create -f environment.yml
```
Then activate the environment by
```
conda activate k-diffusion
```

### Models and Analytic Variances
From the [link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing), download the FFHQ checkpoint ```ffhq_10m.pt```, rename to ```diffusion_ffhq_10m.pt```, and paste it to ```../model_zoo```.

To run guidance based on ```Analytic``` posterior covariance, download the precomputed Monte Carlo estimation from the [link](https://drive.google.com/drive/folders/1D93IZU0ViyExWm1k-L6dRehDHs1jAxGx?usp=drive_link), and paste it to ```./runs```.

To run guidance based on ```DWT-Var``` posterior covariance, download the FFHQ checkpoint ```ffhq_dwt.ckpt``` from the [link](https://drive.google.com/file/d/1ARbLbss9ByMOtF-7cl9_Yd2OupKk-72m/view?usp=drive_link), and paste it to ```../model_zoo```.


### Reproduce Results
From the [link](https://drive.google.com/file/d/1I8at4Y1MPrKV8yPHq_6sn6Et7Elyxavx/view?usp=drive_link), download the validation data (the first 100 images from [FFHQ](https://github.com/NVlabs/ffhq-dataset) and [ImageNet](https://image-net.org/) datasets), unzip and paste it to ```../data```.

For reproducing results on FFHQ dataset in Table 3, run
```bash
bash quick_start/eval_guidance_I.sh ffhq # for Convert, Analytic, TMPD, DPS, PiGDM
bash quick_start/dwt_var/eval_guidance_I.sh 1 # for DWT-Var
```

For reproducing results on FFHQ dataset in Figure 3, run
```bash
bash quick_start/eval_guidance_diffpir.sh ffhq # for DiffPIR
bash quick_start/eval_guidance_II.sh ffhq # for PiGDM, Convert, Analytic
bash quick_start/dwt_var/eval_guidance_II.sh 1 # for DWT-Var
```

For reproducing results on FFHQ dataset in Table 4, run
```bash
bash quick_start/eval_complete_pgdm+mle.sh ffhq convert # for Convert
bash quick_start/eval_complete_pgdm+mle.sh ffhq analytic # for Analytic
bash quick_start/eval_complete_pgdm.sh ffhq # for PiGDM
```

For reproducing results on FFHQ dataset in Figure 4, run
```bash
bash quick_start/eval_complete_dps+mle.sh ffhq convert # for Convert
bash quick_start/eval_complete_dps+mle.sh ffhq analytic # for Analytic
bash quick_start/eval_complete_dps.sh ffhq # for DPS
```


## Citation
If you find this repo helpful, please cite:

```bibtex
@misc{peng2024improving,
      title={Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance}, 
      author={Xinyu Peng and Ziyang Zheng and Wenrui Dai and Nuoqian Xiao and Chenglin Li and Junni Zou and Hongkai Xiong},
      year={2024},
      eprint={2402.02149},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



