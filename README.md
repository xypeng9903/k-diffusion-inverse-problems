# Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance

This repository contains the code and data associated with the paper [Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance](https://arxiv.org/abs/2402.02149).

This code is based on the 

- [K diffusion](https://github.com/crowsonkb/k-diffusion): Provide the code structure.

- [DPS](https://github.com/DPS2022/diffusion-posterior-sampling): Provide the code for degradation opterators.

- [DiffPIR](https://github.com/yuanzhi-zhu/DiffPIR): Provide tools for implementing closed-form solutions.

- [GPyTorch](https://github.com/cornellius-gp/gpytorch): Provide tools for implementing differentiable Gaussian likelihoods, enabling auto-computed Type I guidance (Use `--guidance autoI`).

___________
**Contents**
- [Abstract](#abstract)
- [Brief Introduction](#brief-introduction)
  - [Unified Intepretation of Diffusion-based Solvers to Inverse Problems](#unified-intepretation-of-diffusion-based-solvers-to-inverse-problems)
  - [Solving Inverse Problems With Optimal Posterior Covariance](#solving-inverse-problems-with-optimal-posterior-covariance)
- [Setting Up](#setting-up)
  - [Setup Conda Environment](#setup-conda-environment)
  - [Models and Analytic Variances](#models-and-analytic-variances)
  - [Reproduce Results](#reproduce-results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Abstract
Recent diffusion models provide a promising zero-shot solution to noisy linear inverse problems without retraining for specific inverse problems. In this paper, we reveal that recent methods can be uniformly interpreted as employing a Gaussian approximation with hand-crafted isotropic covariance for the intractable denoising posterior to approximate the conditional posterior mean. Inspired by this finding, we propose to improve recent methods by using more principled covariance determined by maximum likelihood estimation. To achieve posterior covariance optimization without retraining, we provide general plug-and-play solutions based on two approaches specifically designed for leveraging pre-trained models with and without reverse covariance. In addition, we propose a scalable method for learning posterior covariance prediction by leveraging widely-used orthonormal basis for image processing. Experimental results demonstrate that the proposed methods significantly enhance the overall performance and eliminate the need for hyperparameter tuning. 


## Brief Introduction

### Unified Intepretation of Diffusion-based Solvers to Inverse Problems
We provide unified intepretation of previous diffusion-based solvers to inverse problems from the view of approximating the conditional posterior mean $\mathbb{E}[{x}_0|{x}_t,{y}]$. Specifically, we classify them into two categories, Type I and Type II guidance, according to approximation paradigms, as elaborated below.

**Type I guidance.** We classify [DPS](https://arxiv.org/pdf/2209.14687.pdf) and [PiGDM](https://openreview.net/forum?id=9_gsMA8MRKQ) into one category, referred to as Type I guidance, where the conditional posterior mean $\mathbb{E}[{x}_0|{x}_t,{y}]$ is approximated based on the following relationship:

$$
\mathbb{E}[{x}_0|{x}_t,{y}] = \mathbb{E}[{x}_0|{x}_t] + s_t \sigma_t^2 \nabla_{{x}_t} \log p_t({y}|{x}_t)
$$

where $p_t({y}|{x}_t)$ is given by an intractable integral $\mathbb{E}_{p_t({x}_0|{x}_t)}[p({y}|{x}_0)]$. By introducing an isotrophic Gaussian approximation $q_t({x}_0|{x}_t)=\mathcal{N}(\mathbb{E}[{x}_0|{x}_t], r_t^2 I)$ for $p_t({x}_0|{x}_t)$, we can obtain the following approximation by Gaussian marginalization:

$$
p_t({y}|{x}_t) \approx \mathcal{N}({y}|{A}\mathbb{E}[{x}_0|{x}_t], \sigma^2 {I} + r_t^2 {A} {A}^T)
$$

**Type II guidance.** We classify [DiffPIR](https://arxiv.org/pdf/2305.08995.pdf) and [DDNM](https://arxiv.org/pdf/2212.00490.pdf) into the category of Type II guidance, which approximates $\mathbb{E}[{x}_0|{x}_t, y]$ with the solution of the following proximal problem:

$$
\mathbb{E}[{x}_0|{x}_t,{y}] \approx \arg\min_{{x}_0} \lVert {y} - {A} {x}_0 \rVert^2_2  + \frac{\sigma^2}{r_t^2} \lVert {x}_0 - \mathbb{E}[{x}_0|{x}_t] \rVert^2_2
$$

which can be intepreted as compute the mean of an approximate distribution $q_t({x}_0|{x}_t,{y}) \propto p({y}|{x}_0)q_t({x}_0|{x}_t)$  for the conditional posterior $p_t({x}_0|{x}_t,{y})\propto p({y}|{x}_0)p_t({x}_0|{x}_t)$.

### Solving Inverse Problems with Optimal Posterior Covariance

In our study, we generalize the above guidances based on variational Gaussian posterior with general covariance $q_t({x}_0|{x}_t)=\mathcal{N}(\mu_t({x}_t), \Sigma_t({x}_t))$, such that

**Type I guidance.** The likelihood is approximated in a similar way by Gaussian marginalization:

$$
p_t({y}|{x}_t) \approx \mathcal{N}({y}|{A}\mu_t({x}_t), \sigma^2 {I} + {A} \Sigma_t({x}_t) {A}^T)
$$

**Type II guidance.** $\mathbb{E}[{x}_0|{x}_t,{y}]$ is approximated with the solution of the following auto-weighted proximal problem:

$$
\mathbb{E}[{x}_0|{x}_t,{y}] \approx  \arg\min_{{x}_0} \lVert {y} - {A} {x}_0 \rVert^2  + \sigma^2 \lVert {x}_0 - \mu_t({x}_t) \rVert^2_{\Sigma_t^{-1}({x}_t)}
$$


## Setting Up
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

To run guidance based on ```DWT-Var``` covariance, download the FFHQ checkpoint ```ffhq_dwt.ckpt``` from the [link](https://drive.google.com/file/d/1ARbLbss9ByMOtF-7cl9_Yd2OupKk-72m/view?usp=drive_link), and paste it to ```../model_zoo```.


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
bash quick_start/eval_guidance_II.sh ffhq # for Convert, Analytic
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



