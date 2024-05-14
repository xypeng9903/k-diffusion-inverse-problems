## Introduction

This branch contains a unified implementation of Type I and Type II guidance discussed in the paper [Improving Diffusion Models for Inverse Problems Using Optimal Posterior Covariance](https://arxiv.org/abs/2402.02149). Specifically, suppose $\mu_t(\mathbf{x}_t)\approx \mathbb{E}[\mathbf{x}_0|\mathbf{x}_t]$ and $\Sigma^{(1)}_t(\mathbf{x}_t)$ and $\Sigma^{(2)}_t(\mathbf{x}_t)$ are two approximations for $\mathrm{Cov}[\mathbf{x}_0|\mathbf{x}_t]$, then the conditional posterior mean can be approximated by

$$
\mathbb{E}[\mathbf{x}_0|\mathbf{x}_t,\mathbf{y}] \approx \mu_t(\mathbf{x}_t) + \Sigma^{(1)}_t(\mathbf{x}_t) \mathbf{A}^T (\sigma^2 I + \mathbf{A} \Sigma^{(2)}_t(\mathbf{x}_t) \mathbf{A}^T)^{-1} (\mathbf{y} - \mathbf{A}\mu_t(\mathbf{x}_t)) \tag{1}
$$

Let $q_t(\mathbf{x}_0|\mathbf{x}_t)=\mathcal{N}(\mu_t(\mathbf{x}_t),\Sigma_t(\mathbf{x}_t))$. It is not hard to show that, when $\Sigma^{(2)}_t(\mathbf{x}_t) = \Sigma_t(\mathbf{x}_t)$ and $\Sigma^{(1)}_t(\mathbf{x}_t)$ is the TMPD covariance, i.e., $\Sigma^{(1)}_t(\mathbf{x}_t) = \sigma_t^2 \frac{\partial \mu_t(\mathbf{x}_t)}{\partial \mathbf{x}_t}$, Eq. (1) is equivalent to Type I guidance. When $\Sigma^{(1)}_t(\mathbf{x}_t) = \Sigma^{(2)}_t(\mathbf{x}_t) = \Sigma_t(\mathbf{x}_t)$, Eq. (1) is equivalent to Type II guidance (Applying Woodbury matrix identity to Eq. (121) in the paper).

## Quick Start
**Step1:** Download the FFHQ checkpoint ```ffhq_dwt.ckpt``` from the [link](https://drive.google.com/file/d/1ARbLbss9ByMOtF-7cl9_Yd2OupKk-72m/view?usp=drive_link) and paste to ```../model_zoo```.

**Step2:** Run quick start scripts for Type I or Type II guidance:

```bash
bash quick_start/eval_guidance_I.sh 1
bash quick_start/eval_guidance_II.sh 1
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



