#!/usr/bin/env python3

"""Samples from k-diffusion models."""

import argparse
import math

import accelerate
import torch
from tqdm import trange, tqdm

import k_diffusion as K
from condition.condition import ConditionOpenAIDenoiser
from condition.measurements import get_operator

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from condition.diffpir_utils import utils_model
from torch.utils import data
from torchvision import transforms

import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import os

from k_diffusion.external import OpenAIDenoiser


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_yaml(data: dict, file_path: str):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def compute_metrics(hat_x0, x0, loss_fn_vgg):
    def to_eval(x: torch.Tensor):
        return (x[0] / 2 + 0.5).clip(0, 1).detach()
    psnr = peak_signal_noise_ratio(to_eval(x0).cpu().numpy(), to_eval(hat_x0).cpu().numpy(), data_range=1).item()
    ssim = structural_similarity(to_eval(x0).cpu().numpy(), to_eval(hat_x0).cpu().numpy(), channel_axis=0, data_range=1).item()
    lpips = loss_fn_vgg(to_eval(x0), to_eval(hat_x0))[0, 0, 0, 0].item()
    return {
        'psnr': psnr, 
        'ssim': ssim, 
        'lpips': lpips
    }


def calculate_average_metric(metrics_list):
    avg_dict = {}
    count_dict = {}

    for metrics in metrics_list:
        for key, value in metrics.items():
            if key not in avg_dict:
                avg_dict[key] = 0.0
                count_dict[key] = 0
            avg_dict[key] += value
            count_dict[key] += 1

    for key in avg_dict:
        if count_dict[key] > 0:
            avg_dict[key] /= count_dict[key]

    return avg_dict


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=16,
                   help='the batch size')
    p.add_argument('--checkpoint', type=str, default="../model_zoo/diffusion_ffhq_10m.pt",
                   help='the checkpoint to use')
    p.add_argument('--config', type=str, default="configs/config_256x256_ffhq_train.json",
                   help='the model config')
    p.add_argument('-n', type=int, default=1,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--steps', type=int, default=1000,
                   help='the number of denoising steps')
    p.add_argument('--logdir', type=str, default=os.path.join("runs", "analytic_variance"))
    p.add_argument('--data-fraction', type=float, default=0.005)
    
    
    #-----------------------------------------
    # Setup unconditional model and test data
    #-----------------------------------------

    args = p.parse_args()
    config = K.config.load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    extra_args = utils_model.create_argparser(model_config['openai']).parse_args([])

    add_dict_to_argparser(p, args_to_dict(extra_args, model_and_diffusion_defaults().keys()))
    args = p.parse_args()

    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    inner_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    inner_model.load_state_dict(
        dist_util.load_state_dict(args.checkpoint, map_location="cpu")
    )
    inner_model = inner_model.eval().to(device)

    model = OpenAIDenoiser(inner_model, diffusion, device=device)

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']

    # train data
    tf = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x * 2 - 1
    ])
    train_set = K.utils.FolderOfImages(dataset_config['location'], transform=tf)
    train_len = round(len(train_set) * args.data_fraction)
    train_set, _ = data.random_split(train_set, [train_len, len(train_set) - train_len])
    train_dl = data.DataLoader(train_set, args.batch_size, drop_last=True, shuffle=True)

    #-----------------------------------------------------
    # Estimate optimal posterior variance via Monte Carlo
    #-----------------------------------------------------
    
    @torch.no_grad()
    def run():
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        save_yaml(vars(args), os.path.join(args.logdir, 'args.yaml'))

        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        mse_list = []
        
        errors = torch.zeros(len(sigmas), len(train_dl))
        for i, sigma in enumerate(tqdm(sigmas)):
            mse = 0
            for j, batch in enumerate(tqdm(train_dl)):
                x0, = batch
                x0 = x0.to(device)
                hat_x0 = model(x0 + torch.randn_like(x0) * sigma, sigma.repeat(args.batch_size))
                curr_mse = (x0 - hat_x0).pow(2).mean()
                errors[i, j] = curr_mse
                mse += curr_mse
            mse /= len(train_dl)
            mse_list.append(mse)

        mse_list = torch.stack(mse_list).cpu()
        torch.save(
            {'sigmas': sigmas.cpu(), 'mse_list': mse_list, 'errors': errors}, 
            os.path.join(args.logdir, 'recon_mse_test.pt')
        )


    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
