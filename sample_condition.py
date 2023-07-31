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
from condition.utils import utils_model
from torch.utils import data
from torchvision import transforms

import yaml
import matplotlib.pyplot as plt


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--batch-size', type=int, default=1,
                   help='the batch size')
    p.add_argument('--checkpoint', type=str, default="../model_zoo/diffusion_ffhq_10m.pt",
                   help='the checkpoint to use')
    p.add_argument('--config', type=str, default="configs/config_256x256_ffhq.json",
                   help='the model config')
    p.add_argument('--operator-config', type=str, default="configs/inpainting_config.yaml")
    p.add_argument('-n', type=int, default=1,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--steps', type=int, default=50,
                   help='the number of denoising steps')
    p.add_argument('--guidance', type=str, choices=["I", "II"], default="I")
    
    
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

    sigma_min = model_config['sigma_min']
    sigma_max = model_config['sigma_max']

    # test data
    tf = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x * 2 - 1
    ])
    test_set = K.utils.FolderOfImages(dataset_config['location'], transform=tf)
    test_dl = data.DataLoader(test_set, args.batch_size)


    #---------------------------------------------------
    # Setup conditional model and do posterior sampling
    #---------------------------------------------------

    operator_config = load_yaml(args.operator_config)
    operator = get_operator(device=device, **operator_config)

    print(f"Operation: {operator_config['name']} / sigma_s: {operator_config['sigma_s']}")
    
    def run():
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        for i, batch in enumerate(tqdm(test_dl)):
            x0, = batch
            x0 = x0.to(device)
            measurement = operator.forward(x0)
            model = ConditionOpenAIDenoiser(
                inner_model,
                diffusion,
                operator,
                measurement,
                args.guidance,
                device=device
            ).eval()

            def sample_fn(n):
                x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
                x_0 = K.sampling.sample_heun(model, x, sigmas, disable=not accelerator.is_local_main_process)
                return x_0
            hat_x0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, args.n, args.batch_size)

            # save results
            measurement_filename = f"{args.prefix}_img_{i}_measurement.png"
            K.utils.to_pil_image(measurement).save(measurement_filename)
            for j, out in enumerate(hat_x0):
                hat_x0_filename = f"{args.prefix}_img_{i}_hat_x0_sample_{j}.png"
                K.utils.to_pil_image(out).save(hat_x0_filename)
            
    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
