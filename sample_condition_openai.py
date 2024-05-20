#!/usr/bin/env python3

"""Samples from k-diffusion models."""

import argparse
import accelerate
import torch
from tqdm import tqdm
import yaml
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import os
from functools import partial
from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from torch.utils import data
from torchvision import transforms

import k_diffusion as K
from condition.diffpir_utils import utils_model
from condition.condition import ConditionOpenAIDenoiser
from condition.measurements import get_operator


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
    metrics = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips}
    print(metrics)
    return metrics


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
    p.add_argument('--batch-size', type=int, default=1,
                   help='the batch size')
    p.add_argument('--checkpoint', type=str, default="../model_zoo/diffusion_ffhq_10m.pt",
                   help='the checkpoint to use')
    p.add_argument('--config', type=str, default="configs/test_ffhq.json",
                   help='the model config')
    p.add_argument('--operator-config', type=str, default="configs/inpainting_config.yaml")
    p.add_argument('-n', type=int, default=1,
                   help='the number of images to sample')
    p.add_argument('--prefix', type=str, default='out',
                   help='the output prefix')
    p.add_argument('--logdir', type=str, default=os.path.join("runs", f"{__file__[:-3]}", "temp"))
    p.add_argument('--save-img', dest='save_img', action='store_true')
    
    # sampler
    p.add_argument('--steps', type=int, default=50, help='the number of denoising steps')
    p.add_argument('--ode', dest='ode', action='store_true')
    p.add_argument('--euler', dest='euler', action='store_true')
    
    # guidance
    p.add_argument('--guidance', type=str, default="I")
    p.add_argument('--xstart-cov-type', type=str, choices=["analytic", "convert", "pgdm", "dps", "diffpir", "tmpd"], default="convert")
    p.add_argument('--mle-sigma-thres', type=float, default=0.2)
    p.add_argument('--lam', type=float, default=None)
    p.add_argument('--zeta', type=float, default=None)
    p.add_argument('--num-hutchinson-samples', type=int, default=None)
    p.add_argument('--eta', type=float, default=None)



    #-----------------------------------------
    # Setup unconditional model and test data
    #-----------------------------------------

    args = p.parse_args()
    
    assert args.batch_size == 1
    
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
    # Setup conditioned model and do posterior sampling
    #---------------------------------------------------

    operator_config = load_yaml(args.operator_config)
    operator = get_operator(device=device, **operator_config)

    print(f"Operation: {operator_config['name']} / sigma_s: {operator_config['sigma_s']}")
    
    def run():
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        save_yaml(vars(args), os.path.join(args.logdir, 'args.yaml'))

        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
        sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
        recon_mse = torch.load(model_config['recon_mse'])
        metrics_list = []
        
        for i, batch in enumerate(tqdm(test_dl)):
            x0, = batch
            x0 = x0.to(device)
            measurement = operator.forward(x0.clone(), flatten=True)
            model = ConditionOpenAIDenoiser(
                inner_model=inner_model,
                diffusion=diffusion,
                operator=operator,
                measurement=measurement,
                guidance=args.guidance,
                x0_cov_type=args.xstart_cov_type,
                recon_mse=recon_mse,
                lambda_=args.lam,
                zeta=args.zeta,
                eta=args.eta,
                num_hutchinson_samples=args.num_hutchinson_samples,
                mle_sigma_thres=args.mle_sigma_thres,
                device=device
            ).eval()
                
            def sample_fn(n):
                x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
                sampler = partial(K.sampling.sample_heun if not args.euler else K.sampling.sample_euler,
                                  model, x, sigmas, disable=not accelerator.is_local_main_process)
                if not args.ode:
                    x_0 = sampler(s_churn=80, s_tmin=0.05, s_tmax=1, s_noise=1.007)
                else:
                    x_0 = sampler()     
                return x_0
            
            hat_x0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, args.n, args.batch_size)

            # quantitive results
            metrics = compute_metrics(hat_x0, x0, loss_fn_vgg)
            metrics_list.append(metrics)

            # qualitative results
            if args.save_img:
                measurement_filename = os.path.join(args.logdir, f"{args.prefix}_img_{i}_measurement.png")
                K.utils.to_pil_image(measurement[0]).save(measurement_filename)
                for j, out in enumerate(hat_x0):
                    hat_x0_filename = os.path.join(args.logdir, f"{args.prefix}_img_{i}_hat_x0_sample_{j}.png")
                    K.utils.to_pil_image(out).save(hat_x0_filename)

        avg_metrics = calculate_average_metric(metrics_list)
        print(avg_metrics)
        save_yaml(avg_metrics, os.path.join(args.logdir, 'avg_metrics.yaml'))

    try:
        run()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
