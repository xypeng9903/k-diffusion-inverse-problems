import lightning as L
import yaml
import argparse
import torch
from torchvision import transforms
from torch.utils import data
import os
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback
from copy import deepcopy

from guided_diffusion import dist_util
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict
)

from condition.diffpir_utils import utils_model
from k_diffusion.external import OpenAIDenoiserV2
import k_diffusion as K


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_yaml(data: dict, file_path: str):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/train_ffhq_dwt.json')
    p.add_argument('--openai-ckpt', type=str, default=None)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--accumulate-grad-batches', type=int, default=1)
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--num-workers', type=int, default=8)

    args = p.parse_args()
    config = load_yaml(args.config)

    train_config = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'openai_ckpt': args.openai_ckpt,
        'ema_sched': config['ema_sched']
    }

    if args.checkpoint is not None:
        model = OpenAIDenoiser.load_from_checkpoint(args.checkpoint, strict=False, train_config=train_config, model_config=config['model'])
    else:
        model = OpenAIDenoiser(config['model'], train_config)

    tf = transforms.Compose([
            transforms.Resize(config['model']['input_size'], interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(config['model']['input_size']),
            K.augmentation.KarrasAugmentationPipeline(config['model']['augment_prob']),
        ])
    train_set = K.utils.FolderOfImages(config['dataset']['location'], transform=tf)
    train_dl = data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    
    trainer = L.Trainer(
        logger=TensorBoardLogger('runs', f"{__file__[:-3]}"),
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[ExponentialMovingAverage()]
    )
    trainer.fit(model, train_dl)


class OpenAIDenoiser(L.LightningModule):

    def __init__(self, model_config, train_config):
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.train_config = train_config

        inner_model, diffusion = self._create_inner_model(train_config['openai_ckpt'])
        self.model = OpenAIDenoiserV2(inner_model, diffusion, ortho_tf_type=model_config['ortho_tf_type'])
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        self.ema_sched = K.utils.EMAWarmup(**train_config['ema_sched'])
    
    def training_step(self, batch, batch_idx):
        sample_density = K.config.make_sample_density(self.model_config)
        reals, _, _ = batch[0]
        noise = torch.randn_like(reals)
        sigma = sample_density([reals.shape[0]], device=self.device)
        loss = self.model.loss(reals, noise, sigma).mean()
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def on_train_epoch_start(self): 
        self.sample()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_config['lr'])
        return optimizer
    
    @torch.no_grad()
    def sample(self):
        c, (h, w) = self.model_config['input_channels'], self.model_config['input_size']
        sigma_min = float(self.model_config['sigma_min'])
        sigma_max = float(self.model_config['sigma_max'])
        sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=self.device)

        x = torch.randn(1, c, h, w, device=self.device) * self.model_config['sigma_max']
        x_0 = K.sampling.sample_dpmpp_2m(self.model_ema.eval(), x, sigmas)
        
        filename = os.path.join(self.logger.log_dir, f'step_{self.global_step}.png')
        K.utils.to_pil_image(x_0).save(filename)

    def _create_inner_model(self, checkpoint=None):
        args = utils_model.create_argparser(self.model_config['openai']).parse_args([])
        inner_model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        if checkpoint is not None:
            print(f'==> Loading OpenAI checkpoint {checkpoint}')
            inner_model.load_state_dict(
                dist_util.load_state_dict(checkpoint, map_location="cpu")
            )
        return inner_model, diffusion
    
    def ema_update(self):
        self.ema_sched.last_epoch = self.global_step
        ema_decay = self.ema_sched.get_value()
        K.utils.ema_update(self.model, self.model_ema, ema_decay)


class ExponentialMovingAverage(Callback):

    def on_train_batch_end(self, trainer: L.Trainer, model: OpenAIDenoiser, *args):
        model.ema_update()


if __name__ == "__main__":
    main()