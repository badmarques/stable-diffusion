import argparse
import time
import torch
from torch import autocast
from pytorch_lightning import seed_everything
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts

parser = argparse.ArgumentParser()

parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)

parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="specify GPU (cuda/cuda:0/cuda:1/...)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--unet_bs",
    type=int,
    default=1,
    help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
)
parser.add_argument(
    "--turbo",
    action="store_true",
    default=True,
    help="Reduces inference time on the expense of 1GB VRAM",
)
parser.add_argument(
    "--precision", 
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
parser.add_argument(
    "--format",
    type=str,
    help="output image format",
    choices=["jpg", "png"],
    default="png",
)
parser.add_argument(
    "--sampler",
    type=str,
    help="sampler",
    choices=["ddim", "plms"],
    default="plms",
)

opt = parser.parse_args()

if opt.seed == None:
    opt.seed = randint(0, 1000000)
seed_everything(opt.seed)

class NeuralRender():

    def __init__(self, options) -> None:
        
        self.config_yaml = "optimizedSD/v1-inference.yaml"
        self.ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
        self.opt = options
        self.model, self.modelCS, self.modelFS, self.start_code = self.setup_model()


    def setup_model(self):
        sd = self._load_model_from_config(f"{self.ckpt}")
        li, lo = [], []
        for key, value in sd.items():
            sp = key.split(".")
            if (sp[0]) == "model":
                if "input_blocks" in sp:
                    li.append(key)
                elif "middle_block" in sp:
                    li.append(key)
                elif "time_embed" in sp:
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd["model1." + key[6:]] = sd.pop(key)
        for key in lo:
            sd["model2." + key[6:]] = sd.pop(key)

        config = OmegaConf.load(f"{self.config_yaml}")

        model = instantiate_from_config(config.modelUNet)
        _, _ = model.load_state_dict(sd, strict=False)
        model.eval()
        model.unet_bs = opt.unet_bs
        model.cdevice = opt.device
        model.turbo = opt.turbo

        modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = modelCS.load_state_dict(sd, strict=False)
        modelCS.eval()
        modelCS.cond_stage_model.device = opt.device

        modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = modelFS.load_state_dict(sd, strict=False)
        modelFS.eval()
        del sd

        if opt.device != "cpu" and opt.precision == "autocast":
            model.half()
            modelCS.half()

        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=opt.device)

        return model, modelCS, modelFS, start_code

    def _load_model_from_config(self, verbose=False):
        print(f"Loading model from {self.ckpt}")
        pl_sd = torch.load(self.ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        return sd


    def sample(self, prompt=None):
        seeds = ""

        with torch.no_grad():

            if not prompt:
                data = [self.opt.n_samples * [self.opt.prompt]]
            else:
                if isinstance(prompt, list):
                    data = prompt
                    #self.opt.n_samples = len(data)
                else:
                    data = [self.opt.n_samples * [prompt]]

            batch_size = self.opt.n_samples

            all_samples = list()
            for n in trange(self.opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):

                    with autocast("cuda"):
                        self.modelCS.to(opt.device)
                        uc = None
                        if opt.scale != 1.0:
                            uc = self.modelCS.get_learned_conditioning(opt.n_samples *  [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                       
                        subprompts, weights = split_weighted_subprompts(prompts[0])  
                        if len(subprompts) > 1: # Weighted prompts
                            c = torch.zeros_like(uc)
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(len(subprompts)):
                                weight = weights[i]
                                # if not skip_normalize:
                                weight = weight / totalWeight
                                c = torch.add(c, self.modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else:
                            c = self.modelCS.get_learned_conditioning(prompts)

                        shape = [self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f]

                        if self.opt.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelCS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)

                        samples_ddim = self.model.sample(
                            S=self.opt.ddim_steps,
                            conditioning=c,
                            seed=self.opt.seed,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=self.opt.scale,
                            unconditional_conditioning=uc,
                            eta=self.opt.ddim_eta,
                            #x_T=None,
                            x_T=self.start_code,
                            sampler = self.opt.sampler,
                        )

                        self.modelFS.to(self.opt.device)

                        print(samples_ddim.shape)


                        print("showing images")
                        for i in range(batch_size):

                            x_samples_ddim = self.modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                            x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                            img = Image.fromarray(x_sample.astype(np.uint8))

                            plt.imshow(img)
                            plt.show()

                            seeds += str(opt.seed) + ","
                            opt.seed += 1

                        if self.opt.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelFS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                        del samples_ddim
    


renderer = NeuralRender(opt)
#renderer.sample(["A cat sleeping on the computer", "A dog sleeping on the computer"])
renderer.sample("A cat sleeping on the computer")