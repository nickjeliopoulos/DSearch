import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import wandb
import argparse
from datetime import datetime
import PIL
from diffusers import DDIMScheduler
from PIL import Image
from typing import *
from tqdm import tqdm
import json

from sd_pipeline import DPS_continuous_SDPipeline, Decoding_nonbatch_SDPipeline
from compressibility_scorer import CompressibilityScorerDiff, jpeg_compressibility, CompressibilityScorer_modified
from aesthetic_scorer import AestheticScorerDiff, hpsScorer, AestheticScorerDiff_Time, MLPDiff

def measure_torch_device_memory_used_mb(device: torch.device) -> float:
	"""
	Memory usage query - used in wandb logging.
	"""
	if device.type == "cuda":
		free, total = torch.cuda.mem_get_info(device)
		return (total - free) / 1024**2
	else:
		return -1.0

def wandb_log(
	step: int, 
	image, 
	max_fitness, 
	mean_fitness, 
	min_fitness, 
	std_fitness,
	prompt: str, 
	running_time: float, 
	device: torch.device
):
	wandb.log(
		{
			"step": step,
			"pop_best_eval": max_fitness,
			"mean_eval": mean_fitness,
			"min_eval": min_fitness,
			"std_eval": std_fitness,
			"best_img": wandb.Image(image),
			"prompt": prompt,
			"running_time": running_time,
			"memory": measure_torch_device_memory_used_mb(device),
		}
	)

def load_geneval_metadata(prompt_path: str, max_prompts: int = None):
	if prompt_path.endswith(".json"):
		with open(prompt_path, "r") as f:
			data = json.load(f)
	else:
		assert prompt_path.endswith(".jsonl")
		with open(prompt_path, "r") as f:
			data = [json.loads(line) for line in f]

	if "prompt" not in data[0]:
		assert "text" in data[0], "Prompt data should have 'prompt' or 'text' key"
		for item in data:
			item["prompt"] = item["text"]

	if max_prompts is not None:
		data = data[:max_prompts]

	return data

def parse_args():
	parser = argparse.ArgumentParser(description="Inference")
	parser.add_argument("--reward", type=str, default='aesthetic')
	parser.add_argument("--bs", type=int, default=2)
	parser.add_argument("--oversamplerate", type=int, default=1)
	parser.add_argument("--w", type=float, default=1)
	parser.add_argument("--search_schudule", type=str, default="all")
	parser.add_argument("--drop_schudule", type=str, default=None)
	parser.add_argument("--replacerate", type=float, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--duplicate_size",type=int, default=20)  
	parser.add_argument("--variant", type=str, default="PM")
	parser.add_argument("--valuefunction", type=str, default="")
	parser.add_argument("--timesteps", type=int, default=50)
	parser.add_argument("--prompt_path", type=str, default="open_img_pref_sampled_60.jsonl")
	args = parser.parse_args()
	return args

def main(args: argparse.Namespace):
	### Assert!
	assert args.oversamplerate == 1 or args.replacerate == 0.0
	assert args.variant in ['PM', 'MC']

	wandb.init(
		project="inference-dsearch",
		config=vars(args)
	)

	### seed everything
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	DDES_type = None
	if args.oversamplerate > 1:
		DDES_type = 'DDES_E'
	elif args.replacerate > 0:
		DDES_type = 'DDES_R'
	else:
		DDES_type = 'SVDD'

	### Initialization
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	device = torch.device(device)

	pipe = Decoding_nonbatch_SDPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
	pipe.to(device)

	# switch to DDIM scheduler
	pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
	pipe.scheduler.set_timesteps(args.timesteps, device=device)

	# pipe.vae.requires_grad_(False)
	# pipe.text_encoder.requires_grad_(False)
	# pipe.unet.requires_grad_(False)

	# pipe.vae.eval()
	# pipe.text_encoder.eval()
	# pipe.unet.eval()

	if args.reward == 'compressibility':
		if args.variant == 'PM':
			scorer = CompressibilityScorer_modified(dtype=torch.float32)#.to(device)
		elif args.variant == 'MC':
			scorer = CompressibilityScorerDiff(dtype=torch.float32).to(device)
	elif args.reward == 'aesthetic':
		if args.variant == 'PM':
			scorer = AestheticScorerDiff(dtype=torch.float32).to(device)
		elif args.variant == 'MC':
			scorer = AestheticScorerDiff_Time(dtype=torch.float32).to(device)
			if args.valuefunction != "":
				scorer.set_valuefunction(args.valuefunction)
				scorer = scorer.to(device)
	elif args.reward == 'hps':
		if args.variant == 'PM':
			scorer = hpsScorer(inference_dtype=torch.float32, device=device).to(device)
		else:
			raise ValueError("Invalid variant")
	else:
		raise ValueError("Invalid reward")

	### Scorer business
	scorer.requires_grad_(False)
	scorer.eval()

	pipe.setup_scorer(scorer)
	pipe.set_variant(args.variant)
	pipe.set_reward(args.reward)
	pipe.set_parameters(args.bs, args.duplicate_size, args.w, args.search_schudule, args.drop_schudule, args.oversamplerate, args.replacerate)

	### Load prompt data
	prompt_data = load_geneval_metadata(args.prompt_path)

	progress_bar = enumerate(prompt_data)
	# progress_bar = tqdm(prompt_data)

	for prompt_idx, item in progress_bar:
		prompt = [item["prompt"]] * args.bs
		start_time = datetime.now()
		
		images, kl_losses, cur_prompt = pipe(prompt, num_images_per_prompt=1, eta=1.0)

		end_time = datetime.now()

		### Running time in seconds
		running_time = (end_time - start_time).total_seconds()

		### Evaluate each generated image with the scorer and compute statistics
		rewards = scorer(images)

		### Get indices that sort by reward (descending)
		reward_ranked_indices = np.argsort(rewards)[::-1]
		
		images = [images[i] for i in reward_ranked_indices]
		ranked_rewards = rewards[reward_ranked_indices]

		# Compute summary statistics
		mean = ranked_rewards.mean()
		maxv = ranked_rewards.max()
		minv = ranked_rewards.min()
		stdv = ranked_rewards.std()

		### INFO!
		print(f"Prompt {item["prompt"][:12]}... Stats: {args.reward:<5} | mean: {mean:8.4f} | max: {maxv:8.4f} | min: {minv:8.4f} | std: {stdv:8.4f}, took {running_time:.3f}s")

		### Evaluate!

		### Loggit
		wandb_log(
			step=1,
			image=images[0],
			max_fitness=maxv,
			mean_fitness=mean,
			min_fitness=minv,
			std_fitness=stdv,
			prompt=item["prompt"],
			running_time=running_time,
			device=device,
		)
	

if __name__ == "__main__":
	args = parse_args()
	main(args)