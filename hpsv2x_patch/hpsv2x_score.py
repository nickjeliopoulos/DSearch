# img_score_cached.py
import os
import warnings
from typing import Union

import torch
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import huggingface_hub
from hpsv2.utils import root_path, hps_version_map

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Global caches / singletons
# ----------------------------
_model_cache = {
	# Always-on singletons
	"device": "cuda" if torch.cuda.is_available() else "cpu",
	"model": None,
	"preprocess_val": None,
	"tokenizer": None,

	# v2.0-only caching metadata
	"v2_checkpoint_path": None,          # str path to the v2.0 checkpoint once resolved
	"weights_loaded_for_v": None,        # tracks which hps_version weights are currently on the model
}

def _default_v2_ckpt_path() -> str:
	"""Default local path for HPS v2.0 as the original script used."""
	return os.path.join(root_path, "HPS_v2_compressed.pt")

def _ensure_dirs():
	if not os.path.exists(root_path):
		os.makedirs(root_path, exist_ok=True)

def _initialize_singletons():
	"""Create the model/tokenizer/preprocess once."""
	if _model_cache["model"] is None:
		model, _, preprocess_val = create_model_and_transforms(
			'ViT-H-14',
			'laion2B-s32B-b79K',
			precision='amp',
			device=_model_cache["device"],
			jit=False,
			force_quick_gelu=False,
			force_custom_text=False,
			force_patch_dropout=False,
			force_image_size=None,
			pretrained_image=False,
			image_mean=None,
			image_std=None,
			light_augmentation=True,
			aug_cfg={},
			output_dict=True,
			with_score_predictor=False,
			with_region_predictor=False
		)
		_model_cache["model"] = model.to(_model_cache["device"])
		_model_cache["model"].eval()
		_model_cache["preprocess_val"] = preprocess_val
		_model_cache["tokenizer"] = get_tokenizer('ViT-H-14')

def _resolve_checkpoint(hps_version: str, cp: str | None) -> str:
	"""
	Resolve a checkpoint path.
	For v2.0 only, prefer a local file if present to avoid HF traffic.
	"""
	_ensure_dirs()

	if hps_version == "v2.0":
		# If caller provided a path, use it directly.
		if cp:
			return cp

		# If we previously resolved and cached a v2.0 path, reuse it.
		if _model_cache["v2_checkpoint_path"] and os.path.exists(_model_cache["v2_checkpoint_path"]):
			return _model_cache["v2_checkpoint_path"]

		# Prefer default local path if it already exists (no HF call).
		local_default = _default_v2_ckpt_path()
		if os.path.exists(local_default):
			_model_cache["v2_checkpoint_path"] = local_default
			return local_default

		# Otherwise, download once and cache the resolved path.
		# Allow users to force no-network by exporting HPSV2_LOCAL_ONLY=1.
		local_only = os.environ.get("HPSV2_LOCAL_ONLY", "0") == "1"
		if local_only:
			raise FileNotFoundError(
				f"HPSV2_LOCAL_ONLY=1 set but checkpoint not found locally at: {local_default}"
			)

		resolved = huggingface_hub.hf_hub_download(
			"xswu/HPSv2",
			hps_version_map[hps_version],
		)
		_model_cache["v2_checkpoint_path"] = resolved
		return resolved

	# Non-v2.0 behavior: follow original logic (may trigger HF ping).
	if cp is not None:
		return cp
	return huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])

def _maybe_load_weights(hps_version: str, ckpt_path: str):
	"""
	Load weights only if the current model does not already carry weights for this version.
	For v2.0, we cache the fact that weights are already loaded to skip repeated torch.load + load_state_dict.
	For other versions, keep original behavior (load each time to preserve parity).
	"""
	model = _model_cache["model"]
	device = _model_cache["device"]

	if hps_version == "v2.0" and _model_cache["weights_loaded_for_v"] == "v2.0":
		# Already loaded v2.0 weights on the live model; skip.
		return

	# Load from disk and update the model.
	checkpoint = torch.load(ckpt_path, map_location=device)
	state_dict = checkpoint.get("state_dict", checkpoint)
	model.load_state_dict(state_dict)

	if hps_version == "v2.0":
		_model_cache["weights_loaded_for_v"] = "v2.0"
	else:
		# Don't “pin” non-v2.0 loads; match original semantics for other versions.
		_model_cache["weights_loaded_for_v"] = None

def score(
	img_path: Union[list, str, Image.Image],
	prompt: str,
	cp: str | None = None,
	hps_version: str = "v2.0"
) -> list:
	"""
	Compute HPS score(s) with aggressive checkpoint caching for hps_version='v2.0' only.
	Other versions retain the original 'load-each-call' behavior.
	"""
	_initialize_singletons()
	model = _model_cache["model"]
	preprocess_val = _model_cache["preprocess_val"]
	tokenizer = _model_cache["tokenizer"]
	device = _model_cache["device"]

	ckpt_path = _resolve_checkpoint(hps_version, cp)
	_maybe_load_weights(hps_version, ckpt_path)

	def _run_one(img):
		with torch.no_grad():
			if isinstance(img, str):
				image = preprocess_val(Image.open(img)).unsqueeze(0).to(device=device, non_blocking=True)
			elif isinstance(img, Image.Image):
				image = preprocess_val(img).unsqueeze(0).to(device=device, non_blocking=True)
			else:
				raise TypeError('The type of parameter img_path is illegal.')

			text = tokenizer([prompt]).to(device=device, non_blocking=True)
			with torch.amp.autocast("cuda"):
				outputs = model(image, text)
				image_features, text_features = outputs["image_features"], outputs["text_features"]
				logits_per_image = image_features @ text_features.T
				hps_score = torch.diagonal(logits_per_image).cpu().numpy()
			return hps_score[0]

	if isinstance(img_path, list):
		return [_run_one(p) for p in img_path]
	elif isinstance(img_path, (str, Image.Image)):
		return [_run_one(img_path)]
	else:
		raise TypeError('The type of parameter img_path is illegal.')

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
	parser.add_argument('--checkpoint', type=str, default=None,
						help='Path to the model checkpoint. If omitted and hps_version=v2.0, we try local default first.')
	parser.add_argument('--hps-version', type=str, default='v2.0',
						help='HPS version string, default v2.0. Caching only applies to v2.0.')
	args = parser.parse_args()

	# For v2.0: if no explicit --checkpoint, prefer local default path before any hub call.
	if args.checkpoint is None and args.hps_version == "v2.0":
		# If the default file exists locally, we'll stick to it without hub traffic.
		if os.path.exists(_default_v2_ckpt_path()):
			args.checkpoint = _default_v2_ckpt_path()

	# Create a black PIL image 224x224 and pass it to score via args.image_path
	image = Image.new('RGB', (224, 224), (0, 0, 0))
	scores = score(image, args.prompt, args.checkpoint, args.hps_version)
	print('HPSv2 score:', scores)
