# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os

import peft
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video
from tqdm import tqdm

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.lora_utils import configure_lora_for_model
from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.misc import set_seed

# ---------------------------------------------------------------------------
# Reproducibility settings (must be set before any CUDA operations)
# ---------------------------------------------------------------------------
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to the config file")
parser.add_argument("--start_idx", type=int, default=0, help="Start index of prompts to process")
parser.add_argument("--end_idx", type=int, default=None, help="End index of prompts to process (exclusive)")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# ---------------------------------------------------------------------------
# Initialize distributed inference (if launched via torchrun)
# ---------------------------------------------------------------------------
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False
    print(f"Single GPU mode on device {device}")

low_memory = getattr(config, "low_memory", get_cuda_free_memory_gb(device) < 40)
torch.set_grad_enabled(False)

# For reproducibility across different hardware with the same seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# ---------------------------------------------------------------------------
# Initialize pipeline
# ---------------------------------------------------------------------------
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} missing parameters: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# ---------------------------------------------------------------------------
# LoRA support (optional)
# ---------------------------------------------------------------------------
pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

# ---------------------------------------------------------------------------
# Move pipeline to device
# ---------------------------------------------------------------------------
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# ---------------------------------------------------------------------------
# Dataset & dataloader
# ---------------------------------------------------------------------------
extended_prompt_path = getattr(config, "extended_prompt_path", config.data_path)
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
if local_rank == 0:
    print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# ---------------------------------------------------------------------------
# Noise shape from config (defaults: Wan2.1-T2V-1.3B at 480p)
# ---------------------------------------------------------------------------
latent_channels = getattr(config, "latent_channels", 16)
latent_h = getattr(config, "latent_h", 60)
latent_w = getattr(config, "latent_w", 104)

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # Skip if outside specified range
    if idx < args.start_idx:
        continue
    if args.end_idx is not None and idx >= args.end_idx:
        break

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    # Use CPU generator for cross-hardware reproducibility
    generator = torch.Generator(device='cpu').manual_seed(config.seed)
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, latent_channels, latent_h, latent_w],
        generator=generator, device='cpu', dtype=torch.bfloat16,
    ).to(device)

    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=False,
    )

    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    video_out = 255.0 * current_video

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()

    # Determine model type for filename
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "lora"
    elif getattr(config, 'use_ema', False):
        model_type = "ema"
    else:
        model_type = "regular"

    # Save video
    if idx < num_prompts:
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video_out[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()
