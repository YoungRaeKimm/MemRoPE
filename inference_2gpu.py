# 2-GPU Pipeline Parallel Inference for LongLive
# Uses accelerate dispatch_model to split CausalWanModel across 2 GPUs.
# Single process, no torchrun needed.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil

import imageio
import peft
import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.io import write_video
from tqdm import tqdm

from accelerate import dispatch_model

from pipeline import CausalInferencePipeline
from utils.dataset import TextDataset
from utils.lora_utils import configure_lora_for_model
from utils.memory import DynamicSwapInstaller
from utils.misc import set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--start_idx", type=int, default=0, help="Start prompt index (inclusive)")
parser.add_argument("--end_idx", type=int, default=-1, help="End prompt index (exclusive), -1 for all")
parser.add_argument("--profile", action="store_true", help="Enable per-block latency profiling (skip VAE decode)")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# ---------------------------------------------------------------------------
# Device setup: single process, 2 GPUs
# ---------------------------------------------------------------------------
assert torch.cuda.device_count() >= 2, (
    f"2-GPU pipeline parallel requires at least 2 GPUs, found {torch.cuda.device_count()}"
)
gpu0 = torch.device("cuda:0")
gpu1 = torch.device("cuda:1")

set_seed(config.seed)
torch.set_grad_enabled(False)
config.distributed = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def build_device_map(model, gpu0_id=0, gpu1_id=1):
    """Build device_map for CausalWanModel or PeftModel wrapper."""
    # Detect LoRA (PeftModel) wrapper
    is_lora = hasattr(model, 'base_model') and hasattr(model.base_model, 'model')
    prefix = "base_model.model." if is_lora else ""

    device_map = {
        f"{prefix}patch_embedding": gpu0_id,
        f"{prefix}text_embedding": gpu0_id,
        f"{prefix}time_embedding": gpu0_id,
        f"{prefix}time_projection": gpu0_id,
        f"{prefix}head": gpu1_id,
    }
    num_blocks = 30
    split = num_blocks // 2  # 15
    for i in range(num_blocks):
        device_map[f"{prefix}blocks.{i}"] = gpu0_id if i < split else gpu1_id
    return device_map


def get_block_devices(model, num_blocks=30):
    """After dispatch, determine each block's device from its parameters."""
    # Navigate through potential LoRA wrapper
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        blocks = model.base_model.model.blocks
    else:
        blocks = model.blocks

    devices = []
    for i in range(num_blocks):
        param = next(blocks[i].parameters())
        devices.append(param.device)
    return devices


# ---------------------------------------------------------------------------
# Initialize pipeline on CPU (weights loaded to CPU first)
# ---------------------------------------------------------------------------
print("Initializing pipeline on CPU...")
pipeline = CausalInferencePipeline(config, device=torch.device("cpu"))

# ---------------------------------------------------------------------------
# Load generator checkpoint
# ---------------------------------------------------------------------------
if config.generator_ckpt:
    print(f"Loading generator checkpoint: {config.generator_ckpt}")
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
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
    print(f"LoRA enabled with config: {config.adapter}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=True,
    )

    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        print("LoRA weights loaded for generator")

    pipeline.is_lora_enabled = True

# ---------------------------------------------------------------------------
# Convert to bf16 (still on CPU)
# ---------------------------------------------------------------------------
pipeline = pipeline.to(dtype=torch.bfloat16)

# ---------------------------------------------------------------------------
# Text encoder: DynamicSwap CPU offload (targets GPU 0)
# ---------------------------------------------------------------------------
DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu0)

# ---------------------------------------------------------------------------
# Dispatch generator model across 2 GPUs
# ---------------------------------------------------------------------------
inner_model = pipeline.generator.model
device_map = build_device_map(inner_model)
print("Dispatching generator across 2 GPUs with device_map:")
for k, v in sorted(device_map.items()):
    print(f"  {k} -> cuda:{v}")

dispatch_model(inner_model, device_map=device_map)

# dispatch_model's top-level hook moves ALL kwargs to cuda:0, which creates
# copies of kv_cache tensors on cuda:1, breaking in-place cache updates.
# Workaround: store caches as model attributes (not processed by hook).
# Get reference to the inner CausalWanModel for attribute storage.
if hasattr(inner_model, 'base_model') and hasattr(inner_model.base_model, 'model'):
    pipeline._causal_model_ref = inner_model.base_model.model
else:
    pipeline._causal_model_ref = inner_model
print("Cache will be stored as model attributes to bypass dispatch hooks")

# Determine which device each block ended up on
block_devices = get_block_devices(inner_model)
print(f"Block devices: {[str(d) for d in block_devices]}")

# Store on pipeline so KV cache init uses per-block devices
pipeline.block_devices = block_devices

# ---------------------------------------------------------------------------
# VAE on GPU 1 (head output lands on GPU 1, minimizes transfer)
# ---------------------------------------------------------------------------
pipeline.vae.to(device=gpu1)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=config.data_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

os.makedirs(config.output_folder, exist_ok=True)

# Save config alongside outputs for reproducibility
config_save_path = os.path.join(config.output_folder, "config.yaml")
if not os.path.exists(config_save_path):
    shutil.copy2(args.config_path, config_save_path)
    print(f"Config saved to {config_save_path}")

# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------
start_idx = args.start_idx
end_idx = args.end_idx if args.end_idx != -1 else num_prompts
print(f"Processing prompts [{start_idx}, {end_idx})")

for i, batch_data in tqdm(enumerate(dataloader)):
    idx = batch_data['idx'].item()

    if idx < start_idx:
        continue
    if idx >= end_idx:
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

    # Noise on GPU 0 (patch_embedding device)
    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104],
        device=gpu0, dtype=torch.bfloat16
    )

    print(f"[Prompt {idx}] {prompts[0][:80]}...")

    # Determine model type label for output filenames
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "lora"
    elif getattr(config, 'use_ema', False):
        model_type = "ema"
    else:
        model_type = "regular"

    if getattr(config, 'long_video_mode', False):
        # ---------------------------------------------------------------
        # Long video mode: skip VAE decode, then chunked decode + stream write
        # ---------------------------------------------------------------
        latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=False,
            skip_decode=True,
            low_memory=True,
            profile=args.profile,
            vae_device=gpu1,
        )

        # Chunked VAE decode + streaming write
        chunk_size = getattr(config, 'vae_chunk_size', 120)
        num_latent_frames = latents.shape[1]

        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'{idx}-{seed_idx}_{model_type}_2gpu.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')

            writer = imageio.get_writer(output_path, fps=16, codec='libx264', quality=8)
            print(f"  [long_video] Decoding {num_latent_frames} latent frames in chunks of {chunk_size}...")

            for start in range(0, num_latent_frames, chunk_size):
                end = min(start + chunk_size, num_latent_frames)
                chunk_latent = latents[seed_idx:seed_idx+1, start:end].to(gpu1)

                pipeline.vae.model.clear_cache()
                chunk_video = pipeline.vae.decode_to_pixel(chunk_latent, use_cache=False)
                chunk_video = (chunk_video * 0.5 + 0.5).clamp(0, 1)

                # Frame-by-frame write: [B, T, C, H, W] -> iterate T
                frames = (chunk_video[0] * 255).byte().cpu()  # [T, C, H, W]
                for frame in frames:
                    writer.append_data(frame.permute(1, 2, 0).numpy())  # [H, W, C]

                del chunk_latent, chunk_video, frames
                torch.cuda.empty_cache()

            writer.close()
            print(f"  Saved: {output_path}")

        del latents
        torch.cuda.empty_cache()

    else:
        # ---------------------------------------------------------------
        # Standard mode: decode all at once (original behavior)
        # ---------------------------------------------------------------
        if args.profile:
            pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=False,
                skip_decode=True,
                low_memory=True,
                profile=True,
                vae_device=gpu1,
            )
            continue
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=True,
            profile=False,
            vae_device=gpu1,
        )
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        video_out = 255.0 * current_video

        # Clear VAE cache
        pipeline.vae.model.clear_cache()

        # Save video
        if idx < num_prompts:
            for seed_idx in range(config.num_samples):
                if config.save_with_index:
                    output_path = os.path.join(config.output_folder, f'{idx}-{seed_idx}_{model_type}_2gpu.mp4')
                else:
                    output_path = os.path.join(config.output_folder, f'{prompt[:100]}-{seed_idx}.mp4')
                write_video(output_path, video_out[seed_idx], fps=16)
                print(f"  Saved: {output_path}")

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

print("Done!")
