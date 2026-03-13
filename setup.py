from setuptools import setup, find_packages

setup(
    name="memrope",
    version="0.1.0",
    description="MemRoPE: Training-Free Long Video Generation via Memory-Augmented Rotary Position Embedding",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.0",
        "torchvision",
        "diffusers==0.31.0",
        "transformers>=4.49.0",
        "accelerate>=1.1.1",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "easydict",
        "ftfy",
        "omegaconf",
        "einops",
        "sentencepiece",
        "peft",
    ],
)
