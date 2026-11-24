"""
Utility functions for standardized dataset loading, reproducibility,
reconstruction metrics (MSE, PSNR, SSIM), generative FID evaluation,
and inference latency benchmarking.
"""

import gc
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_fashion_mnist_loaders(
    data_dir: str = "data",
    batch_size: int = 128,
    seed: int = 42,
):
    """
    Download and split Fashion-MNIST (70,000 images, 28x28 grayscale in [0, 1])
    into standardized Train (80%), Validation (10%), and Test (10%) DataLoaders.
    """
    tf = transforms.ToTensor()  # [0.0, 1.0]
    full_ds = ConcatDataset([
        datasets.FashionMNIST(data_dir, train=True, transform=tf, download=True),
        datasets.FashionMNIST(data_dir, train=False, transform=tf, download=True),
    ])

    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate_reconstruction(model_fn, dataloader, device: torch.device):
    """
    Evaluate reconstruction fidelity on a DataLoader.
    model_fn(x) must return reconstructed images in [0, 1] with shape [B, 1, 28, 28].
    Returns (test_mse, test_psnr_db, test_ssim).
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    total_mse = 0.0
    total_samples = 0

    for x, _ in dataloader:
        x = x.to(device)
        x_hat = model_fn(x).clamp(0.0, 1.0)
        batch_size = x.size(0)
        total_mse += ((x - x_hat) ** 2).mean().item() * batch_size
        total_samples += batch_size
        psnr_metric.update(x_hat, x)
        ssim_metric.update(x_hat, x)

    return (
        total_mse / total_samples,
        psnr_metric.compute().item(),
        ssim_metric.compute().item(),
    )


def _to_fid_uint8_rgb(imgs_01: torch.Tensor) -> torch.Tensor:
    """
    Convert [B, 1, H, W] float tensor in [0, 1] to [B, 3, H, W] uint8 tensor in [0, 255]
    as required by torchmetrics FrechetInceptionDistance.
    """
    imgs = (imgs_01.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    if imgs.size(1) == 1:
        imgs = imgs.repeat(1, 3, 1, 1)
    return imgs


def clear_gpu_memory(verbose: bool = True) -> None:
    """Free unused Python objects and release cached CUDA memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if verbose:
            print("Allocated:", torch.cuda.memory_allocated() / 1024**3, "GB")
            print("Reserved:", torch.cuda.memory_reserved() / 1024**3, "GB")


@torch.no_grad()
def evaluate_fid(
    sample_fn,
    test_loader,
    device: torch.device,
    num_samples: int = 2000,
    batch_size: int = 128,
) -> float:
    """
    Compute Fréchet Inception Distance (FID) on CPU between real Test Set images and
    synthetic images produced by sample_fn(n) -> tensor [n, 1, 28, 28] in [0, 1].
    Running InceptionV3 on CPU prevents CUDA OOM on low-VRAM GPUs.
    """
    clear_gpu_memory(verbose=False)
    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=False,
        compute_on_cpu=True,
    ).cpu()

    # Real images from test set (kept on CPU for FID)
    real_seen = 0
    for x, _ in test_loader:
        take = min(x.size(0), num_samples - real_seen)
        if take <= 0:
            break
        fid.update(_to_fid_uint8_rgb(x[:take].cpu()), real=True)
        real_seen += take

    # Generated images from prior (sampled on device, immediately moved to CPU)
    fake_seen = 0
    while fake_seen < num_samples:
        cur = min(batch_size, num_samples - fake_seen)
        fake_imgs = sample_fn(cur).detach().cpu()
        fid.update(_to_fid_uint8_rgb(fake_imgs), real=False)
        del fake_imgs
        fake_seen += cur

    score = float(fid.compute().item())
    del fid
    clear_gpu_memory(verbose=False)
    return score


@torch.no_grad()
def measure_latency_ms(fn, device: torch.device, warmup: int = 5, runs: int = 30) -> float:
    """
    Measure average inference latency in milliseconds for fn() (generating 1 batch of 64 samples).
    """
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / runs * 1000.0
