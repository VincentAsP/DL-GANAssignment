"""
PyTorch model architectures for the Deep Learning Take-Home Assignment:
- MLPClassifier & FrozenLatentDecoder (Theoretical limitation proof)
- VanillaAutoencoder
- SparseAutoencoder (L1 sparsity penalty)
- VariationalAutoencoder (VAE)
- VanillaGAN (MLP Generator & Discriminator)
- DCGAN (Deep Convolutional Generator & Discriminator)
- DDPM (Denoising Diffusion Probabilistic Model with lightweight U-Net)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 0. MLP Classifier & Frozen Latent Decoder (Bagian 2: Pembuktian Batasan)
# ==============================================================================
class MLPClassifier(nn.Module):
    """MLP Classifier 784 -> 256 -> 64 -> 10 matching Autoencoder encoder dimensions."""

    def __init__(self, latent_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        z = F.relu(self.fc2(h1))
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))


class FrozenLatentDecoder(nn.Module):
    """Standalone MLP Decoder (64 -> 256 -> 784) trained on frozen z_MLP features."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


# ==============================================================================
# 1. Vanilla Autoencoder
# ==============================================================================
class VanillaAutoencoder(nn.Module):
    """Deterministic Vanilla Autoencoder (784 -> 256 -> latent_dim -> 256 -> 784)."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


# ==============================================================================
# 2. Sparse Autoencoder (L1 Sparsity Regularization)
# ==============================================================================
class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with L1 sparsity penalty on latent representation:
        L = MSE(x, x_hat) + lambda_l1 * mean(|h|)
    """

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.view(x.size(0), -1))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h


def sparse_ae_loss(x_hat: torch.Tensor, x: torch.Tensor, h: torch.Tensor, lambda_l1: float = 1e-3):
    recon = F.mse_loss(x_hat, x)
    sparsity = lambda_l1 * h.abs().mean()
    return recon + sparsity


# ==============================================================================
# 3. Variational Autoencoder (VAE)
# ==============================================================================
class VAE(nn.Module):
    """Convolutional/MLP Variational Autoencoder with Gaussian prior N(0, I)."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc_fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc_fc(x.view(x.size(0), -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
    # Sum over pixels, average over batch (standard ELBO formulation)
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return bce + beta * kld


# ==============================================================================
# 4. Vanilla GAN (MLP / Fully-Connected)
# ==============================================================================
class VanillaGenerator(nn.Module):
    """Fully-Connected Generator mapping z ~ N(0, I) -> 28x28 image in [-1, 1]."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)

    @torch.no_grad()
    def sample_01(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return (self.forward(z) + 1.0) / 2.0


class VanillaDiscriminator(nn.Module):
    """Fully-Connected Discriminator mapping 28x28 image in [-1, 1] -> real/fake logit."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# ==============================================================================
# 5. Deep Convolutional GAN (DCGAN)
# ==============================================================================
def dcgan_weights_init(m: nn.Module) -> None:
    """Radford et al. (2015) weight initialization for DCGAN stability."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class DCGANGenerator(nn.Module):
    """DCGAN Generator: z (100) -> 7x7x256 -> 14x14x128 -> 28x28x1 in [-1, 1]."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # [B, latent_dim, 1, 1] -> [B, 256, 7, 7]
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # [B, 256, 7, 7] -> [B, 128, 14, 14]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # [B, 128, 14, 14] -> [B, 1, 28, 28]
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(dcgan_weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z.view(z.size(0), self.latent_dim, 1, 1))

    @torch.no_grad()
    def sample_01(self, n: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(n, self.latent_dim, device=device)
        return (self.forward(z) + 1.0) / 2.0


class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator: 28x28x1 -> 14x14x64 -> 7x7x128 -> logit."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # [B, 1, 28, 28] -> [B, 64, 14, 14]
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 64, 14, 14] -> [B, 128, 7, 7]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # [B, 128, 7, 7] -> [B, 1, 1, 1]
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
        )
        self.apply(dcgan_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


# ==============================================================================
# 6. Denoising Diffusion Probabilistic Model (DDPM)
# ==============================================================================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.skip(x)


class SimpleUNet28(nn.Module):
    """Lightweight U-Net noise predictor eps_theta(x_t, t) for 28x28 grayscale images."""

    def __init__(self, base_ch: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Downsampling: 28 -> 14 -> 7
        self.down1 = ResBlock2D(1, base_ch, time_dim)
        self.pool1 = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.down2 = ResBlock2D(base_ch, base_ch * 2, time_dim)
        self.pool2 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        # Bottleneck: 7x7
        self.mid = ResBlock2D(base_ch * 2, base_ch * 2, time_dim)

        # Upsampling: 7 -> 14 -> 28
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock2D(base_ch * 4, base_ch, time_dim)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock2D(base_ch * 2, base_ch, time_dim)

        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        d1 = self.down1(x, t_emb)                 # [B, C, 28, 28]
        d2 = self.down2(self.pool1(d1), t_emb)    # [B, 2C, 14, 14]
        m = self.mid(self.pool2(d2), t_emb)       # [B, 2C, 7, 7]
        u2 = self.dec2(torch.cat([self.up2(m), d2], dim=1), t_emb)   # [B, C, 14, 14]
        u1 = self.dec1(torch.cat([self.up1(u2), d1], dim=1), t_emb)  # [B, C, 28, 28]
        return self.out_conv(u1)


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020)
    with linear variance schedule beta_t in [beta_start, beta_end].
    """

    def __init__(
        self,
        timesteps: int = 200,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        base_ch: int = 64,
    ):
        super().__init__()
        self.timesteps = timesteps
        self.model = SimpleUNet28(base_ch=base_ch)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps."""
        c1 = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        c2 = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return c1 * x0 + c2 * noise

    def p_losses(self, x0_neg1_1: torch.Tensor) -> torch.Tensor:
        """Simplified DDPM training objective: MSE(eps, eps_theta(x_t, t))."""
        b = x0_neg1_1.size(0)
        t = torch.randint(0, self.timesteps, (b,), device=x0_neg1_1.device).long()
        noise = torch.randn_like(x0_neg1_1)
        x_t = self.q_sample(x0_neg1_1, t, noise)
        pred_noise = self.model(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample_01(self, n: int, device: torch.device, return_trajectory: bool = False, capture_steps: int = 8):
        """
        Reverse denoising process x_T -> x_0 starting from x_T ~ N(0, I).
        Returns images mapped to [0, 1] (and optional trajectory list for Figur 5).
        """
        x = torch.randn(n, 1, 28, 28, device=device)
        trajectory = []
        capture_set = set(
            int(s) for s in torch.linspace(self.timesteps - 1, 0, capture_steps).round().tolist()
        )

        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=device, dtype=torch.long)
            pred_noise = self.model(x, t)

            beta_t = self.betas[i]
            sqrt_one_minus_ab = self.sqrt_one_minus_alphas_cumprod[i]
            sqrt_recip_a = self.sqrt_recip_alphas[i]

            mean = sqrt_recip_a * (x - (beta_t / sqrt_one_minus_ab) * pred_noise)
            if i > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean

            if return_trajectory and i in capture_set:
                trajectory.append((i, ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu()))

        x_01 = (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        if return_trajectory:
            return x_01, trajectory
        return x_01
