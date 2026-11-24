# Studi Komparatif Empiris Model Generatif (D6421 — Sesi 07)

Repositori eksperimen komparatif enam arsitektur model generatif pada benchmark **Fashion-MNIST** (28×28 grayscale, 70.000 citra, split terstandarisasi 80% Train / 10% Validation / 10% Test, `seed=42`):

1. **Vanilla Autoencoder**
2. **Sparse Autoencoder** (regularisasi L1 $\lambda \|h\|_1$)
3. **Variational Autoencoder (VAE)**
4. **Vanilla GAN** (berbasis MLP / Fully-Connected)
5. **Deep Convolutional GAN (DCGAN)**
6. **Denoising Diffusion Probabilistic Models (DDPM)**

Dilengkapi dengan eksperimen **Pembuktian Batasan Teoretis & Empiris MLP Classifier** untuk *image generation* (Tahap 1–5).

## Struktur Repositori

```text
DL-GANAssignment/
├── model.ipynb           # Notebook utama eksperimen, evaluasi metrik, dan 5 figur wajib
├── src/
│   ├── __init__.py
│   ├── models.py         # Implementasi modular 6 model generatif + MLP Classifier & Frozen Decoder
│   └── utils.py          # Standardisasi dataset, seed, metrik (MSE, PSNR, SSIM, FID), & latency
├── requirements.txt      # Dependensi Python/PyTorch
├── checkpoints/          # Bobot model terlatih (.pt) yang dihasilkan saat eksekusi
├── figures/              # 5 Figur visualisasi komparatif wajib (300 DPI)
└── results/              # Tabel komparasi metrik numerik (metrics_summary.csv)
```

## Cara Menjalankan

1. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Jalankan notebook**:
   Buka `model.ipynb` di Jupyter / VS Code dan jalankan sel secara berurutan (**Run All**).
   - Dataset **Fashion-MNIST** akan diunduh otomatis melalui `torchvision.datasets.FashionMNIST` ke folder `data/`.
   - Seluruh checkpoint disimpan di `checkpoints/`, tabel metrik di `results/metrics_summary.csv`, dan kelima figur wajib di `figures/`.
