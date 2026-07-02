# 📚 Chapter 17: Autoencoders, GANs, and Diffusion Models
### Complete Study Notes — Professor Level

> **All pages analyzed. All concepts covered. Zero shortcuts.**

---

## 🖼️ Visual Gallery (Python-Generated Graphs)

> All visuals are in the [`Visuals/`](Visuals/) folder and embedded in each module.
> Re-generate anytime: `python3 generate_visuals.py`

| # | Graph | Module | File |
|---|-------|--------|------|
| 01 | Autoencoder Architecture: Encoder → Bottleneck → Decoder | 1 | [01_autoencoder_architecture.png](Visuals/01_autoencoder_architecture.png) |
| 02 | AE Reconstruction Loss: Train vs Validation Curve | 1 | [02_reconstruction_loss_curve.png](Visuals/02_reconstruction_loss_curve.png) |
| 03 | Undercomplete vs Overcomplete AE (funnel diagrams) | 1 | [03_undercomplete_overcomplete.png](Visuals/03_undercomplete_overcomplete.png) |
| 04 | PCA (Linear) vs AE (Non-Linear) Latent Space — Swiss Roll | 1 | [04_pca_vs_ae_latent.png](Visuals/04_pca_vs_ae_latent.png) |
| 05 | Sparse AE Activation Histogram vs Standard AE | 2 | [05_sparse_activation_histogram.png](Visuals/05_sparse_activation_histogram.png) |
| 06 | KL-Divergence Sparsity Penalty vs L1 Sparsity Curves | 2 | [06_kl_sparsity_penalty.png](Visuals/06_kl_sparsity_penalty.png) |
| 07 | Denoising AE: Original → Noisy (σ=0.25/0.55) → Reconstruction | 2 | [07_denoising_ae_pipeline.png](Visuals/07_denoising_ae_pipeline.png) |
| 08 | Anomaly Detection via Reconstruction Error Distribution | 2 | [08_anomaly_detection.png](Visuals/08_anomaly_detection.png) |
| 09 | Reparameterization Trick: Blocked vs Flowing Gradients | 3 | [09_reparameterization_trick.png](Visuals/09_reparameterization_trick.png) |
| 10 | KL Divergence: Encoder Posterior vs Prior N(0,1) | 3 | [10_kl_divergence_prior.png](Visuals/10_kl_divergence_prior.png) |
| 11 | VAE ELBO Loss Decomposition Over Training | 3 | [11_vae_elbo_loss.png](Visuals/11_vae_elbo_loss.png) |
| 12 | VAE Latent Manifold: Encoder Means per Digit Class | 3 | [12_vae_latent_manifold.png](Visuals/12_vae_latent_manifold.png) |
| 13 | VAE Latent Space Interpolation: Digit 1 → Digit 0 | 3 | [13_vae_interpolation.png](Visuals/13_vae_interpolation.png) |
| 14 | GAN Min-Max Adversarial Loop Diagram | 4 | [14_gan_minimax_game.png](Visuals/14_gan_minimax_game.png) |
| 15 | GAN Training Loss + Discriminator Accuracy Over Epochs | 4 | [15_gan_training_loss.png](Visuals/15_gan_training_loss.png) |
| 16 | GAN Mode Collapse: Diverse vs Collapsed Generator | 4 | [16_mode_collapse.png](Visuals/16_mode_collapse.png) |
| 17 | DCGAN Generator Architecture with Layer Dimensions | 5 | [17_dcgan_architecture.png](Visuals/17_dcgan_architecture.png) |
| 18 | Transposed Convolution: Zero-Insert → Convolve (Step-by-Step) | 5 | [18_transposed_convolution.png](Visuals/18_transposed_convolution.png) |
| 19 | WGAN vs Standard GAN: JS vs Wasserstein Distance | 5 | [19_wgan_vs_standard.png](Visuals/19_wgan_vs_standard.png) |
| 20 | Noise Schedule: Linear vs Cosine (beta_t, alpha_bar_t) | 6 | [20_noise_schedule.png](Visuals/20_noise_schedule.png) |
| 21 | Diffusion Forward Process: Clean → Pure Noise (with SNR) | 6 | [21_diffusion_forward_process.png](Visuals/21_diffusion_forward_process.png) |
| 22 | Diffusion Reverse Process: Noise → Generated Image | 6 | [22_diffusion_reverse_process.png](Visuals/22_diffusion_reverse_process.png) |
| 23 | U-Net Architecture: Encoder + Skip Connections + Decoder | 6 | [23_unet_architecture.png](Visuals/23_unet_architecture.png) |
| 24 | Classifier-Free Guidance: Quality vs Diversity Trade-off | 6 | [24_cfg_guidance_scale.png](Visuals/24_cfg_guidance_scale.png) |

---

## 🗺️ Master Index

| Module | Topic | File | Pages Covered |
|--------|-------|------|---------------|
| 01 | Efficient Data Representations & Basic Autoencoders | [01_Basic_Autoencoders.md](Detailed_Notes/01_Basic_Autoencoders.md) | pp. 580–598 |
| 02 | Sparse & Denoising Autoencoders | [02_Sparse_and_Denoising_Autoencoders.md](Detailed_Notes/02_Sparse_and_Denoising_Autoencoders.md) | pp. 599–614 |
| 03 | Variational Autoencoders (VAEs) | [03_Variational_Autoencoders.md](Detailed_Notes/03_Variational_Autoencoders.md) | pp. 615–633 |
| 04 | Generative Adversarial Networks (GANs) | [04_Generative_Adversarial_Networks.md](Detailed_Notes/04_Generative_Adversarial_Networks.md) | pp. 634–655 |
| 05 | Deep Convolutional GANs & GAN Variants | [05_DCGAN_and_GAN_Variants.md](Detailed_Notes/05_DCGAN_and_GAN_Variants.md) | pp. 656–675 |
| 06 | Diffusion Models | [06_Diffusion_Models.md](Detailed_Notes/06_Diffusion_Models.md) | pp. 676–695 |

---

## ⚡ One-Page Chapter Summary

### The Timeline / Core Story
```
Dimensionality Reduction (PCA) → Autoencoders (learned non-linear compression)
        ↓
Sparse AE (feature detectors) → Denoising AE (robust representations)
        ↓
Variational AE (probabilistic latent space → generation via sampling)
        ↓
GANs (adversarial game: Generator vs Discriminator → photorealistic synthesis)
        ↓
DCGAN → Progressive GAN → StyleGAN → Conditional GAN (cGAN) → WGAN
        ↓
Diffusion Models (iterative denoising → SOTA image/audio/video generation)
```

### Core Architecture / Math
```
AUTOENCODER:
  Input x → [Encoder f(x)] → z (latent code) → [Decoder g(z)] → x̂
  Loss: ||x - x̂||²  (MSE) or BCE

VAE:
  x → Encoder → μ, σ² → z = μ + ε·σ (reparameterization, ε~N(0,1))
  Loss: Reconstruction Loss + KL Divergence (ELBO)
  KL  = -0.5 * Σ(1 + log σ² - σ² - μ²)

GAN:
  Noise z → Generator G(z) → fake image
  Real/Fake → Discriminator D(x) → probability [0,1]
  Min-Max: min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]

DIFFUSION (DDPM):
  Forward: x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε,    ε~N(0,I)
  Objective: MSE(ε, ε_θ(x_t, t))             predict the noise!
  Reverse:  x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t)·ε_θ) + √β_t·z
```

### Core Code Snippet
```python
# ── Variational Autoencoder (VAE) — Keras Functional API ──
import tensorflow as tf
from tensorflow import keras

# Reparameterization Trick Layer
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean

# Encoder
inputs = keras.Input(shape=[28, 28])
z = keras.layers.Flatten()(inputs)
z = keras.layers.Dense(150, activation="selu")(z)
z = keras.layers.Dense(100, activation="selu")(z)
codings_mean    = keras.layers.Dense(10)(z)   # mu  (NO activation)
codings_log_var = keras.layers.Dense(10)(z)   # log sigma^2  (NO activation)
codings = Sampling()([codings_mean, codings_log_var])
encoder = keras.Model(inputs, [codings_mean, codings_log_var, codings])

# Decoder
dec_in = keras.Input(shape=[10])
x = keras.layers.Dense(100, activation="selu")(dec_in)
x = keras.layers.Dense(150, activation="selu")(x)
x = keras.layers.Dense(28*28, activation="sigmoid")(x)
decoder = keras.Model(dec_in, keras.layers.Reshape([28,28])(x))

# Full VAE + KL Loss
codings_mean, codings_log_var, codings = encoder(inputs)
vae = keras.Model(inputs, decoder(codings))
latent_loss = -0.5 * tf.reduce_sum(
    1 + codings_log_var - tf.exp(codings_log_var) - tf.square(codings_mean), axis=-1)
vae.add_loss(tf.reduce_mean(latent_loss) / (28*28))
vae.compile(loss="binary_crossentropy", optimizer="rmsprop")
```

### Output Target Design Table

| Architecture | Latent Space | Generation | Strength | Weakness |
|---|---|---|---|---|
| Basic AE | Deterministic | Interpolation only | Simple, fast | Not generative |
| Sparse AE | Sparse activations | Limited | Feature detection | Not generative |
| Denoising AE | Robust | Limited | Noise robustness | Not generative |
| VAE | Probabilistic N(μ,σ) | Smooth sampling | Principled generation | Blurry outputs |
| GAN | Arbitrary dist. | Sharp, photorealistic | SOTA image quality | Training instability |
| Diffusion | Noise schedule | Ultra-high quality | SOTA diversity | Slow inference |

---

## 🏆 Top 5 Things to Remember
1. **Autoencoders learn compressed representations** — the bottleneck forces only the most informative features to survive.
2. **VAEs introduce probabilistic latent spaces** — the reparameterization trick (`z = μ + ε·σ`) enables backpropagation through sampling.
3. **GANs are a min-max game** — the Generator tries to fool the Discriminator; training is inherently unstable and requires careful tricks (label smoothing, LeakyReLU, alternating updates).
4. **DCGAN replaced Dense with Conv layers** — strided convolutions (D) and transposed convolutions (G) enable photorealistic synthesis with spatial structure awareness.
5. **Diffusion models are the current SOTA** — they outperform GANs in quality and diversity by learning to reverse a Gaussian noise schedule over T timesteps with a U-Net denoiser.

---

## 🔗 Related Chapters
* **Chapter 15**: RNNs — sequential modeling backbone; recurrent autoencoders.
* **Chapter 16**: Transformers — BERT/GPT use encoder/decoder ideas rooted in AE architecture.

---
*Created for deep-dive studying and interview preparation.*
