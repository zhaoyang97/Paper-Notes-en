---
title: >-
  [Paper Note] Normalizing Flows are Capable Generative Models
description: >-
  [ICML2025][Image Generation][Normalizing Flows] Proposes TarFlow (Transformer AutoRegressive Flow), which implements block autoregressive Normalizing Flows by stacking causal ViTs, breaking the 3 BPD barrier on ImageNet 64×64 for the first time. Through three key techniques—Gaussian noise augmentation, score-based denoising, and guidance—it enables the generation quality of NF models to rival diffusion models for the first time.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Normalizing Flows"
  - "Transformer"
  - "Autoregressive Flows"
  - "Density Estimation"
date: 2026-05-08
content_hash: 794b525fdaefeed0
---

# Normalizing Flows are Capable Generative Models

**Conference**: ICML2025  
**arXiv**: [2412.06329](https://arxiv.org/abs/2412.06329)  
**Code**: [apple/ml-tarflow](https://github.com/apple/ml-tarflow)  
**Area**: Image Generation  
**Keywords**: Normalizing Flows, Transformer, Autoregressive Flows, Image Generation, Density Estimation  
**Authors**: Shuangfei Zhai, Ruixiang Zhang, Preetum Nakkiran, David Berthelot, Jiatao Gu et al. (Apple)

## TL;DR

Proposes TarFlow (Transformer AutoRegressive Flow), which implements block autoregressive Normalizing Flows by stacking causal ViTs, breaking the 3 BPD barrier on ImageNet 64×64 for the first time. Through three key techniques—Gaussian noise augmentation, score-based denoising, and guidance—it enables the generation quality of NF models to rival diffusion models for the first time.

## Background & Motivation

Normalizing Flows (NFs) are a class of exact likelihood-based generative models. They map data distributions to simple priors (such as Gaussian distributions) via invertible transformations, offering advantages like exact likelihood computation, deterministic objectives, and efficient bidirectional computation. However, in recent years, the impact of NFs in practical applications has lagged far behind Diffusion Models and LLMs, and their state-of-the-art (SOTA) performance has long stagnated.

The core question is: **Do NFs have fundamental limitations in their modeling paradigm? Or have we just not yet found the appropriate architecture and training methods to unleash their potential?**

The authors argue for the latter. The historical bottlenecks of NFs lie in:

**Overly complex and constrained architectural design**: Models like Glow and RealNVP use meticulously hand-designed coupling layers, making it difficult to freely scale model capacity.

**Unstable training**: Continuous NFs (e.g., FFJORD) suffer from numerical instability issues.

**Poor generation quality**: Traditional uniform dequantization noise is insufficient to support high-quality sampling.

## Method

### 2.1 Normalizing Flow Basics

NFs model data density via the change-of-variables formula:

$$p_{\text{model}}(x) = p_0(f(x)) \left|\det\left(\frac{df(x)}{dx}\right)\right|$$

where $f: \mathbb{R}^D \to \mathbb{R}^D$ is an invertible transformation, and $p_0$ is a standard Gaussian prior. The MLE training objective is:

$$\min_f \; 0.5\|f(x)\|_2^2 - \log\left|\det\left(\frac{df(x)}{dx}\right)\right|$$

The first term drives the model to map data to latent variables with small norms, while the second term prevents model collapse.

### 2.2 Block Autoregressive Flows

TarFlow is a block generalization of MAF (Masked Autoregressive Flow). The input is represented as a sequence $x \in \mathbb{R}^{N \times D}$, and the flow transformations are stacked over $T$ steps, with each step containing:

1. **Sequence permutation** $\pi^t$: Alternately reversing the sequence direction (directions in odd and even steps are opposite).
2. **Affine transformation**:
$$z_i^{t+1} = (\tilde{z}_i^t - \mu_i^t(\tilde{z}_{<i}^t)) \odot \exp(-\alpha_i^t(\tilde{z}_{<i}^t)), \quad i > 0$$

where $\mu^t, \alpha^t$ are causal functions (the output at the $i$-th position depends only on the previous $i-1$ positions). When $D=1$, this degenerates to standard MAF.

The log-**Jacobian determinant** can be computed efficiently:
$$\log|\det(df^t/dz^t)| = -\sum_{i=1}^{N-1}\sum_{j=0}^{D-1} \alpha_i^t(\tilde{z}_{<i}^t)_j$$

The final training loss is compactly expressed as:
$$\min_f \; 0.5\|z^T\|_2^2 + \sum_{t=0}^{T-1}\sum_{i=1}^{N-1}\sum_{j=0}^{D-1} \alpha_i^t(\tilde{z}_{<i}^t)_j$$

### 2.3 Transformer Autoregressive Flow Architecture

The core innovation is replacing the simple masked MLP in MAF with a **causal Vision Transformer (causal ViT)**. For a $C \times H \times W$ image, it is first partitioned into a sequence of patches ($N = HW/S^2$, $D = CS^2$), and standard causal attention is then utilized to achieve autoregressive transformations at each step.

Key advantages:
- **Simple and modular**: The interior of each flow block is a standard Transformer, with depth and width completely decoupled from the input dimension.
- **Stable training**: Double residual connections (inside the Transformer + between latent variables $z_i^t$) make the training difficulty equivalent to that of a standard Transformer.
- **High scalability**: The number of blocks $T$ and the number of layers per block $K$ can be freely scaled.

### 2.4 Gaussian Noise Augmentation Training

The traditional approach is to add a small amount of uniform noise for dequantization, which the authors find is **far from sufficient**. Key findings:

- The optimal Gaussian noise $\sigma \approx 0.05$ (for pixel values in $[-1,1]$), whereas the standard deviation of traditional uniform noise is only 0.002.
- The essence of noise augmentation: Enriching the training distribution support for the inverse model $f^{-1}$ to avoid out-of-distribution (OOD) issues during sampling.
- Gaussian noise (vs. uniform noise) expands the training distribution support to the entire ambient space.

### 2.5 Score-Based Denoising

Directly sampling after noise-augmented training produces noisy samples. Denoising without additional training is achieved using the Tweedie formula:

$$\hat{x} = y + \sigma^2 \nabla_y \log p_{\text{model}}(y)$$

where $y = f^{-1}(z)$ is the noisy sample. Denoising only requires the TarFlow model itself to compute the score, without needing any extra modules.

### 2.6 Guidance

**Conditional guidance**: Fully consistent with CFG, where class labels are randomly dropped with a probability of 0.1 during training:
$$\tilde{\mu}_i^t = (1+w)\mu_i^t(\cdot; c) - w \cdot \mu_i^t(\cdot; \varnothing)$$

**Unconditional guidance** (introduced in this work): Uses attention temperature $\tau$ to construct degraded predictions that act as "unconditional predictions":
$$\tilde{\mu}_i^t = (1+w)\mu_i^t(\cdot; 1) - w \cdot \mu_i^t(\cdot; \tau)$$

## Key Experimental Results

### Density Estimation: ImageNet 64×64 (BPD ↓)

| Model | Type | BPD |
|------|------|-----|
| Flow Matching | Diff/FM | 3.31 |
| NFDM | Diff/FM | 3.20 |
| VDM | Diff/FM | 3.40 |
| Sparse Transformer | AR | 3.44 |
| Flow++ | Flow | 3.69 |
| Glow | Flow | 3.81 |
| **TarFlow [2-768-8-8]** | **NF** | **2.99** |

First time breaking the 3 BPD barrier! 0.21 lower than the previous strongest counterpart, NFDM.

### Conditional Generation: ImageNet 64×64 (FID ↓)

| Model | Type | FID |
|------|------|-----|
| EDM | Diff/FM | 1.55 |
| ADM (dropout) | Diff/FM | 2.09 |
| BigGAN | GAN | 4.06 |
| **TarFlow (w=2)** | **NF** | **5.7** |

### Conditional Generation: ImageNet 128×128 (FID ↓)

| Model | Type | FID |
|------|------|-----|
| Simple Diffusion | Diff/FM | 1.94 |
| ADM-G | Diff/FM | 2.97 |
| BigGAN-deep | GAN | 5.70 |
| **TarFlow** | **NF** | **5.03** |

### Unconditional Generation: ImageNet 64×64 (FID ↓)

| Model | Type | FID |
|------|------|-----|
| AGM | Diff/FM | 10.07 |
| IC-GAN | GAN | 10.40 |
| **TarFlow** | **NF** | **18.42** |

### Ablation Study

- **VP vs. NVP**: Removing the scale term $\alpha$ (VP mode) deteriorates the FID from 5.7 to 51.0.
- **Channel Coupling vs. Autoregressive**: Replacing with channel coupling deteriorates the FID to 20.4.
- **Depth configuration**: Optimal when $T=K$ (number of blocks = number of layers per block); $T=1$ (one-way autoregressive) completely fails (FID = 267).
- **Noise ablation**: The denoising step achieves the optimal FID around $\sigma=0.05$, and the $\sigma$-robustness of FID after denoising is significantly improved.

### Training Configuration

- Optimizer: AdamW, momentum (0.9, 0.95), cosine learning rate schedule, peak $10^{-4}$.
- Hardware: A100 GPUs, all experiments completed within 14 days.
- Precision: bfloat16 for generation tasks, float32 for likelihood estimation.
- Sampling speed: ~2 minutes for 32 images (single A100 GPU, ImageNet 64×64).

## Highlights & Insights

1. **A victory for architectural minimalism**: Without complex modules like 1x1 convolutions and multi-scale coupling layers, stacking causal ViTs with alternating directions is sufficient to significantly outperform historical baselines.
2. **A bridge between NFs and Diffusion**: Visualization of sampling trajectories shows that the evolution of TarFlow's $z^t$ sequence from noise to images is highly similar to diffusion models, despite having completely different training objectives.
3. **Positive correlation between loss and FID**: A decrease in training loss (likelihood) directly yields improved FID, which is a unique advantage of NFs compared to other generative models.
4. **Compatibility between Guidance and NFs**: Demonstrates for the first time that CFG and unconditional guidance can be directly applied to NF models.

## Limitations & Future Work

1. **Slow sampling speed**: The inverse transformation must perform autoregression step-by-step along the sequence dimension; despite using KV-caching, it is still far slower than the parallel denoising in diffusion models.
2. **FID gap remains**: On conditional ImageNet 64x64, the FID of 5.7 compared to EDM's 1.55 shows a gap of approximately 3-4x.
3. **Weaker unconditional generation**: Unconditional FID of 18.42 is significantly worse than AGM's 10.07.
4. **Limited resolution**: The maximum resolution demonstrated is only 256x256 (AFHQ), lacking validation on high-resolution settings (512+).
5. **Large memory overhead for the denoising step**: Score-based denoising requires caching all intermediate activations for backpropagation.
6. **Guidance schedule under-explored**: The paper initially notes that a linearly increasing $w_i$ is superior, but this was not studied in depth.

## Related Work & Insights

- **Relation to MAF/IAF**: TarFlow is a block generalization of MAF combined with a Transformer backbone replacement.
- **Difference from Flow Matching**: Flow Matching trains an ODE via velocity prediction and requires a large amount of Gaussian noise; TarFlow is directly trained with MLE, using noise that is an order of magnitude smaller.
- **Difference from JetFormer**: JetFormer uses an NF as a tokenizer followed by a two-stage AR Transformer; TarFlow is a single-model, end-to-end framework.
- **Insights**: NFs might have been long-underestimated, primarily due to the lack of a scalable architecture. The introduction of Transformers may offer a similar opportunity for the revival of other "forgotten" classic methods.

## Rating
- Novelty: ⭐⭐⭐⭐ — Simple architectural idea but deep insights; the three sampling techniques (especially unconditional guidance) are quite creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Thorough ablation (noise, denoising, guidance, VP/NVP, depth configuration) across multiple datasets and settings.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear and fluent, complete formulation derivations, well-articulated motivation.
- Value: ⭐⭐⭐⭐ — Injects new vitality into the NF field, though a gap still remains in FID compared to diffusion models, and actual application prospects remain to be seen.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SESaMo: Symmetry-Enforcing Stochastic Modulation for Normalizing Flows](../../ICLR2026/image_generation/sesamo_symmetry-enforcing_stochastic_modulation_for_normalizing_flows.md)
- [\[AAAI 2026\] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment](../../AAAI2026/image_generation/flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](../../ICML2026/image_generation/the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[CVPR 2026\] Improved Mean Flows: On the Challenges of Fastforward Generative Models](../../CVPR2026/image_generation/improved_mean_flows_on_the_challenges_of_fastforward_generative_models.md)
- [\[ICML 2025\] Hessian Geometry of Latent Space in Generative Models](hessian_geometry_of_latent_space_in_generative_models.md)

</div>

<!-- RELATED:END -->
