---
title: >-
  [Paper Note] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution
description: >-
  [AAAI 2026][Image Generation][Real-world super-resolution] DegFlow is proposed to learn continuous degradation trajectories from discrete-scale real HR-LR pairs via a residual autoencoder and latent space Flow Matching. Given only a single HR image at inference, the model synthesizes realistic LR images at arbitrary continuous scales for training super-resolution models, achieving state-of-the-art performance.
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Real-world super-resolution"
  - "degradation modeling"
  - "Flow Matching"
  - "continuous scale"
  - "latent space"
date: 2026-05-08
content_hash: 1bc3ae09e3ffc585
---

# Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution

**Conference**: AAAI 2026
**arXiv**: [2602.04193](https://arxiv.org/abs/2602.04193)  
**Code**: [GitHub](https://github.com/present091/DegFlow)  
**Area**: Image Generation
**Keywords**: Real-world super-resolution, degradation modeling, Flow Matching, continuous scale, latent space

## TL;DR

DegFlow is proposed to learn continuous degradation trajectories from discrete-scale real HR-LR pairs via a residual autoencoder and latent space Flow Matching. Given only a single HR image at inference, the model synthesizes realistic LR images at arbitrary continuous scales for training super-resolution models, achieving state-of-the-art performance.

## Background & Motivation

Deep learning-based super-resolution methods perform well under synthetic degradations such as bicubic downsampling, but generalize poorly to real photographs due to complex nonlinear combinations of unknown blur, noise, and compression artifacts. Existing approaches each have notable drawbacks:

**Hand-crafted degradation pipelines** (Real-ESRGAN, BSRGAN): Augment training data by composing blur kernels, noise, downsampling, and compression artifacts. While improving robustness, they still fail to capture the full complexity of real-world degradations.

**Physical device capture** (RealSR, DRealSR, RealArbiSR): Collect paired data using DSLR zoom lenses. Degradations are realistic but acquisition costs are high and scene diversity is limited.

**Learned degradation models** (DeFlow, RealDGen): Learn the degradation process from a small number of real pairs and synthesize LR images, but lack **explicit scale control** and cannot generate degradations at arbitrary continuous scales.

**InterFlow**: Achieves continuous-scale synthesis by interpolating in the LR latent space, but requires two LR images of different scales as input at inference, limiting its practicality.

DegFlow learns continuous degradations from real HR-LR pairs and **requires only an HR image for inference**, supports arbitrary continuous scales, and produces more realistic degradation results.

## Method

### Overall Architecture

DegFlow follows a two-stage sequential training pipeline:

- **Stage 1 – Residual Autoencoder (RAE)**: Maps images into a compact latent space with HR skip connections to preserve detail.
- **Stage 2 – Latent Flow Matching (LFM)**: Learns continuous degradation trajectories in the latent space.

At inference: HR image → encoder → latent representation → FM model evolving along continuous time steps → decoder → LR image at arbitrary scale.

### Key Designs

#### 1. Residual Autoencoder (RAE)

Encoder $E_\theta$ maps an input image $I \in \mathbb{R}^{C \times H \times W}$ (either HR or LR) to a compact latent code $z = E_\theta(I) \in \mathbb{R}^{Cr^2 \times H/r \times W/r}$, where $r$ is the spatial compression factor.

To compensate for detail loss caused by high compression ratios, **multi-scale residual skip connections** are introduced:

$$\hat{I} = D_\theta(z; H_{\text{HR}})$$

where $H_{\text{HR}} = \{h^{(l)}_{\text{HR}}\}_{l=1}^L$ denotes multi-scale hidden features extracted from the **HR image**. Crucially, only HR features are injected via skip connections regardless of whether the input is HR or LR. This constrains the latent code to encode only **residual information** between LR and HR features (i.e., degradation-specific information), providing an informative representation for the subsequent FM model.

The reconstruction loss is applied to both HR and LR inputs:

$$\mathcal{L}_{\text{Recon}} = \|D_\theta(E_\theta(I_{s_1}); H_{\text{HR}}) - I_{s_1}\|_2^2 + \|D_\theta(E_\theta(I_{s_k}); H_{\text{HR}}) - I_{s_k}\|_2^2$$

#### 2. Latent Flow Matching (LFM)

After freezing the trained RAE, paired HR-LR images are encoded into the latent space and continuous degradation trajectories are constructed.

**Scale normalization**: The degradation level $s_k$ is linearly mapped to a timestamp $t_k = (s_k - s_1)/(s_m - s_1)$, normalized to $[0, 1]$. For example, $\mathcal{S}=\{1,2,4\}$ corresponds to $t_1=0, t_2=1/3, t_4=1$.

**Natural Cubic Spline trajectory**: Simple piecewise linear interpolation deviates from the nonlinear latent manifold and produces discontinuous derivatives that violate the smoothness assumptions of ODEs. The proposed method employs natural cubic splines to interpolate on sub-intervals $[t_k, t_{k+1}]$:

$$\mu_t(\epsilon) = a_k(\epsilon)(t-t_k)^3 + b_k(\epsilon)(t-t_k)^2 + c_k(\epsilon)(t-t_k) + d_k(\epsilon)$$

This guarantees continuity of first- and second-order derivatives, satisfying the regularity requirements of flow matching.

**Perceptual loss (auxiliary supervision)**: Intermediate scales (e.g., ×3.34) have no ground-truth LR images. A third-order Taylor expansion extrapolates predictions at intermediate time steps to the nearest training degradation level $s_{k+1}$, providing perceptual supervision via the LPIPS loss:

$$\mathcal{L}_{\text{LPIPS}} = \text{LPIPS}(I_{s_{k+1}}, D_\theta(\hat{z}_{t_{k+1}}))$$

Gradients from the LPIPS loss are back-propagated to LFM network parameters through the Taylor expansion.

### Loss & Training

**Stage 1 (RAE)**: Adam optimizer, 200k iterations, cosine annealing learning rate $1\text{e-}4 \to 1\text{e-}7$, batch size 16, 256×256 patches with random flipping.

**Stage 2 (LFM)**: Adam optimizer, 400k iterations, cosine annealing $2\text{e-}4 \to 1\text{e-}7$, batch size 32, 256×256 patches.
Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + \lambda \mathcal{L}_{\text{LPIPS}}$, with $\lambda = 0.1$.

Training set: RealSR-V2 Canon training set (×1, ×2, ×4 paired data).

## Key Experimental Results

### Main Results

**Table 2: Fixed-scale SR results (RealSR ×3 test set; no ×3 data used during training)**

| Model | Training Data | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| HAT | RealSR ×3 (Oracle) | 30.71 | 0.8645 | 0.3221 |
| HAT | RealSR ×2, ×4 | 30.39 | 0.8607 | 0.3248 |
| HAT | InterFlow ×2~×4 | 30.65 | 0.8645 | 0.3135 |
| HAT | **Ours ×2~×4** | **30.86** | **0.8668** | **0.3186** |
| MambaIR | RealSR ×3 (Oracle) | 30.62 | 0.8636 | 0.3208 |
| MambaIR | InterFlow ×2~×4 | 30.51 | 0.8625 | 0.3138 |
| MambaIR | **Ours ×2~×4** | **30.73** | **0.8686** | **0.3152** |

The DegFlow-synthesized training set enables HAT to reach 30.86 dB, **surpassing the Oracle trained on real ×3 data** (30.71 dB).

**Table 3: Arbitrary-scale SR results (RealSR ×3 + RealArbiSR multi-scale)**

| Model | Method | RealSR ×3 PSNR | RealArbiSR ×2.5 PSNR |
|---|---|---|---|
| MetaSR | Bicubic | 28.99 | 30.05 |
| MetaSR | InterFlow | 30.42 | 30.71 |
| MetaSR | **Ours** | **30.58** | **30.88** |
| LIIF | InterFlow | 30.44 | 30.70 |
| LIIF | **Ours** | **30.61** | **30.99** |

DegFlow consistently outperforms InterFlow on arbitrary-scale SR while requiring only HR input.

### Ablation Study

**Table 4: Contribution of each component**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| Baseline (piecewise linear trajectory) | 30.58 | 0.8640 | 0.3214 |
| + Nonlinear trajectory (cubic spline) | 30.68 | 0.8652 | 0.3209 |
| + LPIPS third-order Taylor approximation | 30.81 | 0.8662 | 0.3200 |
| + HR skip connections in RAE | **30.86** | **0.8668** | **0.3186** |

Each component contributes clearly: cubic spline > LPIPS perceptual supervision > HR skip connections.

### Key Findings

1. **Timestep–degradation correspondence analysis**: On the RealSR ×3 test set, PSNR and FID peak at $t \approx 0.73$ ($s \approx 3.2$), and CLIP peaks at $t \approx 0.70$ ($s \approx 3.1$), validating that DegFlow learns a meaningful continuous degradation manifold.
2. **External HR data augmentation**: Synthesizing additional LR training data from DIV2K HR images further improves PSNR by 0.14 dB (30.86→31.00), demonstrating the advantage of requiring only HR data to expand the training set.
3. The derivative continuity of cubic splines is critical for ODE solving in flow matching.

## Highlights & Insights

- **Inference requires only an HR image**: In contrast to InterFlow, which requires LR inputs at two scales, DegFlow is more practical with HR-only inference.
- **Continuous degradation manifold**: The combination of cubic splines and flow matching is elegant, striking a balance between continuity and nonlinear modeling.
- **Residual latent space design**: HR skip connections constrain the latent code to encode only degradation information, substantially simplifying the FM model's learning task.
- **LPIPS Taylor approximation**: An elegant solution for providing perceptual supervision at intermediate scales without ground-truth LR images.

## Limitations & Future Work

1. Training depends on real paired datasets (RealSR); generalization may degrade when the target camera domain differs significantly from the training domain.
2. The spatial compression factor $r$ in the RAE requires balancing computational cost against detail preservation.
3. Only two discrete points (×2 and ×4) are used for training; whether additional discrete points further improve the continuous trajectory remains unexplored.
4. No direct comparison with diffusion-based degradation modeling methods (e.g., RealDGen) on degradation realism is provided.

## Related Work & Insights

- Compared to normalizing flow methods such as DeFlow and NAFlow, flow matching is more flexible and does not require invertible architectural constraints.
- Compared to InterFlow, the core advantage of DegFlow lies on the inference side: arbitrary-scale LR images can be generated from HR alone.
- The paradigm of latent-space degradation modeling is generalizable to other degradation tasks including denoising, dehazing, and deblurring.
- The idea of replacing linear interpolation with cubic splines is worth extending within the flow matching framework.

## Rating

- **Novelty**: ★★★★☆ — Latent space flow matching with cubic spline trajectory modeling constitutes a novel degradation modeling approach.
- **Technical Depth**: ★★★★☆ — The two-stage RAE+LFM design is well-motivated; the LPIPS Taylor approximation is clever.
- **Experimental Thoroughness**: ★★★★☆ — Validated on both fixed- and arbitrary-scale SR; surpassing Oracle is a strong result.
- **Writing Quality**: ★★★★☆ — Clear structure with comprehensive comparison tables.
- **Value**: ★★★★★ — HR-only inference, open-source code, directly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[ICML 2026\] Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](../../ICML2026/image_generation/q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)

</div>

<!-- RELATED:END -->
