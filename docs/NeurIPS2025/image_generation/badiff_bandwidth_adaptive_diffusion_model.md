---
title: >-
  [Paper Note] BADiff: Bandwidth Adaptive Diffusion Model
description: >-
  [NeurIPS 2025][Image Generation][bandwidth-adaptive] This paper proposes BADiff—the first bandwidth-adaptive diffusion model—which embeds target entropy constraints as explicit conditions into the diffusion reverse process, coupled with a differentiable entropy regularization loss and an adaptive stopping policy. The model dynamically adjusts generation quality according to real-time bandwidth and terminates sampling adaptively, reducing computational overhead while maintaining perceptual quality. This approach fundamentally avoids the compression artifacts and computational waste inherent in the conventional "high-quality generation → post-compression" pipeline.
tags:
  - NeurIPS 2025
  - Image Generation
  - bandwidth-adaptive
  - diffusion model
  - entropy conditioning
  - early stopping
  - cloud streaming
date: 2026-05-08
content_hash: 84e873a7d6c7f8ee
---

# BADiff: Bandwidth Adaptive Diffusion Model

**Conference**: NeurIPS 2025
**arXiv**: [2510.21366](https://arxiv.org/abs/2510.21366)
**Code**: [GitHub](https://github.com/xzhang9308/BADiff)
**Authors**: Xi Zhang, Hanwei Zhu, Yan Zhong, Jiamang Wang, Weisi Lin (NTU & Alibaba)
**Area**: Diffusion Models / Image Compression / Bandwidth-Adaptive Generation
**Keywords**: bandwidth-adaptive, diffusion model, entropy conditioning, early stopping, cloud streaming

## TL;DR

This paper proposes BADiff—the first bandwidth-adaptive diffusion model—which embeds target entropy constraints as explicit conditions into the diffusion reverse process, coupled with a differentiable entropy regularization loss and an adaptive stopping policy. The model dynamically adjusts generation quality according to real-time bandwidth and terminates sampling adaptively, reducing computational overhead while maintaining perceptual quality. This approach fundamentally avoids the compression artifacts and computational waste inherent in the conventional "high-quality generation → post-compression" pipeline.

## Background & Motivation

**State of the Field**: Diffusion models (DDPM, LDM, etc.) are capable of generating highly faithful images, yet face transmission bandwidth bottlenecks in practical cloud-to-device deployments. The prevailing approach is to generate high-quality images with a diffusion model and then apply lossy compression via BPG or learned image codecs (LIC) before transmission.

**Limitations of Prior Work**:
   - **Cascaded pipelines are wasteful**: Fine-grained textural details painstakingly constructed by the diffusion model are erased by subsequent compression, incurring both computational and quality losses.
   - **Naïve early stopping performs poorly**: Simply reducing the number of diffusion steps introduces visual artifacts and perceptual incoherence, as the model is not trained with early termination in mind.
   - **Generation is bandwidth-agnostic**: Standard diffusion models are entirely unaware of downstream transmission constraints.

**Root Cause**: The generation process and transmission constraints are completely decoupled—the model is oblivious to available bandwidth, producing unnecessarily high-quality outputs that cannot be transmitted as-is, or losing all detail after compression.

**Starting Point**: Diffusion models inherently refine images from coarse to fine—early steps construct global structure while later steps add fine-grained texture. Different bandwidth requirements can thus correspond to different stopping points. The key is to train the model such that it produces perceptually high-quality outputs at any given stopping point.

## Method

### 3.1 Entropy-Conditioned Diffusion

**Core Idea**: The target entropy $H_{\text{target}}$ (a proxy for bandwidth, measured in bpp) is injected as an explicit condition into every step of the diffusion reverse process.

- **Entropy embedding network**: A lightweight MLP $\psi_\eta$ maps the scalar $H_{\text{target}}$ to a 128-dimensional vector $\mathbf{h}$.
- **Condition injection**: In each residual block of the UNet, the entropy embedding is added to the sinusoidal timestep embedding, forming a mixed modulation signal $\mathbf{g}_l(t, H_{\text{target}}) = \mathbf{g}(t) + \mathbf{W}^{(l)} \mathbf{h}$, equivalent to an additive FiLM mechanism.
- **Minimal parameter overhead**: Additional parameters account for less than 0.1%, yet provide the model with a continuous "knob" for controlling generation detail.
- During training, $H_{\text{target}} \sim \mathcal{U}(H_{\min}, H_{\max})$ is sampled randomly to expose the model to a wide range of bandwidth constraints.

### 3.2 Entropy Regularization Loss

Conditioning alone is insufficient—the model may still produce outputs that exceed the bandwidth budget. A differentiable entropy constraint is therefore introduced:

$$\mathcal{L}_{\text{entropy}} = \max(0, H_\phi(\hat{\mathbf{x}}_0) - H_{\text{target}})$$

- **Differentiable neural entropy estimator** $H_\phi$: Inspired by entropy models from learned image compression, it uses a discrete logistic distribution to model pixel-level conditional probabilities, yielding an expected encoding length (bpp) per pixel.
- **Hinge formulation**: Gradients are only propagated when the actual entropy exceeds the budget, avoiding over-regularization.
- **Context extraction**: A hyper-prior combined with autoregressive masked convolutions extracts the causal context $\mathbf{c}_u$.
- **End-to-end differentiable**: Gradients flow from the entropy loss back through the UNet without requiring straight-through estimators.

### 3.3 Calibration Loss

To align the predictions of the entropy estimator $E_\phi$ with an actual codec, an auxiliary calibration loss is introduced:

$$\mathcal{L}_{\text{calibration}} = \frac{1}{|\Omega|} \sum_{u \in \Omega} D_{\text{KL}}(q_u \| p_\phi(\cdot | \mathbf{c}_u))$$

where $q_u(k)$ is derived from pixel-level distributions of a reference end-to-end optimized codec, bringing the predicted entropy values closer to the actual post-encoding bitrate.

### 3.4 Adaptive Sampling Policy

A lightweight MLP policy network $f_\phi$ is introduced to decide at each sampling step whether to terminate:

- **Input**: Spatially average-pooled latent features $\mathbf{h}_t$, current step $t$, and target entropy $H_{\text{target}}$.
- **Output**: Stopping probability $p_t$; sampling terminates when $p_t \geq 0.5$.
- **Supervision signal**: The optimal stopping point is determined offline by running full sampling and computing a per-step cost $\mathcal{C}(t) = \text{entropy} + \beta \cdot \text{distortion} + \gamma \cdot t$, used as a teacher label.
- **Training objective**: $\mathcal{L}_{\text{stop}} = \text{BCE}(y_t, p_t)$
- **Negligible overhead**: < 0.3 ms/step (RTX 4090).

### 3.5 Overall Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{denoise}} + \lambda_{\text{ent}} \mathcal{L}_{\text{entropy}} + \lambda_{\text{cal}} \mathcal{L}_{\text{calibration}} + \lambda_{\text{stop}} \mathcal{L}_{\text{stop}}$$

Default hyperparameters: $\lambda_{\text{ent}}=0.1$, $\lambda_{\text{cal}}=10^{-3}$, $\lambda_{\text{stop}}=10^{-2}$. Trained with the Adam optimizer at lr=$10^{-4}$ for 800k iterations.

## Key Experimental Results

### FID Comparison (Low Bitrate 0.2–0.5 bpp, DDPM Backbone)

| Method | CIFAR-10 | CelebA-HQ | LSUN |
|--------|----------|-----------|------|
| DDPM + BPG (cascade) | 15.2 | 28.5 | 25.7 |
| DDPM + LIC (cascade) | 13.6 | 25.3 | 22.8 |
| Early-Stop + LIC | 22.9 | 35.0 | 31.9 |
| PNDM + LIC | 18.1 | 30.4 | 27.3 |
| **BADiff** | **11.4** | **21.7** | **19.6** |

### Inference Speed (CIFAR-10, ms/image, DDPM Backbone)

| Method | Low Bitrate | Mid Bitrate | High Bitrate |
|--------|-------------|-------------|--------------|
| Cascade + LIC | 115 | 115 | 115 |
| Early-Stop | 58 | 75 | 92 |
| **BADiff** | **65** | **78** | **94** |

BADiff achieves **1.7×** speedup over the cascaded baseline at low bitrate, with substantially better FID.

### Ablation Study (CIFAR-10, Low Bitrate)

| Variant | FID↓ | Δbpp↓ |
|---------|------|-------|
| w/o entropy conditioning | 13.1 | 0.038 |
| w/o hinge loss | 16.2 | 0.055 |
| w/o calibration loss | 18.6 | 0.043 |
| **Full BADiff** | **11.4** | **0.021** |

### High-Resolution Extension

- **512×512**: BADiff FID 6.85 vs. DDPM+LIC 8.45; inference 64.1 ms vs. 121.3 ms
- **1024×1024**: BADiff FID 17.8 vs. DDPM+LIC 21.5; inference 145.6 ms vs. 228.7 ms

### Text-to-Image Extension (Stable Diffusion Backbone)

| Method | Low-BR FID | Mid-BR FID | High-BR FID |
|--------|-----------|-----------|------------|
| SD + BPG | 33.5 | 21.4 | 14.8 |
| SD + LIC | 30.7 | 19.2 | 13.1 |
| **BADiff** | **26.1** | **16.2** | **11.0** |

## Highlights & Insights

- **Paradigm shift**: Bandwidth constraints are moved upstream into the generation process itself, replacing the "carefully generate then aggressively compress" paradigm with one in which the model is aware of bandwidth from the outset.
- **Exploiting the natural properties of diffusion models**: The coarse-to-fine refinement process maps naturally to bandwidth—low bandwidth requires only the coarse structure from early steps, while high bandwidth warrants continued refinement.
- **End-to-end differentiable entropy control**: The neural entropy estimator enables gradient flow through the entropy objective without histogram approximations, yielding more accurate rate control.
- **Solver-agnostic**: The entropy conditioning mechanism is compatible with fast samplers such as PNDM and DPM-Solver.
- **Low cost of teacher label generation**: Labels are computed offline in a single pass, accounting for only 5–8% of a single training epoch.

## Limitations & Future Work

- **Spatially uniform budget**: The current formulation supports only a globally uniform entropy budget, without spatially adaptive bitrate allocation (e.g., allocating more bits to salient regions).
- **Image-only validation**: The framework is not evaluated on video diffusion, where bandwidth constraints are more stringent.
- **No combination with fast solvers**: The paper focuses on DDPM/LDM backbones; integration with PNDM/DPM-Solver is left as future work.
- **Resolution ceiling**: Primary experiments are conducted at 256×256; high-resolution experiments are supplementary.

## Rating

- Novelty: ⭐⭐⭐⭐ First work to directly integrate bandwidth constraints into the diffusion generation process; the combination of entropy conditioning, differentiable entropy loss, and adaptive stopping is a complete and coherent design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across three datasets × three bitrate levels × two backbones, with comprehensive ablations and supplementary high-resolution and T2I experiments.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, methodological derivations are rigorous, and equations and algorithmic pseudocode are complete.
- Value: ⭐⭐⭐⭐ Demonstrates clear practical value for cloud-based image streaming; the paradigm is generalizable to video generation.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[ICCV 2025\] Adaptive Routing of Text-to-Image Generation Requests Between Large Cloud Model and Light-Weight Edge Model](../../ICCV2025/image_generation/adaptive_routing_of_text_to_image_generation_requests_between_large_cloud_model_and_light_weight_edge_model.md)
- [\[NeurIPS 2025\] BlurDM: A Blur Diffusion Model for Image Deblurring](blurdm_a_blur_diffusion_model_for_image_deblurring.md)

<!-- RELATED:END -->
