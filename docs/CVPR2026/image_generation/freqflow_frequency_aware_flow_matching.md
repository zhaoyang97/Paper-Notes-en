---
title: >-
  [Paper Note] Frequency-Aware Flow Matching for High-Quality Image Generation
description: >-
  [CVPR 2026][Image Generation][Flow Matching] FreqFlow explicitly incorporates frequency-domain awareness into the flow matching framework via a dual-branch architecture that processes low-frequency global structures and…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Flow Matching"
  - "Frequency-Domain Awareness"
  - "Dual-Branch Architecture"
  - "Adaptive Weighting"
date: 2026-05-08
content_hash: 8ebb078c609dd08b
---

# Frequency-Aware Flow Matching for High-Quality Image Generation

**Conference**: CVPR 2026
**arXiv**: [2604.15521](https://arxiv.org/abs/2604.15521)  
**Code**: [https://github.com/OliverRensu/FreqFlow](https://github.com/OliverRensu/FreqFlow)  
**Area**: Image Generation
**Keywords**: Flow Matching, Frequency-Domain Awareness, Image Generation, Dual-Branch Architecture, Adaptive Weighting

## TL;DR

FreqFlow explicitly incorporates frequency-domain awareness into the flow matching framework via a dual-branch architecture that processes low-frequency global structures and high-frequency detail information separately, achieving state-of-the-art FID of 1.38 on ImageNet-256.

## Background & Motivation

**Background**: Flow Matching has emerged as one of the dominant frameworks for image generation, achieving high-quality image synthesis by learning a continuous transformation path from Gaussian noise to the data distribution. Models such as SiT and DiT have demonstrated strong performance on large-scale generation tasks.

**Limitations of Prior Work**: Existing flow matching methods inject noise uniformly in the spatial domain; however, noise affects different frequency components in the latent space unevenly. During the reverse process, models tend to reconstruct low-frequency components (global structure) first, with high-frequency components (textures, edges, and other fine details) emerging only in later stages. Nevertheless, no explicit mechanism exists within these models to distinguish and handle different frequency components, leading to blurry generative outputs in fine detail.

**Key Challenge**: Flow matching models operate in the spatial domain, yet the corruption and recovery process inherently affects images in a frequency-uneven manner — a characteristic that is neither explicitly modeled nor effectively exploited. Frequency error analysis reveals that SiT exhibits substantially larger high-frequency error (0.69) than low-frequency error (0.08).

**Goal**: To explicitly incorporate frequency-domain information into the flow matching framework, enabling the model to attend to the appropriate frequency components at each stage of generation.

**Key Insight**: The authors observe that the reverse process of flow matching naturally follows a "low-frequency first, high-frequency second" reconstruction order, consistent with the coarse-to-fine nature of human perception. Explicitly embedding frequency-domain conditioning into the model can reinforce this natural frequency generation sequence.

**Core Idea**: A dedicated frequency branch processes low- and high-frequency components separately; time-dependent adaptive weighting injects frequency-domain information into the spatial branch, realizing frequency-aware flow matching.

## Method

### Overall Architecture

FreqFlow adopts a dual-branch architecture comprising: (1) a **frequency branch** that decomposes the input via the Discrete Fourier Transform (DFT) into low- and high-frequency components to handle global structure and local detail respectively; and (2) a **spatial branch** that synthesizes images in the latent space while receiving the frequency branch outputs as conditioning guidance. Given a noisy image $X_t$, the frequency branch produces low- and high-frequency velocity field predictions, while the spatial branch predicts the complete velocity field under frequency guidance.

### Key Designs

1. **Frequency Branch**:

    - **Function**: Decomposes the input noisy image into the frequency domain and processes low- and high-frequency components separately.
    - **Mechanism**: $X_t$ is transformed to the frequency domain via DFT, and low-pass and high-pass filters are applied to separate frequency components. The frequency branch consists of independent Transformer blocks that model low- and high-frequency components individually and output the corresponding frequency velocity fields, which serve as supervision signals during training.
    - **Design Motivation**: Different frequency components carry different importance at different stages of the generation process; dedicated frequency processing enables more precise control over reconstruction quality within each frequency band.

2. **Time-Dependent Adaptive Weighting**:

    - **Function**: Dynamically balances the contributions of the frequency branch and the spatial branch across different generation stages.
    - **Mechanism**: Learnable time-dependent weights $w(t)$ are introduced so that low-frequency conditioning dominates in early stages (establishing global structure) while high-frequency conditioning is amplified in later stages (refining textural details). The weights adapt to the current timestep, ensuring that frequency-domain information is emphasized at the appropriate stage.
    - **Design Motivation**: The reconstruction process in flow matching naturally follows a low-frequency-first order; adaptive weighting makes this process more precise and efficient.

3. **Dual-Domain Supervision**:

    - **Function**: Trains the model simultaneously in both the frequency and spatial domains.
    - **Mechanism**: The spatial branch employs a standard flow matching loss (velocity field prediction error), while the frequency branch additionally incurs prediction losses on both the low- and high-frequency velocity fields. The losses from both branches are jointly optimized, ensuring the model learns spatial coherence and frequency accuracy simultaneously.
    - **Design Motivation**: A purely spatial-domain loss cannot effectively constrain the accurate reconstruction of frequency components; dual-domain supervision enables effective training in two complementary representation spaces.

### Loss & Training

The total loss is a weighted combination of the spatial-domain flow matching loss and the frequency-domain (low-frequency + high-frequency) velocity field prediction losses. Training follows the standard flow matching paradigm with timesteps sampled uniformly over $t \in [0,1]$.

## Key Experimental Results

### Main Results

| Model | FID ↓ | Parameters |
|-------|-------|------------|
| DiT-XL | 2.17 | 675M |
| SiT-XL | 1.96 | 675M |
| DiMR-G | 1.53 | 1.1B |
| MAR-H | 1.45 | 943M |
| **FreqFlow-L** | **1.44** | 625M |
| **FreqFlow-H** | **1.38** | ~1B |

### Ablation Study

| Configuration | FID |
|---------------|-----|
| Spatial branch only (baseline) | 1.96 |
| + Frequency branch (w/o adaptive weighting) | 1.62 |
| + Time-dependent adaptive weighting | 1.44 |

### Key Findings

- FreqFlow-L surpasses DiT-XL and SiT-XL with fewer parameters (625M vs. 675M), improving FID by 0.73 and 0.52, respectively.
- Frequency error analysis confirms that FreqFlow significantly outperforms SiT on both low-frequency (0.06 vs. 0.08) and high-frequency (0.48 vs. 0.69) errors.
- FreqFlow establishes global structure earlier in the denoising trajectory (reaching minimum log-amplitude at step 200, compared to step 280 for SiT).

## Highlights & Insights

- **Revisiting Flow Matching from a Frequency Perspective**: Extending flow matching from a purely spatial operation to a frequency-aware one is a natural yet previously underexplored direction. Frequency decomposition provides a new analytical tool for understanding and improving generative models.
- **Efficiency Advantage**: FreqFlow-L surpasses larger models with fewer parameters, demonstrating that frequency-domain information constitutes an efficient inductive bias more effective than simply scaling model size.
- **Transfer Potential**: The frequency-aware design paradigm is transferable to other generative tasks requiring multi-scale detail control, such as video generation and 3D generation.

## Limitations & Future Work

- The dual-branch architecture introduces additional computational overhead; lightweight variants of the frequency branch warrant further exploration.
- Evaluation is currently limited to class-conditional generation (ImageNet-256); more complex tasks such as text-to-image generation remain unassessed.
- Frequency decomposition relies on DFT, which may not be optimal for certain aperiodic textures.

## Related Work & Insights

- **vs. SiT**: SiT performs flow matching purely in the spatial domain; FreqFlow adds a frequency branch and adaptive weighting, yielding noticeably superior high-frequency detail.
- **vs. FreeU**: FreeU reweights skip connections in a U-Net to balance frequencies, whereas FreqFlow designs a dedicated frequency processing branch from the ground up in a more systematic manner.
- **vs. DiMR**: DiMR employs a multi-resolution strategy, while FreqFlow uses frequency-domain decomposition; both address the multi-scale problem but from different angles.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing frequency-domain awareness into flow matching is a novel idea, though the methodology is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive comparisons on ImageNet-256 with in-depth frequency error analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Frequency-domain motivation is clearly articulated with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Offers a new direction for improving flow matching models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VeCoR — Velocity Contrastive Regularization for Flow Matching](vecor_--_velocity_contrastive_regularization_for_flow_matching.md)
- [\[CVPR 2026\] EgoFlow: Gradient-Guided Flow Matching for Egocentric 6DoF Object Motion Generation](egoflow_gradient-guided_flow_matching_for_egocentric_6dof_object_motion_generati.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] LeapAlign: Post-Training Flow Matching Models at Any Generation Step by Building Two-Step Trajectories](leapalign_post_training_flow_matching_models_at_any_generation_step.md)
- [\[CVPR 2026\] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation](deco_frequency-decoupled_pixel_diffusion_for_end-to-end_image_generation.md)

</div>

<!-- RELATED:END -->
