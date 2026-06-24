---
title: >-
  [Paper Note] MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation
description: >-
  [CVPR 2025][3D Vision][3D generation] This paper proposes a progressive 3D generation framework using a Pyramid VAE paired with Cascaded MAR (MAR-LR → MAR-HR). By utilizing random masking to accommodate the unordered nature of 3D tokens and employing a condition augmentation strategy to mitigate cumulative errors during resolution scaling, the method achieves state-of-the-art (SOTA) performance among open-source approaches.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D generation"
  - "masked auto-regressive"
  - "pyramid VAE"
  - "cascaded generation"
  - "condition augmentation"
date: 2026-05-08
content_hash: 9dfbd4373d1f8be7
---

# MAR-3D: Progressive Masked Auto-regressor for High-Resolution 3D Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.20519](https://arxiv.org/abs/2503.20519)  
**Code**: [Project Page](https://jinnan-chen.github.io/projects/MAR-3D/)  
**Area**: 3D Vision  
**Keywords**: 3D generation, masked auto-regressive, pyramid VAE, cascaded generation, condition augmentation

## TL;DR

This paper proposes a progressive 3D generation framework using a Pyramid VAE paired with Cascaded MAR (MAR-LR → MAR-HR). By utilizing random masking to accommodate the unordered nature of 3D tokens and employing a condition augmentation strategy to mitigate cumulative errors during resolution scaling, the method achieves state-of-the-art (SOTA) performance among open-source approaches.

## Background & Motivation

**Background**: 3D generation methods generally fall into three paradigms: Large Reconstruction Models (LRMs) that reconstruct directly from a single image but lack generative priors; multi-view diffusion combined with reconstruction, which is limited by multi-view consistency; and 3D VAE integrated with diffusion/AR for direct generation, which is currently the most promising direction.

**Limitations of Prior Work**:
1. **Difficulty in scaling token resolution**: Existing 3D VAEs and generators struggle to preserve geometric details under a limited number of tokens. Directly increasing the number of tokens leads to a quadratic increase in the computational complexity of the Transformer, requiring hundreds of GPUs for training.
2. **Unordered nature of 3D data**: 3D latent tokens lack an inherent sequence, which conflicts with sequential prediction paradigms of autoregressive models.
3. **Vector quantization compression loss**: Traditional VQ methods incur significant compression losses on 3D meshes.

**Key Challenge**: High-quality 3D generation requires a larger token count to represent geometric details, but directly scaling the token complexity faces double challenges of computation and convergence.

**Key Insight**: A progressive strategy—first generating low-resolution tokens to capture the global shape, and then refining them into high-resolution tokens using a super-resolution model.

## Method

### Overall Architecture

MAR-3D consists of two components: Pyramid VAE and Cascaded MAR:
1. **Pyramid VAE**: Encodes multi-resolution point clouds into multi-scale latent tokens, supporting different resolutions of 256/1024 tokens.
2. **MAR-LR**: Generates 256 low-resolution tokens conditioned on image tokens.
3. **MAR-HR**: Generates 1024 high-resolution tokens conditioned on low-resolution tokens and image tokens.
4. Marching Cubes extracts the final mesh from the occupancy field.

### Key Designs

**1. Pyramid VAE**
- **Function**: Downsamples the input point cloud into $K$ levels (16384/4096/1024 points). Each level interacts with learnable queries $\mathbf{S}$ through an independent cross-attention layer. Coarse levels capture structural features while fine levels extract geometric details, which are then summed and processed by self-attention to obtain the latent token $\mathbf{X}$.
- **Mechanism**: $\mathbf{X} = \text{SelfAttn}\left(\sum_{k=1}^{K}\text{CrossAttn}^k(\mathbf{S}, \hat{\mathbf{P}}^k)\right)$. The training objective is the BCE loss (occupancy prediction) + KL divergence (latent regularization).
- **Design Motivation**: Compared with a single-level VAE, the 1024-token Pyramid VAE achieves higher reconstruction quality than a 2048-token single-level VAE, realizing highly efficient token compression.

**2. Cascaded MAR**
- **Function**: MAR-LR and MAR-HR share the same architecture (MAE encoder-decoder + MLP denoising network), with the main difference being that MAR-HR additionally accepts low-resolution tokens as input. Random masks (ratio 0.7-1.0) are utilized during training, and multiple tokens are decoded in parallel in a random order during inference.
- **Mechanism**: Decomposes the joint distribution into two components: temporal (diffusion) and spatial (autoregressive). Each token is supervised using a diffusion loss instead of Cross-Entropy because the latent tokens reside in a continuous space. A cosine schedule is applied during inference to control the token count generated at each step (fewer in the initial stages, more in the later stages).
- **Design Motivation**: 3D latent tokens lack an inherent order; thus, random masking combined with random-order decoding naturally fits this unordered characteristic. The cascaded strategy avoids convergence difficulties associated with directly training on 1024 tokens.

**3. Condition Augmentation**
- **Function**: During MAR-HR training, Gaussian noise is added to the input low-resolution tokens: $x_l' = t\epsilon + (1-t)x_l$, where $t \sim \mathcal{U}(0.4, 0.6)$. During inference, $t$ is fixed to 0.5.
- **Mechanism**: During training, MAR-HR receives "clean" low-resolution tokens encoded by the VAE. However, during inference, it receives "noisy" tokens generated by MAR-LR, resulting in a train-test gap. Adding noise successfully narrows this gap.
- **Design Motivation**: Inspired by cascaded diffusion models, this strategy effectively mitigates cumulative errors. Ablation studies show that without condition augmentation, the F-Score drops from 0.944 to 0.902.

### Loss & Training

- **VAE Loss**: $\mathcal{L}_{\text{vae}} = \text{BCE}(\hat{\mathcal{O}}, \mathcal{D}(\gamma(x), \mathbf{X})) + \lambda_{\text{kl}}\mathcal{L}_{\text{kl}}$
- **MAR Loss**: Diffusion loss $\mathcal{L} = \mathbb{E}[|\epsilon - \epsilon_\theta(x^t|t,z)|^2]$
- **Two-stage Training Data**: Initially trained for 200 epochs on 260K Objaverse meshes, and then fine-tuned for 100 epochs on 60K high-quality meshes.
- **Rotation Augmentation**: Renders 56 conditional views (8 base $\times$ 7 random rotations) for each mesh, synchronously rotating the 3D mesh to ensure image-latent consistency.
- **CFG**: Conditional features are randomly dropped with a probability of 0.1, using a linear CFG schedule $\omega_s = s \cdot \lambda_{cfg} / S$ during inference.

## Key Experimental Results

### Main Results

| Method | GSO F-Score↑ | GSO CD↓ | GSO NC↑ | OmniObj F-Score↑ | OmniObj CD↓ | OmniObj NC↑ |
|---|---|---|---|---|---|---|
| LGM | 0.745 | 0.813 | 0.685 | 0.738 | 0.821 | 0.677 |
| CraftsMan | 0.776 | 0.785 | 0.687 | 0.771 | 0.798 | 0.675 |
| TripoSR | 0.834 | 0.644 | 0.727 | 0.825 | 0.621 | 0.731 |
| InstantMesh | 0.923 | 0.415 | 0.780 | 0.918 | 0.427 | 0.779 |
| **MAR-3D** | **0.944** | **0.351** | **0.835** | **0.931** | **0.364** | **0.826** |

CD is reduced by 15.4% compared with InstantMesh.

### Ablation Study

| Configuration | F-Score↑ | CD↓ | NC↑ |
|---|---|---|---|
| w/o Pyramid VAE | 0.928 | 0.397 | 0.807 |
| w/o condition augmentation | 0.902 | 0.435 | 0.789 |
| w/o MAR-HR | 0.921 | 0.411 | 0.794 |
| w/o rotation augmentation | 0.934 | 0.369 | 0.821 |
| **Full Model** | **0.944** | **0.351** | **0.835** |

### Key Findings

1. **Pyramid VAE is highly efficient and effective**: The 1024-token Pyramid VAE outperforms the 2048-token single-level VAE in reconstruction quality, saving half the token budget while keeping more geometric details.
2. **Condition augmentation is crucial**: Removing condition augmentation degrades the CD from 0.351 to 0.435 (+24%), representing the most significant performance drop among all ablation components.
3. **Comparison between MAR and DiT**: MAR already outperforms DiT under the same budget of 256 tokens. When attempting direct training with 1024 tokens, both struggle to converge; however, MAR successfully scales up using condition augmentation, while the DiT version still shows noticeable noise.
4. **Progressive advantage of autoregressive decoding**: The cosine schedule (generating fewer tokens initially, and more later) outperforms the uniform schedule. This aligns with intuition as initial tokens are harder to predict.

## Highlights & Insights

- **Elegant decomposition strategy**: Decomposing the joint distribution of 3D generation into temporal (per-token diffusion) and spatial (autoregressive) dimensions makes it more scalable than pure DiT.
- **Condition augmentation is a simple yet critical technique**: Simply injecting noise effectively mitigates the cumulative error problem of the cascaded model.
- **Exquisite Pyramid VAE design**: The multi-resolution cross-attention paired with shared queries preserves details while keeping the token count constrained.
- **In-the-wild generalization capability**: Qualitative results demonstrate robust handling of complex topologies, such as holes and thin structures.

## Limitations & Future Work

- Relies on CLIP + DINOv2 to extract image features, which is limited by these models' 3D-aware understanding.
- Only generates geometry (mesh) without addressing texture or material generation.
- Training data is sourced from Objaverse, leaving room for improvement regarding generalization to real-world scanned objects.
- The two-stage cascaded inference increases latency.

## Related Work & Insights

- **3DShape2VecSet**: The pioneer work to encode 3D meshes into shape latents and use a diffusion model for generation.
- **CLAY**: A large-scale 3D diffusion model trained with hundreds of GPUs on a high token count; the progressive strategy proposed in this work serves as a more resource-efficient alternative.
- **MAR (2D)**: Combines autoregressive modeling with diffusion for 2D image generation, which this work extends into the 3D domain.
- **MaskGIT**: The pioneer of parallel prediction paradigm with random ordering, whose core concept is inherited by this work.

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐ The combined design of Pyramid VAE + Cascaded MAR + condition augmentation is highly instructive.  
**Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons against various methods alongside detailed ablations of the VAE and generator.  
**Writing Quality**: ⭐⭐⭐⭐ Clear structure with well-aligned problems, methods, and experiments.  
**Value**: ⭐⭐⭐⭐ SOTA performance among open-source methods, bearing direct practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Structured 3D Latents for Scalable and Versatile 3D Generation](structured_3d_latents_for_scalable_and_versatile_3d_generation.md)
- [\[CVPR 2025\] Event Fields: Capturing Light Fields at High Speed, Resolution, and Dynamic Range](event_fields_capturing_light_fields_at_high_speed_resolution_and_dynamic_range.md)
- [\[CVPR 2025\] Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [\[ECCV 2024\] SemanticHuman-HD: High-Resolution Semantic Disentangled 3D Human Generation](../../ECCV2024/3d_vision/semantichuman-hd_high-resolution_semantic_disentangled_3d_human_generation.md)
- [\[CVPR 2025\] SAR3D: Autoregressive 3D Object Generation and Understanding via Multi-scale 3D VQVAE](sar3d_autoregressive_3d_object_generation_and_understanding_via_multi-scale_3d_v.md)

</div>

<!-- RELATED:END -->
