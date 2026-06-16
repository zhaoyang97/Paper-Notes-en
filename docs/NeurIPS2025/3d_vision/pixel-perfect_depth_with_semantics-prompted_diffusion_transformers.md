---
title: >-
  [Paper Note] Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers
description: >-
  [NeurIPS 2025][3D Vision][Monocular Depth Estimation] This paper proposes Pixel-Perfect Depth, a monocular depth estimation model that performs diffusion generation directly in pixel space (rather than latent space). Thr…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Monocular Depth Estimation"
  - "Pixel-Space Diffusion"
  - "DiT"
  - "Semantics Prompting"
  - "Flying Pixel Removal"
date: 2026-05-08
content_hash: 9d6d6158f4878046
---

# Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers

**Conference**: NeurIPS 2025
**arXiv**: [2510.07316](https://arxiv.org/abs/2510.07316)  
**Code**: [Project Page](https://pixel-perfect-depth.github.io)  
**Area**: 3D Vision
**Keywords**: Monocular Depth Estimation, Pixel-Space Diffusion, DiT, Semantics Prompting, Flying Pixel Removal

## TL;DR

This paper proposes Pixel-Perfect Depth, a monocular depth estimation model that performs diffusion generation directly in pixel space (rather than latent space). Through a Semantics-Prompted DiT (SP-DiT) that incorporates high-level semantic representations from visual foundation models and a cascaded DiT design, the model generates flying-pixel-free depth maps, surpassing all published generative models on five benchmarks.

## Background & Motivation

- **Background**: Monocular depth estimation (MDE) is a foundational task for 3D reconstruction, novel view synthesis, and robotic manipulation. Depth maps produced by current models suffer from pervasive **flying pixels** at object boundaries when converted to point clouds—spurious floating points that severely limit practical applications such as free-viewpoint broadcasting and immersive content creation.

- **Limitations of Prior Work**: The root cause of flying pixels differs by model type:
    - **Discriminative models** (e.g., Depth Anything v2): tend to output **intermediate depth values** (mean bias) between foreground and background at depth-discontinuous edges to minimize regression loss.
    - **Generative models** (e.g., Marigold): can theoretically capture the multimodal distribution at edges, but fine-tuning Stable Diffusion requires a VAE to compress depth maps into latent space—VAE compression **inevitably loses edge sharpness**.

- **Key Challenge**: An intuitive fix is to perform diffusion directly in pixel space. However, the authors find this **extremely challenging**—the core difficulty lies in simultaneously modeling **global semantic coherence** and **fine-grained visual detail** in high-resolution pixel-space generation. SNR analysis confirms that the primary challenge of high-resolution pixel-space diffusion is perceiving and modeling global image structure.

- **Core Idea**: Introduce **high-level semantic representations** from pretrained visual foundation models as prompts (Semantics-Prompted), providing global semantic "anchors" during diffusion so that high-quality depth maps can be stably generated directly in pixel space.

## Method

### Overall Architecture

Input image concatenated with noise → fed into a cascaded DiT (first half uses large patches for global structure; second half uses small patches for detail) → semantic representations extracted from the image are simultaneously injected into the second-half DiT → depth map output directly in pixel space, without any VAE.

### Key Designs

1. **Semantics-Prompted DiT (SP-DiT)**

   Core problem: A vanilla DiT in pixel space cannot simultaneously capture global semantics and local detail (ablation shows NYUv2 AbsRel of 22.5%, nearly unusable).

   Solution: Extract high-level semantic representations $\mathbf{e} = f(\mathbf{c}) \in \mathbb{R}^{T' \times D'}$ from a pretrained visual foundation model $f$, and inject them into DiT tokens:
   $$\mathbf{z'} = h_\phi(\mathbf{z} \oplus \mathcal{B}(\hat{\mathbf{e}}))$$
   where $\mathcal{B}$ denotes bilinear interpolation for spatial resolution alignment, and $h_\phi$ is an MLP fusion layer.

   Key detail — **L2 normalization**: The authors find that the magnitude of semantic representations $\mathbf{e}$ differs greatly from that of DiT tokens, causing training instability when directly concatenated. Simple L2 normalization $\hat{\mathbf{e}} = \mathbf{e}/\|\mathbf{e}\|_2$ resolves this, yielding a dramatic improvement (NYUv2 AbsRel from 22.5% to 4.3%, a 78% gain).

   Compatible VFMs: DINOv2, VGGT, MAE, Depth Anything v2—all yield significant improvements.

2. **Cascaded DiT Design (Cas-DiT)**

   Observation: In DiT, **early blocks handle global/low-frequency structure; later blocks handle high-frequency detail**.

   Based on this, a progressive patch strategy is designed:
   - First $N/2$ blocks (standard DiT): patch size = 16, token count $(H/16)\times(W/16)$ — low computational cost, focused on global structure.
   - Last $N/2$ blocks (SP-DiT): MLP expands to $(H/8)\times(W/8)$ tokens — equivalent to a smaller effective patch size, focused on fine-grained detail.

   Effect: 30% reduction in inference time (on RTX 4090) with further accuracy improvement.

3. **Flow Matching Generative Paradigm**

   Flow Matching is adopted as the generative backbone (rather than DDPM), learning a continuous transformation from noise to depth samples:
   $$\mathbf{x}_t = t \cdot \mathbf{x}_1 + (1-t) \cdot \mathbf{x}_0, \quad \mathbf{v}_t = \mathbf{x}_1 - \mathbf{x}_0$$
   The training objective is an MSE velocity field loss $\|\mathbf{v}_\theta - \mathbf{v}_t\|^2$.

### Loss & Training

- Loss: MSE velocity field loss + gradient matching loss
- Depth preprocessing: log transform followed by min-max normalization to $[-0.5, 0.5]$ using 2%–98% percentile clipping
- The 512-resolution model is trained solely on Hypersim (54K samples); the 1024-resolution model additionally incorporates four datasets
- Pure Transformer architecture with no convolutional layers

## Key Experimental Results

### Main Results: Zero-Shot Relative Depth on Five Benchmarks

| Method | Type | NYU AbsRel↓ | KITTI AbsRel↓ | ETH3D AbsRel↓ | ScanNet AbsRel↓ | DIODE AbsRel↓ |
|--------|------|------------|--------------|--------------|----------------|--------------|
| Marigold | Generative | 5.5 | 9.9 | 6.5 | 6.4 | 10.0 |
| Lotus | Generative | 5.4 | 8.5 | 5.9 | 5.9 | 9.8 |
| DepthAny. v2 | Discriminative | 4.5 | 7.4 | 13.1 | 6.5 | 6.6 |
| **Ours (512)** | Generative | **4.3** | 8.0 | **4.5** | **4.5** | **7.0** |
| **Ours (1024)** | Generative | **4.1** | **7.0** | **4.3** | **4.6** | **6.8** |

### Ablation Study

| Method | NYU AbsRel↓ | ScanNet AbsRel↓ | Inference Time (s) |
|--------|------------|----------------|-------------------|
| DiT (vanilla) | 22.5 | 25.7 | 0.19 |
| SP-DiT | 4.8 | 6.2 | 0.20 |
| SP-DiT + Cas-DiT | **4.3** | **4.5** | **0.14** |

SP-DiT yields a 78% improvement (NYUv2); Cas-DiT further improves accuracy while reducing inference time by 30%.

### Edge-Aware Point Cloud Evaluation

| Method | Chamfer Dist↓ |
|--------|--------------|
| Depth Anything v2 | 0.18 |
| Marigold | 0.17 |
| Depth Pro | 0.14 |
| GT (VAE) | 0.12 |
| **Ours** | **0.08** |

GT (VAE)—i.e., the ground-truth depth encoded and decoded through a VAE—still yields a CD of 0.12, demonstrating that VAE compression itself is the root cause of flying pixels.

### Key Findings

- Pixel-space diffusion with SP-DiT transforms performance from "nearly unusable" (22.5% AbsRel) to "state-of-the-art" (4.3%)—semantic prompting is the key enabler.
- L2 normalization appears simple but has a substantial effect—it resolves the magnitude mismatch between VFM features and DiT tokens.
- All tested VFMs (MAE / DINOv2 / VGGT / DAv2) yield significant performance gains.
- Training from scratch (without relying on SD pretrained weights) using only synthetic data achieves excellent generalization.

## Highlights & Insights

1. **Completely circumvents the VAE bottleneck**—virtually all existing generative depth models are constrained by VAE information loss; this work operates directly in pixel space.
2. The cascaded DiT's "global-first, local-second" design aligns with the hierarchical structure of human visual perception.
3. An **edge-aware point cloud evaluation metric** is proposed, filling the gap left by existing metrics that fail to reflect flying pixel artifacts.
4. Pure Transformer architecture with no convolutions—minimalist design with strong performance.

## Limitations & Future Work

- Multi-step diffusion inference is slower than discriminative models (DepthAny v2: 18 ms vs. PPD: 140 ms); the lightweight PPD-Small variant at 40 ms provides partial mitigation.
- Temporal consistency is lacking when applied to video (inter-frame flickering).
- The method addresses only relative depth; metric depth estimation requires additional adaptation.

## Related Work & Insights

- The semantic prompting mechanism of SP-DiT is applicable to other pixel-space generation tasks (e.g., surface normal estimation, optical flow).
- The cascaded patch strategy suggests that Transformers need not operate at a uniform resolution throughout.
- Comparison with REPA demonstrates that explicit semantic feature concatenation substantially outperforms implicit alignment (REPA AbsRel 17.6 vs. SP-DiT 4.3).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First successful realization of diffusion-based depth estimation in pixel space; SP-DiT and Cas-DiT are novel and effective designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five benchmarks, multi-VFM ablations, edge-aware evaluation, REPA comparison, and a lightweight variant.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is clear, ablations are convincing, and figures are high quality.
- **Value**: ⭐⭐⭐⭐⭐ — Addresses a core pain point in generative depth estimation with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](../../ICCV2025/3d_vision/jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[NeurIPS 2025\] Jasmine: Harnessing Diffusion Prior for Self-Supervised Depth Estimation](jasmine_harnessing_diffusion_prior_for_self-supervised_depth_estimation.md)
- [\[ICCV 2025\] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation](../../ICCV2025/3d_vision/simulating_dual-pixel_images_from_ray_tracing_for_depth_estimation.md)
- [\[CVPR 2026\] Fusion of Depth and Semantics for Probabilistic Floorplan Localization](../../CVPR2026/3d_vision/fusion_of_depth_and_semantics_for_probabilistic_floorplan_localization.md)
- [\[NeurIPS 2025\] Motion4D: Learning 3D-Consistent Motion and Semantics for 4D Scene Understanding](motion4d_learning_3d-consistent_motion_and_semantics_for_4d_scene_understanding.md)

</div>

<!-- RELATED:END -->
