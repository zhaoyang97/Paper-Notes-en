---
title: >-
  [Paper Note] E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training
description: >-
  [CVPR 2026][3D Vision][Self-supervised pre-training] E-RayZer is the first truly self-supervised feed-forward 3D Gaussian reconstruction model. It replaces RayZer's implicit latent scene representation with explicit 3D Gaussians, and incorporates a visual-overlap-based curriculum learning strategy. Under zero 3D annotation conditions, it learns geometrically grounded 3D-aware representations, drastically outperforming RayZer on pose estimation (RPA@5° from ≈0 to 90.8%). On downstream 3D tasks under frozen-backbone probing, it significantly surpasses mainstream pre-trained models such as DINOv3 and CroCo v2, and even rivals the supervised VGGT.
tags:
  - CVPR 2026
  - 3D Vision
  - Self-supervised pre-training
  - 3D Gaussian splatting
  - multi-view reconstruction
  - pose estimation
  - visual representation learning
date: 2026-05-08
content_hash: e953bb4f4e2ec14b
---

# E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training

**Conference**: CVPR 2026
**arXiv**: [2512.10950](https://arxiv.org/abs/2512.10950)
**Code**: [qitaozhao.github.io/E-RayZer](https://qitaozhao.github.io/E-RayZer)
**Area**: 3D Vision
**Keywords**: Self-supervised pre-training, 3D Gaussian splatting, multi-view reconstruction, pose estimation, visual representation learning

## TL;DR

E-RayZer is the first truly self-supervised feed-forward 3D Gaussian reconstruction model. It replaces RayZer's implicit latent scene representation with explicit 3D Gaussians, and incorporates a visual-overlap-based curriculum learning strategy. Under zero 3D annotation conditions, it learns geometrically grounded 3D-aware representations, drastically outperforming RayZer on pose estimation (RPA@5° from ≈0 to 90.8%). On downstream 3D tasks under frozen-backbone probing, it significantly surpasses mainstream pre-trained models such as DINOv3 and CroCo v2, and even rivals the supervised VGGT.

## Background & Motivation

- **Background**: Self-supervised pre-training has driven rapid progress in foundation models for text, 2D images, and video. However, learning 3D-aware representations from multi-view images remains severely underexplored.
- **Limitations of Prior Work**: Dominant 3D vision models rely on pseudo-labels from SfM systems (e.g., COLMAP) for fully supervised training, which is inherently inefficient, imprecise, and unscalable. The predecessor RayZer attempts self-supervised 3D learning via latent-space novel view synthesis, but suffers from a fundamental flaw: its three modules—camera estimation, implicit scene reconstruction, and Transformer-based rendering—are jointly learned in latent space without 3D inductive bias. This allows the model to achieve high-quality synthesis through "shortcut solutions" such as video interpolation. The resulting pose space is neither interpretable nor physically meaningful, as evidenced by near-zero pose estimation accuracy (RPA@5° ≈ 0).
- **Key Challenge**: Without geometric grounding, latent-space self-supervised models can bypass genuine 3D understanding via frame-order interpolation.
- **Goal**: Introduce physically meaningful 3D inductive bias into self-supervised multi-view learning while preserving scalability.
- **Key Insight**: Replace implicit representations with explicit 3D Gaussians to force the model to reason about true 3D geometry through differentiable physical rendering, and design a fine-grained curriculum learning strategy to address the convergence difficulties introduced by explicit 3D.

## Method

### Overall Architecture

E-RayZer takes $V$ multi-view images as input and proceeds through three stages:

1. **Camera parameter prediction**: A multi-view Transformer $f_\theta^{\text{cam}}$ predicts intrinsics $K$ and extrinsics $T$ for all input images.
2. **Explicit 3D reconstruction**: Images are split into reference set $\mathcal{I}_{\text{ref}}$ and target set $\mathcal{I}_{\text{tgt}}$; pixel-aligned 3D Gaussians $\mathcal{G}$ are predicted from reference views.
3. **Self-supervised rendering**: The 3D Gaussians are rendered using self-predicted target-view camera parameters, and a photometric loss is computed against the ground-truth target images.

Training uses 10 input images, with 5 as reference views and 5 as target views. The entire process requires zero 3D annotations.

### Key Designs

1. **Explicit 3D Gaussian Scene Reconstruction (replacing RayZer's implicit representation)**
   - **Function**: Directly predict pixel-aligned 3D Gaussians from reference views as an explicit geometric scene representation.
   - **Mechanism**: A scene Transformer $f_{\psi'}^{\text{scene}}$ encodes posed reference views into multi-view aggregated latent tokens $\mathbf{s}_{\text{ref}}$; a lightweight decoder $f_\omega^{\text{gauss}}$ (single linear layer) then decodes each pixel token into 3D Gaussian parameters—ray distance $d_i$, orientation quaternion $\mathbf{q}_i$, spherical harmonics coefficients $\mathbf{C}_i$, scale $\mathbf{s}_i$, and opacity $\alpha_i$.
   - **Design Motivation**: 3D Gaussians support closed-form differentiable rendering (via a modified gsplat supporting gradient backpropagation through intrinsics $K$), eliminating the need for a learned Transformer renderer (removing RayZer's $f_\phi^{\text{rend}}$). Attention complexity is also reduced from $\mathcal{O}((K_{\text{ref}}hw + n_z)^2)$ to $\mathcal{O}((K_{\text{ref}}hw)^2)$.

2. **Eliminating View Interpolation Shortcuts**
   - **Function**: Prevent the model from learning frame-order-based video interpolation instead of genuine 3D understanding.
   - **Mechanism**: (a) Completely remove RayZer's image index embeddings—the primary cause of interpolation shortcuts. (b) Adopt a VGGT-style local-global alternating attention Transformer, where local attention boundaries naturally define image-camera associations. (c) Use pairwise pose prediction: canonical and target view camera tokens are concatenated to regress relative poses, removing the need to distinguish different types of camera/register tokens.
   - **Design Motivation**: RayZer's uninterpretable pose space stems from image index embeddings providing strong frame-order cues, biasing the model toward interpolation rather than geometric reasoning.

3. **Visual-Overlap-Based Curriculum Learning**
   - **Function**: Address the convergence failure of training explicit 3D from scratch, while adaptively aligning heterogeneous data sources.
   - **Mechanism**: A visual overlap profile $O_u(\Delta t)$ is precomputed for each training sequence (via average pairwise overlap over uniformly sampled frame triplets). During training, an overlap lower bound is linearly annealed: $o(s) = s \cdot o_{\min} + (1-s) \cdot o_{\max}$, transitioning from high-overlap (easy) to low-overlap (hard) samples.
   - Two overlap metrics are used: semantic overlap (DINOv2 cosine similarity, unsupervised) and geometric overlap (UFM co-visibility, trained with 3D annotations). Experiments show comparable performance between the two.
   - **Design Motivation**: RayZer's fixed frame-interval strategy is only a coarse proxy for overlap—the same interval can correspond to vastly different visual overlaps across sequences, and cannot adapt to heterogeneous data sources.

### Loss & Training

- **Photometric self-supervised loss**: $\mathcal{L} = \sum \text{MSE}(I, \hat{I}) + \lambda \cdot \text{Percep}(I, \hat{I})$, where Percep denotes perceptual loss.
- **Architecture**: patch size 16, image resolution 256, 8 layers each for camera and scene Transformers (1 global attention + 1 frame attention per layer), feature dimension 768, 12 attention heads.
- **Training setup**: 8×A100 GPUs, global batch size 192 (24 per GPU), 152K iterations (~198 hours).
- **Learning rate**: Linear warmup to 4e-4 over 3K steps, cosine decay to 0.
- **Curriculum schedule**: Linear progression over the first 86K steps; geometric overlap from 1.0→0.5, semantic overlap from 1.0→0.75.
- **Optimizer**: AdamW ($\beta_1$=0.9, $\beta_2$=0.95), gradient clipping 1.0, steps skipped when gradient norm >5.0.
- **7-dataset sampling ratio**: DL3DV 1.0, CO3Dv2 0.25, RE10K 0.5, MVImgNet 0.25, ARKitScenes 0.5, WildRGB-D 0.25, ACID 0.5.

## Key Experimental Results

### Main Results: Pose Estimation & Novel View Synthesis (Tab. 1)

Comparison against self-supervised/semi-supervised methods. E-RayZer and RayZer are trained fully self-supervised from scratch; SPFSplat is initialized with supervised MASt3R.

| Method | Training Data | WildRGB-D PSNR↑ | WildRGB-D @5°↑ | WildRGB-D @15°↑ | DL3DV @5°↑ | DL3DV @15°↑ |
|--------|--------------|-----------------|----------------|-----------------|------------|-------------|
| SPFSplat | RE10K+extra | 16.7 | 31.5 | 58.0 | 19.5 | 40.6 |
| E-RayZer | RE10K | 21.0 | 40.3 | 89.4 | 21.2 | 55.0 |
| RayZer | DL3DV | 25.9 | 0.0 | 0.2 | 0.0 | 0.6 |
| E-RayZer | DL3DV | 24.3 | 84.5 | 98.4 | 72.0 | 88.4 |
| RayZer | 7 datasets | 26.7 | 0.2 | 9.3 | 0.0 | 1.9 |
| **E-RayZer** | **7 datasets** | **24.9** | **90.8** | **98.6** | **59.9** | **82.9** |

E-RayZer decisively outperforms RayZer on pose estimation (from ≈0% to 60–90%), while achieving comparable NVS quality (PSNR slightly lower by ~2 dB, as RayZer overfits to interpolation rather than true 3D).

### Comparison with Supervised VGGT (Tab. 2, DL3DV training)

| Method | Supervision | DL3DV @5° | RE10K @5° | WildRGB-D @5° | BlendedMVS @5° | NAVI @5° | ScanNet++ @5° |
|--------|------------|-----------|-----------|---------------|----------------|----------|---------------|
| E-RayZer | Self-supervised | 72.0 | 83.0 | 51.1 | 22.9 | 20.7 | 7.7 |
| VGGT* | Supervised | 79.6 | 80.4 | 32.5 | 17.0 | 14.3 | 6.7 |
| VGGT*+E-RayZer init | Supervised | **87.3** | **85.3** | **56.2** | **29.2** | **26.9** | **14.3** |

Self-supervised E-RayZer surpasses supervised VGGT* on multiple OOD datasets (especially under the strict RPA@5° metric), and VGGT* initialized with E-RayZer achieves the best overall results.

### Downstream Task Probing (Tab. 3, Frozen-backbone)

| Pre-training | ScanNet++ AbsRel↓ | ScanNet++ δ<1.25↑ | ScanNet++ @5°↑ | BlendedMVS AbsRel↓ | BlendedMVS @5°↑ |
|-------------|-------------------|-------------------|----------------|---------------------|-----------------|
| DINOv2 | 0.193 | 74.9 | 0.8 | 0.366 | 1.1 |
| DINOv3 | 0.201 | 73.2 | 0.4 | 0.397 | 1.2 |
| CroCo v2 | 0.203 | 73.0 | 1.4 | 0.412 | 1.6 |
| VideoMAE V2 | 0.175 | 76.3 | 0.1 | 0.371 | 1.0 |
| RayZer | 0.161 | 79.3 | 4.7 | 0.351 | 16.7 |
| **E-RayZer** | **0.116** | **87.1** | **13.8** | **0.245** | **26.5** |

Under the frozen-backbone setting, E-RayZer substantially outperforms all baselines: depth estimation AbsRel is 40% lower than DINOv2 (0.116 vs. 0.193), and pose @5° is 17× higher (13.8 vs. 0.8), demonstrating strong 3D spatial awareness in the learned features.

### Ablation Study: Curriculum Learning Strategy (Tab. 6, 7 datasets)

| Curriculum Strategy | PSNR↑ | RPA@5°↑ | RPA@15°↑ | RPA@30°↑ |
|--------------------|-------|---------|----------|----------|
| No curriculum | 15.9 | 2.1 | 21.6 | 40.7 |
| Frame-interval curriculum | 19.1 | 43.8 | 72.1 | 82.9 |
| Semantic overlap curriculum | 19.7 | 58.7 | 81.0 | 89.8 |
| Geometric overlap curriculum | 19.7 | 59.9 | 82.9 | 90.2 |

Training without curriculum nearly collapses (RPA@5° only 2.1%). Frame-interval curriculum is effective but insufficient. Visual overlap curriculum achieves the best results across all metrics. The two overlap variants perform comparably, indicating that unsupervised semantic overlap can substitute for geometric overlap requiring 3D annotations.

### Key Findings

- **Self-supervised and supervised learning are complementary**: Initializing VGGT* with E-RayZer yields consistent improvements (Tab. 2, last row), suggesting the two paradigms learn highly complementary knowledge even when trained on the same data.
- **Data diversity > data volume**: DL3DV (high quality) alone outperforms RE10K; 7-dataset mixture training achieves the best generalization; object-centric datasets require downsampled sampling ratios.
- **Cost of explicit 3D**: NVS quality is slightly lower than RayZer (PSNR ~2 dB lower), which precisely reflects RayZer's overfitting to video interpolation rather than true 3D understanding.
- **Slightly weaker on optical flow**: Tab. 4 shows E-RayZer has EPE 1.254 vs. RayZer's 1.105 on pairwise flow estimation; implicit representations have a natural advantage for low-level motion estimation.

## Highlights & Insights

- The shift from "self-supervised novel view synthesis" to "self-supervised 3D reconstruction" represents a paradigm change: explicit geometric constraints eliminate shortcut solutions, transforming the pose space from uninterpretable to geometrically grounded.
- The curriculum learning strategy is elegantly designed: visual overlap serves as a unified difficulty metric across heterogeneous data sources, automatically adapting to diverse data distributions—more principled and scalable than manually specified frame intervals.
- Self-supervised E-RayZer surpasses some supervised models under strict metrics (RPA@5°), demonstrating that large-scale self-supervision alone can produce geometrically grounded 3D understanding—data diversity and quality are the true drivers of scalability.
- Modifying gsplat to support gradient backpropagation through intrinsics $K$ is a key engineering contribution that enables end-to-end differentiability of the entire pipeline.

## Limitations & Future Work

- Only static scenes are supported, limiting the scale of usable training data—extending to dynamic scenes to leverage general-purpose video is the most important future direction.
- The curriculum strategy assumes continuous video frames with relatively uniform camera motion; performance may degrade for sparse images or large viewpoint changes.
- NVS quality is slightly inferior to the implicit method RayZer, which may be a drawback in applications requiring high-quality rendering.
- Experiments are limited to ViT-Base scale; scaling behavior with larger models remains to be explored.

## Related Work & Insights

- **vs. VGGT** (supervised 3D): E-RayZer demonstrates that self-supervision can match supervised methods in geometric understanding, and the two paradigms are complementary—self-supervised pre-training followed by supervised fine-tuning may be the optimal strategy.
- **vs. DINOv3/CroCo v2** (2D pre-training): The large gap under frozen-backbone probing reveals that 2D visual features lack genuine 3D spatial awareness.
- **vs. SPFSplat** (semi-supervised 3DGS): Even with MASt3R initialization (trained on 14 datasets with supervision), SPFSplat is comprehensively outperformed by E-RayZer, demonstrating the potential of self-supervised learning from scratch.
- The framework of explicit 3D + physical rendering as a self-supervised signal is general and can be extended to other 3D perception tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First truly self-supervised feed-forward 3D Gaussian reconstruction; a paradigm shift from implicit to explicit representations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 9 evaluation datasets, multi-task evaluation (NVS/pose/depth/optical flow), comprehensive ablations, fair comparison with VGGT, and scaling analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic; motivation naturally derived from RayZer's limitations; effective motivation figure for curriculum learning.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for self-supervised pre-training in 3D vision; the complementarity of self-supervised pre-training and supervised fine-tuning carries significant implications.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] 3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds](3d_sans_3d_scans_scalable_pre-training_from_video-generated_point_clouds.md)
- [\[CVPR 2026\] Regularizing INR with Diffusion Prior for Self-Supervised 3D Reconstruction of Neutron Computed Tomography Data](regularizing_inr_with_diffusion_prior_self-supervised_3d_reconstruction_of_neutr.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](../../ICCV2025/3d_vision/rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](../../ICCV2025/3d_vision/towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)

<!-- RELATED:END -->
