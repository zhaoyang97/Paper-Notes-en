---
title: >-
  [Paper Note] RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations
description: >-
  [CVPR 2026][3D Vision][3D Reconstruction] This paper proposes RnG, a unified feed-forward Transformer that leverages reconstruction-guided causal attention to treat KV-Cache as an implicit 3D representation, simultaneously achieving 3D reconstruction and novel-view RGBD generation from sparse unposed images, with inference speeds over 100× faster than diffusion-based methods.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Reconstruction
  - Novel View Synthesis
  - KV-Cache
  - Causal Attention
  - Feed-Forward Transformer
date: 2026-05-08
content_hash: 181504747ffe92c9
---

# RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations

**Conference**: CVPR 2026
**arXiv**: [2603.01194](https://arxiv.org/abs/2603.01194)
**Code**: [https://npucvr.github.io/RnG](https://npucvr.github.io/RnG)
**Area**: 3D Vision / Reconstruction & Generation
**Keywords**: 3D Reconstruction, Novel View Synthesis, KV-Cache, Causal Attention, Feed-Forward Transformer

## TL;DR

This paper proposes RnG, a unified feed-forward Transformer that leverages reconstruction-guided causal attention to treat KV-Cache as an implicit 3D representation, simultaneously achieving 3D reconstruction and novel-view RGBD generation from sparse unposed images, with inference speeds over 100× faster than diffusion-based methods.

## Background & Motivation

1. **Background**: Generalizable 3D reconstruction methods (DUSt3R, VGGT) can recover geometry of visible regions from sparse images but do not model unobserved regions. Novel view synthesis (NVS) methods (LVSM) can generate images from unseen viewpoints but lack consistent 3D structure.
2. **Limitations of Prior Work**: Reconstruction methods produce incomplete outputs (visible regions only); NVS methods lack 3D consistency or require known camera poses. Matrix3D unifies both tasks but its diffusion-based design results in extremely slow inference (27 s/view).
3. **Key Challenge**: How can a single model unify reconstruction and generation while maintaining real-time inference capability?
4. **Goal**: To exploit the latent 3D understanding in reconstruction foundation models and activate it explicitly via neural rendering.
5. **Key Insight**: Transferring reconstruction priors to generation—rather than the conventional direction of using generative priors to assist reconstruction—constitutes a reverse knowledge transfer.
6. **Core Idea**: A causal attention mask prevents source-view tokens from being influenced by target-view tokens, allowing the KV-Cache to naturally serve as a reusable implicit 3D representation.

## Method

### Overall Architecture

Source-view images are tokenized via DINO; target views are encoded as Plücker ray maps. All tokens are processed through 24 alternating global/frame attention layers. Source-view tokens are used for pose estimation, while target-view tokens are decoded via a DPT head to produce RGB images and point maps.

### Key Designs

1. **Reconstruction-Guided Causal Attention**:

    - **Function**: Decouples reconstruction and generation at the attention level.
    - **Mechanism**: A binary mask $M$ is introduced to prevent source-view queries from attending to target-view keys. Source-view tokens attend only to source views (reconstruction), while target-view tokens attend to all views (generation). Both tasks share network parameters but are separated via the attention mask.
    - **Design Motivation**: Reconstruction should guide generation, but generation should not interfere with reconstruction. This design ensures consistent source-view reconstruction results regardless of the target views provided.

2. **KV-Cache as Implicit 3D Representation**:

    - **Function**: Enables efficient two-stage inference.
    - **Mechanism**: Causal attention makes source-view token processing independent of target views. The K/V tokens of source views can thus be cached after the reconstruction stage (~0.2 s); subsequent generation for any target viewpoint requires only a forward pass over target tokens with cached K/V retrieval (<0.1 s).
    - **Design Motivation**: The KV-Cache mechanism makes multi-view generation for the same scene highly efficient, analogous to autoregressive inference in language models.

3. **Reconstruction-Prior-Driven Generation**:

    - **Function**: Leverages 3D reconstruction knowledge to improve novel-view generation quality.
    - **Mechanism**: VGGT pretrained weights and architecture are inherited. Separate RGB and point-map heads decode target-view appearance and geometry respectively. Accumulating point maps across multiple target views yields a complete 3D structure, functioning as a "virtual 3D scanner."
    - **Design Motivation**: Experiments confirm that transferring reconstruction priors to generation is both feasible and effective, and more efficient than transferring from diffusion priors.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{RGB} + \lambda_{pmap}\mathcal{L}_{pmap} + \lambda_c\mathcal{L}_{cam}$. The RGB loss combines MSE and perceptual loss; the point-map loss is uncertainty-weighted L1; the pose loss is Huber loss. Training runs for 40K steps on 8 × A800 GPUs.

## Key Experimental Results

### Main Results

| Method | Pose RA@5↑ | Src Depth Rel↓ | NVS Depth Rel↓ | NVS PSNR↑ | 3D CD↓ |
|--------|-----------|----------------|----------------|-----------|--------|
| RnG (unposed) | 85.1 | 0.584 | 0.717 | 26.28 | 0.0067 |
| VGGT (unposed) | 74.2 | 5.96 | - | - | 0.0260 |
| Matrix3D (unposed) | 43.8 | 9.43 | 9.96 | 18.74 | 0.0670 |
| LVSM (posed) | - | - | - | 27.52 | - |

### Ablation Study

| Configuration | NVS PSNR | Notes |
|---------------|----------|-------|
| Train from scratch (15K) | 20.78 | No reconstruction prior |
| Pretrained initialization (15K) | 24.86 | Reconstruction prior effective |
| Full attention (no causal mask) | 24.86 | Comparable accuracy but no KV-Cache |
| With KV-Cache | 85 ms inference | Without KV-Cache: 213 ms |

### Key Findings

- Unposed RnG approaches the best posed method (LVSM) on NVS while additionally providing pose estimation and 3D geometry.
- The importance of reconstruction priors: pretrained initialization vs. training from scratch yields a gap of 4+ dB.
- Causal attention incurs no accuracy loss while achieving 2.5× inference speedup (213 ms → 85 ms).
- RnG inference is 300× faster than Matrix3D (85 ms vs. 27 s).

## Highlights & Insights

- **Reverse Knowledge Transfer**: The first systematic demonstration that reconstruction priors can effectively bootstrap generation, challenging the conventional paradigm of using generative priors to assist reconstruction.
- **Novel Interpretation of KV-Cache**: The KV-Cache from language models is reinterpreted as an implicit 3D representation—a conceptually elegant formulation.
- **"Virtual 3D Scanner"**: Complete 3D structure can be obtained by accumulating point maps from multi-view queries, without requiring explicit 3D reconstruction algorithms.

## Limitations & Future Work

- The method lacks fine-grained texture detail compared to diffusion-based approaches; incorporating image generation pretraining may address this.
- The world origin definition depends on the intersection of input viewpoints, limiting practical deployment on handheld devices.
- Accumulating 3D geometry from multiple views may introduce noise and inconsistencies.

## Related Work & Insights

- **vs. VGGT**: VGGT reconstructs only visible regions; RnG extends it with generative capability to produce complete 3D.
- **vs. Matrix3D**: Both are unified models, but RnG is a deterministic feed-forward approach (real-time), whereas Matrix3D relies on diffusion (27 s).
- **vs. LVSM**: LVSM requires known poses and provides no 3D geometry; RnG jointly estimates poses, geometry, and appearance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of KV-Cache as a 3D representation is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-task, multi-metric evaluation with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation and intuitive architecture diagrams.
- Value: ⭐⭐⭐⭐⭐ Establishes an efficient paradigm for unifying 3D reconstruction and generation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PR-IQA: Partial-Reference Image Quality Assessment for Diffusion-Based Novel View Synthesis](pr-iqa_partial-reference_image_quality_assessment_for_diffusion-based_novel_view.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)
- [\[CVPR 2026\] Learning 3D Reconstruction with Priors in Test Time](tco_learning_3d_reconstruction_with_priors_in_test_time.md)
- [\[CVPR 2026\] FF3R: Feedforward Feature 3D Reconstruction from Unconstrained Views](ff3r_feedforward_feature_3d_reconstruction_from_unconstrained_views.md)
- [\[CVPR 2026\] Sampling-Aware 3D Spatial Analysis in Multiplexed Imaging](sampling-aware_3d_spatial_analysis_in_multiplexed_imaging.md)

<!-- RELATED:END -->
