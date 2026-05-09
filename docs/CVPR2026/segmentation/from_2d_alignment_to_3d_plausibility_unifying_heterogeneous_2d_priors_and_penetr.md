---
title: >-
  [Paper Note] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction
description: >-
  [CVPR 2026][Segmentation][two-hand reconstruction] This work decouples two-hand reconstruction into 2D structural alignment (fusing keypoint, segmentation, and depth priors) and 3D spatial interaction alignment (a penetration-free diffusion model), achieving an MPJPE of 5.36 mm on InterHand2.6M and substantially outperforming the state of the art.
tags:
  - CVPR 2026
  - Segmentation
  - two-hand reconstruction
  - 2D prior fusion
  - diffusion model
  - penetration elimination
  - occlusion robustness
date: 2026-05-08
content_hash: f0695933d7e16c3a
---

# From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2503.17788](https://arxiv.org/abs/2503.17788)
**Code**: [Project Page](https://gaogehan.github.io/A2P/)
**Area**: Segmentation / 3D Hand Reconstruction
**Keywords**: two-hand reconstruction, 2D prior fusion, diffusion model, penetration elimination, occlusion robustness

## TL;DR

This work decouples two-hand reconstruction into 2D structural alignment (fusing keypoint, segmentation, and depth priors) and 3D spatial interaction alignment (a penetration-free diffusion model), achieving an MPJPE of 5.36 mm on InterHand2.6M and substantially outperforming the state of the art.

## Background & Motivation

Monocular two-hand reconstruction faces two core challenges: (1) complex poses and severe occlusions lead to **ambiguous 2D–3D correspondences**, and existing methods lack effective structural guidance; (2) **interpenetration** occurs frequently in two-hand interaction scenarios, yet no dedicated de-penetration mechanism exists in prior work.

Although vision foundation models (e.g., Sapiens) excel at keypoint detection, segmentation, and depth estimation, directly fine-tuning these large models is costly, and 2D priors are not fully reliable under occlusion. Meanwhile, diffusion models can capture interaction priors but require accurate observation alignment to avoid degenerate solutions.

The authors' core insight is that **2D alignment and 3D alignment should be handled separately**—first using heterogeneous 2D priors to align structure, then employing a generative model to eliminate interpenetration in 3D space.

## Method

### Overall Architecture

The pipeline consists of two stages: Stage 1 performs 2D multi-modal prior alignment, fusing keypoint, segmentation, and depth features from foundation models to guide hand parameter regression; Stage 2 applies a 3D penetration-free diffusion model that maps interpenetrating two-hand poses to physically plausible, collision-free configurations.

### Key Designs

1. **Fusion Alignment Encoder (FAE)**: A lightweight ResNet-50 encoder that distills three types of prior features from the Sapiens foundation model—keypoints $\mathbf{F}_k$, segmentation $\mathbf{F}_s$, and depth $\mathbf{F}_d$—during training. The fused feature $\mathbf{F}_p = \text{Proj}(\mathbf{F}_k, \mathbf{F}_s, \mathbf{F}_d)$ is unified through a learnable projection layer. The FAE aligns $\mathbf{F}_{fa}$ with $\mathbf{F}_p$ via MSE loss. **All foundation model encoders are removed at inference**, enabling encoder-free deployment—preserving multi-prior accuracy while significantly reducing computational overhead. This "distill during training, discard at inference" strategy is particularly elegant.

2. **Two-Hand Penetration-Free Diffusion Model**: A Transformer-based diffusion model conditioned on the interpenetrating two-hand MANO parameters $\mathbf{X}_c$, which learns a generative mapping from penetrating poses to collision-free configurations. Training data is constructed via two strategies: (a) penetrating predictions from a lower-performance model; (b) small perturbations applied to ground-truth poses until penetration occurs. The diffusion loss is $\mathcal{L}_{diffusion} = \|\mathbf{X}_0 - \mathcal{D}(\mathbf{X}_t, \mathbf{X}_c)\|_2$. At inference, IoU and penetration detection are performed first; diffusion is invoked only for samples with confirmed penetration, reducing unnecessary inference overhead.

3. **Collision Gradient Guidance**: Collision gradient guidance is introduced at each reverse diffusion step. Specifically, $\hat{\mathbf{X}}_0$ is obtained from $\mathbf{X}_{t-1}$ via DDIM sampling; mesh vertices are retrieved through the MANO model; collisions are detected using a hybrid distance–orientation criterion: neighboring vertex pairs are first identified by Chamfer distance $\mathbf{N}_{ij} = |\mathbf{V}_{t-1}^i - \mathbf{V}_c^j|^2$, and penetration is then assessed via normal cosine similarity $\cos(\theta_{ij})$. The collision loss uses the GMoF robust function, and $\hat{\mathbf{X}}_0$ is updated via gradient descent as $\hat{\mathbf{X}}_0 = \hat{\mathbf{X}}_0 - \lambda(\delta_i \mathcal{L}_{collision})$.

### Loss & Training

- **Hand regression loss** $\mathcal{L}_{hand}$: L1 supervision over MANO parameters, 3D/2.5D joint coordinates, and 3D relative translation
- **Prior alignment loss** $\mathcal{L}_{prior}(\mathbf{F}_p, \mathbf{F}_{fa})$: MSE distillation for the FAE
- **Total loss** $\mathcal{L}_{total} = \mathcal{L}_{hand} + \mathcal{L}_{prior}$
- The diffusion model is trained separately using an MDM-style diffusion process with 1,000 noise steps and cosine scheduling
- Training hardware: 4×A100; optimizer: AdamW, lr=1e-4, batch size=48

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (4DHands) | Gain |
|--------|------|------|----------|------|
| InterHand2.6M | MRRPE (mm) | **21.60** | 24.58 | -2.98 |
| InterHand2.6M | MPJPE (mm) | **5.36** | 7.49 | -2.13 |
| InterHand2.6M | MPVPE (mm) | **5.58** | 7.72 | -2.14 |
| HIC | MPJPE (mm) | **6.67** | 9.32 | -2.65 |
| HIC | MPVPE (mm) | **6.93** | 9.93 | -3.00 |

### Ablation Study

| Configuration | MPJPE | MPVPE | Note |
|------|---------|------|------|
| Baseline | 7.77 | 7.93 | No additional priors |
| + Key Points | 6.48 | 6.72 | 2D keypoint prior |
| + Segmentation | 6.19 | 6.34 | Segmentation prior added |
| + Depth Prior | 5.74 | 5.98 | Depth prior; significant improvement in Z direction |
| + Penetration-Free Diffusion | **5.36** | **5.58** | Full model |

### Key Findings

- The three 2D priors provide complementary improvements along XY and Z dimensions: keypoints primarily improve XY, while depth primarily improves Z
- The diffusion model yields more pronounced gains in IH (interacting hand) scenarios
- The method still substantially outperforms the state of the art on HIC (unseen data), demonstrating strong generalizability
- Training uses only a small set of datasets, far fewer than 4DHands (3 two-hand + 9 single-hand datasets)

## Highlights & Insights

- The **"distill during training, discard at inference"** FAE design is the paper's most elegant contribution—achieving multi-prior structural guidance at zero additional inference cost
- Penetration detection employs a dual distance–normal criterion, which is more robust than pure distance thresholding
- Framing de-penetration as a conditional generation task (rather than post-hoc optimization) better models the manifold of feasible interactions
- IoU detection gating avoids redundant diffusion inference on non-penetrating samples

## Limitations & Future Work

- The FAE depends on the quality of the Sapiens foundation model, whose priors may be unreliable under extreme occlusion
- Under severe motion blur, additional 2D information may become unreliable; the authors acknowledge that future work could incorporate temporal processing to mitigate this
- The diffusion model introduces additional inference latency; despite IoU gating, real-time performance remains limited (18 FPS with full model vs. 56 FPS without diffusion)
- The approach targets only two-hand scenarios and has not been extended to hand–object interaction or whole-body reconstruction
- Collision detection is based on mesh vertices, and its precision is bounded by mesh resolution

## Related Work & Insights

- **WHAM/TRAM**: Pioneering works that use 2D priors for whole-body reconstruction; this paper is the first to introduce this idea into the two-hand setting
- **InterHandGen**: Uses diffusion priors for two-hand generation but only as a regularization term; this paper directly models the de-penetration mapping, reducing penetration volume from 0.76 to 0.11 and penetration distance from 0.04 to 0.01
- **BUDDI**: Applies diffusion priors to two-person reconstruction; this paper adapts a similar approach at the two-hand scale
- **Zuo et al.**: Captures interaction priors with a VAE but relies on CNN-extracted interaction features and lacks strong geometric constraints
- Inspiration: The FAE "training distillation–inference discard" paradigm is generalizable to other tasks that require multi-modal priors yet demand lightweight inference (e.g., human pose estimation, hand–object interaction)

## Rating

- Novelty: ⭐⭐⭐⭐ First to unify three heterogeneous 2D priors for two-hand reconstruction; penetration-free diffusion modeling is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on three datasets with complete ablations and clear qualitative comparisons
- Writing Quality: ⭐⭐⭐⭐ Logically structured with well-motivated two-stage design
- Value: ⭐⭐⭐⭐ Significant contribution to two-hand reconstruction; the FAE design paradigm is broadly applicable

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](../../ICCV2025/segmentation/wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[CVPR 2026\] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models](occsam_bench_occlusion_robustness_segmentation.md)
- [\[AAAI 2026\] EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization](../../AAAI2026/segmentation/eagle_episodic_appearance-_and_geometry-aware_memory_for_unified_2d-3d_visual_qu.md)
- [\[CVPR 2026\] Making Training-Free Diffusion Segmentors Scale with the Generative Power](making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)
- [\[CVPR 2026\] 3M-TI: High-Quality Mobile Thermal Imaging via Calibration-free Multi-Camera Cross-Modal Diffusion](3m-ti_high-quality_mobile_thermal_imaging_via_calibration-free_multi-camera_cros.md)

<!-- RELATED:END -->
