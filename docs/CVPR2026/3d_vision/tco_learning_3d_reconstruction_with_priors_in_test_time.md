---
title: >-
  [Paper Note] Learning 3D Reconstruction with Priors in Test Time
description: >-
  [CVPR 2026][3D Vision][test-time optimization] This paper proposes Test-time Constrained Optimization (TCO), a framework that improves 3D reconstruction accuracy by treating available priors (camera poses, intrinsics…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "test-time optimization"
  - "3D reconstruction"
  - "multiview transformer"
  - "camera pose"
  - "LoRA"
date: 2026-05-08
content_hash: ad8368c0b39aacf7
---

# Learning 3D Reconstruction with Priors in Test Time

**Conference**: CVPR 2026
**arXiv**: [2604.03878](https://arxiv.org/abs/2604.03878)
**Code**: [https://github.com/cvlab-stonybrook/TCO](https://github.com/cvlab-stonybrook/TCO)
**Area**: 3D Reconstruction
**Keywords**: test-time optimization, 3D reconstruction, multiview transformer, camera pose, LoRA

## TL;DR

This paper proposes Test-time Constrained Optimization (TCO), a framework that improves 3D reconstruction accuracy by treating available priors (camera poses, intrinsics, depth) as output constraints optimized at inference time, without retraining or modifying the architecture of pretrained multiview Transformers.

## Background & Motivation

- **Background**: Feed-forward multiview Transformers (MVTs) such as DUSt3R, VGGT, and π³ can predict depth maps, camera poses, and intrinsics from multiple RGB images in a single forward pass.
- **Limitations of Prior Work**: These models accept only RGB inputs and cannot exploit additional priors (e.g., camera poses from COLMAP, depth from LiDAR). Existing prior-aware methods (e.g., Pow3R, MapAnything) incorporate priors by modifying the network architecture, making them tightly coupled to specific backbones and prior types — retraining is required whenever the backbone or prior type changes, resulting in poor flexibility and high computational cost.
- **Key Challenge**: Feeding priors as network inputs conflates the representation of priors with model architecture, preventing generalization across models and prior types.
- **Goal**: To develop a plug-and-play framework that leverages arbitrary priors at test time without architectural modification or retraining.
- **Key Insight**: Rather than feeding priors as inputs to the network, TCO treats them as constraints on the network's outputs, optimizing the network at inference time to satisfy these constraints.

## Method

### Overall Architecture

TCO leaves the pretrained MVT frozen and applies LoRA fine-tuning to the shared decoder at test time, optimizing a loss composed of a self-supervised objective and prior penalty terms.

### Key Designs

1. **Prior Penalty Terms**: Each available prior is formulated as a constraint on the corresponding predicted modality. For example, a camera pose prior yields $g = \|T_i - T_i^{prior}\| = 0$, and depth priors are handled analogously. These constraints act directly on MVT outputs rather than inputs.

2. **Self-Supervised Objective (Prediction Compatibility)**: A compatibility criterion is defined across multiview predictions — photometric or geometric losses are used to compare renderings from other views against each view's own predictions, ensuring consistency. This prevents the optimization from overfitting to the priors at the expense of overall 3D quality.

3. **LoRA Fine-tuning Strategy**: All prediction heads are frozen; only the shared decoder is fine-tuned via LoRA. This exploits cross-modal synergy — priors available for certain modalities can improve predictions for other modalities through the shared representation.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{self-supervised}} + \sum \mathcal{L}_{\text{prior penalty}}$$

The self-supervised objective may be a photometric loss (reprojection consistency) or a geometric loss (point cloud alignment). LoRA parameters are initialized from zero and converge rapidly.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (TCO) | Base Model | Gain |
|---------|--------|-----------|------------|------|
| ETH3D | Pointmap distance error | >50% reduction | image-only MVT | Significant |
| 7-Scenes | Pointmap distance error | >50% reduction | image-only MVT | Significant |
| NRGBD | Pointmap distance error | >50% reduction | image-only MVT | Significant |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Prior penalty only (no self-supervised) | Poor performance | Prone to overfitting priors |
| Self-supervised only (no priors) | Moderate improvement | Inferior to combined use |
| Full parameter fine-tuning | Unstable | LoRA is preferable |

### Key Findings

- TCO not only substantially outperforms image-only base models, but also surpasses prior-aware feed-forward methods that require retraining (Pow3R, MapAnything).
- TCO effectively utilizes partially available priors.
- The self-supervised objective is critical for preventing overfitting to the provided priors.

## Highlights & Insights

- **Plug-and-play design**: TCO can be applied to any pretrained MVT without architectural modification or retraining.
- The conceptual shift from "priors as inputs" to "priors as constraints" is particularly elegant.
- Fine-tuning only the shared decoder via LoRA cleverly exploits cross-modal synergy.
- The approach aligns well with the broader trend of test-time compute scaling.

## Limitations & Future Work

- Test-time optimization introduces additional inference latency.
- Optimization quality is directly dependent on prior quality; noisy priors may be detrimental.
- Current validation is primarily on indoor scenes; applicability to large-scale outdoor environments remains to be explored.

## Related Work & Insights

- TCO shares the spirit of test-time fine-tuning methods such as Test3R and TTT3R, but offers greater generality.
- The framework provides inspiration for other 3D tasks that require fusion of multimodal priors.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The framing of priors as constraints rather than inputs is a novel and principled contribution.
- **Technical Depth**: ⭐⭐⭐⭐ — The combination of self-supervised objectives, prior constraints, and LoRA is well-motivated and coherent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple datasets and prior types.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play nature ensures broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] BulletGen: Improving 4D Reconstruction with Bullet-Time Generation](bulletgen_improving_4d_reconstruction_with_bullet-time_generation.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[CVPR 2026\] AnchorSplat: Feed-Forward 3D Gaussian Splatting with 3D Geometric Priors](anchorsplat_feed-forward_3d_gaussian_splatting_with_3d_geometric_priors.md)

</div>

<!-- RELATED:END -->
