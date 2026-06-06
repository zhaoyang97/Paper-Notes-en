---
title: >-
  [Paper Note] Towards Source-Aware Object Swapping with Initial Noise Perturbation
description: >-
  [CVPR 2026][Model Compression][Object Swapping] This paper proposes SourceSwap, which generates high-quality pseudo-paired data from single images via frequency-separated initial noise perturbation…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "Object Swapping"
  - "Diffusion Models"
  - "Initial Noise Perturbation"
  - "Self-Supervised"
  - "Cross-Object Alignment"
date: 2026-05-08
content_hash: b6730b1fb453f7a7
---

# Towards Source-Aware Object Swapping with Initial Noise Perturbation

**Conference**: CVPR 2026
**arXiv**: [2602.23697](https://arxiv.org/abs/2602.23697)  
**Code**: None  
**Area**: Model Compression
**Keywords**: Object Swapping, Diffusion Models, Initial Noise Perturbation, Self-Supervised, Cross-Object Alignment

## TL;DR

This paper proposes SourceSwap, which generates high-quality pseudo-paired data from single images via frequency-separated initial noise perturbation, and employs a source-aware dual U-Net architecture to learn cross-object alignment, enabling zero-shot, per-object-fine-tuning-free high-fidelity object swapping.

## Background & Motivation

Object swapping aims to replace a source object in a scene with a reference object, subject to three criteria: object fidelity, scene fidelity, and object-scene harmony.

Limitations of prior work: (1) Test-time fine-tuning methods (DreamEdit, PhotoSwap) require per-object training, resulting in slow inference; (2) Learning-based inpainting methods (AnyDoor, MimicBrush) rely on video/multi-view pseudo-paired data, suffering from blurriness and same-object bias; (3) All existing methods mask out the source object during training, forcing the model to infer object state from background context alone, which prevents learning cross-object alignment.

Core Insight: Retaining the complete source image enables the model to directly learn the alignment relationship between two distinct objects.

## Method

### Overall Architecture

Two stages: (1) initial noise perturbation for pseudo-pair generation → (2) source-aware dual U-Net training.

### Key Designs

#### 1. Frequency-Separated Initial Noise Perturbation

DDIM inversion is applied to the source image $I_s$ to obtain $z_T$, which is then decomposed via FFT into low-frequency $z_T^L$ and high-frequency $z_T^H$ components (cutoff frequency 0.3). Within the source object mask, the high-frequency components are spatially permuted via index shuffling:

$$\hat{z}_T^H[c,k] = \tilde{z}_T^H[c,\pi(k)]$$

Permutation rather than Gaussian resampling preserves the marginal distribution and energy, facilitating seamless blending. Low-frequency components are fixed to maintain structural consistency, while high-frequency permutation alters appearance (color/texture/material). Noise outside the mask remains unchanged.

#### 2. Source-Aware Dual U-Net

The upper branch (reference U-Net) extracts dense features from the reference object without noise injection, yielding cleaner detail representations. The lower branch (denoising U-Net) receives as conditions the complete source image (unmasked), the bounding box mask, and the perturbed source image. The Key/Value tensors from both U-Nets are concatenated at each cross-attention block.

Key design: the source image serves as input, and the perturbed image serves as condition, with the training target being the reconstruction of the original image.

#### 3. Iterative Refinement

The output of each iteration is used as the source image for the next: $I_t^{(k)} = \mathcal{D}(I_r, I_s^{(k)})$. In practice, $k=2$ iterations yield significant improvements in color and texture detail.

### Loss & Training

Built upon SD v1.5 and SD Inpainting v1.5, trained on 40K single-image samples for 10K iterations, requiring approximately 8 hours on a single A100 GPU. The reference U-Net timestep is fixed at 0; VAE and text encoder are frozen.

## Key Experimental Results

### Main Results

| Evaluation Dimension | Metric | SourceSwap Performance |
|---|---|---|
| Object Fidelity | DreamSim ↓ | Pareto-optimal frontier |
| Scene Fidelity | LPIPS ↓ | Pareto-optimal frontier |
| Harmony | MLLM Preference Rate | >62% over all baselines |

### Inference Efficiency

| Method | Inference Time / Sample |
|---|---|
| PhotoSwap | 128.85s + 751.97s pre-training |
| DiptychPrompt | 124.63s |
| AnyDoor | 11.01s |
| **SourceSwap (2 rounds)** | **4.41s** |

### Ablation Study

| Configuration | Effect |
|---|---|
| Without source awareness | Incorrect spatial relationships (floating backpack) |
| Data augmentation only, no perturbation | Model collapse |
| Permuting all frequency components | Structural distortion |
| Permuting low-frequency only | Insufficient appearance variation |
| Gaussian noise resampling | Pasting artifacts, viewpoint conflicts |

### Key Findings

- 40K single-image samples suffice to achieve strong performance, 1–2 orders of magnitude less than AnyDoor (410K) and MimicBrush (10M).
- Learning-based methods consistently outperform training-free methods; task-specific data construction is the key factor.

## Highlights & Insights

1. Frequency-separated noise perturbation is minimal yet effective — requiring only FFT and local permutation.
2. Removing the source mask is a counter-intuitive yet critical design choice — the complete source image actually facilitates cross-object alignment.
3. Training data requirements are 2–3 orders of magnitude smaller than comparable methods.

## Limitations & Future Work

1. The method is built on SD v1.5; upgrading to a stronger backbone model could further improve performance.
2. Object swapping under extreme shape discrepancies may yield limited results.
3. The diversity of perturbations is constrained by the expressiveness of the frequency-separation operation.

## Related Work & Insights

- The approach of manipulating the initial noise space is transferable to tasks such as layout control and style transfer.
- Compared to AnyDoor/MimicBrush: avoids quality issues associated with video-paired training data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Frequency-separated noise perturbation combined with source-aware design is conceptually novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons and sufficient ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated.
- Value: ⭐⭐⭐⭐ A practical zero-shot object swapping solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unlocking ImageNet's Multi-Object Nature: Automated Large-Scale Multilabel Annotation](unlocking_imagenets_multi-object_nature_automated_large-scale_multilabel_annotat.md)
- [\[ICCV 2025\] SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](../../ICCV2025/model_compression/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)
- [\[NeurIPS 2025\] Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization](../../NeurIPS2025/model_compression/find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)
- [\[CVPR 2026\] MARVO: Marine-Adaptive Radiance-aware Visual Odometry](marvo_marine-adaptive_radiance-aware_visual_odometry.md)
- [\[CVPR 2026\] Enhancing Mixture-of-Experts Specialization via Cluster-Aware Upcycling](enhancing_mixture_of_experts_specialization_via_cluster_aware_upcycling.md)

</div>

<!-- RELATED:END -->
