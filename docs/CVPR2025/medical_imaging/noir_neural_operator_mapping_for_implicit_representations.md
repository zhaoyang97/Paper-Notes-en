---
title: >-
  [Paper Note] NOIR: Neural Operator Mapping for Implicit Representations
description: >-
  [CVPR2025][Medical Imaging][Implicit Neural Representations] NOIR reformulates medical image computation tasks as operator learning problems between continuous function spaces. It embeds discrete medical signals into a continuous function space via Implicit Neural Representations (INR), and then learns mappings between functions using Neural Operators (NO), achieving resolution-agnostic segmentation, shape completion, image translation, and synthesis.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "Implicit Neural Representations"
  - "Neural Operator"
  - "Resolution-agnostic"
  - "Medical Image Segmentation"
  - "Image Translation"
date: 2026-05-08
content_hash: 4fba9d570a97425e
---

# NOIR: Neural Operator Mapping for Implicit Representations

**Conference**: CVPR2025  
**arXiv**: [2603.13118](https://arxiv.org/abs/2603.13118)  
**Code**: [GitHub](https://github.com/Sidaty1/NOIR-io)  
**Area**: Medical Image  
**Keywords**: Implicit Neural Representations, Neural Operator, Resolution-agnostic, Medical Image Segmentation, Image Translation

## TL;DR

NOIR reformulates medical image computation tasks as operator learning problems between continuous function spaces. It embeds discrete medical signals into a continuous function space via Implicit Neural Representations (INR), and then learns mappings between functions using Neural Operators (NO), achieving resolution-agnostic segmentation, shape completion, image translation, and synthesis.

## Background & Motivation

- Current deep learning methods for medical image computation are almost entirely based on discrete grids (pixels/voxels), making them highly sensitive to voxel spacing, interpolation methods, and resampling artifacts.
- Medical images naturally originate from continuous physical domains and frequently undergo resampling and transformations in clinical workflows; discrete representations introduce inconsistencies.
- Existing Neural Operator methods (such as FNO) are limited to regular grids and suffer from aliasing errors.
- A resolution-agnostic, task-agnostic, and general framework is needed to unify various downstream medical imaging tasks.

## Method

### Overall Architecture
NOIR consists of two core modules:
1. **Continuous Function Representation Module**: Embeds discrete signals into a continuous function space using INR.
2. **Neural Operator Mapping Module**: Learns mappings between the latent modulation spaces of the implicit representations.

### Continuous Function Representation
- For each discrete signal $f_i^d$, a continuous function $f_i$ is learned and parameterized by an INR $\phi_i(x; \theta, \gamma_i)$.
- $\theta$ represents the dataset-level shared parameters, and $\gamma_i$ represents the signal-specific parameters.
- Instead of being optimized directly, $\gamma_i$ is mapped from a latent vector $z_i$ via a hypernetwork $M_\psi$: $\gamma_i = M_\psi(z_i)$.
- Shared parameters capture general structures (background, anatomy, acquisition characteristics), while $\gamma_i$ encodes individual variations (anatomical variation, pathology).

### Meta-learning Training Scheme
- **Outer Loop**: Optimizes the shared parameters $(\theta, \psi)$.
- **Inner Loop**: Optimizes the latent representation $z_i$ for each signal (initialized to 0, using K-step gradient descent).
- During testing, $(\theta, \psi)$ are frozen, and only the $z^*$ for new signals is optimized.

### Neural Operator Mapping
- Obtain latent representations $z_{in}^i$ and $z_{out}^i$ for the input and output signal pairs, respectively.
- Learn the mapping $\hat{z}_{out} = \mathcal{T}(z_{in})$ using an MLP with residual connections.
- The loss function is MSE: $\mathcal{L}_{NO} = \frac{1}{N}\sum_{i=1}^N \|\mathcal{T}(z_{in}^i) - z_{out}^i\|_2^2$

### ε-ReNO Theoretical Guarantees
- NOIR experimentally satisfies the $\epsilon$-ReNO property proposed by Bartolucci et al.
- The cross-resolution difference of the input latent representation is $\epsilon_{z_{in}} = 4.0 \times 10^{-6}$, and for the output is $\epsilon_{z_{out}} = 2.2 \times 10^{-5}$.
- The segmentation-level error is $\epsilon_{seg} = 1.8 \times 10^{-2}$, demonstrating minimal aliasing errors.

## Key Experimental Results

### Segmentation Tasks

| Dataset | Method | 100% DSC | 10% DSC |
|--------|------|----------|---------|
| Shenzhen | U-Net | 0.95 | 0.75 |
| Shenzhen | NOIR | **0.94** | **0.93** |
| OASIS-4 | U-Net | 0.95 | 0.40 |
| OASIS-4 | NOIR | 0.85 | **0.85** |

- NOIR significantly outperforms all baselines at low resolution (10%): DSC of 0.93 vs 0.75 for U-Net on Shenzhen.
- On OASIS-4 at 10% resolution, NOIR achieves a DSC of 0.85 vs 0.40 for U-Net, showing an extremely prominent advantage.

### Shape Completion (SkullBreak 3D)
- NOIR achieves a DSC of 0.86 at 10% resolution, outperforming 3D U-Net's 0.71.
- At full resolution, NOIR scores 0.87 vs 3D U-Net's 0.96; though there is a gap, it remains stable across resolutions.

### Image Synthesis and Translation

| Task | Method | 100% PSNR |
|------|------|-----------|
| US Synthesis | NOIR | **31.83** |
| US Synthesis | AttU-Net | 28.55 |
| fastMRI | NOIR | 22.87 |
| fastMRI | DDPM | 25.45 |

- In the ultrasound synthesis task, NOIR leads all baselines significantly with a PSNR of 31.83 (+3.28 dB).
- Its performance in the fastMRI translation task is relatively weaker, but remains perfectly stable across resolutions.

### Resolution Robustness
- Across five different resolutions from 32² to 200², the DSC remains stable between 0.93 and 0.94.
- While all baselines suffer from significant performance degradation at low resolutions, NOIR remains almost unaffected.

## Highlights

1. **Paradigm Innovation**: First to systematically unify medical image computation tasks as operator learning between continuous function spaces, covering four major tasks: segmentation, completion, translation, and synthesis.
2. **Resolution-Agnostic**: NOIR maintains highly consistent performance across different resolutions, overcoming traditional methods' dependency on grid resolution.
3. **$\epsilon$-ReNO Validation**: Experiments prove that NOIR satisfies the key theoretical properties of neural operators, exhibiting minimal aliasing errors.
4. **Modular Design**: The input INR, output INR, and NO are trained independently, enabling the INR to be reused across different tasks.
5. **Simple Architecture**: The NO module performs function space mapping using only an MLP with residual connections.

## Limitations

1. **Performance Gap at Full Resolution**: In full-resolution settings for segmentation and shape completion tasks, NOIR performs 1–11% lower than the best grid-based methods.
2. **Dependency on Supervised Learning**: Requires paired labeled data, making it inapplicable to unsupervised scenarios (e.g., registration).
3. **Coupled Latent Dimension and NO Complexity**: High-dimensional latent spaces require larger datasets and more complex NO architectures.
4. **Insufficient Fine-grained Detail Recovery**: Recovery of fine geometric features, such as the orbital region in shape completion, is relatively poor.
5. **Complex Training Workflow**: Requires multi-stage training for the input INR, output INR, and NO, making the overall workflow cumbersome.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Systematically introduces INR + Neural Operator to medical image computation, representing a paradigm-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covering 4 tasks across 5 datasets with complete ablation and theoretical validation, although the dataset scales are relatively small.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical modeling and rigorous mathematical derivations.
- Value: ⭐⭐⭐⭐ — Resolution robustness offers clinical value, but the performance gap at full resolution limits immediate practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Reanimating Images using Neural Representations of Dynamic Stimuli](reanimating_images_using_neural_representations_of_dynamic_stimuli.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](../../NeurIPS2025/medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[ICML 2026\] DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring](../../ICML2026/medical_imaging/discontinuous_galerkin_neural_operator_for_pathology_defocus_deblurring.md)
- [\[ECCV 2024\] I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](../../ECCV2024/medical_imaging/i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)
- [\[NeurIPS 2025\] Convolutional Monge Mapping between EEG Datasets to Support Independent Component Labeling](../../NeurIPS2025/medical_imaging/convolutional_monge_mapping_between_eeg_datasets_to_support_independent_componen.md)

</div>

<!-- RELATED:END -->
