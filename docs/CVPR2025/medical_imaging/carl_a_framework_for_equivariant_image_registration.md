---
title: >-
  [Paper Note] CARL: A Framework for Equivariant Image Registration
description: >-
  [CVPR 2025][Medical Imaging][Image Registration] Proposes CARL (Coordinate Attention with Refinement Layers), a deep registration framework that achieves $[W,U]$-equivariance to translations and rotations via a coordinate attention mechanism. By replacing only the first step in a multi-step registration pipeline with CARL, global $[W,U]$-equivariance is obtained. It matches or exceeds SOTA performance on three medical registration benchmarks (abdomen, lung, and brain)…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Image Registration"
  - "Equivariance"
  - "Coordinate Attention"
  - "Multi-step Registration"
  - "Diffeomorphism"
date: 2026-05-08
content_hash: e2f5271dc7a53071
---

# CARL: A Framework for Equivariant Image Registration

**Conference**: CVPR 2025  
**arXiv**: [2405.16738](https://arxiv.org/abs/2405.16738)  
**Code**: None  
**Area**: Medical Images  
**Keywords**: Image Registration, Equivariance, Coordinate Attention, Multi-step Registration, Diffeomorphism

## TL;DR

Proposes CARL (Coordinate Attention with Refinement Layers), a deep registration framework that achieves $[W,U]$-equivariance to translations and rotations via a coordinate attention mechanism. By replacing only the first step in a multi-step registration pipeline with CARL, global $[W,U]$-equivariance is obtained. It matches or exceeds SOTA performance on three medical registration benchmarks (abdomen, lung, and brain), significantly leading on abdominal registration tasks featuring varying fields of view.

## Background & Motivation

**Background**: Deep learning-based image registration methods (such as VoxelMorph, GradICON, etc.) estimate the spatial correspondence between image pairs by predicting deformation fields using neural networks, achieving strong performance across multiple benchmarks. These methods are typically unsupervised and trained by minimizing image similarity and regularization losses.

**Limitations of Prior Work**: Existing deep registration networks based on displacement field prediction (e.g., VoxelMorph, GradICON) only possess $[U,U]$-equivariance—meaning the registration result remains consistent when both images undergo the exact same transformation. However, they fail when the two input images undergo different transformations ($[W,U]$-equivariance). This is a critical issue in clinical scenarios, where different CT scans often have different fields of view, patient positions, and cropping windows, which is mathematically equivalent to applying different translations/rotations to the two images.

**Key Challenge**: Convolutional networks are naturally $[U,U]$-equivariant to translations (identical translation), but convolutional networks predicting displacement fields cannot yield $[W,U]$-equivariance (different translations). This is because the translation equivariance of convolutions operates on function spaces (where the output values translate accordingly), whereas the equivariance required for registration demands that the output transformation be composed with two different transformations before and after.

**Goal**: To design a deep registration framework that is $[W,U]$-equivariant to independent translations (and optionally rotations) of input images, while retaining the capability to accurately resolve complex local deformations.

**Key Insight**: The attention mechanism is permutation-equivariant, and its attention weights sum to 1. This implies that if the value vectors undergo an affine transformation, the attention output undergoes the exact same affine transformation. Setting the value vectors to coordinates allows calculation of the "centroid of the attention mask," which is the core concept of coordinate attention.

**Core Idea**: Replacing the displacement prediction network in the first step of a multi-step registration pipeline with coordinate attention (standard attention using coordinates as values) yields global $[W,U]$-equivariance. This is because $[W,U]$-equivariance in the first step + $[U,U]$-equivariance in subsequent steps = overall $[W,U]$-equivariance.

## Method

### Overall Architecture

CARL adopts a multi-step multi-resolution registration architecture, with the overall structure defined as $\text{CARL} = \text{TwoStep}\{\text{TwoStep}\{\text{Down}\{\text{TwoStep}\{\text{Down}\{\Xi_\theta\}\}, \Psi_1\}\}, \Psi_2\}, \Psi_3\}$. Here, $\Xi_\theta$ is the proposed coordinate attention network ($[W,U]$-equivariant), $\Psi_i$ represents standard U-Net displacement prediction networks ($[U,U]$-equivariant), and Down is a downsampling operator. The input is a pair of 3D medical images, and the output is an (approximately diffeomorphic) deformation field.

### Key Designs

1. **Coordinate Attention Network $\Xi_\theta$**:

    - **Function**: Estimates $[W,U]$-equivariant deformation fields during the coarse registration stage.
    - **Mechanism**: For a given moving/fixed image pair, features are first extracted using a shared convolutional encoder (utilizing dilated convolutions operating at the same resolution). Standard attention is then applied, taking the fixed image features as Query, moving image features as Key, and the **voxel coordinates of the moving image** as Value. Consequently, the attention output becomes the "weighted centroid of the corresponding position of each fixed voxel in the moving image," directly outputting a coordinate map instead of displacement. Padding and cropping are applied before and after the encoder to mitigate boundary effects, and FlashAttention is employed for highly efficient computation on a $43^3$ volume.
    - **Design Motivation**: Standard attention is permutation-equivariant (the output remains unchanged under identical permutation of Key and Value) and is equivariant to affine transformations of the Value (since weights sum to 1). Translation is both a permutation (in voxel space) and an affine transformation (in coordinate space). Thus, coordinate attention leverages both properties to naturally achieve $[W,U]$-equivariance.

2. **Equivariance Transfer in TwoStep Multi-step Registration**:

    - **Function**: Guarantees the $[W,U]$-equivariance of the overall network.
    - **Mechanism**: The TwoStep operator is defined as $\text{TwoStep}\{\Phi, \Psi\}[I_M, I_F] = \Phi[I_M, I_F] \circ \Psi[I_M \circ \Phi[I_M, I_F], I_F]$, where $\Phi$ is first used for coarse registration, and then the warped moving image alongside the fixed image is fed into $\Psi$ for refinement. The paper rigorously proves that if $\Phi$ is $[W,U]$-equivariant and $\Psi$ is $[U,U]$-equivariant, the cascaded TwoStep pipeline is globally $[W,U]$-equivariant. This property can be recursively applied to any number of steps.
    - **Design Motivation**: A standalone coordinate attention network has limited precision (as it operates on a lower resolution of $43^3$) and requires subsequent refinement. The TwoStep theory guarantees that "first-step $[W,U]$-equivariance + subsequent $[U,U]$-equivariance = overall $[W,U]$-equivariance," allowing the upgrade of the entire GradICON pipeline by simply replacing the first step.

3. **Rotational Equivariance Extension: CARL{ROT}**:

    - **Function**: Extends equivariance from translation to arbitrary rotations.
    - **Mechanism**: Three modifications are made: (1) The encoder averages over 4 axis-aligned $\pi$-rotations, making $\Xi_\theta$ formally equivariant to $\pi$-rotations; (2) The receptive field is enlarged (using an additional convolution with a dilation rate of 8) to stabilize training; (3) SO(3) random rotations are used for data augmentation, with the key trick of "moving" the augmentation inside the loss—applying diffusion regularization to $R^{-1} \circ \Phi[I_M \circ R, I_F \circ Q] \circ Q^{-1}$ instead of directly to $\Phi$, since the Jacobian of the former is close to the identity matrix.
    - **Design Motivation**: Clinical images may differ not only in translation but also in rotation. Although the formal proof only covers translation, the framework can be empirically extended to rotation through data augmentation and architectural modifications.

### Loss & Training

Training is split into two phases: (1) The first 1500 steps use LNCC similarity + diffusion regularization ($\|\nabla \varphi - \mathbf{I}\|_F^2$) as pre-training to stabilize coordinate attention learning; (2) The subsequent 100,000 steps switch to GradICON regularization ($\|\nabla(\text{CARL}[I_M, I_F] \circ \text{CARL}[I_F, I_M]) - \mathbf{I}\|_F^2$) to enforce approximate invertibility. The regularization weight is set to $\lambda=1.5$, and an Adam optimizer is used with a learning rate of 0.0001. An optional 50-step Instance Optimization (IO) can be applied to further boost testing performance.

## Key Experimental Results

### Main Results

| Dataset | Method | DICE↑ / mTRE↓ | %\|J\|<0↓ | Description |
|--------|------|---------------|-----------|------|
| Abdomen1K | ANTs | 45.4% | 0 | Traditional optimization method |
| Abdomen1K | VoxelMorph | 59.3% | - | Classic deep registration |
| Abdomen1K | GradICON | 62.2% | 0.0003 | Direct baseline for CARL |
| Abdomen1K | **CARL** | **75.7%** | - | Only replacing the first step |
| Abdomen1K | **CARL (IO)** | **77.3%** | 0.0001 | + Instance Optimization |
| DirLab Lung | GradICON | 1.57mm | 0.0002 | Classic method |
| DirLab Lung | **CARL** | **1.88mm** | - | Matches SOTA |
| DirLab Lung | **CARL (IO)** | **1.25mm** | 0.0003 | Surpasses most methods |
| HCP Brain | GradICON | ~78.5% | - | Sharp drop after translation |
| HCP Brain | **CARL** | **79.6%** | - | Unaffected after translation |

### Ablation Study

| Configuration | Abdomen1K DICE | HCP DICE | DirLab mTRE | L2R DICE |
|------|---------------|----------|-------------|----------|
| w/o Final Refinement Layer | 74.1% | 78.8% | 2.58mm | 49% |
| with Final Refinement Layer (CARL) | 75.7% | 79.6% | 1.88mm | 50% |

CAM layer insertion ablation (HCP dataset):

| CAM Layer | AP50 | mIoU | Recall |
|--------|------|------|--------|
| Single-layer (16) | 44.7 | 63.2 | 58.6 |
| Dual-layer (11+22, CARL) | 45.9 | 63.7 | 59.7 |
| Triple-layer (8+16+24) | 43.8 | 61.4 | 57.2 |

### Key Findings

- **Breakthrough performance in abdominal registration**: CARL significantly improves the DICE score on Abdomen1K from GradICON's 62.2% to 75.7% (+13.5pp) by replacing only the first-step network. This is due to the large variation in the field-of-view of abdominal CTs, making $[W,U]$-equivariance critical.
- **Equivariance is key, not capacity**: In synthetic experiments on retina images, the non-equivariant GradICON completely fails on the translation-shifted dataset, whereas the equivariant CARL (even when not trained on shifts) generalizes zero-shot. This highlights that equivariance offers a structural advantage.
- **Validation of rotational equivariance**: In HCP brain registration experiments, the DICE score of GradICON drops sharply when images are shifted, whereas CARL remains stable. CARL{ROT} stays unaffected under arbitrary rotations.
- **Necessity of refinement layers**: The standalone coordinate attention network $\Xi_\theta$ lacks precision at low resolutions, necessitating subsequent U-Net refinement.

## Highlights & Insights

- **Elegant integration of theory and practice**: The paper derives the definition of equivariance beginning with closed-form solutions of diffeomorphic registration, constructively proves that coordinate attention achieves $[W,U]$-equivariance, and finally guarantees global equivariance of multi-step networks via the TwoStep theorem. The entire process is backed by rigorous mathematics without sacrificing practical utility.
- **Maximized gains with minimized modifications**: Compared to GradICON, CARL only replaces the first-step network at the coarsest resolution while leaving the rest completely unchanged. This surgical modification makes ablation analysis exceptionally clean and enables easy adoption by other registration frameworks.
- **Diffusion pre-training $\rightarrow$ GradICON regularization** as a staged training strategy is a key finding for stabilizing coordinate attention learning. Diffusion regularization ensures the spatial compactness of early attention masks, while GradICON regularization enforces stronger invertibility constraints later on.

## Limitations & Future Work

- Forward inference takes 2.1 seconds, whereas instance optimization requires 209 seconds; the speed of IO restricts its real-time applicability.
- The formal proof only covers integer voxel translations. For non-integer translations and rotations, the properties are empirically demonstrated rather than strictly proven.
- Boundary effects in the encoder are mitigated by padding but are not completely eliminated, potentially causing errors in scenarios with large displacements.
- CARL's advantage is less pronounced in HCP brain registration than in the abdomen, as brain images are typically standardized beforehand (resulting in aligned fields of view).
- Future directions: (1) Replacing standard convolutional encoders with group-equivariant convolutions to mathematically formalize $[W,U]$-equivariance for a wider range of transformations; (2) Exploring further optimization of FlashAttention to reduce IO overhead.

## Related Work & Insights

- **vs GradICON**: The direct baseline of CARL. Both are identical in refinement layers, loss functions, and hyperparameters, with the only difference lying in the first-step network. The 13.5pp DICE improvement on Abdomen1K is fully attributable to the introduced $[W,U]$-equivariance.
- **vs EasyReg**: EasyReg achieves $[W,U]$-equivariant affine pre-registration of brain images via segmentation and centroid alignment, but relies on high-quality segmentation annotations. CARL is entirely unsupervised and requires no annotations.
- **vs KeyMorph**: KeyMorph achieves equivariant registration by predicting keypoints, but keypoint prediction itself can be unstable. CARL's coordinate attention computes dense correspondences directly in the feature space, improving robustness.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Elegant and profound theoretical contribution of coordinate attention to achieve $[W,U]$-equivariance.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets + synthetic experiments + direct validation of equivariance, providing a highly comprehensive analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation, clear experimental narrative, and extremely detailed supplementary material.
- **Value**: ⭐⭐⭐⭐⭐ Profound impact on medical image registration, particularly in heterogeneous field-of-view scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Scalable Distributed Framework for Multimodal GigaVoxel Image Registration](../../ICLR2026/medical_imaging/a_scalable_distributed_framework_for_multimodal_gigavoxel_image_registration.md)
- [\[CVPR 2025\] EquivAnIA: A Spectral Method for Rotation-Equivariant Anisotropic Image Analysis](equivania_a_spectral_method_for_rotation-equivariant_anisotropic_image_analysis.md)
- [\[CVPR 2025\] SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration](sacb-net_spatial-awareness_convolutions_for_medical_image_registration.md)
- [\[CVPR 2025\] WISE: A Framework for Gigapixel Whole-Slide-Image Lossless Compression](wise_a_framework_for_gigapixel_whole-slide-image_lossless_compression.md)
- [\[CVPR 2025\] CycleULM: A Unified Label-Free Deep Learning Framework for Ultrasound Localisation Microscopy](cycleulm_a_unified_label-free_deep_learning_framework_for_ultrasound_localisatio.md)

</div>

<!-- RELATED:END -->
