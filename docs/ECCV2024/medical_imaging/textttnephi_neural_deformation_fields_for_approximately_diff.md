---
title: >-
  [Paper Note] NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration
description: >-
  [ECCV 2024][Medical Imaging][Neural Implicit Representation] NePhi proposes using neural implicit functions (SIREN) to replace traditional voxel-based deformation fields for representing deformations in image registration. By predicting latent codes via an encoder for fast inference and utilizing instance optimization to enhance accuracy, it matches SOTA precision in 3D lung and brain registration tasks while reducing training memory by fivefold, naturally yielding smooth…
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Neural Implicit Representation"
  - "Diffeomorphic Registration"
  - "Deformation Field"
  - "Latent Code Prediction"
  - "Multi-Resolution Registration"
date: 2026-05-08
content_hash: fd8f947b3f29a3b9
---

# NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration

**Conference**: ECCV 2024  
**arXiv**: [2309.07322](https://arxiv.org/abs/2309.07322)  
**Code**: [https://github.com/uncbiag/NePhi](https://github.com/uncbiag/NePhi)  
**Area**: Medical Image  
**Keywords**: Neural Implicit Representation, Diffeomorphic Registration, Deformation Field, Latent Code Prediction, Multi-Resolution Registration

## TL;DR

NePhi proposes using neural implicit functions (SIREN) to replace traditional voxel-based deformation fields for representing deformations in image registration. By predicting latent codes via an encoder for fast inference and utilizing instance optimization to enhance accuracy, it matches SOTA precision in 3D lung and brain registration tasks while reducing training memory by fivefold, naturally yielding smooth, approximately diffeomorphic deformations.

## Background & Motivation

**Background**: Deep learning-based medical image registration methods predominantly employ **voxel-based deformation fields** to represent spatial transformations, storing a displacement vector at each voxel location. Representative methods such as VoxelMorph and TransMorph have demonstrated promising results.

**Limitations of Prior Work**: Voxel-based deformation fields suffer from several key issues: (1) Memory consumption scales cubically with image resolution, making GPU memory a bottleneck during the training of high-resolution 3D images; (2) Deformation regularity—ensuring that transformations are smooth and topology-preserving (diffeomorphic)—requires auxiliary regularization or complex differential equation solvers; (3) Existing neural deformation field methods (e.g., IDIR), though memory-efficient, rely entirely on test-time optimization (instance optimization), leading to highly computationally expensive and slow inference.

**Key Challenge**: A trade-off exists among registration accuracy, inference speed, memory efficiency, and deformation regularity. Voxel-based methods offer high accuracy but suffer from large memory footprints and poor regularity; purely optimization-based neural field methods are memory-efficient but slow during inference; diffusion-based methods provide good regularity but involve complex computation.

**Goal**: To design a unified framework that simultaneously achieves advantages across all four dimensions: low memory usage, fast inference, high accuracy, and strong deformation regularity.

**Key Insight**: The authors observe that neural implicit representations (such as coordinate-based networks in NeRF) naturally possess continuity and smoothness. Parameterizing the deformation field using a small MLP requires significantly fewer parameters than voxel-based representations, and the continuity of the function inherently guarantees deformation regularity. The key lies in rendering such a representation both generalizable (without needing optimization from scratch for each image pair) and refinement-friendly (supporting instance optimization).

**Core Idea**: To use an encoder to predict latent codes combined with a SIREN decoder to generate continuous deformation fields, unifying fast generalized inference and fine-grained instance optimization.

## Method

### Overall Architecture

The pipeline of NePhi is divided into two phases: the **pre-training phase** and the **inference phase**.

Pre-training phase: Given a pair of moving and fixed images, the encoder extracts features and predicts a low-dimensional latent code $z$. Subsequently, conditioned on $z$, the SIREN decoder outputs the displacement vector $\phi(x)$ for any spatial coordinate $x$. During training, encoder and decoder parameters are jointly optimized using an image similarity loss and a regularization loss.

Inference phase: (1) **Fast Inference**: The encoder predicts the latent code in a single forward pass, and the decoder generates the deformation field, achieving speeds comparable to traditional learning-based methods; (2) **Instance Optimization (IO)**: The encoder and decoder parameters are frozen, and only the latent code $z$ is optimized to further enhance registration accuracy. Since only $z$ (which is extremely low-dimensional, e.g., 256 dimensions) is optimized, the memory consumption of IO is substantially lower than optimizing an entire voxel-based deformation field.

### Key Designs

1. **SIREN Decoder (Deformation Field Generator)**:

    - **Function**: Takes spatial coordinates and a latent code as input, and outputs the displacement vector for that coordinate.
    - **Mechanism**: Uses SIREN (Sinusoidal Representation Network) as the decoder, which is an MLP that utilizes a $\sin$ activation function in each layer. The input is the concatenated coordinate $x \in \mathbb{R}^3$ and latent code $z \in \mathbb{R}^d$, and the output is $\phi(x) \in \mathbb{R}^3$. The periodic activation function of SIREN endows the network with the capability to represent high-frequency details while maintaining the infinite differentiability of the function.
    - **Design Motivation**: Traditional ReLU MLPs have a weak capability for fitting high-frequency signals, whereas SIREN can efficiently represent detail-rich deformation fields via sinusoidal activations. Meanwhile, the parameter count of SIREN is only a few thousand, which is significantly smaller than million-scale voxel-based representations, directly yielding memory benefits.

2. **Hybrid Encoder**:

    - **Function**: Extracts features from image pairs and predicts latent codes to achieve generalized inference.
    - **Mechanism**: The encoder adopts a 3D CNN architecture, taking concatenated moving and fixed images as input and outputting the latent code $z$. The encoder is jointly trained across all image pairs in the training set to learn generalizable feature-extraction capabilities for registration. During inference, a single forward pass provides a reasonable initial latent code.
    - **Design Motivation**: Pure optimization-based methods (e.g., IDIR) require hundreds of iterations per image pair to converge. Integrating an encoder allows the direct prediction of a high-quality initial latent code, which is then refined through a small number of IO steps, balancing speed and accuracy.

3. **Multi-Resolution Registration Strategy**:

    - **Function**: Enhances accuracy in large-deformation scenarios through coarse-to-fine multi-stage registration.
    - **Mechanism**: Divide registration into multiple stages, with each stage operating on sampling coordinates at different spatial resolutions. Coarse deformations are estimated at low resolution first, followed by refinement at high resolution. Each stage shares the same SIREN decoder but uses coordinate samplings of varying granularities.
    - **Design Motivation**: Large-scale deformations are common in medical images (especially lung CTs), and single-resolution registration easily gets trapped in local optima. The multi-resolution strategy decomposes large deformations into progressively refined smaller ones, boosting registration robustness.

### Loss & Training

The training loss consists of two parts:

- **Image Similarity Loss**: Uses negative local normalized cross-correlation (NCC) to measure the similarity between the warped moving image and the fixed image: $\mathcal{L}_{sim} = -\text{NCC}(I_m \circ \phi, I_f)$.
- **Regularization Loss**: Imposes smoothness constraints on the gradient of the deformation field, encouraging positive values of the Jacobian determinant (ensuring local invertibility): $\mathcal{L}_{reg} = \lambda \|\nabla \phi\|^2$.

The total loss is $\mathcal{L} = \mathcal{L}_{sim} + \mathcal{L}_{reg}$. The instance optimization phase utilizes the same loss function but updates only the latent code $z$.

## Key Experimental Results

### Main Results

The paper evaluates the model on a 2D synthetic dataset and two 3D medical datasets: DirLab COPDGene (lung CT) and OASIS (brain MRI).

| Method | Dataset | TRE (mm) ↓ | Dice ↑ | \|%Jac\|≤0 ↓ | Memory (GB) |
|------|--------|-----------|--------|-------------|----------|
| VoxelMorph | DirLab | 3.21 | - | 1.2% | 8.5 |
| TransMorph | DirLab | 2.85 | - | 0.9% | 12.0 |
| IDIR (Opt.) | DirLab | 2.52 | - | 0.01% | 2.0 |
| NePhi (Encoder) | DirLab | 3.05 | - | **0.0%** | **1.7** |
| NePhi + IO | DirLab | **2.48** | - | **0.0%** | **1.7** |
| VoxelMorph | OASIS | - | 0.74 | 1.5% | 8.5 |
| NePhi + IO | OASIS | - | **0.76** | **0.0%** | **1.7** |

### Ablation Study

| Configuration | TRE (mm) ↓ | \|%Jac\|≤0 ↓ | Inference Time |
|------|-----------|-------------|----------|
| NePhi Full (Encoder + IO) | 2.48 | 0.0% | ~30s |
| Encoder Only (No IO) | 3.05 | 0.0% | <1s |
| Optimization Only (No Encoder) | 2.52 | 0.01% | ~5min |
| Single Resolution | 3.38 | 0.0% | ~20s |
| Multi-Resolution | 2.48 | 0.0% | ~30s |
| Voxel Representation + Equivalent Regularization | 2.65 | 0.9% | ~25s |

### Key Findings

- **Deformation regularity is the most vital advantage**: In all configurations of NePhi, the proportion of non-positive Jacobian determinants is 0%, drastically outperforming voxel-based methods (which score 0.9%-1.5%). This demonstrates that the continuity of SIREN naturally produces diffeomorphic deformations.
- **Significant memory saving**: In multi-resolution registration, the training memory of NePhi is only **1/5** of voxel-based methods, as the deformation is parameterized by a small MLP instead of storing an entire voxel grid.
- **Encoder + IO is the optimal combination**: Encoder-only inference is fast but constrained in accuracy, whereas optimization-only yields good accuracy but slow speed. The encoder provides a solid initialization, and a small number of IO steps achieves the optimal trade-off.

## Highlights & Insights

- **Inherent advantages of neural implicit representations for registration**: The infinite differentiability of SIREN naturally ensures the smoothness and approximate diffeomorphism of deformations, without requiring auxiliary regularization techniques or ODE solvers. This insight can be migrated to other tasks demanding smooth transformations.
- **Latent code as a compressed representation of deformation**: Compressing an entire 3D deformation field into a low-dimensional vector not only saves memory but also enables smooth transitions of deformations via latent code interpolation—holding potential application value in atlas construction and longitudinal analysis.
- **"Encoder initialization + sparse optimization" paradigm**: This "amortized + instance-specific" strategy shares underlying concepts with existing work in NeRF (such as pixelNeRF), validating the generalizability of the feed-forward prediction combined with test-time fine-tuning.

## Limitations & Future Work

- **Validation limited to lung and brain**: It has not been validated in more challenging scenarios such as abdominal organs or the heart, where deformations are larger and topological changes are more complex.
- **Frequency sensitivity of SIREN**: The SIREN parameter $\omega_0$ severely affects the outcomes; inappropriate choices may result in over-smoothing or high-frequency oscillation artifacts.
- **Instance optimization still requires time**: Although remarkably faster than pure optimization, IO still takes approximately 30 seconds, which remains insufficient for real-time surgical navigation and similar scenarios.
- **Lack of robustness analysis for extreme deformations**: While the multi-resolution strategy helps, stability under extreme deformations has not been comprehensively discussed.

## Related Work & Insights

- **vs VoxelMorph / TransMorph**: These methods directly regress voxel-based deformation fields, providing good accuracy but suffering from large memory footprints and non-smooth deformations. NePhi replaces discrete voxels with continuous functions, outperforming them across regularity and memory conservation while matching their accuracy through IO.
- **vs IDIR**: IDIR also utilizes neural fields to represent deformations but relies entirely on optimization, rendering inference extremely slow. NePhi incorporates an encoder to achieve generalized prediction, improving inference speed by an order of magnitude.
- **vs Diffeomorphic methods (LDDMM/VoxelMorph-diff)**: Traditional diffeomorphic methods preserve topology via ODE integration, which is computationally expensive. NePhi obtains approximate diffeomorphism "for free" because of the continuous nature of SIREN.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces neural implicit representation into registration frameworks; the integrated design of encoder + decoder + IO is highly novel.
- Experimental Thoroughness: ⭐⭐⭐ The coverage of datasets and comparative methods could be broader; it lacks validation on challenging scenarios like abdominal organs.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-structured, with comprehensive descriptions of the method and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Presents an efficient and regularized new paradigm for medical image registration; its memory efficiency offers practical significance for clinical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adaptive Correspondence Scoring for Unsupervised Medical Image Registration](adaptive_correspondence_scoring_for_unsupervised_medical_ima.md)
- [\[ECCV 2024\] Unsupervised Multi-modal Medical Image Registration via Invertible Translation](unsupervised_multi-modal_medical_image_registration_via_invertible_translation.md)
- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](../../CVPR2025/medical_imaging/thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[CVPR 2025\] SACB-Net: Spatial-Awareness Convolutions for Medical Image Registration](../../CVPR2025/medical_imaging/sacb-net_spatial-awareness_convolutions_for_medical_image_registration.md)
- [\[CVPR 2026\] Learning Diffeomorphism for Medical Image Registration with Time-Embedded Architectures Using Semigroup Regularization](../../CVPR2026/medical_imaging/learning_diffeomorphism_for_medical_image_registration_with_time-embedded_archit.md)

</div>

<!-- RELATED:END -->
