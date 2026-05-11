---
title: >-
  [Paper Note] EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper reformulates the anisotropic slice reconstruction problem in volume electron microscopy (vEM) as a dynamic 3D scene rendering task based on deformable 2D Gaussian splatting, achieving high-fidelity continuous slice synthesis under sparse data conditions via a Teacher-Student pseudo-label mechanism.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - volume electron microscopy
  - anisotropic reconstruction
  - dynamic scene modeling
  - self-supervised learning
date: 2026-05-08
content_hash: de4b7ff0f362345c
---

# EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy

**Conference**: CVPR 2026
**arXiv**: [2512.06684](https://arxiv.org/abs/2512.06684)
**Code**: None
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, volume electron microscopy, anisotropic reconstruction, dynamic scene modeling, self-supervised learning

## TL;DR

This paper reformulates the anisotropic slice reconstruction problem in volume electron microscopy (vEM) as a dynamic 3D scene rendering task based on deformable 2D Gaussian splatting, achieving high-fidelity continuous slice synthesis under sparse data conditions via a Teacher-Student pseudo-label mechanism.

## Background & Motivation

Volume electron microscopy (vEM) enables nanoscale 3D imaging of biological structures; however, directly acquiring isotropic data is prohibitively costly due to the "impossible triangle" trade-off among resolution, field of view, and acquisition time. Data acquired in practice are typically anisotropic—axial (z) resolution is far lower than in-plane (xy) resolution.

Existing deep learning methods attempt to recover isotropy through two paradigms:

**Video frame interpolation**: Interpolating xy slices along the z-axis

**Image super-resolution**: Enhancing xz/yz orthogonal views via super-resolution

Both approaches implicitly assume that tissue structures are approximately isotropic in x/y/z, whereas morphological anisotropy is ubiquitous in real biological samples (e.g., nerve fibers, dendritic spines), causing systematic errors when processing highly directional structures.

**Core motivation**: A reconstruction framework is needed that directly reasons in continuous 3D space without relying on the isotropic assumption.

## Method

### Overall Architecture

EMGauss reformulates the anisotropic volume reconstruction problem as a **dynamic 3D scene rendering problem**:
- The axial slice sequence is treated as a **temporal evolution** of a 2D Gaussian point cloud
- The slice index $t \in [0,1]$ serves as a normalized spatial coordinate along the z-axis
- A deformation MLP learns local geometric changes between adjacent slices

The fundamental building block is derived from Deformable 3D Gaussians, where each Gaussian primitive is parameterized by: opacity $o$, center $\mu$, and covariance matrix $\Sigma$ (decomposed into scaling $S$ and rotation $R$).

### Key Designs

#### 1. Slice-to-3D Dynamic Gaussian Modeling

Starting from a canonical Gaussian set $\mathcal{G}_c$ initialized from observed anisotropic frames, a deformation network $\Phi_\theta$ predicts local offsets for each Gaussian:

$$\Delta\mu_i, \Delta S_i, \Delta o_i = \Phi_\theta(\mu_i, t)$$

**Key constraint design**:
- **Position offset**: Only in-plane displacement is permitted, $\Delta\mu_i = (\Delta x_i, \Delta y_i, 0)$
- **Scaling offset**: Only in-plane scaling is permitted, $\Delta S_i = (\Delta s_x, \Delta s_y, 0)$
- **Opacity**: Dynamic variation along the z-axis is permitted, $\Delta o_i$
- **z-coordinate and z-scaling**: Fixed as global constants $z_0, s_{z,0}$
- **Rotation**: Learnable but time-invariant, i.e., $\Delta R_i = 0$

These constraints ensure axial alignment consistency while allowing in-plane deformation and per-slice appearance variation. At inference, a deformed Gaussian set is obtained by querying intermediate $t$ values and rendered accordingly.

#### 2. Teacher-Student Pseudo-Label Bootstrapping

To address data sparsity where only 10%–20% of axial slices are available:

| Component | Design | Role |
|-----------|--------|------|
| Teacher model | EMA of the Student, decay rate $\alpha=0.995$ | Generates stable pseudo-targets on unseen slices |
| Student model | Actively trained model | Learns to match Teacher pseudo-labels |
| Activation strategy | Activated after training iterations exceed a threshold | Introduces pseudo-supervision only after initial convergence |
| Expansion strategy | Progressively expands from intermediate slices to more positions | Gradually covers unobserved regions |

Pseudo-supervision iterations and real-data supervision iterations alternate to ensure balanced learning and stable convergence.

### Loss & Training

**Three-stage training pipeline**:

| Stage | Iterations | Operations | Purpose |
|-------|------------|------------|---------|
| Warm-up | 2k | Optimize canonical Gaussian set $\mathcal{G}_c$, freeze deformation MLP | Establish a stable radiance baseline |
| Joint training | 1k | Train $\mathcal{G}_c$ and deformation MLP jointly | Capture axial transitions (overfitting risk if too long) |
| Teacher-Student | 15k | Activate EMA Teacher for pseudo-supervision | Improve reconstruction quality on unseen slices |

**Loss functions**:
- RGB photometric supervision: $\ell_1$ loss + D-SSIM regularization
- Pseudo-supervision loss weight increases linearly from 0.1 to 1.0 (between iterations 3k–10k)

## Key Experimental Results

### Main Results

**Datasets**: EPFL (mouse brain, 5 nm isotropic), FIB-25 (Drosophila brain, 8 nm isotropic), FANC (Drosophila nerve cord, real anisotropy ×10)

**Table 2: xy slice reconstruction on synthetically anisotropic datasets**

| Method | EPFL PSNR | EPFL SSIM | EPFL FSIM | FIB-25 PSNR | FIB-25 SSIM | FIB-25 FSIM |
|--------|-----------|-----------|-----------|-------------|-------------|-------------|
| CycleGAN-IR | 22.05 | 0.491 | 0.856 | 22.39 | 0.554 | 0.856 |
| EMDiffuse† | 23.34 | 0.519 | 0.899 | 24.10 | 0.514 | 0.878 |
| IsoVEM | 23.91 | 0.597 | 0.856 | 21.51 | 0.546 | 0.846 |
| **EMGauss** | **26.59** | **0.698** | **0.943** | **27.37** | **0.728** | **0.920** |

†EMDiffuse requires additional data for training

**Table 3: Downstream segmentation results on EPFL dataset (SAM2, IoU)**

| Method | CycleGAN-IR | EMDiffuse | EMGauss |
|--------|-------------|-----------|---------|
| IoU | 0.9099 | 0.9555 | **0.9687** |

### Ablation Study

**Table 4: Ablation of key components (averaged over two isotropic datasets)**

| Configuration | PSNR | SSIM | FSIM |
|---------------|------|------|------|
| w/o Teacher-Student | 25.19 | 0.627 | 0.904 |
| w/o warm-up stage | 25.76 | 0.653 | 0.908 |
| w/o joint training | 24.35 | 0.577 | 0.851 |
| w/o dynamic opacity $\Delta o$ | 25.44 | 0.630 | 0.894 |
| w/ dynamic rotation $\Delta R$ | 25.07 | 0.640 | 0.906 |
| **Full model** | **26.98** | **0.713** | **0.932** |

### Key Findings

1. EMGauss outperforms the best baseline by **~3 dB** in PSNR on both EPFL and FIB-25.
2. On xz/yz slice reconstruction, EMGauss trained only on xy outperforms baselines trained on xz/yz.
3. On the real anisotropic FANC dataset (×10 anisotropy), EMGauss is the only method supporting **continuous generation at arbitrary timesteps**.
4. Removing the joint training stage causes the largest performance drop (−2.6 PSNR) among the three stages.
5. Dynamic opacity is more important than dynamic rotation—static opacity cannot model structural appearance/disappearance, while dynamic rotation introduces temporal jitter.

## Highlights & Insights

1. **Elegance of problem reformulation**: Recasting slice reconstruction as dynamic scene rendering fundamentally avoids the isotropic assumption.
2. **Fully self-contained**: Optimized solely on the anisotropic slices of the target volume, requiring no external datasets or large-scale pretraining.
3. **Continuous generation capability**: Interpolated slices can be synthesized at arbitrary depths, which is infeasible for discrete frame interpolation methods.
4. **Fine-grained control over Gaussian attributes**: Careful design of which attributes vary over time and which remain fixed demonstrates a deep understanding of the problem's nature.

## Limitations & Future Work

1. In the presence of noisy input slices, the number of Gaussian primitives may grow substantially, leading to excessive memory consumption.
2. A lightweight denoising module could be incorporated upstream of the reconstruction pipeline to stabilize optimization.
3. Future work may explore adaptive Gaussian pruning or joint learning with image-space regularizers.
4. Current experiments are validated exclusively in electron microscopy; cross-modality generalization remains to be demonstrated.

## Related Work & Insights

- **Relationship to 3DGS**: The paper cleverly migrates 3DGS from multi-view 3D reconstruction to slice-to-volume reconstruction by reinterpreting the z-axis as a temporal dimension.
- **Distinction from Deformable 3DGS**: The original method targets multi-view reconstruction of dynamic 3D scenes, whereas this work addresses continuous 3D reconstruction from 2D slices.
- **Fundamental distinction from diffusion/GAN methods**: No reliance on cross-domain mapping (xy→xz/yz); instead, continuous 3D geometry is directly modeled.
- **Inspiration**: This paradigm is generalizable to other planar scanning imaging modalities (e.g., CT, MRI).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Original problem reformulation; innovative application of 3DGS in medical imaging
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-dataset, multi-metric validation with downstream tasks and ablations, though cross-modality experiments are absent
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with well-articulated motivation
- Value: ⭐⭐⭐⭐ — Provides a general slice-to-3D reconstruction framework with applicability beyond vEM

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Neural Field-Based 3D Surface Reconstruction of Microstructures from Multi-Detector Signals in Scanning Electron Microscopy](neural_field-based_3d_surface_reconstruction_of_microstructures_from_multi-detec.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](instanthdr_singleforward_gaussian_splatting_for_hi.md)
- [\[CVPR 2026\] RetimeGS: Continuous-Time Reconstruction of 4D Gaussian Splatting](retimegs_continuous-time_reconstruction_of_4d_gaussian_splatting.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)

</div>

<!-- RELATED:END -->
