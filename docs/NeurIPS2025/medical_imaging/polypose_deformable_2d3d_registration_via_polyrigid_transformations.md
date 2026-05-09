---
title: >-
  [Paper Note] PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations
description: >-
  [NeurIPS 2025][Medical Imaging][2D/3D registration] This paper presents PolyPose, a deformable 2D/3D registration method based on polyrigid transformations. Leveraging the anatomical prior that bones are rigid bodies, PolyPose parameterizes complex 3D deformation fields as weighted combinations of multiple rigid transformations in the Lie algebra $\mathfrak{se}(3)$, enabling accurate 3D volumetric registration from as few as two X-ray images without any regularization or hyperparameter tuning.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - 2D/3D registration
  - polyrigid transformation
  - differentiable X-ray rendering
  - intraoperative navigation
  - sparse-view
date: 2026-05-08
content_hash: d2218f9d54a940bd
---

# PolyPose: Deformable 2D/3D Registration via Polyrigid Transformations

**Conference**: NeurIPS 2025
**arXiv**: [2505.19256](https://arxiv.org/abs/2505.19256)
**Code**: [Project Page](https://polypose.csail.mit.edu)
**Area**: Medical Imaging / Medical Image Registration
**Keywords**: 2D/3D registration, polyrigid transformation, differentiable X-ray rendering, intraoperative navigation, sparse-view

## TL;DR

This paper presents PolyPose, a deformable 2D/3D registration method based on polyrigid transformations. Leveraging the anatomical prior that bones are rigid bodies, PolyPose parameterizes complex 3D deformation fields as weighted combinations of multiple rigid transformations in the Lie algebra $\mathfrak{se}(3)$, enabling accurate 3D volumetric registration from as few as two X-ray images without any regularization or hyperparameter tuning.

## Background & Motivation

Estimating the 3D pose of a patient from 2D intraoperative X-ray images is a critical task in image-guided surgery and radiation therapy. Since the number of acquired X-rays is directly related to radiation exposure, only a very limited number of images are available in clinical practice (sparse-view), and scanner geometry further constrains the angular range (limited-angle). While preoperative CT scans are typically available, patient movement between acquisitions causes misalignment between the preoperative CT and intraoperative X-rays.

**Limitations of prior work:**
- **Rigid registration**: Estimates only a global SE(3) transformation, unable to handle articulated motion and soft-tissue deformation.
- **Dense deformation fields**: Employ voxel-wise displacement fields that require extensive per-patient and per-procedure regularization hyperparameter tuning under the severely ill-posed 2D→3D setting, and are prone to anatomically implausible deformations.
- **Learning-based methods**: Require multiple longitudinal CT scans per patient for training, which is infeasible in most clinical scenarios.

The core insight of PolyPose is that human bones are rigid and do not bend. Exploiting this general anatomical prior reduces the number of optimization parameters from the voxel scale $\mathcal{O}(M)$ ($M \approx 10^7$) to the number of rigid bodies $\mathcal{O}(K)$ (as few as $K=3$), fundamentally alleviating the ill-posedness of the problem.

## Method

### Overall Architecture

PolyPose operates in two stages: (1) estimating the camera matrix for each X-ray via rigid registration to an anchor structure; (2) jointly optimizing the poses of all rigid bodies using the estimated camera matrices, constructing a polyrigid deformation field through differentiable rendering.

Given a 3D CT volume $\mathbf{V}$ and a set of 2D X-ray images $\mathbf{I} = \{\mathbf{I}_n\}_{n=1}^N$, the relationship is modeled as: $\mathbf{I}_n = \mathcal{P}(\mathbf{\Pi}_n) \circ \mathbf{V} \circ \mathbf{\Phi}$, where $\mathcal{P}$ is the X-ray projection operator and $\mathbf{\Phi}$ is the 3D deformation field.

### Key Designs

1. **Polyrigid deformation field parameterization**: Let $\{\mathbf{S}_1, \ldots, \mathbf{S}_K\}$ be binary masks of articulated rigid structures in the volume, each associated with an SE(3) transformation $\mathbf{T}_k$. The deformation at any point $\mathbf{x}$ is computed as a weighted combination in the tangent space:

$$\mathbf{\Phi}[\mathbf{T}_1, \ldots, \mathbf{T}_K](\mathbf{x}) = \overline{\mathbf{T}}(\mathbf{x})\tilde{\mathbf{x}}, \quad \overline{\mathbf{T}}(\mathbf{x}) \triangleq \exp\left(\frac{\sum_{k=1}^K w_k(\mathbf{x}) \log \mathbf{T}_k}{\sum_{k=1}^K w_k(\mathbf{x})}\right)$$

   Linearly combining log-transformed matrices in $\mathfrak{se}(3)$ and mapping back to SE(3) ensures that the resulting deformation field is inherently smooth, invertible, and coordinate-frame invariant—a fundamental distinction from naive displacement averaging.

2. **Hyperparameter-free weight function**: Prior methods use $w_k(\mathbf{x}) = \frac{1}{1 + \epsilon d_k^2(\mathbf{x})}$, which involves a tunable $\epsilon$ whose optimal value varies dramatically across structures (e.g., left femur $\epsilon = 10^0$ vs. right femur $\epsilon = 10^{-3}$). PolyPose proposes a gravity-inspired weight function:

$$w_k(\mathbf{x}) = \frac{m_k}{1 + d_k^2(\mathbf{x})}$$

   where $m_k$ is the normalized mass of structure $\mathbf{S}_k$ relative to all structures (estimated from volume). This completely eliminates hyperparameters while handling structures of varying sizes through mass weighting.

3. **Differentiable X-ray rendering**: Based on the Beer–Lambert law, X-ray pixel intensity is modeled as the line integral of linear attenuation coefficients along the ray:

$$\mathbf{I}_n(\mathbf{p}) = \|\mathbf{P} - \mathbf{S}\| \int_0^1 \mathbf{V}(\mathbf{S} + \lambda(\mathbf{P} - \mathbf{S})) d\lambda$$

   Differentiable rendering is achieved via interpolation-based quadrature. A vectorized forward model precomputes a weight matrix $\mathbf{W} \in \mathbb{R}^{M \times K}$, enabling efficient computation of all transformations through batched matrix multiplication: $\hat{\mathbf{\Phi}}(\mathbf{X}) = \exp(\mathbf{W}\hat{\bm{\mathfrak{T}}})\tilde{\mathbf{X}}$.

### Loss & Training

The joint optimization objective maximizes image similarity between rendered and real X-rays:

$$(\hat{\mathbf{T}}_1, \ldots, \hat{\mathbf{T}}_K) = \arg\max_{\mathbf{T}_1, \ldots, \mathbf{T}_K} \frac{1}{N} \sum_{n=1}^N \mathcal{L}(\mathbf{I}_n, \mathcal{P}(\hat{\mathbf{\Pi}}_n) \circ \mathbf{V} \circ \mathbf{\Phi})$$

A multi-scale patch-wise normalized cross-correlation loss is applied to both the original and Sobel-filtered images. The Adam optimizer is used with rotation step size $\beta_{\text{rot}} = 10^{-2}$ and translation step size $\beta_{\text{xyz}} = 10^0$. No regularization terms are required throughout.

## Key Experimental Results

### Main Results (DeepFluoro limited-angle registration, only 2 X-rays, ~30° separation)

| Method | Pelvis Dice ↑ | Left Femur Dice ↑ | Right Femur Dice ↑ | %Folds ↓ |
|--------|--------------|------------------|-------------------|---------|
| **PolyPose** | **0.99** | **0.98** | **0.98** | **0.00%** |
| Dense $\mathbb{R}^3$ | 0.98 | 0.97 | 0.96 | 0.44% |
| xvr (rigid) | 0.99 | 0.96 | 0.94 | 0.00% |
| FireANTs | 0.99 | 0.96 | 0.93 | 0.00% |
| anatomix | 0.95 | 0.93 | 0.92 | 3.01% |
| multiGradICON | 0.83 | 0.86 | 0.77 | 0.00% |

### Ablation Study (deformation parameterization and weight function comparison)

| Configuration | Left Femur Dice | Right Femur Dice | %Folds | Notes |
|---------------|----------------|-----------------|--------|-------|
| PolyPose (Eq.6) | 0.98 | 0.98 | 0.00% | Hyperparameter-free weight function (best) |
| Eq.5, $\epsilon=10^0$ | 0.93 | 0.96 | 0.03% | Good for right femur, poor for left |
| Eq.5, $\epsilon=10^{-3}$ | 0.95 | 0.95 | 0.00% | Good for left femur, poor for right |
| Dense SE(3) | 0.90 | 0.88 | 44.08% | Severe topological defects |

### Key Findings

- From only 2 X-rays with ~30° separation, PolyPose recovers the most accurate 3D deformation fields with zero topological defects.
- On the Head & Neck dataset, PolyPose achieves the best performance not only on rigid structures but also generalizes to soft-tissue organs not directly optimized (thyroid, spinal cord, brain, etc.).
- Although dense deformation models achieve near-perfect image similarity at training viewpoints (NCC ≈ 0.99), they fail to generalize to unseen viewpoints.
- PolyPose is highly robust to erosion of segmentation labels, outperforming baselines even under 3 mm erosion (40–60% volume reduction).

## Highlights & Insights

- **Remarkably concise inductive bias**: The core idea that "bones do not bend" is highly intuitive yet remarkably effective. Reducing optimization parameters from $10^7$ to single digits is a perfect embodiment of the "less is more" principle.
- **Hyperparameter-free design**: The gravity-inspired weight function completely eliminates hyperparameter search, enabling out-of-the-box generalization across procedures and anatomical regions.
- **Theoretical guarantees**: The polyrigid deformation field is inherently diffeomorphic—not enforced through regularization, but as an intrinsic property of the parameterization.

## Limitations & Future Work

- The ability to model extreme soft-tissue deformations far from bones (e.g., abdomen) remains to be validated.
- The diffeomorphic constraint cannot represent certain deformations (e.g., mouth opening), though this may be mitigated by incorporating kinematic chains.
- Segmentation masks of rigid structures in the CT are required; while the method is robust to label quality, this adds a preprocessing step to the pipeline.

## Related Work & Insights

- The polyrigid transformation framework originates from the seminal work of Arsigny et al.; PolyPose successfully extends it to the severely ill-posed 2D/3D registration setting.
- The use of differentiable rendering bridges computer vision (e.g., NeRF) and medical image registration.
- The method has direct clinical value for orthopedic surgical navigation and radiation therapy positioning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The combination of polyrigid parameterization and a hyperparameter-free weight function is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two distinct clinical scenarios (head & neck radiotherapy + orthopedic surgery) with comprehensive quantitative and qualitative evaluation and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are rigorous and clear, with excellent visualizations.
- Value: ⭐⭐⭐⭐⭐ Addresses real clinical pain points with a concise and generalizable approach, demonstrating strong potential for clinical translation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies](../../CVPR2026/medical_imaging/markerbased_3d_reconstruction_of_aggregates_with_a.md)
- [\[NeurIPS 2025\] Surf2CT: Cascaded 3D Flow Matching Models for Torso 3D CT Synthesis from Skin Surface](surf2ct_cascaded_3d_flow_matching_models_for_torso_3d_ct_synthesis_from_skin_sur.md)
- [\[NeurIPS 2025\] Multimodal 3D Genome Pre-training](multimodal_3d_genome_pre-training.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)

<!-- RELATED:END -->
