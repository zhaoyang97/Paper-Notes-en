---
title: >-
  [Paper Note] CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections
description: >-
  [CVPR 2025][Medical Imaging][SDF] CrossSDF is proposed to reconstruct a 3D SDF from 2D cross-sectional Signed Distance Fields. By combining hybrid encoding (hash grid + random Fourier features) and symmetric difference loss, it achieves accurate reconstruction of thin tubular structures (such as blood vessels) for the first time.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "SDF"
  - "cross-section reconstruction"
  - "thin structures"
  - "neural implicit fields"
  - "hash encoding"
date: 2026-05-08
content_hash: df654a86930fb393
---

# CrossSDF: 3D Reconstruction of Thin Structures From Cross-Sections

**Conference**: CVPR 2025  
**arXiv**: [2412.04120](https://arxiv.org/abs/2412.04120)  
**Code**: [Project Page](https://iamsalvatore.io/cross_sdf)  
**Institution**: University of Edinburgh / University of British Columbia  
**Area**: Medical Imaging / 3D Reconstruction  
**Keywords**: SDF, cross-section reconstruction, thin structures, neural implicit fields, hash encoding, medical imaging

## TL;DR
CrossSDF is proposed to reconstruct a 3D SDF from 2D cross-sectional Signed Distance Fields. By combining hybrid encoding (hash grid + random Fourier features) and symmetric difference loss, it achieves accurate reconstruction of thin tubular structures (such as blood vessels) for the first time.

## Background & Motivation

### Background

**Background**: Reconstructing 3D structures from planar cross-sectional data is a classic problem in medical imaging, manufacturing, and topography. CT/MRI scans generate sparse 2D cross-sectional data, from which complete 3D shapes need to be reconstructed.

**Limitations of Prior Work**:

### Limitations of Prior Work

**Limitations of Prior Work**: Point cloud reconstruction methods tend to fail due to sparse data between cross-sections.

### Key Challenge

**Key Challenge**: Traditional interpolation methods (such as OReX) suffer from severe "laddering artifacts," leading to staircase-like structures between parallel cross-sections.

### Mechanism

**Mechanism**: Indicator function methods suffer from normal and smoothness artifacts due to discontinuities, and degrade into an unstable binary classification problem.

### Supplementary Notes

**Supplementary Notes**: All existing methods perform poorly when reconstructing **thin structures** (such as blood vessels and nerve tissue), suffering from either over-smoothing or fragmentation.

**Key Challenge**: In general, 2D SDF and the desired 3D SDF are inconsistent—directly fitting 2D SDF forces 3D surface normals to be parallel to the cross-sections, resulting in "laddering effects".

**Key Insight**: Leveraging only the zero-set information of the 2D SDF (inside/outside classification) guides the 3D SDF learning through a symmetric difference loss, freeing the model from the constraint of "must be a 2D SDF".

**Core Idea**: Symmetric difference loss (releasing in-plane constraints) + hybrid encoding (hash + RFF, eliminating grid artifacts) + adaptive sampling = accurate reconstruction of thin structures.

## Method

### Inputs & Outputs
- **Input**: $n$ arbitrary-oriented 2D planes, with closed contours (intersection lines between cross-sections and target geometry) on each plane.
- **Output**: Neural 3D SDF $f(\mathbf{x}; \boldsymbol{\theta})$, where the zero-set defines the reconstructed geometry.

### Key Designs

1. **Adaptive Sampling**

    - Function: Ensures that the interior of thin structures is sufficiently sampled.
    - Problem: Uniform sampling under-samples structures with small cross-sectional areas; OReX's fixed-radius sampling is also unsuitable for thin structures.
    - Solution: Samples directly inside each contour up to a threshold count, leveraging the inside/outside labels of 2D cross-sections for precise partitioning.
    - Effect: Under-represented contours with small cross-sectional areas are not ignored.

2. **Hybrid Encoding**

    - Function: Combines hash grid encoding $\gamma_{\text{hash}}$ and Random Fourier Feature (RFF) encoding $\gamma_{\text{RF}}$.
    - Hash Grid: Captures fine details but introduces interpolation artifacts at grid boundaries.
    - Random Fourier Features: Encourages smooth representation and reduces grid boundary artifacts.
    - Fusion Method: $\mathbf{z}_{\text{comb}} = \mathbf{z}_{\text{hash}} + \alpha \cdot \mathbf{z}_{\text{RF}}$, where $\alpha$ is a balancing factor.
    - Final Input: $\mathbf{z}_{\text{final}} = [\mathbf{z}_{\text{comb}} | \mathbf{x}]$, concatenating the original 3D coordinates.

3. **Symmetric Difference Loss**

    - Function: Drives optimization only in regions where the predicted and target inside/outside classifications are inconsistent.
    - Core Insight: The unique information regarding 3D geometry contained in the 2D SDF on a cross-section is the inside/outside classification.
    - Two Sets: $\Omega_{\text{on}} = \{\mathbf{x} | f_{2D}(\mathbf{x}) = 0\}$ (zero-set sampling points) and the symmetric difference region.
    - Advantage: The loss is zero when the predicted and target contours completely coincide, freeing the network to autonomously learn the 3D SDF.
    - Effect: Eliminates laddering artifacts caused by directly fitting 2D SDFs.

4. **Eikonal Regularization**

    - Constraints $|\nabla f| = 1$ to guarantee the SDF property.
    - Particularly effective for sparse inputs.

### Overall Architecture
- Two channels: a hash grid channel and an RFF channel, each equipped with a single-layer MLP.
- After fusion, the features are fed into an SDF MLP to output the final distance value.

## Key Experimental Results

### Quantitative Comparison with Existing Methods (Synthetic Data)

| Method | Applicable Scenarios | Thin Structure Preservation | Topological Continuity | Surface Smoothness |
|------|---------|-----------|-----------|-----------|
| OReX | Arbitrary Cross-Sections | Poor | Poor | Poor (Laddering Artifacts) |
| Points2Surf | Point Clouds | Moderate | Moderate | Moderate |
| POCO | Point Clouds | Moderate | Moderate | Good |
| DeepSDF | Latent Space | Poor | Good | Good (Over-smoothed) |
| **CrossSDF** | **Arbitrary Cross-Sections** | **Good** | **Good** | **Good** |

### Ablation Study

| Configuration | Thin Structure Quality | Surface Smoothness | Laddering Artifacts |
|------|-----------|-----------|---------|
| Hash encoding only | Good | Poor (Grid artifacts) | Few |
| RFF encoding only | Poor (Over-smoothed) | Good | Few |
| Directly fitting 2D SDF | Good | Poor | Severe |
| Without adaptive sampling | Poor (Missing small structures) | — | — |
| **Full Method** | **Good** | **Good** | **None** |

### Key Findings
- The symmetric difference loss is critical for eliminating laddering artifacts.
- Hybrid encoding achieves a better balance between detail preservation and smoothness than a single encoding.
- Adaptive sampling is essential for structures with small cross-sectional areas.

## Additional Technical Details

### Network Parameters
- Hash grid resolution: Multi-scale, ranging from $16^3$ to $2048^3$.
- RFF encoding dimension $d$: Set to match the output dimension of the hash encoding.
- Scaling factor $\alpha$: Configured to make the amplitudes of both encodings approximately equal during initialization.
- SDF MLP: Standard 8-layer fully-connected network with skip connections.

### Eikonal Regularization
$$\mathcal{L}_{eik} = \mathbb{E}_{\mathbf{x}}[(|\nabla f(\mathbf{x})| - 1)^2]$$
Applied on randomly sampled points in the 3D space between cross-sections to provide a smoothness prior.

## Highlights & Insights
- The design of the **symmetric difference loss** is highly elegant: it relaxes the geometric constraint from "learning 2D SDF" to "only learning matching inside/outside classifications." This retains adequate supervision signals while liberating the degrees of freedom.
- The **hybrid encoding** solution is concise: one captures detail while the other ensures smoothness, offering strong complementarity.
- Thin vessel reconstruction in medical imaging represents a critical real-world demand, giving high practical value to this method.
- Supports non-parallel cross-sections of arbitrary orientation, demonstrating greater versatility compared to methods restricted to parallel plans.
- The core of the method lies in the design of the loss function; the network architecture requires no special modifications, allowing easy transfer to other implicit field representations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[ICML 2026\] Foundation VAEs for 3D CT Reconstruction, Augmentation, and Generation](../../ICML2026/medical_imaging/foundation_vaes_for_3d_ct_reconstruction_augmentation_and_generation.md)
- [\[ICLR 2026\] UltraGauss: Ultrafast Gaussian Reconstruction of 3D Ultrasound Volumes](../../ICLR2026/medical_imaging/ultragauss_ultrafast_gaussian_reconstruction_of_3d_ultrasound_volumes.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](../../NeurIPS2025/medical_imaging/eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)

</div>

<!-- RELATED:END -->
