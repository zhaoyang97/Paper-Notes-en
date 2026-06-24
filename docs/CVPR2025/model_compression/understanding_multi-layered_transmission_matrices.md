---
title: >-
  [Paper Note] Understanding Multi-layered Transmission Matrices
description: >-
  [CVPR 2025][Model Compression][Transmission matrix] This work analyzes the theoretical foundation of multi-layered transmission matrix approximation from a frequency domain perspective. It reveals that the "missing cone" problem in microscopy unexpectedly becomes an advantage in wavefront shaping scenarios, proving that a small number of SLM layers can achieve effective scattering correction within a limited field of view.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Transmission matrix"
  - "multi-layer correction"
  - "wavefront shaping"
  - "scattering imaging"
  - "missing cone"
date: 2026-05-08
content_hash: 998247c54a088c66
---

# Understanding Multi-layered Transmission Matrices

**Conference**: CVPR 2025  
**arXiv**: [2410.23864](https://arxiv.org/abs/2410.23864)  
**Code**: None  
**Area**: Computational Optics / Wavefront Shaping  
**Keywords**: Transmission matrix, multi-layer correction, wavefront shaping, scattering imaging, missing cone

## TL;DR
This work analyzes the theoretical foundation of multi-layered transmission matrix approximation from a frequency domain perspective. It reveals that the "missing cone" problem in microscopy unexpectedly becomes an advantage in wavefront shaping scenarios, proving that a small number of SLM layers can achieve effective scattering correction within a limited field of view.

## Background & Motivation

**Background**: Wavefront shaping is a core technology for deep biological tissue imaging. Placing spatial light modulators (SLMs) in the optical path can correct aberrations induced by tissue scattering. However, because scattering occurs throughout the entire 3D volume while the SLM is a flat planar device, a single SLM layer can only correct an extremely small field of view (usually just a few micrometers).

**Limitations of Prior Work**: Multi-conjugate correction systems use multiple SLMs to approximate 3D scattering layer-by-layer, but their physical implementation is extremely complex. According to Nyquist sampling, a 200μm thick tissue would require around 100 layers—rendering physical implementation completely infeasible. Previous experiments have achieved at most two SLM layers.

**Key Challenge**: The requirement of too many SLM layers vs. the physical feasibility of implementing only a few layers. Core problem: How many layers are actually needed to achieve effective correction? Is a small number of layers practically meaningful?

**Goal**: To analyze the multi-layered approximation properties of the transmission matrix from both theoretical and experimental perspectives, quantifying the relationship between the number of layers, correction quality, and field of view size.

**Key Insight**: The inherent "missing cone" of microscopes—owing to the limited numerical aperture, the axial resolution is much lower than the lateral resolution. This means that many frequency components in the 3D volume do not participate in the transmission matrix.

**Core Idea**: The missing cone, a fundamental limitation in microscopy, conversely becomes an advantage in wavefront shaping—a large amount of high-frequency axial information does not participate in the transmission matrix. Consequently, the actual number of required layers is far below the Nyquist limit, and the smaller the field of view, the fewer the layers needed.

## Method

### Overall Architecture
A multi-layered slice model of the transmission matrix is established, approximating the 3D scattering volume through alternating planar aberrations and free-space propagation. The layer-number requirement is analyzed from the frequency domain, followed by validation through simulations and real-world experiments.

### Key Designs

1. **Multi-layered Slice Model and Frequency Analysis**:

    - Function: Approximating the 3D scattering volume with M planar aberration layers
    - Mechanism: The transmission matrix is factorized as alternating products of diagonal matrices (representing planar aberrations $\mathcal{D}(\rho_m)$) and propagation matrices ($\mathcal{P}_{\Delta z}$). Under the weak scattering approximation, the transmission matrix becomes a linear function of each layer's aberration. Fourier domain analysis shows that due to the missing cone caused by NA limitations, the actual number of layers needed is far fewer than specified by the Nyquist limit.
    - Design Motivation: To provide theoretical tools to answer the core question of "how many layers are required."

2. **Sparse Approximation under a Finite Field of View**:

    - Function: Proving that restricting the correction field of view can further reduce the number of required layers
    - Mechanism: When the transmission matrix covers only $\Omega_p \times \Omega_p$, it is equivalent to a sparser Fourier domain sampling. While $M=1$ layer corrects approximately $1\times1\mu m$, $M=3$ layers can correct $5\times5\mu m$ (25 times the area), showing a super-linear growth.
    - Design Motivation: To demonstrate that a few layers can significantly accelerate sequential scanning correction within a limited field of view.

3. **Experimental Validation System**:

    - Function: Progressively validating the scope of applicability of the theory
    - Mechanism: Synthetic sphere volume -> precise wave propagation model simulation -> validation using transmission matrices collected in the laboratory from real chicken breast (170μm thick), mouse brain tissues, etc.
    - Design Motivation: To generalize from weak scattering to real-world multi-scattering scenarios.

### Loss & Training
Least-squares optimization fitting: $\mathcal{E}_M = \min_{\rho_1,...,\rho_M} \|\mathcal{T}_{exact} - \mathcal{T}(\rho_1,...,\rho_M)\|^2$. Weak scattering is solved linearly, while strong scattering is optimized using gradient descent.

## Key Experimental Results

### Main Results

| Volume Thickness | Minimum Effective Layers | Field of View Size | Fitting Quality |
|---|---|---|---|
| 40μm | ~6 layers | 40×40μm | Low error |
| 40μm | 3 layers | 5×5μm | Good |
| 40μm | 1 layer | ~1×1μm | Single point only |

### Real Transmission Matrix Validation (Chicken Breast 170μm Thick)

| Number of Layers (M) | Correctable Field of View | Focusing Quality |
|---|---|---|
| 0 | None | Speckle |
| 1 | ~1×1μm | Single-point focus |
| 2 | ~6×6μm | 36x area |
| 6 | ~13×13μm | Good focus |

### Key Findings
- The required number of layers correlates with tissue thickness, but is largely independent of optical density (scattering intensity).
- Although 2-4 SLM layers cannot fully approximate the 3D volume, they can expand the correctable field of view by dozens of times.
- The missing cone structure makes the actual number of required layers far lower than the Nyquist limit.
- The theoretical predictions hold in both weak scattering and multiple scattering scenarios.

## Highlights & Insights
- **Turning a Disadvantage into an Advantage**: The missing cone is a limitation in microscopy but becomes an advantage in wavefront shaping—this shift in perspective is highly inspiring.
- **Practical Analysis Paradigm**: Rather than pursuing perfect 3D reconstruction, this work quantifies "what can be achieved with limited resources," which is highly instructive for designing hardware-constrained systems.
- **Super-linear Layer-to-FOV Relationship**: Even adding just one or two SLM layers can yield significant gains.

## Limitations & Future Work
- The theoretical analysis is primarily based on the weak scattering approximation, and the theoretical derivation is less rigorous for strong scattering in deep tissue scenarios.
- A multi-layer SLM system was not physically constructed for experimental validation.
- Dynamic scattering inside living tissue is not discussed.

## Related Work & Insights
- **vs. Single-layer Wavefront Shaping**: A single layer can only correct micrometer-scale fields of view, whereas multiple layers significantly expand the correction range.
- **vs. Optical Diffraction Tomography (ODT)**: While ODT reconstructs the 3D volume, this work studies the inverse problem—how to explain the transmission matrix using a small number of layers.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight into the relationship between the missing cone and the number of layers is a novel theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated using synthetic simulations and multiple real tissue systems.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations and intuitive physical explanations.
- Value: ⭐⭐⭐⭐ Highly instructive for the design of multi-conjugate wavefront shaping systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Layered Image Vectorization via Semantic Simplification](layered_image_vectorization_via_semantic_simplification.md)
- [\[ACL 2025\] mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding](../../ACL2025/model_compression/mplug_docowl2_doc_compress.md)
- [\[ACL 2025\] Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition](../../ACL2025/model_compression/assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)
- [\[CVPR 2025\] Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](multi-modal_knowledge_distillation-based_human_trajectory_forecasting.md)
- [\[CVPR 2025\] MobileMamba: Lightweight Multi-Receptive Visual Mamba Network](mobilemamba_lightweight_multi-receptive_visual_mamba_network.md)

</div>

<!-- RELATED:END -->
