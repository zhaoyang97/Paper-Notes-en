---
title: >-
  [Paper Note] DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring
description: >-
  [ICML 2026][Medical Imaging][Pathology Image Deblurring] DGNO reformulates defocus deblurring in pathological microscopy as an inverse problem of "spatially-varying integral operators." By adopting the Discontinuous Gale…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Pathology Image Deblurring"
  - "Neural Operators"
  - "Discontinuous Galerkin"
  - "Element-wise Local Operators"
  - "Interface Flux"
date: 2026-05-08
content_hash: 4f357d20d4c0b9ee
---

# DGNO: Discontinuous Galerkin Neural Operator for Pathology Defocus Deblurring

**Conference**: ICML 2026  
**arXiv**: [2605.23282](https://arxiv.org/abs/2605.23282)  
**Code**: https://github.com/DeepMed-Lab-ECNU/Single-Image-Deblur  
**Area**: Medical Imaging / Neural Operators / Image Restoration  
**Keywords**: Pathology Image Deblurring, Neural Operators, Discontinuous Galerkin, Element-wise Local Operators, Interface Flux

## TL;DR
DGNO reformulates defocus deblurring in pathological microscopy as an inverse problem of "spatially-varying integral operators." By adopting the Discontinuous Galerkin (DG) style, it decomposes the global kernel into element-local integral operators and interface numerical fluxes. This preserves the physical interpretability of neural operators while handling inherently local discontinuous blur in pathology images. It outperforms SOTAs such as NAFNet, Restormer, and MambaIRv2 on datasets like BBBC006w1.

## Background & Motivation

**Background**: Defocus deblurring of pathology microscopy images is crucial for downstream cell detection and segmentation (e.g., StarDist). Mainstream deep learning methods utilize CNNs (NAFNet / Cho et al.), Transformers (Restormer / MPT), or Mamba (MambaIRv2) for end-to-end learning. Recent works have also applied Neural Operators (NO) to low-level vision (SRNO / DiffFNO).

**Limitations of Prior Work**: (1) CNNs imply shift-invariance, but pathology defocus is **spatially-varying** due to depth changes, tissue heterogeneity, refractive index variations, and aberrations; shift-invariant convolution is fundamentally inaccurate. (2) Global attention in Transformers lacks physical meaning and fails to explicitly model the blur formation process. (3) Mamba operates at the feature-sequence level and remains physically unstructured. (4) Existing NOs (FNO / SRNO) use globally parameterized kernels, implying smoothness/stationarity assumptions, which are unsuitable for **locally discontinuous blur** (e.g., PSF jumps at tissue region boundaries) in pathology images.

**Key Challenge**: Pathological defocus physically corresponds to a spatially-varying integral operator $g(x,y) = \iint K(x,y;\xi,\eta) h(\xi,\eta)\,d\xi d\eta$, which degrades to convolution only under shift-invariance. In reality, PSFs are both spatially-varying and locally discontinuous. However, existing NOs assume smooth global kernels. Maintaining the physical consistency of NOs while acknowledging local discontinuity remains an open problem.

**Goal**: (1) Formalize defocus deblurring as an NO (function-to-function map) for the first time. (2) Design an NO architecture capable of explicitly handling locally discontinuous blur. (3) Capture local heterogeneity while maintaining global coherence to avoid blocking artifacts.

**Key Insight**: The Numerical solution of Partial Differential Equations (PDEs) uses a mature method called "Discontinuous Galerkin (DG)"—dividing the domain into non-overlapping elements, solving independently within elements, and coupling them via interface numerical fluxes. This approach is naturally suited for locally discontinuous fields, matching the "locally stable + consistent transition between regions" characteristic of pathology images.

**Core Idea**: Decompose the global integral kernel into **element-local volume operators** + **interface numerical fluxes**. The former captures spatial local heterogeneity, while the latter controls information exchange between elements to avoid over-smoothing. This retains the physical consistency of NO while gaining the local discontinuity modeling capability of DG.

## Method

### Overall Architecture

DGNO follows the general NO structure: input image $a(x) \to$ pointwise lift to latent feature field $z_0 = P(a)$ (using a Mamba encoder) $\to$ pass through $T$ layers of DG-style operator layers $\to$ pointwise project to output sharp image $u(x) = Q(z_T)$.

Each operator layer: $z_{t+1}(x) = \sigma(W z_t(x) + (\mathcal{K} z_t)(x))$, where $\mathcal{K}$ is a non-local integral operator. The core innovation of this paper is parameterizing $\mathcal{K}$ in a DG manner.

The domain is partitioned into non-overlapping elements $\{E_e\}$. $\mathcal{K} = \mathcal{K}_{\text{vol}} + \mathcal{K}_{\text{flux}}$: comprising intra-element volume integration and inter-element numerical flux.

### Key Designs

1. **Element Local Volume Integral Operator**:

    - **Function**: Independently learns local integral kernels within each element to characterize local heterogeneous blur in pathology images.
    - **Mechanism**: Applies an intra-element volume integral $(\mathcal{K}_{\text{vol}} z)(x) = \int_{E_e} k_e(x, y) z(y)\,dy$ to features $z|_{E_e}$ on element $E_e$, where each $k_e$ is learned independently. The implementation uses Galerkin-type attention to perform adaptive token aggregation within the element.
    - **Design Motivation**: Global FNO assumes kernel spatial stationarity, which fails for different PSFs in distinct pathology regions (nucleus vs. cytoplasm vs. extracellular matrix). Element-local parameterization allows kernels in different regions to be learned distinctively.

2. **Interface Numerical Flux Operator (with P0DG Lightweight Approximation)**:

    - **Function**: Exchanges information across element boundaries to avoid blocking artifacts while maintaining local discontinuity.
    - **Mechanism**: Defines a numerical flux $\mathcal{K}_{\text{flux}}$ at element interfaces to couple adjacent elements. Two forms are provided: (a) General face-based flux, where each interface has an independent learnable operator; (b) Zero-order DG (P0DG), where interface coupling is derived directly from the element volume operator, saving parameters. The former is more expressive, while the latter is lightweight.
    - **Design Motivation**: Purely independent elements lead to blocking artifacts, while purely global kernels cause over-smoothing. The "local + flux" approach of DG is the standard paradigm for handling discontinuous fields in numerical PDE solutions; this paper transfers it to the discretization scheme of neural operators. P0DG ensures the method remains applicable in resource-constrained scenarios.

3. **Physically-aligned Design**:

    - **Function**: Maintains the physical interpretability of the neural operator.
    - **Mechanism**: The entire DGNO directly corresponds to solving the inverse problem of $g(x,y) = \iint K(x,y;\xi,\eta) h(\xi,\eta)\,d\xi d\eta$ from Fourier optics. Element partitioning corresponds to the "piecewise structural heterogeneity" assumption of pathology images. Element-local corresponds to slowly varying PSFs within regions, while interface flux corresponds to transitions between regions.
    - **Design Motivation**: While CNNs/Transformers are black boxes, NO + DG gives each module a clear physical counterpart, facilitating diagnostics of failure cases and future physical improvements (e.g., introducing explicit PSF priors).

## Key Experimental Results

### Main Results: BBBC006w1 Pathology Defocus Deblurring

| Method | PSNR↑ | SSIM↑ | Params (M) | FLOPs (G) |
|------|------|------|------|---|
| NAFNet | 28.92 | 0.879 | 17.1 | 16 |
| Restormer | 29.47 | 0.886 | 26.1 | 141 |
| MPT | 29.78 | 0.891 | 14.7 | 23 |
| MambaIRv2 | 30.12 | 0.895 | 25.9 | 24 |
| SRNO (NO Baseline) | 30.05 | 0.893 | 16.3 | 19 |
| **DGNO** | **30.86** | **0.907** | 15.8 | **18** |

DGNO significantly leads in both PSNR/SSIM metrics, with parameters and FLOPs comparable to or lower than existing methods.

### Ablation Study: Element Partitioning Granularity

| Number of Elements | PSNR | Note |
|------|------|------|
| 1 (Global kernel, degenerates to SRNO variant) | 30.05 | Loss of locality |
| 4 × 4 | 30.42 | Elements too large |
| 8 × 8 | **30.86** | Optimal |
| 16 × 16 | 30.71 | Elements too small, flux overhead high |
| 32 × 32 | 30.34 | Too fine, blocking artifacts |

An optimal granularity exists, validating the balance between "locality + flux."

### Ablation Study: Interface Flux vs. P0DG

| Configuration | PSNR | Params |
|------|------|------|
| General Interface Flux | 30.86 | 15.8M |
| P0DG (Zero-order approx) | 30.62 | 12.3M |
| No Flux (Purely element-independent) | 29.97 | 14.2M (Blocking artifacts) |

P0DG retains most performance with ~22% parameter savings; pure element independence causes visible blocking artifacts, proving the necessity of flux.

### Key Findings
- **Spatially-varying blur is a key bottleneck for pathology SOTA**: Methods assuming shift-invariance are significantly outperformed by DGNO.
- **Element granularity has a "sweet spot"**: Too large loses locality, while too small increases flux overhead and introduces artifacts.
- **Flux is indispensable**: Removing flux results in learnable kernels but produces blocking artifacts; the DG mathematical structure holds within neural networks.
- **Physical alignment provides interpretability**: Visualizing the kernel of each element allows intuitive observation of blur variation directions (PSF visualizations are provided in the paper appendix).

## Highlights & Insights
- **First work to introduce DG numerical methods to Neural Operators**: Cross-disciplinary innovation by introducing mature tools from numerical PDEs (DG) into NO to handle locally discontinuous fields that standard NOs cannot.
- **Physical framing is the true differentiator**: Unlike other end-to-end methods, DGNO explicitly formulates defocus as an inverse problem of integral operators, providing physical justification for design choices (element-local + flux).
- **Element partitioning provides "structural inductive bias"**: While the inductive bias of CNNs is translation invariance + local connectivity, the inductive bias of DGNO is "piecewise stable + interface coupled," which better fits pathology images. This bias can be generalized to other local heterogeneity problems (e.g., remote sensing clouds, multimodal microscopy fusion).
- **P0DG provides a sweet spot for utility vs. performance**: A -22% parameter reduction with only -0.24 PSNR loss is friendly for practical deployment.

## Limitations & Future Work
- Element partitioning is a fixed grid; adaptive partitioning based on tissue boundaries might yield further improvements.
- Validated only on 2D pathology slides; extensions to 3D pathology volumes (e.g., confocal stacks) have not been attempted.
- PSF is learned implicitly without explicit estimation; joint PSF estimation + deblurring could be considered.
- A rigorous comparison with traditional two-stage methods (PSF estimation + non-blind deconvolution) is missing; physical baselines could be strengthened.
- DG uses element discretization; future work could explore "high-order DG" (hp-refinement) using different basis functions for different elements.

## Related Work & Insights
- **vs. CNN / Transformer / Mamba deblurring**: Those methods imply shift-invariance or lack physical structure; DGNO explicitly models spatially-varying integral operators.
- **vs. FNO / SRNO (Global NO)**: Those assume kernel smoothness and cannot handle local discontinuity; DGNO achieves local expression through DG element partitioning.
- **vs. DG Numerical Methods**: While DG is used in numerical PDE solutions (fluid dynamics, electromagnetics), this is its first systematic application within Neural Operators.
- **Insight**: Use numerical PDE discretization schemes (FEM, DG, HHO, etc.) as an inductive bias library for neural operators; this strategy of "borrowing mathematical physics tools" can be generalized to all scenarios requiring operator learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First DG-NO; introducing numerical PDE discretization to NO for handling local discontinuity is a truly new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Main results + granularity ablation + flux ablation are complete; missing 3D pathology validation.
- Writing Quality: ⭐⭐⭐⭐ Clear derivation from physics $\to$ mathematics $\to$ neural networks; Fig 2 intuitively explains the DG concept.
- Value: ⭐⭐⭐⭐ Pathology image quality directly impacts downstream diagnosis; deblurring is a high-value task. The DG-NO concept itself is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](../../NeurIPS2025/medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)
- [\[ICML 2026\] Rethink the Role of Neural Decoders in Quantum Error Correction](rethink_the_role_of_neural_decoders_in_quantum_error_correction.md)
- [\[ICML 2026\] PathCTM: Thinking in Scales — Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning](thinking_in_scales_accelerating_gigapixel_pathology_image_analysis_via_adaptive_.md)
- [\[CVPR 2026\] Momentum Memory for Knowledge Distillation in Computational Pathology](../../CVPR2026/medical_imaging/momentum_memory_for_knowledge_distillation_in_computational_pathology.md)
- [\[ICLR 2026\] Neuro-Symbolic Decoding of Neural Activity](../../ICLR2026/medical_imaging/neuro-symbolic_decoding_of_neural_activity.md)

</div>

<!-- RELATED:END -->
