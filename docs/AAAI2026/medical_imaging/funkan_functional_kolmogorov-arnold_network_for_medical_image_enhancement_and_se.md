---
title: >-
  [Paper Note] FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation
description: >-
  [AAAI 2026][Medical Imaging][Kolmogorov-Arnold Network] This paper generalizes the Kolmogorov-Arnold representation theorem from finite-dimensional scalar spaces to function spaces (Hilbert spaces), proposing the FunKAN framework. By learning inner functions via Fourier expansion over Hermite basis functions, the framework preserves the spatial structure of image data and outperforms existing KAN variants on MRI enhancement and three medical image segmentation tasks.
tags:
  - AAAI 2026
  - Medical Imaging
  - Kolmogorov-Arnold Network
  - Medical Image Enhancement
  - Medical Image Segmentation
  - MRI De-artifacting
  - Hermite Functions
date: 2026-05-08
content_hash: d733d9194cd2795b
---

# FunKAN: Functional Kolmogorov-Arnold Network for Medical Image Enhancement and Segmentation

**Conference**: AAAI 2026
**arXiv**: [2509.13508](https://arxiv.org/abs/2509.13508)
**Code**: [GitHub](https://github.com/MaksimPenkin/MedicalKAN)
**Area**: Medical Image Analysis / Network Architecture Design
**Keywords**: Kolmogorov-Arnold Network, Medical Image Enhancement, Medical Image Segmentation, MRI De-artifacting, Hermite Functions

## TL;DR

This paper generalizes the Kolmogorov-Arnold representation theorem from finite-dimensional scalar spaces to function spaces (Hilbert spaces), proposing the FunKAN framework. By learning inner functions via Fourier expansion over Hermite basis functions, the framework preserves the spatial structure of image data and outperforms existing KAN variants on MRI enhancement and three medical image segmentation tasks.

## Background & Motivation

Medical image enhancement (e.g., MRI Gibbs ringing artifact removal) and image segmentation (e.g., tumor detection) are two core tasks in medical image analysis. While deep learning has achieved remarkable progress, existing architectures often lack theoretical grounding and exhibit limited adaptability across modalities.

In 2024, Kolmogorov-Arnold Networks (KAN) attracted widespread attention due to their theoretical interpretability. KAN is grounded in the Kolmogorov-Arnold representation theorem: any continuous multivariate function $f(x_1, \ldots, x_n)$ can be decomposed into finite compositions of univariate continuous functions. Unlike MLPs, which use fixed activation functions with learnable weights, KAN employs learnable activation functions (e.g., B-splines), offering greater flexibility and interpretability in function approximation.

However, **the original KAN suffers from a critical structural limitation**: it treats inputs as permutation-invariant scalar sets, requiring 2D feature maps to be flattened into 1D vectors, which entirely destroys the inherent spatial geometry of image data. Although subsequent works such as U-KAN attempt to embed KAN into U-Net, the KAN modules still operate on flattened features, and the loss of spatial priors limits performance gains.

The central insight of this paper is to **generalize the Kolmogorov-Arnold theorem from scalar space $\mathbb{R}^n$ to function space $H^n$ (Hilbert space)**, enabling KAN to operate directly on 2D feature maps and naturally integrate into image processing pipelines. Although this generalization has not been formally proven, its effectiveness is empirically validated.

## Method

### Overall Architecture

The core idea of FunKAN is to treat each 2D feature map $\chi_{l,i}$ as an element of a Hilbert space $H$ (a function defined over an $h \times w$ spatial grid), and model inter-layer mappings via continuous functionals in the dual space $H^*$. The overall architecture consists of three main components: (1) spectral decomposition over Hermite basis functions to parameterize inner functions; (2) an adaptive grid prediction module for dynamically deforming sampling coordinates; and (3) an attention matrix storing Fourier coefficients to enable interpretability.

### Key Designs

1. **Kolmogorov-Arnold Extension to Function Space**:

    - Function: Generalizes the scalar KAN layer $x_{l+1,j} = \sum_i \phi_{l,ji}(x_{l,i})$ to the functional KAN layer $\chi_{l+1,j} = \sum_i \varphi_{l,ji}(\chi_{l,i})$
    - Mechanism: Assumes that a continuous functional $f$ on $H^n$ can be expressed as $f(\chi_1, \ldots, \chi_n) \leadsto \sum_j \zeta_j(\sum_i \varphi_{ji}(\chi_i))$, where $\varphi_{ji} \in H^*$ are continuous linear functionals in the dual space. By the Riesz representation theorem, $H$ and $H^*$ are isomorphic, so each inner function $\varphi_{l,ji}$ is itself an element of $H$, i.e., a 2D function.
    - Design Motivation: Eliminates the flattening operation; each inner function retains the same spatial dimensions $h \times w$ as the input feature map, naturally preserving locality priors.

2. **Fourier Decomposition over Hermite Basis Functions**:

    - Function: Expands each inner function $\varphi_{l,ji}$ using the first $r$ Hermite basis functions: $\varphi_{l,ji} \leadsto \sum_{k=1}^{r} \langle \varphi_{l,ji}, \psi_k \rangle \psi_k$
    - Mechanism: The expansion coefficients $c_{l,ik} = \langle \varphi_{l,i}, \psi_k \rangle$ form an interpretable attention matrix $A_l \in \mathbb{R}^{n \times r}$. By factoring out the $j$ index, the final form is equivalent to a spectrally decomposed $1 \times 1$ convolution: $\chi_{l+1,j} = \sum_i \theta_{l,j}(\sum_k c_{l,ik} \psi_k)$
    - Design Motivation: Hermite functions are chosen for their joint time-frequency localization as eigenfunctions of the Fourier transform; $r=6$ is determined as optimal via grid search. This mirrors the frequency truncation strategy of Fourier Neural Operators, retaining the most informative spectral modes.

3. **Adaptive Grid Prediction Module**:

    - Function: Learns a spatial offset field $\Delta q_l = \{\Delta q_{l,x}, \Delta q_{l,y}\}$ via a residual network to dynamically deform the sampling grid of the Hermite basis functions.
    - Mechanism: A uniform grid $q$ augmented by learned offsets $\Delta q_l$ produces deformed sampling coordinates at which the Hermite basis functions are evaluated. The offset field is generated by pre-activation residual blocks with BatchNorm and ReLU.
    - Design Motivation: Inspired by implicit neural representation architectures, the adaptive grid enables basis functions to better accommodate varying complexity across regions (e.g., denser sampling near image edges).

4. **U-FunKAN Segmentation Architecture**:

    - Function: Embeds the FunKAN backbone into a U-shaped encoder-decoder framework.
    - Mechanism: The encoder consists of four residual blocks (channels 32→64→128→128, with 2× downsampling per stage); three FunKAN blocks ($n=128$, $r=6$) are placed at the bottleneck; the decoder is symmetric with skip connections.
    - Design Motivation: Combines U-Net's multi-scale feature fusion with FunKAN's interpretable spectral decomposition.

### Loss & Training

- MRI Enhancement: MSE loss $\mathcal{L}_{enh} = \frac{1}{N}\sum_i \|I_i^* - I_i^1\|_2^2$
- Segmentation: Weighted BCE + Dice $\mathcal{L}_{segm} = \frac{1}{N}\sum_i [0.1 \cdot CE(I_i^*, I_i^1) + Dice(I_i^*, I_i^1)]$
- Optimizer: Adam ($\beta_1=0.9$, $\beta_2=0.999$) with manual learning rate scheduling $10^{-4} \to 5 \times 10^{-5} \to 10^{-5}$
- Data Augmentation: Gaussian noise ($\sigma=0.01$) added for MRI enhancement; random flipping, rotation, and transposition (probability 0.5) for segmentation.

## Key Experimental Results

### Main Results — MRI Enhancement

Comparison of different KAN backbones under the same convolutional architecture on the IXI dataset for MRI enhancement:

| Method | PSNR↑ | TV↑ | GFLOPs↓ | Params (M)↓ |
|--------|-------|-----|---------|-------------|
| MLP | 37.96 | 1145.57 | 0.19 | 0.01 |
| KAN (B-spline) | 38.10 | 1161.63 | 0.12 | 0.04 |
| ChebyKAN | 38.01 | 1156.56 | 0.12 | 0.03 |
| HermiteKAN | 38.04 | 1161.31 | 0.12 | 0.03 |
| **FunKAN** | **39.05** | **1174.86** | 3.11 | 2.2 |

FunKAN surpasses the best KAN baseline by **0.95 dB** PSNR, a substantial margin.

### Main Results — Medical Segmentation

| Method | BUSI IoU↑ | BUSI F1↑ | GlaS IoU↑ | GlaS F1↑ | CVC IoU↑ | CVC F1↑ |
|--------|-----------|----------|-----------|----------|----------|---------|
| U-Net | 57.22 | 71.91 | 86.66 | 92.79 | 83.79 | 91.06 |
| U-Net++ | 57.41 | 72.11 | 87.07 | 92.96 | 84.61 | 91.53 |
| U-Mamba | 61.81 | 75.55 | 87.01 | 93.02 | 84.79 | 91.63 |
| U-KAN | 63.38 | 76.40 | 87.64 | 93.37 | 85.05 | 91.88 |
| **U-FunKAN** | **68.49** | 77.37 | **88.02** | **93.50** | **85.93** | 91.42 |

U-FunKAN achieves state-of-the-art IoU on all three datasets while requiring only 4.35 GFLOPs (compared to 14.02 for U-KAN and 2087 for U-Mamba).

### Ablation Study

Effect of different channel configurations of U-FunKAN on BUSI segmentation performance:

| C1→C2→C3 | IoU↑ | F1↑ | GFLOPs↓ | Params (M)↓ |
|-----------|------|-----|---------|-------------|
| 32→64→128 | 69.11 | 77.95 | 4.35 | 3.6 |
| 64→96→128 | 69.94 | 78.42 | 10.84 | 4.1 |
| 128→160→256 | 69.49 | 78.39 | 40.42 | 15.7 |
| 256→320→512 | 70.62 | 79.31 | 161.43 | 62.4 |

The default configuration 32→64→128 achieves the optimal balance between efficiency and accuracy. Increasing channel counts yields marginal accuracy gains at a disproportionate computational cost.

### Key Findings

- FunKAN improves over KAN by approximately 1 dB PSNR, whereas KAN improves over MLP by only 0.1 dB, indicating that the gain from preserving spatial structure far exceeds that from learnable activation functions alone.
- U-FunKAN achieves the highest computational efficiency among all compared methods (4.35 GFLOPs) while attaining the best IoU.
- Qualitative analysis shows that KAN reconstructions tend to produce blurring, whereas FunKAN preserves sharper edges and high-frequency details, which is critical for clinical diagnosis.

## Highlights & Insights

- The theoretical contribution is distinctive: generalizing the Kolmogorov-Arnold theorem to Hilbert spaces is a novel formulation; while formal proof is still absent, the assumption is well-motivated and empirically supported.
- The parameterization combining Hermite basis functions and an adaptive grid is elegant: the time-frequency dual localization of Hermite functions suits image processing, and the adaptive grid further enhances flexibility.
- The codebase demonstrates high engineering quality, utilizing PyTorch Lightning, Ruff, and YAML-based configuration management for reproducible experimentation.
- The framework's applicability to both enhancement and segmentation tasks demonstrates its generality.

## Limitations & Future Work

- The Kolmogorov-Arnold extension to function space has not been formally proven and currently remains a well-motivated hypothesis.
- Although acceptable, the parameter count (2.2M) and computational cost (3.11 GFLOPs) of the FunKAN backbone are substantially higher than those of the original KAN (0.04M / 0.12 GFLOPs), diminishing the lightweight advantage.
- The segmentation datasets are relatively small in scale (BUSI: 647 images, GlaS: 165 images, CVC: 612 images); validation on larger-scale datasets is needed.
- Only $r=6$ Hermite basis functions are used, which may be insufficient for more complex scenarios; a systematic ablation on the choice of $r$ is lacking.

## Related Work & Insights

- FunKAN shares the spectral truncation philosophy with Fourier Neural Operators (FNO), but operates in the Hermite domain rather than the Fourier domain.
- The function space extension approach could be applied to other KAN variants requiring spatial awareness.
- The adaptive grid module draws conceptual parallels with deformable convolutions, and future work could explore combining the two.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Decoding with Structured Awareness: Integrating Directional, Frequency-Spatial, and Structural Attention for Medical Image Segmentation](decoding_with_structured_awareness_integrating_directional_frequency-spatial_and.md)
- [\[AAAI 2026\] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](../../CVPR2026/medical_imaging/biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](../../CVPR2026/medical_imaging/apex_adaptive_visual_prompting.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)

</div>

<!-- RELATED:END -->
