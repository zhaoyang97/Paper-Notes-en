---
title: >-
  [Paper Note] Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries
description: >-
  [CVPR 2025][3D Vision][Depth Estimation] This paper proposes a depth estimation method based on a novel image patch representation termed Blurry-Edges. By modeling the smoothness of defocused boundaries, it achieves robust depth estimation under extremely low-light (photon-limited) conditions from a pair of images with different defocus levels, improving noise robustness by over 4 times compared to existing DfD methods.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Depth Estimation"
  - "Defocus Blur"
  - "Low-light"
  - "Image Representation"
  - "Depth from Defocus (DfD)"
date: 2026-05-08
content_hash: 3f1f9a9a3ac46bf3
---

# Blurry-Edges: Photon-Limited Depth Estimation from Defocused Boundaries

**Conference**: CVPR 2025  
**arXiv**: [2503.23606](https://arxiv.org/abs/2503.23606)  
**Code**: [https://blurry-edges.qiguo.org/](https://blurry-edges.qiguo.org/)  
**Area**: 3D Vision  
**Keywords**: Depth Estimation, Defocus Blur, Low-light, Image Representation, Depth from Defocus (DfD)

## TL;DR

This paper proposes a depth estimation method based on a novel image patch representation termed Blurry-Edges. By modeling the smoothness of defocused boundaries, it achieves robust depth estimation under extremely low-light (photon-limited) conditions from a pair of images with different defocus levels, improving noise robustness by over 4 times compared to existing DfD methods.

## Background & Motivation

Depth from Defocus (DfD) is a depth estimation method that does not require active light sources. It features a compact single-lens setup, making it suitable for space-constrained scenarios such as AR/VR, smartphones, and microrobots. However, the core of DfD relies on the precise estimation of spatial image gradients (acting as a proxy for the degree of defocus), which is highly sensitive to image noise. Existing DfD methods typically assume low-noise inputs (noise standard deviation $\le 4$ LSB) and perform poorly in low-light environments.

The core contradiction of this work is: **DfD requires precise spatial gradient information, whereas intense noise in photon-limited scenarios severely degrades gradient estimation**. The authors' **Key Insight** is: instead of directly estimating the defocus level of the entire image, they focus on defocused boundaries by designing a parameterized image patch representation, Blurry-Edges, to explicitly model boundary position, color, and blur level. This allows direct depth computation from the difference in boundary smoothness between a pair of defocused images via a closed-form DfD equation.

## Method

### Overall Architecture

Given a pair of noisy defocused images $I_+, I_-$ acquired with different optical powers, the images are first segmented into overlapping patches. A CNN (local stage) independently predicts the Blurry-Edges representation for each patch. Subsequently, a Transformer Encoder (global stage) applies global consistency optimization. Finally, the outputs are aggregated to generate global boundary maps, color maps, and sparse depth maps. The depth map can be further densified into a dense depth map via post-processing.

### Key Designs

1. **Blurry-Edges Image Patch Representation**:
    - **Function**: Parameterizes an image patch into a stack of multiple layers of wedges with blurred boundaries. Each wedge is described by a vertex position $\mathbf{p}_i$, orientation $\boldsymbol{\theta}_i$, color $\mathbf{c}_i$, and boundary smoothness $\eta_i$.
    - **Mechanism**: Renders the color map of the stacked wedges via alpha compositing, utilizing the error function $\mathrm{erf}$ to model the smooth boundary transitions. The $\alpha$-map of each wedge is defined as $\alpha_i = \frac{1}{2}[1 + \mathrm{erf}(\frac{d_i}{\sqrt{2}\eta_i})]$.
    - **Design Motivation**: In contrast to Field-of-Junction (FoJ)—which can only represent limited structures such as lines, edges, and junctions without modeling boundary smoothness—Blurry-Edges can represent various boundary structures with varying blur levels, providing directly usable defocus cues for DfD.

2. **Closed-Form DfD Depth Equation**:
    - **Function**: Directly calculates depth values from the corresponding boundary smoothness values $\eta_+, \eta_-$ from a pair of defocused images.
    - **Mechanism**: Leveraging a Gaussian PSF convolution model, the difference in smoothness for the same boundary under different optical powers depends solely on depth. By eliminating the texture blur parameter $\xi$, the closed-form depth solution is derived as: $z(\eta_+, \eta_-) = \frac{2\Sigma^2 s^2(\rho_- - \rho_+)}{\eta_+^2 - \eta_-^2 - \Sigma^2 s(\rho_+ - \rho_-)( s\rho_+ + s\rho_- - 2)}$.
    - **Design Motivation**: Avoids pixel-level gradient computation by utilizing boundary-level parameterized smoothness differences instead, which significantly improves noise robustness.

3. **Local-Global Two-Stage Network Architecture**:
    - **Function**: Employs a CNN to locally predict the Blurry-Edges parameters of each patch, followed by a Transformer Encoder to globally optimize consistency.
    - **Mechanism**: The local stage independently processes each patch and solves color parameters using ridge regression. The global stage enforces consistency constraints across all patches for boundary center maps, color maps, and color gradient maps, while ensuring defocus consistency (sharing wedge positions and colors, with differences only in smoothness).
    - **Design Motivation**: The modular design enables independent training. Global optimization addresses the inconsistency in local estimations, mimicking a hierarchical strategy from patch-level to global-level inference.

### Loss & Training

- Local stage loss $\mathcal{L}_\text{local} = \sum_{i=1}^{3} \beta_i \mathbb{E}_{\mathbf{m}}(l_i)$: consists of three terms representing color error, smoothness error, and boundary localization error.
- Global stage loss $\mathcal{L}_\text{global} = \sum_{i=1}^{7} \gamma_i \mathbb{E}_{I_\pm, \mathbf{m}}(g_i)$: contains seven terms, including prediction errors for color, boundary position, boundary smoothness, depth, and neighborhood consistency.
- Two-stage independent training: the local CNN is first trained to convergence, and then the local stage is frozen while the global Transformer is trained.
- The training data consists solely of simple geometric shapes (rectangles, circles, triangles); the model generalizes to the real world without requiring real-world scene data.

## Key Experimental Results

### Main Results

| Method | Type | No. of Images | $\delta 1$ ↑ | RMSE (cm) ↓ | AbsRel (cm) ↓ |
|------|------|--------|-------------|-------------|---------------|
| Focal Track | Sparse | 2 | 0.588 | 6.308 | 4.640 |
| Tang et al. | Sparse | 2 | 0.663 | 6.737 | 4.346 |
| **Ours (Sparse)** | Sparse | 2 | **0.720** | **5.281** | **3.295** |
| PhaseCam3D | Dense | 2 | 0.405 | 9.883 | 8.053 |
| DefocusNet | Dense | 5 | 0.657 | 6.092 | 4.548 |
| DFV-DFF | Dense | 5 | 0.518 | 8.298 | 6.707 |
| DEReD | Dense | 5 | 0.536 | 7.779 | 5.977 |
| **Ours-PP (Dense)** | Dense | 2 | **0.806** | **3.992** | **2.691** |

### Ablation Study

| Configuration (Patch Size) | $\delta 1$ ↑ | RMSE (cm) ↓ | AbsRel (cm) ↓ |
|-------------------|-------------|-------------|---------------|
| $11 \times 11$ | 0.717 | 5.675 | 3.498 |
| $21 \times 21$ (Best) | **0.720** | **5.281** | **3.295** |
| $31 \times 31$ | 0.657 | 6.123 | 4.060 |

### Key Findings

- The proposed method still reliably estimates depth under a noise standard deviation of 18-19 LSB (corresponding to an extremely dark environment of ~80 lux), showing tolerance to noise levels over 4 times higher than prior methods.
- Training only on simple geometries is sufficient to generalize to complex real-world scenes without fine-tuning.
- The Blurry-Edges representation is multi-functional: it simultaneously generates boundary maps, denoised color maps, and depth maps.
- The densified post-processing (Ours-PP) using only 2 images outperforms other dense methods that utilize 5 images.

## Highlights & Insights

- **Representation Innovation**: Blurry-Edges is a significant extension of the Field-of-Junction. By incorporating boundary smoothness modeling, it directly identifies available cues for DfD.
- **Closed-Form Depth Equation**: Calculates depth directly from boundary smoothness, bypassing the noise sensitivity associated with pixel-level gradients.
- **Remarkable Generalization**: Training on simple geometries $\to$ inference on real-world scenes, demonstrating that the parameterized representation of Blurry-Edges possesses a well-suited inductive bias as a prior.
- This method demonstrates that boundary information is significantly more robust than global texture information in noisy environments.

## Limitations & Future Work

- Sparse depth maps are only estimated along boundaries, leaving textureless regions without depth values.
- Densification relies on a post-processing network (U-Net), introducing additional computational overhead.
- The number of wedges is fixed to $l=2$, which may be insufficient for complex junction structures.
- The image resolution is limited to $147 \times 147$, requiring patch-wise processing for large-resolution images.

## Related Work & Insights

- The Field-of-Junction (FoJ) series of work inspired the concept of parameterized image patch representations, but FoJ lacks smoothness modeling.
- Analytical DfD methods (Focal Track, Focal Flow) provide low-computational-cost depth estimation frameworks.
- Learning-based DfD methods (PhaseCam3D, DefocusNet) can generate dense depth maps but lack robustness.
- The "presentation-then-computation" paradigm of this method can be extended to other tasks requiring geometric information extraction from noisy images.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Both the Blurry-Edges representation and the closed-form DfD equation are novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Complete synthetic and real experiments are provided, though the scale of real-world experiments is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear mathematical derivations and rich illustrations.
- **Value**: ⭐⭐⭐⭐ Opens up a new direction for depth estimation under low-light conditions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Video Depth Anything: Consistent Depth Estimation for Super-Long Videos](video_depth_anything_consistent_depth_estimation_for_super-long_videos.md)
- [\[CVPR 2025\] BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation](blade_single-view_body_mesh_estimation_through_accurate_depth_estimation.md)
- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)

</div>

<!-- RELATED:END -->
