---
title: >-
  [Paper Note] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Defocus-to-Focus Adaptation (DoF-Gaussian) introduces a learnable lens imaging model based on geometric optics for 3D Gaussian Splatting representations. By integrating scene-wise depth prior adjustment and a defocus-to-focus adaptation strategy, it reconstructs sharp 3D scenes from shallow depth-of-field (defocused blur) inputs, and supports controllable depth-of-field rendering (including refocusing, aperture adjustment…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Depth-of-Field Control"
  - "Defocus Deblurring"
  - "Lens Imaging Model"
  - "Depth Prior"
  - "Circle of Confusion"
date: 2026-05-08
content_hash: 596da55ac71e88d1
---

# DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2503.00746](https://arxiv.org/abs/2503.00746)  
**Code**: [https://dof-gaussian.github.io/](https://dof-gaussian.github.io/)  
**Area**: 3D Vision / Novel View Synthesis  
**Keywords**: 3D Gaussian Splatting, Depth-of-Field Control, Defocus Deblurring, Lens Imaging Model, Depth Prior, Circle of Confusion

## TL;DR

Defocus-to-Focus Adaptation (DoF-Gaussian) introduces a learnable lens imaging model based on geometric optics for 3D Gaussian Splatting representations. By integrating scene-wise depth prior adjustment and a defocus-to-focus adaptation strategy, it reconstructs sharp 3D scenes from shallow depth-of-field (defocused blur) inputs, and supports controllable depth-of-field rendering (including refocusing, aperture adjustment, and bokeh shape transformation).

## Background & Motivation

**Background**: 3DGS and its variants have achieved great success in novel view synthesis, but they rely on the pinhole camera assumption and require all input images to be fully in focus. However, real-world photographs often contain shallow depth-of-field effects (bokeh blur), which are highly prevalent in daily photography.

**Limitations of Prior Work**:
- 3DGS-based methods suffer significant performance degradation when handling defocused inputs because bokeh blur corrupts geometric accuracy.
- Existing deblurring methods (such as BAGS and Deblurring 3DGS) utilize blur estimation networks but lack a physical lens model, preventing controllable depth-of-field operations such as refocusing or adjusting bokeh shapes.
- NeRF-based methods (such as DoF-NeRF and LensNeRF) incorporate lens models but suffer from slow training and inefficient rendering.
- Existing evaluation datasets only assess deblurring performance, neglecting the accuracy of refocusing and camera parameter learning.

**Key Challenge**: Defocused inputs require explicit modeling of the Circle of Confusion (CoC) to restore sharp scenes. However, an inherent discrepancy exists between the ideal optical CoC and the actual CoC of DSLR cameras; directly modeling the ideal case introduces systematic errors.

**Goal**: How to efficiently reconstruct sharp scenes from defocused inputs within the 3DGS framework while enabling controllable depth-of-field rendering?

## Method

### Overall Architecture

DoF-Gaussian is built upon Mip-Splatting. The pipeline is as follows: (1) Run SfM on shallow depth-of-field input images to obtain sparse depth; (2) Fine-tune a monocular depth network using the sparse depth to obtain a scene-wise depth prior; (3) During 3DGS optimization, introduce learnable lens parameters (aperture $\mathcal{A}$, focal distance $\mathcal{F}$), simulate defocused images from sharp renders via a CUDA-accelerated lens imaging algorithm, and compare them with the inputs for training; (4) During inference, setting the aperture to 0 yields all-in-focus sharp images, while adjusting the aperture and focal distance enables various controllable depth-of-field effects.

### Key Designs

1. **Geometric Optics-based Lens Imaging Model**:
    - Function: Replace the pinhole model with a thin lens model to explicitly model the Circle of Confusion (CoC).
    - Mechanism: For a spatial point $P$ at a distance $d$ from the lens, the diameter of its Circle of Confusion on the image plane is $r(d) = \mathcal{A}|1/\mathcal{F} - 1/d|$. When $d = \mathcal{F}$ (on the focal plane), $r = 0$, rendering a sharp image. A differentiable tanh function is used to substitute the ideal step CoC function to ensure gradient propagation, implemented as a CUDA-parallel forward/backward propagation algorithm.
    - Design Motivation: Explicitly modeling the lens allows learning the real aperture and focal distance of each image, ensuring precise simulation of defocus effects and supporting controllable rendering during inference.

2. **Scene-wise Depth Prior Adjustment**:
    - Function: Provide accurate scene geometry guidance for defocused inputs.
    - Mechanism: Predict depth directly from defocused images can be inaccurate. This approach extracts sparse but robust depth points via COLMAP's SfM, and then fine-tunes the monocular depth network using a silog loss $\mathcal{L}_{silog}$ to match the current scene's scale and layout. This yields the scene-wise depth prior $D_{pred}$, which guides the 3DGS rendered depth via an L2 loss.
    - Design Motivation: SfM point clouds in in-focus regions remain reliable. Utilizing these points as anchors to fine-tune the depth network significantly improves geometry prediction accuracy.

3. **Defocus-to-Focus Adaptation**:
    - Function: Mitigate the discrepancy between the ideal optical CoC and the real camera CoC.
    - Mechanism: Training is split into two stages. For the first $t$ steps, uniform weights across the image are used to model defocus effects and learn accurate lens parameters. Subsequently, a sigmoid weight function $\Psi(x) = 1/(1+e^{-a(x-b)})$ (where $x = |1/\mathcal{F} - 1/d|$) is applied to reweight the loss, assigning higher weights to focused areas (regions with small $x$), while scaling the aperture pixel-wise by $\mathcal{A}' = \mathcal{A} \cdot \Psi$.
    - Design Motivation: Once the lens parameters are learned, the in-focus regions are identified. Shifting the focus to optimizing these sharp regions compensates for the systematic bias of the ideal CoC.

### Loss & Training

Total Loss: $\mathcal{L} = \Psi \odot (\mathcal{L}_{rec} + w_d \mathcal{L}_{depth}) + w_n \mathcal{L}_{normal}$

- $\mathcal{L}_{rec} = (1-\lambda)\mathcal{L}_1(I, C^*) + \lambda \mathcal{L}_{D-SSIM}(I, C^*)$: Defocused render vs. input image
- $\mathcal{L}_{depth} = \|D - D_{pred}\|_2$: Rendered depth vs. scene-wise depth prior ($w_d = 0.01$)
- $\mathcal{L}_{normal}$: Normal consistency loss ($w_n = 0.05$)
- Train for 30,000 iterations, enabling defocus-to-focus adaptation at $t = 10000$.

## Key Experimental Results

### Defocus Deblurring (Deblur-NeRF Dataset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Controllable DoF |
|------|-------|-------|--------|----------|
| Deblur-NeRF | 23.47 | 0.720 | 0.121 | ✗ |
| DoF-NeRF | 22.70 | 0.682 | 0.185 | ✓ |
| DP-NeRF | 23.67 | 0.730 | 0.108 | ✗ |
| BAGS | 23.95 | 0.754 | 0.094 | ✗ |
| Deblurring 3DGS | 23.71 | 0.747 | 0.107 | ✗ |
| **DoF-Gaussian** | **23.97** | **0.756** | **0.093** | **✓** |

DoF-Gaussian achieves state-of-the-art or second-best performance across all deblurring metrics, and is the only 3DGS-based method supporting controllable depth-of-field.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Baseline (No Lens) | 21.31 | 0.636 | 0.239 |
| + Lens Model | 23.05 | 0.728 | 0.109 |
| + Lens + Adaptation | 23.59 | 0.742 | 0.104 |
| + Lens + Depth Prior | 23.42 | 0.738 | 0.098 |
| **Full model** | **23.97** | **0.756** | **0.093** |

Combining the three components step-by-step yields a +2.66 dB PSNR improvement, with the lens model contributing the most (+1.74 dB).

### Key Findings

- On synthetic datasets, DoF-Gaussian's lens parameter learning error is significantly lower than DoF-NeRF's ($\delta_\mathcal{A}$: 0.068 vs 0.196, $\delta_\mathcal{F}$: 0.079 vs 0.256), validating that lens parameters can be accurately learned.
- In all-in-focus input tests, DoF-Gaussian performs comparably to or slightly better than Mip-Splatting (PSNR 27.81 vs 27.05), indicating that the lens model does not introduce degradation under general inputs.
- The scene-wise depth prior achieves 0.44 and 0.53 dB PSNR improvements over configurations without fine-tuning or with sparse depth supervision, respectively.

## Highlights & Insights

- **Physics-driven Elegant Design**: Replace black-box deblurring networks with geometric optics principles. A single lens formula $r(d) = \mathcal{A}|1/\mathcal{F} - 1/d|$ simultaneously addresses both deblurring and controllable depth-of-field.
- **Clever Defocus-to-Focus Adaptation Strategy**: Instead of forcing a fit, it acknowledges the difference between the ideal and real CoC and reallocates optimization weights using learned focal plane information, representing a practical approximation scheme.
- **Rich Interactive Applications**: Enables refocusing, aperture adjustment, CoC shape deformation (e.g., circle to pentagon/hexagon), and dynamic depth-of-field videos. These require only parameter adjustments during inference, making a meaningful exploration for 3DGS towards cinematic rendering.
- Retains the real-time rendering advantages of 3DGS, significantly improving training and rendering efficiency compared to NeRF-based solutions.

## Limitations & Future Work

- The CUDA-implemented CoC simulation exhibits significantly increased computational overhead under large apertures, as each pixel must traverse a larger neighborhood.
- Only ideal circular CoC and post-processing deformations are modeled; aberrations and non-uniform defocusing of real lenses are not directly modeled.
- The depth prior depends heavily on the quality of COLMAP's SfM, which may degrade in texture-sparse or heavily defocused scenes.
- Currently, the aperture and focal distance are learned independently for each image, without modeling the continuity of camera parameters across sequences.
- Joint modeling with other non-ideal conditions, such as HDR or motion blur, is not yet explored.

## Related Work & Insights

- **vs. BAGS**: BAGS models defocus using a blur estimation network but lacks a lens model, failing to support refocusing. DoF-Gaussian's physical model achieves both deblurring and controllable depth-of-field.
- **vs. DoF-NeRF**: This is the closest work, but it suffers from low efficiency due to its NeRF foundation. DoF-Gaussian is built on 3DGS, yielding significantly higher efficiency and more accurate lens parameter learning.
- **vs. Deblurring 3DGS**: It only models blur using scaling factors, lacking detailed physical modeling. DoF-Gaussian's lens model is far more precise.
- **Insight**: Modeling real-world physical imaging principles (rather than relying purely on black-box learning) is an effective path to handling non-ideal inputs, which could be extended to scenarios such as motion blur and low-light environments.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce a complete lens model into 3DGS to achieve controllable depth-of-field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on both real and synthetic datasets, with comprehensive ablation studies and diverse interactive applications.
- Writing Quality: ⭐⭐⭐⭐ Clear physical derivations, with standard diagrams and algorithm pseudocode.
- Value: ⭐⭐⭐⭐ Fills the gap in controllable depth-of-field for 3DGS, with direct applications in photography and cinematic rendering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Geometry Field Splatting with Gaussian Surfels](geometry_field_splatting_with_gaussian_surfels.md)
- [\[CVPR 2025\] DepthSplat: Connecting Gaussian Splatting and Depth](depthsplat_connecting_gaussian_splatting_and_depth.md)
- [\[CVPR 2025\] CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images](cocogaussian_leveraging_circle_of_confusion_for_gaussian_splatting_from_defocuse.md)
- [\[CVPR 2025\] HyperGS: Hyperspectral 3D Gaussian Splatting](hypergs_hyperspectral_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mitigating Ambiguities in 3D Classification with Gaussian Splatting](mitigating_ambiguities_in_3d_classification_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
