---
title: >-
  [Paper Note] InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation
description: >-
  [ICCV 2025][Medical Imaging][3D Gaussian Splatting] InsideOut extends 3D Gaussian Splatting (3DGS) beyond RGB surface modeling to simultaneously represent internal X-ray structures…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "3D Gaussian Splatting"
  - "X-ray Imaging"
  - "Multimodal Fusion"
  - "3D Reconstruction"
  - "Non-Destructive Testing"
date: 2026-05-08
content_hash: 98e4dcebde96b40e
---

# InsideOut: Integrated RGB-Radiative Gaussian Splatting for Comprehensive 3D Object Representation

**Conference**: ICCV 2025
**arXiv**: [2510.17864](https://arxiv.org/abs/2510.17864)  
**Code**: Unavailable  
**Area**: Medical Imaging / 3D Vision
**Keywords**: 3D Gaussian Splatting, X-ray Imaging, Multimodal Fusion, 3D Reconstruction, Non-Destructive Testing

## TL;DR

InsideOut extends 3D Gaussian Splatting (3DGS) beyond RGB surface modeling to simultaneously represent internal X-ray structures, achieving joint representation of RGB appearance and internal radiative structure through hierarchical fitting and an X-ray reference loss.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the dominant approach for high-fidelity 3D scene representation, enabling rapid, high-quality reconstruction of surface appearance from multi-view RGB images. However, existing 3DGS methods focus exclusively on object surfaces and cannot capture internal structures.

**Limitations of Prior Work**: In critical applications such as medical diagnosis, cultural heritage restoration, and industrial quality inspection, surface appearance alone is insufficient — clinicians need to visualize internal organs and lesions, archaeologists need to examine internal structures without damaging artifacts, and engineers need to detect internal defects in manufactured components. All of these scenarios demand a unified 3D representation that integrates surface appearance with internal structural information.

**Key Challenge**: RGB images and X-ray images have fundamentally different data representations — RGB images encode surface reflectance properties, whereas X-ray images reflect internal density distributions based on radiative transmission principles. This modality heterogeneity makes naive fusion of the two modalities challenging. Furthermore, paired RGB–X-ray datasets are extremely scarce.

**Goal**: To design a unified 3DGS framework capable of simultaneously representing RGB surface detail and X-ray-derived internal structure, while addressing the data inconsistency between the two modalities.

**Key Insight**: The authors observe that 3D Gaussian primitives are inherently volumetric spatial distributions that can be simultaneously endowed with surface appearance attributes and radiative transmission attributes. The central challenge is enabling the same set of Gaussians to accurately render both RGB appearance and physically correct X-ray transmission images.

**Core Idea**: Through a hierarchical fitting strategy that progressively aligns RGB and radiative Gaussian splatting, combined with an X-ray reference loss to enforce physical consistency of internal structures, the approach extends 3DGS into a comprehensive 3D representation that models both exterior appearance and interior structure.

## Method

### Overall Architecture

InsideOut takes as input multi-view RGB images and multi-angle X-ray images of the same object. The method first establishes independent representations for RGB Gaussians and Radiative Gaussians, then aligns both into a unified 3D space via a hierarchical fitting pipeline. The final output is a set of dual-attribute Gaussian primitives, each carrying both RGB color/opacity information and X-ray radiative attenuation coefficients, enabling simultaneous rendering of photorealistic RGB images and physically accurate X-ray images.

### Key Designs

1. **Radiative Gaussian Splatting**:

    - Function: Integrates the X-ray imaging process into the Gaussian splatting framework.
    - Mechanism: X-ray image formation results from the cumulative attenuation of rays along their path through an object. The authors combine the Beer–Lambert law with Gaussian splatting by assigning each Gaussian primitive an additional radiative attenuation coefficient $\mu$. X-ray rendering accumulates the attenuation contributions of all Gaussians along each ray to simulate transmission imaging, replacing the alpha blending used in RGB rendering.
    - Design Motivation: The standard alpha compositing of 3DGS cannot correctly simulate the physical X-ray imaging process; the rendering equation must be redefined according to radiative transfer principles.

2. **Hierarchical Fitting**:

    - Function: Aligns RGB Gaussians and Radiative Gaussians into a shared 3D space.
    - Mechanism: A coarse-to-fine three-stage fitting procedure is adopted. In the first stage, the spatial positions and covariance parameters of RGB Gaussians and Radiative Gaussians are optimized independently. In the second stage, a global rigid transformation aligns the two sets of Gaussians into a common coordinate system. In the third stage, all parameters are jointly fine-tuned in the aligned space, so that each Gaussian simultaneously satisfies the rendering constraints of both modalities. This progressive strategy avoids training instability caused by large modality discrepancies in direct joint optimization.
    - Design Motivation: RGB and X-ray camera geometries are typically calibrated independently and initially reside in different coordinate systems, necessitating explicit alignment prior to joint optimization.

3. **X-ray Reference Loss**:

    - Function: Ensures that the reconstructed internal structure is physically consistent with X-ray observations.
    - Mechanism: In addition to standard RGB and X-ray reconstruction losses, an extra reference constraint is introduced. This loss compares X-ray images rendered from different viewpoints, enforcing spatial consistency of the internal radiative distribution within the Gaussian field — i.e., the attenuation distribution of internal structures should be physically correct from any viewing direction. This acts as a multi-view consistency regularization applied specifically to the radiative attributes.
    - Design Motivation: Supervision from a limited number of X-ray viewpoints alone tends to produce blurry or inconsistent artifacts in internal structure representations; the reference loss provides additional geometric constraints.

### Loss & Training

The total loss comprises three components: an RGB reconstruction loss (L1 + SSIM), an X-ray radiative reconstruction loss (measuring the discrepancy between rendered and real X-ray images), and the X-ray reference loss. Training follows the hierarchical strategy in sequential stages, each employing a different combination of loss terms and learning rates.

## Key Experimental Results

### Main Results

The authors collect a new paired RGB–X-ray dataset containing diverse objects (e.g., medical phantoms, industrial parts, artifact replicas) and evaluate both RGB and X-ray rendering quality:

| Task | Metric | InsideOut | RGB-only 3DGS | X-ray-only Reconstruction |
|------|--------|-----------|---------------|--------------------------|
| RGB Rendering | PSNR (dB) | ~30+ | ~31 | N/A |
| RGB Rendering | SSIM | ~0.95 | ~0.96 | N/A |
| X-ray Rendering | PSNR (dB) | ~28+ | N/A | ~25 |
| X-ray Rendering | SSIM | ~0.92 | N/A | ~0.88 |

InsideOut maintains RGB rendering quality close to that of pure RGB 3DGS while substantially surpassing X-ray-only reconstruction methods in X-ray rendering quality.

### Ablation Study

| Configuration | X-ray PSNR | RGB PSNR | Note |
|---------------|-----------|----------|------|
| Full Model | Best | Near-best | Complete InsideOut |
| w/o Hierarchical Fitting | −~2 dB | −~1 dB | Direct joint optimization causes misalignment |
| w/o X-ray Reference Loss | −~1.5 dB | Negligible | Reduced internal structure consistency |
| w/o Radiative Rendering | Significant drop | Unchanged | Alpha blending replaces radiative transport |

### Key Findings

- Hierarchical fitting is critical to performance gains; direct joint optimization suffers from gradient conflicts between modalities that impede convergence.
- The X-ray reference loss primarily improves multi-view consistency of internal structures with negligible negative impact on RGB quality.
- InsideOut's advantages are more pronounced under sparse-view settings, as interior and exterior information can mutually complement each other.

## Highlights & Insights

- **The unified multimodal 3D representation paradigm is highly novel**: Extending 3DGS from visual surfaces to physical interiors breaks the fundamental limitation of "seeing only the outside," opening a new research direction in its own right.
- **The hierarchical fitting strategy elegantly addresses heterogeneous modality alignment**: Rather than forcing end-to-end joint training, the "independent → align → joint" paradigm offers a broadly applicable reference for any multimodal fusion task.
- **The design is transferable to other modality combinations**: For example, RGB + ultrasound, RGB + thermal imaging, or RGB + CT scanning — any scenario requiring simultaneous modeling of surface and internal information can draw on this framework.

## Limitations & Future Work

- Paired multi-view RGB and X-ray data of the same object are required, resulting in high acquisition costs that limit large-scale application.
- X-ray imaging involves radiation safety concerns; practical deployment must consider dosage control.
- The current method assumes static objects and cannot handle deformable structures such as soft tissue.
- Hyperparameters of the hierarchical fitting procedure (e.g., iteration counts and learning rate schedules per stage) require tuning for different object types.
- Future work may explore integrating this method with CT reconstruction to leverage the real-time rendering capability of 3DGS for interactive medical visualization.

## Related Work & Insights

- **vs. 3D Gaussian Splatting**: Standard 3DGS operates solely on RGB images; InsideOut extends it to the X-ray modality by introducing Radiative Gaussians, representing the first attempt to apply 3DGS to multi-physics field modeling.
- **vs. NeRF-based CT Reconstruction**: Works such as Neural Attenuation Fields apply NeRF to CT reconstruction but cannot simultaneously model RGB appearance; InsideOut's dual-modality design is more comprehensive.
- **vs. Traditional Multimodal Fusion**: Conventional methods typically fuse RGB and X-ray data at the 2D level; InsideOut achieves unification at the 3D representation level, enabling synthesis of both modalities from arbitrary viewpoints.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First extension of 3DGS to a dual-modality RGB+X-ray inside-out 3D representation; highly original contribution.
- Experimental Thoroughness: ⭐⭐⭐ The newly collected dataset is limited in scale and comparisons with additional baselines are lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and method description is fluent, though some details require reference to supplementary material.
- Value: ⭐⭐⭐⭐ Opens new application directions for 3DGS in non-destructive testing and medical imaging; enormous potential, though a gap remains before practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GaussianPile: A Unified Sparse Gaussian Splatting Framework for Slice-based Volumetric Reconstruction](../../CVPR2026/medical_imaging/gaussianpile_a_unified_sparse_gaussian_splatting_framework_for_slice-based_volum.md)
- [\[AAAI 2026\] Multivariate Gaussian Representation Learning for Medical Action Evaluation](../../AAAI2026/medical_imaging/multivariate_gaussian_representation_learning_for_medical_action_evaluation.md)
- [\[NeurIPS 2025\] 3D-RAD: A Comprehensive 3D Radiology Med-VQA Dataset with Multi-Temporal Analysis and Diverse Diagnostic Tasks](../../NeurIPS2025/medical_imaging/3drad_a_comprehensive_3d_radiology_medvqa_dataset_with_multi.md)
- [\[AAAI 2026\] PINGS-X: Physics-Informed Normalized Gaussian Splatting with Axes Alignment for Efficient Super-Resolution of 4D Flow MRI](../../AAAI2026/medical_imaging/pings-x_physics-informed_normalized_gaussian_splatting_with_axes_alignment_for_e.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](an_openmind_for_3d_medical_vision_selfsupervised_learning.md)

</div>

<!-- RELATED:END -->
