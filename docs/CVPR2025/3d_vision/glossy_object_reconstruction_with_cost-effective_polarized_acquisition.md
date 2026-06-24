---
title: >-
  [Paper Note] Glossy Object Reconstruction with Cost-effective Polarized Acquisition
description: >-
  [CVPR 2025][3D Vision][Glossy object reconstruction] A cost-effective polarization-assisted 3D reconstruction method is proposed. By simply adding a linear polarizer in front of a standard RGB camera and capturing one polarization image per viewpoint (without the need for polarizing angle calibration), the method recovers high-fidelity geometry and reflectance decomposition of glossy objects via end-to-end optimization of polarization rendering loss in a neural implicit field…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Glossy object reconstruction"
  - "polarization imaging"
  - "neural implicit surfaces"
  - "cost-effective acquisition"
  - "reflectance decomposition"
date: 2026-05-08
content_hash: 096ff26ebc7c8299
---

# Glossy Object Reconstruction with Cost-effective Polarized Acquisition

**Conference**: CVPR 2025  
**arXiv**: [2504.07025](https://arxiv.org/abs/2504.07025)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: Glossy object reconstruction, polarization imaging, neural implicit surfaces, cost-effective acquisition, reflectance decomposition

## TL;DR
A cost-effective polarization-assisted 3D reconstruction method is proposed. By simply adding a linear polarizer in front of a standard RGB camera and capturing one polarization image per viewpoint (without the need for polarizing angle calibration), the method recovers high-fidelity geometry and reflectance decomposition of glossy objects via end-to-end optimization of polarization rendering loss in a neural implicit field.

## Background & Motivation

1. **Background**: Most image-based 3D reconstruction methods assume a Lambertian reflectance model and perform poorly on glossy or specular objects. Existing methods handling glossy objects either require applying a diffuse coating, customizing high-end polarization cameras (such as those using the Sony IMX250MZR sensor), or precisely calibrating multi-angle polarizers.
2. **Limitations of Prior Work**: (1) Polarization methods like PANDORA require expensive, specialized polarimetric cameras, leading to high cost and low portability; (2) Traditional SfP methods are prone to singularities when handling non-manifold meshes; (3) Pure RGB methods (e.g., NeRO, InvRender) suffer from ambiguities when separating diffuse and specular reflections, limiting geometric accuracy.
3. **Key Challenge**: Polarization information is crucial for separating diffuse and specular reflections (as their polarization states are orthogonal). However, traditional methods to acquire polarization information require precise control and calibration of polarization angles, which greatly increases the cost and complexity of the acquisition system.
4. **Goal**: How can polarization information still be leveraged to achieve high-quality geometric reconstruction and reflectance decomposition of glossy objects, while heavily reducing the cost of polarization data acquisition (using only a single polarizer and a standard camera)?
5. **Key Insight**: Although the polarization angle $\phi_{pol}$ is unknown, it can be treated as an optimizable parameter and trained end-to-end alongside the neural network. As long as the Stokes vector model is sufficiently accurate, the polarization angle can be implicitly recovered from multi-view consistency.
6. **Core Idea**: Combine the simplest hardware setup (an RGB camera and a polarizer) with an end-to-end optimization framework based on a polarimetric BRDF neural implicit field to achieve automatic polarization angle estimation and high-quality glossy object reconstruction.

## Method

### Overall Architecture
The input is a set of posed multi-view polarization images (around 40 images, one per view, with unknown polarization angles). The output consists of the object's high-fidelity geometry (mesh extracted via SDF) and reflectance decomposition (diffuse and specular components). The pipeline consists of three steps: (1) Cost-effective data acquisition (shooting with an RGB camera and a fixed-orientation linear polarizer, with camera poses estimated by COLMAP); (2) Neural radiance field representation (VolSDF for implicit surface representation and Ref-NeRF for radiance decomposition); (3) Polarization rendering (calculating Stokes vectors using a pBRDF model, estimating polarization angles, rendering polarization images, and computing loss against the inputs).

### Key Designs

1. **Cost-effective Polarized Acquisition System**:
    - **Function**: To acquire multi-view polarization images at extremely low cost without precise polarization angle calibration.
    - **Mechanism**: A linear polarizer is fixed in front of a standard RGB camera (Sony A6400, 4K), preserving its orientation while capturing about 40 images around the object handheld. There is no need to rotate the polarizer to multiple angles, nor is there a need to calibrate the alignment between the polarizer and the camera. Poses are estimated using COLMAP after $4\times$ downsampling.
    - **Design Motivation**: Traditional polarization methods require capturing images at four angles (e.g., 0°, 45°, 90°, 135°), or using expensive on-chip polarization sensors. The proposed method only requires one polarization image per view and treats the polarization angle as an optimization parameter within the network, fundamentally eliminating the need for precise calibration.

2. **Polarimetric BRDF-based Neural Rendering**:
    - **Function**: To associate Stokes vectors and polarization states with surface properties through a physics-driven polarization rendering model.
    - **Mechanism**: VolSDF is used as the geometric backbone (querying SDF values and normals), and DiffuseNet and SpecularNet are employed to encode diffuse and specular radiance, respectively. Based on the pBRDF model, the polarization of the diffuse component is governed by the Fresnel transmission coefficient $T$, and the specular component is governed by the Fresnel reflection coefficient $R$, with their polarization angles being orthogonal. The final output Stokes vector $\mathbf{s}^{out}$ encodes the complete polarization state, which is transformed via a Mueller matrix to render the polarization image $I_{\phi_{pol}}^{out} = \frac{1}{2}\mathbf{s}_{\phi_{pol}}^{out}[0]$.
    - **Design Motivation**: Unlike direct supervision on polarization parameters (such as supervising AoP/DoP in PANDORA), this method indirectly learns polarization states through an end-to-end rendering loss. This permits network training with only a single polarization image per view, significantly reducing data requirements.

3. **Automatic Angle Estimation and Theoretical Sufficiency Analysis**:
    - **Function**: To automatically recover polarization angles through network optimization without prior knowledge of the polarizer angles.
    - **Mechanism**: The polarized intensity $I_{\phi_{pol}} = I_{un}(1 + \rho\cos(2\phi - 2\phi_{pol}))$ is a sinusoidal function of the polarization angle. Given the Stokes vector and the captured polarization image, the polarization angle $\phi_{pol}$ can be solved. The authors' theoretical analysis indicates that: besides the polarization angle, the unknowns include the normal (2 parameters), diffuse coefficient $k_d$ (3 parameters), specular coefficient $k_s$ (3 parameters), and roughness $\eta$ (1 parameter), totaling 10 unknowns. Each viewpoint provides 3 constraints (RGB), making the system overdetermined with just 4 viewpoints.
    - **Design Motivation**: This theoretically guarantees the feasibility of the "single polarization image per view" strategy, providing a mathematical foundation for cost-effective acquisition. In experiments, the polarization angle estimation error is $<5^\circ$.

### Loss & Training
The total loss is a weighted sum of three terms: $\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_{mask} + 0.1\mathcal{L}_{eikonal}$. The RGB loss $\mathcal{L}_{rgb}$ compares the rendered polarization images with the inputs using an $\ell_1$ loss, filtered with GT masks to remove background noise. The mask loss $\mathcal{L}_{mask}$ utilizes binary cross-entropy (BCE) to supervise the predicted foreground masks. The Eikonal regularization term $\mathcal{L}_{eikonal}$ constrains the norm of the SDF gradients to be 1, ensuring a valid signed distance field.

## Key Experimental Results

### Main Results
Quantitative evaluations are conducted on a self-collected dataset of glossy objects (RedOx, GreenOx, Cat, Horse, Lays, etc.) against several SOTA methods. Ground truth (GT) geometry is captured using an industrial 3D scanner after objects are coated with a diffuse spray.

| Object | Metric | **Ours** | NeRO | InvRender | NVDiffRec | PhySG |
|------|------|----------|------|-----------|-----------|-------|
| RedOx | PSNR/CD | **30.88/2.23e-4** | 19.88/2.04e-3 | 22.47/2.28e-2 | 30.86/0.3005 | 16.42/2.36e-2 |
| GreenOx | PSNR/CD | **31.02/1.17e-4** | 16.98/1.08e-3 | 27.32/1.78e-2 | 30.66/0.2638 | 18.39/1.43e-2 |
| Cat | PSNR/CD | **24.83/9.88e-5** | 24.51/9.31e-3 | 22.32/1.82e-3 | 23.61/0.5936 | 16.32/1.48e-3 |
| Horse | PSNR/CD | **27.97/2.07e-4** | 22.22/1.20e-3 | 24.92/1.13e-3 | 27.15/0.1315 | 16.59/1.31e-3 |
| Lays | PSNR/CD | **30.82/1.01e-3** | 26.68/1.04e-3 | 25.61/1.21e-3 | 29.31/0.1152 | 17.41/2.66e-3 |

Comparison with the polarimetric method PANDORA on a synthetic Bust model: Mixed PSNR 26.53 vs 26.86, normal MAE 4.227° vs 4.096°. Performance is highly competitive, yet this method only requires a **single polarization image per view** instead of four.

### Ablation Study

| Configuration | RedOx PSNR | RedOx CD | Description |
|------|-----------|----------|------|
| Full model | 30.88 | 2.23e-4 | Full polarization rendering |
| w/o polarization | 26.29 | 3.01e-3 | No polarization, degrades to an enhanced version of Ref-NeRF |
| Diffuse only | 25.03 | 1.06e-3 | Diffuse only, degrades to VolSDF with masking |

### Key Findings
- Polarization information is highly critical for geometric reconstruction: omitting polarization increases CD from 2.23e-4 to 3.01e-3 (a 13.5x degradation), indicating that polarization cues successfully constrain normal estimation.
- While the "diffuse only" configuration shows better CD than the "without polarization" baseline (1.06e-3), its PSNR drops (25.03), suggesting that modeling the specular component is vital for high rendering quality.
- Although NVDiffRec achieves a high PSNR (30.86), its CD remains extremely poor (0.3005), illustrating that geometric estimation can fail severely even when producing visually pleasing rendered results—high PSNR does not equate to accurate geometry.
- The proposed method demonstrates robust performance under different polarizing angles, with estimated angle errors $<5^\circ$.

## Highlights & Insights
- **Extreme Cost-Effectiveness**: A cheap linear polarizer (costing only a few dollars) upgrades a standard camera to a polarimetric acquisition system without any calibration. This design philosophy of "minimal hardware modification augmented by algorithmic compensation" is highly appealing for practical applications.
- **Elegant Proof of Theoretical Sufficiency**: A concise analysis showing that with 10 unknowns and 3 constraints per view, 4 views are theoretically sufficient. This convincingly explains "why a single polarization image per view is enough".
- **Polarization Angle as an Optimizable Parameter**: Transforming a physical parameter that historically required precise measurement into a learnable parameter in the network is a successful case of "replacing calibration with optimization". This approach is highly transferable to other tasks where physical parameter calibration is difficult.

## Limitations & Future Work
- Currently, the polarizer orientation must remain fixed during acquisition. If polarizer directions vary across different viewpoints, the methods should theoretically still function but would require more views.
- The method does not handle transparent or translucent objects, focusing exclusively on opaque glossy surfaces.
- It exhibits color bleeding artifacts (the color from specular highlights may bleed onto adjacent diffuse regions).
- The GT geometry is obtained by scanning objects with a diffuse coating. The alignment between the scan and reconstruction coordinate systems relies heavily on manual initialization and non-rigid ICP, which might introduce evaluation errors.
- The potential of utilizing more modern 3D representations (such as 3D Gaussian Splatting) as the geometric backbone has not yet been explored.

## Related Work & Insights
- **vs PANDORA**: PANDORA relies on dedicated polarization cameras to capture complete four-directional polarization info and directly supervises the Stokes vectors. In contrast, this method requires only one polarization image per view and learns indirectly through rendering losses, vastly lowering cost while matching quality.
- **vs NeRO**: NeRO is tailored for highly reflective objects, achieving good geometry but sub-par reflectance decomposition. The proposed method utilizes polarization cues to simultaneously boost both geometric accuracy and decomposition quality.
- **vs Ref-NeRF/VolSDF**: This method incorporates a polarization rendering layer on top of these baselines, essentially using polarization physics priors to enhance the disentanglement of normals and materials.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of cost-effective polarization acquisition and end-to-end polarization angle optimization is a novel system-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprises synthetic and real-world data, multi-method comparisons, and complete ablation studies. The GT acquisition protocol is standardized.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical analysis, clear pipeline diagrams, and self-consistent physical derivations.
- Value: ⭐⭐⭐⭐ Highly practical, lowering the hardware entry barrier for polarimetric 3D reconstruction and showing potential for application on consumer-grade devices like smartphones.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ICTPolarReal: A Polarized Reflection and Material Dataset of Real World Objects](../../CVPR2026/3d_vision/ictpolarreal_a_polarized_reflection_and_material_dataset_of_real_world_objects.md)
- [\[CVPR 2025\] Floating No More: Object-Ground Reconstruction from a Single Image](floating_no_more_object-ground_reconstruction_from_a_single_image.md)
- [\[CVPR 2025\] GS-2DGS: Geometrically Supervised 2DGS for Reflective Object Reconstruction](gs-2dgs_geometrically_supervised_2dgs_for_reflective_object_reconstruction.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](../../ICCV2025/3d_vision/easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[CVPR 2025\] High-fidelity 3D Object Generation from Single Image with RGBN-Volume Gaussian Reconstruction Model](high-fidelity_3d_object_generation_from_single_image_with_rgbn-volume_gaussian_r.md)

</div>

<!-- RELATED:END -->
