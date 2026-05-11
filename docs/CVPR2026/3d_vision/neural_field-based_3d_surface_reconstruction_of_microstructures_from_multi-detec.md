---
title: >-
  [Paper Note] Neural Field-Based 3D Surface Reconstruction of Microstructures from Multi-Detector Signals in Scanning Electron Microscopy
description: >-
  [CVPR 2026][3D Vision][Scanning Electron Microscopy] This paper proposes NFH-SEM, a neural field-based hybrid framework that embeds the physical model of electron scattering in SEM into a neural field optimization pipeli…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Scanning Electron Microscopy"
  - "3D Reconstruction"
  - "Neural Fields"
  - "Microstructures"
  - "Photometric Stereo"
date: 2026-05-08
content_hash: 27adfc99568832c6
---

# Neural Field-Based 3D Surface Reconstruction of Microstructures from Multi-Detector Signals in Scanning Electron Microscopy

**Conference**: CVPR 2026
**arXiv**: [2508.04728](https://arxiv.org/abs/2508.04728)
**Code**: [https://github.com/zju3dv/NFH-SEM](https://github.com/zju3dv/NFH-SEM)
**Area**: 3D Vision
**Keywords**: Scanning Electron Microscopy, 3D Reconstruction, Neural Fields, Microstructures, Photometric Stereo

## TL;DR
This paper proposes NFH-SEM, a neural field-based hybrid framework that embeds the physical model of electron scattering in SEM into a neural field optimization pipeline, enabling high-fidelity 3D surface reconstruction of microstructures from multi-view, multi-detector SEM images. The framework achieves self-calibration and shadow-robust reconstruction at nanometer-scale accuracy (478 nm stacked features, 782 nm pollen textures, 1.559 μm fracture steps).

## Background & Motivation

1. **Background**: Scanning electron microscopy (SEM) is widely used in materials science, biology, and industrial manufacturing, producing high-resolution micro/nanoscale images. However, SEM images are inherently 2D intensity distributions of secondary electrons (SE) or backscattered electrons (BSE), and do not directly encode 3D information. Existing SEM 3D reconstruction methods fall into two main categories: multi-view methods (SfM+MVS) and single-view methods (photometric stereo, PS).

2. **Limitations of Prior Work**:
   - Multi-view methods frequently fail in weakly textured or repetitive regions commonly found in microscopic specimens.
   - Single-view PS methods require reference samples for detector calibration and are highly sensitive to shadow artifacts—shadowed regions cause distorted gradient estimates.
   - Hybrid methods that combine both approaches are still limited by calibration requirements and shadow artifacts, and their 2D heightmap representation cannot capture complex microstructures.
   - Learning-based methods (NeuS, 3DGS, feed-forward reconstruction) either lack large-scale SEM training data and fail to generalize, or rely on RGB optical rendering models that cannot exploit the geometric cues encoded in SEM signals.

3. **Key Challenge**: The physics of SEM signal generation (electron scattering) differs fundamentally from conventional RGB imaging, yet existing 3D reconstruction methods either ignore SEM physics (multi-view methods) or rely on simplified physical models requiring complex calibration procedures (single-view methods).

4. **Goal**: To design a neural field reconstruction framework capable of automatically learning SEM imaging physics, self-calibrating detector parameters, and autonomously separating shadow regions.

5. **Key Insight**: Model BSE signal scattering and detector response as a learnable forward model, embedded within the volume rendering pipeline of an SDF-based neural field, and jointly optimized with geometry.

6. **Core Idea**: By embedding a learnable BSE forward model into neural field optimization, the framework achieves self-calibration of SEM imaging physics and automatic shadow separation, yielding high-fidelity microscale 3D reconstruction.

## Method

### Overall Architecture
The input consists of multi-view, multi-detector SEM images (one SE image and four 4Q-BSE images per viewpoint). The pipeline proceeds in two stages: (1) SfM+MVS on multi-view SE images to obtain coarse initial geometry and camera parameters; (2) using the coarse geometry as initialization, an SDF-based neural field jointly optimizes geometry and BSE forward model parameters by fusing multi-view depth priors and 4Q-BSE photometric information. The output is a high-fidelity 3D surface mesh.

### Key Designs

1. **Learnable BSE Forward Model**:

   - **Function**: Maps predicted surface normals to 4Q-BSE intensities, enabling BSE images to directly supervise geometric learning in the neural field.
   - **Mechanism**: Traditional PS methods use $I_i(n) = d_i \cos(\varphi_i - \varphi_n)\tan(\theta_n) + c_i$ to compute gradients directly from BSE images, requiring calibration of $c/d$. NFH-SEM inverts this paradigm—a forward model with 16 learnable parameters, $\mathcal{F}_i(n) = \mathbf{R}(\theta_n)[d_i\cos(\varphi_i-\varphi_n)\sin(\theta_n) + c_i\cos(\theta_n)] + e_i$, maps normals to BSE intensities. The emission amplification term $\mathbf{R}(\theta)$ replaces the conventional $\sec(\theta)$ with a fourth-order polynomial to better fit real BSE responses. Each of the four quadrants has independent $c, d, e$ parameters and shares polynomial coefficients $p$.
   - **Design Motivation**: (1) Traditional analytic models are insufficiently accurate (confirmed by ablation); (2) the learnable formulation enables self-calibration without reference samples; (3) embedding the forward model into volume rendering effectively backpropagates gradient information into the neural field.

2. **Iterative Shadow Separation**:

   - **Function**: Automatically detects and excludes shadow regions in BSE images to prevent shadow contamination of geometric reconstruction.
   - **Mechanism**: Large discrepancies between the forward model output $\mathcal{F}(\hat{n};\hat{\Phi})$ and the actual BSE image $b$ are observed predominantly in shadowed regions, since shadows cannot be modeled by surface normal functions alone. A dynamic binary shadow mask is defined as $S = (|\mathcal{F}(\hat{n};\hat{\Phi}) - b| < \alpha d)$, where the threshold $\alpha d$ is updated dynamically with parameter $d$ during training. Since $d$ controls the sensitivity of BSE intensity to surface normals, setting the threshold proportionally prevents geometry-induced intensity variations from being misclassified as shadows.
   - **Design Motivation**: Shadows are pervasive in 4Q-BSE images, and supervising with shadow-corrupted images leads to severe geometric distortion. Iterative separation establishes a positive feedback loop—better shadow masks yield cleaner supervision, which in turn produces more accurate geometry and forward model estimates, enabling better shadow detection.

3. **Three-Stage Training Strategy**:

   - **Function**: Stably integrates multi-source geometric cues into the neural field.
   - **Mechanism**: **Stage I**: Initializes coarse geometry priors using only depth loss $\mathcal{L}_d$ and SDF regularization $\mathcal{R}_s$. **Stage II**: Introduces BSE loss $\mathcal{L}_{BSE}(1)$ (without shadow masking) and forward model regularization $\mathcal{R}_\Phi$ to jointly learn the normal-to-BSE mapping. **Stage III**: Activates the dynamic shadow mask $\mathcal{L}_{BSE}(S)$ to refine geometry and model parameters. Each stage runs for only 1,000 iterations, with total training taking approximately 2 minutes.
   - **Design Motivation**: Directly co-optimizing all components leads to instability; establishing a geometric prior before progressively introducing photometric supervision and shadow handling is essential.

4. **BSE Forward Model Regularization**:

   - **Function**: Constrains the four quadrant parameters from diverging excessively.
   - **Mechanism**: The variance of each parameter group $c, d, e$ is computed separately and summed as the regularization term $\mathcal{R}_\Phi = \text{Var}(c) + \text{Var}(d) + \text{Var}(e)$, encouraging quadrant consistency while permitting small deviations due to manufacturing tolerances.
   - **Design Motivation**: The four BSE detector quadrants are nominally symmetric by design, so their parameters should be close, yet manufacturing and installation tolerances introduce small differences.

### Loss & Training
The total loss is $\mathcal{L} = \lambda_1 \mathcal{L}_d + \lambda_2 \mathcal{R}_s + \lambda_3 \mathcal{L}_{BSE} + \lambda_4 \mathcal{R}_\Phi$, where $\mathcal{L}_d$ is a weighted depth loss (weighted by MVS confidence), $\mathcal{R}_s$ is the standard unit-norm SDF gradient constraint, and $\mathcal{L}_{BSE}$ is the MAE loss on the 4Q-BSE images.

## Key Experimental Results

### Main Results (Qualitative Comparison on Real Datasets)
Evaluated on TPL microstructures (Wukong, Lucy, Lion), peach pollen, and silicon carbide particles:
- Multi-view baselines recover only coarse shapes and fail to reconstruct smooth base surfaces and fine details (e.g., filaments, stacked steps, pollen textures).
- Single-view PS baselines recover limited texture but suffer from severe global shape distortion.
- Six learning-based methods (NeuS, 2DGS, PGSR, DN-Splatter, VGGT, MapAnything) all fail substantially when applied directly.
- NFH-SEM accurately recovers 478 nm printed layer stacking (Lucy sample), 782 nm pollen adhesion texture, and 1.559 μm fracture steps.

### Ablation Study (Simulated Dataset, Unit: nm)

| Configuration | Chamfer ↓ | Normal Angular Error ↓ | BSE Model Error ↓ |
|---|---|---|---|
| Input coarse model | 25.11 | 7.85° | - |
| Single-view PS | 512.22 | 12.99° | - |
| w/o BSE-$\mathcal{F}$ (direct gradient supervision) | 135.61 | 7.48° | - |
| w/o Poly-$\mathbf{R}$ (simplified emission model) | 19.96 | 4.34° | 7.16 |
| w/o 4Q-Var (shared quadrant parameters) | 19.90 | 3.91° | 1.35 |
| w/o S-Mask (no shadow mask) | 29.38 | 4.36° | 0.61 |
| **Full model** | **17.48** | **3.70°** | **0.27** |

### Key Findings
- The learnable forward model is the most critical component—replacing it with direct gradient supervision (w/o BSE-$\mathcal{F}$) degrades Chamfer distance by 7.75×.
- The polynomial emission term (Poly-$\mathbf{R}$) reduces BSE modeling error from 7.16 to 0.27 compared to the simplified $\sec(\theta)$ formulation.
- Removing the shadow mask increases Chamfer distance from 17.48 to 29.38, demonstrating that shadow separation is essential for geometric accuracy.
- Shadow detection achieves an average accuracy of 81.7%.
- The entire training requires approximately 2 minutes on a single RTX 4090 GPU across 3,000 iterations.

## Highlights & Insights
- **Paradigm of embedding physical models into neural fields**: Rather than computing gradients directly from physics formulas to serve as supervision, the physical model is embedded as a differentiable layer within the optimization—a paradigm generalizable to other domains requiring specialized imaging physics (e.g., X-ray, ultrasound).
- **Elegant realization of self-calibration**: By treating detector parameters as learnable variables and optimizing them jointly, the framework eliminates the cumbersome reference-sample calibration required by traditional methods, substantially lowering the barrier to practical use.
- **Positive feedback mechanism in iterative shadow separation**: Using forward model residuals to define the shadow mask and adaptively adjusting the threshold via the physical parameter $d$ forms a self-reinforcing cycle—a particularly elegant engineering design.

## Limitations & Future Work
- The framework assumes a homogeneous electron emission coefficient, which may not hold for multi-material composite specimens.
- Severely occluded microporous structures where all quadrants are shadowed may be irrecoverable.
- Charging effects in low-conductivity samples cause pixel drift that may compromise multi-view alignment.
- The dataset, while pioneering, is limited in scale (only three sample categories).
- Future extensions may include piecewise emission coefficient estimation for heterogeneous materials and validation across a broader range of specimen types.

## Related Work & Insights
- **vs. Agisoft Metashape (multi-view baseline)**: Multi-view methods rely solely on SE images and fail in weakly textured regions; NFH-SEM additionally exploits the photometric information in 4Q-BSE images to compensate for insufficient feature matching.
- **vs. Single-view PS**: PS methods require calibration and are severely affected by shadows; NFH-SEM addresses both fundamental limitations through the learnable forward model and shadow separation.
- **vs. NeuS/3DGS**: These methods are built on RGB rendering models and cannot interpret the geometric encoding in SEM signals; NFH-SEM bridges this domain gap by embedding SEM physics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First complete adaptation of neural field methods to SEM imaging physics; self-calibration and shadow separation strategies are elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Both real and simulated data are evaluated with comprehensive ablations, though quantitative ground-truth comparison on real data is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — SEM physical background is clearly introduced, method derivations are rigorous, and figures are well-crafted.
- Value: ⭐⭐⭐⭐⭐ — Significant application value for microscale 3D characterization in materials science and biology; pioneering work at the intersection of SEM and neural fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EMGauss: Continuous Slice-to-3D Reconstruction via Dynamic Gaussian Modeling in Volume Electron Microscopy](emgauss_continuous_slice-to-3d_reconstruction_via_dynamic_gaussian_modeling_in_v.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)
- [\[CVPR 2026\] Neu-PiG: Neural Preconditioned Grids for Fast Dynamic Surface Reconstruction on Long Sequences](neu-pig_neural_preconditioned_grids_for_fast_dynamic_surface_reconstruction_on_l.md)
- [\[ICLR 2026\] LiTo: Surface Light Field Tokenization](../../ICLR2026/3d_vision/lito_surface_light_field_tokenization.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)

</div>

<!-- RELATED:END -->
