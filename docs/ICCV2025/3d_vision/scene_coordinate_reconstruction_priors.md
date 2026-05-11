---
title: >-
  [Paper Note] Scene Coordinate Reconstruction Priors
description: >-
  [ICCV 2025][3D Vision][Scene Coordinate Regression] This paper proposes a probabilistic training framework for scene coordinate regression (SCR) that introduces hand-crafted depth distribution priors and a learned prior…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Scene Coordinate Regression"
  - "Reconstruction Priors"
  - "3D Diffusion Models"
  - "Visual Relocalization"
  - "Structure from Motion"
date: 2026-05-08
content_hash: 7618a28a69276520
---

# Scene Coordinate Reconstruction Priors

**Conference**: ICCV 2025
**arXiv**: [2510.12387](https://arxiv.org/abs/2510.12387)
**Code**: [nianticspatial.github.io/scr-priors](https://nianticspatial.github.io/scr-priors/)
**Area**: 3D Vision
**Keywords**: Scene Coordinate Regression, Reconstruction Priors, 3D Diffusion Models, Visual Relocalization, Structure from Motion

## TL;DR

This paper proposes a probabilistic training framework for scene coordinate regression (SCR) that introduces hand-crafted depth distribution priors and a learned prior based on a 3D point cloud diffusion model, significantly improving scene reconstruction quality, camera pose estimation, and downstream task performance under insufficient multi-view constraints.

## Background & Motivation

Scene coordinate regression (SCR) models are a powerful class of implicit scene representations widely used in visual relocalization and structure from motion (SfM). The core idea is to train a scene-specific network that maps image patches to 3D scene coordinates. The ACE framework can train efficient SCR models within minutes, and ACE0 further enables self-supervised SfM.

However, SCR fundamentally relies on classical triangulation principles—whether explicit or implicit—and triangulation degenerates under insufficient multi-view constraints, such as in texture-poor regions, repetitive structures, or reflective surfaces. This leads to noisy, distorted, or severely outlying predicted 3D scene points, ultimately degrading pose estimation and novel view synthesis quality.

The authors observe that degenerate scene representations clearly violate plausible geometry—for example, half the point cloud of a room scattering into open space, or camera trajectories passing through walls. Such implausibility can be corrected through high-level reconstruction priors. Yet existing SCR methods make almost no use of scene-agnostic prior knowledge; only ACE's feature encoder is pretrained, but as a low-level component it cannot guarantee global consistency of the final scene representation.

## Method

### Overall Architecture

The core contribution of this paper is to reinterpret SCR training as maximum likelihood learning. Given mapping images $\mathcal{I}_M$ and camera poses $\mathbf{h}^*$, the posterior probability of a scene point $\mathbf{y}$ is:

$$-\log p(\mathbf{y} \mid \mathbf{h}^*, \mathcal{I}_M) \propto -\log p(\mathbf{h}^*, \mathcal{I}_M \mid \mathbf{y}) - \log p(\mathbf{y})$$

where the first term corresponds to the reprojection loss $L_{\text{reproj}}$ and the second term is the prior $L_{\text{reg}} = -\log p(\mathbf{y})$. The key innovation lies in **jointly optimizing** the reprojection error and the prior, rather than alternating between an initialization loss and a reprojection loss as in the original ACE.

The overall pipeline builds on the ACE framework, decomposing the SCR model into a scene-agnostic feature extractor $f_B$ (pretrained and frozen) and a scene-specific regression head $f_H$. During training, mini-batches are sampled from a feature buffer; after predicting 3D scene points, both the reprojection loss and prior regularization are applied. The priors are used only during training and do not affect inference efficiency.

### Key Designs

**1. Depth Distribution Prior (RGB)**

This prior models a plausible distribution over the depth values of predicted scene coordinates. A Laplacian distribution $\text{Lap}(d \mid \mu, b)$ is adopted, with mean $\mu=1.73$ m and bandwidth $b=60$ cm fitted on the ScanNet training set. Two usage modes are proposed:

- **Negative Log-Likelihood (NLL)**: Directly computes $\log p(\mathbf{y}_i) = \lambda_{\text{reg}} \log \text{Lap}(d_i \mid \mu, b)$ for each pixel's depth value, equivalent to an L1 loss pulling depths toward the empirical mean.
- **Wasserstein Distance (WD)**: Computes the Wasserstein distance between the mini-batch depth set $\{d_i\}$ and the target Laplace distribution, leveraging ACE's random mini-batch sampling to approximate the overall depth distribution, thereby constraining not only the mean but also the variance.

**2. Depth Prior (RGB-D)**

When RGB-D input is available, the broad depth distribution is replaced by a narrow distribution centered at the measured depth $d_i^*$: $\log p(\mathbf{y}_i) = \lambda_{\text{reg}} \log \text{Lap}(d_i \mid d_i^*, b')$, with tolerance bandwidth $b'=10$ cm. Compared to DSAC\*'s hard-switching strategy (switching to reprojection loss within a 10 cm threshold), this probabilistic soft constraint provides more stable optimization.

**3. Point Cloud Diffusion Prior (RGB)**

A 3D point cloud denoising diffusion model is pretrained to encode knowledge of plausible indoor scene layouts. Using the score matching relationship, the noise estimate of the diffusion model is proportional to the gradient of the log-likelihood:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\lambda_{\text{reg}} \epsilon_\theta(\mathbf{x}_\tau, \tau)$$

The architecture uses PVCNN (Point-Voxel CNN), trained on 706 ScanNet training scenes, randomly sampling 5,120 points per step, with 200 total diffusion steps and 100k iterations on a single V100.

Integration strategy: the diffusion prior is applied after step 5k of ACE training; the diffusion timestep $T/20$ is aligned to step 5k and linearly interpolated to 0; points with reprojection error below 30 pixels are excluded from the prior (considered already sufficiently constrained by multi-view observations).

### Loss & Training

The final optimization objective is: $L = L_{\text{reproj}} + L_{\text{reg}}$

where $L_{\text{reg}}$ is one of the three priors above. The prior is balanced by weight $\lambda_{\text{reg}}$. The diffusion prior acts directly on the optimization of scene coordinates in gradient form.

## Key Experimental Results

### Main Results: SfM Reconstruction (ScanNet + Indoor6)

| Method | Reg. Rate↑ | ATE/RPE (cm)↓ | Median Error (cm/°)↓ | PSNR 1/7↑ | PSNR 60/60↑ |
|---|---|---|---|---|---|
| ACE0 (RGB) | 98.1% | 26.6/4.0 | 19.7/9.0 | 30.2 | 22.3 |
| + Laplace NLL | **98.9%** | **25.4/3.5** | **17.5/8.8** | 30.2 | 22.2 |
| + Laplace WD | 98.7% | 25.9/3.6 | 17.5/6.8 | 30.3 | 21.7 |
| + Diffusion | 98.6% | 26.5/3.8 | 18.8/8.9 | 30.2 | 22.4 |
| ACE0 + DSAC\* (RGB-D) | 96.2% | 29.2/6.0 | 20.9/5.9 | 30.0 | 21.9 |
| + Laplace NLL (RGB-D) | **98.9%** | **18.3/3.5** | **12.8/4.4** | **30.6** | **22.9** |

The diffusion prior is particularly effective on the Indoor6 dataset: registration rate improves from 57.1% to 61.8% (+4.7%), and PSNR improves from 13.5 to 14.6 dB (+1.1 dB).

### Main Results: Relocalization (7Scenes)

| Method | Training Time | Model Size | Avg. Acc. (5cm/5°)↑ |
|---|---|---|---|
| ACE | 5 min | 4 MB | 97.1% |
| + Laplace NLL | 4.5 min | 4 MB | 97.3% |
| + Laplace WD | 4.5 min | 4 MB | 97.2% |
| + Diffusion | 8 min | 4 MB | **97.7%** |
| GLACE | 6 min | 9 MB | 95.6% |
| + Diffusion | 9 min | 9 MB | 95.9% |

On the most challenging Stairs scene, the diffusion prior improves ACE's accuracy from 81.9% to 86.2% (+4.3%).

### Ablation Study

- The **depth distribution prior** simplifies the optimization objective, slightly reducing training time (5 min → 4.5 min) while improving performance.
- The **diffusion prior** adds approximately 3 minutes of training time (5 min → 8 min) but provides the strongest global regularization signal.
- The **RGB-D prior**'s probabilistic soft constraint significantly outperforms DSAC\*'s hard-switching strategy: ATE drops from 29.2 cm to 18.3 cm.
- Priors are used only during training and **do not affect inference time or model size**.
- The diffusion model is trained on only ~700 indoor scenes with limited generation quality, yet it is already sufficiently effective as a prior.
- Skipping the diffusion prior for points with reprojection error below 30 pixels avoids interfering with already well-constrained regions.

## Highlights & Insights

1. The **probabilistic reinterpretation** is the most elegant contribution of this paper—unifying SCR training under a maximum a posteriori (MAP) framework makes the introduction of various priors both natural and theoretically grounded, eliminating the need to design ad-hoc regularization strategies case by case.
2. The idea of using **diffusion models as priors rather than generators** is highly inspiring: even a 3D diffusion model with limited generation quality can effectively guide reconstruction through the gradient of its learned distribution, reducing requirements on the fidelity of 3D diffusion models.
3. The **prior-at-training, zero-overhead-at-inference** design is highly practical and perfectly preserves the inference efficiency advantages of the ACE family.
4. The RGB-D prior results demonstrate that probabilistic soft constraints are better suited than hard-switching strategies for fusing multimodal information—an insight generalizable to other multimodal fusion scenarios.
5. The alignment strategy between the diffusion prior and the SCR training process (starting at step 5k, linearly interpolating timesteps, skipping low-error points) is elegantly designed and reflects a deep understanding of the characteristics of both processes.

## Limitations & Future Work

1. **Validation limited to indoor scenes**: Priors are fitted/trained on ScanNet; outdoor scenes require different depth distribution models and more diverse training data.
2. **Limited diffusion model expressiveness**: Using PVCNN trained on only ~700 scenes, the generated point clouds lack detail and have limited capacity to encode complex scene layouts.
3. **Diffusion prior lacks conditioning signals**: The unconditional diffusion model provides only generic priors; incorporating image or semantic conditioning could enable more precise guidance.
4. Results on the Indoor6 dataset exhibit high variance, making definitive conclusions difficult.
5. The depth distribution prior assumes a Laplace family, which may be insufficiently flexible for multimodal depth distributions (e.g., multi-story scenes).

## Related Work & Insights

- **ACE/ACE0/GLACE series**: The foundational framework of this paper, demonstrating the efficiency of SCR methods. The proposed priors can be seamlessly integrated into any derivative of these frameworks.
- **DiffusioNeRF**: Inspired the idea of using diffusion models as reconstruction priors, but DiffusioNeRF operates on 2.5D patches and requires rendering, incurring high overhead. This paper operates directly in 3D space and is more efficient.
- **DUSt3R/MASt3R**: Feed-forward methods achieving strong reconstruction priors through large-scale pretraining, but difficult to scale to large image collections. The prior strategy proposed here is complementary.
- **DSAC\***: An early method for incorporating depth information into SCR; the probabilistic framework proposed here offers a more elegant alternative.
- **PVCNN**: An efficient architecture for 3D point cloud processing; this paper demonstrates its viability for scene-level diffusion.

## Rating

- Novelty: ⭐⭐⭐⭐ — Probabilistically reinterpreting SCR training and introducing multiple priors, with clear reasoning and theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, multiple prior variants, dual-task validation across SfM and relocalization, with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation is clearly articulated, the probabilistic framework is elegantly derived, and experiments are well organized.
- Value: ⭐⭐⭐⭐ — The plug-and-play prior strategy offers direct practical value to the SCR community, and the probabilistic perspective provides methodological inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAS: Segment Any 3D Scene with Integrated 2D Priors](sas_segment_any_3d_scene_with_integrated_2d_priors.md)
- [\[ICCV 2025\] Proactive Scene Decomposition and Reconstruction](proactive_scene_decomposition_and_reconstruction.md)
- [\[ICCV 2025\] Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction](back_on_track_bundle_adjustment_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
