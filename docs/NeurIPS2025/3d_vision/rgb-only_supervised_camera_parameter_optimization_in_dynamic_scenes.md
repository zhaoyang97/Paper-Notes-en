---
title: >-
  [Paper Note] RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes
description: >-
  [NeurIPS 2025][3D Vision][camera parameter estimation] ROS-Cam proposes a camera parameter (focal length + pose) optimization method for dynamic scenes supervised solely by a single RGB video. It achieves state-of-the-ar…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "camera parameter estimation"
  - "dynamic scene"
  - "RGB-only supervision"
  - "outlier-aware optimization"
  - "visual odometry"
date: 2026-05-08
content_hash: d27e2922c6e87fe6
---

# RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes

**Conference**: NeurIPS 2025
**arXiv**: [2509.15123](https://arxiv.org/abs/2509.15123)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: camera parameter estimation, dynamic scene, RGB-only supervision, outlier-aware optimization, visual odometry

## TL;DR
ROS-Cam proposes a camera parameter (focal length + pose) optimization method for dynamic scenes supervised solely by a single RGB video. It achieves state-of-the-art accuracy and fastest runtime across 5 datasets via three key contributions: a patch-wise tracking filter for sparse, robust correspondences; a Cauchy distribution-based outlier-aware joint optimization that adaptively down-weights moving objects; and a two-stage optimization strategy grounded in Softplus/convex minimax analysis.

## Background & Motivation

**Background**: COLMAP is the de facto standard for camera parameter estimation in static scenes, but handling dynamic scenes requires ground-truth motion masks to exclude moving objects. Numerous improved methods have emerged (categorized in Table 1), yet the vast majority rely on additional priors: GT focal length (CF-3DGS, Nope-NeRF), GT motion masks (GFlow, LEAP-VO), metric depth (DROID-SLAM), or GT 3D point clouds and poses (DUSt3R, Monst3R, Cut3R). Such priors are generally unavailable in casually captured videos.

**Limitations of Prior Work**: (a) The few existing RGB-only methods (VGGSfM, FlowMap, casualSAM) either cannot handle dynamic scenes or rely on multiple pretrained dense prediction models (RAFT/CoTracker/MiDAS) for pseudo-supervision — failure of any single model degrades overall performance; (b) none can adaptively exclude moving outliers without GT motion priors; (c) computational latency is high.

**Key Challenge**: Can camera focal length and pose be estimated accurately and efficiently in dynamic scenes using only RGB video — the most minimal form of supervision? This simultaneously requires resolving three challenges: sparse and robust tracking correspondences, adaptive exclusion of moving objects, and efficient stable optimization convergence.

**Key Insight**: The method establishes maximally sparse hinge-based tracking correspondences (relying only on a point-tracking model rather than dense predictors), models uncertainty via a Cauchy distribution to down-weight moving outlier point pairs (rather than segmenting or detecting them), and employs a two-stage optimization strategy to balance fast and precise convergence. **Core Idea**: Extract the most robust information (sparse high-gradient tracking points) with minimal dependencies (PT model only), and jointly optimize camera parameters and 3D calibration points using the most robust uncertainty model (heavy-tailed Cauchy distribution).

## Method

### Overall Architecture
Given an $N$-frame RGB video, ROS-Cam first extracts $H$ sparse, robust tracking trajectories from the output of a PT model via Patch-wise Tracking Filters, which serve as pseudo-supervision. Each trajectory corresponds to a learnable 3D calibration point $P^{cali}$. The method then jointly optimizes the calibration points, focal length $f$, rotation quaternions $Q$, translations $t$, and uncertainty parameters $\Gamma$. The estimated camera parameters are finally fed into 4DGS for 4D scene reconstruction.

### Key Designs

1. **Patch-wise Tracking Filters**:

    - **Function**: Extract the sparsest and most robust tracking trajectories from a pretrained point-tracking model's output as pseudo-supervision for optimization.
    - **Mechanism**: A cascade of four filters — (a) *Patch-wise Texture Filter*: divides the image into $w \times w$ patches, computes intensity variance per patch, and retains only texture-rich patches (high variance = easier to track); (b) *Patch-wise Gradient Filter*: selects the pixel with the maximum gradient magnitude within each selected patch as the tracking point; (c) *Visibility Filter*: discards trajectories that become invisible at any frame (avoiding re-appearance tracking errors due to occlusion); (d) *Patch-wise Distribution Filter*: when multiple trajectories fall within the same patch, retains only the one with the highest gradient to ensure spatially uniform coverage.
    - **Design Motivation**: The attention mechanism of PT models tracks texture-rich/high-gradient points more accurately (exploiting rather than fighting the PT model's characteristics). Maximal sparsity significantly reduces the number of learnable parameters and computational cost while improving robustness.

2. **Outlier-aware Joint Optimization**:

    - **Function**: Adaptively reduce the influence of moving object correspondences on optimization without relying on any motion prior or mask.
    - **Mechanism**: Each 3D calibration point $P^{cali}_h$ is associated with an uncertainty parameter $\Gamma_h$ (kept positive via Softplus). An Average Cumulative Projection (ACP) Error is proposed, accumulating and averaging the projection error of each calibration point across all frames. The Cauchy Loss is formulated as $\mathcal{L} = \frac{1}{H}\sum \log\!\left(\Gamma + E^2_{ACP}/\Gamma\right)$, where calibration points on moving objects incur high ACP error due to triangulation inconsistency, yielding large learned $\Gamma$ and thus being down-weighted. Rotation is parameterized via quaternions (avoiding orthogonality constraints).
    - **Design Motivation**: The Cauchy distribution handles heavy tails (large errors from moving outliers) more effectively than the Gaussian distribution, and its log-likelihood form naturally yields a Cauchy Loss that is inherently robust to large errors. Associating uncertainty with sparse 3D points (rather than 2D pixels) drastically reduces parameter count — casualSAM has $424 \times 270 \times 480$ uncertainty parameters on NeRF-DS scenes, whereas ROS-Cam has only 440.

3. **Two-stage Optimization Strategy**:

    - **Function**: Accelerate convergence and avoid local minima.
    - **Mechanism**: *Stage 1* — fixes $\Gamma^{raw}=1$ (no uncertainty learning) and optimizes only $P^{cali}/f/Q/t$, leveraging the approximation $\text{Softplus}(1)\approx\ln 2$ for rapid convergence to a coarse solution. *Stage 2* — initializes $\Gamma^{raw}$ from Stage 1 ACP errors (based on the optimal solution $x^*=\sqrt{O}$ of the convex sub-term $\Phi = x + O/x$ in the Cauchy Loss), then jointly optimizes all parameters, enabling correct down-weighting of moving objects for further refinement.
    - **Design Motivation**: Joint optimization of $\Gamma$ from scratch is prone to unstable convergence due to the non-convexity of the Cauchy Loss. The two-stage strategy is derived from analytical examination of the asymptotic behavior of Softplus and the convex sub-term of the Cauchy Loss — a theoretically grounded engineering design rather than a heuristic trick.

### Loss & Training
Total loss: $\mathcal{L}_{total} = \mathcal{L}_{cauchy} + R_{depth}$. $\mathcal{L}_{cauchy}$ is the Cauchy Loss (primary term); $R_{depth} = \frac{1}{N}\sum -\text{ReLU}(P^{proj\text{-}homo}[:,3])$ is a depth regularization term encouraging positive depth. Stage 1 runs for 200 iterations; Stage 2 runs for 50 iterations.

## Key Experimental Results

### Main Results

| Method | Supervision | NeRF-DS PSNR↑ | DAVIS PSNR↑ | TUM ATE↓ | TUM RPE_t↓ | Runtime |
|--------|-------------|---------------|-------------|----------|-----------|---------|
| COLMAP (w/ mask) | GT motion mask | 32.17 | — | — | — | 1.5 h |
| casualSAM | RGB-only | 21.23 | 19.03 | 0.071 | 0.010 | 10.5 h |
| Robust-CVD | RGB-only | — | — | 0.153 | 0.026 | — |
| **ROS-Cam** | **RGB-only** | **33.55** | **22.29** | **0.065** | **0.010** | **0.83 h** |

| Method | iPhone Avg. PSNR↑ | Type |
|--------|------------------|------|
| Record3D | ~25.5 | LiDAR sensor |
| COLMAP (w/o mask) | ~21.0 | RGB-only (static) |
| **ROS-Cam** | **~25.2** | RGB-only |

### Ablation Study

| Configuration | NeRF-DS PSNR↑ | Notes |
|---------------|---------------|-------|
| Full (ROS-Cam) | 33.55 | Complete method |
| w/o two-stage | 25.95 | Removed two-stage → unstable convergence |
| w/o $\Gamma$ | 26.44 | Removed uncertainty → cannot exclude moving outliers |
| w/o $E_{ACP}$ | 23.56 | Removed ACP error → worst performance |
| w/o texture filter | 25.99 | Degraded tracking point quality |
| w/o gradient filter | 26.04 | Degraded tracking point quality |
| w/o distribution filter | 26.02 | Non-uniform clustering of tracking points |

### Key Findings
- ROS-Cam achieves PSNR 33.55 on NeRF-DS, surpassing COLMAP with GT motion masks (32.17) — pure RGB supervision outperforms GT mask supervision.
- Runtime scales approximately linearly (~1/800 hours per frame), whereas COLMAP scales near-exponentially — the advantage grows for longer videos.
- Pose accuracy on TUM-dynamics (ATE = 0.065) is competitive with or superior to methods using stronger supervision, such as DROID-SLAM (0.043, requires GT focal length + metric depth) and Monst3R (0.098, requires GT 3D point clouds).
- Ablation results confirm that every component contributes significantly; the ACP error and uncertainty parameters are the two most critical designs.

## Highlights & Insights
- **"Minimal supervision = maximal generalization" philosophy**: By minimizing reliance on pretrained models and external priors, the method avoids the cascading risk of any single prior source failing.
- The sparse association of uncertainty parameters with 3D points (rather than 2D pixels) is an elegant engineering decision — reducing parameter count by orders of magnitude while maintaining performance.
- The choice of Cauchy Loss is theoretically motivated (heavy-tail robustness) and naturally complements the ACP error, forming a self-consistent robust estimation framework.
- The two-stage optimization is not a simple coarse-to-fine scheme, but a theoretically guided design derived from analytical examination of Softplus asymptotic behavior and the convex sub-term of the Cauchy Loss.

## Limitations & Future Work
- Assumes a pinhole camera model with constant focal length; not applicable to fisheye or zoom lenses.
- Extreme dynamic scenes (where nearly all objects are in motion with very few static points) may cause triangulation degeneracy.
- The accuracy ceiling of RGB-only methods remains below that of direct measurement approaches such as LiDAR.
- Performance on some high-speed motion sequences in MPI-Sintel (e.g., ambush_4/5) falls below casualSAM.

## Related Work & Insights
- **Sparse vs. dense trade-off**: This work demonstrates that for camera estimation, "sparse but high-quality" correspondences significantly outperform "dense but noisy" pseudo-supervision.
- **Cauchy distribution in robust estimation**: The framework is generalizable to other optimization problems requiring outlier resistance.
- **Impact on 4D reconstruction pipelines**: ROS-Cam can serve as a front-end replacement for COLMAP in any 4D reconstruction method, with particular value in dynamic scenes.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of minimal supervision and Cauchy robust estimation constitutes a distinctive design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Five datasets (3 real + 1 synthetic + 1 iPhone), multi-dimensional evaluation (NVS + pose + runtime), and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Method motivation is clearly articulated with solid theoretical analysis.
- Value: ⭐⭐⭐⭐ — Directly applicable to 3D/4D reconstruction of casually captured dynamic videos.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[NeurIPS 2025\] Galactification: Painting Galaxies onto Dark Matter Only Simulations Using a Transformer-Based Model](galactification_painting_galaxies_onto_dark_matter_only_simulations_using_a_tran.md)
- [\[NeurIPS 2025\] Every Camera Effect, Every Time, All at Once: 4D Gaussian Ray Tracing for Physics-based Camera Effect Data Generation](every_camera_effect_every_time_all_at_once_4d_gaussian_ray_tracing_for_physics-b.md)
- [\[CVPR 2026\] PointTPA: Dynamic Network Parameter Adaptation for 3D Scene Understanding](../../CVPR2026/3d_vision/pointtpa_dynamic_network_parameter_adaptation_for_3d_scene_understanding.md)

</div>

<!-- RELATED:END -->
