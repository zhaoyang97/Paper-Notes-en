---
title: >-
  [Paper Note] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos
description: >-
  [CVPR 2025][3D Vision][Dynamic videos] MegaSaM achieves accurate, fast, and robust estimation of camera parameters and depth maps from casually captured dynamic videos by integrating monocular depth priors, motion probability maps, and uncertainty-aware global BA into a deep visual SLAM framework, significantly outperforming existing methods on both synthetic and real-world datasets.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Dynamic videos"
  - "visual SLAM"
  - "camera tracking"
  - "depth estimation"
  - "uncertainty-aware"
date: 2026-05-08
content_hash: 4ca427b58ce0c831
---

# MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos

**Conference**: CVPR 2025  
**arXiv**: [2412.04463](https://arxiv.org/abs/2412.04463)  
**Code**: [https://mega-sam.github.io](https://mega-sam.github.io) (Project Page)  
**Area**: 3D Vision  
**Keywords**: Dynamic videos, visual SLAM, camera tracking, depth estimation, uncertainty-aware

## TL;DR
MegaSaM achieves accurate, fast, and robust estimation of camera parameters and depth maps from casually captured dynamic videos by integrating monocular depth priors, motion probability maps, and uncertainty-aware global BA into a deep visual SLAM framework, significantly outperforming existing methods on both synthetic and real-world datasets.

## Background & Motivation

**Background**: Recovering camera parameters and scene geometry from videos (SfM/SLAM) is a fundamental problem in computer vision. Traditional methods work well in static scenes with large baseline motion but often fail on casually captured handheld videos.

**Limitations of Prior Work**: Casual videos present three major challenges: (1) limited camera motion parallax (e.g., near-pure rotation, near-static motion) leading to insufficient geometric constraints; (2) moving objects in the scene violating the static scene assumption; (3) unknown and highly diverse focal lengths. Existing solutions are either computationally expensive (CasualSAM requires per-sequence network fine-tuning, RoDynRF relies on costly NeRF optimization), fragile in dynamic scenes (Particle-SfM, LEAP-VO rely on motion segmentation but lack robustness), or lack accuracy (MonST3R).

**Key Challenge**: There exists a triangular trade-off among accuracy, speed, and robustness. Accurate methods are too slow, fast methods are insufficiently robust, and robust methods have limited accuracy.

**Goal**: How to achieve accurate and robust camera tracking and depth estimation from casual videos containing dynamic objects, limited parallax, and unknown focal lengths, while maintaining high speed?

**Key Insight**: The authors find that the deep visual SLAM framework with a differentiable BA layer (DROID-SLAM) inherently provides a high-quality foundation for accuracy and efficiency. By "meticulously modifying" its training and inference schemes, it can handle complex dynamic scenes. The key lies in correctly integrating external priors (monocular depth, motion segmentation) and parameter observability analysis.

**Core Idea**: Introducing motion probability maps to downweight dynamic objects + monocular depth initialization + uncertainty-aware global BA within a deep visual SLAM framework to address the SfM problem in casual dynamic videos through a three-pronged approach.

## Method

### Overall Architecture
MegaSaM is built upon the differentiable BA layer of DROID-SLAM, maintaining two state variables: per-frame disparity maps and camera poses, which are updated by iteratively minimizing the weighted reprojection error. The system is split into two modules: a front-end (sliding-window BA for real-time keyframe tracking) and a back-end (global BA for refining all frames). Three key components are integrated: learning motion probability maps to handle dynamic objects, integrating monocular depth priors to assist initialization and regularization, and adaptively deciding whether to use depth regularization based on uncertainty analysis. Finally, a consistent depth optimization can be optionally run to obtain high-resolution depth maps.

### Key Designs

1. **Motion Probability Map Learning and Two-Stage Training**:

    - **Function**: Identify and downweight dynamic objects in the video, preventing them from corrupting the static scene assumption of BA.
    - **Mechanism**: A subnet $F_m$ is added to predict the per-frame motion probability map $\mathbf{m}_i$ based on the current frame and its neighboring keyframes. This motion probability map is multiplied by the pairwise optical flow confidence $\hat{\mathbf{w}}_{ij}$ to form the final weight in BA: $\tilde{\mathbf{w}}_{ij} = \hat{\mathbf{w}}_{ij} \mathbf{m}_i$. The training adopts a two-stage strategy: the first stage pre-trains the original SLAM network F on static videos (learning optical flow and confidence); the second stage freezes F and fine-tunes only $F_m$ on dynamic videos (learning motion probability maps), supervised by cross-entropy loss + camera loss.
    - **Design Motivation**: Directly training the entire system end-to-end on dynamic videos causes instability, as the differentiable BA layer is highly sensitive to gradient noise generated by dynamic objects. The two-stage strategy decouples motion learning from correspondence learning, which crucially ensures training stability.

2. **Monocular Depth Prior Integration and Initialization**:

    - **Function**: Provide geometric constraints under limited parallax conditions, resolving degradation issues when parallax is insufficient.
    - **Mechanism**: DepthAnything is used to predict per-frame affine-invariant disparity, and UniDepth's metric depth is used to estimate the global scale and shift alignment: $D_i^{align} = \hat{\alpha} D_i^{rel} + \hat{\beta}$. During the training stage, disparity is initialized with GT scale/shift; during the inference stage, it is aligned using UniDepth's metric depth via median alignment. The focal length is initialized to the median predicted by UniDepth. A depth regularization term is added to the cost function of front-end BA: $\mathcal{C} = \sum \| \hat{\mathbf{u}}_{ij} - \mathbf{u}_{ij} \|^2_{\Sigma} + w_d \sum \| \hat{\mathbf{d}}_i - D_i^{align} \|^2$.
    - **Design Motivation**: The original constant initialization (all 1s) in DROID-SLAM fails to converge to the correct solution under low parallax and complex dynamic scenes. Combining two complementary monocular depth models (DepthAnything provides accurate and consistent depth, while UniDepth provides the scene scale) offers reliable initialization and regularization.

3. **Uncertainty-Aware Global BA**:

    - **Function**: Adaptively decide whether to use depth regularization and focal length optimization during global BA.
    - **Mechanism**: The epistemic uncertainty of parameters is estimated using Laplace approximation: $\Sigma_\theta \approx \text{diag}(-\mathbf{H}(\theta^*))^{-1}$, where H is the Hessian matrix of the BA cost function. The intuition is: if perturbing a parameter has almost no effect on the reprojection error, the parameter is unobservable, indicating high uncertainty. Specifically, the median of the disparity Hessian $\text{med}(\text{diag}(\mathbf{H_d}))$ and the focal length Hessian $H_f$ are computed. The depth regularization weight is adaptively set to $w_d = \gamma_d \exp(-\beta_d \cdot \text{med}(\text{diag}(\mathbf{H_d})))$, reinforcing depth regularization when disparity uncertainty is high. When focal length uncertainty is high ($H_f < \tau_f$), focal length optimization is frozen.
    - **Design Motivation**: This resolves a key dilemma: if a video has a sufficient baseline, depth regularization is redundant or even harmful (introducing monocular depth error); if the video is near-pure rotation, it will degrade without depth regularization. Uncertainty analysis provides an automatic decision mechanism without the need for manual parameter tuning.

### Loss & Training
Two-stage training: (1) Static pre-training on TartanAir (163 scenes) + Kubric static (5K videos), with 7-frame sequences, using camera loss + optical flow loss. (2) Dynamic fine-tuning on Kubric dynamic (11K videos), freezing F and training only $F_m$, using camera loss + motion cross-entropy loss. AdamW optimizer is used, taking about 4 days on 8 A100 GPUs. Trained strictly on synthetic data, with zero-shot generalization to real-world videos.

## Key Experimental Results

### Main Results

**Sintel Dataset Camera Estimation:**

| Method | ATE↓ (Calibrated) | RRE↓ (Calibrated) | ATE↓ (Uncalibrated) | Time |
|------|------------|------------|--------------|------|
| CasualSAM | 0.036 | 0.20 | 0.067 | 1.6min |
| Particle-SfM | 0.062 | 1.26 | 0.057 | 21s |
| MonST3R | - | - | 0.078 | 1.0s |
| **MegaSaM** | **0.018** | **0.04** | **0.023** | **1.0s** |

**DyCheck Dataset Camera Estimation:**

| Method | ATE↓ (Calibrated) | RRE↓ (Calibrated) | ATE↓ (Uncalibrated) | Time |
|------|------------|------------|--------------|------|
| LEAP-VO | 0.167 | 0.09 | - | 0.8s |
| ACE-Zero | 0.062 | 0.11 | 0.056 | 1.6s |
| MonST3R | - | - | 0.690 | 1.0s |
| **MegaSaM** | **0.020** | **0.05** | **0.020** | **0.8s** |

### Ablation Study

| Configuration | Description |
|------|------|
| W/o monocular depth initialization | Camera and scene geometry degrade severely under low-parallax scenes (Fig. 2a) |
| W/o uncertainty-aware BA | Uniformly using/not using depth regularization is sub-optimal (Fig. 2b) |
| Full Configuration | Adaptive regularization yields the best results (Fig. 2c) |
| W/o two-stage training | Direct training on dynamic videos causes training instability |

### Key Findings
- MegaSaM achieves an ATE of only 0.018 (calibrated) on Sintel, representing a 50% reduction compared to the runner-up CasualSAM (0.036), while being 100 times faster (1s vs 1.6min).
- On DyCheck real-world dynamic videos, MegaSaM's ATE = 0.020, which is significantly better than MonST3R (0.690), as MonST3R degrades severely in long video dynamic scenes.
- Uncertainty-aware BA is critical: it automatically deactivates depth constraints when there is a sufficient baseline to avoid introducing noise, and activates depth constraints under low parallax to prevent degradation.
- Trained strictly on synthetic data (TartanAir + Kubric), MegaSaM generalizes zero-shot to real-world dynamic videos, demonstrating strong generalizability.

## Highlights & Insights
- **Uncertainty-driven adaptive regularization**: This is an elegant "autopilot" design—automatically adjusting regularization intensity by analyzing diagnostic information from the Hessian matrix, avoiding manual hyperparameter tuning for different video types.
- **Two-stage decoupled training philosophy**: The strategy of decoupling motion learning from correspondence learning can be generalized to other differentiable optimization systems that need to handle out-of-domain noise.
- **Key engineering insight**: Combining two complementary monocular depth models (DepthAnything for high accuracy + UniDepth for scale guidance) yields much better results than using a single model.

## Limitations & Future Work
- Dependence on external monocular depth models (DepthAnything + UniDepth), which themselves might be unreliable in certain scenes.
- The motion probability map is trained on synthetic data; extremely complex real-world dynamic scenes may exceed its generalization capability.
- The 1.3 FPS execution rate of the consistent depth optimization component is still relatively slow for long videos.
- No clear timeline is provided for code and pre-trained model open-source plans.

## Related Work & Insights
- **vs DROID-SLAM**: MegaSaM is built upon the differentiable BA core of DROID-SLAM, but extends it from static to dynamic scenes through three key modifications (motion probability maps, depth priors, and uncertainty-aware BA).
- **vs MonST3R**: MonST3R follows the DUSt3R global point cloud prediction paradigm, which performs decently on short videos but degrades seriously on long videos. In contrast, MegaSaM's SLAM framework naturally supports incremental processing of long videos.
- **vs CasualSAM**: CasualSAM requires per-video fine-tuning of the monocular depth network, which is highly time-consuming (minute-scale). MegaSaM does not require network fine-tuning and is two orders of magnitude faster.

## Rating
- Novelty: ⭐⭐⭐⭐ The design concepts of uncertainty-aware BA and the two-stage training strategy are clever; although individual components are not entirely net-new, the combination is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive baseline comparisons, covering synthetic/real datasets, calibrated/uncalibrated scenes, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Sufficient technical details with clear mathematical formulations and derivations.
- Value: ⭐⭐⭐⭐⭐ Provides a practical solution with an excellent balance of accuracy, speed, and robustness for SfM on casual dynamic videos.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds](mosca_dynamic_gaussian_fusion_from_casual_videos_via_4d_motion_scaffolds.md)
- [\[CVPR 2025\] MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion](mp-sfm_monocular_surface_priors_for_robust_structure-from-motion.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](../../ICCV2025/3d_vision/longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[CVPR 2025\] Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)
- [\[CVPR 2025\] Dense-SfM: Structure from Motion with Dense Consistent Matching](dense-sfm_structure_from_motion_with_dense_consistent_matching.md)

</div>

<!-- RELATED:END -->
