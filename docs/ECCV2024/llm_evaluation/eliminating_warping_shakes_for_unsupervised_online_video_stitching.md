---
title: >-
  [Paper Note] Eliminating Warping Shakes for Unsupervised Online Video Stitching
description: >-
  [ECCV 2024][LLM Evaluation][Video stitching] Defines a new problem in video stitching termed "warping shake" (temporal shaking in non-overlapping regions when extending image stitching to video). This work proposes StabStitch, the first unsupervised online video stitching framework. By simultaneously generating and smoothing stitching trajectories, it achieves both video stitching and stabilization, reaching a real-time speed of 28.2 ms/frame.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "Video stitching"
  - "video stabilization"
  - "warping shake"
  - "unsupervised learning"
  - "online processing"
date: 2026-05-08
content_hash: bd0e89adbc995bc8
---

# Eliminating Warping Shakes for Unsupervised Online Video Stitching

**Conference**: ECCV 2024  
**arXiv**: [2403.06378](https://arxiv.org/abs/2403.06378)  
**Code**: Yes ([https://github.com/nie-lang/StabStitch](https://github.com/nie-lang/StabStitch))  
**Area**: LLM Evaluation  
**Keywords**: Video stitching, video stabilization, warping shake, unsupervised learning, online processing

## TL;DR

Defines a new problem in video stitching termed "warping shake" (temporal shaking in non-overlapping regions when extending image stitching to video). This work proposes StabStitch, the first unsupervised online video stitching framework. By simultaneously generating and smoothing stitching trajectories, it achieves both video stitching and stabilization, reaching a real-time speed of 28.2 ms/frame.

## Background & Motivation

Video stitching technology is used to combine videos with limited fields of view from different perspectives into panoramic or wide-view scenes, and is widely applied in autonomous driving, video surveillance, and virtual reality. This paper focuses on the most common and challenging scenario of hand-held camera video stitching.

**Core Problem – Warping Shake**:

The authors identify an ignored yet critical problem: when directly applying image stitching algorithms to video frame-by-frame, although the stitched result of each individual frame appears natural and reasonable, obvious content shaking occurs between temporally adjacent frames in **non-overlapping regions**. The key points are:

**Shaking does not originate from the source videos**: Thanks to modern video stabilization technologies (both hardware and software), source videos captured by handheld cameras are typically stable.

**Shaking originates from stitching warps**: Image stitching methods (e.g., LPC, UDIS++) estimate the spatial warps of each frame independently. Although they maintain good structures in each frame, the frame-to-frame warps are discontinuous in the temporal domain, leading to content shaking in the warped non-overlapping regions.

**Definition**: warping shake = temporal instability in non-overlapping regions caused by temporally unsmooth spatial warps, regardless of whether the source videos are stable.

**Limitations of Prior Work**:

- **Outdated Assumption**: Prior methods like Nie et al. assume that handheld camera videos inevitably suffer from severe shaking and must stabilize each source video individually, which is inconsistent with the reality where modern portable devices widely feature built-in stabilization.
- **Complex Optimization**: Jointly optimizing stitching and stabilization requires constructing complex non-linear solving systems and multi-round iterative optimizations for different parameters, resulting in extremely slow inference speeds.
- **Poor Robustness**: The complex optimization process is highly demanding on input video quality (requiring sufficiently many, accurate, and uniformly distributed matching points), making the system fragile in practical applications.
- **Offline Processing**: All existing methods require the full video as input and cannot perform real-time online processing.

**Core Idea of StabStitch**: Since the source videos are inherently stable, the problem is simplified to **eliminating the shaking introduced by stitching warps**, rather than the traditional approach of stabilizing both source videos and stitching simultaneously. By drawing inspiration from the concept of camera trajectories in video stabilization, the "stitching trajectory" in video stitching is derived and smoothed to achieve joint stitching and stabilization.

## Method

### Overall Architecture

StabStitch consists of three core components:

1. Temporal Warp Model: Estimates motions between adjacent frames of the same video.
2. Spatial Warp Model: Estimates spatial alignment across different viewpoints.
3. Warp Smoothing Model: Smoothes the stitching trajectory to eliminate warping shakes.

### Key Designs

1. **Temporal Warp Model (TNet)**: Uses a CNN to estimate the motion of grid control points $m^T(t)$ between adjacent frames.

    - Simplified from the UDIS++ architecture, using ResNet18 instead of ResNet50, and replacing the global correlation layer with a local correlation layer (cost volume) since motion between adjacent frames is typically small.
    - Optimization objective: $\mathcal{L}^{tmp} = \mathcal{L}_{alignment} + \lambda^{tmp} \mathcal{L}_{distortion}$
    - **Design Motivation**: Using a CNN instead of traditional feature point methods to estimate motion makes it more robust in challenging scenarios like low light and low texture.

2. **Stitch-Meshflow (Stitching Trajectory Derivation)**: Innovatively combines spatial and temporal warps to derive stitching trajectories.

    - Camera trajectory (in video stabilization) is defined as the accumulation of temporal motion: $C_i(t) = m_i(1) + m_i(2) + \cdots + m_i(t)$
    - The spatial warp model (SNet) estimates the grid motion $m^S$ between reference and target frames, with an additional motion consistency constraint:
    $$\mathcal{L}_{consis.} = \frac{1}{(U+1)(V+1)} \sum_{i} \|m_i(t) - m_i(t-1) - \mu^{spt}\|_2$$
    - **Stitching Motion Derivation**: Maps temporal motion to the warped space via Thin Plate Spline (TPS) transformation to obtain the stitching motion:
    $$s(t) = \mathcal{T}_{M^{Rig} \to M^S(t-1)}(M^T(t)) - M^S(t)$$
    - The stitching trajectory is the accumulation of stitching motion: $S_i(t) = s_i(1) + s_i(2) + \cdots + s_i(t)$
    - **Design Motivation**: To align frame $t$ and frame $t-1$ in the warped video, one cannot directly use the temporal motion of the source video. It is necessary to consider the coordinate transformation of frame $t-1$ after spatial warping. By "projecting" temporal motion into the warped space via TPS and subtracting it from the actual spatial warp of the current frame, the residual motion between each frame in the warped video is obtained.

3. **Warp Smoothing Model (SmoothNet)**: A 3D convolutional network that takes a sequence of stitching trajectories, spatial grid sequences, and overlap masks as input, and outputs the smoothing increment $\Delta$.

    - Smoothed trajectory: $\hat{S} = S + \Delta$
    - Smoothed spatial grid: $\hat{M}^S = M^S - \Delta$
    - **Comprehensive Design of Four Optimization Objectives**:
        - **Data Term** $\mathcal{L}_{data} = \|(\hat{S}-S)(\alpha \cdot OP + 1)\|_2$: Smoothed trajectories should not deviate too far from the original trajectories, with larger weights applied to the overlapping region (multiplied by $\alpha \cdot OP + 1$) to preserve alignment quality.
        - **Smoothness Term** $\mathcal{L}_{smoothness}$: Constrains the trajectory position at any time step to be at the midpoint of the preceding and succeeding steps, implicitly regularizing for no sudden rotation and consistent translation magnitudes.
        - **Spatial Consistency Term** $\mathcal{L}_{space} = \frac{1}{N} \sum_t \mathcal{L}_{distortion}(\hat{M}^S(t))$: Ensures that the changes of all $(U+1) \times (V+1)$ trajectories remain consistent, preventing spatial distortions caused by independent optimization.
        - **Online Collaborative Term** $\mathcal{L}_{online} = \frac{1}{N-1} \sum_t \|\hat{S}^{(\xi)}(t) - \hat{S}^{(\xi+1)}(t-1)\|_2$: Ensures consistent smoothed trajectories of overlapping frames in adjacent sliding windows.

### Loss & Training

The three models are trained independently:

**Temporal Warp**: $\mathcal{L}^{tmp} = \mathcal{L}_{alignment} + \lambda^{tmp} \mathcal{L}_{distortion}$ ($\lambda^{tmp}=5$)

**Spatial Warp**: $\mathcal{L}^{spt} = \mathcal{L}_{alignment} + \lambda^{spt} \mathcal{L}_{distortion} + \omega^{spt} \mathcal{L}_{consis.}$ ($\lambda^{spt}=10, \omega^{spt}=0.1$)

**Warp Smoothing**: $\mathcal{L}^{smooth} = \mathcal{L}_{data} + \lambda^{smooth} \mathcal{L}_{smoothness} + \omega^{smooth} \mathcal{L}_{space}$ ($\lambda^{smooth}=50, \omega^{smooth}=10$), with the additional term $\mathcal{L}_{online}$ in the online mode.

Control point resolution is $(6+1) \times (8+1)$, and sliding window length is 7 frames.

## Key Experimental Results

### Main Results

Comparison with image stitching methods on the StabStitch-D dataset (PSNR/SSIM):

| Method | Regular | Low-Light | Low-Texture | Fast-Moving | Average |
|------|---------|-----------|-------------|-------------|---------|
| LPC | 24.22/0.812 | Failed | Failed | 23.88/0.813 | - |
| UDIS++ | 23.19/0.785 | 31.09/0.936 | 29.98/0.906 | 21.56/0.756 | 27.19/0.859 |
| UDIS++* (retrained) | 24.63/0.829 | 34.26/0.957 | 32.81/0.920 | **24.78/0.819** | 29.78/0.891 |
| **StabStitch** | **24.64/0.832** | **34.51/0.958** | **33.63/0.927** | 23.36/0.787 | **29.89/0.890** |

User preference study compared with video stitching methods (excluding failed cases of Nie et al.):

| Prefers StabStitch | Prefers Nie et al. | No Preference |
|--------------|--------------|--------|
| **30.47%** | 6.25% | 63.28% |

### Ablation Study

| Configuration | Alignment↑ | Distortion↓ | Stability↓ |
|------|-----------|-------------|------------|
| Basic Stitching | 30.67/0.902 | 0.784 | 81.57 |
| + $\mathcal{L}_{consis.}$ | **30.75/0.903** | 0.804 | 60.32 |
| + $\mathcal{L}_{consis.}$ + Warp Smoothing | 29.89/0.890 | **0.674** | **48.74** |

Inference speed analysis (RTX 4090Ti):

| SNet | TNet | Trajectory Gen. | SmoothNet | Warping | Blending | Total |
|------|------|---------|-----------|---------|----------|------|
| 11.5ms | 10ms | 1.1ms | 1ms | 4.4ms | 0.2ms | **28.2ms** |

### Key Findings

- **Warping shake indeed exists and is severe**: Even with stable input videos, the stability score of baseline image stitching is as high as 81.57, which StabStitch reduces to 48.74.
- **Motion consistency constraint is effective**: $\mathcal{L}_{consis.}$ reduces the stability score from 81.57 to 60.32, while slightly improving alignment quality.
- **Sacrificing slight alignment for substantial stability**: The full version of StabStitch only experiences a 2.8% drop in alignment performance (30.75 -> 29.89), but improves stabilization by 40.2%.
- **Extremely robust across scenarios**: Nie et al. failed on 10 out of 20 test video pairs (in low-light and low-texture scenarios), whereas StabStitch succeeded on all of them.
- **Real-time performance**: 28.2 ms/frame vs. over 40 minutes for a 26-second video in Nie et al., yielding a speedup of tens of thousands of times.

## Highlights & Insights

- Precisely defines the new problem of "warping shake"—where seemingly stable inputs become unstable after stitching. This phenomenon is highly common in engineering practice but has not been systematically studied prior to this work.
- The derivation of Stitch-Meshflow is a key technical contribution: it cleverly integrates spatial and temporal warps into a stitching trajectory, allowing mature video stabilization techniques to be directly applied to video stitching.
- The 3D convolutional smoothing network design is simple yet highly efficient (only 3 layers), highlighting the effectiveness of the unsupervised learning scheme.
- The online mode requires only a 1-frame latency, achieved through sliding windows and online collaborative constraints.

## Limitations & Future Work

- Assumes stable inputs as a prerequisite: if the source videos themselves have large-scale shaking (e.g., extreme motion), they must be stabilized individually first.
- The alignment performance in fast-moving scenarios is inferior to the retrained version of UDIS++ (23.36 vs. 24.78), as smoothing inevitably sacrifices some alignment quality.
- The dataset size is relatively limited (100+ video pairs); a larger dataset could potentially yield further improvements.
- Only supports stitching two videos, and has not been extended to multi-video panoramic stitching.

## Related Work & Insights

- The trajectory representation of MeshFlow is innovatively extended to the stitching scenario; the generalization paradigm from camera path to stitching path is highly inspirational.
- The spatial warp network of UDIS++ is efficiently modified (ResNet50 -> ResNet18, global correlation -> local correlation), balancing both performance and efficiency.
- Resolves the traditional dilemma of choosing between "stitch then stabilize" vs. "stabilize then stitch" via a unified single-step framework.
- The constructed StabStitch-D dataset can serve as a standard evaluation benchmark in the field of video stitching.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to define the warping shake problem; novel derivation of Stitch-Meshflow.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive with multi-scenario evaluations, user studies, speed analyses, and complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem definition, rigorous formula derivations, and highly illustrative figures.
- **Value**: ⭐⭐⭐⭐⭐ High practical value—creates a real-time online video stitching system, offering open-source code and datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SIGMA: Sinkhorn-Guided Masked Video Modeling](sigma_sinkhorn-guided_masked_video_modeling.md)
- [\[ECCV 2024\] Deep Cost Ray Fusion for Sparse Depth Video Completion](deep_cost_ray_fusion_for_sparse_depth_video_completion.md)
- [\[ECCV 2024\] Distribution Alignment for Fully Test-Time Adaptation with Dynamic Online Data Streams](distribution_alignment_for_fully_test-time_adaptation_with_dynamic_online_data_s.md)
- [\[ECCV 2024\] ColorMNet: A Memory-based Deep Spatial-Temporal Feature Propagation Network for Video Colorization](colormnet_a_memory-based_deep_spatial-temporal_feature_propagation_network_for_v.md)
- [\[NeurIPS 2025\] Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator](../../NeurIPS2025/llm_evaluation/your_pre-trained_llm_is_secretly_an_unsupervised_confidence_calibrator.md)

</div>

<!-- RELATED:END -->
