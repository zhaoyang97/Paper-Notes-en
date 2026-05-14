---
title: >-
  [Paper Note] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection
description: >-
  [Human Understanding] > SeeKer proposes to autoregressively factorize the joint density of skeleton sequences at the keypoint level…
tags:
  - "Human Understanding"
date: 2026-05-08
content_hash: 61c2267d673175ae
---

# Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection

- **Conference**: ICCV 2025
- **arXiv**: 2506.18368
- **Code**: https://github.com/adelic99/seeker
- **Area**: Human Understanding
- **Keywords**: Video Anomaly Detection, Skeleton Sequences, Autoregressive Density Estimation, Keypoint-Level, Gaussian Distribution

## TL;DR

> SeeKer proposes to autoregressively factorize the joint density of skeleton sequences at the keypoint level, detecting abnormal human behaviors by predicting conditional Gaussian distributions over subsequent keypoints. It substantially outperforms existing methods on the UBnormal and MSAD-HR benchmarks.

## Background & Motivation

Video anomaly detection is a safety-critical computer vision task widely applied in medical monitoring, workplace safety, and public surveillance. Skeleton-based methods offer distinct advantages due to their low dimensionality, anonymity, and domain invariance.

However, existing skeleton-based anomaly detection methods (e.g., STG-NF, MoCoDAD) suffer from the following limitations:

**Neglecting the compositional nature of skeletons**: These methods treat skeletons as monolithic graph structures and apply graph neural networks for holistic likelihood estimation, overlooking the fact that skeletons are composed of individual keypoints.

**Lack of causal modeling**: They fail to fully exploit the inherent causal structure of skeleton sequences in the temporal domain.

**Inability to handle keypoint detection uncertainty**: Anomaly scores cannot account for confidence variations in skeleton keypoint detectors, leading to false positives under occlusion or detection errors.

The authors observe that abnormal human behaviors typically manifest as unusual body poses—i.e., keypoints appearing at unlikely positions. This motivates a more natural modeling approach: density estimation at the keypoint level.

## Method

### Overall Architecture

The core idea of SeeKer (Sequential Keypoint Density Estimator) is to predict the conditional multivariate Gaussian distribution over subsequent keypoint positions given preceding keypoints. Applying this approach across the entire skeleton sequence yields an autoregressive density factorization at the keypoint level.

Each skeleton $X_t$ is an $N \times D$ matrix ($N=18$ keypoints, $D=2$ coordinate dimensions). The joint density of a skeleton sequence $\mathbf{X}$ is factorized via two levels of autoregression:

**Level 1 — Temporal autoregression**:

$$p_\theta(\mathbf{X}) = \prod_{t=1}^{T} p_\theta(X_t | \mathbf{X}_\Delta)$$

where $\mathbf{X}_\Delta$ denotes the skeletons from the past $\Delta$ frames.

**Level 2 — Keypoint autoregression**:

$$p_\theta(X_t | \mathbf{X}_\Delta) = \prod_{n=1}^{N} p_\theta(X_{t,n} | X_{t,<n}, \mathbf{X}_\Delta)$$

The conditional distribution of each keypoint $X_{t,n}$ is defined as a multivariate Gaussian:

$$p_\theta(X_{t,n} | X_{t,<n}, \mathbf{X}_\Delta) := \mathcal{N}(X_{t,n} | \boldsymbol{\mu}_\theta, \Sigma_\theta)$$

### Loss & Training

The training objective maximizes sequence likelihood. The simplified loss function is:

$$L(\theta; \mathcal{D}) \cong \sum_{\mathbf{X},t,n} (X_{t,n} - \boldsymbol{\mu}_\theta)^\top \Sigma_\theta^{-1} (X_{t,n} - \boldsymbol{\mu}_\theta) + \ln \det \Sigma_\theta$$

The first term is the Mahalanobis distance; the second is the log-determinant of the covariance matrix, serving as a regularizer to prevent degenerate solutions.

### Composite Anomaly Score

The base anomaly score is the sum of negative log-likelihoods over all keypoints:

$$s'(X_t | \mathbf{X}_\Delta) = -\sum_n \ln p_{\theta_{\text{MLE}}}(X_{t,n} | X_{t,<n}, \mathbf{X}_\Delta)$$

The confidence-weighted anomaly score incorporating keypoint detector confidence:

$$s(X_t | \mathbf{X}_\Delta) = -\sum_n c_{t,n} \ln p_{\theta_{\text{MLE}}}(X_{t,n} | X_{t,<n}, \mathbf{X}_\Delta)$$

where $c_{t,n} \in [0,1]$ is the keypoint detection confidence, integrating detection uncertainty into the final decision. In multi-person scenarios, the maximum anomaly score within a frame is used.

### Autoregressive Architecture

A causally masked fully-connected network is adopted, with causal constraints enforced via fixed block-triangular masks, allowing coordinates within the same keypoint to be mutually dependent. Experiments confirm that this simple architecture outperforms Transformers and RNNs.

## Key Experimental Results

### Main Results

| Dataset | Method | AUROC | AP |
|--------|------|-------|-----|
| UBnormal (Full) | STG-NF | 71.8 | 62.7 |
| | MoCoDAD | 68.3 | - |
| | MULDE | 72.8 | - |
| | **SeeKer** | **77.9** | **80.3** |
| ShanghaiTech (Full) | STG-NF | 85.9 | 77.6 |
| | **SeeKer** | **85.5** | **80.0** |
| MSAD-HR | STG-NF | 55.7 | 56.5 |
| | **SeeKer** | **61.1** | **60.1** |

On UBnormal, SeeKer outperforms STG-NF by **6.1pp** in AUROC and **17.6pp** in AP; on MSAD-HR, the AUROC gain is **5.4pp**.

### Ablation Study

| Covariance Type | UBnormal (Full) | UBnormal (HR) | ShanghaiTech (Full) | ShanghaiTech (HR) |
|-----------|----------------|---------------|--------------------|--------------------|
| Fixed (Identity) | 61.4 | 63.4 | 74.1 | 74.4 |
| Diagonal Learnable | 77.1 | 78.1 | 85.3 | 86.2 |
| Full Learnable | **77.9** | **78.9** | **85.5** | **86.9** |

Learned covariance improves over fixed covariance by more than **10pp**, underscoring the importance of covariance learning.

### Key Findings

1. **Conceptually simple yet highly effective**: The core idea of SeeKer—keypoint-level autoregressive Gaussian density estimation—is straightforward yet achieves state-of-the-art performance across multiple benchmarks.
2. **Strong interpretability**: Anomaly scores can be decomposed into per-keypoint contributions, enabling localization of the specific joints triggering anomaly detection.
3. **Effective confidence weighting**: Weighting by keypoint detector confidence yields greater robustness under occlusion and detection errors.
4. **Keypoint ordering invariance**: Experiments confirm that different keypoint orderings do not affect model expressiveness.

## Highlights & Insights

- Refining density estimation from "holistic skeleton" to "per-keypoint" is the key innovation, yielding both interpretability and finer-grained anomaly detection.
- The confidence-weighted anomaly score elegantly incorporates uncertainty from all components in the pipeline.
- The masked fully-connected model outperforms more complex Transformers, highlighting the importance of task-appropriate architecture design.
- Operating in the original 2D space makes prediction visualization straightforward and intuitive.

## Limitations & Future Work

- Performance depends on the quality of the external skeleton detector (AlphaPose) and tracker, limiting effectiveness in crowded scenes (e.g., MSAD).
- Only 2D skeleton information is used; depth-directional anomalous motions cannot be modeled.
- The conditional Gaussian assumption may be insufficiently flexible to represent complex multimodal keypoint distributions.
- The temporal window $\Delta$ is fixed and cannot dynamically adapt to actions of varying speeds.

## Related Work & Insights

- **Skeleton-based anomaly detection**: STG-NF (graph normalizing flows), MoCoDAD (diffusion model prediction), MULDE (energy model + denoising score matching).
- **Video anomaly detection**: Methods based on deep features, optical flow, and multimodal fusion.
- **Autoregressive density estimation**: Transfer of sequence modeling ideas from NLP/speech to skeleton sequences.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐⭐ |
| Clarity | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐ |
| Overall | 8.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/human_understanding/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)

</div>

<!-- RELATED:END -->
