---
title: >-
  [Paper Note] Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning
description: >-
  [CVPR2026][LLM Evaluation][visual geometry] This paper proposes a **Factored Flow Prediction** module that predicts optical flow from the geometric latent of a source view and the pose latent of a target view, enabling unlabeled videos to serve as supervisory signals for 3D geometry learning. The method achieves state-of-the-art performance across 8 benchmarks covering both static and dynamic scenes.
tags:
  - CVPR2026
  - LLM Evaluation
  - visual geometry
  - factored flow
  - 3D reconstruction
  - unlabeled video
  - correspondence learning
  - dynamic scenes
date: 2026-05-08
content_hash: 0ae3f16fb4a2c89b
---

# Flow3r: Factored Flow Prediction for Scalable Visual Geometry Learning

**Conference**: CVPR2026  
**arXiv**: [2602.20157](https://arxiv.org/abs/2602.20157)  
**Code**: [flow3r-project.github.io](https://flow3r-project.github.io/)  
**Area**: LLM Evaluation  
**Keywords**: visual geometry, factored flow, 3D reconstruction, unlabeled video, correspondence learning, dynamic scenes

## TL;DR

This paper proposes a **Factored Flow Prediction** module that predicts optical flow from the geometric latent of a source view and the pose latent of a target view, enabling unlabeled videos to serve as supervisory signals for 3D geometry learning. The method achieves state-of-the-art performance across 8 benchmarks covering both static and dynamic scenes.

## Background & Motivation

1. **Feed-forward 3D reconstruction relies on expensive annotations**: Methods such as DUSt3R, VGGT, and π³ require dense depth and camera pose supervision, which is prohibitively costly to obtain, especially for in-the-wild dynamic scenes.
2. **Annotated data cannot scale**: Unlike LLMs or ViTs, which can be trained on massive unlabeled data via self-supervised objectives, 3D geometry learning is constrained by annotation scale and cannot benefit from scaling in the same way as language or vision.
3. **Existing optical flow supervision (VGGT tracking head) is insufficiently effective**: VGGT predicts optical flow via patch-feature matching, which encourages visually discriminative features but does not directly promote learning of pose or geometry.
4. **Projective flow is unstable and fails on dynamic scenes**: Computing flow by projecting predicted pointmaps with estimated camera parameters is highly sensitive to geometric errors and cannot model scene motion.
5. **Unlabeled video is a vast untapped resource**: The internet contains massive quantities of unannotated monocular video; leveraging 2D correspondences therein as supervision could dramatically expand training data.
6. **2D dense correspondence models are mature**: Models such as UFM, RoMa, and CoTracker can provide high-quality pseudo-label optical flow for arbitrary image pairs, forming the foundation for exploiting unlabeled video.

## Method

### Overall Architecture

Flow3r augments a standard multi-view Transformer (e.g., VGGT/π³) with a **factored flow prediction head**. During training, annotated 3D datasets (~34K sequences) and unlabeled video data (~800K sequences) are mixed to jointly optimize the model.

### Factored Flow Prediction

The core observation is that for static scenes, the optical flow from a source view to a target view depends solely on the **scene geometry of the source view** and the **camera pose of the target view**. This motivates an asymmetric flow prediction formulation:

$$\hat{\mathbf{F}}_{i \rightarrow j} = \Phi_{\text{flow}}(\mathbf{g}_i, \mathbf{c}_j)$$

- $\mathbf{g}_i$: per-patch geometric features output by the multi-view Transformer for the source view
- $\mathbf{c}_j$: camera token of the target view (global pose feature)
- $\mathbf{c}_j$ modulates $\mathbf{g}_i$, which is then decoded into dense optical flow via a DPT head

**Key advantages**:
- Gradients from the flow supervision flow directly into both the geometry branch and the pose branch, promoting learning in both
- Operating in latent space avoids dependence on explicit geometric decoding required by projective approaches, yielding greater robustness
- Naturally extends to dynamic scenes, as the flow implicitly encodes both camera motion and scene motion

### Comparison with Alternative Designs

| Design | Principle | Drawback |
|--------|-----------|----------|
| flow-tracking (VGGT) | Predicts flow via patch-feature matching between two views | Enhances visual discriminability only; does not promote geometry/pose learning |
| flow-projective | Projects predicted pointmaps with camera parameters | Sensitive to errors; limited to static scenes |
| **flow-factored (Ours)** | Decodes from geometric latent + pose latent | Information bottleneck limits standalone flow accuracy, but yields the best geometric supervision |

### Loss & Training

- **Annotated data**: camera pose loss $\mathcal{L}_{\text{cam}}$ + geometry loss $\mathcal{L}_{\text{geo}}$ (including optimally aligned pointmap loss)
- **Flow loss** (applicable to both annotated and unlabeled data): robust Charbonnier regression loss, weighted by a co-visibility mask:

$$\mathcal{L}_{\text{flow}} = \frac{1}{\sum_p \mathbf{C}[p]} \sum_p \mathbf{C}[p] \cdot \ell_{\text{robust}}(\|\hat{\mathbf{u}}_{i\to j}[p] - \mathbf{u}_{i\to j}[p]\|_2)$$

- Pseudo-label optical flow for unlabeled data is generated by the pretrained UFM model

Two-stage training:
1. **Freeze the backbone** and train only the newly added flow prediction head (on annotated data)
2. **Unfreeze the full model** and fine-tune end-to-end on annotated + unlabeled data

## Key Experimental Results

### Main Results — Dynamic Scenes (Tab. 2)

| Method | Kinetics RPE-t↓ | EPIC RPE-t↓ | Sintel MSE↓ | Bonn f-score↑ |
|--------|:---:|:---:|:---:|:---:|
| DUSt3R | 0.063 | 0.110 | 0.622 | 0.800 |
| CUT3R | 0.027 | 0.081 | 0.676 | 0.899 |
| VGGT | 0.038 | 0.049 | 0.595 | 0.884 |
| π³ | 0.023 | 0.043 | 0.523 | 0.905 |
| **Flow3r** | **0.018** | **0.037** | **0.426** | **0.954** |

Flow3r achieves the best results on all metrics across all four dynamic benchmarks, with substantial improvements in both pose and geometry.

### Main Results — Static Scenes (Tab. 3)

| Method | 7-Scenes RTA↑ | 7-Scenes MSE↓ | NRGBD f-score↑ | ScanNet RTA↑ |
|--------|:---:|:---:|:---:|:---:|
| π³ | 87.69 | 0.169 | 0.983 | 91.14 |
| **Flow3r** | **91.66** | **0.102** | **0.992** | 92.89 |

Gains from dynamic data also transfer to static scenes; MSE on 7-Scenes drops from 0.169 to 0.102 (↓40%).

### Ablation Study — Scaling Unlabeled Data (Tab. 4)

| Annotated | Unlabeled | RRA@30↑ | MSE↓ |
|:---------:|:---------:|:---:|:---:|
| 11K | 0 | 66.01 | 0.637 |
| 11K | 3K | 76.26 | 0.598 |
| 11K | 10K | 78.45 | 0.560 |
| 11K | 20K | 81.12 | 0.532 |
| 44K (annotated only) | 0 | 78.68 | 0.565 |

11K annotated + 20K unlabeled outperforms 44K annotated-only, demonstrating that unlabeled video supervised via optical flow can substitute for expensive 3D annotations.

### Ablation Study — Flow Prediction Mechanism Comparison (Tab. 1)

- flow-tracking (VGGT-style) yields almost no improvement in geometric quality
- flow-projective even degrades performance
- flow-factored consistently outperforms the baseline and alternative designs on both static and dynamic scenes

## Highlights & Insights

- **Factored flow prediction** is an elegant and effective design: the information bottleneck forces the geometric latent to capture true 3D structure and the pose latent to capture true camera motion
- The method exhibits strong **generality**: it can be plugged into both VGGT and π³ architectures and brings consistent gains in both cases
- **Scaling behavior is clear and monotonic**: performance improves steadily with the amount of unlabeled data, demonstrating the scalability of the approach
- Achieves comprehensive state-of-the-art results across 8 benchmarks spanning static and dynamic scenes, with the largest gains on in-the-wild dynamic video where annotations are scarce

## Limitations & Future Work

- The method depends on pretrained models such as UFM for optical flow pseudo-labels; if the 2D correspondence model fails in certain domains, the approach is correspondingly limited
- Robustness may degrade on complex dynamic scenes containing multiple independently moving objects
- Current experiments are conducted at ~800K sequences; the effect of scaling to 10M–100M sequences has not been verified
- Factored flow prediction achieves lower standalone flow accuracy than direct patch matching (the cost of the information bottleneck) and cannot be used as an independent optical flow estimator

## Related Work & Insights

| Method | Supervision Type | Dynamic Support | Pose Annotation Required | Core Idea |
|--------|-----------------|:---------------:|:------------------------:|-----------|
| DUSt3R | Fully supervised | ✗ | ✓ | Two-view pointmap regression |
| VGGT | Fully supervised + tracking head | ✓ | ✓ | Multi-view Transformer + patch-matching flow |
| π³ | Fully supervised | ✓ | ✓ | Local coordinate prediction + permutation equivariance |
| CUT3R | Fully supervised | ✓ | ✓ | Streaming multi-view inference |
| MegaSAM | Optimization-based | ✓ | ✗ | Monocular depth prior + per-video optimization |
| **Flow3r** | **Semi-supervised** | **✓** | **Partial** | **Factored flow + unlabeled video scaling** |

## Rating

- Novelty: ⭐⭐⭐⭐ — The insight behind factored flow prediction is concise and profound; the asymmetric design is theoretically well-motivated
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 8 benchmarks, 3 ablated designs, scaling curves, multi-backbone validation, and qualitative comparisons
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, well-articulated motivation
- Value: ⭐⭐⭐⭐⭐ — Identifies a viable path for scaling 3D geometry learning, with significant implications for the field

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](reframing_long-tailed_learning_via_loss_landscape_geometry.md)
- [\[CVPR 2026\] Free-Grained Hierarchical Visual Recognition](free-grained_hierarchical_visual_recognition.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)
- [\[CVPR 2026\] Semi-Supervised Conformal Prediction With Unlabeled Nonconformity Score](semi-supervised_conformal_prediction_with_unlabeled_nonconformity_score.md)
- [\[AAAI 2026\] BCWildfire: A Long-term Multi-factor Dataset and Deep Learning Benchmark for Boreal Wildfire Risk Prediction](../../AAAI2026/llm_evaluation/bcwildfire_a_long-term_multi-factor_dataset_and_deep_learning_benchmark_for_bore.md)

<!-- RELATED:END -->
