---
title: >-
  [Paper Note] Balancing Multimodal Domain Generalization via Gradient Modulation and Projection
description: >-
  [AAAI2026][Video Understanding][Multimodal Domain Generalization] This paper proposes a Gradient Modulation Projection (GMP) strategy that addresses inter-modality optimization imbalance and inter-task gradient conflicts…
tags:
  - "AAAI2026"
  - "Video Understanding"
  - "Multimodal Domain Generalization"
  - "Gradient Modulation"
  - "Gradient Projection"
  - "Optimization Imbalance"
date: 2026-05-08
content_hash: 436f19c6d297ead6
---

# Balancing Multimodal Domain Generalization via Gradient Modulation and Projection

**Conference**: AAAI2026
**arXiv**: [2603.14175](https://arxiv.org/abs/2603.14175)
**Code**: To be confirmed
**Area**: Video Understanding
**Keywords**: Multimodal Domain Generalization, Gradient Modulation, Gradient Projection, Optimization Imbalance

## TL;DR

This paper proposes a Gradient Modulation Projection (GMP) strategy that addresses inter-modality optimization imbalance and inter-task gradient conflicts in multimodal domain generalization (MMDG) through two components: Inter-modality Gradient Decoupled Modulation (IGDM) and Conflict-Adaptive Gradient Projection (CAGP), achieving state-of-the-art performance on multiple benchmarks.

## Background & Motivation

Multimodal domain generalization (MMDG) aims to leverage the complementary advantages of multiple modalities such as video and audio, enabling models to generalize to unseen domains at test time. In real-world applications—such as cross-environment action recognition and audio-visual event detection—test data frequently originates from different devices and environments, making effective domain generalization critical.

However, **optimization imbalance** is pervasive in multimodal learning: different modalities converge at different rates during training, leading to uneven gradient contributions—certain modalities dominate the learning process while others are suppressed. Experiments (Table 1) demonstrate that individual modality branches in joint training perform significantly worse than their independently trained counterparts, indicating that existing MMDG training strategies fail to fully exploit each modality's capacity.

More critically, existing balancing strategies (e.g., OGM-GE, Grad-Blending) regulate each modality's gradient contribution solely based on **source-domain classification performance**. This overlooks a key insight: a modality that exhibits strong classification ability on the source domain does not necessarily learn well-generalizable domain-invariant features, and may therefore generalize poorly to the target domain. Table 1 clearly illustrates this point—conventional methods yield notable improvements on the source domain but marginal gains on the target domain.

## Core Problem

This paper identifies and addresses two categories of imbalance in MMDG:

1. **Inter-Modality Imbalance**: Persistent disparities in gradient magnitudes across modalities cause stronger modalities to dominate optimization while weaker modalities remain under-optimized. Conventional methods balance solely based on classification gradient magnitudes, ignoring domain-invariance objectives, and may suppress modalities that are critical for cross-domain generalization.

2. **Inter-Task Conflicts**: The classification loss gradient $g_c^m$ and the domain adversarial loss gradient $g_d^m$ frequently point in opposing directions (negative cosine similarity), giving rise to gradient conflicts. The severity of such conflicts varies across modalities (e.g., severe in the video modality, mild in the audio modality), and a uniform conflict resolution strategy cannot accommodate these modality-specific differences.

## Method

### Overall Architecture: GMP

GMP comprises two core components—IGDM and CAGP—addressing inter-modality imbalance and inter-task conflicts, respectively.

### Component 1: Inter-Modality Gradient Decoupled Modulation (IGDM)

The core mechanism of IGDM is **decoupled modulation**—separately modulating the classification gradient and the domain-invariance gradient rather than applying a unified scaling factor. The procedure is as follows:

**Step 1: Compute Dual Confidence Metrics**

- **Semantic Confidence** $q_i^m$: measures the classification certainty of modality $m$ for sample $i$, taken as the softmax probability assigned to the ground-truth class by the classifier.
- **Domain Confidence** $c_i^m$: measures the domain discrimination certainty of modality $m$, taken as the probability assigned to the ground-truth domain label by the domain discriminator. A lower domain confidence indicates that the modality has learned more domain-invariant features.

**Step 2: Compute Disparity Ratios**

Over each mini-batch, two ratios between modalities are computed:

- $\rho_t^m$: semantic confidence ratio; $\rho_t^m > 1$ indicates that modality $m$ is stronger in classification.
- $\sigma_t^m$: domain confidence ratio; $\sigma_t^m > 1$ indicates that modality $m$ is stronger in domain invariance.

**Step 3: Decoupled Modulation Coefficients**

- Classification gradient modulation coefficient $k_t^m = 1 - \tanh(\alpha_k \cdot \rho_t^m)$ (when $\rho_t^m > 1$), used to suppress the classification gradient of the modality that dominates classification.
- Domain gradient modulation coefficient $p_t^m = 1 - \tanh(\alpha_p \cdot \sigma_t^m)$ (when $\sigma_t^m > 1$), used to suppress the domain gradient of the modality that dominates domain invariance.

In this way, classification gradients and domain gradients are modulated **independently** rather than scaled by a single unified coefficient, enabling finer-grained control.

### Component 2: Conflict-Adaptive Gradient Projection (CAGP)

CAGP handles inter-task conflicts on the modulated gradients. Its three key design principles are:

1. **Conflict Awareness**: Projection is triggered only when $\hat{g}_c^m \cdot \hat{g}_d^m < 0$; otherwise, the original gradients are retained.
2. **Modality Specificity**: Conflict detection and projection are performed independently for each modality.
3. **Weak-Task Protection**: A relative task strength ratio $\Gamma_t^m = \rho_t^m / \sigma_t^m$ is used to determine which task is dominant. When $\Gamma_t^m > 1$ (classification is dominant), the classification gradient is projected onto the orthogonal complement of the domain gradient; when $\Gamma_t^m < 1$ (domain invariance is dominant), the domain gradient is projected onto the orthogonal complement of the classification gradient. The weaker task's gradient direction is always preserved in full; only the conflicting component of the stronger task's gradient is removed.

## Key Experimental Results

### Benchmark Datasets

- **EPIC-Kitchens**: Kitchen action recognition dataset with video and audio modalities.
- **HAC**: Audio-visual dataset with video and audio modalities.

### Comparison with Existing Gradient Strategies (Table 2)

| Method | EPIC-Kitchens | HAC |
|------|:---:|:---:|
| Base (concatenation fusion) | 55.06 | 61.86 |
| OGM-GE | 55.71 | 62.83 |
| Grad-Blending | 55.49 | 62.66 |
| **GMP (Ours)** | **57.36** | **64.91** |

GMP outperforms the best baseline by +1.65% on EPIC-Kitchens and +2.08% on HAC.

### Integration with MMDG Methods (Table 3)

When integrated as a plug-and-play module into RNA-Net, MOOSA, SimMMDG, and CMRF, GMP yields consistent improvements. SimMMDG+GMP achieves 62.03% and 69.11% on the two datasets, respectively.

### Single-Modality Generalization Improvement

The video branch improves from 48.86% to 52.33% (+3.47%), and the audio branch from 34.15% to 35.88% (+1.73%). Conventional joint training causes the video branch to underperform its independently trained counterpart by 6.12%; GMP reduces this gap to 2.65%.

### Ablation Study (Table 4)

- IGDM alone: EPIC 55.98%, HAC 63.05%
- CAGP alone: EPIC 55.34%, HAC 63.41%
- Full combination: EPIC 57.36%, HAC 64.91%, confirming the complementarity of the two components
- Replacing decoupled modulation with unified modulation: performance drops to 54.97%/62.50%
- Removing either confidence metric ($k_t^m$ or $p_t^m$): notable performance degradation in both cases
- Fixed projection direction or PCGrad: both inferior to the adaptive CAGP

## Highlights & Insights

- **First work to analyze MMDG from an optimization perspective**, identifying the fundamental reason why conventional balancing strategies fail in MMDG settings (focusing solely on classification while neglecting generalization).
- **Elegant decoupled modulation design**: separately modulating the two types of gradients using semantic confidence and domain confidence, providing finer granularity than unified modulation.
- **Weak-task-protective gradient projection** is intuitively sound—when two objectives conflict, prioritizing the learning progress of the weaker objective is a principled strategy.
- **Plug-and-play generality**: GMP integrates seamlessly into various existing MMDG methods and consistently yields improvements.
- Ablation experiments are comprehensive, theoretical analysis is clear, and t-SNE visualizations intuitively demonstrate the effectiveness of the approach.

## Limitations & Future Work

- Validation is limited to two modalities (video and audio); generalization to additional modalities (e.g., text, IMU, depth maps) remains unverified.
- Hyperparameters $\alpha_k$ and $\alpha_p$ require tuning over $[0, 1]$; sensitivity across different datasets warrants further investigation.
- The quality of the domain discriminator directly affects the reliability of domain confidence; instability in discriminator training may degrade IGDM's effectiveness.
- The datasets used are relatively small-scale (EPIC-Kitchens, HAC); performance on large-scale datasets has yet to be validated.
- The gradient projection operations introduce additional computational overhead, though the paper does not discuss efficiency implications.

## Related Work & Insights

| Dimension | Conventional Methods (OGM-GE, etc.) | GMP (Ours) |
|:--|:--|:--|
| Balancing Criterion | Based solely on classification performance | Considers both classification and domain invariance |
| Gradient Modulation | Unified modulation | Decoupled modulation (classification and domain gradients modulated separately) |
| Conflict Resolution | None / PCGrad uniform handling | Adaptive projection + weak-task protection |
| Target Domain Performance | Limited gains (+0.43%~+0.65%) | Substantial gains (+2.30%) |
| Generality | Standalone method | Plug-in module compatible with existing MMDG methods |

GMP is orthogonal and complementary to architecture- and representation-focused MMDG methods such as SimMMDG and MOOSA, providing gains at the optimization level.

The application of **multi-objective optimization perspectives to multimodal learning** is a promising direction; the decoupled modulation framework proposed here is generalizable to other multi-task multimodal scenarios. The weak-task-protective projection is related to multi-task learning methods such as PCGrad, but its task-strength-adaptive design for MMDG is novel. The design rationale behind domain confidence as a metric can inspire future work requiring assessment of domain-invariance quality. For the video understanding community, this paper offers a gradient-level paradigm for preventing modality collapse and suppression in multimodal joint training.

## Rating

- Novelty: ⭐⭐⭐⭐ (First optimization-perspective approach to MMDG; decoupled modulation and adaptive projection are novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Complete ablations, multi-baseline comparisons, integration experiments, and visualizations; limited dataset variety)
- Writing Quality: ⭐⭐⭐⭐ (Theoretical derivations are clear; motivation is well-substantiated)
- Value: ⭐⭐⭐⭐ (Strong plug-and-play generality; meaningful research direction)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Lifelong Domain Adaptive 3D Human Pose Estimation](lifelong_domain_adaptive_3d_human_pose_estimation.md)
- [\[AAAI 2026\] StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis](stegavar_privacy-preserving_video_action_recognition_via_steganographic_domain_a.md)
- [\[AAAI 2026\] Uncovering Zero-Shot Generalization Gaps in Time-Series Foundation Models Using Real-World Videos](uncovering_zero-shot_generalization_gaps_in_time-series_foundation_models_using_.md)
- [\[AAAI 2026\] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation](emovid_a_multimodal_emotion_video_dataset_for_emotion-centric_video_understandin.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](../../CVPR2026/video_understanding/openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)

</div>

<!-- RELATED:END -->
