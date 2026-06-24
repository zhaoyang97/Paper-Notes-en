---
title: >-
  [Paper Note] Cross-Domain Learning for Video Anomaly Detection with Limited Supervision
description: >-
  [ECCV2024][LLM Pretraining][Video Anomaly Detection] Proposes a weakly-supervised cross-domain learning (CDL) framework that integrates unlabeled external videos into training via an uncertainty-driven pseudo-labeling mechanism, significantly improving the cross-domain generalization capability of video anomaly detection.
tags:
  - "ECCV2024"
  - "LLM Pretraining"
  - "Video Anomaly Detection"
  - "Cross-Domain Learning"
  - "Pseudo-Labeling"
  - "uncertainty estimation"
  - "Weakly-Supervised Learning"
date: 2026-05-08
content_hash: 8a0ad43c463c9326
---

# Cross-Domain Learning for Video Anomaly Detection with Limited Supervision

**Conference**: ECCV2024  
**arXiv**: [2408.05191](https://arxiv.org/abs/2408.05191)  
**Code**: TBD  
**Area**: LLM Pre-training  
**Keywords**: Video Anomaly Detection, Cross-Domain Learning, Pseudo-Labeling, uncertainty estimation, Weakly-Supervised Learning

## TL;DR

Proposes a weakly-supervised cross-domain learning (CDL) framework that integrates unlabeled external videos into training via an uncertainty-driven pseudo-labeling mechanism, significantly improving the cross-domain generalization capability of video anomaly detection.

## Background & Motivation

Video anomaly detection (VAD) aims to automatically localize anomalous events in videos (such as accidents, explosions, and other safety threats) and is a core task in video surveillance. Existing methods are divided into unsupervised (modeling with normal videos only) and weakly-supervised (utilizing video-level labels). Weakly-supervised methods perform exceptionally well within a single domain, but their performance drops significantly in cross-domain scenarios due to three reasons: (1) anomaly definitions are context-dependent, and simple transfer cannot capture cross-domain differences; (2) anomalous events are rare, leading to class imbalance, which becomes more severe during cross-domain transfer; (3) the amount of weakly labeled data is limited, constraining the model's ability to learn open-set anomalies.

Existing cross-domain VAD methods are based on unsupervised techniques and lack explicit modeling of anomaly features, resulting in performance that is insufficient to meet practical demands. Meanwhile, since a large number of unlabeled videos are readily available, combining limited weakly-supervised data with abundant unlabeled data represents an important and practical research direction.

## Core Problem

How to leverage a large number of unlabeled external videos to improve the cross-domain and open-set generalization capabilities of VAD models when only a small amount of weakly labeled source domain data is available? The key challenge is that pseudo-labels of external data contain noise; direct use leads to confirmation bias, requiring an adaptive mechanism to quantify and suppress the uncertainty of pseudo-labels.

## Method

### Overall Architecture

The CDL framework consists of two stages of iterative training:

1. **CDL Step 0**: Train the main model $P_m$ (based on CLIP backbone) and the auxiliary model $P_a$ (based on I3D backbone) separately on the weakly labeled data $\mathcal{D}_l$ using Ranking Loss, then generate segment-level pseudo-labels for the external data $\mathcal{D}_u$.
2. **CDL Step k (k>0)**: Iteratively train on $\mathcal{D}_l \cup \mathcal{D}_u$. Each CDL step contains multiple epochs, with the uncertainty regularization scores recomputed every epoch to adaptively weight the loss of external data. Pseudo-labels are regenerated at the end of each CDL step to progressively improve their quality.

### Dual-Backbone Feature Extraction

- **CLIP backbone** (ViT-B/32): Extracts frame-level features, pooled into a fixed number of $n_s$ segments via bilinear interpolation.
- **I3D backbone**: Extracts segment-level 3D convolutional features.

The two backbones possess contrasting inductive biases (Transformer vs. 3D CNN), providing diversified predictions for uncertainty estimation. During inference, only the CLIP backbone is used.

### Prediction Bias Estimation

Decomposes the prediction bias of external data into two parts: the discrepancy between model prediction and pseudo-label (optimizable term) and the discrepancy between pseudo-label and true label (treated as constant). BCE loss is used to estimate the prediction bias of each segment.

### High-Dimensional Space Uncertainty Estimation

Traditional methods calculate prediction variance in the probability space (binary classification posterior probability), but VAD is a binary task where the probability distribution support is limited. This paper proposes quantifying uncertainty in high-dimensional feature space:

- Take the segment representations $Z_m$ and $Z_a$ from the penultimate layer of $P_m$ and $P_a$.
- Calculate the cosine similarity between segments.
- Obtain the uncertainty regularization score via exponential transformation: $s^j = e^{\tau(\langle z_m^j, z_a^j \rangle - 1)}$.

Higher scores (consistent encoding between the two models) indicate more reliable pseudo-labels; lower scores indicate greater uncertainty.

### Training Objectives

The loss on external data is the uncertainty-weighted BCE loss plus a cosine similarity regularization term:

$$\mathcal{L}_{\text{ext}} = \mathbb{E}[S \cdot \mathcal{L}_{\text{bce}} - \lambda_3 \cdot \langle Z_m, Z_a \rangle]$$

The total loss consists of Ranking Loss (labeled data) plus external data loss: $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{rank}} + \lambda_4 \cdot \mathcal{L}_{\text{ext}}$

### Inference

Use $P_m$ to compute segment-level anomaly scores, uniformly mapping them to the frame level according to the number of frames.

## Key Experimental Results

### Cross-Domain Experiments

**UCF-Crime as the source domain, XD-Violence as the cross-domain target** (Table 2):

| Method | Feature | XDV AP(%) |
|---|---|---|
| zxVAD (Unsupervised SOTA) | - | 40.68 |
| SSRL (Weakly-supervised In-domain SOTA) | I3D | 51.60 |
| CDL (UCF+HACS) | CLIP | **65.14** |
| CDL (UCF+XDV) | CLIP | **68.37** |

Achieves a **+27.69%** improvement on XDV compared to the unsupervised cross-domain SOTA, and a **+16.77%** improvement compared to the best weakly-supervised in-domain method.

**XD-Violence as the source domain, UCF-Crime as the cross-domain target** (Table 3):

| Method | UCF-R AUC(%) |
|---|---|
| zxVAD (Unsupervised SOTA) | 63.61 |
| CDL (XDV+HACS) | **88.50** |

Achieves an absolute improvement of **+24.89%** on UCF-R compared to the unsupervised cross-domain SOTA.

### Open-Set Experiments (Table 4)

When trained with only 1 class of anomaly, CDL achieves an 85.39% AUC on UCF-R (compared to 84.32% without CDL), consistently outperforming all baselines.

### Ablation Study (Table 5, open-set c=1)

| External Data | Uncertainty Weighting | Cosine Loss | UCF-R AUC(%) |
|---|---|---|---|
| ✗ | ✗ | ✗ | 84.32 |
| ✓ | ✗ | ✗ | 84.67 |
| ✓ | ✓ | ✗ | 84.80 |
| ✓ | ✓ | ✓ | **85.39** |

Each component has a positive contribution, with the cosine similarity loss providing the largest gain (+0.59%).

## Highlights & Insights

- **Practical Scenario Modeling**: The combination of weakly-supervised + unlabeled external data is highly aligned with actual deployment needs, being much more practical than purely unsupervised cross-domain methods.
- **High-Dimensional Uncertainty Estimation**: Calculating dual-model prediction differences in the feature space rather than the probability space (which has limited support in binary classification) is more robust.
- **Adaptive Pseudo-Label Refinement**: The uncertainty score serves as an automatic threshold to dynamically weight the loss, avoiding manual hyperparameter tuning and exhibiting a strong negative correlation with pseudo-label quality.
- **Test Set Re-labeling**: Discovered severe annotation noise in the UCF-Crime test set (the proportion of anomaly frames increased from 7.58% to 16.55%), providing a more accurate UCF-R annotation.

## Limitations & Future Work

- Relies on two backbones (I3D and CLIP), incurring high training overhead (though inference only uses CLIP).
- When XDV is used as the source domain, the in-domain performance on XDV (78.61%) is lower than the in-domain SOTA (80.67%), potentially due to in-domain underfitting caused by the simple architecture.
- Iterative pseudo-label refinement requires 40 CDL steps, so training efficiency needs improvement.
- Evaluated only between large-scale datasets with similar definitions (UCF-Crime / XD-Violence), without covering small-scale datasets with large differences in anomaly definitions (such as ShanghaiTech).
- The choice of the amount of external data (11,000 videos) was determined via ablation, lacking theoretical guidance.

## Related Work & Insights

| Dimension | Ours (CDL) | zxVAD | RTFM / S3R etc. |
|---|---|---|---|
| Supervision | Weakly-supervised | Unsupervised | Weakly-supervised |
| External Data | ✓ (Unlabeled) | ✓ (Generate pseudo-anomaly frames) | ✗ |
| Cross-domain Capability | Strong (Significant improvement) | Moderate | Weak (Severe performance drop) |
| Uncertainty Modeling | High-dimensional feature space | None | None |
| Inference Backbone | CLIP | Custom | I3D |

### Inspirations & Connections

- The idea of quantifying uncertainty via dual-model prediction differences can be generalized to other weakly-supervised/semi-supervised video understanding tasks.
- Uncertainty estimation in high-dimensional feature space is more suitable for binary or few-class tasks than in probability space, which is worth exploring in other tasks with low numbers of classes.
- The test set re-labeling work reminds us that the annotation quality of benchmark datasets directly affects the fairness of methodology evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ — Weakly-supervised cross-domain + uncertainty-driven pseudo-labeling is a novel combination; high-dimensional uncertainty estimation is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage of cross-domain, open-set, ablation, correlation analysis, and CDF evolution.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, complete derivation of formulas.
- Value: ⭐⭐⭐⭐ — Strong practicality, significant cross-domain improvement, but scenario coverage is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning](../../ICML2025/llm_pretraining/revisiting_continuity_of_image_tokens_for_cross-domain_few-shot_learning.md)
- [\[AAAI 2026\] Learning Procedural-aware Video Representations through State-Grounded Hierarchy Unfolding](../../AAAI2026/llm_pretraining/learning_procedural-aware_video_representations_through_state-grounded_hierarchy.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[NeurIPS 2025\] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](../../NeurIPS2025/llm_pretraining/learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)
- [\[ICLR 2026\] What Scales in Cross-Entropy Scaling Law?](../../ICLR2026/llm_pretraining/what_scales_in_cross-entropy_scaling_law.md)

</div>

<!-- RELATED:END -->
