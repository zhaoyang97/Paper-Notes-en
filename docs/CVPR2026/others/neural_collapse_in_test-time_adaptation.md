---
title: >-
  [Paper Note] Neural Collapse in Test-Time Adaptation
description: >-
  [CVPR 2026][Neural Collapse] This work extends Neural Collapse (NC) theory from the class level to the sample level, discovering the NC3+ phenomenon (sample feature embeddings align with their corresponding classifier we…
tags:
  - "CVPR 2026"
  - "Neural Collapse"
  - "Test-Time Adaptation"
  - "Out-of-Distribution Robustness"
  - "Feature-Classifier Alignment"
  - "Hybrid Objective"
date: 2026-05-08
content_hash: 27c41a09e4c4c8a7
---

# Neural Collapse in Test-Time Adaptation

**Conference**: CVPR 2026
**arXiv**: [2512.10421](https://arxiv.org/abs/2512.10421)  
**Code**: [https://github.com/Cevaaa/NCTTA](https://github.com/Cevaaa/NCTTA)  
**Area**: Others (Out-of-Distribution Generalization / Test-Time Adaptation)
**Keywords**: Neural Collapse, Test-Time Adaptation, Out-of-Distribution Robustness, Feature-Classifier Alignment, Hybrid Objective

## TL;DR
This work extends Neural Collapse (NC) theory from the class level to the sample level, discovering the NC3+ phenomenon (sample feature embeddings align with their corresponding classifier weights). Building on this, it identifies feature-classifier misalignment at the sample level as the root cause of performance degradation under distribution shift, and proposes NCTTA, which employs a hybrid objective combining geometric proximity and prediction confidence to guide feature re-alignment, achieving a 14.52% improvement over Tent on ImageNet-C.

## Background & Motivation

1. **Background**: Test-Time Adaptation (TTA) has become a practical approach for handling distribution shift. Major methods include prototype-based methods (SHOT, T3A), consistency regularization methods (MEMO, CoTTA), normalization-layer methods (NOTE, SAR), and entropy minimization methods (Tent, EATA, DeYO).

2. **Limitations of Prior Work**: While these methods achieve strong empirical performance through algorithmic optimization at inference time, they generally lack a theoretical understanding of the root cause of model degradation under distribution shift—knowing *what works* without knowing *why*.

3. **Key Challenge**: Neural Collapse (NC) theory reveals the elegant geometric structure of trained DNNs (class means ↔ classifier weight alignment), but its analysis relies on class labels and the full training set to compute class means—both of which are unavailable in TTA, where only unlabeled mini-batches of test data are accessible.

4. **Goal**
    - Extend NC theory to the sample level to make it applicable to TTA scenarios
    - Explain performance degradation under distribution shift from an NC perspective
    - Propose a theoretically motivated TTA method

5. **Key Insight**: Since NC3 states that "class means align with classifier weights," and since during late-stage training intra-class variance approaches zero (NC1), each individual sample's feature should also align with its corresponding classifier weight—this is NC3+.

6. **Core Idea**: Performance degradation equals sample features drifting away from the correct classifier weights. Therefore, the central task of TTA is re-alignment. Since pseudo-labels are unreliable, a hybrid objective combining geometric proximity and prediction confidence is used as a substitute.

## Method

### Overall Architecture
NCTTA updates model parameters at test time via a hybrid-objective-guided contrastive alignment mechanism. The input is an unlabeled test mini-batch; the output is adapted model predictions. The core pipeline: (1) compute FCA distances between sample features and all classifier weights; (2) construct a hybrid objective from FCA distances and prediction confidence; (3) select the top-$k$ most likely correct classes as positive samples and treat the rest as negatives; (4) apply an NC-guided alignment loss to attract positives and repel negatives.

### Key Designs

1. **NC3+: Sample-Level Alignment Collapse**

    - **Function**: Extends NC theory from the class level to the sample level, providing a theoretical foundation for TTA.
    - **Mechanism**: Defines the FCA distance $d_{ij} = \|\frac{\mathbf{h}_i}{\|\mathbf{h}_i\|_2} - \frac{w_j}{\|w_j\|_2}\|_2$ as the normalized Euclidean distance between a sample feature embedding $\mathbf{h}_i$ and the $j$-th class classifier weight $w_j$. It is theoretically shown that under cross-entropy loss, the ground-truth FCA distance $d_{iy_i}$ decreases monotonically toward zero. This is empirically validated on ImageNet-100 with multiple backbones: the G-FCA distance consistently decreases throughout training.
    - **Design Motivation**: NC3 requires class means, which depend on fully labeled data and are unavailable in TTA. NC3+ only requires a single sample's feature and the classifier weights, making it perfectly suited for the TTA setting.

2. **NC3+-Based Explanation of Performance Degradation**

    - **Function**: Explains why OOD samples are misclassified from the perspective of FCA distance.
    - **Mechanism**: Analysis of OOD data reveals two distinct shifts in distance distributions. For correctly classified samples, the G-FCA distance $d_{iy_i}^{\text{correct}}$ remains small (features still align with the correct weights). For misclassified samples, the G-FCA distance $d_{iy_i}^{\text{wrong}}$ increases substantially (features drift away from the correct weights), while the P-FCA distance $d_{i\hat{y}_i}^{\text{wrong}}$ decreases (features drift toward incorrect weights). This gap widens as corruption severity increases.
    - **Design Motivation**: Establishes a quantitative link between feature-classifier misalignment and performance degradation, pointing to re-alignment as the central task of TTA.

3. **NCTTA: Contrastive Alignment with a Hybrid Objective**

    - **Function**: Explicitly guides feature embeddings to re-align with the correct classifier weights during the TTA phase.
    - **Mechanism**: Since pseudo-labels are unreliable when features have already drifted, the predicted label $\hat{y}_i$ cannot directly specify the alignment target. NCTTA constructs a hybrid target $\widetilde{\mathbf{y}}_i = (1-\alpha)\hat{d}_i + \alpha p_i$, where $\hat{d}_i$ is the softmax-normalized FCA distance (geometric proximity) and $p_i$ is the predicted probability (confidence), with $\alpha$ balancing the two. The top-$k$ classes ranked by $\widetilde{\mathbf{y}}_i$ form the positive set $\mathcal{T}_i$, and an NC-guided alignment loss $\mathcal{L}_{\text{NC}}$ attracts positives and repels negatives. A dynamic weight $\lambda_i$ is also introduced, jointly controlled by an entropy metric and the P-FCA distance, to modulate the loss contribution of each sample.
    - **Design Motivation**: Pure pseudo-labels ($\alpha=1, k=1$) yield high error rates under severe shift; pure geometric proximity may be misled by anomalous features. The hybrid scheme is more robust than either alone, and top-$k$ instead of top-1 further increases tolerance to errors.

### Loss & Training
The total loss is $\mathcal{L}_{\text{total}}(x_i) = \lambda_i \cdot \mathbb{I}_{x_i \in S_{\text{ENT}}} \cdot (\mathcal{L}_{\text{ENT}}(x_i) + \mathcal{L}_{\text{NC}}(x_i))$, where $S_{\text{ENT}}$ is the entropy-filtered sample set (excluding high-entropy predictions), $\mathcal{L}_{\text{ENT}}$ is the standard entropy minimization loss, and $\mathcal{L}_{\text{NC}}$ can be instantiated in three forms: InfoNCE-style, L2-style, or Triplet-style.

## Key Experimental Results

### Main Results

| Method | CIFAR-10-C Avg (ResNet50) | ImageNet-C Avg (ViT-B/16) |
|--------|--------------------------|--------------------------|
| no_adapt | 57.39 | 38.88 |
| Tent | 75.19 | 51.87 |
| EATA | 74.04 | 63.91 |
| SAR | 74.67 | 53.97 |
| NOTE | 71.03 | 39.15 |
| MEMO | 68.85 | 45.38 |
| DeYO | 76.65 | 63.49 |
| **NCTTA** | **78.16** | **66.46** |

NCTTA outperforms Tent by 14.59% and DeYO by 2.97% on ImageNet-C.

### Ablation Study

| $\mathcal{L}_{\text{NC}}$ Form | ImageNet-C Contrast (Sev-5) |
|------|------|
| InfoNCE-style | **Best** |
| L2-style | Slightly lower |
| Triplet-style | Lowest |

| $\alpha$ | $k=1$ | $k=3$ | $k=5$ | Note |
|------|-------|-------|-------|------|
| 0.0 (geometry only) | Low | Medium | Medium | Pure FCA distance insufficient |
| 0.5 (hybrid) | Medium | **Best** | Medium | Balances geometry and confidence |
| 1.0 (confidence only) | Lowest | Low | Low | Pure pseudo-labels unreliable |

### Key Findings
- NCTTA achieves the best or second-best performance across nearly all corruption types, demonstrating strong generalization.
- The InfoNCE-style loss is most effective, likely because its contrastive gradients are more informative.
- $\alpha=0.5, k=3$ is the optimal configuration, underscoring the importance of balancing geometry and confidence with a moderate top-$k$ range.
- On the Waterbirds dataset, worst-group accuracy improves from 70.87% (no_adapt) / 75.65% (DeYO) to 76.56%, demonstrating effectiveness against subpopulation shift.
- NCTTA also achieves the best average results in cross-domain PACS experiments.

## Highlights & Insights
- **The bridge between NC theory and TTA is remarkably natural**: NC3+ is a direct corollary of NC3 when NC1 (intra-class variance → 0) holds, yet no prior work had explicitly identified or exploited this. This sample-level perspective perfectly accommodates the TTA constraint of having only unlabeled mini-batches.
- **The hybrid objective design is elegant**: Using geometric proximity to "correct" unreliable pseudo-labels is a well-motivated strategy. Under severe shift, pseudo-label error rates are high, yet geometric neighborhood relations retain a degree of reliability—making the two signals complementary.
- **A complete chain from theory to method to experiment**: The logical progression from NC3+ theoretical discovery → explanation of performance degradation → method design → experimental validation is exceptionally clear and complete, serving as a strong exemplar of theory-driven method design.

## Limitations & Future Work
- The theoretical proof of NC3+ assumes cross-entropy loss and standard TTA conditions; its applicability to models trained with other objectives (e.g., contrastive pre-training) is not discussed.
- NCTTA currently requires iterating over all $K$ classifier weights to compute FCA distances, which may introduce non-trivial computational overhead for large-scale tasks (e.g., ImageNet-21K).
- In continual TTA scenarios where model parameters are updated continuously and classifier weights change over time, whether the assumptions underlying NC3+ remain valid warrants further analysis.
- Label-space shift (open-set TTA) is not considered.

## Related Work & Insights
- **vs. Tent**: Tent performs entropy minimization only, without exploiting the geometric structure between features and classifiers. NCTTA surpasses Tent by 14.59% on ImageNet-C, demonstrating that geometry-guided alignment is more effective than pure entropy minimization.
- **vs. DeYO**: DeYO improves performance through more refined sample selection but still lacks an alignment mechanism. NCTTA further improves by 2.97%.
- **vs. EATA**: EATA also applies entropy filtering but lacks NC-guided alignment. NCTTA outperforms EATA by 4.12% on CIFAR-10-C.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ NC3+ is a new theoretical discovery; the bridge from theory to method is highly elegant
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across multiple datasets and backbones with detailed ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and intuitive visualizations
- Value: ⭐⭐⭐⭐ Provides a new theoretical perspective and practical method for the TTA community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Private and Stable Test-Time Adaptation with Differential Privacy](../../ICML2026/others/private_and_stable_test-time_adaptation_with_differential_privacy.md)
- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](../../ICML2026/others/tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)
- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](../../ICML2026/others/on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](../../NeurIPS2025/others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](vit3_unlocking_test_time_training_in_vision.md)

</div>

<!-- RELATED:END -->
