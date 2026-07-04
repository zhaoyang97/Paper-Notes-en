---
title: >-
  [Paper Note] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation
description: >-
  [ECCV2024][active learning] The BUAL framework is proposed to push unknown-class samples toward high-confidence regions and known-class samples toward low-confidence regions using Random Label Negative Learning. Combined with a bidirectional uncertainty sampling strategy, the framework effectively selects highly informative known-class samples under open-set scenarios.
tags:
  - "ECCV2024"
  - "active learning"
  - "Open-Set Annotation"
  - "Negative Learning"
  - "uncertainty estimation"
date: 2026-05-08
content_hash: 024799eaf513bd33
---

# Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation

**Conference**: ECCV2024  
**arXiv**: [2402.15198](https://arxiv.org/abs/2402.15198)  
**Code**: [GitHub](https://github.com/zongcc/BUAL)  
**Area**: Others  
**Keywords**: active learning, Open-Set Annotation, Negative Learning, uncertainty estimation

## TL;DR

The BUAL framework is proposed to push unknown-class samples toward high-confidence regions and known-class samples toward low-confidence regions using Random Label Negative Learning. Combined with a bidirectional uncertainty sampling strategy, the framework effectively selects highly informative known-class samples under open-set scenarios.

## Background & Motivation

- **Core Goal of Active Learning**: Iteratively select the most informative samples from an unlabeled data pool for annotation, training an efficient model with minimal labeling cost.
- **Limitations of the Closed-Set Assumption**: Traditional AL methods assume that the categories in the unlabeled pool are identical to those of the target task. However, in real-world scenarios, a large number of unknown-class (open-set) samples are often mixed in.
- **Dilemma of Existing Methods**:
    - Traditional uncertainty-based methods (LC / Margin / Entropy) tend to select low-confidence samples; however, unknown-class samples also exhibit low confidence, leading to frequent false selections.
    - Open-set annotation methods (such as CCAL and LfOSA) prioritize samples most likely to belong to known classes; however, these are often "easy samples" that the model has already mastered, offering limited utility for training.
    - Both types of methods are sensitive to the openness ratio: OSA methods perform worse than random sampling at a low openness ratio, while traditional methods fail at a high openness ratio.

## Core Problem

**How to perform sample selection in open-set scenarios that simultaneously satisfies the dual objectives of "high informativeness" and "belonging to known classes"?**

Key Insight: If unknown-class samples can be pushed to high-confidence regions, existing uncertainty-based AL methods can be directly applied to open-set scenarios, as the remaining samples in the low-confidence regions will be highly informative known-class samples.

## Method

### 1. Random Label Negative Learning (RLNL)

**Core Idea**: Finetune the model using Negative Learning (complementary label learning) to achieve separation of known and unknown class samples in the confidence space.

**Training Workflow**:

- **First Phase (Positive Training)**: Train a $K$-class classifier $f_p(\cdot)$ (positive classifier) normally using cross-entropy with the labeled known-class data $D_l^{kno}$.
- **Second Phase (Negative Finetuning)**: Replace the final classification head and finetune to obtain $f_n(\cdot)$ (negative classifier) using Negative Learning loss.

**Negative Learning Loss Function**:

$$\ell_{NL}(f, \bar{y}) = -\sum_{k=1}^{K} \bar{y}_k \log(1 - p_k)$$

**Random Label Assignment Strategy**:

- For labeled known-class samples: Uniformly sample complementary labels from $\mathcal{Y} \setminus y^l$ (excluding the ground-truth label).
- For unlabeled samples: Uniformly and randomly sample labels from $\mathcal{Y}$ (all classes).
- Resample labels in each training iteration.

**Why does RLNL work?**

- Unlabeled **known-class samples** have a $1/K$ probability of being assigned to their correct labels, incurring heavy penalties that push them into low-confidence regions. Meanwhile, their overlap with labeled data in the feature space imposes implicit constraints based on prior knowledge.
- Unlabeled **unknown-class samples** can never be assigned correct labels (as their true classes are not in $\mathcal{Y}$). Under mini-batch gradient updates, they oscillate away from the decision boundary toward high-confidence regions.
- t-SNE visualization experiments verify that after RLNL, representations of unknown-class samples are distinctly far from the decision boundary, while known-class samples remain near the labeled data.

### 2. Bidirectional Uncertainty (BU) Sampling Strategy

Since the negative classifier $f_n(\cdot)$ is unstable during training and its predictions oscillate across epochs:

- Collect predictions of $f_n$ every $m$ epochs, and average them over $t$ collections to obtain $\mathcal{P}^-$.
- Simultaneously obtain positive predictions $p^+$ using $f_p$.

**Bidirectional Uncertainty Sampling Formula**:

$$x^* = \arg\max_x \; p_{K+1}^{aux}(x) \cdot unc_n + r \cdot [1 - p_{K+1}^{aux}(x)] \cdot unc_p$$

where:

- $unc_p$: Uncertainty of the positive classifier (more accurate for known-class samples).
- $unc_n$: Uncertainty of the negative classifier (more effective at distinguishing unknown classes).
- $p_{K+1}^{aux}$: **Local balance factor**, obtained from an auxiliary $K+1$-class classifier, where higher values indicate a higher probability that the sample belongs to an unknown class.
- $r$: **Global balance factor**, representing the proportion of known-class samples in the last query round, reflecting the current openness of the data pool.

**Adaptive Degeneration**: When there are no unknown-class samples, $r=1$ and $p_{K+1}^{aux}=0$, and the formula degenerates into standard uncertainty sampling.

### 3. Three Concrete Instantiations

- **B-LC (Bidirectional Least Confident)**: Based on the complement of the maximum predicted probability.
- **B-Margin**: Based on the difference between the top-two predicted probabilities.
- **B-Entropy**: Based on predictive entropy.

## Key Experimental Results

**Datasets**: CIFAR-10, CIFAR-100, Tiny-ImageNet, with the openness ratio set to 0.2/0.4/0.6/0.8.

**Main Results** (average accuracy in the final round):

| Method | CIFAR-10 (0.6) | CIFAR-100 (0.6) | Tiny-ImageNet (0.6) |
|---|---|---|---|
| Random | 87.2 | 58.7 | 50.9 |
| Margin | 89.0 | 58.8 | 50.8 |
| LfOSA | 87.0 | 62.4 | 52.4 |
| CCAL | 88.0 | 64.7 | 50.3 |
| **B-Margin** | **92.6** | **68.3** | **55.7** |

- BUAL performs best across all openness ratios and is **insensitive** to variations in openness.
- Although LfOSA achieves the highest identification rate for known classes, the queried samples highly overlap with labeled data (validated via t-SNE), indicating they are easy samples already mastered by the model.
- Ablation Study: Using only $unc_p$ yields 87.5%, using only $unc_n$ yields 89.4%, combining them bidirectionally yields 90.8%, and adding the two balance factors yields 92.5%.

## Highlights & Insights

1. **Ingenious Concept**: Instead of directly identifying unknown classes, it "pushes away" unknown classes using RLNL, making traditional uncertainty-based methods naturally applicable to open-set scenarios.
2. **Clear Theoretical Intuition**: It exploits the asymmetry where known-class samples have a probability of being assigned correct labels and thus get penalized, whereas unknown-class samples can never be assigned correct labels.
3. **High Versatility**: BUAL is a framework that can extend any uncertainty-based AL method to open-set scenarios.
4. **Adaptive Mechanism**: The global ($r$) and local ($p_{K+1}^{aux}$) balance factors ensure the stability of the method across different openness levels.

## Limitations & Future Work

1. **Computational Overhead**: Training three classifiers ($f_p, f_n, f_{aux}$) is required, and multiple predictions must be collected and averaged during the RLNL phase.
2. **Subset Sampling**: The random sampling of $D_{sub}$ may introduce bias, especially under extreme class imbalances.
3. **Limited to Image Classification**: Open-set active learning has not been explored for other tasks such as NLP, object detection, or segmentation.
4. **Convergence of Negative Learning**: The paper acknowledges that predictions from $f_n$ are unstable and require multiple averages, without providing convergence guarantees.
5. **Assumption of Known Openness Ratio**: Although adaptively estimated via $r$, the estimation in the initial rounds may be inaccurate.

## Related Work & Insights

| Method | Strategy Type | Core Idea | Robustness to Openness |
|---|---|---|---|
| LC / Margin / Entropy | Uncertainty | Select low-confidence samples | Fails at high openness |
| Coreset / BADGE | Diversity / Hybrid | Select representative samples in distribution | Feature discrepancy of unknown classes leads to false selection |
| CCAL | Contrastive Learning | Select samples semantically similar to known classes | Worse than random at low openness |
| LfOSA | MAV Modeling | Select samples with high maximum activation values | Selects easy samples |
| DIAS | Open-Set Recognition | Identify unknown classes first and then filter | Poor identification capability with limited labeled data |
| **BUAL** | Bidirectional Uncertainty | Push away unknown classes + bidirectional sampling | **Stable under all openness ratios** |

## Related Work & Insights

- Negative Learning has been applied in noisy label learning. This work creatively transfers it to open-set active learning, providing a valuable paradigm for **cross-task methodology transfer**.
- The design of dual balance factors (global + local) is a general adaptive strategy that can be extended to other sampling problems requiring multi-objective balancing.
- There is a potential connection to the field of out-of-distribution (OOD) detection: RLNL is essentially an unsupervised method for OOD signal enhancement.

## Rating
- Novelty: ⭐⭐⭐⭐ — The RLNL concept is novel, and utilizing negative learning to push away unknown classes is an innovative contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets $\times$ four openness ratios, thorough ablation studies, and comprehensive visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, intuitive illustrations, and standard logical flow.
- Value: ⭐⭐⭐⭐ — Provides a practical framework to extend closed-set AL methods to open-set scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](../../CVPR2025/others/rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)
- [\[ECCV 2024\] FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information](fisherrf_active_view_selection_and_mapping_with_radiance_fields_using_fisher_inf.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](../../CVPR2025/others/instance-wise_supervision-level_optimization_in_active_learning.md)

</div>

<!-- RELATED:END -->
