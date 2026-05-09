---
title: >-
  [Paper Note] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry
description: >-
  [NeurIPS 2025 (Workshop)][Active Learning] This paper proposes NCAL-R, which leverages the Neural Collapse (NC) geometry emerging in the terminal training phase of deep networks. Two scoring metrics—Class Mean Alignment Perturbation (CMAP) and Feature Fluctuation (FF)—are designed for sample selection, making active learning more reliable under label noise and distribution shift. The method consistently outperforms conventional AL baselines on ImageNet-100 and CIFAR-100.
tags:
  - NeurIPS 2025 (Workshop)
  - Active Learning
  - Neural Collapse
  - Feature Geometry
  - Noise-Robust Learning
  - OOD Generalization
date: 2026-05-08
content_hash: 0bef719509c76d02
---

# Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry

**Conference**: NeurIPS 2025 (Workshop)
**arXiv**: [2510.09740](https://arxiv.org/abs/2510.09740)
**Code**: [https://github.com/Vision-IIITD/NCAL](https://github.com/Vision-IIITD/NCAL)
**Area**: Active Learning / Reliable Machine Learning
**Keywords**: Active Learning, Neural Collapse, Feature Geometry, Noise-Robust Learning, OOD Generalization

## TL;DR

This paper proposes NCAL-R, which leverages the Neural Collapse (NC) geometry emerging in the terminal training phase of deep networks. Two scoring metrics—Class Mean Alignment Perturbation (CMAP) and Feature Fluctuation (FF)—are designed for sample selection, making active learning more reliable under label noise and distribution shift. The method consistently outperforms conventional AL baselines on ImageNet-100 and CIFAR-100.

## Background & Motivation

**Background**: Active Learning (AL) reduces annotation costs by prioritizing the most informative samples. Mainstream strategies include uncertainty-based, diversity-based, and representativeness-based methods.

**Limitations of Prior Work**: Traditional AL methods perform well under ideal conditions but face three challenges in realistic scenarios: (1) *Label noise*—annotators make mistakes, and AL heuristics (especially uncertainty-based ones) tend to repeatedly select mislabeled samples, amplifying errors; (2) *Distribution shift*—when training and test distributions differ, conventional selection strategies fail; (3) *Poor transferability*—many methods require task-specific tuning.

**Key Challenge**: Samples with high uncertainty may be genuinely informative, or they may simply be mislabeled or OOD. Conventional methods cannot distinguish "valuable uncertainty" from "harmful uncertainty."

**Goal**: How to select samples that reinforce inter-class separation and expose genuinely ambiguous regions when labels are unreliable and distribution shift may be present?

**Key Insight**: In the terminal training phase, deep network features exhibit Neural Collapse (NC)—within-class features collapse to their class mean, and class means arrange into an equiangular tight frame. This structured geometric information provides a selection signal beyond traditional heuristics: samples that perturb inter-class geometry (high CMAP) are valuable, while samples whose features fluctuate strongly across training (high FF) signal true ambiguity.

**Core Idea**: Use the geometric structure of Neural Collapse to identify samples with structural influence on the feature space, replacing conventional uncertainty/diversity heuristics.

## Method

### Overall Architecture

At each AL round, NCAL-R trains the model on the current labeled set until the NC phase, then computes CMAP and FF scores for each sample in the unlabeled pool. The two scores are normalized and averaged into a composite score, and the top-$k$ samples are selected for annotation. No auxiliary networks, pseudo-labels, or task-specific tuning are required; the method is applicable to any backbone that provides feature embeddings.

### Key Designs

1. **Class Mean Alignment Perturbation (CMAP)**

   - **Function**: Quantifies the degree to which a candidate sample perturbs the inter-class geometric structure.
   - **Mechanism**: The Class Mean Alignment (CMA) is defined as the average cosine similarity between all pairs of class means. For a candidate sample $x$ predicted to belong to class $c$, the updated class mean $\tilde{\mu}_t^c$ is computed assuming $x$ is added to the labeled set, and the change in CMA is $\mathrm{CMAP}(x) = \mathrm{CMA}(\mathcal{L}_t \cup x) - \mathrm{CMA}(\mathcal{L}_t)$. Through algebraic simplification, this reduces to an efficient dot product: $(\bar{\tilde{\mu}}_t^c - \bar{\mu}_t^c)^\top (M_t - \bar{\mu}_t^c)$. A high CMAP indicates that annotating the sample would substantially alter the inter-class mean relationships, helping reduce inter-class correlation and thus an upper bound on generalization error.
   - **Design Motivation**: Based on the theoretical result of Jin et al. (2020), the generalization error upper bound is related to weight correlations. Under NC, class means align with classifier weights; therefore, minimizing CMA serves as a proxy for minimizing generalization error.

2. **Feature Fluctuation (FF)**

   - **Function**: Captures the instability of a sample's representation during training.
   - **Mechanism**: Given multiple checkpoints $\{\theta_t\}_{t=T_i}^{T_f}$ in the terminal training phase, FF counts the number of predicted label changes for sample $x$ across consecutive checkpoints: $\mathrm{FF}(x) = \sum_{t=T_i+1}^{T_f} \mathbf{1}[\hat{y}_t(x) \neq \hat{y}_{t-1}(x)]$. A high FF indicates that predictions continue to oscillate even during the NC phase when most features have stabilized, identifying samples near the true decision boundary.
   - **Design Motivation**: Traditional uncertainty metrics (e.g., entropy) are snapshots at a single moment, whereas FF measures stability across time, better distinguishing "temporarily uncertain" (low FF) from "intrinsically boundary-adjacent" (high FF) samples.

3. **Joint Acquisition Strategy**

   - **Function**: Integrates structural influence and prediction instability into a unified score.
   - **Mechanism**: CMAP and FF are each normalized by their mean and standard deviation, then averaged: $\text{Score}(x) = (\text{CMAP}(x) + \text{FF}(x))/2$. The top-$k$ samples are selected, ensuring that chosen samples both structurally impact feature geometry and reside in genuinely ambiguous regions.
   - **Design Motivation**: CMAP targets inter-class structural optimization while FF targets ambiguity discovery; the two are complementary.

## Key Experimental Results

### Main Results (CIFAR-10, OOD Detection AUROC, ImageNet-100 Training)

| Method | 10% | 15% | 20% | 25% | 30% | 35% |
|--------|-----|-----|-----|-----|-----|-----|
| Random | 77.18 | 80.57 | 84.13 | 85.45 | 86.89 | 87.82 |
| CoreSet | 81.56 | 83.73 | 85.66 | 87.10 | 88.29 | 88.95 |
| CDAL | 81.78 | 84.28 | 85.90 | 86.34 | 87.98 | 88.92 |
| **NCAL** | **82.49** | **85.55** | **87.89** | **89.15** | **90.53** | **91.53** |

### OOD Generalization (30% Label Budget, Linear Probing after ImageNet-100 Training)

| Method | ImgNet-R | CIFAR100 | Flowers | NINCO | CUB | Avg |
|--------|----------|----------|---------|-------|-----|-----|
| Random | 18.06 | 41.64 | 58.69 | 64.23 | 37.84 | 46.95 |
| CDAL | 17.56 | 41.98 | 58.13 | 65.87 | 38.53 | 47.21 |
| **NCAL** | **19.27** | **43.78** | **60.87** | **67.66** | **40.01** | **48.98** |
| 100% data | 20.01 | 45.31 | 61.77 | 69.90 | 42.29 | 50.87 |

### GCD (Generalized Category Discovery, 60-40 Known-Novel Split)

| Method | All Classes | Old Classes | New Classes |
|--------|------------|-------------|-------------|
| Random | 33.20 | 50.34 | 20.35 |
| CoreSet | 32.23 | 49.98 | 18.92 |
| **NCAL** | **35.07** | **51.95** | **23.05** |

### Key Findings

- NCAL consistently outperforms all baselines across all label budgets (10%–35%), with larger gains at low budgets.
- OOD generalization improves by approximately 2% on average, indicating that NC-guided feature spaces are more transferable.
- Novel class discovery accuracy improves by +2.1 points over the best baseline, suggesting that NCAL's feature space naturally accommodates new categories.
- Inter-class distance analysis shows that NCAL achieves a mean inter-class distance of 15.944 (vs. 15.114 for Random), reflecting better class separation.
- Under long-tailed distributions, NCAL improves by approximately 3% (45.15% vs. 42.30%), demonstrating that geometry-guided selection is effective for imbalanced data.

## Highlights & Insights

- **Transforming Neural Collapse from an explanatory theory into a practical tool**: NC has primarily been used to understand training dynamics; this paper is the first to systematically apply it to AL sample selection, opening a practical direction for NC. This methodology is transferable to curriculum learning and data selection.
- **Elegant derivation of CMAP**: By exploiting the NC condition that class means approximately align with classifier weights, the generalization error upper bound is converted into a geometric measure in feature space, further simplified to an efficient dot product computation—theoretically principled and computationally practical.
- **Lightweight design requiring no additional components**: No auxiliary networks, pseudo-labels, or specific architectures are needed; only feature embeddings and training checkpoints are required.

## Limitations & Future Work

- As a workshop paper, the experimental scale is limited (ResNet-18 backbone, largest dataset ImageNet-100); performance on large-scale models and datasets remains unknown.
- FF requires storing multiple training checkpoints, with storage and computation costs scaling with model size and number of checkpoints.
- NC theory requires training to near-zero error, but models may not fully reach the NC state in practice, potentially weakening the theoretical guarantees of CMAP.
- Comparisons with more recent AL methods (e.g., BADGE, BAIT) are absent.

## Related Work & Insights

- **vs. CoreSet**: CoreSet pursues coverage diversity in feature space; NCAL pursues structural optimization of inter-class geometry. The latter is more theoretically grounded and shows greater advantages on GCD tasks.
- **vs. CDAL**: CDAL (Contextual Diversity) considers contextual diversity but still relies on static snapshot-based selection. NCAL's FF introduces a temporal instability signal.
- **vs. ActiveOOD (SISOMe)**: ActiveOOD relies on OOD-filtering heuristics and underperforms in closed-set AL; NCAL's unified framework is effective in both OOD and closed-set settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Guiding AL with Neural Collapse is a novel entry point with elegant theoretical derivation.
- **Experimental Thoroughness**: ⭐⭐⭐ — Workshop format limits experimental scale; large-model validation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Concise and clear, with compact mathematical derivations.
- **Value**: ⭐⭐⭐⭐ — Opens a new application direction for NC in AL; the CMAP+FF design methodology is methodologically inspiring.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The Persistence of Neural Collapse Despite Low-Rank Bias](the_persistence_of_neural_collapse_despite_low-rank_bias.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[ICCV 2025\] Joint Asymmetric Loss for Learning with Noisy Labels](../../ICCV2025/others/joint_asymmetric_loss_for_learning_with_noisy_labels.md)
- [\[NeurIPS 2025\] Neural Collapse in Cumulative Link Models for Ordinal Regression: An Analysis with Unconstrained Feature Model](neural_collapse_in_cumulative_link_models_for_ordinal_regression_an_analysis_wit.md)
- [\[ICLR 2026\] Active Learning for Decision Trees with Provable Guarantees](../../ICLR2026/others/active_learning_for_decision_trees_with_provable_guarantees.md)

<!-- RELATED:END -->
