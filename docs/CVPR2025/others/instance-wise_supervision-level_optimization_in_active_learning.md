---
title: >-
  [Paper Note] Instance-wise Supervision-level Optimization in Active Learning
description: >-
  [CVPR 2025][Active Learning] This paper proposes the ISO (Instance-wise Supervision-level Optimization) framework. In active learning, it not only selects which samples to annotate but also automatically determines the optimal annotation level (exact vs. coarse labels) for each sample. Through a value-cost ratio (VCR) and a diversity-aware batch selection algorithm, it achieves over 10% higher accuracy than traditional active learning under a fixed budget constraint.
tags:
  - "CVPR 2025"
  - "Active Learning"
  - "Weak Supervision"
  - "Annotation Budget Optimization"
  - "Instance-wise Optimization"
  - "Multi-level Annotation"
date: 2026-05-08
content_hash: 8f3675ce92ebefd6
---

# Instance-wise Supervision-level Optimization in Active Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.06517](https://arxiv.org/abs/2503.06517)  
**Code**: [https://github.com/matsuo-shinnosuke/ISOAL](https://github.com/matsuo-shinnosuke/ISOAL)  
**Area**: Others/Active Learning  
**Keywords**: Active Learning, Weak Supervision, Annotation Budget Optimization, Instance-wise Optimization, Multi-level Annotation

## TL;DR

This paper proposes the ISO (Instance-wise Supervision-level Optimization) framework. In active learning, it not only selects which samples to annotate but also automatically determines the optimal annotation level (exact vs. coarse labels) for each sample. Through a value-cost ratio (VCR) and a diversity-aware batch selection algorithm, it achieves over 10% higher accuracy than traditional active learning under a fixed budget constraint.

## Background & Motivation

**Background**: Active learning (AL) maximizes annotation efficiency by iteratively selecting the most valuable samples for annotation. Prevailing methods either select samples with the highest model "uncertainty" (e.g., Margin, Entropy) or cover the data distribution based on diversity (e.g., Coreset). Another direction is weakly supervised learning (WSL), which reduces annotation costs by using coarse-grained but low-cost labels (e.g., using "sparrow" instead of "house sparrow").

**Limitations of Prior Work**: (1) Traditional AL methods only consider a single annotation level (exact labels), failing to exploit the fact that "weak labels are cheaper" to obtain more data. (2) APFWA, the only method combining AL and weak supervision, dynamically adjusts the ratio of weak to exact annotations, but this adjustment relies on global parameters estimated after random sampling—ignoring which specific samples are more suitable for weak annotation and which require exact annotation.

**Key Challenge**: Under a fixed budget constraint, deciding whether to annotate one more weak label (cheap) or exact label (expensive but highly informative) should not be a globally uniform decision, but should vary in an instance-wise manner. The uncertainty of some samples stems from confusion within a superclass (requiring exact labels), while for others, it arises from confusion across superclasses (where weak labels suffice).

**Goal**: To design a unified framework that simultaneously optimizes "which samples to select" and "what annotation level to use for each sample" to maximize annotation efficiency under a fixed budget.

**Key Insight**: Treat supervision-level optimization as a resource allocation problem, where each sample has a "value-cost ratio" (VCR) under weak/exact annotation, and select a batch of diverse samples with high VCRs.

**Core Idea**: Compute the value-cost ratio (VCR = model improvement $\times$ uncertainty / cost) of both weak and exact annotation for each unlabeled sample, and then use an approximation algorithm of determinantal point processes to select batches with both high value and diversity.

## Method

### Overall Architecture

ISO executes the following steps in each active learning round: (1) Compute two VCRs (one for exact and one for weak annotation) for each sample in the unlabeled pool; (2) Represent each sample as VCR-weighted feature vectors; (3) Select a subset of samples that maximizes the volume spanned by the selected vectors under a budget constraint, using a k-means++ based batch selection algorithm; (4) Assign the selected samples to either the exact or weak annotation pool based on the vector type; (5) Train the model using data from both pools before proceeding to the next round.

### Key Designs

1. **Value-Cost Ratio (VCR) Estimation**:

    - **Function**: Evaluate the "cost-effectiveness" for each sample under each annotation level.
    - **Mechanism**: VCR is defined as $v_f(x) = M_f \cdot u_f(x) / C_f$, $v_w(x) = M_w \cdot u_w(x) / C_w$, where $C_f, C_w$ are the annotation costs (preset), $u_f(x), u_w(x)$ represent the uncertainty of the sample on the exact/weak classification heads (percentile-normalized margin values), and $M_f, M_w$ denote the model improvement per unit of data. The model improvement $M$ is estimated by partitioning the current annotated data into $K=5$ subsets, progressively training the model to evaluate the performance curve, and taking the weighted average slope divided by the amount of data to obtain the contribution per sample.
    - **Design Motivation**: Uncertainty indicates "how much this sample needs to be annotated", model improvement shows "how much overall progress this type of annotation can still bring", and cost indicates "how expensive this annotation is". Combining the three provides a comprehensive assessment of cost-effectiveness. Percentile normalization prevents outliers from distorting the scale.

2. **Diversity-Aware Batch Selection**:

    - **Function**: Select a batch of samples with both high VCR and diversity under budget constraints.
    - **Mechanism**: Each sample is represented by two vectors (weak-annotated version and exact-annotated version): $v_f(x) \cdot \tilde{f}(x)$ and $v_w(x) \cdot \tilde{f}(x)$, where $\tilde{f}(x)$ is the normalized feature vector. The objective is to select a set of vectors to maximize the spanned "volume" (equivalent to maximizing the determinant of the Gram matrix). This allows both high VCR and high diversity to increase the objective volume. The selection process uses k-means++ style sequential sampling, where the selection probability at each step is proportional to the squared distance to the nearest already-selected vector, until the budget $B$ is exhausted. Selecting an exact-version vector costs $C_f$, while a weak-version vector costs $C_w$.
    - **Design Motivation**: Greedily selecting the highest VCR leads to choosing a cluster of similar samples because high-uncertainty samples often gather together. Determinant maximization is the optimization objective of Determinantal Point Processes (DPP), naturally balancing "high value" and "diversity". As an efficient approximation algorithm for DPP, k-means++ can operate under budget constraints. The vector magnitude (VCR) and direction (features) simultaneously encode value and spatial location information.

3. **Automatic Decision of Supervision Level**:

    - **Function**: Unify the decision of "which annotation level to select" into the vector selection process.
    - **Mechanism**: The key insight is that the two vectors for the same sample (exact and weak versions) have the same direction but different magnitudes (different VCRs). During the k-means++ sampling process, whichever version of the vector is selected determines the annotation level. If the weak-annotation VCR is higher (meaning the weak label is more cost-effective), the weak-version vector is longer, resulting in a higher probability of being selected. If a sample has exceptionally high uncertainty for exact annotation, the exact-version vector may be longer.
    - **Design Motivation**: This elegantly merges "sample selection" and "annotation level selection"—originally two independent optimization problems—into a single vector selection problem, avoiding the suboptimal two-stage strategy of first determining the ratio and then selecting samples.

### Loss & Training

A two-stage training strategy is adopted: first train the feature extractor and the weak classification head using the weakly annotated data, and then train the exact classification head using the exactly annotated data. The feature extractor $f$ is a ResNet18 encoder, and each of the two classification heads is a linear layer. The loss function is cross-entropy.

## Key Experimental Results

### Main Results

CIFAR-100 (weak annotation cost $C_w = 1/2$), classification accuracy (%):

| Method | Round 1 | Round 3 | Round 5 |
|------|---------|---------|---------|
| Random | ~15 | ~22 | ~28 |
| Margin | ~16 | ~24 | ~30 |
| BADGE | ~17 | ~25 | ~31 |
| APFWA | ~18 | ~28 | ~36 |
| **ISO (Ours)** | **~20** | **~32** | **~42** |

At Round 5, ISO outperforms traditional AL methods by **10%+** and exceeds APFWA by approximately **6%**. While traditional methods require 5 rounds (budget of 5000) to reach 30%, ISO only requires 3 rounds (budget of 3000).

### Ablation Study

| Configuration | CIFAR-100 Round 5 | CUB200 Round 5 | Description |
|------|-------------------|----------------|------|
| **Full ISO** | **~42%** | **~34%** | Full model |
| w/o uncertainty | ~38% | ~28% | Remove instance-wise uncertainty; VCR only uses global model improvement |
| w/o diversity | ~39% | ~31% | Greedily select the highest VCR without considering diversity |

### Influence of Different Weak Annotation Costs

| $C_w$ | CIFAR-100 Round 5 | CUB200 Round 5 |
|-------|-------------------|----------------|
| 1/2 | ~42% | ~34% |
| 1/4 | ~48% | ~38% |
| 1/8 | ~55% | ~43% |

### Key Findings

- **The cheaper the weak annotation, the greater the utility of ISO**: Accuracy with $C_w = 1/8$ is over 13% higher than with $C_w = 1/2$, since a lower weak-annotation cost allows annotating more samples under the same budget.
- **Instance-wise uncertainty is particularly crucial on CUB200** (dropping by 6% when removed), because fine-grained bird classification requires deciding whether an exact label is needed for each specific sample.
- **Diversity contributes more on CIFAR-100**, as there are many classes (100 classes) and a wide data distribution. Failing to consider diversity easily leads to selecting clusters within a specific confounded region.
- **Comparison with fixed-ratio AL + weak supervision baselines**: The optimal ratio varies with $C_w$ (60/40 is optimal when $C_w=1/2$, while 20/80 is optimal when $C_w=1/8$). ISO automatically identifies the suitable ratio, consistently matching or exceeding the optimal fixed ratio.
- **Traditional AL methods that only use exact labels improve significantly after introducing weak labels, proving the value of weak supervision for AL.**

## Highlights & Insights

- **Unified Optimization via Vector Representation**: Unifiedly formulating both "sample selection" and "annotation level selection" into a selection problem within the same vector space is the most elegant design of this work. A single sample is represented by two vectors (weak/exact versions), and the selected version directly dictates the annotation level, eliminating the need for extra hyperparameters or two-stage strategies.
- **Online Estimation of Model Improvement in VCR**: Estimating the "marginal utility of one additional sample" by incrementally partitioning the training data is simple yet effective in capturing the current marginal contributions of different annotation types, with a controllable computational overhead (requiring training on only $K=5$ subsets).
- **Extensibility to More Annotation Levels**: Although the paper only evaluates two levels (weak and exact), the framework can naturally scale. For example, in segmentation tasks, it could involve pixel-level, bounding-box, and image-level annotations, with one vector representing each level.

## Limitations & Future Work

- **Evaluation Limited to Classification Tasks**: The paper acknowledges that the concept of ISO is applicable to tasks like segmentation (strong supervision = mask vs. weak supervision = bbox), but did not experimentally verify it. The information increment brought by weak labels in segmentation might differ substantially from classification.
- **Constraint on the Number of Annotation Levels**: Although extensible, the computational cost of the current VCR estimation method (incremental training) would grow rapidly with more levels.
- **Stability of Model Improvement Estimation**: The estimation of $M_f, M_w$ depends on small validation sets and incremental training, which can be unstable in early active learning rounds when the dataset size is small. Although the paper mitigates this by averaging over 3 runs, the fundamental issue persists.
- **Prerequisite of Superclass Structure**: The definition of weak labels (superclasses) needs to be predefined. In CUB200, the authors defined 70 superclasses based on naming suffixes; in practical applications, such superclass structures might not be trivially available.
- **Small Scale of Datasets**: Both CIFAR-100 and CUB200 are small-scale datasets, and the performance on large-scale datasets (such as ImageNet) remains unexplored.

## Related Work & Insights

- **vs. APFWA**: APFWA also combines weak and exact annotations, but only performs global ratio optimization (e.g., "40% weak annotations in this round") without considering which sample is more suited for which annotation level. ISO optimizes at the instance level and consistently outperforms APFWA.
- **vs. BADGE**: BADGE combines uncertainty and diversity in the gradient space and is the state-of-the-art for single-level annotation. ISO incorporates the annotation-level dimension on top of BADGE, further enhancing efficiency.
- **vs. Coreset**: A pure diversity-based approach that ignores uncertainty. ISO's vector representation elegantly unifies the diversity selection of Coreset and the uncertainty selection of Margin.

## Rating

- Novelty: ⭐⭐⭐⭐ Instance-wise annotation level optimization is a novel problem formulation, and the unified vector representation is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐ Two datasets with multiple ablations, but lacks large-scale datasets and non-classification tasks.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, detailed algorithm description, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Introduces a new dimension (annotation level) to active learning, holding substantial practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[CVPR 2026\] Mitigating Instance Entanglement in Instance-Dependent Partial Label Learning](../../CVPR2026/others/mitigating_instance_entanglement_in_instance-dependent_partial_label_learning.md)
- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)
- [\[ICML 2025\] AutoAL: Automated Active Learning with Differentiable Query Strategy Search](../../ICML2025/others/autoal_automated_active_learning_with_differentiable_query_strategy_search.md)
- [\[NeurIPS 2025\] Reliable Active Learning from Unreliable Labels via Neural Collapse Geometry](../../NeurIPS2025/others/reliable_active_learning_from_unreliable_labels_via_neural_collapse_geometry.md)

</div>

<!-- RELATED:END -->
