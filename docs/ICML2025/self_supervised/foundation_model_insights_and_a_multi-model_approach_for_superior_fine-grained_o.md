---
title: >-
  [Paper Note] Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection
description: >-
  [ICML 2025][Self-Supervised Learning][Subset Selection] This paper systematically investigates the advantages and disadvantages of using Foundation Models (FMs) to replace conventional Information Extractors (IEs) for subset selection. It is found that FMs significantly outperform traditional IEs on fine-grained datasets. Consequently, the RAM-APL method is proposed to utilize multiple FMs (DINOv2 + CLIP) to jointly measure sample importance from both intra-class and inter-cl…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Subset Selection"
  - "Foundation Models"
  - "Fine-Grained Classification"
  - "Multi-Model Fusion"
  - "Data-Efficient Training"
date: 2026-05-08
content_hash: 43e5c1c9e035557d
---

# Foundation Model Insights and a Multi-Model Approach for Superior Fine-Grained One-shot Subset Selection

**Conference**: ICML 2025  
**arXiv**: [2506.14473](https://arxiv.org/abs/2506.14473)  
**Code**: [GitHub](https://github.com/ZhijingWan/RAM-APL)  
**Area**: Self-Supervised Learning  
**Keywords**: Subset Selection, Foundation Models, Fine-Grained Classification, Multi-Model Fusion, Data-Efficient Training

## TL;DR

This paper systematically investigates the advantages and disadvantages of using Foundation Models (FMs) to replace conventional Information Extractors (IEs) for subset selection. It is found that FMs significantly outperform traditional IEs on fine-grained datasets. Consequently, the RAM-APL method is proposed to utilize multiple FMs (DINOv2 + CLIP) to jointly measure sample importance from both intra-class and inter-class dimensions, achieving SOTA results on three fine-grained datasets.

## Background & Motivation

Coreset selection (subset selection) aims to select a small, representative subset from large-scale training data to reduce training costs without significantly sacrificing model performance. Traditional one-shot subset selection methodologies rely on pre-trained models as **Information Extractors (IEs)** to obtain features, gradients, or uncertainty scores. However, traditional IEs require pre-training on the target dataset, posing a **dataset dependency** issue that restricts their application to large-scale, novel datasets.

Foundation Models (FMs)—such as DINOv2, CLIP, SigLIP, and EVA-CLIP—with their powerful generalization capabilities, are promising candidates to replace traditional IEs and establish a **dataset-agnostic** subset selection pipeline. Nevertheless, prior work (Xie et al., 2023) discovered that directly utilizing FMs does not always outperform conventional IEs, raising two core questions:

1. Under what conditions do FMs surpass traditional IEs?
2. Is the performance across different FMs consistent?

This paper answers both questions through extensive experiments and proposes a novel method based on these findings.

## Method

### Overall Architecture

The proposed method comprises three levels:

**Level 1: Single-Model Study**  
Experiments are conducted across five datasets (CIFAR-10, CIFAR-10N, CIFAR-10I, Oxford-IIIT Pet, and Pet-N) using three categories of IEs (target-dataset pre-trained models `model-TD`, TinyImageNet pre-trained models `model-TIN`, and single FMs) in conjunction with four classic selection algorithms (MIN, K-Center Greedy, Graph Cut, and Moderate\_DS) at sampling rates of 10%, 30%, and 50%. Three key observations are derived:

- **Observation 1**: FMs exhibit limited advantages on noisy, coarse-grained datasets.
- **Observation 2**: FMs significantly and consistently outperform traditional IEs on fine-grained datasets (regardless of whether they contain noise).
- **Observation 3**: Different FMs yield varying subset selection performances, and higher accuracy of an FM on downstream tasks does not necessarily imply superior subset selection.

**Level 2: Multi-Model Pipeline**  
Based on Observations 2 and 3, a new pipeline is proposed that jointly utilizes multiple FMs as IEs, avoiding the additional step of selecting the optimal single FM required in single-model pipelines.

**Level 3: RAM-APL Method**  
Under the multi-model pipeline, a sample importance metric is designed to simultaneously consider both **intra-class distribution** and **inter-class distribution**.

### Key Designs

#### RAnking Mean (RAM) — Intra-class Distribution Metric

The core challenge addressed by RAM is: how to fuse features extracted by different FMs with mismatching dimensions and misaligned feature spaces?

**Core Idea**: Map features from unaligned feature spaces to a unified **distance ranking space**.

Detailed steps:

1. Extract the feature set $\mathcal{F}^i$ for each FM $M_F^i$.
2. Compute the central feature $\tilde{F}_c^i$ for each class $c$ (mean of all sample features within the class).
3. Compute the Euclidean distance $d(F_j^i, \tilde{F}_c^i)$ from each sample to its corresponding class center.
4. Sort the samples based on distance within each class to obtain a ranking value $r_j^i$.
5. Average and normalize the rankings across all FMs: $\bar{r}_j = \frac{1}{m \times |S|} \sum_{i=1}^{m} r_j^i$.

A smaller ranking mean indicates that the sample is closer to the class prototype, representing stronger intra-class representativeness. This ranking space mapping elegantly bypasses the issue of dimensionality misalignment.

#### Accuracy of Pseudo-class Labels (APL) — Inter-class Distribution Metric

APL measures **whether a sample is prone to misclassification into other classes** across different FM feature spaces.

Detailed steps:

1. For each FM $M_F^i$, calculate the central features for all $C$ classes.
2. For each sample, compute distances to all class centers, assigning the class of the nearest center as the **pseudo-class label** $\tilde{y}_j^i$.
3. If the pseudo-class label equals the ground-truth label, score $\varphi_j^i = 1$; otherwise, $\varphi_j^i = 0$.
4. Average the scores across all FMs: $\bar{\varphi}_j = \frac{1}{m} \sum_{i=1}^{m} \varphi_j^i$.

A lower $\bar{\varphi}_j$ indicates that the sample is more easily misclassified, shares more similarity with other classes, and represents a hard sample on the inter-class decision boundary.

#### Final Score and Selection

The two dimensions are linearly combined to compute the final score:

$$Score = W_1 \times \bar{\mathcal{R}} + W_2 \times (1 - \bar{\varphi})$$

Samples with the lowest scores are selected for the subset.

### Loss & Training

**Adaptive Weighting Mechanism**: $W_1$ and $W_2$ are dynamically adjusted based on the sampling rate $p$:

$$W_1 = \alpha + (1-\alpha) \times \frac{1}{1 + e^{\beta(p - 0.5)}}, \quad W_2 = 1 - W_1$$

Design motivation:

- **Low sampling rates** (e.g., 1%, 10%): $W_1$ is larger, prioritizing "easy" samples with strong intra-class representativeness, which benefits early model optimization.
- **High sampling rates** (e.g., 50%, 70%): $W_2$ gradually increases, soft-introducing more hard samples near inter-class boundaries to bolster the model's ability to distinguish fine-grained nuances.
- Key constraint: $W_1 > W_2$ is always maintained, meaning intra-class evaluation always dominates.

Default hyperparameters: $\alpha = 0.2$, $\beta = 1$.

**Target Model Training**: SGD optimizer, batch size 128, initial learning rate 0.1 with Cosine decay, momentum 0.9, weight decay $5 \times 10^{-4}$, trained for 200 epochs, with random crop to 224×224 and random horizontal flip.

## Key Experimental Results

### Main Results

Compared to 12 baseline methods across three fine-grained datasets, reporting the average accuracy gain at various sampling rates (relative to the Random baseline):

| Dataset | Metric | RAM-APL | Second Best (GC) | Gain |
|---|---|---|---|---|
| Oxford-IIIT Pet | Average Gain | **+3.74%** | +1.52% | +2.22% |
| Food-101 | Average Gain | **+4.44%** | +3.04% | +1.40% |
| CUB-200-2011 | Average Gain | **+6.40%** | +2.78% | +3.62% |

RAM-APL outperforms all baseline methods across all datasets and sampling rates.

### Ablation Study

| Configuration | 1% | 50% | 70% | Explanation |
|---|---|---|---|---|
| MIN (Model-TD) | 5.6±0.7 | 40.3±2.6 | 55.2±2.7 | Traditional IE baseline |
| MIN (CLIP) | 5.6±0.2 | 45.9±1.8 | 56.3±0.7 | Single FM as IE |
| MIN (DINOv2) | 6.2±0.1 | 46.8±2.0 | 60.5±2.9 | Single FM, DINOv2 is superior |
| RAM (CLIP+DINOv2) | 5.9±0.3 | 47.1±1.4 | 56.5±2.7 | RAM-only fusion, competitive with single models |
| **RAM-APL** | **6.5±0.4** | **47.5±1.9** | 58.7±2.2 | Full method, optimal at 1% and 50% |

Feature fusion strategy comparison (Pet dataset):

| Fusion Strategy | 1% | 30% | 50% | 70% |
|---|---|---|---|---|
| Concatenate | 5.9±0.4 | 31.7±1.3 | 47.7±3.0 | 57.8±1.2 |
| **RAM-APL (Ranking Fusion)** | **6.5±0.4** | **32.4±2.9** | 47.5±1.9 | **58.7±2.2** |

### Key Findings

1. **FMs hold a decisive advantage on fine-grained datasets**: On Pet and Pet-N, FMs act as the optimal IE in 9 out of 12 configurations, whereas they are optimal in only 4/12 configurations on CIFAR-10N.
2. **FM downstream performance $\neq$ subset selection performance**: Although EVA-CLIP demonstrates the strongest zero-shot classification performance on Pet, it is not the optimal IE under any selection method.
3. **Multi-model outperforms single-model**: The combination of DINOv2 + CLIP strikes the best balance between efficiency and accuracy.
4. **Ranking space fusion excels over feature concatenation**: This is particularly evident at higher sampling rates.
5. **Increasing the number of FMs is not always beneficial**: Utilizing all four FMs does not yield superior performance compared to the DINOv2 + CLIP dual-model setup.
6. **Adaptive weighting outperforms equal-weight fusion**: The sigmoid weighting strategy with $\alpha=0.2, \beta=1$ yields better results than $W_1=W_2=1$.

## Highlights & Insights

- **Ranking space mapping is the core innovation**: By mapping unaligned high-dimensional features to integer rankings before averaging, RAM elegantly resolves the misalignment issue in multi-model feature fusion without requiring additional alignment networks or projection layers.
- **Dual-perspective metric of intra-class + inter-class distributions**: Current methods typically focus solely on intra-class distribution (e.g., geometric methods) or decision boundaries (e.g., margin methods). RAM-APL comprehensively accounts for both dimensions.
- **Strong intuition behind sampling-rate adaptive weights**: At low sampling rates, prioritizing "easy samples" facilitates model convergence, while progressively introducing "hard samples" at higher sampling rates enhances generalization. This aligns closely with curriculum learning paradigms.
- **Rigorous experimental design**: The single-model study spans 5 datasets $\times$ 4 algorithms $\times$ 3 types of IEs $\times$ 3 sampling rates = 180 experimental setups, providing robust empirical support for the conclusions.
- **Simplicity and efficiency**: The method requires no additional training, no fine-tuning of FMs, and no gradient calculation—relying solely on feature extraction, distance computation, and ranking.

## Limitations & Future Work

1. **Limited to image classification evaluation**: Whether the method generalizes to other downstream tasks, such as object detection or segmentation, remains to be verified.
2. **Restricted to fine-grained datasets**: The advantages of FMs are less pronounced on coarse-grained and noisy datasets, making RAM-APL primarily suitable for fine-grained scenarios.
3. **Pseudo-label dependency on ground truth**: Evaluating the correctness of pseudo-class labels in APL requires ground-truth labels, which is inapplicable in unsupervised or weakly-supervised settings.
4. **Manual selection of FM combinations**: Currently, DINOv2 + CLIP is used as the default combination; there is a lack of automated selection/weighting mechanisms for FMs.
5. **Fixed weight function formulation**: The parameters $\alpha$ and $\beta$ in the sigmoid weight function are manually tuned; more adaptive strategies have not been explored.
6. **Limited consideration of class imbalance**: The method is evaluated under class-balanced sampling, and its robustness to severe class imbalance has not been thoroughly discussed.

## Related Work & Insights

- **Moderate\_DS (Xia et al., 2023)**: Selecting samples with "moderate" distance to the class center serves as the inspiration for the RAM component in this study.
- **TDDS (Zhang et al., 2024)**: Requiring 90 epochs to extract training dynamics highlights the high computational cost of traditional IEs.
- **Swayamdipta et al., 2020**: The data map concept—"easy instances/samples facilitate optimization"—inspired the adaptive weighting strategy that transitions from selecting easy to hard samples.
- **DeepCore (Guo et al., 2022)**: A unified experimental framework for coreset selection; baseline methods in this study are implemented based on it.
- **DINOv2 + CLIP Complementarity**: DINOv2 excels at visual representation, while CLIP excels at semantic alignment, yielding highly complementary feature spaces.

## Rating

| Dimension | Rating | Comments |
|---|---|---|
| Novelty | 7/10 | Integrating multiple FMs for subset selection is a novel direction, though ranking means and pseudo-label accuracy themselves are relatively straightforward. |
| Technical Depth | 7/10 | The method is simple yet effective, and the mathematical derivations are clear. However, theoretical analysis is limited, lacking an in-depth explanation of why FMs perform better specifically on fine-grained data. |
| Experimental Thoroughness | 9/10 | Very systematic, covering single-model studies, multi-model experiments, ablations, parameter sensitivity, and fusion strategy comparisons. |
| Writing Quality | 8/10 | Clear structure with a cohesive narrative flow spanning observation $\rightarrow$ motivation $\rightarrow$ methodology. |
| Value | 7/10 | Simple and easy to implement, though practical applicability is confined to fine-grained classification. |
| **Overall** | **7.5/10** | A solid empirical work with a clean, effective methodology and comprehensive experiments. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] A Bayesian Model Selection Criterion for Selecting Pretraining Checkpoints](a_bayesian_model_selection_criterion_for_selecting_pretraining_checkpoints.md)
- [\[ICML 2025\] Griffin: Towards a Graph-Centric Relational Database Foundation Model](griffin_towards_a_graph-centric_relational_database_foundation_model.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](../../ICML2026/self_supervised/infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[ICML 2025\] What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models](what_has_a_foundation_model_found_using_inductive_bias_to_probe_for_world_models.md)
- [\[CVPR 2026\] Nonparametric Deep Fine-grained Clustering with Low-Rank Guided Vision-Language Model](../../CVPR2026/self_supervised/nonparametric_deep_fine-grained_clustering_with_low-rank_guided_vision-language_.md)

</div>

<!-- RELATED:END -->
