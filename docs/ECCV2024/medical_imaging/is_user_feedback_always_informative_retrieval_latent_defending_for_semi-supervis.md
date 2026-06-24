---
title: >-
  [Paper Note] Is User Feedback Always Informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data
description: >-
  [ECCV 2024][Medical Imaging][Semi-supervised Domain Adaptation] It is discovered that user feedback is not always beneficial in domain adaptation. "Negatively Biased Feedback" (NBF), which biases towards correcting erroneous model predictions, leads to performance degradation in existing semi-supervised domain adaptation methods. To address this, Retrieval Latent Defending (RLD) is proposed to balance supervision signals by introducing pseudo-labeled defending samples to each…
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Semi-supervised Domain Adaptation"
  - "Negatively Biased Feedback"
  - "Source-Free"
  - "User Feedback"
  - "Pseudo-labeling"
date: 2026-05-08
content_hash: 425f9609d0b68b16
---

# Is User Feedback Always Informative? Retrieval Latent Defending for Semi-Supervised Domain Adaptation without Source Data

**Conference**: ECCV 2024  
**arXiv**: [2407.15383](https://arxiv.org/abs/2407.15383)  
**Code**: [Yes](https://github.com/junha1125/RLD-SemiSDA)  
**Area**: Medical Images / Domain Adaptation  
**Keywords**: Semi-supervised Domain Adaptation, Negatively Biased Feedback, Source-Free, User Feedback, Pseudo-labeling

## TL;DR

It is discovered that user feedback is not always beneficial in domain adaptation. "Negatively Biased Feedback" (NBF), which biases towards correcting erroneous model predictions, leads to performance degradation in existing semi-supervised domain adaptation methods. To address this, Retrieval Latent Defending (RLD) is proposed to balance supervision signals by introducing pseudo-labeled defending samples to each mini-batch.

## Background & Motivation

Deep learning models often suffer performance degradation in the deployment domain (target domain) due to domain shift. Semi-Supervised Domain Adaptation (SemiSDA) utilizes a small amount of labeled target data and a large amount of unlabeled target data for adaptation. However, existing work carries a **critically overlooked issue**:

**Feedback Bias Assumption**: All existing SemiSDA methods assume that labeled data is a **random subset** of the target domain. However, in real-world ML products, user feedback originates from **interactive error-correction**—users are more inclined to provide feedback when the model makes mistakes. For instance, radiologists are more likely to record X-ray images misdiagnosed by the model because diagnostic accuracy directly affects patient survival.

**Negatively Biased Feedback (NBF)**: This phenomenon is supported by cognitive psychology—humans react more strongly to negative events (i.e., erroneous model predictions). The resulting labeled dataset is **biased** within each class distribution—concentrating in hard sample regions near the model's decision boundaries rather than being uniformly distributed.

**Unexpected Impact of NBF**: Intuitively, feedback correcting more errors should lead to better adaptation. However, experiments reveal that under NBF conditions, using SOTA SemiSDA methods like AdaMatch actually **decreases** performance (e.g., from 67.6% to 64.5% on DomainNet-126). The reason is that during the adaptation process of SemiSDA methods, the decision boundaries are **dominated** by the biased distribution of labeled data, causing the decision boundaries to shift away from the true class clusters.

**Source-free constraint**: Considering data privacy and edge device limitations, source data is unavailable, which further intensifies the difficulty of adaptation.

This is the **first** study to reveal and analyze the impact of user feedback bias on domain adaptation.

## Method

### Overall Architecture

Retrieval Latent Defending (RLD) is a **plug-and-play** method that can be combined with any SemiSDA baseline. The core idea is to introduce additional "latent defending samples" retrieved from a candidate bank into each training mini-batch, alongside the biased labeled data and unlabeled data, to balance the distribution of supervising signals.

### Key Designs

1. **Modeling and Validation of Negatively Biased Feedback (NBF)**:

    - Simulation experiments on Blob synthetic datasets visualize the impact of NBF: Random Feedback (RF) is uniformly distributed within class clusters, whereas NBF concentrates near the decision boundary.
    - Pseudo-labeling achieves 91.7% accuracy under RF, but only 88.1% under NBF.
    - Cause Analysis: The distribution of labeled data significantly contributes to the post-adaptation decision boundaries (indicated by red arrows in figures); the biased distribution leads to sub-optimal decision boundaries.
    - Design Motivation: This uncovers the implicit assumption of existing SemiSDA methods being **sensitive to the spatial distribution of labeled data**.

2. **Candidate Bank Generation**: Performed once before the start of each epoch.

    - Freeze the current model and generate pseudo-labels for all unlabeled data $X_t^{ulb}$.
    - $\hat{y}_{ulb}^n = \arg\max_c [f_\theta(x_{ulb}^n)]_c$
    - Retain only the **top-p%** samples with the highest softmax probabilities within each class (default is $p=40\%$).
    - The filtering step avoids introducing inaccurate pseudo-labels.
    - As adaptation progresses, model improvement $\rightarrow$ pseudo-label quality increases $\rightarrow$ target candidate bank is progressively refined.
    - Design Motivation: Utilize high-confidence pseudo-labeled samples as proxies for the true uniform distribution to provide a balancing force against biased labeled data.

3. **Defending Sample Selection**: Performed in each iteration.

    - For each labeled sample $(x_{lb}^b, y_{lb}^b)$ in the mini-batch, randomly select $k$ pseudo-labeled samples of the same class from the candidate bank (default $k=3$).
    - Selection Strategy: Class-aware random selection (validated as optimal in ablation studies).
    - Effect: Selected defending samples "surround" the biased labeled data in the feature space, forming a more balanced class representation.
    - Design Motivation: Prevent supervision signals from overly relying on the spatial locations of biased labeled samples.

### Loss & Training

The total loss consists of three parts:

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{sup} + \mathcal{L}_{unsup}}_{\text{baseline}} + \underbrace{\frac{1}{k \cdot B} \sum_{b=1}^{k \cdot B} \mathcal{H}(\hat{y}_{LD}^b, f_\theta(x_{LD}^b))}_{\text{retrieval latent defending}}$$

- $\mathcal{L}_{sup}$: Cross-entropy loss for labeled data.
- $\mathcal{L}_{unsup}$: Consistency regularization loss for unlabeled data (specific form depends on the baseline).
- RLD Loss: Cross-entropy loss on the pseudo-labels of defending samples.
- Hyperparameters: $k=3$ (3 defending samples paired per labeled sample), $p=0.4$ (top-40% filtering rate), uniformly set across all experiments.
- The ratio of unlabeled samples in the mini-batch is reduced (from 1:7 to 1:4) to assign more weight to defending and labeled samples.

## Key Experimental Results

### Main Results — DomainNet-126 (ResNet-50, 378 feedback)

| Method | RF | NBF | NBF w/ RLD | Gain |
|------|-----|-----|-----------|------|
| AdaMatch | 67.6 | 64.5 | **72.0** | +7.5 |
| FixMatch | 67.6 | 63.4 | **73.2** | +9.8 |
| FreeMatch | 73.8 | 72.0 | **74.8** | +2.8 |
| FlexMatch | 73.3 | 71.4 | **74.7** | +3.3 |
| CDAC | 68.3 | 64.6 | **73.2** | +8.6 |

NBF consistently causes a 1-4% performance degradation, which RLD not only recovers but **surpasses RF** performance.

### Ablation Study — Defending Sample Selection Strategy (FreeMatch, ResNet-50)

| Selection Strategy | Accuracy | Baseline (w/o RLD) |
|---------|-------|------------------|
| Random (class-agnostic) | 74.1 | 72.0 |
| **Random (class-aware)** | **74.8** | 72.0 |
| K-means cluster center | 74.6 | 72.0 |
| Farthest cosine distance | 74.0 | 72.0 |

### Medical Images — MIMIC-CXR-V2 (PA→AP, DenseNet-121)

| Method | RF (AUROC) | NBF | NBF w/ RLD | NBF-CE | NBF-CE w/ RLD |
|------|-----------|-----|-----------|--------|---------------|
| Source model | .7738 | - | - | - | - |
| Pseudo-Label | .7850 | .7691 | **.7884** | .7639 | **.7875** |

NBF-CE (where clinicians are more likely to provide feedback on high-confidence incorrect predictions) degrades performance more severely than NBF (.7691 $\rightarrow$ .7639), but RLD effectively mitigates both.

### Key Findings

- **Universality of NBF**: Across the three tasks of image classification, semantic segmentation, and medical image diagnosis, NBF consistently causes performance degradation in existing SemiSDA methods.
- **Generality of RLD**: RLD is effective when combined with 7 different SemiSDA/SemiSL baselines without requiring modifications to the core strategies of the baselines.
- **Greater Advantage with Scarce Feedback**: With only 1 feedback per class, RLD yields an improvement of up to +4.9% (vs. +1.2% with 15 feedback per class), demonstrating greater value in real-world applications where feedback is scarce.
- **Negative vs. Positive Feedback**: Interestingly, when combined with RLD, NBF performs even better than Pure Positive Feedback (PPF) — because NBF contains novel knowledge about model deficiencies.
- **Mini-batch Ratio**: Reducing the proportion of unlabeled samples (1:4 outperforms 1:7) is more beneficial as it prioritizes reliable information.

## Highlights & Insights

1. **Problem Discovery Outweighs Methodology**: NBF is a practical problem overlooked by the entire SemiSDA field. The interdisciplinary analysis of cognitive psychology and ML is highly convincing.
2. **Simplicity and Effectiveness**: RLD only requires adding high-confidence pseudo-labeled samples in the mini-batch, without modifying any baseline algorithm.
3. **Practical Significance in Medical Scenarios**: The feedback patterns of radiologists conform well to the NBF assumption. Naive adaptation may lead to model performance degradation, which directly affects patient safety.
4. **NBF-CE Scenario**: Further analysis on how physicians are more likely to provide feedback on high-confidence errors adds depth and practical relevance to the work.

## Limitations & Future Work

1. The candidate bank is updated once per epoch, increasing computational overhead as the dataset scale grows.
2. The quality of pseudo-labels relies on the current model, which can be unreliable in the early stages of adaptation.
3. Only feedback in the form of annotations is considered; in practice, other feedback formats like thumbs up/down or ratings also exist.
4. NBF in the experiments is simulated (by randomly selecting incorrect predictions) and has not been validated in real-world user interaction environments.
5. The number of defending samples $k$ and the filtering rate $p$ are uniformly set for all experiments, lacking an adaptive adjustment mechanism.

## Related Work & Insights

- **Difference from Active Domain Adaptation (ActiveDA)**: ActiveDA involves the machine selecting the most informative samples to request annotations, whereas this work focuses on biased feedback voluntarily provided by users.
- **Failure of Pseudo-Labeling Methods**: Pseudo-labeling methods (FixMatch, FreeMatch) fail particularly severely under NBF, as biased labeled data shifts the pseudo-label distribution.
- **Connections to Curriculum Learning**: The philosophy of curriculum learning and adaptive thresholds aligns with the strategy of RLD to reduce unlabeled samples and prioritize reliable information.
- **Future Directions**: Future research could incorporate uncertainty estimation to adaptively adjust the selection and weights of defending samples.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — For first uncovering and systematically analyzing the NBF phenomenon, presenting a highly insightful and practically valuable problem definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers three task domains, seven baselines, two architectures, varying amounts of feedback, alongside comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — The visualization analysis of simulation experiments is intuitive and clear, with a progressive articulation of the problem.
- **Value**: ⭐⭐⭐⭐⭐ — Directly addresses real-world challenges in ML product deployment, holding particularly high value in medical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Semi-supervised Deep Transfer for Regression without Domain Alignment](../../ICCV2025/medical_imaging/semi-supervised_deep_transfer_for_regression_without_domain_alignment.md)
- [\[ECCV 2024\] Alternate Diverse Teaching for Semi-supervised Medical Image Segmentation](alternate_diverse_teaching_for_semi-supervised_medical_image_segmentation.md)
- [\[CVPR 2025\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](../../CVPR2025/medical_imaging/semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[CVPR 2026\] Tell2Adapt: A Unified Framework for Source Free Unsupervised Domain Adaptation via Vision Foundation Model](../../CVPR2026/medical_imaging/tell2adapt_a_unified_framework_for_source_free_unsupervised_domain_adaptation_vi.md)
- [\[ECCV 2024\] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)

</div>

<!-- RELATED:END -->
