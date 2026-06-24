---
title: >-
  [Paper Note] A Unified Framework for Heterogeneous Semi-supervised Learning
description: >-
  [CVPR 2025][LLM Pretraining][Heterogeneous Semi-supervised Learning] This paper proposes a new problem setting termed Heterogeneous Semi-Supervised Learning (HSSL), where labeled and unlabeled data originate from domains with different distributions, and the goal is to train a model that generalizes well to both domains. By expanding the C-class problem into a 2C-class classification task (where the same semantic class in different domains is treated as distinct classes)…
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "Heterogeneous Semi-supervised Learning"
  - "Cross-Domain Pseudo-Labeling"
  - "Prototype Alignment"
  - "Progressive Mixup"
  - "2C-Class Classification"
date: 2026-05-08
content_hash: cc308958be492447
---

# A Unified Framework for Heterogeneous Semi-supervised Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.00286](https://arxiv.org/abs/2503.00286)  
**Code**: None  
**Area**: Semi-supervised Learning / Domain Adaptation  
**Keywords**: Heterogeneous Semi-supervised Learning, Cross-Domain Pseudo-Labeling, Prototype Alignment, Progressive Mixup, 2C-Class Classification

## TL;DR

This paper proposes a new problem setting termed Heterogeneous Semi-Supervised Learning (HSSL), where labeled and unlabeled data originate from domains with different distributions, and the goal is to train a model that generalizes well to both domains. By expanding the C-class problem into a 2C-class classification task (where the same semantic class in different domains is treated as distinct classes), this work provides a unified solution integrating Weighted Moving Average (WMA) pseudo-labeling, cross-domain prototype alignment, and progressive cross-domain Mixup.

## Background & Motivation

**Background**: Traditional semi-supervised learning (SSL) assumes that labeled and unlabeled data share the same distribution, while unsupervised domain adaptation (UDA) focuses solely on performance in the target domain. Neither is applicable to scenarios where labeled and unlabeled data come from different domains and generalization is required across both domains.

**Limitations of Prior Work**: In practical scenarios such as medical imaging and aerial photography, labeled data are sourced from high-end equipment/research hospitals, while unlabeled data are obtained from various different devices in rural clinics—leading to significant distribution differences. Standard SSL methods fail when ignoring distribution discrepancies, whereas UDA sacrifices source domain performance by only optimizing for the target domain.

**Key Challenge**: HSSL concurrently faces three major challenges: (1) different feature distributions across domains $p_L(x|y) \neq p_U(x|y)$; (2) different label distributions across domains $p_L(y) \neq p_U(y)$; and (3) testing sets comprising a mixture of both domains.

**Core Idea**: The C semantic classes are expanded into 2C fine-grained classes (the first C corresponding to the labeled domain, and the remaining C corresponding to the unlabeled domain). A unified 2C-class classifier is leveraged to naturally handle domain differences, supported by three components to facilitate cross-domain knowledge transfer.

## Method

### Overall Architecture

Pre-train a C-class model $\rightarrow$ generate initial pseudo-labels for unlabeled data $\rightarrow$ expand to a 2C-class model $\rightarrow$ joint training: supervised loss (labeled set) + WMA pseudo-label loss (unlabeled set) + cross-domain prototype alignment loss + progressive Mixup loss.

### Key Designs

1. **Weighted Moving Average (WMA) Pseudo-Labeling**:

    - **Function**: Adaptive updating of pseudo-labels for unlabeled data.
    - **Mechanism**: $\hat{y}_i^t = \beta \hat{y}_i^{t-1} + (1-\beta) h(f(x_i^u))$, where $\beta=0.8$. Only pseudo-labels with confidence exceeding the threshold $\epsilon$ participate in training.
    - **Design Motivation**: Due to the cross-domain gap, initial pseudo-labels are highly noisy. WMA provides smooth and adaptive label updates to prevent oscillation.

2. **Cross-Domain Semantic Prototype Alignment**:

    - **Function**: Aligning prototype vectors of the same semantic class across the two domains.
    - **Mechanism**: Compute the prototype $p_k$ of the $k$-th class in the labeled domain and $p_{C+k}$ of the $k$-th class in the unlabeled domain. A symmetric contrastive loss is employed to pull corresponding class pairs closer and push non-corresponding class pairs apart:
      $$\mathcal{L}_{pa} = -\sum_{k=1}^{C} [\log \frac{\exp(\cos(p_k, p_{C+k})/\tau)}{\sum_{k'\neq k} \exp(\cos(p_k, p_{C+k'})/\tau)} + ...]$$
    - **Design Motivation**: Facilitating knowledge transfer by leveraging the shared semantic relations between the two domains.

3. **Progressive Cross-Domain Mixup**:

    - **Function**: Generate cross-domain synthetic samples to bridge the two domains.
    - **Mechanism**: $x^m = \lambda x^u + (1-\lambda) x^l$. The key lies in progressive scheduling—$\lambda \sim \psi(t) \times \text{Beta}(\alpha, \alpha)$, where $\psi(t) = 0.5 + t/(2T)$. In the early stages of training, $\lambda \in [0, 0.5)$ biases toward the labeled domain, and later gradually expands to $[0, 1]$ for equal fusion.
    - **Design Motivation**: In the early stages of training, pseudo-labels are unreliable, necessitating a bias toward the labeled domain. As training progresses and pseudo-label quality improves, the weight of the unlabeled domain is gradually increased.

### Loss & Training

$\mathcal{L}_{total} = \mathcal{L}_{cl}^L + \lambda_{pl} \mathcal{L}_{pl}^U + \lambda_{pa} \mathcal{L}_{pa} + \lambda_{Mixup} \mathcal{L}_{Mixup}$

## Key Experimental Results

### Main Results: Office-31 (ResNet-50)

| Method | W/A | A/W | D/A | D/W | W/D | Average |
|------|-----|-----|-----|-----|-----|------|
| Supervised | 68.6 | 82.8 | 35.5 | 96.9 | 98.2 | 77.8 |
| FixMatch (SSL) | 69.1 | 83.4 | 53.7 | 98.1 | 98.2 | 81.5 |
| SimMatch (SSL) | 71.1 | 84.1 | 68.6 | 96.8 | 98.8 | 84.3 |
| MCC+Sup (UDA) | 71.5 | 88.8 | 67.6 | 81.7 | 99.5 | 83.0 |
| BiAdopt | 70.2 | 85.0 | 67.1 | 94.2 | 98.5 | 82.0 |
| **Uni-HSSL** | **73.1** | **90.2** | **72.1** | **100** | **100** | **87.5** |

### Ablation Study (Average Accuracy on Office-Home, Appendix)

Ablation experiments validate the contributions of the three components: WMA pseudo-labeling, prototype alignment, and progressive Mixup.

### Key Findings

- Uni-HSSL achieves a mean accuracy of 87.5% on Office-31, outperforming the strongest SSL method SimMatch by 3.2% and the strongest UDA method MCC+Sup by 4.5%.
- The improvement is most significant on the most challenging D→A task (DSLR→Amazon, which exhibits the largest domain gap): 72.1% vs. SimMatch 68.6%.
- The 2C-class classification strategy is effective: it naturally distinguishes sample origins without requiring an extra domain classifier.
- Uni-HSSL also achieves leading performance on VisDA (large-scale synthetic→real) and ISIC-2019 (medical images).
- UDA methods (CDAN+Sup) perform worse than the Supervised baseline on certain tasks, since optimizing solely for the target domain in UDA compromises source domain performance.

## Highlights & Insights

- **Value of the Problem Definition**: HSSL bridges the SSL and UDA communities and aligns more closely with real-world scenarios, where diverse data sources are the norm.
- **Elegance of the 2C-Class Expansion**: Simple yet effective—treating "the same concept in different domains" as distinct classes allows the model to naturally learn intra-domain and inter-domain feature differences, bypassing the need for complex domain classifiers.
- **Curriculum Learning Philosophy of Progressive Mixup**: Reflects the curriculum learning principle of "easy-to-hard"—relying initially on labeled data and progressively placing trust in pseudo-labels.

## Limitations & Future Work

- It assumes that both domains share the exact same C classes, making it unable to handle scenarios where the class sets do not completely overlap.
- The 2C-class classification may lead to classifier capacity bottlenecks when the number of classes is large.
- Prototype alignment relies on reliable pseudo-labels, meaning prototype quality may be suboptimal in the early stages of training.
- It is only validated on image classification and has not been extended to dense prediction tasks such as object detection or segmentation.

## Related Work & Insights

- **vs. BiAdopt**: BiAdopt also addresses heterogeneous SSL but employs independent components to handle different domains separately, whereas Uni-HSSL utilizes a unified 2C-class framework for end-to-end training.
- **vs. FixMatch/FlexMatch**: Standard SSL methods assume identical distributions and yield limited performance when the domain gap is large (e.g., only 53.7% on D→A). The cross-domain components of Uni-HSSL significantly alleviate this limitation.
- **vs. UDA**: UDA only optimizes the target domain, which leads to degraded source domain performance under the HSSL setting.

## Rating

- Novelty: ⭐⭐⭐⭐ The problem definition is valuable and the 2C-class expansion is simple and effective. However, individual components (pseudo-labeling, prototype alignment, Mixup) are combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on 4 datasets with comparisons against multiple baselines, but lacks detailed ablation studies and large-scale data validation.
- Writing Quality: ⭐⭐⭐⭐ The problem formulation is clear and the method description is comprehensive, though it contains numerous mathematical notations.
- Value: ⭐⭐⭐⭐ Provides a unified solution for SSL in heterogeneous data scenarios, offering broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Density Ratio Estimation-based Bayesian Optimization with Semi-Supervised Learning](../../ICML2025/llm_pretraining/density_ratio_estimation-based_bayesian_optimization_with_semi-supervised_learni.md)
- [\[NeurIPS 2025\] Mouse-Guided Gaze: Semi-Supervised Learning of Intention-Aware Representations for Reading Detection](../../NeurIPS2025/llm_pretraining/mouse-guided_gaze_semi-supervised_learning_of_intention-aware_representations_fo.md)
- [\[ICML 2025\] A Square Peg in a Square Hole: Meta-Expert for Long-Tailed Semi-Supervised Learning](../../ICML2025/llm_pretraining/a_square_peg_in_a_square_hole_meta-expert_for_long-tailed_semi-supervised_learni.md)
- [\[ACL 2025\] An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](../../ACL2025/llm_pretraining/dual_stage_curriculum_learning_sequence_labeling.md)
- [\[NeurIPS 2025\] Heterogeneous Adversarial Play in Interactive Environments](../../NeurIPS2025/llm_pretraining/heterogeneous_adversarial_play_in_interactive_environments.md)

</div>

<!-- RELATED:END -->
