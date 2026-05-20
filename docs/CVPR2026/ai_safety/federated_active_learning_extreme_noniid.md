---
title: >-
  [Paper Note] Federated Active Learning Under Extreme Non-IID and Global Class Imbalance
description: >-
  [CVPR 2026][AI Safety][Federated Learning] This paper systematically analyzes the impact of global class imbalance and client heterogeneity on query model selection in federated active learning (FAL)…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "Federated Learning"
  - "Active Learning"
  - "non-IID"
  - "Class Imbalance"
  - "Query Selection"
  - "Class-Fair Sampling"
  - "Prototype-Guided"
date: 2026-05-08
content_hash: f70f321c0e1b7f53
---

# Federated Active Learning Under Extreme Non-IID and Global Class Imbalance

**Conference**: CVPR 2026
**arXiv**: [2603.10341](https://arxiv.org/abs/2603.10341)  
**Code**: [GitHub](https://github.com/chenchenzong/FairFAL)  
**Area**: AI Security
**Keywords**: Federated Learning, Active Learning, non-IID, Class Imbalance, Query Selection, Class-Fair Sampling, Prototype-Guided

## TL;DR

This paper systematically analyzes the impact of global class imbalance and client heterogeneity on query model selection in federated active learning (FAL), derives three core Observations, and proposes FairFAL—a class-fair FAL framework featuring adaptive query model selection, prototype-guided pseudo-labeling, and two-stage uncertainty-diversity balanced sampling—consistently outperforming all baselines across five benchmark datasets.

## Background & Motivation

- **Background**: Federated Learning (FL) enables collaborative training without sharing raw data, while Active Learning (AL) reduces annotation cost through selective labeling. FAL combines both paradigms—decentralized clients collaboratively identify the most informative unlabeled samples under privacy constraints. This is particularly relevant in domains such as medical imaging and autonomous driving, where annotation is expensive and data privacy is critical.
- **Limitations of Prior Work**: Existing FAL research exhibits three blind spots: (1) client heterogeneity is treated purely as a data partitioning issue, with an implicit assumption of roughly balanced global class distributions; (2) there is no systematic criterion for selecting between the two naturally available query models in FAL—the globally aggregated model versus the locally trained model; (3) under the compound condition of global long-tail distribution and extreme non-IID, existing sampling strategies systematically favor head classes. Recent methods such as LoGo, KAFAL, and IFAL account for non-IID but do not explicitly address global class imbalance. IFAL even falls below random sampling on CIFAR-100 with $\rho=20$ (26.82 vs. 27.44), underscoring the severity of the problem.
- **Key Challenge**: In FAL, the global model benefits from superior feature representations (cross-client aggregation) but loses discriminability in uncertainty sampling due to over-smoothed predictions; the local model is more sensitive to client-specific decision boundaries but reflects long-tail skewness when the global distribution is highly imbalanced. The relative advantage of each model depends on the interplay between global imbalance ratio $\rho$ and client heterogeneity $\alpha$, precluding any fixed choice.
- **Goal**: To design a FAL framework that adaptively selects the query model and explicitly promotes class-fair sampling under the challenging setting of extreme non-IID ($\alpha=0.1$) and global long-tail distribution ($\rho=20$).
- **Key Insight**: The paper begins with a systematic empirical analysis—comparing global and local model sampling behaviors under four $(\alpha, \rho)$ combinations on CIFAR-10 (4 combinations × 2 strategies × 5 seeds), using a triple statistical analysis of AULC, Wilcoxon test, and Hodges-Lehmann effect size—and derives three Observations that motivate each framework component.
- **Core Idea**: Class-balanced sampling capability—especially acquisition of minority classes—is the most consistent predictor of FAL performance, surpassing uncertainty or diversity alone.

## Method

### Overall Architecture

FairFAL augments the standard FedAvg framework with three cooperative components, executed sequentially during the query phase of each FAL round: (1) **Adaptive Model Selection**—determines whether each client uses the global or local model as the query selector, based on the global imbalance coefficient $\bar{\gamma}$ and the local-global divergence $d_k$; (2) **Prototype-Guided Pseudo-Labeling**—computes class prototypes from global model features and assigns pseudo-labels to unlabeled samples to form class-level candidate pools; (3) **Two-Stage Uncertainty-Diversity Balanced Sampling**—selects top uncertain samples per class in the first stage, then applies $k$-center diversity filtering in gradient embedding space. Standard federated training updates follow each AL query round.

### Key Designs

1. **[Adaptive Model Selection]**:

    - **Function**: Automatically determines per client whether the global or local model serves as the query selector, without additional privacy leakage (only a scalar $\gamma_k$ is uploaded).
    - **Mechanism**: Two key quantities are estimated—the **global class imbalance ratio $\gamma_k$** and the **local-global distribution divergence $d_k$**. For $\gamma_k$: in the first AL round (where labeled data approximates IID via random sampling), a class-balanced subset $\mathcal{B}^{(k)}$ is constructed per client (by upsampling to equalize class counts), the global model is used to obtain softmax priors $\hat{\pi}_g^{(k)}$, and $\gamma_k = \min_c \hat{\pi}_{g,c} / \max_c \hat{\pi}_{g,c} \in (0,1]$ is computed, where values closer to 1 indicate greater balance. The server averages all $\gamma_k$ to obtain $\bar{\gamma}$ (only scalars are transmitted, ensuring privacy), fixed after the first round. For $d_k$: the global and local models are each applied to the same balanced subset, and normalized symmetric divergence is computed as $d_k = \frac{1}{C}\sum_c \frac{|\hat{\pi}_{g,c} - \hat{\pi}_{\ell,c}^{(k)}|}{\hat{\pi}_{g,c} + \hat{\pi}_{\ell,c}^{(k)}} \in [0,1]$, updated each round. The model selection score $s_k = 1 - \frac{1}{2}(d_k + \bar{\gamma})$: the global model is used when $s_k > \delta = 0.75$ (severe global imbalance + homogeneous clients); otherwise the local model is used.
    - **Design Motivation**: Driven by Observation 1—under uncertainty sampling, local models generally outperform global models (aggregation of locally diverse queries naturally yields a globally balanced query set), except when the global distribution is severely imbalanced and clients are homogeneous (local queries then reflect global long-tail skewness). Ablations confirm that adaptive selection $\mathcal{M}^{(k)}$ consistently outperforms its counterpart $\tilde{\mathcal{M}}^{(k)}$ (59.33 vs. 58.49), and is insensitive to threshold $\delta$ (variation < 0.5% for $\delta \in [0.65, 0.85]$).

2. **[Prototype-Guided Pseudo-Labeling]**:

    - **Function**: Assigns class-aware pseudo-labels to unlabeled samples, partitioning the unlabeled pool into class-level subsets $\tilde{\mathcal{D}}_{U,c}^{(k)}$ as the basis for subsequent class-balanced sampling.
    - **Mechanism**: The penultimate-layer feature extractor $\phi^g(\cdot)$ of the global model is used to extract L2-normalized features $\mathbf{z}_i^{(k)}$. For each class $c$, a prototype $\boldsymbol{\mu}_c^{(k)}$ is computed as the mean feature of labeled samples in that class. Pseudo-labels are then assigned to unlabeled samples via cosine similarity: $\hat{y}^{(k)}(x) = \arg\max_c \langle \mathbf{z}^{(k)}(x), \boldsymbol{\mu}_c^{(k)} \rangle$. This bypasses classifier decision boundary shift under long-tail conditions—class assignment in feature space is more robust than using logits directly.
    - **Design Motivation**: Jointly motivated by Observation 2 (class-balanced sampling strongly aligns with performance) and Observation 3 (global model features are of superior quality). Ablations confirm: global prototypes outperform local prototypes (59.95 vs. 59.14), as the global model provides cleaner class separation in feature space.

3. **[Uncertainty-Diversity Balanced Sampling]**:

    - **Function**: Ensures informativeness and diversity simultaneously under a class-balanced constraint, avoiding redundant clustering of high-uncertainty samples.
    - **Mechanism**: **Stage 1 (Intra-class Uncertainty Filtering)**—for each class's candidate subset, entropy is computed using the adaptively selected query model, and the top-$\kappa \cdot b_c$ most uncertain samples form an overcomplete candidate pool $\mathcal{H}_c^{(k)}$ ($\kappa = 4$, $b_c$ is the uniform per-class budget). **Stage 2 (Gradient Embedding $k$-center)**—gradient embeddings $\mathbf{g}^{(k)}(x) = \psi(x; \phi^g, f^g)$ (gradients of the classification loss with respect to global classifier parameters) are computed for each candidate, and greedy $k$-center selection of $b_c$ samples is performed with labeled samples as anchors, minimizing the maximum coverage radius.
    - **Design Motivation**: Directly selecting the most uncertain samples leads to severe redundancy—high-uncertainty samples tend to cluster in narrow regions of feature space. The two-stage design first ensures informativeness (excluding "safe" samples) and then ensures diversity (covering feature space). $\kappa = 4$ provides sufficient candidate diversity; ablations show negligible sensitivity to $\kappa$ ($\kappa = 2/3/4$ differ by < 0.3%).

### Loss & Training

Standard FedAvg: 100 communication rounds, 5 local epochs per client. 4-layer CNN backbone, SGD (momentum=0.9, weight decay=1e-5, batch size=64), learning rate 0.01 with 10× decay after 75 rounds. FAL proceeds for 9 query rounds: 5% random labeling in the first round, 5% queried per subsequent round. 10 clients, Dirichlet partitioning ($\alpha = 0.1$ / $\alpha = 100$), global imbalance ratio $\rho = 20$. All experiments averaged over 5 random seeds.

## Key Experimental Results

### Main Results

**Natural Image Datasets ($\alpha = 0.1$, $\rho = 20$, final-round test accuracy %)**:

| Method | Type | FMNIST | CIFAR-10 | CIFAR-100 |
|--------|------|--------|----------|-----------|
| Random | Baseline | 85.60 | 55.70 | 27.44 |
| Entropy | Uncertainty | 86.28 | 57.18 | 27.44 |
| BADGE | Hybrid | 86.65 | 58.20 | 27.39 |
| Coreset | Diversity | 86.15 | 57.63 | 27.81 |
| KAFAL | FAL | 87.05 | 60.01 | 27.84 |
| LoGo | FAL | 86.98 | 59.68 | 27.95 |
| IFAL | FAL | 86.80 | 57.51 | 26.82 |
| **FairFAL** | **FAL** | **87.37** | **60.44** | **29.20** |

**Medical Datasets ($\alpha = 0.1$, natural long-tail distribution)**:

| Method | OctMNIST | DermaMNIST |
|--------|----------|------------|
| Random | 68.30 | 72.32 |
| KAFAL | 70.40 | 73.27 |
| LoGo | 70.00 | 73.62 |
| IFAL | 68.40 | 72.97 |
| **FairFAL** | **72.80** | **73.77** |

### Ablation Study

**Incremental Component Addition (CIFAR-10, accuracy %)**:

| Configuration | ($\alpha$=0.1, $\rho$=20) | ($\alpha$=100, $\rho$=20) |
|---------------|---------------------------|--------------------------|
| Model selection $\mathcal{M}^{(k)}$ only | 59.33 | 63.65 |
| Opponent model $\tilde{\mathcal{M}}^{(k)}$ | 58.49 | 61.89 |
| + Local prototypes | 59.14 | 63.39 |
| + Global prototypes | 59.95 | 64.02 |
| + Two-stage $\kappa=2$ | 60.61 | 64.60 |
| + Two-stage $\kappa=3$ | 60.38 | 64.58 |
| **+ Two-stage $\kappa=4$ (Final)** | **60.44** | **64.57** |
| + Two-stage $\kappa=5$ | 60.28 | 64.17 |

**Statistical Analysis of Global vs. Local Query Selector (Hodges-Lehmann Effect Size, percentage points pp)**:

| ($\alpha$, $\rho$) | Entropy Winner | Entropy HL (pp) | Coreset Winner | Coreset HL (pp) |
|--------------------|----------------|-----------------|----------------|-----------------|
| (0.1, 1) | Local | 201 | Global | 28 |
| (100, 1) | Local | 50 | Global | 21 |
| (0.1, 20) | Local | 66 | Global | 50 |
| (100, 20) | **Global** | **106** | **Global** | **92** |

### Key Findings

- **Adaptive model selection is effective**: $\mathcal{M}^{(k)}$ consistently outperforms its counterpart (+0.84 under high heterogeneity, +1.76 under near-homogeneous settings).
- **Global prototypes > local prototypes** (59.95 vs. 59.14): the global model yields cleaner class separation in feature space.
- **Method is insensitive to $\kappa$** ($\kappa=2/3/4$ differ by < 0.3%) and to $\delta$ (< 0.5% variation over $[0.65, 0.85]$).
- **FL-framework agnostic**: FairFAL consistently achieves the best performance under FedProx and SCAFFOLD as well.
- **Advantage grows with task difficulty** (FMNIST +0.32 → CIFAR-100 +1.25).
- **IFAL falls below Random on CIFAR-100** (26.82 vs. 27.44)—methods lacking class-balancing mechanisms completely fail under complex long-tail scenarios.
- **Visualization of Observation 2**: The cumulative minority-class sampling ratio of global vs. local models closely tracks final accuracy, with the gap stabilizing after minority-class sampling diverges.

## Highlights & Insights

- The core insight is remarkably clear and generalizable: class-balanced sampling capability is a more consistent predictor of FAL performance than uncertainty or diversity.
- The derivation of the three Observations is instructive: (1) local models generally outperform global models under uncertainty sampling (except when the global distribution is severely imbalanced and clients are homogeneous); (2) class-balanced sampling strongly aligns with final performance (with a clear causal direction); (3) global model features are consistently of higher quality under diversity-based sampling.
- Prototype-guided pseudo-labeling circumvents long-tail classifier decision boundary shift—class assignment in feature space avoids logit bias.
- Privacy-preserving design is carefully considered—clients upload only the scalar $\gamma_k$, and all estimates are computed from local labeled data.
- The statistical analysis methodology (AULC + Wilcoxon test + Hodges-Lehmann effect size) provides rigorous statistical support for the Observations, rather than relying solely on mean comparisons.

## Limitations & Future Work

- Adaptive selection is a hard switch ($s_k > \delta$: global; otherwise: local); soft blending—e.g., weighted fusion of query results from both models—has not been explored.
- Uniform per-class budget allocation ($b_c$ equal across classes) may be suboptimal under extreme long-tail distributions; dynamically allocating larger budgets to tail classes based on class scarcity warrants investigation.
- Validation is limited to image classification with a 4-layer CNN; applicability to more complex tasks (object detection, segmentation) and deeper backbones (ResNet-50, ViT) remains unknown.
- $\bar{\gamma}$ is estimated only in the first round and fixed thereafter, without updating as the global distribution shifts due to active querying in subsequent rounds.
- The method assumes the first-round query is random to approximate IID initialization; in practice, initial labeled data may already be skewed.
- No comparison with generative model-based data augmentation approaches, an alternative strategy for addressing class imbalance.

## Related Work & Insights

- **vs. LoGo (CVPR'23)**: Performs local clustering followed by global uncertainty scoring—a two-stage approach that does not explicitly address class balance; FairFAL explicitly guarantees per-class sampling via prototype pseudo-labeling.
- **vs. KAFAL (ICCV'23)**: Leverages global-local prediction discrepancy to identify "knowledge-inconsistent" samples—however, high discrepancy does not imply minority-class membership; FairFAL adds a class-aware layer.
- **vs. IFAL**: Falling below Random on CIFAR-100+$\rho=20$ provides the most direct evidence of failure without class-balancing mechanisms.
- Prototype learning, imported from few-shot and contrastive learning, is applied here to class-aware sampling in FAL—the FL global model serves as a naturally high-quality feature extractor, even when the classification head is biased.
- The core challenge in transitioning from AL to FAL lies not only in privacy constraints, but in the compound difficulty of "dual model selection + global long-tail distribution."

## Rating

- **Novelty**: ⭐⭐⭐⭐ The framework is designed from systematic empirical analysis, with each component clearly motivated by a corresponding Observation; the methodology is exemplary.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets, eleven baselines, and comprehensive ablations (model selection, prototype quality, $\kappa$, $\delta$, uncertainty measure, FL framework, number of clients).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Observation-driven structure is clear, statistical analysis is rigorous, and every design choice is experimentally justified.
- **Value**: ⭐⭐⭐⭐ Directly actionable for the FAL community; the core insight (class balance > uncertainty/diversity) carries broad significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2026\] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[CVPR 2026\] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning](proxyfl_a_proxy-guided_framework_for_federated_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
