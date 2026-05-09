---
title: >-
  [Paper Note] Federated Active Learning Under Extreme Non-IID and Global Class Imbalance
description: >-
  [CVPR 2026][AI Safety][Federated Active Learning] This paper systematically investigates the query model selection problem in federated active learning (FAL), identifies class-balanced sampling as the key performance factor, and proposes FairFAL — a framework achieving fair and efficient FAL via adaptive model selection, prototype-guided pseudo-labeling, and uncertainty-diversity balanced sampling.
tags:
  - CVPR 2026
  - AI Safety
  - Federated Active Learning
  - Non-IID
  - Class Imbalance
  - active learning
  - Long-Tailed Distribution
date: 2026-05-08
content_hash: 3b1d2c57cf4995f7
---

# Federated Active Learning Under Extreme Non-IID and Global Class Imbalance

**Conference**: CVPR 2026
**arXiv**: [2603.10341](https://arxiv.org/abs/2603.10341)
**Code**: [chenchenzong/FairFAL](https://github.com/chenchenzong/FairFAL)
**Area**: AI Safety / Federated Learning
**Keywords**: Federated Active Learning, Non-IID, Class Imbalance, active learning, Long-Tailed Distribution

## TL;DR

This paper systematically investigates the query model selection problem in federated active learning (FAL), identifies class-balanced sampling as the key performance factor, and proposes FairFAL — a framework achieving fair and efficient FAL via adaptive model selection, prototype-guided pseudo-labeling, and uncertainty-diversity balanced sampling.

## Background & Motivation

Federated active learning (FAL) combines the privacy guarantees of federated learning with the label efficiency of active learning, yet faces two severely underexplored challenges in realistic deployments:

**Global class imbalance**: Real-world federated systems typically exhibit long-tailed global distributions, where rare but critical classes appear sparsely across clients.

**Extreme client heterogeneity**: Data distributions vary drastically across clients (extreme Non-IID).

Existing FAL methods (e.g., LoGo, KAFAL, IFAL) have begun addressing Non-IID settings, but generally treat heterogeneity only as a data partitioning problem, implicitly assuming a relatively balanced global label distribution. Under long-tailed global distributions, existing acquisition strategies struggle to capture minority-class samples, leading to wasteful annotation budgets.

This paper raises a fundamental question: **In FAL, which model — global or local — is better suited as the query selector, and how does this relate to class-balanced sampling?**

## Method

### Overall Architecture

FairFAL is built upon three empirical observations and comprises three corresponding core components:

- **Observation 1**: For uncertainty sampling, the local model generally outperforms the global model except when the global distribution is severely imbalanced and clients are approximately homogeneous → Adaptive Model Selection
- **Observation 2**: Regardless of which model is used, better class-balanced sampling (especially minority-class acquisition) consistently leads to higher final performance → Class-Aware Sampling
- **Observation 3**: For diversity sampling, the global model consistently outperforms the local model across all settings → Global Feature-Guided Diversity

### Key Designs

1. **Adaptive Model Selection**: The method estimates the degree of global imbalance and local-global distribution divergence via lightweight prediction discrepancy, then adaptively selects the query model.

   **Global class imbalance estimation**: For each client, a class-balanced subset $\mathcal{B}^{(k)}$ is constructed, and the global model's predicted prior $\hat{\boldsymbol{\pi}}_g^{(k)}$ is used to estimate the imbalance ratio:
    $\gamma_k = \frac{\min_{c \in \mathcal{C}_k^+} \hat{\pi}_{g,c}}{\max_{c \in \mathcal{C}_k^+} \hat{\pi}_{g,c}} \in (0,1]$
   Each client uploads the scalar $\gamma_k$; the server averages them to obtain the global coefficient $\bar{\gamma}$ (computed only in the first round).

   **Local-global distribution divergence estimation**:
    $d_k = \frac{1}{C}\sum_{c=1}^{C}\frac{|\hat{\pi}_{g,c} - \hat{\pi}_{\ell,c}^{(k)}|}{\hat{\pi}_{g,c} + \hat{\pi}_{\ell,c}^{(k)}}$

   **Model selection score**: $s_k = 1 - \frac{1}{2}(d_k + \bar{\gamma})$. The global model is selected when $s_k > \delta = 0.75$; otherwise the local model is used. Intuitively, when the global distribution is severely imbalanced (small $\bar{\gamma}$) and the local distribution closely mirrors the global one (small $d_k$), $s_k$ is large and the global model is preferred.

2. **Prototype-Guided Pseudo-Labeling**: Class prototypes are constructed from global model features to provide more reliable class assignments, overcoming classifier bias induced by imbalanced data.

   Class prototype: $\boldsymbol{\mu}_c^{(k)} = \frac{1}{|\mathcal{D}_{L,c}^{(k)}|}\sum_{y_i=c} \mathbf{z}_i^{(k)}$, where $\mathbf{z}_i^{(k)} = \frac{\phi^g(x_i)}{\|\phi^g(x_i)\|_2}$ denotes the $\ell_2$-normalized feature from the global model.

   Pseudo-labels are assigned via cosine similarity: $\hat{y}^{(k)}(x) = \arg\max_c \langle \mathbf{z}^{(k)}(x), \boldsymbol{\mu}_c^{(k)} \rangle$

   The unlabeled pool is partitioned into per-class subsets $\widetilde{\mathcal{D}}_{U,c}^{(k)}$ based on pseudo-labels, forming the foundation for subsequent class-aware sampling.

3. **Two-Stage Balanced Sampling**: Uncertainty and diversity are jointly optimized within a class-balanced framework.

   **Stage 1 — Per-class candidate selection**: A uniform budget $b_c^{(k)}$ is allocated per class; the top $\kappa \cdot b_c^{(k)}$ highest-uncertainty samples form an over-complete candidate pool $\mathcal{H}_c^{(k)}$ ($\kappa = 4$).

   **Stage 2 — Diversity refinement**: $k$-center sampling is applied in the gradient embedding space of the global model $\mathbf{g}^{(k)}(x) = \psi(x; \phi^g, f^g)$, minimizing the maximum distance:
    $\mathcal{Q}_c^{(k)} = \arg\min_{\mathcal{Q}'} \max_{x \in \mathcal{H}_c^{(k)}} \min_{a \in \mathcal{A}_c^{(k)} \cup \mathcal{Q}'} d(\mathbf{g}^{(k)}(x), \mathbf{g}^{(k)}(a))$
   A greedy $k$-center algorithm is used to obtain a 2-approximation solution.

### Loss & Training

- Standard federated training: FedAvg framework with local SGD
- Each FAL round consists of a complete federated training phase followed by active querying
- 5% of the training data is queried for annotation per round
- The first round uses random querying; subsequent rounds apply the FairFAL strategy

## Key Experimental Results

### Main Results

Datasets: FMNIST / CIFAR-10 / CIFAR-100, global imbalance ratio $\rho = 20$, 10 clients.

**CIFAR-10, final-round accuracy (α=0.1, ρ=20)**:

| Method | 15% | 25% | 35% | 45% |
|--------|-----|-----|-----|-----|
| Random | 47.24 | 50.46 | 54.29 | 55.70 |
| KAFAL | 49.99 | 56.34 | 58.41 | 60.01 |
| LoGo | 51.56 | 56.35 | 58.30 | 59.68 |
| IFAL | 47.76 | 52.67 | 55.62 | 57.51 |
| **FairFAL** | **52.12** | **56.90** | **59.62** | **60.44** |

**Medical datasets (α=0.1)**: On OctMNIST, FairFAL achieves 72.80% vs. KAFAL 70.40%; on DermaMNIST, FairFAL achieves 73.77% vs. LoGo 73.62%.

FairFAL consistently outperforms all baselines across all datasets and heterogeneity settings, with larger gains observed as task difficulty increases.

### Ablation Study

| Configuration | (α=0.1, ρ=20) | (α=100, ρ=20) | Note |
|---------------|:---:|:---:|------|
| Adaptive model selection $\mathcal{M}^{(k)}$ | 59.33 | 63.65 | Correct query model selected |
| Alternative model $\widetilde{\mathcal{M}}^{(k)}$ | 58.49 | 61.89 | Wrong selection → −0.84~1.76% |
| + Class-aware sampling (Local prototypes) | 59.14 | 63.39 | Local prototype quality is lower |
| + Class-aware sampling (Global prototypes) | 59.95 | 64.02 | Global prototypes more accurate (+0.63~0.81%) |
| + Two-stage balanced sampling (κ=2) | 60.61 | 64.60 | κ=2 marginally better but difference is small |
| + Two-stage balanced sampling (κ=4, Final) | **60.44** | **64.57** | Full FairFAL with a more flexible candidate pool |

### Key Findings

- **Generality of observations**: The pattern that class-balanced sampling leads to better performance holds consistently across all experimental settings.
- **Necessity of adaptive selection**: Using the "wrong" model degrades performance by 0.84–1.76% relative to the correct selection.
- **Global prototypes outperform local prototypes**: Global model features yield more discriminative and globally consistent representations.
- **Validation on medical data**: FairFAL achieves the best performance on OctMNIST and DermaMNIST (naturally long-tailed), attaining 72.80% vs. 70.40%.
- **Collapse of existing methods**: Under α=100 (near-homogeneous clients), methods lacking explicit class-balancing mechanisms (e.g., IFAL) perform even worse than random sampling.

## Highlights & Insights

1. **Systematic empirical study**: This is the first work to systematically investigate global vs. local query model selection in FAL, presenting three valuable observations validated via rigorous statistical testing (Wilcoxon test + Hodges-Lehmann estimator) rather than simple mean comparisons.
2. **Observation-driven design**: Each component has a clear empirical motivation with transparent design rationale.
3. **Privacy preservation**: Adaptive model selection only requires uploading the scalar $\gamma_k$, introducing no additional privacy leakage.
4. **Practical modularity**: The framework is modular with composable components; the $\kappa$ hyperparameter exhibits low sensitivity.

## Limitations & Future Work

1. **Fixed threshold $\delta = 0.75$**: This may lack flexibility for specific scenarios; adaptive adjustment warrants investigation.
2. **First-round assumption**: The method assumes the first-round randomly queried labeled set approximates IID, which may not hold under extreme imbalance.
3. **Classification-only validation**: Performance on more complex tasks such as detection and segmentation remains unexplored.
4. **Client scale**: Only the 10-client configuration is tested.
5. **Class count limitation**: CIFAR-100 covers only 100 classes; performance under very large label spaces (e.g., ImageNet-21k) is not verified.

## Related Work & Insights

- **BADGE**: A classic two-stage uncertainty-diversity sampling method; FairFAL extends this paradigm by incorporating class-aware mechanisms.
- **LoGo**: A FAL method combining local clustering with global uncertainty scoring, but without consideration of global class imbalance.
- **KAFAL/IFAL**: Leverage global-local prediction discrepancy to guide acquisition, but lack class-balancing designs and fail under extreme imbalance.
- **Key insight from this paper**: Class balance is central to FAL performance, rather than solely pursuing uncertainty or diversity; the representational advantage of the global model can be leveraged for prototype computation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Empirical observations are substantive; method design follows clear theoretical logic.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Five datasets, multiple configurations, statistical testing, and complete ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Excellent structure: observation → design → validation, with fluent exposition.
- **Value**: ⭐⭐⭐⭐ Fills a critical gap in FAL under extreme imbalance and Non-IID settings, with practical implications for real-world deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2026\] FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)

<!-- RELATED:END -->
