---
title: >-
  [Paper Note] ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning
description: >-
  [CVPR 2026][AI Safety][Federated Learning] ProxyFL is proposed as a framework that leverages classifier weights as unified proxies to simultaneously mitigate external heterogeneity (cross-client distribution discrepancy)…
tags:
  - "CVPR 2026"
  - "AI Safety"
  - "Federated Learning"
  - "Semi-Supervised Learning"
  - "Data Heterogeneity"
  - "Proxy Learning"
  - "Pseudo-Labels"
date: 2026-05-08
content_hash: d413c33e52e81378
---

# ProxyFL: A Proxy-Guided Framework for Federated Semi-Supervised Learning

**Conference**: CVPR 2026
**arXiv**: [2602.21078](https://arxiv.org/abs/2602.21078)  
**Code**: [DuowenC/FSSLlib](https://github.com/DuowenC/FSSLlib)  
**Area**: AI Security
**Keywords**: Federated Learning, Semi-Supervised Learning, Data Heterogeneity, Proxy Learning, Pseudo-Labels

## TL;DR

ProxyFL is proposed as a framework that leverages classifier weights as unified proxies to simultaneously mitigate external heterogeneity (cross-client distribution discrepancy) and internal heterogeneity (distribution mismatch between labeled and unlabeled data) in federated semi-supervised learning, achieving substantial improvements over existing FSSL methods across multiple benchmarks.

## Background & Motivation

Federated semi-supervised learning (FSSL) enables multiple clients to collaboratively train a global model using limited labeled data and abundant unlabeled data while preserving privacy. The core challenge lies in two dimensions of data heterogeneity:

**External Heterogeneity**: Distribution discrepancies across clients (non-IID). Existing methods mitigate this via dynamic aggregation weights, but naive averaging can be skewed by outlier clients, causing deviation from the global class distribution.

**Internal Heterogeneity**: Distribution mismatch between labeled and unlabeled data within a single client. Existing methods typically discard low-confidence samples to avoid pseudo-label errors, resulting in insufficient training data participation.

Through empirical analysis, the authors identify two key observations: (1) simple averaging of classifier weights is susceptible to outlier shifts and fails to effectively capture the global class distribution; (2) as heterogeneity increases, more unlabeled samples are excluded from training, despite their potential to improve performance.

## Method

### Overall Architecture

ProxyFL defines the learnable weights $\boldsymbol{\Omega}_m = \{\omega_m^c\}_{c=1}^C$ of the final fully connected layer as class proxies, which model both local and global class distributions. The framework comprises two core modules: **Global Proxy Tuning (GPT)** on the server side and **Indecisive-Categories Proxy Learning (ICPL)** on the client side. The proxies are part of the model parameters, introducing no additional privacy risks or communication overhead.

### Key Designs

1. **Global Proxy Tuning (GPT)**: Explicitly optimizes the global proxy on the server to fit cross-client class distributions. The global proxy is first initialized via simple averaging $\overline{\boldsymbol{\Omega}}_{\mathcal{G}}$, then further fine-tuned via a contrastive objective that pulls the global proxy $\boldsymbol{\Omega}_{\mathcal{G}}^c$ toward same-class local proxies and pushes it away from different-class ones:

$$\mathcal{L}_{\text{GPT}} = \sum_{c=1}^{C}\sum_{m=1}^{M} -\log \frac{e^{-\phi(\boldsymbol{\Omega}_{\mathcal{G}}^c, \omega_m^c)}}{e^{-\phi(\boldsymbol{\Omega}_{\mathcal{G}}^c, \omega_m^c)} + \sum_{c' \neq c} e^{-\phi(\boldsymbol{\Omega}_{\mathcal{G}}^c, \omega_m^{c'})}}$$

   The computational complexity is only $O(Q \times M \times C^2 \times d)$, approximately 0.4 GFLOPs on CIFAR-100—equivalent to a single image inference pass and thus negligible.

2. **Indecisive-Categories Proxy Learning (ICPL)**: Rather than discarding low-confidence unlabeled samples or assigning a single pseudo-label, this module constructs an "indecisive category set" $\xi_i$. For each low-confidence sample $\mathbf{u}_i^{\text{lc}}$, any class whose global logit $\overline{\mathbf{y}}_i(c)$ exceeds the global class prior $\mathcal{P}_{\mathcal{G}}'(\mathbf{Y}(c))$ is included in $\xi_i$:

$$\xi_i = \{c \mid \overline{\mathbf{y}}_i(c) > \mathcal{P}_{\mathcal{G}}'(\mathbf{Y}(c))\}$$

   The prior $\mathcal{P}_{\mathcal{G}}'$ is aggregated from the prediction preferences of each client's model, imposing higher thresholds for majority classes and lower thresholds for minority classes to dynamically regulate the indecisive category range.

3. **Positive-Negative Proxy Pool**: Positive and negative proxy pools are constructed based on each sample's category set $\xi_i$. For high-confidence samples, the positive proxy is the classifier weight corresponding to the pseudo-label class $\omega_i^{\text{hc}} = \omega_k^{\hat{y}_i}$; for low-confidence samples, it is a weighted sum of the indecisive category proxies $\omega_i^{\text{lc}} = \sum_{c' \in \xi_i} \tilde{\mathbf{y}}_i(c') \times \omega_k^{c'}$. Negative samples are drawn from samples in the batch whose category sets have no overlap with the current sample. A contrastive objective ensures that all samples—including low-confidence ones—participate in training.

### Loss & Training

The total loss comprises local and global components:

$$\mathcal{L} = \underbrace{\mathcal{L}_s + \alpha \mathcal{L}_u + \beta \mathcal{L}_{\text{ICPL}}}_{\text{local}} + \underbrace{\mathcal{L}_{\text{GPT}}}_{\text{global}}$$

- $\mathcal{L}_s$: Cross-entropy loss on labeled data
- $\mathcal{L}_u$: KL divergence loss on high-confidence unlabeled data (strongly augmented predictions vs. pseudo-labels)
- $\mathcal{L}_{\text{ICPL}}$: Contrastive loss over all unlabeled data
- $\alpha, \beta$ are both set to 1

## Key Experimental Results

### Main Results

10% label rate; heterogeneity controlled via Dirichlet distribution (smaller $\alpha$ indicates higher heterogeneity):

| Dataset | $\alpha$ | Metric (Acc) | ProxyFL | Prev. SOTA (SAGE) | Gain |
|--------|---------|-----------|---------|----------------|------|
| CIFAR-10 | 0.1 | Acc | 88.56 | 87.05 | +1.51 |
| CIFAR-100 | 0.1 | Acc | 57.50 | 54.18 | +3.32 |
| SVHN | 0.1 | Acc | 95.09 | 93.85 | +1.24 |
| CINIC-10 | 0.1 | Acc | 77.98 | 74.59 | +3.39 |
| CIFAR-100 | 0.5 | Acc | 58.75 | 55.82 | +2.93 |
| CINIC-10 | 0.5 | Acc | 78.96 | 75.74 | +3.22 |

On SVHN and CINIC-10 ($\alpha=0.1$), ProxyFL approaches the fully supervised upper bound of FedAvg-SL.

### Ablation Study

| Configuration | CIFAR-10 ($\alpha$=0.1) | CIFAR-100 ($\alpha$=0.1) | Notes |
|------|------------------------|--------------------------|------|
| Baseline (GPL) | 84.56 | 48.96 | FedAvg+FixMatch-GPL |
| +GPT | 87.59 | 54.86 | Global proxy tuning yields significant gains |
| +ICPL | 87.81 | 57.21 | Low-confidence sample participation is effective |
| +GPT+ICPL | **88.56** | **57.50** | Two modules are complementary, achieving optimal performance |

Comparison of indecisive category set designs ($\alpha=0.1$):

| Strategy | CIFAR-100 | SVHN | Notes |
|------|-----------|------|------|
| Top-1 | 55.66 | 94.56 | Single pseudo-label |
| Top-5 | 56.58 | 94.71 | Fixed top-5 categories |
| $\mathcal{P}_{\mathcal{G}}'(\mathbf{Y})$ | **57.21** | **94.82** | Dynamic prior threshold achieves best results |

### Key Findings

- **Convergence speed**: ProxyFL reaches 50% accuracy on CIFAR-100 ($\alpha=0.1$) in only 177 rounds, a 3.18× speedup over LPL's 562 rounds.
- **Recall rate of the indecisive category set** substantially exceeds the precision of single pseudo-labels, validating the set-based strategy.
- **Proxy vs. prototype**: The proxy approach outperforms FedProto+FSSL variants across all datasets without introducing privacy risks (prototypes are susceptible to feature inversion attacks).

## Highlights & Insights

- The novel use of classifier weights as "proxies" to unify the handling of both external and internal heterogeneity avoids the privacy leakage risks inherent in prototype-based methods.
- The indecisive category set is an elegant design—rather than hard-coding a single pseudo-label, it preserves uncertainty and allows contrastive learning to handle ambiguity naturally.
- The server-side tuning overhead of the GPT module is minimal (roughly equivalent to a single image inference), making it highly practical.

## Limitations & Future Work

- Validation is limited to image classification; extension to more complex tasks such as detection and segmentation remains unexplored.
- Experiments cover only the Labels-at-All-Clients scenario; other FSSL settings such as Labels-at-Partial-Clients are not addressed.
- The prior distribution $\mathcal{P}_{\mathcal{G}}'$ for the indecisive category set accumulates across global communication rounds and may be unstable in early rounds.
- The number of clients is fixed at 20; scalability to larger-scale federated settings is not investigated.

## Related Work & Insights

- **FedDure / SAGE**: Current FSSL state of the art; ProxyFL builds upon these by introducing a proxy mechanism.
- **FedProto**: Represents class distributions via prototypes but carries privacy leakage risks, as features can be reconstructed via inversion.
- **FixMatch**: A standard SSL baseline whose high-confidence filtering strategy leads to data waste in FSSL settings.
- Proxy learning has established applications in metric learning; this work is the first to introduce it into FSSL for addressing heterogeneity.

## Rating

- Novelty: ⭐⭐⭐⭐ (Unified proxy framework addressing both internal and external heterogeneity with conceptual clarity)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 datasets × 3 heterogeneity levels with comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear problem–observation–solution derivation logic)
- Value: ⭐⭐⭐⭐ (Substantive contribution to the FSSL field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[CVPR 2026\] FedDAP: Domain-Aware Prototype Learning for Federated Learning under Domain Shift](feddap_domain-aware_prototype_learning_for_federated_learning_under_domain_shift.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[CVPR 2026\] SubFLOT: Submodel Extraction for Efficient and Personalized Federated Learning via Optimal Transport](subflot_submodel_extraction_for_efficient_and_personalized_federated_learning_vi.md)
- [\[CVPR 2026\] FecalFed: Privacy-Preserving Poultry Disease Detection via Federated Learning](fecalfed_privacy-preserving_poultry_disease_detection_via_federated_learning.md)

</div>

<!-- RELATED:END -->
