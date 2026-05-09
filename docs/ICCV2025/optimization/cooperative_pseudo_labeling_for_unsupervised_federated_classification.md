---
title: >-
  [Paper Note] Cooperative Pseudo Labeling for Unsupervised Federated Classification
description: >-
  [ICCV 2025][Optimization][Unsupervised Federated Learning] FedCoPL is the first work to extend unsupervised federated learning (UFL) to classification tasks. It addresses CLIP's inherent bias and label shift challenges via a cooperative pseudo labeling strategy (global assignment ensuring class balance) and a partial prompt aggregation protocol (aggregating only visual prompts while keeping text prompts local).
tags:
  - ICCV 2025
  - Optimization
  - Unsupervised Federated Learning
  - Pseudo Labeling
  - CLIP
  - Prompt Tuning
  - Label Shift
date: 2026-05-08
content_hash: d6457a4a4bc03407
---

# Cooperative Pseudo Labeling for Unsupervised Federated Classification

**Conference**: ICCV 2025
**arXiv**: [2510.10100](https://arxiv.org/abs/2510.10100)
**Code**: [https://github.com/krumpguo/FedCoPL](https://github.com/krumpguo/FedCoPL)
**Area**: Optimization / Federated Learning
**Keywords**: Unsupervised Federated Learning, Pseudo Labeling, CLIP, Prompt Tuning, Label Shift

## TL;DR

FedCoPL is the first work to extend unsupervised federated learning (UFL) to classification tasks. It addresses CLIP's inherent bias and label shift challenges via a cooperative pseudo labeling strategy (global assignment ensuring class balance) and a partial prompt aggregation protocol (aggregating only visual prompts while keeping text prompts local).

## Background & Motivation

Federated learning (FL) is a distributed learning paradigm for preserving data privacy. Conventional FL requires labeled data on each client, which is costly and demands domain expertise. While unsupervised federated learning (UFL) has been explored, **prior UFL methods focus exclusively on representation learning and clustering, and cannot be directly applied to classification**.

The emergence of vision-language models such as CLIP has changed this landscape—their strong zero-shot classification capability reduces dependence on labeled data. However, directly applying CLIP within a UFL framework faces two core challenges:

### Challenge 1: CLIP's Inherent Bias + Unknown Label Distribution

CLIP is pre-trained on web-scale data and exhibits preference toward certain classes. Directly using CLIP to generate pseudo labels leads to:
- Over-selection of preferred classes, even when pseudo labels are incorrect
- Unknown true label distributions on each client (only unlabeled data available)
- Severely imbalanced global training—some classes are over-represented while others are neglected

**Example**: Suppose the true label distribution on a client is $[0.1, 0.1, 0.8]$; CLIP-generated pseudo labels may yield $[0.6, 0.05, 0.35]$—low accuracy and unrepresentative of the true distribution.

### Challenge 2: Aggregation Conflicts from Label Shift

In federated settings, clients have heterogeneous label distributions (label skew), causing divergent model update directions. Naively averaging prompt parameters introduces conflicts and degrades performance.

**Key Insight**:
1. Each client estimates its local pseudo label distribution → the server globally reassigns labels → achieving cross-class balance.
2. Exploiting the characteristic difference between visual and text prompts—visual prompts are similar across clients (learning generic image representations), while text prompts diverge significantly (encoding class-specific knowledge)→ aggregate only visual prompts.

## Method

### Overall Architecture

FedCoPL consists of two core strategies:
1. **Cooperative Pseudo Labeling (CoPL)**: Clients estimate distributions → upload to server → global adjustment → redistribution to clients.
2. **Partial Prompt Aggregation**: Visual prompts are uploaded and aggregated; text prompts are retained locally.

### Key Designs

#### 1. Cooperative Pseudo Labeling (CoPL)

**Step 1: Local Distribution Estimation**

Each client $k$ applies dual filtering based on confidence and entropy to select an estimation set from its unlabeled data $D_k$:

$$D_k^{est} = \{(x, \hat{y}) \mid \max_c p_c(x) > \tau_1, \; Ent(p(x)) < \tau_2\}$$

where $\tau_1$ and $\tau_2$ are set to the median values of confidence and entropy, respectively.

**Design Motivation**: High confidence ensures pseudo label reliability; low entropy ensures unambiguous predictions. Using the median as a dynamic threshold avoids manual hyperparameter tuning.

**Step 2: Statistical Upload**

Each client computes the per-class pseudo label count to obtain an estimated distribution $U_k^{est}$ (a $C$-dimensional vector) and uploads it to the server. Only statistical information—not raw data—is transmitted, preserving privacy.

**Step 3: Global Assignment**

The server allocates a per-class global budget $M = \frac{\sum_k \sum_c u_{k,c}}{C}$ proportionally to each client:

$$\tilde{u}_{k,c} = \left\lceil \frac{u_{k,c}}{\sum_{i \in K} u_{i,c}} \cdot M \right\rceil$$

**Key Insight**: Proportional allocation rather than absolute values is used—even when estimated distributions are noisy, proportional relationships remain representative.

**Step 4: Local Selection**

Client $k$ selects, for each class, the samples with the highest predicted probabilities according to the adjusted capacity $\tilde{U}_k$, forming the training set $\tilde{D}_k$.

The entire pseudo label generation–estimation–assignment–selection pipeline is re-executed every $Q$ rounds (default $Q=5$) using the latest model.

#### 2. Partial Prompt Aggregation Protocol

Through empirical analysis, the authors identify an important phenomenon: **text prompts exhibit significantly greater divergence across clients than visual prompts**.

Measured via drift diversity and cosine distance:
- Visual prompts show low divergence → they learn generic image representations.
- Text prompts show high divergence → they encode class-specific knowledge tied to local label distributions.

Based on this observation, the protocol proposes:
- **Visual prompts $P_k^v$**: uploaded to the server for weighted average aggregation:
  $$P^v = \sum_{k=1}^K \frac{\tilde{n}_k}{\sum_i \tilde{n}_i} \cdot P_k^v$$
- **Text prompts $P_k^t$**: retained locally to achieve personalization.

The aggregated visual prompt is broadcast to all clients as initialization for the next round.

**Additional Benefit**: Transmitting only visual prompts also reduces communication overhead by 50%.

### Loss & Training

The overall objective function is:

$$\min_{P^v, \{P_k^t\}} \sum_{k=1}^K \mathbb{E}_{(x, \hat{y}) \in \tilde{D}_k} \ell_{ce}(g(P^v, P_k^t; x), \hat{y})$$

- The CLIP backbone (ViT-B/32 or ViT-B/16) is frozen; only prompts are trained.
- SGD optimizer with learning rate 0.1 and cosine annealing.
- 20 communication rounds, 10 local training epochs per round.

## Key Experimental Results

### Main Results

**Dirichlet-based label shift ($\beta=0.1$), ViT/B-32 backbone**:

| Method | PL | DTD | RESISC45 | CUB | UCF101 | CIFAR10 | CIFAR100 | Avg. |
|--------|----|-----|----------|-----|--------|---------|---------|------|
| CLIP Zero-shot | - | 43.24 | 54.51 | 51.28 | 61.00 | 86.93 | 64.17 | 60.19 |
| PromptFL + FPL | FPL | 45.79 | 59.76 | 47.29 | 64.39 | 87.51 | 63.26 | 61.33 |
| pFedPrompt + CPL | CPL | 44.22 | 61.76 | 47.23 | 65.59 | 90.26 | 65.63 | 62.45 |
| FedOPT + CPL | CPL | 37.43 | 51.10 | 46.61 | 58.93 | 91.08 | 57.87 | 57.17 |
| **FedCoPL** | **CoPL** | **60.89** | **75.76** | **56.09** | **73.20** | **95.38** | **73.59** | **72.49** |

FedCoPL achieves an average accuracy of **72.49%**, outperforming the best baseline by approximately **10%**. Gains are particularly pronounced on RESISC45 (75.76% vs. 65.95%) and UCF101 (73.20% vs. 65.59%).

### Ablation Study

| Confidence Filtering | Entropy Filtering | Global Assignment | DTD | RESISC45 | CUB | UCF101 |
|---------------------|------------------|------------------|-----|----------|-----|--------|
| ✗ | ✗ | ✗ | 45.79 | 59.76 | 47.29 | 64.39 |
| ✗ | ✗ | ✓ | 46.35 | 69.60 | 51.98 | 65.31 |
| ✓ | ✓ | ✗ | (not evaluated independently) | - | - | - |
| **✓** | **✓** | **✓** | **60.89** | **75.76** | **56.09** | **73.20** |

**Prompt aggregation strategy ablation**:

| Aggregate Text | Aggregate Visual | DTD | RESISC45 | CUB | UCF101 |
|---------------|-----------------|-----|----------|-----|--------|
| ✗ | ✗ (both local) | 50.45 | 61.72 | 49.08 | 66.02 |
| ✓ | ✓ (both aggregated) | 47.34 | 61.68 | 49.08 | 64.97 |
| ✓ | ✗ (text only) | 55.31 | 67.12 | 51.76 | 69.53 |
| ✗ | **✓ (visual only)** | **60.89** | **75.76** | **56.09** | **73.20** |

Aggregating only visual prompts yields the best performance, validating the hypothesis regarding the divergent characteristics of visual and text prompts.

### Key Findings

1. **Global assignment is the core contributor**: The global assignment strategy alone contributes approximately +10% on RESISC45, resolving class imbalance caused by CLIP's inherent bias.
2. **CoPL is plug-and-play**: Replacing the pseudo labeling component with CoPL in other federated frameworks yields a **17.09%** improvement for FedOPT + CoPL on RESISC45.
3. **Advantage grows with more clients**: As the number of clients scales from 5 to 50, baseline methods degrade substantially, while FedCoPL remains stable.
4. **Pseudo label accuracy**: CoPL achieves a pseudo label accuracy of **96.07%** on CIFAR10 (vs. FPL at 91.02% and CPL at 92.18%).

## Highlights & Insights

- **First extension of UFL to classification**: Opens a new research direction by leveraging the zero-shot capability of VLMs to compensate for the absence of labels.
- **Proportional allocation outperforms absolute values**: Even when estimates are inaccurate, proportional information preserves representativeness—a generalizable principle applicable to broader settings.
- **Privacy enhancement**: Only pseudo label distribution statistics (not precise class distributions) are uploaded, derived from CLIP predictions and thus non-invertible to raw data.
- **Discovery of visual/text prompt divergence**: Provides empirical evidence and design principles for federated prompt learning.

## Limitations & Future Work

- The method assumes a uniform global label distribution, which may not hold in real-world scenarios.
- Theoretical analysis of convergence, privacy, and fairness is absent (acknowledged by the authors as future work).
- The default median-based values for $\tau_1$ and $\tau_2$ may be suboptimal, though experiments suggest low sensitivity.
- Application to larger-scale VLMs (e.g., GPT-4V) and more complex tasks (e.g., segmentation, detection) remains unexplored.

## Related Work & Insights

- **UPL / FPL** [Huang et al., 2022; Menghini et al., 2023]: Centralized unsupervised prompt learning baselines.
- **CPL** [Zhang et al., 2024]: Generates multiple candidate pseudo labels to improve accuracy.
- **FedOPT** [Li et al., 2024]: Employs optimal transport to fuse global and local text prompts.
- **Orchestra** [Lubana et al., 2022]: Global clustering method for unsupervised federated representation learning.
- Insight: The modality-specific separation in CLIP (generic visual vs. specialized textual) may be exploitable in a wider range of federated learning scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — First extension of UFL to classification; the problem formulation is pioneering.
- Technical Depth: ⭐⭐⭐⭐ — Global assignment strategy and partial aggregation protocol are well-motivated and principled.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 6 datasets, two shift types, multiple baselines, and comprehensive ablations.
- Practicality: ⭐⭐⭐⭐ — Label-free, communication-efficient, and privacy-preserving; well-suited for real-world federated scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Interpretable Queries for Explainable Image Classification with Information Pursuit](learning_interpretable_queries_for_explainable_image_classification_with_informa.md)
- [\[NeurIPS 2025\] Deep Taxonomic Networks for Unsupervised Hierarchical Prototype Discovery](../../NeurIPS2025/optimization/deep_taxonomic_networks_for_unsupervised_hierarchical_prototype_discovery.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](../../NeurIPS2025/optimization/optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](federated_continual_instruction_tuning.md)
- [\[ICCV 2025\] Class-Wise Federated Averaging for Efficient Personalization](class-wise_federated_averaging_for_efficient_personalization.md)

</div>

<!-- RELATED:END -->
