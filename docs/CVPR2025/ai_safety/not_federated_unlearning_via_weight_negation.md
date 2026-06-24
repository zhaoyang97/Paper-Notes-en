---
title: >-
  [Paper Note] NoT: Federated Unlearning via Weight Negation
description: >-
  [CVPR 2025][AI Safety][Federated Unlearning] This paper proposes the NoT algorithm, which achieves unlearning by multiplying the weights of specific layers of the global model by -1 (negating) to disrupt inter-layer co-adaptation, followed by fine-tuning with retained data to recover performance. It requires no extra storage or access to target data, and significantly outperforms seven baseline methods on CIFAR-10/100 and Caltech-101 with the lowest communication and computat…
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Federated Unlearning"
  - "Weight Negation"
  - "Inter-layer Co-adaptation"
  - "Privacy Protection"
  - "Model Perturbation"
date: 2026-05-08
content_hash: 160d6ea14637ceb5
---

# NoT: Federated Unlearning via Weight Negation

**Conference**: CVPR 2025  
**arXiv**: [2503.05657](https://arxiv.org/abs/2503.05657)  
**Code**: None  
**Area**: AI Safety / Federated Unlearning  
**Keywords**: Federated Unlearning, Weight Negation, Inter-layer Co-adaptation, Privacy Protection, Model Perturbation

## TL;DR
This paper proposes the NoT algorithm, which achieves unlearning by multiplying the weights of specific layers of the global model by -1 (negating) to disrupt inter-layer co-adaptation, followed by fine-tuning with retained data to recover performance. It requires no extra storage or access to target data, and significantly outperforms seven baseline methods on CIFAR-10/100 and Caltech-101 with the lowest communication and computational overheads.

## Background & Motivation

**Background**: Federated learning (FL) preserves data privacy through distributed training, but faces a growing demand for data deletion—regulations like GDPR grant users the "right to be forgotten," requiring models to eliminate the contributions of specific participants. Federated Unlearning (FU) has emerged to address this need.

**Limitations of Prior Work**: Existing FU methods have distinct drawbacks. The first category (e.g., FedEraser, FUKD) requires storing historical model updates, incurring extra storage overhead and privacy risks. The second category (e.g., PGD, MoDE, FCU) achieves unlearning through gradient modification but requires the participation of target clients and incurs high computational costs. Crucially, many methods rely on direct access to the target data, which may no longer be available.

**Key Challenge**: Effective unlearning requires sufficiently large perturbations to model parameters to "forget" target data, but excessive perturbations require intensive fine-tuning to recover performance—namely, the trade-off between unlearning thoroughness and recovery efficiency.

**Goal**: To design an unlearning algorithm that requires neither extra storage nor access to target data, while guaranteeing both unlearning effectiveness and fast recovery capability.

**Key Insight**: The authors approach this from the perspective of inter-layer co-adaptation, observing that neural network performance highly depends on the coordinate relationship of parameters across layers. If this coordination is disrupted, the model loses its acquired knowledge.

**Core Idea**: To use weight negation (multiplying by -1) to maximize the disruption of inter-layer co-adaptation, while theoretically proving that the negated model retains "layer-wise optimality" (LWOP), enabling rapid performance recovery during subsequent fine-tuning.

## Method

### Overall Architecture
NoT follows a two-step paradigm of "perturbation + fine-tuning." When a client initiates an unlearning request: (1) The server performs weight negation on specified layers of the global model: $\theta'_\ell = -\theta^*_\ell$; (2) Federated fine-tuning is performed on the negated model using the retained data to recover key knowledge. The entire process does not require storing historical updates or accessing target data.

### Key Designs

1. **Weight Negation**:

    - **Function**: Disrupts the model's inter-layer co-adaptation by multiplying the parameters of specified layers by -1.
    - **Mechanism**: For each layer $\ell$ in the selected layer set $\mathscr{L}_{\text{neg}}$, execute $\theta'_\ell = -\theta^*_\ell$ while retaining other layers. After negation, the output of layers using ReLU activation changes fundamentally—originally positive pre-activation values become negative and are clipped to 0, while originally negative ones become positive and are retained, which is equivalent to "flipping" the feature selection pattern.
    - **Design Motivation**: Theorem 2 proves that under mild assumptions, weight negation causes the largest change in activation values among all equal-norm perturbations (i.e., $\mathbb{E}\|\sigma(Y) - \sigma(-Y)\|^2$ is an upper bound for any other perturbation), thereby representing the strongest perturbation method.

2. **Unlearning via Loss Gap**:

    - **Function**: Provides a theoretical foundation for the "perturbation + fine-tuning" paradigm and quantifies the unlearning speed.
    - **Mechanism**: Defines the loss gap $\delta(\theta) = |\mathcal{L}_{D_r}(\theta) - \mathcal{L}_{D_u}(\theta)|$ to measure the degree of unlearning. Theorem 1 provides a lower bound for the minimum fine-tuning time required to achieve target unlearning, showing that a larger initial loss and a larger Hessian spectrum lead to faster unlearning. Natural unlearning (fine-tuning directly without perturbation) is extremely slow due to the initial loss being near-optimal and gradients being small.
    - **Design Motivation**: A strong perturbation is required to accelerate the unlearning process, rather than relying on slow natural unlearning.

3. **Resilient Perturbation**:

    - **Function**: Guarantees that the negated model can be recovered quickly.
    - **Mechanism**: A resilient perturbation must satisfy two conditions: (C2a) Jacobian control: Theorem 3 proves that for layers $\ell > \mathscr{L}_{\text{neg}}$, the Wasserstein distance of the gradient Jacobian is controlled by $TV(Y_-; -Y_-)$, which approaches 0 in the wide network limit; (C2b) layer-wise optimality preservation (LWOP): Theorem 4 proves that if $\mathscr{L}_{\text{neg}}$ is an antichain in the computing graph's partially ordered set and does not contain the maximal element, negation preserves LWOP.
    - **Design Motivation**: Random parameter resetting is also a strong perturbation, but lacks resilience and recovers extremely slowly. Weight negation, by retaining the absolute value information of parameters and the distribution of the Jacobian spectrum, yields much faster recovery.

### Loss & Training
Fine-tuning after negation is performed using the standard FedAvg algorithm on the retained data $D_r$. For client-level unlearning, the target client does not participate in fine-tuning; for class-level or instance-level unlearning, the target client participates using its remaining data. By default, only the first-layer weights are negated—this is sufficient to alter low-level feature representations, which propagates through fine-tuning and leads to significant updates in deeper parameters.

## Key Experimental Results

### Main Results
Comparison with 7 baselines on CIFAR-10, CIFAR-100, and Caltech-101 in an IID setting, with 10 clients and 1 client requesting unlearning:

| Dataset / Model | Method | Retain Acc (Δ↓) | Forget Acc (Δ↓) | Test Acc (Δ↓) | MIA (Δ↓) | Avg Gap ↓ |
|---|---|---|---|---|---|---|
| CIFAR-10 / CNN | Retrain (oracle) | 91.66 (0.00) | 83.05 (0.00) | 82.32 (0.00) | 50.23 (0.00) | 0.00 |
| | FT | 92.48 (0.82) | 85.56 (2.51) | 82.36 (0.04) | 50.90 (0.67) | 1.01 |
| | MoDE | 92.56 (0.90) | 85.25 (2.20) | 82.31 (0.01) | 50.70 (0.47) | 0.90 |
| | **NoT** | **91.69 (0.03)** | **83.86 (0.81)** | **82.65 (0.33)** | **50.23 (0.00)** | **0.29** |
| CIFAR-100 / CNN | Retrain | 72.32 (0.00) | 53.31 (0.00) | 54.28 (0.00) | 49.70 (0.00) | 0.00 |
| | FCU | 73.40 (1.08) | 56.68 (3.37) | 55.37 (1.09) | 50.03 (0.33) | 1.47 |
| | **NoT** | **72.25 (0.07)** | **55.22 (1.91)** | **55.23 (0.95)** | **49.63 (0.07)** | **0.75** |
| Caltech-101 / ViT | Retrain | 99.73 (0.00) | 48.29 (0.00) | 48.02 (0.00) | 49.67 (0.00) | 0.00 |
| | FT | 99.96 (0.23) | 94.23 (45.94) | 48.75 (0.73) | 73.80 (24.13) | 17.76 |
| | **NoT** | **99.70 (0.03)** | **50.81 (2.52)** | **47.83 (0.19)** | **50.07 (0.40)** | **0.79** |

### Non-IID Experiments
Caltech-101 / ViT, Dirichlet $\beta=0.1$:

| Method | Retain Δ↓ | Forget Δ↓ | Test Δ↓ | MIA Δ↓ | Avg Gap ↓ |
|---|---|---|---|---|---|
| FT | 1.38 | 36.24 | 2.19 | 15.67 | 13.87 |
| PGD | 1.30 | 32.14 | 1.69 | 13.53 | 12.16 |
| **NoT** | **0.71** | **1.40** | **0.37** | **0.97** | **0.86** |

### Ablation Study

| Perturbation Method | Avg Gap ↓ | Description |
|---|---|---|
| Weight Negation (NoT) | 0.29 | Negating the first layer, optimal |
| Gaussian Noise | 1.55 | Gaussian noise perturbation, slow recovery |
| Random Reinit | 3.23 | Random reinitialization, slowest recovery |
| Negation (all layers) | 0.84 | Negating all layers, vanishing gradient affects recovery |
| Negation (last layer) | 1.12 | Negating only the last layer, insufficient unlearning |

### Key Findings
- **Negating the first layer works best**: The first layer controls low-level feature extraction; negating it cascade-affects deeper parameter updates via fine-tuning, achieving effective unlearning.
- **Lowest communication/computation overhead**: NoT's communication volume is about 50-60% of Retrain, and computation is about 50-55%.
- **Particularly prominent advantage on ViT**: FT fails to unlearn almost completely (Forget Acc 94.23% vs. Retrain 48.29%), whereas NoT reaches 50.81%.
- **Supports three unlearning granularities**: Client-level, class-level, and instance-level unlearning are all effective without changing the algorithm.

## Highlights & Insights
- **Minimalist design, profound theory**: Multiplying weights by -1 seems simple and aggressive, but is backed by a complete theoretical framework—Theorem 2 proves it is the strongest perturbation, and Theorems 3-4 prove it retains resilience. This style of "simple method + rigorous proof" is exemplary.
- **Zero extra requirements**: No need to store historical updates, no need to access data, and no participation required from the target client. Compared to FedEraser (which stores updates from all rounds) and PGD (which requires the target client to perform gradient reversal), NoT has an extremely low deployment barrier in practice.
- **Leveraging inter-layer co-adaptation**: Translating inter-layer dependency, which is typically treated as a black box, into a controllable unlearning tool. This perspective can be transferred to scenarios such as model safety and backdoor removal.

## Limitations & Future Work
- The theoretical Theorem 4 does not cover the LWOP proof for the ReLU activation function (since ReLU is neither odd nor even), leaving only a conjectured "approximate LWOP."
- On ResNet-18/CIFAR-100, the Avg Gap (4.73) is higher than that of FCU (2.51); recovery after negation might be slower in complex models and fine-grained classification scenarios.
- Only IID and simple non-IID distributions are considered, leaving performance under extreme heterogeneous data distributions unknown.
- Systematic defense effects against model backdoor attacks are not discussed.

## Related Work & Insights
- **vs. FedEraser**: Requires storing historical model states, which incurs high storage costs and privacy risks; NoT requires no storage.
- **vs. PGD**: Performs gradient reversal on target data and constrains update magnitudes, requiring target client participation; NoT only requires server-side operations.
- **vs. MoDE**: Degrades performance using a randomly initialized model, which is conceptually similar but lacks theoretical behavior guarantees; NoT is supported by rigorous theory.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The idea of weight negation is extremely simple yet effective, with a comprehensive theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, three architectures, and multiple settings, but larger-scale experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivation and well-organized experiments.
- Value: ⭐⭐⭐⭐ High practical value, easy deployment, but the scope of application is limited to federated learning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] WARP: Weight Teleportation for Attack-Resilient Unlearning Protocols](../../ICLR2026/ai_safety/warp_weight_teleportation_for_attack-resilient_unlearning_protocols.md)
- [\[ICML 2026\] Forgetting is Not Deletion: An Investigation of Reversibility in LLM Machine Unlearning](../../ICML2026/ai_safety/unlearning_isnt_deletion_investigating_reversibility_of_machine_unlearning_in_ll.md)
- [\[CVPR 2025\] Towards Source-Free Machine Unlearning](towards_source-free_machine_unlearning.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[CVPR 2025\] A Simple Data Augmentation for Feature Distribution Skewed Federated Learning](a_simple_data_augmentation_for_feature_distribution_skewed_federated_learning.md)

</div>

<!-- RELATED:END -->
