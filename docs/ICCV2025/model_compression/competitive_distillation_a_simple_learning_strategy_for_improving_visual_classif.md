---
title: >-
  [Paper Note] Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification
description: >-
  [ICCV 2025][Model Compression][Competitive Distillation] This paper proposes a competitive distillation strategy in which, during multi-network joint training…
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Competitive Distillation"
  - "Knowledge Distillation"
  - "Mutual Learning"
  - "Stochastic Perturbation"
  - "Collective Intelligence"
date: 2026-05-08
content_hash: e2c8a379d878b156
---

# Competitive Distillation: A Simple Learning Strategy for Improving Visual Classification

**Conference**: ICCV 2025
**arXiv**: [2506.23285](https://arxiv.org/abs/2506.23285)
**Code**: N/A
**Area**: Model Compression & Knowledge Distillation
**Keywords**: Competitive Distillation, Knowledge Distillation, Mutual Learning, Stochastic Perturbation, Collective Intelligence

## TL;DR

This paper proposes a competitive distillation strategy in which, during multi-network joint training, the best-performing network is dynamically selected as the teacher at each iteration. Combined with a stochastic perturbation mechanism that introduces mutation operations analogous to genetic algorithms, the approach achieves significant improvements in visual classification performance.

## Background & Motivation

Knowledge distillation has been widely adopted to accelerate network training and improve performance. Existing approaches fall into two main categories: (1) conventional model distillation, which uses a fixed teacher network to guide a student network; and (2) Deep Mutual Learning (DML), in which multiple networks act as each other's teacher and student by aligning their predicted distributions. However, DML suffers from a counterintuitive problem: **the best-performing network should not learn from weaker ones**, yet DML's bidirectional alignment forces all networks to learn from one another, resulting in suboptimal learning directions.

The core motivation of this paper is that **the learning direction should be determined dynamically at each training iteration**—only the top-performing network should serve as the teacher, while all others learn from it, thereby preventing strong networks from being dragged down by weaker ones. This idea is inspired by the competitive processes observed in collective intelligence.

## Method

### Overall Architecture

Competitive distillation organizes a group of networks $\Theta = \{\Theta_i | i=1,2,...,n\}$ ($n \geq 2$) to jointly train on the same task. At each iteration, competitive optimization selects the current best network as the teacher, while the remaining networks serve as students and distill knowledge from it. Stochastic perturbation is additionally introduced to simulate the "mutation" operation in genetic algorithms, helping networks escape local optima.

### Key Designs

1. **Competitive Optimization**:
   At each training iteration, all networks compute the cross-entropy loss $L_C$ on the current batch. **The network with the lowest loss is designated as the teacher** $\Theta_T^t$, and the remaining networks become students $\Theta_S^t$. The teacher network updates its parameters using only the classification loss, whereas student networks additionally incorporate distillation and feature losses:
   $$\Theta_i^{t+1} \leftarrow \begin{cases} \Theta_i^t - \gamma \frac{\partial L_{C_i}}{\partial \Theta_i^t}, & \text{if } \Theta_i^t = \Theta_T^t \\ \Theta_i^t - \gamma \left(\frac{\partial L_{C_i}}{\partial \Theta_i^t} + \frac{\partial L_{D_i}}{\partial \Theta_i^t} + \frac{\partial L_{F_i}}{\partial \Theta_i^t}\right), & \text{otherwise} \end{cases}$$
   The teacher and student roles **switch dynamically at every iteration**, giving all networks an opportunity to act as the teacher.

2. **Stochastic Perturbation**:
   Inspired by the mutation operator in genetic algorithms, one network is randomly selected at each iteration and its input images are perturbed. Perturbations are sampled from a predefined processing pool $P = \{P_r | r=1,...,R\}$, which includes image mixing, splicing, noise injection, data warping, and random cropping, among others. These perturbation parameters are more aggressive than standard data augmentation (e.g., cropping scale set to $[0.3, 0.7]$).
   The core insight is that **beneficial mutations are naturally preserved** (a network that performs better after perturbation becomes the teacher, propagating its knowledge to all students), whereas **harmful mutations are automatically discarded** (a network that performs worse is not selected as the teacher, so its erroneous knowledge does not propagate).

3. **Multi-Level Knowledge Transfer**:
   Student networks receive two forms of supervisory signal from the teacher:
   - Distillation loss $L_D$: aligns the soft-label distributions of the student and teacher via KL divergence.
   - Feature loss $L_F$: aligns the feature maps of the student and teacher via L2 distance.

### Loss & Training

- **Teacher network loss**: $L_{\theta_i}^T = L_{C_i}$ (classification loss only)
- **Student network loss**: $L_{\theta_i}^S = L_{C_i} + \alpha L_{D_i} + \beta L_{F_i}$, where $\alpha = \beta = 1$
- Classification loss $L_C$: standard cross-entropy
- Distillation loss $L_D = D_{KL}(p_t \| p_i)$, measuring the divergence between student and teacher predicted distributions
- Feature loss $L_F = \sum_{j=1}^M \|F_i - F_t\|_2^2$, matching intermediate-layer feature representations
- Optimizer: SGD with Nesterov momentum, initial learning rate 0.1, batch size 128

## Key Experimental Results

### Main Results

| Dataset | Network Combination | Independent | DML | Competitive Distillation | Gain (vs. Ind.) |
|---|---|---|---|---|---|
| CIFAR-100 | ResNet-32 × 2 | 68.73 | 70.87 | 71.45 | +2.72 |
| CIFAR-100 | ResNet-56 + ResNet-152 | 73.76 / 76.66 | 75.31 / 78.64 | 76.76 / 79.80 | +3.00 / +3.14 |
| CIFAR-100 | CeiT-B × 2 | 86.28 | 87.76 | 88.05 | +1.77 |
| ImageNet | Vim-S × 2 | 80.46 | 81.37 | 82.04 | +1.58 |
| ImageNet | CeiT-B + ViT-B | 85.47 / 84.25 | 86.11 / 85.34 | 86.56 / 86.27 | +1.09 / +2.02 |
| Market-1501 | ResNet-56 + ResNet-152 | 88.57 / 94.77 | 90.43 / 95.32 | 91.28 / 96.17 | +2.71 / +1.40 |

### Ablation Study

| Configuration | Net1 Acc. (CIFAR-100) | Notes |
|---|---|---|
| Competitive Optimization (Backbone) | 71.13 | Basic competitive selection mechanism |
| + Feature Loss $L_F$ | 71.24 | With feature alignment |
| + Stochastic Perturbation | 71.37 | With mutation operation |
| + $L_F$ + Perturbation (Full) | 71.45 | All components; best performance |
| DML + $L_F$ + Perturbation | 71.09 | Same components under DML framework perform worse |

### Key Findings

- Competitive distillation is effective across three architecture families: CNNs (ResNet, MobileNet, WRN), Transformers (ViT, CeiT), and Mamba (Vim).
- Three-network groups ($n=3$) yield larger gains than two-network groups ($n=2$).
- Heterogeneous network combinations (e.g., ResNet-32 + WRN28-12) also benefit substantially.
- The approach generalizes to cross-task scenarios such as person re-identification (Market-1501).

## Highlights & Insights

- **Remarkably simple idea**: merely changing the rule for selecting "who acts as the teacher" leads to substantial improvements over DML and related methods, underscoring the critical role of learning direction in training effectiveness.
- **Elegant design of stochastic perturbation**: the competitive selection mechanism automatically filters out harmful perturbations and retains beneficial ones, eliminating the need for manually designed perturbation strategies.
- **Strong generality**: the method is applicable to CNNs, Transformers, and Mamba architectures; to classification, ReID, and detection tasks; and to both homogeneous and heterogeneous network combinations.

## Limitations & Future Work

- Simultaneous training of multiple networks requires GPU memory and computational cost proportional to $n$ times that of independent training.
- Only one network is used for inference after training, leaving the training resources invested in the remaining networks effectively unutilized.
- Competitive selection is based solely on the loss of the current batch, which may lead to suboptimal teacher selection in the presence of data noise.
- The effectiveness of the approach on larger-scale models (e.g., ViT-L, LLMs) remains unexplored.

## Related Work & Insights

- The key distinction from DML lies in **unidirectional** vs. **bidirectional** knowledge flow: DML enforces bidirectional alignment among all networks, whereas competitive distillation ensures that knowledge flows strictly from stronger to weaker networks.
- The stochastic perturbation mechanism is analogous to the mutation-selection process in evolutionary strategies; incorporating additional evolutionary algorithm strategies (e.g., crossover, elitism) is a promising direction.
- Combining the competitive mechanism with self-distillation to construct "virtual competitors" within a single network is another avenue worth exploring.

## Rating

- **Novelty**: ⭐⭐⭐ The core idea is simple and intuitive; the degree of innovation is modest, but the results are solid.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers diverse architectures, multiple tasks, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Logically clear with intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Simple and practical; high applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Cross-Architecture Distillation Made Simple with Redundancy Suppression](cross-architecture_distillation_made_simple_with_redundancy_suppression.md)
- [\[ICCV 2025\] Soft Separation and Distillation: Toward Global Uniformity in Federated Unsupervised Learning](soft_separation_and_distillation_toward_global_uniformity_in_federated_unsupervi.md)
- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[ICCV 2025\] Gain-MLP: Improving HDR Gain Map Encoding via a Lightweight MLP](gain-mlp_improving_hdr_gain_map_encoding_via_a_lightweight_mlp.md)
- [\[ICCV 2025\] CIARD: Cyclic Iterative Adversarial Robustness Distillation](ciard_cyclic_iterative_adversarial_robustness_distillation.md)

</div>

<!-- RELATED:END -->
