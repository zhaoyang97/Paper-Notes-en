---
title: >-
  [Paper Note] Backdoor Mitigation by Distance-Driven Detoxification
description: >-
  [ICCV 2025][AI Safety][backdoor defense] This paper proposes Distance-Driven Detoxification (D3), which reformulates backdoor defense as a constrained optimization problem — maximizing the distance between the fine-tuned model weights and the poisoned initial weights, subject to a constraint that the clean sample loss does not exceed a threshold. This allows the model to effectively escape the "backdoor region," achieving best or second-best defense performance across 7 state…
tags:
  - "ICCV 2025"
  - "AI Safety"
  - "backdoor defense"
  - "fine-tuning"
  - "distance-driven"
  - "constrained optimization"
  - "model purification"
date: 2026-05-08
content_hash: ef221e1ebced2364
---

# Backdoor Mitigation by Distance-Driven Detoxification

**Conference**: ICCV 2025
**arXiv**: [2411.09585](https://arxiv.org/abs/2411.09585)  
**Code**: None (evaluated on the BackdoorBench platform)  
**Area**: AI Safety
**Keywords**: backdoor defense, fine-tuning, distance-driven, constrained optimization, model purification

## TL;DR

This paper proposes Distance-Driven Detoxification (D3), which reformulates backdoor defense as a constrained optimization problem — maximizing the distance between the fine-tuned model weights and the poisoned initial weights, subject to a constraint that the clean sample loss does not exceed a threshold. This allows the model to effectively escape the "backdoor region," achieving best or second-best defense performance across 7 state-of-the-art attacks.

## Background & Motivation

Backdoor attacks secretly implant backdoors during training, causing models to produce targeted misclassifications on trigger-bearing inputs while behaving normally on clean inputs. Post-training defense aims to purify an already-trained model that may have been backdoored.

The authors conduct an in-depth analysis of why conventional fine-tuning fails:

**Objective Mismatch**: The ideal defense objective should simultaneously minimize clean loss and maximize backdoor loss. However, vanilla fine-tuning only minimizes clean loss and completely ignores backdoor loss.

**Backdoor Region Trap**: By visualizing loss curves along the trajectory from initial weights to fine-tuned weights, the authors find that vanilla fine-tuning frequently converges to a region where both clean loss and backdoor loss are low — i.e., the model appears to perform well on clean data, yet the backdoor remains effective.

**Key Insight**: Extrapolating the weights further along the fine-tuning direction ($t>1$) can substantially increase backdoor loss without significantly affecting clean loss, thereby reducing attack success rate.

The theoretical explanation for this finding is based on a second-order Taylor expansion: the initial poisoned model is a local minimum of the backdoor loss, and the Hessian is positive semi-definite, so the backdoor loss grows approximately quadratically with distance.

## Method

### Overall Architecture

D3 formalizes backdoor defense as a constrained optimization problem, seeking a model that maximally departs from the poisoned initial weights while keeping the loss on clean data within a controllable bound. By converting the constraint into a regularization term, the problem is solved efficiently via projected gradient descent (PGD).

### Key Designs

1. **Constrained Optimization Formulation**:

    - Original form: $\max_{\theta} d(\theta, \theta_{init})$, subject to $\mathbb{E}[\ell(f_\theta(x), y)] \leq \epsilon$
    - The objective maximizes weight distance; the constraint bounds the clean data loss to threshold $\epsilon$.
    - Core Idea: escaping the low-value region of backdoor loss by moving away from the initial weights.

2. **Three Practical Challenges and Responses**:

    - **Overfitting**: Large deviations from pre-trained weights may degrade generalization. Solution: measure distance only over a subset of weights $\theta_s$ (e.g., linear layers), preserving pre-trained knowledge in remaining layers.
    - **Weight Scaling Loophole**: Simply scaling weights yields large distances without changing model predictions (since argmax is scale-invariant). Solution: add constraint $\theta_s \in \mathcal{S}$ to bound the weight norm, enforced via projection operator $\mathcal{P}$.
    - **Constraint Complexity**: Loss evaluation in DNNs is inherently non-convex and computationally intensive. Solution: convert the hard constraint into a regularization penalty term.

3. **Final Optimization Objective**:

    - $\min_{\theta:\theta_s \in \mathcal{S}} -d(\theta_s, \theta_{init,s}) + \lambda \cdot \max(0, \mathcal{L}_{cl}(\theta) - \epsilon)$
    - The first term maximizes the Frobenius-norm distance between selected weights and initial weights.
    - The second term penalizes violations of the clean performance constraint: activated only when clean loss exceeds $\epsilon$.
    - $\lambda=10$ controls the trade-off between distance and clean performance; $\epsilon=0.1$ is the loss threshold.
    - $\theta_s$ is chosen as the linear layer weights (generalizable across architectures).

### Loss & Training

- Solved via projected gradient descent (PGD): each iteration performs unconstrained gradient descent followed by projection to enforce $\theta_s \in \mathcal{S}$ (constraining the Frobenius norm).
- Default retain dataset size is 5% of the training set.
- Introduces negligible additional overhead compared to vanilla fine-tuning — only the weight distance computation is added.

## Key Experimental Results

### Main Results — CIFAR-10 PreAct-ResNet18

| Attack | No Defense ASR | FT ASR | FT-SAM ASR | SAU ASR | **D3 ASR** | D3 ACC |
|--------|---------------|--------|-----------|---------|-----------|--------|
| BadNets | 95.03 | 1.48 | 2.28 | 1.33 | **0.74** | 90.77 |
| Blended | 99.92 | 96.11 | 11.61 | 1.57 | **0.22** | 92.29 |
| WaNet | 89.73 | 17.10 | 1.31 | 0.58 | **0.04** | 93.31 |
| LF | 99.28 | 78.44 | 6.89 | 0.71 | **1.31** | 92.37 |
| Input-aware | 98.26 | 1.72 | 1.54 | 0.93 | **0.06** | 92.96 |
| SIG | 98.27 | 2.37 | 0.57 | 1.84 | **0.00** | 89.99 |
| SSBA | 97.86 | 74.79 | 3.20 | 0.81 | **0.46** | 91.93 |
| **Average** | 96.91 | 38.86 | 3.91 | 1.04 | **0.46** | 91.93 |

D3 achieves an average ASR of only 0.46%, substantially outperforming SAU (1.04%) and FT-SAM (3.91%).

### Ablation Study — Robustness Analysis

| Condition | BadNets ACC/ASR | Blended ACC/ASR | WaNet ACC/ASR |
|-----------|----------------|-----------------|---------------|
| Poison rate 1% | 92.18/0.68 | 92.85/0.24 | - |
| Poison rate 10% | 90.77/0.74 | 92.99/0.22 | - |
| Poison rate 50% | 86.90/1.51 | 89.01/0.03 | - |
| Retain set 1% | 88.57/2.31 | 90.64/2.86 | 91.96/1.42 |
| Retain set 5% | 90.77/0.74 | 92.29/0.22 | 93.31/0.04 |
| Retain set 10% | 90.97/0.44 | 92.61/0.01 | 93.53/0.11 |
| Generated data (CIFAR-5m) | 90.42/1.11 | 92.16/0.20 | 92.85/0.04 |

### Resistance to Adaptive Attacks

| Attack | SAM Perturbation Budget | FT ASR | FT-SAM ASR | **D3 ASR** |
|--------|------------------------|--------|-----------|-----------|
| BadNets | 1.0 | 26.30 | 17.74 | **0.76** |
| BadNets | 3.0 | 71.24 | 54.79 | **1.24** |
| Blended | 1.0 | 71.80 | 72.71 | **0.14** |
| Blended | 3.0 | 82.17 | 91.93 | **2.74** |
| WaNet | 3.0 | 21.38 | 18.87 | **1.48** |

When an adversary uses SAM to push backdoor weights toward flat minima, FT and FT-SAM fail while D3 remains effective.

### Key Findings

- D3 achieves the lowest ASR on 6 out of 7 attacks, remaining below 1% on the remaining one.
- As poison rate varies from 1% to 50%, D3 consistently maintains ASR below 2%.
- A retain dataset of only 1% of the training set suffices for effective defense.
- D3 also works effectively with generated data (CIFAR-5m), enhancing practical deployability.
- D3 runs faster than most defense methods with minimal additional overhead.
- T-SNE visualizations confirm that D3 restores poisoned samples to their correct clusters.
- Weight difference histograms show that D3's solutions are indeed farther from the initial weights than those of vanilla fine-tuning.

## Highlights & Insights

- The problem analysis is exceptionally thorough: the discovery of the "backdoor region trap" and the theoretical explanation via second-order Taylor expansion are highly convincing.
- The method is remarkably simple — it requires no trigger reconstruction, no teacher network, and no complex adversarial training; it merely adds distance regularization on top of standard fine-tuning.
- The identification of and response to three practical challenges (overfitting, scaling loophole, constraint complexity) reflects comprehensive design consideration.
- The adaptive attack experiments are particularly critical — when the adversary uses SAM to make the backdoor more robust, FT-SAM completely fails while D3 remains effective.

## Limitations & Future Work

- Moving the model away from initial weights in D3 may slightly degrade clean accuracy (average ACC is marginally lower than FT-SAM).
- Distance is measured only over linear layer weights; backdoor information embedded in other layers may not be fully addressed.
- The selection of $\lambda$ and $\epsilon$ lacks an adaptive mechanism; tuning may be required for different scenarios.
- Effectiveness on larger-scale models (e.g., backdoor defense for LLMs) remains to be validated.

## Related Work & Insights

- The closest related work is FT-SAM — FT-SAM improves fine-tuning via sharpness-aware minimization, but can still be bypassed by adaptive attacks; D3 fundamentally changes the optimization objective.
- Compared to methods requiring trigger reconstruction such as NC, i-BAU, and SAU, D3 avoids the computational overhead of trigger reverse engineering.
- The distance-driven idea may extend to other security scenarios, such as adversarial example defense or data poisoning defense.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The distance-driven optimization perspective is entirely novel, with deep theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 attacks × 3 datasets × 3 architectures × 8 baselines, including adaptive attack analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem observation → theoretical explanation → method design → experimental validation is complete and coherent.
- Value: ⭐⭐⭐⭐ The method is concise and efficient, with a low barrier to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Controllable Feature Whitening for Hyperparameter-Free Bias Mitigation](controllable_feature_whitening_for_hyperparameter-free_bias_mitigation.md)
- [\[CVPR 2026\] Eliminate Distance Differences Induced by Backdoor Attacks: Layer-Selective Training and Clipping to Mask Backdoor Models](../../CVPR2026/ai_safety/eliminate_distance_differences_induced_by_backdoor_attacks_layer-selective_train.md)
- [\[NeurIPS 2025\] Reconstruction and Secrecy under Approximate Distance Queries](../../NeurIPS2025/ai_safety/reconstruction_and_secrecy_under_approximate_distance_queries.md)
- [\[ICCV 2025\] Backdoor Attacks on Neural Networks via One-Bit Flip](backdoor_attacks_on_neural_networks_via_one_bit_flip.md)
- [\[NeurIPS 2025\] Robust Graph Condensation via Classification Complexity Mitigation](../../NeurIPS2025/ai_safety/robust_graph_condensation_via_classification_complexity_mitigation.md)

</div>

<!-- RELATED:END -->
