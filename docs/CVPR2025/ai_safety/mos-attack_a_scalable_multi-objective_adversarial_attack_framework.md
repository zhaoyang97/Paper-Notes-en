---
title: >-
  [Paper Note] MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework
description: >-
  [CVPR 2025][AI Safety][Adversarial Attack] This paper proposes the MOS-Attack framework, which models adversarial attacks as a multi-objective set-based optimization problem. By incorporating smooth max/min approximations, it enables the joint optimization of multiple loss functions and automatically discovers synergy patterns among them, outperforming existing state-of-the-art single-objective and ensemble attacks on CIFAR-10 and ImageNet.
tags:
  - "CVPR 2025"
  - "AI Safety"
  - "Adversarial Attack"
  - "Multi-Objective Optimization"
  - "Loss Function Synergy"
  - "Set-based Optimization"
  - "Robustness Evaluation"
date: 2026-05-08
content_hash: 747e6a6944a0be8b
---

# MOS-Attack: A Scalable Multi-Objective Adversarial Attack Framework

**Conference**: CVPR 2025  
**arXiv**: [2501.07251](https://arxiv.org/abs/2501.07251)  
**Code**: [GitHub](https://github.com/pgg3/MOS-Attack)  
**Area**: AI Safety  
**Keywords**: Adversarial Attack, Multi-Objective Optimization, Loss Function Synergy, Set-based Optimization, Robustness Evaluation

## TL;DR

This paper proposes the MOS-Attack framework, which models adversarial attacks as a multi-objective set-based optimization problem. By incorporating smooth max/min approximations, it enables the joint optimization of multiple loss functions and automatically discovers synergy patterns among them, outperforming existing state-of-the-art single-objective and ensemble attacks on CIFAR-10 and ImageNet.

## Background & Motivation

The essence of adversarial attacks lies in maximizing a non-differentiable 0-1 loss function, which is approximated in practice using differentiable surrogate losses (e.g., Cross-Entropy, DLR). Existing attack methods (e.g., FGSM, PGD, APGD, ACG) are single-objective attacks, meaning they optimize only one surrogate loss at a time.

**Core Problem**: Different surrogate loss functions exhibit varying capacities to approximate the 0-1 loss, and they possess **synergistic or conflicting** relationships. Simply combining multiple loss functions linearly fails to fully exploit these relationships. Although prior works have attempted multi-loss strategies (such as alternating optimization or multi-objective targeted losses), they lack a **systemic multi-objective optimization framework** and a **theoretical understanding of loss function interactions**.

**Key Challenges**:
- Directly allocating independent adversarial examples for each loss function is prohibitively expensive (number of samples = number of loss functions).
- How to simultaneously optimize multiple objectives using fewer samples than the number of loss functions.
- How to automatically discover which loss functions "assist" one another (synergy) to streamline the attack.

## Method

### Overall Architecture

MOS-Attack separates the adversarial attack into two phases: (1) **Multi-objective set-based optimization**: Given $m$ loss functions, $K$ adversarial examples ($K < m$) are used to simultaneously optimize all objectives to generate a solution set approximating Pareto optimality; (2) **Synergy pattern mining**: The correspondence between dominant samples and loss functions is analyzed to automatically discover synergy patterns among loss functions, thereby constructing a streamlined multi-objective attack (e.g., MOS-3*).

### Key Designs

**1. Smooth Set-based Optimization**

This approach stems from Tchebycheff decomposition but addresses its three main issues: (a) Complexity—replacing decomposition schemes that require $> m$ samples with $K < m$ samples; (b) Weight ambiguity—fixing weights to an all-one vector; (c) Non-differentiability—using smooth max/min operators to approximate extreme value operations.

The final optimization objective is: 

$$g(\Delta) = -\mu \log \left( \sum_{i=1}^m \left( \sum_{k=1}^K \exp(f_i(\delta_k)/\mu) \right)^{-1} \right)$$

where $\mu$ is a smoothing parameter, $f_i$ is the $i$-th loss function, and $\delta_k$ is the $k$-th perturbation. This formula elegantly expresses the concept of "taking the maximum over $K$ samples for each loss, and then the minimum across $m$ losses" in a differentiable form. Each sample can "focus on" different loss functions across different dimensions, giving rise to the concept of "virtual adversarial examples."

**2. APGD-based Implementation**

The smooth set-based optimization problem is embedded into the APGD framework: (a) Simultaneously optimizing $X$ (the set of adversarial examples) and $\Delta$ (the set of perturbations), since $\nabla_X g = \nabla_{\Delta} g$; (b) Set projection—projecting samples individually into the $\ell_\infty$ ball; (c) Inheriting APGD's momentum updates and adaptive step-size decay strategies, including checkpoint inspection and step-size halving mechanisms. No additional hyperparameters are required.

**3. Automated Synergy Pattern Mining**

A two-step pipeline: (a) **Determining dominant samples**—proposing a bi-objective optimization problem: minimizing the optimization gap between using a subset and the full set (approximated using joint smooth operators) + minimizing the subset size ($L_0 \to L_1$ relaxation), solved via gradient methods for the indicator vector $\beta$; (b) **Determining synergy patterns**—for each dominant sample, checking which loss functions yield normalized values exceeding a threshold of $C \times$ the maximum value, recording this as the sample's "loss synergy combination," and computing frequencies across the entire dataset.

### Loss & Training

Eight surrogate loss functions are used: four classic losses (Cross Entropy, Marginal Loss, DLR, Boosted CE) + four losses discovered through automated search (from AutoLoss and Tightening works). These eight loss functions cover diverse operations in logit and probability spaces.

## Key Experimental Results

### Main Results: Attack Success Rate (Table 3)

**CIFAR-10 ($\epsilon=8/255$):**

| Method | Avg Rank | ID0 (R-18)↑ | ID2 (R-18)↑ | ID9 (WR-70-16)↑ |
|------|:---:|:---:|:---:|:---:|
| APGD-CE (1 restart) | 5.92 | 39.17 | 41.57 | 31.43 |
| ACG-CW (5 restarts) | 4.00 | 42.45 | 43.10 | 32.54 |
| APGD-All (1×8) | 1.67 | 42.78 | 44.16 | 33.50 |
| **MOS-8 (K=5)** | **1.33** | **42.77** | **44.18** | **33.51** |

**ImageNet ($\epsilon=4/255$):**

| Method | Avg Rank | ID12 (R-18)↑ | ID13 (R-50)↑ | ID16 (WR-50-2)↑ |
|------|:---:|:---:|:---:|:---:|
| APGD-CE (1 restart) | 6.00 | 70.60 | 61.38 | 59.02 |
| ACG-CW (5 restarts) | 4.00 | 72.94 | 62.74 | 58.92 |
| APGD-All (1×8) | 1.40 | 74.38 | 64.92 | 61.26 |
| **MOS-8 (K=5)** | **1.60** | **74.52** | **64.94** | **61.14** |

MOS-8 achieves an average rank of 1.33 (best) on CIFAR-10, attaining performance comparable to or exceeding APGD-All (which takes the best of 8 independent attacks) using only 5 samples.

### Ablation Study

**MOS Upper Bound Analysis (Table 5)**: The gap between MOS-8 ($K=1$) and its theoretical upper bound is only 0.3%-0.8%, which further shrinks to 0.1%-0.3% for MOS-8 ($K=8$), demonstrating the high-fidelity approximation of smooth set-based optimization.

**Synergy Pattern Discovery (Fig. 2)**:
- On CIFAR-10, the most frequent pattern is the joint occurrence of {Loss5, Loss6, Loss7} (around 30%), followed by {Loss5, Loss6} (around 15%).
- Losses 4–7 discovered by search consistently perform best, yielding the highest ASR when optimized individually in APGD-All.
- MOS-3* (utilizing only 3 losses) constructed based on synergy analysis still outperforms 5-restart single-objective attacks like ACG-CW.

### Key Findings

- **Multi-Objective > Single-Objective**: Even the strongest single-objective ACG-CW (100 steps) achieves optimality on only 3 out of 17 models.
- **Efficiency Advantage**: MOS-8 ($K=5$) uses only 5 adversarial examples compared to 8 in APGD-All (a 37.5% efficiency gain) while yielding comparable or better performance.
- Searched loss functions (IDs 4–7) systematically outperform classic loss functions (IDs 0–3).
- As models become more complex (e.g., WR-70-16), the performance gap between MOS and single-objective attacks narrows, indicating more uniform robustness in stronger models.

## Highlights & Insights

1. **Elegance in Problem Formulation**: By formulating adversarial attacks as multi-objective set-based optimization, the application of smooth max/min operators renders the originally combinatorial optimization problem differentiable and solvable via gradient-based methods.
2. **Parameter-Free Design**: The framework introduces no auxiliary hyperparameters requiring tuning. The weight vector is fixed to an all-one vector, leaving the number of adversarial examples $K$ as the sole configurable variable.
3. **Automated Discovery of Synergy Patterns**: This not only enhances attack effectiveness but, more importantly, provides a systematic understanding of the relationships between different surrogate loss functions—revealing which losses "naturally pair up" and which "operate independently."

## Limitations & Future Work

- Although the smoothing parameter $\mu$ is fixed, its optimal selection might vary across different models and datasets.
- The discovery of synergy patterns depends on the initial pool of eight loss functions; a larger library of losses might yield different patterns.
- The method is validated only under the $\ell_\infty$ constraint, leaving its effectiveness under other norm constraints (e.g., $\ell_2$) unexplored.
- While the theoretical computational cost increases only by a constant factor, the batched forward and backward propagation of $K$ samples demands higher GPU memory.
- The validation is primarily conducted on robustness evaluation for classification tasks, leaving its applicability to recognition, detection, or segmentation tasks unexplored.

## Related Work & Insights

- **APGD/AutoAttack**: This serves as the base framework for MOS, onto which MOS incorporates multi-objective optimization capabilities.
- **ACG**: An advanced optimization scheme utilizing conjugate gradients, serving as the strongest single-objective attack baseline.
- **AutoLoss/Tightening**: Methods for automated search of surrogate loss functions, which provided Losses 4–7.
- **Insights**: The multi-objective optimization perspective can be extended to adversarial training (the defense side). Synergy pattern analysis can guide the design of robustness evaluation standards (e.g., selecting which loss functions to use as benchmarks). The smooth set-based optimization technique can also be applied to other AI safety problems requiring simultaneous optimization of multiple conflicting objectives.

## Rating

⭐⭐⭐⭐ — Designs an adversarial attack framework starting from multi-objective optimization theory. The mathematical formulation is elegant, and the engineering implementation is clean (with minimal modification to APGD). Its effectiveness is systematically validated across 17 models. The synergy pattern mining provides scientific value extending beyond mere "tool development."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/ai_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[ICLR 2026\] Concept-based Adversarial Attack: a Probabilistic Perspective](../../ICLR2026/ai_safety/concept-based_adversarial_attack_a_probabilistic_perspective.md)
- [\[CVPR 2026\] DASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples](../../CVPR2026/ai_safety/dash_a_meta-attack_framework_for_synthesizing_effective_and_stealthy_adversarial.md)
- [\[ICML 2025\] Understanding Model Ensemble in Transferable Adversarial Attack](../../ICML2025/ai_safety/understanding_model_ensemble_in_transferable_adversarial_attack.md)

</div>

<!-- RELATED:END -->
