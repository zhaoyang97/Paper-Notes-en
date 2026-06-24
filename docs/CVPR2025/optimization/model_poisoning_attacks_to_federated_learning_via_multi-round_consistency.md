---
title: >-
  [Paper Note] Model Poisoning Attacks to Federated Learning via Multi-Round Consistency
description: >-
  [CVPR 2025][Optimization][federated learning] This work identifies that existing model poisoning attacks in federated learning cancel each other out due to cross-round directional inconsistency. It proposes PoisonedFL, which achieves a multi-round consistent attack through a fixed random direction vector, dynamic magnitude adjustment, and a hypothesis testing mechanism, bypassing 8 SOTA defenses without requiring any real client information.
tags:
  - "CVPR 2025"
  - "Optimization"
  - "federated learning"
  - "model poisoning"
  - "multi-round consistency"
  - "untargeted attack"
  - "Byzantine robustness"
date: 2026-05-08
content_hash: 4a715132c56d41b2
---

# Model Poisoning Attacks to Federated Learning via Multi-Round Consistency

**Conference**: CVPR 2025  
**arXiv**: [2404.15611](https://arxiv.org/abs/2404.15611)  
**Code**: [xyq7/PoisonedFL](https://github.com/xyq7/PoisonedFL/)  
**Area**: Optimization  
**Keywords**: federated learning, model poisoning, multi-round consistency, untargeted attack, Byzantine robustness

## TL;DR

This work identifies that existing model poisoning attacks in federated learning cancel each other out due to cross-round directional inconsistency. It proposes PoisonedFL, which achieves a multi-round consistent attack through a fixed random direction vector, dynamic magnitude adjustment, and a hypothesis testing mechanism, bypassing 8 SOTA defenses without requiring any real client information.

## Background & Motivation

**Background**: Federated Learning (FL) allows multiple clients to collaboratively train models without sharing raw data. However, its distributed nature makes it inherently vulnerable to model poisoning attacks, where malicious clients upload carefully crafted gradient updates to disrupt the global model.

**Core Problem**: Existing model poisoning attacks suffer from two major limitations:
1. **Suboptimal performance**: Existing attacks (e.g., LIE, Fang, Min-Max) only ensure malicious update consistency within a single round. The attack directions (+1/-1) of parameters frequently flip across multiple rounds (high flipping rate), leading to mutual cancellation of attack effects over time.
2. **Overly strong assumptions**: Most attacks require acquiring real client model updates or local data, requiring large-scale compromise of real client devices.

**Key Insight**: This paper observes that when a model is shifted extensively along a random direction, its accuracy degrades severely. Therefore, if the aggregated updates across multiple rounds can accumulate along the same random direction, the final model will eventually be pushed to random-guess levels. In contrast, existing attacks fail to achieve this due to direction inconsistency, which significantly weakens the multi-round cumulative effect.

## Method

### Overall Architecture

The core idea of PoisonedFL is to transmit direction-consistent malicious updates $\mathbf{g}_i^t = \mathbf{k}^t \odot \mathbf{s}$ from all fake clients, where $\mathbf{s}$ is a randomly chosen and fixed direction vector (with each dimension being +1 or -1) initialized at the start, and $\mathbf{k}^t$ is a dynamically adjusted non-negative magnitude vector. Direction consistency ensures that malicious updates accumulate over multiple rounds instead of canceling out, while the dynamic magnitude adjustment prevents the updates from being filtered by defenses.

### Key Design 1: Multi-Round Consistency Optimization Objective

- **All-round optimization**: Maximize the total accumulated update $\|\sum_{t=1}^T \mathbf{g}^t\|$ subject to $\text{sign}(\sum_t \mathbf{g}^t) = \mathbf{s}$.
- **Reformulation to round-by-round**: Impose the constraint $\text{sign}(\mathbf{g}^t) = \mathbf{s}$ in each round to ensure the aggregated updates point in the same direction, naturally accumulating the magnitude.
- **Core Difference**: The aggregation directions of existing attacks frequently flip across rounds, canceling out the overall effect. PoisonedFL maintains direction consistency to superimpose the attack effects round by round.

### Key Design 2: Dynamic Magnitude Adjustment

The magnitude vector $\mathbf{k}^t = \lambda^t \cdot \mathbf{v}^t$ is decomposed into two parts:

- **Unit magnitude vector $\mathbf{v}^t$**: This is estimated by subtracting the normalized malicious update from the previous global model difference $\mathbf{g}^{t-1}$ to approximate the genuine clients' distribution. This aligns the dimensional distribution of the malicious updates with that of the benign clients, evading anomaly detection.
- **Scaling factor $\lambda^t = c^t \cdot \|\mathbf{w}^{t-1} - \mathbf{w}^{t-2}\|$**: It is set proportional to the magnitude of the aggregated update from the previous round, preventing the update from being excessively large or small.

### Key Design 3: Adaptive $c^t$ Regulation via Hypothesis Testing

- **Null hypothesis $H_0$**: The attack has not succeeded in the past $e$ rounds (i.e., the direction $\mathbf{s}$ has not manifested in the aggregated updates).
- **Alternative hypothesis $H_1$**: The attack was successful.
- **Testing method**: Count the number of dimensions $X$ in $\mathbf{w}^{t-1} - \mathbf{w}^{t-e}$ that align with the direction of $\mathbf{s}$. Under $H_0$, $X \sim Bin(d, 0.5)$. If $p > 0.01$, the alternative hypothesis $H_1$ is rejected, lowering the magnitude to $c^t = \beta \cdot c^{t-1}$ ($\beta < 1$) to improve stealthiness.
- **Function**: Adjusts the attack strength adaptively without knowing which defense is deployed on the server.

### Loss & Training

Since this paper introduces an attack method, there is no traditional training loss. The core is to solve the optimization problem round-by-round:
$$\max_{\mathbf{g}_i^t} \|\mathbf{g}^t\|, \quad \text{s.t.} \; \text{sign}(\mathbf{g}^t) = \mathbf{s}$$

This is solved approximately using the three components described above.

## Key Experimental Results

### Main Results

PoisonedFL vs. 7 attacks × 9 defenses × 5 datasets. A higher test error rate (%) indicates a more successful attack (Part of Table 1):

| Defenses \ Attacks | No Attack | Fang | LIE | Min-Max | MPAF | **PoisonedFL** |
|---|---|---|---|---|---|---|
| **MNIST-FedAvg** | 2.11 | 13.66 | 2.28 | 97.89 | 90.04 | **90.02** |
| **MNIST-Multi-Krum** | 2.13 | 5.98 | 2.34 | 6.23 | 2.80 | **75.28** |
| **MNIST-FLTrust** | 3.43 | 4.00 | 3.41 | 12.56 | 3.43 | **88.65** |
| **MNIST-FLAME** | 2.86 | 2.66 | 2.61 | 2.60 | 2.72 | **88.59** |
| **MNIST-FLCert** | 3.34 | 4.61 | 2.83 | 4.57 | 6.46 | **88.06** |
| **FashionMNIST-FLTrust** | 16.73 | 17.52 | 12.50 | 21.32 | 16.83 | **88.41** |
| **Purchase-Multi-Krum** | 11.09 | 16.78 | 11.87 | 14.56 | 12.11 | **73.59** |

- PoisonedFL achieves the highest or near-highest test error rates across all dataset × defense combinations.
- Especially against the strongest defenses (FLTrust, FLAME, FLCert) where other attacks are almost completely ineffective, PoisonedFL still achieves 70-90% error rates.

### Ablation Study

- Removing multi-round consistency (i.e., changing the random direction every round) leads to a significant performance drop.
- Uniform magnitude (all dimensions of $\mathbf{v}^t$ are identical) vs. dynamic $\mathbf{v}^t$ vs. complete method: Dynamic estimation significantly outperforms uniform magnitude.
- Removing hypothesis testing (fixed $c^t$): Reduced effectiveness against certain defenses.

### Key Findings

1. **FL systems are far more vulnerable than previously assumed**: Even with SOTA defenses deployed, PoisonedFL (which operates without any information on benign clients) can still degrade the model to a random guess.
2. The fundamental reason for the failure of prior attacks is not insufficient strength per round, but rather the cross-round direction inconsistency that cancels out the cumulative effect.
3. Even if targeted defenses are designed (e.g., detecting direction consistency), PoisonedFL can still evade them through fine-tuning $\mathbf{s}$.
4. FLDetector (which detects multi-round inconsistency) is completely ineffective against PoisonedFL because PoisonedFL's malicious updates are inherently consistent across rounds.

## Highlights & Insights

1. **Deep Insight**: Identifies the fundamental flaw of "mutual cancellation" in existing attacks, assessing the cumulative effects of multi-round training from a global perspective. The idea is simple yet highly impactful.
2. **Minimal Assumptions**: Only requires injecting fake clients, without necessitating knowledge of defenses or access to real client data — representing the most realistic threat model.
3. **Elegant Use of Hypothesis Testing**: Leverages statistical methods to automatically judge whether the attack is filtered, eliminating the need for any assumptions about defenses.
4. **Extreme Simplicity**: The core mechanism is just fixing a random direction vector and keeping it consistent across rounds. It is surprisingly simple but achieves devastating results.

## Limitations & Future Work

1. Primarily evaluated on untargeted attack scenarios; its applicability to targeted attacks (backdoor attacks) remains unexplored.
2. Evaluations are mainly focused on image classification; its efficacy in other domains such as NLP and recommendation systems requires further validation.
3. The adversary needs to be able to inject fake clients, which may not be feasible in certain closed FL systems.
4. Does not consider additional defense mechanisms like Differential Privacy, which might mitigate the attack effects.

## Related Work & Insights

- **LIE/Fang/Min-Max/Min-Sum**: Attacks that require real client information. This work demonstrates that even with such advantages, they underperform compared to PoisonedFL.
- **MPAF**: An attack that also does not require real client information, but lacks multi-round consistency, resulting in vastly inferior performance.
- **FLTrust/FLAME/FLCert**: Various SOTA defenses, all of which are bypassed by PoisonedFL.
- **Insight**: Taking the "cumulative effect" perspective in security research is crucial. Perturbations that seem mild in a single round can lead to catastrophic impacts when accumulated over multiple rounds via direction consistency. This suggests that defenders must also focus on consistency detection across temporal dimensions.

## Rating

⭐⭐⭐⭐ — Deep insight (identifying the multi-round self-cancellation issue), with an extremely simple yet devastatingly effective method. Bypasses all SOTA defenses under the weakest assumptions, sounding an alarm for the FL security community. One star is deducted because it only focuses on untargeted attacks, and the experimental settings are somewhat academic (small models + standard datasets).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](../../NeurIPS2025/optimization/efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[CVPR 2025\] Federated Learning with Domain Shift Eraser](federated_learning_with_domain_shift_eraser.md)
- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](../../NeurIPS2025/optimization/fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)
- [\[CVPR 2025\] SCOPE: Semantic Coreset with Orthogonal Projection Embeddings for Federated Learning](scope_semantic_coreset_with_orthogonal_projection_embeddings_for_federated_learn.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](../../CVPR2026/optimization/dc-merge_improving_model_merging_with_directional_consistency.md)

</div>

<!-- RELATED:END -->
