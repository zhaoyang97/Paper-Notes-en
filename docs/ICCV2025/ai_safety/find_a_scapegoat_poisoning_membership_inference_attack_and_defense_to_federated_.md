---
title: >-
  [Paper Note] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning
description: >-
  [ICCV 2025][AI Safety][Federated Learning] This paper proposes FedPoisonMIA, a poisoning-based membership inference attack for federated learning that maximizes angular deviation, along with a defense mechanism called Angular Trimmed-mean (ATM) that filters malicious gradients via angular distance.
tags:
  - ICCV 2025
  - AI Safety
  - Federated Learning
  - Membership Inference Attack
  - Poisoning Attack
  - Byzantine-Robust Aggregation
  - Privacy Preservation
date: 2026-05-08
content_hash: b850e0801536c4f9
---

# Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning

**Conference**: ICCV 2025
**arXiv**: [2507.00423](https://arxiv.org/abs/2507.00423)
**Code**: None
**Area**: AI Safety / Federated Learning Privacy
**Keywords**: Federated Learning, Membership Inference Attack, Poisoning Attack, Byzantine-Robust Aggregation, Privacy Preservation

## TL;DR

This paper proposes FedPoisonMIA, a poisoning-based membership inference attack for federated learning that maximizes angular deviation, along with a defense mechanism called Angular Trimmed-mean (ATM) that filters malicious gradients via angular distance.

## Background & Motivation

**Background**: Federated Learning (FL) enables multiple clients to collaboratively train a global model without sharing raw data, and is regarded as a key paradigm for privacy preservation. However, the distributed nature of FL makes it vulnerable to poisoning attacks, where malicious clients can submit carefully crafted updates to influence the global model.

**Limitations of Prior Work**: Most poisoning attacks in FL focus on compromising model integrity (e.g., degrading accuracy), with insufficient attention to privacy attacks. Poisoning Membership Inference Attacks (PMIA) manipulate gradients to infer whether a specific sample exists in a benign client's dataset, but existing PMIA methods have limited effectiveness against Byzantine-robust aggregation mechanisms.

**Key Challenge**: PMIA is inherently harder to design than conventional poisoning attacks — it must extract private information without significantly degrading global model performance (which would trigger detection). Gradients computed with wrong labels produce excessively large angular deviations, making them易 to filter by robust aggregation rules.

**Key Insight**: On the attack side, the paper crafts malicious gradients to maximize angular deviation while ensuring it does not exceed the maximum angular difference among benign gradients; on the defense side, angular deviation serves as the discriminative criterion for filtering malicious updates.

**Core Idea**: The attack is formulated as a constrained optimization problem: maximize the angular deviation between the malicious gradient and benign gradients, subject to the constraint that this deviation does not exceed the maximum pairwise angle among benign gradients. A greedy algorithm selects mask samples to conceal the attack gradient, and a scaling coefficient $\alpha$ is jointly optimized.

## Method

### Overall Architecture

**Attack Framework — FedPoisonMIA**: The attacker holds an attack sample set $D_{attack}$ and a mask sample set $D_{mask}$, and constructs the final malicious gradient as $g_{malicious} = \alpha \cdot g_{attack} + g_{mask}$, where $g_{attack}$ is computed from wrongly labeled samples and $g_{mask}$ is computed from carefully selected correctly labeled samples.

**Defense Framework — ATM**: The server computes the average pairwise angle between each gradient and all other gradients, removes the $2b$ gradients with the largest average angles, and aggregates the remainder by taking the mean.

### Key Designs

1. **Attack Gradient Generation (Step I)**: The true labels of samples in $D_{attack}$ are replaced with random incorrect labels, and the gradient $g_{attack}$ is computed accordingly. The wrong labels cause the loss on the target samples to increase; if a benign client holds the same samples (with correct labels), the loss is pulled back, enabling membership inference via the loss discrepancy.

2. **Greedy Mask Sample Selection (Step II)**: A subset $\hat{D}_{mask}$ is selected from $D_{mask}$ by solving:
$$\arg\max_{\hat{D}_{mask}} \max_{i \in \mathcal{B}} \angle(g_{malicious}, g_{benign}^i)$$
subject to: $\max_{i \in \mathcal{B}} \angle(g_{malicious}, g_{benign}^i) \leq \max_{i,j \in \mathcal{B}} \angle(g_{benign}^i, g_{benign}^j)$

   Since exhaustive enumeration of all subsets is NP-hard, a greedy strategy is adopted that iteratively adds the sample maximizing the objective, until the target count $\lfloor \gamma |D_{mask}| \rfloor$ is reached.

3. **Scaling Coefficient Optimization (Step III)**: With $\hat{D}_{mask}$ fixed, $\alpha$ is optimized to maximize angular deviation while satisfying the angular constraint.

4. **ATM Defense Mechanism**:
   - Compute pairwise angles $\theta_{i,j}$ among all gradients
   - Compute the average angle for each gradient: $\bar{\theta}_k = \frac{1}{|G|-1} \sum \theta_{k,l}$
   - Sort gradients by average angle in ascending order and remove the $2b$ with the largest values
   - Aggregate the remaining gradients by averaging

### Loss & Training

ATM aggregation formula: $\bar{g} = \frac{1}{|G| - 2b} \sum_{g \in G'} g$, where $G'$ is the set of gradients after removing outliers.

Convergence guarantee (Theorem 1): The $L_2$ deviation of ATM is upper-bounded by $\frac{2(n-m)(b+1)\sigma^2}{(n-b-m)^2}$, ensuring the aggregated result converges toward the true mean.

## Key Experimental Results

### Main Results: Attack Accuracy (C=0.8, Full-knowledge)

| Dataset | Attack | FedAvg | Median | Trimmed-mean | ATM |
|--------|---------|--------|--------|-------------|-----|
| Texas100 IID | Passive | 0.643 | 0.583 | 0.630 | 0.600 |
| Texas100 IID | GA | 0.826 | 0.820 | 0.810 | 0.766 |
| Texas100 IID | AGREvader | 0.804 | 0.761 | 0.756 | 0.741 |
| Texas100 IID | **FedPoisonMIA** | **0.897** | **0.913** | **0.880** | **0.803** |
| CIFAR-10 IID | **FedPoisonMIA** | **0.913** | **0.753** | **0.857** | **0.713** |

FedPoisonMIA achieves the highest attack accuracy across all datasets and defenses. ATM consistently reduces attack accuracy to the lowest level among all evaluated aggregation rules.

### Ablation Study: Greedy vs. Random Selection (Texas100)

| Selection Strategy | FedAvg | Median | Trimmed-mean | ATM |
|---------|--------|--------|-------------|-----|
| Random | 0.771 | 0.744 | 0.751 | 0.751 |
| **Greedy** | **0.884** | **0.807** | **0.884** | **0.880** |

Greedy mask selection improves attack accuracy by over 20% compared to random selection, validating the critical role of carefully chosen mask samples.

### Key Findings

- **Attack Strength**: FedPoisonMIA improves attack accuracy by 8.0–16.0% over the best baseline under Texas100 Non-IID, and consistently achieves the best performance under more complex defenses such as Fang, Multi-Krum, and DeepSight.
- **Effectiveness under Partial Knowledge**: When only the malicious client's own gradients are available (no access to benign gradients), attack accuracy drops by at most 8.0%.
- **Defense Effectiveness**: ATM achieves the lowest attack accuracy in most settings without introducing additional computation overhead on clients. ATM sorts on the client-count dimension (tens of clients), whereas Median/Trimmed-mean sort on the parameter dimension (hundreds of thousands), giving ATM a significant computational advantage.
- **IID vs. Non-IID**: Attack accuracy is higher under IID settings, as gradient consistency makes it easier for the attack to exploit benign updates.

## Highlights & Insights

- **Unified Attack-Defense Design**: The attack formulation (maximizing angular deviation) naturally motivates the defense strategy (filtering based on angular deviation), yielding a logically coherent framework.
- **Practical Threat Model**: The attacker does not need to know which aggregation rule the server uses, and remains effective under partial-knowledge settings.
- **Computational Efficiency of ATM**: By sorting on the client-count dimension rather than the parameter dimension, ATM offers substantially lower overhead compared to Median and Trimmed-mean.

## Limitations & Future Work

- ATM reduces attack accuracy to the 0.7–0.8 range but does not fully eliminate the risk of privacy leakage, leaving room for further improvement.
- The attack requires the attacker to possess target samples or similar ones; the practical difficulty of acquiring such samples is not thoroughly discussed.
- Adaptive attacks (with knowledge of ATM) can still achieve non-trivial accuracy, though lower than FedPoisonMIA.
- Attack effectiveness degrades under asynchronous FL but remains significant, indicating the need for stronger defenses in that setting.

## Related Work & Insights

- AGREvader [Zhang 2023] also employs mask gradients to conceal the attack, but uses Euclidean distance as the constraint; FedPoisonMIA adopts angular distance instead, achieving better results.
- Traditional Byzantine-robust methods (Median, Trimmed-mean, Multi-Krum) are primarily designed to defend against data/model integrity attacks, and offer limited protection against membership inference attacks — a largely overlooked security dimension.
- Differential privacy (DP) can also reduce attack effectiveness but at the cost of significant degradation in model accuracy.

## Rating

- Novelty: ⭐⭐⭐⭐ — The angular-deviation maximization attack paradigm is novel, with a coherent attack-defense co-design
- Technical Depth: ⭐⭐⭐⭐ — Rigorous optimization formulation and convergence analysis
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 datasets, 8+ defense mechanisms, IID/Non-IID, synchronous/asynchronous, multiple metrics
- Practical Value: ⭐⭐⭐⭐ — Exposes a realistic and previously underappreciated privacy threat in federated learning

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](fedvla_federated_vision-language-action_learning_with_dual_gating_mixture-of-exp.md)
- [\[ICCV 2025\] Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning](active_membership_inference_test_amint_enhancing_model_auditability_with_multi-t.md)
- [\[ICLR 2026\] Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning](../../ICLR2026/ai_safety/hide_and_find_a_distributed_adversarial_attack_on_federated_graph_learning.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_double-blind_federated_adaptation_of_foundation_models.md)

<!-- RELATED:END -->
