---
title: >-
  [Paper Note] Towards Trustworthy Federated Learning with Untrusted Participants
description: >-
  [ICML 2025][AI Safety][Federated Learning] The CafCor algorithm is proposed, which injects correlated noise achieved through shared randomness among participants and integrates a novel robust aggregation method called CAF. This achieves a privacy-utility trade-off close to Central Differential Privacy (CDP), without trusting the server and in the presence of malicious participants.
tags:
  - "ICML 2025"
  - "AI Safety"
  - "Federated Learning"
  - "Differential Privacy"
  - "Robust Aggregation"
  - "Byzantine Fault Tolerance"
  - "Correlated Noise"
date: 2026-05-08
content_hash: 213beeee831032f5
---

# Towards Trustworthy Federated Learning with Untrusted Participants

**Conference**: ICML 2025  
**arXiv**: [2505.01874](https://arxiv.org/abs/2505.01874)  
**Code**: None (the paper mentions plans to open-source)  
**Area**: AI Security  
**Keywords**: Federated Learning, Differential Privacy, Robust Aggregation, Byzantine Fault Tolerance, Correlated Noise

## TL;DR

The CafCor algorithm is proposed, which injects correlated noise achieved through shared randomness among participants and integrates a novel robust aggregation method called CAF. This achieves a privacy-utility trade-off close to Central Differential Privacy (CDP), without trusting the server and in the presence of malicious participants.

## Background & Motivation

In distributed learning, **privacy** and **robustness** are two critical requirements for building trustworthy systems, but their intersection remains under-explored.

Existing solutions face a fundamental trade-off:
- **Local Differential Privacy (LDP)**: Does not trust the server, but suffers from extreme utility loss, with a privacy-utility trade-off of $\Theta(d/(n\epsilon^2))$.
- **Central Differential Privacy (CDP)**: Assumes a fully trusted server, achieving much better utility with a trade-off of $\Theta(d/(n^2\epsilon^2))$.
- CDP is $n$ times better than LDP, but fully trusting the server is unrealistic in practice.

**Core Problem**: Can a utility close to CDP be achieved without trusting the server and in the presence of malicious participants?

The **SecLDP** (Secret-based Local Differential Privacy) model proposed in this paper is key: each pair of participants shares a random seed unknown to others. This can be established via a single round of encrypted communication, which is a much weaker trust assumption than CDP.

## Method

### Overall Architecture

**CafCor** = Correlated noise + CAF aggregation, consisting of two core components:

1. **Correlated Noise Injection**: Utilizes shared randomness among participants.
2. **Covariance-bound Agnostic Filter (CAF) Aggregation**: A novel robust aggregation method.

Workflow (Algorithm 1):
- The server broadcasts the model $\theta_t$.
- Each honest worker $i$ samples a mini-batch and computes the clipped gradient $g_t^{(i)}$.
- Two types of noise are injected: independent noise $\bar{v}_t^{(i)} \sim N(0, \sigma_{\text{ind}}^2 I_d)$ + correlated noise $\sum_j v_t^{(ij)}$, where $v_t^{(ij)} = -v_t^{(ji)}$.
- After adding momentum, the messages are sent to the server.
- The server aggregates the received messages using CAF and updates the model.

### Key Designs

**1. Correlated Noise Mechanism**

Core Idea: Each pair of workers $(i,j)$ shares a random seed to generate symmetric noise $v_t^{(ij)} = -v_t^{(ji)} \sim N(0, \sigma_{\text{cor}}^2 I_d)$.

- **Cancellation upon Aggregation**: When averaging gradients across all honest workers, the correlated noise cancels out.
- **Privacy Protection**: The server cannot obtain the shared seeds, so individual messages from workers appear highly noisy.
- **Collusion Robustness**: Even if malicious actors collude, as long as there is at least one non-colluding honest worker, the correlated noise shared with them provides privacy.

The required noise scale is only $\sigma_{\text{cor}}^2 = \Theta(1/(n\epsilon^2))$, whereas LDP requires $\sigma_{\text{ind}}^2 = \Theta(1/\epsilon^2)$—a difference of $n$ times.

**2. CAF Aggregation (Algorithm 2)**

CAF filters out malicious inputs via iterative downweighting:
- Initialize all input weights to 1.
- In each round, calculate the weighted mean and empirical covariance matrix.
- Find the direction corresponding to the largest eigenvalue (direction of maximum variance).
- Reduce the weights of inputs based on their projection along this direction.
- Record the weighted mean corresponding to the minimum-maximum eigenvalue.
- Terminate when the total weight drops to $n-2f$.

**Key Advantages of CAF**:
- Does not assume any prior knowledge about the covariance of honest inputs (unlike Diakonikolas et al., 2017).
- The error bound depends on the maximum eigenvalue rather than the trace, which is $d$ times better than trimmed mean/median.
- Computational complexity is $O(f(nd^2 + d^3))$, which can be reduced to $O(fnd \log d)$ in practice using the power method.

**3. Momentum Update**

Workers maintain a local momentum $m_t^{(i)} = \beta_{t-1} m_{t-1}^{(i)} + (1-\beta_{t-1}) \tilde{g}_t^{(i)}$, sending momentum instead of raw gradients. The momentum coefficient is $\beta_t = 1 - 24L\gamma_t$.

### Loss & Training

- Assume the loss function is $L$-smooth, and can be strongly convex or non-convex.
- Gradient clipping $\text{Clip}(g; C) = g \cdot \min\{1, C/\|g\|\}$ with sensitivity bound $C$.
- Learning rate strategy: for strongly convex cases, $\gamma_t = 10/(\mu(t + 240L/\mu))$; for non-convex cases, $\gamma_t = \min\{1/(24L), \sqrt{3L_0}/(16\bar{\sigma}\sqrt{LT})\}$.

## Key Experimental Results

### Main Results

On MNIST and Fashion-MNIST, with $n=100$ workers and $f \in \{5,10\}$ malicious workers:

| Threat Model | MNIST (f=5) | MNIST (f=10) | Fashion-MNIST (f=5) |
|---|---|---|---|
| CDP (Baseline Upper Bound) | ~83% | ~82% | ~72% |
| CafCor-SecLDP | **83%** | ~78% | **72%** |
| CafCor-ByzLDP | ~75%| ~60% | ~71% |
| DSGD-LDP | ~30% | ~30% | ~50% |

CafCor-SecLDP approaches CDP performance when $f=5$, significantly outperforming LDP.

### Ablation Study

**Comparison of Aggregation Methods** (Figure 2, $n-f=10$ honest workers):
- CafCor (CAF) vs CWTM, CWMED, GM, Multi-Krum, Meamed.
- Under Sign Flipping and Label Flipping attacks:
    - CAF achieves optimal performance across all attacks, reaching near 90% accuracy under LF attacks.
    - CWTM/CWMED/GM only achieve around 50% accuracy under LF attacks.
    - Replacing CAF with a trimmed mean degrades the theoretical convergence rate by a factor of $n$.

### Key Findings

1. **Privacy-Utility Trade-off**: $O((f+1)d/(n^2\epsilon^2))$, which matches CDP's $\Theta(d/(n^2\epsilon^2))$ when $f=O(1)$.
2. **Strongly Convex Convergence Rate**: $O(\kappa G^2/\mu + L\bar{\sigma}^2/(\mu^2T) + L^2L_0/(\mu^2T^2))$.
3. **Non-Convex Convergence Rate**: $O(\kappa G^2 + \bar{\sigma}\sqrt{LL_0}/\sqrt{T} + LL_0/T)$.
4. Strictly outperforms LDP when $f$ is sublinear in $n$.

## Highlights & Insights

1. **Introduction of SecLDP Model**: Finds a practical middle ground between LDP and CDP, where the correlated noise scheme only requires a single round of pre-communication.
2. **Dimensional Advantage of CAF Aggregation**: The error bound does not include an extra dimension factor $d$, which is a key advantage over trimmed mean in high-dimensional scenarios.
3. **No Covariance Prior Required**: CAF only needs to know the upper bound of the number of malicious workers, without requiring assumptions on the statistical properties of honest inputs.
4. **Fine-grained Collusion Analysis**: The parameter $q$ in Theorem 4.1 allows precise control over scenarios with $0$ to $f$ colluding malicious actors.
5. **Graceful Degradation**: If shared randomness is unavailable, setting $\sigma_{\text{cor}}=0$ naturally falls back to LDP while maintaining SOTA performance.

## Limitations & Future Work

1. **Communication Cost**: Seed exchange requires $O(n^2)$ pairwise communications. Although this is a one-time process with a small payload, it may still present a bottleneck for large-scale systems.
2. **Optimality of SecLDP Unknown**: The paper poses "What is the optimal utility under SecLDP? Does CafCor achieve it?" as an open question.
3. **Privacy Budget Allocation**: The settings of $\sigma_{\text{ind}}$ and $\sigma_{\text{cor}}$ rely on assumptions about cell collusion status, which can be difficult to determine in practice.
4. **Simple Datasets**: Experiments are conducted on MNIST/Fashion-MNIST, with no validation on more complex datasets (e.g., CIFAR-10 or NLP tasks).
5. **$2f$ Iterations in CAF**: Although running in polynomial time, it still incurs overhead when $f$ is large.

## Related Work & Insights

- **Three-way Trade-off in LDP + Robustness** [Allouah et al., 2023c]: Proved a fundamental trade-off among privacy, robustness, and utility under LDP. This work breaks through this limitation.
- **Centralized Robust Mean Estimation** [Hopkins et al., 2022; Liu et al., 2021]: Achieved near-optimal error bounds under CDP.
- **Correlated Noise Schemes** [Sabater et al., 2022; Allouah et al., 2024]: Prior works did not provide utility guarantees under adversarial settings.
- **SMEA Aggregation** [Allouah et al., 2023c]: Satisfies the same high-dimensional robust criterion but has exponential computational complexity. CAF serves as its polynomial-time alternative.
- Provides a new paradigm for research that simultaneously addresses privacy and robustness in federated learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The combination of correlated noise and robust aggregation is unique, and the SecLDP model is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive comparison across multiple attack types, threat models, and aggregation methods.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous theory and clear experimental details.
- Value: ⭐⭐⭐⭐⭐ — Addresses the core pain points of privacy and robustness in federated learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[ICML 2025\] Generalization in Federated Learning: A Conditional Mutual Information Framework](generalization_in_federated_learning_a_conditional_mutual_information_framework.md)
- [\[ICLR 2026\] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](../../ICLR2026/ai_safety/beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)
- [\[ICLR 2026\] Convergent Differential Privacy Analysis for General Federated Learning](../../ICLR2026/ai_safety/convergent_differential_privacy_analysis_for_general_federated_learning.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](../../NeurIPS2025/ai_safety/mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)

</div>

<!-- RELATED:END -->
