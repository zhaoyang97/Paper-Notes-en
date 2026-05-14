---
title: >-
  [Paper Note] Provable Watermarking for Data Poisoning Attacks
description: >-
  [NeurIPS 2025][AI Safety][watermarking] This paper proposes two provable watermarking schemes—post-poisoning watermarking and poisoning-concurrent watermarking—that provide transparency declaration mechanisms for data po…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "watermarking"
  - "data poisoning"
  - "backdoor attack"
  - "availability attack"
  - "provable guarantee"
date: 2026-05-08
content_hash: b7f4b0437af321e7
---

# Provable Watermarking for Data Poisoning Attacks

**Conference**: NeurIPS 2025
**arXiv**: [2510.09210](https://arxiv.org/abs/2510.09210)
**Code**: None
**Area**: AI Security
**Keywords**: watermarking, data poisoning, backdoor attack, availability attack, provable guarantee

## TL;DR

This paper proposes two provable watermarking schemes—post-poisoning watermarking and poisoning-concurrent watermarking—that provide transparency declaration mechanisms for data poisoning attacks. The theoretical analysis demonstrates that, under specific watermark length conditions, both watermark detectability and poisoning effectiveness can be simultaneously guaranteed.

## Background & Motivation

1. **The dual nature of data poisoning**: Data poisoning attacks have traditionally been regarded as security threats, but have increasingly been adopted for legitimate purposes, including dataset ownership verification (backdoor attacks), prevention of unauthorized data use (unlearnable examples), and artistic copyright protection (NightShade/Glaze).

2. **Risk of misuse**: Authorized users may inadvertently train on poisoned data, leading to misunderstandings and disputes. Poisoning generators therefore bear a responsibility to transparently disclose the presence of poisoning traces to authorized users.

3. **Limitations of existing detection methods**: Existing backdoor and availability attack detection methods are diverse and mutually incompatible, making it difficult to distribute them as a unified framework to authorized users. More critically, these methods largely rely on heuristic training algorithms and lack provable poisoning declaration mechanisms.

4. **Applicability of watermarking**: Watermarking techniques have been widely applied in copyright protection and AI-generated content detection, making them a promising solution that allows poisoning generators to declare the existence of poisoning in a provable manner.

5. **Two practical scenarios**: Two distinct requirements arise in practice—one in which a poisoning generator delegates watermark creation to a third-party entity (post-poisoning watermarking), and one in which the generator simultaneously produces both the watermark and the poisoning (poisoning-concurrent watermarking).

6. **Theoretical gap**: Despite extensive independent research on data poisoning and watermarking, applying watermarking schemes to data poisoning attacks remains unexplored, with no existing theoretical guarantees for such a setting.

## Method

### Overall Architecture

The paper defines data poisoning as a mapping $\delta^p: [0,1]^d \to [-\epsilon_p, \epsilon_p]^d$ and watermarking as a mapping $\delta^w: [0,1]^d \to [-\epsilon_w, \epsilon_w]^q$, where $q \leq d$ denotes the watermark length. The detection mechanism uses a secret key $\zeta$ to compute the inner product $\zeta^T x$ to determine whether a sample has been watermarked.

### Post-Poisoning Watermarking

- **Scenario**: A third-party entity generates watermarks for an already-poisoned dataset.
- **Total perturbation**: $\delta_x = \delta_x^p + \delta_x^w$, with perturbation budget $\epsilon_w + \epsilon_p$.
- **Sample-level theory** (Theorem 4.1): When the watermark length $q > \frac{1}{\epsilon_w}\sqrt{2d\log\frac{1}{\omega}}$, i.e., $q = \Omega(\sqrt{d}/\epsilon_w)$, watermarked data can be distinguished from clean data with high probability.
- **Universal version** (Theorem 4.9): When $q = \Theta(\sqrt{d}/\epsilon_w)$ and sample size $N = \Omega(d)$, the guarantee holds for the majority of samples and extends to the entire distribution.

### Poisoning-Concurrent Watermarking

- **Scenario**: The watermark generator is also the poisoning generator, enabling control over dimension allocation.
- **Dimension separation**: Watermarking and poisoning operate on disjoint dimensions, with poisoning dimensions $\mathcal{P} = [d] \setminus \mathcal{W}$.
- **Total perturbation**: $\max\{\epsilon_w, \epsilon_p\}$, which is smaller than the post-poisoning case of $\epsilon_w + \epsilon_p$.
- **Sample-level theory** (Theorem 4.3): A watermark length of $q = \Omega(1/\epsilon_w^2)$ suffices to ensure detectability, requiring a shorter watermark than the post-poisoning scheme.
- **Universal version** (Theorem 4.13): $q = \Theta(1/\epsilon_w^2)$ is sufficient to hold for the majority of samples.

### Poisoning Effectiveness Guarantees

- **Post-poisoning case** (Theorem 5.2): For an $L$-layer feedforward network with Xavier initialization, when $d$ and $N$ are sufficiently large, the influence of watermarking on poisoning effectiveness can be bounded by a negligible error term, without additional constraints on $q$.
- **Concurrent case** (Theorem 5.6): An additional condition $q = O(\sqrt{d}/\epsilon_p)$ is required; otherwise, the watermark dominates and significantly degrades poisoning effectiveness.
- **Final watermark length ranges**: Post-poisoning uses $\Theta(\sqrt{d}/\epsilon_w)$; concurrent uses $\Theta(1/\epsilon_w^2)$ to $O(\sqrt{d}/\epsilon_p)$.

### Threat Model Instantiation

Using an autonomous driving dataset as a running example, the paper constructs a complete deployment scenario: data owner Alice publishes a dataset with both poisoning and watermarking; authorized user Bob obtains the key via a secure channel to verify identity and remove poisoning; malicious user Chad can neither successfully train a model nor remove the watermark. An HMAC-based key management scheme is also designed to prevent forgery attacks.

## Key Experimental Results

### Experimental Setup

- **Attack methods**: Backdoor attacks (Narcissus, AdvSc) + availability attacks (UE, AP)
- **Datasets**: CIFAR-10, CIFAR-100, Tiny-ImageNet
- **Models**: ResNet-18, ResNet-50, VGG-19, DenseNet121, WRN34-10, MobileNet v2
- **Watermark length**: 0–3000; watermark/poisoning budgets are 16/255 (backdoor) and 8/255 (availability)

### Backdoor Attack Watermarking Results (Table 1 — ResNet-18, CIFAR-10)

| Watermark Length $q$ | Narcissus Post-Poisoning Acc/ASR/AUROC | Narcissus Concurrent Acc/ASR/AUROC | AdvSc Post-Poisoning Acc/ASR/AUROC | AdvSc Concurrent Acc/ASR/AUROC |
|:---:|:---:|:---:|:---:|:---:|
| 0 (Baseline) | 94.69/95.04/- | 94.69/95.04/- | 92.80/95.53/- | 92.80/95.53/- |
| 500 | 94.95/93.11/0.9509 | 94.70/95.03/0.9968 | 93.18/97.43/0.9218 | 92.89/95.79/0.9986 |
| 1000 | 94.40/92.43/0.9974 | 94.32/92.03/0.9992 | 93.05/94.41/0.9809 | 93.38/84.39/0.9995 |
| 2000 | 94.55/90.37/1.0000 | 94.89/22.46/1.0000 | 93.40/79.97/0.9994 | 92.38/30.05/1.0000 |
| 3000 | 94.93/90.02/1.0000 | 94.72/9.75/1.0000 | 93.10/74.82/1.0000 | 93.04/9.97/1.0000 |

### Availability Attack Watermarking Results (Table 2 — ResNet-18, CIFAR-10)

| Watermark Length $q$ | UE Post-Poisoning Acc↓/AUROC↑ | UE Concurrent Acc↓/AUROC↑ | AP Post-Poisoning Acc↓/AUROC↑ | AP Concurrent Acc↓/AUROC↑ |
|:---:|:---:|:---:|:---:|:---:|
| 0 (Baseline) | 10.79/- | 10.79/- | 8.53/- | 8.53/- |
| 500 | 11.71/0.7810 | 10.02/0.9930 | 8.71/0.8623 | 15.84/0.8931 |
| 1000 | 11.37/0.9499 | 9.42/0.9991 | 10.58/0.9742 | 21.87/0.9949 |
| 2000 | 9.06/0.9992 | 10.03/1.0000 | 10.48/0.9987 | 38.62/1.0000 |
| 3000 | 9.99/1.0000 | 91.79/1.0000 | 13.52/1.0000 | 93.40/1.0000 |

**Key Findings**:
- Post-poisoning watermarking: poisoning effectiveness is well-preserved across all values of $q$ (ASR declines only marginally), validating Theorem 5.2.
- Poisoning-concurrent watermarking: poisoning effectiveness degrades sharply at large $q$ (e.g., ASR drops to ~22–30% at $q=2000$), validating the necessity of the condition $q = O(\sqrt{d}/\epsilon_p)$.
- Detection performance (AUROC) of the concurrent scheme consistently surpasses that of the post-poisoning scheme at equivalent $q$, corroborating the theoretical conclusion that $\Omega(1/\epsilon_w^2) < \Omega(\sqrt{d}/\epsilon_w)$.

## Highlights & Insights

- **Pioneering contribution**: This work is the first to introduce watermarking into the domain of data poisoning attacks, providing transparency guarantees for "poisoning for good" use cases.
- **Theoretical rigor**: A complete theoretical guarantee chain is established, from sample-level to universal and distributional settings, with precise conditions ensuring both watermark detectability and poisoning effectiveness.
- **Complementary schemes**: The post-poisoning scheme suits third-party trust scenarios (where larger watermark lengths are preferable), while the concurrent scheme is tailored for self-administered poisoning (requiring a balance between watermark and poisoning length).
- **End-to-end deployment design**: Beyond theory and algorithms, the paper also proposes engineering-level security measures such as HMAC-based key management.

## Limitations & Future Work

- **Sufficient but not necessary conditions**: The theory provides only sufficient conditions on watermark length; necessary conditions remain an open problem.
- **Linear detection mechanism**: Detection relies on a simple inner-product threshold, which may lack robustness.
- **Narrow valid range in concurrent watermarking**: The effective range of $q$, i.e., $[\Theta(1/\epsilon_w^2), O(\sqrt{d}/\epsilon_p)]$, may be restrictively narrow in practice.
- **Limited adversarial robustness**: Although the appendix discusses defenses such as data augmentation, image regeneration, differentially private noise, and diffusion-based purification, a systematic robustness analysis is not conducted in depth.

## Related Work & Insights

| Dimension | Ours | Li et al. (2023) Backdoor Dataset Watermarking |
|:---|:---|:---|
| Watermark–poisoning relationship | Explicitly separates watermarking from poisoning; proposes two schemes | Treats the backdoor attack itself as the watermark signal |
| Theoretical guarantees | Provides provable conditions for detectability and poisoning effectiveness | Lacks formal theoretical guarantees |
| Attack coverage | Unified framework for backdoor and availability attacks | Limited to backdoor attacks |
| Generality | General; does not depend on specific attack algorithms | Tightly coupled to specific backdoor attack methods |

| Dimension | Ours | NightShade/Glaze |
|:---|:---|:---|
| Objective | Declaring the presence of poisoning; providing transparency | Protecting artists' copyright from being learned by generative models |
| Watermark mechanism | Detectable watermark independent of poisoning | No explicit watermark mechanism |
| Theoretical framework | Rigorous probabilistic argumentation | Empirical approach |
| Detection capability | Authorized users can verify poisoning | No active detection mechanism |

## Rating

- ⭐⭐⭐⭐ **Novelty**: First to introduce watermarking into data poisoning scenarios; problem formulation is novel and practically motivated.
- ⭐⭐⭐⭐ **Technical Depth**: Complete theoretical chain from sample-level to distributional guarantees; proofs are rigorous.
- ⭐⭐⭐⭐ **Experimental Thoroughness**: Covers 4 attack types, multiple models and datasets, with thorough ablation studies.
- ⭐⭐⭐ **Practicality**: Theory is promising, but constraints on the valid watermark length range and robustness concerns limit real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning](fedfact_a_provable_framework_for_controllable_group-fairness_calibration_in_fede.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](private_zeroth-order_optimization_with_public_data.md)
- [\[NeurIPS 2025\] Incentivizing Time-Aware Fairness in Data Sharing](incentivizing_time-aware_fairness_in_data_sharing.md)
- [\[NeurIPS 2025\] Factor Decorrelation Enhanced Data Removal from Deep Predictive Models](factor_decorrelation_enhanced_data_removal_from_deep_predictive_models.md)
- [\[NeurIPS 2025\] Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)

</div>

<!-- RELATED:END -->
