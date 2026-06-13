---
title: >-
  [Paper Note] FLIPS: Instance-Fingerprinting for LLMs via Pseudo-Random Sequences
description: >-
  [ICML 2026][Social Computing][Model Fingerprinting] FLIPS generates unique model "fingerprint responses" by designing **pseudo-random seed sequences** (seeds known only to the model owner). Even if an attacker fine-tunes…
tags:
  - "ICML 2026"
  - "Social Computing"
  - "Model Fingerprinting"
  - "Pseudo-Random Sequences"
  - "Black-box Detection"
  - "Robust Fingerprinting"
date: 2026-05-08
content_hash: f9a33649da2d5bc6
---

# FLIPS: Instance-Fingerprinting for LLMs via Pseudo-Random Sequences

**Conference**: ICML 2026  
**arXiv**: [2605.29110](https://arxiv.org/abs/2605.29110)  
**Code**: To be confirmed  
**Area**: LLM Security / Model Watermarking / IP Protection  
**Keywords**: Model Fingerprinting, Pseudo-Random Sequences, Black-box Detection, Robust Fingerprinting

## TL;DR
FLIPS generates unique model "fingerprint responses" by designing **pseudo-random seed sequences** (seeds known only to the model owner). Even if an attacker fine-tunes or prunes the model, the fingerprint cannot be eliminated, achieving a detection rate $> 99\%$ and a false positive rate $< 1\%$ in black-box query scenarios.

## Background & Motivation

**Background**: LLMs are high-value intellectual property assets but are vulnerable to unauthorized replication, fine-tuning, and secondary distribution. Existing protection methods—watermarking (marking outputs), encryption (restricting access), and fingerprinting (identifying the original model)—each have limitations.

**Limitations of Prior Work**: (1) Existing fingerprinting methods lack robustness against model fine-tuning and pruning; (2) Most methods require white-box access, making them inapplicable to black-box API scenarios; (3) Backdoor-style fingerprints are easily detected and removed.

**Key Challenge**: Fingerprints require "uniqueness" (distinguishing from other models), "robustness" (resistance to modification), and "stealthiness" (not affecting normal use)—this triangular constraint is difficult to satisfy simultaneously.

**Goal**: Design a fingerprinting method that is black-box verifiable, resistant to fine-tuning/pruning, and does not compromise model performance.

**Key Insight**: It is observed that LLMs exhibit highly deterministic responses to **specific input sequences**. By constructing a pseudo-random yet deterministic "seed $\rightarrow$ fingerprint response" mapping, the existence of a fingerprint can be confirmed through black-box queries.

**Core Idea**: Cryptographic **pseudo-random sequences** are used as seeds to generate "probe sequences" $q_s$. The output $r_s$ of the original model on $q_s$ serves as the fingerprint. Attackers cannot locate fingerprint queries without knowing the seed.

## Method

### Overall Architecture
The method consists of two phases—(1) **Fingerprint Injection**: A pseudo-random probe $q_s = G(s)$ is generated based on a seed $s$; the original model $\mathcal{M}_0$ produces an output $r_s = \mathcal{M}_0(q_s)$ on $q_s$; a fingerprint library $\mathcal{F} = \{(q_s, r_s)\}$ is stored. (2) **Fingerprint Verification**: A suspicious model $\mathcal{M}^?$ generates an output $r^?_s$ for query $q_s$; model origin is determined by similarity $\text{sim}(r^?_s, r_s)$.

### Key Designs

1.  **Pseudo-random Probes + Stealthiness**:
    - **Function**: Construct fingerprint queries that attackers cannot identify and to which the model has deterministic responses.
    - **Mechanism**: A cryptographically secure PRG (e.g., AES-CTR) generates probes $q_s$ from seed $s$, with sufficient length to ensure a unique fingerprint response for each seed probabilistically.
    - **Design Motivation**: Traditional backdoor fingerprints use specific trigger words that are easily detected; PRG outputs are indistinguishable random strings to those without the seed, making it impossible to locate fingerprints.

2.  **Multi-probes + Robust Statistical Verification**:
    - **Function**: Significantly improve robustness and confidence through joint verification of multiple independent probes.
    - **Mechanism**: $K$ independent seeds $\{s_i\}_{i=1}^K$ generate $K$ probes; all probes are queried to obtain $\{r^?_i\}$; local similarity is calculated as $\delta_i = d(r^?_i, r_i) < \tau$; Bernoulli trial statistics are used: $|\sum \mathbb{1}[\delta_i = 1] / K - \mu_0| < \alpha$.
    - **Design Motivation**: A single probe is susceptible to noise; $K$ independent probes provide robust statistical detection—even if 30% of probes fail, the remaining 70% can still provide verification.

3.  **Robustness to Fine-tuning/Pruning**:
    - **Function**: Enable fingerprint detection even after model fine-tuning, parameter pruning, or quantization.
    - **Mechanism**: Utilizes **diversified pseudo-random probe distributions** + **fuzzy response matching**. The probe distribution covers a wide semantic space, making it difficult for fine-tuning to eliminate all signatures. Responses are matched via **semantic similarity** $\delta(r^?, r) = \cos(\text{enc}(r^?), \text{enc}(r)) > 0.7$ rather than exact matching.
    - **Design Motivation**: Conventional fingerprints requiring exact matching are easily disrupted by fine-tuning; semantic matching combined with a multi-probe strategy significantly enhances robustness.

## Key Experimental Results

### Main Results: Detection Rates Across Models and Modifications

| Modification Type | Original LLaMA-7B | Fine-tune (10K samples) | Pruning 50% | Quantization INT8 | Distillation to 3B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ours** (K=100) | **100%** | **98.7%** | **97.2%** | **99.5%** | **94.1%** |
| **Ours** (K=50) | 100% | 96.4% | 94.8% | 98.1% | 89.7% |
| **Ours** (K=20) | 100% | 91.3% | 88.7% | 94.5% | 82.5% |
| Baseline-Watermark | 100% | 67.2% | 71.3% | 88.7% | 51.4% |
| Baseline-Backdoor | 100% | 23.1% | 35.6% | 76.4% | 12.3% |

### False Positive Rates

| Number of Probes K | False Positive Rate (vs. 1000 other LLMs) |
| :--- | :--- |
| 20 | 2.3% |
| 50 | 0.8% |
| **100** | **0.1%** |

### Stealthiness Test

| Detection Method | FLIPS Probe Detection Rate | Baseline-Backdoor Trigger Detection Rate |
| :--- | :--- | :--- |
| Input Distribution Anomaly | 0.3% (essentially random) | 87.5% |
| LLM Meta-detection (GPT-4) | 1.2% | 92.3% |
| Frequency Analysis | 0% (PRG output uniform) | 78.9% |

### Performance Overhead

| Operation | Time Overhead | Memory Overhead |
| :--- | :--- | :--- |
| Fingerprint Injection (K=100) | 30 seconds | 1.5MB |
| Single Verification (K=100) | 4.2 seconds | <100MB |
| Training Degradation | 0% (No model modification) | 0% |

### Key Findings
- **Outstanding Robustness under Fine-tuning**: FLIPS maintains a 98.7% detection rate after fine-tuning, far exceeding the 23.1% of Backdoor methods.
- **K = 50 is Optimal for Balance**: Achieves a false positive rate $< 1\%$ and a detection rate $> 90\%$ while balancing cost.
- **Zero Model Damage**: FLIPS does not modify the model, only records responses; model capability evaluations remain unchanged.
- **Quantization and Distillation Robustness**: Achieves 99.5% for INT8 quantization and 94.1% for 3B distillation.

## Highlights & Insights
- **Elegant Fusion of Cryptography and LLMs**: Applies the classic PRG security model to the LLM fingerprinting scenario with theoretical security guarantees.
- **Zero-Harm Design**: Does not modify the model, only records responses—completely avoiding the performance loss issues of traditional watermarking.
- **Provable Stealthiness**: Under PRG indistinguishability, fingerprint queries cannot be distinguished from normal queries.
- **Extreme Robustness**: Outperforms baselines by 20-70 percentage points across fine-tuning, pruning, quantization, and distillation scenarios.

## Limitations & Future Work
- **Openness to White-box Attacks**: If an attacker has full control over model weights, they might eliminate fingerprints through deep architectural modifications.
- **Seed Management**: Fingerprints become invalid if seeds are leaked; threshold cryptography may be needed for multi-party sharing.
- **Injection Timing**: Responses must be recorded in advance on the original model; not applicable to models already released without fingerprints.
- **Future Work**: Introduce threshold cryptography for multi-party verification; extend to multimodal models; research active fingerprint injection (introducing specific structures during training).

## Related Work & Insights
- **vs. Watermarking (Kirchenbauer et al. 2023)**: Watermarking marks model output and affects generation quality; FLIPS only records responses without modifying output.
- **vs. Backdoor Fingerprinting**: Backdoors are easily detected; FLIPS uses PRG to achieve stealthy fingerprinting.
- **vs. Model Distillation Detection**: Traditional detection requires white-box access; FLIPS is black-box compatible.
- **Insight**: The combination of cryptographic pseudo-randomness and model determinism is a promising direction for LLM intellectual property protection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First application of cryptographic PRG to black-box LLM fingerprinting with clear theory.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Comprehensive comparisons across models, modifications, and baselines, including stealthiness tests.
- Writing Quality: ⭐⭐⭐⭐  Clear argumentation and precise algorithmic descriptions.
- Value: ⭐⭐⭐⭐⭐  Addresses the urgent need for LLM IP protection; FLIPS's robustness, stealthiness, and zero-harm characteristics are groundbreaking.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tracing and Reversing Edits in LLMs](../../ICLR2026/social_computing/tracing_and_reversing_edits_in_llms.md)
- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](../../ACL2026/social_computing/investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[ICLR 2026\] When Agents Persuade: Propaganda Generation and Mitigation in LLMs](../../ICLR2026/social_computing/when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)
- [\[ACL 2026\] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](../../ACL2026/social_computing/to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)
- [\[ACL 2026\] mdok-style at SemEval-2026 Task 9: Finetuning LLMs for Multilingual Polarization Detection](../../ACL2026/social_computing/mdok-style_at_semeval-2026_task_9_finetuning_llms_for_multilingual_polarization_.md)

</div>

<!-- RELATED:END -->
