---
title: >-
  [Paper Note] Exploring the Limits of Strong Membership Inference Attacks on Large Language Models
description: >-
  [NeurIPS 2025][LLM Safety][Membership Inference Attack] This work presents the first extension of strong membership inference attacks (LiRA) to GPT-2-scale LLMs ranging from 10M to 1B parameters, training over 4,000 reference models. Four key findings are revealed: strong MIAs can succeed on LLMs but with limited effectiveness (AUC < 0.7), and a substantial fraction of per-sample decisions are **indistinguishable from random coin flips** under training randomness.
tags:
  - NeurIPS 2025
  - LLM Safety
  - Membership Inference Attack
  - LLM Privacy
  - LiRA
  - Differential Privacy
  - Pre-trained Language Models
date: 2026-05-08
content_hash: e1fda81adcc0601c
---

# Exploring the Limits of Strong Membership Inference Attacks on Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.18773](https://arxiv.org/abs/2505.18773)
**Code**: None (Internal experiments at Google DeepMind)
**Area**: AI Security
**Keywords**: Membership Inference Attack, LLM Privacy, LiRA, Differential Privacy, Pre-trained Language Models

## TL;DR

This work presents the first extension of strong membership inference attacks (LiRA) to GPT-2-scale LLMs ranging from 10M to 1B parameters, training over 4,000 reference models. Four key findings are revealed: strong MIAs can succeed on LLMs but with limited effectiveness (AUC < 0.7), and a substantial fraction of per-sample decisions are **indistinguishable from random coin flips** under training randomness.

## Background & Motivation

**Background**: Membership inference attacks (MIAs) are a central methodology for evaluating privacy leakage in machine learning models. The strongest MIAs (e.g., LiRA) require training large numbers of reference models, which is computationally prohibitive at the scale of LLMs. As a result, prior work either employs **weak attacks** (which do not train reference models, such as fine-tuning attacks) or runs strong attacks on **small models**.

**Limitations of Prior Work**:
   - Weak attacks have been shown to be **brittle** — often performing no better than random guessing
   - Insights from strong attacks on small models **do not generalize** to modern LLMs
   - The true performance ceiling of the strongest MIAs on pre-trained LLMs remains unknown

**Key Challenge**: Is the inefficacy of weak attacks caused by the absence of reference models, or does MIA face **fundamental difficulty** on LLMs? This question has not been answered.

**Goal**: Through large-scale computational investment, establish the first benchmark for strong MIAs on pre-trained LLMs and answer the above question.

**Key Insight**: Train 4,000+ GPT-2 reference models (10M to 1B parameters) using the C4 dataset (50M+ samples, three orders of magnitude larger than prior work), and systematically run LiRA attacks.

**Core Idea**: Commit unprecedented computational resources to establish, for the first time, a performance baseline for strong MIAs at LLM scale, and introduce the *flip rate* metric to expose per-sample decision instability masked by aggregate metrics.

## Method

### Overall Architecture

- **Attack Method**: LiRA (Likelihood Ratio Attack), one of the strongest reference-model-based MIAs
- **Target Models**: GPT-2 architecture, 10M–1B parameters
- **Training Data**: Subset of the C4 dataset, up to approximately 100M samples
- **Reference Models**: 128 models (64 IN + 64 OUT), each trained on a random subset of size $N$ sampled from a fixed dataset of size $2N$

### Key Designs

#### 1. LiRA Attack Pipeline

- **Function**: For each query sample $x$, collect two sets of statistics from reference models: $\{s(f,x): f \in \Phi_{\text{IN}}(x)\}$ and $\{s(f,x): f \in \Phi_{\text{OUT}}(x)\}$
- **Mechanism**: Fit two distributions $p_{\text{IN}}$ and $p_{\text{OUT}}$, and compute the likelihood ratio score $\Lambda(x) = p_{\text{IN}}(s(h,x)) / p_{\text{OUT}}(s(h,x))$
- **Observation Statistic**: Model loss
- **Decision Rule**: $b(x) = \mathbf{1}\{\Lambda(x) \geq \tau\}$, where threshold $\tau$ is calibrated on non-members

#### 2. Per-Sample Flip Rate

- **Function**: Measures the stability of MIA decisions under training randomness
- **Core Formula**:
$$\text{flip}_\eta(x) \coloneqq \Pr_{r,r' \sim \mu}[b_r^{(\eta)}(x) \neq b_{r'}^{(\eta)}(x)]$$
  $B = 127$ target models trained with different random seeds are used to measure decision consistency.
- **Design Motivation**: Aggregate metrics such as AUC may obscure per-sample decision instability. A flip rate $\approx 0.5$ implies decisions **are no better than a coin flip**.
- **Statistical Test**: Two-sided exact binomial test, $H_0: \theta = 0.5$; at $\alpha = 0.05$, the threshold is $\widehat{\text{flip}}_{127} \gtrsim 0.487$

#### 3. Compute-Optimal Model Configuration

- **Chinchilla Scaling**: Training tokens = 20 × number of model parameters
- **Single-Epoch Training**: Simulates realistic LLM training conditions
- **Model Scales**: 10M, 44M, 85M, 140M, 302M, 489M, 604M, 1018M

### Evaluation Metrics

- **ROC-AUC**: Threshold-independent attack success rate
- **TPR @ fixed FPR**: True positive rate at low false positive rates
- **Per-sample flip rate**: Stability of individual decisions

## Key Experimental Results

### Main Results: LiRA on Chinchilla-Optimal Models

| Model Size | Training Samples | AUC |
|------------|-----------------|-----|
| 10M | ~200K | 0.592 |
| 85M | ~1.7M | **0.699** |
| 140M | ~7M | 0.678 |
| 302M | ~15M | 0.689 |
| 489M | ~24M | 0.547 |
| 604M | ~30M | 0.654 |
| 1018M | ~50M | 0.553 |

**Key Finding**: MIA vulnerability exhibits a **non-monotonic relationship** with model size — larger models are not necessarily more vulnerable.

### Ablation Study

| Variable | Setting | AUC |
|----------|---------|-----|
| Number of reference models (140M) | 1 IN → 256 IN | 0.540 → 0.680 |
| Training epochs (140M) | 1 epoch → 10 epochs | 0.573 → 0.797 |
| Training set size (140M) | 50K–10M | Peak 0.753 (at 1M) |
| Fixed data, varying model size | 10M–1018M, 8.3M samples | TPR increases monotonically |
| 44M model: half data 2 epochs vs. full data 1 epoch | — | 0.744 vs. 0.620 |

### Flip Rate Analysis (302M Model, ~500K Samples)

| FPR | Coin-flip decisions (members) | Coin-flip decisions (non-members) |
|-----|------------------------------|----------------------------------|
| 0.001 | ~15.4% | Negligible |
| 0.02 | ~18.4% | ~0.03% |
| Relaxed threshold (flip ≥ 0.4), FPR = 0.02 | ~39.8% | ~0.2% |

### Extraction vs. MIA Correlation

- Among the 1,000 samples with the highest LiRA scores, 713 are confirmed members
- However, the maximum suffix extraction probability is only ~0.0067
- Most samples have negative log-probability > 100, corresponding to probability ~$10^{-44}$
- **Conclusion**: MIA success and data extractability are **uncorrelated**

### Key Findings

1. **Strong MIAs can succeed on LLMs**: LiRA substantially outperforms the random baseline (AUC ≈ 0.55–0.70)
2. **But success is limited under realistic settings**: AUC < 0.7 for all models under Chinchilla-optimal configurations
3. **Multiple epochs significantly increase vulnerability**: AUC rises from 0.62 to 0.74 after 2 epochs
4. **Instability revealed by flip rate is striking**: Even at FPR = 0.001, approximately 15% of true positive decisions are indistinguishable from coin flips
5. **Timing of sample exposure affects vulnerability**: Samples encountered later in training are more vulnerable

## Highlights & Insights

1. **Unprecedented computational scale**: 4,000+ reference models spanning 10M to 1B parameters — this investment alone constitutes a major contribution
2. **Flip rate is a highly insightful new metric**: It exposes the truth masked by aggregate metrics — many seemingly successful attack decisions are mere statistical noise
3. **Decoupling of MIA and extraction**: Challenges the common assumption chain that "MIA success = memorization = extractability"
4. **Non-monotonic model size–vulnerability relationship**: Refutes the intuition that larger models are inherently more vulnerable
5. **Establishes a performance upper bound for weak attacks**: Informs the community of the maximum achievable performance of weak attacks

## Limitations & Future Work

1. **GPT-2 architecture only**: Results are not validated on truly large-scale LLMs (e.g., GPT-4, Llama)
2. **C4 dataset**: A single dataset that may not be representative of other training distributions
3. **Fixed attack method**: Newer MIA variants are not explored (RMIA underperforms in the appendix)
4. **High computational cost of flip rate analysis**: Requires 127 replicas of the target model
5. **Caution required when interpreting practical privacy implications**: Limited MIA success does not warrant claims that LLMs are private
6. **Non-monotonic relationship not fully understood**: The authors hypothesize it is related to Chinchilla scaling laws and training hyperparameters

## Related Work & Insights

- **Relationship to weak attacks**: The failure of weak attacks (fine-tuning-based, loss-based) is indeed partly attributable to the absence of reference models; however, the limited success of strong attacks also suggests that **MIA faces fundamental challenges on LLMs**
- **Implications for dataset contamination detection**: MIAs are frequently used to detect benchmark contamination, but the findings of this paper suggest such methods may be unreliable
- **Broader inspiration**: The flip rate methodology can be generalized to other ML security evaluations to assess the decision stability of attacks

## Rating

⭐⭐⭐⭐⭐ (5/5)
- Substantial workload, deep insights, and fills an important gap in the literature
- The flip rate methodology contributes independently of the specific experimental results
- Directly influences research directions in privacy evaluation and ML security

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](../../ICLR2026/llm_safety/membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)
- [\[NeurIPS 2025\] Distributive Fairness in Large Language Models: Evaluating Alignment with Human Values](distributive_fairness_in_large_language_models_evaluating_alignment_with_human_v.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[ACL 2026\] Jailbreaking Large Language Models with Morality Attacks](../../ACL2026/llm_safety/jailbreaking_large_language_models_with_morality_attacks.md)
- [\[NeurIPS 2025\] AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text](agentstealth_reinforcing_large_language_model_for_anonymizing_user-generated_tex.md)

<!-- RELATED:END -->
