---
title: >-
  [Paper Note] ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification
description: >-
  [ICML2025][Reinforcement Learning][LLM self-correction] This paper proposes the ReVISE framework, which introduces a special token `[refine]` and a two-stage curriculum learning scheme (first learning self-verification, then learning self-correction). This enables LLMs to introspectively verify and correct their own reasoning trajectories at test-time without requiring external verifiers or complex RL training.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "LLM self-correction"
  - "test-time scaling"
  - "preference learning"
  - "curriculum learning"
  - "self-verification"
date: 2026-05-08
content_hash: 8e9bb57b74d99418
---

# ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification

**Conference**: ICML2025  
**arXiv**: [2502.14565](https://arxiv.org/abs/2502.14565)  
**Code**: [github.com/seunghyukoh/revise](https://github.com/seunghyukoh/revise)  
**Authors**: Hyunseok Lee, Seunghyuk Oh, Jaehyung Kim, Jinwoo Shin, Jihoon Tack (KAIST)
**Area**: Reinforcement Learning  
**Keywords**: LLM self-correction, test-time scaling, preference learning, curriculum learning, self-verification

## TL;DR

This paper proposes the ReVISE framework, which introduces a special token `[refine]` and a two-stage curriculum learning scheme (first learning self-verification, then learning self-correction). This enables LLMs to introspectively verify and correct their own reasoning trajectories at test-time without requiring external verifiers or complex RL training.

## Background & Motivation

**Core Problem**: In complex reasoning tasks steps, errors in early steps tend to accumulate progressively, while LLMs exhibit severely insufficient capability to detect and correct their own errors (self-awareness). The autoregressive nature of generation also restricts the model's ability to look back and revise previous steps.

**Limitations of Prior Work**:

**External Verifier Methods** (e.g., Luo et al., 2024): Rely on large-scale external models to perform verification and trigger regeneration, resulting in high computational overhead.

**RL Methods** (e.g., SCoRe, Kumar et al., 2024): Suffer from unstable training and massive computational demands (approx. 1.5 million generations / 3000 steps), without explicitly modeling the verification of intermediate reasoning steps.

**Self-Refine Methods** (Madaan et al., 2023): Suffer from performance degradation on complex tasks, and LLMs fundamentally lack inherent self-correction capabilities (as demonstrated by Huang et al., 2024).

**Key Challenge**: Can LLMs be equipped with an intrinsic mechanism that explicitly verifies their own reasoning processes and corrects errors accordingly?

## Method

### 3.1 Problem Formulation and the `[refine]` Token

Given an input $x$, the model first generates an initial output $y_{\text{init}} \sim \mathcal{M}(\cdot|x)$, and then predicts a verification token $v \in \{[\text{eos}], [\text{refine}]\}$:

- If $v = [\text{eos}]$: The model deems the answer correct and terminates generation.
- If $v = [\text{refine}]$: The model deems the answer incorrect and continues to generate the refined reasoning $y_{\text{refined}} \sim \mathcal{M}(\cdot|[\text{refine}], y_{\text{init}}, x)$.

**Key Advantage**: The softmax probability of the model's verification token can be directly extracted to serve as the self-verification confidence.

### 3.2 Two-Stage Curriculum Learning

Both stages utilize a combined SFT + DPO loss, avoiding unstable RL training.

**Stage 1: Learning Self-Verification**

Multiple responses are sampled from the initial model $\mathcal{M}_0$ for each input. Correct/incorrect paths are distinguished based on the ground-truth to construct preference pairs:

- Correct response $\hat{y} = y_{\text{correct}}$ $\rightarrow$ Preference $(x, \hat{y} \oplus [\text{eos}], \hat{y} \oplus [\text{refine}])$
- Incorrect response $\hat{y} = y_{\text{wrong}}$ $\rightarrow$ Preference $(x \oplus \hat{y}, [\text{refine}], [\text{eos}])$

Objective function:

$$\mathcal{L}_{\text{verify}} = \mathcal{L}_{\text{SFT}}(\mathcal{D}_{\text{verify}}) + \lambda \mathcal{L}_{\text{Pref}}(\mathcal{D}_{\text{verify}})$$

where the implicit reward in the DPO loss is $r(x,y) = \beta \log \frac{\mathcal{M}(y|x)}{\mathcal{M}_0(y|x)}$, and $\lambda = 0.1$.

**Stage 2: Learning Self-Correction**

Starting from $\mathcal{M}_1$ obtained in Stage 1, a new preference dataset $\mathcal{D}_{\text{correct}}$ is constructed:

- Correct response: Same as Stage 1, encouraging `[eos]`
- Incorrect response $\hat{y} = y_{\text{wrong}}$: The positive sample is $[\text{refine}] \oplus y$ (concatenated with ground-truth), and the negative sample is $[\text{eos}]$

$$\mathcal{L}_{\text{correct}} = \mathcal{L}_{\text{SFT}}(\mathcal{D}_{\text{correct}}) + \lambda \mathcal{L}_{\text{Pref}}(\mathcal{D}_{\text{correct}})$$

The key to curriculum learning: Decoupling self-verification and self-correction into two stages, preventing conflict between the objectives of the two tasks.

### 3.3 Verification-Confidence-Aware Sampling

During inference, the softmax probability of the `[eos]` token, $c_i = \mathcal{M}([\text{eos}]|y_i, x)$, is leveraged as the confidence, replacing the equal-weight counting in traditional majority voting:

$$y^* = \arg\max_{y \in \mathcal{Y}} \sum_{i: y_i = y} c_i$$

That is, summing up the confidence scores for identical answers, and selecting the answer with the highest cumulative confidence.

## Key Experimental Results

**Models**: Llama-3.2-1B, Llama-3.1-8B (base/non-instruct versions)  
**Datasets**: GSM8K (8.8K training set), MATH (trained on MetaMath 50K subset), MBPP (CoT generated via GPT-4o)  
**Baselines**: SFT, RFT, STaR+ (STaR + SFT data), DPO, SCoRe  
**Training**: AdamW, lr ∈ {1e-4, 1e-5}, cosine decay, 1 epoch  

### Main Results (Table 1)

| Method | Llama-3.2-1B GSM8K Maj@1/5 | MATH-500 Maj@1/5 | Llama-3.1-8B GSM8K Maj@1/5 | MATH-500 Maj@1/5 |
|------|:---:|:---:|:---:|:---:|
| Few-shot CoT | 5.7 / 7.2 | 3.0 / 3.2 | 56.7 / 58.3 | 23.4 / 23.2 |
| SFT | 22.1 / 26.4 | 10.4 / 11.4 | 58.2 / 64.8 | 27.8 / 33.2 |
| RFT | 26.2 / 28.6 | 12.6 / 12.8 | 58.9 / 65.3 | 30.8 / 35.6 |
| STaR+ | 26.2 / 29.9 | 11.4 / 13.4 | 59.2 / 64.9 | 30.4 / 32.8 |
| **ReVISE** | **28.1 / 32.8** | **13.4 / 14.8** | **61.6 / 69.2** | **33.6 / 37.6** |

### Coding Task MBPP (Table 2, Llama-3.2-1B)

| Method | Pass@1 |
|------|:---:|
| Few-shot CoT | 24.5 |
| SFT | 30.0 |
| STaR+ | 30.7 |
| **ReVISE** | **33.1** |

### Performance on Instruct-tuned Models (Table 3, Llama-3.2-1B-Instruct)

| Method | GSM8K | GSM240K |
|------|:---:|:---:|
| Zero-shot CoT | 48.6 | 48.6 |
| SFT | 41.9 | 54.8 |
| RFT | 44.0 | 50.9 |
| **ReVISE** | **52.3** | **59.4** |

Note: SFT/RFT performance on instruct-tuned models actually falls below zero-shot CoT (due to catastrophic forgetting). ReVISE avoids this issue because gold labels are only utilized in the second attempt for correction rather than direct SFT.

### Comparison with SCoRe (Table 6, Gemma-2-2B, MATH-500)

| Method | Accuracy | Training Efficiency |
|------|:---:|:---:|
| SCoRe | 23.0% | ×1 |
| ReVISE | 23.2% | **30$\times$ more efficient** |
| ReVISE + iter2 | 25.8% | **15$\times$ more efficient** |

ReVISE only requires generating 1 reasoning path per sample (50K times in total), whereas SCoRe requires approximately 1.5 million generations.

### Quantification of Verification Ability (Table 7, Llama-3.2-1B, GSM8K AUROC)

| Method | AUROC |
|------|:---:|
| V-STaR (External Verifier) | 69.5% |
| **ReVISE (Intrinsic Verification)** | **76.0%** |

### Ablation Study

- **Effectiveness of Curriculum Learning**: Without curriculum 22.6% $\rightarrow$ Stage 1 only ~26% $\rightarrow$ Full ReVISE 28.1% (GSM8K Maj@1)
- **Criticality of DPO Loss**: Removing DPO leads to a 10.3% performance drop.
- **Iterative Correction**: Supports multiple refinement rounds (1 $\rightarrow$ 2 $\rightarrow$ 3 times), with consistent accuracy gains for the 8B model on MATH-500.
- **Cross-Domain Generalization**: Training on MATH $\rightarrow$ Evaluation on GSM8K; 8B model registers ReVISE 61.5% > SFT 60.3%.

## Highlights & Insights

1. **Simple yet Effective Design**: Introducing only a single special token `[refine]` and replacing RL with preference learning successfully achieves both self-verification and self-correction capabilities.
2. **Highly Efficient Training**: Reduces training compute by 30$\times$ compared to SCoRe, showing a significant efficiency advantage under comparable performance.
3. **Built-in Confidence Signal**: The probability of the `[eos]` token naturally serves as the verification confidence, eliminating the need for separate verifier training.
4. **Friendly to Instruct-tuning**: Avoids catastrophic forgetting, since gold labels are only used as the "refined second attempt" rather than direct SFT targets.
5. **Inherent Support for Test-Time Scaling**: The refinement mechanism itself serves as a form of compute scaling, which yields further gains when combined with confidence-weighted voting.

## Limitations & Future Work

1. **Slight Degradation of Verification Ability in Stage 2**: AUROC drops slightly from the optimal value achieved in Stage 1, indicating a mild catastrophic forgetting issue.
2. **Correction Data Dependent on Ground-Truth**: The positive samples in Stage 2 directly concatenate the ground-truth label, which limits application in unlabeled scenarios.
3. **Only Verges on Verifying Final Answer Correctness**: Lacks intermediate process-level verification, which might limit its effectiveness on longer reasoning chains.
4. **Limited Experimental Scale**: Only evaluated on 1B and 8B models, leaving its effectiveness on larger models or reasoning-specific models unverified.
5. **Single-step Correction Training**: Although multiple refinements can be performed during inference, training is only tailored to a single correction. Training schemes for multi-round refinement warrant further exploration.
6. **Relatively Simple Benchmark Tasks**: GSM8K/MATH-500 are standard mathematical reasoning tasks; evaluation has not been conducted on more challenging competition problems or multimodal reasoning.

## Related Work & Insights

- **Backtracking (Zhang et al., 2024c)**: Similarly introduces a reset token, but focuses on safety scenarios rather than reasoning correction.
- **STaR / V-STaR**: Self-improvement paradigms that do not perform explicit verification.
- **SCoRe (Kumar et al., 2024)**: RL-based self-correction, which shows comparable performance but with a 30$\times$ higher training cost.
- **Self-Refine (Madaan et al., 2023)**: Leverages iterative feedback at inference time, but degrades performance on complex tasks.
- **Insights**: The design of the `[refine]` token can be extended to other scenarios requiring "self-censorship" or review (e.g., safety, code review). The curriculum learning strategy of decomposing complex tasks is also highly referable.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of the `[refine]` token and the two-stage curriculum learning is simple yet highly effective. Unifying self-verification and self-correction under preference learning is a compelling idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations (curriculum learning, DPO, confidence-aware sampling, iterative correction, cross-domain generalization, instruct-tuning) and comparisons with strong baselines like SCoRe, though missing verification on larger models and harder tasks.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured, with standardized mathematical formulations and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Provides a self-correction training paradigm that is substantially more efficient than RL, offering high practical utility. However, in the era of advanced reasoning models like o1/DeepSeek-R1, the competitiveness of this method requires further evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Test-Time Adaptation with Binary Feedback](test-time_adaptation_with_binary_feedback.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[ICLR 2026\] DuPO: Enabling Reliable Self-Verification via Dual Preference Optimization](../../ICLR2026/reinforcement_learning/dupo_enabling_reliable_self-verification_via_dual_preference_optimization.md)
- [\[ICLR 2026\] P-GenRM: Personalized Generative Reward Model with Test-time User-based Scaling](../../ICLR2026/reinforcement_learning/p-genrm_personalized_generative_reward_model_with_test-time_user-based_scaling.md)

</div>

<!-- RELATED:END -->
