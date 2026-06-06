---
title: >-
  [Paper Note] Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics
description: >-
  [ACL 2026][LLM/NLP][Sampling Strategy] Min-k Sampling detects a "semantic cliff" (the boundary between high-confidence candidates and low-quality tail noise) by analyzing the local structure of sorted logit distributions…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Sampling Strategy"
  - "Temperature Invariance"
  - "Semantic Cliff Detection"
  - "Dynamic Truncation"
  - "Logit Space"
date: 2026-05-08
content_hash: d35a62f9a910f2ec
---

# Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics

**Conference**: ACL 2026  
**arXiv**: [2604.11012](https://arxiv.org/abs/2604.11012)  
**Code**: [https://github.com/YecanLee/Mink](https://github.com/YecanLee/Mink)  
**Area**: LLM Evaluation  
**Keywords**: Sampling Strategy, Temperature Invariance, Semantic Cliff Detection, Dynamic Truncation, Logit Space

## TL;DR

Min-k Sampling detects a "semantic cliff" (the boundary between high-confidence candidates and low-quality tail noise) by analyzing the local structure of sorted logit distributions. It achieves strict temperature-invariant truncation, maintaining robust reasoning and creative writing quality even under extreme temperatures.

## Background & Motivation

**Background**: The quality of LLM text generation is highly dependent on decoding sampling strategies. Mainstream methods such as Top-k, Top-p (nucleus sampling), and Min-p balance diversity and accuracy through truncation in probability space. Recently, Top-$n\sigma$ moved operations to the logit space to achieve temperature invariance.

**Limitations of Prior Work**: (1) Probability space methods (Top-k/p/Min-p) are extremely sensitive to temperature—noise rates exceed 90% when temperature exceeds 2.0 and collapse entirely at temperature 10.0. (2) Although Top-$n\sigma$ is temperature-invariant, it relies on global standard deviation $\sigma$, which is easily disturbed by massive long-tail noise tokens, failing to precisely locate fine-grained confidence differences among high-confidence candidates. (3) Top-$n\sigma$ is highly sensitive to the hyperparameter $n$ ($n=1.0$ introduces noise, while $n=2.0$ amplifies it).

**Key Challenge**: Temperature scaling simultaneously controls two effects that should be independent: increasing diversity among reasonable candidates (desired) and introducing noise tokens from the tail (undesired). An ideal truncation mechanism should decouple these two effects.

**Goal**: Design a dynamic truncation strategy that is temperature-invariant, insensitive to hyperparameters, and capable of accurately capturing the model's confidence boundary.

**Key Insight**: Analyze the local morphology of the sorted logit sequence rather than global statistics. In a sequence where logits are arranged from high to low, a "semantic cliff" exists—a sharp drop from meaningful candidate tokens to irrelevant noise tokens. The truncation boundary is determined by detecting the position of this cliff.

**Core Idea**: Use position-weighted relative decay rates to detect the maximum drop point (semantic cliff) of sorted logits. This calculation is strictly invariant to temperature scaling, achieving complete decoupling of truncation decisions from temperature.

## Method

### Overall Architecture

At each generation step, the model's logit vector $\mathbf{I} \in \mathbb{R}^{|V|}$ is obtained. After sorting in descending order, the position-weighted relative decay rate is calculated to find the maximum decay position as the truncation point $k$. Logits beyond this position are set to $-\infty$, and the remaining logits are used for sampling after temperature scaling and softmax. The entire truncation decision is completed before temperature scaling and is strictly invariant to $T$.

### Key Designs

1.  **Weighted Relative Decay**:
    - **Function**: Precisely locates the semantic cliff within the sorted logit sequence.
    - **Mechanism**: After sorting logits in descending order, the dynamic range $R_l = l_1 - l_{|V|}$ is calculated. Then, for each position, the weighted relative decay rate is computed as $w_i = \frac{l_i - l_{i+1}}{R_l} \cdot \frac{1}{i}$. Here, $(l_i - l_{i+1})/R_l$ is the normalized local drop magnitude, and $1/i$ is a position weight emphasizing changes at the head. The truncation point is $k_{cliff} = \arg\max_i w_i$.
    - **Design Motivation**: The $1/i$ weight is based on the empirical observation that the most meaningful probability drops usually occur at the head of the distribution. Ablation studies verify that removing $1/i$ significantly introduces tail noise, while removing $R_l$ normalization causes model collapse at high temperatures. Linear decay $1/i$ outperforms logarithmic or quadratic decay.

2.  **Dynamic Fallback**:
    - **Function**: Prevents truncation from degenerating to $k=1$ under extremely flat distributions.
    - **Mechanism**: When the model is highly uncertain, the logit values are nearly uniform, the $w_i$ sequence has no obvious peak, and $\arg\max$ may collapse to $i^*=1$. A fallback candidate size is defined as $k_{fallback} = \lfloor \tau / R_l \rfloor$, and the final $k = \max(k_{cliff}, k_{fallback})$. $\tau$ is a small constant that automatically provides a reasonable minimum exploration range when $R_l$ is very small (extremely flat distribution).
    - **Design Motivation**: Ablation studies show that the fallback mechanism does not affect performance on structured reasoning tasks (where the main mechanism is robust enough) but serves as a necessary safety net in high-entropy open scenarios.

3.  **Strict Temperature Invariance Proof**:
    - **Function**: Theoretically guarantees that truncation decisions are completely unaffected by temperature.
    - **Mechanism**: Temperature scaling $l'_i = l_i/T$ does not change the ranking. Normalized decay $d'_i = (l'_i - l'_{i+1})/(l'_1 - l'_{|V|}) = (l_i - l_{i+1})/(l_1 - l_{|V|}) = d_i$, as $T$ cancels out in the fraction. Weighted decay $w'_i = d'_i/i = d_i/i = w_i$. Therefore, $\arg\max_i w'_i = \arg\max_i w_i$, making the truncation position $k$ invariant to $T$.
    - **Design Motivation**: Temperature invariance is a core design goal, ensuring that temperature only controls diversity within the selected candidate set without affecting the composition of the candidate set itself.

### Loss & Training

Ours is an inference-time method and requires no training. The default hyperparameter $\tau = 3.0$ is used, which is insensitive in experiments. Truncation is performed before temperature scaling, and the truncated logits are processed via $l'/T$ and softmax to generate the sampling distribution.

## Key Experimental Results

### Main Results

**Math Reasoning Accuracy (EM%, LLaMA3-8B-Instruct, GSM8K)**

| Method | T=1.0 | T=3.0 | T=5.0 | T=10.0 |
| :--- | :--- | :--- | :--- | :--- |
| Top-k | 75.44 | 9.86 | 0.08 | 0.00 |
| Top-p | 76.27 | 2.50 | 0.00 | 0.00 |
| Min-p | 75.36 | 28.81 | 0.38 | 0.00 |
| Top-$n\sigma$ | 75.44 | 72.55 | 74.07 | 73.77 |
| **Ours** | **77.39** | **76.02** | **76.48** | **74.79** |

**AQuA Reasoning Accuracy (EM%, LLaMA3-8B-Instruct)**

| Method | T=1.0 | T=5.0 | T=10.0 |
| :--- | :--- | :--- | :--- |
| Top-$n\sigma$ | 48.03 | 44.49 | 49.61 |
| **Ours** | **50.00** | **50.39** | **46.06** |

**Creative Writing Win Rate (%, LLaMA3-8B-Instruct vs Greedy)**

| Method | T=1.0 | T=3.0 | T=10.0 |
| :--- | :--- | :--- | :--- |
| Top-k | 50.80 | 1.40 | - |
| Top-p | 49.60 | 0.00 | - |
| Top-$n\sigma$ | 52.40 | 51.40 | 50.00 |
| **Ours** | **58.60** | **53.60** | **52.80** |

### Ablation Study

**Human Evaluation (200 pairwise comparisons, Ours vs Top-$n\sigma$)**

| Preference | LLaMA3-8B | Qwen3-4B | Total |
| :--- | :--- | :--- | :--- |
| Ours Win | 41 | 34 | 75 (37.5%) |
| Top-$n\sigma$ Win | 33 | 34 | 67 (33.5%) |
| Tie | 26 | 32 | 58 (29.0%) |

### Key Findings

- Traditional methods almost completely collapse when T > 2.0 (noise rate > 90%), whereas Ours maintains a GSM8K accuracy of 74.79% even at T = 10.0.
- Ours outperforms all baselines under the standard T = 1.0 setting (GSM8K 77.39% vs. Top-$n\sigma$ 75.44%), indicating that semantic cliff detection is advantageous even at normal temperatures.
- In creative writing, Ours achieves a 58.60% win rate at T = 1.0, significantly outperforming all methods.
- In human evaluation, Ours shows a clear advantage on LLaMA3 (41 vs. 33) but remains even on Qwen3.
- The hyperparameter $\tau$ has minimal impact on performance within the 1.0-10.0 range, validating the claim of low hyperparameter sensitivity.

## Highlights & Insights

- Completely decoupling "truncation" and "temperature," two previously entangled effects, is an elegant methodological solution.
- The proof of temperature invariance is simple and powerful: by dividing both the numerator and denominator by $T$, it requires only a few lines of derivation.
- The algorithm is extremely simple—the core code is less than 10 lines, and the computational overhead is negligible.
- The design of the $1/i$ position weight is intuitive and thoroughly validated by experiments.

## Limitations & Future Work

- Binomial tests did not show a statistically significant difference between Ours and Top-$n\sigma$ in human evaluation.
- The selection of $\tau$ for the fallback mechanism lacks theoretical guidance in extreme cases.
- Validation was only performed on the LLaMA3 and Qwen3 series; other architectures were not tested.
- Future work could explore combining semantic cliff detection with acceleration techniques like speculative decoding.

## Related Work & Insights

- A new advancement in logit space methods following Top-$n\sigma$, but replacing global $\sigma$ with local statistics.
- Shares a naming convention with Min-p (dynamic truncation in probability space) but is entirely different: Ours operates in logit space and is temperature-invariant.
- Provides a new paradigm for decoding strategy design: focusing on the local structure of the distribution rather than global features.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of semantic cliff detection is novel and theoretically elegant, though it is an improvement over existing logit space methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers reasoning and creative writing tasks, 4 models, and includes human evaluation, though the practical utility of extreme temperature scenarios remains questionable.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, the temperature invariance proof is concise, and experiments align closely with the theory.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Leveraging Pretrained Language Models as Energy Functions for Glauber Dynamics Text Diffusion](leveraging_pretrained_language_models_as_energy_functions_for_glauber_dynamics_t.md)
- [\[ACL 2026\] GRASS: Gradient-based Adaptive Layer-wise Importance Sampling for Memory-Efficient LLM Fine-tuning](grass_gradient-based_adaptive_layer-wise_importance_sampling_for_memory-efficien.md)
- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](../../AAAI2026/llm_nlp/stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ICML 2026\] Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](../../ICML2026/llm_nlp/geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)
- [\[ICML 2026\] A Geometric Relation of the Error Introduced by Sampling a Language Model's Output Distribution to its Internal State](../../ICML2026/llm_nlp/a_geometric_relation_of_the_error_introduced_by_sampling_a_language_models_outpu.md)

</div>

<!-- RELATED:END -->
