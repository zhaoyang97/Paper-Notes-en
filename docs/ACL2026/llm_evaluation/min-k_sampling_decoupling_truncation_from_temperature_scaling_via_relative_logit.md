---
title: >-
  [Paper Note] Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics
description: >-
  [ACL 2026][LLM Evaluation][Sampling Strategy] Min-k Sampling detects "semantic cliffs" — the boundary between high-confidence candidate tokens and low-quality tail noise — by analyzing the local structure of the sorted logit distribution. This yields strictly temperature-invariant truncation that maintains robust performance on reasoning and creative writing tasks even under extreme temperatures.
tags:
  - ACL 2026
  - LLM Evaluation
  - Sampling Strategy
  - Temperature Invariance
  - Semantic Cliff Detection
  - Dynamic Truncation
  - Logit Space
date: 2026-05-08
content_hash: 06d260e0b8cb0128
---

# Min-k Sampling: Decoupling Truncation from Temperature Scaling via Relative Logit Dynamics

**Conference**: ACL 2026  
**arXiv**: [2604.11012](https://arxiv.org/abs/2604.11012)  
**Code**: [https://github.com/YecanLee/Mink](https://github.com/YecanLee/Mink)  
**Area**: LLM Evaluation  
**Keywords**: Sampling Strategy, Temperature Invariance, Semantic Cliff Detection, Dynamic Truncation, Logit Space

## TL;DR

Min-k Sampling detects "semantic cliffs" — the boundary between high-confidence candidate tokens and low-quality tail noise — by analyzing the local structure of the sorted logit distribution. This yields strictly temperature-invariant truncation that maintains robust performance on reasoning and creative writing tasks even under extreme temperatures.

## Background & Motivation

**State of the Field**: The quality of LLM text generation heavily depends on decoding sampling strategies. Mainstream methods such as Top-k, Top-p (nucleus sampling), and Min-p balance diversity and accuracy through truncation in probability space. More recently, Top-$n\sigma$ shifts operations to logit space to achieve temperature invariance.

**Limitations of Prior Work**: (1) Probability-space methods (Top-k/p/Min-p) are highly sensitive to temperature — noise rates exceed 90% when temperature surpasses 2.0, and generation collapses entirely at temperature 10.0. (2) Although Top-$n\sigma$ is temperature-invariant, it relies on the global standard deviation $\sigma$, which is easily distorted by a large number of long-tail noise tokens, making it difficult to precisely locate fine-grained confidence differences among high-confidence candidates. (3) Top-$n\sigma$ is highly sensitive to the hyperparameter $n$ ($n=1.0$ introduces noise; $n=2.0$ amplifies it).

**Root Cause**: Temperature scaling simultaneously governs two effects that should be independent — increasing diversity among plausible candidates (desirable) and injecting noise tokens from the tail (undesirable). An ideal truncation mechanism should decouple these two effects.

**Paper Goals**: To design a dynamic truncation strategy that is simultaneously temperature-invariant, robust to hyperparameter choices, and capable of precisely capturing model confidence boundaries.

**Starting Point**: Rather than relying on global statistics, the paper analyzes the local morphology of the sorted logit sequence. In a descending logit sequence, a "semantic cliff" exists — a sharp drop from meaningful candidate tokens to irrelevant noise tokens. The truncation boundary is determined by detecting the location of this cliff.

**Core Idea**: A position-weighted relative decay rate is used to detect the point of maximum drop (semantic cliff) in the sorted logit sequence. This computation is strictly invariant to temperature scaling, achieving complete decoupling of truncation decisions from temperature.

## Method

### Overall Architecture

At each generation step, the model's logit vector $\mathbf{I} \in \mathbb{R}^{|V|}$ is obtained, sorted in descending order, and then the position-weighted relative decay rate is computed to identify the maximum decay position as the truncation point $k$. Logits beyond this position are set to $-\infty$, and the remaining logits undergo temperature scaling and softmax for sampling. The entire truncation decision is completed prior to temperature scaling and is strictly invariant with respect to $T$.

### Key Designs

1. **Weighted Relative Decay Detection**:

    - **Function**: Precisely locates the semantic cliff in the sorted logit sequence.
    - **Mechanism**: After sorting logits in descending order, the dynamic range $R_l = l_1 - l_{|V|}$ is computed. For each position, the weighted relative decay rate is calculated as $w_i = \frac{l_i - l_{i+1}}{R_l} \cdot \frac{1}{i}$, where $(l_i - l_{i+1})/R_l$ is the normalized local drop magnitude and $1/i$ is a positional weight emphasizing changes at the head of the distribution. The truncation point is $k_{cliff} = \arg\max_i w_i$.
    - **Design Motivation**: The $1/i$ weight is grounded in empirical observation that the most meaningful probability drops typically occur at the head of the distribution. Ablation studies confirm that removing $1/i$ significantly introduces tail noise, while removing the $R_l$ normalization causes model collapse at high temperatures. Linear decay $1/i$ outperforms logarithmic and quadratic alternatives.

2. **Dynamic Fallback Mechanism**:

    - **Function**: Prevents truncation from degenerating to $k=1$ under extremely flat distributions.
    - **Mechanism**: When the model is highly uncertain, logit values are nearly uniform and the $w_i$ sequence has no prominent peak, causing $\arg\max$ to collapse to $i^*=1$. A fallback candidate size is defined as $k_{fallback} = \lfloor \tau / R_l \rfloor$, and the final truncation point is $k = \max(k_{cliff}, k_{fallback})$. The small constant $\tau$ automatically provides a reasonable minimum exploration range when $R_l$ is very small (i.e., the distribution is extremely flat).
    - **Design Motivation**: Ablation studies show that the fallback mechanism does not affect performance on structured reasoning tasks (where the primary mechanism is sufficiently robust), but is a necessary safety net in high-entropy open-ended scenarios.

3. **Proof of Strict Temperature Invariance**:

    - **Function**: Theoretically guarantees that truncation decisions are completely unaffected by temperature.
    - **Mechanism**: Temperature scaling $l'_i = l_i/T$ does not alter the ranking. The normalized decay $d'_i = (l'_i - l'_{i+1})/(l'_1 - l'_{|V|}) = (l_i - l_{i+1})/(l_1 - l_{|V|}) = d_i$, since dividing both numerator and denominator by $T$ cancels out. The weighted decay $w'_i = d'_i/i = d_i/i = w_i$; therefore $\arg\max_i w'_i = \arg\max_i w_i$, and the truncation point $k$ is invariant to $T$.
    - **Design Motivation**: Temperature invariance is the central design objective, ensuring that temperature controls only the diversity within the selected candidate set, without affecting its composition.

### Loss & Training

Min-k is an inference-time method requiring no training. The default hyperparameter is $\tau = 3.0$, and experiments demonstrate insensitivity to its value. Truncation is applied before temperature scaling; the truncated logits are then processed via $l'/T$ and softmax to produce the sampling distribution.

## Key Experimental Results

### Main Results

**Mathematical Reasoning Accuracy (EM%, LLaMA3-8B-Instruct, GSM8K)**

| Method | T=1.0 | T=3.0 | T=5.0 | T=10.0 |
|--------|-------|-------|-------|--------|
| Top-k | 75.44 | 9.86 | 0.08 | 0.00 |
| Top-p | 76.27 | 2.50 | 0.00 | 0.00 |
| Min-p | 75.36 | 28.81 | 0.38 | 0.00 |
| Top-$n\sigma$ | 75.44 | 72.55 | 74.07 | 73.77 |
| **Min-k** | **77.39** | **76.02** | **76.48** | **74.79** |

**AQuA Reasoning Accuracy (EM%, LLaMA3-8B-Instruct)**

| Method | T=1.0 | T=5.0 | T=10.0 |
|--------|-------|-------|--------|
| Top-$n\sigma$ | 48.03 | 44.49 | 49.61 |
| **Min-k** | **50.00** | **50.39** | **46.06** |

**Creative Writing Win Rate (%, LLaMA3-8B-Instruct vs. Greedy)**

| Method | T=1.0 | T=3.0 | T=10.0 |
|--------|-------|-------|--------|
| Top-k | 50.80 | 1.40 | - |
| Top-p | 49.60 | 0.00 | - |
| Top-$n\sigma$ | 52.40 | 51.40 | 50.00 |
| **Min-k** | **58.60** | **53.60** | **52.80** |

### Ablation Study

**Human Evaluation (200 Pairwise Comparisons, Min-k vs. Top-$n\sigma$)**

| Preference | LLaMA3-8B | Qwen3-4B | Total |
|------------|-----------|----------|-------|
| Min-k Wins | 41 | 34 | 75 (37.5%) |
| Top-$n\sigma$ Wins | 33 | 34 | 67 (33.5%) |
| Tie | 26 | 32 | 58 (29.0%) |

### Key Findings

- Conventional methods nearly completely collapse at T>2.0 (noise rate >90%), while Min-k maintains 74.79% accuracy on GSM8K at T=10.0.
- Min-k also outperforms all baselines at the standard setting of T=1.0 (GSM8K 77.39% vs. Top-$n\sigma$ 75.44%), demonstrating that semantic cliff detection is advantageous even at normal temperatures.
- In creative writing, Min-k achieves a win rate of 58.60% at T=1.0, significantly outperforming all competing methods.
- In human evaluation, Min-k shows a clear advantage over LLaMA3 (41 vs. 33) but performs on par with Qwen3.
- The hyperparameter $\tau$ has minimal impact on performance across the range 1.0–10.0, validating the claimed low hyperparameter sensitivity.

## Highlights & Insights

- The complete decoupling of "truncation" and "temperature" — two effects that have long been entangled — represents an elegant methodological contribution.
- The proof of temperature invariance is concise and compelling: numerator and denominator both divide by $T$, requiring only three lines of derivation.
- The algorithm is extremely simple — the core implementation requires no more than 10 lines of code, with negligible computational overhead.
- The design intuition behind the $1/i$ positional weight is clear and well-validated by ablation experiments.

## Limitations & Future Work

- Binomial testing does not reveal statistically significant differences between Min-k and Top-$n\sigma$ in human evaluation.
- The choice of $\tau$ in the fallback mechanism under extreme conditions lacks theoretical guidance.
- Validation is limited to the LLaMA3 and Qwen3 model families; other architectures have not been tested.
- Future work could explore combining semantic cliff detection with acceleration techniques such as speculative decoding.

## Related Work & Insights

- Min-k represents a new development in logit-space methods following Top-$n\sigma$, replacing the global $\sigma$ with a local statistic.
- Despite the similar name, Min-k is fundamentally different from Min-p (dynamic truncation in probability space): Min-k operates in logit space and is temperature-invariant.
- The work offers a new paradigm for decoding strategy design: focusing on the local structure of the distribution rather than its global characteristics.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The semantic cliff detection idea is novel and theoretically elegant, though the overall contribution is an incremental improvement over existing logit-space methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage spans reasoning and creative writing tasks across four models with human evaluation included, though the practical utility of extreme temperature scenarios remains questionable.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, the temperature invariance proof is concise, and experimental results closely support the theoretical claims.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] OptScale: Probabilistic Optimality for Inference-time Scaling](../../AAAI2026/llm_evaluation/optscale_probabilistic_optimality_for_inference-time_scaling.md)
- [\[CVPR 2026\] Enhancing Out-of-Distribution Detection with Extended Logit Normalization](../../CVPR2026/llm_evaluation/enhancing_out-of-distribution_detection_with_extended_logit_normalization.md)
- [\[AAAI 2026\] Towards a Rigorous Understanding of the Population Dynamics of the NSGA-III: Tight Runtime Bounds](../../AAAI2026/llm_evaluation/towards_a_rigorous_understanding_of_the_population_dynamics_of_the_nsga-iii_tigh.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/llm_evaluation/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ICLR 2026\] Disentangling Shared and Private Neural Dynamics with SPIRE: A Latent Modeling Framework for Deep Brain Stimulation](../../ICLR2026/llm_evaluation/disentangling_shared_and_private_neural_dynamics_with_spire_a_latent_modeling_fr.md)

<!-- RELATED:END -->
