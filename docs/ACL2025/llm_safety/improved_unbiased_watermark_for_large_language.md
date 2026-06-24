---
title: >-
  [Paper Note] Improved Unbiased Watermark for Large Language Models
description: >-
  [ACL 2025][LLM Safety][watermark] This paper proposes MCmark, a family of multi-channel unbiased watermarking algorithms. By dividing the vocabulary into $l$ segments and boosting token probabilities within the selected segment to embed statistical signals, MCmark preserves the original output distribution of the LLM while improving detectability by over 10% compared to existing unbiased watermarks.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "watermark"
  - "unbiased"
  - "LLM"
  - "MCmark"
  - "multi-channel"
  - "detectability"
date: 2026-05-08
content_hash: c2e360ce89089adc
---

# Improved Unbiased Watermark for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2502.11268](https://arxiv.org/abs/2502.11268)  
**Code**: [https://github.com/RayRuiboChen/MCMark](https://github.com/RayRuiboChen/MCMark)  
**Area**: AI Safety  
**Keywords**: watermark, unbiased, LLM, MCmark, multi-channel, detectability

## TL;DR
This paper proposes MCmark, a family of multi-channel unbiased watermarking algorithms. By dividing the vocabulary into $l$ segments and boosting token probabilities within the selected segment to embed statistical signals, MCmark preserves the original output distribution of the LLM while improving detectability by over 10% compared to existing unbiased watermarks.

## Background & Motivation

**Background**: As the text generation capabilities of AI models surpass those of humans, verifying the source of AI-generated content has become crucial. Watermarking techniques embed statistical signals into the generated text to distinguish between human and AI-generated text. Biased watermarks (such as KGW) enhance detectability by adding a bias $\delta$ to green list tokens, but distort the original distribution, thereby degrading text quality. Unbiased watermarks keep the original distribution intact, but suffer from weaker detectability.

**Limitations of Prior Work**: Existing of unbiased watermarking methods face three practical issues—(1) The $\gamma$-reweight method by Hu et al. requires access to the LM prompt and API for detection, which is not model-agnostic; (2) EXP-edit detection requires thousands of inferences, incurring extremely high time costs; (3) Although DiPmark is model-agnostic, its detection accuracy is significantly lower than biased watermarks (e.g., when KGW at $\delta=2.0$ achieves TPR@FPR=0.1% of 98.79%, DiPmark only achieves 78.77%).

**Key Challenge**: Unbiased watermarks are inherently weaker in detectability than biased watermarks because they do not alter the distribution, resulting in weaker statistical signals. Existing methods are all based on the "red-green list dichotomy" (two distribution channels), which limits the upper bound of signal strength.

**Goal**: Informally achieve a significant increase in detectability and robustness while strictly maintaining unbiasedness. The core idea is to generalize from 2 channels to $l$ channels.

**Key Insight**: The authors introduce the abstract concept of "distribution channels," formalizing the watermarking problem as a constrained optimization problem: maximizing the cumulative probability of generating tokens from the corresponding segment under the channel selected by the watermark key, while constraining each channel to be a valid probability distribution whose expectation equals the original distribution.

**Core Idea**: Divide the vocabulary evenly into $l$ segments, construct $l$ distribution channels, and boost the token probability of the corresponding segment in each channel. By solving for the closed-form optimal solution, detectability is maximized while maintaining unbiasedness.

## Method

### Overall Architecture
The pipeline of MCmark is divided into two phases: generation and detection.
- **Generation Phase**: Given the original LLM distribution $P_M(\cdot|x_{1:t})$, the vocabulary $V$ is evenly divided into $l$ segments $V_1, \dots, V_l$. Based on a secret key $\mathsf{sk}$ and the n-gram of the preceding $n$ tokens acting as the watermark key, a distribution channel $P_i$ is pseudo-randomly selected, and the next token is sampled from $P_i$.
- **Detection Phase**: Given the generated text and the secret key, the channel index $i_t$ for each step is recovered, and the statistic $\Phi = \sum_{t=1}^{T} \mathbf{1}_{x_t \in V_{i_t}}$ is computed. Under the null hypothesis (no watermark), $\Phi$ follows a binomial distribution (with success probability $1/l$), and hypothesis testing is used to determine the presence of a watermark.

### Key Designs

1. **Distribution Channel Concept**:

    - **Function**: Abstraction of the watermarked distribution space into a set of "channels" $\{P_1, \dots, P_l\}$, where each channel is a complete probability distribution.
    - **Mechanism**: In channel $P_i$, the probabilities of tokens belonging to $V_i$ are boosted, while the probabilities of tokens in other segments are reduced accordingly. The watermark key determines which channel is selected. The critical constraint is that the weighted average of all channels must recover the original distribution: $\frac{1}{l} \sum_{i=1}^{l} P_i(x) = P_M(x|x_{1:t})$.
    - **Design Motivation**: Prior methods (DiPmark, $\gamma$-reweight, STA-1) inherently use only 2 channels (red-green list), which limits signal strength. Generalizing to $l$ channels allows for a more significant reallocation of probabilities.

2. **Closed-Form Solution to the Optimization Problem**:

    - **Function**: Find the channel probability distribution that maximizes detectability while satisfying the unbiasedness constraint.
    - **Mechanism**: Define the optimization objective as $\max \sum_{i=1}^{l} P_{i,V_i}$ (maximizing the total probability of each channel on its corresponding segment), subject to the constraints that row sums equal 1 (validity) and column sums equal $l \cdot P_{V_j}$ (unbiasedness). The optimal solution is closed-form: $P_{i,V_i} = \min\{1, l \cdot P_{V_i}\}$, and for $i \ne j$, $P_{i,V_j} = \frac{(1 - lP_{V_i})_+ (lP_{V_j} - 1)_+}{\sum_k (1 - lP_{V_k})_+}$.
    - **Design Motivation**: Avoid numerical optimization and compute the closed-form solution directly to ensure time efficiency comparable to methods like KGW. Moreover, this is theoretically proven to be globally optimal.

3. **Watermark Key Design and n-gram Strategy**:

    - **Function**: Select the channel using a pseudo-random seed generated from the secret key $\mathsf{sk}$ and the n-gram of the preceding $n$ tokens.
    - **Mechanism**: Adopts a key generation strategy consistent with the KGW method, ensuring that the channel selection for each token appears random but can be deterministically recovered.
    - **Design Motivation**: The n-gram dependency makes the channel selection context-dependent, increasing the unpredictability of the watermark.

### Theoretical Analysis: Expected True-Negative Rate

The authors define the Expected True-Negative Rate ($E_{TN}$) to quantify detectability. When $l=2$:
- DiPmark: $E_{TN}^{\text{DiP}} = \max\{P_{V_r} - \alpha, 0\} + \max\{P_{V_r} - (1-\alpha), 0\}$
- STA-1: $E_{TN}^{\text{STA}} = P_{V_r}^2$
- MCmark: $E_{TN}^{\text{MCmark}} = |1/2 - P_{V_r}|$

Under the assumption that $P_{V_r}$ is uniformly distributed in $[0,1]$, the expected $E_{TN}$ of MCmark is $1/4$, which is lower than DiPmark's $(\alpha - 1/2)^2 + 1/4$ and STA-1's $1/3$, and has the smallest variance ($1/48$). This indicates that MCmark can embed detectable signals more stably.

### Robustness-Detectability Trade-off

Increasing $l$ yields a trade-off:
- **Detectability**: Moderately increasing $l$ improves detectability (more channels $\rightarrow$ more extreme probability reallocation).
- **Robustness**: As $l$ increases, each segment $V_i$ becomes smaller. If an attacker modifies a token, the probability of it landing in the correct segment is only $1/l$, which is beneficial for robustness. However, an excessively large $l$ leads to an uneven distribution of $P_{V_i}$ (some segments have extremely low probabilities), which instead increases false negatives.

In experiments, $l=20$ achieves the best balance between detectability and robustness.

### Loss & Training
MCmark does not require training. Watermarks are embedded during inference by modifying the sampling strategy.

## Key Experimental Results

### Main Results: Detectability Comparison (LLaMA2 + C4 Dataset)

| Method | TPR@FPR=1% | TPR@FPR=0.1% | TPR@FPR=0.01% | Median p-value |
|------|-----------|-------------|--------------|----------------|
| KGW (δ=2.0, biased) | 99.34% | 98.79% | 97.79% | 6.58e-22 |
| Unigram (δ=2.0, biased) | 99.88% | 99.52% | 98.93% | 5.39e-25 |
| DiPmark (α=0.4) | 87.66% | 78.77% | 71.77% | 1.24e-7 |
| DiPmark (α=0.3) | 81.88% | 69.88% | 61.65% | 5.28e-6 |
| STA-1 | 84.93% | 71.58% | 57.76% | 2.66e-5 |
| γ-reweight | 89.17% | 81.79% | 75.83% | 4.47e-8 |
| EXP-edit | 89.01% | 86.35% | - | 2.00e-4 |
| **MCmark (l=20)** | **98.96%** | **98.38%** | **97.69%** | **8.10e-30** |

Among unbiased watermarks, MCmark improves TPR@FPR=0.1% by over 14% (98.38% vs 81.79%), comparable to the strongest biased watermark, KGW ($\delta=2.0$).

### Robustness Comparison (Token Substitution Attack, TPR@FPR=0.1%)

| Method | ε=0.05 | ε=0.1 | ε=0.2 |
|------|--------|-------|-------|
| DiPmark (α=0.4) | 69.63% | 58.13% | 29.06% |
| STA-1 | 60.84% | 47.15% | 21.35% |
| EXP-edit | 81.35% | 78.27% | 74.88% |
| **MCmark (l=20)** | **97.11%** | **96.07%** | **88.79%** |

### Robustness to GPT Paraphrasing Attack

| Method | TPR@FPR=1% | Median p-value | AUC |
|------|-----------|----------------|-----|
| DiPmark (α=0.4) | 6.4% | 6.03e-1 | 0.4921 |
| STA-1 | 11.6% | 2.32e-1 | 0.6850 |
| EXP-edit | 17.9% | 2.30e-1 | 0.6879 |
| **MCmark (l=20)** | **48.0%** | **1.26e-2** | **0.8592** |

### Ablation Study: Impact of Channel Number $l$

| Configuration | Description |
|------|------|
| l=2 | Degenerates to a red-green list; detectability is close to DiPmark, though the theoretical optimal solution is already superior to DiPmark. |
| l=5~20 | Detectability continues to increase, while robustness remains high. |
| l=20 | The best balance point, where both TPR and robustness are optimal. |
| l>100 | The p-value of detectability continues to drop, but robustness begins to decrease significantly. |
| l=32000 (=\|V\|) | Each segment contains only 1 token, degenerating to Gumbel-max with the worst robustness. |

### Key Findings
- The channel number $l$ is the core hyperparameter: $l=20$ is the practical sweet spot. Under this setting, detectability almost matches the strongest biased watermark (KGW with $\delta=2.0$), while robustness far exceeds all unbiased baselines.
- Unbiasedness validation: Across the entire range of $l$ from 2 to 32000, MCmark performs consistently with the watermark-free baseline on translation (BLEU) and summarization (perplexity) tasks, validating the theoretical unbiasedness guarantee.
- Under GPT paraphrasing attacks, MCmark retains an AUC of 0.8592, vastly outperforming counterparts (where the highest is only 0.6879). This is highly meaningful in practice, as users are most likely to use GPT rewriting to remove watermarks.
- Under GPT back-translation attacks, MCmark's TPR@FPR=0.01% is as high as 81.2%, while DiPmark is only 19.8% and STA-1 is 11.1%.
- Under DIPPER attacks (the strongest text rewriting attack), all methods degrade significantly, but MCmark (AUC=0.695) remains the best.

## Highlights & Insights
- **The concept of distribution channels is extremely elegant**: It unifies unbiased watermarking as an optimization problem of "how to allocate probabilities into $l$ channels", resulting in a core algorithm with a single closed-form formula (Eq. 3). This highly concise formalization provides both theoretical guarantees and ease of implementation.
- **The generalization from binary to multi-way division is highly intuitive**: Prior methods default to red-green binary divisions. MCmark shows this is not necessary—signal density increases substantially at $l=20$. This simple but counter-intuitive observation is the most significant contribution of the paper.
- **Theorization of the detectability-robustness trade-off**: A clear trade-off analysis is provided—larger $l$ increases detectability but weakens robustness (single-token modifications are more likely to disrupt signals). This framework can guide the design of other watermarking methods.
- **Transfer potential**: The multi-channel design concept can translate to image or audio watermarking—as long as there is a concept of discrete tokens, segmented boosting can be applied.

## Limitations & Future Work
- **Adaptive selection of $l$**: The paper fixes $l=20$, but the optimal $l$ may vary across different text lengths and domains. An adaptive selection strategy for $l$ (dynamically adjusting based on the entropy of the vocabulary probability distribution) could further improve performance.
- **Vocabulary splitting strategy**: Currently, a uniform random split is used. Grouping tokens based on semantic similarity (making tokens in each segment semantically related) might improve text naturalness without sacrificing detectability.
- **Insufficient validation in long-text scenarios**: Experiments are mostly conducted on short texts (C4 dataset snippets). Performance on long documents (such as academic papers or reports) remains unexplored.
- **Game-theoretic play against strong rewriting attacks**: Although the AUC under GPT paraphrasing is 0.8592, the TPR@FPR=0.01% is still only 20.2%, indicating room for improvement in high-precision requirement scenarios.

## Related Work & Insights
- **vs KGW/Unigram (Biased Watermarks)**: KGW achieves strong detectability by adding a bias $\delta$ to the green list, but alters the text distribution. MCmark achieves comparable detectability without changing the distribution (TPR close to KGW $\delta=2.0$ when $l=20$), alongside a theoretical guarantee of unbiasedness.
- **vs DiPmark**: DiPmark is a special case of MCmark when $l=2$. By generalizing to $l>2$, MCmark gains substantial improvements (TPR@FPR=0.1% increases from 78.77% to 98.38%).
- **vs EXP-edit**: EXP-edit detection requires thousands of LLM inferences and has no theoretical FPR guarantee. MCmark's detection is one-pass and has a strict upper bound on the false positive rate.
- **vs STA-1**: STA-1 optimizes text quality for low-entropy settings. MCmark maintains its advantage across all entropy levels and is more general (requiring no adjustment for specific scenarios).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Generalizing from binary to multi-way division is intuitively simple but yields surprisingly strong results with solid theoretical contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Thorough ablation studies and coverages across multiple attacks (token substitution, GPT paraphrasing, back-translation, DIPPER), multiple models, and datasets.
- **Writing Quality**: ⭐⭐⭐⭐ Clear mathematical derivations and organized experiments, though the notation system occasionally requires cross-referencing.
- **Value**: ⭐⭐⭐⭐ Significant push toward the practical applicability of LLM watermarking, offering clear practical guidelines with $l=20$ and open-sourced code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)
- [\[ICLR 2026\] Analyzing and Evaluating Unbiased Language Model Watermark](../../ICLR2026/llm_safety/analyzing_and_evaluating_unbiased_language_model_watermark.md)
- [\[ICML 2025\] De-mark: Watermark Removal in Large Language Models](../../ICML2025/llm_safety/de-mark_watermark_removal_in_large_language_models.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2025\] MEGen: Generative Backdoor into Large Language Models via Model Editing](megen_generative_backdoor_into_large_language_models_via_model_editing.md)

</div>

<!-- RELATED:END -->
