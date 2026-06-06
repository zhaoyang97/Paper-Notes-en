---
title: >-
  [Paper Note] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition
description: >-
  [ICML 2026][LLM Pretraining][scaling law] The authors propose InfoLaw: redefining "pretraining" as a process of "bucket-wise information accumulation…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "scaling law"
  - "data quality"
  - "data repetition"
  - "data recipe"
  - "information content"
date: 2026-05-08
content_hash: 19032d53197e6fd5
---

# InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition

**Conference**: ICML 2026  
**arXiv**: [2605.02364](https://arxiv.org/abs/2605.02364)  
**Code**: None  
**Area**: LLM Pretraining / Scaling Law / Data Recipe  
**Keywords**: scaling law, data quality, data repetition, data recipe, information content

## TL;DR
The authors propose InfoLaw: redefining "pretraining" as a process of "bucket-wise information accumulation," where the information in each bucket equals "quality density $f_d$ × unique token count $M_d$ × $\log K$" multiplied by an exponentially decaying factor with respect to repetition $R_d$. The final validation loss is expressed as $L = \alpha\cdot\text{info}^{-\beta}$, which can be fitted on 252M-1.2B and extrapolated to 7B / 425B tokens with an average error of 0.15% and a maximum of 0.96%. This formulation can be directly used to search for the optimal data recipe.

## Background & Motivation

**Background**: The Chinchilla scaling law expresses loss as $L = E + A/N^\alpha + B/D^\beta$, which allows precise extrapolation when data is abundant. However, in practice, overtraining has become mainstream (e.g., LLaMA / Qwen series), and high-quality tokens are insufficient, necessitating repetition or mixing with lower-quality data.

**Limitations of Prior Work**: (i) The standard scaling law systematically underestimates large model loss under repetition (Fig 1: power-law fitted on 252M-1.2B deviates significantly at 2.5B); (ii) Different mixture recipes (e.g., more high-quality with less diversity, or more low-quality with greater diversity) fall on different curves, making cross-recipe comparison impossible; (iii) Optimal recipe search can only be done via small-scale grid search, but the optimal recipe at small and large scales is inconsistent—this leaves data recipe decisions stuck at "if you can't predict, you can only blindly burn GPUs."

**Key Challenge**: Higher quality yields higher per-token value, but limited tokens must be repeated, and repetition brings diminishing returns; compute alone cannot simultaneously capture the three variables of "quality × repetition × scale."

**Goal**: To build a data-aware scaling law that can extrapolate loss across four dimensions: (mixture recipe, model size, training tokens, overtrain ratio), and search for the optimal data recipe without additional experiments.

**Key Insight**: Change the axis! Since compute $C$ is insufficient, construct a new "effective data signal"—view training as "information accumulation," so that different mixtures yield the same loss at the same information content, naturally collapsing all experimental points onto a single power-law curve.

**Core Idea**: Information content = ($f_d M_d \log K$) × (1 − $e^{-\lambda(N) R_d/\log K}$); the first term is the "potential information" in the data, the second is the "proportion learned by the model." All mixture × scale × repetition experiments are unified on the $L$-info plane as a single power law $L = \alpha\cdot\text{info}^{-\beta}$.

## Method

### Overall Architecture
(1) Deduplicate Common Crawl globally to obtain 3.7T tokens, score with two quality classifiers (FineWeb-edu + DCLM), and split into 6 buckets by percentile (0-5%, 5-20%, 20-40%, 40-60%, 60-80%, 80-100%); (2) Design LayerMix sampling: assign a set of weights $w=[w_0,...,w_5]$ (with $w_d\geq w_{d+1}$ and sum to 1), sample $K$ tokens from source $S$ according to $w$ to form the training set, with repetition per bucket $R_d = w_d K / M_d$ ($M_d = \min(w_d K, B_d S)$); (3) Train 9 models (252M-1.2B) × 3 LayerMix recipes (HQ/MQ/LQ), fixed 3.6× overtrain, for 27 runs to collect loss data; (4) Fit four sets of parameters: $f_d, \lambda(N), \alpha, \beta$; (5) Validate extrapolation on 1.5B-7B, and use the fitted curve to search for the optimal recipe.

### Key Designs

1. **LayerMix Sampling: Explicit Parameterization of "Quality × Repetition"**:

    - **Function**: Uses a set of weights $w=[w_0,\ldots,w_5]$ to simultaneously control "quality distribution" and "degree of repetition," allowing pretraining to be decomposed into independent contributions from 6 buckets.
    - **Mechanism**: First, split into 6 buckets by quality percentile, with fixed source proportions $B=[0.05, 0.15, 0.20, 0.20, 0.20, 0.20]$; when constructing the training set, the $d$-th bucket packs $K_d = w_d K$ tokens, with unique token count $M_d = \min(K_d, B_d S)$, and average repetition $R_d = K_d / M_d$, so $R_d=1$ when $K_d\leq S_d$, otherwise $>1$. The constraint $w_d\geq w_{d+1}$ enforces priority for high-quality buckets.
    - **Design Motivation**: Traditional scaling laws assume tokens are non-repetitive and of uniform quality, making it impossible to separate their contributions; LayerMix turns the data recipe into a 6-dimensional parameter space, where each $(w_d, R_d)$ can independently estimate its marginal contribution to final loss, laying the foundation for decomposing loss into "additive information."

2. **Information Accumulation Formula: Quality Density + Exponential Decay + Log Normalization**:

    - **Function**: Expresses "how much effective information is gained from reading a segment of data" in a unified formula, reflecting quality, repetition, model capacity, and training scale.
    - **Mechanism**: First, use first-order exponential decay to model "information gain from reading the same document for the $t$-th time" $I_{i\_\text{part}}(t, \lambda(N)) = I_i\cdot\lambda(N)e^{-\lambda(N)t}$; integrating gives "total information after $T$ reads" $I_{i,\text{total}}(T) = I_i(1-e^{-\lambda(N)T})$, reflecting diminishing returns from repetition. Add $\log K$ normalization (empirically necessary for cross-scale generalization) to get $I_{i,\text{total}} = I_i\log K(1-e^{-\lambda(N)T/\log K})$. Finally, sum over all buckets: $\text{info}(w, K, S, f, \lambda(N)) = \sum_d f_d M_d \log K\cdot(1 - e^{-\lambda(N) R_d/\log K})$, where $f_d = e^{-\theta d}$ is monotonically decreasing quality density, and $\lambda(N) = a\ln N + b$ is the learning rate determined by model capacity.
    - **Design Motivation**: Decouples "information density" and "learning ability"—$f_d$ describes the data's potential, $\lambda(N)$ describes the model's ability to absorb data, and the $(1-e^{-\cdot})$ term naturally yields the intuition that "small models saturate early, large models can extract more information."

3. **Two-Step Fitting of $f_d$ and $\lambda(N)$**:

    - **Function**: Determines parameterization without overfitting the 27 experimental points, and extrapolates to unseen $N$.
    - **Mechanism**: (i) Assume $f_d = e^{-\theta d}$ ($\theta > 0$ ensures monotonic decrease), treat $\lambda(N)$ as a discrete variable, sample 100,000 sets $(\theta, \{\lambda_N\})$ from parameter space, use Spearman correlation $\rho_s(L, \text{info})$ as the fitting metric to select optimal $(\theta^*, \lambda_N^*)$, yielding $\theta^*=0.922$; (ii) Fit $\lambda(N)=a\ln N+b$ using each model's $\lambda_N^*$ as data points, resulting in $a^*=0.140, b^*=0.018$; the logarithmic function remains monotonically saturating in the extrapolation range, matching the intuition that "larger models have marginally increasing learning rates without divergence." Final loss-info fit: $\alpha=3.7373, \beta=0.0441$.
    - **Design Motivation**: Spearman is a monotonicity metric, insensitive to absolute scale, aligning with the semantic "info monotonically corresponds to loss"; enforcing parametric forms for $f_d, \lambda(N)$ rather than table lookup ensures controllable extrapolation to 1.5B-7B.

### Loss & Training

All 27 fit experiments use a fixed overtrain ratio $m=3.6$, Transformer + SwiGLU + RoPE, 250k vocabulary, bf16; extrapolation experiments use $m'=25$ with 1.2B/640B tokens for validation. The fitted InfoLaw and 100k candidate mixture samples are used to directly select the optimal $w$ (no further training required).

## Key Experimental Results

### Main Results

| Evaluation Scenario | Configuration | Mean/Max Absolute Error |
|---------------------|--------------|------------------------|
| Unseen LayerMix (MLQ/MHQ) ×252M-1.2B | Within-range mixture | Perfect collapse onto curve |
| 1.5B-2.5B × MQ/LQ/HQ | Model extrapolation | Mean 0.15% / Max 0.96% |
| 7B × MLQ/MHQ × 300B tokens | Unseen mixture + scale | Mean 0.15% / Max 0.96% |
| 25× overtrain (1.2B, 640B tokens) | Cross overtrain domain | Fitted curves nearly parallel, only intercept shifts |
| 2.5B optimal recipe search | $w^*=[0.50, 0.49, 0.01, 0, 0, 0]$ | Outperforms 4 baselines |

### Ablation Study

| Setting | Key Metric | Description |
|---------|------------|-------------|
| No $\log K$ normalization | Cannot generalize across scales | Validates necessity of $\log K$ (Appendix B) |
| $\lambda(N)$ as power-law / exp | Poor extrapolation | Log function fits best |
| Traditional scaling law $L(C)$ | Systematic underestimation for large models | Fig 1 shows clear deviation at 7B |
| 1.2B + 25 random mixtures | Pearson 0.76 | InfoLaw can rank unseen recipes |

### Key Findings
- **Small models / few tokens prefer quality, large models / many tokens prefer diversity**: Table 2 shows that for 7B at 300B tokens, optimal $w_0=0.548$, $w_1=0.444$; at 1000B tokens, $w_0$ drops to 0.395, $w_2$ rises to 0.214. This refutes the naive intuition that "more high quality is always better."
- **Marginal returns of repetition decay exponentially**: Under the HQ recipe, the top 5% bucket is repeated 16×, under MQ 10×; initially, both have similar loss, but HQ converges more slowly later, matching the saturation of $1-e^{-\lambda R/\log K}$ at large $R$.
- **Info collapse is the core empirical evidence**: The originally scattered 27 points (different $w, N, K$) collapse into a straight line on the $L$-info plot (Fig 3f), providing the most intuitive evidence for InfoLaw.
- **Overtrain only shifts intercept, not slope**: The $m=3.6$ and $m'=25$ curves are nearly parallel in log-log space, meaning InfoLaw does not require refitting $\beta$ for each overtrain ratio.

## Highlights & Insights
- **Changing the x-axis is an elegant idea**: When compute $C$ alone lacks explanatory power, instead of adding more terms, switch to a composite axis—info—that accommodates "quality × repetition." Compared to Chinchilla's approach of "adding N and D terms," InfoLaw is more "data-centric."
- **$\log K$ normalization is a key hack**: The appendix empirically shows that without it, cross-scale generalization fails, suggesting that scaling law design should consider not only model size but also the dilution effect of data scale on learning rate.
- **Small models as recipe searchers**: Using 252M-1.2B as an "experimental platform" plus 100k cheap mixture candidates allows direct recipe selection for 7B, representing a data-efficient "prior scheduling" paradigm.
- **One formula answers two questions**: It can both predict large model loss and select data recipes; the former is "diagnosis," the latter "decision," extending scaling law usage from passive to active.

## Limitations & Future Work
- Fitted data only covers three mixtures for 252M-1.2B; not validated on MoE, long context, or code/math specialized subsets.
- Quality bucketing depends on the average of two external classifiers; biases in the classifiers propagate to $f_d$, with no sensitivity analysis performed.
- Whether $\lambda(N) = a\ln N + b$ remains monotonically saturating for extremely large $N$ is unverified; the paper only tests up to 7B, so extrapolation risk for >100B is unknown.
- Only validation perplexity / five-task average is used as loss; whether "factual knowledge" or "reasoning depth" still monotonically correspond to info is unknown.
- Curriculum order (hard-to-easy vs. random) is not considered; LayerMix defaults to random packing.

## Related Work & Insights
- **vs Chinchilla (Hoffmann 2022)**: Chinchilla assumes unlimited data and only considers $N$ and $D$; InfoLaw decomposes $D$ into "quality × repetition × bucket proportion," providing greater accuracy when data is limited.
- **vs Muennighoff 2023 (scaling law under data constraints with repetition)**: They introduce $R_{\text{D}}^*$ as an effective repetition coefficient for single-source repetition; InfoLaw extends this to multi-bucket mixtures and incorporates quality density $f_d$.
- **vs RegMix (Liu 2024)**: RegMix uses small proxy models for regression-based recipe selection, requiring many proxies; InfoLaw uses a single info-loss power law, eliminating the need for further training during search.
- **vs DataComp-LM / FineWeb (Penedo, Li)**: These works provide high-quality data pools but do not address "proportional mixing"; InfoLaw offers a principled recipe selector that can be seamlessly integrated after DataComp.

## Rating
- Novelty: ⭐⭐⭐⭐ The "info as coordinate" perspective is refreshing, with formula construction and physical intuition (diminishing returns from repetition + log normalization) highly aligned.
- Experimental Thoroughness: ⭐⭐⭐⭐ 27 fits + 1.5B-7B multi-scale extrapolation + 25× overtrain + recipe search validation, with coverage density among the highest in pretraining papers; the only regret is evaluation only on standard dense Transformers, not MoE.
- Writing Quality: ⭐⭐⭐⭐ Fig 1 directly reveals the problem via loss-C, Fig 3 visually demonstrates info collapse, and formula derivations are clear; the appendix completes the normalization ablation.
- Value: ⭐⭐⭐⭐ Provides practical pretraining teams with a "small-scale fitting → large-scale recipe selection" tool, with huge potential for GPU savings and high industry value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Softplus Attention with Re-weighting Boosts Length Extrapolation in Large Language Models](softplus_attention_with_re-weighting_boosts_length_extrapolation_in_large_langua.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[ICML 2026\] Predicting Large Model Test Losses with a Noisy Quadratic System](predicting_large_model_test_losses_with_a_noisy_quadratic_system.md)
- [\[ICML 2026\] Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](decomposing_the_basic_abilities_of_large_language_models_mitigating_cross-task_i.md)

</div>

<!-- RELATED:END -->
