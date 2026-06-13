---
title: >-
  [Paper Note] InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition
description: >-
  [ICML 2026][LLM Pretraining][scaling law] The authors propose InfoLaw, which redefines "pre-training" as a process of "accumulating information by buckets." The information quantity per bucket is defined as "quality dens…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "scaling law"
  - "data quality"
  - "data repetition"
  - "data recipe"
  - "information quantity"
date: 2026-05-08
content_hash: ff241af2600f69b7
---

# InfoLaw: Information Scaling Laws for Large Language Models with Quality-Weighted Mixture Data and Repetition

**Conference**: ICML 2026  
**arXiv**: [2605.02364](https://arxiv.org/abs/2605.02364)  
**Code**: None  
**Area**: LLM Pre-training / Scaling Law / Data Recipe  
**Keywords**: scaling law, data quality, data repetition, data recipe, information quantity

## TL;DR
The authors propose InfoLaw, which redefines "pre-training" as a process of "accumulating information by buckets." The information quantity per bucket is defined as "quality density $f_d \times$ unique tokens $M_d \times \log K$," multiplied by a factor that decays exponentially with the number of repetitions $R_d$. By expressing the validation loss as $L = \alpha\cdot\text{info}^{-\beta}$, the model can be fitted on 252M-1.2B parameters and extrapolated to 7B models or 425B tokens. It achieves an average error of 0.15% (max 0.96%) and can be directly used to search for optimal data recipes.

## Background & Motivation

**Background**: Chinchilla-style scaling laws express loss as $L = E + A/N^\alpha + B/D^\beta$, which accurately extrapolates when data is abundant. However, in practice, overtraining has become mainstream (e.g., LLaMA / Qwen series), where high-quality tokens are insufficient, necessitating repetition or mixing with lower-quality data.

**Limitations of Prior Work**: (i) Standard scaling laws systematically underestimate the loss of large models under repetition (Fig 1: a power-law fitted on 252M-1.2B significantly deviates at 2.5B); (ii) Different mixture recipes (e.g., more high-quality with less diversity vs. less low-quality with more diversity) fall on different curves, making cross-recipe comparison impossible; (iii) Finding the optimal recipe requires small-scale grid searches, but optimal recipes at small scales often do not align with those at large scales—trapping data recipe decisions in a state where one must "blindly burn GPUs" without predictive capabilities.

**Key Challenge**: Higher quality implies higher value per instance, but limited tokens must be repeated, leading to diminishing returns. The single axis of "compute" is no longer sufficient to simultaneously characterize the three variables: "quality × repetition × scale."

**Goal**: To build a data-aware scaling law that can extrapolate loss across four dimensions (mixture recipe, model size, training tokens, and overtraining ratio) and search for the optimal data recipe without running additional experiments.

**Key Insight**: Change the axis! Since compute $C$ is insufficient, the authors construct a new "effective data signal." Training is viewed as "information accumulation," allowing different mixtures to produce the same loss at the same information level, which naturally collapses all experimental points onto the same power-law line.

**Core Idea**: Information quantity is defined as $(f_d M_d \log K) \times (1 - e^{-\lambda(N) R_d/\log K})$, where the former term represents the "potential information" in the data and the latter represents the "proportion learned by the model." This unifies all mixture × scale × repetition experiments onto a single power-law $L = \alpha\cdot\text{info}^{-\beta}$ on the $L$-info plane.

## Method

### Overall Architecture
(1) Perform global de-duplication on 3.7T tokens from Common Crawl. Use two quality classifiers (FineWeb-edu + DCLM) to score and partition the data into 6 buckets based on percentiles (0-5%, 5-20%, 20-40%, 40-60%, 60-80%, 80-100%). (2) Design LayerMix sampling: given a set of weights $w=[w_0,...,w_5]$ (where $w_d\geq w_{d+1}$ and $\sum w_d = 1$), sample $K$ tokens from source $S$ to form the training set. Repetition for each bucket is $R_d = w_d K / M_d$, where $M_d = \min(w_d K, B_d S)$. (3) Train 27 runs featuring 9 model scales (252M-1.2B) × 3 LayerMix recipes (HQ/MQ/LQ) with a fixed 3.6× overtraining ratio to collect loss data. (4) Fit four sets of parameters: $f_d, \lambda(N), \alpha, \beta$. (5) Validate extrapolation on 1.5B-7B models and use the fitted curve to search for optimal recipes.

### Key Designs

1.  **LayerMix Sampling: Explicitly Parametrizing "Quality × Repetition"**:
    - **Function**: Uses a weight vector $w$ to simultaneously control "quality distribution" and "repetition level," allowing one pre-training run to be decomposed into independent contributions from 6 buckets.
    - **Mechanism**: Data is split into 6 buckets by quality, with fixed source pool proportions $B$. The training set for bucket $d$ contains $K_d = w_d K$ tokens, using $M_d = \min(K_d, B_d S)$ unique tokens, resulting in an average repetition $R_d = K_d / M_d$. $R_d=1$ if $K_d\leq S_d$, otherwise $R_d>1$. The constraint $w_d\geq w_{d+1}$ prioritizes high-quality buckets.
    - **Design Motivation**: Traditional scaling laws assume non-repeating, uniform quality data, failing to separate their contributions. LayerMix transforms the data recipe into a 6D parameter space, where the marginal contribution of $(w_d, R_d)$ for each dimension can be estimated independently, enabling the decomposition of loss into "additive information."

2.  **Information Accumulation Formula: Quality Density + Exponential Decay + Log Normalization**:
    - **Function**: Provides a unified formula for "how much effective information is gained from reading a segment of data," reflecting quality, repetition, model capacity, and training scale.
    - **Mechanism**: First, apply first-order exponential decay to model the information gain of reading the same document for the $t$-th time: $I_{i\_\text{part}}(t, \lambda(N)) = I_i\cdot\lambda(N)e^{-\lambda(N)t}$. Integration yields total accumulated info: $I_{i,\text{total}}(T) = I_i(1-e^{-\lambda(N)T})$, reflecting "diminishing returns of repetition." Incorporating $\log K$ normalization (found empirically necessary for cross-magnitude generalization) gives $I_{i,\text{total}} = I_i\log K(1-e^{-\lambda(N)T/\log K})$. Finally, sum across all buckets: $\text{info}(w, K, S, f, \lambda(N)) = \sum_d f_d M_d \log K\cdot(1 - e^{-\lambda(N) R_d/\log K})$, where $f_d = e^{-\theta d}$ is the monotonically decreasing quality density and $\lambda(N) = a\ln N + b$ is the learning rate determined by model capacity.
    - **Design Motivation**: Decouples "information density" from "learning ability"—$f_d$ describes the potential of the data itself, while $\lambda(N)$ describes the model's capacity to digest data. The $(1-e^{-\cdot})$ term naturally captures why small models saturate early while large models can extract more information.

3.  **Two-step Fitting Process for $f_d$ and $\lambda(N)$**:
    - **Function**: Determines parametric forms and extrapolates to unseen $N$ without overfitting the 27 experimental points.
    - **Mechanism**: (i) Assume $f_d = e^{-\theta d}$ ($\theta > 0$) and treat $\lambda(N)$ as discrete variables. Sample 100k sets of $(\theta, \{\lambda_N\})$ from the parameter space and select the optimal $(\theta^*, \lambda_N^*)$ using Spearman correlation $\rho_s(L, \text{info})$ as the fit metric, yielding $\theta^*=0.922$. (ii) Fit the $\lambda_N^*$ values to $\lambda(N)=a\ln N+b$, resulting in $a^*=0.140, b^*=0.018$. The logarithmic function remains monotonic and saturating in the extrapolation range, consistent with the intuition that larger models have marginally increasing but non-divergent learning rates. The final loss-info fit yields $\alpha=3.7373, \beta=0.0441$.
    - **Design Motivation**: Spearman correlation measures monotonic consistency, which is insensitive to absolute scales and fits the semantic that "info monotonically corresponds to loss." Enforcing parametric forms for $f_d$ and $\lambda(N)$ ensures controllable extrapolation to 1.5B-7B.

### Loss & Training
All 27 fitting experiments used a fixed overtraining ratio $m=3.6$, Transformer + SwiGLU + RoPE, 250k vocabulary, and bf16. Extrapolation experiments used $m'=25$ with 1.2B models and 640B tokens. The fitted InfoLaw was used with 100k mixture candidate samplings to directly select the optimal $w$ (no retraining required).

## Key Experimental Results

### Main Results

| Evaluation Scenario | Configuration | Avg/Max Absolute Error |
|---------|------|--------------------|
| Unseen LayerMix (MLQ/MHQ) × 252M-1.2B | In-distribution mixture | Perfect collapse to curve |
| 1.5B-2.5B × MQ/LQ/HQ | Model extrapolation | Avg 0.15% / Max 0.96% |
| 7B × MLQ/MHQ × 300B tokens | Unseen mixture + scale | Avg 0.15% / Max 0.96% |
| 25× overtrain (1.2B, 640B tokens) | Cross-overtraining domain | Parallel fit with intercept shift |
| 2.5B Optimal Recipe Search | $w^*=[0.50, 0.49, 0.01, 0, 0, 0]$ | Defeats 4 baselines |

### Ablation Study

| Setting | Key Metric | Description |
|------|---------|------|
| No $\log K$ normalization | Fails across scales | Confirms $\log K$ is essential (Appendix B) |
| $\lambda(N)$ using power-law / exp | Poor extrapolation | Logarithmic function fits best |
| Traditional scaling law $L(C)$ | Large model underestimation | Significant deviation at 7B in Fig 1 |
| 1.2B + 25 random mixtures | Pearson 0.76 | InfoLaw can rank unseen recipes |

### Key Findings
- **Small models / few tokens prefer quality; large models / many tokens prefer diversity**: Table 2 shows that for 7B with 300B tokens, optimal $w_0=0.548, w_1=0.444$. At 1000B tokens, $w_0$ drops to 0.395 and $w_2$ rises to 0.214. This contradicts the naive intuition that "more high-quality data is always better."
- **Diminishing marginal returns of repetition are exponential**: In the HQ recipe, the top 5% bucket is repeated 16×, while in MQ it is repeated 10×. Losses are similar early on, but HQ converges slower later, aligning with the saturation of $1-e^{-\lambda R/\log K}$ at large $R$.
- **Info collapse is the core empirical evidence**: The 27 scattered points (different $w, N, K$) collapse into a straight line on the $L$-info plot (Fig 3f), providing the most intuitive evidence for InfoLaw.
- **Overtraining shifts intercept, not slope**: The curves for $m=3.6$ and $m'=25$ are nearly parallel in the log-log plane, meaning InfoLaw does not need to re-fit $\beta$ for every overtraining ratio.

## Highlights & Insights
- **Elegant axis shifting**: When compute $C$ lacks explanatory power, rather than adding more terms, the authors switch to a synthetic axis—info—that accommodates "quality × repetition." Compared to the Chinchilla approach of adding $N$ and $D$ terms, InfoLaw is more "data-centric."
- **$\log K$ normalization is a key hack**: The empirical evidence in the appendix showing its necessity for cross-scale generalization suggests that scaling law designs should consider the dilution effect of "data scale itself on the learning rate."
- **Small models as recipe searchers**: Using 252M-1.2B models as an "experimental platform" + 100k cheap mixture samplings to select recipes for 7B models represents a paradigm of data-efficient "prior scheduling."
- **One formula, two questions**: InfoLaw predicts large model loss and selects data recipes. The former is "diagnosis," the latter is "decision-making," extending the use of scaling laws from passive to active.

## Limitations & Future Work
- Fitting data only covers 252M-1.2B and three mixtures; it has not been verified on MoE, long context, or specialized subsets like code/math.
- Quality bucketing relies on the average of two external classifiers; biases in the classifiers will propagate to $f_d$, and no sensitivity analysis was conducted.
- Whether $\lambda(N) = a\ln N + b$ remains monotonically saturating at extremely large $N$ is unknown; the paper only validates up to 7B.
- Loss is defined only by validation perplexity / five-task average; it is unknown if high-level capabilities like "factual knowledge" or "reasoning depth" remain monotonically mapped to info.
- Curriculum ordering (e.g., easy-to-hard vs. random) is not considered; LayerMix defaults to random packing.

## Related Work & Insights
- **vs. Chinchilla (Hoffmann 2022)**: Chinchilla assumes infinite data and considers only $N$ and $D$. InfoLaw decomposes $D$ into "quality × repetition × bucket ratio," making it more accurate in data-constrained scenarios.
- **vs. Muennighoff 2023 (Scaling Law under Data Constraints)**: They introduced a repetition efficiency coefficient $R_{\text{D}}^*$ for single-source repetition; InfoLaw extends this to multi-bucket mixtures and incorporates quality density $f_d$.
- **vs. RegMix (Liu 2024)**: RegMix uses small proxy models to regress and select recipes, requiring many proxy training runs. InfoLaw uses a single info-loss power law, requiring no extra training during search.
- **vs. DataComp-LM / FineWeb (Penedo, Li)**: They provide high-quality data pools but do not specify "how to mix in proportion." InfoLaw provides a principled recipe selector that can seamlessly integrate with DataComp.

## Rating
- Novelty: ⭐⭐⭐⭐ "Info as a coordinate" is a fresh perspective; formula construction aligns well with physical intuition.
- Experimental Thoroughness: ⭐⭐⭐⭐ 27 fits + 1.5B-7B multi-scale extrapolation + 25× overtrain + recipe search validation; high density for a pre-training paper, though MoE is absent.
- Writing Quality: ⭐⭐⭐⭐ Fig 1 reveals the problem directly, Fig 3 proves the conclusion with info collapse, and logic is clear; appendix covers necessary ablations.
- Value: ⭐⭐⭐⭐ Provides practical pre-training teams with a tool for "small-scale fitting → large-scale recipe selection," with significant potential to save GPU resources.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](../../NeurIPS2025/llm_pretraining/scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Dropout Universality: Scaling Laws and Optimal Scheduling at the Edge-of-Chaos](dropout_universality_scaling_laws_and_optimal_scheduling_at_the_edge-of-chaos.md)
- [\[ICML 2026\] On Training Large Language Models for Long-Horizon Tasks: An Empirical Study of Horizon Length](on_training_large_language_models_for_long-horizon_tasks_an_empirical_study_of_h.md)
- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](../../ACL2025/llm_pretraining/davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](../../ACL2025/llm_pretraining/large_vocabulary_size_improves_large_language_models.md)
- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)

</div>

<!-- RELATED:END -->
