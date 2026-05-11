---
title: >-
  [Paper Note] Adaptive Prediction-Powered AutoEval with Reliability and Efficiency Guarantees
description: >-
  [NeurIPS 2025][Model Compression][LLM evaluation] This paper proposes the R-AutoEval+ framework, which introduces an adaptive weighting mechanism within the testing-by-betting framework to dynamically regulate reliance o…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "LLM evaluation"
  - "prediction-powered inference"
  - "autoevaluator"
  - "e-value"
  - "testing-by-betting"
date: 2026-05-08
content_hash: b8af98ab592ee106
---

# Adaptive Prediction-Powered AutoEval with Reliability and Efficiency Guarantees

**Conference**: NeurIPS 2025
**arXiv**: [2505.18659](https://arxiv.org/abs/2505.18659)
**Code**: [https://github.com/kclip/R_AutoEval_plus](https://github.com/kclip/R_AutoEval_plus)
**Area**: Model Compression
**Keywords**: LLM evaluation, prediction-powered inference, autoevaluator, e-value, testing-by-betting

## TL;DR
This paper proposes the R-AutoEval+ framework, which introduces an adaptive weighting mechanism within the testing-by-betting framework to dynamically regulate reliance on LLM-judge-generated synthetic data. It is the first method to simultaneously guarantee evaluation reliability and sampling efficiency no worse than approaches using only real data under finite samples, validated across three scenarios: LLM quantization, prompt selection, and inference budget allocation.

## Background & Motivation

**Background**: Selecting AI models (e.g., LLMs) requires accurate performance estimation. Traditional evaluation (Eval) relies on costly human-annotated real data and is unbiased; AutoEval uses LLM-as-judge to generate large amounts of synthetic data at low cost but may introduce systematic bias. **Limitations of Prior Work**: Recent prediction-powered inference (PPI) methods (R-AutoEval) correct synthetic data bias using a small amount of real data to obtain reliability guarantees, but a critical issue remains — when judge quality is poor, R-AutoEval's sampling efficiency is lower than the purely real-data method R-Eval, meaning synthetic data becomes counterproductive. **Key Challenge**: Judge quality is unknown a priori, necessitating a method that fully exploits synthetic data when the judge is reliable and automatically degrades to the real-data-only method when it is not. **Key Insight**: The paper combines the multi-strategy betting mechanism of e-values with the regularization coefficient $\rho$ from PPI++, using online learning to automatically discover the optimal synthetic data weight.

## Method

### Overall Architecture
R-AutoEval+ conducts model evaluation within the testing-by-betting framework. Given a target risk level $\alpha$, evaluation is formalized as a hypothesis testing problem: null hypothesis $H_0: R > \alpha$ (model risk exceeds threshold) vs. alternative $H_1: R \leq \alpha$ (model meets requirement). By sequentially processing $n$ real data points and constructing an e-value statistic $E_n$, the null hypothesis is rejected when $E_n \geq 1/\delta$, guaranteeing a false rejection probability of $\leq \delta$. The key innovation of R-AutoEval+ is maintaining $S$ candidate dependency factors $\rho_s \in \{0, \ldots, 1\}$, where $\rho = 0$ corresponds to pure R-Eval and $\rho = 1$ to R-AutoEval, with adaptive weights used to online-select the optimal $\rho$.

### Key Designs

1. **PPI++ Effective Observation Construction**:

   - Function: Fuses real and synthetic data into a single unbiased risk estimate.
   - Mechanism: For each candidate factor $\rho_s$, the effective observation is constructed as $\ell_{s,i}^f = (\rho_s / r) \cdot \sum \ell(\tilde{X}, f(\tilde{X})) + \ell(X_i, Y_i) - \rho_s \cdot \ell(X_i, f(X_i))$, where the first term is the weighted contribution of synthetic data and the latter two terms are bias correction terms. The estimator is unbiased and bounded.
   - Design Motivation: $\rho_s$ controls the degree of reliance on synthetic data — larger values increase dependence, potentially reducing variance (with a good judge) or increasing it (with a poor judge); $\rho_s = 0$ discards synthetic data entirely, reducing to standard Eval.

2. **Exponential-Weight Adaptive Update Mechanism**:

   - Function: Online learning of optimal allocation weights for each candidate factor $\rho_s$.
   - Mechanism: $S$ independent e-value sets $\{E_{s,i}\}$ are maintained, and weights are updated as $w_{s,i} = w_{s,0} \cdot E_{s,i-1} / \sum w_{s',0} \cdot E_{s',i-1}$ — $\rho_s$ values with greater accumulated evidence (i.e., higher e-values) receive larger weights.
   - Design Motivation: This is equivalent to the exponential-weight forecasting algorithm, which has a sublinear regret guarantee (Lemma 1) and can identify the optimal dependency level after processing $O(\log S)$ samples.

3. **Multi-Strategy E-Value Fusion**:

   - Function: Fuses observations from $S$ candidate strategies into a single valid e-value statistic.
   - Mechanism: The global e-value is defined as $E_n = \prod_{i=1}^n \sum_{s=1}^S w_{s,i} \cdot (1 - \lambda_{s,i} \cdot (\ell_{s,i}^f - \alpha))$; the convex combination form guarantees $\mathbb{E}[E_n \mid R > \alpha] \leq 1$.
   - Design Motivation: Betting variables $\lambda_{s,i}$ are set adaptively via Universal Portfolio (UP) or WSR strategies; UP satisfies sublinear regret while WSR is computationally more efficient.

### Loss & Training
The method requires no training. Betting variables are set adaptively via the UP strategy (discretizing the continuous domain of $\lambda$ into 10,000 grid points) or the WSR strategy (based on online variance estimation). Initial weights are set to uniform distribution $w_{s,0} = 1/S$.

## Key Experimental Results

### Main Results

| Task / Judge | Metric | R-AutoEval+ | R-Eval | R-AutoEval | Gain |
|---|---|---|---|---|---|
| GSM8K / GPT-4.1 (93%) | Avg. tokens | **856.13** | 983.34 | 883.99 | −127 vs. R-Eval |
| GSM8K / Llama-3.3-70B (89%) | Avg. tokens | **847.05** | 983.34 | 854.42 | −136 vs. R-Eval |
| GSM8K / BitNet (35%) | Avg. tokens | **942.47** | 983.34 | 950.27 | −41 vs. R-Eval |
| TriviaQA Quantization / Llama-3.3-BF16 | Model size | **Smallest** | Medium | Potentially larger | Selects smaller model |
| Instruct-Induction | Prompt length | **Shortest** | Baseline | Judge-dependent | Consistently optimal |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| $S=2$ vs. $S=5$ vs. $S=10$ vs. $S=20$ | 871.67→855→856→856 tokens | Performance saturates at $S \geq 5$ |
| UP vs. WSR betting strategy | Comparable performance | WSR is faster to compute |
| Data ordering sensitivity | R-AutoEval+: 6.88% vs. R-Eval: 8.62% | Normalized deviation; R-AutoEval+ is more robust |
| Same-family judge Qwen3-32B | R-AutoEval 24 tokens worse than R-Eval | Preference leakage makes bias correction harder |

### Key Findings
- R-AutoEval+ maintains reliability (risk $\leq \alpha = 0.1$) across all scenarios while selecting more efficient models.
- Weight evolution heatmaps visually demonstrate adaptivity: with a high-quality judge ($\gamma = 0.99$), weights concentrate at $\rho \approx 0.9$; with a low-quality judge ($\gamma = 0.7$), they concentrate at $\rho \approx 0$.
- Same-family LLM judge–evaluatee combinations reduce AutoEval effectiveness due to preference leakage.

## Highlights & Insights
- The first LLM automatic evaluation method to simultaneously provide finite-sample reliability and sampling efficiency guarantees.
- Theorem 3 rigorously proves that sample complexity $\leq \min\{\text{R-Eval}, \text{R-AutoEval}\}$, ensuring no worst-case degradation.
- The discovery of preference leakage is noteworthy: same-family judges, despite higher nominal accuracy, are harder to correct for bias and prove less effective than cross-family judges.
- The optional stopping property of e-values makes the framework naturally suited for online/streaming evaluation scenarios.

## Limitations & Future Work
- Unlabeled real data is required to generate synthetic evaluation results.
- The candidate factor set $\{\rho_s\}$ is fixed and discretized in advance, precluding continuous optimization.
- The efficiency guarantee holds only when $\delta$ is sufficiently small (i.e., under high reliability requirements).
- Computational complexity is $S$ times that of R-AutoEval ($O(SnG)$), though this is negligible relative to LLM inference costs.

## Related Work & Insights
- **PPI/PPI++**: Foundational semi-supervised inference framework; R-AutoEval+ adds adaptive $\rho$ selection.
- **Testing-by-betting**: E-values support optional stopping and optional continuation, offering greater flexibility than p-values.
- **Active evaluation**: An orthogonal direction — adaptively selecting real data vs. adaptively weighting synthetic data — the two approaches are composable.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining the e-value betting framework with PPI++ represents an innovative cross-disciplinary contribution; the adaptive weight update elegantly resolves the core tension.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three LLM evaluation scenarios plus rich ablations (S, strategy, judge quality, ordering sensitivity, confidence intervals).
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations; visualizations in Examples 1/2 effectively aid understanding.
- Value: ⭐⭐⭐⭐ Establishing statistical guarantees for LLM automatic evaluation carries significant practical importance; code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAT: Bridging RNN Efficiency and Attention Accuracy via Chunk-based Sequence Modeling](rat_bridging_rnn_efficiency_and_attention_accuracy_via_chunk-based_sequence_mode.md)
- [\[NeurIPS 2025\] Adaptive Stochastic Coefficients for Accelerating Diffusion Sampling](adaptive_stochastic_coefficients_for_accelerating_diffusion_sampling.md)
- [\[NeurIPS 2025\] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning](twilight_adaptive_attention_sparsity_with_hierarchical_top-p_pruning.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[NeurIPS 2025\] AdmTree: Compressing Lengthy Context with Adaptive Semantic Trees](admtree_compressing_lengthy_context_with_adaptive_semantic_trees.md)

</div>

<!-- RELATED:END -->
