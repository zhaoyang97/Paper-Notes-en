---
title: >-
  [Paper Note] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation
description: >-
  [ACL 2026][LLM Evaluation][prompt difficulty] The authors formalize LLM benchmarking as a **hierarchical Bayesian estimation problem**—specifically, prompt difficulty $p_i \sim \mathbb{P}(\mu,\sigma)$, where the accuracy of $k$ generations for each prompt follows Bernoulli$(p_i)$. It is theoretically proven that $k>1$ samples can reduce within-prompt variance to $
tags:
  - ACL 2026
  - LLM Evaluation
  - prompt difficulty
  - data map
date: 2026-05-08
content_hash: 7bfe16b2eb7eab6f
---
# Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation

**Conference**: ACL 2026 Findings  
**arXiv**: [2502.08943](https://arxiv.org/abs/2502.08943)  
**Code**: To be confirmed  
**Area**: LLM Evaluation / Statistical Inference / Benchmark Methodology  
**Keywords**: Multiple Sampling, Hierarchical Models, Prompt Difficulty, Mislabel Detection, Data Map

## TL;DR
The authors formalize LLM benchmarking as a **hierarchical Bayesian estimation problem**—specifically, prompt difficulty $p_i \sim \mathbb{P}(\mu,\sigma)$, where the accuracy of $k$ generations for each prompt follows Bernoulli$(p_i)$. It is theoretically proven that $k>1$ samples can reduce within-prompt variance to $\frac{1}{nk}$, leading to derived prompt-level difficulty scores $\mathbb{P}(\text{correct})$ and a data map capable of detecting mislabels (achieving a 44.4% hit rate on GSM8K).

## Background & Motivation

**Background**: Current LLM benchmark evaluations (LiveBench, WildBench, and OpenLLM Leaderboard use greedy; TrustLLM, MT-Bench, and AlpacaEval use random sampling) are almost entirely based on **one generation per prompt** to calculate benchmark scores.

**Limitations of Prior Work**: (1) Greedy decoding is deterministic, which is inconsistent with real-world deployment scenarios involving temperature sampling, leading to overestimated or underestimated benchmark scores. (2) Even with random sampling, the variance of a single generation is extreme—empirical findings show that on GSM8K, the difference between the best and worst runs for Llama-3.1 8B can be as high as **18.6pp**! (3) Single generations only provide a binary 0/1, making it impossible to define prompt-level difficulty or answer "which questions are harder."

**Key Challenge**: LLM generation is inherently stochastic, yet current evaluation paradigms treat it as a **deterministic output**. This practice of "discarding randomness" makes leaderboard rankings completely unreliable on small benchmarks (e.g., IFEval with only 541 prompts), where the true gap between two models might be smaller than the sampling noise.

**Goal**: (a) Provide a statistically sound estimator for benchmark evaluation; (b) Derive useful prompt-level signals (difficulty, annotation quality) using the by-products of multiple generations.

**Key Insight**: Treat benchmarking as an **estimation problem**—the goal is to estimate $\mu = \mathbb{E}[p_i]$, meaning $k>1$ sampling is a simple form of variance reduction. While seemingly obvious, the community has not formally adopted this perspective.

**Core Idea**: **Formalize benchmarking as a hierarchical model $p_i \sim \mathbb{P}(\mu,\sigma), \, y_{i,j} \sim \text{Bernoulli}(p_i)$, where multiple sampling directly reduces variance and naturally provides prompt difficulty scores.**

## Method

### Overall Architecture

This paper reinterprets LLM benchmarking as a statistical estimation problem: the input consists of 0/1 correctness records for $n$ prompts, each sampled $k$ times, and the goal is to accurately estimate the true benchmark score $\mu=\mathbb{E}[p_i]$. The method first formulates the evaluation as a two-layer hierarchical model (prompt difficulty $p_i$ follows a population distribution, and each generation follows Bernoulli$(p_i)$), deriving the variance decomposition of the moment estimator $\hat\mu$ to explain "why multiple sampling reduces variance." Subsequently, the by-products of $k$ generations—empirical accuracy $\hat p_i$ and semantic consistency $\mathbb{S}$ for each prompt—are plotted on a 2D plane to output prompt-level difficulty scores and a data map that identifies mislabels. The entire method requires no model training, only multiple samplings during the inference stage.

### Key Designs

**1. Hierarchical Model + Variance Decomposition: Providing a Rigorous Basis for Multiple Sampling**

Current benchmarks report scores as point estimates without error bars, failing to answer "whether this ranking is trustworthy." This paper defines $p_i \sim \mathbb{P}(\mu,\sigma;\theta)$ ($i=1,\dots,n$) and $y_{i,j} \sim \text{Bernoulli}(p_i)$ ($j=1,\dots,k$). The moment estimator $\hat\mu = \frac{1}{nk}\sum_{i,j}y_{i,j}$ is an unbiased estimator of $\mu$, and its variance can be decomposed as:

$$\text{Var}(\hat\mu) = \underbrace{\tfrac{1}{nk}(\mu-\mu^2-\sigma^2)}_{\text{within-prompt}} + \underbrace{\tfrac{1}{n}\sigma^2}_{\text{between-prompt}}.$$

The first term approaches zero as $k$ increases, while the second term represents inherent noise determined by $n$. The 95% CI is given by the CLT as $\hat\mu \pm 1.96\sqrt{\widehat{\text{Var}(\hat\mu)}}$. This decomposition informs the user about the stability of the reported score based on $k$, while revealing that IRT (1PL model) is a parameterized special case of $\mathbb{P}(\mu,\sigma)$ within this framework.

**2. Prompt-level Difficulty Score $\mathbb{P}(\text{correct})$: Comparable Difficulty for Each Prompt**

Previously, labeling prompt difficulty relied on manual effort (e.g., MATH's 5 levels) or multi-LLM IRT fitting. This paper directly takes $\hat p_i = \frac{1}{k}\sum_{j=1}^{k} y_{i,j}$ as the estimated probability of the target LLM being correct on the $i$-th prompt. As $k \to \infty$, $\hat p_i \to p_i$. Plotting the distribution of these continuous [0,1] difficulty scores reveals benchmark properties: reasoning-heavy benchmarks like MMLU-Pro, IFEval, and MuSR show a diffuse density over $[0,1]$ (indicating the LLM behaves like random sampling on many items), while simple tasks like GSM8K show distinct spikes near 0 and 1 (stable behavior). This subjective difficulty, derived from a single target LLM, is better suited for diagnosing specific model weaknesses than "cross-model objective difficulty."

**3. Data Map for Mislabel Detection ($\mathbb{P}(\text{correct}) \times \mathbb{S}(\text{consistency})$): Catching Benchmark Errors**

Beyond correctness, the paper calculates a semantic consistency negative entropy $\mathbb{S}(\text{consistency})=\sum_{c=1}^{C}\text{Prop}_c \log \text{Prop}_c$. The $k$ generations are grouped into $C$ semantic clusters, where $\text{Prop}_c$ is the proportion of each cluster; higher values indicate more consistent answers. The core hypothesis is: **Low $\mathbb{P}(\text{correct})$ + High $\mathbb{S}(\text{consistency})$** prompts are likely mislabeled or ambiguous—the LLM confidently and consistently provides an answer that disagrees with the ground truth. This is a reverse application of the self-consistency intuition (Wang et al. 2022) that "true difficult problems should have multiple reasoning paths." Empirically, filtering for $\hat p_i \le 0.1$ and $\mathbb{S} \ge -0.8$ on GSM8K identified 18 prompts, 44.4% of which were confirmed as mislabeled or ambiguous upon manual review.

### Loss & Training
No models are trained. All experiments use $k=50$ samples during inference (temperature=0.7, top-p=1.0, 0-shot CoT). The moment estimator is closed-form and requires no iteration.

## Key Experimental Results

### Main Results: Variance Comparison of 4 LLMs across 4 Benchmarks (k=1 vs k=50, SE in %)

| Benchmark | n | Llama 3.1 8B Greedy | Sample (k=50) | Δ(k=1) | Llama 3.1 70B Greedy | Sample (k=50) | Δ(k=1) |
|---|---|---|---|---|---|---|---|
| MMLU-Pro | 12,187 | 46.2 (0.45) | 46.1 (0.39) | 10.0 | 63.8 (0.44) | 63.4 (0.40) | 3.9 |
| GSM8K | 1,319 | 86.1 (0.95) | 85.6 (0.68) | **18.6** | 95.6 (0.56) | 95.3 (0.45) | 4.8 |
| IFEval | 541 | 74.5 (1.87) | 71.1 (1.51) | 8.3 | 82.6 (1.64) | 80.2 (1.42) | 5.9 |
| MuSR | 756 | 24.8 (1.65) | 29.0 (1.00) | 8.2 | 56.3 (1.80) | 57.9 (1.40) | 5.4 |

| Benchmark | Qwen 2.5 7B Greedy | Sample (k=50) | Δ(k=1) | Ministral 8B Greedy | Sample (k=50) | Δ(k=1) |
|---|---|---|---|---|---|---|
| MMLU-Pro | 53.3 (0.45) | 53.0 (0.36) | 1.3 | 39.7 (0.44) | 36.3 (0.29) | 1.5 |
| GSM8K | 90.2 (0.82) | 90.2 (0.65) | 2.3 | 86.1 (0.95) | 84.9 (0.73) | 3.1 |
| IFEval | 72.6 (1.92) | 71.2 (1.64) | 5.9 | 51.4 (2.15) | 49.8 (1.65) | 5.6 |
| MuSR | 49.2 (1.82) | 50.9 (0.98) | 8.3 | 49.7 (1.82) | 50.8 (0.91) | 8.6 |

→ Δ(k=1) represents the gap between the best and worst runs when k=1, reaching **up to 18.6pp**! This indicates that single sampling is unreliable for distinguishing models. Greedy and Sample (k=50) can also differ by 3-4pp (e.g., Llama 8B on GSM8K: 86.1 vs 85.6).

### Ablation Study: Impact of $k$ on Confidence Interval Width (IFEval, synthetic)

| k | 95% CI Width (IFEval, relative to k=50 oracle) | Description |
|---|---|---|
| k=1 | ~3.6 pp | Single sampling; extremely wide CI often fails to cover the true value. |
| k=5 | ~1.6 pp | CI narrows significantly. |
| k=10 | ~1.1 pp | Approaches the level of k=20. |
| k=20 | ~0.8 pp | Diminishing marginal returns. |
| k=50 | 0 (oracle) | Full sampling used as reference. |
| Greedy | Constant ~2-3 pp gap | consistently deviates from the true value. |

### Key Findings
- **LLMs perform like random samplers on reasoning tasks**: The $\mathbb{P}(\text{correct})$ distribution for MMLU-Pro / IFEval / MuSR is diffuse over $[0,1]$, suggesting models are "guessing" on many questions; simple tasks like GSM8K exhibit spikes near 0 and 1.
- **Larger models are more stable**: Llama 70B shows smaller Δ(k=1) than 8B across all 4 benchmarks (4.8 vs 18.6 on GSM8K), indicating that increasing scale improves both accuracy and variance.
- **Temperature affects smaller models more significantly**: As T increases from 0.4 to 1.0, the 8B model's $\mathbb{P}(\text{correct})$ distribution becomes more diffuse, while the 70B model remains largely unchanged.
- **k=10 is sufficient**: Synthetic experiments show CI narrows sharply from k=1 to 10, with diminishing returns from 10 to 50. A default of k≈10 is recommended for engineering practice.
- **44.4% of 18 flagged prompts were true mislabels**: Even widely used, high-quality benchmarks like GSM8K have a ~5% error rate; this method identifies suspicious samples at minimal cost.

## Highlights & Insights
- **Value of the estimation perspective**: A simple $\hat\mu \pm 1.96 \sqrt{\text{Var}}$ decomposition transforms "leaderboard reliability" from intuition into a computable problem. All leaderboard maintainers should include these error bars.
- **Variance decomposition guides sampling frequency**: Within-prompt variance $\propto \frac{1}{nk}$, between-prompt $\propto \frac{1}{n}$. If $n$ is large, multiple sampling has diminishing returns; if $n$ is small (e.g., IFEval), multiple sampling provides huge benefits.
- **IRT as a specialized parameterization**: By embedding the popular Item Response Theory into this framework, the authors unify multiple definitions of "prompt difficulty" with theoretical elegance.
- **The "Low Accuracy + High Consistency = Mislabel" trick**: This reverse application of self-consistency identifies dataset quality issues nearly for free. This data map concept can be extended to any generative benchmark.
- **Subjective vs. Objective Difficulty**: The authors clarify that "hard for Model A" and "universally hard" are distinct; this work focuses on the former, which is more useful for model-specific diagnostics.

## Limitations & Future Work
- The authors acknowledge: (1) Inference cost increases $k$-fold; "minimum sufficient $k$" needs further study. (2) The model assumes independent sampling between prompts, but prompts often share subjects/sources (unmodeled correlation). (3) The true positive rate for mislabel detection is ~50%, requiring more precise semantic metrics.
- Observations: (a) The method assumes ground truth is available to judge correctness, which is not directly applicable to open-ended generation (e.g., MT-Bench). (b) $\mathbb{S}(\text{consistency})$ is easy to compute on GSM8K via final answer clustering, but open-ended QA requires NLI/embeddings, with unknown accuracy. (c) Experiments were limited to 4 LLMs (8-70B); results for much larger or smaller models are unverified.
- Improvement ideas: (a) **Adaptive sampling**—automatically sample more for high-variance prompts and less for low-variance ones (optimal allocation). (b) Incorporate prompt covariance structures. (c) Expand mislabel detection into an ensemble of multiple models and semantic metrics to push TPR above 80%.

## Related Work & Insights
- **vs. Miller 2024 (Adding Error Bars to Evals)**: A concurrent work also mentions reducing variance through multiple sampling but remains conceptual; this paper provides rigorous theory, empirical evidence, and derivative applications.
- **vs. Song et al. 2024 (Good/Bad/Greedy)**: They noted the gap between greedy and sampling; this paper provides a statistical model explanation for why they differ.
- **vs. Polo et al. 2024 (TinyBenchmarks)**: They used IRT across multiple LLMs to estimate difficulty; this paper proves IRT is a 1PL special case of its hierarchical model and proposes a lighter single-LLM version.
- **vs. Swayamdipta et al. 2020 (Dataset Cartography)**: That work used training dynamics for classification data maps; this work builds generative data maps using inference $\mathbb{P}(\text{correct})$ × $\mathbb{S}$, following a similar philosophy in a different context.
- **Insight**: Any benchmark with stochastic output should report error bars; this paper provides a ready-to-use toolbox. The data map idea can be extended to LLM agent benchmarks (e.g., SWE-bench) to locate problematic cases.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple but insightful—brings statistical variance decomposition into LLM evaluation with a seamless flow from theory to application.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 LLMs × 4 benchmarks × k=50 + temperature ablation + GSM8K mislabel case study; well-validated but focused (open-ended generation not covered).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematics, intuitive charts, rigorous lemmas, and natural extensions to applications.
- Value: ⭐⭐⭐⭐⭐ Directly promotes a paradigm shift in LLM benchmark evaluation; error bar reporting should become a standard for leaderboards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](k-metbench_a_multi-dimensional_benchmark_for_fine-grained_evaluation_of_expert_r.md)
- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)
- [\[ACL 2026\] Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind](beyond_fixed_psychological_personas_state_beats_trait_but_language_models_are_st.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)

</div>

<!-- RELATED:END -->
