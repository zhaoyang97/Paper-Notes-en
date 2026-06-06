---
title: >-
  [Paper Note] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Multiple sampling] The authors formalize LLM benchmarking as a **hierarchical Bayesian estimation problem**—where prompt difficulty $p_i \sim \mathbb{P}(\mu,\sigma)$…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Multiple sampling"
  - "Hierarchical model"
  - "prompt difficulty"
  - "Mislabel detection"
  - "data map"
date: 2026-05-08
content_hash: f60060a7416e4731
---

# Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation

**Conference**: ACL 2026  
**arXiv**: [2502.08943](https://arxiv.org/abs/2502.08943)  
**Code**: To be confirmed  
**Area**: LLM Evaluation / Statistical Inference / Benchmark Methodology  
**Keywords**: Multiple sampling, Hierarchical model, prompt difficulty, Mislabel detection, data map

## TL;DR
The authors formalize LLM benchmarking as a **hierarchical Bayesian estimation problem**—where prompt difficulty $p_i \sim \mathbb{P}(\mu,\sigma)$, and the $k$-generation accuracy for each prompt follows $\text{Bernoulli}(p_i)$. They theoretically prove that $k>1$ sampling can compress within-prompt variance to $\frac{1}{nk}$ and derive prompt-level difficulty scores $\mathbb{P}(\text{correct})$ and a data map capable of detecting annotation errors (44.4% hit rate on GSM8K).

## Background & Motivation

**Background**: Current LLM benchmark evaluations (greedy used by LiveBench, WildBench, OpenLLM Leaderboard; random sampling used by TrustLLM, MT-Bench, AlpacaEval) are almost entirely based on **one generation per prompt** to calculate scores.

**Limitations of Prior Work**: (1) Greedy decoding is deterministic and inconsistent with real-world scenarios involving temperature-based sampling, causing benchmark scores to be over- or underestimated. (2) Even with random sampling, single-generation variance is extreme—empirical evidence shows that for Llama-3.1 8B on GSM8K, the gap between best/worst runs can reach **18.6pp**! (3) Single generation only yields a binary 0/1, making it impossible to define prompt-level difficulty or answer "which questions are harder."

**Key Challenge**: LLM generation is inherently stochastic, yet current evaluation paradigms treat it as **deterministic output**. This "discarding of stochasticity" makes leaderboard rankings untrustworthy for small benchmarks (e.g., IFEval with only 541 prompts), where the real performance gap between models might be smaller than the sampling noise.

**Goal**: (a) Provide a statistically sound estimator for benchmark evaluation; (b) Derive useful prompt-level signals (difficulty, annotation quality) from the by-products of multiple generations.

**Key Insight**: Treat benchmarking as an **estimation problem**—the objective is to estimate $\mu = \mathbb{E}[p_i]$, so $k>1$ sampling becomes a simple variance reduction task. This perspective is seemingly obvious but has not been formally investigated by the community.

**Core Idea**: **Formalize benchmarking as a hierarchical model $p_i \sim \mathbb{P}(\mu,\sigma), \, y_{i,j} \sim \text{Bernoulli}(p_i)$. Multiple sampling directly reduces variance and naturally provides prompt difficulty scores.**

## Method

### Overall Architecture
This is essentially a statistical modeling plus application suite:

1. Express LLM benchmarking as a two-level hierarchical model;
2. Derive the variance decomposition for the moment estimator $\hat\mu$;
3. Use $\hat p_i = \frac{1}{k}\sum_j y_{i,j}$ as the prompt-level difficulty score $\mathbb{P}(\text{correct})$;
4. Plot $\mathbb{P}(\text{correct})$ and semantic consistency $\mathbb{S}(\text{consistency})$ on a 2D plane to create a data map, using the bottom-left and bottom-right regions to detect mislabels.

### Key Designs

1. **Hierarchical Model + Variance Decomposition**:

    - **Function**: Provide a rigorous theoretical basis for the necessity of multiple sampling.
    - **Mechanism**: Assume $p_i \sim \mathbb{P}(\mu,\sigma;\theta)$ for $i=1,\dots,n$, and $y_{i,j} \sim \text{Bernoulli}(p_i)$ for $j=1,\dots,k$. The moment estimator $\hat\mu = \frac{1}{nk}\sum_{i,j}y_{i,j}$ is an unbiased estimator of $\mu$. The variance can be decomposed as $\text{Var}(\hat\mu) = \underbrace{\frac{1}{nk}(\mu-\mu^2-\sigma^2)}_{\text{within-prompt}} + \underbrace{\frac{1}{n}\sigma^2}_{\text{between-prompt}}$. The first term vanishes as $k$ increases, while the second is intrinsic noise determined by $n$. Based on CLT, a 95% CI is provided as $\hat\mu \pm 1.96\sqrt{\widehat{\text{Var}(\hat\mu)}}$.
    - **Design Motivation**: Current benchmarks report scores as point estimates without error bars. This decomposition explicitly tells users the stability of a reported score given $k$. It also reveals that IRT (1PL model) is a parametric special case of $\mathbb{P}(\mu,\sigma)$ within this framework.

2. **Prompt-level Difficulty Score $\mathbb{P}(\text{correct})$**:

    - **Function**: Provide a continuous [0,1] difficulty score for each prompt to allow for cross-prompt comparison.
    - **Mechanism**: Use $\hat p_i = \frac{1}{k}\sum_{j=1}^{k} y_{i,j}$ as the estimated probability of correctness for the LLM on the $i$-th prompt; $\hat p_i \to p_i$ as $k \to \infty$. Visualizing distributions (Fig 1) immediately reveals benchmark properties: reasoning-heavy benchmarks like MMLU-Pro, IFEval, and MuSR show **diffuse density** on $[0,1]$ (indicating random-like sampling), while simple benchmarks like GSM8K show sharp peaks near 0 and 1 (stable behavior).
    - **Design Motivation**: Previous methods assigned difficulty either manually (e.g., MATH categories) or via multi-LLM IRT fitting (Polo et al.). Ours uses single target LLM multiple sampling to obtain subjective difficulty specifically for that LLM, making it better for diagnosing specific model weaknesses than "cross-model objective difficulty."

3. **Data Map for Mislabel Detection ($\mathbb{P}(\text{correct})$ × $\mathbb{S}(\text{consistency})$)**:

    - **Function**: Use by-products to detect annotation errors or semantic ambiguities within the benchmark.
    - **Mechanism**: In addition to correctness, calculate semantic consistency entropy $\mathbb{S}(\text{consistency})=\sum_{c=1}^{C}\text{Prop}_c \log \text{Prop}_c$ (negative entropy; higher means more consistent) by clustering $k$ generations into $C$ semantic clusters. Hypothesis: Prompts with **low $\mathbb{P}(\text{correct})$ + high $\mathbb{S}(\text{consistency})$** are likely mislabeled or ambiguous—the LLM consistently and confidently provides an answer that disagrees with the ground truth.
    - **Design Motivation**: Inverse the intuition of self-consistency (Wang et al. 2022) which suggests that "true difficult problems have multiple reasoning paths"—stable but "wrong" answers are likely mislabeled. Empirically, using $\hat p_i \le 0.1$ and $\mathbb{S} \ge -0.8$ identified 18 prompts on GSM8K, 44.4% of which were confirmed as mislabeled/ambiguous upon human review.

### Loss & Training
No models are trained. All experiments use $k=50$ sampling during inference (temperature=0.7, top-p=1.0, 0-shot CoT). The moment estimator is closed-form and requires no iteration.

## Key Experimental Results

### Main Results: Variance Comparison for 4 LLMs Across 4 Benchmarks (k=1 vs k=50, SE in %)

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

→ Δ(k=1) represents the gap between the best and worst runs when $k=1$, reaching up to **18.6pp**! This shows that single-sample runs cannot reliably distinguish models. Greedy and sample (k=50) can also differ by 3-4pp (e.g., Llama 8B on GSM8K).

### Ablation Study: Impact of k on 95% Confidence Interval Width (IFEval, Synthetic)

| k | 95% CI Width (IFEval, relative to k=50 oracle) | Description |
|---|---|---|
| k=1 | ~3.6 pp | Single sample; CI is very wide and often fails to cover reality |
| k=5 | ~1.6 pp | CI narrows significantly |
| k=10 | ~1.1 pp | Approaches the stability of $k=20$ |
| k=20 | ~0.8 pp | Diminishing marginal returns |
| k=50 | 0 (oracle) | Reference point using full sampling |
| Greedy | Persistent ~2-3 pp gap | Constant deviation from reality |

### Key Findings
- **LLMs perform like random samplers on reasoning tasks**: The $\mathbb{P}(\text{correct})$ distributions for MMLU-Pro / IFEval / MuSR are diffuse on $[0,1]$, suggesting models are often "guessing" rather than "knowing." Simple tasks like GSM8K show peaks near 0/1.
- **Large models are more stable**: Llama 70B shows smaller Δ(k=1) than 8B across all benchmarks (4.8 vs 18.6 on GSM8K), indicating that scale $\uparrow$ improves accuracy $\uparrow$ and reduces variance $\downarrow$.
- **Temperature affects small models more**: For 8B models, increasing $T=0.4 \to 1.0$ makes the $\mathbb{P}(\text{correct})$ distribution more diffuse, while 70B remains almost unchanged.
- **$k=10$ is sufficient**: Synthetic experiments (Fig 2) show sharp CI narrowing from $k=1 \to 10$, with diminishing returns after. $k \approx 10$ is recommended for engineering practice.
- **44.4% of suspicious prompts are mislabeled**: Even high-quality benchmarks like GSM8K contain ~5% error rates; this method identifies suspect samples at minimal cost.

## Highlights & Insights
- **The perspective of benchmarking as an estimation problem is valuable**: A simple $\hat\mu \pm 1.96 \sqrt{\text{Var}}$ decomposition transforms the trustworthiness of leaderboard rankings from intuition into a calculable metric. All leaderboard maintainers should include error bars.
- **Variance decomposition dictates sampling strategy**: within-prompt variance $\propto \frac{1}{nk}$ and between-prompt $\propto \frac{1}{n}$. If the benchmark $n$ is already large, multiple sampling has diminishing returns; if $n$ is small (e.g., IFEval), multiple sampling is highly beneficial.
- **IRT as a special case**: The authors embed the popular Item Response Theory into their framework, unifying multiple definitions of "prompt difficulty" with theoretical elegance.
- **"Low accuracy + high consistency = mislabel" trick**: This inverse application of self-consistency detects dataset quality issues at zero additional cost. The data map concept can be generalized to any generative benchmark.
- **Subjective vs. objective difficulty**: The authors distinguish between "hard for Model A" and "universally hard." Focus on the former is more useful for model-specific diagnostics.

## Limitations & Future Work
- Authors acknowledge: (1) $k$-fold sampling increases inference cost $k$ times; (2) the assumption of independent sampling across prompts doesn't account for prompts from similar subjects; (3) the True Positive Rate for mislabel detection is ~50%, requiring better semantic metrics.
- Our observations: (a) Methods assume a categorical ground truth logic, not directly applicable to open-ended generation (like MT-Bench) involving LLM-as-a-judge (introducing a new variance source); (b) $\mathbb{S}(\text{consistency})$ is easy to cluster in GSM8K but requires embedding/NLI for open QA; (c) scale coverage is limited to 8-70B models.
- Improvement ideas: (a) **adaptive sampling**—allocate more samples to high-variance prompts; (b) incorporate prompt covariance structures; (c) expand mislabel detection to an ensemble of multiple models and semantic metrics to improve TPR beyond 80%.

## Related Work & Insights
- **vs Miller 2024 (Adding Error Bars to Evals)**: Concurrent work also mentions multiple sampling to reduce variance but remains conceptual; Ours provides rigorous theory, empirical validation, and applications.
- **vs Song et al. 2024 (Good/Bad/Greedy)**: They highlight the greedy vs sampling gap; Ours provides a statistical model explaining why the gap exists.
- **vs Polo et al. 2024 (TinyBenchmarks)**: They use IRT across multiple LLMs to estimate difficulty; Ours proves IRT is a special 1PL case and offers a lightweight single-LLM version.
- **vs Swayamdipta et al. 2020 (Dataset Cartography)**: That work uses training dynamics for classification data maps; Ours uses inference $\mathbb{P}(\text{correct}) \times \mathbb{S}$ for generative data maps.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet profound—applies statistical variance decomposition to LLM evaluation with seamless theoretical and practical integration.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 LLMs × 4 benchmarks × k=50 + temperature ablation + GSM8K case study; thorough but limited to closed-form correctness.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear math, intuitive charts, rigorous lemmas, and natural applications.
- Value: ⭐⭐⭐⭐⭐ Directly promotes a paradigm shift in LLM benchmarking; error bars should become a standard for leaderboards.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](beyond_static_benchmarks_synthesizing_harmful_content_via_persona-based_simulati.md)
- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks](benchmarker_an_education-inspired_toolkit_for_highlighting_flaws_in_multiple-cho.md)
- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)

</div>

<!-- RELATED:END -->
