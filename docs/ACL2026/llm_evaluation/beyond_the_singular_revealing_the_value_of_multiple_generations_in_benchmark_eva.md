---
title: >-
  [Paper Note] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation
description: >-
  [ACL 2026][LLM Evaluation][prompt difficulty] The authors formalize LLM benchmarking as a **hierarchical Bayesian estimation problem**—prompt difficulty $p_i \sim \mathbb{P}(\mu,\sigma)$, and the accuracy of $k$ generations per prompt follows Bernoulli$(p_i)$. It is theoretically proven that using $k>1$ samples reduces within-prompt variance to $\frac{1}{nk}$, and
tags:
  - ACL 2026
  - LLM Evaluation
  - prompt difficulty
  - data map
date: 2026-05-08
content_hash: f447da555fbd925c
---
# Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation

**Conference**: ACL 2026 Findings  
**arXiv**: [2502.08943](https://arxiv.org/abs/2502.08943)  
**Code**: To be confirmed  
**Area**: LLM Evaluation / Statistical Inference / Benchmark Methodology  
**Keywords**: Multi-sampling, Hierarchical models, prompt difficulty, Mislabel detection, data map

## TL;DR
The authors formalize LLM benchmarking as a **hierarchical Bayesian estimation problem**—prompt difficulty $p_i \sim \mathbb{P}(\mu,\sigma)$, and the accuracy of $k$ generations per prompt follows Bernoulli$(p_i)$. It is theoretically proven that using $k>1$ samples reduces within-prompt variance to $\frac{1}{nk}$, and this leads to the derivation of prompt-level difficulty scores $\mathbb{P}(\text{correct})$ and a "data map" capable of detecting mislabeled instances (with a 44.4% hit rate on GSM8K).

## Background & Motivation

**Background**: Current LLM benchmark evaluations (LiveBench, WildBench, OpenLLM Leaderboard use greedy; TrustLLM, MT-Bench, AlpacaEval use random sampling) are almost entirely based on **one generation per prompt** to calculate benchmark scores.

**Limitations of Prior Work**: (1) Greedy decoding is deterministic, which is inconsistent with real-world scenarios using temperature sampling, leading to overestimated or underestimated benchmark scores. (2) Even with random sampling, single-generation variance is extreme—the authors empirically found a gap of **18.6pp** between the best and worst runs for Llama-3.1 8B on GSM8K. (3) Single generations provide only a binary 0/1 result, failing to define prompt-level difficulty or identify "which questions are harder."

**Key Challenge**: LLM generation is inherently stochastic, yet current evaluation paradigms treat it as **deterministic output**. This disregard for stochasticity makes leaderboard rankings entirely unreliable for small benchmarks (e.g., IFEval with only 541 prompts)—the true performance gap between two models might be smaller than the sampling noise.

**Goal**: (a) Provide a statistically sound estimator for benchmark evaluation; (b) Derive useful prompt-level signals (difficulty, annotation quality) as by-products of multiple generations.

**Key Insight**: Treat benchmarking as an **estimation problem**—the goal is to estimate $\mu = \mathbb{E}[p_i]$, meaning $k>1$ sampling is a simple variance reduction technique. This perspective is seemingly obvious but has not been formally adopted by the community.

**Core Idea**: **Formalize benchmarking as a hierarchical model $p_i \sim \mathbb{P}(\mu,\sigma), \, y_{i,j} \sim \text{Bernoulli}(p_i)$, where multi-sampling directly reduces variance and naturally yields prompt difficulty scores.**

## Method

### Overall Architecture

This paper refines LLM benchmarking as a statistical estimation problem: the input consists of $n$ prompts with $k$ binary (0/1) correctness records per prompt, and the goal is to accurately estimate the true benchmark score $\mu=\mathbb{E}[p_i]$. The method begins by describing evaluation as a two-layer hierarchical model (prompt difficulty $p_i$ follows a population distribution, and each generation follows Bernoulli$(p_i)$), deriving the variance decomposition of the moment estimator $\hat\mu$ to explain "why more sampling reduces variance." Subsequently, the by-products of $k$ generations—empirical accuracy $\hat p_i$ and semantic consistency $\mathbb{S}$—are plotted on a 2D plane to output prompt-level difficulty scores and a "data map" for detecting mislabels. The entire method requires no model training, only multiple samplings during inference.

### Key Designs

**1. Hierarchical Model + Variance Decomposition: Providing a rigorous basis for "why more sampling is needed"**

Current benchmarks report scores as point estimates without error bars, making it impossible to determine if a ranking is reliable. This paper sets $p_i \sim \mathbb{P}(\mu,\sigma;\theta)$ ($i=1,\dots,n$) and $y_{i,j} \sim \text{Bernoulli}(p_i)$ ($j=1,\dots,k$). The moment estimator $\hat\mu = \frac{1}{nk}\sum_{i,j}y_{i,j}$ is an unbiased estimator of $\mu$, with its variance decomposable as:

$$\text{Var}(\hat\mu) = \underbrace{\tfrac{1}{nk}(\mu-\mu^2-\sigma^2)}_{\text{within-prompt}} + \underbrace{\tfrac{1}{n}\sigma^2}_{\text{between-prompt}}.$$

The former term converges to zero as $k$ increases, while the latter is the intrinsic noise determined by $n$. The Central Limit Theorem (CLT) provides a 95% CI: $\hat\mu \pm 1.96\sqrt{\widehat{\text{Var}(\hat\mu)}}$. This decomposition explicitly quantifies the reporting stability and reveals that Item Response Theory (specifically the 1PL model) is a parametric special case of $\mathbb{P}(\mu,\sigma)$ within this framework.

**2. Prompt-level Difficulty Score $\mathbb{P}(\text{correct})$: Assigning comparable difficulty to each prompt**

Previously, labeling prompt difficulty required manual effort (e.g., MATH's 5 levels) or multi-LLM IRT fitting (e.g., Polo et al.). This paper directly uses $\hat p_i = \frac{1}{k}\sum_{j=1}^{k} y_{i,j}$ as the estimate of the target LLM's success probability on the $i$-th prompt, where $\hat p_i \to p_i$ as $k \to \infty$. Visualizing the distribution of this [0,1] continuous difficulty score (Fig 1) immediately reveals benchmark properties: reasoning-heavy benchmarks like MMLU-Pro, IFEval, and MuSR show a diffuse density over $[0,1]$ (meaning the LLM behaves like random sampling on many questions), while simple tasks like GSM8K show sharp peaks near 0 and 1 (stable behavior). This subjective difficulty, derived from a single target LLM, is better suited for diagnosing specific model weaknesses than "cross-model objective difficulty."

**3. Data map for mislabel detection ($\mathbb{P}(\text{correct}) \times \mathbb{S}(\text{consistency})$): Identifying benchmark labeling errors**

In addition to correctness, the paper calculates a semantic consistency negative entropy $\mathbb{S}(\text{consistency})=\sum_{c=1}^{C}\text{Prop}_c \log \text{Prop}_c$ by clustering $k$ generations into $C$ semantic clusters. High proportion values indicate consistent answers. The key hypothesis is that **Low $\mathbb{P}(\text{correct})$ + High $\mathbb{S}(\text{consistency})$** prompts are likely mislabeled or ambiguous—the LLM confidently and consistently provides an answer that contradicts the ground truth. This is an inverse application of the self-consistency (Wang et al. 2022) intuition that "truly hard problems should have multiple reasoning paths"—stable but "wrong" outputs are likely due to incorrect labels. Empirically, using $\hat p_i \le 0.1$ and $\mathbb{S} \ge -0.8$ on GSM8K identified 18 prompts, 44.4% of which were confirmed as mislabeled or ambiguous upon manual review.

### Loss & Training
No training involved. All experiments used $k=50$ samples during inference (temperature=0.7, top-p=1.0, 0-shot CoT). The moment estimator is closed-form and requires no iteration.

## Key Experimental Results

### Main Results: Variance comparison across 4 LLMs and 4 benchmarks (k=1 vs k=50, SE unit %)

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

→ Δ(k=1) shows the gap between the best and worst runs for $k=1$, reaching **up to 18.6pp**! This demonstrates that single sampling cannot reliably distinguish models. Greedy and sample (k=50) results can also differ by 3-4pp (e.g., Llama 8B on GSM8K).

### Ablation Study: Impact of k on Confidence Interval (CI) Width (IFEval, Synthetic)

| k | 95% CI Width (IFEval, relative to k=50 oracle) | Description |
|---|---|---|
| k=1 | ~3.6 pp | Single sample; CI is extremely wide and often fails to cover the true value |
| k=5 | ~1.6 pp | CI narrows significantly |
| k=10 | ~1.1 pp | Near-equivalence to k=20 |
| k=20 | ~0.8 pp | Diminishing marginal returns |
| k=50 | 0 (oracle) | Full sampling used as reference |
| Greedy | ~2-3 pp gap | Consistently deviates from the true value |

### Key Findings
- **LLMs perform like random sampling on reasoning tasks**: The $\mathbb{P}(\text{correct})$ distributions for MMLU-Pro, IFEval, and MuSR are diffuse over $[0,1]$, indicating models are "guessing" on many problems. Only simple tasks like GSM8K show peaks near 0 and 1.
- **Larger models are more stable**: Llama 70B has smaller Δ(k=1) than 8B across all benchmarks (e.g., 4.8 vs 18.6 on GSM8K), indicating that scaling increases accuracy and decreases variance.
- **Temperature impacts small models more significantly**: For 8B models, $\mathbb{P}(\text{correct})$ becomes more diffuse as temperature $T$ increases from 0.4 to 1.0; 70B remains almost unchanged.
- **k=10 is sufficient**: Synthetic experiments (Fig 2) show the CI narrows sharply from $k=1 \to 10$, with diminishing returns beyond that. Defaulting to $k \approx 10$ is recommended for practice.
- **44.4% of 18 suspect prompts were mislabeled**: Even widely used, high-quality benchmarks like GSM8K have ~5% error rates; this method identifies suspect samples at minimal cost.

## Highlights & Insights
- **Valuable perspective on benchmarking as an estimation problem**: A simple $\hat\mu \pm 1.96 \sqrt{\text{Var}}$ decomposition turns reliability from intuition into a computable metric. Leaderboard maintainers should include these error bars.
- **Variance decomposition explicitly dictates sampling count**: Within-prompt variance $\propto \frac{1}{nk}$, and between-prompt $\propto \frac{1}{n}$. For large $n$, multi-sampling returns diminish; for small $n$ (like IFEval), returns are massive.
- **IRT as a special parameterization**: The integration of item response theory into this framework unifies various definitions of "prompt difficulty" with theoretical elegance.
- **"Low accuracy + High consistency = mislabel" trick**: An elegant inverse application of self-consistency for zero-cost dataset auditing.
- **Subjective vs objective difficulty**: The authors clarify that "hard for Model A" and "generally hard" are distinct; focusing on the former is more useful for model-specific diagnostics.

## Limitations & Future Work
- The authors acknowledge: (1) Inference cost increases $k$-fold; (2) Assumes prompt independence, ignoring subject correlation; (3) Mislabel detection hit rate is only ~50%, requiring better semantic metrics.
- Own observations: (a) Assumes ground truth correctness, which does not directly apply to open-ended generation (e.g., MT-Bench) without a judge; (b) Consistency clustering is simple for GSM8K but requires NLI or embeddings for open QA, whose accuracy is unverified; (c) Experiments only involve 4 LLMs (8-70B).
- Improvement ideas: (a) **adaptive sampling**—allocate more samples to high-variance prompts; (b) modeling prompt covariance structures; (c) ensembling multi-model and multi-semantic metrics for mislabel detection.

## Related Work & Insights
- **vs Miller 2024 (Adding Error Bars to Evals)**: A concurrent conceptual work; this paper provides more rigorous theory, empirical evidence, and derivative applications (difficulty, data map).
- **vs Song et al. 2024 (Good/Bad/Greedy)**: They noted the greedy vs sampling gap; this paper provides the statistical model explanation.
- **vs Polo et al. 2024 (TinyBenchmarks)**: They used IRT across multiple LLMs for difficulty; this paper proves IRT is a 1PL special case and offers a lightweight single-LLM version.
- **vs Swayamdipta et al. 2020 (Dataset Cartography)**: Adapts classification data maps to the generative domain.
- **Inspiration**: Every stochastic benchmark should report error bars; the data map approach can be extended to locate flawed cases in agent benchmarks (e.g., SWE-bench).

## Rating
- Novelty: ⭐⭐⭐⭐ Simple but precise angle—applying statistical variance decomposition to LLM evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 LLMs × 4 benchmarks × k=50 + temperature ablation + GSM8K case study; thorough but specific (open generation not covered).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematics, intuitive charts, rigorous lemmas.
- Value: ⭐⭐⭐⭐⭐ Directly drives a paradigm shift in LLM benchmarking; error bar reporting should become standard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ACL 2026\] Beyond Marginal Distributions: A Framework to Evaluate the Representativeness of Demographic-Aligned LLMs](beyond_marginal_distributions_a_framework_to_evaluate_the_representativeness_of_.md)
- [\[ACL 2026\] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](k-metbench_a_multi-dimensional_benchmark_for_fine-grained_evaluation_of_expert_r.md)
- [\[ACL 2026\] Beyond Reproduction: A Paired-Task Framework for Assessing LLM Comprehension and Creativity in Literary Translation](beyond_reproduction_a_paired-task_framework_for_assessing_llm_comprehension_and_.md)
- [\[ACL 2026\] Beyond Fixed Psychological Personas: State Beats Trait, but Language Models are State-Blind](beyond_fixed_psychological_personas_state_beats_trait_but_language_models_are_st.md)
- [\[ACL 2026\] Teaching Language Models to Forecast Research Success Through Comparative Idea Evaluation](teaching_language_models_to_forecast_research_success_through_comparative_idea_e.md)

</div>

<!-- RELATED:END -->
