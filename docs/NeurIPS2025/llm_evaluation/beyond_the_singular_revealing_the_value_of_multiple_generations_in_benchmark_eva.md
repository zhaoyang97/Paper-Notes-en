---
title: >-
  [Paper Note] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation
description: >-
  [NeurIPS 2025][LLM Evaluation][hierarchical statistical model] This paper formalizes LLM benchmark evaluation as a hierarchical statistical model, theoretically demonstrates that multiple stochastic generations ($k>1$) reduce the variance of benchmark score estimates, and introduces a prompt-level difficulty metric $\mathbb{P}(\text{correct})$ along with data maps for benchmark quality control.
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "hierarchical statistical model"
  - "multiple generations"
  - "benchmark variance analysis"
  - "prompt difficulty quantification"
  - "label error detection"
  - "data maps"
date: 2026-05-08
content_hash: 5367a80c53335c18
---

# Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation

**Conference**: NeurIPS 2025
**arXiv**: [2502.08943](https://arxiv.org/abs/2502.08943)  
**Authors**: Wenbo Zhang, Hengrui Cai (UC Irvine), Wenyu Chen (Meta)
**Area**: LLM Evaluation
**Keywords**: hierarchical statistical model, multiple generations, benchmark variance analysis, prompt difficulty quantification, label error detection, data maps

## TL;DR
This paper formalizes LLM benchmark evaluation as a hierarchical statistical model, theoretically demonstrates that multiple stochastic generations ($k>1$) reduce the variance of benchmark score estimates, and introduces a prompt-level difficulty metric $\mathbb{P}(\text{correct})$ along with data maps for benchmark quality control.

## Background & Motivation
Current LLM benchmark evaluation suffers from three core issues:

**Inconsistent generation strategies**: Some benchmarks (LiveBench, WildBench, OpenLLM Leaderboard) use greedy decoding, while others (TrustLLM, MT Bench, Alpaca Eval) use stochastic sampling, leading to significant discrepancies in evaluation results.

**High variance from single-generation evaluation**: Whether greedy or stochastic, current evaluation relies on a single generation to represent model performance, introducing substantial sampling variance in benchmark score estimates—particularly unreliable on small datasets.

**Inability to quantify prompt difficulty**: Single-generation evaluation cannot answer questions such as "which prompt is harder?", limiting the understanding of benchmark data composition.

The paper's core insight is that benchmark evaluation is fundamentally a **parameter estimation problem** that can be formalized via a hierarchical statistical model, and that multiple generations are key to reducing estimation variance and obtaining finer-grained information.

## Method

### Hierarchical Statistical Model
Given an LLM parameterized by $\theta$ and a benchmark $\mathcal{D}=\{x_i\}_{i=1}^n$ containing $n$ prompts, the difficulty $p_i$ of each prompt is drawn from an unknown benchmark difficulty distribution, and the correctness of each generation follows a Bernoulli distribution:

$$p_i \sim \mathbb{P}(\mu, \sigma; \theta), \quad y_{i,j} \sim \text{Bernoulli}(p_i)$$

where $p_i$ is the probability that the LLM correctly answers the $i$-th prompt (i.e., the latent difficulty of the prompt), and $y_{i,j}$ is the correctness indicator for the $j$-th generation. Method-of-moments estimators: $\hat{p}_i = \frac{\sum_j y_{i,j}}{k}$, $\hat{\mu} = \frac{\sum_i \hat{p}_i}{n}$.

### Core Theory: Variance Decomposition (Lemma 2.1)
The benchmark score estimator $\hat{\mu}$ is an unbiased estimator of $\mu$, with variance decomposed as:

$$\text{Var}(\hat{\mu}) = \underbrace{\frac{1}{nk}(\mu - \mu^2 - \sigma^2)}_{\text{Within-prompt variance}} + \underbrace{\frac{1}{n}\sigma^2}_{\text{Between-prompt variance}}$$

- **Within-prompt variance**: Arises from stochastic sampling of the same prompt; decreases linearly to zero as $k$ increases.
- **Between-prompt variance**: Arises from the distributional variation in prompt difficulty $p_i$; determined solely by $n$ and independent of $k$.

This decomposition explains why multiple generations are effective: they eliminate the first variance term, while the second is determined by the intrinsic properties of the benchmark. Based on the CLT, a 95% confidence interval can be constructed as $\hat{\mu} \pm 1.96\sqrt{\widehat{\text{Var}(\hat{\mu})}}$.

### Prompt-Level Difficulty Metric: $\mathbb{P}(\text{correct})$
The prompt-level difficulty metric is defined as $\mathbb{P}(\text{correct}) = p_i$, where a higher $p_i$ indicates an easier prompt. Its estimator $\hat{p}_i$ converges to the true value as $k$ increases. Compared to IRT models, the proposed approach does not require evaluation results from multiple LLMs; it estimates **subjective difficulty** (i.e., difficulty relative to the target model) using only multiple generations from the target model itself, making it more targeted.

### Data Maps and Label Error Detection
A semantic consistency metric is introduced: $\mathbb{S}(\text{consistency}) = \sum_{c=1}^C \text{Prop}_c \log \text{Prop}_c$, i.e., the negative entropy of semantic clusters (higher values indicate greater consistency). The $k$ generations are clustered into $C$ semantic groups, and the proportion of each group is computed.

**Key assumption**: Prompts with low $\mathbb{P}(\text{correct})$ but high $\mathbb{S}(\text{consistency})$ may be mislabeled—if the model consistently produces a "wrong" answer, the label itself may be erroneous (contradicting the self-consistency principle).

## Key Experimental Results

### Experimental Setup
- **Benchmarks**: MMLU-Pro (12,187 prompts), GSM8K (1,319), IFEval (541), MuSR (756)
- **Models**: Llama 3.1 8B/70B Instruct, Qwen 2.5 7B Instruct, Ministral 8B Instruct
- **Sampling**: Temperature 0.7, top-p 1.0, 50 generations per prompt, 0-shot CoT

### Greedy vs. Stochastic Sampling

| Benchmark | Model | Greedy (SE) | Sampling k=50 (SE) | Single-sample Δ(k=1) |
|-----------|-------|-------------|--------------------|-----------------------|
| MMLU-Pro | Llama 8B | 46.2 (0.45) | 46.1 (0.39) | 10.0 |
| GSM8K | Llama 8B | 86.1 (0.95) | 85.6 (0.68) | 18.6 |
| IFEval | Llama 8B | 74.5 (1.87) | 71.1 (1.51) | 8.3 |
| MuSR | Llama 8B | 24.8 (1.65) | 29.0 (1.00) | 8.2 |
| GSM8K | Llama 70B | 95.6 (0.56) | 95.3 (0.45) | 4.8 |
| MuSR | Llama 70B | 56.3 (1.80) | 57.9 (1.40) | 5.4 |

Key findings:
- Significant score gaps exist between greedy decoding and stochastic sampling (e.g., 3.4 points on GSM8K and 4.2 points on MuSR for Llama 8B).
- Single stochastic generation is highly unstable: Δ(k=1) reaches 18.6 on GSM8K for Llama 8B, implying that the best and worst single runs can differ by up to 18.6 points.
- Multiple sampling (k=50) substantially reduces SE compared to greedy decoding (MuSR: 1.65→1.00; GSM8K: 0.95→0.68).

### Distribution Characteristics of $\mathbb{P}(\text{correct})$
- **Difficult tasks** (MMLU-Pro, IFEval, MuSR): $\mathbb{P}(\text{correct})$ distributions are diffuse, with density roughly uniform over [0,1], indicating that LLMs behave close to random samplers on complex reasoning tasks.
- **Easier tasks** (GSM8K): Distributions are bimodal, concentrated near 0 and 1, reflecting lower uncertainty.
- **Larger models are more stable**: Llama 70B exhibits the most stable behavior across all benchmarks, with $\mathbb{P}(\text{correct})$ distributions more concentrated in the tails.

### Effect of Temperature
- Smaller models (8B) are sensitive to temperature: as temperature increases from 0.4 to 1.0, the $\mathbb{P}(\text{correct})$ distribution becomes more diffuse.
- Larger models (70B) are relatively insensitive to temperature, with distribution shapes remaining largely unchanged.

### Label Error Detection (GSM8K Case Study)
Using data maps from Llama 70B, prompts with $\mathbb{P}(\text{correct}) \leq 0.1$ and $\mathbb{S}(\text{consistency}) \geq -0.8$ are selected, yielding 18 candidates. Manual inspection reveals:
- **44.4% confirmed problematic** (22.2% with incorrect labels + 22.2% with ambiguous questions admitting multiple reasonable interpretations).
- This detection rate is achieved using only a single LLM and simple semantic measures.

### Ranking Reliability (Appendix F)
Comparing Llama 8B vs. Mistral 8B on GPQA: with multiple generations, Mistral consistently outperforms; however, with a single generation, there is a 20% probability of producing an incorrect ranking—demonstrating that model rankings are susceptible to sampling variance.

## Related Work & Insights
The 1PL IRT model $\mathbb{P}(y_{li}=1|\theta_l, b_i) = \sigma(\theta_l - b_i)$, when conditioned on a single LLM, is equivalent to $p_i$ in this paper (up to a sigmoid transformation). However, IRT requires joint evaluation across multiple LLMs to estimate parameters, whereas the proposed method requires only multiple generations from the target LLM itself, making it substantially more practical.

## Highlights & Insights
1. **Clear theoretical contribution**: The variance decomposition in Lemma 2.1 is concise and elegant, directly establishing the mathematical value of multiple generations—eliminating within-prompt variance.
2. **Strong practical utility**: The data map approach detects label errors using only a single model and simple clustering, and can be directly applied to benchmark quality control.
3. **Exposes fragility of existing evaluation**: The Δ(k=1) figures are striking; an 18.6-point swing on GSM8K suggests that current leaderboard rankings may be highly unreliable.
4. **Subjective vs. objective difficulty**: The paper points out that IRT's "objective" difficulty (averaged across models) may introduce bias when evaluating a specific model—prompt difficulty should be model-specific.

## Limitations & Future Work
- **High computational cost**: The inference overhead of 50 generations × $n$ prompts is substantial; the paper does not explore the minimum number of generations needed for sufficiently reliable estimates.
- **Overly strong independence assumption**: Prompts are assumed to be i.i.d., yet in practice prompts from the same source or topic may be correlated.
- **Limited label detection precision**: A 44.4% true positive rate means that more than half of flagged prompts are actually unproblematic, indicating a high false positive rate.
- **Evaluation limited to MCQ and short-answer settings**: For more open-ended generation tasks (e.g., summarization, translation), measuring semantic consistency requires more sophisticated approaches.

## Rating
⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond the Surface: Enhancing LLM-as-a-Judge Alignment with Human via Internal Representations](beyond_the_surface_enhancing_llm-as-a-judge_alignment_with_human_via_internal_re.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] Risk Management for Mitigating Benchmark Failure Modes: BenchRisk](risk_management_for_mitigating_benchmark_failure_modes_benchrisk.md)
- [\[NeurIPS 2025\] PFΔ: A Benchmark Dataset for Power Flow under Load, Generation, and Topology Variations](pfδ_a_benchmark_dataset_for_power_flow_under_load_generation_and_topology_variat.md)

</div>

<!-- RELATED:END -->
