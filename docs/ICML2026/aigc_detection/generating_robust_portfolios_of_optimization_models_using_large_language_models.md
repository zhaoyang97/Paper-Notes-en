---
title: >-
  [Paper Note] Generating Robust Portfolios of Optimization Models using Large Language Models
description: >-
  [ICML 2026][AIGC Detection][Optimization modeling] This paper proposes a lightweight, training-free algorithm that utilizes a single LLM to perform roles as both a "stochastic generator" and a "scoring evaluator." By pac…
tags:
  - "ICML 2026"
  - "AIGC Detection"
  - "Optimization modeling"
  - "candidate portfolios"
  - "LLM evaluation"
  - "human-in-the-loop"
  - "coverage guarantees"
date: 2026-05-08
content_hash: c024354cc4cb7a0f
---

# Generating Robust Portfolios of Optimization Models using Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.27013](https://arxiv.org/abs/2605.27013)  
**Code**: None  
**Area**: Optimization Modeling / LLM-as-Generator / LLM-as-Judge  
**Keywords**: Optimization modeling, candidate portfolios, LLM evaluation, human-in-the-loop, coverage guarantees

## TL;DR
This paper proposes a lightweight, training-free algorithm that utilizes a single LLM to perform roles as both a "stochastic generator" and a "scoring evaluator." By packing candidate optimization models into a portfolio until their cumulative generation probability reaches a prefix sum of $1-\alpha$, it theoretically proves that if *either* the generator or the evaluator aligns with human preferences, the portfolio is guaranteed to contain a high-quality optimization model. Empirical results on NL4LP using GPT demonstrate that this portfolio-based approach consistently outperforms random sampling in worst-case scenarios.

## Background & Motivation
**Background**: Formalizing real-world decision problems (resource allocation, scheduling, planning) into mathematical optimization models is the most difficult bottleneck in Operations Research, as it requires expertise in both the business domain and optimization modeling itself. Recently, several works have emerged for automated optimization modeling using LLMs (OptiMUS, LLMOPT, Autoformulation, ORLM, etc.), typically focusing on end-to-end fine-tuning or designing reward/objective functions.

**Limitations of Prior Work**: These methods almost exclusively output a **single** optimization model. Due to the inherent randomness and hallucination rates of LLMs, single-model quality has no guarantee. Improving reliability often requires expensive retraining or RLHF. Consequently, decision-makers cannot judge the quality of the output model nor do they have secondary options as a backup.

**Key Challenge**: LLMs possess two distinct capabilities for optimization modeling—as a **stochastic generator** (providing diverse candidates through multiple samplings) and as a **reasoning evaluator/judge** (scoring candidates based on world knowledge). Existing research tends to use only one (either sampling or judging), failing to unify them. If the generator is biased, the judge cannot recover the result, and vice versa.

**Goal**: To output a set (portfolio) of optimization models without training or fine-tuning, while providing **theoretical coverage guarantees**: as long as either the generator or the evaluator is consistent with human rankings, the portfolio will contain a high-quality model, supporting a human-in-the-loop selection process.

**Key Insight**: The authors observe that the generation probability $p(o)$ from the generator and the rank $\pi_e(d)$ from the evaluator are **independent** signals. By combining them—specifically, by truncating based on cumulative generator probability while following the evaluator's ranking—the signals can back each other up.

**Core Idea**: Add candidates to the portfolio starting from the highest evaluator rank until the cumulative **generation probability** reaches $1-\alpha$. This truncation ensures the portfolio benefits from both "evaluator ranking coverage" and "generation probability coverage."

## Method

### Overall Architecture
Given a natural language description $d$:

1.  **Generation Phase**: Treat the LLM as a generator $g$ and sample $N$ times (e.g., $N=50$). Each sample produces a candidate optimization model $o \in \mathcal{O}$ consisting of natural language explanation and Python code. The generation probability $p(o)$ is estimated using normalized token-level log-probabilities.
2.  **Evaluation Phase**: Switch the same LLM to an evaluator role $e$. Execute the code for each candidate, feed the output and problem description back into the prompt, and let the LLM assign a score (1–100). Average the scores across multiple runs (e.g., 4 times) to get the rank $\pi_e(d) = (o_{(1)^e}, o_{(2)^e}, \ldots)$.
3.  **Portfolio Construction**: Accumulate candidates into the portfolio based on the rank $\pi_e(d)$, stopping when the sum of their generation probabilities first $\geq 1-\alpha$. Formally, $\mathcal{P}(d;\alpha)=\{o_{(i)^e}\}_{i=1}^{k^*(\alpha)}$, where $k^*(\alpha)=\inf\{k:\sum_{i=1}^k p(o_{(i)^e}) \geq 1-\alpha\}$.
4.  **Decision-maker Backup**: The decision-maker selects one model from these $k^*$ candidates, which is a manageable set with theoretical quality guarantees.

### Key Designs

1.  **Dual Cutoff via "Probability Truncation + Rank Pruning"**:
    *   **Function**: Uses evaluator ranking to determine the "insertion order" and generator cumulative probability to determine the "stopping point."
    *   **Mechanism**: Traverses candidates by evaluation rank while maintaining a cumulative sum $S_k=\sum_{i=1}^k p(o_{(i)^e})$. Once $S_k \geq 1-\alpha$, the process stops. The intuition is: if the evaluator is reliable, the top candidates contain a good model; if the evaluator is unreliable but the generator is reliable, top positions will likely contain good models due to their higher sampling probability.
    *   **Design Motivation**: Traditional methods using only top-k scores or top-p probabilities are fragile if one component fails; this combination allows either signal to compensate for the other.

2.  **Unified Coverage Definition and "OR" Alignment Hypothesis**:
    *   **Function**: Quantifies portfolio quality as coverage $c(\mathcal{P})=\frac{1}{k}\sum_{i=1}^k \mathbb{I}\{o_{(i)^*}\in \mathcal{P}\}$, measuring how many top-k human-ranked candidates are in the portfolio.
    *   **Mechanism**: Defines **Evaluator Alignment** ($\pi_e(d)=\pi^*(d)$) and **Generator Alignment** ($i\leq j \Rightarrow p(o_{(i)^*})\geq p(o_{(j)*})$). It is proved that: (i) If the evaluator is aligned, $c(\mathcal{P})=1$ for any generator; (ii) If the generator is aligned, $c(\mathcal{P})>\frac{1-2\alpha}{k^*(\alpha)}>0$ even with a poor evaluator for $\alpha \in (0, 1/2)$.
    *   **Design Motivation**: Earlier works required both components to be reliable (an "AND" condition). This work relaxes this to an "OR" condition—guarantees hold if at least one role (generation or evaluation) aligns with human preferences.

3.  **Dual-Role Single Model + Code Execution Feedback**:
    *   **Function**: Uses the same LLM for both generation and evaluation to minimize costs.
    *   **Mechanism**: The generator outputs Python code; the evaluator first **executes** the code to get numerical results, then scores the model using "description + model + execution output."
    *   **Design Motivation**: Optimization correctness must be verified by execution. Purely semantic evaluation can be deceived by syntactically correct but logically flawed models.

### Loss & Training
The approach requires **no training, no fine-tuning, and no RLHF**. It relies entirely on prompted sampling. The only hyperparameter is $\alpha$, which controls the trade-off between coverage and portfolio size. Theoretical proofs rely on Lemma: under evaluator alignment, probability accumulation to $1-\alpha$ covers at least the top $k^*$ human choices. Lower bounds are derived in Appendix A.

## Key Experimental Results

### Main Results

**Synthetic Data**: Candidate space $|\mathcal{O}|=K \in \{10, 20, 50, 100\}$. Generator settings: Aligned / Weakly Aligned / Uniform / Misaligned. Evaluator error rates $\epsilon \in \{0, 0.3, 0.5, 0.7, 1\}$.

| Setting ($K=100$) | $\alpha$ Range | Empirical Coverage | vs. Theory ($\frac{1-2\alpha}{k^*}$) |
|---|---|---|---|
| Weakly Aligned generator, $\epsilon=0$ | $(0, 0.5)$ | $\geq 1-\alpha$ | Far above theoretical bound |
| Weakly Aligned generator, $\epsilon=0.5$ | $(0, 0.5)$ | $\approx 1-\alpha$ | Satisfies Prop. 3.6 |
| Aligned generator, $\epsilon=1.0$ (worst judge) | $(0, 0.5)$ | High | High coverage at cost of larger size |

**Real-world Data (NL4LP 25 Tasks)**: Generator = `gpt-5.4-nano` (50 samples); Judge = `gpt-5.4`. Quality measured by the **worst-case** score within the portfolio.

| Portfolio Size $s$ | Ours (LLM-as-evaluator) | Ours (generator-prob-as-evaluator) | Random Portfolio |
|---|---|---|---|
| 2 | Significantly Superior | Moderately Superior | Baseline |
| 4 | Significantly Superior | Moderately Superior | Baseline |
| 8 | Significantly Superior | Moderately Superior | Baseline |

### Ablation Study

| Configuration | Coverage Behavior | Description |
|---|---|---|
| Full: reasoning evaluator + prob truncation | Highest worst-case score | Complete method |
| w/o evaluator (rank by prob only) | Scores drop but still > random | Loses execution feedback but retains prob signal |
| w/o prob truncation (fixed top-k) | No $\alpha$ slider, no guarantee | Equivalent to standard LLM-as-judge baseline |
| Random Portfolio | Lowest worst-case score | Standard baseline, outperformed at all sizes $s$ |

### Key Findings
- **"OR" Alignment Hypothesis validated**: Even with a reversed evaluator ($\epsilon=1$), coverage remains strictly positive if the generator is at least weakly aligned.
- **Coverage/Size Trade-off**: Stronger generator alignment leads to higher coverage for the same $\alpha$; stronger evaluator alignment minimizes portfolio size.
- **Code Execution Improves Reasoning**: Reasoning evaluators outperform pure probability-based ranking, indicating that execution feedback provides critical signals beyond semantics.

## Highlights & Insights
- **Dual-role, Single Model, OR-type Guarantee**: The unification of generator and judge into a single stopping rule with minimal overhead is highly efficient and robust.
- **Relaxed Alignment Requirements**: Switching from an "AND" to an "OR" requirement for component reliability is a significant methodological shift, applicable to code generation, SQL synthesis, and RL reward shaping.
- **Execution-Informed Evaluation**: Integrating traditional solver outputs into the LLM evaluation process transforms the judge from "looking correct" to "calculating correct."
- **User-tunable Interface**: The $\alpha$ parameter provides a clear interface for human-in-the-loop systems to balance risk (coverage) against effort (number of options).

## Limitations & Future Work
- **Probabilistic Accuracy**: Estimating $p(o)$ via token log-probs may be inaccurate for long sequences or tail-end candidates.
- **Loose Theoretical Bound**: The $\frac{1-2\alpha}{k^*}$ bound is much looser than empirical findings; tighter instance-dependent bounds are needed.
- **Alignment Verification**: Defining "alignment" requires knowing human ground truth, which is unavailable in practice; cheaper proxies for alignment are needed.
- **Benchmarking**: Direct head-to-head comparisons against end-to-end systems like OptiMUS on "best-in-portfolio" vs. "single-output" metrics would further strengthen the conclusions.

## Related Work & Insights
- **vs. OptiMUS / LLMOPT**: Those focus on single-model output through agent collaboration or tuning. Ours provides a set with robustness guarantees.
- **vs. Eureka / Text2Reward**: Those use environment feedback to optimize single reward functions; we extend this to full optimization models and set-based outputs.
- **vs. OPRO**: While OPRO uses LLMs as black-box optimizers, this work uses LLMs to **generate** models for traditional solvers, maintaining interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐ The cumulative truncation rule for "OR" alignment is a clean and elegant contribution.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic experiments are strong, but real-world benchmarks could be larger.
- Writing Quality: ⭐⭐⭐⭐ Clear definitions, logical flow, and intuitive visualizations.
- Value: ⭐⭐⭐⭐ The methodology is easily transferable to any structural generation task requiring validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators](feature-augmented_transformers_for_robust_ai-text_detection_across_domains_and_g.md)
- [\[NeurIPS 2025\] ASCIIBench: Evaluating Language-Model-Based Understanding of Visually-Oriented Text](../../NeurIPS2025/aigc_detection/asciibench_evaluating_language-model-based_understanding_of_visually-oriented_te.md)
- [\[ICLR 2026\] CLARC: C/C++ Benchmark for Robust Code Search](../../ICLR2026/aigc_detection/clarc_cc_benchmark_for_robust_code_search.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](../../NeurIPS2025/aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[ICML 2026\] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)

</div>

<!-- RELATED:END -->
