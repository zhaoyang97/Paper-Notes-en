---
title: >-
  [Paper Note] CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges
description: >-
  [ACL 2025][LLM Evaluation][Selection Bias] Proposed CalibraEval, an unsupervised inference-time debiasing method. By formulating the debiasing problem as an optimization task, CalibraEval utilizes a Non-parametric Order-preserving Algorithm (NOA) to learn a calibration function that maps the observed probability distribution of LLM judges to an unbiased distribution, effectively mitigating selection bias in LLMs-as-Judges.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Selection Bias"
  - "Calibration"
  - "Non-parametric Algorithm"
  - "Debiasing"
date: 2026-05-08
content_hash: 1565481a125b1f26
---

# CalibraEval: Calibrating Prediction Distribution to Mitigate Selection Bias in LLMs-as-Judges

**Conference**: ACL 2025  
**arXiv**: [2410.15393](https://arxiv.org/abs/2410.15393)  
**Code**: [github.com/CSHaitao/CalibraEval](https://github.com/CSHaitao/CalibraEval)  
**Area**: LLM Evaluation  
**Keywords**: LLM Evaluation, Selection Bias, Calibration, Non-parametric Algorithm, Debiasing

## TL;DR

Proposed CalibraEval, an unsupervised inference-time debiasing method. By formulating the debiasing problem as an optimization task, CalibraEval utilizes a Non-parametric Order-preserving Algorithm (NOA) to learn a calibration function that maps the observed probability distribution of LLM judges to an unbiased distribution, effectively mitigating selection bias in LLMs-as-Judges.

## Background & Motivation

The "LLMs-as-Judges" paradigm utilizes powerful LLMs (such as GPT-4) for the automatic evaluation of generated text quality, and is particularly widely applied in pairwise comparison scenarios. However, LLM judges suffer from severe selection bias:

**Position Bias**: LLMs tend to prefer options at specific positions (such as the first or the last).

**Token Bias**: LLMs may assign higher probabilities to specific option identifiers (such as A or B).

These two biases collectively constitute selection bias, leading to inconsistent evaluation results when option positions or identifiers are swapped, which severely undermines the validity and fairness of the evaluation.

Limitations of Prior Work:
- Discarding inconsistent judgments or labeling them as "ties" loses evaluation details.
- Multi-round interaction/discussion methods are costly and have uncertain effectiveness.
- The Pride method assumes a simple linear multiplicative relationship between the biased and unbiased distributions, which is overly simplistic.
- Supervised methods require labeled data, exhibiting poor scalability.

## Method

### Overall Architecture

The core idea of CalibraEval is: rather than directly modeling the generation mechanism of the bias (i.e., the exact form of $f(\cdot)$), it learns a calibration function $g(\cdot)$ to map the observed biased probability distribution directly to the unbiased distribution:

$$P_{debiased}(t_i|I,X_0) = g(P_{observed}(t_i|I,X_0))$$

### Key Designs

1. **Four Combinations to Formulate the Optimization Objective**: For two options $o_1, o_2$ and two identifiers $t_1, t_2$ in a pairwise comparison, four permutations are constructed (default order, swapped positions, swapped identifiers, and swapped both). The optimization objective is established based on the intuition that "an unbiased judge should yield consistent results across all four combinations":

    $\min_{g \in \mathcal{G}} \sum_{i=1}^{K} [g(s_0^i) + g(s_2^i) - 1]^2 + [g(s_0^i) - g(s_1^i)]^2 - \lambda[g(s_0^i) - g(s_2^i)]^2$

    - The first term ensures consistency after swapping identifiers.
    - The second term ensures consistency after swapping positions.
    - The third term is a regularization term, preventing convergence to the trivial solution $g(\cdot) = 0.5$.

2. **Non-parametric Order-preserving Algorithm (NOA)**: The key algorithm to solve the aforementioned NP optimization problem. Its core assumption is that the calibration function $g(\cdot)$ preserves order for the same identifier—specifically, a higher observed probability should correspond to a higher unbiased probability. The detailed steps are as follows:

    - Collect three types of probability values (default, swapped positions, swapped identifiers) for $K$ samples in the estimation set.
    - Merge and sort all probability values, appending the boundary conditions $z_0=0, z_M=1$.
    - Introduce parameters $d_k$ initialized as $z_k$, and define the order-preserving mapping function via a softmax-like formulation:
    $g(z_k) = \frac{\sum_{i=0}^{k} \exp(d_i)}{\sum_{i=0}^{M} \exp(d_i)}$
    - Iteratively optimize the parameters $d_k$ using gradient descent, normalizing after each iteration to ensure a unique solution.
    - Upon convergence, fit the continuous calibration function $g^*(\cdot)$ using weighted least squares coupled with the PAVA algorithm.

3. **Label-free, Inference-time Calibration**: CalibraEval requires no explicit labels, learning the calibration function solely based on the relationships of the predictive distributions under different permutations. The calibration function can be computed after observing all test samples or estimated using a subset.

### Loss & Training

- The optimization objective is formulated as above, where $\lambda$ is the regularization hyperparameter.
- Gradient descent update: $d_k^{(new)} = d_k^{(old)} - \gamma \frac{\partial L}{\partial d_k}$
- Normalization constraint: $\sum_{i=0}^{M} d_i = 0$ after each iteration.
- Continuous function fitting: PAVA (Pool Adjacent Violators Algorithm).

## Key Experimental Results

### Main Results

| Method | Kappa↑ | ICC(2,k)↑ | ICC(3,k)↑ |
|------|--------|-----------|-----------|
| Llama-3-8B (No Debiasing) | 31.14 | 71.14 | 77.16 |
| + DI | 25.15 | 66.95 | 72.25 |
| + CC | 25.31 | 60.62 | 67.47 |
| + DC | 33.21 | 74.58 | 76.88 |
| + Pride | 37.35 | 76.83 | 78.20 |
| + **CalibraEval** | **39.16** | **83.38** | **84.30** |

Results on Qwen-72B:

| Method | Kappa↑ | ICC(2,k)↑ | ICC(3,k)↑ |
|------|--------|-----------|-----------|
| Qwen-72B (No Debiasing) | 77.47 | 92.63 | 93.06 |
| + Pride | 77.86 | 92.81 | 93.19 |
| + **CalibraEval** | **79.98** | **96.24** | **96.68** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Only first term (identifier consistency) | Kappa improved but limited | Single constraint is insufficient |
| Only first + second terms (no regularization) | Risk of trivial solutions | Regularization term is necessary |
| Complete three terms | Optimal | Terms of objective are complementary |
| Varying estimation set size | Stable with $K \ge 100$ | Insensitive to sample size |

### Key Findings

1. **CalibraEval consistently outperforms all baselines**: CalibraEval continually leads across various consistency and accuracy metrics on multiple models, including Llama-3-8B, Llama-3.1-8B, Qwen-14B, Qwen-72B, and ChatGPT.
2. **Limitations of Pride**: Pride assumes a linear multiplicative relation, which even leads to performance degradation on certain models.
3. **Instability of DI (Discard Inconsistent) and CC (Content-level Calibration)**: These simple methods perform inconsistently across different models and datasets.
4. **Cross-model Generalization**: The calibration function learned by CalibraEval remains effective under different prompt templates, option identifiers, and few-shot settings.
5. **Large Models also Benefit**: Even for large models like Qwen-72B, CalibraEval still improves ICC by 3-4 percentage points.
6. **Minimal Computational Overhead**: It only requires computing a lightweight calibration function at inference time, without additional training.

## Highlights & Insights

1. **Elegant Problem Formulation**: Formulating the debiasing problem as an optimization task and designing the objective function using the intuition that "unbiased evaluations should be consistent under four permutations" is logically natural and rigorous.
2. **Flexibility of the Non-parametric Approach**: NOA does not assume a specific form of bias and narrows the solution space using order-preserving constraints, making it much more general than Pride's linear multiplicative assumption.
3. **Practicality of the Label-free Design**: In real-world scenarios, the cost of obtaining human labels is the precise motivation for using LLM judges. Thus, a label-free method holds substantial practical value.
4. **Exquisite Design of Softmax Order-preserving Mapping**: The cumulative softmax expression naturally satisfies the order-preserving constraints and is differentiable, facilitating gradient optimization.

## Limitations & Future Work

- Requires running inference on four permutations for each sample (increasing inference cost by approximately 3x). Although more efficient than multi-round discussion methods, it still incurs additional overhead.
- The order-preserving assumption might not fully hold in extreme bias scenarios.
- Focuses only on pairwise comparison scenarios, without extending to pointwise evaluations or multi-option comparisons.
- The calibration function is learned for a specific model, which requires recalibration when changing models.
- Co-utilization with model fine-tuning methods (such as BCT) remains unexplored.
- The selection of the $\lambda$ hyperparameter relies on empirical tuning, lacking an automatic selection mechanism.

## Related Work & Insights

- **Relation to Pride (Zheng et al., 2023)**: Pride assumes $P_{observed} \propto P_{prior} \times P_{debiased}$ (linear multiplication), which is a special case of CalibraEval. CalibraEval avoids this simplifying assumption and directly learns a general mapping function.
- **Relation to Contextual Calibration**: The latter calibrates token probabilities of LLMs in classification tasks, while CalibraEval extends a similar concept to pairwise comparison evaluation scenarios.
- **Insights for LLM Evaluation Pipelines**: CalibraEval can serve as a standard post-processing step within any LLM-as-Judge pipeline to enhance evaluation reliability.
- **Potential Impact on RLHF**: LLMs are also used for preference judgments in RLHF, where bias calibration could improve the quality of reward models.

## Rating

- Novelty: ⭐⭐⭐⭐ The methodology of formulating the debiasing problem as an optimization task is novel, and the order-preserving design of the NOA algorithm is exquisite.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation with multiple models, benchmarks, and metrics, showcasing robust generalizability (across prompts, tokens, and few-shot variations).
- Writing Quality: ⭐⭐⭐⭐ Derivations are clear and the motivation is well-defined, though some mathematical notations are dense.
- Value: ⭐⭐⭐⭐⭐ Resolves a critical bottleneck in LLM evaluations; the method is plug-and-play with extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ODP-Bench: Benchmarking Out-of-Distribution Performance Prediction](../../ICCV2025/llm_evaluation/odp-bench_benchmarking_out-of-distribution_performance_prediction.md)
- [\[ACL 2025\] Are Bias Evaluation Methods Biased?](are_bias_evaluation_methods_biased.md)
- [\[ACL 2025\] JuStRank: Benchmarking LLM Judges for System Ranking](justrank_llm_judge_system_ranking.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)

</div>

<!-- RELATED:END -->
