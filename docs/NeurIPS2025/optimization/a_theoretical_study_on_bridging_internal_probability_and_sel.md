---
title: >-
  [Paper Note] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning
description: >-
  [NeurIPS 2025][Optimization][Self-Consistency] This paper proposes the first theoretical framework for sampling-based test-time scaling methods…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Self-Consistency"
  - "Perplexity"
  - "Test-time Scaling"
  - "Confidence Estimation"
  - "LLM Reasoning"
date: 2026-05-08
content_hash: 4910a00ce6905c19
---

# A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2510.15444](https://arxiv.org/abs/2510.15444)  
**Code**: [https://wnjxyk.github.io/RPC](https://wnjxyk.github.io/RPC)  
**Area**: Optimization
**Keywords**: Self-Consistency, Perplexity, Test-time Scaling, Confidence Estimation, LLM Reasoning

## TL;DR
This paper proposes the first theoretical framework for sampling-based test-time scaling methods, decomposing reasoning error into estimation error and model error. It reveals the limitations of Self-Consistency (slow convergence) and Perplexity (large model error), and introduces the RPC method that combines the strengths of both, achieving comparable reasoning performance on 7 benchmarks with only 50% of the sampling cost.

## Background & Motivation
Test-time scaling improves LLM reasoning capability by allocating additional computational resources at inference time. Sampling-based methods—which generate multiple reasoning paths and select the best—have been widely adopted. Existing confidence estimation approaches fall into two categories: (1) **consistency-based methods**, exemplified by Self-Consistency (SC), which estimate answer confidence via majority voting; and (2) **probability-based methods**, exemplified by Perplexity (PPL), which directly leverage LLM internal probabilities. Despite their empirical success, these methods lack rigorous theoretical analysis to understand their mechanisms, inherent limitations, and directions for improvement. This paper aims to fill this theoretical gap.

## Core Problem
What are the fundamental limitations of Self-Consistency and Perplexity respectively in sampling-based test-time scaling? Can a method be found that simultaneously achieves the advantages of both—fast estimation error convergence and low model error?

## Method

### Overall Architecture
Given a reasoning problem $(x, y)$, the LLM samples $n$ reasoning paths $\tilde{t}_1, \ldots, \tilde{t}_n$. A confidence estimation function $\hat{p}$ evaluates the credibility of each candidate answer, and the answer with the highest confidence is ultimately selected. The paper establishes a theoretical framework that decomposes reasoning error (squared error) into two independent components, analyzes the shortcomings of existing methods, and proposes improvements.

### Key Designs

1. **Reasoning Error Decomposition (Proposition 1)**: For any answer $\hat{y}$, the reasoning error decomposes as:
    $\mathcal{E}_{\hat{p}}(\hat{y}) = \underbrace{\mathbb{E}[(\hat{p}(\hat{y}|x) - p(\hat{y}|x))^2]}_{\text{Estimation Error}} + \underbrace{(p(\hat{y}|x) - \mathbb{I}[\hat{y}=y])^2}_{\text{Model Error}}$
    - **Estimation Error**: depends on the number of samples and the confidence estimation strategy; can be reduced through better estimation methods.
    - **Model Error**: determined by the LLM's intrinsic reasoning capability; fixed and irreducible.

2. **Theoretical Analysis of SC and PPL**:

    - **SC** (Proposition 2): Estimation error is $\frac{1}{n} p(\hat{y}|x)(1-p(\hat{y}|x))$, converging only at the linear rate $O(1/n)$; performance degrades under limited sampling. However, model error is relatively low.
    - **PPL** (Proposition 3): Estimation error converges at the exponential rate $(1-p(\hat{t}|x))^n$, but model error is larger than SC (due to the absence of a consistency aggregation function over equivalent answers). The convergence advantage also degrades when probabilities are extremely low.

3. **Perplexity Consistency (PC)**: The core innovation—integrating LLM internal probabilities into the Self-Consistency framework. For each answer $\hat{y}$, the estimated confidence is the sum of probabilities over all corresponding reasoning paths:
    $\hat{p}^{(\text{PC})}(\hat{y}|x) = \sum_{\tilde{t} \in \mathcal{R}} \mathbb{I}[g(\tilde{t})=\hat{y}] \cdot p(\tilde{t}|x)$
    Theoretical guarantee (Theorem 4): PC's estimation error converges at the exponential rate $\alpha^n$ (where $\alpha = 1 - \frac{1}{k}p(\hat{y}|x)$), while maintaining the same low model error as SC.

4. **Reasoning Pruning (RP)**: Addresses the degradation of PC on low-probability answers. When $p(\hat{y}|x) \to 0$, exponential convergence degrades to linear. RP automatically filters low-probability reasoning paths by modeling the probability distribution:
    - Fits a mixture of two Weibull distributions to the probability distribution of all sampled paths.
    - Estimates parameters via maximum likelihood estimation and computes the posterior probability of each path belonging to the high-probability component.
    - Removes paths with $P_{\text{High}} < 0.5$, with a truncated mean as a fallback.
    - Theorem 7 proves that under the optimal threshold, RP achieves optimal error reduction with high probability.

5. **RPC Method**: Applies Reasoning Pruning to filter low-probability paths, then computes confidence via Perplexity Consistency. RPC is a hyperparameter-free, plug-and-play method.

### Loss & Training
RPC is a post-hoc method that requires no modification to the LLM architecture or training procedure. The Weibull mixture distribution parameters are fitted on-the-fly via maximum likelihood estimation over the sampled paths for each problem.

## Key Experimental Results

| Dataset | Metric | RPC | SC | PPL | Gain (vs SC) |
|---------|--------|-----|----|-----|--------------|
| MATH | Acc | 51.95 | 50.57 | 46.99 | +1.38 |
| MathOdyssey | Acc | 31.62 | 28.25 | 27.35 | +3.37 |
| OlympiadBench | Acc | 11.14 | 11.07 | 7.27 | +0.07 |
| AIME | Acc | 9.74 | 9.40 | 5.96 | +0.34 |

Efficiency comparison (minimum number of samples required to match SC's best performance):

| Dataset | SC Requires | RPC Requires | Sampling Reduction |
|---------|------------|--------------|-------------------|
| MATH | 64 | 32 | −50.0% |
| MathOdyssey | 112 | 32 | −71.4% |
| OlympiadBench | 128 | 64 | −50.0% |
| AIME | 128 | 48 | −62.5% |

### Ablation Study
- **PC module contribution**: PC alone improves convergence rate but suffers from degradation on some datasets (e.g., MathOdyssey).
- **RP module contribution**: RP yields the most significant gains on MATH and MathOdyssey, effectively addressing the degradation caused by low-probability paths.
- **Cross-model generalization**: Effective on InternLM2-Math 1.8B, 7B, and DeepSeek-Math 7B.
- **Cross-task generalization**: Effective on code generation (HumanEval/MBPP/APPS) and logical reasoning (GPQA/LogiQA).
- **Compatibility with R1 models**: On DeepSeek-R1-Distill-Qwen-7B (MATH, 16 samples): RPC (61.11) vs. SC (57.22) vs. PPL (60.04).
- **Compatibility with advanced methods**: RPC+ESC and RPC+BoN(RM) both outperform their respective baselines.
- **High-temperature sampling**: At T=1.3, SC performance degrades while RPC continues to improve (MATH: RPC 53.12 vs. SC 50.65).
- **Hyperparameter robustness**: Performance remains stable across different initializations and parameter ranges.
- **Computational overhead**: RPC's additional cost is negligible (0.036s vs. 0.006s per question), far below LLM inference time.

## Highlights & Insights
- **First theoretical framework**: Clearly decomposes reasoning error in sampling-based test-time scaling into estimation error and model error, providing principled guidance for method design.
- **Precise diagnosis**: Theoretical analysis precisely identifies SC's linear convergence bottleneck and PPL's large model error, rather than offering vague qualitative observations.
- **Elegant integration**: PC achieves both exponential convergence and low model error through a simple probability-weighted consistency formula.
- **Automated pruning**: The Weibull mixture distribution automatically models the probability distribution without requiring manual threshold tuning, enhancing practicality.
- **50% sampling cost reduction**: Substantially reduces the number of LLM inference calls while maintaining equivalent performance, directly saving computational resources.

## Limitations & Future Work
- **Strong theoretical assumptions**: The framework assumes LLM sampling follows a Bernoulli distribution and that sampled paths are mutually distinct, which may not strictly hold in practice.
- **Limited ceiling as a post-hoc method**: Since RPC does not modify model training, performance gains are bounded by the quality of the sampled reasoning paths.
- **Only two representative methods analyzed**: The theoretical framework has the potential to analyze additional methods (e.g., MCTS, reward model scoring), but this is not explored in the paper.
- **Sampling strategy not investigated**: Theory suggests that diversity in sampling is important, but the paper does not explore how to design better sampling strategies.
- **Applicability of Weibull mixture**: Assuming two Weibull distributions for all problems may be an oversimplification.

## Related Work & Insights
- **vs. SC (Wang et al., 2022)**: This paper theoretically demonstrates that SC's estimation error converges only linearly; RPC elevates this to exponential convergence via probability weighting.
- **vs. CISC (Taubenfeld et al., 2025)**: CISC also explores the combination of confidence and self-consistency, but this paper provides a more complete theoretical framework with explicit error decomposition.
- **vs. ESC (Li et al., 2024)**: ESC reduces SC cost via early stopping; RPC improves efficiency through the estimation method itself. The two approaches are complementary and can be combined.
- **vs. TTSC (Huang et al., 2025)**: TTSC performs self-calibration, consistent with the core ideas of this paper, but this paper provides a theoretical explanation.

The error decomposition framework can be transferred to test-time scaling analysis in multimodal reasoning. The probability-weighted consistency idea in PC can be extended to reward model scoring scenarios—substituting reward scores for LLM internal probabilities. The Weibull mixture modeling approach in RP can be applied to other settings that require distinguishing high- from low-quality samples.

## Rating
- Novelty: ⭐⭐⭐⭐ First theoretical framework for sampling-based test-time scaling; method design is theory-driven.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 datasets, multiple model scales/architectures, extensive ablations, and compatibility tests with advanced methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear; proposition–theorem–remark structure is rigorous; experimental presentation is well-organized.
- Value: ⭐⭐⭐⭐ Theoretical analysis provides a valuable cognitive framework for the field; RPC is simple, effective, and directly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Do Neural Networks Need Gradient Descent to Generalize? A Theoretical Study](do_neural_networks_need_gradient_descent_to_generalize_a_theoretical_study.md)
- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)
- [\[NeurIPS 2025\] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees](verbalized_algorithms.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[ICLR 2026\] ∇-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space](../../ICLR2026/optimization/nabla-reasoner_llm_reasoning_via_test-time_gradient_descent_in_latent_space.md)

</div>

<!-- RELATED:END -->
