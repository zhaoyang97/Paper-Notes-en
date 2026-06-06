---
title: >-
  [Paper Note] Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation
description: >-
  [ICML 2026][LLM Evaluation][Symbolic Regression] DoLQ inserts a "Scientist Agent" into the search loop of LLM-based symbolic regression to perform simultaneous qualitative (physical plausibility) and quantitative (ablati…
tags:
  - "ICML 2026"
  - "LLM Evaluation"
  - "Symbolic Regression"
  - "Ordinary Differential Equations"
  - "LLM Agent"
  - "Physical Interpretability"
  - "Hypothesis Search"
date: 2026-05-08
content_hash: d8d5b48ca4fba0b3
---

# Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation

**Conference**: ICML 2026  
**arXiv**: [2605.07323](https://arxiv.org/abs/2605.07323)  
**Code**: https://github.com/Bon99yun/DoLQ  
**Area**: Scientific Discovery / Symbolic Regression / LLM Multi-Agent  
**Keywords**: Symbolic Regression, Ordinary Differential Equations, LLM Agent, Physical Interpretability, Hypothesis Search

## TL;DR
DoLQ inserts a "Scientist Agent" into the search loop of LLM-based symbolic regression to perform simultaneous qualitative (physical plausibility) and quantitative (ablative MSE contribution) evaluations. This forces LLM-SR from producing "low-error but bloated and physically absurd" candidates toward equations that are both numerically precise and structurally compact.

## Background & Motivation
**Background**: Reconstructing Ordinary Differential Equations (ODEs) from observational data is a core problem in scientific machine learning. Early methods like SINDy relied on predefined basis function libraries and sparse regression; subsequent symbolic regression (SR) utilized evolutionary algorithms or Transformers (e.g., ODEformer) for automated search. The latest generation, such as LLM-SR and LASR, employs LLMs as candidate generators, leveraging scientific priors to significantly compress the search space.

**Limitations of Prior Work**: Existing methods evaluate candidate equations primarily through numerical metrics like MSE and complexity. However, Figure 1 presents a critical counterexample: two equations with nearly identical MSE can differ drastically in extrapolation performance and physical meaning. LLM-SR frequently outputs "garbage bag" equations containing over ten terms, where the true terms are diluted by physically meaningless ones, leading to mutual interference during parameter optimization.

**Key Challenge**: Purely numerical evaluation only reflects "current fitting quality" but cannot determine whether a specific term corresponds to a real physical mechanism. The latter is precisely what determines stable extrapolation in ID-Ext / OOD intervals.

**Goal**: (1) Explicitly inject physical plausibility into the search loop; (2) Actively prune terms that "appear to contribute but are actually overfitting"; (3) Stably recover structures in multi-dimensional ODEs (including 4D systems like Glider).

**Key Insight**: LLMs possess both numerical reasoning capabilities and domain knowledge. Instead of using them solely as "Nominators," an independent LLM can act as a "Reviewer" to score candidates from orthogonal semantic and numerical dimensions, feeding this feedback into the next round of the nominator's prompt.

**Core Idea**: A dual-track review system (Qualitative + Quantitative) $\rightarrow$ Double Veto: remove semantically unreasonable terms immediately and remove terms with no numerical contribution after a delay. This guides the search toward compact equations that are both physically plausible and numerically optimal.

## Method

### Overall Architecture
DoLQ is a three-agent iterative closed loop, operating through a "Nominate $\rightarrow$ Optimize $\rightarrow$ Review" cycle:

1. **Sampler Agent**: Receives the system description $\mathcal{T}$ and Scientist feedback, then produces executable Python code snippets for several candidate terms (e.g., `params[0] * np.sin(x[1])`) along with natural language physical explanations for each term.
2. **Parameter Optimizer**: Instantiates symbolic terms into differentiable skeleton functions $f_j(t, \boldsymbol{x}; \boldsymbol{\theta})$ and fits parameters $\boldsymbol{\theta}$ using a hybrid optimizer.
3. **Scientist Agent**: Performs quantitative (ablation MSE) and qualitative (semantic alignment) reviews of the optimized equations, yielding keep/hold/remove decisions and cumulative insights to be fed back to the Sampler.

The outer loop executes for 100 rounds. In each round, the Sampler proposes multiple candidate hypotheses, which are evaluated in parallel, and the "Global Best" is continuously updated.

### Key Designs

1. **Dual-Track Review by Scientist Agent**:
    - **Function**: Determines whether to keep or remove each candidate term.
    - **Quantitative Track**: Performs ablation for each term by setting its coefficient to zero and recalculating the residual MSE. If the error rises significantly $\rightarrow$ good; if it remains unchanged $\rightarrow$ neutral; if it decreases $\rightarrow$ bad (indicating the term is "point-hunting" through overfitting). In Figure 2, `np.sin(x1)` is judged good because the MSE increases by 78.2% upon its removal, while `x1**4` is judged bad because the MSE decreases.
    - **Qualitative Track**: The LLM reads the system description $\mathcal{T}$, the symbolic form of the term, and the Sampler's explanation, scoring it as good/neutral/bad. The criterion is whether it corresponds to physical mechanisms mentioned in the description (e.g., "air resistance" corresponding to $-cx^2$ is good).
    - **Track Fusion**: Terms that are semantically bad are removed immediately. Terms must be good in both tracks to be kept; others are held. Any term held for two consecutive rounds is automatically removed. Removed skeletons (with parameters replaced by placeholders) are added to a blacklist to prevent regeneration, featuring a probabilistic forgetting mechanism to avoid permanent "mis-kills."
    - **Design Motivation**: Numerical MSE alone retains spurious terms that fit noise, while semantic review alone might miss "unfamiliar but necessary" forms. Independent tracks with mutual veto power are the core differentiators from standard LLM-SR.

2. **Structured Prompting & Parallel Hypotheses**:
    - **Function**: Enables the LLM to propose "executable and interpretable" candidates rather than just symbols.
    - **Mechanism**: The prompt is divided into: (a) task definition and system description $\mathcal{T}$; (b) Scientist feedback, including cumulative insights, term-by-term reviews, and blacklisted skeletons; (c) output format constraints, forcing each term into a Python format like `params[0] * np.cos(x[0])` with a physical justification. 
    - **Design Motivation**: (1) Executable formats bypass symbolic parsing; (2) Mandatory justifications externalize the LLM's "implicit physical knowledge" for the Scientist's review; (3) Parallel hypotheses and blacklisting prevent entrapment in a single search path.

3. **Hybrid DE + BFGS Parameter Optimization**:
    - **Function**: Moves correct skeletons from local optima to global optima.
    - **Mechanism**: Primarily uses Differential Evolution (DE) for global search in the parameter space, followed by BFGS for local refinement. Three strategies (BFGS only, DE only, Hybrid) are run simultaneously, and the one with the lowest MSE is selected. The objective is the residual MSE: $$\sum_i (\dot{x}_j(t_i) - f_j(t_i, \boldsymbol{x}; \boldsymbol{\theta}))^2$$.
    - **Design Motivation**: BFGS is highly sensitive to initialization. Correct skeletons are often rejected because poor initial values lead to poor MSE, causing the Scientist to incorrectly remove them. The DE backup prevents such "false negatives" where the skeleton is correct but parameters are not found.

### Loss & Training
This is a training-free framework without backpropagation. The core "training signal" is the Scientist Agent's term-wise keep/hold/remove decisions, which modify the Sampler's distribution through prompt rewriting. Performance is measured using Normalized MSE (NMSE) and success rates. Each system is run three times, and the best is taken over 100 rounds for fair comparison with baselines.

## Key Experimental Results

### Main Results
Using Gemini 2.5 Flash Lite as the backbone, the method was evaluated on 8 systems from ODEbench. Dimensional average NMSE (Residual / Integral) comparison:

| System | Metric | ICSR | LASR | LLM-SR | EDL | **DoLQ** |
|------|------|------|------|--------|-----|----------|
| SIR(2D) ID | Residual | 2.8e-8 | 1.8e-8 | 1.7e-8 | 18.6 | **1.7e-8** |
| CDIMA(2D) ID | Integral | 3.8e-4 | 2.9e-1 | 3.1e-3 | 1.3e5 | **2.4e-8** |
| Glider(4D) ID | Residual | 2.6e-2 | 2.5e-2 | 9.95e-7 | 1.5e4 | **1.2e-6** |
| CDIMA(2D) ID-Ext | Integral | 2.6e-4 | 4.9e1 | 9.4e-3 | NaN | **1.2e-7** |

CDIMA is a nonlinear saturation system where all baselines collapse during extrapolation (ID-Ext), while DoLQ remains stable at the 1e-7 magnitude—a 5 to 6 order of magnitude improvement, highlighting that correct structure is essential for stable extrapolation. Success rates for both NMSE tests and structural term tests show DoLQ achieving SOTA performance.

### Ablation Study

| Configuration | Convergence Iteration (Glider 2D) | Note |
|------|---------------------|------|
| Full DoLQ | Round 27 | Full Scientist feedback |
| w/o Scientist | Round 62 | MSE feedback only, no qualitative |
| BFGS only | Failed | Correct skeletons fail to find parameters |
| DE only | Unstable | Global search without fine-tuning |
| **DE + BFGS** | Stable Convergence | Hybrid strategy as used in Figure 8 |

Ablations show that ground truth terms are judged as "keep" much more frequently than others. Even if misclassified as "remove," the probabilistic forgetting allows them to resurface and eventually converge.

### Key Findings
- **Qualitative Review acts as an Accelerator**: Removing the Scientist Agent more than doubles the convergence time because the Sampler lacks guidance on "which physical direction to explore."
- **Token Efficiency**: DoLQ consumes fewer tokens than LLM-SR on 4D systems because it maintains compact equations, whereas LLM-SR's redundant terms lead to increasingly long prompts.
- **CDIMA as a Watershed**: For non-polynomial saturation terms like $\frac{4 x_0 x_1}{x_0^2 + 1}$, pure evolution (LASR) and LLM-SR fail. Only DoLQ recovers it by identifying semantic clues like "saturation" through the Scientist.
- **Hybrid Optimization Prevents False Negatives**: Correct skeletons are often rejected by BFGS alone due to local minima; DE provides the necessary backbone for structural discovery.

## Highlights & Insights
- **"Reviewer Agent" as a Pattern for LLM4Science**: Using LLMs as both generators and reviewers, but isolating them with independent prompts and contexts, reduces LLM overconfidence/hallucination via adversarial design.
- **Asymmetric Rules**: Qualitative = Veto power for immediate removal of nonsense; Quantitative = Delayed decision for terms with no current contribution, preventing accidental deletion of useful terms.
- **Probabilistic Blacklist**: Reviving removed skeletons periodically is a rare design in SR that helps counter "early-stage bias" in LLMs.
- **Execution-Ready Expressions**: Outputting Python code directly instead of LaTeX minimizes parser errors and is a highly practical engineering choice.

## Limitations & Future Work
- **Noise Sensitivity**: Using finite differences for $\dot{x}$ makes the Residual MSE vulnerable to noise; future work could use integral-based estimations.
- **Reasoning Fixation**: The Scientist may prematurely anchor on a specific physical interpretation, leading to sub-optimal local convergence.
- **Reliance on LLM Priors**: If descriptions are vague or the system is rare (e.g., novel materials), the qualitative review's reliability may degrade.
- **PDE Generalization**: Extending to PDEs is difficult due to the lack of universal solvers compared to ODEs.

## Related Work & Insights
- **vs LLM-SR**: Both use LLMs as Samplers, but LLM-SR prioritizes MSE and tends to stack terms, whereas DoLQ enforces structural compactness and excels in extrapolation.
- **vs LASR**: Evolution handles polynomials well but lacks the semantic capability to explore non-polynomial forms like sine/cosine or rational saturations found in CDIMA.
- **Insight**: The "Reviewer Agent" framework is transferable to other scientific tasks like theorem proving step validation, semantic linting in code synthesis, or fact-checking in automated literature reviews.

## Rating
- Novelty: ⭐⭐⭐⭐ High value in the dual-track review incremental design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong comparison across 8 systems and multiple baselines, though OOD and heavy noise tests are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline, well-structured examples, and logical flow.
- Value: ⭐⭐⭐⭐ Provides a reusable template for LLM-driven scientific discovery with significant performance gains in complex systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Resolution Diagnostics for Paired LLM Evaluation](resolution_diagnostics_for_paired_llm_evaluation.md)
- [\[ICML 2026\] REAL: Integrating Regression-Aware Rewards into RL for Fine-Grained Human-Centric LLM Evaluation](real_regression-aware_reinforcement_learning_for_llm-as-a-judge.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)
- [\[ICLR 2026\] BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation](../../ICLR2026/llm_evaluation/biasscope_towards_automated_detection_of_bias_in_llm-as-a-judge_evaluation.md)
- [\[ACL 2026\] Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference](../../ACL2026/llm_evaluation/statistically_reliable_llm-based_ranking_evaluation_via_prediction-powered_infer.md)

</div>

<!-- RELATED:END -->
