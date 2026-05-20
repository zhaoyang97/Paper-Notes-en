---
title: >-
  [Paper Note] Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation
description: >-
  [ICML 2026][Scientific Computing][Symbolic Regression] DoLQ inserts a "Scientist Agent" into the search loop of LLM symbolic regression, performing both qualitative (physical plausibility) and quantitative (ablation MSE…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Symbolic Regression"
  - "Ordinary Differential Equations"
  - "LLM Agent"
  - "Physical Interpretability"
  - "Hypothesis Search"
date: 2026-05-08
content_hash: 77c2980ab80c97c6
---

# Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation

**Conference**: ICML 2026  
**arXiv**: [2605.07323](https://arxiv.org/abs/2605.07323)  
**Code**: https://github.com/Bon99yun/DoLQ  
**Area**: Scientific Discovery / Symbolic Regression / LLM Multi-Agent  
**Keywords**: Symbolic Regression, Ordinary Differential Equations, LLM Agent, Physical Interpretability, Hypothesis Search

## TL;DR
DoLQ inserts a "Scientist Agent" into the search loop of LLM symbolic regression, performing both qualitative (physical plausibility) and quantitative (ablation MSE contribution) evaluations on candidates. This approach forces LLM-SR, which typically produces "low-error but bloated and physically nonsensical" equations, to converge to equations that are both numerically accurate and structurally compact.

## Background & Motivation
**Background**: Inferring ordinary differential equations (ODEs) from observational data is a core problem in scientific machine learning. Early methods like SINDy relied on predefined basis function libraries and sparse regression; subsequent symbolic regression (SR) approaches used evolutionary algorithms or Transformers (ODEformer) for automated search; the latest generation employs LLM-based methods such as LLM-SR and LASR, leveraging LLMs as candidate generators to dramatically reduce the search space via scientific priors.

**Limitations of Prior Work**: All existing methods evaluate candidate equations solely on numerical metrics—MSE and complexity. Figure 1 in the paper provides a key counterexample: two equations with nearly identical MSEs can behave entirely differently in extrapolation and physical meaning. LLM-SR often outputs "garbage bag" equations with over ten terms, where the true terms are diluted by physically meaningless ones, causing parameter optimization interference.

**Key Challenge**: Purely numerical evaluation only indicates "how well the current fit is," but not "whether a term corresponds to a real physical mechanism," the latter being crucial for stable extrapolation in ID-Ext/OOD regimes.

**Goal**: (1) Explicitly inject physical plausibility into the search loop; (2) Actively prune terms that "appear useful but are actually overfitting"; (3) Reliably recover structure in multi-dimensional ODEs (even 4D Glider).

**Key Insight**: LLMs possess both numerical reasoning and domain knowledge. Rather than using them solely as "nominators," an independent LLM can serve as a "reviewer," scoring candidates from both semantic and numerical perspectives, with reviewer feedback injected into the next round of prompts for the nominator.

**Core Idea**: Dual-track qualitative + quantitative review → double veto: semantically unreasonable terms are immediately removed, numerically unhelpful terms are delayed for removal, guiding the search toward equations that are both physically plausible and numerically optimal.

## Method

### Overall Architecture
DoLQ is a three-agent iterative closed loop, revolving around "nomination → optimization → review":

1. **Sampler Agent**: Consumes system description $\mathcal{T}$ and Scientist feedback, outputs executable Python code snippets for candidate terms (e.g., `params[0] * np.sin(x[1])`) and a natural language physical explanation for each term.
2. **Parameter Optimizer**: Instantiates symbolic terms as differentiable skeleton functions $f_j(t, \boldsymbol{x}; \boldsymbol{\theta})$, fitting parameters $\boldsymbol{\theta}$ using a hybrid optimizer.
3. **Scientist Agent**: Performs quantitative (ablation MSE) and qualitative (semantic alignment) reviews on optimized equations, producing keep/hold/remove decisions and cumulative insights, which are fed back to the Sampler.

The outer loop runs for 100 iterations; in each round, the Sampler generates multiple candidate hypotheses in parallel, with the global optimum ("Global Best") continuously updated.

### Key Designs

1. **Scientist Agent's Dual-Track Review**:

    - **Function**: Determines for each candidate term whether to keep or remove.
    - **Quantitative Track**: Performs ablation on each term—sets its coefficient to zero and recalculates residual MSE. If error increases significantly → good; remains nearly unchanged → neutral; decreases → bad (indicates the term is "gaming the score," introducing overfitting). For example, in Figure 2, removing `np.sin(x1)` increases MSE by 78.2% (good), while removing `x1**4` decreases MSE (bad).
    - **Qualitative Track**: The LLM reads the system description $\mathcal{T}$, the symbolic form of the term, and the Sampler's physical explanation, scoring as good/neutral/bad based on whether the term corresponds to a physical mechanism mentioned in the description (e.g., "air resistance" corresponding to $-cx^2$ is good).
    - **Fusion**: Semantic bad is immediately removed; only terms rated good by both tracks are kept; all others are held; terms held for two consecutive rounds are automatically removed. Removed skeletons (with parameters replaced by placeholders) are blacklisted to prevent regeneration, but a probabilistic forgetting mechanism periodically clears the blacklist to avoid false negatives.
    - **Design Motivation**: Solely relying on MSE retains many spurious terms that merely fit noise; relying only on semantics misses "unfamiliar but necessary" terms. The independent dual-track, with mutual veto when necessary, distinguishes this work from LLM-SR.

2. **Sampler's Structured Prompt + Parallel Hypotheses**:

    - **Function**: Ensures the LLM proposes "executable + explainable" candidate terms, not just symbolic expressions.
    - **Mechanism**: The prompt is divided into three parts—(a) task definition + system description $\mathcal{T}$; (b) Scientist feedback, including cumulative insights, term-by-term review results, and the blacklist of removed skeletons; (c) output format constraints, requiring each term to be written as Python code (e.g., `params[0] * np.cos(x[0])`) plus a physical justification. Multiple hypotheses are generated per round, each providing a set of independent terms for each dimension, facilitating parallel evaluation and ablation without cross-contamination.
    - **Design Motivation**: (1) Executable format eliminates the need for symbolic parsing by the Parameter Optimizer; (2) Mandatory justification makes the LLM's "implicit physical knowledge" explicit for Scientist review; (3) Parallel hypotheses and blacklisting prevent the search from getting stuck on a single path.

3. **DE + BFGS Hybrid Parameter Optimization**:

    - **Function**: Recovers the equation skeleton from local to global optimum.
    - **Mechanism**: Differential Evolution (DE) is first used for global search in parameter space to find a feasible region; BFGS then performs local refinement. Three strategies (BFGS only / DE only / hybrid) are run in parallel, with the lowest MSE selected. The optimization target is residual MSE $\sum_i (\dot{x}_j(t_i) - f_j(t_i, \boldsymbol{x}; \boldsymbol{\theta}))^2$ rather than integral MSE, as the latter can be contaminated across dimensions by early errors in multi-dimensional systems.
    - **Design Motivation**: BFGS is highly sensitive to initialization; structurally correct skeletons are often wrongly rejected due to poor initial values, misleading the Scientist to remove good structures—a catastrophic false negative. DE serves as a safety net to avoid "correct skeleton but missed parameters" (see Figure 8: BFGS alone fails, hybrid succeeds).

### Loss & Training
This is a training-free framework with no backpropagation. The core "training signal" is the Scientist Agent's per-term keep/hold/remove decisions, which rewrite the prompt to alter the Sampler's distribution in the next round. Parameter optimization targets residual MSE (Equation 2); final performance is measured by normalized MSE (NMSE) and success rate. Each system is run three times, taking the best result, with a fixed search budget of 100 rounds for fair comparison with baselines.

## Key Experimental Results

### Main Results
Gemini 2.5 Flash Lite is used as the backbone, evaluated on 8 systems from ODEbench. Average NMSE (Residual / Integral) across dimensions:

| System | Metric | ICSR | LASR | LLM-SR | EDL | **DoLQ** |
|--------|--------|------|------|--------|-----|----------|
| SIR(2D) ID | Residual | 2.8e-8 | 1.8e-8 | 1.7e-8 | 18.6 | **1.7e-8** |
| CDIMA(2D) ID | Integral | 3.8e-4 | 2.9e-1 | 3.1e-3 | 1.3e5 | **2.4e-8** |
| Glider(4D) ID | Residual | 2.6e-2 | 2.5e-2 | 9.95e-7 | 1.5e4 | **1.2e-6** |
| CDIMA(2D) ID-Ext | Integral | 2.6e-4 | 4.9e1 | 9.4e-3 | NaN | **1.2e-7** |

CDIMA is a nonlinear saturation system; all baselines collapse on ID-Ext (LASR rises to 48.5), but DoLQ remains stable at the $1e-7$ level, a difference of 5–6 orders of magnitude—demonstrating that correct structure is essential for stable extrapolation. For SIR, since the equation form can be directly inferred from the description, all methods achieve very low errors, making them indistinguishable.

Aggregate success rates across 8 systems (Figure 4): For NMSE test (integral NMSE < $1e-3$) and Term test (structure matches ground truth), DoLQ achieves SOTA.

### Ablation Study

| Configuration | Glider(2D) Convergence Iteration | Description |
|---------------|-------------------------------|-------------|
| Full DoLQ | 27th round | Complete Scientist Agent feedback |
| w/o Scientist | 62nd round | Only MSE fed to Sampler, no qualitative review |
| BFGS only | Failed | Cannot find parameters even with correct structure |
| DE only | Unstable | Global search but insufficient refinement |
| **DE + BFGS** | Stable convergence | Hybrid strategy in Figure 8 |

Figure 7 shows that ground truth terms are rated as keep by the Scientist far more frequently than other terms; even if occasionally misjudged as remove, the probabilistic forgetting mechanism allows them to revive in later rounds, ensuring eventual convergence.

### Key Findings
- **Qualitative Review as Accelerator**: Removing the Scientist more than doubles convergence time, as the Sampler lacks guidance on "which physical direction to explore," leading to repeated trial and error with obviously unreasonable terms.
- **Reverse Token Efficiency**: Figure 5 shows LLM-SR consumes far more tokens than DoLQ on Glider(4D), as redundant terms lengthen the prompt, making DoLQ more efficient.
- **CDIMA as Watershed**: For non-polynomial saturation terms like $\frac{4 x_0 x_1}{x_0^2 + 1}$, pure evolutionary (LASR) and pure LLM-SR both fail; only DoLQ, with the Scientist recognizing semantic cues like "should have saturation," can reliably recover the structure.
- **Hybrid Optimizer Avoids False Negatives**: BFGS alone often gets stuck in local minima even with correct skeletons, causing the Scientist to mistakenly remove good structures; DE serves as the hidden backbone for structure discovery.

## Highlights & Insights
- **"Reviewer Agent" as a General Pattern for LLM4Science**: Using LLMs as both generator and reviewer, with independent prompts and contexts for each role, this "self-adversarial" design reduces LLM overconfidence. It can be directly transferred to other scientific search tasks (e.g., protein design, chemical reaction pathways).
- **Qualitative = Immediate Veto, Quantitative = Delayed Decision**: The asymmetric fusion rule is ingenious—semantically incorrect terms are "killed instantly" as they are not worth further exploration; numerically unhelpful terms are given two rounds of hold before removal, avoiding premature deletion of potentially useful terms.
- **Probabilistic Forgetting Blacklist**: Periodically reviving deleted skeletons is a rare design in SR, effectively countering LLM "early bias."
- **Executable Symbolic Expressions**: Having the LLM output directly as `params[0] * np.sin(x[0])` instead of LaTeX avoids parser errors—a highly practical engineering trick.

## Limitations & Future Work
- **Sensitivity to Numerical Differentiation Noise**: Using finite differences to estimate $\dot{x}$, residual MSE is severely affected by observation noise; the authors plan to use integral estimates in the future.
- **Reasoning Fixation**: The Scientist occasionally anchors prematurely to a physical explanation (e.g., insisting certain dynamics are drag-dominated), causing the search to converge early to suboptimal structures.
- **Dependence on LLM Physical Priors**: When descriptions are vague or systems are rare (e.g., novel material systems), the Scientist's qualitative review reliability drops, reducing the method to "MSE-only."
- **Difficulty Generalizing to PDEs**: The method relies on mature ODE solvers; the lack of general PDE solvers is a direct obstacle.
- **Fixed Search Budget of 100 Rounds**: Wasteful for simple systems (SIR), possibly insufficient for complex (higher-dimensional) systems; adaptive budgeting is a natural extension.

## Related Work & Insights
- **vs LLM-SR (Shojaee 2025)**: Both use LLMs as Samplers, but LLM-SR relies only on MSE, tending to accumulate terms; DoLQ, with the Scientist, controls term count and is stable in extrapolation (5 orders of magnitude difference on CDIMA).
- **vs LASR (Grayeli 2024)**: Evolutionary algorithm + concept library, performs well on polynomial systems but lacks exploration of non-polynomial forms (sin/cos/rational), structurally unable to recover CDIMA.
- **vs ODEformer (d'Ascoli 2024)**: Pure Transformer end-to-end generation, does not utilize semantic descriptions, and cannot control physically meaningful terms; DoLQ explicitly leverages $\mathcal{T}$.
- **Insights**: The "second LLM as reviewer" framework can be extended to (1) intermediate step verification in mathematical theorem proving; (2) semantic-level linting in program synthesis; (3) paragraph-level fact-checking in automated review writing.

## Rating
- Novelty: ⭐⭐⭐⭐ Adding "qualitative + quantitative dual-track review" to the LLM-SR framework is a clear and crucial incremental innovation, though the "dual-agent review" pattern is becoming popular in LLM4Science.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 ODE systems, 4 SOTA baselines, ID + ID-Ext dual regimes, thorough ablation; the only shortcoming is the lack of OOD evaluation and extensive real-noise experiments.
- Writing Quality: ⭐⭐⭐⭐ Figure 1's counterexample and Figure 2's pipeline are well-coordinated, method details are clearly explained, with formulas and examples interleaved.
- Value: ⭐⭐⭐⭐ Provides a reusable "reviewer agent" template for LLM-driven scientific discovery; the 5-order-of-magnitude gap on CDIMA is highly valuable for real scientific applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](../../ICLR2026/llm_evaluation/llm_unlearning_with_llm_beliefs.md)
- [\[ICML 2026\] CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing](crispedit_low-curvature_projections_for_scalable_non-destructive_llm_editing.md)
- [\[ICLR 2026\] Multi-LLM Adaptive Conformal Inference for Reliable LLM Responses](../../ICLR2026/llm_evaluation/multi-llm_adaptive_conformal_inference_for_reliable_llm_responses.md)
- [\[ICML 2026\] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)
- [\[ICLR 2026\] Preference Leakage: A Contamination Problem in LLM-as-a-judge](../../ICLR2026/llm_evaluation/preference_leakage_a_contamination_problem_in_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
