---
title: >-
  [Paper Note] From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates
description: >-
  [ICML 2026][LLM Reasoning][Polynomial Inequality] NSPI allows LLMs to propose approximate Sum-of-Squares (SOS) structure conjectures, which are then refined into rigorous SOS decompositions with rational coefficients via…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Polynomial Inequality"
  - "SOS Decomposition"
  - "Lean Formalization"
  - "Neuro-symbolic"
  - "GRPO"
date: 2026-05-08
content_hash: 7ce813f69bcaf77e
---

# From LLM-Generated Conjectures to Lean Formalizations: Automated Polynomial Inequality Proving via Sum-of-Squares Certificates

**Conference**: ICML 2026  
**arXiv**: [2605.15445](https://arxiv.org/abs/2605.15445)  
**Code**: Not yet public  
**Area**: LLM Reasoning / Neuro-symbolic / Automated Theorem Proving  
**Keywords**: Polynomial Inequality, SOS Decomposition, Lean Formalization, Neuro-symbolic, GRPO

## TL;DR
NSPI allows LLMs to propose approximate Sum-of-Squares (SOS) structure conjectures, which are then refined into rigorous SOS decompositions with rational coefficients via Gauss–Newton iteration and rational recovery. Finally, automated machine verification is performed using Lean's `linear_combination` + `positivity` tactics, extending inequality proving scalability up to 10 variables.

## Background & Motivation

**Background**: Polynomial inequalities are fundamental tools in optimization, control, and combinatorics. Proving $f(x) \ge 0$ primarily follows two routes: purely symbolic routes (Maple, Z3, SOS+SDP) and recently emerging LLM formal proof routes (DeepSeek-Prover-V2, Goedel-Prover, Kimina-Prover) that directly generate Lean/Isabelle tactics.

**Limitations of Prior Work**: Purely symbolic methods perform adequately in low dimensions (3-4 variables) but face combinatorial explosion as the number of variables increases—SDP matrix dimensions grow by $\binom{n+d}{d}$, and Maple solves only 1.7% at 10 variables. LLM-based methods rely on formal training corpora, but Lean proofs for high-dimensional inequalities have almost no public data; DS-Prover-v2 drops to 0% for 5+ variables.

**Key Challenge**: Symbolic methods are **precise but not scalable**; LLMs are **scalable but cannot prove**—SDP outputs numerical matrices with floating-point errors that Lean does not accept, while tactics written by LLMs are difficult to execute in high-dimensional spaces. Neuro-symbolic works like AlphaGeometry/AIPS only use LLMs as search heuristics and do not let LLMs directly produce symbolic objects.

**Goal**: To upgrade the LLM to a **symbolic conjecture generator**—it directly outputs approximate SOS structures, while the symbolic end is responsible for refining them into exact certificates verifiable by Lean.

**Key Insight**: It is observed that the "structure" of an SOS certificate (which monomials are included in each squared term) is easier to guess than the "coefficients," and coefficient refinement is guaranteed by theorems (Peyrl–Parrilo 2008 rational recovery theorem). Thus, the task is split into three parts: "LLM guesses structure + Symbolic refines coefficients + Lean verifies."

**Core Idea**: LLM guesses approximate SOS $\rightarrow$ Gauss–Newton iteration + Rational recovery yields an exact rational Gram matrix $\rightarrow$ Lean `linear_combination + positivity` for one-click verification, providing an end-to-end pipeline from heuristic discovery to machine-verifiable proof.

## Method

### Overall Architecture
NSPI is a three-stage pipeline: (1) **Neural Conjecture**: The LLM receives a non-negative polynomial $f(x)$ and outputs an approximate SOS representation $\hat f(x) = \sum_i \hat f_i(x)^2$, ranked by numerical error; (2) **Symbolic Correction**: Top-K conjectures undergo Gauss–Newton numerical refinement followed by rational recovery to obtain an exact rational Gram matrix; (3) **Formal Verification**: A predefined Lean template assembles the SOS certificate into a complete proof, calling `linear_combination` to verify the equality and `positivity` to verify non-negativity. Compared to pure SOS-SDP, this pipeline leverages LLM structural priors to bypass combinatorial explosion; compared to pure LLM proofs, the symbolic end provides precision guarantees.

### Key Designs

1.  **Dual-track SOS Data Synthesis + Two-stage Training**:
    - **Function**: Prepares millions of training pairs for the SOS structure conjecturer and enables the LLM to learn to "output a reasonable SOS skeleton given $f$" through SFT cold-start + GRPO curriculum RL.
    - **Mechanism**: Randomly sampling $f_i(x)$ then summing squares produces non-integer coefficients and coefficient explosion, so the process is reversed from a PSD Gram matrix $\widetilde G$. **Computation-driven**: Either apply spectral shifting to a random symmetric integer matrix $G$ as $\widetilde G = G - \lfloor \lambda_{\min} \rfloor I \succeq 0$, write as a factor form $\widetilde G = L^\top D L$, or use LMI optimization $\max \lambda$ s.t. $G - \lambda I \succeq 0$ followed by integerization. **Structure-driven**: Use diagonally dominant (dd) or scaled dd matrices, which are naturally PSD by Gershgorin disk theorem, parameterized as $\widetilde G = \sum_i \eta_i u_i u_i^\top$ where $u_i$ has at most two non-zero $\pm 1$ elements. Training stage 1 (SFT) cold-starts on 1M synthetic pairs; stage 2 organizes unsolved samples into a curriculum for GRPO, where Reward = Accuracy $R_{\text{Accuracy}} = 1/(1 + \alpha \|f - \hat f\|_2)$ + Format reward + Algebraic structural consistency penalty (encouraging non-zero monomial set matches).
    - **Design Motivation**: Writing rational SOS directly is uncontrollable for LLMs, but guessing approximate structures is easier. Structure-driven data ensures integer coefficients and monomial sets are controlled, avoiding degradation where the model "cannot remember large coefficients."

2.  **Gauss–Newton Refinement + Dual-regime Rational Recovery**:
    - **Function**: Refines approximate SOS from LLMs into exact rational Gram matrices for Lean.
    - **Mechanism**: First, extract the monomial basis $\mathbf v(x)$ and initial floating-point Gram matrix $\mathbf G$ from $\hat f(x)$, using Cholesky $\mathbf G \approx L L^\top$ to get $\hat f(\mathbf x) \approx \sum_i (\sum_\alpha c_{i,\alpha} \mathbf x^\alpha)^2$. Then, Gauss–Newton iteration solves for perturbations $\Delta c_{i,\alpha}$ such that the backward error $\theta = \|\hat f(\mathbf x) - \mathbf v(x)^\top \mathbf G \mathbf v(x)\|$ falls below threshold $\tau$. **Dual-regime rational recovery**: For interior cases (Gram matrix strictly inside PSD cone), project the matrix onto the SOS equality constraint affine subspace and rationalize per Peyrl–Parrilo. For boundary cases (numerical rank deficiency), apply truncated $LDL^\top$ decomposition + simultaneous Diophantine approximation to recover rational vectors based on rank structure rather than direct global rationalization.
    - **Design Motivation**: SDP solvers only provide numerical solutions where rounding breaks PSD; GN allows numerical solutions to converge near machine precision, followed by a theoretically guaranteed rational recovery to avoid the fragility of repeated SDP+rounding.

3.  **Lean Formalization Template + `llm_ineq` Tactic**:
    - **Function**: Compiles exact SOS certificates into Lean 4 proofs for kernel checking.
    - **Mechanism**: Authors developed a Lean tactic `llm_ineq`. Given a target polynomial and a rational SOS certificate, it automatically splits into two obligations: (a) Polynomial equality $p = \sum_i k_i q_i^2$ verified by the `linear_combination` tactic after normal form expansion; (b) SOS expression non-negativity proven by the `positivity` tactic applying `sq_nonneg`, `mul_nonneg`, `add_nonneg`, etc. The template is universal; as long as the symbolic end provides a correct SOS, the Lean end closes the loop.
    - **Design Motivation**: Restricting Lean complexity to two standard tactics avoids dependence on LLM-written tactics—this ensures that in high-dimensional scenarios, if the SOS is correct, Lean will pass, shifting the reliability bottleneck from "LLM's Lean capability" to "SOS structure accuracy."

### Loss & Training
The SFT phase uses standard next-token loss. The GRPO phase rewards are $R = R_{\text{Accuracy}} + R_{\text{Format}} - R_{\text{Struct-Penalty}}$. Structural penalties include soft penalties (symmetric difference of non-zero monomial sets) and hard penalties (for higher-order terms unseen in training). The curriculum buckets data by the number of iterations required by symbolic solvers, fed from easy to hard.

## Key Experimental Results

### Main Results
Pass rates within 1 hour on PolyIneqBench (522 challenging inequalities, $n=3$ to $n=10$):

| Variables | Maple | Z3 | DS-Prover-v2 | Goedel-v2 | Kimina | Gemini-3-Pro | **NSPI (Ours)** |
|-----------|-------|----|--------------|-----------|--------|--------------|-----------------|
| n=3       | 97.6% | 97.6% | 42.9%     | 20.2%     | 36.9%  | 22.6%        | (See Paper)     |
| n=4       | 39.0% | 32.5% | 2.6%      | 5.2%      | 5.2%   | 24.7%        | —               |
| n=5       | 26.7% | 23.3% | 0%        | 0%        | 0%     | 36.7%        | —               |
| n=7       | 6.7%  | 1.7%  | 0%        | 0%        | 1.7%   | 15.0%        | —               |
| n=10      | 1.7%  | 0%    | 0%        | 0%        | 0%     | 6.7%         | —               |

Key Trends: Purely symbolic methods (Maple/Z3) excel at $n=3$ but drop to 0–1.7% at $n=10$. Pure LLM provers fail at $n \ge 5$. General models like Gemini-3-Pro are more stable in high dimensions, indicating the utility of heuristics. NSPI, as a neuro-symbolic method, pushes provability to the 10-variable level.

### Ablation Study
| Configuration | Impact | Description |
|---------------|--------|-------------|
| SFT only (no GRPO) | Severe degradation (n≥6) | Cold-start only replicates training distribution; fails to generalize to hard samples. |
| No structural penalty | Monomial mismatch↑, refinement failure↑ | LLM outputs SOS with unseen monomials, reducing GN convergence. |
| No Gauss–Newton | Accuracy in boundary cases↓ | Numerical error exceeds Peyrl–Parrilo $\delta$ threshold; rationalization fails PSD. |

### Key Findings
- If the SOS structure is wrong, symbolic refinement cannot save it; conversely, if the structure is correct, refinement + rational recovery almost always succeeds—hence the focus on structural consistency in rewards.
- The dual-regime rational recovery (interior vs. boundary) is essential: high-dimensional polynomials often lie on the PSD cone boundary (rank significantly smaller than matrix dimension); interior projection alone loses rank structure.
- The division of roles is critical: the LLM does not attempt to write Lean proofs directly (the root cause of LLM-prover failure in high dimensions), but focuses on symbolic intermediate conjectures.

## Highlights & Insights
- **Upgrading LLM to "Symbolic Object Generator"**: Unlike AlphaGeometry/AIPS which treat neural networks as search guides, this work proves LLMs can directly produce symbolic intermediate representations (SOS structures), involving the neural end in certificate construction rather than just retrieval.
- **Reverse Synthesis via Gram Matrices**: Instead of squaring and expanding (which explodes coefficients), generating from structured PSD matrices allows precise control over monomial sets. This "reverse dataset construction" is transferable to any generation task with algebraic constraints.
- **Shifting the Reliability Bottleneck**: Reliability is moved from "LLM's Lean skill" to "SOS structure accuracy" + "rational recovery validity." The latter is backed by Peyrl–Parrilo theorems, while the former is improved by data/RL—a paradigm of pushing uncertainty into theoretically guaranteed sub-modules.

## Limitations & Future Work
- The framework currently covers **unconstrained** polynomial inequalities; constrained cases (Positivstellensatz, Putinar form) are not yet handled.
- The LLM requires task-specific training (1M synthetic data + GRPO); it is not a plug-and-play general prover. Switching to constrained SOS or matrix inequalities requires a new data pipeline.
- Coverage is limited to the SOS-provable subset—polynomials that are non-negative but not SOS (e.g., Motzkin) remain out of reach; extension to Schmüdgen/Putinar multipliers is needed.
- Evaluation only reaches $n=10$ with a 1-hour budget; evidence for truly large-scale engineering inequalities (dozens of variables, fractions/transcendental functions) is still needed.

## Related Work & Insights
- **vs AIPS / AlphaGeometry**: Both are neuro-symbolic, but AIPS/AG use neural networks for search heuristics or auxiliary constructions, while the symbolic end handles main reasoning. NSPI has the LLM generate the symbolic certificate structure directly.
- **vs DeepSeek-Prover-V2 / Kimina-Prover**: Pure LLM formalization provers write Lean tactics end-to-end. These are strong for competition problems but fail high-dimensional inequalities due to data scarcity. NSPI bypasses this via "LLM writes symbols, Lean uses templates."
- **vs Classical SOS-SDP (Parrilo, Lasserre)**: Pure SDP is theoretically complete but explodes with $\binom{n+d}{d}$. NSPI uses LLM to provide a "likely feasible" sparse SOS structure, reducing the SDP to a low-dimensional sub-problem, effectively pruning the search space with neural priors.

## Rating
- Novelty: ⭐⭐⭐⭐ Placed LLM in the role of symbolic object generator with dual-regime recovery.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered $n=3$ to $n=10$, but lacks constrained scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear layering of neural, symbolic, and formal stages.
- Value: ⭐⭐⭐⭐ Provides a truly scalable engineering solution for high-dimensional inequality proving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Automated Formal Proofs of Combinatorial Identities via Wilf–Zeilberger Guidance and LLMs](automated_formal_proofs_of_combinatorial_identities_via_wilf-zeilberger_guidance.md)
- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[ICML 2026\] R2-Router: A New Paradigm for LLM Routing with Reasoning](r2-router_a_new_paradigm_for_llm_routing_with_reasoning.md)
- [\[ICML 2026\] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning](beyond_two-stage_training_cooperative_sft_and_rl_for_llm_reasoning.md)
- [\[ICML 2026\] TRACE: Evaluating LLM CoT Reasoning Process Quality with the Toulmin Argumentation Model](trace_toulmin-based_reasoning_assessment_through_constructive_elements_for_llm_c.md)

</div>

<!-- RELATED:END -->
