---
title: >-
  [Paper Note] Cyclic Counterfactuals under Shift–Scale Interventions
description: >-
  [NeurIPS 2025][Causal Inference][cyclic causal models] This paper establishes a theoretical framework for counterfactual reasoning under shift–scale soft interventions in cyclic (non-DAG) structural causal models (SCMs).…
tags:
  - "NeurIPS 2025"
  - "Causal Inference"
  - "cyclic causal models"
  - "counterfactual reasoning"
  - "soft interventions"
  - "contraction mappings"
date: 2026-05-08
content_hash: 1f0b90f3df697d6e
---

# Cyclic Counterfactuals under Shift–Scale Interventions

**Conference**: NeurIPS 2025
**arXiv**: [2510.25005](https://arxiv.org/abs/2510.25005)
**Code**: None
**Area**: Causal Inference
**Keywords**: causal inference, cyclic causal models, counterfactual reasoning, soft interventions, contraction mappings

## TL;DR

This paper establishes a theoretical framework for counterfactual reasoning under shift–scale soft interventions in cyclic (non-DAG) structural causal models (SCMs). It proves that a global contraction condition guarantees unique solvability of cyclic SCMs and derives sub-Gaussian concentration inequalities for counterfactual distributions.

## Background & Motivation

**Background**: Counterfactual reasoning is one of the central problems in causal inference. The vast majority of counterfactual frameworks (e.g., Pearl's do-calculus, twin networks) assume that the causal structure is a directed acyclic graph (DAG), i.e., that there are no feedback loops among variables.

**Limitations of Prior Work**: However, feedback loops are ubiquitous in real-world systems—positive/negative feedback in gene regulatory networks, the mutual influence of consumption and income in macroeconomic models, and predator–prey dynamics in ecosystems. The causal structure of these systems contains cycles, violating the DAG assumption. In cyclic SCMs, the structural equations may have no unique solution (or no solution at all), making the very definition of counterfactual inference problematic. Furthermore, existing theory primarily considers hard interventions ($do$-interventions), which fix a variable to a constant value, whereas real-world policies are often soft interventions—such as "increase every individual's drug dosage by 20%" or "reduce every student's class size by 5"—which depend on the individual's original value and cannot be expressed as $do(X=x)$.

**Key Challenge**: (1) Uniqueness of counterfactual reasoning in cyclic systems lacks theoretical guarantees; (2) shift–scale soft interventions are more expressive than hard interventions, yet their theoretical foundations remain underdeveloped.

**Goal**: Under what conditions does the counterfactual distribution under shift–scale interventions in cyclic causal models exist and is unique? Do such interventions possess algebraic stability (composability)? How concentrated is the distribution of counterfactual outcomes?

**Key Insight**: The paper draws on the contraction mapping principle (Banach fixed-point theorem) from dynamical systems theory to provide sufficient conditions for the unique solvability of cyclic SCMs, and then proves that shift–scale interventions preserve contractivity.

**Core Idea**: A global contraction condition is employed to uniformly handle counterfactual reasoning under soft interventions in cyclic SCMs, extending unique solvability from DAGs to cyclic graphs satisfying the contraction condition.

## Method

### Overall Architecture

The theoretical architecture of the paper is organized into four layers: (1) proving that contractive SCMs are simple SCMs (uniquely solvable for all subsets of variables); (2) proving that bounded shift–scale interventions preserve contractivity; (3) proving closure under composition of such interventions; (4) deriving concentration inequalities for counterfactual distributions under Gaussian noise and Lipschitz regularity conditions.

### Key Designs

1. **Global Contraction Condition → Unique Solvability (Theorem 1)**:

    - Function: Provides a verifiable sufficient condition for the unique solvability of cyclic SCMs.
    - Mechanism: If the structural equations $f: \mathcal{X} \times \mathcal{E} \to \mathcal{X}$ of an SCM satisfy global $\kappa$-contraction ($\kappa < 1$), i.e., $\|f(x,e) - f(y,e)\|_p \leq \kappa \|x - y\|_p$, then for any subset of variables $\mathcal{O}$, a unique fixed-point solution exists. The proof applies the Banach fixed-point theorem: for fixed exogenous variables $e$ and non-$\mathcal{O}$ variables, $f_\mathcal{O}$ is a $\kappa$-contraction on the complete metric space $\mathcal{X}_\mathcal{O}$, hence a unique fixed point exists. Pointwise convergence of Picard iteration guarantees the measurability of the solution map.
    - Design Motivation: The closure result for simple SCMs in Bongers et al. (2021) assumes simplicity without providing a sufficient condition for it. This paper fills that critical gap.

2. **Shift–Scale Interventions Preserve Contractivity (Theorem 2)**:

    - Function: Proves that the twin SCM after intervention remains uniquely solvable, ensuring that the counterfactual distribution is well-defined.
    - Mechanism: A shift–scale intervention replaces $X_j \leftarrow a_j f_j(x,e) + b_j$; when $|a_j| \leq 1$, the intervened map $\tilde{g}$ remains $\kappa$-contractive. The key step is that the diagonal scaling matrix $D = \text{diag}(a_j)$ satisfies $\|Du\|_p \leq \|u\|_p$ (since $|a_j| \leq 1$), so $\|\tilde{g}(u,e) - \tilde{g}(v,e)\|_p \leq \|f(u,e) - f(v,e)\|_p \leq \kappa\|u-v\|_p$.
    - Design Motivation: Ensuring that the intervened model remains well-posed is a necessary prerequisite for establishing a theory of counterfactual inference.

3. **Compositional Closure and Concentration Inequalities (Proposition 1 & 2)**:

    - Function: Proves that multiple shift–scale interventions can be composed into a single equivalent intervention, and that the counterfactual distribution is concentrated around its mean.
    - Mechanism: (Prop. 1) The composition of multiple shift–scale interventions is equivalent to a single intervention with $a_j^{\text{comp}} = \prod_r a_j^{(r)}$ and a corresponding affine shift; since $|a_j^{(r)}| \leq 1$, we have $|a_j^{\text{comp}}| \leq 1$, preserving contractivity. (Prop. 2) When the exogenous noise is Gaussian, the solution map $\Phi$ is $L = \frac{\sqrt{2}}{1-\kappa}$-Lipschitz, and by the Gaussian Lipschitz concentration inequality, $\mathbb{P}(h(\mathbf{X},\mathbf{X}') - \mathbb{E}[h] \geq t) \leq \exp\!\left(-\frac{t^2}{4(1-\kappa)^{-2}\sigma^2}\right)$.
    - Design Motivation: Compositional closure ensures algebraic stability in sequential intervention analysis; the concentration inequality provides a quantitative uncertainty bound for counterfactual outcomes.

### Loss & Training

This paper is purely theoretical; no training procedure is involved.

## Key Experimental Results

### Main Results

The paper illustrates the theory through a cyclic consumption–income economic model:

| Quantity | Observational Distribution | Post-Intervention Distribution | Change |
|---------|-----------|-----------|------|
| $\mathbb{E}[C]$ | 1.5625 | 2.024 | +29% |
| $\mathbb{E}[I]$ | 1.125 | 2.048 | +82% |
| $\text{Corr}(C,I)$ | 0.75 | 0.69 | −8% |
| Contraction constant $\kappa$ | 0.6403 | 0.5936 | Remains $<1$ |

The intervention applies a scale of $\alpha=0.8$ and a shift of $\beta=1.0$ to income $I$ (simulating a fiscal reform: dampening the feedback effect of consumption on income while providing a fixed income subsidy).

### Ablation Study

| Condition | Result | Note |
|------|------|------|
| $\|a\| \leq 1$ | Contractivity preserved | Guaranteed by theorem |
| $\|a\| > 1$ but $\kappa_{\max} < 1$ | Still solvable | Extended condition in Remark 1 |
| $\kappa_{\max} \geq 1$ | Uniqueness not guaranteed | Requires additional analysis |
| Gaussian noise | Sub-Gaussian concentration | Proposition 2 |
| Heavy-tailed noise | Only polynomial concentration | Not covered in this paper |

### Key Findings

- The spectral norm of the system matrix $A$ directly determines contractivity: $\|A\|_2 < 1$ is a sufficient condition.
- Shift–scale interventions preserve contractivity via diagonal scaling matrices, and the contraction constant does not increase.
- The concentration of the counterfactual distribution deteriorates sharply as $\kappa \to 1$ (concentration parameter $\propto (1-\kappa)^{-2}$).
- In linear cyclic models, the counterfactual response map is affine, yielding a closed-form solution.

## Highlights & Insights

- **Contraction mapping as a master key**: The application of the Banach fixed-point theorem to cyclic causal models is elegant—it reframes the "solvability" of feedback loops as a "contractivity" problem, with a single key unlocking three doors: uniqueness, measurability, and closure of the twin SCM. This framework extends naturally to any class of interventions that preserves contractivity.
- **Theoretical foundations for soft interventions**: Shift–scale interventions strictly generalize hard interventions (where $a=0, b=\xi$ is a special case), providing rigorous mathematical foundations for more flexible causal queries such as "increase everyone's dosage by 20%."
- **Practical utility of concentration inequalities**: The sub-Gaussian tail bound yields confidence intervals for counterfactual predictions, which are particularly valuable in high-stakes settings such as medical decision-making.

## Limitations & Future Work

- **Global contraction condition is restrictive**: Real-world systems may satisfy contractivity only locally, yet the paper requires it to hold globally. Theories based on local or piecewise contraction remain to be explored.
- **Scale factor restriction $|a_j| \leq 1$**: Amplifying interventions ($|a_j| > 1$) require verifying the additional condition $\kappa_{\max} < 1$. Stochastic policies and nonlinear interventions are not covered.
- **Gaussian noise assumption**: The concentration inequalities rely on Gaussian noise; only polynomial concentration can be obtained for heavy-tailed distributions.
- **Lack of empirical validation**: The paper presents only a two-variable linear economic toy model; no validation on real biological systems or gene regulatory networks is provided.
- **Causal discovery not addressed**: The causal graph is assumed to be known; learning cyclic causal structures from data is an independent open problem.

## Related Work & Insights

- **vs. Bongers et al. (2021)**: Bongers et al. establish closure results for simple SCMs (under do-interventions, marginalization, and twinning) but do not provide sufficient conditions for simplicity. This paper fills that gap using the contraction condition and further extends the framework to soft interventions.
- **vs. Rothenhäusler et al. (2015)**: They use shift interventions to learn cyclic causal graphs but do not address counterfactual inference or unique solvability.
- **vs. Lorch et al. (2024)**: They model shift–scale interventions in causal systems via stationary diffusion processes, but focus on continuous time and do not address counterfactuals.

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic treatment of counterfactual reasoning under soft interventions in cyclic SCMs.
- Experimental Thoroughness: ⭐⭐ — Only a two-variable toy model; lacks validation on real-world applications.
- Writing Quality: ⭐⭐⭐⭐ — Theorem proofs are clear and rigorous, though notation is heavy.
- Value: ⭐⭐⭐ — Fills a theoretical gap in counterfactual reasoning under soft interventions in cyclic causal models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Bi-Level Decision-Focused Causal Learning for Large-Scale Marketing Optimization](bi-level_decision-focused_causal_learning_for_large-scale_marketing_optimization.md)
- [\[CVPR 2026\] Retrieving Counterfactuals Improves Visual In-Context Learning](../../CVPR2026/causal_inference/retrieving_counterfactuals_improves_visual_in-context_learning.md)
- [\[ICLR 2026\] Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models](../../ICLR2026/causal_inference/distributional_equivalence_in_linear_non-gaussian_latent-variable_cyclic_causal_.md)
- [\[ICLR 2026\] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Perturbations](../../ICLR2026/causal_inference/rfeval_benchmarking_reasoning_faithfulness_under_counterfactual_perturbations.md)
- [\[CVPR 2026\] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](../../CVPR2026/causal_inference/fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)

</div>

<!-- RELATED:END -->
