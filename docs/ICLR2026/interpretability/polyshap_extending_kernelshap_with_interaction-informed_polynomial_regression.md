---
title: >-
  [Paper Note] PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression
description: >-
  [ICLR 2026][Interpretability][Shapley values] This paper proposes PolySHAP, which extends KernelSHAP's linear approximation to higher-order polynomial regression to capture nonlinear feature interactions…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Shapley values"
  - "Explainable AI"
  - "Polynomial regression"
  - "Feature interactions"
  - "KernelSHAP"
date: 2026-05-08
content_hash: a845fe31b75d3254
---

# PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression

**Conference**: ICLR 2026
**arXiv**: [2601.18608](https://arxiv.org/abs/2601.18608)
**Code**: [GitHub](https://github.com/FFmgll/PolySHAP)
**Area**: Interpretability
**Keywords**: Shapley values, Explainable AI, Polynomial regression, Feature interactions, KernelSHAP

## TL;DR

This paper proposes PolySHAP, which extends KernelSHAP's linear approximation to higher-order polynomial regression to capture nonlinear feature interactions, thereby improving the estimation accuracy of Shapley values. The paper further provides a theoretical proof that paired sampling is equivalent to second-order PolySHAP, offering the first rigorous explanation for the superior performance of this widely used heuristic.

## Background & Motivation

Shapley values are among the most fundamental game-theoretic tools in explainable AI for quantifying individual feature contributions to model predictions. However, exact computation requires $2^d$ game evaluations for a model with $d$ features, making it computationally prohibitive. KernelSHAP circumvents this exponential cost by approximating the game function $\nu$ with a linear function, but such linear approximations are inherently unable to capture nonlinear interaction effects among features, limiting estimation accuracy.

Furthermore, paired sampling — a widely adopted heuristic strategy that substantially improves KernelSHAP's estimation quality — has lacked a satisfactory theoretical explanation for its effectiveness. This paper provides a unified theoretical framework and practical solution to both problems through the lens of polynomial regression.

## Method

### Overall Architecture

The core idea of PolySHAP is to extend KernelSHAP's linear approximation to higher-order polynomials, incorporating interaction terms to capture nonlinear feature relationships. The procedure is as follows:
1. Define an interaction frontier $\mathcal{I}$ specifying the set of interaction terms to be modeled.
2. Fit the polynomial via weighted least squares.
3. Convert the PolySHAP representation back to Shapley values using a theoretical formula.

### Key Designs

1. **PolySHAP Interaction Representation**: The game function is approximated by a polynomial containing interaction terms. The PolySHAP representation $\phi^{\mathcal{I}} \in \mathbb{R}^{d'}$ (where $d' = d + |\mathcal{I}|$) is obtained by solving a weighted least squares problem:
$$\phi^{\mathcal{I}}[\nu] := \arg\min_{\phi \in \mathbb{R}^{d'}: \langle\phi,\mathbf{1}\rangle = \nu(D)} \sum_{S \subseteq D} \mu(S)\left(\nu(S) - \sum_{T \in D \cup \mathcal{I}} \phi_T \prod_{j \in T} \mathbb{1}[j \in S]\right)^2$$
The PolySHAP representation is then converted to Shapley values via Theorem 4.3: $\phi_i^{SV}[\nu] = \phi_i^{\mathcal{I}} + \sum_{S \in \mathcal{I}: i \in S} \frac{\phi_S^{\mathcal{I}}}{|S|}$. **Design Motivation**: A more expressive polynomial approximation of the game function yields more accurate Shapley value estimates.

2. **Paired Sampling Equivalence Theorem (Theorem 5.1)**: The paper proves that under paired sampling (simultaneously sampling $S$ and $D \setminus S$), the output of KernelSHAP is **exactly equal** to that of second-order PolySHAP (2-PolySHAP), meaning that paired KernelSHAP implicitly captures all second-order interactions. This provides the first theoretical explanation for the substantial accuracy gains from paired sampling. **Design Motivation**: To supply a rigorous theoretical foundation for the paired sampling heuristic prevalent in practice.

3. **$k$-Additive Interaction Frontier**: The frontier $\mathcal{I}_{\leq k} = \{S \subseteq D : 2 \leq |S| \leq k\}$ is defined to progressively include higher-order interaction terms ($k$-PolySHAP). The case $k=1$ reduces to KernelSHAP, while $k=2$ incorporates all pairwise interactions. For high-dimensional settings, a **partial interaction frontier** $\mathcal{I}_\ell$ is introduced to selectively include a subset of higher-order terms when the computational budget does not support a full $k$-th order expansion.

4. **Leverage Score Sampling**: A leverage score sampling strategy is adopted, drawing subsets according to leverage scores rather than Shapley weights. Under a budget of $m = O(d' \log(d'/\delta) + d'/({\epsilon\delta}))$, this guarantees approximation quality with probability $1-\delta$.

### Loss & Training

PolySHAP solves a constrained weighted least squares problem, where the constraint enforces the efficiency property (Shapley values sum to $\nu(D)$). The constrained problem is reduced to an unconstrained one via the projection matrix $\mathbf{P}_{d'}$. A border trick is employed to enumerate small-cardinality subsets exhaustively rather than by sampling.

## Key Experimental Results

### Main Results

Experiments are conducted on 15 diverse explanation games spanning tabular, image, and language domains, with $d$ ranging from 8 to 101, comparing PolySHAP against multiple baselines.

| Dataset/Game | Metric | PolySHAP (3rd-order) | KernelSHAP | Gain |
|---|---|---|---|---|
| Housing ($d=8$) | MSE | Best | Baseline | Substantial reduction |
| Adult ($d=14$) | MSE | Best | Baseline | Substantial reduction |
| Estate ($d=15$) | MSE | Best | Baseline | Substantial reduction |
| Cancer ($d=30$) | MSE | Best | Baseline | Substantial reduction |
| CG60 ($d=60$) | MSE | Marginal improvement | Baseline | Limited gain (high-dim.) |

### Ablation Study

| Configuration | Key Metric (MSE) | Notes |
|---|---|---|
| 1-PolySHAP (= KernelSHAP) | Baseline | No interaction terms |
| 2-PolySHAP | Significant improvement | All pairwise interactions included |
| 2-PolySHAP (50%) | Moderate improvement | Only 50% of pairwise interactions |
| 3-PolySHAP | Best | Third-order interactions; largest gains in low-dim. settings |
| Paired KernelSHAP vs. Paired 2-PolySHAP | Identical | Empirically validates Theorem 5.1 |
| Paired 3-PolySHAP vs. Paired 4-PolySHAP | Nearly identical | Suggests analogous equivalences at higher orders |

### Key Findings

- Incorporating any number of interaction terms consistently improves Shapley value approximation quality.
- Under paired sampling, KernelSHAP automatically attains 2-PolySHAP performance; thus, the practical gains of PolySHAP over paired KernelSHAP begin to manifest from third-order interactions onward.
- In high-dimensional settings ($d \geq 60$), the number of feasible third-order interaction terms is limited, resulting in smaller gains.
- RegressionMSR is the only baseline competitive with PolySHAP, but it relies on XGBoost tree models and exhibits instability on certain games.

## Highlights & Insights

- **Significant theoretical contribution**: The equivalence between paired sampling and 2-PolySHAP is an elegant theoretical result that resolves a long-standing practical puzzle.
- **Naturally elegant methodology**: The extension from linear to polynomial approximation is conceptually clean and preserves consistency guarantees.
- **Unified perspective**: KernelSHAP, Faith-SHAP, and $k_{ADD}$-SHAP are subsumed within a single framework.
- **Projection lemma**: The technical projection lemma (Lemma A.1) plays a central role in the proofs of multiple theorems.

## Limitations & Future Work

- In high-dimensional settings, the number of third-order interaction combinations grows as $\binom{d}{3}$, severely limiting the number of feasible interaction terms.
- The conjecture that paired $k$-PolySHAP is equivalent to $(k+1)$-PolySHAP (for odd $k$) remains unproven.
- The interaction frontier selection is generic (adding all terms up to a given order) and does not exploit problem-specific interaction structure.
- Runtime analysis remains largely theoretical; efficiency in large-scale practical applications warrants further investigation.

## Related Work & Insights

- Compared to RegressionMSR (Witter et al., 2025): PolySHAP maintains consistency without requiring an additional regression adjustment step.
- Relationship to $k_{ADD}$-SHAP (Pelegrina et al., 2023): PolySHAP simplifies and generalizes its convergence proofs.
- Future directions: Interaction detection methods (e.g., Tsang et al., 2020) or graph-structural information could be leveraged to construct more informed interaction frontiers.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The polynomial extension is natural but not groundbreaking; the paired sampling equivalence theorem is the standout contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Fifteen games covering tabular, image, and language domains with comprehensive baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, figures are intuitive, and the narrative is well-structured.
- **Value**: ⭐⭐⭐⭐ Represents a substantive advance in Shapley value estimation for XAI; the theoretical explanation of paired sampling carries broad implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Scaling Laws for Symbolic Regression](../../NeurIPS2025/interpretability/towards_scaling_laws_for_symbolic_regression.md)
- [\[NeurIPS 2025\] Are Greedy Task Orderings Better Than Random in Continual Linear Regression?](../../NeurIPS2025/interpretability/are_greedy_task_orderings_better_than_random_in_continual_linear_regression.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](provably_explaining_neural_additive_models.md)
- [\[ICLR 2026\] Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test](grokking_in_llm_pretraining_monitor_memorization-to-generalization_without_test.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text-Driven Reasoning](dynamic_reflections_probing_video_representations_with_text_alignment.md)

</div>

<!-- RELATED:END -->
