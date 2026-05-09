---
title: >-
  [Paper Note] Variance Computation for Weighted Model Counting with Knowledge Compilation Approach
description: >-
  [AAAI 2026][Weighted Model Counting] This paper treats the weights in weighted model counting (WMC) as random variables with associated variances, and proposes a polynomial-time algorithm for computing WMC variance on structured d-DNNF representations. It further proves intractability of this problem on structured DNNF, d-DNNF, and FBDD (unless P=NP), and applies the framework to quantify parameter uncertainty in Bayesian network inference.
tags:
  - AAAI 2026
  - Weighted Model Counting
  - Knowledge Compilation
  - Variance Computation
  - Bayesian Networks
  - Structured d-DNNF
date: 2026-05-08
content_hash: 179cf015b9ef4bbd
---

# Variance Computation for Weighted Model Counting with Knowledge Compilation Approach

**Conference**: AAAI 2026
**arXiv**: [2601.03523](https://arxiv.org/abs/2601.03523)
**Code**: [https://github.com/nttcslab/variance-wmc](https://github.com/nttcslab/variance-wmc)
**Area**: Knowledge Compilation / Probabilistic Inference / Weighted Model Counting
**Keywords**: Weighted Model Counting, Knowledge Compilation, Variance Computation, Bayesian Networks, Structured d-DNNF

## TL;DR

This paper treats the weights in weighted model counting (WMC) as random variables with associated variances, and proposes a polynomial-time algorithm for computing WMC variance on structured d-DNNF representations. It further proves intractability of this problem on structured DNNF, d-DNNF, and FBDD (unless P=NP), and applies the framework to quantify parameter uncertainty in Bayesian network inference.

## Background & Motivation

**State of the Field**: Knowledge compilation is a core technique for converting propositional formulas into compact and tractable representations. The most important query in this setting is weighted model counting (WMC)—computing the weighted sum over all satisfying assignments of a Boolean function. WMC has been widely applied to probabilistic inference tasks including Bayesian networks, factor graphs, and probabilistic programming.

**Limitations of Prior Work**: In practice, parameters of probabilistic models are typically learned from data, and when training data is insufficient, the parameters themselves carry uncertainty. However, conventional WMC inference treats weights as fixed real numbers and entirely ignores parameter uncertainty, making inference results potentially unreliable—there is no way to assess how much the output may fluctuate.

**Root Cause**: Bayesian statistics can measure uncertainty by assigning distributions to parameters, treating inference results as random variables, and computing their variance. However, the tractability of WMC variance computation across knowledge compilation representations has been almost entirely unknown. Prior to this work, only Nakamura et al. (2022) addressed a special case on OBDDs for network reliability analysis.

**Paper Goals**: (a) Can WMC variance be computed in polynomial time on more general knowledge compilation representations? (b) On which representations does this problem become intractable? (c) How can variance computation be applied to Bayesian network inference for uncertainty quantification?

**Starting Point**: The authors systematically investigate the tractability boundaries of the VC (variance computation) and CVC (covariance computation) queries across various NNF subclasses from the perspective of the knowledge compilation map.

**Core Idea**: By leveraging the variable decomposition structure induced by a vtree, the paper derives a recursive decomposition of the covariance formula with memoization, achieving polynomial-time WMC variance computation on structured d-DNNF. Intractability on less constrained representations is established via reductions.

## Method

### Overall Architecture

The input is a Boolean function $f$ represented as a structured d-DNNF, along with the expectations, variances, and covariances of positive weights $P_x$ and negative weights $N_x$ for each Boolean variable $x$. The output is the variance $\mathrm{V}[W_f]$ of the WMC $W_f$.

The overall pipeline consists of three stages:
- **Preprocessing**: Build the LCA data structure ($O(|\mathcal{V}|)$) and precompute auxiliary data required by ADJEXP and ADJCOV ($O(|\mathcal{V}|^2)$).
- **Expectation Computation**: Precompute the expected value $\mathrm{E}[W_{f_\gamma}]$ for each node using the standard WMC algorithm ($O(|\alpha|)$).
- **Recursive Covariance Computation**: Recursively decompose and memoize $\mathrm{Cov}[W_{f_\alpha}, W_{f_\beta}]$ via Algorithm 1; the final variance is $\mathrm{V}[W_f] = \mathrm{Cov}[W_f, W_f]$.

Total complexity is $O(|\alpha|^2 + |\mathcal{V}|^2)$.

### Key Designs

1. **Formal Definition of WMC Variance (VC/CVC Queries)**:

    - Function: Each variable's weight $(P_x, N_x)$ is treated as a random variable. The variance computation query VC is defined as computing $\mathrm{V}[W_f^\mathcal{V}]$, and the covariance computation query CVC as computing $\mathrm{Cov}[W_f^\mathcal{V}, W_g^\mathcal{V}]$.
    - Mechanism: Weight pairs $(P_x, N_x)$ and $(P_y, N_y)$ for distinct variables $x \neq y$ are assumed mutually independent, while $P_x$ and $N_x$ for the same variable may be correlated. Under this assumption, $\mathrm{E}[W_f]$ reduces to standard WMC, while the variance requires a new algorithm.
    - Design Motivation: The independence assumption naturally holds under the parameter independence assumption of Bayesian networks and enables derivation of tractable decomposition formulas.

2. **Vtree-Guided Recursive Covariance Decomposition (Algorithm 1)**:

    - Function: Recursively compute the WMC covariance between any two nodes $\alpha, \beta$ in a structured t-d-DNNF.
    - Mechanism: Three cases are distinguished based on the relationship between $\mathsf{d}(\alpha)$ and $\mathsf{d}(\beta)$ in the vtree:
        - **Case I** (no ancestor–descendant relationship): The product independence formula $\mathrm{Cov}[AX, BY] = \mathrm{Cov}[A,B]\mathrm{Cov}[X,Y] + \mathrm{Cov}[A,B]\mathrm{E}[X]\mathrm{E}[Y] + \mathrm{E}[A]\mathrm{E}[B]\mathrm{Cov}[X,Y]$ decomposes the covariance into the left and right vtree subtrees.
        - **Case II** ($\alpha$ is a $\vee$-node): Determinism guarantees covariance additivity $\mathrm{Cov}[A+B, C] = \mathrm{Cov}[A,C] + \mathrm{Cov}[B,C]$, decomposing to child nodes.
        - **Case III** ($\alpha$ is a $\wedge$-node): Structured decomposability decomposes $\beta$ along the vtree's left and right subtrees, followed by application of the product formula.
    - Design Motivation: The vtree provides a hierarchical decomposition of the variable set, enabling precise tracking of variable set changes during recursion—a key technical distinction from the OBDD-based algorithm of Nakamura et al. (2022).
    - Memoization: All computed values $\mathrm{Cov}[\alpha, \beta]$ are stored in $\mathtt{c}[\alpha, \beta]$ to avoid redundant recursive calls, ensuring total complexity $O(|\alpha||\beta|)$.

3. **Auxiliary Functions ADJEXP and ADJCOV (Variable Set Adjustment)**:

    - Function: Adjust the variable set of WMC during recursion to align subproblems with inconsistent variable scopes.
    - Mechanism: Since $W_f^\mathcal{V}$ depends on $\mathcal{V}$, when a node's scope is smaller than the current vtree node's scope, the contribution of missing variables must be completed using the true function. The preprocessing stage computes auxiliary quantities such as $\prod_{x \in S}(\mu_{P_x} + \mu_{N_x})$, enabling adjustment in $O(1)$ time.
    - Design Motivation: This is the core technical challenge in generalizing from OBDD to structured t-d-DNNF—whereas OBDD variables follow a linear order, vtrees are tree-structured, making variable set management considerably more complex.

### Theoretical Analysis

**Tractability (Theorem 7)**: When $f$ and $g$ are given as structured t-d-DNNFs $\alpha$ and $\beta$ sharing the same vtree, CVC can be solved in $O(|\alpha||\beta| + |\mathcal{V}|^2)$ time, and VC in $O(|\alpha|^2 + |\mathcal{V}|^2)$ time.

**Intractability (Theorem 11)**: When Boolean functions are represented as structured DNNF, d-DNNF, or FBDD, both VC and CVC are intractable (unless P=NP). The proof strategy is as follows:
- For st-DNNF: Model counting (CT) is reduced to VC by choosing special weights such that $|A_f| = \lceil V[W_f]/(4^n - 1)\rceil$.
- For d-DNNF/FBDD: Sentence entailment (SE) is reduced to CVC, and then Lemma 15 further reduces CVC to VC.

This implies that d-DNNF and FBDD support polynomial-time WMC but not polynomial-time variance computation—a surprising and theoretically significant tractability separation result.

## Key Experimental Results

### Main Results: Variance Computation on 70 Binary Bayesian Networks

Seventy binary Bayesian networks (3 to 122 random variables) were obtained from the bnRep database, encoded as CNFs using ENC2, and compiled into SDDs (a subset of structured t-d-DNNF). Parameter uncertainty was modeled using Beta distributions ($\theta=10$).

| Network | # Variables | SDD Size | Compilation Time (s) | Variance Computation Time (s) |
|---------|-------------|----------|----------------------|-------------------------------|
| projectmanagement | 26 | 3888 | 0.500 | 0.025 |
| GDIpathway2 | 28 | 2755 | 0.784 | 0.021 |
| grounding | 36 | 3397 | 2.387 | 0.017 |
| engines | 12 | 1804 | 0.240 | 0.011 |
| windturbine | 122 | 2043 | 1.380 | 0.009 |

**Key Finding**: After SDD compilation, variance computation takes at most 0.025 seconds; even including compilation time, the most expensive network requires approximately 10 seconds in total.

### Parameter Sensitivity Analysis: algalactivity2 Network

The variance of $\Pr(\text{Chl\_a}=0)$ is computed, and the variance of each parameter is reduced to 1/10 of its original value one at a time, to assess the impact on marginal probability variance.

| Parameter with Reduced Variance | Resulting Marginal Variance |
|---------------------------------|----------------------------|
| DO\|pH₀,Te₀ | 0.002887 |
| Chl_a\|C₁,DO₀,N₀,Te₁ | 0.003532 |
| Te\|P₀ | 0.003554 |
| pH\|Te₀ | 0.003592 |
| Chl_a\|C₁,DO₁,N₁,Te₀ | 0.003674 |
| (no reduction) | 0.003904 |

### Key Findings

- **Highly Efficient Variance Computation**: After SDD compilation, variance computation for all 70 networks completes within milliseconds, validating the practical tractability of the algorithm.
- **Non-Obvious Parameter Influence**: The parameters most influencing the variance of $\Pr(\text{Chl\_a}=0)$ include not only the conditional probabilities of Chl_a itself but also those of DO, Te, and pH, demonstrating that it is impossible to identify which parameters most affect inference uncertainty without actually computing the variance.
- **Practical Decision-Making Value**: With a mean of 0.5281 and standard deviation of 0.0625 for $\Pr(\text{Chl\_a}=0)$, variance of this magnitude materially affects decisions such as determining whether $\Pr(\text{Chl\_a}=0) \leq 0.55$.

## Highlights & Insights

- **Elegant Extension of the Knowledge Compilation Map**: Embedding VC/CVC queries into the tractability framework of the knowledge compilation map yields a sharp boundary—tractable on structured t-d-DNNF, intractable on all less constrained representations. This clean demarcation is theoretically elegant.
- **Surprising Separation for d-DNNF/FBDD**: Despite supporting polynomial-time WMC, d-DNNF and FBDD cannot tractably compute variance—establishing that variance computation is fundamentally harder than expectation computation and requires the additional constraint of structured decomposability.
- **Non-Trivial Generalization from OBDD to Structured t-d-DNNF**: Vtree-guided variable set management is the key technical contribution. The generalization from linear variable ordering (OBDD) to tree-structured decomposition (structured t-d-DNNF) broadens the applicability of the algorithm and incidentally improves known results for network reliability analysis from pathwidth to treewidth.
- **Practical Uncertainty Quantification Tool**: Variance computation enables quantitative answers to the question of which poorly estimated parameters most affect inference results, providing direct practical value for guiding data collection and parameter learning.

## Limitations & Future Work

- **CVC Requires a Shared Vtree**: Covariance computation requires both structured t-d-DNNFs to respect the same vtree, which restricts certain applications. The authors conjecture that CVC is intractable without a shared vtree but provide no proof.
- **Only Second-Order Moments Considered**: The tractability of higher-order moments (e.g., skewness, kurtosis) is not explored, despite their relevance in some decision-making contexts.
- **Compilation Bottleneck**: SDD compilation itself may become a bottleneck (the most expensive network takes 10 seconds to compile), and SDD size may grow exponentially. Variance computation is polynomial only given an already-compiled Boolean representation.
- **Parameter Independence Assumption**: While natural for Bayesian networks, this assumption may not hold in other applications such as probabilistic programming.
- **Evaluation Limited to Binary Bayesian Networks**: Although a theoretical extension to multi-valued variables is provided in Appendix B, experiments are conducted exclusively on binary networks.

## Related Work & Insights

- **vs. Nakamura et al. (2022)**: The prior work is restricted to OBDDs and the specific setting of network reliability analysis. This paper extends the algorithm to structured t-d-DNNF (a strict superset), generalizes the problem to arbitrary WMC, and theoretically improves the complexity bound from pathwidth to treewidth.
- **vs. Probabilistic Circuits**: Probabilistic circuits can represent the distribution of a random variable $X$ and compute its moments, but the present paper requires computing the variance of the probability value $\Pr(X=a)$ itself as a random variable—a different level of abstraction that probabilistic circuits currently cannot address.
- **vs. Credal Networks**: Credal networks handle uncertainty by computing upper and lower bounds on marginal probabilities, but inference is NP-hard even for networks of constant treewidth. By contrast, the variance computation proposed in this paper is polynomial-time under equivalent structural conditions.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically embedding variance computation queries into the knowledge compilation map is a novel perspective, though the overall approach is a natural extension of the existing framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Experiments cover 70 real-world Bayesian networks along with a parameter sensitivity case study, providing good coverage, but comparisons against baselines such as Monte Carlo estimation are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Formal definitions are precise, proofs are rigorous, case studies are intuitive, and the overall structure is highly professional.
- Value: ⭐⭐⭐⭐ Solid theoretical contributions (sharp tractability boundaries) with clear application prospects (uncertainty quantification in Bayesian networks), though the target audience is relatively narrow (knowledge compilation and probabilistic inference communities).

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Tractable Weighted First-Order Model Counting with Bounded Treewidth Binary Evidence](tractable_weighted_first-order_model_counting_with_bounded_treewidth_binary_evid.md)
- [\[AAAI 2026\] Model Counting for Dependency Quantified Boolean Formulas](model_counting_for_dependency_quantified_boolean_formulas.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[AAAI 2026\] Structural Approach to Guiding a Present-Biased Agent](structural_approach_to_guiding_a_present-biased_agent.md)
- [\[AAAI 2026\] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics](data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)

<!-- RELATED:END -->
