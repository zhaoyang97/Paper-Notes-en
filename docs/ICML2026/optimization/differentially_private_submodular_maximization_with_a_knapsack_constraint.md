---
title: >-
  [Paper Note] Differentially Private Submodular Maximization with a Knapsack Constraint
description: >-
  [ICML 2026][Optimization][Submodular Maximization] This paper introduces differentially private algorithms for submodular maximization with a knapsack constraint (SMK). For monotone objectives, it achieves the optimal $(1-1/e)$ approximation while improving the additive error from polynomial dependency on $n$ to polylogarithmic dependency and reducing query complexity from exponential to polynomial. Additionally, it provides the first DP algorithm with provable guarantees ($1…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Submodular Maximization"
  - "Knapsack Constraint"
  - "Differential Privacy"
  - "Approximation Algorithms"
  - "Combinatorial Optimization"
date: 2026-05-08
content_hash: 821c3f38cf0a8700
---

# Differentially Private Submodular Maximization with a Knapsack Constraint

**Conference**: ICML 2026  
**arXiv**: [2606.14951](https://arxiv.org/abs/2606.14951)  
**Code**: To be confirmed  
**Area**: Optimization / Differential Privacy  
**Keywords**: Submodular Maximization, Knapsack Constraint, Differential Privacy, Approximation Algorithms, Combinatorial Optimization

## TL;DR
This paper introduces differentially private algorithms for submodular maximization with a knapsack constraint (SMK). For monotone objectives, it achieves the optimal $(1-1/e)$ approximation while improving the additive error from polynomial dependency on $n$ to polylogarithmic dependency and reducing query complexity from exponential to polynomial. Additionally, it provides the first DP algorithm with provable guarantees ($1/4$ approximation) for non-monotone objectives.

## Background & Motivation

**Background**: Submodular maximization, which characterizes "diminishing marginal returns," is a cornerstone of discrete optimization. It is widely applied in machine learning tasks like feature selection, recommendation systems, influence maximization, and sample selection. A classic variant is **knapsack-constrained submodular maximization (SMK)**: each element $v$ has a cost $c(v)$, and the goal is to maximize a submodular function $f(S)$ subject to $\sum_{v\in S}c(v)\le B$. When these tasks involve sensitive personal data (e.g., medical feature selection), rigorous **differential privacy (DP)** guarantees are required alongside strong utility.

**Limitations of Prior Work**: DP submodular maximization has primarily focused on **cardinality** and **matroid** constraints. To the authors' knowledge, the only prior work addressing knapsack constraints is Sadeghi & Fazel (2021), which has three major drawbacks: ① it relies on the **exact evaluation of the multilinear extension**, requiring $2^{|N|}$ queries in the worst case (exponential) under the value-oracle model; ② it only supports **monotone** functions, excluding non-monotone applications involving diversity or regularized penalties; ③ its utility guarantee includes an additive error with **polynomial dependency on the ground set size $n$** and scales with the **minimum element cost $c_{\min}$** ($\Delta/(c_{\min}\varepsilon)$), both of which are relatively loose.

**Key Challenge**: Privatizing combinatorial SMK algorithms faces two difficulties not present in cardinality/matroid constraints. First, greedy selection in SMK uses the **density score** $f(u\mid S)/c(u)$; varying costs lead to **highly disparate score sensitivities**. Directly applying the Exponential Mechanism would make the error proportional to the worst-case sensitivity $\Delta\cdot B/c_{\min}\ge \Delta k$, introducing an overhead factor of $k$. Second, the optimal Two-Guess-Greedy algorithm requires $O(n^2)$ enumeration iterations. Using standard advanced composition to distribute the privacy budget causes the error to grow polynomially with $n$, failing to reach the SOTA polylogarithmic dependency.

**Goal**: To improve the **approximation ratio, additive error, and query complexity** for both monotone and non-monotone SMK, bridging the gap with cardinality/matroid constraint settings.

**Key Idea**: The authors use a "private selection mechanism with fine-grained sensitivity" and "treating parallel enumeration structures as a set of DP subroutines" to solve these challenges. The former ensures the error depends only on $\Delta f$ rather than the worst-case density sensitivity, while the latter bypasses standard composition penalties to achieve polylogarithmic dependency.

## Method

### Overall Architecture
The paper presents three algorithms for three different settings, all sharing a common privatization logic. For the **monotone** case, privatization is built upon the **Two-Guess-Greedy** framework by Feldman et al. (2023)—which enumerates two elements from the optimal solution and then runs density greedy—resulting in an optimal $(1-1/e)$ approximation (Algorithm 2). A faster variant running a single density greedy yields a $1/2$ approximation (Algorithm 7). For the **non-monotone** case, the algorithm adopts the "density greedy + random deletion with fixed probability" paradigm (Han et al., 2021), providing the first guaranteed DP algorithm with a $1/4$ expected approximation (Algorithm 3). All three replace the greedy "max density selection" step with a noisy private choice and use two techniques to control error and privacy budget.

To understand the framework, one must understand why the non-private Two-Guess-Greedy achieves $(1-1/e)$: pure density greedy lacks constant approximation guarantees for knapsack constraints. Sviridenko (2004) fixed this by "guessing" three elements of OPT, and Feldman et al. reduced this to two. This requires enumerating all $O(n^2)$ element pairs, running density greedy for each, and taking the best. This "enumerate-and-select" parallel structure is repurposed here for a privacy advantage.

### Key Designs

**1. Handling "Non-uniform Density Sensitivity" with GRNM to make error dependent only on $\Delta f$**

In knapsack greedy, the element with the highest density $f(u\mid S)/c(u)$ is selected, but varying costs $c(u)$ result in vastly different sensitivities for density scores. Standard Exponential Mechanism or Report-Noisy-Max (RNM) errors are proportional to the **maximum sensitivity across all candidates** $\Delta=\max_u\Delta_{q(u;\cdot)}$, which is $\Delta f\cdot B/c_{\min}$ here. Since $B/c_{\min}\ge k$ (where $k$ is the maximum cardinality of a feasible solution), this introduces a factor of $k$. The authors instead use the **Generalized Report-Noisy-Max (GRNM)** from Raskhodnikova & Smith (2016). It applies RNM to "carefully constructed auxiliary scores" and provides a fine-grained utility bound **based on each candidate's own sensitivity**. With probability $1-\beta$, it outputs $u^*$ such that $q(u^*)\ge \max_u\{q(u)-(2\Delta_{q(u;\cdot)}/\varepsilon)\log(|C|/\beta)\}$, effectively penalizing high-sensitivity candidates without forcing all candidates to suffer from the worst-case sensitivity. The resulting additive error is proportional to $\Delta f$ (rather than $\Delta f\cdot B/c_{\min}$), matching the cardinality constraint setting.

**2. Treating Two-Guess-Greedy's parallel enumeration as a "set of DP subroutines" to reduce composition penalty to polylog**

The $O(n^2)$ enumeration steps in Two-Guess-Greedy are a privacy "composition bottleneck." Using advanced composition makes the error grow polynomially with $n$. The key observation is that these $O(n^2)$ steps are **highly parallel**, fitting the framework of "selecting the best output from a set of DP subroutines." Each subroutine is a private execution of density greedy. Based on the **Generalized Selection Framework** by Cohen et al. (2023), where the privacy loss is **independent of the total number of subroutines**, the authors **randomize the number of times each enumeration step is invoked**. This bypasses the standard composition penalty and compresses the error dependency on $n$ to **polylogarithmic**. This is a fundamental improvement over the polynomial dependency in Sadeghi & Fazel.

**3. Robustness for non-monotone cases: "No shutdown for negative marginals + RNM for best observed solution"**

Privatizing non-monotone SMK involves three additional obstacles. ① **Composition bottleneck due to indefinite selection counts**: Since selected elements can be randomly discarded, the algorithm might perform up to $n$ selections, causing the error to grow with $\sqrt{n}$ under standard composition. The authors prove the required number of private selections is **stochastically dominated by the sum of independent geometric random variables**, concentrating around $O(k)$. ② **Robustness to negative marginal returns**: Non-private analysis (Han et al., 2021) relies on selecting elements with "strictly positive marginal gain." However, DP noise can lead to the accidental selection of elements with negative gains. The authors modify the algorithm to **not stop when no positive gain elements exist, but to continue adding feasible elements**, finally using RNM to select the **approximate best solution observed**. The analysis focuses on "early iterations where large gains still exist," proving that large-contribution elements are selected with high probability even if noise masks the sign of the marginal gain, achieving a $1/4$ approximation. ③ **Non-uniform density sensitivity**: Resolved using GRNM as in the monotone case.

### Mechanism: Medical Feature Selection under Privacy Budget
Example 1.1 in the paper illustrates the mechanism. Given a sensitive medical dataset $\{(x_i,y_i)\}_{i=1}^n$, where each individual has feature vectors $x_i$ and a disease label $y_i$, and each feature $j$ has a cost $c_j$, the goal is to select a set $S$ such that $\sum_{j\in S}c_j\le B$ to maximize a submodular utility $f_D(S)$ (e.g., mutual information). $f_D$ depends on the sensitive data $D$ with sensitivity $\Delta f$. A non-private Two-Guess-Greedy could leak whether a specific individual has the disease. The proposed private algorithm enumerates all feature pairs as "guesses," runs density greedy on the remainder, and uses GRNM instead of selecting the absolute max density. Features that are cheap but have high sensitivity (e.g., specific to one person) are penalized to avoid deterministic selection. Finally, it uses the Cohen framework to privately select the best feature set. This ensures $(\varepsilon, \delta)$-DP while keeping additive error at $\tilde{O}(\Delta f\cdot k^{1.5}/\varepsilon)$.

### Loss & Training
This is a theoretical work; there is no training. Privacy analysis is built on standard tools: DP definition $\Pr[A(D)\in S]\le e^\varepsilon\Pr[A(D')\in S]+\delta$; Laplace Mechanism; Basic/Advanced Composition Theorems; and the Report-Noisy-Max (RNM) mechanism along with its generalized version (GRNM) and Cohen et al.'s Generalized Selection. Submodular sensitivity is defined as $\Delta f=\max_{S\subseteq N, D\sim D'}|f_D(S)-f_{D'}(S)|$.

## Key Experimental Results

> As a theoretical paper, "experiments" refer to the comparison of theoretical guarantees. The table below summarizes the three main results against the only previous DP knapsack work ($n$=ground set size, $k$=max cardinality, $L=\max_u f(u)$, $\beta$=failure probability, normalized budget $B=1$ so $1/c_{\min}\ge k$).

### Main Results Comparison ($(\varepsilon,\delta)$-DP, $\Delta$-sensitive objective)

| Algorithm | Monotone? | Approx. Ratio | Additive Error (Magnitude) | Query Complexity |
| :--- | :--- | :--- | :--- | :--- |
| Algorithm 2 (Ours i) | Yes | $1-1/e$ (Optimal) | $\frac{\Delta k^{1.5}}{\varepsilon}\sqrt{\log\frac{n}{\delta\beta}\log\frac{n}{\beta}}$ | $O(\beta^{-1}n^3 k)$ |
| Algorithm 7 (Ours ii) | Yes | $1/2$ | $\frac{\Delta k^{1.5}}{\varepsilon}\sqrt{\log\frac1\delta\log\frac n\beta}$ | $O(nk)$ |
| Algorithm 3 (Ours iii) | No | $1/4$ (Expectation) | $\frac{\Delta k}{\varepsilon}\sqrt{(k+\log\frac1\delta)\log\frac1\delta\log\frac n\beta}$ | $O(nk)$ |
| Sadeghi & Fazel 2021 | Yes | $1-1/e$ | $\frac{\Delta}{c_{\min}\varepsilon}\sqrt{\frac1\beta\log\frac1\delta\, n\log\frac n\beta}+\frac{\beta L}{c_{\min}^2}$ | $O(2^n)$ |

### Key Findings
- **Lower Bound Perspective**: The expected additive error lower bound for $(\varepsilon,\delta)$-DP under cardinality constraints is $\Omega(kc\log(\varepsilon/\delta)/\varepsilon)$. The error in this paper is $\tilde{O}(\sqrt{k})$ higher, but it handles a more general class of submodular functions, and the $k$ dependency matches the best result for polynomial-time algorithms under cardinality constraints.
- **Optimal Approximation Ratio**: Monotone $(1-1/e)$ is information-theoretically optimal; non-monotone $1/4$ matches current SOTA combinatorial non-private guarantees.
- **Utility-Efficiency Trade-off**: For monotone targets, Algorithm 2 uses $O(\beta^{-1}n^3k)$ queries for the optimal $(1-1/e)$, while Algorithm 7 uses $O(nk)$ queries for a $1/2$ ratio, providing a trade-off for large-scale instances.

## Highlights & Insights
- **"Parallel Enumeration as Subroutines" is the highlight**: Interpreting the $O(n^2)$ enumeration of Two-Guess-Greedy as a selection problem from DP subroutines allows the use of Cohen's framework where privacy loss is independent of the number of subroutines. This provides a clear path for privatizing other enumeration-based combinatorial optimization algorithms.
- **Fine-grained Sensitivity is Essential for Cost Heterogeneity**: Knapsack constraints' inherent difficulty relative to cardinality is non-uniform density sensitivity. Using GRNM to "charge" each candidate based on its own sensitivity is the key to reducing error from $\Delta k$ to $\Delta$.
- **Handling Non-monotonicity with Post-hoc RNM**: Rather than requiring strictly positive gains at each step (which DP noise masks), the algorithm continues selection and uses RNM to pick the best observed state, shifting analysis focus to early iterations.

## Limitations & Future Work
- **Additive Error Gap**: The monotone additive error contains $k^{1.5}$, leaving a $\sqrt{k}$ gap with the $\Omega(k)$ lower bound. Whether this can be eliminated is an open question.
- **Complexity of Optimal Monotone Approx**: Algorithm 2 requires $O(\beta^{-1}n^3k)$ queries, which is computationally expensive for massive datasets.
- **Purely Theoretical**: No empirical validation on real-world tasks like medical feature selection was provided.
- **Sensitivity Assumption**: Relies on the bounded-sensitivity $\Delta$ submodular class; objectives with unbounded or hard-to-bound sensitivity are not directly applicable.
- **Centralized DP Model**: The analysis assumes a trusted server holding dataset $D$. Whether similar utility can be maintained in Local DP (LDP) or Shuffle DP models remains unexplored.

## Related Work & Insights
- **vs Sadeghi & Fazel (2021)**: The only previous SMK work. It relied on exact multilinear extensions ($2^n$ queries), was monotone only, and had polynomial error dependency on $n$. This paper replaces it with combinatorial greedy (poly query), polylog($n$) error, and support for non-monotone cases.
- **vs Mitrovic et al. (2017)**: Achieved $(1-1/e)$ with $\tilde{O}(\Delta k^{1.5}/\varepsilon)$ error for **cardinality** constraints. This paper brings similar error and approximation levels to the more difficult **knapsack** constraint.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First non-monotone DP-SMK; "parallel enumeration to subroutines" insight).
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive theory with alignment to bounds, but lacks empirical data).
- Writing Quality: ⭐⭐⭐⭐ (Clear structure mapping challenges to solutions).
- Value: ⭐⭐⭐⭐ (Fills a major gap in DP submodular optimization).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[ICML 2026\] Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICLR 2026\] Efficient Submodular Maximization for Sums of Concave over Modular Functions](../../ICLR2026/optimization/efficient_submodular_maximization_for_sums_of_concave_over_modular_functions.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)

</div>

<!-- RELATED:END -->
