---
title: >-
  [Paper Note] Conditional Independent Component Analysis for Estimating Causal Structure with Latent Variables
description: >-
  [ICLR2026][Causal Inference][Causal Discovery] This paper proposes the principle of Conditional Independent Component Analysis (CICA)—extracting components that are mutually independent given a set of latent variables—and proves that by selecting its sparsest solution and applying row permutation, one can identify latent variable positions and all causal edges in linear non-Gaussian acyclic models with latent confounders, thereby breaking the reliance on "purity assumptions"…
tags:
  - "ICLR2026"
  - "Causal Inference"
  - "Causal Discovery"
  - "Latent Variables"
  - "Conditional Independent Component Analysis"
  - "Non-Gaussian Linear Model"
  - "Identifiability"
date: 2026-05-08
content_hash: 8e22f9a88a151e52
---

# Conditional Independent Component Analysis for Estimating Causal Structure with Latent Variables

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=TAOpnCPnjg](https://openreview.net/forum?id=TAOpnCPnjg)  
**Code**: To be confirmed  
**Area**: Causal Discovery / Latent Variable Structure Learning  
**Keywords**: Causal Discovery, Latent Variables, Conditional Independent Component Analysis, Non-Gaussian Linear Model, Identifiability

## TL;DR
This paper proposes the principle of Conditional Independent Component Analysis (CICA)—extracting components that are mutually independent given a set of latent variables—and proves that by selecting its sparsest solution and applying row permutation, one can identify latent variable positions and all causal edges in linear non-Gaussian acyclic models with latent confounders, thereby breaking the reliance on "purity assumptions" required by methods like GIN/TIN.

## Background & Motivation

**Background**: Recovering causal structures from observational data is a core task in scientific discovery. When unmeasured latent confounders exist, traditional methods relying on the "causal sufficiency" (no latent confounders) assumption fail. For handling latent variables, the linear Gaussian family has developed methods based on rank-deficiency constraints (Silva 2006, Huang 2022, Dong 2023, etc.) to recover structures up to the Markov equivalence class; the linear non-Gaussian family utilizes high-order statistics to provide additional asymmetric information, represented by GIN conditions (Xie 2020) and TIN conditions (Dai 2022).

**Limitations of Prior Work**: Most of these methods rely on the "purity assumption" to simplify the problem—requiring each latent variable to have enough **pure children** (variable sets that are conditionally independent given the latent variable and independent of the rest of the graph) and usually **prohibiting causal edges between observed variables**. Once the structure is "impure" (observed variables have edges between them or are simultaneously influenced by multiple latents), these methods cannot distinguish fundamentally different graphs. The paper uses a three-node example (Figures 1a and 1b, where $X_1, X_2, X_3$ share both a latent $L$ and noise $E_1$) to show that most methods fail to identify the true graph. Theoretically, only overcomplete ICA (OICA) and high-order cumulant-based methods could distinguish them, but the former relies on EM for approximate inference (computationally expensive and prone to local optima), while the latter is highly sensitive to outliers and requires massive samples for reliable estimation.

**Key Challenge**: There is a trade-off between identifiability and practicality—identifying complex (impure) structures requires OICA or high-order cumulants, which are computationally and statistically unfriendly.

**Key Insight**: The authors analyze why GIN/TIN fail on Figures 1a/1b and identify the root cause as their reliance only on **unilateral projection** $\omega^\top Y \perp\!\!\!\perp Z$ (finding a linear combination on the $Y$ side independent of the entire $Z$). When every pair of observed variables shares both latents and noise, unilateral projection cannot eliminate dependencies from both sources simultaneously. However, **bilateral projection** $\omega_1^\top Y \perp\!\!\!\perp \omega_2^\top Z$ (finding combinations on both sides) leaves additional identifiable traces—the paper proves that independence patterns expressible by unilateral projection are a subset of bilateral projection (Lemma 1), so bilateral projection carries richer identification signals.

**Core Idea**: Instead of fully separating all latent sources like OICA, it is better to explicitly "factor out the shared influence," requiring components to be **conditionally independent given a latent vector**. This generative principle is formalized as Conditional Independent Component Analysis (CICA): finding an invertible transform $W$ such that the coordinates of $Z=WX$ are mutually independent given some latent variable $L$. This allows optimization via tractable proxy targets (rank-deficiency) to avoid high-order cumulants while theoretically inducing the bilateral projections needed for identification.

## Method

### Overall Architecture
The problem is set as a linear structural causal model $V=BV+E$, where $V=X \cup L$ contains observed variables $X=\{X_i\}_{i=1}^m$ and latent variables $L=\{L_i\}_{i=1}^d$, $E$ are mutually independent non-Gaussian exogenous noises, and the adjacency matrix $B$ encodes direct causal effects. Equivalently $V=AE$, where $A=(I-B)^{-1}$ is the mixing matrix. The goal is to recover the complete causal graph $G$ (latent positions + all causal edges) from observed data $X$ only.

The pipeline operates as follows: First, "finding causal structure" is transformed into "solving CICA"—searching for an invertible $W$ such that $Z=WX$ is conditionally independent given latents; since the conditioning set is latent and not directly testable, a **differentiable proxy target** (rank-deficiency of covariance submatrices / Triad constraints) is used to make CICA optimizable; the resulting CICA solutions are not unique (having permutation, scaling, and "whether the conditioning set is latent or noise" ambiguity), but the paper proves the **sparsest** CICA solution corresponds exactly to the true causal matrix $I-B_{X,X}$ between observed variables; finally, these conclusions are extended recursively from single causal clusters to the entire hierarchy to obtain the full CICA-LiNGAM algorithm.

### Key Designs

**1. CICA Principle and Identifiability: Replacing "Full Independence" with "Conditional Independence"**

The generative assumption of CICA (Assumption 1) is: observations $X=AS$, sources $S=ML+E$, where $L$ are $p$ latent variables, $E$ are independent non-Gaussian, and $E \perp\!\!\!\perp L$. A $p$-th order CICA solution is defined as: there exist $p$ latent variables $L$ such that the components of $Z=WX$ are mutually independent given $L$ (Definition 5). When $p=0$, conditional independence reduces to full independence, making CICA equivalent to classical ICA. The key value of this step is that it **only requires "separation up to conditional independence" rather than full separation of all latent sources like OICA**—doing one step less is sufficient for identification.

Does this definition introduce new, complex mixing ambiguities? Lemma 2 provides the answer: for two valid CICA solutions $W_1, W_2$, there must exist a permutation matrix $P_\pi$ and a non-singular diagonal matrix $D$ such that $W_2=P_\pi D W_1$. That is, CICA only adds a **"latent set $L$ selection" ambiguity** compared to ICA, with the rest remaining permutation and scaling ambiguities. More importantly, Lemma 3 proves that any $p$-th order CICA solution can induce bilateral projections $\omega_1^\top Y' \perp\!\!\!\perp \omega_2^\top Z'$ (as long as subset sizes exceed $p$), which are exactly the signals needed to identify impure structures like Figures 1a/1b—linking "CICA solvability" with the "existence of bilateral projections."

**2. Rank-deficiency / Triad Proxy Targets: Making Untestable Conditional Independence Optimizable**

In the CICA definition, the conditioning set $L$ is latent and cannot be directly written as a testable objective. The paper borrows from Huang 2022 / Dong 2023, using **rank-deficiency constraints** to characterize conditional independence (Lemma 4): let $p=p_{\min}(X)$, $m \ge 2p+2$, and $X'=WX$; then $W$ is a $p$-th order CICA solution if and only if for any pair of disjoint subsets $X_1, X_2$ in $X'$ each containing $p+1$ coordinates, $\det(\Sigma_{X_1,X_2})=0$, where $\Sigma=\mathrm{Cov}(X')$. The intuition is that conditional independence is equivalent to the rank collapse of covariance submatrices. $m \ge 2p+2$ is not a hard constraint and can be relaxed by replacing covariance with high-order cumulant tensors.

When $p_{\min}(X)=1$, there is a proxy with even weaker conditions—the Triad constraint (Lemma 5): define pseudo-residuals $E_{(i,j|k)}:=\mathrm{Cov}(X_j,X_k)X_i-\mathrm{Cov}(X_i,X_k)X_j$; if $E_{(i,j|k)} \perp\!\!\!\perp X_k$, then $\{X_i,X_j\}$ and $X_k$ satisfy the Triad constraint. $W$ is a 1st-order CICA solution if and only if every ordered triplet of $X'$ satisfies this constraint. Since determinants and dependence measures (like HSIC) are differentiable, these lemmas provide optimization criteria for CICA. $p_{\min}(X)$ does not need to be given a priori and can be determined within the algorithm using GIN conditions (Lemma 11).

**3. Sparsest Solution = True Causal Matrix: Using Sparsity to Resolve Latent Set Ambiguity**

The "latent set ambiguity" left by Lemma 2 is a core challenge: when $W$ is a CICA solution, its conditioning set **is not necessarily the true latent confounders** $\mathrm{LPa}(X)$, but could be the exogenous noise of an observed variable. As shown in Figure 2, for $\{X_1,X_2,X_3\}$, $W_1$ uses $L$ as the conditioning set and $W_2$ uses noise $E_1$ as the conditioning set; both are valid 1st-order CICA solutions, but $W_2$ effectively swaps the roles of $L$ and $E_1$, making its solution matrix **denser**.

The paper introduces sparsity as a discriminative signal. Lemma 6/7 states: $I-B_{X,X}$ itself is a $p_{\min}(X)$-th order CICA solution with the true latent confounders $\mathrm{LPa}(X)$ as the conditioning set, and using the graph's acyclicity eliminates permutation/scaling ambiguity to uniquely recover $B_{X,X}$ from $W$. Then Lemma 8 proves: $I-B_{X,X}$ has the fewest non-zero elements ($\ell_0$ minimal) among all CICA solutions. It is worth emphasizing that the paper **does not assume the true causal structure itself is sparse** (it can be arbitrarily dense)—"sparsest solution corresponds to the true structure" is not a prior assumption but a **provably emergent property** within the CICA framework. Lemma 9 further provides a uniqueness condition (Condition 2: every observed variable has a "sibling that is mutually confounded but not directly caused by it"), ensuring the sparsest solution uniquely corresponds to $I-B_{X,X}$; even if Condition 2 fails, the support matrices remain identical, so the causal **structure** between observed variables is still identifiable. Combining these yields Theorem 1: all latents in $\mathrm{LPa}(X)$, edges from latents to observed variables, and edges between observed variables are identifiable.

**4. CICA-LiNGAM: Bottom-up Recursion from Causal Clusters to Hierarchical Graphs**

A single CICA only solves one causal cluster. For hierarchical structures where latents might not have observed children, the paper uses a **proxy variable selection** strategy (Lemma 10): take the highest causal order node in the current latent child set $S$, and use the corresponding component $Z_k$ of the sparsest CICA solution as a proxy for that latent variable, thereby recursively extending Theorem 1 to the whole hierarchy (Theorem 2: under Condition 1, the entire graph $G$ is fully identifiable, including latent positions and causal relationships).

The implemented algorithm CICA-LiNGAM (Algorithm 1) uses a bottom-up loop: maintain an active variable set $A$ (initially $X$), and in each round—① Use GIN conditions (Lemma 11) to identify causal clusters in the active set; ② Find the sparsest CICA solution for each cluster (Lemma 4 or 5), apply permutation to make the diagonal non-zero, and derive intra-cluster causal structure per Theorem 1; ③ Merge clusters sharing the same latent parent and determine if new latents need introduction; ④ Update the active set per Lemma 10; until $A$ is empty, returning the complete graph $G$.

The paper also clarifies the relationship between CICA and Independent Subspace Analysis (ISA) (§3.5): Theorem 3 proves that ISA **lacks sufficient information** in the presence of latent confounders—ISA only groups variables into "as independent as possible" irreducible subspaces but **does not constrain the connections within the subspace**. Thus, Figures 1a/1b belong to the same equivalence class under ISA ("ISA equivalent") and cannot be distinguished; CICA fills this gap by imposing constraints within the subspace (e.g., the sparsest 1st-order CICA solution for $\{X_2,X_4\}$ can identify the edge $X_2 \to X_4$).

## Key Experimental Results

### Main Results
Synthetic data were generated from four typical graph structures (Cases 1–4, including fully impure settings) satisfying Condition 1, with sample sizes $N \in \{5k, 10k, 20k\}$. Causal strengths were sampled uniformly from $[-2, -0.5] \cup [0.5, 2]$, and non-Gaussian noise was the square of an exponential distribution. Results are averaged over 10 runs. Baselines: RLCD, PO-LiNGAM, LaHME, CDHS. Metrics: Latent count error (↓), Correct ordering rate (↑), Causal edge F1 (↑).

Performance at 20k samples (Error ↓ / Ordering ↑ / F1 ↑):

| Graph Structure | Metric | Ours | LaHME | PO-LiNGAM | RLCD | CDHS |
|-----------------|--------|------|-------|-----------|------|------|
| Case 1 | Error↓ / Order↑ / F1↑ | **0.00** / **0.75** / **0.77** | 0.00 / 0.50 / 0.67 | 0.00 / 0.50 / 0.67 | 1.00 / 0.00 / 0.00 | 0.40 / 0.60 / 0.60 |
| Case 2 (Impure) | Error↓ / Order↑ / F1↑ | **0.00** / **0.66** / **0.72** | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 |
| Case 3 | Error↓ / Order↑ / F1↑ | 0.10 / **0.61** / **0.78** | 0.10 / 0.40 / 0.65 | 0.20 / 0.40 / 0.65 | 0.00 / 0.58 / 0.73 | 2.00 / 0.00 / 0.00 |
| Case 4 (Impure) | Error↓ / Order↑ / F1↑ | 0.10 / **0.68** / **0.74** | 0.10 / 0.36 / 0.67 | 2.00 / 0.00 / 0.00 | 0.70 / 0.20 / 0.23 | 2.00 / 0.00 / 0.00 |

Most notably, on **fully impure** Cases 2 and 4, baselines almost completely failed—CDHS produced no valid output because the "same-source proxy" condition was violated, LaHME/PO-LiNGAM's clustering steps failed, and RLCD was inapplicable to Cases 1/2 due to the requirement of at least 4 observed variables for rank tests. Only the proposed method consistently identified impure connections across all cases.

### Sparsity Gap Simulation
To verify the stability of "sparsest solution = true graph" under finite samples, the paper enumerates all $2^{n(n-1)/2}$ graphs for $n \in \{3,4,5,6\}$ observed variables and 1 latent variable with a fixed causal order $X_1 \to \cdots \to X_n$, counting non-zero elements in $n+1$ valid CICA solutions (from sparse to dense):

| $n$ | Graphs | Sum0 (True) | Sum1 | Sum2 | … | Densest |
|-----|--------|-------------|------|------|---|---------|
| 3 | 8 | 36 | 43 | 50 | … | 59 |
| 4 | 64 | 448 | 531 | 602 | … | 809 |
| 5 | 1024 | 10240 | 12015 | 13368 | … | 19758 |
| 6 | 32768 | 442368 | 513675 | 563799 | … | 895521 |

For $n=6$, the true solution $I-B_{X,X}$ averages $13.5$ non-zeros (Sum0/Graphs), which is **2.176 fewer non-zeros** (16.12% relative increase) than the second-sparsest solution. This indicates that the sparsity gap is a universal "generational gap," providing practical stability and robustness guarantees for CICA to find the true graph.

### Key Findings
- **Major Gain from Handling "Impure" Structures**: In scenarios like Case 2/4 where observed variables have edges and share latents, all baselines fail (error in latents/zero output), whereas CICA maintains correct identification—the core breakthrough over GIN/TIN/pure-assumption methods.
- **Sparsity Gap Remains Stable with Scale**: From $n=3$ to $n=6$, the true solution is always the sparsest and maintains a separable gap from the next sparsest, supporting the reliability of $\ell_0$ minimization.
- **Theoretical Consistency Caveat**: Synthetic graphs were constructed to satisfy identifiability (Condition 1/2), verifying algorithm behavior under "assumption met" scenarios; real-world experiments are in Appendix C and not detailed in the main text.

## Highlights & Insights
- **"Separating until conditional independence" is an elegant simplification**: OICA tries to separate all latent sources (expensive/unstable), while CICA only requires conditional independence given latents. This "doing one step less" yields a differentiable rank-deficiency proxy while preserving identifiability—the central pivot of the paper.
- **Sparsity as an Emergence, Not an Assumption**: The paper emphasizes that true graphs can be arbitrarily dense; the fact that "sparsest CICA solution = true solution" is **proven** within the framework (Lemma 8/9), not an imposed Occam's razor. This allows the method to work on dense true graphs, distinguishing it from works that assume structural sparsity.
- **Perspective of Unilateral vs. Bilateral Projection**: Attributing GIN/TIN failures to "unilateral projection" and proving it is a subset of bilateral projection (Lemma 1) is a clear and generalizable framework—applicable to any task using independence constraints for identification.
- **Bridging Representation Learning Uncertainty to Causal Identification via Sparsity**: CICA extends ICA's permutation/scaling ambiguity with a "latent set ambiguity" and then resolves it with sparsity. This approach of "relaxing representation then tightening via structural preference" provides a template for other latent variable models.

## Limitations & Future Work
- **Linear Mixing Assumption**: Core results are built on $X=AS$ linear mixing and LiNGAM; extending to non-linear settings is the main future direction (potentially using Chen 2024b/2025b).
- **Large Sample Requirement**: Synthetic experiments used 5k–20k samples; reliable estimation of high-order information and non-Gaussianity naturally requires more samples.
- **Strong Identifiability Conditions**: Condition 1 (3 neighbors, 2 children per latent), Condition 2 ("sibling" constraints), and rank-faithfulness must hold, which are hard to verify on real data.
- **Thin Evidence on Real Data**: The main text focuses on synthetic data, with real-world experiments relegated to Appendix C, making it hard to judge robustness against real-world noise/non-stationarity directly from the main body.

## Related Work & Insights
- **vs. GIN (Xie 2020) / TIN (Dai 2022)**: Both rely on unilateral projection $\omega^\top Y \perp\!\!\!\perp Z$ and pure children; this paper proves unilateral is a subset of bilateral and uses $\omega_1^\top Y \perp\!\!\!\perp \omega_2^\top Z$ for richer signals, identifying impure structures where GIN/TIN fail.
- **vs. OICA (Eriksson & Koivunen 2004) & High-order Cumulants**: Theoretically handle impurity but OICA is computationally expensive/unstable, and cumulants are outlier-sensitive/sample-intensive. CICA uses differentiable rank-deficiency/Triad proxies for a better balance.
- **vs. Rank-deficiency Family (Silva 2006 / Huang 2022 / Dong 2023, RLCD)**: Mostly linear Gaussian recovering up to Markov equivalence; this paper grafts rank-deficiency onto non-Gaussian + CICA to identify specific causal directions and coefficients. RLCD also requires ≥4 observed variables, while CICA handles smaller clusters.
- **vs. ISA (Local ISA-LiNG, Dai 2024)**: ISA only separates to irreducible subspaces without intra-subspace constraints, lacking information for latent confounders (Theorem 3, Figs 1a/1b are ISA equivalent); CICA fills this by imposing conditional independence within subspaces.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes CICA as a new principle, systematically breaking purity assumptions with "conditional independence + sparsest solution"; the unilateral vs. bilateral analysis is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive comparison with four baselines and sparsity gap simulations, though real data is in the appendix and samples are large.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory progresses logically (definition → ambiguity → sparsity-based disambiguation → recursive algorithm) with well-placed examples and clear arguments.
- Value: ⭐⭐⭐⭐⭐ Provides a tool for non-Gaussian causal discovery with latent confounders that balances identifiability and feasibility; results are solid and extensible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Causal Structure Learning in Hawkes Processes with Complex Latent Confounder Networks](causal_structure_learning_in_hawkes_processes_with_complex_latent_confounder_net.md)
- [\[AAAI 2026\] CaDyT: Causal Structure Learning for Dynamical Systems with Theoretical Score Analysis](../../AAAI2026/causal_inference/causal_structure_learning_for_dynamical_systems_with_theoretical_score_analysis.md)
- [\[ICLR 2026\] CARL: Preserving Causal Structure in Representation Learning](carl_preserving_causal_structure_in_representation_learning.md)
- [\[ICLR 2026\] Beyond DAGs: A Latent Partial Causal Model for Multimodal Learning](beyond_dags_a_latent_partial_causal_model_for_multimodal_learning.md)
- [\[ICLR 2026\] Score-based Greedy Search for Structure Identification of Partially Observed Causal Models](score-based_greedy_search_for_structure_identification_of_partially_observed_cau.md)

</div>

<!-- RELATED:END -->
