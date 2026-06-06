---
title: >-
  [Paper Note] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function
description: >-
  [ICML 2026][data-driven algorithm design] This work employs "real algebraic geometry + first-order logic quantifier elimination" to provide the first provable generalization bound for multi-dimensional hyperparameter tun…
tags:
  - "ICML 2026"
  - "data-driven algorithm design"
  - "pseudo-dimension"
  - "quantifier elimination"
  - "multi-dimensional hyperparameter"
  - "semi-algebraic"
date: 2026-05-08
content_hash: d3f0b8013b825251
---

# Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function

**Conference**: ICML 2026  
**arXiv**: [2602.02406](https://arxiv.org/abs/2602.02406)  
**Code**: None  
**Area**: Learning Theory / Automated Hyperparameter Tuning / Data-driven Algorithm Design  
**Keywords**: data-driven algorithm design, pseudo-dimension, quantifier elimination, multi-dimensional hyperparameter, semi-algebraic

## TL;DR
This work employs "real algebraic geometry + first-order logic quantifier elimination" to provide the first provable generalization bound for multi-dimensional hyperparameter tuning, extending the Balcan 2025 framework—which was limited to one-dimensional scalar hyperparameters—to arbitrary $p$ dimensions, bilevel validation losses, approximate inner optimization, and other practical scenarios. It also provides the first matching lower bound.

## Background & Motivation
**Background**: In industry, hyperparameter tuning mainly relies on grid search, Bayesian optimization, and Hyperband. However, theoretical guarantees either only apply to discrete grids or assume the loss is smooth with respect to hyperparameters, which is rarely the case in practice (where losses are often piecewise, non-differentiable, and discontinuous). Since Balcan 2020, data-driven algorithm design has formulated hyperparameter selection as empirical risk minimization over an unknown problem distribution $\mathcal D$ and studied its pseudo-dimension.

**Limitations of Prior Work**: The strongest existing result (Balcan et al. 2025) has four main limitations: (i) It only handles one-dimensional scalar hyperparameters $\mathcal A=\mathbb R$, as its geometric arguments rely on oscillation/monotonicity of 1D curves; (ii) It only addresses the degenerate case where "training loss = validation loss" ($f\equiv g$), violating basic model selection principles; (iii) It requires strong regularity conditions such as ELICQ to avoid pathological boundary topologies; (iv) It lacks matching lower bounds, making it unclear whether the bounds are tight.

**Key Challenge**: In real ML pipelines, multiple regularizers are typically stacked (e.g., elastic net $L_1+L_2$, weighted group lasso, weighted fused lasso), so $\alpha\in\mathbb R^p$ is the norm. In higher dimensions, the critical set is no longer a simple curve but a high-dimensional manifold, rendering geometric arguments ineffective. Thus, "multi-dimensional + bilevel validation" is a fundamental challenge that theory has yet to resolve.

**Goal**: (1) Establish a general learning-theoretic complexity framework that handles arbitrary $p$-dimensional hyperparameters and arbitrary bilevel validation ($f\neq g$); (2) Provide matching lower bounds; (3) Apply the framework to new learnable classes (weighted group/fused lasso) to demonstrate generality.

**Key Insight**: Abandon geometry and fully embrace model theory/real algebraic geometry. It is observed that for piecewise polynomial structures, the implicitly defined loss $\ell_\alpha(x)=\min_\theta f(x,\alpha,\theta)$ can be written as a "polynomial first-order logic (FOL)" formula. Basu et al. 2006's quantifier elimination algorithm can convert any fixed-depth FOL into a quantifier-free system of polynomial inequalities, enabling the use of the Goldberg-Jerrum (GJ) framework to estimate pseudo-dimension.

**Core Idea**: Encode the "inner optimization + outer validation" bilevel structure as a polynomial FOL $(\forall\theta)(\exists\theta')[\dots]$, perform quantifier elimination to obtain a quantifier-free formula (QFF), and then use the GJ framework to compute the pseudo-dimension. This upgrades previous geometric analysis (limited to 1D curves) to high-dimensional algebraic analysis.

## Method

### Overall Architecture
The paper follows a logical chain: (1) Presents a new general tool (Thm 4.1): any function class describable by a polynomial FOL with fixed quantifier depth $K$ has pseudo-dimension bounded by $\mathcal O(p\prod(d_k{+}1)\log M + p^2\prod d_k\log\Delta)$, where $M$ is the number of atomic polynomials and $\Delta$ is the maximal degree; (2) Applies (1) to the training loss scenario $f\equiv g$ to obtain the upper bound in Thm 5.1, $\mathcal O(pd\log(M_f{+}T_f{+}d)+p^2d\log\Delta_f)$, and uses bit-extraction + stabilization arguments to provide the lower bound $\Omega(pd\log\Delta_f)$ (Thm 5.2); (3) Extends to bilevel validation $f\neq g$ to obtain Thm 6.1 and the $\epsilon$-approximate inner version (Prop 6.2); (4) Removes dependence on $d$ when an explicit solution path exists (§7); (5) Instantiates two new learnable problems: weighted group lasso and weighted fused lasso (§8).

### Key Designs

1. **Polynomial FOL → Pseudo-dimension General Tool (Thm 4.1)**:

    - **Function**: Translates the broad condition of "describable by a finite-quantifier polynomial logic" directly into a pseudo-dimension upper bound, serving as the engine of the paper.
    - **Mechanism**: For any threshold $t$, if $\mathbb I(f_\alpha(x)\geq t)$ is equivalent to a fixed $K$-layer FOL $(Q_1\theta^{[1]})\dots(Q_K\theta^{[K]}) P(\alpha,\theta^{[1]},\dots,\theta^{[K]})$, Basu 2006's quantifier elimination yields an equivalent QFF with at most $M^{\prod(d_k+1)}\Delta^{\mathcal O(p)\prod d_k}$ atomic polynomials and degree at most $\Delta^{\mathcal O(\prod d_k)}$. Feeding this into the GJ framework gives the pseudo-dimension upper bound $\mathcal O(p\prod(d_k+1)\log M + p^2\prod d_k\log\Delta)$. Compared to the original Goldberg-Jerrum 1993 bound $\mathcal O(p(p{+}q)\prod d_k(\log M{+}\log\Delta))$, this result removes dependence on data dimension $q$ and improves $p\log M$ to $\log M$ without the $p$ factor, yielding significant differences when $q\gg p$ or the class has exponentially many pieces.
    - **Design Motivation**: After abandoning geometry, a "language" is needed that is strong enough to describe implicit optimization $\arg\min$ but weak enough for algorithmic quantifier elimination; polynomial FOL fits precisely—almost all semi-algebraic losses can be encoded, and the Basu algorithm ensures elimination complexity is only exponential in $K$ and polynomial in other parameters.

2. **FOL Encoding of Implicit Loss (Core of Thm 5.1 / 6.1 Proofs)**:

    - **Function**: Expresses implicit conditions involving $\arg\min$, such as "$\ell_\alpha(x)=\min_\theta f(x,\alpha,\theta)\geq t$" and "$\ell_\alpha(x)=\inf_{\theta\in\mathcal S(x,\alpha)}g(x,\alpha,\theta)\geq t$", as explicit polynomial FOL.
    - **Mechanism**: For the training loss case, uses $\Phi_{x,t}(\alpha)\triangleq(\forall\theta\in\mathbb R^d)[(\theta\in\Theta)\Rightarrow f_x(\alpha,\theta)\geq t]$ with quantifier depth $K=1$; for bilevel validation, expands $(A\Rightarrow B)\equiv(\neg A\lor B)$ and the "no better candidate" property of $\arg\min$ to obtain $(\forall\theta)(\exists\theta')[\theta\notin\Theta\lor g_x(\alpha,\theta)\geq t\lor(\theta'\in\Theta\land f_x(\alpha,\theta')<f_x(\alpha,\theta))]$ with $K=2$. For $\epsilon$-approximate inner optimization, simply replace $<$ with $f(x,\alpha,\theta)>f(x,\alpha,\theta')+\epsilon$ in the third term (Prop 6.2).
    - **Design Motivation**: The geometric method of Balcan 2025 essentially treats $\theta$ as a hidden dimension to be eliminated via curve intersection counting; here, $\forall/\exists$ quantifiers encode $\theta,\theta'$ directly in logic, letting quantifier elimination handle $\theta$ automatically, without manual geometric arguments for each problem.

3. **Matching Lower Bound (Thm 5.2)**:

    - **Function**: Proves that the leading term $pd\log\Delta_f$ in Thm 5.1 is tight and cannot be further reduced.
    - **Mechanism**: Inspired by Bartlett 2019's bit-extraction technique, constructs $N=pdB$ one-hot triplets $x^{(j,i,b)}\in\{0,1\}^{p\times d\times B}$ ($B=\lfloor\log_2 K\rfloor$, $K=\lfloor\Delta_f/2\rfloor$), and designs $f(x,\alpha,\theta)$ as the sum of three terms: "locks $\theta$ to grid points $\{0,\dots,K{-}1\}^d$", "aligns specified coordinates of $\alpha$ with the base-$K$ encoding of $\theta$", and "inserts a bit-extracting polynomial $E_c$". Stabilization arguments ensure the continuous optimization $\arg\min$ and the discrete grid $\mathcal K^d$ differ by less than $0.1$, so a threshold $\tau=0.25$ can realize any $2^N$ bit labeling—shattering $N=\Omega(pd\log\Delta_f)$.
    - **Design Motivation**: The bilevel $\arg\min$ prevents direct use of Bartlett 2019's classic bit extraction (continuous optimization may not land on discrete points), so $C\sum_m\prod_k(\theta_m-k)^2$ is used to strongly anchor $\theta$ to grid points, and the bit-extraction polynomial $E_c$ is scaled to be one order of magnitude smaller, forming a two-level structure: "main term enforces discreteness + secondary term encodes bits".

### Loss & Training
No models are trained; the results are sample complexity upper and lower bounds in the sense of statistical learning theory. Specifically: for $N\geq N(\epsilon,\delta)=\mathcal O(H^2/\epsilon^2\cdot(\text{Pdim}+\log(1/\delta)))$ i.i.d. problem instances, ERM yields $\hat\alpha$ such that $\mathbb E[\ell_{\hat\alpha}]\leq\inf_\alpha\mathbb E[\ell_\alpha]+\epsilon$.

## Key Experimental Results
This is a purely theoretical work; "experiments" refer to learnability results for several new problems. The "experiment table" is a mapping from "problem type" to "bound type".

### Main Results

| Problem Type | Previous Result | This Work | Gain |
|---------|---------|----------|------|
| 1D hyperparameter + $f\equiv g$ | Geometric bound (Balcan 2025) | $\mathcal O(pd\log(M_f{+}T_f{+}d)+p^2d\log\Delta_f)$, removes ELICQ | Generalization + weaker assumptions |
| Multi-dimensional ($p\geq 2$) + $f\equiv g$ | None | Upper bound + $\Omega(pd\log\Delta_f)$ lower bound | 0→learnable |
| Multi-dimensional + $f\neq g$ bilevel validation | None | $\mathcal O(pd^2\log M_{\text{tot}}+p^2d^2\log\Delta_{\text{tot}})$ | 0→learnable |
| Approximate inner $\epsilon$-min | None | Same order as exact | 0→learnable |
| Weighted Group / Fused LASSO | No piecewise polynomial | Still learnable as semi-algebraic | 0→learnable |

### Ablation Study

| Simplification | Bound Order | Note |
|---------|-----------|------|
| Full FOL encoding (Thm 5.1) | $pd\log(\cdot)+p^2d\log\Delta$ | baseline |
| Explicit solution path (§7) | Removes $d$ dependence | For LASSO/Ridge, can reduce order, partially matches known lower bounds |
| Directly applying Goldberg-Jerrum 1993 | $p(p+q)\prod d_k(\log M+\log\Delta)$ | Adds $q$ and an extra $p$ factor, possibly several orders of magnitude worse |

### Key Findings
- Bilevel $f\neq g$ incurs an extra $d$ factor compared to single-level $f\equiv g$ ($pd\to pd^2$), due to the additional $\exists\theta'$ quantifier; approximate inner optimization does not further increase the order.
- The removal of the $p$ factor in $p\log M$ is due to a tighter "region counting → shattering" analysis, not the elimination algorithm itself.
- Weighted group lasso, even if not piecewise polynomial, remains learnable as a semi-algebraic class, indicating the FOL perspective covers much more than classical geometric bounds.
- When explicit solution paths exist (e.g., LASSO's LARS path), the $d$ factor can be completely removed, matching the lower bound $\Omega(pd\log\Delta_f)$ in Thm 5.2 in the optimal case.
- Quantifier depth $K$ is the only parameter with exponential complexity; thus, moving from bilevel ($K=2$) to trilevel ($K=3$) causes a qualitative jump, suggesting that "meta-meta-learning" is unlikely to admit practical bounds.

## Highlights & Insights
- "Expressing implicit optimization problems as FOL and letting quantifier elimination prove generalization bounds for you" is a highly transferable paradigm—the same idea applies to bilevel optimization, meta-learning, and statistical complexity analysis of implicit deep models.
- Combining Bartlett 2019's bit extraction with the discretization penalty $\sum_m\prod_k(\theta_m-k)^2$ provides a reusable template for lower bounds in function classes with inner $\arg\min$.
- Provides the first learnability guarantees for "weighted group/fused lasso" losses that are not piecewise-polynomial but are semi-algebraic, greatly expanding the analyzable family for data-driven hyperparameter tuning.
- Removing dependence on data dimension $q$ in Thm 4.1 is a significant improvement: in modern ML applications where $q\gg p$ (e.g., image/tabular data), the bound can improve by several orders of magnitude.
- The authors clearly distinguish between "problems with explicit solution paths (LASSO)" and "black-box only" cases, providing strictly smaller bounds for the former, allowing theory to tighten as problem structure allows.

## Limitations & Future Work
- The upper bound is quadratic in $p,d$ rather than linear, which may not be tight for modern deep learning scenarios with large hyperparameter/model parameter counts.
- The quantifier elimination approach is exponential in quantifier depth $K$, so for trilevel or higher bilevel (meta-meta-learning), the bound grows rapidly.
- Only sample complexity is provided; no concrete optimization algorithm is given—ERM for non-smooth objectives remains challenging and requires heuristic methods like SGD.
- No empirical experiments: theoretical results have not yet been compared with grid search, BO, or Hyperband on real data.

## Related Work & Insights
- **vs Balcan et al. 2025**: Their geometric curve arguments restrict to $p=1$ and $f\equiv g$; this work uses FOL + QE to simultaneously address $p\geq 1$ and $f\neq g$.
- **vs Goldberg-Jerrum 1993**: Thm 4.1 here strictly improves on GJ's FOL pseudo-dim bound, removing $q$ and an extra $p$ factor.
- **vs Bartlett et al. 2019 (bit extraction)**: Reuses their lower bound framework, but anchors continuous optimization to discrete grids via stabilization penalties, a key extension of the original technique.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Real algebraic geometry + FOL to tackle multi-dimensional + bilevel hyperparameter tuning—a methodological breakthrough in theory
- Experimental Thoroughness: ⭐⭐ No empirical results, only problem instantiations to demonstrate the framework
- Writing Quality: ⭐⭐⭐⭐ Clear structure, each theorem is preceded by a motivation section, proof sketches are provided in the main text
- Value: ⭐⭐⭐⭐ Provides a general upper bound tool for the entire data-driven algorithm design community, with long-term value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Provably Data-Driven Projection Method for Quadratic Programming](../../AAAI2026/others/provably_data-driven_projection_method_for_quadratic_programming.md)
- [\[ICLR 2026\] An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](../../ICLR2026/others/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)
- [\[AAAI 2026\] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer](../../AAAI2026/others/synweather_weather_observation_data_synthesis_across_multiple_regions_and_variab.md)
- [\[ICML 2026\] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)
- [\[ICLR 2026\] t-SNE Exaggerates Clusters, Provably](../../ICLR2026/others/t-sne_exaggerates_clusters_provably.md)

</div>

<!-- RELATED:END -->
