---
title: >-
  [Paper Note] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function
description: >-
  [ICML 2026][learning_theory][data-driven algorithm design] This paper utilizes "real algebraic geometry + first-order logic quantifier elimination" to provide the first provable generalization bound for multi-dimensional hyper-parameter tuning. It extends the Balcan 2025 framework, which was limited to scalar hyper-parameters, to arbitrary $p$-dimensions, bi-level validation l
tags:
  - ICML 2026
  - learning_theory
  - data-driven algorithm design
  - pseudo-dimension
  - quantifier elimination
  - multi-dimensional hyperparameter
  - semi-algebraic
date: 2026-05-08
content_hash: 5e6a3a421b1bb26e
---
# Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function

**Conference**: ICML 2026  
**arXiv**: [2602.02406](https://arxiv.org/abs/2602.02406)  
**Code**: None  
**Area**: Learning Theory / Automated Hyper-parameter Tuning / Data-driven Algorithm Design  
**Keywords**: data-driven algorithm design, pseudo-dimension, quantifier elimination, multi-dimensional hyperparameter, semi-algebraic

## TL;DR
This paper utilizes "real algebraic geometry + first-order logic quantifier elimination" to provide the first provable generalization bound for multi-dimensional hyper-parameter tuning. It extends the Balcan 2025 framework, which was limited to scalar hyper-parameters, to arbitrary $p$-dimensions, bi-level validation loss, and approximate inner-level optimization, while providing the first matching lower bound.

## Background & Motivation
**Background**: Hyper-parameter tuning in industry primarily relies on grid search, Bayesian optimization, and Hyperband. Theoretically, these either hold only for discrete grids or assume the loss is smooth with respect to hyper-parameters (which in reality is often piecewise, non-differentiable, or discontinuous). Since Balcan 2020, data-driven algorithm design has framed hyper-parameter selection as "Empirical Risk Minimization over an unknown problem distribution $\mathcal D$" and studied its pseudo-dimension.

**Limitations of Prior Work**: The strongest existing results (Balcan et al. 2025) have four limitations: (i) they can only handle scalar hyper-parameters $\mathcal A=\mathbb R$, as their geometric arguments rely on the oscillation or monotonicity of 1D curves; (ii) they only handle the degenerate case where "training loss = validation loss" ($f\equiv g$), violating basic model selection principles; (iii) they require strong regularity conditions like ELICQ to prevent pathological boundary topology; (iv) they lack matching lower bounds to determine if the bounds are tight.

**Key Challenge**: Practical ML pipelines almost always stack multiple regularization terms (Elastic Net $L_1+L_2$, weighted group lasso, weighted fused lasso, etc.), making $\alpha\in\mathbb R^p$ the norm. However, in multiple dimensions, critical sets are no longer simple curves but high-dimensional manifolds, causing geometric arguments to fail completely. Thus, "multi-dimensional + bi-level validation" is an unavoidable yet theoretically unsolved challenge.

**Goal**: (1) Establish a general learning theory complexity framework capable of handling arbitrary $p$-dimensional hyper-parameters and bi-level validation $f\neq g$; (2) Provide lower bounds that match the upper bounds; (3) Apply the framework to new learnable classes (weighted group/fused lasso) to demonstrate generality.

**Key Insight**: Abandon geometry and embrace model theory and real algebraic geometry. Observed that under a piecewise polynomial structure, the implicitly defined loss $\ell_\alpha(x)=\min_\theta f(x,\alpha,\theta)$ can be written as a "polynomial first-order logic (FOL)" formula. The quantifier elimination algorithm from Basu et al. 2006 can transform any fixed-depth FOL into a quantifier-free system of polynomial inequalities, which can then be used with the Goldberg-Jerrum (GJ) framework to estimate pseudo-dimension.

**Core Idea**: Encode the bi-level structure of "inner optimization + outer validation" into a polynomial FOL $(\forall\theta)(\exists\theta')[\dots]$. Perform quantifier elimination to obtain a Quantifier-Free Formula (QFF), and use GJ to calculate the pseudo-dimension, upgrading the previous 1D geometric analysis to high-dimensional algebraic analysis.

## Method

### Overall Architecture
The paper follows a logical chain: (1) Introduce a new general tool (Thm 4.1): for any function class describable by a polynomial FOL with fixed quantifier layers $K$, the pseudo-dim is controlled by $\mathcal O(p\prod(d_k{+}1)\log M + p^2\prod d_k\log\Delta)$ ($M$ is the number of atomic polynomials, $\Delta$ is the maximum degree); (2) Apply (1) to the training loss scenario $f\equiv g$ to obtain the Thm 5.1 upper bound $\mathcal O(pd\log(M_f{+}T_f{+}d)+p^2d\log\Delta_f)$, and provide a lower bound $\Omega(pd\log\Delta_f)$ (Thm 5.2) via bit-extraction and stabilization; (3) Extend to bi-level validation $f\neq g$ (Thm 6.1) and $\epsilon$-approximate inner-level versions (Prop 6.2); (4) Further remove the dependence on $d$ when an explicit solution path exists (§7); (5) Instantiate two new learnable problems: weighted group lasso and weighted fused lasso (§8).

### Key Designs

**1. Polynomial FOL → Pseudo-dimension General Tool (Thm 4.1): Translating "describable by finite-quantifier polynomial logic" into pseudo-dimension upper bounds**

After abandoning geometry, a language is needed that is strong enough to describe the $\arg\min$ of implicit optimization yet weak enough to be algorithmically eliminated. Polynomial FOL fits this niche, as almost all semi-algebraic losses can be encoded. For any threshold $t$, if $\mathbb I(f_\alpha(x)\geq t)$ is equivalent to a $(Q_1\theta^{[1]})\dots(Q_K\theta^{[K]}) P(\alpha,\theta^{[1]},\dots,\theta^{[K]})$ form with $K$ layers, applying the quantifier elimination algorithm yields an equivalent quantifier-free formula. This results in a pseudo-dim upper bound of $\mathcal O(p\prod(d_k+1)\log M + p^2\prod d_k\log\Delta)$. Compared to the original GJ 1993 bound, this removes the dependence on data dimension $q$ and a factor of $p$ from $p\log M$, providing significant improvements when $q\gg p$.

**2. FOL Encoding of Implicit Loss (Core step in Thm 5.1 / 6.1): Using $\forall/\exists$ to write implicit conditions involving $\arg\min$ as explicit logic formulas**

The geometric method in Balcan 2025 treats $\theta$ as a hidden dimension to be eliminated via curve intersection counting, requiring manual design for every problem. This paper uses quantifiers to include $\theta, \theta'$ in the logic, letting quantifier elimination handle them automatically. The training loss version uses $\Phi_{x,t}(\alpha)\triangleq(\forall\theta\in\mathbb R^d)[(\theta\in\Theta)\Rightarrow f_x(\alpha,\theta)\geq t]$ ($K=1$). The bi-level validation version expands the condition "not optimal equals existence of a better candidate" into $(\forall\theta)(\exists\theta')[\theta\notin\Theta\lor g_x(\alpha,\theta)\geq t\lor(\theta'\in\Theta\land f_x(\alpha,\theta')<f_x(\alpha,\theta))]$ ($K=2$). This shifts "manual elimination in geometry" to "automatic elimination in logic."

**3. Matching Lower Bound (Thm 5.2): Using "discretization penalty + bit-extraction" to prove the $pd\log\Delta_f$ dominant term is tight**

Bi-level $\arg\min$ prevents the direct use of classic bit extraction (continuous optimization might not hit specific discrete points). This paper constructs $N=pdB$ one-hot triplets $x^{(j,i,b)}\in\{0,1\}^{p\times d\times B}$ and designs $f(x,\alpha, \theta)$ with three terms: one uses $C\sum_m\prod_k(\theta_m-k)^2$ to force $\theta$ to the grid $\{0,\dots,K{-}1\}^d$, another aligns $\alpha$ coordinates with base-$K$ encodings of $\theta$, and a third inserts a bit-extracting polynomial $E_c$. This two-layer structure—"primary term forces discretization + secondary term implements bit encoding"—combined with stabilization arguments, achieves a shatter of $N=\Omega(pd\log\Delta_f)$.

### Loss & Training
This paper is purely theoretical and does not train a model; it provides upper and lower bounds for sample complexity in the sense of statistical learning. Formally: given $N\geq N(\epsilon,\delta)=\mathcal O(H^2/\epsilon^2\cdot(\text{Pdim}+\log(1/\delta)))$ i.i.d. problem instances, ERM yields $\hat\alpha$ satisfying $\mathbb E[\ell_{\hat\alpha}]\leq\inf_\alpha\mathbb E[\ell_\alpha]+\epsilon$.

## Key Experimental Results

### Main Results

| Problem Type | Previous Results | Ours | Gain |
| :--- | :--- | :--- | :--- |
| 1D Hyper-param + $f\equiv g$ | Geometric bound (Balcan 2025) | $\mathcal O(pd\log(M_f{+}T_f{+}d)+p^2d\log \Delta_f)$, no ELICQ | Generalized + weakened assumptions |
| Multi-dim ($p\geq 2$) + $f\equiv g$ | None | Upper bound + $\Omega(pd\log\Delta_f)$ lower bound | 0 → Learnable |
| Multi-dim + $f\neq g$ Bi-level | None | $\mathcal O(pd^2\log M_{\text{tot}}+p^2d^2\log\Delta_{\text{tot}})$ | 0 → Learnable |
| Approx. Inner $\epsilon$-min | None | Same order as exact | 0 → Learnable |
| Weighted Group / Fused LASSO | Not piecewise polynomial | Learnable within semi-algebraic class | 0 → Learnable |

### Ablation Study

| Simplified Condition | Bound Order | Description |
| :--- | :--- | :--- |
| Full FOL Encoding (Thm 5.1) | $pd\log(\cdot)+p^2d\log\Delta$ | Baseline |
| Explicit solution path (§7) | Removes dependence on $d$ | Lower order for LASSO/Ridge, partially matching lower bounds |
| Naive Goldberg-Jerrum 1993 | $p(p+q)\prod d_k(\log M+\log\Delta)$ | Includes $q$ and an extra $p$ factor; potentially orders of magnitude worse |

### Key Findings
- Bi-level $f\neq g$ incurs an additional factor of $d$ compared to single-level $f\equiv g$ ($pd \to pd^2$), arising from the extra $\exists\theta'$ quantifier layer. Approximate inner optimization does not further double this.
- Removing the $p$ factor in $p\log M$ stems from tighter "region counting → shattering" analysis rather than the elimination algorithm itself.
- Weighted group lasso is learnable despite not being piecewise polynomial because it remains within the semi-algebraic class, showing the FOL perspective is broader than geometric bounds.
- When an explicit solution path exists (e.g., LARS for LASSO), the $d$ factor can be completely removed, matching the $\Omega(pd\log\Delta_f)$ lower bound in the best case.
- The number of quantifier layers $K$ is the only parameter with exponential dependence in the complexity order; thus, bilevel ($K=2$) to trilevel ($K=3$) represents a qualitative leap, suggesting that "meta-meta-learning" is difficult to bound.

## Highlights & Insights
- "Representing implicit optimization as FOL and letting quantifier elimination prove the generalization bound" is a highly transferable paradigm applicable to bilevel optimization and meta-learning.
- Combining bit extraction with a "discretization penalty" like $\sum_m\prod_k(\theta_m-k)^2$ provides a reusable template for lower bounds of function classes with inner $\arg\min$.
- Providing the first learning guarantees for "weighted group/fused lasso" significantly broadens the family of analyzable data-driven hyper-parameter tuning problems.
- Removing the dependence on data dimension $q$ in Thm 4.1 is significant; in modern ML where $q \gg p$, the bound improves by several orders of magnitude.

## Limitations & Future Work
- The upper bound is a quadratic polynomial in $p, d$, which may not be tight for modern deep learning where parameters are very large.
- The quantifier-elimination path is exponentially complex in the number of quantifier layers $K$, causing bounds for $K \geq 3$ to explode.
- Only provides sample complexity, not a specific optimization algorithm; solving ERM on non-smooth targets remains a challenge.
- Absence of empirical experiments; theoretical results have not yet been compared with grid search/BO/Hyperband on real data.

## Related Work & Insights
- **vs Balcan et al. 2025**: They used 1D geometric curves, restricting $p=1$ and $f\equiv g$; this work opens $p\geq 1$ and $f\neq g$ via FOL + QE.
- **vs Goldberg-Jerrum 1993**: Thm 4.1 is strictly superior to the GJ FOL pseudo-dim bound by removing $q$ and a factor of $p$.
- **vs Bartlett et al. 2019 (bit extraction)**: Reuses the lower bound framework but extends it by anchoring continuous optimization to a discrete grid using a stabilization penalty.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses real algebraic geometry and FOL to solve multi-dimensional bi-level tuning; a breakthrough in theoretical methodology.
- Experimental Thoroughness: ⭐⭐ No empirical validation; uses problem instances only to demonstrate the framework.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with motivation paragraphs for each theorem and proof outlines in the main text.
- Value: ⭐⭐⭐⭐ Provides a general upper bound tool for the data-driven algorithm design community with high long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](../../ICLR2026/learning_theory/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)
- [\[ICML 2026\] Tree-Structured Orthonormal Decomposition of the Aitchison Simplex](tree-structured_orthonormal_decomposition_of_the_aitchison_simplex.md)
- [\[ICML 2026\] The Data Manifold under the Microscope](the_data_manifold_under_the_microscope.md)
- [\[ICML 2026\] On the Robustness of Langevin Dynamics to Score Function Error](on_the_robustness_of_langevin_dynamics_to_score_function_error.md)
- [\[ICML 2026\] Understanding the Parameter Space Geometry of Transformers Encoding Boolean Functions](understanding_the_parameter_space_geometry_of_transformers_encoding_boolean_func.md)

</div>

<!-- RELATED:END -->
