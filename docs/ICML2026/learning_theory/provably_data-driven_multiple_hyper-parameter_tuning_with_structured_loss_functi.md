---
title: >-
  [Paper Note] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function
description: >-
  [ICML 2026][learning_theory][data-driven algorithm design] This paper employs "real algebraic geometry + first-order logic (FOL) quantifier elimination" to provide the first provable generalization bounds for multi-dimensional hyperparameter tuning. It generalizes the Balcan 2025 framework—which was previously limited to one-dimensional scalar hyperparameters—to cover arbitrar
tags:
  - ICML 2026
  - learning_theory
  - data-driven algorithm design
  - pseudo-dimension
  - quantifier elimination
  - multi-dimensional hyperparameter
  - semi-algebraic
date: 2026-05-08
content_hash: 2c938fa0360b5afe
---
# Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function

**Conference**: ICML 2026  
**arXiv**: [2602.02406](https://arxiv.org/abs/2602.02406)  
**Code**: None  
**Area**: Learning Theory / Automated Hyperparameter Tuning / Data-driven Algorithm Design  
**Keywords**: data-driven algorithm design, pseudo-dimension, quantifier elimination, multi-dimensional hyperparameter, semi-algebraic

## TL;DR
This paper employs "real algebraic geometry + first-order logic (FOL) quantifier elimination" to provide the first provable generalization bounds for multi-dimensional hyperparameter tuning. It generalizes the Balcan 2025 framework—which was previously limited to one-dimensional scalar hyperparameters—to cover arbitrary $p$-dimensional parameters, bi-level validation loss, and approximate inner optimization, while establishing a matching lower bound.

## Background & Motivation
**Background**: In industrial settings, hyperparameter tuning primarily relies on grid search, Bayesian optimization, and Hyperband. However, theoretical guarantees typically hold only for discrete grids or assume that the loss is smooth with respect to hyperparameters (whereas real losses are often piecewise, non-differentiable, or discontinuous). The "data-driven algorithm design" paradigm initiated by Balcan 2020 formulates hyperparameter selection as empirical risk minimization (ERM) over an unknown problem distribution $\mathcal D$ and investigates its pseudo-dimension.

**Limitations of Prior Work**: The current strongest results (Balcan et al. 2025) suffer from four limitations: (i) they only handle one-dimensional scalar hyperparameters $\mathcal A=\mathbb R$ because their geometric arguments rely on the oscillation or monotonicity of 1D curves; (ii) they only handle the degenerate case where training loss equals validation loss ($f\equiv g$), violating basic model selection principles; (iii) they require strong regularity conditions like ELICQ to prevent boundary topological pathologies; and (iv) they lack matching lower bounds, making it unclear if the bounds are tight.

**Key Challenge**: Practical ML pipelines almost always involve multiple stacked regularization terms (Elastic Net $L_1+L_2$, weighted group lasso, weighted fused lasso, etc.), where $\alpha\in\mathbb R^p$ is the norm. In higher dimensions, critical sets are no longer simple curves but high-dimensional manifolds, causing geometric arguments to fail entirely. Thus, "multi-dimensional + bi-level validation" remains a critical yet theoretically unresolved bottleneck.

**Goal**: (1) Establish a universal learning theory complexity framework capable of handling arbitrary $p$-dimensional hyperparameters and bi-level validation $f\neq g$. (2) Provide lower bounds matching the upper bounds. (3) Apply the framework to new learnable classes (weighted group/fused lasso) to demonstrate generality.

**Key Insight**: Moving away from geometry to fully embrace model theory and real algebraic geometry. It is observed that under a piecewise polynomial structure, the implicitly defined loss $\ell_\alpha(x)=\min_\theta f(x,\alpha,\theta)$ can be written as a "polynomial first-order logic (FOL)" formula. The quantifier elimination algorithm of Basu et al. 2006 can transform any fixed-depth FOL into a quantifier-free system of polynomial inequalities, which can then be used with the Goldberg-Jerrum (GJ) framework to estimate pseudo-dimension.

**Core Idea**: Encode the bi-level structure of "inner optimization + outer validation" into a polynomial FOL formula $(\forall\theta)(\exists\theta')[\dots]$. Perform quantifier elimination to obtain a Quantifier-Free Formula (QFF), and then apply GJ to calculate the pseudo-dimension, thereby upgrading the previous 1D geometric analysis into high-dimensional algebraic analysis.

## Method

### Overall Architecture
The paper follows a logical chain: (1) Introduce Thm 4.1 as a new general tool: for any function class describable by a polynomial FOL with fixed quantifier depth $K$, its pseudo-dimension is controlled by $\mathcal O(p\prod(d_k{+}1)\log M + p^2\prod d_k\log\Delta)$, where $M$ is the number of atomic polynomials and $\Delta$ is the maximum degree. (2) Apply (1) to the training loss scenario $f\equiv g$ to obtain the Thm 5.1 upper bound $\mathcal O(pd\log(M_f{+}T_f{+}d)+p^2d\log\Delta_f)$ and provide a matching lower bound $\Omega(pd\log\Delta_f)$ through bit-extraction and stabilization arguments (Thm 5.2). (3) Extend this to bi-level validation $f\neq g$ in Thm 6.1 and an $\epsilon$-approximate inner version in Prop 6.2. (4) Further remove reliance on $d$ when an explicit solution path exists (§7). (5) Instantiate two new learnable problems: weighted group lasso and weighted fused lasso (§8).

### Key Designs

**1. Polynomial FOL → Pseudo-dimension General Tool (Thm 4.1): Direct translation from poly-logic to pseudo-dimension bounds**

By abandoning geometry, a language is needed that is strong enough to describe the implicit $\arg\min$ of optimization yet weak enough for algorithmic quantifier elimination. Polynomial FOL serves this purpose, as nearly all semi-algebraic losses can be encoded. For any threshold $t$, if $\mathbb I(f_\alpha(x)\geq t)$ is equivalent to a $(Q_1\theta^{[1]})\dots(Q_K\theta^{[K]}) P(\alpha,\theta^{[1]},\dots,\theta^{[K]})$ form with $K$ layers, the quantifier-free formula obtained has at most $M^{\prod(d_k+1)}\Delta^{\mathcal O(p)\prod d_k}$ atomic polynomials with degree $\leq \Delta^{\mathcal O(\prod d_k)}$. This yields a tighter pseudo-dimension bound compared to the original GJ 1993 bound, removing the dependency on data dimension $q$ and the $p$ factor from $p \log M$, which is significant when $q \gg p$.

**2. FOL Encoding of Implicit Loss (Mechanism of Thm 5.1 / 6.1): Using $\forall/\exists$ to explicate $\arg\min$ conditions**

The geometric method in Balcan 2025 treats $\theta$ as a hidden dimension to be eliminated via curve intersection counting, requiring manual geometric proofs for every problem. Ours replaces this with quantifiers for $\theta, \theta'$ in the logic language, letting quantifier elimination automate the process. Training loss uses $\Phi_{x,t}(\alpha)\triangleq(\forall\theta\in\mathbb R^d)[(\theta\in\Theta)\Rightarrow f_x(\alpha,\theta)\geq t]$ ($K=1$). The bi-level version utilizes $(A\Rightarrow B)\equiv(\neg A\lor B)$ and the property that a point is an $\arg\min$ unless there exists a better candidate, expanding to $(\forall\theta)(\exists\theta')[\theta\notin\Theta\lor g_x(\alpha,\theta)\geq t\lor(\theta'\in\Theta\land f_x(\alpha,\theta')<f_x(\alpha,\theta))]$ ($K=2$). For $\epsilon$-approximate inner optimization, the condition is updated to $f(x,\alpha,\theta)>f(x,\alpha,\theta')+\epsilon$ (Prop 6.2). This transition from manual geometric elimination to automated logical elimination is the key to solving higher-dimensional bi-level problems.

**3. Matching Lower Bound (Thm 5.2): Proving the dominance of $pd\log\Delta_f$ via discretization penalties + bit-extraction**

The bi-level $\arg\min$ prevents the direct use of Bartlett's 2019 classic bit extraction (as continuous optimization may not hit specific discrete points). Ours constructs $N=pdB$ one-hot triplets $x^{(j,i,b)}\in\{0,1\}^{p\times d\times B}$ and designs $f(x,\alpha,\theta)$ with three terms: a penalty $C\sum_m\prod_k(\theta_m-k)^2$ to force $\theta$ onto the grid $\{0,\dots,K{-}1\}^d$, a term to align $\alpha$ coordinates with base-$K$ encodings of $\theta$, and a bit-extracting polynomial $E_c$ with a smaller weight. This two-layer structure—"primary term forces discretization + secondary term implements bit encoding"—combined with stabilization arguments ensures that the continuous $\arg\min$ stays within $0.1$ of the discrete grid, allowing the shattering of $N=\Omega(pd\log\Delta_f)$ instances with threshold $\tau=0.25$.

### Loss & Training
Ours does not train a model but provides sample complexity bounds in the sense of statistical learning. Formally: given $N\geq N(\epsilon,\delta)=\mathcal O(H^2/\epsilon^2\cdot(\text{Pdim}+\log(1/\delta)))$ i.i.d. problem instances, the ERM solution $\hat\alpha$ satisfies $\mathbb E[\ell_{\hat\alpha}]\leq\inf_\alpha\mathbb E[\ell_\alpha]+\epsilon$.

## Key Experimental Results

### Main Results

| Problem Type | Previous Results | Ours | Gain |
| :--- | :--- | :--- | :--- |
| 1D Hyperparameter + $f\equiv g$ | Geometric bound (Balcan 2025) | $\mathcal O(pd\log(M_f{+}T_f{+}d)+p^2d\log\Delta_f)$, no ELICQ | Generalization + relaxed assumptions |
| Multi-dim ($p\geq 2$) + $f\equiv g$ | None | Upper bound + $\Omega(pd\log\Delta_f)$ lower bound | 0 → Learnable |
| Multi-dim + $f\neq g$ Bi-level | None | $\mathcal O(pd^2\log M_{\text{tot}}+p^2d^2\log\Delta_{\text{tot}})$ | 0 → Learnable |
| Approximate Inner $\epsilon$-min | None | Same order as exact | 0 → Learnable |
| Weighted Group / Fused LASSO | Not piecewise polynomial | Learnable within semi-algebraic class | 0 → Learnable |

### Ablation Study

| Simplified Condition | Bound Order | Note |
| :--- | :--- | :--- |
| Full FOL Encoding (Thm 5.1) | $pd\log(\cdot)+p^2d\log \Delta$ | baseline |
| Explicit solution path (§7) | Remove dependence on $d$ | For LASSO/Ridge, matches lower bound |
| Direct GJ 1993 application | $p(p+q)\prod d_k(\log M+\log\Delta)$ | Adds $q$ and a factor of $p$; significantly looser |

### Key Findings
- Bi-level $f\neq g$ incurs an additional factor of $d$ compared to single-level $f\equiv g$ ($pd \to pd^2$), arising from the extra $\exists\theta'$ quantifier layer. Approximate inner optimization does not double this complexity further.
- The removal of the $p$ factor from $p \log M$ stems from tighter "cell counting → shattering" analysis rather than the elimination algorithm itself.
- Weighted group lasso is learnable despite not being piecewise polynomial because it remains semi-algebraic, showing that the FOL perspective is far broader than geometric bounds.
- When an explicit solution path exists (e.g., LARS for LASSO), the $d$ factor can be completely removed, achieving an optimal bound matching the $\Omega(pd\log\Delta_f)$ lower bound.
- The number of quantifier layers $K$ is the only parameter where complexity grows exponentially; thus, moving from bi-level ($K=2$) to tri-level ($K=3$) significantly increases the bound, suggesting that "meta-meta-learning" is difficult to bound usefully.

## Highlights & Insights
- The paradigm of "writing implicit optimization as FOL and letting quantifier elimination prove the generalization bound" is highly transferable—it can be used for bi-level optimization, meta-learning, and the complexity analysis of implicit deep models.
- Combining Bartlett 2019's bit extraction with a "discretization penalty" provides a reusable template for lower bounds of function classes containing an inner $\arg\min$.
- Providing the first learning guarantees for non-piecewise-polynomial losses like "weighted group/fused lasso" greatly broadens the family of analyzable data-driven hyperparameter tuning problems.
- Removing dependency on data dimension $q$ in Thm 4.1 is a major practical improvement, as modern ML applications often have $q \gg p$, leading to bounds that are orders of magnitude tighter.
- The distinction between "analytic solution paths (LASSO)" and "black-box optimization" allows the theory to tighten as problem structure becomes clearer.

## Limitations & Future Work
- Upper bounds are quadratic rather than linear in $p$ and $d$, which might be loose for modern deep learning where both are large.
- The quantifier-elimination path is exponentially complex with respect to the number of quantifier layers $K$, causing bounds for tri-level or higher structures to swell.
- Only sample complexity is provided; no specific optimization algorithm is given. ERM on non-smooth objectives remains difficult and relies on SGD-type heuristics.
- Absence of empirical experiments: Theoretical results have not been compared against grid search/BO/Hyperband on real-world data.

## Related Work & Insights
- **vs Balcan et al. 2025**: They use 1D geometric proofs limited to $p=1$ and $f\equiv g$; Ours uses FOL + QE to handle $p \geq 1$ and $f \neq g$ simultaneously.
- **vs Goldberg-Jerrum 1993**: Thm 4.1 is strictly tighter for FOL pseudo-dimension bounds by removing $q$ and one factor of $p$.
- **vs Bartlett et al. 2019**: Reuses the bit extraction framework but anchors continuous optimization to a discrete grid via stabilization penalties, a key extension for $\arg\min$-based classes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Uses real algebraic geometry and FOL to solve multi-dimensional bi-level tuning; a methodological breakthrough.
- Experimental Thoroughness: ⭐⭐ Purely theoretical; framework demonstrated via problem instances only.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with motivational sections before theorems; proof sketches provided in-text.
- Value: ⭐⭐⭐⭐ Provides general upper bound tools for the data-driven algorithm design community; high long-term value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](../../ICLR2026/learning_theory/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](../../NeurIPS2025/learning_theory/adaptive_data_analysis_for_growing_data.md)
- [\[ICML 2026\] Expectation Consistency Loss: Rethink Confidence Calibration under Covariate Shift](expectation_consistency_loss_rethink_confidence_calibration_under_covariate_shif.md)
- [\[ICLR 2026\] Function Spaces Without Kernels: Learning Compact Hilbert Space Representations](../../ICLR2026/learning_theory/function_spaces_without_kernels_learning_compact_hilbert_space_representations.md)
- [\[ICML 2025\] Multiple-Policy Evaluation via Density Estimation](../../ICML2025/learning_theory/multiple-policy_evaluation_via_density_estimation.md)

</div>

<!-- RELATED:END -->
