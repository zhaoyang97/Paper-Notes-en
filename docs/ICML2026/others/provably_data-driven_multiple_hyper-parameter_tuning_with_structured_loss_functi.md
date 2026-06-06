---
title: >-
  [Paper Note] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function
description: >-
  [ICML 2026][data-driven algorithm design] This paper uses "real algebraic geometry + first-order logic quantifier elimination" to provide the first provable generalization bound for multi-dimensional hyperparameter tunin…
tags:
  - "ICML 2026"
  - "data-driven algorithm design"
  - "pseudo-dimension"
  - "quantifier elimination"
  - "multi-dimensional hyperparameter"
  - "semi-algebraic"
date: 2026-05-08
content_hash: 94d5e732ff93df07
---

# Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function

**Conference**: ICML 2026  
**arXiv**: [2602.02406](https://arxiv.org/abs/2602.02406)  
**Code**: None  
**Area**: Learning Theory / Automated Machine Learning / Data-driven Algorithm Design  
**Keywords**: data-driven algorithm design, pseudo-dimension, quantifier elimination, multi-dimensional hyperparameter, semi-algebraic

## TL;DR
This paper uses "real algebraic geometry + first-order logic quantifier elimination" to provide the first provable generalization bound for multi-dimensional hyperparameter tuning. It generalizes the Balcan 2025 framework, which was previously limited to one-dimensional scalar hyperparameters, to various practical scenarios including arbitrary $p$ dimensions, bilevel validation loss, and approximate inner optimization, while establishing the first matching lower bound.

## Background & Motivation
**Background**: Hyperparameter tuning in industry primarily relies on the "trio" of grid search, Bayesian optimization, and Hyperband. However, theoretically, these either hold only for discrete grids or assume that the loss is smooth with respect to hyperparameters (whereas in practice, it is often piecewise, non-differentiable, and discontinuous). Data-driven algorithm design since Balcan 2020 formulates hyperparameter selection as "Empirical Risk Minimization (ERM) over an unknown problem distribution $\mathcal{D}$" and investigates its pseudo-dimension.

**Limitations of Prior Work**: The strongest existing results (Balcan et al. 2025) have four major limitations: (i) they can only handle one-dimensional scalar hyperparameters $\mathcal{A}=\mathbb{R}$, as their geometric arguments rely on the oscillation or monotonicity of 1D curves; (ii) they can only handle the degenerate case where "training loss = validation loss" ($f\equiv g$), violating basic model selection principles; (iii) they require strong regularity conditions such as ELICQ to prevent pathological boundary topology; (iv) they lack matching lower bounds, making it impossible to determine if the bounds are tight.

**Key Challenge**: Practical Machine Learning (ML) pipelines almost always involve stacking multiple regularization terms (e.g., Elastic Net $L_1+L_2$, weighted group lasso, weighted fused lasso), where $\alpha\in\mathbb{R}^p$ is standard. However, in higher dimensions, critical sets are no longer simple curves but high-dimensional manifolds, causing geometric arguments to fail completely. Therefore, "multi-dimensional + bilevel validation" is a critical but theoretically unresolved challenge.

**Goal**: (1) Establish a general learning theory complexity framework capable of handling arbitrary $p$-dimensional hyperparameters and arbitrary bilevel validation ($f\neq g$); (2) Provide lower bounds that match the upper bounds; (3) Apply the framework to new learnable classes (weighted group/fused lasso) to demonstrate its universality.

**Key Insight**: Abandon geometry and embrace model theory/real algebraic geometry. It is observed that under a piecewise polynomial structure, the implicitly defined loss $\ell_\alpha(x)=\min_\theta f(x,\alpha,\theta)$ can be expressed as a "polynomial first-order logic (FOL)" formula. The quantifier elimination algorithm from Basu et al. 2006 can transform any fixed-depth FOL into a quantifier-free system of polynomial inequalities, which can then be used within the Goldberg-Jerrum (GJ) framework to estimate pseudo-dimension.

**Core Idea**: Encode the bilevel structure of "inner optimization + outer validation" into a polynomial FOL $(\forall\theta)(\exists\theta')[\dots]$, perform quantifier elimination to obtain a Quantifier-Free Formula (QFF), and then apply GJ to calculate the pseudo-dimension. This upgrades previous geometric analysis limited to 1D curves to high-dimensional algebraic analysis.

## Method

### Overall Architecture
The paper follows a logical chain: (1) It presents a new general tool (Thm 4.1): for any function class describable by a polynomial FOL with fixed quantifier levels $K$, its pseudo-dimension is controlled by $\mathcal{O}(p\prod(d_k{+}1)\log M + p^2\prod d_k\log\Delta)$, where $M$ is the number of atomic polynomials and $\Delta$ is the maximum degree; (2) It applies (1) to the training loss scenario ($f\equiv g$) to obtain the Thm 5.1 upper bound $\mathcal{O}(pd\log(M_f{+}T_f{+}d)+p^2d\log\Delta_f)$ and provides a lower bound $\Omega(pd\log\Delta_f)$ (Thm 5.2) using bit-extraction and stabilization arguments; (3) It generalizes this to bilevel validation ($f\neq g$) in Thm 6.1 and an $\epsilon$-approximate inner version in Prop 6.2; (4) It further removes the dependency on $d$ when an explicit solution path exists (§7); (5) It instantiates two new learnable problems: weighted group lasso and weighted fused lasso (§8).

### Key Designs

1.  **Polynomial FOL → Pseudo-dimension Universal Tool (Thm 4.1)**:
    *   **Function**: Translates the broad condition of "being describable by finite-quantifier polynomial logic" directly into a pseudo-dimension upper bound, serving as the core engine of the work.
    *   **Mechanism**: For any threshold $t$, if $\mathbb{I}(f_\alpha(x)\geq t)$ is equivalent to a fixed $K$-level FOL $(Q_1\theta^{[1]})\dots(Q_K\theta^{[K]}) P(\alpha,\theta^{[1]},\dots,\theta^{[K]})$, applying Basu 2006's quantifier elimination yields an equivalent QFF. The number of atomic polynomials is $\leq M^{\prod(d_k+1)}\Delta^{\mathcal{O}(p)\prod d_k}$ and the degree is $\leq \Delta^{\mathcal{O}(\prod d_k)}$. Feeding this into the GJ framework yields a pseudo-dimension upper bound of $\mathcal{O}(p\prod(d_k+1)\log M + p^2\prod d_k\log\Delta)$. Compared to the original Goldberg-Jerrum 1993 bound $\mathcal{O}(p(p{+}q)\prod d_k(\log M{+}\log\Delta))$, this result removes the dependency on the data dimension $q$ and removes the $p$ factor in front of $\log M$, which is significant when $q\gg p$ or when classes contain exponential pieces.
    *   **Design Motivation**: To replace geometry, a language was needed that is strong enough to describe the $\arg\min$ of implicit optimization but weak enough to be algorithmically eliminated; polynomial FOL fits these requirements as almost all semi-algebraic losses can be encoded, and the Basu algorithm guarantees that elimination complexity is exponential only in $K$.

2.  **FOL Encoding of Implicit Losses (Core Mechanism for Thm 5.1 / 6.1)**:
    *   **Function**: Represents implicit conditions involving $\arg\min$, such as "$\ell_\alpha(x)=\min_\theta f(x,\alpha,\theta)\geq t$" and "$\ell_\alpha(x)=\inf_{\theta\in\mathcal{S}(x,\alpha)}g(x,\alpha,\theta)\geq t$", as explicit polynomial FOL.
    *   **Mechanism**: For the training loss version, it uses $\Phi_{x,t}(\alpha)\triangleq(\forall\theta\in\mathbb{R}^d)[(\theta\in\Theta)\Rightarrow f_x(\alpha,\theta)\geq t]$, where $K=1$. For the bilevel validation version, it utilizes $(A\Rightarrow B)\equiv(\neg A\lor B)$ and the property that non-optimality equals the existence of a better candidate to obtain $(\forall\theta)(\exists\theta')[\theta\notin\Theta\lor g_x(\alpha,\theta)\geq t\lor(\theta'\in\Theta\land f_x(\alpha,\theta')<f_x(\alpha,\theta))]$, where $K=2$. The $\epsilon$-approximate version simply modifies the third term to $f(x,\alpha,\theta)>f(x,\alpha,\theta')+\epsilon$ (Prop 6.2).
    *   **Design Motivation**: Balcan 2025's geometric method essentially treats $\theta$ as an eliminable hidden dimension and counts curve intersections. This work directly incorporates $\theta, \theta'$ into the logic language using $\forall/\exists$, effectively letting quantifier elimination handle $\theta$ without requiring manual geometric arguments for each specific problem.

3.  **Matching Lower Bound (Thm 5.2)**:
    *   **Function**: Proves that the $pd\log\Delta_f$ dominant term in Thm 5.1 is tight.
    *   **Mechanism**: Inspired by the bit-extraction technique of Bartlett 2019, it constructs $N=pdB$ one-hot triplets $x^{(j,i,b)}\in\{0,1\}^{p\times d\times B}$ ($B=\lfloor\log_2 K\rfloor$, $K=\lfloor\Delta_f/2\rfloor$). It designs $f(x,\alpha,\theta)$ as a sum of three terms: "locking $\theta$ to the grid points $\{0,\dots,K{-}1\}^d$", "aligning the specified coordinates of $\alpha$ with the base-$K$ encoding of $\theta$", and "inserting a bit-extracting polynomial $E_c$". A "stabilization" argument ensures the gap between the $\arg\min$ of continuous optimization and the discrete grid $\mathcal{K}^d$ is $<0.1$, allowing any $2^N$ bit labels to be realized with threshold $\tau=0.25$, thus shattering $N=\Omega(pd\log\Delta_f)$.
    *   **Design Motivation**: Bilevel $\arg\min$ prevented the direct use of Bartlett 2019's bit extraction (as continuous optimization might not land on discrete points). Thus, $C\sum_m\prod_k(\theta_m-k)^2$ is used to force $\theta$ onto the grid, with the bit-extraction polynomial $E_c$ set to an order of magnitude smaller than $C$, creating a structure where the main term forces discretization and the secondary term implements bit encoding.

### Loss & Training
This work does not train models but provides sample complexity bounds in the context of statistical learning. Formally, ERM is performed on $N\geq N(\epsilon,\delta)=\mathcal{O}(H^2/\epsilon^2\cdot(\text{Pdim}+\log(1/\delta)))$ i.i.d. problem instances to obtain $\hat\alpha$ such that $\mathbb{E}[\ell_{\hat\alpha}]\leq\inf_\alpha\mathbb{E}[\ell_\alpha]+\epsilon$.

## Key Experimental Results

### Main Results

| Problem Type | Previous Results | Ours | Gain |
| :--- | :--- | :--- | :--- |
| 1D Hyperparameter + $f\equiv g$ | Geometric bound (Balcan 2025) | $\mathcal{O}(pd\log(M_f{+}T_f{+}d)+p^2d\log\Delta_f)$, no ELICQ | Generality + Weaker assumptions |
| Multi-dimensional ($p\geq 2$) + $f\equiv g$ | None | Upper bound + $\Omega(pd\log\Delta_f)$ lower bound | 0 → Learnable |
| Multi-dimensional + $f\neq g$ Bilevel | None | $\mathcal{O}(pd^2\log M_{\text{tot}}+p^2d^2\log\Delta_{\text{tot}})$ | 0 → Learnable |
| Approximate inner $\epsilon$-min | None | Identical order to exact | 0 → Learnable |
| Weighted Group / Fused LASSO | Not piecewise polynomial | Learnable within semi-algebraic class | 0 → Learnable |

### Ablation Study

| Simplified Condition | Bound Order | Note |
| :--- | :--- | :--- |
| Full FOL encoding (Thm 5.1) | $pd\log(\cdot)+p^2d\log\Delta$ | Baseline |
| Explicit solution path (§7) | Removes dependency on $d$ | Order reduction for LASSO/Ridge; partially matches known lower bounds |
| Direct Goldberg-Jerrum 1993 | $p(p+q)\prod d_k(\log M+\log\Delta)$ | Includes $q$ and an extra $p$ factor; significantly larger |

### Key Findings
- Bilevel $f\neq g$ incurs an additional $d$ factor compared to single-level $f\equiv g$ ($pd\to pd^2$) due to the extra $\exists\theta'$ quantifier level; approximate inner optimization does not further double the complexity.
- The removal of the $p$ factor in $p\log M$ is due to tighter "region counting → shattering" analysis rather than the elimination algorithm itself.
- Weighted group lasso remains learnable as it falls within the semi-algebraic class even if it is not piecewise polynomial, indicating that the FOL perspective has broader coverage than classical geometric bounds.
- When an explicit solution path exists (e.g., LARS for LASSO), the $d$ factor can be entirely removed, matching the $\Omega(pd\log\Delta_f)$ lower bound in the best case.
- The number of quantifier levels $K$ is the only parameter with exponential dependence, meaning bilevel ($K=2$) to trilevel ($K=3$) induces qualitative growth, suggesting that "meta-meta-learning" is difficult to bound effectively.

## Highlights & Insights
- The paradigm of "writing implicit optimization as FOL and letting quantifier elimination prove generalization bounds" is highly transferable; the same approach can be applied to bilevel optimization, meta-learning, and the statistical complexity of implicit deep models.
- Combining Bartlett 2019's bit extraction with a "discretization penalty" $\sum_m\prod_k(\theta_m-k)^2$ provides a reusable template for constructing lower bounds for function classes involving an inner argmin.
- Providing the first learning guarantees for "weighted group / fused lasso" losses, which are not piecewise-polynomial but semi-algebraic, significantly expands the scope of analyzable data-driven hyperparameter tuning problems.
- Removing the dependency on data dimension $q$ in Thm 4.1 is important: in modern ML where $q\gg p$, the bound may improve by several orders of magnitude.
- The authors distinguish between "analytic optimization paths (LASSO)" and "black-box" scenarios, providing strictly tighter bounds for the former to reflect how theory can tighten with problem structure.

## Limitations & Future Work
- The upper bound is quadratic in $p, d$ rather than linear, which may not be tight for modern deep learning where both hyperparameter and model parameter counts are large.
- The quantifier-elimination path is exponentially complex with respect to the number of quantifier levels $K$, so bounds for trilevel or higher bilevel structures (meta-meta-learning) will expand rapidly.
- Only sample complexity is provided; no specific optimization algorithms are given. Solving ERM on non-smooth targets remains difficult and requires SGD-type heuristics.
- Empirical validation is absent: theoretical results have not been compared with grid search, BO, or Hyperband on real datasets.

## Related Work & Insights
- **vs Balcan et al. 2025**: Uses 1D geometric curves for proofs, limited to $p=1$ and $f\equiv g$; this work uses FOL + QE to handle $p\geq 1$ and $f\neq g$ simultaneously.
- **vs Goldberg-Jerrum 1993**: Thm 4.1 is strictly superior to the GJ FOL pseudo-dimension bound, removing $q$ and one $p$ factor.
- **vs Bartlett et al. 2019 (bit extraction)**: Reuses the lower bound framework but introduces a stabilization penalty to anchor continuous optimization to discrete grids.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A breakthrough in theoretical methodology by using real algebraic geometry and FOL for multi-level tuning.
- Experimental Thoroughness: ⭐⭐ Lacks empirical evidence; relies on problem instantiation to demonstrate the framework.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with motivation paragraphs for each theorem and high-level proof sketches.
- Value: ⭐⭐⭐⭐ Provides a general upper bound tool with high long-term value for the data-driven algorithm design community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Provably Data-Driven Projection Method for Quadratic Programming](../../AAAI2026/others/provably_data-driven_projection_method_for_quadratic_programming.md)
- [\[ICLR 2026\] An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](../../ICLR2026/others/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)
- [\[ICML 2026\] Adaptive Preconditioners Trigger Loss Spikes in Adam](adaptive_preconditioners_trigger_loss_spikes_in_adam.md)
- [\[ICML 2026\] Advantages of Non-Smooth Components in Vision Transformer Fine-Tuning](vision_transformer_finetuning_benefits_from_non-smooth_components.md)
- [\[AAAI 2026\] SynWeather: Weather Observation Data Synthesis across Multiple Regions and Variables via a General Diffusion Transformer](../../AAAI2026/others/synweather_weather_observation_data_synthesis_across_multiple_regions_and_variab.md)

</div>

<!-- RELATED:END -->
