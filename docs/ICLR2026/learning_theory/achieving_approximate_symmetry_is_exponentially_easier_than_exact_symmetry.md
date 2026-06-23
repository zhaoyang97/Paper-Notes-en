---
title: >-
  [Paper Note] Achieving Approximate Symmetry Is Exponentially Easier than Exact Symmetry
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper defines a quantifiable cost for "forcing model symmetry"—**averaging complexity**—and proves an exponential separation: under standard conditions, forcing **exact symmetry** requires a number of queries linear in the group size $|G|$, while forcing **approximate symmetry** requires only logarithmic queries $
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 16356d52a7dbbbc8
---
# Achieving Approximate Symmetry Is Exponentially Easier than Exact Symmetry

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=ncOJYFcleS](https://openreview.net/forum?id=ncOJYFcleS)  
**Code**: To be confirmed  
**Area**: Learning Theory / Geometric Machine Learning  
**Keywords**: Approximate symmetry, averaging complexity, group representation theory, equivariant learning, Fourier analysis

## TL;DR
This paper defines a quantifiable cost for "forcing model symmetry"—**averaging complexity**—and proves an exponential separation: under standard conditions, forcing **exact symmetry** requires a number of queries linear in the group size $|G|$, while forcing **approximate symmetry** requires only logarithmic queries $O(\log|G|/\varepsilon)$. This provides the first theoretical explanation for "why approximate symmetry is often more cost-effective than exact symmetry in practice."

## Background & Motivation
**Background**: The core idea of geometric machine learning is to incorporate observed structures in scientific data (permutation symmetry in point clouds, sign-flip symmetry in spectral methods, rotational symmetry in robotics, various symmetries in molecular data) into models as strong inductive biases. Numerous methods exist to **exactly** encode symmetry: model-agnostic approaches include group averaging, data augmentation, canonicalization, and frame averaging; model-specific approaches include equivariant CNNs, equivariant GNNs, and Deep Sets. These methods consistently improve sample complexity and generalization.

**Limitations of Prior Work**: Exact symmetry is often "overkill" in many real-world scenarios. Desired reflection symmetry in medical imaging is rarely perfect; results may be only **slightly** sensitive to such transformations. In symmetry discovery, only partial symmetry information is known. In these cases, hard-coding exact invariance can **limit the universality and expressivity of the model**. Consequently, practitioners have turned to **approximate symmetry**: allowing models to violate symmetry to a certain degree in exchange for better flexibility, robustness, and the ability to utilize symmetry in semi-supervised settings—an approach that proves effective empirically.

**Key Challenge**: Despite the empirical success of approximate symmetry, its **theoretical foundation is almost non-existent**. There is no unified characterization of "how difficult it is to achieve approximate symmetry," nor a **direct comparison between exact and approximate approaches on the same scale**. Intuition suggests that approximate symmetry, as a "relaxed" version of invariance, should be easier to achieve, but this intuition has not been formalized into a theorem.

**Goal**: To formalize the vague "complexity of achieving approximate symmetry" and answer a quantitative question: given a model, what are the complexities of forcing approximate versus exact symmetry? Is there a fundamental separation between the two?

**Key Insight**: The authors assume a learner can only access the model in a **black-box manner** and is only allowed **linear post-processing**. Given a pre-selected function $f$ and group element $g$, an oracle returns the transformed function $x\mapsto f(gx)$; this interaction is called an **action query (AQ)**. The learner uses linear combinations of these query responses to achieve (approximate) symmetry with as few queries as possible. The number of queries serves as the "budget" and a natural measure of complexity.

**Core Idea**: Using "the minimum number of action queries required to achieve symmetry (i.e., the size of the minimum averaging scheme)" as a unified scale, the paper proves exact symmetry requires $\Theta(|G|)$ while approximate symmetry requires only $O(\log|G|/\varepsilon)$, establishing an exponential separation.

## Method
As a pure theory paper, the "method" consists of a set of **definitions + theorems + proof strategies**: abstracting the cost of "forcing symmetry through averaging" into averaging complexity, then using group representation theory and Fourier analysis on groups to provide lower bounds for exact symmetry and upper bounds for approximate symmetry.

### Overall Architecture
The logical chain is: **①** Define an averaging scheme $\omega:G\to\mathbb{R}$ and its induced averaging operator $E_\omega$, unifying "symmetrization" as a "weighted linear combination of $\{f(g^{-1}x)\}_g$"; **②** Define exact, weak approximate, and strong approximate symmetries, along with their corresponding averaging complexities $\mathrm{AC}_{\mathrm{ex}}$, $\mathrm{AC}_{\mathrm{wk}}$, and $\mathrm{AC}_{\mathrm{st}}$ (the **minimum scheme size** required); **③** Prove a lower bound for exact symmetry $\mathrm{AC}_{\mathrm{ex}}=|G|$ (Thm 13, using symmetric tensor powers to expand the function class and irreducible decompositions from representation theory to show that "$\omega$ must be the uniform distribution"); **④** Prove an upper bound for approximate symmetry $O(\log|G|/\varepsilon)$ (Thm 15, using random sampling and large deviation bounds for the operator norm of random matrices); **⑤** Substitute the same function class into both ③ and ④ to obtain the exponential separation.

The averaging operator takes the form:

$$(E_\omega[f])(x) := \sum_{g\in G}\omega(g)\,f(g^{-1}x),\qquad \|\omega\|_{\ell_1(G)}=\sum_{g}\omega(g)=1,$$

The scheme size $\mathrm{size}(\omega):=\#\{g:\omega(g)\neq 0\}$ is the count of non-zero weights, representing the number of action queries. Note that $\omega$ **does not depend** on the specific $x$, otherwise it would degenerate into (weighted) frame averaging, and the concept of complexity would lose meaning—this is the key constraint that frames the problem as "universal linear combination."

### Key Designs

**1. Averaging Complexity: Turning "Forcing Symmetry" into a Countable Query Budget**

The primary difficulty was the lack of a unified measure in literature for the difficulty of achieving approximate symmetry, making comparison impossible. The authors' approach is to completely **operator-ize and count** the symmetrization process: any scheme forcing symmetry can be written as a set of weights $\omega(g)$ for a linear combination of transformed functions $f(g^{-1}x)$; the "cost" is defined as the **minimum number of non-zero weights** needed. The three complexities are unified as constrained minimization problems:

$$\mathrm{AC}_{\mathrm{ex}}(F):=\min_\omega \mathrm{size}(\omega)\quad \text{s.t.}\quad (E_\omega[f])(gx)=(E_\omega[f])(x),\ \forall f,g,x.$$

This definition is effective because it **transforms a geometric/algebraic symmetry problem into a combinatorial optimization problem**: counting "the minimum number of group elements whose weighted average can symmetrize any $f$," a value that can be precisely characterized via group representation theory. The trivial upper bound is clear—querying all $g\in G$ (uniform averaging $\omega\equiv 1/|G|$) always achieves exact symmetry, so all three complexities $\le|G|$; the real question is when **sublinear** performance is possible.

**2. Weak / Strong Approximate Symmetry: Tuning Precision with $L^2$ Norms**

Exact symmetry requires $f(gx)=f(x)$ everywhere, which is too rigid. The authors use the $L^2(X)$ norm to measure the "distance from symmetry" and a contraction factor $\varepsilon>0$ to compress the "non-symmetric components" of the function. **Weak approximation** requires the symmetry error to be compressed by a factor of $\varepsilon$ on **average** over group elements:

$$\mathbb{E}_g\!\!\int_X\!|(E_\omega[f])(x)-(E_\omega[f])(gx)|^2 d\mu \le \varepsilon\, \mathbb{E}_g\!\!\int_X\!|f(x)-f(gx)|^2 d\mu;$$

**Strong approximation** requires this to hold for **every** $g$ (removing $\mathbb{E}_g$). When $\varepsilon=0$, both revert to exact symmetry. This $\varepsilon$-contraction design is key because it turns "degree of approximation" into a **continuously adjustable knob**, allowing analysis of how complexity scales with $\varepsilon$. Choosing the $L^2$ norm allows the results to connect directly to generalization theories for approximate symmetric regression. Proposition 10 provides basic properties: all are non-increasing with $\varepsilon$, and $\mathrm{AC}_{\mathrm{wk}}\le\mathrm{AC}_{\mathrm{st}}\le\mathrm{AC}_{\mathrm{ex}}\le|G|$, with $\mathrm{AC}_{\mathrm{wk}}(F,4\varepsilon)\le\mathrm{AC}_{\mathrm{st}}(F,4\varepsilon)\le\mathrm{AC}_{\mathrm{wk}}(F,\varepsilon)$—the latter relates strong and weak approximations within a constant factor, allowing focus on weak approximation.

**3. Linear Lower Bound for Exact Symmetry: Using Symmetric Tensor Powers and Irreducible Decomposition**

To prove exact symmetry is "expensive," one must exclude trivial function classes (e.g., constant functions). The authors introduce two tools. First is the **faithful group action assumption** (Assumption 11): for every non-identity element $g$, there exist $f\in F, x\in X$ such that $f(gx)\neq f(x)$. Second is the **symmetric tensor power** $\widetilde{\mathrm{Sym}}^{\otimes k}(F)=\bigoplus_{\ell=0}^k\mathrm{Sym}^{\otimes\ell}(F)$, which expands a base function class $F$ into a larger class spanned by the "pointwise product of up to $k$ functions in $F$"—for example, if $F$ consists of linear functions on $\mathbb{R}^d$, $\widetilde{\mathrm{Sym}}^{\otimes k}(F)$ is the space of polynomials of degree $\le k$. Thm 13 asserts that there exists an integer $K$ with an explicit closed-form upper bound such that:

$$\mathrm{AC}_{\mathrm{ex}}\!\big(\widetilde{\mathrm{Sym}}^{\otimes k}(F)\big)=|G|,\qquad \forall k\ge K,$$

where $K\le\min\{|G|,\ \sum_{\lambda\in\Lambda}(M_\lambda-1)\}$, with $\Lambda$ being the set of eigenvalues of all $\rho(g)$ and $M_\lambda$ the maximum multiplicity of $\lambda$. The proof logic (§5.1) is elegant: by complete reducibility, the representation $\rho$ decomposes into a direct sum of irreducible representations (irreps). Thinking of the averaging scheme $\omega$ as a signal on the group, its Fourier transform $\widehat\omega(\pi)=\sum_g\omega(g)\pi(g)^*$ is taken. Averaging on $\widetilde{\mathrm{Sym}}^{\otimes k}(F)$ yields exact symmetry if and only if $\widehat\omega(\pi)=0$ for every **non-trivial** irrep appearing in any $\mathrm{Sym}^{\otimes k}(\rho)$. For $k=K$, **every non-trivial irrep will appear in some tensor power**, forcing $\widehat\omega$ to be zero everywhere except at the trivial irrep. By Fourier inversion, $\omega$ must be the uniform measure $1/|G|$, requiring $|G|$ non-zero weights. Intuition: the larger the function class, the more non-trivial frequencies must be "killed," eventually leaving the uniform average as the only choice.

**4. Logarithmic Upper Bound for Approximate Symmetry: Random Sampling and Matrix Concentration**

To prove approximate symmetry is "cheap," the same Fourier perspective is used, but the goal shifts to constructing $\omega$ such that $\|\widehat\omega(\pi)\|_{\mathrm{op}}\le\varepsilon$ for all non-trivial irreps with a small $\mathrm{size}(\omega)$. Thm 15 provides:

$$\mathrm{AC}_{\mathrm{st}}(F,\varepsilon)=O\!\Big(\tfrac{\log|G|}{\varepsilon}\Big),\qquad \mathrm{AC}_{\mathrm{wk}}(F,\varepsilon)=O\!\Big(\tfrac{\log|G|}{\varepsilon}\Big),$$

with hidden constants at most $8/3\approx 2.67$ (weak) or $32/3\approx 10.67$ (strong). The proof (§5.2) uses a **probabilistic construction**: independently and uniformly sample $n$ group elements, denote their empirical distribution as $\Omega$, and form a block diagonal matrix $\Xi:=\bigoplus_{\pi\ \text{non-trivial}}\widehat\Omega(\pi)$. Since $\mathbb{E}[\Xi]=0$ and each non-trivial $\pi$ satisfies $\|\widehat\Omega(\pi)\|_{\mathrm{op}}\le\|\Xi\|_{\mathrm{op}}$, the problem reduces to **controlling the operator norm of a zero-mean random matrix**. Standard large deviation bounds show that if $n\ge c\,\log\dim(\Xi)/\varepsilon$ (for some universal constant $c$), then $\|\Xi\|_{\mathrm{op}}\le\varepsilon$ holds with high probability. Since $\dim(\Xi)\le|G|$, we obtain $O(\log|G|/\varepsilon)$. This upper bound is powerful because it **holds uniformly for all function classes**, regardless of Assumption 11 or tensor power construction—specifically including the classes that require linear complexity in Thm 13. Between the two: for the same function class, exact requires $|G|$ while approximate requires only $O(\log|G|)$, establishing the **exponential separation**. Remark 16 further notes that this logarithmic upper bound is tight up to constants.

### Example: Exact Complexity on Permutation Groups
Take $X=\mathbb{R}^d$, $G=S_d$ acting by coordinate permutation, and $F$ as all linear functions. Here $\rho(g)$ is a permutation matrix. If $g$ has cycle lengths $(\ell_1,\dots,\ell_t)$, its eigenvalues are the $\ell_j$-th roots of unity. A counting argument gives $\sum_{\lambda\in\Lambda}M_\lambda=\sum_{q=1}^d\varphi(q)\lfloor d/q\rfloor=d(d+1)/2$ (where $\varphi$ is Euler's totient function). Thus, for polynomial degree $K=\tfrac{d(d+1)}{2}-1$, forcing exact symmetry already requires the full $|G|=d!$ averages—whereas approximate symmetry only requires $O(\log d!)=O(d\log d)$, an astronomical difference.

## Key Experimental Results
This is a pure theory paper without benchmark experiments, supported instead by a **schematic illustration** and a **complexity comparison table**.

### Core Conclusion Comparison (Theoretical Bounds)

| Symmetry Type | Averaging Complexity | Scale | Source |
| :--- | :--- | :--- | :--- |
| Exact Symmetry $\mathrm{AC}_{\mathrm{ex}}$ | $=|G|$ (when $k\ge K$) | Linear | Thm 13 |
| Weak Approx. $\mathrm{AC}_{\mathrm{wk}}(F,\varepsilon)$ | $O(\log|G|/\varepsilon)$, constant $\le 8/3$ | Logarithmic | Thm 15 |
| Strong Approx. $\mathrm{AC}_{\mathrm{st}}(F,\varepsilon)$ | $O(\log|G|/\varepsilon)$, constant $\le 32/3$ | Logarithmic | Thm 15 |
| Approx. Lower Bound | $\Omega_\varepsilon(\log|G|)$ | Log. (Tight) | Remark 16 |

### Key Findings

| Property | Content | Explanation |
| :--- | :--- | :--- |
| Monotonicity | $\mathrm{AC}_{\mathrm{wk}}\le\mathrm{AC}_{\mathrm{st}}\le\mathrm{AC}_{\mathrm{ex}}\le|G|$ | Stronger symmetry is more expensive (Prop 10) |
| Strong/Weak Link | $\mathrm{AC}_{\mathrm{st}}(F,4\varepsilon)\le\mathrm{AC}_{\mathrm{wk}}(F,\varepsilon)$ | Interconvertible within constant factors |
| Class Sensitivity | $F_1\subseteq F_2\Rightarrow \mathrm{AC}(F_1)\le\mathrm{AC}(F_2)$ | Exact symmetry becomes harder as the function class grows |

- **Visual Validation (Fig 1)**: For a 2D rotation group with 100 elements, an anisotropic function $f(x,y)$ averaged over $|S|=5\approx\log(100)$ random rotations (approximate) is visually indistinguishable from the result using all $|S|=100$ rotations (exact)—empirically corroborating the $|S|\approx\log|G|$ theory.
- **Source of Exponential Separation**: Exact symmetry requires $\widehat\omega$ to be **exactly zero** at all non-trivial frequencies, a rigid constraint met only by the uniform average. Approximate symmetry only requires these frequencies to be **suppressed below $\varepsilon$**, a flexible constraint met by logarithmic random sampling. This rigidity vs. flexibility is the root of the $|G|$ vs. $\log|G|$ gap.
- **Universality**: The approximate upper bound does not depend on the faithful action assumption or tensor power construction; it holds for any finite-dimensional function class, including complex-valued functions.

## Highlights & Insights
- **Operator-izing Averaging Cost**: The cleverest part of the "averaging complexity" measure is transforming a geometric/algebraic "forced symmetry" problem into a combinatorial "minimum group queries" problem, solvable via Fourier analysis.
- **Dual Bounds on the Same Class**: Many separation results compare different classes ("Class A is hard, Class B is easy"). This work provides both $|G|$ lower bounds and $\log|G|$ upper bounds on the **exact same function class**, making the separation undeniable.
- **Frequency Suppression Perspective**: Viewing the averaging scheme as a group signal and characterizing symmetrization capability via Fourier spectrum sparsity/operator norms is a portable toolset likely applicable to data augmentation or canonicalization analysis.
- **Explaining Empirical Phenomena**: Practitioners have long observed that approximate symmetry is flexible and robust, but couldn't explain why it was "cheaper." This paper provides the "exponentially cheaper" answer, turning engineering intuition into a theorem.

## Limitations & Future Work
- **Limited to Finite Groups**: All conclusions rely on $G$ being a finite group for Fourier analysis and $|G|$ counting. Extending to infinite groups (e.g., continuous rotations $SO(d)$) is a natural but difficult challenge that may require new tools.
- **Gap Between Abstract Model and Real Training**: Averaging complexity measures query costs in an abstract "black-box + linear post-processing" interaction. This does not directly translate to computational/sample costs in real neural network training; it is an **abstract indicator in the sense of lower/upper bounds**.
- **Lack of Empirical Benchmarking**: Aside from the schematic in Figure 1, the paper does not verify on real datasets that models with $\log|G|$ averages perform better on downstream tasks.
- **Future Directions**: Applying this framework to analyze the complexity of data augmentation; replacing the $L^2$ norm with other metrics; and investigating if the dependency on $\varepsilon$ can be improved beyond $1/\varepsilon$.

## Related Work & Insights
- **vs. Frame Averaging / Canonicalization**: These are specific exact equivariance methods. This paper doesn't propose a new method but establishes a unified complexity scale for all "averaging-based symmetrization" and points out that exact methods naturally pay a $|G|$ price in the abstract model.
- **vs. Soft / Partial / Approximate Equivariance Methods** (Finzi 2021, Kim 2023, Park 2025, etc.): These works show empirically that approximate symmetry is "flexible and robust." This paper provides the theoretical defense—proving for the first time that approximate symmetry is **exponentially easier to achieve** than exact symmetry.
- **vs. Learning Hardness Under Symmetry** (Kiani 2024, etc.): While both are symmetry theories, Kiani et al. focus on "how hard it is to learn under symmetry constraints," whereas this paper focuses on "how hard it is to force symmetry into a model," providing the exponential separation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to unify exact and approximate symmetry in a complexity framework and prove exponential separation.
- Experimental Thoroughness: ⭐⭐⭐ Pure theory paper; only one schematic and no downstream empirical tests, but theoretical bounds are tight.
- Writing Quality: ⭐⭐⭐⭐ Systematic progression from definitions to theorems; the Fourier perspective is clear.
- Value: ⭐⭐⭐⭐⭐ Provides a rigorous theoretical explanation for a widely observed empirical phenomenon; tools are transferable to broader geometric ML analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking](to_augment_or_not_to_augment_diagnosing_distributional_symmetry_breaking.md)
- [\[ICLR 2026\] Reducing Symmetry Increase in Equivariant Neural Networks](reducing_symmetry_increase_in_equivariant_neural_networks.md)
- [\[ICLR 2026\] Branch and Bound Search for Exact MAP Inference in Credal Networks](branch_and_bound_search_for_exact_map_inference_in_credal_networks.md)
- [\[ICLR 2026\] Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling](expressive_power_of_implicit_models_rich_equilibria_and_test-time_scaling.md)
- [\[ICLR 2026\] Better Bounds for the Distributed Experts Problem](better_bounds_for_the_distributed_experts_problem.md)

</div>

<!-- RELATED:END -->
