---
title: >-
  [Paper Note] Submodular Function Minimization with Dueling Oracle
description: >-
  [ICLR 2026][Optimization & Theory][Paper Note] This paper investigates submodular function minimization (SFM) in a setting where only noisy pairwise comparison feedback ("which of the two sets has a larger function value") is available (dueling oracle), without any access to direct function values. The authors construct subgradient estimators using the Lovász exten
tags:
  - ICLR 2026
  - Optimization & Theory
date: 2026-05-08
content_hash: 7e06db791dca3cc6
---
# Submodular Function Minimization with Dueling Oracle

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=BeMtzSH1d7](https://openreview.net/forum?id=BeMtzSH1d7)  
**Code**: Included in supplementary materials (no public repository link provided)  
**Area**: Optimization / Submodular Optimization / Learning Theory  
**Keywords**: Submodular function minimization, dueling oracle, Lovász extension, stochastic gradient descent, upper and lower bounds

## TL;DR
This paper investigates submodular function minimization (SFM) in a setting where only noisy pairwise comparison feedback ("which of the two sets has a larger function value") is available (dueling oracle), without any access to direct function values. The authors construct subgradient estimators using the Lovász extension and SGD. For linear transfer functions, they provide an $O(n^{3/2}/\sqrt{T})$ error bound with a matching lower bound (optimal within a restricted algorithm class). For sigmoid transfer functions, they utilize Firth bias correction to achieve an $O(n^{7/5}/T^{2/5})$ error bound.

## Background & Motivation
**Background**: Submodular function minimization (SFM) is a core problem in combinatorial optimization, machine learning, and operations research, reducible to problems like graph cuts, clustering, image segmentation, and price optimization. Classical approaches (ellipsoid method, combinatorial strongly polynomial algorithms) and more recent methods that convert SFM to convex minimization via Lovász extension and then run SGD (Hazan & Kale 2012, Ito 2019) **all assume a value oracle**—querying a set $S$ yields an exact or noisy function value $f(S)$.

**Limitations of Prior Work**: In many real-world scenarios, exact function values are unattainable or unreliable. For instance, in preference labeling for Large Language Models, RLHF, or recommendation systems, humans find it difficult to provide reliable absolute scores but **can easily express relative preferences** like "A is better than B." Existing research on "dueling oracles" only covers multi-armed bandits and convex optimization. **SFM under pure dueling feedback without any value oracle has never been studied prior to this work.**

**Key Challenge**: The viability of Lovász extension-based SGD depends on the ability to construct **unbiased subgradient estimators**. Prior works rely on value oracle feedback to directly calculate these estimates. However, a dueling oracle returns only a noisy binary sign $o\in\{\pm 1\}$, where the probability of correctness is modulated by a **transfer function** $\rho$: the larger the difference in function values, the more reliable the sign; as the difference diminishes, it becomes a coin flip. When $\rho$ is nonlinear (e.g., sigmoid), it is impossible to derive an unbiased subgradient from such feedback, and biased estimators accumulate errors in SGD.

**Goal**: Extend the solvability of SFM from "value oracles" to "dueling oracles" and provide **algorithms + error upper bounds + information-theoretic lower bounds** for two critical transfer functions—linear $\rho(x)=ax$ and sigmoid $\rho(x)=2/(1+e^{-bx})-1$.

**Key Insight**: The response probability of a dueling oracle is $\Pr(o=+1)=\tfrac12+\tfrac12\rho(f(S)-f(S'))$. In the linear case, $\rho$ is proportional to the value difference, making $\mathbb{E}[o/a]$ exactly equal to the difference, which naturally yields an unbiased subgradient. The sigmoid case corresponds to a **logistic regression model** (matching the Bradley–Terry preference model), where Firth bias correction from statistics can be borrowed to suppress the estimator's bias to $O(1/k^2)$.

**Core Idea**: Construct subgradient estimators (unbiased for linear, low-bias for sigmoid) using dueling feedback and run projected SGD on the Lovász extension. For sigmoid, Firth penalized likelihood estimation is introduced to counter bias accumulation, and the learning rate $\eta$ and the number of repeated queries $k$ per step are redesigned to balance the three components: the SGD term, variance, and bias.

## Method

### Overall Architecture
The algorithmic skeleton across the paper follows a single pipeline: **Use the Lovász extension to transform discrete SFM into convex minimization → Perform iterations via projected SGD on the hypercube $K=[0,1]^n$ → Estimate a subgradient at each step using dueling oracle feedback → Average the iterates and perform random threshold rounding to recover a set.** Innovations lie in "how to estimate subgradients from dueling feedback" and "how to configure parameters to minimize error." Only the estimator and parameter settings change depending on whether the transfer function is linear or sigmoid; the main loop remains consistent.

Specifically, the Lovász extension $\hat f(w)$ extends a submodular function $f$ defined on $2^{[n]}$ to $K$, satisfying $\min_{X} f(X)=\min_{w\in K}\hat f(w)$ (Theorem 1). Since $\hat f$ is piecewise linear, the subgradient is constant within each linear region. At point $w$, by sorting components in descending order to obtain permutation $\pi$ and setting $B_i=\{\pi(1),\dots,\pi(i)\}$, the $i$-th component of the subgradient is $g_{\pi(i)}=f(B_i)-f(B_{i-1})$ (Proposition 2). **Key Observation**: This component is exactly the value difference between two adjacent nested sets, which is precisely what the dueling oracle compares. Thus, querying the pair $(B_{\tau(i)},B_{\tau(i)-1})$ provides information for the $i$-th subgradient component. SGD updates follow $w^{(t+1)}=\Pi_K(w^{(t)}-\eta\hat g_t)$. After $T'$ steps, the average $\bar w$ is taken, and a set $\hat S$ is produced via random thresholding $z \sim U[0,1]$, ensuring $\mathbb{E}[f(\hat S)]=\hat f(\bar w)$.

### Key Designs

**1. Unbiased Subgradients via Dueling Feedback: A "Free Lunch" Under Linear Transfer Functions**

In the linear SFM case, the core problem is obtaining subgradients without function values. The authors exploit the structure of Proposition 2. Under a linear transfer function $\rho(x)=ax$, the dueling oracle returns $o_{ti}\in\{\pm1\}$ for the query $(B_{\tau(i)},B_{\tau(i)-1})$ such that $\mathbb{E}[o_{ti}]=\rho(f(B_{\tau(i)})-f(B_{\tau(i)-1}))=a\,g_i$. Thus, $\hat g_{ti}=o_{ti}/a$ serves as an **unbiased estimator**: $\mathbb{E}[\hat g_{ti}]=g_i$. This alignment between "pairwise comparison" and "piecewise linear subgradients" allows the use of standard SGD analysis, setting $T'=T/n$ steps and $\eta=\frac{a}{2\sqrt{nT}}$.

**2. Firth Bias Correction: Rescuing Unbiasedness for Sigmoid Transfer Functions**

The sigmoid case $\rho(x)=2/(1+e^{-bx})-1$ is nonlinear, so $\mathbb{E}[o]$ and the value difference share a nonlinear relationship, making it theoretically impossible to construct an unbiased subgradient estimator. While prior dueling convex optimization works controlled bias using $\beta$-smoothness and $\alpha$-strong convexity, the Lovász extension is neither smooth nor strongly convex.

The authors reframe the problem as **logistic regression**: for a query $(B_{\tau(i)},B_{\tau(i)-1})$ repeated $k$ times with $k_+, k_-$ counts for $+1/-1$, the parameter $\theta=g_i$ follows a logistic model. Standard Maximum Likelihood Estimation (MLE) yields $O(1/k)$ bias. By applying **Firth's (1993) penalized likelihood method**—adding $\frac12\log|I(\theta)|$ to the log-likelihood—the first-order bias is eliminated for exponential family natural parameters. For the logistic case, this yields:
$$\hat\theta^*=\frac1b\log\frac{k_++\tfrac12}{k_-+\tfrac12}.$$
This "half-count" adjustment reduces bias to $O(1/k^2)$, yielding the estimator used in Algorithm 2.

**3. Three-Way Balanced Parameters: Managing Bias as a Third Error Source**

With the loss of unbiasedness, the analysis becomes more complex. Standard SFM analysis balances two terms: "distance contraction" $\propto n/\eta$ and "variance" $\propto \eta\sum\|\hat g_t\|^2$. In the sigmoid case, a **third term—subgradient estimation bias**—emerges, along with a new control variable: the number of repeats $k$.

The authors provide a parameter rule to jointly optimize these three terms. Solving the combined error bound yields $\eta \propto (n^2/T^2)^{1/5}$, $T' \propto T^{4/5}/n^{4/5}$, and $k \propto T^{1/5}/n^{1/5}$. This configuration results in the $O(n^{7/5}/T^{2/5})$ bound. The degradation from $1/2$ to $2/5$ in the exponent of $T$ is the formal "price" of non-linear bias.

**4. Lower Bounds under Pairwise Queries**

The authors prove matching lower bounds by extending the hard instance construction from Ito (2019). The proof explicitly incorporates the "distribution over query pairs" to bound the KL divergence between distributions induced by different instances.

Under **Restriction 1** (queries must have symmetric difference $|S_t\triangle S_t'|=O(1)$, which Algorithms 1 and 2 satisfy), the lower bound is $\Omega(n^{3/2}/\sqrt T)$, **matching the linear algorithm's upper bound and proving Algorithm 1 is optimal within this class**. Without this restriction, the bound is $\Omega(n/\sqrt T)$, suggesting possible improvement of at most $O(1/\sqrt n)$.

### Loss & Training
There is no traditional training loss; the objective is the convex minimization of the Lovász extension $\hat f$. "Training" consists of projected SGD iterations. Hyperparameters are determined analytically: $\eta=\frac{a}{2\sqrt{nT}}$ for linear, and the 3-way balanced ratios for sigmoid.

## Key Experimental Results

### Main Results

| Transfer Function | Upper Bound (Ours) | Lower Bound (Restriction 1) | Lower Bound (Unrestricted) | Bound Match |
| :--- | :--- | :--- | :--- | :--- |
| Linear $\rho(x)=ax$ | $O(n^{3/2}/\sqrt T)$ | $\Omega(n^{3/2}/\sqrt T)$ | $\Omega(n/\sqrt T)$ | $T$ matches; optimal in class |
| Sigmoid $\rho(x)=\tfrac{2}{1+e^{-bx}}-1$ | $O(n^{7/5}/T^{2/5})$ | $\Omega(n^{3/2}/\sqrt T)$ | $\Omega(n/\sqrt T)$ | Gap in $T$ ($2/5$ vs $1/2$) |
| General Nonlinear | $O(n^{4/3}/T^{1/3})$ | $\Omega(n^{3/2}/\sqrt T)$ | $\Omega(n/\sqrt T)$ | Larger gap |

### Key Findings
- **Linear is "clean"**: Since unbiased estimators are possible, upper and lower bounds match perfectly in $T$.
- **The cost of sigmoid is in the exponent**: Bias reduction via Firth only reaches $O(1/k^2)$, resulting in the degradation of the $T$ exponent from $1/2$ to $2/5$.
- **$b$ cannot be extreme**: The error scales poorly when $b \to 0^+$ (pure noise) or $b \to +\infty$ (vanishing gradient info).
- Numerical experiments in Appendix B suggest that linear algorithms remain effective for smooth nonlinear $\rho$ when exploring small neighborhoods near zero.

## Highlights & Insights
- **Structural Alignment**: Aligning pairwise comparisons with nested set differences in the Lovász extension is the paper's most elegant contribution.
- **Statistical Transfer**: Applying Firth bias correction to optimization is highly transferable to RLHF or preference alignment scenarios.
- **Three-Way Analysis**: The paradigm of jointly optimizing learning rate and repetition counts provides a blueprint for optimization problems where unbiasedness is unattainable.

## Limitations & Future Work
- **Sigmoid Gap**: The gap between $T^{-2/5}$ and $T^{-1/2}$ remains an open question; the authors conjecture $O(1/\sqrt T)$ may be achievable.
- **Known $\rho$ Assumption**: The algorithm requires knowing parameters $a$ or $b$. Real-world applications would need parameter-free versions.
- **Purely Theoretical**: The lack of end-to-end experiments in real recommendation or RLHF environments means the practical engineering gains remain to be verified.

## Related Work & Insights
- **vs Hazan & Kale (2012)**: This work improves on their $O(n/T^{1/3})$ error by using dueling feedback instead of full value bandit feedback, achieving $O(n^{3/2}/\sqrt T)$ in the linear case.
- **vs Ito (2019)**: This work reaches the same error rate as Ito's noisy value oracle setting despite the significantly weaker "comparison-only" feedback.
- **vs Bradley–Terry / RLHF**: The sigmoid setting directly maps to the Bradley–Terry model, providing a theoretical foundation for combinatorial optimization from preference feedback.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
<!-- RELATED:END --></div>

## Related Papers

- [\[ICLR 2026\] Activation Function Design Sustains Plasticity in Continual Learning](activation_function_design_sustains_plasticity_in_continual_learning.md)
- [\[ICLR 2026\] Reducing Contextual Stochastic Bilevel Optimization via Structured Function Approximation](reducing_contextual_stochastic_bilevel_optimization_via_structured_function_appr.md)
- [\[ICLR 2026\] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization](towards_understanding_the_calibration_benefits_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Hyperbolic Aware Minimization: Implicit Bias for Sparsity](hyperbolic_aware_minimization_implicit_bias_for_sparsity.md)
- [\[ICLR 2026\] Efficient Submodular Maximization for Sums of Concave over Modular Functions](efficient_submodular_maximization_for_sums_of_concave_over_modular_functions.md)

</div>

<!-- RELATED:END -->
