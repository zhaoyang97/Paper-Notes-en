---
title: >-
  [Paper Note] Is Spurious Correlation Removal Always Learnable?
description: >-
  [ICML 2026][Learning Theory][Spurious Correlation] This paper demonstrates that removing spurious correlations may be "computationally non-learnable" even when the invariant structure is "statistically identifiable" in ideal scenarios. It proves the existence of a family of multi-environment instances where brute-force search recovers the invariant direction with polynomial samples, but any polynomial-time algorithm achieving constant precision would resolve a widely believed…
tags:
  - "ICML 2026"
  - "Learning Theory"
  - "Invariant Learning"
  - "Computational Complexity"
  - "Spurious Correlation"
  - "Computational-Statistical Gap"
  - "Environmental Diversity"
  - "Minimax Rate"
date: 2026-05-08
content_hash: cd02b0d955215578
---

# Is Spurious Correlation Removal Always Learnable?

**Conference**: ICML 2026  
**arXiv**: [2606.12930](https://arxiv.org/abs/2606.12930)  
**Code**: To be confirmed  
**Area**: Learning Theory / Invariant Learning / Computational Complexity  
**Keywords**: Invariant Learning, Spurious Correlation, Computational-Statistical Gap, Environmental Diversity, Minimax Rate

## TL;DR
This paper demonstrates that removing spurious correlations may be "computationally non-learnable" even when the invariant structure is "statistically identifiable" in ideal scenarios. It proves the existence of a family of multi-environment instances where brute-force search recovers the invariant direction with polynomial samples, but any polynomial-time algorithm achieving constant precision would resolve a widely believed hard sparse recovery problem. Simultaneously, the paper characterizes identifiability, minimax rates, and sample complexity phase transitions using an "environmental diversity" parameter $\gamma$.

## Background & Motivation

**Background**: Machine learning models often rely on spurious correlations that hold only in training distributions but fail under distribution shifts (e.g., classifying based on background color rather than the object). Invariant learning (e.g., IRM, REx) attempts to utilize "multi-environment" data to preserve only features whose predictive relationship with the label is stable across environments. It has been theoretically proven that under certain diversity assumptions, this invariant structure is **statistically identifiable**.

**Limitations of Prior Work**: Despite identifiability, invariant methods perform inconsistently in practice—succeeding on some datasets while degrading to Empirical Risk Minimization (ERM) on others. The community often attributes such inconsistency to "poor optimization" or "insufficient environments."

**Key Challenge**: The authors investigate a more fundamental question: Could this inconsistency be **unavoidable even in ideal cases where invariance is fully identifiable**? In other words, failures might not be optimization issues but the **computational hardness** itself—where the global objective has a unique maximum, but no polynomial-time algorithm can find it.

**Goal**: (1) Prove the separation of "computational feasibility $\neq$ statistical feasibility" in spurious correlation removal within a clean linear-Gaussian setting; (2) Identify a structural quantity that explains "when learning is easy vs. hard"; (3) Provide quantitative relationships between this quantity and identifiability, optimal estimation rates, and sample complexity thresholds.

**Key Insight**: Formalizing "spurious correlation removal" as **recovering the invariant subspace** $V_{\mathrm{inv}}$ from multi-environment data. This step places the problem into a framework with mature tools: one side connects to average-case complexity (hardness assumptions of planted clique / sparse recovery) to prove lower bounds; the other side connects to minimax theory on Grassmann manifolds to provide upper bounds.

**Core Idea**: In short—**"wrapping" a widely believed hard sparse recovery primitive into a valid multi-environment spurious correlation instance**, thereby "porting" its computational hardness to invariant subspace recovery; then using a scalar "environmental diversity" $\gamma$ to link statistical identifiability with sample rates.

## Method

### Overall Architecture

The paper does not propose a new algorithm but instead maps out a **feasibility map** for "spurious correlation removal," supported by two pillars.

The first is the **computational lower bound** (Section 3): Starting from a black-box, samplable "supervised sparse recovery primitive" (inspired by average-case reductions like planted clique / sparse CCA), a family of multi-environment spurious correlation instances is constructed. In these instances, the invariant direction is **statistically recoverable via brute-force search**, but **polynomial-time recovery would contradict the hardness of the primitive**. The key constraint is that the construction must be "samplable"—generating samples in random polynomial time without accessing the hidden sparse direction or its support.

The second is the **statistical upper bound** (Section 4): Under linear-Gaussian structures, it characterizes when $V_{\mathrm{inv}}$ is identifiable and what the minimax rate of estimation error is. The core quantity is **environmental diversity** $\gamma$ introduced in Definition 2.3—measuring the variance of perturbations to spurious correlations across environments. Insufficient diversity renders $V_{\mathrm{inv}}$ unidentifiable (even with infinite samples), while sufficient diversity makes the problem well-posed with a sharp finite-sample phase transition.

The second pillar also reveals a counter-intuitive insight: the **"diversity" of environments is more critical than their "quantity"**—a few highly diverse environments can be more useful than a large number of similar ones.

### Key Designs

**1. Formalizing Spurious Correlation Removal as Linear-Gaussian Invariant Subspace Recovery**

To discuss "learnability," the problem must be anchored as a mathematical object. The paper defines Spurious Correlation (SC) instances (Definition 2.1) by orthogonally decomposing $\mathbb{R}^d$ into an invariant subspace $V_{\mathrm{inv}}$ (dimension $k$) and a spurious subspace $V_{\mathrm{sp}}$. Input is split as $X = X_{\mathrm{inv}} + X_{\mathrm{sp}}$, satisfying: (C1) the conditional mechanism $\mathbb{P}_e(Y \mid X_{\mathrm{inv}})$ is **identical** across environments ("invariance"); (C2) $\mathbb{P}_e(X_{\mathrm{sp}} \mid X_{\mathrm{inv}})$ **differs** between at least one pair of environments ("spuriousness"—it drifts). Statistical analysis adds the standard linear-Gaussian structure $Y = \langle w^*, X_{\mathrm{inv}} \rangle + \epsilon$.

This formalization transforms the vague task of "removing spurious correlations" into "finding the unique $k$-dimensional subspace on the Grassmannian $\mathrm{Gr}(k,d)$ that keeps the predictive mechanism invariant." The performance metric is subspace distance $\operatorname{dist}(V, V') := \|P_V - P_{V'}\|_F$.

**2. Samplable Hard Instance Construction: Embedding Sparse Recovery into SC Instances**

This is the technical core. The paper assumes a black-box **samplable supervised sparse recovery primitive** (Hypothesis 3.2): a random polynomial-time sampler $\mathcal{R}$ exists. In simple terms, any polynomial-time algorithm that could recover a constant overlap with the hidden sparse direction $v$ would solve the underlying hard sparse recovery problem.

Construction 3.3 **wraps** this primitive into an SC instance: each environment independently runs $\mathcal{R}(A)$ to get $(Z_t^{(e)}, Y_t^{(e)})$, then constructs the input:

$$X^{(e,t)} = (Z_t^{(e)},\, W^{(e,t))), \qquad W^{(e,t)} = \mu^{(e)} Y_t^{(e)} + \eta^{(e,t)},$$

setting $V_{\mathrm{inv}} = \mathrm{span}\{(v, 0)\}$. The spurious block $W^{(e)}$ uses environment-dependent coefficients $\mu^{(e)}$ to create drift, satisfying (C2); while $Y$ depends on $Z$ only through $v^\top Z$, ensuring (C1). This construction **never touches the hidden direction**, utilizing only independent calls to $\mathcal{R}(A)$ and public Gaussian noise.

To ensure $V_{\mathrm{inv}}$ is uniquely identified, the paper defines an **invariance-predictivity joint score** $S(V) = A(V) - \lambda T(V)$, where $A(V)$ measures predictive power and $T(V)$ **penalizes drifting spurious directions** using covariance differences across environments. Lemma 3.5 proves that $V_{\mathrm{inv}}$ maintains a constant margin $c_{\mathrm{mar}}$ in $S(V)$.

The resulting gap (Theorem 3.8 / 3.9) shows: (a) $V_{\mathrm{inv}}$ is the unique maximum of $S(V)$, hence **identifiable**; (b) Brute-force search recovers it with polynomial samples $N = \tilde{O}(\mathrm{poly}(m)/c_{\mathrm{mar}}^2)$; (c) However, any polynomial-time algorithm achieving $\operatorname{dist}(\hat V, V_{\mathrm{inv}}) \leq \delta_0$ would solve the hard primitive, causing a contradiction. Notably, the gap exists even for **$k=1$** and **only 2 environments**.

**3. Environmental Diversity Parameter $\gamma$ and Identifiability**

The statistical protagonist is diversity $\gamma$ (Definition 2.3). For spurious coordinate $j$, let $\mu_{e,j}(x) = \mathbb{E}_e[X_{\mathrm{sp},j} \mid X_{\mathrm{inv}} = x]$. Defining the range across environments as $\Delta_j(x) = \max_e \mu_{e,j}(x) - \min_e \mu_{e,j}(x)$, then:

$$\gamma := \big(\mathbb{E}[\max_j \Delta_j(X_{\mathrm{inv}})^2]\big)^{1/2}.$$

In mean-drift settings, this simplifies to $\gamma = \max_{e,e'} \|\mu_{\mathrm{sp}}^{(e)} - \mu_{\mathrm{sp}}^{(e')}\|_\infty$—how far apart different environments push the spurious correlation. When $\gamma = 0$ (no distinguishable change in spurious conditional means), **$V_{\mathrm{inv}}$ is unidentifiable even with infinite data** (Corollary 4.7). The paper provides an empirical proxy $\hat\gamma$ (median of the cross-environment range of feature-label correlations) for practical diagnosis.

**4. Minimax Rates and Phase Transitions: Translating Diversity to $1/\gamma^2$ Sample Complexity**

Under sufficient diversity and local regularity, the paper provides the optimal estimation rate (Theorem 4.11):

$$\mathbb{E}[\operatorname{dist}(\hat V, V_{\mathrm{inv}})^2] = \Theta\!\left(\frac{k(d-k)}{n|\mathcal{E}|}\right),$$

where $k(d-k)$ is the intrinsic degrees of freedom of a $k$-dimensional subspace. Under label-induced drift, a **sharp phase transition** exists (Theorem 4.15): the critical sample size

$$n^* \propto \frac{k(d-k)}{|\mathcal{E}|\,\gamma^2},$$

In the supercritical regime ($n > 2n^*$), error $\lesssim k(d-k)/(n|\mathcal{E}|\gamma^2)$; in the subcritical regime ($n < n^*/2$), the error of any estimator is bounded by a constant. This $1/\gamma^2$ scaling quantitatively verifies that "a few highly diverse environments beat many similar ones."

## Key Experimental Results

### Main Results: Computational-Sample Gap on Synthetic Hard Instances

| Phenomenon | Brute-force Search (Exp. Time) | Best Poly-time Method | Theoretical Explanation |
|------|---------------------|-------------------|----------|
| Samples for target accuracy | Far fewer | Far more | Theorem 3.9 Gap |
| $\gamma = 0$ | Unrecoverable | Unrecoverable | Corollary 4.7 Unidentifiable |
| Samples crossing $n^*$ | Error drops sharply | — | Theorem 4.15 Transition |

Synthetic experiments (Figure 1/2) show that on planted hard instances, brute-force search achieves accuracy with significantly fewer samples than poly-time methods, illustrating the "statistically feasible, computationally infeasible" gap. A phase transition leap in accuracy occurs as samples exceed $n^* \propto k(d-k)/(|\mathcal{E}|\gamma^2)$.

### Key Findings
- **Gap appears at $k=1$**: Hardness does not require high-dimensional invariant subspaces; even the simplest 1D case is computationally hard.
- **Diversity > Quantity**: If $\gamma = 0$, learning is impossible with infinite samples. As $\gamma$ increases, the critical sample size decreases by $1/\gamma^2$.
- **Failure may be Hardness, not Hyperparameters**: Before attributing invariant learning failure to optimization, one should **estimate environmental diversity $\hat\gamma$**.

## Highlights & Insights
- **Clean Counter-example for "Identifiable $\neq$ Learnable"**: While previous work focused on statistical identifiability, this paper uses samplable reductions to prove "identifiable but poly-time unrecoverable" instances, clarifying the computational-statistical gap in spurious correlation removal.
- **Samplability as a Key Technical Tool**: The construction cleverly uses black-box calls to the primitive without accessing the hidden direction, allowing average-case complexity to be ported to the representation learning context.
- **$\gamma$ as both Theory and Diagnostic**: The transition from a population definition to an empirical proxy $\hat\gamma$ provides a practical tool to screen datasets before training.

## Limitations & Future Work
- **Hardness is Conditional**: The lower bound depends on Hypothesis 3.2. While inspired by reductions like sparse CCA, it is a hardness assumption rather than an unconditional theorem.
- **Idealized Linear-Gaussian Setting**: Real-world spurious correlations are more complex. $\gamma$ as a coordinate-wise scalar might oversimplify scenarios with entangled features.
- **Efficiency of Upper-Bound Estimators**: Many minimax rates in the paper are achieved by "inefficient estimators." General efficient algorithms that reach optimal rates without specific structural assumptions remain absent.

## Related Work & Insights
- **vs. IRM / REx**: While prior methods proposed objectives to approximate invariant structures, this paper asks if the global optima of such objectives can be efficiently found, providing a hardness-based answer.
- **vs. Rosenfeld et al.**: Previous work proved $V_{\mathrm{inv}}$ is statistically identifiable under diversity; this paper extends this by showing identifiability is necessary but not sufficient for computational learnability.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ High. Solidifies the computational-statistical gap in OOD generalization.
- **Experimental Thoroughness**: ⭐⭐⭐ Moderate. Theory-focused; synthetic validation is robust, but real-world benchmarks are limited.
- **Writing Quality**: ⭐⭐⭐⭐ Very good. Clearly delineates when learning is possible vs. impossible.
- **Value**: ⭐⭐⭐⭐ High. Provides a diagnostic perspective for why invariant learning fails.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Deep Learning with Learnable Product-Structured Activations](../../ICLR2026/learning_theory/deep_learning_with_learnable_product-structured_activations.md)
- [\[ICLR 2026\] Does Weak-to-strong Generalization Happen under Spurious Correlations?](../../ICLR2026/learning_theory/does_weak-to-strong_generalization_happen_under_spurious_correlations.md)
- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[ICML 2026\] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering](simple_algorithms_for_bad_triangle_transversals_with_applications_to_correlation.md)
- [\[ICLR 2026\] Efficient Testing for Correlation Clustering: Improved Algorithms and Optimal Bounds](../../ICLR2026/learning_theory/efficient_testing_for_correlation_clustering_improved_algorithms_and_optimal_bou.md)

</div>

<!-- RELATED:END -->
