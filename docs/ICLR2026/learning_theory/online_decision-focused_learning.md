---
title: >-
  [Paper Note] Online Decision-Focused Learning
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper generalizes Decision-Focused Learning (DFL) from the i.i.d. batch setting to an online setting where objective functions vary over time. By employing two strategies—regularizing the inner problem for differentiability and utilizing approximate oracles with perturbations to handle non-convexity—the authors pr
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 0f64a7744c3723be
---
# Online Decision-Focused Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FJhtHBphCt](https://openreview.net/forum?id=FJhtHBphCt)  
**Code**: To be confirmed  
**Area**: Learning Theory / Online Learning / Decision-Focused Learning  
**Keywords**: Decision-focused learning, Online learning, Static/Dynamic regret, Bi-level optimization, Perturbed leader

## TL;DR
This paper generalizes Decision-Focused Learning (DFL) from the i.i.d. batch setting to an online setting where objective functions vary over time. By employing two strategies—regularizing the inner problem for differentiability and utilizing approximate oracles with perturbations to handle non-convexity—the authors propose DF-FTPL and DF-OGD algorithms, providing provable static and dynamic regret upper bounds, respectively.

## Background & Motivation
**Background**: Many real-world decision problems involve making choices under uncertainty. The mainstream "predict-the-optimize" framework first trains a predictive model on historical data and then feeds the predictions into an optimization problem to guide decisions. Decision-Focused Learning (DFL, or smart predict-then-optimize) goes further: instead of simply optimizing prediction accuracy, the model directly minimizes the loss incurred by downstream decisions, making it more robust to prediction errors.

**Limitations of Prior Work**: Existing theoretical analyses of DFL are restricted to the **batch setting**, assuming that data are pre-collected i.i.d. samples and the objective function remains fixed. In reality, environments are dynamic and data distributions drift, making this assumption invalid.

**Key Challenge**: Moving DFL to an online setting compounds two types of difficulties. First, the DFL objective $f_t(\theta)=\langle \bar g_t(X_t), w_t^\star(\theta)\rangle$ involves an inner problem of "minimizing a linear function over a polytope," where the optimal solution $w_t^\star(\theta)$ jumps between a finite number of vertices. Consequently, the **gradient is either zero or undefined**, prohibiting standard first-order methods. Second, due to the bi-level structure, the loss is **generally non-convex**. Online learning further requires the objective to change arbitrarily over time. Prior work on online bi-level optimization relies on smoothness assumptions, which are incompatible with the DFL structure.

**Goal**: Establish the first set of provable theoretical guarantees for online decision-focused learning and design algorithms that are practically executable.

**Key Insight**: The authors formulate the problem as a sequential bi-level optimization, where each round's inner problem involves solving a linear program over a polytope and the outer problem optimizes the resulting decision cost. Since gradients are non-existent, the inner problem is regularized into a differentiable surrogate. To handle non-convexity, the approach leverages the "approximate oracle + perturbation" strategy common in recent non-convex online learning research.

**Core Idea**: By combining inner-decision regularization with approximate oracles and perturbations to bypass non-convexity, DFL is transformed into a method capable of online updates with provably sub-linear regret convergence.

## Method

### Overall Architecture
Consider a sequence of $T$ decision rounds. In each round $t$, nature determines the distribution of covariates $X_t$ and a cost function $\bar g_t$. The decision-maker **first observes $X_t$** (without seeing the true cost $\bar g_t(X_t)$), predicts the cost using a parameterized model $g(\theta_t, X_t)$, and takes an action in a convex polytope $W=\mathrm{Conv}(v_1,\dots,v_K)$:

$$w_t^\star(\theta_t)\in\arg\min_{w\in W}\langle g(\theta_t,X_t), w\rangle .$$

At the end of the round, the true cost $\bar g_t(X_t)$ is revealed, and the decision-maker updates the parameters to $\theta_{t+1}$. The core of decision-focused learning is to minimize the downstream decision cost $\langle \bar g_t(X_t), w_t^\star(\theta)\rangle$ during parameter updates, which is a **bi-level optimization**.

Since the inner optimal solution jumps between vertices, the outer objective $f_t$ is neither differentiable nor convex. The overall logic is: ① Add a regularization term $R$ to the inner objective to obtain a differentiable surrogate solution $\tilde w_t(\theta)$ residing in the strict interior of the polytope, making the surrogate objective $\tilde f_t$ differentiable. ② Update the non-convex surrogate objective using an "approximate offline oracle + random perturbation," yielding two algorithms: DF-FTPL (perturbing cumulative loss for static regret) and DF-OGD (perturbing gradient evaluation points for dynamic regret). This framework is purely theoretical, centered on regularization choices, oracle assumptions, and regret proofs.

The two regret measures are defined as:

$$R^s_T=\sum_{t\in[T]} f_t(\theta_t)-\inf_{\theta\in\Theta}\sum_{t\in[T]}\mathbb E[f_t(\theta)],\qquad R^d_T=\sum_{t\in[T]} F_t(\theta_t)-\sum_{t\in[T]}\inf_{\theta\in\Theta} F_t(\theta),$$

where $F_t(\theta)=\mathbb E[f_t(\theta)\mid \mathcal H_{t-1}]$. Static regret compares against the best fixed strategy, while dynamic regret compares against an optimal sequence of oracles per round, using conditional expectations to prevent the comparator from overfitting to $X_t$.

### Key Designs

**1. Regularizing the Inner Problem for Differentiability: Smoothing Vertex Solutions**
The non-differentiability stems from $w_t^\star(\theta)$ jumping across vertex set $\{v_1,\dots,v_K\}$. Following Wilder et al. (2019), a regularization term $R$ is added to the inner objective to yield surrogate decisions:

$$\tilde w_t(\theta)\in\arg\min_{w\in W}\big\{\langle g(\theta,X_t),w\rangle+\alpha_t R(w)\big\},$$

minimizing the surrogate objective $\tilde f_t(\theta)=\langle\bar g_t(X_t),\tilde w_t(\theta)\rangle$ with gradient $\nabla\tilde f_t(\theta)=\nabla\tilde w_t(\theta)^\top\bar g_t(X_t)$. For general polytopes $W=\{w: A^\top w\preceq b\}$, a **log-barrier** $R(w)=-\sum_{i=1}^n\ln(b_i-A_i^\top w)$ is used. For simplexes, **negative entropy** $R_0(w)=\sum_i w_i\ln w_i$ is used, where $\tilde w_t$ becomes a softmax. There is a fundamental trade-off: larger $\alpha_t$ improves smoothness but increases deviation from $w_t^\star$.

**2. Approximate Oracle + Perturbation to Bypass Non-convexity**
Even with strongly convex $R$, $\tilde f_t$ remains **generally non-convex**, rendering standard online convex optimization ineffective. The authors assume access to a $\xi$-approximate offline oracle $O_\xi$ that outputs $\vartheta=O_\xi(f)$ such that $f(\vartheta)\le\inf_\theta f(\theta)+\xi$. This reflects the reality that global optima are hard to reach in non-convex landscapes, where $\xi$ measures solution quality.

Importantly, since DFL gradients are zero or undefined, the concept of local regret is **meaningless**, necessitating the oracle + perturbation route. Shifting cumulative costs with random noise stabilizes updates in non-convex settings.

**3. DF-FTPL: Perturbing Cumulative Loss for Static Regret**
DF-FTPL (Algorithm 1) is derived from Follow-the-Perturbed-Leader. In each round, a noise vector $\sigma_t$ with exponential distribution $\sigma_{j,t}\sim\mathrm{Exp}(\eta)$ is sampled, and the oracle computes:

$$\theta_{t+1}=O_\xi\Big(\sum_{i=1}^t \tilde f_i-\langle\sigma_t,\cdot\rangle\Big).$$

Theorem 1 provides the static regret bound under assumptions H1 and H2: with constant $\alpha_t=\alpha$, the average regret $T^{-1}\mathbb E[R^s_T]$ is $\tilde O(m^{3/4}n^{3/2}T^{-1/4}+\xi)$. The regret depends polynomially on the parameter dimension $m$ but only via $\ln\ln(d)$ on the decision space dimension $d$, making it efficient for high-dimensional decision spaces.

**4. DF-OGD: Perturbing Gradient Evaluation Points for Dynamic Regret**
To handle non-stationary environments, DF-OGD (Algorithm 2) uses Online Gradient Descent. Each round samples $\delta_t\sim\mathrm{Unif}[0,1]$ to evaluate the gradient at a random point $u_t$ between the current parameter $\theta_t$ and the oracle's near-optimal solution $\vartheta_t$. Theorem 2 provides a dynamic regret bound of $T^{-1}\mathbb E[R^d_T]=\tilde O(\mathbb E[\sqrt n(1+P_T)^{1/4}T^{-1/4}]+\xi)$, where $P_T$ is the path length (drift of near-optimal solutions). Notably, this bound is **independent of parameter dimension $m$**.

> **Key Challenge (Margin Condition H2)**: There exists $C_0>0, \beta\in(0,1]$ such that $\mathbb P\big(\inf_{j\ne I_t(\theta)}\{u_j(\theta,X_t)-u_{I_t(\theta)}(\theta,X_t)\}\ge\varepsilon\mid\mathcal H_{t-1}\big)\ge 1-C_0\varepsilon^\beta$. This margin condition ensures the optimal decision is identifiable, controlling the distance between $w_t^\star$ and $\tilde w_t$.

## Key Experimental Results

### Main Results (Knapsack Problem)
Experimentation utilizes a knapsack problem where items are selected based on predicted costs. Synthetic data involve non-linear, non-stationary processes, while the decision-maker uses a misspecified linear model. Comparisons include PF-OGD (Predict-Focused) and online SPO (Smart Predict-then-Optimize).

| Metric | DF-FTPL / DF-OGD (Ours) | PF-OGD | online SPO | Conclusion |
|------|------|------|------|------|
| Avg. Decision Cost $t^{-1}\sum \bar g$ | Lower (Better) | Higher | Higher | Ours performs better in both static/dynamic settings |
| Avg. Prediction Error (MSE) | Higher | Lower | Mid | Aligns with DFL goal: optimize decision, not accuracy |

### Key Findings
- **Verification of DFL's nature**: The algorithms produce lower decision costs despite higher MSE, confirming that DFL prioritizes decision quality over statistical accuracy, especially when models are misspecified.
- **Superiority**: The proposed methods outperform both PFL and the widely used online SPO upper-bound surrogate.

## Highlights & Insights
- **Decoupling Hurdles**: The separation of "regularization for differentiability" and "oracles/perturbation for non-convexity" provides a clean framework applicable to other bi-level online problems with linear inner programs.
- **Asymmetric Dimension Dependence**: The bounds have extremely weak dependence on decision dimension $d$ ($\ln\ln d$), making them suitable for high-dimensional tasks.
- **Robust Dynamic Regret**: Achieving sub-linear dynamic regret under conditions where functions are non-Lipschitz and non-stationary is a significant contribution, facilitated by the margin assumption H2.

## Limitations & Future Work
- **Convergence Rate**: The $T^{-1/4}$ rate is slower than the $T^{-1/2}$ seen in smooth single-layer settings, a cost of the bi-level non-differentiable trade-off.
- **Oracle Dependence**: The theory relies on the availability of a $\xi$-approximate oracle and the H2 margin condition, which may be difficult to verify in complex models.
- **Synthetic Scale**: Testing was conducted on synthetic knapsack tasks with linear predictors; validation on large-scale real-world operations research tasks is needed.

## Related Work & Insights
- **Comparison to Batch DFL**: Previous works assume fixed i.i.d. data; this paper is the first to extend DFL to online settings with arbitrary distribution shifts.
- **Comparison to Online Bi-level Optimization**: Prior works require smoothness; the proposed regularization approach bridges the gap for the non-smooth DFL structure.
- **Comparison to Non-convex Online Learning**: While drawing from FTPL/oracle literature, this work tackles the specific challenges of bi-level non-differentiability, yielding a $T^{-1/4}$ rate.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

- **Smart Predict-then-Optimize** (Elmachtoub & Grigas, 2022)
- **Melding the Data-Decisions Pipeline** (Wilder et al., 2019)
- **Follow the Perturbed Leader** (Suggala & Netrapalli, 2020)

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Online Decision Making with Generative Action Sets](online_decision_making_with_generative_action_sets.md)
- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Oracle-Efficient Hybrid Online Learning with Constrained Adversaries](oracle-efficient_hybrid_learning_with_constrained_adversaries.md)
- [\[ICLR 2026\] Conformalized Decision Risk Assessment](conformalized_decision_risk_assessment.md)
- [\[ICLR 2026\] Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms](decision-theoretic_approaches_for_improved_learning-augmented_algorithms.md)

</div>

<!-- RELATED:END -->
