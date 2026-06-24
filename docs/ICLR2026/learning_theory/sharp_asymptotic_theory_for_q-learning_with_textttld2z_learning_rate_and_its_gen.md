---
title: >-
  [Paper Note] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization
description: >-
  [ICLR 2026][Learning Theory][Q-learning] This paper provides the first complete set of asymptotic theories for Q-learning using "Linear Decay to Zero" (LD2Z, $\eta_{t,n}=\eta(1-t/n)$) and its power-law generalization (PD2Z-$\nu$, $\eta_{t,n}=\eta(1-t/n)^\nu$). This includes sharp non-asymptotic error bounds, a Central Limit Theorem (CLT) for tail Polyak-Ruppert averaging estimators, and a strong invariance principle (time-consistent Gaussian approximation) for partial sum pro…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Reinforcement Learning Theory"
  - "Q-learning"
  - "Learning Rate Scheduling"
  - "LD2Z"
  - "Non-asymptotic Error Bounds"
  - "Strong Invariance Principle"
date: 2026-05-08
content_hash: b43d0b35c60bc9bc
---

# Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=WjEAMyLDoh](https://openreview.net/forum?id=WjEAMyLDoh)  
**Code**: The paper mentions a GitHub repository (specific URL not provided)  
**Area**: Learning Theory / Reinforcement Learning Theory  
**Keywords**: Q-learning, Learning Rate Scheduling, LD2Z, Non-asymptotic Error Bounds, Strong Invariance Principle

## TL;DR
This paper provides the first complete set of asymptotic theories for Q-learning using "Linear Decay to Zero" (LD2Z, $\eta_{t,n}=\eta(1-t/n)$) and its power-law generalization (PD2Z-$\nu$, $\eta_{t,n}=\eta(1-t/n)^\nu$). This includes sharp non-asymptotic error bounds, a Central Limit Theorem (CLT) for tail Polyak-Ruppert averaging estimators, and a strong invariance principle (time-consistent Gaussian approximation) for partial sum processes. It theoretically explains why this "two-stage" step size enjoys both the fast forgetting of initial values from constant step sizes and the asymptotic convergence guarantees of polynomial step sizes.

## Background & Motivation
**Background**: As a classic model-free method for estimating optimal MDP policies, the statistical properties of Q-learning (asymptotic/non-asymptotic error bounds, CLT, functional CLT) have been extensively studied. However, these theoretical works almost exclusively assume step sizes are either **constant** ($\eta_t\equiv\eta$) or **polynomial decay** ($\eta_t=\eta t^{-\alpha}$).

**Limitations of Prior Work**: Both classes of step sizes have significant drawbacks. Constant step sizes converge quickly and "forget" initial values rapidly but converge to a **stationary distribution** around $Q^\star$—meaning a non-negligible **asymptotic bias** exists unless eliminated by techniques like jackknife. Polynomial decay step sizes are theoretically elegant (achieving asymptotic normality under Polyak-Ruppert averaging) but converge **extremely slowly**: the rate of "forgetting" initial values is only $\exp(-ct^{1-\alpha})$, which slows down as $\alpha$ approaches $1$.

**Key Challenge**: There is a trade-off between "fast forgetting of initial values" and "zero asymptotic bias." Constant step sizes achieve the former, and polynomial step sizes achieve the latter; no step size previously analyzed in theory combined both.

**Goal**: Empirically, the LD2Z step size commonly used in the deep learning community (e.g., in BERT, LLaMA, and various "knee schedules" with warm-up) performs exceptionally well, but its asymptotic/statistical properties are a vacuum, especially in the context of Q-learning. This paper aims to answer: (1) what is the non-asymptotic convergence rate for LD2Z and its generalizations in Q-learning; and (2) whether valid statistical inference (CLT + bootstrap) can be performed.

**Key Insight**: Instead of directly analyzing the $\nu=1$ LD2Z, the authors first consider a more general power-law family, PD2Z-$\nu$ ($\eta_{t,n}=\eta(1-t/n)^\nu$, where LD2Z is the special case $\nu=1$). This allows them to address a deeper question: "is linear ($\nu=1$) sufficient, or should $\nu$ be tuned with iterations?"

**Core Idea**: By treating Q-learning as a Stochastic Approximation (SA) targeting the Bellman equation, the authors **incrementally establish** theory for PD2Z-$\nu$ step sizes—from non-asymptotic moment bounds to tail PR averaging CLT and then the strong invariance principle—proving that this family possesses "best-of-both-worlds" attributes.

## Method

### Overall Architecture
This is a **purely theoretical** work with no algorithmic pipeline to diagram. The "method" consists of a hierarchical chain of proofs. Under standard regularity assumptions (bounded $p$-th moment of rewards + local attraction basin condition near the optimal policy), the authors analyze the synchronous Q-learning iteration:

$$Q_{t,n}=(1-\eta_{t,n})Q_{t-1,n}+\eta_{t,n}\widehat{B}_tQ_{t-1,n},\qquad \widehat{B}_t\ \text{is the empirical Bellman operator}$$

Using PD2Z-$\nu$ step sizes, they produce three layers of results: (1) **Non-asymptotic error bounds** for any $p\ge 2$ moment, revealing a two-stage rate of "fast transient decay + slow convergence decay"; (2) a **tail Polyak-Ruppert averaging** estimator with proven **asymptotic normality**; (3) a **strong invariance principle** (time-consistent Gaussian approximation) for the partial sum process to support bootstrap inference.

### Key Designs

**1. Two-stage Non-asymptotic Error Bound for PD2Z-$\nu$**

This is the foundation of the paper (Theorem 3.1). It addresses the lack of analytical characterization for the "fast drop then stabilization" observed empirically with LD2Z. The authors prove that for $\eta<\frac{2(1-\gamma)}{(1-\gamma)^2+2(p-1)\gamma^2}$, the $t$-th iteration satisfies:
$$\|Q_{t,n}-Q^\star\|_p\ \le\ \exp\!\big(-c_3\eta t(1-n^{-1})^\nu\big)\,|Q_0-Q^\star|\ +\ \begin{cases}\sqrt{C_1}\,\sqrt{\eta_{t,n}}, & t\le n-\tfrac{2}{(c_3\eta)^{1/(\nu+1)}}n^{\nu/(\nu+1)}\\[4pt]\sqrt{C_2}\,n^{-\frac{\nu}{2(\nu+1)}}, & t> n-\tfrac{2}{(c_3\eta)^{1/(\nu+1)}}n^{\nu/(\nu+1)}\end{cases}$$
Key insight: **Two regimes, two rates**. In the **transient phase** ($t\lesssim n^c, c<1$), $\eta_{t,n}\asymp 1$, step size behaves like a constant, and the bias dissipates exponentially. After crossing the threshold into the **convergence phase**, the error is locked at $n^{-\nu/(2(\nu+1))}$. The contrast in forgetting speed is sharp: PD2Z-$\nu$ forgets at $\exp(-ct)$ (exponentially), whereas polynomial steps (Theorem 3.3) only achieve $\exp(-ct^{1-\alpha})$.

**2. Optimal Scaling of $\nu$: Why $\nu=1$ (Pure Linear) is Sufficient**

Corollary 3.2 simplifies the bound to $\exp(-c_3\eta(1-n^{-1})^\nu t)|Q_0-Q^\star|+O(\sqrt{\eta_{t,n}}\vee n^{-\nu/(2(\nu+1))})$ and shows that at $t=n$, the minimum of the right-hand side regarding $\nu$ occurs at $\nu\asymp\log_2\log n$. This mechanism (Remark 3.3) is a trade-off: the constant $C_2$ increases with $\nu$, while the decay term $n^{-\nu/(2(\nu+1))}$ decreases. $\log_2\log n$ is effectively constant for any realistic $n$, theoretically **justifying a fixed $\nu$ independent of iterations**—specifically $\nu=1$ for LD2Z.

**3. Tail Polyak-Ruppert Averaging + CLT: Bypassing Constant-Phase Bias**

Standard PR averaging (averaging over all $n$ steps) **fails** under LD2Z. The authors analyze the process in segments: when $t\le n/2$, the step size $\eta_{t,n}\ge\eta/2^\nu$ is still "constant-level," meaning the early iterations do not converge to $Q^\star$ and lack Gaussianity. Averaging the entire trajectory leads to errors much larger than $Q^\star$ (as shown in Figure 4). Thus, the authors propose averaging only the **latest iterations**, defining the tail PR estimator:
$$\bar{Q}_n=\frac{1}{\lfloor cn^{\nu/(\nu+1)}\rfloor}\sum_{t=n-\lfloor cn^{\nu/(\nu+1)}\rfloor+1}^{n}Q_{t,n},$$
proviing (Theorem 3.5) that $n^{\nu/(2(\nu+1))}(\bar{Q}_n-Q^\star)\xrightarrow{w}N(0,\Sigma)$.

**4. Strong Invariance Principle for Gaussian Bootstrap**

To bypass the analytical complexity of $\Sigma$, the authors establish a **strong invariance principle** (strong Gaussian approximation). Since $(Q_{t,n})$ is **non-stationary**, standard Brownian motion approximations fail. They utilize a non-stationary Gaussian process that matches the covariance structure:
$$Y_t=(I-\eta_{t,n}G)Y_{t-1}+\eta_{t,n}\aleph_t,\qquad G=I-\gamma H_{\pi^\star}.$$
Theorem 4.1 proves that one can construct $Q^c_{t,n}\overset{D}{=}Q_{t,n}$ such that the partial sum process of $Q^c_l-Q^\star$ is closely approximated by $Y_l$. This allows for Gaussian bootstrap by running multiple independent $Y_t$ chains to perform time-consistent inference on $\bar{Q}_n$.

## Key Experimental Results

Experiments are numerical simulations performed on a $4\times4$ FrozenLake grid world ($\gamma=0.1$, rewards $10$ or $5$ for specific states, $-1$ for out-of-bounds, 0.9 transition probability) with Monte-Carlo repetitions.

### Main Results (Comparison of Learning Rates, §5.2)

| Comparison | Setting | Phenomenon |
|------------|---------|------------|
| LD2Z vs. Poly Decay vs. Constant | $\eta=0.05, n=5000, B=1000$ chains | LD2Z significantly outperforms poly decay (Fig 1). Constant step converges to a stationary distribution with bias; PD2Z-$\nu$ outperforms constant throughout. |
| Transient Error $|Q_{t,n}-Q^\star|_\infty$ | $1000\le t\le n$, Mean ±1 SD | Increasing $\nu$ ($\nu=2,3$) **slightly** reduces error when $t < n$. |
| Terminal Error $|Q_{n,n}-Q^\star|_\infty$ | $n\in\{500,\dots,2500\}$ | Terminal error for $\nu\in\{1,2,3\}$ is **nearly identical**, justifying $\nu=1$. |

### Time-Consistent Approximation and CLT Verification (§5.3–5.4)

| Experiment | Configuration | Key Finding |
|------------|---------------|-------------|
| Strong Invariance Q–Q Plot | LD2Z $\eta_{t,n}=0.05(1-t/n)$ | Max partial sums of $Q^c_l-Q^\star$ quantiles match the Gaussian approximation $\sum Y_l$ perfectly. |
| Poly Step Approximation | $\eta_t=0.05t^{-0.65}$ | Covariance-matching approximation outperforms Brownian motion (Functional CLT) in sup-norm consistency. |
| Tail PR vs. Standard PR | $n\in\{1000,\dots,5000\}$ | $L_\infty$ error of $\bar{Q}_n$ (tail) is significantly lower than $\tilde{Q}_n$ (full), which is contaminated by the early constant-step phase. |

### Key Findings
- **The two-stage rate is real**: Errors first decay exponentially to shed the initial bias and then stabilize at $n^{-\nu/(2(\nu+1))}$ during the convergence phase.
- **$\nu$ has negligible impact on final accuracy**: The theoretically optimal $\nu\asymp\log\log n$ grows so slowly that $\nu=1,2,3$ produce near-identical results.
- **Averaging must be tail-based**: Including the early "constant step-like" phase in PR averaging prevents convergence; tail averaging restores asymptotic normality.

## Highlights & Insights
- **Generalization strategy**: By generalizing LD2Z to PD2Z-$\nu$, the authors provide a rigorous theoretical justification for an engineering heuristic (using $\nu=1$).
- **Covariance-matching Gaussian process**: This technique is more precise than functional CLTs for non-stationary SA/SGD processes and can be adapted to other settings.
- **Tail PR Averaging**: This identifies a critical failure mode of standard PR averaging for decay-to-zero schedules, serving as a warning for future SA inference.
- **Weak Assumptions**: Requires only finite $p$-th moments of rewards rather than bounded rewards, utilizing sharp Burkholder inequalities.

## Limitations & Future Work
- **Fixed $n$ requirement**: LD2Z requires knowing the total number of steps $n$ in advance, making it primarily suitable for **offline RL**. Extending this to **online RL** is an open direction.
- **Analytical $\Sigma$ complexity**: Since the asymptotic covariance of tail PR averaging is difficult to handle analytically, one must rely on bootstrap. The Berry-Esseen bounds for this CLT remain for future work.
- **Scale and Horizon**: Experiments are limited to a small grid with $\gamma=0.1$. In scenarios with $\gamma \to 1$ (long horizon), the constant factors and sample complexity may degrade significantly.

## Related Work & Insights
- **vs. Goldreich et al. (2025)**: They analyzed LD2Z for strongly convex SGD; **Ours** extends this to the more complex Q-learning setting and adds CLT and strong invariance.
- **vs. Li et al. (2023b) / Chen et al. (2020b)**: These works utilize functional CLT for polynomial steps; **Ours** shows that covariance-matching is sharper for sup-norm approximations and demonstrates the exponential forgetting speed of LD2Z.
- **vs. Li et al. (2024a)**: **Ours** achieves a similar sample complexity rate for large $\nu$ but under the much weaker assumption of finite $p$-th moment rewards.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Admissible Heuristics for A*: Theory and Practice](learning_admissible_heuristics_for_a_theory_and_practice.md)
- [\[ICLR 2026\] Towards a Sharp Analysis of Offline Policy Learning for f-Divergence-Regularized Contextual Bandits](towards_a_sharp_analysis_of_offline_policy_learning_for_f-divergence-regularized.md)
- [\[ICML 2026\] Performative Learning Theory](../../ICML2026/learning_theory/performative_learning_theory.md)
- [\[ICLR 2026\] Almost Bayesian: Dynamics of SGD Through Singular Learning Theory](almost_bayesian_dynamics_of_sgd_through_singular_learning_theory.md)
- [\[ICLR 2026\] Pretrain–Test Task Alignment Governs Generalization in In-Context Learning](pretraintest_task_alignment_governs_generalization_in_in-context_learning.md)

</div>

<!-- RELATED:END -->
