---
title: >-
  [Paper Note] Lipschitz Bandits with Stochastic Delayed Feedback
description: >-
  [ICLR 2026][learning_theory][Lipschitz bandit] This paper presents the first systematic study of the learning problem for Lipschitz bandits in continuous arm spaces under stochastic delayed feedback. It proposes the Delayed Zooming algorithm for bounded delays (maintaining a sub-optimality gap bound of $\Delta(x) \leq 6r_t(x)$ via a lazy update mechanism) and the D
tags:
  - ICLR 2026
  - learning_theory
  - Lipschitz bandit
date: 2026-05-08
content_hash: 23d3505bab22c6d9
---
# Lipschitz Bandits with Stochastic Delayed Feedback

**Conference**: ICLR 2026  
**arXiv**: [2510.00309](https://arxiv.org/abs/2510.00309)  
**Code**: None  
**Area**: Online Learning / Bandit Algorithms  
**Keywords**: Lipschitz bandit, delayed feedback, zooming algorithm, phased pruning, regret lower bounds, quantiles

## TL;DR

This paper presents the first systematic study of the learning problem for Lipschitz bandits in continuous arm spaces under stochastic delayed feedback. It proposes the Delayed Zooming algorithm for bounded delays (maintaining a sub-optimality gap bound of $\Delta(x) \leq 6r_t(x)$ via a lazy update mechanism) and the DLPP phased pruning strategy for unbounded delays (where regret is linked to the delay quantile $Q(p)$). Furthermore, it establishes an instance-dependent lower bound proving that DLPP is near-optimal.

## Background & Motivation

**Background**: Lipschitz bandits extend classical Multi-Armed Bandits (MAB) to continuous metric spaces $(\mathcal{A}, \mathcal{D})$, where the reward function $\mu$ satisfies the 1-Lipschitz condition. The classical Zooming algorithm achieves the optimal regret rate of $\tilde{O}(T^{(d_z+1)/(d_z+2)})$ in the delay-free setting through adaptive discretization, where $d_z$ is the zooming dimension. While delayed feedback has been extensively studied in MAB, linear bandits, and kernel bandits, the problem remains entirely unexplored for Lipschitz bandits.

**Limitations of Prior Work**: (1) The combination of continuous arm spaces and delayed feedback introduces dual complexity—each sampling point represents a neighborhood, and its estimation depends on delayed observations. (2) The core analysis of the Zooming algorithm relies on the property that "confidence radii change only when an arm is pulled"—this property is broken under delay because confidence radii of unpulled arms shrink when delayed rewards arrive. (3) Finite-arm delay methods (e.g., Delayed-UCB1) do not involve coverage arguments for continuous spaces and cannot be directly generalized.

**Key Challenge**: How to maintain efficient adaptive exploration of a continuous metric space when feedback is delayed and potentially permanently missing?

**Goal**: Design Lipschitz bandit algorithms that achieve near-optimal regret rates under both bounded and unbounded stochastic delays and prove their optimality.

**Key Insight**: Categorize the problem into two sub-problems based on delay support—using "zooming + lazy update" for bounded delays, and a "phased accumulation of reliable feedback + elimination" strategy for unbounded delays.

**Core Idea**: Control the rate of confidence radius shrinkage via lazy updates (for bounded delays), or relate the number of pulls to observations through phased round-robin sampling and the $Q(p)$ quantile (for unbounded delays) to recover the optimal delay-free regret rate.

## Method

### Overall Architecture

The problem addressed in this paper is conducting Lipschitz bandit optimization on a continuous arm space where rewards are not returned immediately but after a stochastic delay, or may never arrive. The problem is formalized as a triplet $(\mathcal{A}, \mathcal{D}, \mu)$, where $\mathcal{A}$ is a compact doubling metric space, $\mathcal{D}$ is the metric, and $\mu: \mathcal{A} \to [0,1]$ is an unknown 1-Lipschitz reward function. At each round $t$, an arm $x_t$ is selected, generating a reward $y_t = \mu(x_t) + \epsilon_t$ (where $\epsilon_t$ is sub-Gaussian noise). However, this reward is observed only after a stochastic delay $\tau_t \sim f_\tau$ (independent of arms and rewards). The objective is to minimize cumulative regret $R(T) = \sum_{t=1}^T (\mu^* - \mu(x_t))$.

The work bifurcates based on the severity of the delay into two algorithmic pipelines, complemented by a lower bound. For bounded delays ($\tau_t \leq \tau_{\max}$), the algorithm follows the adaptive discretization framework of classical Zooming but employs a **Delayed Zooming** lazy update mechanism to lock the speed of confidence radius shrinkage, recovering the optimal regret rate. For unbounded delays (including permanent feedback loss), where maintaining real-time confidence radii is infeasible, the **DLPP** strategy is used to "prune after accumulating sufficient reliable feedback," linking regret to the delay quantile $Q(p)$ rather than worst-case scenarios. Finally, a matching **instance-dependent lower bound** proves the near-optimality of DLPP.

### Key Designs

**1. Delayed Zooming: Locking the Confidence Radius Shrinkage Rate via Lazy Updates (Bounded Delay $\tau_t \leq \tau_{\max}$)**

The classical Zooming analysis has a critical vulnerability: confidence radii only change when an arm is "pulled," ensuring they are strictly synchronized with the pull count. Under delay, rewards from previously pulled arms may arrive later, causing the confidence radius of an arm **not currently being pulled** to shrink, which collapses the coverage-activation argument. Delayed Zooming addresses this in two steps. First, it calculates the confidence radius based on actual observation counts $v_t(x)$ rather than pull counts $n_t(x)$:

$$r_t(x) = \sqrt{\frac{4\log T + 2\log(2/\delta)}{1 + v_t(x)}}$$

Thus, the radius reflects only acquired information. Second, it introduces a **lazy update mechanism**: a buffer queue $Q[x]$ is maintained for each active arm. Let $s$ be the last time the arm was pulled. If accumulated observations lead to $v_t(x)+1 > 4 v_s(x)$, subsequent feedback is buffered and the radius is not updated until the arm is pulled again. By capping the observation growth rate at "no more than 4 times the previous count," the algorithm guarantees $r_t(x) \geq \tfrac{1}{2} r_s(x)$—ensuring the radius shrinks by at most half between pulls. The cost is the relaxation of the sub-optimality gap from $\Delta(x) \leq 3r_t(x)$ to $\Delta(x) \leq 6r_t(x)$, but the coverage argument is restored, resulting in a regret bound of $\tilde{O}\big(T^{\frac{d_z+1}{d_z+2}} + \tau_{\max} T^{\frac{d_z}{d_z+2}}\big)$, where delay contributes only an additive term.

**2. DLPP: Pruning after Accumulating Sufficient Reliable Feedback (Unbounded/Missing Feedback $\tau = \infty$)**

When delays are potentially unbounded or feedback is entirely lost, real-time radius updates become meaningless. DLPP abandons real-time updates for a phased elimination strategy. In phase $m$, the algorithm maintains a set of covering balls $\mathcal{B}_m$ with radius $r_m = 2^{-m}$, performing uniform round-robin sampling until each ball accumulates

$$v_m = \frac{2\log T + \log(2/\delta)}{2r_m^2}$$

observations. At the end of the phase, pruning occurs based on empirical means: any region where $\hat\mu_m^* - \hat\mu_m(B) \geq 8r_m$ is eliminated, and surviving regions are subdivided for the next phase. The challenge is that sampling follows pull counts, while decisions depend on observation counts, which are out of sync. DLPP uses Chernoff inequalities to relate their probabilities:

$$\Pr\!\Big(v_{t+Q(p)}(B) \leq \tfrac{p}{2}\, n_t(B)\Big) \leq \exp\!\Big(-\tfrac{p}{8}\, n_t(B)\Big)$$

Crucially, waiting $Q(p)$ rounds ensures that at least a $p/2$ proportion of pulled arms result in available observations. Consequently, regret is naturally linked to the **delay quantile** $Q(p)$ (where a $p$ proportion of feedback arrives within $Q(p)$ rounds). Even if some feedback is permanently missing, the algorithm functions with a regret bound of:

$$R(T) \lesssim \min_{p \in (0,1]} \left\{ \frac{1}{p} T^{\frac{d_z+1}{d_z+2}} \Big(c\log\tfrac{T}{\delta}\Big)^{\frac{1}{d_z+2}} + Q(p) \right\}$$

Optimizing over $p$ allows the bound to adapt to the "central mass" of the delay distribution rather than its worst case.

**3. Instance-Dependent Lower Bound: Proving DLPP Near-Optimality**

The paper provides a lower bound matching the upper bound's form, demonstrating that DLPP is near-optimal up to logarithmic factors. The construction uses a specific family of delay distributions: delay is a fixed $\tau_0$ with probability $p$, and $\infty$ otherwise. Using Bernoulli sampling coupling, the delayed Lipschitz bandit is reduced to a delay-free version—simulating an algorithm that sees instant feedback with probability $p$—thereby transferring the delay-free lower bound. The final result is:

$$R(T) \gtrsim \frac{T^{(d_z+1)/(d_z+2)}(c\log T)^{1/(d_z+2)}}{p\log T} - \frac{1}{p} + \bar\Delta \cdot Q(p)$$

where $\bar\Delta = \int_\mathcal{A} \Delta(x) \big/ \int_\mathcal{A} 1$ is the average sub-optimality gap. The term $\bar\Delta \cdot Q(p)$ represents the unavoidable regret during the initial "blind flight" period of $Q(p)$ rounds where no feedback is received, matching the $Q(p)$ term in the upper bound.

### Loss & Training

Both algorithms are theoretical and do not require gradient-based training. Key parameters are derived theoretically: the lazy update threshold $4v_s(x)$ for Delayed Zooming, and the required observations $v_m$ and pruning threshold $8r_m$ per phase for DLPP. Regret rate optimization is achieved by choosing $\rho = (\log T / T)^{1/(d_z+2)}$ (bounded delay) or $M = \frac{\log(T/(c\log T))}{d_z+2}$ (DLPP).

## Key Experimental Results

### Main Results (3 Reward Functions × 2 Delay Distributions, $T=60000$, 30 Trials)

| Reward Function | Algorithm | No Delay | Uniform $\mathbb{E}[\tau]=20$ | Uniform $\mathbb{E}[\tau]=50$ | Geometric $\mathbb{E}[\tau]=20$ | Geometric $\mathbb{E}[\tau]=50$ |
|-----------------|-----------|:--------:|:--------------------------:|:--------------------------:|:----------------------------:|:----------------------------:|
| Trig (1D)       | Delayed Zooming | 138.97   | 154.55                     | 171.07                     | 159.30                       | 152.98                       |
| Trig (1D)       | DLPP      | 304.60   | 314.87                     | 326.71                     | 312.44                       | 325.74                       |
| Sine (1D)       | Delayed Zooming | 130.64   | 137.31                     | 148.69                     | 132.88                       | 144.08                       |
| Sine (1D)       | DLPP      | 178.05   | 195.35                     | 209.97                     | 186.28                       | 208.80                       |
| 2D Function     | Delayed Zooming | 1445.86  | 1843.05                    | 1858.45                    | 1463.38                      | 1828.15                      |
| 2D Function     | DLPP      | **1120.64** | **1159.85** | **1136.46** | **1120.63** | **1142.55** |

### Theoretical Results Comparison

| Setting | Algorithm | Regret Upper Bound | Consistency Check |
|---------|-----------|--------------------|-------------------|
| Bounded Delay $\tau_{\max}$ | Delayed Zooming | $\tilde{O}(T^{(d_z+1)/(d_z+2)} + \tau_{\max} T^{d_z/(d_z+2)})$ | $\tau_{\max}=0$ recovers classical Lipschitz bandit |
| Bounded Delay $d_z=0$ | Delayed Zooming | $O(\sqrt{cT\log T} + c\tau_{\max})$ | Recovers finite-arm MAB with delay |
| Unbounded Delay | DLPP | $\min_p \{ \frac{1}{p} T^{(d_z+1)/(d_z+2)} (\cdot)^{1/(d_z+2)} + Q(p) \}$ | Depends on median $\tau_{\text{med}}$ when $p=0.5$ |
| Lower Bound | — | $\frac{R}{p\log T} - \frac{1}{p} + \bar\Delta \cdot Q(p)$ | Matches DLPP upper bound up to log factors |

### Key Findings

- Both algorithms maintain sub-linear regret under bounded/unbounded delays, with delay-induced extra regret being merely an additive term.
- Delayed Zooming yields lower regret in 1D scenarios, while DLPP's pruning and discretization strategy is superior in 2D (where DLPP even outperforms Zooming without delay).
- Delayed Zooming performs well in geometric distribution (unbounded) experiments, though theoretical guarantees are currently limited to bounded delays—an open problem.
- The DLPP regret curve is piecewise linear, stemming from the phase-based uniform sampling and periodic regret accumulation.

## Highlights & Insights

- **Technical Contribution of Lazy Update**: While "using $v_t$ instead of $n_t$ + buffering" seems simple, its analysis is non-trivial. It is proven that the doubling of the constant in $\Delta(x) \leq 6r_t(x)$ is precise—when $v_t(x)+1 > 4v_s(x)$, $r_t(x)$ could drop below $r_s(x)/2$, necessitating an update halt.
- **Elegance of Quantile Characterization**: The DLPP regret bound adapts to the "central mass" of the delay distribution via $\min_{p \in (0,1]}$. Selecting $p=0.5$ yields median dependence, while smaller $p$ can handle heavy-tailed distributions.
- **Lower Bound Matching**: The lower bound construction using Bernoulli coupling—simulating a delay-free algorithm with probability $p$—shows that delayed regret is at least $\Omega(R/(p\log T))$. Combined with the uninformative $\bar\Delta \cdot Q(p)$ term for the first $Q(p)$ rounds, it demonstrates that DLPP is nearly unimprovable.

## Limitations & Future Work

- Theoretical guarantees for Delayed Zooming are restricted to bounded delays; extending this to unbounded delays is an explicit open problem.
- DLPP requires a coverage oracle; calculating covers in high-dimensional spaces can be computationally expensive.
- Experiments are restricted to 1D and 2D synthetic functions (trig, sine, bimodal), lacking realistic application scenarios.
- Both algorithms require knowledge of the time horizon $T$; anytime versions are not discussed.
- Adversarial delay scenarios are not considered.

## Related Work & Insights

- **vs. Delayed-UCB1 (Joulani et al., 2013)**: Finite-arm delayed MAB; their analysis lacks coverage-activation mechanisms, making direct generalization impossible.
- **vs. BLiN (Feng et al., 2022)**: The first batched Lipschitz bandit, but limited to instant/batch feedback and axis-aligned cube partitioning in $[0,1]^d$.
- **vs. Lancewicki et al. (2021)**: Characterized regret for unbounded delay MAB using quantile functions; this work generalizes that approach to continuous spaces and proves a matching lower bound for DLPP.

## Rating

- Novelty: ⭐⭐⭐⭐ First to explore the intersection of Lipschitz bandits and delayed feedback, though the technical route builds on combinations of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid theoretical foundations (upper + matching lower bounds), with sufficient but synthetic experimental validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definitions, well-structured algorithms, and elegant theorem statements.
- Value: ⭐⭐⭐⭐ Significant theoretical contribution to bandit literature, filling an important gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bi-Lipschitz Autoencoder With Injectivity Guarantee](bi-lipschitz_autoencoder_with_injectivity_guarantee.md)
- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)
- [\[ICLR 2026\] Data-to-Energy Stochastic Dynamics](data-to-energy_stochastic_dynamics.md)

</div>

<!-- RELATED:END -->
