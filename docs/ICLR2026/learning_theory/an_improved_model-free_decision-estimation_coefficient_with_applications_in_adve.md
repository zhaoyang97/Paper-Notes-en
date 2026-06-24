---
title: >-
  [Paper Note] An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs
description: >-
  [ICLR 2026][Learning Theory][Decision-Estimation Coefficient (DEC)] This paper proposes Dig-DEC—a model-free Decision-Estimation Coefficient driven purely by information gain without optimism. It is consistently no larger than the existing optimistic DEC, enabling the first model-free learning for mixed MDPs (stochastic transitions + adversarial rewards) under bandit feedback, while tightening regret rates for online function estimation from $T^{3/4}/T^{5/6}/T^{2/3}$ to $T^{2…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Reinforcement Learning Theory"
  - "Decision-Estimation Coefficient (DEC)"
  - "Model-free Learning"
  - "Adversarial MDPs"
  - "Information Gain"
  - "Regret Bound"
date: 2026-05-08
content_hash: fbf2673debb65a41
---

# An Improved Model-free Decision-estimation Coefficient with Applications in Adversarial MDPs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=lbLAgGF8OO](https://openreview.net/forum?id=lbLAgGF8OO)  
**Code**: To be confirmed  
**Area**: Learning Theory / Reinforcement Learning Theory  
**Keywords**: Decision-Estimation Coefficient (DEC), Model-free Learning, Adversarial MDPs, Information Gain, Regret Bound

## TL;DR
This paper proposes Dig-DEC—a model-free Decision-Estimation Coefficient driven purely by information gain without optimism. It is consistently no larger than the existing optimistic DEC, enabling the first model-free learning for mixed MDPs (stochastic transitions + adversarial rewards) under bandit feedback, while tightening regret rates for online function estimation from $T^{3/4}/T^{5/6}/T^{2/3}$ to $T^{2/3}/T^{7/9}/\sqrt{T}$.

## Background & Motivation

**Background**: The complexity of online decision-making (including various RL settings) has recently been unified under the **Decision-Estimation Coefficient (DEC)** framework. [FKQR21, FGH23] proposed the DEC and the accompanying **Estimation-to-Decision (E2D)** algorithmic principle: in "Decision Making with Structured Observations (DMSO)" problems, where a learner selects policy $\pi_t$ and observes $o_t \sim M_t(\cdot|\pi_t)$, the DEC provides both upper and lower bounds on regret.

**Limitations of Prior Work**: A gap of $\log|\mathcal{M}|$ exists between the upper and lower bounds in [FGH23] ($\mathcal{M}$ is the model class). This gap represents the **cost of model estimation**—the lower bound reflects the difficulty of "decision/exploration," while the upper bound carries the overhead of "estimating the entire model class." Since many practical RL algorithms are **model-free and value-based** (estimating only the value function class $\mathcal{F}$ rather than the model), the cost should theoretically only involve $\log|\mathcal{F}|$. To address this, [FGQ+23] proposed the **optimistic DEC**, reducing the upper bound from $|\mathcal{M}|$ to only depend on the value function class $|\mathcal{F}|$.

**Key Challenge**: However, the optimistic DEC relies on the **optimism principle** to drive exploration (incorporating the optimistic value $V_\phi(\pi_\phi)$ into the objective). The original spirit of the DEC framework was to explore **purely based on information gain**. Optimism can be non-essential and suboptimal in certain cases and, more critically, it only handles **stochastic environments**. In adversarial environments (where rewards change per round), optimism-based updates require the **explicit construction of reward estimators**, which is impossible under bandit feedback. [LWZ25] applied the DEC framework to mixed MDPs (stochastic transitions, adversarial rewards) but provided only "model-based algorithms (high estimation error)" or "model-free algorithms requiring full-information reward feedback," leaving the **model-free + bandit feedback case as an open problem**.

**Goal**: (1) Design a model-free DEC that removes optimism and returns to pure information gain, with complexity consistently no worse than optimistic DEC; (2) Resolve model-free bandit learning for mixed MDPs; (3) Improve the rates of online function estimation.

**Key Insight**: The authors observe that the role of optimism in the objective can be **replaced by two information gain terms**: a KL **regularization term** to stabilize the posterior (replacing the optimistic value) and a KL **information gain term** to capture distributional differences ignored by the optimistic DEC.

**Core Idea**: By replacing the optimistic value $V_\phi(\pi_\phi)$ with "dual information gain," the resulting complexity metric, denoted as **Dig-DEC**, is consistently $\le$ optimistic DEC. It can be arbitrarily smaller in special cases and naturally adapts to adversarial rewards since it no longer requires reward estimators.

## Method

### Overall Architecture

The setting is **DMSO (Decision Making with Structured Observations)**: model space $\mathcal{M}$, policy space $\Pi$, observation space $\mathcal{O}$, and value function $V_M: \Pi \to [0,1]$. Each round, the environment selects $M_t$ (unobserved), the learner selects $\pi_t$ and observes $o_t$. Regret against a reference policy $\pi^\star$ is $\mathrm{Reg}(\pi^\star) = \sum_t \big(V_{M_t}(\pi^\star) - V_{M_t}(\pi_t)\big)$.

Ours follows the **infoset + $\Phi$-restricted environment** formulation from [LWZ25] to unify all cases: partitioning $\mathcal{M} \times \Pi$ into disjoint subsets $\phi$, each corresponding to a unique optimal policy $\pi_\phi$. Stochastic MDPs, mixed MDPs, and fully adversarial settings are special cases of this $\Phi$-restricted environment—differing only in whether the adversary can change the model within $\phi^\star$ per round (Stochastic: $M^\star$ fixed; Mixed: transition $P^\star$ fixed, reward $R_t$ arbitrary; Adversarial: arbitrary per round).

The method is a **unified minimax + posterior update** loop (Algorithm 1). Given an infoset distribution $\rho_t \in \Delta(\Phi)$, solve the saddle-point objective with a **general divergence $D$**:
$$\mathrm{AIR}^{\Phi,D}_\eta(p,\nu;\rho) = \mathbb{E}_{\pi\sim p} \mathbb{E}_{(M,\pi^\star)\sim\nu} \Big[V_M(\pi^\star) - V_M(\pi) - \tfrac{1}{\eta}D_\pi(\nu\|\rho)\Big],$$
$$p_t, \nu_t = \arg\min_{p\in\Delta(\Pi)} \max_{\nu\in\Delta(\Psi)} \mathrm{AIR}^{\Phi,D}_\eta(p,\nu;\rho_t),$$
execute $\pi_t \sim p_t$, observe $o_t$, and perform `PosteriorUpdate` to get $\rho_{t+1}$. Regret is decomposed into two parts (Theorem 6):
$$\mathbb{E}[\mathrm{Reg}(\pi_{\phi^\star})] \le \underbrace{\mathbb{E} \Big[ \textstyle\sum_t \min_p \max_\nu \mathrm{AIR}^{\Phi,D}_\eta \Big]}_{\le\, T \cdot \text{dig-dec}} + \underbrace{\mathbb{E}[\mathrm{Est}] / \eta}_{\text{Estimation Error}}.$$
The first part is controlled by the Dig-DEC complexity (decision difficulty), and the second part $\mathrm{Est}$ is controlled by `PosteriorUpdate` (estimation difficulty). The paper follows two main lines: **replacing the first part with Dig-DEC (removing optimism)** and **obtaining smaller $\mathrm{Est}$ via better estimators**.

### Key Designs

**1. Dig-DEC: Replacing Optimistic Value with "Dual Information Gain"**

To address the bottleneck where optimistic DEC must rely on optimism and cannot handle adversarial settings, ours instantiates a specific $D$ within the general divergence framework:
$$D_\pi(\nu\|\rho) = \mathbb{E}_{M\sim\nu} \mathbb{E}_{o\sim M(\cdot|\pi)} \Big[\mathrm{KL} \big(\nu_\phi(\cdot|\pi,o), \rho\big) + \mathbb{E}_{\phi\sim\rho} \big[\bar D^\pi(\phi\|M)\big] \Big].$$
The resulting first-part complexity is **Dig-DEC (dual information gain DEC)**:
$$\text{dig-dec}^{\Phi,D}_\eta = \max_{\rho} \min_{p} \max_{\nu} \mathbb{E}_{\pi\sim p} \mathbb{E}_{(M,\pi^\star)\sim\nu} \Big[V_M(\pi^\star) - V_M(\pi) - \tfrac{1}{\eta} \mathbb{E}_o[\mathrm{KL}(\nu_\phi(\cdot|\pi,o), \rho)] - \tfrac{1}{\eta} \mathbb{E}_{\phi\sim\rho}[\bar D^\pi(\phi\|M)] \Big].$$
The "dual" in the name refers to the **two information gain measures** in the objective (KL term + $\bar D$ term). Crucially, the KL term is decomposed: $\mathrm{KL}(\nu_\phi, \rho) + \mathbb{E}_o [\mathrm{KL}(\nu_\phi(\cdot|\pi,o), \nu_\phi)]$. The first part is a **regularizer** preventing the marginal of $\nu$ from deviating too far from $\rho$—this item **replaces the optimistic value $V_\phi(\pi_\phi)$**, thereby removing optimism. The second part is the **information gain term**, capturing elements missed by optimistic DEC: since common $\bar D$ (like bilinear divergence or squared Bellman error) are **mean-based** and ignore distributional differences, this KL gain captures differences at the distribution level, allowing Dig-DEC to be **strictly better** than optimistic DEC (see the toy example in Key Findings).

**2. Unified Analysis via Bregman Divergence: Handling Arbitrary Divergence $D$**

Previous analyses [XZ23, LWZ25] fixed the posterior update to $\rho_{t+1}(\phi) = \nu_t(\phi|\pi_t, o_t)$, relying on a "constructive minimax theorem" **only applicable to strictly convex divergences (like KL)**. Ours uses an **argument based on first-order optimality and Bregman divergence**: utilizing the fact that $\nu_t$ is the best response to $p_t$, we obtain
$$\mathbb{E}_{\pi\sim p_t} \Big[V_{M_t}(\pi_{\phi^\star}) - V_{M_t}(\pi) - \tfrac{1}{\eta} D_\pi(\delta_{M_t,\pi_{\phi^\star}}\|\rho_t) \Big] \le \max_\nu \mathrm{AIR}^{\Phi,D}_\eta(p_t, \nu; \rho_t) - \tfrac{1}{\eta} \mathbb{E}_{\pi\sim p_t} \big[\mathrm{Breg}_{D_\pi(\cdot\|\rho_t)}(\delta_{M_t,\pi_{\phi^\star}}, \nu_t)\big],$$
where $\mathrm{Breg}_F(x,y) = F(x) - F(y) - \langle \nabla F(y), x-y \rangle \ge 0$. This formulation **naturally connects with standard mirror descent analysis**, allowing the framework to handle general (non-KL) divergences and making the algorithm more flexible. For instance, when reproducing the full-information mixed MDP results of [LWZ25], ours can select $D$ such that $\mathrm{Est}$ does not even involve $\log|\Phi|$, whereas the original paper required a more complex two-layer algorithm.

**3. Two Estimation Errors $\bar D$ and Superior `PosteriorUpdate`: Minimizing $\mathrm{Est}$**

The rate of the second part $\mathrm{Est}$ depends on the choice of $\bar D$. Ours provides two sets corresponding to two problem types, both improving upon the estimators in [FGQ+23]:

- **Average Estimation Error $\bar D_{av}$** (Standard realizability, $\bar D$ instantiated as average Bellman error): Aiming to approximate $\sum_h (\mathbb{E}_{\pi_k, M^\star}[\ell_h(\phi; o_h)])^2$. [FGQ+23] used a **biased** estimator $(\frac{1}{\tau}\sum_i \ell_h)^2$. Ours **splits the $\tau$ samples** of an epoch into two halves to construct an **unbiased** estimator:
$$L_k(\phi) = \sum_h \Big(\tfrac{2}{\tau} \textstyle\sum_{i=1}^{\tau/2} \ell_h(\phi; o_h^i) \Big) \Big(\tfrac{2}{\tau} \sum_{i=\tau/2+1}^{\tau} \ell_h(\phi; o_h^i) \Big),$$
where the independence between the two halves ensures the expectation of the product is the squared truth without bias, leading to sharper concentration and reducing $\mathrm{Est}$ from $\sqrt T$ to $\mathbb{E}[\mathrm{Est}] \lesssim N\log|\Phi|\, T^{1/3}$ (Theorem 7).

- **Squared Estimation Error $\bar D_{sq}$** (Requires **Bellman completeness**, $\bar D$ instantiated as squared Bellman error): **No batching is required** here. Using a **two-time-scale, two-layer learning structure with a biased loss at the top layer** (refining [FGQ+23, AZ22]), $\mathrm{Est}$ is reduced to a **constant level** $\mathbb{E}[\mathrm{Est}] \lesssim \log^2|\Phi|$ (Theorem 11), with simpler analysis.

**4. Non-inferiority and Strict Improvement: The Theoretical Position of Dig-DEC**

To demonstrate that removing optimism is always beneficial or neutral, ours provides two comparisons. First, for any non-negative $\bar D$, $\text{dig-dec}^{\Phi,D}_\eta \le \mathrm{dec}^\Phi_\eta$ (the model-free DEC of [LWZ25]). More importantly, **Theorem 13** shows that in the stochastic case, $\text{dig-dec}^{\Phi,D}_\eta \le \text{o-dec}^{\Phi,D}_\eta + \eta$. Since the DEC is usually of order $(\eta d)^\alpha$ with $\alpha \le 1$, the $+\eta$ term does not change the rate. Second, **Theorem 14** provides a **3-arm bandit** counterexample: for any $T \ge 1, \eta \le 1$, the algorithm of [FGQ+23] yields $\max_a \mathbb{E}[\mathrm{Reg}(a)] \ge \Omega(\sqrt T)$, while ours is $\le 1$. This proves that the **improvement from the KL info gain term capturing distributional differences can be arbitrarily large**.

## Key Experimental Results

As this is a purely theoretical paper, the "key data" refers to the **regret bounds** in various settings. Core improvements (compared with [FGQ+23] under the same settings):

### Improvements in Online Function Estimation Rates

| Estimation Type | Sub-case | [FGQ+23] | Ours | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Avg Estimation Error | on-policy | $T^{3/4}$ | $T^{2/3}$ | Unbiased split-sample estimator |
| Avg Estimation Error | off-policy | $T^{5/6}$ | $T^{7/9}$ | Same as above |
| Squared Error (Bellman complete) | — | $T^{2/3}$ | $\sqrt T$ | Two-time-scale, constant $\mathrm{Est}$ |

The $\sqrt T$ result under Bellman complete MDPs marks the **first time DEC-based methods match the performance of optimism-based methods** (e.g., [JLM21, XFB+23]).

### Regret Bounds for Specific MDP Classes (Partial)

| Environment | Setting | $\bar D$ | Regret $\mathbb{E}[\mathrm{Reg}]$ |
| :--- | :--- | :--- | :--- |
| Stochastic / bilinear on-policy | Bellman complete | $D_{sq}$ | $H\sqrt{dT\log|\Phi|}$ |
| Stochastic / coverable | Bellman complete | $D_{sq}$ | $H\sqrt{dT\log|\Phi|}$ |
| Stochastic / bilinear on-policy | — | $D_{av}$ | $H\sqrt{d\log|\Phi|}\,T^{2/3}$ |
| Mixed (adversarial rewards) / bilinear on-policy | — | $D_{av}$ | $d(H^5\log|\Phi|)^{1/4}T^{5/6}$ |
| Mixed / bilinear on-policy | Bellman complete | $D_{sq}$ | $d(H^5\log^2|\Phi|)^{1/4}T^{3/4}$ |
| Mixed / coverable | Bellman complete | $D_{sq}$ | $d(H^5\log^2|\Phi|)^{1/4}T^{3/4}$ |

### Key Findings
- **This provides the first sub-linear regret for model-free bandit learning in mixed MDPs (stochastic transitions, adversarial rewards, bandit feedback, linear rewards)**, resolving the open problem in [LWZ25]. Moving away from optimism—avoiding explicit reward estimators—is the key to handling adversarial rewards.
- All regret bounds **depend only on the value function class $\log|\Phi|$ and not the model class $|\mathcal{M}|$**, fulfilling the "model-free" promise.
- The framework **generalizes and simplifies the AIR analysis** of [XZ23, LWZ25]. A single Bregman/mirror descent argument covers stochastic, mixed, and fully adversarial settings while supporting non-KL divergences.

## Highlights & Insights
- **"Replacing optimism with two information gain terms" is the core insight**: Decomposing the ad-hoc optimistic value $V_\phi(\pi_\phi)$ into "KL regularization (replacing optimism) + KL information gain (capturing distribution differences)" is conceptually clean and leads to strict improvements.
- **Removing optimism unlocks adversarial settings**: A theoretical simplification (discarding optimism) happens to bypass the engineering deadlock where reward estimators cannot be constructed under adversarial rewards. This serves as a beautiful example of theoretical simplification expanding applicability.
- **Unbiased split-sample estimator**: Splitting samples within an epoch and multiplying them to get an unbiased estimate of a squared expectation is a simple yet effective trick transferable to other "squared expectation estimation" scenarios.
- **Transferability of the unified framework**: Replacing the "constructive minimax theorem" with Bregman/mirror descent logic frees DEC analysis from the restriction of "strictly convex divergences," paving the way for more flexible algorithms.

## Limitations & Future Work
- **Assumptions 3 / 4 are major constraints**: Mixed settings only cover **linear rewards with known features**, where the partition $\Phi$ ensures a unique mapping from reward to value. For **low-rank MDPs with unknown reward features**, this partitioning would cause $\log|\Phi|$ to grow polynomially with the number of possible features, whereas [LMWZ24] only requires logarithmic growth.
- **Unmatched rates remain**: Under average estimation error, off-policy only reaches $T^{7/9}$, and mixed cases mostly stop at $T^{3/4} \sim T^{5/6}$, remaining distant from $\sqrt T$. Whether these can be tightened under weaker assumptions is unknown.
- **Purely theoretical without numerical verification**: All conclusions are regret bounds; no experimental evidence is provided for constants or practical performance.
- **Computational complexity is not discussed**: Solving a saddle point over $\Delta(\Pi) \times \Delta(\Psi)$ and performing posterior updates per round may pose scaling challenges.

## Related Work & Insights
- **vs Optimistic DEC / Optimistic E2D [FGQ+23]**: They use optimism to drive exploration and only handle stochastic environments. Ours replaces optimism with dual information gain, yields complexity $\le$ theirs (Theorem 13), and handles adversarial rewards.
- **vs [LWZ25] (AIR / DEC for Mixed MDPs)**: They first tackled mixed MDPs using DEC but were restricted to model-based algorithms or full-information reward feedback. Ours fills the "model-free + bandit feedback" gap and simplifies their AIR framework.
- **vs Model-based DEC / E2D [FKQR21, FGH23]**: Their bounds include $\log|\mathcal{M}|$ (model estimation cost). Ours takes a model-free approach, with bounds containing only $\log|\Phi|$.
- **vs Adversarial DEC [FRSS22, XZ23]**: In fully adversarial settings, decision complexity often explodes, becoming nearly vacuous in MDPs (becoming $\log|\Pi|$). The mixed setting is a more controllable middle ground where ours establishes model-free bandit results.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Dig-DEC replaces optimism with dual information gain and resolves the long-standing model-free bandit problem for mixed MDPs; both the concept and results are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ A theoretical paper providing regret bounds across stochastic/mixed settings with strict improvements and counterexamples. No numerical experiments, though some rates are not yet fully tightened.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from framework to complexity to estimation to applications. Definitions and bounds are clear, though notation is dense.
- Value: ⭐⭐⭐⭐⭐ Unifies and advances DEC theory. The insight of removing optimism and the unbiased estimator have independent transferable value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dimension-Free Decision Calibration for Nonlinear Loss Functions](dimension-free_decision_calibration_for_nonlinear_loss_functions.md)
- [\[ICLR 2026\] Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms](decision-theoretic_approaches_for_improved_learning-augmented_algorithms.md)
- [\[ICLR 2026\] Improved High-Dimensional Estimation with Langevin Dynamics and Stochastic Weight Averaging](improved_high-dimensional_estimation_with_langevin_dynamics_and_stochastic_weigh.md)
- [\[ICLR 2026\] Near-Optimal Sample Complexity Bounds for Constrained Average-Reward MDPs](near-optimal_sample_complexity_bounds_for_constrained_average-reward_mdps.md)
- [\[ICLR 2026\] Online Decision-Focused Learning](online_decision-focused_learning.md)

</div>

<!-- RELATED:END -->
