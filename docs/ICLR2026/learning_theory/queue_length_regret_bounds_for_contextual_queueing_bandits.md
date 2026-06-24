---
title: >-
  [Paper Note] Queue Length Regret Bounds for Contextual Queueing Bandits
description: >-
  [ICLR 2026][Learning Theory][Contextual Queueing Bandits] This paper proposes the "Contextual Queueing Bandits" framework—scheduling tasks with heterogeneous contextual features while learning unknown service rates online. By employing a "policy-switching queue + coupling" argument, it decomposes queue length regret and proves that the algorithm CQB-ε achieves a decaying regret of $\tilde{O}(T^{-1/4})$ under stochastic contexts, while CQB-Opt achieves polylogarithmic regret o…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Online Learning"
  - "Queueing Bandits"
  - "Contextual Queueing Bandits"
  - "Queue Length Regret"
  - "Policy-Switching Queues"
  - "Coupling Argument"
  - "Logistic Bandits"
date: 2026-05-08
content_hash: fd586342ce11b513
---

# Queue Length Regret Bounds for Contextual Queueing Bandits

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bBPOcj5dOg](https://openreview.net/forum?id=bBPOcj5dOg)  
**Code**: TBD  
**Area**: Learning Theory / Online Learning / Queueing Bandits  
**Keywords**: Contextual Queueing Bandits, Queue Length Regret, Policy-Switching Queues, Coupling Argument, Logistic Bandits

## TL;DR
This paper proposes the "Contextual Queueing Bandits" framework—scheduling tasks with heterogeneous contextual features while learning unknown service rates online. By employing a "policy-switching queue + coupling" argument, it decomposes queue length regret and proves that the algorithm CQB-ε achieves a decaying regret of $\tilde{O}(T^{-1/4})$ under stochastic contexts, while CQB-Opt achieves polylogarithmic regret of $O(\log^2 T)$ under adversarial contexts.

## Background & Motivation

**Background**: Service platforms such as cloud scheduling, recommendation systems, ride-hailing, call centers, and LLM inference are essentially queueing systems: tasks wait in queues, and a scheduler assigns tasks to servers. When service rates are unknown, "learning while scheduling" becomes a core problem. The mainstream approach models this as queueing bandits and evaluates policy performance using **queue length regret**—the difference in queue length between the current policy and the optimal policy.

**Limitations of Prior Work**: Existing work on queue length regret (e.g., Krishnasamy 2016) assumes all tasks are homogeneous and ignores individual task context features. In modern platforms, tasks are naturally heterogeneous (varying in size, user profiles, and device compatibility). Although Kim & Oh (2024) introduced contexts, they require a **fixed number of discrete queues** where all tasks in a queue share the same context, failing to support arbitrary heterogeneous contexts; moreover, they optimize the cumulative weights of MaxWeight rather than true queue length regret.

**Key Challenge**: Allowing each task to have different context features introduces a fundamental analytical obstacle: **queue state misalignment**. The current policy $\pi$ and the optimal policy $\pi^*$ may process tasks in different orders, leading to completely different "remaining task feature lists" in the queues at the same moment. Existing analyses rely on the premise that $\mathcal{X}_t^* = \mathcal{X}_t$ (consistent queue states); once features are heterogeneous, this premise collapses, making it impossible to compare regrets directly at each time step.

**Goal**: (1) Establish a queueing bandit framework that supports heterogeneous contexts; (2) Provide a queue length regret decomposition tool that remains analytical under queue state misalignment; (3) Design algorithms and provide provable decaying/polylogarithmic regret bounds for both stochastic and adversarial contexts.

**Core Idea**: Use a family of **policy-switching queues** (which follow the current policy until a specific round and then switch to the optimal policy) for a telescopic decomposition. Then, use **coupling** to align adjacent switching queues, transforming the overall regret into a sum of products: "instantaneous error of suboptimal choices in a single round × long-term impact of that divergence on the final queue difference," thereby bypassing queue state misalignment.

## Method

### Overall Architecture

Problem Setup: Single queue, $K$ servers, discrete time $t\in[T]$. In each round, the agent observes the queue state $X_t\subset\mathbb{R}^d$ (set of feature vectors for remaining tasks), selects a task $x_t\in X_t$, and matches it with a server $k_t$. Service completion is determined by a logistic model: the departure rate is $\mu(x_t^\top\theta^*_{k_t})$, where $\mu(z)=(1+e^{-z})^{-1}$ and $\theta^*_k$ are unknown parameters for server $k$. The queue evolves according to:

$$X_{t+1}=X_t\setminus\{x_t:D(t)=1\}\cup\{x(t):A(t)=1\},\quad Q(t+1)=[Q(t)+A(t)-D(t)]^+$$

($A(t)$ is the arrival with mean $\lambda$; $D(t)$ is the departure). The goal is to control the **queue length regret** $R_T=\mathbb{E}[Q(T)-Q^*(T)]$, where $\pi^*$ is the optimal policy that knows all $\theta^*_k$ and selects the task-server pair with the maximum departure rate each round.

The "Method" consists of two components: a **universal analytical framework** (policy-switching queues + coupling + regret decomposition) that simplifies queue length regret into a manageable sum of products, and **two specific algorithms** (CQB-ε for stochastic contexts, CQB-Opt for adversarial contexts) that achieve decaying and polylogarithmic bounds within this framework. The analytical framework is the core innovation for solving state misalignment.

### Key Designs

**1. Policy-Switching Queues + Coupling Argument: Aligning Misaligned Queues for Decomposition**

This is the core solution to queue state misalignment. Directly comparing $Q(t)$ and $Q^*(t)$ is unfeasible due to different processing orders. The authors define the **policy-switching queue** $Q(t_1,t_2)$ as follows: follow the current policy from $t=1$ to $t_1$, then switch to the optimal policy from $t_1+1$ to $t_2$. At the boundaries, it reduces to $Q(t_2-1, t_2) = Q(t_2)$ and $Q(0, t_2) = Q^*(t_2)$. Thus, the queue length regret is split into a telescopic sum:

$$R_T=\mathbb{E}[Q(T)-Q^*(T)]=\sum_{t=1}^{T-1}\mathbb{E}[Q(t,T)-Q(t-1,T)],$$

where each term is the difference between two queues whose switching times differ by one round. To compare these adjacent queues, the authors construct a **coupling process**: for arrivals and departures in each round, shared uniform random numbers $U_{i,1},U_{i,2}\sim\text{Unif}(0,1)$ are drawn, ensuring coupled queues $Q^+,Q^-$ share the same arrival and departure threshold decisions. Coupling preserves marginal distributions, so $R_T=\sum_{t=1}^{T-1}\mathbb{E}[Q^+(t)-Q^-(t)]$. Since the coupled queues follow identical policies before round $t-1$, their states are consistent at time $t$—the misalignment is "aligned." Defining $\psi(t,T):=Q^+(t)-Q^-(t)$, it can be proved that $\psi(t,T)\in\{-1,0,1\}$ (Lemma 4.1), and its expectation is decomposed into the product of two factors (Lemma 4.2):

$$\mathbb{E}[\psi(t,T)]\le \underbrace{\sqrt{\mathbb{E}\big[(\mu((x^*_t)^\top\theta^*_{k^*_t})-\mu(x_t^\top\theta^*_{k_t}))^2\big]}}_{=:m_t}\cdot\underbrace{\sqrt{\mathbb{E}[\tilde\psi(t,T)]}}_{=:\delta_t}.$$

$m_t$ is the **instantaneous error** caused by selecting a suboptimal task-server pair, and $\delta_t$ is the **long-term impact** of this divergence on the terminal queue difference. Thus, $R_T\le\sum_t m_t\delta_t$. This decomposition is independently valuable as it decouples "short-term mismatch" from "long-term ripple effects."

**2. CQB-ε Two-Stage Algorithm: Extracting $\tilde{O}(T^{-1/4})$ Decay Bound via Monotonicity + Chebyshev Sum Inequality**

Simply having $R_T\le\sum m_t\delta_t$ is insufficient: using Cauchy-Schwarz and the elliptic potential lemma directly yields $\tilde{O}(\sqrt T)$, which is typical for contextual bandits but **non-decaying**. The key observation is that the two factors have opposite monotonic trends: as estimation improves, $m_t$ (instantaneous error) should decrease with $t$, while $\delta_t$ (long-term impact) should increase with $t$ as early divergences are "smoothed out" by the same optimal policy over the remaining $T-t-1$ rounds. The authors design CQB-ε: a **pure exploration phase** for $\tau=O(d\log T/(\sigma_0^4\epsilon^2))$ rounds where new tasks are assigned to servers in a round-robin fashion to bound the uncertainty $\|x\|_{V^{-1}_{t-1,k}}$ (Lemma 5.2); and an **ε-exploration phase** that performs random exploration with probability $\varepsilon=T^{-1/2}$, otherwise selecting tasks optimistically using the UCB rule $(x_t,k_t)=\arg\max\,\mu(x^\top\hat\theta_{t-1,k})+\beta_{t-1,k}\|x\|_{V^{-1}_{t-1,k}}$.

With this algorithm, the authors prove $m_t\le M_t$ (where $\{M_t\}$ is non-increasing) and $\delta_t\le\Delta_t$ (where $\{\Delta_t\}$ is non-decreasing). Summing the product of a non-increasing and a non-decreasing sequence allows the use of the **Chebyshev Sum Inequality**:

$$R_T\le\sum_{t=1}^{T-1}M_t\Delta_t\le\frac{1}{T-1}\Big(\sum_{t=1}^{T-1}M_t\Big)\Big(\sum_{t=1}^{T-1}\Delta_t\Big).$$

By proving $\sum M_t=\tilde O(T^{3/4})$ and $\sum\Delta_t=O(\log T)$, their product divided by $T-1$ yields the decay bound in Theorem 5.5, with the dominant term being $\tilde O(T^{-1/4})$. This shows that the queue length difference between the current and optimal policies vanishes as $T$ increases.

**3. CQB-Opt for Adversarial Contexts: Counting Bad Rounds + Weighting Processes for $O(\log^2 T)$**

Under the adversarial setting, contexts are chosen by an adversary, removing the i.i.d. assumption. The core premise of Design 2—that uncertainty is uniformly bounded after $\tau$ rounds—fails: **bad rounds** (where $\|x_t\|_{V^{-1}_{t-1,k_t}}>\epsilon/(4\beta_{t-1,k_t})$) might persist until the end, making $m_t$ non-monotonic. The authors switch tactics: first, **counting the total number of bad rounds** via a counting version of the elliptic potential lemma ($|B'|\le O(d^2\log^2 T/\epsilon^2)$); second, handling the randomness of bad rounds by defining a $\mathcal{G}_t$-measurable **weighting process** $V(t)=\alpha^{-B'(t-1)}e^{\eta Q(t)}$ ($\alpha>1,\eta>0$), which provides tail bounds for $Q(t)$. Returning to the decomposition framework, the adversarial setting uses Cauchy-Schwarz: $R_T\le\sqrt{\sum m_t^2}\sqrt{\sum\delta_t^2}$, where $\sum m_t^2=O(d\log T)$ and $\sum\delta_t^2$ is bounded using a threshold round $\omega'=O(d^2\log^2 T/\epsilon^3)$. The final Theorem 6.3 gives $R_T=O\!\big(d^2\log^2 T/\epsilon^{1.5}+d\log T/\epsilon^2\big)$, a polylogarithmic regret at the cost of logarithmic factors compared to the stochastic setting.

## Key Experimental Results

### Main Results

Stochastic instance parameters: $\lambda=0.7, \epsilon=0.1, K=5, d=5, \kappa=10$; features $x$ and parameters $\theta^*_k$ sampled from $\text{Unif}(-1,1)$. Each algorithm ran on $N=10$ instances for $T=5000$ rounds.

| Policy | Queue Length Behavior | Description |
|------|--------------|------|
| Random Policy | Linear Growth | No learning, queue diverges |
| Optimal Policy $\pi^*$ | Lowest Baseline | Known $\theta^*_k$ |
| CQB-ε (Ours) | Plateau then Drop, Approaches Opt | Switches to ε-greedy after pure exploration |
| CQB-Opt (Ours) | Decays toward Optimal | Adapted for adversarial contexts |
| 4 Extra Baselines | See Appendix A | Comparison targets |

Both proposed algorithms converge toward the optimal queue length after a certain number of rounds, while the random policy continues to grow linearly, verifying the theoretical decay.

### Ablation Study

| Configuration | Key Observation | Description |
|------|----------|------|
| CQB-ε, $\epsilon\in\{0.05,0.1,0.15\}$ | Smaller $\epsilon$ leads to longer exploration and later drop | $\tau$ is inversely proportional to $\epsilon$ |
| CQB-Opt, $\epsilon\in\{0.05,0.1,0.15\}$ | Larger $\epsilon$ leads to faster convergence | Lower load enables faster optimization |

(Note: $\epsilon$ refers to the traffic slack, not the ε-greedy exploration rate $\varepsilon=T^{-1/2}$.)

### Key Findings
- Larger $\epsilon$ (traffic slack) leads to faster convergence for both algorithms, consistent with the theoretical dependence of $\tau$ and $\omega'$ on $\epsilon$—lighter loads are easier to stabilize.
- The CQB-ε curve shows a "plateau (queue accumulation during exploration) then sharp drop" pattern, confirming the two-stage structure.
- The linear divergence of the random policy highlights the necessity of "learning while scheduling"; without learning, the queue cannot be stabilized.

## Highlights & Insights
- **Policy-switching queues transform "incomparable" to "comparable"**: The telescopic decomposition compares only adjacent queues differing by one round in switching time. Coupled with alignment via coupling, this is key to bypassing queue state misalignment, offering a lesson for any online decision problem where processing order affects state.
- **Decoupled product decomposition**: $R_T\le\sum m_t\delta_t$ separates the instantaneous cost of a wrong choice from its long-term ripple effects, a decomposition the authors claim has independent value.
- **Using monotonicity + Chebyshev Sum Inequality for decay bounds**: Identifying the opposite trends of $m_t$ (non-increasing) and $\delta_t$ (non-decreasing) allows the use of the Chebyshev inequality to compress the naive $\tilde O(\sqrt T)$ into a decaying $\tilde O(T^{-1/4})$.
- **Dual techniques for stochastic vs. adversarial**: The stochastic setting relies on "uncertainty smoothing → two-stage + monotonicity," while the adversarial setting relies on "bad round counting + weighting processes → Cauchy-Schwarz," demonstrating how to switch proof tools within the same framework.

## Limitations & Future Work
- **Single Queue Assumption**: Currently only handles a single queue; multi-queue extensions, where state misalignment is more complex, are left for future work.
- **Lack of Lower Bounds**: Only upper bounds are provided; it is unclear if $\tilde O(T^{-1/4})$ or $O(\log^2 T)$ is optimal.
- **Strong Structural Assumptions**: Relies on a logistic model, traffic slack $\epsilon>0$, and stochastic covariance lower bounds. Dependencies on $1/(\sigma_0^8\epsilon^5)$ can be poor under heavy loads or ill-conditioned covariance.
- **Operational Constraints**: Lacks practical constraints like maximum waiting time.
- **Scale of Experiments**: $d=K=5$, $T=5000$, and synthetic data suggest a gap from real-world large-scale platforms.

## Related Work & Insights
- **vs. Classical Queueing Bandits (Krishnasamy 2016)**: They pioneered queue length regret and decay bounds but for homogeneous tasks without context, where $\mathcal{X}_t^*=\mathcal{X}_t$ always holds. This work is the first to provide decay rates for queue length regret with heterogeneous contexts.
- **vs. Kim & Oh (2024)**: They also perform context-aware scheduling but restrict discrete contexts and optimize a proxy MaxWeight objective rather than performing a direct queue length regret analysis.
- **vs. Logistic Bandits (Faury 2020)**: This work borrows logistic parameter estimation scripts, but differs fundamentally as its action set is **endogenous and policy-dependent** (past actions affect current state), requiring new techniques to handle misalignment and optimize queue length rather than cumulative reward.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First contextual queueing bandit framework + novel policy-switching/coupling technique.
- Experimental Thoroughness: ⭐⭐⭐ Synthetic data, small scale, mainly for theoretical verification.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from challenges to technical theorems.
- Value: ⭐⭐⭐⭐ Introduction of heterogeneous contexts into queue length regret analysis; tools are transferable to other online decision/queueing learning problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)
- [\[ICLR 2026\] Quantitative Bounds for Length Generalization in Transformers](quantitative_bounds_for_length_generalization_in_transformers.md)
- [\[ICLR 2026\] Diversified Multinomial Logit Contextual Bandits](diversified_multinomial_logit_contextual_bandits.md)
- [\[ICLR 2026\] Efficient Best-of-Both-Worlds Algorithms for Contextual Combinatorial Semi-Bandits](efficient_best-of-both-worlds_algorithms_for_contextual_combinatorial_semi-bandi.md)
- [\[ICLR 2026\] Contextual Multi-Armed Bandits with Minimum Aggregated Revenue Constraints](contextual_multi-armed_bandits_with_minimum_aggregated_revenue_constraints.md)

</div>

<!-- RELATED:END -->
