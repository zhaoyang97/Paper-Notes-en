---
title: >-
  [Paper Note] FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs
description: >-
  [ICML 2026][Reinforcement Learning][Push-Pull/AB] This paper proposes FAB—the first purely first-order algorithm for distributed bilevel optimization over time-varying directed graphs—combining AB/Push-Pull communication…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Push-Pull/AB"
  - "Time-Varying Directed Graphs"
  - "Bilevel Optimization"
  - "First-Order Algorithm"
  - "Consensus Error"
date: 2026-05-08
content_hash: 8d417307e4052a5e
---

# FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs

**Conference**: ICML 2026  
**arXiv**: [2605.06328](https://arxiv.org/abs/2605.06328)  
**Code**: https://anonymous.4open.science/r/FAB-HE66 (anonymous)  
**Area**: Distributed Optimization / Bilevel Optimization / Reinforcement Learning  
**Keywords**: Push-Pull/AB, Time-Varying Directed Graphs, Bilevel Optimization, First-Order Algorithm, Consensus Error

## TL;DR
This paper proposes FAB—the first purely first-order algorithm for distributed bilevel optimization over time-varying directed graphs—combining AB/Push-Pull communication with value function penalization, providing a non-asymptotic $\mathcal{O}(K^{-2/3})$ convergence rate, and incidentally resolving the long-standing open problem of AB/Push-Pull convergence rates in nonconvex, time-varying directed graph settings.

## Background & Motivation
**Background**: Decentralized optimization has become a mainstream paradigm in distributed machine learning, evolving from early undirected graph algorithms (EXTRA, Exact-Diffusion, gradient tracking), to directed graph methods like Push-Sum, Push-DIGing, and further to AB/Push-Pull protocols that combine row-stochastic matrix $A$ and column-stochastic matrix $B$. To address real-world constraints such as communication delays, stragglers, and satellite networks, recent work has extended these methods to **time-varying directed graphs**.

**Limitations of Prior Work**: When applying these decentralized algorithms to machine learning tasks, hyperparameter tuning (e.g., regularization coefficient $\lambda$) is notoriously difficult. The authors observe that, rather than brute-force grid search, treating hyperparameters as upper-level variables in a bilevel optimization framework is more effective. However, distributed bilevel optimization has only been studied on static networks; there is no provable method for time-varying directed graphs. Even in single-level settings, the convergence rate of AB/Push-Pull under time-varying directed graphs and nonconvex objectives remains uncharacterized.

**Key Challenge**: (1) Dynamic, unbalanced communication introduces persistent consensus bias; in the time-varying setting, the fundamental eigenvector $\pi$ is no longer fixed, rendering the key tool of time-invariant $\pi$ from static analysis ineffective. (2) The penalty parameter $\lambda$ in bilevel optimization must be large to approximate the original problem, but increasing $\lambda$ amplifies consensus error, potentially causing divergence—there is a hard trade-off between "approximation accuracy vs. consensus stability."

**Goal**: To design a purely first-order, provably convergent distributed bilevel algorithm that works under the most general setting of time-varying directed graphs, nonconvex upper-level, and strongly convex lower-level, while also filling the gap in AB/Push-Pull convergence rates for nonconvex, time-varying settings.

**Key Insight**: The value function penalization method $\min_{x,y} F(x,y)+\lambda(G(x,y)-\min_z G(x,z))$ reformulates the bilevel problem as a single-level one, avoiding second-order Hessian-vector products. Introducing an auxiliary variable $z$ to track $y^*(x)$ further decomposes the problem into a distributable min-max form, which naturally aligns with the push/pull dual-matrix communication of AB/Push-Pull.

**Core Idea**: Use a row-stochastic matrix $A^k$ to pull decision variables $(x,y,z)$, and a column-stochastic matrix $B^k$ to push gradient tracking variables $(t_x,t_y,t_z)$. Embed the gradient descent-ascent updates of the value function penalization objective directly into the AB/Push-Pull framework, simultaneously addressing bilevel structure, time-varying directed communication, and purely first-order gradients.

## Method

### Overall Architecture
FAB rewrites the distributed bilevel problem with nonconvex upper-level and $\mu$-strongly convex lower-level,
$\min_x \mathcal{F}^*(x)=F(x,y^*(x))$, $y^*(x)=\arg\min_y G(x,y)$,
as an equivalent min-max form with auxiliary variables: $\min_{x,y}\max_z \frac{1}{n}\sum_i \mathcal{L}_i(x,y,z)$, where $\mathcal{L}_i = f_i(x,y)+\lambda(g_i(x,y)-g_i(x,z))$. Each agent $i$ maintains two sets of variables per iteration: decision variables $(x_i^k,y_i^k,z_i^k)$ and gradient tracking variables $(t_{x,i}^k,t_{y,i}^k,t_{z,i}^k)$. Each round involves pulling (row-stochastic matrix $A^k$) to aggregate decisions from in-neighbors, local evaluation of $\nabla \mathcal{L}_i$, and pushing (column-stochastic matrix $B^k$) to transmit gradient tracking quantities, in a three-step loop.

### Key Designs

1. **Coupling AB/Push-Pull with Value Function Penalization**:

    - **Function**: Seamlessly extends single-level AB/Push-Pull communication primitives to the bilevel setting, avoiding Hessian computation or inversion.
    - **Mechanism**: After reformulating $\min_{x,y}\max_z F+\lambda(G(x,y)-G(x,z))$, $y$ is updated via gradient descent to approach the upper-level optimum, $z$ via gradient ascent to track $y^*(x)$. All three are handled by the AB/Push-Pull pull-evaluate-push steps, sharing the same row/column stochastic matrices $A^k,B^k$, with no second-order information required.
    - **Design Motivation**: Previous distributed bilevel methods either relied on second-order derivatives (Hessian-vector products are hard to compute in decentralized settings) or only worked on static undirected graphs. This design transforms all exchanged signals into gradient quantities and reuses the dual-matrix push-pull structure, perfectly matching time-varying directed communication.

2. **Gradient Tracking Triplet**:

    - **Function**: Recovers unbiased estimates of the global average gradient on dynamic, unbalanced communication graphs, suppressing consensus drift due to agent heterogeneity.
    - **Mechanism**: For each of $(x,y,z)$, maintain a tracking variable $t$ with update rule $t^{k+1}_i = \sum_j b_{ij}^k t_j^k + d_i^{k+1} - d_i^k$, where $d_i^k = \nabla_{\cdot} \mathcal{L}_i$. Column-stochasticity ensures $\sum_i t_{\cdot,i}^k = \sum_i d_{\cdot,i}^k$, aligning each agent's update direction with the global average gradient.
    - **Design Motivation**: Time-varying directed graphs lack a time-invariant root eigenvector $\pi$, so naive consensus averaging leaves persistent bias. Gradient tracking, via the difference term $d^{k+1}-d^k$, "self-corrects" this bias and is key to extending analysis from strong convexity to nonconvexity.

3. **Fine-Tuned Trade-off between Penalty $\lambda$ and Step Size $\eta$**:

    - **Function**: Identifies the provably optimal trade-off between "large $\lambda$ → accurate bilevel approximation" and "large $\lambda$ → amplified consensus error → smaller step size required."
    - **Mechanism**: Theoretical Lyapunov analysis yields a descent inequality of the form $\|\nabla \mathcal{F}^*\|^2 + \frac{8\underline{c}n}{5a^n}\mathcal{C}_{b,3}\lambda \mathbf{V}_D^k \leq \frac{4\mathcal{C}_{gap}}{\lambda^2}+\dots$, where approximation error decays as $\lambda^{-2}$, but consensus error is linearly amplified by $\lambda$. Setting $\lambda = \mathcal{O}(K^{1/3})$, $\eta = \mathcal{O}(K^{-1/3})$ balances both at $K^{-2/3}$.
    - **Design Motivation**: This is the theoretical core of the paper—explicitly rebalancing the common "the larger $\lambda$ the better" intuition for bilevel penalization within the distributed consensus error framework, and providing actionable asymptotic tuning formulas.

### Loss & Training
Local penalty $\mathcal{L}_i(x,y,z)=f_i(x,y)+\lambda(g_i(x,y)-g_i(x,z))$, with $\lambda=\mathcal{O}(K^{1/3})$; step sizes $\eta_x^k,\eta_y^k,\eta_z^k=\mathcal{O}(K^{-1/3})$. The lower-level $g_i$ is assumed $\mu$-strongly convex, $L_{g,1}$-smooth, and Hessian Lipschitz; the upper-level $f_i$ only requires $L_{f,1}$-smoothness and a lower bound (can be nonconvex). The communication graph is strongly connected at each step (or $C$-connected), and $A^k,B^k$ have all nonzero entries lower bounded by $a,b>0$.

## Key Experimental Results

### Main Results

| Task | Network/Setting | Baselines | FAB Performance |
|------|----------------|-----------|-----------------|
| Distributed RL Policy Evaluation (Linear Bellman) | $\nu\in\{0.1,0.3,0.5\}$, noise $\omega\in\{1,2,3\}$ | SGP-DL / Push-SAGA-DL / Push-ASGD-DL / AB-DL | Lowest loss and fastest convergence across all connectivity and noise combinations |
| Fashion-MNIST Data Hyper-cleaning (MLP, 203k params) | $cr\in\{0.4,0.5,0.6\}$, $\rho\in\{0.1,0.5,1\}$ | Same as above | Test accuracy leads under all corruption/heterogeneity settings |
| IMDB Data Hyper-cleaning (BERT-110M finetuning) | Time-varying directed graph | Same as above | Maintains advantage even in large-model NLP settings |
| MNIST Hyperparameter Tuning (Validation Set Adversarial Corruption) | 100 agents, $cr=0.2,0.4$ | SGP+grid / Push-DIGing+grid / AB+grid | Test accuracy significantly higher than single-level baselines with fixed $\lambda=0.2$; $\lambda$ adapts dynamically during training |

### Ablation Study

| Dimension | Observation | Conclusion |
|-----------|-------------|------------|
| Increasing network size $n$ | Convergence degrades but not exponentially | Matches theoretical worst-case $(ab)^{-n}$, but actual decay is milder |
| Decreasing connectivity parameter $\nu$ (sparser) | Both FAB and baselines slow down, but FAB degrades less | Bilevel + gradient tracking is more robust to sparse topologies |
| Increasing noise $\omega$ | Single-level baselines oscillate significantly | Bilevel structure acts as built-in adaptive regularization, absorbing gradient noise |
| Peak memory | FAB is on par with single-level baselines | First-order design avoids memory blowup common in second-order methods |

### Key Findings
- **Theoretical open problem resolved**: Reducing FAB to the single-level case (no $\lambda$) yields AB/Push-Pull with $\mathcal{O}((ab)^{-n}K^{-1})$ convergence rate for time-varying directed graphs and nonconvex objectives, matching centralized gradient descent's $K^{-1}$ rate.
- **Bilevel vs grid search**: In motivating experiments, FAB significantly outperforms any single-level method with fixed $\lambda$ under strong label noise ($cr=0.4$), and the $\lambda$ trajectory clearly shows adaptive decay, demonstrating the practical value of bilevel hyperparameter tuning.
- **Network size factor $(ab)^{-n}$**: While theoretically exponential, experiments (Figure 6(a)) show only slow degradation, consistent with similar constants in worst-case analyses of subgradient-push and Push-Pull in the literature.

## Highlights & Insights
- Integrates "distributed optimization" + "bilevel optimization" + "time-varying directed graphs"—each a challenging direction—into a single framework with the first non-asymptotic convergence guarantee, demonstrating both substantial workload and novelty.
- Uses auxiliary variable $z$ to flatten the $\min\max\min$ structure into a single min-max, elegantly matching the AB/Push-Pull dual-matrix design—transferable to other nested problems (meta-learning, adversarial robustness, actor-critic).
- Analysis framework is modular: restricting to the "single-level + time-varying directed + nonconvex" subset immediately yields AB/Push-Pull convergence rates, a rare "prove more, get more" byproduct in algorithm papers.
- Engineering-wise, the 100% first-order + gradient tracking design can be directly embedded into PyTorch DDP/RPC, requiring no second-order derivatives, making it deployment-friendly.

## Limitations & Future Work
- The $(ab)^{-n}$ factor in the convergence constant, while worst-case, still suggests potential performance collapse for very large agent counts or extremely sparse communication; future work may require task-specific tighter analysis or communication compression.
- $\lambda=\mathcal{O}(K^{1/3})$ requires knowing $K$ in advance; the authors suggest adapting centralized increasing-$\lambda$ schemes (Kwon 2023), but no strict guarantee is given for the distributed case.
- The lower-level strong convexity assumption does not hold in many RL/meta-learning tasks (e.g., nonconvex value functions); relaxing to lower-level PL or nonconvexity is a natural direction.
- The largest model tested is BERT-base (110M); end-to-end feasibility for large models + cross-region time-varying communication (e.g., real satellite or federated edge networks) remains to be validated.

## Related Work & Insights
- **vs AB/Push-Pull series (Xin & Khan 2018; Pu 2021; Saadatniaki 2020; Nedić 2025)**: These works focus on single-level, mainly strongly convex settings; this paper addresses bilevel, nonconvex + time-varying. A simplified variant of the analysis also resolves their open problems.
- **vs Distributed Bilevel (Yang 2022, Zhu 2024, Chen 2025)**: Prior work mostly relies on second-order derivatives (Hessian-vector) and only works on static graphs; this is the first first-order method for both static and time-varying directed graphs.
- **vs Centralized Bilevel (Kwon 2023, Chen 2025a)**: Their value function penalization inspired this reformulation, but this paper additionally handles the mutual amplification of consensus error and $\lambda$, a distributed-specific challenge.
- **vs Push-DIGing / SGP**: Also targets time-varying directed graphs, but single-level; this paper shows that introducing bilevel structure allows the same communication primitives to enable adaptive hyperparameter tuning, far beyond traditional ERM.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First distributed bilevel algorithm for time-varying directed graphs + incidentally resolves AB/Push-Pull nonconvex time-varying open problem
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers hyperparameter tuning, data hyper-cleaning, RL policy evaluation across CV/NLP, but lacks ultra-large-scale federated experiments
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, challenge analysis and design motivation are well explained, Algorithm 1 is concise and readable
- Value: ⭐⭐⭐⭐ Directly applicable to decentralized ML systems, satellite networks, and federated hyperparameter automation scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Bilevel Optimization over Saddle Points of Zero-Sum Markov Games](bilevel_optimization_over_saddle_points_of_zero-sum_markov_games.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICML 2026\] Parameter-free Dynamic Regret: Time-varying Movement Costs, Delayed Feedback, and Memory](parameter-free_dynamic_regret_time-varying_movement_costs_delayed_feedback_and_m.md)
- [\[AAAI 2026\] Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](../../AAAI2026/reinforcement_learning/do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [\[ICML 2026\] Laplacian Representations for Decision-Time Planning](laplacian_representations_for_decision-time_planning.md)

</div>

<!-- RELATED:END -->
