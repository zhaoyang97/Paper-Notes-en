---
title: >-
  [Paper Note] FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs
description: >-
  [ICML 2026][Reinforcement Learning][Push-Pull/AB] This paper proposes FAB—the first purely first-order algorithm for distributed bilevel optimization over time-varying directed graphs. By combining AB/Push-Pull communica…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Push-Pull/AB"
  - "Time-varying directed graphs"
  - "Bilevel optimization"
  - "First-order algorithms"
  - "Consensus error"
date: 2026-05-08
content_hash: 52328dbc34a488ab
---

# FAB: A First-Order AB-based Gradient Algorithm for Distributed Bilevel Optimization over Time-Varying Directed Graphs

**Conference**: ICML 2026  
**arXiv**: [2605.06328](https://arxiv.org/abs/2605.06328)  
**Code**: https://anonymous.4open.science/r/FAB-HE66 (anonymous)  
**Area**: Distributed Optimization / Bilevel Optimization / Reinforcement Learning  
**Keywords**: Push-Pull/AB, Time-varying directed graphs, Bilevel optimization, First-order algorithms, Consensus error

## TL;DR
This paper proposes FAB—the first purely first-order algorithm for distributed bilevel optimization over time-varying directed graphs. By combining AB/Push-Pull communication with the value function penalty method, it provides a non-asymptotic $\mathcal{O}(K^{-2/3})$ convergence rate and simultaneously resolves the long-standing open problem regarding the convergence rate of AB/Push-Pull in non-convex scenarios over time-varying directed graphs.

## Background & Motivation
**Background**: Decentralized optimization has become a mainstream paradigm in distributed machine learning, expanding from early undirected graph algorithms (EXTRA, Exact-Diffusion, Gradient Tracking) to Push-Sum and Push-DIGing on directed graphs, and further to the AB/Push-Pull protocol integrating row-stochastic matrix $A$ and column-stochastic matrix $B$. Recently, this has been extended to **time-varying directed graphs** to address real-world constraints such as communication delays, stragglers, and satellite networks.

**Limitations of Prior Work**: Tuning hyperparameters (e.g., regularization coefficient $\lambda$) in distributed machine learning tasks is often arduous. The authors observe that instead of brute-force grid search, it is more effective to treat hyperparameters as upper-level variables in bilevel optimization. However, distributed bilevel optimization has only been studied on static networks, and no provable methods exist for the time-varying directed setting. Furthermore, even in single-level scenarios, the convergence rate of AB/Push-Pull under non-convex objectives on time-varying directed graphs has remained uncharacterized.

**Key Challenge**: (1) Dynamic unbalanced communication introduces continuously drifting consensus bias. In time-varying settings, the Perron eigenvector $\pi$ is no longer fixed, rendering the key weapon of $\pi$-invariance from traditional static analysis ineffective. (2) To approximate the original problem, the penalty parameter $\lambda$ must be large; however, larger $\lambda$ exacerbates consensus error, potentially leading to divergence—creating a hard trade-off between "approximation accuracy vs. consensus stability."

**Goal**: Design a purely first-order, provably convergent distributed bilevel algorithm capable of working in the general setting of time-varying directed graphs with non-convex upper-level and strongly convex lower-level objectives, while completing the convergence theory for AB/Push-Pull in non-convex time-varying settings.

**Key Insight**: The value function penalty method $\min_{x,y} F(x,y)+\lambda(G(x,y)-\min_z G(x,z))$ transforms the bilevel problem into a single-level one, avoiding second-order Hessian-vector products. By introducing an auxiliary variable $z$ to track $y^*(x)$, the problem is decomposed into a distributable min-max form, which naturally fits the push/pull dual-matrix communication of AB/Push-Pull.

**Core Idea**: Use a row-stochastic matrix $A^k$ to pull decision variables $(x,y,z)$ and a column-stochastic matrix $B^k$ to push gradient tracking variables $(t_x,t_y,t_z)$. The gradient descent-ascent updates of the value function penalty objective are directly embedded into the AB/Push-Pull framework, simultaneously addressing the bilevel structure, time-varying directed communication, and first-order gradients.

## Method

### Overall Architecture
FAB reformulates the distributed bilevel problem with a non-convex upper-level and $\mu$-strongly convex lower-level objective $\min_x \mathcal{F}^*(x)=F(x,y^*(x))$, where $y^*(x)=\arg\min_y G(x,y)$, into an equivalent min-max form with auxiliary variables: $\min_{x,y}\max_z \frac{1}{n}\sum_i \mathcal{L}_i(x,y,z)$, where $\mathcal{L}_i = f_i(x,y)+\lambda(g_i(x,y)-g_i(x,z))$. Each agent $i$ maintains two sets of variables per round: decision variables $(x_i^k,y_i^k,z_i^k)$ and gradient tracking variables $(t_{x,i}^k,t_{y,i}^k,t_{z,i}^k)$. Each round follows a cycle: pull (via row-stochastic matrix $A^k$) to aggregate decisions from incoming neighbors, local evaluation of $\nabla \mathcal{L}_i$, and push (via column-stochastic matrix $B^k$) to transmit gradient tracking quantities.

### Key Designs

1.  **Coupling of AB/Push-Pull + Value Function Penalty**:

    - **Function**: Seamlessly extends the single-level AB/Push-Pull communication primitives to bilevel scenarios, avoiding Hessian or inversion computations.
    - **Mechanism**: After rewriting as $\min_{x,y}\max_z F+\lambda(G(x,y)-G(x,z))$, $y$ uses gradient descent to approximate the upper-level optimum, while $z$ uses gradient ascent to track $y^*(x)$. All variables are updated using the pull-evaluate-push steps of AB/Push-Pull, sharing the same row/column stochastic matrices $A^k, B^k$ without requiring second-order information.
    - **Design Motivation**: Distributed bilevel methods previously relied on second-order derivatives (Hessian-vector products are difficult to compute accurately in decentralized settings) or were limited to static undirected graphs. This design converts all exchanged signals into gradient quantities and reuses the push-pull structure of dual stochastic matrices, which fits time-varying directed communication.

2.  **Gradient Tracking Triplet**:

    - **Function**: Recovers an unbiased estimate of the global average gradient on dynamic, unbalanced communication graphs, suppressing consensus drift caused by agent heterogeneity.
    - **Mechanism**: Maintains a tracking variable $t$ for each variable $(x,y,z)$, with the update rule $t^{k+1}_i = \sum_j b_{ij}^k t_j^k + d_i^{k+1} - d_i^k$, where $d_i^k = \nabla_{\cdot} \mathcal{L}_i$. Column stochasticity ensures $\sum_i t_{\cdot,i}^k = \sum_i d_{\cdot,i}^k$, aligning the direction of each agent with the global average gradient.
    - **Design Motivation**: Time-varying directed graphs lack a time-invariant Perron eigenvector $\pi$; simple consensus averaging leaves persistent bias. Gradient tracking "self-corrects" this bias via the difference term $d^{k+1}-d^k$, serving as a key tool for extending analysis from strongly convex to non-convex cases.

3.  **Exquisite Balance between Penalty $\lambda$ and Step Size $\eta$**:

    - **Function**: Identifies a provable optimal trade-off between "large $\lambda \to$ accurate bilevel approximation" and "large $\lambda \to$ amplified consensus error $\to$ requires smaller step size."
    - **Mechanism**: Theoretical Lyapunov analysis shows a descent inequality of the form $\|\nabla \mathcal{F}^*\|^2 + \frac{8\underline{c}n}{5a^n}\mathcal{C}_{b,3}\lambda \mathbf{V}_D^k \leq \frac{4\mathcal{C}_{gap}}{\lambda^2}+\dots$, where the approximation error decays at $\lambda^{-2}$ while the consensus error is linearly amplified by $\lambda$. Setting $\lambda = \mathcal{O}(K^{1/3})$ and $\eta = \mathcal{O}(K^{-1/3})$ enables convergence at $K^{-2/3}$.
    - **Design Motivation**: This is the theoretical core—re-evaluating the "the larger the $\lambda$ the better" heuristic for bilevel penalties within the framework of distributed consensus errors and providing an actionable asymptotic parameter formula.

### Loss & Training
Local penalty $\mathcal{L}_i(x,y,z)=f_i(x,y)+\lambda(g_i(x,y)-g_i(x,z))$, with $\lambda=\mathcal{O}(K^{1/3})$; step sizes $\eta_x^k,\eta_y^k,\eta_z^k=\mathcal{O}(K^{-1/3})$. The lower-level $g_i$ is assumed $\mu$-strongly convex, $L_{g,1}$-smooth, and Hessian Lipschitz; the upper-level $f_i$ is $L_{f,1}$-smooth and lower-bounded (non-convex). The communication graph is assumed strongly connected (or $C$-connected) at each step, with non-zero elements of $A^k, B^k$ lower-bounded by $a,b>0$.

## Key Experimental Results

### Main Results

| Task | Network/Setting | Baselines | FAB Performance |
|------|-----------|---------|----------|
| Distributed RL Policy Evaluation (Linear Bellman) | $\nu\in\{0.1,0.3,0.5\}$, Noise $\omega\in\{1,2,3\}$ | SGP-DL / Push-SAGA-DL / Push-ASGD-DL / AB-DL | Lowest loss and fastest convergence across all connectivity and noise combinations |
| Fashion-MNIST Data Hyper-cleaning (MLP, 203k params) | $cr\in\{0.4,0.5,0.6\}$, $\rho\in\{0.1,0.5,1\}$ | Same as above | Leads in test accuracy across all corruption rates and heterogeneity levels |
| IMDB Data Hyper-cleaning (BERT-110M fine-tuning) | Time-varying directed graph | Same as above | Maintains advantage in large-model NLP settings |
| MNIST Hyperparameter Tuning (Adversarial corruption) | 100 agents, $cr=0.2,0.4$ | SGP+grid / Push-DIGing+grid / AB+grid | Test accuracy significantly higher than single-level baselines with fixed $\lambda=0.2$; $\lambda$ adaptive trajectory evolves with training |

### Ablation Study

| Dimension | Observation | Conclusion |
|------|------|------|
| Network size $n$ increases | Convergence performance decreases but not exponentially | Consistent with the worst-case bound $(ab)^{-n}$; actual decay is more moderate |
| Connectivity $\nu$ decreases (sparser) | Both FAB and baselines slow down, but FAB degrades less | Bilevel + gradient tracking is more robust to sparse topologies |
| Noise $\omega$ increases | Single-level baselines oscillate significantly | Bilevel structure acts as a built-in adaptive regularizer, absorbing gradient noise |
| Peak Memory | FAB is on the same order of magnitude as single-level baselines | First-order design avoids the memory explosion typical of second-order methods |

### Key Findings
- **Theoretical open problems resolved simultaneously**: Reducing FAB to the single-level case (without $\lambda$) yields the $\mathcal{O}((ab)^{-n}K^{-1})$ convergence rate for AB/Push-Pull in time-varying directed non-convex scenarios, matching the $K^{-1}$ rate of centralized gradient descent.
- **Bilevel vs. grid search**: In the motivating experiment, FAB significantly outperforms any fixed-$\lambda$ single-level method under strong label noise ($cr=0.4$), and the $\lambda$ training trajectory clearly shows adaptive decay from large to small, proving the value of bilevel hyperparameter tuning.
- **Network scale factor $(ab)^{-n}$**: While theoretically exponential, experimental Figure 6(a) indicates only a slow degradation, consistent with constants appearing in worst-case analyses of subgradient-push and Push-Pull in literature.

## Highlights & Insights
- Integrates "distributed optimization", "bilevel optimization", and "time-varying directed graphs"—three distinct challenges—into a single framework with the first non-asymptotic convergence guarantee.
- The use of auxiliary variable $z$ to flatten the $\min\max\min$ structure into a single min-max layer aligns elegantly with the row/column dual-matrix design of AB/Push-Pull. This can be extended to other nested problems (meta-learning, adversarial robustness, actor-critic).
- The analytical framework is modular: taking the "single-level + time-varying directed + non-convex" subset immediately provides AB/Push-Pull convergence rates, a rare byproduct in algorithmic papers.
- The 100% first-order + gradient tracking design allows for direct embedding into PyTorch DDP/RPC without second-order derivative support, making it user-friendly for practical deployment.

## Limitations & Future Work
- The $(ab)^{-n}$ factor in the convergence constant, though a worst-case scenario, suggests potential performance drops for massive agent counts and extremely sparse communication; future work requires task-specific tighter analysis or communication compression.
- $\lambda=\mathcal{O}(K^{1/3})$ requires prior knowledge of $K$; a possible improvement is an increasing $\lambda$ scheme (Kwon 2023) adapted for the distributed setting.
- The strongly convex lower-level assumption does not hold in many RL/meta-learning tasks (e.g., non-convex value functions); relaxing this to PL or non-convex lower levels is a natural direction.
- The largest model tested was BERT-base (110M); end-to-end feasibility for larger models with cross-regional time-varying communication (e.g., satellite networks) remains to be verified.

## Related Work & Insights
- **vs. AB/Push-Pull series (Xin & Khan 2018; Pu 2021; Saadatniaki 2020; Nedić 2025)**: These works focus on single-level, primarily strong convexity; this paper tackles bilevel, non-convex, and time-varying settings, while filling the open problems they left.
- **vs. Distributed Bilevel (Yang 2022, Zhu 2024, Chen 2025)**: Prior work mostly relied on second-order derivatives (Hessian-vector) and static graphs; FAB is the first first-order solution applicable to both static and time-varying directed graphs.
- **vs. Centralized Bilevel (Kwon 2023, Chen 2025a)**: Their value function penalty informs the reformulation, but this paper additionally addresses the amplification effect between consensus error and $\lambda$.
- **vs. Push-DIGing / SGP**: While both target time-varying directed graphs, they are single-level; FAB demonstrates that with a bilevel structure, the same communication primitives can be used for hyperparameter adaptation beyond traditional ERM.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First distributed bilevel algorithm for time-varying directed graphs + resolves open problem for non-convex AB/Push-Pull
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers hyperparameter tuning, hyper-cleaning, and RL policy evaluation across CV and NLP, though lacks ultra-large-scale federated testing
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation, well-explained motivations, and compact Algorithm 1
- Value: ⭐⭐⭐⭐ Direct application potential in decentralized ML systems, satellite networks, and automated hyperparameter optimization in federated learning

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
