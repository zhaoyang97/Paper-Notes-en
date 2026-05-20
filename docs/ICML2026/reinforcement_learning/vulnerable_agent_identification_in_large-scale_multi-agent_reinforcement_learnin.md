---
title: >-
  [Paper Note] Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Mean-field MARL] This paper investigates the bi-level NP-hard problem of "identifying the K most vulnerable agents in a large-scale MARL system with N agents." The problem is formulate…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Mean-field MARL"
  - "Vulnerable Agent ID"
  - "Fenchel-Rockafellar"
  - "NP-hard Combinatorial Optimization"
  - "Robust Bellman"
date: 2026-05-08
content_hash: 474a5e2a0093ada1
---

# Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2509.15103](https://arxiv.org/abs/2509.15103)  
**Code**: https://github.com/Waken-dream/VAI  
**Area**: Reinforcement Learning / Multi-Agent / Adversarial Robustness  
**Keywords**: Mean-field MARL, Vulnerable Agent ID, Fenchel-Rockafellar, NP-hard Combinatorial Optimization, Robust Bellman

## TL;DR
This paper investigates the bi-level NP-hard problem of "identifying the K most vulnerable agents in a large-scale MARL system with N agents." The problem is formulated as HAD-MFC (Hierarchical Adversarial Decentralized Mean Field Control). The authors use the Fenchel-Rockafellar transform to fold the lower-level worst-case adversarial policy training into a regularized "robust mean-field Bellman operator," and convert the upper-level combinatorial selection into an MDP with dense rewards, solvable via greedy or RL methods. They prove the decomposition preserves optimality and outperform baselines in 17 out of 18 tasks.

## Background & Motivation

**Background**: Mean-field MARL (Yang 2018, Subramanian 2022) approximates "other agents" with a mean-field distribution, enabling MARL to scale to thousands of agents—applied in swarm robotics, power grid voltage control, and taxi dispatch. However, in real deployments, **a minority of agents dropping out, being attacked, or suffering hardware failures** is inevitable.

**Limitations of Prior Work**: (1) Existing MARL robustness studies are mostly small-scale—10 agents yield only $\binom{10}{1}=10$ attack scenarios, but 1000 agents have $\binom{1000}{100} \approx 10^{139}$, leading to combinatorial explosion; (2) Influence Maximization (IM) algorithms assume known graph structure and propagation rules, which MARL lacks; (3) Existing MARL attack methods (GMA-FGSM, RTCA) rely on random selection or differential evolution, ineffective for large-scale mean-field systems.

**Key Challenge**: This is a **bi-level coupled** problem—the upper level selects $K$ out of $N$ agents (NP-hard combinatorial optimization), while the lower level trains **worst-case adversarial policies** for these $K$ agents (a mean-field MARL problem). The two levels are interdependent: the upper-level selection depends on the lower-level's destructive potential, and vice versa. Direct bi-level RL does not converge, and combinatorial enumeration is infeasible.

**Goal**: (1) Mathematically define the problem as HAD-MFC and prove NP-hardness; (2) Find a way to **estimate the reward drop after attack** without actually training adversarial policies; (3) Convert the upper-level combinatorial problem into an MDP with dense rewards, efficiently solved by greedy or RL; (4) Prove the decomposition preserves global optimality.

**Key Insight**: Observation 1—Under mean-field approximation, the worst-case value of agent $i$'s Bellman operator, when perturbed by $\epsilon^i$ and with $\xi$ fraction of teammates perturbed, can be modeled with an $\ell_p$ ball constraint; Observation 2—The Fenchel-Rockafellar transform can convert the "inner min problem" into an "outer regularization term," turning the hard task of training a worst-case adversary into a single TD learning step on cooperative trajectories.

**Core Idea**: Compress "training the worst-case adversarial policy" into "learning a robust V function with $\epsilon$ and $\xi$ on cooperative trajectories," then use this V as a reward signal to drive an upper-level selection MDP—tractable and optimality-preserving.

## Method

### Overall Architecture
HAD-MFC formalization: $\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \mathcal{A}, \mathcal{P}, R, \mu_0, \nu_0, \gamma\rangle$. Each agent $i$ follows a well-trained cooperative policy $\pi_\beta$ by default; if selected into the attack set $\mathcal{K}$, its policy becomes $\hat{\pi}^i = \epsilon^i \pi_\alpha^i + (1-\epsilon^i) \pi_\beta^i$. The attacker's objective $\min_{\mathcal{K}} \min_{\pi_\alpha} J(\pi_\alpha, \pi_\beta)$ is bi-level NP-hard. The overall pipeline: (1) Collect trajectories $\tau \sim \pi_\beta$ under the cooperative policy and learn $Q^i(s^i, a^i, \mu, \nu)$ offline; (2) Use Fenchel-Rockafellar to derive the "regularized mean-field Bellman operator" $\mathcal{B}^R_{\epsilon^i, \xi}$ and learn $V^i(s^i, \mu, \epsilon^i, \xi)$ offline; (3) Use the difference in $V^i$ as the reward to formulate the upper-level combinatorial problem as an MDP, then sequentially select $K$ agents using greedy (VAI-Greedy) or DQN (VAI-RL).

### Key Designs

1. **Fenchel-Rockafellar Decoupling: Robust Bellman Operator**:

    - **Function**: Folds "training a worst-case $\pi_\alpha$" into a mean-field Bellman operator with a regularization term, eliminating the need for lower-level RL training.
    - **Mechanism**: Given the perturbed policy $\hat{\pi}^i = \epsilon^i \pi_\alpha^i + (1-\epsilon^i) \pi_\beta^i$ and perturbed mean-field action $\nu(a) = \xi \nu_\alpha(a) + (1-\xi)\nu_\beta(a)$, with $\hat{\pi}_\alpha^i = \hat{\pi}^i - \pi_\beta^i$ constrained by $\|\hat{\pi}_\alpha^i\|_p \le \epsilon^i$ and $\hat{\nu}_\alpha \lesssim \xi$. Applying the Fenchel-Rockafellar transform (convex conjugate duality) to the robust Bellman inequality $V^i \le (\mathcal{B}^{\hat{\pi}} V^i)$ yields the **regularized mean-field Bellman operator**: $\mathcal{B}^R_{\epsilon^i, \xi} V^i = (\mathcal{B}^{\pi_\beta} V^i) - (\epsilon^i + \xi + \epsilon^i \xi) \|Q^i\|_q$, where $1/p + 1/q = 1$. **Key point**: This is an **exact** transformation (as long as the uncertainty set is convex, proper, and lower semi-continuous, which $\ell_p$ balls satisfy), introducing no approximation. The learned $V^i(s^i, \mu, \epsilon^i, \xi)$ is equivalent to "the worst-case expected return for agent $i$ under $\epsilon^i$ perturbation and $\xi$ fraction of team perturbed." The operator is also a contraction (Proposition 4.3).
    - **Design Motivation**: Directly training the worst-case adversary requires solving the RL problem $\min_{\pi_\alpha} J(\pi_\alpha, \pi_\beta)$, and retraining for each $\mathcal{K}$, which is computationally infeasible. Fenchel-Rockafellar replaces the "inner min" with an "outer regularization," so the entire training only needs **cooperative trajectories**—$\pi_\alpha$ never needs to exist.

2. **Upper-Level as MDP with Dense Reward**:

    - **Function**: Reformulates the NP-hard $\binom{N}{K}$ combinatorial selection as a sequential selection MDP, enabling both greedy and RL solutions.
    - **Mechanism**: The MDP is defined as $\mathcal{M} = \langle \boldsymbol{\mathcal{S}}, \epsilon, \mathcal{N}, \tilde{\mathcal{P}}, \tilde{R}, \gamma\rangle$, where at each step, an agent is added to the attack set $\mathcal{K}_k = \mathcal{K}_{k-1} \cup n_k$. **Reward is defined as "the expected system return drop after adding this agent"**: $r_k = \frac{1}{N}\sum_i (V^i(s_0^i, \mu_0, \epsilon^i_{k-1}, \xi_{k-1}) - V^i(s_0^i, \mu_0, \epsilon^i_k, \xi_k))$, where $V^i$ is the robust V learned above. This is equivalent to "the expected system reward loss per added agent," providing **dense signals** (unlike traditional combinatorial optimization with only terminal rewards). DQN (VAI-RL) or greedy selection (VAI-Greedy) is then used. Proposition 4.5 proves this decomposition preserves the original HAD-MFC optimum.
    - **Design Motivation**: Traditional K-subset selection problems have sparse rewards (only at the end), making training slow; using robust V's stepwise difference as reward densifies the signal, reducing the NP-hard problem to an MDP. VAI-Greedy requires no training ("no-learning baseline"); VAI-RL uses DQN to capture long-term dependencies, and the paper shows RL significantly outperforms greedy when the number of attackers is large (e.g., in Battle with 144 agents and 36 attackers, RL outperforms Greedy by ~30%).

3. **Unified TD Loss for Offline Learning of V and Q**:

    - **Function**: Grounds robust V learning as standard TD loss, enabling the entire pipeline to be completed on cooperative trajectories.
    - **Mechanism**: The TD loss is $\min \mathbb{E}_{\tau \sim \pi_\beta}(V^i(s^i, \mu, \epsilon^i, \xi) - r - \gamma V^i(s'^i, \mu', \epsilon^i, \xi) + (\epsilon^i \xi + \epsilon^i + \xi)\|Q^i(s^i, a^i_\beta, \mu, \nu_\beta)\|_q)^2$, where $\epsilon \sim U[0, 2^{1/p}], \xi \sim \text{Bernoulli}(\xi)$. $Q^i$ is learned and fixed under the cooperative policy; $V^i$ is newly learned. Both use only cooperative trajectories, **no further environment interaction** (Proposition 4.4 provides geometric intuition: $\epsilon^i \xi \|Q^i\|_q$ is the worst-case first-order deviation of Q within the $\ell_p$ ball).
    - **Design Motivation**: Black-box attack assumption—the attacker cannot access the victim's policy parameters, only cooperative trajectories; this TD loss requires only such data, matching real-world threat models. Random sampling of $\epsilon$ and $\xi$ enables V to learn a family of value functions under different perturbations, allowing queries for any $(\epsilon, \xi)$ configuration.

### Loss & Training
Cooperative Q: First, train $Q^i$ under the cooperative policy $\pi_\beta$ using MF-Q (Battle) or MF-AC (Taxi), then freeze it. Robust V: Train $V^i(\epsilon^i, \xi)$ using the above TD loss. Upper-level: VAI-Greedy selects the agent with the highest reward at each step; VAI-RL uses DQN to sequentially select K agents. All baselines (Random, DC, Bi-level RL, PIANO, RTCA) share the same network architecture and hyperparameters, with five random seeds.

## Key Experimental Results

### Main Results
Three environments (Battle, Taxi Matching, Vicsek flocking dynamics) with 18 subtasks (different map sizes × number of attackers). Key rows from the table (Battle ↓ lower is stronger attack; Vicsek ↑ higher means closer to target policy):

| Environment/Scale | #Adv | Random | DC | PIANO | RTCA | VAI-Greedy | VAI-RL |
|-------------------|------|--------|-----|-------|------|------------|--------|
| Battle-64         | 32   | -152.89| -160.51 | -175.24 | -192.78 | -214.40 | **-929.88** |
| Battle-144        | 72   | -1809  | -2014   | -2313  | -2221  | -2579   | **-2837**   |
| Taxi-100          | 36   | 884.49 | 867.62  | 793.71 | 860.58 | 770.14  | **652.10**  |
| Vicsek-400        | 200  | -295.13| -313.55 | -290.53| -287.53| -256.44 | -275.62     |

VAI-RL achieves -929.88 on Battle-64 with 32 adversaries, 5× stronger than the next best baseline (-214 → -929), indicating the ability to find **catastrophic failure** combinations when enough attackers are present. Outperforms baselines in 17/18 tasks.

### Ablation Study

| Configuration | Description | Effect |
|---------------|-------------|--------|
| Random        | Randomly select K agents | Weak baseline |
| DC            | Degree centrality, select most neighbors | Strong in rule-based, weak in MARL (central agents not always most vulnerable) |
| Bi-level RL   | Both levels trained end-to-end with RL | Weaker than VAI (no explicit reward signal) |
| PIANO         | GNN + RL sequential selection | Does not consider worst-case adversary |
| RTCA          | Differential evolution | Only effective at small scale |
| VAI-Greedy    | Greedy only, no RL | Close to RL with few attackers |
| VAI-RL        | DQN at upper level | Significantly outperforms greedy with many attackers |

### Key Findings
- **VAI-RL outperforms Greedy with many attackers**: RL wins 10 out of 18 tasks, with the gap concentrated in scenarios with many attackers (e.g., Battle 144 + 72 adversaries, RL inflicts +260 more damage than Greedy). Reason: Greedy only considers immediate reward, RL models long-term agent interactions.
- **DC (degree centrality) fails**: In Battle, frontline agents are more vulnerable than those in the central crowd, but DC tends to select central agents, resulting in poor performance. This shows graph-based heuristics are unreliable in large-scale MARL.
- **PIANO/Bi-level RL are ineffective**: Existing learning-based baselines cannot solve the worst-case selection problem, as they do not explicitly model "adversarial policy after agent selection."
- **Applicable to rule-based systems (Vicsek)**: By converting rule-based agents to Boltzmann policies, the Q function can be estimated, extending the method to robustness analysis beyond MARL.
- **Agent-level vulnerability is interpretable**: Figure 1 visualizes each agent's contribution under $\epsilon=1$, directly identifying "which positions/roles are key vulnerabilities."

## Highlights & Insights
- **Fenchel-Rockafellar transforms RL into offline learning**—a clever move that compresses "training adversarial policies" into "TD learning with a regularization term on cooperative trajectories," applicable to any "min-max RL" scenario (adversarial RL, robust MDP, robust SAC).
- The trick of converting NP-hard combinatorial optimization into a dense reward MDP (reward = marginal gain per added element) is general and can be applied to feature selection, point cloud downsampling, subset DPP, and all "K-subset" problems.
- The method is **fully black-box**—requires neither victim policy parameters nor the true environment model, only cooperative trajectories, matching real-world attacker capabilities.
- Proposition 4.4's geometric intuition ($\epsilon^i \xi \|Q^i\|_q$ as the worst-case first-order deviation of Q within the $\ell_p$ ball) gives physical meaning to the regularization term, providing intuitive justification.
- The "vulnerability visualization" (Figure 1) is a valuable byproduct—providing system designers with a direct map: "these agents are key nodes, prioritize their protection," guiding fault-tolerant design in real deployments.

## Limitations & Future Work
- Learning $V^i$ relies on function approximation; as noted in Remark 3, this is the only source of approximation, but the propagation from "V function error → agent selection error" is not systematically characterized; if Q is poorly learned, the method fails.
- $\epsilon^i = 1$ assumes extreme attacks (full agent control), while partial perturbations are more common in practice; although other $\epsilon$ values are discussed in Appendix D.2, deeper real-world defense use cases are not explored.
- The upper-level MDP's reward is the difference in V; **if V's error exceeds the difference, the gradient signal is distorted**; may be unstable in large-scale, sparse-reward environments.
- Experiments are limited to ≤400 agents; scalability to truly "tens of thousands" (e.g., city-wide networks) needs further validation.
- VAI-RL uses DQN, which is naturally suited for discrete action spaces (agent selection), but for large $K$ and exploding state spaces, whether PPO/actor-critic is better remains unexplored.

## Related Work & Insights
- **vs RTCA (Zhou & Liu 2023)**: RTCA uses differential evolution to select vulnerable agents in small-scale MARL; VAI works for large-scale mean-field. The key difference is VAI's theoretical decoupling guarantee, while RTCA is black-box search.
- **vs Influence Maximization (Kempe et al. 2003)**: IM assumes known graph and propagation rules; VAI is graph-free and rule-free, inferring via the V function—essentially IM in a "rule-free learnable" RL setting.
- **vs Bi-level RL (Vezhnevets et al. 2017)**: Direct bi-level RL does not converge in this problem (signal too sparse); VAI uses Fenchel-Rockafellar for explicit decoupling, avoiding bi-level training.
- **vs PR-MDP (Tessler et al. 2019)**: PR-MDP formalizes "perturbed proportion $\epsilon$" but does not address combinatorial selection; VAI extends PR-MDP to mean-field + combinatorial selection.
- **vs GMA-FGSM**: That is gradient-based adversarial attack; VAI is Q/V-based interpretable selection, providing rankings rather than attacking a single group.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Fenchel-Rockafellar folding worst-case RL into cooperative TD learning is truly innovative; converting NP-hard combinatorial optimization to dense reward MDP is elegant; Proposition 4.5's proof of lossless decomposition is a core theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 3 environments, 18 subtasks, 5 baselines, generalization to rule-based systems, agent-level visualization—already solid; lacks larger-scale and different $\epsilon$ stress tests.
- Writing Quality: ⭐⭐⭐⭐ — Propositions and proof logic are clearly connected; however, the Fenchel-Rockafellar section is challenging for non-RL readers and lacks worked examples.
- Value: ⭐⭐⭐⭐⭐ — Robustness evaluation for large-scale MARL deployments in robotics, power grids, and urban traffic is a real need; this is the first efficient tool for locating vulnerable nodes, with clear industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](../../AAAI2026/reinforcement_learning/explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[ICML 2026\] NonZero: Interaction-Guided Exploration for Multi-Agent Monte Carlo Tree Search](nonzero_interaction-guided_exploration_for_multi-agent_monte_carlo_tree_search.md)
- [\[ICLR 2026\] Continuous-Time Value Iteration for Multi-Agent Reinforcement Learning](../../ICLR2026/reinforcement_learning/continuous-time_value_iteration_for_multi-agent_reinforcement_learning.md)
- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](../../ICLR2026/reinforcement_learning/spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)

</div>

<!-- RELATED:END -->
