---
title: >-
  [Paper Note] Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Mean-field MARL] This paper investigates the bi-level NP-hard problem of "identifying the $K$ most vulnerable agents in a large-scale MARL system with $N$ agents." The problem is model…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Mean-field MARL"
  - "Vulnerable Agent ID"
  - "Fenchel-Rockafellar"
  - "NP-hard Combinatorial Optimization"
  - "Robust Bellman"
date: 2026-05-08
content_hash: 751481879d91917c
---

# Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2509.15103](https://arxiv.org/abs/2509.15103)  
**Code**: https://github.com/Waken-dream/VAI  
**Area**: Reinforcement Learning / Multi-Agent Systems / Adversarial Robustness  
**Keywords**: Mean-field MARL, Vulnerable Agent ID, Fenchel-Rockafellar, NP-hard Combinatorial Optimization, Robust Bellman  

## TL;DR
This paper investigates the bi-level NP-hard problem of "identifying the $K$ most vulnerable agents in a large-scale MARL system with $N$ agents." The problem is modeled as HAD-MFC (Hierarchical Adversarial Decentralized Mean Field Control). By utilizing the Fenchel-Rockafellar transform, the training of the lower-level worst-case adversarial policy is folded into a "robust mean-field Bellman operator" with a regularization term. The upper-level combinatorial selection problem is then transformed into an MDP with dense rewards, solved via greedy or RL methods. The decomposition is proven to preserve optimality, and the method outperforms baselines in 17 out of 18 tasks.

## Background & Motivation

**Background**: Mean-field MARL (Yang 2018, Subramanian 2022) enables scaling to thousands of agents by approximating "other agents" as a mean-field distribution. It has been applied in robot swarm control, power grid voltage control, and taxi scheduling. However, in real-world deployments, **a minority of agents going offline, being attacked, or experiencing hardware failure** is inevitable.

**Limitations of Prior Work**: (1) Existing MARL robustness research focuses on small scales—a 10-agent system has only $\binom{10}{1}=10$ attack scenarios, while a 1,000-agent system has $\binom{1000}{100} \approx 10^{139}$ scenarios, causing combinatorial explosion. (2) Influence Maximization (IM) algorithms assume known graph structures and propagation rules, which are unavailable in MARL. (3) Existing MARL attack methods (GMA-FGSM, RTCA) rely on random selection or differential evolution, which are ineffective for large-scale mean-field systems.

**Key Challenge**: This is a **bi-level coupled** problem. The upper level must select $K$ out of $N$ agents (NP-hard combinatorial optimization), while the lower level must train the **worst-case adversarial policies** for these $K$ agents (a mean-field MARL problem). These levels are interdependent: the upper-level choice depends on the potential damage the lower level can cause, and the lower-level training depends on which agents were selected. Direct bi-level RL fails to converge, and combinatorial enumeration is infeasible.

**Goal**: (1) Formally define the problem as HAD-MFC and prove its NP-hardness. (2) Find a proxy to estimate "reward drop after attack" **without actually training adversarial policies**. (3) Transform the upper-level combinatorial problem into an MDP with dense rewards for efficient greedy or RL solving. (4) Prove that this decomposition does not compromise global optimality.

**Key Insight**: Observation 1—Under mean-field approximation, the "worst-case value" of the Bellman operator for the $i$-th agent under an $\epsilon^i$ perturbation and a $\xi$ proportion of perturbed peers can be modeled using $\ell_p$-ball constraints. Observation 2—The Fenchel-Rockafellar transform can convert the "inner minimization problem" into an "outer regularization term," transforming the complex task of training a worst-case adversary into TD learning on cooperative trajectories.

**Core Idea**: Compress the "training of the worst-case adversarial policy" into "learning a robust $V$-function containing $\epsilon$ and $\xi$ on cooperative trajectories." This $V$-function is then used as a reward signal to drive an upper-level selection MDP, achieving both computational efficiency and optimality preservation.

## Method

### Overall Architecture
HAD-MFC Formalization: $\mathcal{G} = \langle \mathcal{N}, \mathcal{S}, \mathcal{A}, \mathcal{P}, R, \mu_0, \nu_0, \gamma\rangle$. Each agent $i$ follows a well-trained cooperative policy $\pi_\beta$ by default. If selected for the attack set $\mathcal{K}$, the policy becomes $\hat{\pi}^i = \epsilon^i \pi_\alpha^i + (1-\epsilon^i) \pi_\beta^i$. The attacker's objective $\min_{\mathcal{K}} \min_{\pi_\alpha} J(\pi_\alpha, \pi_\beta)$ is bi-level NP-hard. The overall pipeline: (1) Sample trajectories $\tau \sim \pi_\beta$ and learn $Q^i(s^i, a^i, \mu, \nu)$ offline. (2) Derive the "regularized mean-field Bellman operator" $\mathcal{B}^R_{\epsilon^i, \xi}$ via Fenchel-Rockafellar and learn $V^i(s^i, \mu, \epsilon^i, \xi)$ offline. (3) Formulate the upper-level combinatorial problem as an MDP using the difference in $V^i$ as rewards, solving it sequentially via VAI-Greedy or VAI-RL (DQN).

### Key Designs

1. **Fenchel-Rockafellar Decoupling: Robust Bellman Operator**:
    - **Function**: Folds "training a worst-case $\pi_\alpha$" into "a mean-field Bellman operator with a regularization term," eliminating lower-level RL training.
    - **Mechanism**: Let the perturbed policy be $\hat{\pi}^i = \epsilon^i \pi_\alpha^i + (1-\epsilon^i) \pi_\beta^i$ and the perturbed mean-field action be $\nu(a) = \xi \nu_\alpha(a) + (1-\xi)\nu_\beta(a)$. Defining $\hat{\pi}_\alpha^i = \hat{\pi}^i - \pi_\beta^i$ constrained by $\|\hat{\pi}_\alpha^i\|_p \le \epsilon^i$ and $\hat{\nu}_\alpha \lesssim \xi$, applying the Fenchel-Rockafellar transform (convex conjugate duality) to the robust Bellman inequality $V^i \le (\mathcal{B}^{\hat{\pi}} V^i)$ yields the **regularized mean-field Bellman operator**: $\mathcal{B}^R_{\epsilon^i, \xi} V^i = (\mathcal{B}^{\pi_\beta} V^i) - (\epsilon^i + \xi + \epsilon^i \xi) \|Q^i\|_q$, where $1/p + 1/q = 1$.  
    - **Key**: This is an **exact** transformation provided the uncertainty set is convex, proper, and lower semi-continuous (satisfied by $\ell_p$ balls). The learned $V^i(s^i, \mu, \epsilon^i, \xi)$ is equivalent to the "worst-case expected return for agent $i$ when it is perturbed by $\epsilon^i$ and a $\xi$ proportion of its peers are perturbed."
    - **Design Motivation**: Training a worst-case adversary requires solving an RL problem for every possible $\mathcal{K}$, which is computationally impossible. Fenchel-Rockafellar replaces the "inner min" with "outer regularization," requiring only **cooperative trajectories** for training—$\pi_\alpha$ never needs to be explicitly trained.

2. **Upper-level Transformation to MDP with Dense Reward**:
    - **Function**: Reformulates the NP-hard $\binom{N}{K}$ selection problem into a sequential MDP tractable by Greedy and RL methods.
    - **Mechanism**: Define MDP $\mathcal{M} = \langle \boldsymbol{\mathcal{S}}, \epsilon, \mathcal{N}, \tilde{\mathcal{P}}, \tilde{R}, \gamma\rangle$. At each step, an agent is added to the attack set $\mathcal{K}_k = \mathcal{K}_{k-1} \cup n_k$. The **reward is defined as the drop in expected system return**: $r_k = \frac{1}{N}\sum_i (V^i(s_0^i, \mu_0, \epsilon^i_{k-1}, \xi_{k-1}) - V^i(s_0^i, \mu_0, \epsilon^i_k, \xi_k))$. This provides **dense signals** unlike aggregate rewards in traditional optimization. The MDP is solved using DQN (VAI-RL) or by greedily picking the agent with the highest reward (VAI-Greedy). Proposition 4.5 proves this decomposition maintains the original HAD-MFC optimal solution.
    - **Design Motivation**: Sparse rewards in traditional K-subset selection lead to slow training. Using robust $V$-function differences spreads reward signals across every step. VAI-Greedy serves as a no-learning baseline, while VAI-RL captures long-term dependencies, outperforming Greedy as the number of attackers increases.

3. **Unified Offline Learning of V and Q via TD Loss**:
    - **Function**: Implements robust $V$ learning as standard TD loss, allowing the pipeline to be completed using only cooperative trajectories.
    - **Mechanism**: The TD loss is $\min \mathbb{E}_{\tau \sim \pi_\beta}(V^i(s^i, \mu, \epsilon^i, \xi) - r - \gamma V^i(s'^i, \mu', \epsilon^i, \xi) + (\epsilon^i \xi + \epsilon^i + \xi)\|Q^i(s^i, a^i_\beta, \mu, \nu_\beta)\|_q)^2$, where $\epsilon \sim U[0, 2^{1/p}], \xi \sim \text{Bernoulli}(\xi)$. $Q^i$ is fixed and learned under the cooperative policy, while $V^i$ is learned to represent returns under various perturbation levels.
    - **Design Motivation**: Black-box attack assumption—attackers cannot access victim policy parameters and must use cooperative trajectories. This TD loss fits this realistic threat model without further environment interaction.

### Loss & Training
Cooperative Q: Pre-trained using MF-Q (Battle) or MF-AC (Taxi) under policy $\pi_\beta$ and frozen. Robust V: Trained using the TD loss specified above. Upper Level: VAI-Greedy selects the highest reward agent per step; VAI-RL uses DQN to select $K$ agents sequentially. All baselines (Random, DC, Bi-level RL, PIANO, RTCA) share the same network architecture and hyperparameters across five random seeds.

## Key Experimental Results

### Main Results
Testing in three environments (Battle, Taxi Matching, Vicsek Dynamics) across 18 sub-tasks. Key results (Battle ↓ lower is better/stronger attack; Vicsek ↑ higher is better/closer to target):

| Env/Scale | # Adv | Random | DC | PIANO | RTCA | VAI-Greedy | VAI-RL |
|----------|-------|--------|-----|-------|------|------------|--------|
| Battle-64 | 32 | -152.89 | -160.51 | -175.24 | -192.78 | -214.40 | **-929.88** |
| Battle-144 | 72 | -1809 | -2014 | -2313 | -2221 | -2579 | **-2837** |
| Taxi-100 | 36 | 884.49 | 867.62 | 793.71 | 860.58 | 770.14 | **652.10** |
| Vicsek-400 | 200 | -295.13 | -313.55 | -290.53 | -287.53 | -256.44 | -275.62 |

VAI-RL achieves -929.88 in Battle-64 with 32 adversaries, 5x stronger than the second-best baseline (-214), indicating it finds vulnerable combinations that cause total system failure. It outperforms baselines in 17/18 tasks.

### Ablation Study

| Config | Description | Effect |
|------|------|------|
| Random | Random selection of $K$ agents | Weak baseline |
| DC | Degree Centrality (picking most connected agents) | Effective in rule-based systems, weak in MARL |
| Bi-level RL | End-to-end RL for both levels | Weaker than VAI (lacks explicit signals) |
| PIANO | GNN + RL sequential selection | Does not account for worst-case adversaries |
| RTCA | Differential Evolution | Only effective for small scales |
| VAI-Greedy | Greedy only, no RL | Close to RL for small attacker counts |
| VAI-RL | Upper-level DQN | Significantly outperforms Greedy for large attacker counts |

### Key Findings
- **VAI-RL outperforms Greedy with more attackers**: RL wins 10/18 tasks, specifically where attacker counts are high (e.g., +260 improvement in Battle-144). RL models long-term synergistic effects between agents that Greedy ignores.
- **DC (Degree Centrality) failure**: In Battle, frontline agents are more vulnerable than those in the central crowd. DC tends to pick central agents, resulting in poor performance. This suggests graph heuristics are unreliable in large MARL systems.
- **PIANO/Bi-level RL insufficiency**: Existing learning baselines fail to solve worst-case selection because they do not explicitly model potential adversarial behavior of the selected agents.
- **Rule-based system support**: By converting rule-based agents into Boltzmann policies, Q-functions can be estimated, allowing VAI to extend to non-MARL robustness analysis.
- **Interpretable Vulnerability**: Figure 1 visualizes each agent's contribution under $\epsilon=1$, highlighting critical vulnerabilities based on position/role.

## Highlights & Insights
- Using the **Fenchel-Rockafellar transform to turn a min-max RL problem into offline TD learning** is a powerful innovation. It bypasses the engineering difficulty of training adversaries and is portable to other min-max RL scenarios (Robust MDP, Robust SAC).
- The strategy of converting NP-hard combinatorial optimization into an MDP with dense rewards (reward = gain difference) is a **universal trick** applicable to feature selection, point cloud downsampling, and subset DPP problems.
- The method is **entirely black-box**, requiring no victim policy parameters or environment models, only cooperative trajectories. This aligns with realistic threat models.
- The **geometric intuition** (Proposition 4.4) showing that the regularization term corresponds to the worst-case first-order deviation of $Q$ in the $\ell_p$ ball provides a solid theoretical grounding.
- The "Vulnerability Visualization" provides significant value for system designers, identifying critical nodes for prioritized protection in fault-tolerant design.

## Limitations & Future Work
- The learning of $V^i$ relies on function approximation, which is the singular source of error. The paper does not systematically characterize how $V$-function error propagates to selection error.
- The $\epsilon^i = 1$ "total control" assumption is extreme. Real-world scenarios often involve partial perturbations.
- Stable gradient signals for the upper-level MDP depend on the precision of $V$ differences; if the $V$ approximation error exceeds the difference, the signal becomes noisy.
- Experiments are restricted to $\le 400$ agents. Scalability to "million-agent" systems (e.g., city-wide traffic) requires further validation.
- VAI-RL uses DQN, which suits discrete agent selection. For massive $K$ or state spaces, PPO/Actor-Critic alternatives remain unexplored.

## Related Work & Insights
- **vs RTCA (Zhou & Liu 2023)**: RTCA uses differential evolution for small MARL. VAI scales to mean-field systems with theoretical decoupling guarantees versus RTCA's black-box search.
- **vs Influence Maximization (Kempe et al. 2003)**: IM assumes known graphs/rules. VAI is graph-agnostic, inferring vulnerability via $V$-functions (RL version of IM for "rule-less" scenarios).
- **vs Bi-level RL (Vezhnevets et al. 2017)**: Direct bi-level RL often fails to converge due to signal sparsity. VAI avoids this through explicit Fenchel-Rockafellar decoupling.
- **vs PR-MDP (Tessler et al. 2019)**: PR-MDP formalizes perturbation ratios but does not solve combinatorial selection. VAI extends PR-MDP to mean-field + selection.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Converting worst-case RL to cooperative TD learning via Fenchel-Rockafellar is genuinely innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 18 tasks across 3 environments with multiple baselines and rule-based generalization is robust, though more stress tests on $\epsilon$ would be beneficial.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear connection between propositions and proofs, though the Fenchel-Rockafellar section is mathematically dense for non-RL specialists.
- **Value**: ⭐⭐⭐⭐⭐ — High industrial value for assessing the robustness of large-scale MARL in robot swarms, grids, and traffic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Interaction-Breaking Adversarial Learning Framework for Robust Multi-Agent Reinforcement Learning](interaction-breaking_adversarial_learning_framework_for_robust_multi-agent_reinf.md)
- [\[ICML 2026\] LLM-Guided Communication for Cooperative Multi-Agent Reinforcement Learning](llm-guided_communication_for_cooperative_multi-agent_reinforcement_learning.md)
- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)
- [\[ICML 2026\] Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control](plug-and-play_benchmarking_of_reinforcement_learning_algorithms_for_large-scale_.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](../../AAAI2026/reinforcement_learning/explaining_decentralized_multi-agent_reinforcement_learning_policies.md)

</div>

<!-- RELATED:END -->
