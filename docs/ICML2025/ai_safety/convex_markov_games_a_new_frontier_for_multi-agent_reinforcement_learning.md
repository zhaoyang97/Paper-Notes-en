---
title: >-
  [Paper Note] Convex Markov Games: A New Frontier for Multi-Agent Reinforcement Learning
description: >-
  [ICML2025][AI Safety][Convex Markov Game] Proposes the **Convex Markov Game (cMG)** framework, generalizing single-agent convex MDPs to multi-agent settings, which allows general convex preferences over occupancy measures (such as entropy, KL divergence, fairness penalties, and safety constraints). It proves the existence of pure-strategy Nash equilibria and designs a differentiable Projected Gradient Loss (PGL) algorithm to approximate equilibria.
tags:
  - "ICML2025"
  - "AI Safety"
  - "Convex Markov Game"
  - "Nash Equilibrium"
  - "Occupancy Measure"
  - "Multi-Agent RL"
  - "convex optimization"
  - "safety"
  - "fairness"
date: 2026-05-08
content_hash: a65850b288f9b85c
---

# Convex Markov Games: A New Frontier for Multi-Agent Reinforcement Learning

**Conference**: ICML2025  
**arXiv**: [2410.16600](https://arxiv.org/abs/2410.16600)  
**Code**: Not open-sourced  
**Area**: AI Safety  
**Keywords**: Convex Markov Game, Nash Equilibrium, Occupancy Measure, Multi-Agent RL, convex optimization, safety, fairness

## TL;DR

Proposes the **Convex Markov Game (cMG)** framework, generalizing single-agent convex MDPs to multi-agent settings, which allows general convex preferences over occupancy measures (such as entropy, KL divergence, fairness penalties, and safety constraints). It proves the existence of pure-strategy Nash equilibria and designs a differentiable Projected Gradient Loss (PGL) algorithm to approximate equilibria.

## Background & Motivation

- **MDP $\to$ Convex MDP**: In single-agent RL, convex MDP (cMDP) allows the objective to be a general convex function of the occupancy measure (such as maximizing exploration entropy), rather than just linear reward accumulation.
- **Markov Games (MG)**: The classic framework for multi-agent RL, but the objectives remain limited to linear rewards.
- **Gap**: The intersection of multi-agent and convex objectives ($n > 1$ with convex losses) lacked a formal definition, let alone a proof of the existence of Nash equilibria.
- **Real-world Demand**: Objectives such as behavior diversity (creativity), imitation learning, fairness, and safety constraints naturally violate timestep additivity and cannot be modeled by standard MGs.

| | Linear Loss | Convex Loss |
|---|---|---|
| n=1 | MDP | Convex MDP |
| n>1 | Markov Game | **Convex Markov Game (Ours)** |

## Method

### 1. Definition of Convex Markov Games

A cMG is defined as a tuple $\mathcal{G} = \langle \mathcal{S}, \mathcal{A}, P, u, \gamma, \mu_0 \rangle$, where the key difference is that the utility function $u_i$ is a **concave function** of the occupancy measure $\mu_i$ (equivalent to minimizing a convex loss):

$$u_i : \left(\prod_{j=1}^{n} \Delta^{\mathcal{S} \times \mathcal{A}_j}\right) \to \mathbb{R}$$

The occupancy measure is defined as:

$$\mu^{\pi}(s, a) = (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t \mathbb{P}(s_t = s, a_t = a \mid \mu_0, \pi, P)$$

The state occupancy can be recovered via the matrix equation: $\mu^s(\pi) = (1-\gamma)[I - \gamma P^{\pi}]^{-1} \mu_0$.

### 2. Existence of Nash Equilibrium

- **Mixed-strategy NE** (Proposition 1): Compact and convex policy space + continuous utility $\to$ directly derived from Glicksberg's theorem.
- **Pure-strategy NE** (Theorem 1, Core Contribution): The best-response set is convex in the occupancy measure space but **non-convex** in the policy space (breaking the premise of Kakutani's fixed-point theorem). The paper resorts to topological arguments—proving that the best-response set is **contractible**, and completes the proof using the more general existence theorem of Debreu (1952).

### 3. Equilibrium Computation: Projected Gradient Loss (PGL)

**Exploitability** definition:

$$\epsilon = \max_{i} \left[\max_{z \in \mathcal{M}_i(\pi_{-i})} u_i(z, \pi_{-i}) - u_i(\mu_i, \pi_{-i})\right]$$

Directly minimizing exploitability requires solving $n$ convex programs, which is computationally expensive. The paper derives an **upper bound** after introducing entropy regularization:

$$\epsilon_i(\pi) \leq \tau \log(|\mathcal{S}||\mathcal{A}_i|) + \sqrt{2} \|\Pi_{T\mathcal{U}_i}(\nabla_{\mu_i}^{i\tau})\|$$

where $\Pi_{T\mathcal{U}_i}$ is the projection operator onto the tangent space of the feasible set of occupancy measures:

$$\Pi_{T\mathcal{U}_i} = I - A^\top (A A^\top)^{-1} A$$

This defines a differentiable loss function:

$$\mathcal{L}^{\tau}(\pi) = \sum_i \|\Pi_{T\mathcal{U}_i}(\nabla_{\mu_i}^{i\tau})\|^2$$

**PGL Algorithm**: Optimizes $\mathcal{L}^{\tau}$ using Adam in the policy logit space, gradually annealing the temperature $\tau \to 0$. This combines the advantages of both perspectives—independent, simple projections from the policy perspective, and a convex upper bound from the occupancy measure perspective.

### 4. Four Types of Application Scenarios

| Application | Utility Function Form | Core Idea |
|---|---|---|
| **Creativity** | $u_i = r_i^\top \mu_i + \tau H(\mu_i)$ | Entropy reward encourages diverse behaviors |
| **Imitation** | $u_i = r_i^\top \mu_i - \tau \,d_{\text{KL}}(\mu_i \| \mu_i^{\text{ref}})$ | KL regularization aligns with reference policies |
| **Fairness** | $u_i = r_i^\top \mu_i - (\text{state frequency difference})^2$ | Penalizes uneven visitation frequencies |
| **Safety** | $u_i = r_i^\top \mu_i - c \cdot \max(0, \mu_i(s_{\text{danger}}, a_{\text{fast}}) - 0.1)$ | Non-smooth convex penalty restricts dangerous behaviors |

## Key Experimental Results

| Experimental Scenario | Method | Exploitability | Key Findings |
|---|---|---|---|
| Multi-agent pathfinding | PGL | $\epsilon \approx 1.7$ ($\approx 7\%$ utility) | Learners coordinate to pass through bottleneck corridors |
| Iterated Prisoner's Dilemma (IPD) | PGL | Converges to $\approx 0$ | Discovered reciprocal cooperative strategies (utility = 0.47 > DD's 0.33) |
| Iterated Public Goods Game (IPGG) | PGL | Converges to $\approx 0$ | Found conditional contribution strategies (utility = 0.03 > 0 of always defecting) |
| Bach-Stravinsky Fairness | PGL | $\epsilon \le 2.5 \times 10^{-5}$ | 60/40 fair attendance, frequency difference $< 10^{-5}$ |
| Warehouse Robot Safety | PGL (no safety loss) | $\epsilon \le 3.4 \times 10^{-2}$ | Fast actions account for 69% |
| Warehouse Robot Safety | PGL (with safety loss) | $\epsilon \le 1.0 \times 10^{-3}$ | Fast actions decrease to 42%, safety behaviors improve |
| IPD Human Imitation | PGL + KL Annealing | $\epsilon = 1.4 \times 10^{-4}$ | Policies are close to humans but with higher utility (0.48 vs. 0.46) and 3 orders of magnitude lower exploitability |

Baseline Comparison: Direct minimization of exploitability (cvxpylayers) **crashed** due to numerical instability; Round-Robin converged but found simpler, state-independent equilibria; sgamesolver similarly only found state-independent NE of the underlying NFGs. PGL uniquely discovered **state-dependent symmetric policies**.

## Highlights & Insights

1. **Filling Theoretical Gaps**: Formally defines cMG and proves the existence of pure-strategy NE for the first time. The topological argument (contractibility $\to$ fixed point) bypasses the non-applicability of both the Bellman equation and Kakutani's theorem.
2. **Unified Framework**: Four major applications—creativity, imitation, fairness, and safety—are treated unifiedly under the same mathematical language.
3. **Transient Imitation Effect**: The entropy reward during the high-temperature annealing phase forces agents to explore cooperative states. Although the reward vanishes at low temperatures, cooperative behavior is "locked in"—representing an ingenious equilibrium selection mechanism.
4. **Experimental Findings**: The reciprocity strategy discovered in IPD is structurally similar to tit-for-tat but more robust (with extremely low exploitability). In the warehouse scenario, the convex safety penalty effectively reduces the frequency of dangerous actions.
5. **Differentiable End-to-End**: The loss function, projection operators, and occupancy measure mapping are all differentiable, supporting automatic differentiation optimization.

## Limitations & Future Work

1. **Centralized + Requiring Known Dynamics**: The current algorithm assumes complete knowledge of transition probabilities and cannot be directly applied to model-free / decentralized settings.
2. **Difficulty in Unbiased Estimation of the Projection Operator**: Unlike NFGs, the projection operator $\Pi_{T\mathcal{U}_i}$ in cMG depends on other players' policies, making it difficult to construct unbiased estimators.
3. **No Convergence Guarantees**: PGL performs gradient descent on non-convex exploitability; it is only empirically effective without theoretical convergence rates.
4. **Scalability to be Verified**: Experiments are limited to small-scale tabular games (dozens of states) and have not yet been extended to deep RL settings.
5. **NE Computation Itself is PPAD-hard**: cMG strictly subsumes MG, meaning its computational complexity is at least as high. The annealing heuristic of PGL does not guarantee finding a global optimum.

## Related Work & Insights

- **Convex MDPs** (Zhang et al., 2020; Zahavy et al., 2021): The direct single-agent predecessor of this work.
- **Equilibrium Computation in Markov Games** (Fink 1964; Eibelshäuser & Poensgen 1919): The homotopy methods in sgamesolver inspired the temperature annealing in PGL.
- **RLHF and Markov Games** (Wu et al., 2025): LLM alignment has been formalized as Markov Games; cMG is expected to provide richer modeling tools.
- **Multi-Agent Exploration** (Zahavy et al., 2023): Discovered creative plays in Chess using convex MDPs, which this work generalizes to multi-agent equilibria.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Fills the theoretical gap from cMDP to multi-agent settings; the topological proof of pure-strategy NE existence is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Seven scenarios cover four types of applications, but all are small-scale tabular experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous theory and intuitive illustrations (especially the occupancy measure space visualization in Figure 1).
- Value: ⭐⭐⭐⭐⭐ — Opens up a new direction for MARL; the unified framework has broad potential applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Sample-Efficient Distributionally Robust Multi-Agent Reinforcement Learning via Online Interaction](../../ICLR2026/ai_safety/sample-efficient_distributionally_robust_multi-agent_reinforcement_learning_via_.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[ICLR 2026\] Expressiveness of Multi-Neuron Convex Relaxations in Neural Network Certification](../../ICLR2026/ai_safety/expressiveness_of_multi-neuron_convex_relaxations_in_neural_network_certificatio.md)
- [\[ICML 2025\] Collaborative Mean Estimation Among Heterogeneous Strategic Agents: Individual Rationality, Fairness, and Truthful Contribution](collaborative_mean_estimation_among_heterogeneous_strategic_agents_individual_ra.md)
- [\[ICML 2025\] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](adaptive_multi-prompt_contrastive_network_for_few-shot_out-of-distribution_detec.md)

</div>

<!-- RELATED:END -->
