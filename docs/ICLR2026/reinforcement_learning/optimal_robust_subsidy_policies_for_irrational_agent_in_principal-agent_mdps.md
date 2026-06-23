---
title: >-
  [Paper Note] Optimal Robust Subsidy Policies for Irrational Agent in Principal-Agent MDPs
description: >-
  [ICLR 2026][Reinforcement Learning][Paper Note] This paper investigates how a principal can design subsidies within an MDP framework to guide a **potentially partially irrational** agent. It proves that when the agent is "globally $\epsilon$-incentive compatible," the seemingly complex bi-level minimax problem can be equivalently reduced to **one-dimensional concave
tags:
  - ICLR 2026
  - Reinforcement Learning
date: 2026-05-08
content_hash: 14ce3f42376aa506
---
# Optimal Robust Subsidy Policies for Irrational Agent in Principal-Agent MDPs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZO6Iwd3BZ7](https://openreview.net/forum?id=ZO6Iwd3BZ7)  
**Code**: To be confirmed (Theoretical paper, no open-source code)  
**Area**: Reinforcement Learning / Mechanism Design / Principal-Agent MDP  
**Keywords**: Principal-agent problem, subsidy mechanism, bounded rationality, incentive compatibility, robust optimization

## TL;DR
This paper investigates how a principal can design subsidies within an MDP framework to guide a **potentially partially irrational** agent. It proves that when the agent is "globally $\epsilon$-incentive compatible," the seemingly complex bi-level minimax problem can be equivalently reduced to **one-dimensional concave optimization**. Conversely, when incentive compatibility constraints are refined to a "per-state" basis, the problem either leads to non-Markovian policies or becomes NP-hard.

## Background & Motivation
**Background**: The principal-agent problem (often modeled as a Stackelberg game) is a core paradigm in economics and governance—where a government uses tax incentives or subsidies to guide individual behavior toward socially beneficial outcomes, yet market participants ultimately maximize their own private utility. In machine learning, RLHF for aligning LLMs shares this structure: the principal (designer) wants the agent's (model's) behavior to align with social values but cannot directly control the agent's decisions. Extending this to MDPs yields a sequential version where the "principal issues state-action subsidies, the agent selects a policy accordingly, and the principal seeks to maximize their own return."

**Limitations of Prior Work**: Previous works embedding the principal-agent problem into MDPs (contract-based models, reward shaping) almost exclusively assume **perfect rationality**—the agent will always select the policy that maximizes its own cumulative return (including subsidies). However, in reality, agents often deviate from perfect rationality: humans have limited cognition, incomplete information, and behavioral biases; RL approximation algorithms may yield sub-optimal policies due to insufficient exploration or limited compute. Once the agent can "deviate," the principal's carefully designed subsidies may fail completely.

**Key Challenge**: The fundamental difficulty the principal faces is "how to shape the agent's behavior under the premise that the principal cannot directly control the agent and the agent's response is unknown (potentially worst-case)." The perfect rationality assumption makes the problem solvable but not robust; relaxing this assumption expands the agent's response set, where optimal responses might be stochastic or even history-dependent, making the principal's optimization a difficult bi-level game.

**Goal**: Design a **robust subsidy mechanism** that maximizes the principal's cumulative expected return under the worst-case agent response. The analysis is broken down into three progressive agent models: perfectly rational agent (baseline), global $\epsilon$-IC agent, and per-state $\epsilon$-IC agent.

**Key Insight**: The authors formalize "bounded rationality" as an **incentive compatibility tolerance $\epsilon$**—the agent is willing to accept any policy whose cumulative utility falls within $\epsilon$ of its own optimal value. Robustness is characterized via minimax (principal vs. worst-case agent response). The insight is that $\epsilon$-IC encapsulates "irrationality" into a quantifiable relaxation set, which can potentially be tamed using duality and convex analysis.

**Core Idea**: The principal's robust subsidy design is formulated as $\max_{\Delta r}\min_{\pi}V_P^{\pi,\Delta r}$. It is proven that under global $\epsilon$-IC, this can be **reduced to a one-dimensional concave optimization**; meanwhile, it is revealed that two natural definitions of per-state IC both destroy tractability.

## Method

### Overall Architecture
This is a **theoretical paper**. The framework is not a data pipeline but a logical chain of "problem modeling + incremental analysis of rationality levels." The principal-agent interaction is modeled as a finite-horizon, time-inhomogeneous MDP: $M=\langle S,A,H,P,r_P,r_A,\hat s\rangle$, where each $(s,a,h)$ provides a reward $r_P$ to the principal and $r_A$ to the agent. The principal commits to a non-negative subsidy mechanism $\Delta r:S\times A\times H\to\mathbb R_{\ge0}$. After subsidies, effective rewards become $r_P^{\Delta r}=r_P-\Delta r$ and $r_A^{\Delta r}=r_A+\Delta r$ (subsidies are transfer payments from the principal to the agent). After observing $\Delta r$, the agent selects a policy from a feasible set $\Pi(\Delta r)$ determined by its rationality level. The principal's robust goal is to counter the worst-case agent response:

$$\mathrm{OPT}\triangleq\max_{\Delta r\in R_\Delta}\ \min_{\pi\in\Pi(\Delta r)}\ V_P^{\pi,\Delta r}(\hat s,h{=}0)$$

The core methodology involves analyzing three different definitions of $\Pi(\Delta r)$ corresponding to three rationality models, evaluating the solvability and optimal subsidy structure of the principal's minimax problem for each. A central quantity throughout is **social welfare** $r_{sw}\triangleq r_P+r_A$, which is invariant to subsidies $\Delta r$. Thus, the "social welfare maximizing action" $a\in\arg\max_{a'}Q^*_{sw}(s,a',h)$ serves as the natural benchmark for incentive alignment.

### Key Designs

**1. Modeling robust subsidy design as a minimax principal-agent MDP with social welfare as the alignment benchmark**

The first contribution is the modeling itself. It addresses the limitation of prior work that either assumed perfect rationality or treated subsidy costs as external constraints, failing to characterize worst-case guarantees when the "agent deviates." The authors define the principal's return as $V_P^{\pi,\Delta r}$ and the agent's adversarial response as a policy in the feasible set that minimizes the principal's return: $\pi^{\Delta r}\in\arg\min_{\pi\in\Pi(\Delta r)}V_P^{\pi,\Delta r}(\hat s,0)$. Standard Bellman operators $(T_uV)(s,a,h)=u(s,a,h)+\sum_{s'}P(s'|s,a,h)V(s',h{+}1)$ are instantiated for $r_P$, $r_A$, and $r_{sw}$. The key insight is that since social welfare $r_{sw}=r_P+r_A$ is independent of subsidies, the "maximum reachable social welfare" provides a natural upper bound for the principal's return.

**2. Perfectly rational agent (baseline): Optimal subsidy targets social welfare maximizing actions, and Principal Profit = Max Social Welfare − Agent's No-Subsidy Self-Interest Value**

As a baseline, the authors analyze the perfectly rational agent ($\pi \in \Pi_0(\Delta r)$ implies $V_A^{\pi,\Delta r}(\hat s,0)\ge V_A^{\Delta r}(\hat s,0)$). Theorem 3.1 shows the principal's optimal return is exactly:

$$V^*_{sw}(\hat s,h{=}0)-V_A^{\Delta r=0}(\hat s,h{=}0),$$

meaning the max reachable social welfare minus what the agent could get on its own without subsidies. The intuition is that the total value cannot exceed the social welfare bound, and the agent won't accept anything worse than its "independent value." One optimal subsidy is $\Delta r^*(s,a,h)=V_A^{\Delta r=0}(s,h)-Q_A^{\Delta r=0}(s,a,h)$, which **flattens** the agent's adjusted Q-values across actions, making them indifferent, and then uses a tie-breaking rule to push the agent toward the principal's preferred actions. Proposition 3.2 further shows it suffices to subsidize only **social-welfare-maximizing actions** ($\Delta r_{sw}$).

**3. Global $\epsilon$-IC agent: Reducing bi-level minimax to 1D concave optimization (Core Theorem)**

This is the technical core. A global $\epsilon$-IC agent (Definition 4.1) only requires that the **cumulative reward loss over the full horizon** does not exceed $\epsilon$: $V_A^{\pi,\Delta r}(\hat s,0)\ge V_A^{\Delta r}(\hat s,0)-\epsilon$. Unlike perfect rationality, this agent's worst-case response can be a **stochastic policy**, making the principal's problem a non-trivial bi-level program. Direct solution faces two obstacles: the agent's feasible set $M(\Delta r)$ (rewritten as an occupancy measure $\mu$ constraint) varies with $\Delta r$, and the function $f(\Delta r)=\min_\mu\sum\mu(r_P-\Delta r)$ is **not concave** in $\Delta r$.

The solution involves: treating the inner problem as a Linear Program (LP) over $\mu$, introducing dual variables ($\alpha\ge0$ for the $\epsilon$-IC constraint, $V$ for transitions), and **swapping the order of $\max_{\Delta r}$ and $\max_{\alpha,V}$**. For fixed $\alpha,V$, the objective is monotonic in $\Delta r$, allowing $\Delta r$ to be determined by the constraint boundary. Substituting back and letting $x=\frac{\alpha}{1+\alpha}\in[0,1)$, the problem collapses into the 1D concave optimization of Theorem 4.1:

$$\max_{x\in[0,1)}F(x)=x\,V^*_{sw}(\hat s,0)-V^*_x(\hat s,0)-\frac{x}{1-x}\epsilon,$$

where $V^*_x(s,h)\triangleq\max_\pi\{xV^\pi_{sw}(s,h)-V_P^{\pi,\Delta r=0}(s,h)\}$. Since $F$ is a max over linear functions of $x$, it is **concave** on $[0,1)$ and solvable via first-order methods. The optimal subsidy retains the $V-Q$ structure. Remarking that as $\epsilon \to 0$ and $x^* \to 1$, the result recovers the perfectly rational case.

**4. Per-state $\epsilon$-IC agent: Impossibility results for two natural definitions**

The third model tightens IC from "cumulative" to "local to each state." Two formalizations are shown to be intractable. **Value-consistent per-state $\epsilon$-IC** (Definition 5.1) requires $V_A^{\pi,\Delta r}(s,h)\ge V_A^{\Delta r}(s,h)-\epsilon$ for all $s,h$. The issue is that the agent's worst-case response can be **non-Markovian**, significantly lowering the principal's return. **Greedy per-state $\epsilon$-IC** (Definition 5.2) uses a look-ahead greedy constraint to ensure Markovian responses, but Theorem 5.1 proves that computing the principal's optimal subsidy in this case is **NP-hard**.

### Loss & Training
While there is no training process, the paper provides a **social welfare loss bound** under optimal subsidies. The social welfare gap $\delta_{sw}$ is the difference between max reachable welfare and welfare achieved under $\Delta r^*$. Proposition 4.3 proves that for a given $\epsilon$ and optimal $x^* \in (0,1)$, $\delta_{sw} = \frac{\epsilon}{1-x^*}$ and is bounded by $O(\sqrt{\epsilon})$.

## Key Experimental Results

### Main Results: Solvability and Optimal Subsidy Structure

| Agent Model | Feasible Set Constraint | Worst-case Response | Principal Problem Solvability | Optimal Subsidy Structure |
| :--- | :--- | :--- | :--- | :--- |
| Perfectly Rational (Baseline) | $V_A^{\pi,\Delta r}\ge V_A^{\Delta r}$ | Deterministic | Closed-form (Thm 3.1) | Subsidize SW-max actions; Profit $=V^*_{sw}-V_A^{\Delta r=0}$ |
| Global $\epsilon$-IC | Cumulative loss $\le\epsilon$ | Stochastic (Mixture involving $\pi_{sw}$) | **1D Concave Optimization** (Thm 4.1) | $\Delta r^*=V^*_{x^*}-Q^*_{x^*}$; concentrated on SW-max actions |
| Per-state $\epsilon$-IC (Value-consistent) | $V_A^{\pi,\Delta r}(s,h)\ge V_A^{\Delta r}(s,h)-\epsilon$ $\forall s,h$ | **Non-Markovian** | Intractable / Not polynomial size | Non-Markovian subsidies can achieve higher values |
| Per-state $\epsilon$-IC (Greedy) | Look-ahead greedy $\epsilon$-IC | Polynomial size | **NP-hard** (Thm 5.1) | — |

### Key Findings
- **Dimensionality reduction as a core contribution**: The bi-level minimax under global $\epsilon$-IC, which is initially neither a standard minimax nor concave, is reduced to a single-variable concave function on $[0,1)$ through dual LP transformation and variable substitution.
- **Structural stability of optimal subsidies**: In both perfectly rational and global $\epsilon$-IC cases, optimal subsidies focus on social-welfare-maximizing actions and follow a $V-Q$ structure.
- **The "granularity" of rationality determines difficulty**: Moving from global to per-state IC causes a phase transition from "tractable" to "intractable" (either non-Markovian or NP-hard).

## Highlights & Insights
- **Turning non-concave bi-level problems into 1D concave optimization**: The technique of dualizing the inner occupancy measure LP and then swapping optimization orders is a reusable pattern for robust mechanism design where subsidies couple with the feasible set.
- **Social welfare invariance as a lever**: Since $r_{sw} = r_P + r_A$ is independent of $\Delta r$, max social welfare serves as a fixed anchor for all theoretical results.
- **Clarification of solvability boundaries**: The negative results for per-state IC provide a clear map of where "bounded rationality" modeling remains computationally practical.

## Limitations & Future Work
- **Ours assumes prior knowledge**: The principal is assumed to know the agent's reward $r_A$ and tolerance $\epsilon$, which is often unrealistic. The authors suggest "learning-based settings" (principal learns $r_A$ or $\epsilon$ through interaction) as future work.
- **Per-state IC impracticality**: Due to NP-hardness and non-Markovian issues, accurate per-state "step-by-step" rationality modeling lacks a practical algorithm within this framework.
- **Theoretical focus**: There are no numerical experiments or real-world RLHF validation. It remains to be seen how well $\epsilon$-IC approximates actual human biases or approximate RL training behavior.

## Related Work & Insights
- **vs. Contract-based models**: Unlike existing works that assume perfect rationality and often find optimal contracts to be NP-hard or history-dependent, this work makes irrationality a first-class citizen and finds that global $\epsilon$-IC remains tractable.
- **vs. Reward shaping / Reward design**: Previous reward shaping works often treat subsidy costs as external constraints (e.g., fixed budget), leading to NP-hardness. By integrating the subsidy cost directly into the principal's utility (objective = reward - subsidy), this paper enables the reduction to concave optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically introducing $\epsilon$-IC to principal-agent MDPs and providing a 1D reduction is highly original.
- Experimental Thoroughness: ⭐⭐⭐ Comprehensive for a theory paper, but lacks numerical/empirical validation.
- Writing Quality: ⭐⭐⭐⭐ Logical progression through the three models is very clear.
- Value: ⭐⭐⭐⭐ Provides a solvability map for robust incentive design under irrationality, relevant for RLHF and mechanism design.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICLR 2026\] Potentially Optimal Joint Actions Recognition for Cooperative Multi-Agent Reinforcement Learning](potentially_optimal_joint_actions_recognition_for_cooperative_multi-agent_reinfo.md)
- [\[ICLR 2026\] Inter-Agent Relative Representations for Multi-Agent Option Discovery](inter-agent_relative_representations_for_multi-agent_option_discovery.md)
- [\[ICML 2025\] Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals](../../ICML2025/reinforcement_learning/learning_to_incentivize_in_repeated_principal-agent_problems_with_adversarial_ag.md)
- [\[ICLR 2026\] Multi-Agent Guided Policy Optimization](multi-agent_guided_policy_optimization.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](../../AAAI2026/reinforcement_learning/explaining_decentralized_multi-agent_reinforcement_learning_policies.md)

</div>

<!-- RELATED:END -->
