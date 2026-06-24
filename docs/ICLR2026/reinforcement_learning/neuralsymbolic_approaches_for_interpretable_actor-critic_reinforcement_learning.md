---
title: >-
  [Paper Note] Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Actor-Critic] NSAC replaces the black-box actor in A2C with "additive rule ensembles." It uses a neural network critic for value estimation, while a set of IF-THEN rules directly handles decision-making. Rules are learned online via policy gradients and Orthogonal Gradient Boosting (OGB), achieving performance comparable to black-box methods like DQN, PPO, and A2C while maintaining intrinsic interpretability.
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Actor-Critic"
  - "Rule Ensembles"
  - "Interpretability"
  - "A2C"
  - "Orthogonal Gradient Boosting"
  - "Neuro-Symbolic"
date: 2026-05-08
content_hash: 06b7ed6d932795a7
---

# Neural+Symbolic Approaches for Interpretable Actor-Critic Reinforcement Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0XIsA0PxJM](https://openreview.net/forum?id=0XIsA0PxJM)  
**Code**: TBD  
**Area**: Interpretable Reinforcement Learning / Neuro-Symbolic  
**Keywords**: Actor-Critic, Rule Ensembles, Interpretability, A2C, Orthogonal Gradient Boosting, Neuro-Symbolic  

## TL;DR
NSAC replaces the black-box actor in A2C with "additive rule ensembles." It uses a neural network critic for value estimation, while a set of IF-THEN rules directly handles decision-making. Rules are learned online via policy gradients and Orthogonal Gradient Boosting (OGB), achieving performance comparable to black-box methods like DQN, PPO, and A2C while maintaining intrinsic interpretability.

## Background & Motivation
- **Background**: Actor-critic methods based on neural networks (A2C, PPO) perform excellently in high-dimensional state-action spaces due to their powerful function approximation capabilities. However, the actor is essentially a black box, failing to reveal "why a certain action was chosen." This deficiency hinders deployment in scenarios requiring transparency and compliance, such as healthcare, finance, and law.
- **Limitations of Prior Work**: Symbolic RL attempts to provide transparent decision-making but generally face three issues: (i) **Dependence on pre-defined knowledge** (expert rules, pre-trained teachers), making them fail when switching domains; (ii) **Distortion in post-hoc explanations**: distilling decision trees/rules from a black-box teacher involves a size-fidelity trade-off, and post-hoc explainers (like SHAP) indicate correlation rather than causality, where minor model perturbations can flip results; (iii) **Overly complex expressions**: symbolic regression/program synthesis produce expressions with trigonometric functions, logarithms, and program syntax that are difficult for humans to simulate.
- **Key Challenge**: It is difficult to balance the **scalability/adaptability** of neural networks with the **transparency/traceability** of symbolic systems—pure symbolic methods struggle with complexity, while pure neural methods lack interpretability.
- **Goal**: Construct an actor-critic framework where the actor consists directly of human-readable rules and is **learned directly** from environmental interactions (without pre-defined knowledge or post-hoc distillation), all while maintaining performance comparable to black-box RL.
- **Core Idea**: **[Neural Critic + Symbolic Actor]** assigns the neural network to handle "computationally intensive but explanation-free" value estimation, while additive rule ensembles handle "explanation-required" decision-making; **[Additive instead of Hierarchical]** uses flat rule ensembles instead of decision trees to bypass tree scalability bottlenecks; **[Online Rule Mining via Orthogonal Gradient Boosting]** uses OGB to automatically discover rule conditions using only basic feature comparisons.

## Method

### Overall Architecture
NSAC maintains the A2C actor-critic feedback loop: the critic remains a neural network $V_\phi(s)$, updated by minimizing TD error; the actor is replaced by "one additive rule ensemble per action." Each action $a$ corresponds to an ensemble $f_a(s)=\sum_{j=1}^{k} w_{a,j}q_{a,j}(s)$, which directly predicts the advantage of that action, followed by a softmax to obtain the policy $\pi(a|s)=\frac{\exp(f_a(s))}{\sum_{a'}\exp(f_{a'}(s))}$. In each step, the advantage $A_t$ calculated by the critic serves as the learning signal. Policy gradients are applied to both selected and unselected actions, and rule sets are iteratively refreshed through "rule replacement + OGB for new rules + fully-corrective weight recalculation," ensuring the symbolic policy reflects real decisions while remaining compact.

```mermaid
flowchart LR
    S[State s] --> C[Neural Critic V_\phi]
    S --> A[Symbolic Actor: Additive rule ensembles per action f_a]
    A -->|softmax| P[Policy \pi(a|s)]
    P --> ACT[Execute action a]
    ACT --> ENV[Environment]
    ENV -->|r, s'| TD[TD Error \delta = r + \gamma V_{s'} - V_s]
    C --> TD
    TD -->|Update| C
    TD -->|Advantage A_t| RU[Rule replacement + OGB rule discovery + Fully corrective weights]
    RU --> A
```

### Key Designs

**1. Additive rule ensembles as actor: Replacing neural policy heads with IF-THEN rules to bypass decision tree scalability bottlenecks.** Unlike the hierarchical structure of decision trees (which become hard to read as they grow), NSAC maintains a **flat additive rule ensemble** for each action. Each rule is a conjunction of several Boolean propositions $q_i(x)=\prod_{j=1}^{c_i}p_{i,j}(x)$ (where each $p_{i,j}$ is a threshold comparison $\mathbb{I}[s\cdot x^{(j)}\le s\cdot x_l^{(j)}]$), and the total output is a linear combination of rules weighted by $w_i$. Since each rule contributes equally and can be read independently as "IF condition THEN weight," the model naturally satisfies the three interpretability criteria proposed by Murdoch et al.: simulatability, modularity, and low complexity.

**2. Policy gradient updates for rules within the A2C framework: Treating $(w,q)$ as differentiable parameters with branching based on action selection.** The actor loss is a regularized policy gradient objective $\nabla L_\lambda(w,q)=-\mathbb{E}[\nabla_{w,q}\log\pi(a_t|s_t,w,q)A_t]+\lambda\nabla_{w,q}\|w\|_2^2$. By merging $\theta=(w;q)$, second-order gradients are derived and distinguished for two cases: when the ensemble $f_a$ corresponds to the action actually executed ($a=a_t$), $\nabla L = \mathbb{E}[-\nabla A(s_t,a_t)(1-\pi(a_t|s))]$; when it does not ($a\ne a_t$), $\nabla L = \mathbb{E}[\nabla A(s_t,a_t)\pi(a_t|s)]$. This branching—"pushing selected actions toward the advantage and suppressing others"—aligns rule ensemble updates with actual sampled actions. The critic minimizes TD error as usual with $L_V(\phi)=\mathbb{E}[(R_t+\gamma V(s_{t+1})-V_\phi(s_t))^2]$. The paper also provides theoretical proof of NSAC's convergence to local optima.

**3. Orthogonal Gradient Boosting (OGB) for online rule discovery + fully-corrective weight recalculation: Ensuring rule generalization and optimal weighting.** Instead of simply stacking rules, each iteration uses the OGB objective $\text{obj}_{\text{ogb}}(q)=|g_\perp^\top q|/(\|q_\perp\|+\epsilon)$ to find new rules in the candidate dimensions, where $g_\perp=g-BB^\top g$ is the projection of the gradient onto the orthogonal complement of existing rules. This forces new rules to provide "new direction" information rather than redundancy, selecting rules that are more general with better risk-complexity trade-offs. Accompanied by "rule replacement," the lowest-weight rules are deleted, and all weights are recalculated using fully-corrective optimization (solving a convex problem on the $k\times k$ Gram matrix $Q^\top Q$). The cost is $O(d^2nk)$ per rule, $k$ times more than standard gradient boosting ($O(d^2n)$), but controllable in interpretable scenarios with small rule sets.

## Key Experimental Results

### Main Results (Mean Return ± SD, 10 random seeds)
Compared black-box RL (Q-table/DQN/A2C/PPO/SDSAC/SACBBF/Rainbow) with symbolic methods (SYMPOL/πaffine-D/D-SDT) on 5 classic Gym environments + the Sinergym HVAC building control benchmark.

| Environment | DQN | A2C | PPO | Rainbow | SYMPOL (Best Symbolic) | **NSAC** |
|---|---|---|---|---|---|---|
| MountainCar-v0 | -135.07 | -157.51 | -150.40 | -137.76 | -200 | **-132.25** |
| Acrobot-v1 | -112.68 | -98.93 | -82.63 | -89.75 | -80.02 | -87.71 |
| CartPole-v1 | 161.00 | 453.51 | 498.73 | 498.53 | 500 | 499.14 |
| Blackjack-v1 | -0.06 | -0.07 | -0.06 | -0.06 | -0.06 | -0.06 |
| Postman | 31.91 | 24.23 | 34.35 | 34.67 | 25.34 | 27.14 |
| HVAC-1Zone | -1445367 | -1334562 | -1865276 | -1465732 | -1478783 | **-1251321** |
| HVAC-5Zone | -1876253 | -1984843 | -1676288 | -1602142 | -1586352 | **-1463601** |

- NSAC **achieved best-in-field results** on MountainCar and both HVAC tasks, while matching the strongest black-box baselines in other tasks.
- Symbolic baselines (SYMPOL/πaffine-D/D-SDT) compete with neural methods in simple tasks but **collapse at scale** (e.g., πaffine-D drops to -425 on Acrobot and only 109 on CartPole), whereas NSAC remains stable across all environments and significantly leads all symbolic methods on the difficult HVAC task.

### Ablation Study (CartPole-v1, Number of Rules × Warm Start)

| Configuration | Observation |
|---|---|
| Rules per action: 5/10/**12**/20/30/40/50 | **12 rules + warm start** yielded highest return (≈500); <10 rules lacks coverage, while too many overfit to noise. |
| Warm start vs. None | Warm start is more critical with fewer rules, allowing faster convergence and avoidance of poor local optima. |

### Key Findings
- Rule count and performance are non-monotonic: A "complexity-expressivity" sweet spot exists (12 rules/action), confirming that interpretable models are not necessarily better when larger.
- Learned rules can be directly read (e.g., "IF vel < 4 and pos > 8 → w=0.32"), supporting qualitative strategy diagnosis (RQ3).

## Highlights & Insights
- **Clear Philosophy of Division**: Assigning "computation-heavy" value estimation to neural networks and "explanation-heavy" decision-making to symbolic rules avoids the dilemma of "pure symbolic complexity vs. pure neural opacity," providing a clean implementation of neuro-symbolic RL.
- **Direct Rule Learning from Environment**: Rules are not distilled from black-box teachers and do not require pre-defined knowledge; thus, they reflect the **actual decision process** rather than a post-hoc approximation—directly addressing the weakness of post-hoc explainers like SHAP, which "only flag correlation and flip under perturbation."
- **Formal Criteria for Interpretability**: Rule policies correspond to simulatability/modularity/low-complexity criteria rather than just "looking readable," backed by convergence proofs.
- **Real HVAC Benchmark**: Achieving the best results in high-dimensional, strongly coupled multi-zone building control with comfort constraints demonstrates that additive rule ensembles can handle more than just toy tasks.

## Limitations & Future Work
- **Discrete Actions + Softmax**: The framework is based on one rule ensemble per action with softmax; how to extend this to continuous action spaces is not fully detailed.
- **Computational Overhead**: OGB is $k$ times more expensive than standard gradient boosting ($O(d^2nk)$), and fully-corrective weights cost $O(k^2n)$, which is only controllable with small rule sets; costs rise as rule count increases.
- **Restricted Rule Conditions**: Propositions are conjunctions of threshold comparisons, which may have limited expressivity for tasks requiring non-linear or relational conditions.
- **Lack of Human-Centric Validation**: While the paper argues for readability using formal criteria, there are no human-subject experiments to prove that "humans actually trust/use it more."
- **Environment Scale**: Still biased toward classic control and a single real-world benchmark, lacking testing in large-scale or visual input domains.

## Related Work & Insights
- **Tree-Policy RL** (Native tree / Distillation tree / D-SDT): Interpretable but faces size-fidelity and scalability issues; NSAC swaps "hierarchical trees" for "additive flat rules."
- **Symbolic Policy Discovery** (Neural-guided DSP, Genetic Programming, Program Synthesis πaffine): Avoids pre-defined knowledge but outputs complex mathematical/program expressions; NSAC sticks to "basic threshold comparisons + IF-THEN" for human readability.
- **Additive Rule Ensembles / OGB** (Friedman-Popescu, Yang et al. 2024): Ours is the first to systematically embed rule ensembles from supervised learning into the policy gradient loop of actor-critic.
- **Insight**: Neuro-symbolic approaches don't need to be "end-to-end differentiable for everything." Dividing responsibilities based on "whether an explanation is needed" might be a more pragmatic route for trustworthy RL.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of additive rule ensembles + OGB within A2C policy gradients for an endogenous interpretable actor is new and includes convergence proofs and interpretability criteria; however, components (A2C, rule ensembles, OGB) are existing, making it a clever integration rather than a brand-new mechanism.
- **Experimental Thoroughness**: ⭐⭐⭐ — Covers 5 classic environments + real HVAC, compares black-box vs. symbolic baselines, uses 10 seeds, and includes rule count/warm start ablations; however, environments are relatively small, and continuous/visual domains are missing, with no human-centric validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivations and critiques of "post-hoc explanation distortion" are sharp, derivations are complete, and graphical comparisons are intuitive.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for high-risk RL scenarios requiring transparent decisions: "no performance loss + readable rules + direct learning," with convincing leadership in HVAC tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Finite-Time Analysis of Actor-Critic Methods with Deep Neural Network Approximation](finite-time_analysis_of_actor-critic_methods_with_deep_neural_network_approximat.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] Chunking the Critic: A Transformer-based Soft Actor-Critic with N-Step Returns](chunking_the_critic_a_transformer-based_soft_actor-critic_with_n-step_returns.md)
- [\[ICLR 2026\] Flowing Through States: Neural ODE Regularization for Reinforcement Learning](flowing_through_states_neural_ode_regularization_for_reinforcement_learning.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)

</div>

<!-- RELATED:END -->
