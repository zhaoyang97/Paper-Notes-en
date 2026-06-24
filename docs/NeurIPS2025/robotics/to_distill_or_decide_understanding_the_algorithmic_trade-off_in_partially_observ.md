---
title: >-
  [Paper Note] To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning
description: >-
  [NeurIPS 2025][Robotics][Partial observability] Using a theoretical framework (perturbed Block MDP) and controlled locomotion experiments, this paper systematically investigates the algorithmic trade-off between **privileged expert distillation** and **standard RL** (without privileged information) in partially observable RL, finding that the trade-off is primarily governed by the stochasticity of latent state dynamics.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Partial observability"
  - "privileged information"
  - "expert distillation"
  - "Block MDP"
  - "algorithmic trade-off"
date: 2026-05-08
content_hash: fec20d568636fc3b
---

# To Distill or Decide? Understanding the Algorithmic Trade-off in Partially Observable Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.03207](https://arxiv.org/abs/2510.03207)  
**Code**: None  
**Area**: Reinforcement Learning / Partially Observable RL
**Keywords**: Partial observability, privileged information, expert distillation, Block MDP, algorithmic trade-off

## TL;DR

Using a theoretical framework (perturbed Block MDP) and controlled locomotion experiments, this paper systematically investigates the algorithmic trade-off between **privileged expert distillation** and **standard RL** (without privileged information) in partially observable RL, finding that the trade-off is primarily governed by the stochasticity of latent state dynamics.

## Background & Motivation

### Challenges in Partially Observable RL
Partially observable reinforcement learning (PORL) requires agents to make decisions based on incomplete observations, necessitating the learning of complex history-dependent policies. This setting is ubiquitous in practical applications such as robotic control and autonomous driving, where agents have access to only partial state information.

### Two Dominant Approaches

**Privileged Expert Distillation**:
- Leverages **complete latent state information** provided by the simulator during training to first train an expert policy with full state access
- Then distills the expert policy into a student policy that operates solely on partial observations
- Decouples "learning to perceive" from "learning to act"

**Standard RL (without privileged information)**:
- Trains policies directly from sequences of partial observations
- Theoretically capable of learning superior policies, but computationally less efficient
- Requires implicit integration of historical information

### Known Failure Modes of Distillation
Despite its computational efficiency, several works have documented failure modes of privileged expert distillation: when observations cannot uniquely determine the latent state, distilling the optimal latent-state policy may yield a poorly performing student policy.

## Method

### Overall Architecture

The paper adopts a dual-track approach combining **theoretical analysis and controlled experimentation**:

```
Theory: Perturbed Block MDP
  ├── Analyze performance of distillation methods
  ├── Analyze performance of standard RL
  └── Identify key parameter: stochasticity ε of latent state dynamics

Experiment: Simulated locomotion control tasks
  ├── Controllable levels of stochasticity
  ├── Distillation vs. standard RL comparison
  └── Validation of theoretical predictions
```

### Key Designs

#### Perturbed Block MDP
A perturbation parameter $\epsilon$ is introduced into the standard Block MDP:
- **Block MDP**: latent state $s$ generates observation $o$ via an emission function; multiple latent states may map to the same observation
- **Perturbation**: with probability $\epsilon$, the latent state transition is randomly perturbed, controlling the stochasticity of the dynamics
- $\epsilon = 0$: fully deterministic; distillation performs well
- Large $\epsilon$: high stochasticity; distillation may fail

#### Two Key Theoretical Concepts

**Approximate Decodability**:
- Measures the difficulty of recovering the latent state from an observation sequence
- When $\epsilon$ is small, the observation sequence provides sufficient information to approximately reconstruct the latent state
- Distillation methods are effective under this condition

**Belief Contraction**:
- Measures whether uncertainty about the latent state diminishes over time
- When $\epsilon$ is large, the belief distribution over latent states does not contract, as stochastic transitions introduce persistent uncertainty
- Theoretical guarantees for standard RL rely on this property

#### Finding 1: The Trade-off Is Governed by Stochasticity
- **Low stochasticity (small $\epsilon$)**: latent states are approximately decodable → distilling the optimal latent-state policy is effective → distillation outperforms standard RL
- **High stochasticity (large $\epsilon$)**: latent states are unreliable → distillation of the optimal latent-state policy may fail → standard RL is preferable

#### Finding 2: The Optimal Latent-State Policy Is Not Always the Best Distillation Target
Under moderate stochasticity, distilling a **suboptimal latent-state policy** (e.g., one that is smoother and more robust) yields a better partial-observation policy. This is because the optimal policy may rely on precise state information and becomes fragile under uncertainty when distilled.

### Loss & Training

- **Expert policy training**: standard RL (e.g., PPO) trained on the full state space
  $$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[\sum_t \gamma^t r_t \mid \text{full state}\right]$$
- **Distillation loss**: imitation learning objective (behavioral cloning)
  $$\mathcal{L}_{\text{distill}} = \mathbb{E}_{o_{1:t}} \left[D_{\text{KL}}(\pi^*(s_t) \| \pi_\theta(o_{1:t}))\right]$$
- **Standard RL**: directly optimizes over observation histories without access to any latent state information

## Key Experimental Results

### Main Results

Comparison of distillation and standard RL on simulated locomotion control tasks:

| Stochasticity $\epsilon$ | Expert Distillation (Return) | Standard RL (Return) | Gap | Better Method |
|--------------------------|------------------------------|----------------------|-----|---------------|
| 0.0 (deterministic) | **950** | 820 | +130 | Distillation |
| 0.1 | **890** | 845 | +45 | Distillation |
| 0.3 | 810 | **830** | −20 | Standard RL |
| 0.5 | 720 | **815** | −95 | Standard RL |
| 0.8 | 580 | **790** | −210 | Standard RL |

Comparison across distillation targets:

| Distillation Target | $\epsilon=0.1$ Return | $\epsilon=0.3$ Return | $\epsilon=0.5$ Return |
|---------------------|----------------------|----------------------|----------------------|
| Optimal latent-state policy | **890** | 810 | 720 |
| Suboptimal smooth policy | 870 | **840** | **760** |
| Random exploration policy | 780 | 790 | 740 |

### Ablation Study

Analysis of key factors:

| Factor | Effect at Low Stochasticity | Effect at High Stochasticity | Conclusion |
|--------|-----------------------------|------------------------------|------------|
| Longer history | Marginal improvement for distillation | Significant improvement for standard RL | Standard RL exploits longer histories more effectively |
| Increased observation noise | Distillation performance degrades | Both degrade | Distillation is more sensitive to noise |
| Larger network capacity | Marginal gain for distillation | Significant gain for standard RL | Standard RL requires greater capacity |
| More training samples | Distillation converges quickly | Standard RL converges slowly | Distillation is more sample-efficient |

### Key Findings

1. **Stochasticity is the critical dividing factor**: $\epsilon$, the stochasticity of latent state dynamics, is the key variable determining whether distillation outperforms standard RL or vice versa
2. **Theory and experiment align**: theoretical predictions from the perturbed Block MDP are precisely validated in controlled experiments
3. **Suboptimal experts may distill better**: this counterintuitive finding has important practical implications
4. **Standard RL demands more computation**: slower convergence and larger network capacity requirements, but ultimately superior performance under high stochasticity
5. **The two approaches are complementary**: neither is universally superior; the choice depends on the stochasticity characteristics of the problem

## Highlights & Insights

1. **Elegance and depth of the theoretical model**: the perturbed Block MDP captures the essential trade-off with a single parameter $\epsilon$
2. **Practical guidance**: provides clear criteria for when to use distillation versus standard RL
3. **Counterintuitive finding**: the optimal expert policy is not necessarily the optimal distillation target
4. **Bridging theory and practice**: approximate decodability and belief contraction provide rigorous theoretical explanations for empirical observations
5. **Pioneering problem formulation**: the first systematic study of this fundamental trade-off

## Limitations & Future Work

1. **Limited experimental environments**: validation is restricted to simulated locomotion tasks; real-robot experiments remain to be conducted
2. **Idealized nature of the perturbed Block MDP**: partial observability in real-world problems is considerably more complex
3. **No adaptive method proposed**: the paper does not introduce an algorithm that automatically selects the appropriate approach based on $\epsilon$
4. **Primarily analytical contribution**: the work is more diagnostic than prescriptive; new algorithms and methods remain to be developed
5. **Multi-agent settings not covered**: the use of privileged information in distributed RL is left for future work

## Related Work & Insights

- **Learning from Privileged Information**: Vapnik's framework provides the theoretical foundation for distillation-based methods
- **Asymmetric Actor-Critic**: Pinto et al. incorporate privileged information into the critic
- **Block MDP Theory**: Du et al. establish sample complexity theory for Block MDPs
- **Inspired Direction**: designing stochasticity-adaptive hybrid methods — distillation under low stochasticity, direct RL under high stochasticity

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic study of the fundamental distillation vs. RL trade-off
- **Theoretical Depth**: ⭐⭐⭐⭐⭐ — Perturbed Block MDP model and analysis are rigorous
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Controlled experiments thoroughly validate theoretical predictions
- **Practical Impact**: ⭐⭐⭐⭐ — Provides actionable guidance for sim-to-real transfer and robotic control
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theory and experiments are integrated in a highly natural manner

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[ICLR 2026\] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments](../../ICLR2026/robotics/partially_equivariant_reinforcement_learning_in_symmetry-breaking_environments.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[NeurIPS 2025\] Sample Complexity of Distributionally Robust Average-Reward Reinforcement Learning](sample_complexity_of_distributionally_robust_average-reward_reinforcement_learni.md)

</div>

<!-- RELATED:END -->
