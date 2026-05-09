---
title: >-
  [Paper Note] HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning
description: >-
  [AAAI2026][Reinforcement Learning][multi-agent RL] This paper proposes HCPO, an algorithm that enhances the expressiveness and exploration efficiency of multi-agent joint policies by introducing a conductor mechanism, constructing a Gaussian mixture model-like joint policy framework, and providing monotonic improvement guarantees for two-level policy updates.
tags:
  - AAAI2026
  - Reinforcement Learning
  - multi-agent RL
  - cooperative MARL
  - joint policy optimization
  - hierarchical framework
  - trust region
date: 2026-05-08
content_hash: ffa9031face53f9d
---

# HCPO: Hierarchical Conductor-Based Policy Optimization in Multi-Agent Reinforcement Learning

**Conference**: AAAI2026
**arXiv**: [2511.12123](https://arxiv.org/abs/2511.12123)
**Area**: Reinforcement Learning
**Keywords**: multi-agent RL, cooperative MARL, joint policy optimization, hierarchical framework, trust region

## TL;DR
This paper proposes HCPO, an algorithm that enhances the expressiveness and exploration efficiency of multi-agent joint policies by introducing a conductor mechanism, constructing a Gaussian mixture model-like joint policy framework, and providing monotonic improvement guarantees for two-level policy updates.

## Background & Motivation
Efficient exploration is critical for joint policy optimization in cooperative MARL. Existing CTDE paradigms (e.g., MAPPO, QMIX) suffer from two core problems:

- **Limited joint policy expressiveness**: Most methods assume the joint policy factorizes as a product of independent per-agent policies $\boldsymbol{\pi}(\boldsymbol{a}|s) = \prod_i \pi^i(a^i|s)$, restricting the expressive capacity of the policy space.
- **Uncoordinated independent exploration**: Agents explore independently, making it difficult to coordinate the discovery of high-value joint policies.
- **Limitations of existing hierarchical methods**: MAVEN relies on the monotonicity assumption of QMIX; COPA requires communication at execution time; skill discovery methods depend on variational inference.

## Method

### Conductor-Based Joint Policy Framework
Inspired by a coach directing players in a soccer match, a centralized conductor is introduced to provide a shared instruction $M$ to the entire team:

$$\boldsymbol{\pi}_{\text{mar}}(\boldsymbol{a}|s) \triangleq \mathbb{E}_{M \sim w(\cdot|s)} \boldsymbol{\pi}(\boldsymbol{a}|s, M)$$

- The conductor policy $w(\cdot|s)$ selects one of $K$ discrete instructions based on the global state.
- Given instruction $M$, the joint policy decomposes into a product of conditionally independent policies: $\boldsymbol{\pi}(\boldsymbol{a}|s,M) = \prod_{i=1}^N \pi^i(a^i|s,M)$.
- The overall structure forms a mixture policy analogous to a Gaussian mixture model, substantially enhancing expressiveness.

### Advantage Function Decomposition
The joint advantage function is decomposed into a conductor level and an agent level:

$$A_{\boldsymbol{\pi}_{\text{mar}}}(s, \boldsymbol{a}) = A_{\boldsymbol{\pi}_{\text{mar}}}(M|s) + A_{\boldsymbol{\pi}_{\text{mar}}}(\boldsymbol{a}|s, M)$$

- **Instruction advantage** $A(M|s)$: evaluates the relative quality of instruction $M$ over alternatives.
- **Joint action advantage** $A(\boldsymbol{a}|s,M)$: evaluates the quality of the joint action given the instruction.

### Two-Level Policy Update
1. **Conductor policy update**: Maximizes the instruction advantage subject to a KL divergence constraint:

$$w_{k+1} = \arg\max_{\bar{w}} \left[\mathbb{E}_{s,M\sim\bar{w}} A(M|s) - C \cdot D_{\text{KL}}^{\max}(w_k, \bar{w})\right]$$

2. **Sequential agent policy update**: For each instruction $M^j$, agents are updated one by one in a randomly permuted order, leveraging conditional advantage decomposition (Lemma 2) to decompose the joint advantage into a sum of per-agent marginal advantages.

### Decentralized Execution
- A centralized conductor is used during training; each agent is equipped with a local conductor $w^i(\cdot|o^i)$.
- The centralized conductor's policy is distilled into local conductors via a cross-entropy loss.
- At execution time, each agent relies solely on local observations and its local conductor, requiring no communication.

### Theoretical Guarantee
The paper proves $J(\boldsymbol{\pi}_{\text{mar},k+1}) \geq J(\boldsymbol{\pi}_{\text{mar},k})$, i.e., the joint policy performance improves monotonically, without relying on the monotonicity assumption of QMIX.

## Key Experimental Results

### SMAC (StarCraft II)
Evaluated on 5 maps (5 seeds), HCPO achieves a 90% win rate first on all maps with the lowest standard deviation.

### MA-MuJoCo
- HalfCheetah-v2-2×3: Final return is **23.42%** higher than the second-best algorithm HAA2C.
- t-SNE visualizations show that HCPO covers a broader state space in early training, confirming superior exploration.
- Walker2d-v2-6×1: Entropy analysis and average nearest-neighbor distance further validate HCPO's exploration advantage.

### MPE (Multi-Agent Particle Environment)
- Policy improvement is fastest during early training (0–2M steps), indicating high cooperative efficiency.
- HCPO demonstrates greater stability compared to HATRPO and A2PO.

### Ablation Study
- Removing the conductor leads to lower win rates and slower convergence.
- The number of instructions $K$ requires balancing performance against computational cost.
- A random conductor (uniform instruction output) yields significantly degraded performance, validating the effectiveness of the learned instruction distribution.
- The median return of the local conductor approaches that of the centralized conductor.

## Highlights & Insights
- **Mixture policy representation**: Modeling the joint policy as a mixture distribution breaks the expressiveness bottleneck of factorized independent policies.
- **Strict monotonic improvement guarantee**: Theoretical guarantees are established without relying on the QMIX monotonicity assumption.
- **Decentralized execution**: Policy distillation eliminates the need for communication at execution time.
- **Unified framework**: Trust region methods are organically integrated with sequential updates and the hierarchical mechanism.

## Limitations & Future Work
- Applicable only to on-policy algorithms, limiting sample efficiency; the authors plan to incorporate off-policy methods in future work.
- The instruction space is discrete ($K$ instructions); continuous instruction spaces remain unexplored.
- Conductor distillation introduces additional training overhead.

## Rating
- Novelty: ⭐⭐⭐⭐ — The conductor-based mixture policy framework is novel in MARL, with complete theoretical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Full coverage across three major benchmarks (SMAC/MA-MuJoCo/MPE) with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and rigorous theoretical derivations, though notation is dense.
- Value: ⭐⭐⭐⭐ — Offers new perspectives on policy expressiveness and coordinated exploration in MARL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Behaviour Policy Optimization: Provably Lower Variance Return Estimates for Off-Policy Reinforcement Learning](behaviour_policy_optimization_provably_lower_variance_return_estimates_for_off-p.md)
- [\[AAAI 2026\] DiffOP: Reinforcement Learning of Optimization-Based Control Policies via Implicit Policy Gradients](diffop_reinforcement_learning_of_optimization-based_control_policies_via_implici.md)
- [\[AAAI 2026\] Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction](thinker_training_llms_in_hierarchical_thinking_for_deep_search_via_multi-turn_in.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[AAAI 2026\] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)

</div>

<!-- RELATED:END -->
