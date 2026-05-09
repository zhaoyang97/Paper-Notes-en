---
title: >-
  [Paper Note] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization
description: >-
  [ICLR 2026][LLM Agent] This paper proposes EMPO2, an RL framework that combines an external memory module with hybrid on-policy/off-policy updates. By leveraging memory-guided exploration and knowledge distillation to internalize exploration gains into model parameters, EMPO2 achieves improvements of 128.6% and 11.3% over GRPO on ScienceWorld and WebShop, respectively.
tags:
  - ICLR 2026
  - LLM Agent
  - Reinforcement Learning
  - Exploration
  - External Memory
  - Hybrid Policy Optimization
date: 2026-05-08
content_hash: f48c87d065d9ff5a
---

# Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization

**Conference**: ICLR 2026
**arXiv**: [2602.23008](https://arxiv.org/abs/2602.23008)
**Code**: [https://github.com/agent-lightning/empo2](https://github.com/agent-lightning/empo2)
**Area**: Agent
**Keywords**: LLM Agent, Reinforcement Learning, Exploration, External Memory, Hybrid Policy Optimization

## TL;DR
This paper proposes EMPO2, an RL framework that combines an external memory module with hybrid on-policy/off-policy updates. By leveraging memory-guided exploration and knowledge distillation to internalize exploration gains into model parameters, EMPO2 achieves improvements of 128.6% and 11.3% over GRPO on ScienceWorld and WebShop, respectively.

## Background & Motivation

**State of the Field**: LLM agents learn decision-making in interactive environments via reinforcement learning (e.g., GRPO), but the core bottleneck is **insufficient exploration** — agents over-rely on pretrained knowledge and struggle to discover novel states that require active searching.

**Limitations of Prior Work**: (a) Pure parameter-update RL methods (e.g., GRPO) converge prematurely to suboptimal solutions on tasks requiring long-horizon exploration; (b) non-parametric methods (e.g., Reflexion) improve decisions through reflective memory but saturate quickly with fixed parameters and cannot continue improving; (c) offline RL and SFT methods depend on large amounts of expert trajectories or external resources such as GPT-4.

**Root Cause**: Parametric updates can internalize knowledge but lack exploration incentives; external memory can facilitate exploration but cannot expand intrinsic capabilities. Each approach has inherent limitations, and no unified framework exists.

**Paper Goals**: How can LLM agents autonomously explore novel environments during online RL while internalizing the experience gained through exploration into model parameters?

**Starting Point**: Non-parametric memory updates can bootstrap parametric updates — the agent first acquires high-quality trajectories via memory-guided exploration, then distills this knowledge into a memory-free policy through off-policy updates.

**Core Idea**: Self-generated memory tips serve as exploration scaffolding. Through hybrid on/off-policy updates, the exploration capability afforded by memory is progressively internalized into model weights.

## Method

### Overall Architecture
EMPO2 operates in two modes during rollout (with/without memory) and two modes during updates (on-policy/off-policy), yielding three learning configurations. Given task description $u$ and environment state $s_t$, the agent outputs natural language action $a_t$. The agent interacts with the environment over multiple steps to generate trajectories, then optimizes via GRPO-style policy gradients using reward signals.

### Key Designs

1. **Self-Generated Memory Module**:

    - Function: After each episode, the agent reflects on its trajectory using the current policy $\pi_\theta$ to generate tips stored in memory buffer $\mathcal{M}$.
    - Mechanism: $\text{tip}_i \sim \pi_\theta(s_t, u, \text{tip-generation prompt})$; tips are retrieved via cosine similarity, with up to 10 retrieved per step.
    - Design Motivation: Unlike Reflexion, the tips here are not an end in themselves but serve as intermediate exploration scaffolding to guide parametric updates.

2. **Dual-Mode Rollout**:

    - Function: During rollout, memory-augmented prompting is used with probability $p$, and standard prompting with probability $1-p$.
    - Mechanism: In memory mode, $a_{t+1} \sim \pi_\theta(\cdot | s_t, u, \text{tips}_t)$; in memory-free mode, $a_{t++1} \sim \pi_\theta(\cdot | s_t, u)$.
    - Design Motivation: Memory rollouts produce high-quality exploratory trajectories, while memory-free rollouts preserve the policy's independent reasoning capability.

3. **Hybrid On/Off-Policy Updates**:

    - Function: Memory-augmented trajectories undergo off-policy updates with probability $q$ (tips removed) or on-policy updates with probability $1-q$ (tips retained).
    - Mechanism: In off-policy mode, the conditional log-probability $\log\pi_\theta(a_t|s_t,u,\text{tips})$ used during rollout is replaced by $\log\pi_\theta(a_t|s_t,u)$, constituting **reward-guided knowledge distillation** — high-reward trajectories are reinforced ($\hat{A}_t > 0$) and low-reward ones suppressed, training the policy to reproduce memory-assisted behavior without memory at inference time.
    - Design Motivation: On-policy updates ensure stable learning; off-policy updates transfer memory-driven exploration capability into intrinsic model knowledge.

4. **Off-Policy Training Stabilization (Token Masking)**:

    - Function: Advantage terms are masked for tokens whose policy probability falls below threshold $\delta$.
    - Mechanism: An indicator function $\mathbf{1}_{\pi_\theta(a_t|s_t,u) \geq \delta}$ is incorporated into the loss.
    - Design Motivation: Low-probability tokens cause the importance sampling ratio $\rho$ to explode, leading to gradient NaNs; the masking mechanism effectively prevents training collapse.

5. **Intrinsic Rewards**:

    - Function: Additional rewards $r_{\text{intrinsic}} = 1/n$ are granted based on state novelty, where $n$ is the number of similar historical states.
    - Design Motivation: Encourages the agent to explore novel states even in the absence of extrinsic rewards, maintaining policy entropy.

### Loss & Training
A GRPO-based clipped surrogate loss with token masking and KL regularization:
$$\mathcal{L} = \mathbb{E}\left[\frac{1}{NT}\sum_{i,t}\min(\rho_\theta^{(i,t)} A_t^{(i)}, \text{clip}(\rho, 1\pm\epsilon) A_t^{(i)}) \cdot \mathbf{1}_{\pi_\theta \geq \delta}\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

## Key Experimental Results

### Main Results

**ScienceWorld** (19 tasks, Qwen2.5-7B-Instruct):

| Method | Avg. Score | vs GRPO |
|------|---------|---------|
| Naive (zero-shot) | -61.3 | - |
| Reflexion (non-parametric) | 17.1 | - |
| Retrospex (offline RL) | 33.8 | - |
| GRPO (online RL) | 33.2 | baseline |
| **EMPO2** | **75.9** | **+128.6%** |

**WebShop**:

| Method | Score | Success Rate |
|------|-------|-------------|
| GRPO | 79.3 | 66.1% |
| GiGPO w/o std | 86.2 | 75.2% |
| **EMPO2** | **88.3** | **76.9%** |

### Ablation Study

| Configuration | Performance | Notes |
|------|---------|------|
| Full EMPO2 | Highest | All three modes intact |
| w/o off-policy | Significant drop | Exploration gains cannot be internalized without knowledge distillation |
| w/o on-policy w/ memory | Drop | Reduced stability without memory-augmented on-policy updates |

### Key Findings
- EMPO2 achieves a perfect score of 100 on 7 of 19 ScienceWorld tasks, whereas GRPO peaks at 78.2.
- The most substantial gains appear on Electricity tasks (power-component: 15.1→94.3), as these tasks require the highest degree of exploration.
- In OOD experiments, EMPO2 adapts to new tasks with only a few memory-aided steps (average improvement 136%), while GRPO is unstable.
- The off-policy and on-policy-with-memory modes are complementary: the former handles knowledge distillation, the latter ensures stable learning.

## Highlights & Insights
- **Memory as Exploration Scaffolding**: Rather than relying on memory directly for inference, EMPO2 uses high-quality trajectories produced with memory to distill knowledge into model parameters via off-policy updates, eliminating the need for memory at test time. This "assist-then-internalize" paradigm is particularly elegant.
- **Token Masking for Off-Policy Stability**: A simple yet effective solution to the importance sampling ratio explosion problem in off-policy LLM training, transferable to other off-policy LLM training scenarios.
- **Few-Shot Task Transfer**: The trained model acquires a meta-capability of "exploring with memory," enabling rapid adaptation to novel tasks in just a few steps — suggesting that EMPO2 learns a general exploration strategy rather than task-specific patterns.

## Limitations & Future Work
- Validation is limited to Qwen2.5-7B; larger models and different architectures remain untested.
- Memory retrieval relies on simple cosine similarity; more advanced RAG mechanisms could yield further improvements.
- Evaluation is restricted to text-based interactive environments (ScienceWorld, WebShop); mathematical reasoning and code generation scenarios are not explored.
- Off-policy updates depend on importance sampling; alternative off-policy techniques (e.g., V-trace) could be investigated.

## Related Work & Insights
- **vs. Reflexion**: Reflexion performs only non-parametric updates (fixed weights + memory); EMPO2 unifies memory exploration with parametric learning, surpassing Reflexion's performance ceiling.
- **vs. GRPO/GiGPO**: These methods update parameters without memory-assisted exploration, leading to insufficient exploration on tasks that require discovering novel states.
- **vs. Knowledge Distillation**: Conventional distillation is offline (teacher→student); EMPO2 performs online self-distillation, where the teacher (memory-augmented policy) and student (memory-free policy) share parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of memory and hybrid policy optimization is novel, though individual components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two environments with ablation and OOD testing, though additional benchmarks would strengthen the evaluation.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, figures are informative, and mathematical formulations are rigorous.
- Value: ⭐⭐⭐⭐ Provides a practical and effective exploration-augmented RL training framework for LLM agents.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](harnessing_uncertainty_entropy-modulated_policy_gradients_for_long-horizon_llm_a.md)
- [\[NeurIPS 2025\] Group-in-Group Policy Optimization for LLM Agent Training](../../NeurIPS2025/llm_agent/groupingroup_policy_optimization_for_llm_agent_training.md)
- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](../../ACL2026/llm_agent/memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ICLR 2026\] PhyScensis: Physics-Augmented LLM Agents for Complex Physical Scene Arrangement](physcensis_physics-augmented_llm_agents_for_complex_physical_scene_arrangement.md)
- [\[ICLR 2026\] REMem: Reasoning with Episodic Memory in Language Agents](remem_reasoning_with_episodic_memory_in_language_agent.md)

<!-- RELATED:END -->
