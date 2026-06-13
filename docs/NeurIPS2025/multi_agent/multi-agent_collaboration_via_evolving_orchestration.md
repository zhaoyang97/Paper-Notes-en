---
title: >-
  [Paper Note] Multi-Agent Collaboration via Evolving Orchestration
description: >-
  [NeurIPS 2025][Multi-Agent][Multi-Agent Collaboration] This paper proposes a "Puppeteer" multi-agent collaboration paradigm in which a centralized orchestrator learns via RL to dynamically select which agent to activate…
tags:
  - "NeurIPS 2025"
  - "Multi-Agent"
  - "Multi-Agent Collaboration"
  - "Dynamic Orchestration"
  - "Reinforcement Learning"
  - "Topology Evolution"
  - "Collective Reasoning"
date: 2026-05-08
content_hash: d6ee76c67a4468b9
---

# Multi-Agent Collaboration via Evolving Orchestration

**Conference**: NeurIPS 2025
**arXiv**: [2505.19591](https://arxiv.org/abs/2505.19591)  
**Code**: [https://github.com/OpenBMB/ChatDev/tree/puppeteer](https://github.com/OpenBMB/ChatDev/tree/puppeteer)  
**Area**: Reinforcement Learning
**Keywords**: Multi-Agent Collaboration, Dynamic Orchestration, Reinforcement Learning, Topology Evolution, Collective Reasoning

## TL;DR

This paper proposes a "Puppeteer" multi-agent collaboration paradigm in which a centralized orchestrator learns via RL to dynamically select which agent to activate at each reasoning step. The approach simultaneously improves performance and efficiency on both closed-domain and open-domain tasks, and reveals that evolved topologies tend toward more compact cyclic structures.

## Background & Motivation

**Background**: LLM-based multi-agent systems (MAS) tackle complex problems by combining diverse models, reasoning strategies, and tools. Existing approaches such as ChatDev and MacNet rely on predefined static topologies (chain, tree, DAG, etc.).

**Limitations of Prior Work**: Static organizational structures cannot adapt to growing task complexity or increasing agent counts, leading to coordination overhead, redundant computation, and poor communication efficiency. For instance, a 50-node mesh-structured MAS may require 10 hours to generate a few hundred lines of code.

**Key Challenge**: How can collaboration effectiveness be maximized while minimizing computational overhead? Static topologies are inherently incapable of satisfying both objectives simultaneously.

**Goal**: Design a dynamic, evolvable multi-agent orchestration mechanism that selects the optimal agent activation sequence in real time based on task state, and continuously optimizes this selection via RL.

**Key Insight**: Drawing an analogy to puppetry—a puppeteer (orchestrator) operates behind the scenes, dynamically pulling strings (activating agents) according to the plot (task state), learning to tighten effective strings and cut unnecessary ones.

**Core Idea**: Multi-agent collaboration is formulated as a Markov decision process, in which a centralized orchestrator learns via REINFORCE to dynamically select agent activation sequences, jointly optimizing task quality and inference efficiency.

## Method

### Overall Architecture

The system comprises a set of heterogeneous agents $\mathcal{A} = \{(m, r, t)\}$ (model × reasoning strategy × toolset) and a centralized orchestrator $\pi_\theta$. At each timestep $t$, the orchestrator observes the global state $S_t$ and selects an agent $a_t$ to activate; the agent produces an output that updates the system state. This process continues until a termination condition is met. The outputs of all agents are aggregated to produce a final answer. The orchestrator is updated via REINFORCE using reward signals collected upon task completion.

### Key Designs

1. **Centralized Puppeteer**

    - **Function**: Replaces autonomous agent-to-agent selection with unified scheduling by a centralized orchestrator.
    - **Mechanism**: The orchestrator samples $a_t \sim \pi(S_t, \tau)$ based on the global state and task description to select the next agent to activate; agents are responsible only for executing inference, not for scheduling decisions.
    - **Design Motivation**: Decentralized agent-level partner selection introduces quadratic coordination overhead as agent count grows. Centralized orchestration decouples agent selection from agent-internal behavior, substantially improving scalability.

2. **Serialized Orchestration**

    - **Function**: "Unrolls" graph-structured collaboration topologies into temporal decision sequences.
    - **Mechanism**: Rather than searching the full topology space, the orchestration process is modeled as a sequential decision $a_0 \to a_1 \to \ldots \to a_T$. Crucially, sequences can be "folded" back into directed graphs (agents as nodes, orchestration order as edges), supporting cyclic structures and branching connections.
    - **Design Motivation**: Combinatorial explosion in the high-dimensional graph topology space makes exhaustive search intractable. Serialization ensures the Markov property $\mathbb{P}(a_{t+1}|S_0,...,S_{t+1}) = \mathbb{P}(a_{t+1}|S_{t+1})$, enabling direct optimization via RL.

3. **Adaptive Evolution via RL**

    - **Function**: Continuously optimizes the orchestrator policy through REINFORCE.
    - **Mechanism**: The reward balances quality and efficiency: $R_t = r - \lambda \cdot C_T$ (terminal) or $\gamma \cdot R_{t+1} - \lambda \cdot C_t$ (intermediate), where $C_t = F \cdot \log(1 + t/\varphi)$ is a step-dependent cost penalty. $\lambda$ controls the accuracy–efficiency trade-off.
    - **Design Motivation**: Without cost penalization, the system degenerates into a conventional large-scale collaboration framework (potentially higher performance but at extreme computational cost). Through RL, the orchestrator gradually learns to achieve goals with more compact agent sequences.

4. **Emergent Topology Evolution**

    - **Function**: Analyzes structural changes in the collaboration topology during orchestrator training.
    - **Mechanism**: Two key trends are identified—**compactification** (graph density increases, communication concentrates on a small number of hub agents) and **cyclification** (cyclic structures increase, supporting repeated verification and iterative refinement).
    - **Design Motivation**: These emergent phenomena explain why Puppeteer improves both efficiency and quality simultaneously—not by simply reducing agent count, but by enabling agents to form more effective feedback loops.

### Loss & Training

- REINFORCE gradient estimate: $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{n=1}^N (\sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|S_t)) \cdot R(\tau)$
- The orchestrator is initialized from Llama-3.1-Nemotron-70B-Reward.
- Default settings: episode length 4, parallel rollouts 3, $\lambda=0.1$, $\gamma=0.99$.

## Key Experimental Results

### Main Results (Titan Subspace)

| Method | GSM-Hard | MMLU-Pro | SRDD | CommonGen | Average |
|--------|----------|----------|------|-----------|---------|
| LLaMA-3.1-405B | 0.135 | 0.760 | 0.606 | 0.812 | 0.578 |
| GPT-4-Turbo | 0.275 | 0.680 | 0.624 | 0.763 | 0.586 |
| AFlow | 0.540 | 0.750 | 0.648 | 0.822 | 0.690 |
| MacNet | 0.291 | 0.480 | 0.423 | **0.882** | 0.519 |
| EvoAgent | 0.425 | 0.540 | 0.173 | 0.860 | 0.499 |
| **Puppeteer (evolved)** | **0.700** | **0.830** | **0.764** | 0.799 | **0.773** |

### Efficiency Analysis

| Metric | Initial Phase | After Evolution | Change |
|--------|--------------|-----------------|--------|
| Titan avg. token consumption | ~30K | ~15K | **−50%** |
| Titan avg. agent count | ~3.5 | ~2.0 | **−43%** |
| Mimas avg. agent count | ~3.5 | ~3.5 | Unchanged (smaller models require more reasoning steps) |

### Key Findings

- **Simultaneous gains in performance and efficiency**: Puppeteer (evolved) achieves an average of 0.773 on Titan vs. 0.690 for the strongest baseline AFlow (+8.3%), while reducing token consumption by 50%.
- **Heterogeneous agents outperform single-model agents**: Puppeteer (multi-model) consistently outperforms Puppeteer-Mono (single-model), benefiting from the complementarity of heterogeneous agents.
- **Titan and Mimas subspaces adopt distinct evolution strategies**: The Titan subspace improves efficiency by reducing agent count; the Mimas subspace does so by selecting lower-cost agents while maintaining agent count, as smaller models require longer reasoning chains.
- **Topology evolves from sparse chains to compact cycles**: Early training phases exhibit predominantly disjoint chains; after evolution, cyclic structures and cross-connections emerge, enabling deeper internal feedback.

## Highlights & Insights

- **Serialized orchestration + RL is an elegant design**: It sidesteps the combinatorial explosion of graph topology search by recasting the problem as sequential decision-making—a paradigm transferable to any system requiring dynamic module composition (e.g., MoE routing, pipeline optimization).
- **Simultaneously improving performance and efficiency breaks a common trade-off**: The cost penalty in the reward design teaches the orchestrator that "less is more"—fewer but more precisely selected agent invocations yield better outcomes.
- **Emergent cyclic topologies as a natural reflection mechanism**: RL autonomously discovers iterative verification patterns analogous to Reflexion/Self-Refine, demonstrating that such patterns need not be hand-engineered but arise naturally as optimal collaboration strategies.
- **Divergent evolution strategies between Mimas and Titan**: This reveals how agent capability shapes optimal orchestration—stronger agents can terminate earlier, while weaker agents require more rounds but can be selected at lower cost.

## Limitations & Future Work

- The episode length is fixed at 4, limiting the ability to handle more complex tasks; termination timing should ideally also be learned by the RL agent.
- REINFORCE suffers from high variance and low sample efficiency; PPO or Actor-Critic methods could offer improvements.
- The orchestrator, as a standalone model, introduces additional inference overhead (though substantially less than that of the agents themselves).
- Evaluation is limited to 4 benchmarks; validation on more complex multi-step planning tasks (e.g., WebArena, SWE-bench) is lacking.
- Explicit inter-agent communication mechanisms are not explored; agents currently interact only indirectly through the global state.

## Related Work & Insights

- **vs. MacNet**: MacNet employs a static DAG topology, whereas Puppeteer generates topologies dynamically. MacNet performs strongly on CommonGen (dense interaction benefits creative tasks) but poorly on GSM-Hard, which demands precise reasoning.
- **vs. AFlow**: AFlow applies MCTS to optimize the code representation of single-agent workflows; Puppeteer optimizes multi-agent activation sequences—these operate at different levels of abstraction and could in principle be combined.
- **vs. EvoAgent**: EvoAgent uses evolutionary algorithms to automatically generate agents, but the topology remains static; Puppeteer's dynamic orchestration comprehensively outperforms it across all tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The Puppeteer paradigm and the serialized orchestration design are innovative, though the idea of RL-based agent orchestration is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation spans 4 benchmarks (closed- and open-domain) across two agent subspaces (Titan and Mimas), with rich topology evolution analysis.
- Writing Quality: ⭐⭐⭐⭐ The puppetry analogy is vivid and intuitive, though the heavy use of notation can be demanding.
- Value: ⭐⭐⭐⭐⭐ Provides a principled solution for the automated organizational design of multi-agent systems.

## Key Experimental Results

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies](gaudp_reinventing_multi-agent_collaboration_through_gaussian-image_synergy_in_di.md)
- [\[NeurIPS 2025\] 3D-Agent: Tri-Modal Multi-Agent Collaboration for Scalable 3D Object Annotation](3d-agenttri-modal_multi-agent_collaboration_for_scalable_3d_object_annotation.md)
- [\[NeurIPS 2025\] Thought Communication in Multiagent Collaboration](thought_communication_in_multiagent_collaboration.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](../../ACL2026/multi_agent/agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[NeurIPS 2025\] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)

</div>

<!-- RELATED:END -->
