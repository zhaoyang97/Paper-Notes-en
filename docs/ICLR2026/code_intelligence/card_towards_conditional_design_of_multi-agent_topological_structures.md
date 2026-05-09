---
title: >-
  [Paper Note] CARD: Towards Conditional Design of Multi-agent Topological Structures
description: >-
  [ICLR 2026][Multi-Agent Communication Topology] CARD proposes a Conditional Agentic Graph Designer framework that adaptively designs multi-agent communication topologies based on dynamic environment signals—including model capability changes, tool availability, and knowledge source updates—via a conditional variational graph encoder and environment-aware optimization. The approach consistently outperforms static and prompt-based baselines on HumanEval, MATH, and MMLU.
tags:
  - ICLR 2026
  - Multi-Agent Communication Topology
  - Conditional Graph Generation
  - Graph Neural Networks
  - Dynamic Environment Signals
  - Agent Collaboration
date: 2026-05-08
content_hash: d5c68c1fc8edec2d
---

# CARD: Towards Conditional Design of Multi-agent Topological Structures

**Conference**: ICLR 2026
**arXiv**: [2603.01089](https://arxiv.org/abs/2603.01089)
**Code**: [https://github.com/Warma10032/CARD](https://github.com/Warma10032/CARD)
**Area**: Code Intelligence
**Keywords**: Multi-Agent Communication Topology, Conditional Graph Generation, Graph Neural Networks, Dynamic Environment Signals, Agent Collaboration

## TL;DR
CARD proposes a Conditional Agentic Graph Designer framework that adaptively designs multi-agent communication topologies based on dynamic environment signals—including model capability changes, tool availability, and knowledge source updates—via a conditional variational graph encoder and environment-aware optimization. The approach consistently outperforms static and prompt-based baselines on HumanEval, MATH, and MMLU.

## Background & Motivation
LLM-based multi-agent systems have demonstrated strong performance on tasks such as code generation and collaborative reasoning, yet their effectiveness and robustness depend critically on the communication topology among agents. Two key deficiencies characterize existing approaches: (1) communication topologies are typically **fixed** (e.g., chain, hierarchical) or **statically learned**, ignoring real-world dynamic factors such as model upgrades, API changes, tool capability updates, and knowledge source refreshes; and (2) no systematic protocol exists for describing and adapting to these environmental changes. The root cause is that static topologies cannot accommodate deployment-time dynamics, while manual topology redesign is not scalable. The paper's starting point is to frame multi-agent topology design as a **conditional graph generation problem**, wherein environment signals drive the adaptive construction of topologies.

## Method

### Overall Architecture
The CARD framework first defines AMACP (Adaptive Multi-Agent Communication Protocol), a protocol for adaptive multi-agent communication. Building on this protocol, CARD models the multi-agent system as a dynamic graph structure: each agent is a node, and inter-agent communication relationships are edges (including spatial and temporal edges). The core of CARD is a conditional graph generator that takes as input a task description and dynamic environment signals (LLM capability profiles, tool availability, and knowledge source status) and produces an optimized communication topology graph. At runtime, agents perform message passing and collaborative reasoning according to the generated topology.

### Key Designs

1. **AMACP (Adaptive Multi-Agent Communication Protocol)**: Defines communication specifications for multi-agent systems under dynamic environments. The core mechanism of AMACP is to explicitly encode environment state as conditional signals and incorporate them into the topology design process. The protocol specifies three categories of dynamic environment signals: **(a) LLM Profiles**: characterizing the capability of the LLM used by each agent (e.g., reasoning strength, code generation ability, knowledge coverage); **(b) Tool Capabilities**: describing the functionality and status of available tools (e.g., whether a search engine is accessible, the version of a code executor); **(c) Knowledge Sources**: describing the reachability and freshness of external knowledge bases. When any signal changes, AMACP triggers topology regeneration to ensure the system adapts to the current environment.

2. **Conditional Variational Graph Encoder**: The core component of CARD. This encoder maps agent features (roles, LLM capabilities) and environment signals into a latent space representation of the graph. Concretely, a GCN (Graph Convolutional Network) performs feature aggregation over agent nodes, and a conditional variational approach is used to learn the distribution over topologies. The generation process involves two types of edges: **(a) Spatial Edges**: defining the information flow among agents within a single reasoning step—specifying which agent should pass its output to which other agent; **(b) Temporal Edges**: defining information transfer across reasoning rounds—specifying how an agent's output at round $t$ influences other agents' inputs at round $t+1$. Through variational inference, the encoder can sample multiple candidate topologies given environmental conditions and select the optimal one via the optimization objective.

3. **Environment-Aware Optimization**: During training, CARD employs an environment-condition-modulated loss function that optimizes not only task accuracy but also the robustness of the topology across diverse environment configurations. Training data covers multiple environment configurations (different LLM combinations, varying tool availability, etc.), enabling CARD to learn topologies that perform well under each configuration. At runtime, CARD rapidly generates a topology adapted to currently perceived environment signals without retraining—achieved through the generative capacity of the conditional VAE.

4. **Multi-Agent Execution Engine**: After topology generation, the system organizes collaborative agent execution according to the graph structure. Each agent node is assigned a specific role (e.g., CodeWriter, MathSolver, Analyzer) and equipped with corresponding external tools (search, code execution, RAG, etc.). Messages flow along spatial and temporal edges, supporting multi-round iterative reasoning. The framework is compatible with multiple LLMs including GPT-4, Claude, DeepSeek, and Llama.

### Loss & Training
Training proceeds in two stages: (1) a fully connected (FullConnected) configuration is used to initialize the agent system, and performance data collected across diverse environment configurations serves as training signal; (2) CARD's conditional graph generator learns the mapping from environment signals to optimal topologies. The optimization objective jointly considers task accuracy and topology efficiency (communication overhead). GCN-based feature aggregation is combined with the ELBO loss of the conditional VAE to train the graph generator. Evaluation uses 5 agents, 10 iterations, and a batch size of 8.

## Key Experimental Results

### Main Results

| Dataset | Metric | CARD | FullConnected | Chain | Random | Note |
|---------|--------|------|---------------|-------|--------|------|
| HumanEval | Pass Rate | Best | 2nd | Poor | Worst | Code generation |
| MATH | Accuracy | Best | 2nd | Poor | Poor | Mathematical reasoning |
| MMLU | Accuracy | Best | 2nd | Poor | Medium | General knowledge |

CARD consistently outperforms static topology baselines and prompt-based dynamic baselines across all three benchmarks, with the advantage becoming more pronounced when the environment changes (e.g., agent model downgrade, tool unavailability).

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Without environment condition signals | Significant drop | Degrades to unconditional graph generation; cannot adapt to environment changes |
| Spatial edges only | Moderate drop | Lacks temporal information flow; multi-round reasoning is limited |
| Temporal edges only | Larger drop | Lacks intra-round inter-agent collaboration; performance on complex tasks decreases |
| Without GNN feature aggregation | Drop | Agent features not enhanced by graph structure; generated topology quality degrades |
| FullConnected baseline | High but below CARD | High communication overhead; lacks adaptability when environment changes |

### Key Findings
- Topologies generated by CARD approach or surpass the performance of full connectivity under normal environments while incurring significantly lower communication overhead—sparse but precise communication outperforms redundant full connectivity.
- In model downgrade scenarios (e.g., replacing one agent's LLM from GPT-4 to a weaker model), CARD automatically adjusts the topology to reduce reliance on the degraded agent, resulting in a much smaller performance drop compared to fixed topologies.
- Conditional variational generation allows multiple candidate topologies to be produced for the same task, with the optimal one selected, improving system robustness.
- CARD's advantage grows as the number of agents increases, since the design space of static topologies grows exponentially in large-scale agent systems.
- CARD draws on GDesigner's design philosophy, but the key distinction is the introduction of environment condition signals, enabling adaptation to dynamic changes in real-world deployment.

## Highlights & Insights
- **Analogy from NAS to agent topology search**: Just as Neural Architecture Search (NAS) automates network design, CARD automates the design of multi-agent collaboration structures, with conditional generation enabling adaptation to dynamic environments.
- **Fusion of graph generation and agent systems**: Combining the graph structure learning capabilities of GNNs with LLM-based agent systems represents a promising research direction.
- **Generality of the AMACP protocol**: Although this work focuses on code, math, and knowledge tasks, the AMACP protocol is designed to be general and can extend to any scenario requiring multi-agent collaboration.
- **Explicit modeling of environment signals**: This is the key innovation distinguishing CARD from prior work such as GDesigner—CARD learns not only "which topology is good" but also "which topology is good under which conditions."

## Limitations & Future Work
- Validation is currently limited to three task types—code generation, mathematical reasoning, and knowledge QA—without coverage of more complex multi-agent collaboration scenarios (e.g., debate, creative writing).
- Defining and collecting environment signals may pose challenges in practical deployment—how to automatically detect changes in model capabilities remains an open question.
- Training the conditional VAE requires sufficient data collected across diverse environment configurations, leading to relatively high initial training costs.
- The number of agents is fixed at 5; scenarios involving dynamic addition or removal of agents are not explored.
- Quantitative comparisons with existing methods such as GDesigner lack sufficient detail, possibly due to space constraints.
- Topology generation is currently one-shot; dynamic topology adjustment during execution is not explored.

## Related Work & Insights
- **GDesigner**: The primary reference work, which uses GNNs to design multi-agent topologies but lacks conditional adaptation capability. CARD extends this by introducing conditional generation.
- **MetaGPT / AutoGen / CrewAI**: Popular multi-agent frameworks that employ fixed topologies (e.g., SOP-based pipelines) and struggle to adapt to environmental changes.
- **CVAE for graph generation**: Methods such as Graph VAE provide the theoretical foundation for CARD's conditional graph generator.
- **Insight**: The "meta-design" of multi-agent systems—designing how to design agent systems—remains an underexplored direction, and CARD opens a new path through conditional design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VeriMaAS: Automated Multi-Agent Workflows for RTL Design](../../NeurIPS2025/code_intelligence/automated_multi-agent_workflows_for_rtl_design.md)
- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)
- [\[ACL 2026\] MARS2: Scaling Multi-Agent Tree Search via Reinforcement Learning for Code Generation](../../ACL2026/code_intelligence/mars2_scaling_multi-agent_tree_search_via_reinforcement_learning_for_code_genera.md)
- [\[AAAI 2026\] Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction](../../AAAI2026/code_intelligence/extracting_events_like_code_a_multi-agent_programming_framework_for_zero-shot_ev.md)
- [\[ICLR 2026\] Improving Code Localization with Repository Memory](improving_code_localization_with_repository_memory.md)

<!-- RELATED:END -->
