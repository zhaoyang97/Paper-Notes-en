---
title: >-
  [Paper Note] InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios
description: >-
  [ICLR 2026][LLM Agent][multi-agent system] This paper proposes InfiAgent, a DAG-based pyramidal multi-agent framework that achieves automated hierarchical task decomposition, dual-audit quality assurance, intelligent routing, and self-evolution through an agent-as-a-tool mechanism, outperforming ADAS by an average of 9.9% across multiple reasoning benchmarks.
tags:
  - ICLR 2026
  - LLM Agent
  - multi-agent system
  - DAG
  - agent-as-a-tool
  - self-evolution
  - task decomposition
  - hierarchical agent
date: 2026-05-08
content_hash: 39deedc678af49ac
---

# InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios

**Conference**: ICLR 2026
**arXiv**: [2509.22502](https://arxiv.org/abs/2509.22502)
**Code**: To be released
**Area**: LLM Agent
**Keywords**: multi-agent system, DAG, agent-as-a-tool, self-evolution, task decomposition, hierarchical agent

## TL;DR

This paper proposes InfiAgent, a DAG-based pyramidal multi-agent framework that achieves automated hierarchical task decomposition, dual-audit quality assurance, intelligent routing, and self-evolution through an agent-as-a-tool mechanism, outperforming ADAS by an average of 9.9% across multiple reasoning benchmarks.

## Background & Motivation

Current LLM-based agent systems face several core challenges:

**Manual design bottleneck**: Existing agents require carefully crafted workflows, prompts, and iterative tuning, demanding expertise in both LLM technology and domain knowledge, which severely limits cross-industry scalability.

**High coordination overhead**: Traditional multi-agent architectures adopt peer-to-peer collaboration models that allow free interaction among agents, leading to increased coordination overhead, deadlocks, and unpredictable behavior.

**Lack of adaptability**: Existing frameworks rely on manually authored templates and lack systematic reasoning capabilities, making them unable to autonomously adapt to new tasks and optimization requirements.

**Stability issues**: As system complexity increases, unpredictable inter-agent interactions, resource conflicts, and emergent behavioral instability become increasingly prominent.

InfiAgent aims to provide a general framework that automatically adapts to diverse problem domains without extensive manual configuration.

## Method

### Overall Architecture

InfiAgent is designed as a DAG (Directed Acyclic Graph)-based decomposition and routing system. Unlike traditional architectures that emphasize direct execution, most agents in InfiAgent focus on **planning and routing**, delegating actual execution to a pool of lower-level functional agents. Each higher-level agent acts as a planner for a small set of specialized lower-level agents, orchestrating their collaboration and merging results. The overall architecture takes a pyramidal form, with a top-level Router responsible for redirecting all user queries to avoid layer-by-layer search.

### Key Designs

#### 1. Agent-as-a-Tool Mechanism and Intelligent Routing

The core principle is to treat agents as tools for decomposition and routing. When task $T_0$ is submitted to the top-level agent $\alpha$:

- Appropriate lower-level agents $\{A_1, A_2, \ldots, A_k\}$ are automatically identified.
- The task is reformulated into subtasks $\{T_1, T_2, \ldots, T_k\}$.
- Each lower-level agent is invoked as a specialized tool by the higher-level agent.

The decomposition process is formalized as:

$$T^{(l)} \mapsto \{T_1^{(l+1)}, T_2^{(l+1)}, \ldots, T_{k_l}^{(l+1)}\}$$

To preserve the simplicity of each agent, the framework enforces a strict constraint: $k_l \leq K_{\max}$, where $K_{\max} = 5$, ensuring that no individual agent faces excessive coordination complexity even as the system grows exponentially.

#### 2. Architectural Scalability

The framework leverages depth to achieve exponential growth in functional agents: with an average branching factor $b$, the number of functional agents reachable at depth $L$ is $N_{\text{func}} \approx b^L$. This grants top-level agents broad generalization capability while requiring each intermediate agent to reason over only a limited number of child nodes.

#### 3. Dual-Audit Quality Assurance

InfiAgent implements a two-level auditing mechanism:

- **Execution-level audit**: Continuously monitors and validates each agent's output, maintaining a quality score $Q_i^{(t+1)} = \alpha \cdot Q_i^{(t)} + (1-\alpha) \cdot \text{validate}(O_i^{(t)})$.
- **System-level audit**: Maintains system stability through built-in review mechanisms and retrospective summarization, while performing context compression to reduce token consumption.

#### 4. Lightweight Communication and Context Control

Agents exchange only file descriptors and metadata: $M_{i \to j} = (\text{addr}, \text{desc})$. The execution context is decomposed into four structured components:

- **System prompt context** $C_{\text{sys}}$: Predefined prompts that guide agent behavior.
- **Long-term memory index** $C_{\text{LM}}$: Compressed file descriptors and indices, $C_{\text{LM}} = \text{compress}(\{d(f_i) \mid f_i \in \mathcal{F}\})$.
- **Short-term shared memory** $C_{\text{SM}}$: A dynamic call stack recording the active agent invocation tree.
- **Compressed environment interaction context** $C_{\text{ENV}}$: Automatically compressed when token length approaches threshold $\tau$.

By enforcing $|C| \ll |H|$ (where $H$ denotes the complete history log), context length is kept bounded.

#### 5. Self-Evolution Mechanism

Evolution occurs at three levels:

- **Model level**: A Git-style workflow in which multiple lightweight models execute in parallel; a Judge model $J$ evaluates their outputs before merging into the main branch $B_{\text{main}}^{(t+1)} = \text{merge}(B_{\text{main}}^{(t)}, \{\Delta m_i^{(t)} \mid J(\Delta m_i^{(t)}) = 1\})$.
- **Agent level**: High-quality training data from the main branch is used to continuously update all parallel models $m_i^{(t+1)} \leftarrow \text{train}(m_i^{(t)}, D(B_{\text{main}}^{(t)}))$.
- **Topology level**: The DAG topology is dynamically restructured based on performance patterns and emerging task requirements, enabling the formation of domain-level expert models.

### Loss & Training

The framework is optimized through a quality scoring mechanism and self-evolution feedback loop. The core objective is to maximize task completion quality $Q_i$ while minimizing context length $|C|$.

## Key Experimental Results

### Main Results

Performance on five benchmarks (all using GPT-4o-mini as the backbone model):

| Method | DROP | HumanEval | MBPP | GSM8K | MATH | Avg. |
|--------|------|-----------|------|-------|------|------|
| IO (GPT-4o-mini) | 68.3 | 87.0 | 71.8 | 92.7 | 48.6 | 73.68 |
| CoT | 78.5 | 88.6 | 71.8 | 92.4 | 48.8 | 76.02 |
| CoT SC (5-shots) | 78.8 | 91.6 | 73.6 | 92.7 | 50.4 | 77.42 |
| MedPrompt | 78.0 | 91.6 | 73.6 | 90.0 | 50.0 | 76.64 |
| Self Refine | 70.2 | 87.8 | 69.8 | 89.6 | 46.1 | 72.70 |
| ADAS | 76.6 | 82.4 | 53.4 | 90.8 | 35.4 | 67.72 |
| **InfiAgent** | **82.4** | 89.3 | 71.8 | **93.1** | 35.6 | 74.44 |

### Ablation Study

InfiHelper case study — quality evaluation comparison (peer review on a 1–10 scale):

| System | Representative Paper | Best Score | Avg. Score |
|--------|----------------------|------------|------------|
| AI-Researcher | Multiple VQ-VAE/GCN | 6 | 4.75 |
| Zochi | Tempest Jailbreak | 6 | 6.0 |
| Sakana-AI | Compositional Regularization | 4 | 4.0 |
| **InfiHelper** | Adaptive Multi-Scale DAS | **7** | **5.67** |

### Key Findings

1. **Superior complex reasoning**: DROP reaches 82.4%, surpassing the best baseline CoT SC (78.8%) by 3.6 percentage points, validating the advantage of the agent-as-a-tool mechanism on multi-step reasoning tasks.
2. **Strong math and code performance**: GSM8K achieves 93.1% (highest overall); HumanEval at 89.3% is highly competitive.
3. **Limited performance on specialized mathematics**: MATH scores only 35.6%, as the overhead of the tool-calling framework consumes model capacity that could otherwise be devoted to direct mathematical reasoning.
4. **Average improvement of 9.9% over ADAS**: Validates that hierarchical decomposition outperforms end-to-end automatic generation.

## Highlights & Insights

1. **Elegant agent-as-a-tool abstraction**: Treating agents uniformly as tools achieves unprecedented modularity and reusability, enabling the same framework to handle tasks ranging from scientific research to software engineering.
2. **Pyramid structure with Router design**: The Router directly redirects user queries, avoiding layer-by-layer search and significantly improving efficiency.
3. **Self-evolution as the standout contribution**: The Git-style model evolution workflow enables the system to optimize autonomously without human intervention.
4. **Practical lightweight communication design**: Passing only (addr, desc) pairs ensures bounded context length, addressing context bloat in long-running tasks.
5. **Papers generated by InfiHelper passed human peer review at a top IEEE venue**, demonstrating the system's practical value in real-world research scenarios.

## Limitations & Future Work

1. **Suboptimal MATH benchmark performance**: For challenging problems requiring focused derivation rather than multi-step decomposition, the overhead of the tool-calling framework becomes a liability.
2. **Experiments use a uniform backbone model**: Heterogeneous model collaboration is sacrificed for the sake of fair comparison.
3. **InfiHelper case study lacks detailed quantitative evaluation**: Assessment relies primarily on AI reviewer scores.
4. **Convergence and stability of the self-evolution mechanism are insufficiently analyzed.**
5. **Lack of direct comparison with more recent multi-agent frameworks such as AutoGen and MetaGPT.**

## Related Work & Insights

- **ADAS** (Hu et al., 2024): An end-to-end framework for automatic agent generation; InfiAgent improves upon it by an average of 9.9% through structured decomposition.
- **AgentGym / EvoAgent**: Self-improving and evolving agents; InfiAgent extends this line of work by adding topology-level evolution.
- **NADER** (Yang et al., 2025): A collaborative architecture design framework emphasizing algorithmic and structural evolution for building resilient systems.
- **Insights**: The agent-as-a-tool abstraction is generalizable to broader domains; the Git-style self-evolution workflow is worth adopting in other systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Agent-as-a-Tool and the pyramidal DAG architecture represent interesting system-level design innovations.
- **Value**: ⭐⭐⭐⭐ — The framework is readily extensible and deployable; the InfiHelper case study demonstrates real-world applicability.
- **Experimental Thoroughness**: ⭐⭐⭐ — Benchmark experiments are adequate, but the case study evaluation is relatively weak.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, with complete mathematical formalization and intuitive illustrations.
- **Overall**: ⭐⭐⭐⭐ (7.5/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)
- [\[ICLR 2026\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ICLR 2026\] Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](multi-agent_design_optimizing_agents_with_better_prompts_and_topologies.md)

</div>

<!-- RELATED:END -->
