---
title: >-
  [Paper Note] Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS
description: >-
  [ACL2026][LLM Reasoning][LLM-MAS] This paper proposes Evo-Attacker, which models tool return tampering in LLM multi-agent systems as a long-horizon reinforcement learning problem with dynamic attack memory. It employs At…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "LLM-MAS"
  - "tool attack"
  - "attack memory"
  - "GRPO"
  - "red-teaming"
date: 2026-05-08
content_hash: bb0b937508bfdc5a
---

# Evo-Attacker: Memory-Augmented Reinforcement Learning for Long-Horizon Tool Attacks on LLM-MAS

**Conference**: ACL2026  
**arXiv**: [2605.25389](https://arxiv.org/abs/2605.25389)  
**Code**: No public code link found in cache  
**Area**: LLM Security / Multi-Agent Systems / Tool Call Robustness  
**Keywords**: LLM-MAS, tool attack, attack memory, GRPO, red-teaming

## TL;DR
This paper proposes Evo-Attacker, which models tool return tampering in LLM multi-agent systems as a long-horizon reinforcement learning problem with dynamic attack memory. It employs Attack-Flow GRPO to optimize retrieve, reflect, and modify decisions, significantly reducing system success rates across multi-architecture and multi-task benchmarks.

## Background & Motivation
**Background**: LLM multi-agent systems (LLM-MAS) accomplish complex tasks like code generation, deep research, and web operations through multiple roles and external tools. Tool returns are typically treated as factual evidence by agents, serving as critical inputs for planning, reasoning, and collaboration chains.

**Limitations of Prior Work**: Many security studies focus on explicit malicious instructions in user inputs or inter-agent messages, but the tool channel is more covert. In reality, network transmissions, third-party APIs, or search services can be compromised. If tool returns are slightly tampered with, errors can propagate along the multi-round collaboration, ultimately causing global task failure without necessarily triggering input safety filters.

**Key Challenge**: Multi-agent tool attacks must generalize across diverse tasks and tool schemas while identifying critical moments during long interactions. Static templates or single-step injections are easily validated by subsequent agents; purely online exploration faces sparse terminal rewards and long-horizon credit assignment.

**Goal**: Construct a unified red-teaming framework capable of identifying key tool calls under a limited intervention budget, generating contextually consistent perturbations using historical success, and optimizing the entire attack reasoning flow via reinforcement learning.

**Key Insight**: The attacker is treated as a gray-box adversary: it can only monitor and modify the tool returns of a single target agent, without visibility into the internal states or private messages of other agents. This restriction aligns with real-world toolchain risks and forces the attack strategy to efficiently utilize local observations.

**Core Idea**: Transform successful attack trajectories into dynamic memory. The attack strategy then performs a Retrieve-Reflect-Modify cycle during each tool call, with intermediate reasoning steps reinforced by the terminal failure reward.

## Method
Evo-Attacker is a framework designed for evaluation and red-teaming. Its focus is not on generating fixed templates but on modeling tool return risks as long-term decision-making: when to remain idle, when to continue collecting context, when to modify a specific tool return, and how to ensure the modification appears consistent within the task context.

### Overall Architecture
The system first formalizes the LLM-MAS as a dynamic graph $\mathcal{G}=(\mathcal{A},\mathcal{E})$, where nodes are agents and edges are communication channels. Each agent may call a tool at a certain time step, represented as $(id,args,r)$, where $r$ is the return value.

The attacker adopts a gray-box threat model: it only controls the tool channel of a single target agent, intercepting outgoing tool calls and original returns. Within a budget $B$, it can replace specific returns with perturbed values. The objective is to maximize the final system failure probability $\mathbb{E}[J(o)]$, where $J(o)=1$ denotes task failure.

Evo-Attacker consists of three stages. The first stage constructs the Attack Memory by collecting trajectories that successfully cause system failure. The second stage performs memory-augmented attacks, retrieving similar experiences based on the current call and historical interactions to reflect on intervention suitability before modifying. The third stage uses Attack-Flow GRPO to optimize the Retrieve-Reflect-Modify reasoning chain.

### Key Designs
1.  **Dynamic Attack Memory**:
    *   **Function**: Stores transferable successful attack experiences across tasks.
    *   **Mechanism**: During the exploration phase, if an interaction leads to system failure, the context $\mathcal{X}_{ctx}=(\mathcal{G},\mathcal{T},a^*)$ and attack trace $T_{trace}$ are saved as a memory entry, including the original call, modification logic, and perturbed return.
    *   **Design Motivation**: Static templates generalize poorly due to diverse multi-agent tasks and tool schemas. Memory transforms historical vulnerability patterns into retrievable experiences, helping attackers quickly locate similar risk surfaces in new scenarios.

2.  **Retrieve-Reflect-Modify Reasoning**:
    *   **Function**: Enables the attacker to decide if, when, and how to intervene based on context.
    *   **Mechanism**: The Retrieve stage generates queries to fetch top-$k$ similar memories; the Reflect stage determines if historical patterns are transferable and outputs Attack, Continue, or NoOp; the Modify stage selects specific tool calls and generates modification instructions.
    *   **Design Motivation**: Applying historical templates directly often results in contextual inconsistency, while random interventions waste budget. The reflection step acts as a filter for transferability and attackability.

3.  **Attack-Flow GRPO Long-Horizon Optimization**:
    *   **Function**: Addresses the issue where success/failure is only observed at the end of the episode.
    *   **Mechanism**: An attack episode is viewed as a trajectory of alternating attacker actions and environment responses. The reward is defined as $R(\zeta)=\mathbb{I}(J(o_{sys})=1)+\lambda R_{struct}(\zeta)$, which is then broadcast to the Retrieve/Reflect/Modify tokens. GRPO calculates group-relative advantage using $G$ rollouts under the same task context.
    *   **Design Motivation**: Tool attack success often depends on multi-round planning; rewarding only the last modification is insufficient. Propagating the final signal to all attacker decisions strengthens early retrieval and timing judgment.

### Loss & Training
The optimization objective only applies to tokens generated by the attacker; tool returns and victim agent messages are masked. The paper uses Qwen3-14B as the victim agent backbone and Qwen3-8B as the Evo-Attacker backbone. The intervention budget is set to $B=3$, retrieval count $k=5$, and GRPO uses $G=8$ parallel rollouts with $\lambda=0.5$ and a learning rate of $1e-6$. The initial attack memory is bootstrapped from 500 samples of the WebShop training set and 50 samples from HumanEval.

## Key Experimental Results

### Main Results
| Architecture / Task Group | Baseline (No Attack) | Post Evo-Attacker | Gain (Drop) | Conclusion |
|---------------------------|----------------------|-------------------|-------------|------------|
| Flat / MAB.code           | 66.2                 | 39.2              | 27.0        | Code tasks are significantly affected by cross-schema attacks |
| Flat / HumanEval          | 67.5                 | 38.6              | 28.9        | Strong effect even on strict syntax tasks |
| Flat / MAB.research       | 80.0                 | 54.6              | 25.4        | Errors propagate through deep research chains |
| Chain / WebShop           | 62.8                 | 33.2              | 29.6        | Largest drop in shopping/web tasks under chain architecture |
| Hierarchical / WebArena   | 35.8                 | 18.4              | 17.4        | Hierarchical filters are present but not fully protective |

### Ablation Study
| Configuration  | Key Metrics (Code/Research/Web) | Description |
|----------------|---------------------------------|-------------|
| w/o Attack     | 67.4 / 58.7 / 48.3             | Baseline without attack |
| w/o Retrieval  | 52.5 / 50.3 / 34.4             | Generalization drops without attack memory |
| w/o Reflection | 49.9 / 47.2 / 32.9             | Budget wasted on low-value steps without suitability judgment |
| w/o RL         | 55.2 / 51.0 / 36.9             | Weakest long-horizon planning without Attack-Flow GRPO |
| Full           | 40.9 / 40.1 / 26.0             | Maximum drop when all three modules are combined |

### Key Findings
*   Evo-Attacker outperforms baselines like Forced Output, InjecAgent, Web Fraud, and Prompt Infection across Flat, Chain, and Hierarchical architectures, showing its advantage is not limited to specific web templates.
*   All three modules (Retrieval, Reflection, RL) are critical. The significantly weakened performance without RL supports the author's argument regarding long-horizon credit assignment.
*   Attack effectiveness generally increases as the budget moves from 1 to 5, reflection depth from 1 to 5, and retrieved memories from 0 to 20, indicating that more test-time computation yields stronger red-teaming results.
*   Cross-model experiments show GPT/Gemini are effective as attackers in zero-shot settings; smaller open-source models like Llama/Ministral optimized with Attack-Flow GRPO can meet or exceed the performance of closed-source attackers.

## Highlights & Insights
*   The paper elevates the tool security problem from "injecting a malicious string" to "local fact pollution within long-horizon task trajectories." This is more reflective of the real risks in multi-agent + tool ecosystems.
*   The concept of Attack Memory is valuable for the defense side. Just as attackers reuse vulnerability patterns, defenders should build risk memories of tool returns to detect similar anomalies and high-risk call sites.
*   The Reflection module highlights the importance of budget constraints. Real-world attacks and defenses do not process every step but identify critical nodes; this perspective is closer to system security than single-step prompt injection.
*   The token masking design in Attack-Flow GRPO is elegant: it only optimizes the attacker’s own decisions without treating environment responses as learnable actions. This approach is a useful reference for other agentic RL optimizations.

## Limitations & Future Work
*   The method incurs high training and inference costs, particularly due to deliberative reasoning and GRPO rollouts increasing token and compute consumption.
*   Stealthiness is primarily evaluated using LLM-based detectors, lacking coverage of deterministic defenses like cryptographic signatures, tool return validation, or whitelist parameter filtering.
*   Experiments are conducted in controlled simulated environments, not covering real commercial services, actual network attack chains, or complex supply chain security constraints.
*   Future defense research could explore tool return signatures, cross-agent consistency checking, memory-aware anomaly detection, least-trust tool interfaces, and rollback mechanisms for task-critical nodes.

## Related Work & Insights
*   **vs. Single-Agent Tool Attacks**: Methods like Forced Output and InjecAgent often take a single-step or single-agent view; this work focuses on propagation and validation bypass within multi-agent chains.
*   **vs. Web Fraud / Prompt Infection**: These rely heavily on web pages or fixed templates, whereas Evo-Attacker achieves generalization across code, research, and web tasks via memory retrieval and reflection.
*   **vs. Optimization-based Prompt Attack**: GCG and AutoDAN are mostly used for static single-round triggers; this paper shifts optimization to the interaction process and tool return channels.
*   **Insight**: When evaluating LLM-MAS security, one must not only test user input filters; tool outputs, third-party APIs, search results, and intermediate files should all be included in the threat model.

## Rating
*   Novelty: ⭐⭐⭐⭐ Utilizing memory-augmented reasoning and GRPO for multi-agent tool channel red-teaming is a fresh problem setting.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple architectures, tasks, baselines, and ablations, though real-world defense coverage remains limited.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure and explicit threat model; some result tables are highly dense.
*   Value: ⭐⭐⭐⭐ Highly relevant for LLM agent security and tool-call defense, particularly for building defense evaluation sets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ToolMATH: A Math Tool Benchmark for Realistic Long-Horizon Multi-Tool Reasoning](../../ICML2026/llm_reasoning/toolmath_a_math_tool_benchmark_for_realistic_long-horizon_multi-tool_reasoning.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ACL 2026\] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](sppo_sequence-level_ppo_for_long-horizon_reasoning_tasks.md)
- [\[AAAI 2026\] Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning](../../AAAI2026/llm_reasoning/beyond_react_a_planner-centric_framework_for_complex_tool-au.md)
- [\[ACL 2026\] Revisiting Entropy in Reinforcement Learning for Large Reasoning Models](revisiting_entropy_in_reinforcement_learning_for_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
