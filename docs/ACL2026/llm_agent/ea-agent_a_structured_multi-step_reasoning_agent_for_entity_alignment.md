---
title: >-
  [Paper Note] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment
description: >-
  [ACL 2026][LLM Agent][entity alignment] This paper proposes EA-Agent, which decomposes entity alignment (EA) into a structured multi-step reasoning process. Through planning and execution over a tool pool (triple selector + alignment tool + reflector), EA-Agent achieves interpretable alignment decisions. Combined with reward-guided offline policy optimization for continuous improvement of planning capability, it achieves up to 3.17% Hits@1 improvement on DBP15K while reducing efficiency issues caused by redundant triples.
tags:
  - ACL 2026
  - LLM Agent
  - entity alignment
  - knowledge graph
  - multi-step reasoning
  - tool planning
  - reward-guided optimization
date: 2026-05-08
content_hash: ef6ee6929cbab09c
---

# EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment

**Conference**: ACL 2026
**arXiv**: [2604.11686](https://arxiv.org/abs/2604.11686)
**Code**: [GitHub](https://github.com/YXNan0110/EA-Agent)
**Area**: LLM Agent
**Keywords**: entity alignment, knowledge graph, multi-step reasoning, tool planning, reward-guided optimization

## TL;DR
This paper proposes EA-Agent, which decomposes entity alignment (EA) into a structured multi-step reasoning process. Through planning and execution over a tool pool (triple selector + alignment tool + reflector), EA-Agent achieves interpretable alignment decisions. Combined with reward-guided offline policy optimization for continuous improvement of planning capability, it achieves up to 3.17% Hits@1 improvement on DBP15K while reducing efficiency issues caused by redundant triples.

## Background & Motivation

**State of the Field**: Entity alignment is a foundational technique for knowledge fusion, aiming to identify nodes across different knowledge graphs that refer to the same real-world entity. Traditional approaches rely on knowledge representation learning (e.g., TransE, GCN-Align), but exhibit limited performance under noisy or sparsely supervised settings. Recent LLM-based methods (e.g., ChatEA, LLMEA) have improved performance by leveraging semantic understanding.

**Limitations of Prior Work**: (1) Existing LLM-based EA methods treat LLMs as black-box decision makers, lacking interpretability—it is difficult to determine which information is critical for alignment decisions; (2) directly feeding large numbers of attribute and relation triples leads to excessively long prompts and high inference costs; (3) many triples are redundant or noisy, which interferes with decision-making.

**Root Cause**: There is a need to leverage LLMs' powerful semantic understanding while simultaneously addressing the issues of black-box opacity and efficiency under large-scale triple inputs.

**Paper Goals**: To design a reasoning-driven agent framework that achieves interpretable, controllable, and efficient entity alignment through multi-step tool planning and execution.

**Starting Point**: EA is framed as a multi-step decision problem—first selecting the most informative triples, then making alignment decisions, and finally performing reflective verification under uncertainty.

**Core Idea**: A tool pool (attribute/relation triple selectors + alignment tool + reflector) + path planning + reward-guided offline policy optimization.

## Method

### Overall Architecture
Three stages: (1) **Path Planning**: the agent autonomously plans a tool invocation path based on structural features of the source entity and candidate similarity scores; (2) **Tool Invocation**: triple selection, alignment decision, and reflective verification are executed in the planned order; (3) **Agent Optimization**: a reward function evaluates path quality → offline policy update → iterative improvement.

### Key Designs

1. **Attribute/Relation Triple Selector**:

    - Function: filters redundant triples before LLM reasoning, retaining the most discriminative information.
    - Mechanism: The attribute selector applies an entropy-based criterion $H(a) = -\sum p(v)\log p(v)$—attributes whose values are more uniformly distributed (lower entropy) across candidate entities have stronger discriminative power. The relation selector uses inverse-frequency weighting $I(r) = \log(N/(\text{freq}(r)+1))$—rarer relations are more discriminative. Predefined important attributes are also preserved.
    - Design Motivation: Feeding all triples indiscriminately wastes tokens and introduces noise. The selector acts as an information bottleneck, retaining only critical signals.

2. **Reward-Guided Path Optimization**:

    - Function: continuously improves the agent's tool planning strategy.
    - Mechanism: The reward function $\gamma = \gamma_\mu + c \cdot \gamma_{\text{ref}} + \gamma_e$ comprises three components: (1) alignment correctness $\gamma_\mu$ (primary); (2) reflection quality $\gamma_{\text{ref}}$ (rewards successful corrections, penalizes erroneous modifications, lightly penalizes redundant reflection); (3) path efficiency $\gamma_e = e^{-\beta \cdot l}$ (penalizes excessively long paths). The policy is optimized via reward-guided offline SFT by rewriting paths.
    - Design Motivation: Single-round planning may produce redundant or inefficient paths. The closed-loop cycle of "plan → execute → evaluate → update" ensures continuous policy improvement.

3. **Reflector (conditionally activated)**:

    - Function: verifies and corrects uncertain alignment results.
    - Mechanism: An LLM-based module activated only when candidate similarity scores indicate ambiguity. It re-evaluates candidates based on prior context and provides a revised prediction.
    - Design Motivation: Not all alignments require reflection—direct decision-making is more efficient for straightforward cases, and additional verification is triggered only under uncertainty.

## Key Experimental Results

### Main Results (DBP15K)

| Method | FR-EN Hits@1 | JA-EN Hits@1 | ZH-EN Hits@1 |
|--------|-------------|-------------|-------------|
| GCN-Align | ~40 | ~40 | ~40 |
| TEA | ~90 | ~90 | ~85 |
| ChatEA | ~92 | ~91 | ~88 |
| **EA-Agent** | **~95** | **~94** | **~91** |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| w/o triple selection | Token consumption increases substantially; slight performance degradation |
| w/o reflector | Error rate increases on uncertain cases |
| w/o path optimization | Planning strategy becomes unstable; more redundant tool invocations |
| **Full EA-Agent** | Optimal performance + highest efficiency |

### Key Findings
- **EA-Agent achieves state-of-the-art on all datasets**, with up to 3.17% improvement in Hits@1 and consistent MRR gains.
- **The triple selector significantly reduces token consumption** while maintaining or even improving performance—confirming that a large proportion of triples are indeed redundant.
- **Path optimization substantially improves planning quality**: after 3 iterations, both path efficiency and alignment accuracy improve steadily.
- **Conditional activation of the reflector is the optimal strategy**: always-on activation is inferior to on-demand activation.
- **Interpretability**: every alignment decision is traceable to a specific tool invocation path and the corresponding key triples.

## Highlights & Insights
- **Framing EA as a multi-step tool planning problem** opens up the agent paradigm for knowledge graph tasks.
- **The three-component reward function design** is highly practical: it balances correctness, reflection quality, and efficiency, avoiding optimization bias toward any single objective.
- **The triple selector's use of information-theoretic criteria** (entropy and inverse frequency) is a simple yet effective approach that can be directly transferred to other KG tasks.

## Limitations & Future Work
- The approach depends on TEA to generate initial candidate lists, so candidate quality caps overall performance.
- Path optimization requires multiple iterations, leading to relatively high training costs.
- Validation is conducted only on cross-lingual EA; monolingual or cross-domain EA remains unexplored.
- The tool pool is manually designed; whether new tools can be discovered automatically is an open question.
- The reflector's judgments may introduce new hallucinations.

## Related Work & Insights
- **vs. ChatEA**: ChatEA formats KG structure using code, but alignment decisions remain black-box. EA-Agent achieves interpretable decisions through tool planning.
- **vs. LLMEA**: LLMEA directly inputs all triples, whereas EA-Agent selects before aligning, yielding higher efficiency.
- **vs. general agent frameworks**: EA-Agent specializes the agent paradigm for KG tasks; both the tool design and reward function are task-specific.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing the agent paradigm into EA is novel, though individual components (tool planning, LoRA fine-tuning) are not new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets + 10 baselines + ablation + efficiency analysis + interpretability case studies.
- Writing Quality: ⭐⭐⭐⭐ RQ-driven structure; formalization is clear.
- Value: ⭐⭐⭐⭐ Provides methodological inspiration for LLM applications in the KG domain.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] From Query to Counsel: Structured Reasoning with a Multi-Agent Framework and Dataset for Legal Consultation](from_query_to_counsel_structured_reasoning_with_a_multi-agent_framework_and_data.md)
- [\[AAAI 2026\] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment](../../AAAI2026/llm_agent/arcane_a_multi-agent_framework_for_interpretable_and_configurable_alignment.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)

<!-- RELATED:END -->
