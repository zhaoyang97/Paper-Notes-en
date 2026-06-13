---
title: >-
  [Paper Note] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment
description: >-
  [ACL 2026][Graph Learning][Entity Alignment] EA-Agent is proposed to decompose Entity Alignment (EA) into a structured multi-step reasoning process. By planning and executing a tool pool (triplet selector + alignment too…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "Entity Alignment"
  - "Knowledge Graph"
  - "Multi-step Reasoning"
  - "Tool Planning"
  - "Reward-guided Optimization"
date: 2026-05-08
content_hash: c709a60650dfe618
---

# EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment

**Conference**: ACL 2026  
**arXiv**: [2604.11686](https://arxiv.org/abs/2604.11686)  
**Code**: [GitHub](https://github.com/YXNan0110/EA-Agent)  
**Area**: LLM Agent  
**Keywords**: Entity Alignment, Knowledge Graph, Multi-step Reasoning, Tool Planning, Reward-guided Optimization

## TL;DR
EA-Agent is proposed to decompose Entity Alignment (EA) into a structured multi-step reasoning process. By planning and executing a tool pool (triplet selector + alignment tool + reflector), it achieves interpretable alignment decisions. Combined with reward-guided offline policy optimization for continuous improvement of planning capabilities, it improves Hits@1 on DBP15K by up to 3.17% while mitigating efficiency issues caused by redundant triplets.

## Background & Motivation

**Background**: Entity alignment is a fundamental technology for knowledge fusion, aiming to identify nodes in different knowledge graphs that refer to the same entity. Traditional methods based on knowledge representation learning (e.g., TransE, GCN-Align) show limited performance in noisy or sparse supervised scenarios. Recent LLM-based methods (e.g., ChatEA, LLMEA) utilize semantic understanding to improve performance.

**Limitations of Prior Work**: (1) Existing LLM-based EA methods treat LLMs as black-box decision-makers, lacking interpretability—making it difficult to determine which information is critical for alignment decisions; (2) Directly inputting large numbers of attribute and relation triplets results in excessively long prompts and high inference costs; (3) Many triplets are redundant or even noisy, which can interfere with judgment.

**Key Challenge**: The need to utilize the powerful semantic understanding of LLMs while addressing the issues of black-box non-interpretability and the efficiency of large-scale triplet processing.

**Goal**: Design a reasoning-driven Agent framework that achieves interpretable, controllable, and efficient entity alignment through multi-step tool planning and execution.

**Key Insight**: Treat EA as a multi-step decision-making problem—selecting the most informative triplets first, then making alignment decisions, and finally performing reflection for verification when uncertainty exists.

**Core Idea**: Tool pool (Attribute/Relation Triplet Selector + Alignment Tool + Reflector) + Path Planning + Reward-guided Offline Policy Optimization.

## Method

### Overall Architecture
Three stages: (1) **Path Planning**: The Agent autonomously plans tool invocation paths based on the structural features of the source entity and candidate similarity scores; (2) **Tool Invocation**: Triplet selection, alignment decision, and reflection verification are executed in the planned order; (3) **Agent Optimization**: A reward function evaluates path quality → offline policy update → iterative improvement.

### Key Designs

1.  **Attribute/Relation Triplet Selector**:

    - **Function**: Filters redundant triplets before LLM reasoning to retain the most discriminative information.
    - **Mechanism**: The attribute selector uses an entropy-based criterion $H(a) = -\sum p(v)\log p(v)$—a more uniform distribution of attribute values among candidate entities (lower entropy) indicates stronger discriminative power. The relation selector uses inverse frequency weighting $I(r) = \log(N/(\text{freq}(r)+1))$—rare relations are more discriminative. Predefined important attributes are also retained.
    - **Design Motivation**: Inputting all triplets without selection wastes tokens and introduces noise. The selector acts as an information bottleneck, preserving only critical signals.

2.  **Reward-guided Path Optimization**:

    - **Function**: Continuously improves the Agent's tool planning strategy.
    - **Mechanism**: The reward function $\gamma = \gamma_\mu + c \cdot \gamma_{\text{ref}} + \gamma_e$ comprises three components: (1) Alignment correctness $\gamma_\mu$ (core); (2) Reflection rationality $\gamma_{\text{ref}}$ (rewarding successful corrections, penalizing incorrect modifications, and slightly penalizing redundant reflections); (3) Path efficiency $\gamma_e = e^{-\beta \cdot l}$ (penalizing excessively long paths). Strategies are optimized by rewriting paths via offline SFT under reward guidance.
    - **Design Motivation**: Single-turn planning may generate redundant or inefficient paths. The closed-loop "planning → execution → evaluation → update" iteration ensures continuous strategy improvement.

3.  **Reflector (Conditional)**:

    - **Function**: Verifies and corrects alignment results for uncertain cases.
    - **Mechanism**: An LLM-based module activated only when candidate similarity scores indicate ambiguity. It re-evaluates candidates based on previous context and provides revised predictions.
    - **Design Motivation**: Not all alignments requires reflection—direct decisions are more efficient for simple cases, and extra verification is only triggered when uncertainty exists.

## Key Experimental Results

### Main Results (DBP15K)

| Method | FR-EN Hits@1 | JA-EN Hits@1 | ZH-EN Hits@1 |
|------|-------------|-------------|-------------|
| GCN-Align | ~40 | ~40 | ~40 |
| TEA | ~90 | ~90 | ~85 |
| ChatEA | ~92 | ~91 | ~88 |
| **Ours (EA-Agent)** | **~95** | **~94** | **~91** |

### Ablation Study

| Configuration | Description |
|------|-------------|
| w/o Triplet Selection | Token consumption increases significantly; performance drops slightly |
| w/o Reflector | Error rate for uncertain cases increases |
| w/o Path Optimization | Planning strategy becomes unstable; redundant tool calls increase |
| **Full EA-Agent** | Optimal performance + maximum efficiency |

### Key Findings
- **EA-Agent achieves SOTA on all datasets**, with Hits@1 improvements up to 3.17% and consistent MRR gains.
- **The triplet selector significantly reduces token consumption** while maintaining or even improving performance—proving that many triplets are indeed redundant.
- **Path optimization significantly enhances planning quality**: Both path efficiency and alignment accuracy improve steadily after 3 iterations.
- **Conditional activation of the reflector is the optimal strategy**: Always enabling it is less effective than demand-based activation.
- **Interpretability**: Every alignment decision can be traced back to a specific tool invocation path and key triplets.

## Highlights & Insights
- **Modeling EA as a multi-step tool planning problem** opens up the application space for the Agent paradigm in knowledge graph tasks.
- **The three-component reward function design** is highly practical: it balances correctness, reflection rationality, and efficiency, avoiding optimization bias toward a single goal.
- **The triplet selector utilizing information theory criteria** (entropy and inverse frequency) is a simple yet effective solution that can be directly migrated to other KG tasks.

## Limitations & Future Work
- Dependent on TEA for generating the initial candidate list; candidate quality limits the upper bound.
- Path optimization requires multiple iterations, leading to high training costs.
- Validated only on cross-lingual EA; same-language or cross-domain EA remains to be explored.
- The tool pool is manually designed; can new tools be discovered automatically?
- Reflector judgments may introduce new hallucinations.

## Related Work & Insights
- **vs ChatEA**: ChatEA uses code to format KG structures but remains a black-box decision-maker. EA-Agent achieves interpretable decisions through tool planning.
- **vs LLMEA**: LLMEA inputs all triplets directly; EA-Agent is more efficient by selecting before aligning.
- **vs General Agent Frameworks**: EA-Agent specializes the Agent paradigm for KG tasks; tool design and reward functions are task-specific.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing the Agent paradigm to EA is new, though individual components (tool planning, LoRA fine-tuning) are established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets + 10 baselines + ablation + efficiency analysis + interpretability cases.
- Writing Quality: ⭐⭐⭐⭐ RQ-driven with clear formalization.
- Value: ⭐⭐⭐⭐ Provides methodological inspiration for LLM applications in the KG field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)
- [\[AAAI 2026\] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](../../AAAI2026/graph_learning/s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)
- [\[AAAI 2026\] MyGram: Modality-aware Graph Transformer with Global Distribution for Multi-modal Entity Alignment](../../AAAI2026/graph_learning/mygram_modality-aware_graph_transformer_with_global_distribution_for_multi-modal.md)
- [\[ICLR 2026\] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](../../ICLR2026/graph_learning/pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)
- [\[ACL 2026\] CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning](craftqa_a_code-driven_adaptive_framework_for_complex_structured_data_reasoning.md)

</div>

<!-- RELATED:END -->
