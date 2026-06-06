---
title: >-
  [Paper Note] Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue
description: >-
  [ACL 2026][Dialogue Systems][Common Ground Establishment] This paper proposes the IndiRef benchmark to evaluate the ability of dialogue systems to establish and utilize persistent common ground through "relational refere…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Common Ground Establishment"
  - "Relational Reference"
  - "Situated Dialogue"
  - "Reinforcement Learning"
  - "Dialogue Memory"
date: 2026-05-08
content_hash: 12ac629b6f9bc318
---

# Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue

**Conference**: ACL 2026  
**arXiv**: [2601.09365](https://arxiv.org/abs/2601.09365)  
**Code**: [GitHub](https://github.com/biswesh/IndiRef)  
**Area**: Reinforcement Learning  
**Keywords**: Common Ground Establishment, Relational Reference, Situated Dialogue, Reinforcement Learning, Dialogue Memory

## TL;DR

This paper proposes the IndiRef benchmark to evaluate the ability of dialogue systems to establish and utilize persistent common ground through "relational references" (e.g., "the cafe next to the park we went to yesterday"). It finds that existing LLMs do not exceed 50% accuracy even under full-context conditions, and improves performance by 15-20% through synthetic data combined with GRPO reinforcement learning.

## Background & Motivation

**Background**: In dialogue, common ground refers to the shared knowledge, beliefs, and assumptions accumulated between participants. Recently, LLMs have demonstrated the ability to perform basic dialogue acts (such as confirmation and response), but whether these behaviors represent true understanding remains uncertain.

**Limitations of Prior Work**: (1) Existing LLMs may merely "simulate" understanding by generating plausible responses rather than actually establishing and utilizing common ground—a phenomenon termed the "illusion of understanding"; (2) As dialogue history grows, systems must rely on memory management techniques to retrieve information from established common ground, but existing methods (summarization, RAG, knowledge graphs) perform poorly when handling complex relational references; (3) There is a lack of effective benchmarks to measure the ability of dialogue systems to establish persistent, usable common ground.

**Key Challenge**: In situated dialogue, entities often lack unique referential expressions (e.g., the same room can be called "the room with the TV" or "the room in front of the bathroom"), and referential relationships involve multi-dimensional relational reasoning across space, time, and attributes. Existing representation methods fail to adequately capture these inter-entity relationships.

**Goal**: (1) Propose a benchmark based on relational reference resolution to evaluate the common ground establishment capabilities of dialogue systems; (2) Evaluate the effectiveness of common representation methods; (3) Improve the system's dialogue understanding through synthetic data and reinforcement learning.

**Key Insight**: Inspired by Kruijt and Vossen (2022), this work utilizes "relational references" common in human dialogue (referring to entities through spatial, temporal, or attribute-based relationships) as a probe for common ground capability—if a model can correctly resolve such references, it indicates the establishment of effective common ground.

**Core Idea**: Treat "complex relational reference resolution" as the core metric for measuring common ground establishment in dialogue systems, and enhance the multi-step reasoning capabilities of LLMs through synthetic situated dialogue data and GRPO training.

## Method

### Overall Architecture

The framework comprises three research questions: (1) Benchmarking—proposing the IndiRef benchmark consisting of 400 QA pairs based on relational references; (2) Representation Evaluation—comparing summarization, chunked retrieval, and ontological common ground representation methods under resource-constrained conditions; (3) Performance Enhancement—improving model performance through synthetic data generation and GRPO reinforcement learning. The input is the situated dialogue history, and the output is the correct answer to relational reference queries.

### Key Designs

1. **IndiRef Benchmark**:

    - **Function**: Evaluates the ability of dialogue systems to utilize common ground through relational references.
    - **Mechanism**: Based on two dialogue datasets (Meetup and Spot the Difference), 400 QA pairs (100 per category) were manually constructed, covering four reference types: Temporal (e.g., "The Thai restaurant we went to after watching Spider-Man"), Spatial (e.g., "The bottle on the table"), Attribute (e.g., "The yellow house"), and Inferential Common Ground (understanding implicit information). The design is adversarial—it includes multiple entities of the same type to prevent simple keyword matching and tests perspective taking through demonstrative pronouns (yours/mine).
    - **Design Motivation**: Existing benchmarks only test immediate dialogue acts (like confirmation) and cannot measure whether the system has truly established persistent, exploitable common ground.

2. **Comparison of Common Ground Representations (Writer-Reader-Generator Framework)**:

    - **Function**: Evaluates different common ground storage and retrieval methods in resource-constrained scenarios.
    - **Mechanism**: Adopts a $W$ (Write)-$R$ (Read)-$G$ (Generate) framework. Three methods are compared: (a) Summarization—compressing dialogue history into a summary $s_t$; (b) Chunked Retrieval—slicing dialogue into overlapping utterance chunks $c_i$ (7 utterances, stride 3) and retrieving the top-k most relevant chunks; (c) Ontology-based—using an Agent to extract entities, attributes, relations, and speaker information to build structured knowledge, retrieving information via multi-step queries (RAG[n]→Process→Final). Both sparse (BM25) and dense (NV-Embed-V2) embeddings were tested.
    - **Design Motivation**: Since the full history cannot fit into the context window in real-world long dialogue scenarios, it is necessary to evaluate which representation method best preserves relational information.

3. **Synthetic Data Generation + GRPO Training**:

    - **Function**: Addresses the scarcity of training data for situated dialogue and enhances model reasoning.
    - **Mechanism**: Employs an "Environment-First, Dialogue-Second" three-stage generation process: (a) Procedurally build a simulated world where two navigators explore and record spatio-temporal facts; (b) Use a script controller to synchronize the navigators' experiences and generate dialogue scripts, where the LLM is only responsible for generating utterances under script constraints; (c) Deterministically extract QA pairs from ground truth facts. After generating approximately 600 QA pairs, Llama 3.1-8B is trained using GRPO, providing positive rewards for correct answers.
    - **Design Motivation**: Existing LLMs lack training data for situated dialogue, and direct generation of dialogue by LLMs leads to unreliable reasoning; thus, reasoning logic is delegated to a procedural script controller.

### Loss & Training

Training is performed using Group Relative Policy Optimization (GRPO). The reward function is based on answer correctness—positive rewards are given when the model-generated answer matches the predefined ground truth. The training data consists of approximately 600 QA pairs from synthetic dialogue scenarios.

## Key Experimental Results

### Main Results

**Full Context Baselines (Performance of LLMs on IndiRef, FEM/LLM-as-Judge)**

| Model | Temporal Ref. | Spatial Ref. | Attribute Ref. | Inferential CG |
|------|---------|---------|---------|---------|
| Gemma2-2B | 0.20/0.18 | 0.18/0.16 | 0.24/0.26 | 0.26/0.16 |
| Llama3.1-8B | 0.38/0.32 | 0.46/0.38 | 0.46/0.44 | 0.20/0.20 |
| Gemma2-27B | 0.50/0.44 | 0.58/0.56 | 0.48/0.44 | 0.28/0.26 |
| Qwen-QWQ-32B | 0.38/0.32 | 0.52/0.38 | 0.44/0.40 | 0.40/0.40 |

**Comparison of Representations in Resource-Constrained Scenarios (Llama3.1-8B, Meetup)**

| Method | Temporal | Spatial | Attribute | Inference |
|------|------|------|------|------|
| Full Context Baseline | 0.38/0.32 | 0.46/0.38 | 0.46/0.44 | 0.20/0.20 |
| Summary | 0.32/0.28 | 0.34/0.26 | 0.30/0.25 | 0.28/0.18 |
| Chunk (NV-Embed) | 0.24/0.20 | 0.08/0.06 | 0.16/0.08 | 0.22/0.24 |
| Chunk (BM25) | 0.26/0.24 | 0.20/0.16 | 0.20/0.18 | 0.24/0.26 |
| Agent Ontology | 0.40/0.36 | 0.38/0.34 | 0.38/0.30 | 0.24/0.22 |

### Ablation Study

**Effect of GRPO Training (Llama3.1-8B)**

| Configuration | Temporal | Spatial | Attribute | Inference |
|------|------|------|------|------|
| Original (Full Context) | 0.38/0.32 | 0.46/0.38 | 0.46/0.44 | 0.20/0.20 |
| In-Context Learning | 0.60/0.56 | 0.58/0.54 | 0.62/0.58 | 0.42/0.34 |
| GRPO Training | 0.58/0.52 | 0.66/0.54 | 0.62/0.60 | 0.46/0.42 |

**Agent Ontology + GRPO Training**

| Configuration | Temporal | Spatial | Attribute | Inference |
|------|------|------|------|------|
| w/o GRPO | 0.40/0.36 | 0.38/0.34 | 0.38/0.30 | 0.24/0.22 |
| w/ GRPO | 0.48/0.46 | 0.44/0.42 | 0.52/0.44 | 0.36/0.38 |

### Key Findings

- Even under full-context conditions, the strongest model (Gemma2-27B) did not exceed 58% accuracy in any category, indicating that relational reference resolution is highly challenging for current LLMs.
- All resource-constrained representation methods underperformed the full-context baseline; information loss is the core issue.
- The Agent Ontology method outperformed summarization and chunking, suggesting that multi-step retrieval and explicit entity-relation modeling aid in understanding context.
- Reasoning-focused models (Qwen-QWQ) performed best in the Inferential Common Ground category (0.40) but average in others, and frequently suffered from hallucinations.
- GRPO training improved performance by 15-20% across both Meetup and STD datasets, proving that synthetic data training can transfer across different scenarios.

## Highlights & Insights

- Using "relational reference resolution" as a probe for common ground capability is an ingenious design—it transforms abstract "understanding" into a quantifiable QA task.
- The "Environment-First" approach to synthetic data generation is worth emulating—delegating reasoning logic to a procedural controller and language generation to the LLM ensures factual correctness of the data.
- The finding that sparse embeddings (BM25) slightly outperform dense embeddings for named entity retrieval provides valuable reference for RAG system design.

## Limitations & Future Work

- The IndiRef benchmark is small (400 QA pairs), and manual construction limits scalability.
- GRPO training was only conducted on an 8B parameter model; larger models might benefit more.
- The domain of the synthetic data is narrow (primarily navigation), and its generalizability to other situated dialogues remains to be verified.
- The Agent Ontology approach tends to merge information from different participants in scenarios with similar images (STD).

## Related Work & Insights

- **vs Dialog State Tracking (DST)**: DST uses slot-value pairs to represent task-oriented dialogue states but lacks the flexibility to handle inter-entity relations; the relational references in this work require richer representations.
- **vs Knowledge Graph Methods**: KGs can model entity relationships, but entities in situated dialogue often lack stable referential expressions. This paper's ontology approach partially addresses this through event logs and multi-step queries.
- **vs RAG Methods**: RAG relies on similarity retrieval, but in relational references, the semantics of the question may differ significantly from the segment containing the answer, leading to retrieval failure.

## Rating

- Novelty: ⭐⭐⭐⭐ Using "relational reference" as a probe for common ground is a unique perspective; the synthetic data method is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison of multiple representation methods and models, though the dataset size is small.
- Writing Quality: ⭐⭐⭐⭐⭐ Three research questions progress logically, experimental design is clear, and analysis is in-depth.
- Value: ⭐⭐⭐⭐ Reveals fundamental flaws in dialogue systems regarding common ground establishment, providing evaluation directions for embodied dialogue and social robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition](../../CVPR2026/dialogue/evolutionary_multimodal_reasoning_via_hierarchical_semantic_representation_for_i.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)
- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)
- [\[ACL 2026\] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue](context-agent_dynamic_discourse_trees_for_non-linear_dialogue.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)

</div>

<!-- RELATED:END -->
