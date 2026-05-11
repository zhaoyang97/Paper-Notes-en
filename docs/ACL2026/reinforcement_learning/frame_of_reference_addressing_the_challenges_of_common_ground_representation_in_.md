---
title: >-
  [Paper Note] Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue
description: >-
  [ACL 2026][Reinforcement Learning][Common Ground] This paper introduces the IndiRef benchmark for evaluating dialogue systems' ability to establish and exploit persistent common ground through "relational references" (e.…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Common Ground"
  - "Relational Reference"
  - "Situated Dialogue"
  - "Dialogue Memory"
date: 2026-05-08
content_hash: 18a27691151a2c3b
---

# Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue

**Conference**: ACL 2026
**arXiv**: [2601.09365](https://arxiv.org/abs/2601.09365)
**Code**: [GitHub](https://github.com/biswesh/IndiRef)
**Area**: Reinforcement Learning
**Keywords**: Common Ground, Relational Reference, Situated Dialogue, Reinforcement Learning, Dialogue Memory

## TL;DR

This paper introduces the IndiRef benchmark for evaluating dialogue systems' ability to establish and exploit persistent common ground through "relational references" (e.g., "the café next to the park we visited yesterday"). Experiments show that existing LLMs achieve no more than 50% accuracy even under full-context conditions, and a combination of synthetic data generation and GRPO reinforcement learning training yields performance improvements of 15–20%.

## Background & Motivation

**Background**: In dialogue, common ground refers to the accumulated shared knowledge, beliefs, and assumptions among conversation participants. Recent LLMs have demonstrated the ability to perform certain basic dialogue acts (e.g., acknowledgment, response), yet whether these behaviors reflect genuine understanding remains uncertain.

**Limitations of Prior Work**: (1) Existing LLMs may merely simulate understanding by generating plausible responses rather than truly establishing and utilizing common ground—an "illusion of understanding"; (2) As dialogue history grows, systems must rely on memory management techniques to retrieve information from established common ground, but existing approaches (summarization, RAG, knowledge graphs) perform poorly when handling complex relational references; (3) Effective benchmarks for measuring a dialogue system's ability to build persistent, usable common ground are lacking.

**Key Challenge**: In situated dialogue, entities often lack a unique referring expression (e.g., the same room may be called "the room with the TV" or "the room in front of the bathroom"), and reference resolution involves multi-dimensional relational reasoning across spatial, temporal, and attributive dimensions. Existing representation methods fail to adequately capture these inter-entity relationships.

**Goal**: (1) Propose a benchmark based on relational reference resolution to evaluate dialogue systems' common ground establishment capabilities; (2) Assess the effectiveness of commonly used common ground representation methods; (3) Improve dialogue understanding through synthetic data and reinforcement learning.

**Key Insight**: Inspired by Kruijt and Vossen (2022), the paper leverages "relational references" frequent in human dialogue—where entities are referred to via spatial, temporal, or attributive relations—as a probe for common ground capability. If a model can correctly resolve such references, it demonstrably maintains effective common ground.

**Core Idea**: Treating the resolution of complex relational references as the central metric for evaluating dialogue systems' common ground establishment, and enhancing LLMs' multi-step reasoning capacity via synthetic situated dialogue data combined with GRPO training.

## Method

### Overall Architecture

The framework addresses three research questions: (1) **Benchmarking**—introducing the IndiRef benchmark comprising 400 relational reference-based question–answer pairs; (2) **Representation evaluation**—comparing summarization, chunk-based retrieval, and ontology-based methods for common ground representation under resource-constrained conditions; (3) **Performance improvement**—enhancing model performance through synthetic data generation and GRPO reinforcement learning training. The input is a situated dialogue history; the output is a correct answer to a relational reference question.

### Key Designs

1. **IndiRef Benchmark**:

    - **Function**: Evaluates dialogue systems' ability to leverage common ground through relational references.
    - **Mechanism**: Built upon two dialogue datasets (Meetup and Spot the Difference), the benchmark consists of 400 manually constructed question–answer pairs (100 per category), covering four reference types: temporal reference (e.g., "the Thai restaurant we went to after watching Spider-Man"), spatial reference (e.g., "the bottle on the table"), attributive reference (e.g., "the yellow house"), and inferential common ground (understanding of implied information). The benchmark is designed adversarially—containing multiple entities of the same type to prevent simple keyword matching—and tests perspective-taking ability through deictic expressions (your/my).
    - **Design Motivation**: Existing benchmarks only assess immediate dialogue acts (e.g., acknowledgment) and cannot measure whether systems have truly established persistently usable common ground.

2. **Common Ground Representation Comparison (Writer–Reader–Generator Framework)**:

    - **Function**: Evaluates different common ground storage and retrieval methods under resource-constrained scenarios.
    - **Mechanism**: A $W$ (Write)–$R$ (Read)–$G$ (Generate) framework is adopted, comparing three methods: (a) Summarization—compressing dialogue history into a summary $s_t$; (b) Chunk-based retrieval—segmenting dialogue into overlapping utterance chunks $c_i$ (7 utterances, stride 3) and retrieving the top-k most relevant chunks; (c) Ontology-based method—using an agent to extract entities, attributes, relations, and speaker information into structured knowledge, with multi-step querying (RAG[n]→Process→Final) for retrieval. Both sparse (BM25) and dense (NV-Embed-V2) embedding strategies are evaluated.
    - **Design Motivation**: In real long-dialogue scenarios, the full history cannot fit within the context window, making it necessary to assess which representation method best preserves relational information.

3. **Synthetic Data Generation + GRPO Training**:

    - **Function**: Addresses the scarcity of situated dialogue training data and enhances model reasoning.
    - **Mechanism**: An "environment-first, dialogue-second" three-stage generation pipeline is employed: (a) Simulated worlds are programmatically constructed, with two navigators exploring and recording spatiotemporal facts; (b) A script controller synchronizes the navigators' experiences and generates dialogue scripts, with LLMs responsible only for producing utterances within script constraints; (c) Question–answer pairs are deterministically extracted from ground-truth facts. Approximately 600 question–answer pairs are generated, after which GRPO is used to train Llama 3.1-8B, rewarding correct answers.
    - **Design Motivation**: Existing LLMs lack training data for situated dialogue, and generating dialogue directly with LLMs produces unreliable reasoning; therefore, the reasoning logic is delegated to a programmatic script controller.

### Loss & Training

GRPO (Group Relative Policy Optimization) is used for training. The reward function is based on answer correctness—a positive reward is granted when the model's generated answer matches the predefined ground-truth answer. Training data comprises approximately 600 question–answer pairs derived from synthetic dialogue scenarios.

## Key Experimental Results

### Main Results

**Full-Context Baseline (LLM Performance on IndiRef, FEM/LLM-as-Judge)**

| Model | Temporal | Spatial | Attributive | Inferential |
|-------|----------|---------|-------------|-------------|
| Gemma2-2B | 0.20/0.18 | 0.18/0.16 | 0.24/0.26 | 0.26/0.16 |
| Llama3.1-8B | 0.38/0.32 | 0.46/0.38 | 0.46/0.44 | 0.20/0.20 |
| Gemma2-27B | 0.50/0.44 | 0.58/0.56 | 0.48/0.44 | 0.28/0.26 |
| Qwen-QWQ-32B | 0.38/0.32 | 0.52/0.38 | 0.44/0.40 | 0.40/0.40 |

**Comparison of Representation Methods under Resource-Constrained Conditions (Llama3.1-8B, Meetup)**

| Method | Temporal | Spatial | Attributive | Inferential |
|--------|----------|---------|-------------|-------------|
| Full-context baseline | 0.38/0.32 | 0.46/0.38 | 0.46/0.44 | 0.20/0.20 |
| Summarization | 0.32/0.28 | 0.34/0.26 | 0.30/0.25 | 0.28/0.18 |
| Chunk (NV-Embed) | 0.24/0.20 | 0.08/0.06 | 0.16/0.08 | 0.22/0.24 |
| Chunk (BM25) | 0.26/0.24 | 0.20/0.16 | 0.20/0.18 | 0.24/0.26 |
| Agent Ontology | 0.40/0.36 | 0.38/0.34 | 0.38/0.30 | 0.24/0.22 |

### Ablation Study

**Effect of GRPO Training (Llama3.1-8B)**

| Configuration | Temporal | Spatial | Attributive | Inferential |
|---------------|----------|---------|-------------|-------------|
| Original (full-context) | 0.38/0.32 | 0.46/0.38 | 0.46/0.44 | 0.20/0.20 |
| In-Context Learning | 0.60/0.56 | 0.58/0.54 | 0.62/0.58 | 0.42/0.34 |
| GRPO Training | 0.58/0.52 | 0.66/0.54 | 0.62/0.60 | 0.46/0.42 |

**Agent Ontology + GRPO Training**

| Configuration | Temporal | Spatial | Attributive | Inferential |
|---------------|----------|---------|-------------|-------------|
| Without GRPO | 0.40/0.36 | 0.38/0.34 | 0.38/0.30 | 0.24/0.22 |
| With GRPO | 0.48/0.46 | 0.44/0.42 | 0.52/0.44 | 0.36/0.38 |

### Key Findings

- Even under full-context conditions, the strongest model (Gemma2-27B) does not exceed 58% accuracy in any category, demonstrating that relational reference resolution poses a fundamental challenge for current LLMs.
- All resource-constrained representation methods underperform the full-context baseline, with information loss being the central bottleneck.
- The agent ontology method outperforms both summarization and chunk-based retrieval, indicating that multi-step retrieval and explicit entity–relation modeling contribute to contextual understanding.
- The reasoning-oriented model (Qwen-QWQ) achieves the best performance on the inferential common ground category (0.40) but performs comparably or worse on other categories and frequently produces hallucinations.
- GRPO training yields 15–20% improvements on both the Meetup and STD datasets, demonstrating that synthetic data training transfers across different scenarios.

## Highlights & Insights

- Using "relational reference resolution" as a probe for common ground capability is an elegant design choice—it operationalizes the abstract notion of "understanding" into a quantifiable QA task.
- The "environment-first" approach to synthetic data generation is noteworthy: delegating reasoning logic to a programmatic controller while assigning language generation to LLMs ensures factual correctness in the data.
- The finding that sparse embeddings (BM25) slightly outperform dense embeddings for named entity retrieval offers a useful reference for RAG system design.

## Limitations & Future Work

- The IndiRef benchmark is relatively small (400 question–answer pairs), and manual construction limits its scalability.
- GRPO training is conducted only on an 8B-parameter model; larger models may yield greater benefits.
- The synthetic data covers a narrow domain (primarily navigation scenarios), and generalization to other situated dialogue contexts remains to be verified.
- The agent ontology method tends to conflate information from different participants in similar-image scenarios (STD).

## Related Work & Insights

- **vs. Dialog State Tracking (DST)**: DST represents task-oriented dialogue states using slot–value pairs but lacks the flexibility to handle inter-entity relations; the relational references in this paper demand richer representations.
- **vs. Knowledge Graph Methods**: Knowledge graphs can model entity relations, but entities in situated dialogue often lack stable referring expressions; the proposed ontology-based method partially addresses this through event logs and multi-step querying.
- **vs. RAG Methods**: RAG relies on similarity-based retrieval, but in relational reference scenarios the semantics of the question and those of the relevant passage may diverge substantially, leading to retrieval failures.

## Rating

- Novelty: ⭐⭐⭐⭐ Using "relational references" as a probe for common ground capability is a distinctive perspective; the synthetic data generation method is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple representation methods and models are compared, though the dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The three research questions are organized progressively; experimental design is clear and analysis is thorough.
- Value: ⭐⭐⭐⭐ Exposes fundamental deficiencies in dialogue systems' common ground establishment, providing an evaluation direction for embodied dialogue and social robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)
- [\[ACL 2026\] A Survey of Reinforcement Learning for Large Language Models under Data Scarcity: Challenges and Solutions](a_survey_of_reinforcement_learning_for_large_language_models_under_data_scarcity.md)
- [\[ACL 2026\] UniCreative: Unifying Long-form Logic and Short-form Sparkle via Reference-Free Reinforcement Learning](unicreative_unifying_long-form_logic_and_short-form_sparkle_via_reference-free_r.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](../../ICLR2026/reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)

</div>

<!-- RELATED:END -->
