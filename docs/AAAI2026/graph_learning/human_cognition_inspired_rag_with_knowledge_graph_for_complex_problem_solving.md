---
title: >-
  [Paper Note] Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving
description: >-
  [AAAI 2026][Graph Learning][RAG] This paper proposes CogGRAG, a human cognition-inspired knowledge graph-based RAG framework that substantially improves LLM accuracy and reliability on complex Knowledge Graph Question An…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "RAG"
  - "Knowledge Graph"
  - "Question Answering"
  - "Mind Map"
  - "Self-Verification"
date: 2026-05-08
content_hash: b20284767703fda1
---

# Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving

**Conference**: AAAI 2026
**arXiv**: [2503.06567](https://arxiv.org/abs/2503.06567)  
**Code**: [https://github.com/cy623/RAG.git](https://github.com/cy623/RAG.git)  
**Area**: Graph Learning
**Keywords**: RAG, Knowledge Graph, Question Answering, Mind Map, Self-Verification

## TL;DR

This paper proposes CogGRAG, a human cognition-inspired knowledge graph-based RAG framework that substantially improves LLM accuracy and reliability on complex Knowledge Graph Question Answering (KGQA) tasks through three stages: top-down mind map decomposition, hierarchical structured retrieval, and dual-LLM self-verification reasoning.

## Background & Motivation

### Limitations of Prior Work in RAG

LLMs continue to face challenges in knowledge integration and complex reasoning, and remain prone to hallucination. While RAG mitigates some of these issues by incorporating external knowledge, existing methods have the following fundamental limitations:

**Limitations of traditional RAG (vector similarity)**:
   - Treats knowledge fragments as independent units, failing to capture contextual dependencies and semantic relationships
   - Does not support multi-step reasoning, making it difficult to handle complex questions

**Limitations of existing Graph RAG methods**:
   - **Lack of holistic reasoning structure**: Methods such as Graph-CoT and GNN-RAG employ iterative/sequential reasoning pipelines, where each step depends on the previous one, causing errors to propagate progressively and become unrecoverable
   - **Lack of verification mechanisms**: In the face of retrieval errors or insufficient knowledge coverage, LLMs may still generate inaccurate responses without the ability to self-detect or self-correct

### Core Idea of CogGRAG

Inspired by the Dual-Process Theory of human cognition:
- **System 1**: Intuitive reasoning (rapid answer generation)
- **System 2**: Monitoring and error correction (verifying answer correctness)

CogGRAG unifies question decomposition, knowledge retrieval, and reasoning within a single graph-structured cognitive framework:

**Plan before execute**: A complete reasoning plan (mind map) is constructed prior to retrieval, rather than reasoning while retrieving

**One-shot retrieval instead of iterative retrieval**: All required knowledge is retrieved at once based on a global mind map

**Dual-LLM verification**: A reasoning LLM and a verification LLM collaborate, simulating human self-reflection

## Method

### Overall Architecture

CogGRAG consists of three stages:
1. Top-down Decomposition → constructing a tree-structured mind map
2. Structured Knowledge Retrieval → local + global two-level retrieval
3. Bottom-up Reasoning with Self-Verification → dual-LLM collaboration

### Key Designs

#### 1. **Top-Down Question Decomposition**: Constructing a Tree-Structured Mind Map

**Function**: Recursively decomposes the original complex question into hierarchically organized, semantically coherent sub-questions, forming a mind map $\mathcal{M}$.

**Mechanism**:
- Each node in the mind map is a tuple $m = (q, t, s)$: sub-question $q$, depth level $t$, status $s \in \{\text{Continue}, \text{End}\}$
- LLM is used for recursive decomposition:
$$\{(q_j^{t+1}, s_j^{t+1})\}_{j=1}^N = \text{Decompose}(q^t, p_\theta, \text{prompt}_{dec})$$
- Decomposition stops when all leaf nodes are marked as End, indicating atomic questions have been reached

**Design Motivation** (illustrated by example):
- Question: "When did the football manager who recruited Beckham manage Manchester United?"
- Traditional RAG cannot resolve the intermediate entity "Ferguson" (not directly mentioned in the question)
- CogGRAG decomposes into: "Who recruited Beckham?" → "When did that manager coach Manchester United?"
- Decomposition surfaces key intermediate entities and constructs a more accurate reasoning path

**Advantages**: Planning is completed before retrieval and reasoning, explicitly separating "planning" from "execution," enabling global reasoning and reducing error propagation.

#### 2. **Structured Knowledge Retrieval**: Local + Global Two-Level Retrieval Strategy

**Function**: Retrieves external knowledge supporting reasoning from the knowledge graph $\mathcal{G}$ based on the mind map.

**Two-level information extraction**:
- **Local-level**: Entities, entity-relation pairs, and triples associated with individual sub-questions
    - e.g., "Which manager recruited Beckham?" → entity (David Beckham), triple (manager, recruited, David Beckham)
- **Global-level**: Semantic dependencies across multiple sub-questions, represented as interconnected subgraphs
    - e.g., [(manager, recruited, David Beckham), (manager, manage, Manchester United)]

**Retrieval pipeline**:
1. LLM extracts key information from the mind map: $\mathcal{K} = \text{Extract}(\mathcal{M}, p_\theta, \text{prompt}_{ext})$
2. For each entity $e \in \mathcal{K}$, expand to its neighborhood in $\mathcal{G}$ to obtain a candidate triple set $\tilde{\mathcal{T}}$
3. Semantic similarity filtering: $\mathcal{T} = \{\tau \in \tilde{\mathcal{T}}, k \in \mathcal{K} \mid \text{sim}(\tau, k) > \varepsilon\}$

**Design Motivation**: Unlike step-by-step retrieval in methods such as Graph-CoT, CogGRAG performs one-shot retrieval based on the global mind map, avoiding error accumulation from sequential retrieval failures while ensuring retrieval completeness and contextual coherence.

#### 3. **Bottom-Up Reasoning with Dual-LLM Self-Verification**: Simulating Human Cognitive Reflection

**Function**: Answers sub-questions bottom-up starting from leaf nodes; each answer is reviewed by a verification LLM, and incorrect answers are regenerated.

**Dual-LLM architecture**:
- **$\text{LLM}_{res}$**: Responsible for bottom-up reasoning and answer generation based on the mind map and retrieved triple set
- **$\text{LLM}_{ver}$**: Responsible for evaluating the validity of generated answers (consistency, factual grounding, logical coherence)

**Reasoning pipeline**:
1. Starting from the deepest sub-questions, generate candidate answers:
$$a^t = \text{LLM}_{res}(\mathcal{T}, q^t, \hat{\mathcal{M}}, \text{prompt}_{res})$$
where $\hat{\mathcal{M}}$ denotes the set of sub-questions that have been answered and verified.

2. The verification module judges:
$$\delta^t = \text{LLM}_{ver}(q^t, a^t, \hat{\mathcal{M}}, \text{prompt}_{ver})$$

3. If verification fails, re-reasoning is performed:
$$\hat{a}^t = \begin{cases} \text{LLM}_{res}(\mathcal{T}, q^t, \hat{\mathcal{M}}, \text{prompt}_{rethink}) & \text{if } \delta^t = \text{False} \\ a^t & \text{otherwise} \end{cases}$$

4. Recursively aggregate to the root node: $A = \hat{a}^0$

**Selective abstention mechanism**: When the reasoning LLM cannot produce a reliable answer based on $\mathcal{T}$, it explicitly returns "I don't know" rather than hallucinating.

## Key Experimental Results

### Main Results

#### Three General KGQA Benchmarks (backbone: LLaMA2-13B)

| Method Type | Method | HotpotQA (RL/EM/F1) | CWQ (RL/EM/F1) | WebQSP (RL/EM/F1) |
|-------------|--------|---------------------|----------------|-------------------|
| LLM-only | Direct | 19.1/17.3/18.7 | 31.4/28.8/31.7 | 51.4/47.9/53.5 |
| LLM-only | CoT | 23.3/20.8/22.1 | 35.1/32.7/33.5 | 55.2/51.6/55.3 |
| LLM+KG | CoT+KG | 28.7/25.4/26.9 | 42.2/37.6/40.8 | 52.8/48.1/50.5 |
| Graph RAG | ToG | 29.3/26.4/29.6 | 49.1/46.1/47.7 | 54.6/57.4/56.1 |
| Graph RAG | RoG | 30.7/28.1/30.4 | 55.3/51.8/54.7 | **65.2/62.8/67.2** |
| Graph RAG | GoG | 31.5/30.1/31.1 | 55.7/52.4/54.8 | 65.5/59.1/63.6 |
| **Ours** | **CogGRAG** | **34.4/30.7/35.5** | **56.3/53.4/55.8** | 59.8/56.1/58.9 |

CogGRAG achieves overall best performance on HotpotQA and CWQ. On WebQSP it is surpassed by RoG; the authors attribute this to data leakage due to the dataset's wide usage.

#### Different Backbone Models

| Backbone | HotpotQA RL | CWQ RL | WebQSP RL |
|----------|-------------|--------|-----------|
| CogGRAG w/ Qwen2.5-7B | 28.4 | 50.5 | 53.2 |
| CogGRAG w/ LLaMA3-8B | 32.1 | 53.5 | 57.2 |
| CogGRAG w/ LLaMA2-13B | 34.4 | 56.3 | 59.8 |
| CogGRAG w/ Qwen2.5-32B | **40.5** | **66.5** | **74.1** |

Performance improves steadily with model scale, demonstrating good scalability of the method.

#### Domain-Specific Dataset (GRBENCH, backbone: LLaMA2-13B)

| Method | E-commerce (RL/EM/F1) | Literature | Academic | Healthcare |
|--------|-----------------------|------------|----------|------------|
| LLaMA2-13B | 7.1/6.8/6.9 | 5.4/5.1/5.3 | 5.4/4.7/5.1 | 4.3/3.1/3.6 |
| Graph-CoT | 26.4/24.0/25.3 | 26.7/23.3/24.9 | 19.3/14.8/16.9 | **28.1/25.2/26.7** |
| **CogGRAG** | **30.2/28.7/29.5** | **32.4/30.1/31.3** | **23.6/21.5/22.7** | 27.4/25.6/26.2 |

Strong performance is maintained on non-general-domain KGs. The near-trivial performance of standalone LLMs (5–7%) confirms the critical importance of external knowledge graphs.

### Ablation Study

| Configuration | HotpotQA | CWQ | WebQSP | Note |
|---------------|----------|-----|--------|------|
| CogGRAG-nd (w/o decomposition) | significant drop | significant drop | significant drop | decomposition contributes most |
| CogGRAG-ng (w/o global retrieval) | moderate drop | moderate drop | moderate drop | global retrieval aids cross-subquestion association |
| CogGRAG-nv (w/o verification) | slight drop | slight drop | slight drop | verification improves reliability |
| **CogGRAG (full)** | **best** | **best** | **best** | all three modules are indispensable |

The decomposition module contributes most substantially, confirming the validity of the "plan before execute" paradigm.

#### Hallucination Evaluation (HotpotQA)

| Method | Correct Rate ↑ | Abstention Rate ↑ | Hallucination Rate ↓ |
|--------|---------------|-------------------|----------------------|
| LLaMA2-13B | 19.1% | 25.7% | **55.2%** |
| ToG | 29.3% | 20.2% | 50.5% |
| MindMap | 27.9% | 22.4% | 49.7% |
| **CogGRAG** | **34.4%** | **40.6%** | **25.0%** |

CogGRAG substantially reduces the hallucination rate to 25%, while raising the abstention rate from ~20% to 40.6%, indicating that the model has learned to respond with "I don't know" when knowledge is insufficient rather than fabricating answers.

### Key Findings

1. Decomposing complex questions into reasoning plans is critical for KGQA
2. Graph RAG methods substantially outperform simple LLM+KG approaches, particularly on complex questions
3. The self-verification mechanism effectively reduces the hallucination rate by 50%
4. One-shot global retrieval avoids the error accumulation associated with iterative retrieval

## Highlights & Insights

1. **Cognitive science inspiration**: Integrating Dual-Process Theory into the RAG framework reflects a theoretically grounded design rather than a purely engineering trick
2. **Decoupling planning from execution**: First constructing a mind map, then performing one-shot retrieval, and finally reasoning bottom-up — the three-stage decoupling makes errors controllable
3. **Selective abstention**: Teaching LLMs to say "I don't know" is more practical than merely reducing hallucinations
4. **Controllable inference time**: Despite the introduction of self-verification, one-shot retrieval avoids iterative overhead, keeping total inference time comparable to baselines

## Limitations & Future Work

1. The dual-LLM architecture increases inference cost, requiring two models to be run
2. Decomposition quality depends on LLM capability; weaker models may produce low-quality mind maps
3. CogGRAG does not surpass RoG on WebQSP, possibly because the dataset is too simple to benefit from complex decomposition
4. The similarity threshold $\varepsilon = 0.7$ is manually set; adaptive threshold selection warrants exploration
5. The verification mechanism allows only one rethink attempt; multi-round iterative verification could be explored

## Related Work & Insights

- **ToG (Think-on-Graph)**: Iterative beam-search-based reasoning over knowledge graph paths
- **Graph-CoT**: Iterative graph reasoning framework, whose core weakness is error propagation
- **MindMap**: Although the name is similar, the nature is fundamentally different — MindMap is a knowledge organization tool, whereas the mind map in CogGRAG is a reasoning planning tool
- **GoT (Graph of Thoughts)**: Models LLM reasoning as an arbitrary graph but does not involve external knowledge retrieval

## Rating

- Novelty: ⭐⭐⭐⭐ — The unified framework design combining cognitive inspiration, decomposition, retrieval, and reasoning is original, though individual components are not entirely novel in isolation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, multiple backbones, ablation study, hallucination analysis, and inference time analysis constitute a very comprehensive evaluation
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich figures and tables
- Value: ⭐⭐⭐⭐ — Provides an effective framework for combined RAG + KG reasoning, though additional KG infrastructure is required

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG](../../NeurIPS2025/graph_learning/remindrag_low-cost_llm-guided_knowledge_graph_traversal_for_efficient_rag.md)
- [\[CVPR 2026\] Graph-to-Frame RAG: Visual-Space Knowledge Fusion for Training-Free and Auditable Video Reasoning](../../CVPR2026/graph_learning/graph-to-frame_rag_visual-space_knowledge_fusion_for_training-free_and_auditable.md)
- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](../../ACL2026/graph_learning/graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[ACL 2026\] ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection](../../ACL2026/graph_learning/compliancenlp_knowledge-graph-augmented_rag_for_multi-framework_regulatory_gap_d.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](../../NeurIPS2025/graph_learning/gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)

</div>

<!-- RELATED:END -->
