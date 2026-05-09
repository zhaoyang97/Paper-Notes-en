---
title: >-
  [Paper Note] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction
description: >-
  [AAAI 2026][Reinforcement Learning][RAG] This paper proposes TAdaRAG, a task-adaptive RAG framework that performs on-the-fly knowledge graph construction via intent-driven template routing, supervised fine-tuning, and REINFORCE-based reinforcement learning. It addresses three core limitations of conventional RAG—chunking-induced hallucination, broken reasoning chains, and irrelevant information interference—achieving state-of-the-art performance on 6 public benchmarks and 1 commercial scenario benchmark.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - RAG
  - Knowledge Graph
  - Task Adaptation
  - REINFORCE
  - Long-context Understanding
date: 2026-05-08
content_hash: c41d8df965679063
---

# TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction

**Conference**: AAAI 2026
**arXiv**: [2511.12520](https://arxiv.org/abs/2511.12520)
**Code**: [github.com/IAAR-Shanghai/TAdaRAG](https://github.com/IAAR-Shanghai/TAdaRAG)
**Area**: Reinforcement Learning
**Keywords**: RAG, Knowledge Graph, Task Adaptation, REINFORCE, Long-context Understanding

## TL;DR

This paper proposes TAdaRAG, a task-adaptive RAG framework that performs on-the-fly knowledge graph construction via intent-driven template routing, supervised fine-tuning, and REINFORCE-based reinforcement learning. It addresses three core limitations of conventional RAG—chunking-induced hallucination, broken reasoning chains, and irrelevant information interference—achieving state-of-the-art performance on 6 public benchmarks and 1 commercial scenario benchmark.

## Background & Motivation

### State of the Field

Retrieval-Augmented Generation (RAG) enhances LLM output quality by retrieving external knowledge and represents the dominant paradigm for mitigating hallucination. However, current RAG systems face three critical bottlenecks in practical deployment:

### Three Core Problems

**Problem 1: Hallucination from Chunk Truncation.** Long retrieved documents must be split into chunks due to input window constraints, causing information loss from complete knowledge. For example, when a legal clause is split into multiple chunks, each chunk is incomplete, preventing the model from correctly integrating information and leading to factual errors.

**Problem 2: Broken Reasoning Chains.** Discrete chunks fail to capture the inherent logical relationships within a corpus. In multi-hop reasoning tasks (e.g., HotpotQA, 2WikiMQA), answers require cross-document reasoning, but independent chunks lack structured logical connections, resulting in incoherent reasoning chains.

**Problem 3: Irrelevant Information Interference.** Conventional RAG retrieves unstructured text containing numerous details irrelevant to the query, which interferes with the model's ability to extract key information and degrades practical utility.

### Distinction from Existing Graph-Enhanced RAG

Methods such as GraphRAG, HippoRAG, and PathRAG leverage knowledge graphs to organize information, but they **rely on pre-built KGs**—requiring manual maintenance, lacking scalability, and exhibiting poor adaptability to new domains or tasks. The core innovation of TAdaRAG lies in **integrating KG construction into the inference process (rather than the retrieval stage)**, enabling real-time, task-adaptive dynamic knowledge graph generation.

## Method

### Overall Architecture

TAdaRAG adopts a two-stage training pipeline:
- **Stage 1**: Supervised knowledge extraction fine-tuning (SFT) — learns high-quality KG extraction capabilities.
- **Stage 2**: Task-adaptive KG construction (RL) — optimizes KG construction via the REINFORCE algorithm to maximize downstream task performance.

At inference time, the model dynamically constructs a task-adaptive KG conditioned on the query and retrieved documents, which is then integrated into the generation pipeline to produce the final answer.

### Key Designs

#### 1. **Intent-Driven Template Routing Mechanism**

Pre-trained language models frequently produce irrelevant or redundant entities during entity extraction, particularly in industrial settings. TAdaRAG designs domain-specific extraction templates to standardize knowledge extraction:

- The application domain of the input text (health, legal, news, etc.) is first identified.
- Intent detection is performed via prompting to select the corresponding template $t$.
- The template specifies: which entity types are required for the domain, entity description norms, and definitions of inter-entity relations.

An instruction set $I = \{q, r, t\}$ (query, external knowledge, template) is constructed, and a strong LLM (GPT-4o/DeepSeek) is invoked to perform knowledge extraction and generate high-quality KGs.

Based on this procedure, an SFT dataset is constructed covering 4 question domains, 7 sub-datasets, and 9,548 samples in total.

#### 2. **Parallel Subgraph Construction + Mixing Network**

The model constructs $p$ parallel subgraphs $g_i = \{g_i^1, g_i^2, \ldots, g_i^p\}$ for each input, using learnable tokens `<|startextraction|>` and `<|endextraction|>` to demarcate the knowledge extraction boundaries.

A **Mixing Network** is employed to fuse information from representations with and without KG:

Given an instruction-answer pair $(x_i, y_i)$ and subgraph $g_i^k$:

1. Compute the graph-free hidden state $H_{i,j}^{\text{base}}$ and the graph-conditioned hidden state $H_{i,j,k}^{\text{graph}}$.
2. Compute the fusion weight via a three-layer MLP with ReLU activations:
    $\omega_{i,j,k} = \text{MLP}(\text{concat}(H_{i,j}^{\text{base}}, H_{i,j,k}^{\text{graph}}))$
3. Compute the weighted combination of log-likelihoods:
    $l_{i,j,k}^{\text{mix}} = \omega_{i,j,k} \cdot l_{i,j,k}^{\text{w/ graph}} + (1-\omega_{i,j,k}) \cdot l_{i,j,k}^{\text{w/o graph}}$

This enables the model to autonomously determine when to rely on the KG and when to answer directly.

#### 3. **REINFORCE Optimization for Graph Construction**

The objective is to find the optimal subgraph $\tilde{g}^{(i)}$ that maximizes $\pi_\theta(y_i | x_i, \tilde{g}^{(i)})$.

The reward function is defined as:

$$R_{i,k} = \max(0, \mathcal{L}_i^{\text{base}} - \mathcal{L}_{i,k}^{\text{graph}} - \bar{R}_i)$$

Intuitively, a positive reward is granted when incorporating the KG reduces the loss beyond the average improvement across all subgraphs. $\bar{R}_i$ serves as a baseline computed from the mean gain over all subgraphs.

The REINFORCE loss is:
$$\mathcal{L}^{\text{REINFORCE}} = -R_{i,k} \cdot \log\pi_\theta(g_i^k | x_i)$$

### Loss & Training

The overall loss function is:

$$\mathcal{L} = \alpha \cdot \mathcal{L}^{\text{base}} + (1-\alpha) \cdot \mathcal{L}^{\text{graph}} + \beta \cdot \mathcal{L}^{\text{REINFORCE}}$$

- $\mathcal{L}^{\text{base}}$: loss for direct answering without KG (preserves the model's independent answering capability).
- $\mathcal{L}^{\text{graph}}$: loss for KG-assisted answering (learns to integrate KG information).
- $\mathcal{L}^{\text{REINFORCE}}$: optimizes the quality of KG construction.

Training configuration:
- Backbone models: Mistral-7B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct.
- Stage 1 (SFT): 5 epochs, maximum sequence length 20480, learning rate 5e-5, cosine schedule.
- Stage 2 (RL): 3 epochs, learning rate 5e-7, ZeRO-2, AdamW, bfloat16.
- Sampling temperature $T=0.6$ (training), greedy decoding (evaluation).
- Maximum KG length: 2048 tokens.
- Total training time: approximately 16 hours (8×A100 80GB); SFT 4 hours, RL 12 hours.

## Key Experimental Results

### Main Results

**Based on Mistral-7B-Instruct (F1 / ROUGE-L)**:

| Method | Health | Biology | Legal | HotpotQA | 2WikiMQA | GovReport |
|--------|--------|---------|-------|----------|----------|-----------|
| NaïveRAG | 34.80 | 34.10 | 35.80 | 37.60 | 20.60 | 27.40 |
| GraphRAG | 35.60 | 34.80 | 37.65 | 38.00 | 36.50 | 25.60 |
| MEMORAG | 37.40 | 35.70 | 51.20 | 42.90 | 30.30 | 31.60 |
| **TAdaRAG (w/ reinforce)** | **40.77*** | **39.31*** | 49.88 | **44.83*** | **39.31*** | **36.41*** |

**Based on Qwen2.5-7B-Instruct**:

| Method | Health | Biology | Legal | HotpotQA | 2WikiMQA | GovReport |
|--------|--------|---------|-------|----------|----------|-----------|
| MEMORAG | 36.87 | 36.00 | 47.60 | 37.99 | 35.32 | 31.13 |
| **TAdaRAG (w/ reinforce)** | **42.38*** | **40.75*** | 46.83 | **49.23*** | **43.79*** | **36.95*** |

\* denotes statistically significant improvement at $p < 0.01$.

### Ablation Study

| Stage | Health | Biology | 2WikiMQA | GovReport | Note |
|-------|--------|---------|----------|-----------|------|
| NaïveRAG | 34.80 | 34.10 | 20.60 | 27.40 | Baseline |
| w/ graph (prompted KG) | 38.19 | 36.87 | 38.48 | 33.72 | Prompting alone yields substantial gains |
| w/ sft (SFT fine-tuning) | 40.00 | 38.92 | 38.86 | 35.39 | SFT further improves extraction quality |
| **w/ reinforce (full)** | **40.77** | **39.31** | **39.31** | **36.41** | RL achieves best performance across all datasets |

**KG scale variation across training stages** (Mistral-7B):

| Stage | Health Graph Size | Health Entity Count | HotpotQA Graph Size | HotpotQA Entity Count |
|-------|------------------|--------------------|--------------------|----------------------|
| Base (prompted) | 7303 | 58.3 | 1894 | 16.1 |
| SFT | 5146 | 50.0 | 573 | 12.5 |
| Reinforce | **2006** | **44.2** | **257** | **10.1** |

As training progresses, the KG becomes increasingly compact — REINFORCE learns to retain only task-relevant key information.

### Key Findings

1. **KG construction evolves from coarse to fine**: graph size decreases substantially from Stage 1 to Stage 2 (7303→2006) and entity count also drops (58→44), while performance continues to improve — RL effectively learns to "distill."
2. **Largest gains on 2WikiMQA** (20.60→39.31, +18.71), indicating that structured KGs are most beneficial for multi-hop reasoning.
3. **Optimal number of parallel subgraphs** is 3; too few lacks diversity, too many introduces noise (though Qwen2.5 is more robust to this).
4. **Commercial scenario validation**: On NowNewsQA (news QA), TAdaRAG significantly outperforms PathRAG in both conciseness (8.25 vs. 7.63) and factuality (8.45 vs. 7.85) ($p < 0.0001$).
5. **LLM-based evaluation strongly correlates with human evaluation**: Pearson correlation coefficients range from 0.706 to 0.925.

## Highlights & Insights

- **Relocating KG construction from the retrieval stage to the inference stage** is the core innovation, enabling genuinely "real-time" and "task-adaptive" knowledge graph generation.
- **REINFORCE enables the model to automatically learn to distill KGs** — rather than manually specifying extraction rules, the system is optimized end-to-end.
- **The Mixing Network design is elegant**: it allows the model to autonomously determine, at the token level, whether to rely on the KG or answer directly.
- The system has been **deployed in a commercial product (Xinyu AI Search)**, demonstrating practical production value.
- The framework generalizes across diverse scenarios ranging from open-domain QA to legal and medical domains and long-document summarization.

## Limitations & Future Work

- **On-the-fly KG construction incurs additional computational overhead** — generating a KG at every inference step results in higher latency compared to standard RAG.
- The approach partially relies on **manually designed domain templates** (even if only for cold-start initialization), limiting the degree of full automation.
- Performance on the Legal dataset falls below MEMORAG (49.88 vs. 51.20), possibly due to the specialized nature of legal text requiring more domain-specific templates.
- The optimal number of parallel subgraphs (3) is determined empirically, lacking theoretical justification.
- Validation is currently limited to 7B and 14B scale models; the effectiveness and efficiency trade-offs for larger models remain unexplored.

## Related Work & Insights

- **Fundamental distinction from GraphRAG**: GraphRAG pre-constructs a global KG with community summaries, whereas TAdaRAG constructs task-specific KGs on demand at inference time.
- **Complementary to PathRAG**: PathRAG extracts key paths from an indexed graph, while TAdaRAG constructs an entirely new KG from raw text.
- **MEMORAG** compresses the database using a memory module and generates retrieval cues; it serves as the strongest baseline.
- **Growing role of RL in RAG**: from retrieval strategy optimization to knowledge structure optimization as in this work, RL is playing an increasingly important role in the RAG paradigm.
- The paper offers insights for "Agentic RAG": agents dynamically selecting knowledge representation forms (text vs. KG vs. summary).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of real-time task-adaptive KG construction is novel, though individual components (intent routing, SFT, REINFORCE) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6+1 datasets, 3 backbone models, statistical significance testing, human evaluation, and commercial deployment validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated (three illustrative case figures), and the methodology is presented with clear hierarchy.
- **Value**: ⭐⭐⭐⭐⭐ — Substantial improvements on multi-hop reasoning (+18.71) and long-document summarization (+9.01), with confirmed commercial deployment.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](../../CVPR2026/reinforcement_learning/reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](../../NeurIPS2025/reinforcement_learning/knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [\[AAAI 2026\] PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning](pa-fas_towards_interpretable_and_generalizable_multimodal_face_anti-spoofing_via.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)

<!-- RELATED:END -->
