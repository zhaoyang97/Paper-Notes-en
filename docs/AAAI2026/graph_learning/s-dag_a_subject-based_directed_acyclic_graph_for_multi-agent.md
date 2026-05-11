---
title: >-
  [Paper Note] S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning
description: >-
  [AAAI 2026][Graph Learning][Subject-level analysis] This paper proposes S-DAG, which uses a GNN to identify relevant subjects and their dependencies from a given question…
tags:
  - "AAAI 2026"
  - "Graph Learning"
  - "Subject-level analysis"
  - "directed acyclic graph"
  - "GNN reasoning"
  - "expert model composition"
  - "heterogeneous reasoning"
date: 2026-05-08
content_hash: b7d6737be02f94e8
---

# S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning

**Conference**: AAAI 2026
**arXiv**: [2511.06727](https://arxiv.org/abs/2511.06727)
**Code**: [https://github.com/WanyuGroup/AAAI2026_S-DAG](https://github.com/WanyuGroup/AAAI2026_S-DAG)
**Area**: Graph Learning
**Keywords**: Subject-level analysis, directed acyclic graph, GNN reasoning, expert model composition, heterogeneous reasoning

## TL;DR
This paper proposes S-DAG, which uses a GNN to identify relevant subjects and their dependencies from a given question, constructing a directed acyclic graph. Subject nodes are matched to the most capable expert LLMs (14 domain-specific models of 7–13B parameters), and collaborative reasoning proceeds in DAG topological order (supporting subjects → dominant subject). The resulting small-model pool surpasses GPT-4o-mini (59.73 vs. 58.52) and approaches the performance of a 72B model.

## Background & Motivation

**Background**: Existing MoE/routing methods (e.g., MoE Router, GraphRouter) select models at the **task level**—choosing one model or Top-k models for an entire question. However, many questions span multiple subjects (e.g., a problem involving physics, mathematics, and chemistry simultaneously), making task-level routing too coarse-grained.

**Limitations of Prior Work**: Multi-Agent Debate allows multiple models to deliberate but does not differentiate their respective expertise; Symbolic-MoE selects Top-k models by skill but ignores inter-subject dependencies (e.g., solving a physics problem may require mathematical derivation first).

**Key Challenge**: Three issues must be addressed simultaneously: (a) identifying which subjects a question involves; (b) determining the information flow among subjects (which supports which); (c) matching each subject to the most capable model.

**Key Insight**: Model multi-subject problems as a DAG—subjects as nodes, dependencies as directed edges (supporting → dominant)—and assign the best expert model to each node.

**Core Idea**: GNN-based subject-level DAG construction + model capability profiling and matching + DAG topological-order multi-agent collaborative reasoning.

## Method

### Overall Architecture
Two stages: (1) a GNN constructs the S-DAG (nodes = subjects, edges = dependencies); (2) expert models are assigned and collaborate in DAG topological order.

### Key Designs

1. **S-DAG Construction (GNN-based)**:

    - The question is encoded by BERT → fused with embeddings of 15 candidate subjects → GNN directed message passing updates node features → a node classifier determines subject relevance and an edge classifier determines dependency direction.
    - Ground-truth S-DAG: an LLM (qwen-turbo) scores the weight of each of the 15 subjects (averaged over 3 rounds); edges point from lower-weight (supporting) to higher-weight (dominant) subjects.

2. **LLM Profiling**:

    - 14 domain expert models (DeepseekMath-7B, BioMistral-7B, Qwen2.5-Coder-7B, etc.).
    - A per-model per-subject capability matrix $C_{ij}$ is constructed using 200 randomly sampled test questions.
    - Each subject node is assigned the highest-scoring model.

3. **DAG-guided Collaboration**:

    - **Subject Expert Agent** (root nodes): processes the original question from its own domain perspective.
    - **Supporting Agent** (intermediate nodes): integrates outputs from upstream agents with its own expertise.
    - **Dominant Agent** (terminal node): synthesizes all inputs and produces the final answer.

## Key Experimental Results

### Main Results

| Method | MMLU-Pro | GPQA | MedMCQA | Avg. |
|--------|----------|------|---------|------|
| GPT-4o-mini (CoT) | 49.42 | 47.31 | 78.82 | 58.52 |
| Symbolic-MoE | 48.13 | 45.92 | 78.55 | 57.53 |
| MAD (Qwen2.5-7b) | 45.82 | 46.81 | 76.55 | 56.39 |
| **S-DAG (7–13B pool)** | **50.98** | **49.82** | 78.38 | **59.73** |
| Qwen2.5-72B (CoT) | 50.81 | 48.98 | **80.44** | 60.08 |

### Ablation Study

| Configuration | Avg. Accuracy | Inference Time | LLM Calls |
|---------------|--------------|----------------|-----------|
| w/o GNN + random model | 41.12% | 14.21s | 5.1 |
| w/ GNN + random model | 42.19% | 14.82s | 4.1 |
| w/o GNN + capability matching | 53.51% | 14.53s | 5.1 |
| Fully connected graph | 57.29% | **38.45s** | **8.2** |
| **S-DAG (full)** | **59.73%** | **15.02s** | **4.1** |

### Key Findings
- **Small model pool surpasses GPT-4o-mini**: S-DAG (7–13B) 59.73% vs. GPT-4o-mini 58.52%, demonstrating that collaborative small models can substitute large models.
- **Model capability matching is the most critical factor**: random matching yields only 42.19%, while capability matching reaches 53.51% (+11.3 pp), with GNN-based DAG further improving it to 59.73%.
- **DAG outperforms fully connected graph in both accuracy and efficiency**: 59.73% vs. 57.29% (higher accuracy), 15s vs. 38.5s (2.5× faster), 4.1 vs. 8.2 LLM calls (halved)—redundant communication is in fact harmful.
- **GNN's value lies in noise filtering**: LLM-generated S-DAG labels are noisy and inconsistent; the trained GNN produces more robust graph structures.

## Highlights & Insights
- **Subject-level granularity** is finer than task-level routing—different subjects within the same question can be handled by different expert models, realizing true specialization.
- **DAG topological constraints** improve both accuracy and efficiency: supporting subjects are processed first and the dominant subject integrates their results, eliminating the redundant communication of fully connected graphs.
- **The paradigm of replacing large models with a small-model pool** has significant practical deployment value—the total parameter count of 14 models at 7–13B is far smaller than a single 72B model.

## Limitations & Future Work
- Evaluation is limited to MCQ benchmarks; open-ended generation tasks are not assessed.
- The 15 candidate subjects may lack sufficient granularity (e.g., "Medicine" could be divided into finer sub-disciplines).
- The model capability profile is estimated from only 200 samples, which is a relatively small sample size.
- Deploying 14 models simultaneously introduces non-trivial engineering complexity.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Subject-level DAG construction + capability matching + topological collaboration constitutes a new paradigm for multi-agent reasoning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three benchmarks, nine baselines, and comprehensive ablations, though limited to MCQ format.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and ablation analysis is thorough.
- **Value**: ⭐⭐⭐⭐⭐ The result of a small-model pool surpassing a large model has practical deployment value; the finding that DAG outperforms fully connected graphs carries theoretical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](assemble_your_crew_automatic_multi-agent_communication_topol.md)
- [\[AAAI 2026\] Spiking Heterogeneous Graph Attention Networks](spiking_heterogeneous_graph_attention_networks.md)
- [\[ICLR 2026\] Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](../../ICLR2026/graph_learning/pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)
- [\[AAAI 2026\] EchoLess: Label-Based Pre-Computation for Memory-Efficient Heterogeneous Graph Learning](echoless_label-based_pre-computation_for_memory-efficient_heterogeneous_graph_le.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)

</div>

<!-- RELATED:END -->
