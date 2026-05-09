---
title: >-
  [Paper Note] ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG
description: >-
  [NeurIPS 2025][Graph Learning][Knowledge Graph] This paper proposes ReMindRAG, a KG-RAG system that combines LLM-guided KG traversal (node exploration + exploitation) with a training-free memory replay mechanism. The system stores LLM traversal experience in edge embeddings, enabling significant reduction in LLM calls for subsequent similar queries (~50% cost reduction) while improving answer accuracy (5%–10% gain).
tags:
  - NeurIPS 2025
  - Graph Learning
  - Knowledge Graph
  - RAG
  - LLM-Guided Graph Traversal
  - Memory Replay
  - Training-Free Memory
date: 2026-05-08
content_hash: bcac3306fe26ca8f
---

# ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG

**Conference**: NeurIPS 2025
**arXiv**: [2510.13193](https://arxiv.org/abs/2510.13193)
**Code**: [Available](https://github.com/kilgrims/ReMindRAG)
**Area**: Graph Learning / RAG
**Keywords**: Knowledge Graph, RAG, LLM-Guided Graph Traversal, Memory Replay, Training-Free Memory

## TL;DR

This paper proposes ReMindRAG, a KG-RAG system that combines LLM-guided KG traversal (node exploration + exploitation) with a training-free memory replay mechanism. The system stores LLM traversal experience in edge embeddings, enabling significant reduction in LLM calls for subsequent similar queries (~50% cost reduction) while improving answer accuracy (5%–10% gain).

## Background & Motivation

RAG systems augment LLM generation by retrieving external knowledge. Knowledge graph-based RAG (KG-RAG) leverages graph structures to represent inter-entity relationships, offering advantages in multi-hop reasoning and long-range dependency tasks.

However, existing KG-RAG methods face a **fundamental tension between effectiveness and cost efficiency**:

**Traditional graph search algorithms** (PageRank, GNNs, etc.): low cost but unable to capture fine-grained semantic relationships, resulting in suboptimal performance.

**LLM-guided graph traversal** (e.g., Plan-on-Graph): leverages LLM semantic understanding for traversal decisions, achieving strong performance but requiring repeated LLM calls, leading to high token consumption and long inference latency.

**Core Challenge**: How can a KG-RAG system maintain the high effectiveness of LLM-guided traversal while substantially reducing LLM invocation costs?

Limitations of existing approaches:
- Traditional graph pruning methods may discard critical information
- Path planning methods offer limited efficiency gains
- No existing method exploits historical traversal experience to accelerate future queries

## Method

### Overall Architecture

ReMindRAG consists of two core modules:
1. **LLM-Guided KG Traversal**: Balances local depth-first search and global search through node Exploration and node Exploitation.
2. **Memory Replay**: Encodes LLM traversal experience into KG edge embeddings in a training-free manner, enabling direct reuse for subsequent similar queries.

### Key Designs

#### 1. LLM-Guided KG Traversal (Exploration + Exploitation)

**Function**: Starting from seed nodes, the LLM iteratively expands a subgraph until it contains sufficient information to answer the query.

**Node Exploration**: At each step, the LLM evaluates all nodes in the current subgraph $S$ and selects the target node $a$ most likely to lead to the answer, ensuring global search rather than depth-first search along a single path.

**Node Exploitation**: Given the selected node $a$, the optimal expansion node $p$ is chosen from its neighbor set $S_a$, and $p$ along with the corresponding edge is added to subgraph $S$.

**Design Motivation**: Analogous to the exploration-exploitation trade-off in multi-armed bandits — exploration ensures the possibility of global optimality, while exploitation ensures depth along promising directions.

#### 2. Memory Replay: Encoding Traversal Experience into KG Edge Embeddings

**What to memorize**: Effective paths (those that successfully locate the answer) and ineffective paths (those that do not contribute to the answer) are distinguished and extracted from the final subgraph via DFS.

**How to memorize**: Each edge embedding vector $v$ is updated via a closed-form formula:

Weight function: $\delta(\mathbf{x}) = \frac{2}{\pi}\cos(\frac{\pi}{2}\|\mathbf{x}\|_2)$

- Effective path reinforcement: $\hat{v} = v + \delta(v) \cdot \frac{q}{\|q\|_2}$ (update along the query embedding direction)
- Ineffective path penalization: $\hat{v} = v - \delta(\frac{v \cdot q}{\|q\|_2}) \cdot \frac{v \cdot q}{\|q\|_2}$ (reduce the component along the query direction)

**Two key properties**:
- **Fast Wakeup**: When the edge embedding norm is small (initial zero vector), $\delta$ is large (≈ $2/\pi$), enabling rapid learning.
- **Damped Update**: When the edge embedding norm is large (accumulated experience), $\delta$ is small, providing resistance to perturbation from subsequent updates.

**Design Motivation**: Inspired by the way LLMs "memorize" world knowledge in their parameters, this mechanism encodes traversal experience into graph edge embeddings in a training-free manner.

#### 3. Memory-Based Efficient Subgraph Expansion

Prior to LLM-guided traversal, memory is used to expand the initial subgraph. DFS is performed from each seed node, with the decision to include a neighbor determined by the match between edge embeddings and the query:

$$w_{N_{from},N_{to}} = \alpha \cdot \text{sim}(\text{emb}(N_{from}), \text{emb}(N_{to})) + (1-\alpha) \cdot \frac{\text{emb}(q) \cdot v_{N_{from},N_{to}}}{\|\text{emb}(q)\|}$$

Expansion proceeds when $w > \lambda$; otherwise the branch is pruned. This allows similar queries to rapidly locate the answer subgraph via memory, without additional LLM calls.

### Theoretical Analysis

**Memory Capacity Theorem**: When the embedding dimension $d$ is sufficiently large, edge embeddings can effectively store traversal information for a set of semantically similar queries within a maximum angular distance $\theta$:

$$\theta \leq \lim_{d\to\infty} [2\arcsin(\sqrt{\frac{1}{2}}\sin(\arccos(\lambda)))]$$

The theoretical upper bound is $\lambda = 0.775$, consistent with the cosine similarity threshold of 0.6 commonly used in prior work for semantic similarity judgment.

## Key Experimental Results

### Main Results: System Effectiveness Evaluation

Under two backbone models, GPT-4o-mini and Deepseek-V3:

| Method | Long-Dep. QA (avg) | Multi-Hop QA (avg) | Simple QA (avg) |
|--------|-------------------|-------------------|----------------|
| LLM Only | 22.82% | 47.43% | 24.11% |
| NaiveRAG | 41.95% | 57.73% | 57.18% |
| GraphRAG | 17.95% | 40.83% | 14.58% |
| LightRAG | 45.30% | 49.48% | 56.16% |
| HippoRAG2 | 38.93% | 66.50% | 72.18% |
| Plan-on-Graph | 23.62% | 53.19% | 35.58% |
| **ReMindRAG** | **58.39%** | **76.80%** | **76.84%** |

ReMindRAG significantly outperforms all baselines across all three task types, leading the best baseline by approximately 12% on long-dependency QA and 10% on multi-hop QA.

### Memory Efficiency and Cost Experiments (Ablation)

Multi-round memory + cost savings (Multi-Hop QA, GPT-4o-mini):

| Configuration | Accuracy | Tokens/Query | Notes |
|--------------|----------|-------------|-------|
| No memory | 74.22% | 10.16K | Initial state |
| 1-round memory (Same) | 79.78% | 7.73K | Cost ↓24%, Acc. ↑5.6% |
| 2-round memory (Same) | 84.04% | 6.56K | Cost ↓35%, Acc. ↑9.8% |
| 3-round memory (Same) | 87.62% | 5.89K | Cost ↓42%, Acc. ↑13.4% |
| 1-round memory (Similar) | 75.53% | 9.88K | Effective for semantically equivalent queries |
| 1-round memory (Different) | 72.34% | 10.57K | Different queries also partially benefit |

### Key Findings

1. **Self-correction capability**: Accuracy continues to improve across memory rounds (up to +13%), and even when the initial answer is incorrect, the memory mechanism enables automatic correction by penalizing ineffective paths.
2. **Memory stability**: Under an extreme alternating test with different datasets (Origin/Different rounds), the system maintains stable performance.
3. **Dimension robustness**: Truncating embedding dimensions from 768 to 12 has negligible impact on performance, demonstrating that the memory mechanism is dimension-insensitive.
4. **Substantial cost reduction**: Token consumption decreases by approximately 58% after 3 rounds of memory.

## Highlights & Insights

1. **Training-free memory mechanism**: No training process is required; edge embeddings are updated directly via closed-form formulas, making the approach both elegant and efficient.
2. **Fast Wakeup + Damped Update**: The cosine weight function is carefully designed — enabling rapid learning at initialization and stable retention after sufficient accumulation.
3. **Self-correction as a fortuitous benefit**: The memory mechanism not only reduces cost but also improves accuracy by iteratively penalizing incorrect paths and reinforcing correct ones.
4. **Theoretical-practical consistency**: The memory capacity has theoretical guarantees, and experiments confirm robustness independent of embedding dimensionality.

## Limitations & Future Work

1. **First-query LLM cost remains high**: The initial traversal without memory still requires multiple LLM calls.
2. **Risk of memory contamination**: If the initial answer is entirely incorrect, erroneous paths may be memorized, requiring multiple correction rounds.
3. **Dependency on KG construction quality**: System performance is sensitive to the quality of the underlying KG — errors in entity or relation extraction will propagate into the memory.
4. **Scalability**: The overhead of DFS expansion and edge embedding storage should be evaluated as KG scale increases substantially.

## Related Work & Insights

- Compared to HippoRAG2: ReMindRAG achieves more precise retrieval through LLM-guided traversal rather than relying on traditional graph algorithms.
- The memory replay concept is inspired by cognitive science, but the implementation via edge embedding storage is a novel contribution.
- Cold-start issues could potentially be addressed by pre-initializing memory with FAQ-style queries.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (The combination of training-free memory replay and LLM-guided traversal is highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multi-task, multi-backbone, multi-round memory, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear figures, rich case studies)
- Value: ⭐⭐⭐⭐⭐ (Simultaneously improves effectiveness and efficiency; high practical utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning](smore_structural_mixture_of_residual_experts_for_parameter-efficient_llm_fine-tu.md)
- [\[AAAI 2026\] Human Cognition Inspired RAG with Knowledge Graph for Complex Problem Solving](../../AAAI2026/graph_learning/human_cognition_inspired_rag_with_knowledge_graph_for_complex_problem_solving.md)
- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](../../ICLR2026/graph_learning/entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Efficient AC Power Flow Prediction in Power Grids](graph_neural_networks_for_efficient_ac_power_flow_prediction_in_power_grids.md)

</div>

<!-- RELATED:END -->
