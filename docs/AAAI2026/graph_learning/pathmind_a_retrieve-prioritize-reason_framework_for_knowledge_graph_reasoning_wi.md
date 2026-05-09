---
title: >-
  [Paper Note] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models
description: >-
  [AAAI 2026][Graph Learning][Knowledge Graph Reasoning] This paper proposes PathMind, a framework following the Retrieve-Prioritize-Reason paradigm. It identifies important reasoning paths via a semantics-aware path prioritization function that jointly considers cumulative cost and estimated future cost (inspired by A*), and then enhances faithful and interpretable LLM reasoning through a two-stage training strategy comprising task-specific instruction tuning and path-level preference alignment. PathMind achieves state-of-the-art performance on complex reasoning tasks while consuming significantly fewer tokens.
tags:
  - AAAI 2026
  - Graph Learning
  - Knowledge Graph Reasoning
  - LLM
  - Path Prioritization
  - Retrieval Augmentation
  - Preference Alignment
date: 2026-05-08
content_hash: 0d2df3e7e14840d4
---

# PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.14256](https://arxiv.org/abs/2511.14256)
**Code**: [github.com/liuyudiy/PathMind](https://github.com/liuyudiy/PathMind)
**Area**: Graph Learning
**Keywords**: Knowledge Graph Reasoning, LLM, Path Prioritization, Retrieval Augmentation, Preference Alignment

## TL;DR

This paper proposes PathMind, a framework following the Retrieve-Prioritize-Reason paradigm. It identifies important reasoning paths via a semantics-aware path prioritization function that jointly considers cumulative cost and estimated future cost (inspired by A*), and then enhances faithful and interpretable LLM reasoning through a two-stage training strategy comprising task-specific instruction tuning and path-level preference alignment. PathMind achieves state-of-the-art performance on complex reasoning tasks while consuming significantly fewer tokens.

## Background & Motivation

### Problem Definition

Knowledge Graph Reasoning (KGR) aims to infer new knowledge or answer complex queries over a knowledge graph $\mathcal{G} = (\mathcal{E}, \mathcal{R}, \mathcal{T})$. Given a query $q$ and KG $\mathcal{G}$, the goal is to design a function $f$ that predicts the answer $a = f(q, \mathcal{G})$.

### Two Dominant Paradigms and Their Limitations

#### 1. Retrieval-Augmented Paradigm
- Retrieves query-relevant triples or multi-hop paths from the KG and serializes them as text input to an LLM.
- **Key limitation**: Reasoning paths are extracted indiscriminately without assessing each path's importance to answer generation, potentially introducing irrelevant noise that misleads the LLM.
- Example: For the query "Who are Amazon's competitors?", the path $Amazon \xrightarrow{invest\_in} Retail \xrightarrow{invest\_by} Walmart$ clearly reveals a competitive relationship, whereas $Amazon \xrightarrow{partner} Google \xleftarrow{partner} Walmart$ may misleadingly imply a cooperative relationship.

#### 2. Synergy-Augmented Paradigm
- Employs LLMs as agents that iteratively interact with the KG to dynamically explore reasoning paths.
- **Key limitation**: The large search space demands numerous retrieval operations and repeated LLM calls, incurring high computational overhead that severely limits scalability and practicality.
- Example: ToG requires 11.6 LLM calls per query; PoG requires 9.0.

### Paper Goals

- Design an effective path prioritization mechanism inspired by the A* algorithm.
- Jointly account for **cumulative cost** from the query to the current node and **estimated future cost** to the target.
- Complete reasoning with only a single LLM call.

## Method

### Overall Architecture

PathMind consists of three core modules:
1. **Subgraph Retrieval**: Extracts a query subgraph and encodes it as a graph representation.
2. **Path Prioritization**: Identifies important reasoning paths using a priority function.
3. **Knowledge Reasoning**: Generates accurate and consistent responses via a two-stage training strategy.

### Key Designs

#### 1. Subgraph Retrieval Module

**Query Subgraph Extraction**:
- For each topic entity $e_q \in \mathcal{O}_q$ in query $q$, retrieve its $k$-hop neighborhood $\mathcal{N}_k(e_q)$.
- Take the union of neighborhoods as the subgraph node set: $\mathcal{E}_q = \bigcup_{e_q \in \mathcal{O}_q} \mathcal{N}_k(e_q)$.
- Extract edges connecting subgraph nodes to construct the query subgraph $\mathcal{G}_q$.

**Graph Representation Learning**:
- A GNN learns node and relation representations via message passing and aggregation:
  $$\bm{m}_e^{(l)} = \text{AGG}^{(l)}(\{\bm{W}_r^{(l)} \bm{h}_{e'}^{(l-1)} \mid (e', r, e) \in \mathcal{T}_q\})$$
  $$\bm{h}_e^{(l)} = \text{UPDATE}^{(l)}(\bm{h}_e^{(l-1)}, \bm{m}_e^{(l)})$$

#### 2. Path Prioritization Module (Core Contribution)

Inspired by the A* algorithm, a semantics-aware path priority function is designed. The overall priority score is:
$$s_q(e) = \sigma(\text{MLP}(\bm{d}(q,e) + \bm{f}(e,a)))$$

**Cumulative Cost $d(q,e)$**:
- Measures the aggregated path cost from the query to the current entity:
$$\bm{d}(q,e) = \sum_{\pi \in \Pi_{q \rightsquigarrow e}} \sum_{(e_{i-1}, r_i, e_i) \in \pi} \bm{w}_q(e_{i-1}, r_i, e_i)$$
- where $\bm{w}_q(e_{i-1}, r_i, e_i) = (\bm{h}_{e_{i-1}} \bm{W}_{r_i} \bm{h}_{e_i})^\top \bm{q}$ is the query-conditioned semantic representation of a triple.

**Estimated Future Cost $f(e,a)$**:
- Since the target answer $a$ is unknown at inference time, it is reparameterized using topic entities and query relations:
$$\bm{f}(e,a) = \bm{f}([\bm{d}(q,e), \bm{q}])$$
- Intuition: the remaining cost is estimated by comparing the current representation with the query — if $\bm{d}(q,e)$ is close to $\bm{q}$, the remaining cost approaches zero.

**Two Key Challenges Addressed**:
1. KGs are heterogeneous graphs rather than grid graphs — edges represent semantic relations rather than geometric distances → semantic distances are defined using GNN-learned representations.
2. KGs are large-scale → retrieval scope is restricted to the query subgraph $\mathcal{G}_q$.

**Learning Objective**:
$$\mathcal{L} = -\sum_{e \in \mathcal{A}_q} \log(s_q(e)) - \sum_{e \in \mathcal{G}_q \backslash \mathcal{A}_q} \log(1 - s_q(e))$$

Top-$K$ entities are iteratively selected over $T$ iterations (WebQSP: $T=2$; CWQ: $T=4$; $K=3$).

#### 3. Knowledge Reasoning Module (Two-Stage Training)

**Stage 1: Task-Specific Instruction Tuning (SFT)**
- Input: query $q$ + important reasoning paths $\Pi_q$
- Output: corresponding answer set $\mathcal{A}_q$
- Loss: $\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(q, \mathcal{A}_q) \sim \mathcal{D}_{\text{SFT}}}[\log P_\phi(\mathcal{A}_q | q, \Pi_q)]$

**Stage 2: Path-Level Preference Alignment (DPO)**
- Constructs preference pairs: $\Pi_q^w$ (preferred paths = retrieved important paths) vs. $\Pi_q^l$ (dispreferred paths = remaining candidate paths in the subgraph).
- DPO loss:
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\mathcal{M}(\Pi_q^w | q)}{\mathcal{M}(\Pi_q^l | q)} - \beta \log \frac{\mathcal{M}_{\text{sft}}(\Pi_q^w | q)}{\mathcal{M}_{\text{sft}}(\Pi_q^l | q)}\right)\right]$$

### Training Details

- **LLM backbone**: Llama3.1-8B
- **Subgraph retrieval**: 3-hop neighborhood
- **GNN**: randomly initialized; BERT encodes query representations
- **Training**: 3 epochs, batch size = 2, learning rate = 2e-5, warmup ratio = 3e-2
- **DPO**: learning rate = 5e-6, $\beta = 0.1$
- **Maximum input length**: 2048 tokens
- **Hardware**: 2 × NVIDIA A800 GPUs

## Key Experimental Results

### Main Results

| Method | Type | WebQSP Hits@1 | WebQSP F1 | CWQ Hits@1 | CWQ F1 |
|--------|------|---------------|-----------|------------|--------|
| ReaRev | Traditional Retrieval | 0.764 | 0.709 | 0.529 | 0.478 |
| GPT-4o | Direct LLM Reasoning | 0.618 | 0.436 | 0.382 | 0.329 |
| ToG | Synergy-Augmented | 0.826 | — | 0.685 | — |
| RoG | Retrieval-Augmented | 0.857 | 0.708 | 0.626 | 0.562 |
| GNN-RAG* | Retrieval-Augmented | 0.864 | 0.690 | 0.673 | 0.591 |
| SubgraphRAG | Retrieval-Augmented | 0.866 | 0.706 | 0.472 | 0.570 |
| EPERM | Retrieval-Augmented | 0.888 | 0.724 | 0.662 | 0.589 |
| GCR* | Retrieval-Augmented | 0.883 | 0.654 | 0.686 | 0.532 |
| **PathMind** | **Retrieval-Augmented** | **0.895** | **0.728** | **0.707** | **0.614** |

\*Reproduced with Llama3.1-8B. PathMind surpasses GNN-RAG on CWQ by 5.1% in Hits@1 and 3.9% in F1.

### Ablation Study

| Variant | WebQSP Hits@1 | CWQ Hits@1 | CWQ F1 | Note |
|---------|---------------|------------|--------|------|
| **PathMind (Full)** | **0.895** | **0.707** | **0.614** | — |
| w/o Prioritization | 0.840 | 0.643 | 0.561 | Removing path prioritization causes large performance drop |
| w/o Alignment | 0.871 | 0.672 | 0.586 | Removing DPO yields suboptimal results |
| w/o Training | 0.668 | 0.413 | 0.274 | Removing two-stage training causes severe degradation |
| Random Paths | 0.356 | 0.268 | 0.079 | Random paths nearly fail |
| Shortest Paths | 0.854 | 0.662 | 0.578 | Shortest paths underperform prioritized paths |
| **Important Paths** | **0.895** | **0.707** | **0.614** | Prioritized paths are optimal |

### Efficiency Comparison

| Method | Hits@1 (%) | Avg. Time (s) | LLM Calls | Input Tokens |
|--------|-----------|---------------|-----------|--------------|
| ToG | 75.1 | 16.14 | 11.6 | 7,069 |
| PoG | 87.3 | 16.80 | 9.0 | 5,518 |
| RoG | 85.7 | 2.60 | 2 | 521 |
| GNN-RAG | 86.4 | 1.52 | 1 | 414 |
| GCR | 88.3 | 3.60 | 2 | 231 |
| **PathMind** | **89.5** | **2.23** | **1** | **216** |

PathMind requires only 1 LLM call and 216 input tokens, achieving the best trade-off between performance and efficiency.

### Key Findings

1. **Path prioritization is the core component**: Removing the path prioritization module reduces CWQ Hits@1 by 6.4%, demonstrating that identifying important paths is critical.
2. **Greater gains on complex reasoning**: PathMind's advantage is more pronounced on CWQ, which requires multi-hop reasoning, than on WebQSP, which is dominated by single-hop queries.
3. **$K=3$ is optimal**: Too many nodes ($K>3$) introduce irrelevant entities that obscure key information, causing F1 to decline.
4. **Cross-LLM generalization**: PathMind achieves strong results across Qwen2-7B, Llama2-7B, and Llama3.1-8B.
5. **Scalability verified**: As the number of reasoning hops and answers increases, PathMind consistently outperforms RoG by filtering out irrelevant paths and reducing interference.
6. **Cumulative cost contributes more**: Between the two components of the path priority function, cumulative cost contributes more than estimated future cost (0.878 vs. 0.831 Hits@1).

## Highlights & Insights

1. **Elegant analogy of A* in KG reasoning**: "Geometric distance" in graph search is replaced by GNN-learned "semantic distance," transferring path-planning intuitions to knowledge reasoning.
2. **Exceptional token efficiency**: State-of-the-art performance is achieved with only 216 input tokens — more than 30× fewer than synergy-augmented methods.
3. **Path-level application of DPO**: Preference alignment is applied to reasoning paths rather than answers — preferred paths vs. remaining candidate paths.
4. **Interpretable reasoning via case study**: Figure 5 illustrates how PathMind correctly identifies two-hop reasoning paths and produces accurate answers in the presence of noisy paths.

## Limitations & Future Work

1. **Subgraph retrieval bottleneck**: A 3-hop neighborhood may omit critical information (Case 3 demonstrates incorrect predictions caused by missing paths).
2. **Static path prioritization**: The learned priority function is fixed at inference time and cannot dynamically adjust based on intermediate reasoning results.
3. **Fixed LLM input length**: The 2048-token limit may become a bottleneck when the number of paths is large.
4. **Evaluation on only two datasets**: Validation is limited to WebQSP and CWQ; broader domain coverage (e.g., biomedical KGs) is lacking.
5. **GNN representation quality**: The accuracy of path prioritization depends on the quality of entity and relation representations learned by the GNN.

## Related Work & Insights

- **Comparison with RoG**: RoG employs a planning-retrieval-reasoning framework to generate relation paths; PathMind additionally introduces path importance assessment.
- **Comparison with GNN-RAG**: GNN-RAG retrieves shortest paths between topic entities and answer candidates but does not evaluate path importance.
- **Comparison with PoG**: PoG proposes self-correcting adaptive planning but requires 9 LLM calls.
- **Insight**: Path quality matters more than quantity — precisely selecting a small number of important paths outperforms providing a large pool of candidate paths.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of A*-inspired path prioritization and DPO-based path alignment is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extensive baseline comparisons, detailed ablations, efficiency analysis, case studies, and scalability analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear logical flow, excellent figure and table design, and complete appendix.
- Value: ⭐⭐⭐⭐ — Substantial contribution to the KGR field; the path prioritization idea is generalizable to other RAG scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[AAAI 2026\] NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation](notam-evolve_a_knowledge-guided_self-evolving_optimization_framework_with_llms_f.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](self-adaptive_graph_mixture_of_models.md)
- [\[NeurIPS 2025\] Dynamic Bundling with Large Language Models for Zero-Shot Inference on Text-Attributed Graphs](../../NeurIPS2025/graph_learning/dynamic_bundling_with_large_language_models_for_zero-shot_inference_on_text-attr.md)

</div>

<!-- RELATED:END -->
