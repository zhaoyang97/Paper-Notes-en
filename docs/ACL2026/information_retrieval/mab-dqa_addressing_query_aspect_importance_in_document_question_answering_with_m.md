---
title: >-
  [Paper Note] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits
description: >-
  [ACL 2026][Document Question Answering] This paper proposes MAB-DQA, a framework that decomposes complex queries into multiple aspect sub-queries, dynamically evaluates the importance of each aspect via a multi-armed bandit mechanism (Thompson Sampling), and redistributes retrieval budgets accordingly, achieving significant improvements in retrieval precision and answer accuracy for multimodal document question answering.
tags:
  - ACL 2026
  - Document Question Answering
  - Multi-Armed Bandits
  - Query Decomposition
  - Multimodal RAG
  - Hypergraph Reasoning
date: 2026-05-08
content_hash: ffca6c0e2c87c71b
---

# MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits

**Conference**: ACL 2026
**arXiv**: [2604.08952](https://arxiv.org/abs/2604.08952)
**Code**: [GitHub](https://github.com/ElephantOH/MAB-DQA)
**Area**: Document Question Answering & Information Retrieval
**Keywords**: Document Question Answering, Multi-Armed Bandits, Query Decomposition, Multimodal RAG, Hypergraph Reasoning

## TL;DR

This paper proposes MAB-DQA, a framework that decomposes complex queries into multiple aspect sub-queries, dynamically evaluates the importance of each aspect via a multi-armed bandit mechanism (Thompson Sampling), and redistributes retrieval budgets accordingly, achieving significant improvements in retrieval precision and answer accuracy for multimodal document question answering.

## Background & Motivation

- **State of the Field**: Document Question Answering (DQA) requires AI systems to generate answers from documents based on user queries and is a core task in document understanding. State-of-the-art methods such as ColPali and MoloRAG adopt a vision-query Late Interaction paradigm, computing similarity scores by summing the maximum dot products between query tokens and document image patches.
- **Limitations of Prior Work**: The "max-pooling + summation" operation in Late Interaction assigns equal weight to all query tokens, failing to distinguish the importance of different aspects within a query as humans naturally do. This causes low-importance but high-frequency keywords (e.g., a company name like "Best Buy") to produce spuriously high similarity scores on irrelevant pages, while pages containing genuinely critical evidence are ranked lower.
- **Root Cause**: Multimodal RAG typically retains only a small number of candidate pages (e.g., Top-4), making it easy to overlook content with high informational value but low visual saliency. The authors find that 19.8% of samples in MMLongBench and 27.8% in LongDocURL exhibit retrieval errors caused by neglecting critical query conditions.
- **Paper Goals**: Explicitly model the varying importance of multiple implicit aspects within a query, dynamically allocate retrieval attention, and prioritize evidence pages containing key information.
- **Starting Point**: Each sub-query is treated as an "arm" in a multi-armed bandit, with preliminary VLM reasoning feedback serving as the reward signal. An explore-exploit strategy adaptively allocates retrieval budget to high-value aspects.
- **Core Idea**: A three-stage progressive DQA pipeline comprising query decomposition, Thompson Sampling-driven dynamic retrieval budget allocation, and hypergraph reflective reasoning.

## Method

### Overall Architecture

MAB-DQA consists of three core stages: (1) Query-Aware Page Hypergraph Construction — decomposing the query into aspect sub-queries and constructing a hypergraph over document pages; (2) MAB-Guided Retrieval — dynamically selecting high-value aspects for page retrieval via Thompson Sampling; and (3) Hypergraph-based Reflective Reasoning Agent (HRRA) — generating and verifying the final answer through multi-stage validation.

### Key Designs

1. **Query-Aware Page Hypergraph**:
   - **Function**: Jointly models inter-page relationships and the multi-aspect structure of the query.
   - **Mechanism**: A query-agnostic page graph $G$ is first constructed based on inter-page similarity. A VLM then decomposes the original query $q$ into $M$ aspect sub-queries $\{q_1, \ldots, q_M\}$. For each sub-query, the Top-$\theta_H$ pages are retrieved to form candidate set $C_j$; pages ranked higher under the sub-query than under the global query are selected to construct hyperedge $\hat{E}_j$. The resulting hypergraph $H = (V_G, \{\hat{E}_j\} \cup E_G)$ encodes both inter-page edges and aspect hyperedges.
   - **Design Motivation**: Ordinary graphs cannot express the group relationship between a single sub-query and multiple pages; hyperedges naturally model the structure of "one aspect associated with a group of pages."

2. **MAB-Guided Retrieval**:
   - **Function**: Dynamically evaluates the importance of each aspect and allocates retrieval budget to high-value aspects.
   - **Mechanism**: Each sub-query $Q_j$ is treated as an arm maintaining a $\text{Beta}(\alpha_j, \beta_j)$ distribution. At each round, Thompson Sampling selects an arm, retrieves pages from the corresponding hyperedge, and uses VLM-assessed page relevance $s_\text{vlm} \in [0, 1]$ as reward to update the Beta parameters. The composite page score is $\text{score}(p_i) = (1-\alpha)\cdot\max\text{LI} + \alpha\cdot s_\text{vlm} + \beta[(1-\lambda)\cdot h_i + \lambda\cdot\bar{s}_\text{cb}]$, where $\bar{s}_\text{cb}$ is the mean Thompson Sampling confidence of associated sub-queries.
   - **Design Motivation**: The informational value of different query aspects varies substantially; fixed weights cannot capture this dynamic variation. The explore-exploit balance of MAB naturally suits this scenario.

3. **Hypergraph-based Reflective Reasoning Agent (HRRA)**:
   - **Function**: Generates and verifies the final answer from retrieved evidence pages.
   - **Mechanism**: An "initial answer–verification–refinement" pipeline is employed. An initial answer is generated from retrieved evidence; if inconsistencies or evidence gaps are detected, the system re-enters hypergraph construction to build a query-focused subgraph for a reflection loop.
   - **Design Motivation**: Single-pass generation may miss information or produce hallucinations; the reflective reasoning mechanism provides multi-stage validation.

### Loss & Training

- The framework is an inference-time method and involves no model training.
- Beta distribution parameters are updated online: $(\alpha_j, \beta_j) \leftarrow (\alpha_j + s_\text{vlm},\ \beta_j + 1 - s_\text{vlm})$.
- Key hyperparameters: $\alpha=0.8$ (VLM evaluation weight), $\beta=0.1$ (hyperparameter scaling), $\lambda=0.75$ (balance between page degree and arm confidence), $\theta_G=0.8$ (page graph edge threshold), $\theta_H=10$ (hyperedge capacity), $m=20$ (retrieval iterations).

## Key Experimental Results

### Main Results

| Method | MMLongBench | LongDocURL | FetaTab | PaperTab | Avg. |
|---|---|---|---|---|---|
| Qwen-2.5-VL-7B (Direct) | 0.204 | 0.398 | 0.350 | 0.112 | 0.266 |
| MDocAgent | 0.315 | 0.527 | 0.598 | 0.227 | 0.417 |
| MoloRAG+ | 0.372 | 0.528 | 0.600 | 0.195 | 0.424 |
| **MAB-DQA** | **0.399** | **0.564** | **0.638** | **0.269** | **0.468** |
| Gain | +7.25% | +5.22% | +6.33% | +18.50% | +10.38% |

Retrieval performance (Top-3, MMLongBench): MAB-DQA outperforms MoloRAG and MoloRAG+ across all metrics — Recall (69.53), Precision (34.32), NDCG (41.05), and MRR (72.94).

### Ablation Study

| Variant | MMLongBench | LongDocURL | FetaTab | PaperTab | Avg. Gain |
|---|---|---|---|---|---|
| ColPali (Baseline) | 0.296 | 0.554 | 0.537 | 0.152 | 0.0% |
| + MABR | 0.388 | 0.543 | 0.609 | 0.226 | +22.8% |
| + HRRA | 0.395 | 0.561 | 0.624 | 0.236 | +26.5% |
| MAB-DQA (Full) | 0.399 | 0.564 | 0.638 | 0.269 | +33.1% |

### Key Findings

- **Significant variance in query aspect importance**: Approximately 20–28% of samples exhibit retrieval errors due to neglected critical conditions, confirming a systematic deficiency in uniformly weighted Late Interaction.
- **Largest gain on PaperTab (+18.5%)**: Tasks involving document structure and table comprehension benefit most from aspect-aware retrieval.
- **MABR and HRRA are complementary**: MABR provides adaptive retrieval focused on key aspects (+22.8%); HRRA further corrects errors through reflective verification (+26.5%); their combination reaches +33.1%.
- **Generalizes across VLM backbones**: Consistent improvements are observed on Qwen2.5-VL-7B, LLaVA-13B, Qwen3-30B, and Qwen3-32B.

## Highlights & Insights

- **Precise problem formulation**: The paper clearly characterizes the overlooked issue of uniform query aspect weighting in Late Interaction, supported by visualization heatmaps and statistical evidence (Issue column).
- **Elegant MAB formulation**: Casting query aspect importance estimation as a multi-armed bandit problem is well-motivated; Thompson Sampling's explore-exploit balance aligns naturally with the retrieval setting.
- **Inference-time method, training-free**: The entire framework operates at inference time without additional training data or fine-tuning, making it plug-and-play.
- **Hypergraph modeling of page-aspect relations**: More expressive than ordinary graphs for capturing the group structure of "one aspect associated with multiple pages."

## Limitations & Future Work

- Heavy reliance on the underlying VLM — performance degrades if the VLM performs poorly in specialized domains (e.g., legal, medical).
- The framework involves numerous hyperparameters ($\alpha, \beta, \lambda, \theta_G, \theta_H, m$), currently selected via grid search; the authors plan to introduce Bayesian optimization for automatic tuning.
- Only Thompson Sampling is evaluated; no comparison against other bandit strategies such as UCB or $\varepsilon$-Greedy.
- Retrieval time complexity is $O(m \cdot T_\text{VLM})$, with VLM calls growing linearly with the number of iterations, which may become a bottleneck for large-scale documents.

## Related Work & Insights

- **ColPali (Faysse et al., 2024)**: A vision-language embedding model serving as the retrieval backbone in this work.
- **MoloRAG (Wu et al., 2025)**: Also employs graph structures and VLM evaluation for multimodal DQA; the strongest baseline, but with fixed retrieval budgets and no distinction among aspect importance.
- **MBA-RAG (Tang et al., 2025)**: Also applies MAB but in a unimodal setting, using MAB to select retrieval strategies rather than query aspects.
- **GraphRAG (Edge et al., 2025)**: A representative graph-augmented RAG method; MAB-DQA extends this direction with hypergraphs and dynamic budget allocation.
- The aspect-aware retrieval idea proposed in this paper is generalizable to broader multimodal RAG scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing MAB for query aspect importance modeling is a novel perspective; the hypergraph construction also offers a distinctive contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 4 benchmarks, multiple baselines, ablation studies, sensitivity analyses, and cross-VLM validation.
- Writing Quality: ⭐⭐⭐⭐ Clear figures, well-motivated problem formulation, and persuasive visual case studies.
- Value: ⭐⭐⭐⭐ Inference-time, training-free, plug-and-play design offers strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)

<!-- RELATED:END -->
