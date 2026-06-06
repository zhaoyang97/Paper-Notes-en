---
title: >-
  [Paper Note] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits
description: >-
  [ACL 2026][Information Retrieval & RAG][Document VQA] The MAB-DQA framework is proposed to decompose complex queries into multiple aspect sub-queries. It utilizes a Multi-Armed Bandit mechanism (Thompson Sampling) to dyn…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Document VQA"
  - "Multi-Armed Bandits"
  - "Query Decomposition"
  - "Multi-modal RAG"
  - "Hypergraph Reasoning"
date: 2026-05-08
content_hash: 3819a4e8cc64293b
---

# MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits

**Conference**: ACL 2026  
**arXiv**: [2604.08952](https://arxiv.org/abs/2604.08952)  
**Code**: [GitHub](https://github.com/ElephantOH/MAB-DQA)  
**Area**: Document Question Answering and Information Retrieval  
**Keywords**: Document VQA, Multi-Armed Bandits, Query Decomposition, Multi-modal RAG, Hypergraph Reasoning

## TL;DR

The MAB-DQA framework is proposed to decompose complex queries into multiple aspect sub-queries. It utilizes a Multi-Armed Bandit mechanism (Thompson Sampling) to dynamically evaluate the importance of each aspect and reallocate the retrieval budget, significantly improving retrieval precision and answer accuracy in multi-modal document question answering.

## Background & Motivation

- **Background**: Document Question Answering (DQA) requires AI to generate answers from documents based on user queries and is a core task in document understanding. Current state-of-the-art methods (e.g., ColPali, MoloRAG) adopt the vision-query Late Interaction paradigm, calculating similarity scores by summing the maximum dot products between query tokens and document image patches.
- **Limitations of Prior Work**: The "max-pooling + summation" operation in Late Interaction assigns equal weight to all query tokens, failing to distinguish between the importance of different aspects in a query as humans do. This leads to high false similarity for low-importance but high-frequency keywords (e.g., company names like "Best Buy") on irrelevant pages, while pages containing crucial evidence are ranked lower.
- **Key Challenge**: Multi-modal RAG typically retains only a few candidate pages (e.g., Top-4), causing information-rich but visually non-salient content to be easily overlooked. Statistics show that 19.8% of samples in MMLongBench and 27.8% in LongDocURL exhibit retrieval errors due to ignoring key query conditions.
- **Goal**: Explicitly model the varying importance of multiple implicit aspects within a query and dynamically allocate retrieval attention to prioritize evidence pages containing critical information.
- **Key Insight**: Treat each sub-query as an "arm" of a Multi-Armed Bandit (MAB) and use step-wise VLM reasoning feedback as a reward signal, adaptively allocating the retrieval budget to high-value aspects through an exploration-exploitation strategy.
- **Core Idea**: A three-stage progressive DQA process consisting of query decomposition, Thompson Sampling-driven dynamic retrieval budget allocation, and hypergraph reflective reasoning.

## Method

### Overall Architecture

MAB-DQA consists of three core phases: (1) Query-Aware Page Hypergraph construction—decomposing the query into aspect sub-queries and constructing a hypergraph with document pages; (2) Multi-Armed Bandit Guided Retrieval—dynamically selecting high-value aspects for page retrieval using Thompson Sampling; (3) Hypergraph-based Reflective Reasoning Agent—generating final answers through multi-stage verification.

### Key Designs

1. **Query-Aware Page Hypergraph**:
    - **Function**: Unified modeling of document page relationships and the multi-aspect structure of the query.
    - **Mechanism**: First, a query-agnostic page graph $G$ is built (based on inter-page similarity). Then, a VLM decomposes the original query $q$ into $M$ aspect sub-queries $\{q_1, \dots, q_m\}$. For each sub-query, a set of candidate pages $C_j$ is formed by retrieving the Top-$\theta_H$ pages. Pages ranked higher under the sub-query than the global query are selected to form a hyperedge $\hat{E}_j$. The final hypergraph $H = (V_G, \{\hat{E}_j\} \cup E_G)$ includes both inter-page edges and aspect hyperedges.
    - **Design Motivation**: Standard graphs cannot represent the group relationship between a sub-query and multiple pages; hyperedges are naturally suited for modeling "one aspect associated with a set of pages."

2. **MAB-Guided Retrieval**:
    - **Function**: Dynamically assess aspect importance and allocate retrieval budget to high-value aspects.
    - **Mechanism**: Each sub-query $Q_j$ acts as an arm, maintaining a $\text{Beta}(\alpha_j, \beta_j)$ distribution. In each round, an arm is selected via Thompson Sampling to retrieve pages from the corresponding hyperedge. A VLM evaluates page relevance $s_{vlm} \in [0,1]$ as a reward to update Beta parameters. The comprehensive page score is calculated as: $\text{score}(p_i) = (1-\alpha) \cdot \max LI + \alpha \cdot s_{vlm} + \beta[(1-\lambda) \cdot h_i + \lambda \cdot \bar{s}_{cb}]$, where $\bar{s}_{cb}$ is the mean Thompson Sampling confidence of associated sub-queries.
    - **Design Motivation**: Information value varies greatly across query aspects, and fixed weights fail to capture this; the exploration-exploitation balance of MAB is natively suited for this scenario.

3. **Hypergraph-based Reflective Reasoning Agent (HRRA)**:
    - **Function**: Generate and verify final answers from retrieved evidence pages.
    - **Mechanism**: Employs an "initial answer-verification-optimization" pipeline. If inconsistency or evidence gaps are detected, the agent enters a reflection loop to construct a query-focused subgraph for further reasoning.
    - **Design Motivation**: Single-pass generation may miss information or hallucinate; a reflective reasoning mechanism provides multi-stage verification assurance.

### Loss & Training

- This framework is an inference-time method and does not involve model training.
- Online update of Beta distribution parameters: $(\alpha_j, \beta_j) \leftarrow (\alpha_j + s_{vlm}, \beta_j + 1 - s_{vlm})$.
- Key hyperparameters: $\alpha=0.8$ (VLM evaluation weight), $\beta=0.1$ (hyperparameter scaling), $\lambda=0.75$ (balance between page degree and arm confidence), $\theta_G=0.8$ (edge threshold), $\theta_H=10$ (hyperedge capacity), $m=20$ (retrieval iterations).

## Key Experimental Results

### Main Results

| Method | MMLongBench | LongDocURL | FetaTab | PaperTab | Average |
|---|---|---|---|---|---|
| Qwen-2.5-VL-7B (Direct) | 0.204 | 0.398 | 0.350 | 0.112 | 0.266 |
| MDocAgent | 0.315 | 0.527 | 0.598 | 0.227 | 0.417 |
| MoloRAG+ | 0.372 | 0.528 | 0.600 | 0.195 | 0.424 |
| **MAB-DQA** | **0.399** | **0.564** | **0.638** | **0.269** | **0.468** |
| **Gain** | +7.25% | +5.22% | +6.33% | +18.50% | +10.38% |

Retrieval performance (Top-3, MMLongBench): MAB-DQA outperforms MoloRAG and MoloRAG+ across all metrics: Recall (69.53), Precision (34.32), NDCG (41.05), and MRR (72.94).

### Ablation Study

| Variant | MMLongBench | LongDocURL | FetaTab | PaperTab | Average Gain |
|---|---|---|---|---|---|
| Colpali (Baseline) | 0.296 | 0.554 | 0.537 | 0.152 | 0.0% |
| + MABR | 0.388 | 0.543 | 0.609 | 0.226 | +22.8% |
| + HRRA | 0.395 | 0.561 | 0.624 | 0.236 | +26.5% |
| MAB-DQA (Full) | 0.399 | 0.564 | 0.638 | 0.269 | +33.1% |

### Key Findings

- **Significant variation in query aspect importance**: Approximately 20-28% of samples exhibit retrieval errors due to ignored key conditions, proving systemic flaws in uniformly weighted Late Interaction.
- **Highest gain on PaperTab (+18.5%)**: Tasks involving document structure and table understanding benefit most from aspect-aware retrieval.
- **MABR and HRRA are complementary**: MABR provides adaptive retrieval focusing on key aspects (+22.8%), while HRRA further corrects errors via reflective verification (+26.5%), reaching +33.1% combined.
- **Generalization across VLM backbones**: Consistent improvements are observed across Qwen2.5-VL-7B, LLaVa-13B, Qwen3-30B, and Qwen3-32B.

## Highlights & Insights

- **Precise Problem Definition**: Clearly characterizes the overlooked issue of "uniform query aspect weighting" in Late Interaction, supported by visualization heatmaps and statistical data.
- **Clever MAB Modeling**: Translates the estimation of query aspect importance into a MAB problem, where Thompson Sampling's exploration-exploitation balance fits the retrieval scenario perfectly.
- **Inference-time, Training-free**: The framework operates during inference without needing additional training data or fine-tuning, making it plug-and-play.
- **Hypergraph Modeling of Page-Aspect Relations**: More effective than standard graphs in representing the group structure of "one aspect associated with multiple pages."

## Limitations & Future Work

- Strong dependency on the underlying VLM—performance may be limited if the VLM performs poorly in specific domains (e.g., legal, medical).
- High number of hyperparameters ($\alpha, \beta, \lambda, \theta_G, \theta_H, m$), currently selected via grid search; authors plan to introduce Bayesian optimization for automatic tuning.
- Only Thompson Sampling is used; other bandit strategies like UCB or $\epsilon$-Greedy have not been compared.
- Retrieval time complexity is $O(m \cdot T_{VLM})$; the number of VLM calls grows linearly with iterations, potentially causing a bottleneck for large-scale documents.

## Related Work & Insights

- **ColPali (Faysse et al., 2024)**: Vision-language embedding model serving as the retrieval backbone for this work.
- **MoloRAG (Wu et al., 2025)**: Uses graph structures and VLM evaluation for multi-modal DQA. It serves as a strong baseline but uses a fixed retrieval budget and ignores aspect importance.
- **MBA-RAG (Tang et al., 2025)**: Employs MAB for uni-modal RAG to select retrieval strategies rather than query aspects.
- **GraphRAG (Edge et al., 2025)**: Representative work for graph-enhanced RAG; MAB-DQA further introduces hypergraphs and dynamic budget allocation.
- The aspect-aware retrieval logic can be extended to general multi-modal RAG scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing MAB to model query aspect importance is a novel perspective, and the hypergraph construction is unique.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 4 benchmarks, multiple baselines, ablation studies, sensitivity analysis, and cross-VLM validation.
- Writing Quality: ⭐⭐⭐⭐ Clear diagrams, well-argued motivation, and persuasive visualized case studies.
- Value: ⭐⭐⭐⭐ High practicality as an inference-time, training-free, plug-and-play method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ACL 2026\] FinRAG-12B: A Production-Validated Recipe for Grounded Question Answering in Banking](finrag-12b_a_production-validated_recipe_for_grounded_question_answering_in_bank.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)

</div>

<!-- RELATED:END -->
