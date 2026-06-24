---
title: >-
  [Paper Note] ANRE: Analogical Replay for Temporal Knowledge Graph Forecasting
description: >-
  [ACL 2025][Time Series][Temporal Knowledge Graph] This paper proposes the ANRE (Analogical Replay) method, which retrieves structurally similar "analogical events" from historical knowledge graph snapshots as reasoning clues to assist future event forecasting in temporal knowledge graphs, achieving significant performance improvements across multiple benchmark datasets.
tags:
  - "ACL 2025"
  - "Time Series"
  - "Temporal Knowledge Graph"
  - "Link Prediction"
  - "Analogical Reasoning"
  - "Historical Event Replay"
  - "Graph Completion"
date: 2026-05-08
content_hash: 85f908b6ce9455bb
---

# ANRE: Analogical Replay for Temporal Knowledge Graph Forecasting

**Conference**: ACL 2025  
**Area**: Time Series  
**Keywords**: Temporal Knowledge Graph, Link Prediction, Analogical Reasoning, Historical Event Replay, Graph Completion

## TL;DR
This paper proposes the ANRE (Analogical Replay) method, which retrieves structurally similar "analogical events" from historical knowledge graph snapshots as reasoning clues to assist future event forecasting in temporal knowledge graphs, achieving significant performance improvements across multiple benchmark datasets.

## Background & Motivation

**Background**: Temporal Knowledge Graphs (TKGs) extend traditional knowledge graphs to the temporal dimension. The forecasting task aims to predict events that might occur at future timestamps (i.e., link prediction) based on historical graph snapshots. Existing methods are generally divided into graph neural network-based methods (such as RE-GCN, TiRGN) and sequence model-based methods (such as CyGNet), which learn temporal patterns of entities and relations by encoding historical graph structures.

**Limitations of Prior Work**: (1) Most approaches only focus on the direct historical interactions of target entities, ignoring the behavioral patterns of **other entities** under similar contexts within the graph—these "analogical events" contain rich reasoning clues. (2) Existing temporal encoding methods struggle to capture long-range periodic patterns and cross-entity structural similarities. (3) When the target entity's historical information is sparse, model performance drops sharply.

**Key Challenge**: The essence of TKG forecasting is "predicting the future based on the past", but each entity has limited historical information. If the historical experience of all entities across the entire graph can be utilized to assist reasoning (analogous to how humans predict unknown events through analogy), the amount of available information can be substantially expanded.

**Goal**: Design an analogical replay mechanism to automatically retrieve structurally similar events from historical graph snapshots for a given query and leverage these analogical events as an extra reasoning context to enhance forecasting.

**Key Insight**: Humans recall similar historical scenarios to make decisions when facing new situations. Inspired by this, the authors model TKG forecasting as an analogical reasoning process: finding "precedents" in history to infer potential outcomes of the current situation.

**Core Idea**: Retrieve analogical events by defining event similarity metrics, and then fuse the embedded representations of the retrieved events into the graph encoding process to provide cross-entity, cross-temporal auxiliary signals for the current prediction.

## Method

### Overall Architecture
The pipeline of ANRE consists of three phases: (1) Analogical Retrieval: retrieve analogical events from historical snapshots based on structural and semantic similarity; (2) Analogical Encoding: encode the retrieved analogical events into auxiliary representations; (3) Enhanced Prediction: fuse the analogical representations with standard graph encodings to output final link prediction probabilities.

### Key Designs

1. **Analogical Event Retrieval Module**:

    - **Function**: Retrieve structurally similar "precedent" events for the current query from historical TKG snapshots.
    - **Mechanism**: Given a query triplet $(s, r, ?, t)$, search for completed historical triplets $(s', r', o', t')$ at past timestamps that share the same or similar relation $r$. Rank them based on entity embedding similarity and relation consistency to select the Top-K most relevant analogical events. The similarity calculation comprehensively considers relation semantic similarity and entity structural role matching.
    - **Design Motivation**: Events with the same relation type often follow similar patterns (e.g., a "visit" relation typically involves specific types of subjects and objects). Exploiting such patterns can provide reasoning cues for sparse entities.

2. **Analogical Encoding and Fusion Module**:

    - **Function**: Convert retrieved analogical events into representations usable for enhancing predictions.
    - **Mechanism**: For the K retrieved analogical events, use an attention mechanism to assign weights and aggregate them based on their relevance to the current query. The attention weights consider temporal decay (giving higher weight to recent events) and structural matching degree. The aggregated analogical representation is then fused with the entity representations output by the standard graph encoder via a gating mechanism.
    - **Design Motivation**: Since different analogical events carry varying reference values, the attention mechanism automatically weights them; the gating mechanism allows the model to leverage analogical information more when it is helpful and less when it is not.

3. **Temporal-Aware Graph Encoder**:

    - **Function**: Encode structural and temporal information in historical graph snapshots.
    - **Mechanism**: Apply a Relation Graph Convolutional Network (R-GCN) at each timestamp $t$ to encode the current graph structure, and then pass information along the time dimension via a GRU to obtain temporal-aware entity representations for each timestamp. R-GCN aggregates neighbor information, while GRU captures temporal evolution trends.
    - **Design Motivation**: Combining graph structure encoding with temporal encoding is a standard paradigm in TKG modeling, ensuring high-quality base representations.

### Loss & Training
Cross-entropy loss is adopted to train and predict the probability distribution of the target entity over the candidate entity set. The analogical retrieval module and the prediction module are optimized end-to-end during training.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | RE-GCN | TiRGN | CyGNet | Gain |
|--------|------|------|--------|-------|--------|------|
| ICEWS14 | MRR | 0.412 | 0.369 | 0.385 | 0.348 | +7.0% |
| ICEWS18 | MRR | 0.348 | 0.315 | 0.332 | 0.299 | +4.8% |
| WIKI | MRR | 0.523 | 0.478 | 0.502 | 0.455 | +4.2% |
| YAGO | MRR | 0.618 | 0.582 | 0.598 | 0.561 | +3.3% |
| ICEWS14 | Hits@1 | 0.325 | 0.285 | 0.298 | 0.262 | +9.1% |
| ICEWS14 | Hits@10 | 0.578 | 0.532 | 0.551 | 0.518 | +4.9% |

### Ablation Study

| Configuration | MRR (ICEWS14) | Description |
|------|--------------|------|
| Full ANRE | 0.412 | Full model |
| w/o Analogical Retrieval | 0.385 | Removing the analogical module drops performance to the level of TiRGN |
| w/o Temporal Decay | 0.398 | Treating recent and distant analogies equally drops performance by 3.4% |
| w/o Gated Fusion | 0.395 | Direct concatenation reduces performance by 4.1% |
| Random Retrieval Replacement | 0.370 | Random retrieval yields poor results, showing the quality of retrieval is key |
| K=3 Analogical Events | 0.405 | Effective even with fewer analogical events |
| K=10 Analogical Events | 0.412 | Optimal value of K |
| K=20 Analogical Events | 0.408 | Too many events introduce noise instead |

### Key Findings
- The analogical retrieval module is the core contribution; removing it drops performance back to the baseline level.
- For entities with sparse historical interactions (fewer than 5 historical events), the improvement of ANRE is particularly significant (MRR increases by 12%+), validating the value of the analogical mechanism in cold-start scenarios.
- Temporal decay has a significantly positive impact on performance, indicating that recent analogical events have higher reference value than distant ones.
- The optimal retrieval count is around K=10; too few events lead to insufficient information, while too many introduce noise.

## Highlights & Insights
- The core idea of analogical reasoning is highly intuitive and aligns with human cognition—we regularly judge new situations based on similar cases in history. Formalizing this concept into a retrieval-augmented framework for TKGs is an elegant design.
- The analogical retrieval mechanism can naturally generalize to other temporal prediction tasks, such as social network link prediction and financial event forecasting.
- The gated fusion mechanism allows the model to adaptively decide whether to adopt the analogical information, avoiding the risk of forcibly introducing noise.

## Limitations & Future Work
- The time complexity of analogical retrieval grows linearly with history length, which may become a bottleneck on ultra-large-scale graphs.
- Currently, only relation-level similarity is considered; finer-grained entity attribute matching may yield further improvements.
- Comparison with recent LLM-based knowledge graph reasoning methods is not included.
- Future work can combine analogical retrieval with causal inference, upgrading "correlation-based analogy" to "causality-based analogy".

## Related Work & Insights
- **vs RE-GCN**: RE-GCN only encodes the direct history of the target entity, whereas ANRE expands the information source via analogical retrieval.
- **vs CyGNet**: CyGNet utilizes repetitive events via a copy mechanism, while ANRE goes further to utilize events that are structurally similar but not completely identical.
- **vs RAG Paradigm**: ANRE can be viewed as retrieval-augmented generation (RAG) in the TKG domain, retrieving analogical events instead of text documents.

## Rating
- Novelty: ⭐⭐⭐⭐ Applying analogical reasoning to TKG forecasting is a novel perspective, and the retrieval-guided approach is applicable across domains.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested across four datasets with multiple baselines and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Highly clear motivation and well-structured methodology description.
- Value: ⭐⭐⭐⭐ Provides a new research direction for TKG forecasting, with a highly generalizable retrieval-augmented concept.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models](g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)
- [\[ACL 2026\] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](../../ACL2026/time_series/stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[NeurIPS 2025\] Graph-based Neural Space Weather Forecasting](../../NeurIPS2025/time_series/graph-based_neural_space_weather_forecasting.md)
- [\[ACL 2025\] CTPD: Cross-Modal Temporal Pattern Discovery for Enhanced Multimodal Electronic Health Records Analysis](ctpd_cross-modal_temporal_pattern_discovery_for_enhanced_multimodal_electronic_h.md)

</div>

<!-- RELATED:END -->
