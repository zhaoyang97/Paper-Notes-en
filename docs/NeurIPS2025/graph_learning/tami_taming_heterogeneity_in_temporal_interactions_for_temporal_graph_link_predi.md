---
title: >-
  [Paper Note] TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction
description: >-
  [NeurIPS 2025][Graph Learning][Temporal graphs] This paper is the first to systematically identify the heterogeneity problem in temporal graph interactions (interaction intervals follow a power-law distribution), and proposes the TAMI framework comprising two modules—Log Time Encoding (LTE) and Link History Aggregation (LHA)—that can be seamlessly integrated into existing TGNNs, consistently improving link prediction performance across 16 datasets with gains of up to 87.05%.
tags:
  - NeurIPS 2025
  - Graph Learning
  - Temporal graphs
  - link prediction
  - time encoding
  - heterogeneity
  - interaction frequency
date: 2026-05-08
content_hash: a8b7ba0afa2fc553
---

# TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2510.23577](https://arxiv.org/abs/2510.23577)
**Code**: [GitHub](https://github.com/Alleinx/TAMI_temporal_graph)
**Area**: Graph Learning
**Keywords**: Temporal graphs, link prediction, time encoding, heterogeneity, interaction frequency

## TL;DR

This paper is the first to systematically identify the heterogeneity problem in temporal graph interactions (interaction intervals follow a power-law distribution), and proposes the TAMI framework comprising two modules—Log Time Encoding (LTE) and Link History Aggregation (LHA)—that can be seamlessly integrated into existing TGNNs, consistently improving link prediction performance across 16 datasets with gains of up to 87.05%.

## Background & Motivation

**Background**: Temporal graph link prediction is a fundamental task for predicting future node interactions in continuous-time dynamic graphs. Mainstream approaches include temporal neighbor aggregation-based TGNNs (DyGFormer, GraphMixer, TGAT, TGN) and random walk-based methods (CAWN).

**Limitations of Prior Work**:
   - Temporal interactions are inherently **heterogeneous**: a small fraction of node pairs account for the majority of interaction events, and interaction intervals follow power-law/highly skewed distributions (skewness of 2.385 on the UCI dataset).
   - Existing time encoding functions use sinusoidal functions $\bm{z}(t) = \cos(\Delta t \times \bm{\omega})$; when the $\Delta t$ distribution is highly right-skewed, the frequency parameter $\bm{\omega}$ becomes difficult to learn.
   - Node embeddings rely solely on the most recent $m$ interactions, causing them to be dominated by high-frequency neighbors while low-frequency interaction information is forgotten.

**Key Challenge**: High-frequency interactions dominate model learning, yet predicting low-frequency interactions is equally important (e.g., eating turkey on Thanksgiving vs. eating burgers daily). Existing methods exhibit large performance gaps between high-frequency and low-frequency interaction prediction.

**Goal**: (1) Alleviate the skewness problem in time encoding; (2) Prevent the interaction history of node pairs from being forgotten.

**Key Insight**: Observing that interaction intervals follow a power-law distribution, logarithmic transformation is applied to compress the skewness; an explicit pair-level interaction history is additionally maintained.

**Core Idea**: Apply logarithmic transformation to "tame" the skewness in time encoding, and use explicit link history aggregation to prevent low-frequency interactions from being forgotten.

## Method

### Overall Architecture

TAMI = LTE (Log Time Encoding) + any existing TGNN + LHA (Link History Aggregation)

It can seamlessly replace the time encoding and link prediction modules of any TGNN.

### Key Designs

#### LTE: Log Time Encoding

- **Function**: Applies a logarithmic transformation to the time delta prior to time encoding, compressing the right-skewed distribution.
- **Core Formula**:
$$\bm{z}_l(t) = \cos(\Delta t_l \times \bm{\omega}), \quad \Delta t_l = \ln(1 + \Delta t)$$
  compared to the original encoding $\bm{z}(t) = \cos(\Delta t \times \bm{\omega})$.
- **Theoretical Basis** (Proposition 1): If $\Delta t \sim \text{Pareto}(\alpha, x_m)$, the skewness of $\ln(1+\Delta t)$ is determined solely by $\alpha$. The logarithmic transformation effectively reduces skewness (from 2.385 to −1.14 on the UCI dataset).
- **Design Motivation**: High skewness causes the frequency parameter $\bm{\omega}$ to overfit high-frequency interactions while underfitting low-frequency ones. LTE makes the time delta distribution more balanced, facilitating learning of the frequency parameters.

#### LHA: Link History Aggregation

- **Function**: Explicitly maintains an embedding history of the most recent $k$ interactions for each node pair.
- **Core Formula**:
$$\bm{r}_{uv}^\tau = \gamma \times \bm{c}_{uv} + (1-\gamma) \times \bm{r}_{uv}^{t_1}$$
  where $\bm{c}_{uv} = \text{MLP}([\bm{h}_u; \bm{h}_v])$ encodes the current state, $\bm{r}_{uv}^{t_1}$ is the most recent historical edge embedding, and $\gamma$ controls the forgetting rate.
- **Final Link Prediction**:
$$p_{uv} = \text{MLP}([\bm{h}_u; \bm{h}_v; \bm{h}_{uv}])$$
  The historical embedding is concatenated with node embeddings and fed into the predictor.
- **Design Motivation**: When a node pair does not appear in each other's recent neighborhood ("Exclusive" case), their interaction history is completely lost. LHA compensates by explicitly storing and utilizing this history.
- **Space Complexity**: GPU overhead is only $O(bd_r)$ (where $b$ is the batch size); history is stored on CPU and loaded on demand.

### Loss & Training

- LTE replaces the original time encoding; LHA wraps around the link predictor.
- A most-recent aggregator is used to simplify $\bm{h}_{uv} = \bm{r}_{uv}^{t_1}$.
- $M_{uv}(\cdot)$ is updated after each interaction, maintaining at most $k$ history entries.

## Key Experimental Results

### Main Results: 13 Classic Datasets + 3 TGB Datasets

**AP improvement under random negative sampling** (representative datasets):

| Baseline → +TAMI | Enron | Lastfm | UCI | UN Vote |
|-------------|-------|--------|-----|---------|
| GraphMixer → | 82.26→90.97 (+10.59%) | 75.56→88.13 (+16.64%) | 93.38→96.20 (+3.02%) | 52.20→57.74 (+10.61%) |
| DyGFormer → | 92.46→92.66 (+0.22%) | 93.01→94.03 (+1.10%) | 95.66→96.72 (+1.11%) | 55.88→56.02 (+0.25%) |

**Maximum improvements under different negative sampling strategies**:
- Random sampling: up to 16.64% (Lastfm)
- Historical sampling: up to 38.48% (UN Vote)
- Inductive sampling: up to 54.13% (UN Vote)

### TGB Dataset Performance

| Dataset | DyGFormer MRR | +TAMI MRR | Gain |
|--------|--------------|-----------|------|
| tgbl-wiki | 0.798 (rank 1) | 0.815 (rank 1) | +2.13% |
| tgbl-review | 0.224 (rank 6) | 0.419 (rank 2) | **+87.05%** |
| tgbl-coin | 0.752 (rank 2) | 0.794 (rank 1) | +5.59% |

### Ablation Study

| Module | Avg. Gain (GraphMixer) | Avg. Gain (DyGFormer) |
|------|-------------------|-------------------|
| LTE only | ~0.5% | ~0.7% |
| LHA only | ~4.2% | ~0.2% |
| LTE+LHA (TAMI) | **~4.5%** | **~0.8%** |

LHA contributes more to GraphMixer (due to its weaker neighbor aggregation), while LTE contributes more to DyGFormer (which already has strong aggregation but whose time encoding is still affected by skewness).

### Performance Grouped by Interaction Frequency

GraphMixer on the UCI dataset, grouped by average interaction interval:
- Node pairs with interval < 100: originally high AP; small improvement from TAMI.
- Node pairs with interval > 10000: originally very low AP; **LHA improves AP by 25%+**, with LTE providing further gains.

### Key Findings

1. TAMI improves performance on all 13 classic datasets and 3 TGB datasets, with **no degradation on any**.
2. The largest gains occur under inductive sampling (up to 54.13%), as more unseen node pairs require historical information.
3. LTE not only improves accuracy but also accelerates training—total training time is reduced by up to 76.7%.
4. "Exclusive" node pairs (those not in each other's recent neighborhood) show the largest prediction improvement, validating the design motivation of LHA.
5. The 87.05% MRR improvement on tgbl-review is particularly striking, indicating severe long-tail interaction issues in that dataset.

## Highlights & Insights

1. **Precise problem formulation**: The first work to systematically identify and quantify the heterogeneity of temporal graph interactions and its impact on TGNNs.
2. **Simple yet effective solutions**: LTE is merely a $\ln(1+\cdot)$ transformation; LHA is a straightforward embedding history maintenance mechanism.
3. **Plug-and-play**: Can be directly integrated into any TGNN framework without modifying the underlying architecture.
4. **Training efficiency**: LTE makes frequency parameters easier to learn, accelerating convergence and reducing training time by up to 76.7%.
5. **Theoretical support**: The theoretical analysis of skewness reduction via logarithmic transformation under the Pareto distribution is concise and compelling.

## Limitations & Future Work

1. LHA maintains historical embeddings for all node pairs; the total space $O(Nd_r)$ may become a bottleneck for very large-scale graphs.
2. $\gamma$ and $k$ are manually tuned hyperparameters; adaptive learning of these values may be preferable.
3. The most-recent aggregator may be overly simplistic—an attention-based aggregator could potentially capture more complex historical patterns.
4. The theoretical analysis is limited to the Pareto distribution; applicability to other heavy-tailed distributions is not discussed.
5. Comparisons with more recent TGNN methods (e.g., frequency-aware methods from 2024) are absent.

## Related Work & Insights

- **DyGFormer** [Yu et al.]: Aggregates a large number of single-hop temporal neighbors via attention; TAMI achieves further improvements on top of this baseline.
- **GraphMixer** [Cong et al.]: MLP-Mixer with neighbor mean pooling; TAMI's LHA yields the largest improvements for this method.
- **CAWN** [Wang et al.]: Causal anonymous walks; random sampling may miss low-frequency interactions.
- **TGB** [Huang et al.]: Standardized temporal graph benchmark; TAMI advances DyGFormer to first place on multiple datasets.

## Rating

⭐⭐⭐⭐

The method is concise and effective, backed by thorough experiments (16 datasets, 3 negative sampling strategies, 2 TGNN integrations). The analysis of heterogeneity in temporal graph interactions addresses an important gap. Although the two modules—logarithmic transformation and link history—are simple, their intuition is clear and their empirical impact is significant. The main weaknesses are a relatively straightforward theoretical analysis and limited technical novelty (the method essentially combines a log transformation with pair-level memory).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)
- [\[ICLR 2026\] Revisiting Node Affinity Prediction in Temporal Graphs](../../ICLR2026/graph_learning/revisiting_node_affinity_prediction_in_temporal_graphs.md)
- [\[NeurIPS 2025\] Spatio-Temporal Directed Graph Learning for Account Takeover Fraud Detection](spatio-temporal_directed_graph_learning_for_account_takeover_fraud_detection.md)
- [\[AAAI 2026\] UniHR: Hierarchical Representation Learning for Unified Knowledge Graph Link Prediction](../../AAAI2026/graph_learning/unihr_hierarchical_representation_learning_for_unified_knowledge_graph_link_pred.md)

</div>

<!-- RELATED:END -->
