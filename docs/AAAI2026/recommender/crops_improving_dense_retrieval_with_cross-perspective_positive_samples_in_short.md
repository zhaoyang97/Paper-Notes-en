---
title: >-
  [Paper Note] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search
description: >-
  [Recommender Systems] This paper proposes CroPS, a data engine that enriches positive sample sets from three complementary perspectives—query reformulation behavior, recommender system interactions…
tags:
  - "Recommender Systems"
date: 2026-05-08
content_hash: 7a69d5e8551abf2a
---

# CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search

## Basic Information

- **Paper Link**: [arXiv:2511.15443](https://arxiv.org/abs/2511.15443)
- **Authors**: Ao Xie, Jiahui Chen, Quanzhi Zhu, Xiaoze Jiang, Zhiheng Qin, Enyun Yu, Han Li (Kuaishou Technology)
- **Conference**: AAAI 2026
- **Code**: Unavailable
- **Area**: Information Retrieval / Recommender Systems / Short-Video Search

## TL;DR

This paper proposes CroPS, a data engine that enriches positive sample sets from three complementary perspectives—query reformulation behavior, recommender system interactions, and LLM world knowledge—combined with Hierarchical Label Assignment (HLA) and the H-InfoNCE loss function, to break the filter bubble effect in industrial-scale dense retrieval systems. CroPS has been fully deployed in Kuaishou Search.

## Background & Motivation

### Core Problem: Filter Bubbles Induced by Self-Reinforcing Training Paradigms

Industrial short-video search systems commonly adopt a dual-encoder architecture for dense retrieval, with training data derived from historical exposure and interaction logs: videos clicked or watched by users serve as positive samples, while unexposed or filtered videos serve as negatives. This **self-reinforcing training paradigm** has a fundamental flaw—only content previously exposed by the system has any chance of becoming a positive sample, while semantically relevant but never-retrieved content is systematically excluded from the positive set and may even be incorrectly labeled as negative.

The paper illustrates this with a concrete example: when a user searches for "transformer," videos about electrical power transformers—though semantically relevant—are incorrectly treated as negatives because they have never been exposed, due to the dominance of deep learning content in historical data. This bias causes the model's retrieval behavior to become increasingly conservative and homogeneous, continuously degrading user experience.

### Limitations of Prior Work

Prior research has focused primarily on two directions: (1) architectural improvements, such as ColBERT's late interaction design; and (2) negative sampling strategies, such as ANCE's dynamic hard negatives and TriSampler. However, none of these approaches escape the self-reinforcing training paradigm—regardless of how negatives are sampled, positive samples remain confined to the historical exposure set, leaving the root cause of the filter bubble unaddressed.

### Paper Goals

The authors identify that **positive sample enrichment is a severely underexplored yet highly promising direction**. By introducing semantically relevant positive samples from beyond the historical exposure set across multiple perspectives, the data-level filter bubble boundary can be effectively broken. This insight constitutes the central motivation of CroPS.

## Method

### Overall Architecture

CroPS consists of three main modules:

1. **CroPS Data Engine**: Enriches positive sample sets from three complementary perspectives (query level, system level, and world knowledge level), forming $\mathcal{P} = \mathcal{P}_0 \cup \mathcal{P}_1 \cup \mathcal{P}_2 \cup \mathcal{P}_3$
2. **Hierarchical Label Assignment (HLA)**: Assigns hierarchical labels from 0 to 5 to positive and negative samples from different sources, replacing conventional binary labels
3. **H-InfoNCE Loss**: A loss function supporting multi-level contrastive learning, both efficient and compatible with HLA

### Key Design 1: Three-Perspective Positive Sample Enrichment

**(1) Query-Level Positive Sample Enrichment ($\mathcal{P}_1$)**

This perspective leverages user query reformulation behavior. When users are dissatisfied with initial search results, they issue semantically similar follow-up queries within a short time window (90 seconds). CroPS treats videos interacted with under reformulated queries as potential positive samples for the original query. A pretrained 6-layer Transformer semantic discriminator $\theta(\cdot)$ evaluates the relevance between the original query and candidate videos, with threshold $\alpha = 0.6$:

$$\mathcal{P}_1 = \bigcup_{q_i \in \mathcal{Q}} \{d_{ij} \in \mathcal{D}_i \mid \theta(q, d_{ij}) > \alpha\}$$

The key insight is that reformulation behavior itself encodes signals about what users are truly seeking. These positive samples typically fall within the retrieval blind spots of the original query, precisely supplementing what the self-reinforcing paradigm omits.

**(2) System-Level Positive Sample Enrichment ($\mathcal{P}_2$)**

This perspective breaks the data barrier between the search system and the recommender system. For a query $q$, the set of users who issued that query $\mathcal{U}$ is identified, and videos interacted with in the recommendation feed around the query timestamp (up to 100 per user) are retrieved. The same semantic discriminator then filters for semantically relevant videos:

$$\mathcal{P}_2 = \bigcup_{u_i \in \mathcal{U}} \{d_{ij} \in \mathcal{D}_i \mid \theta(q, d_{ij}) > \alpha\}$$

Recommender system interaction data is typically fresher and more closely aligned with individual user interests, thus complementing search data.

**(3) World Knowledge Enrichment ($\mathcal{P}_3$)**

This perspective uses an LLM (Qwen2.5-14B) as a "pseudo-retriever." A one-shot strategy is adopted: the LLM is provided with a query and a known relevant video as an example, and is prompted to generate descriptions of other videos matching the query as synthetic positive samples. A total of 35 million synthetic positive samples are generated. This strategy simulates the behavior of users seeking information from external sources when the platform fails to satisfy them, injecting external semantic associations and factual knowledge into the training process.

### Key Design 2: Hierarchical Label Assignment (HLA)

Positive samples from different sources vary in reliability and importance; treating them uniformly leads to suboptimal learning. HLA partitions samples into six levels (0–5):

| Level | Sample Type | Semantics |
|:---:|:---:|:---:|
| 5 | Query reformulation positives | Most directly reflect precise user intent |
| 4 | System-level positives / World knowledge positives / Clicked videos | Strong relevance signal |
| 3 | Ranking-stage exposed but unclicked videos | Moderate relevance |
| 2 | Ranking-stage unexposed videos | Weak / uncertain relevance |
| 1 | Videos filtered between pre-ranking and ranking | Low relevance |
| 0 | In-batch negatives | Irrelevant |

Query reformulation positives receive the highest label (5) because reformulation behavior represents users' active correction after dissatisfaction with initial results, making subsequent interactions the most faithful reflection of underlying information needs. Assigning the highest weight guides the model to proactively understand the ambiguity behind vague queries, thereby reducing reformulation frequency.

### Key Design 3: H-InfoNCE Loss

Standard InfoNCE assumes binary relevance (positive/negative) and cannot exploit the multi-level supervision signals provided by HLA. H-InfoNCE introduces a level-aware contrastive structure: for a positive sample with label $l$, only samples with labels strictly lower than $l$ are treated as negatives:

$$\mathcal{L} = -\sum_{d_i \in \mathcal{S}} \log \frac{\exp(\text{sim}(q, d_i) / \tau)}{\sum_{d_j \in \{d_i\} \cup \{d_k \in S | l_i > l_k\}} \exp(\text{sim}(q, d_j) / \tau)}$$

The implementation uses a masking matrix to filter incomparable samples and organizes inputs with label-indexed data structures. Contrastive losses across all levels are computed in a single forward pass, achieving speed comparable to standard InfoNCE.

## Key Experimental Results

### Table 1: Main Results on CPSQA Dataset

| Method | Recall@100 CT (%) | Recall@100 QR (%) | NDCG@4 (%) |
|:---:|:---:|:---:|:---:|
| BM25 | 42.9 | 22.5 | 64.8 |
| DPR | 56.0 | 30.7 | 66.5 |
| ANCE | 56.9 | 31.3 | 67.1 |
| ADORE+STAR | 59.4 | 31.9 | 67.4 |
| TriSampler | 59.8 | 32.2 | 66.9 |
| FS-LR | 59.6 | 33.0 | 66.0 |
| **CroPS** | **69.1** | **40.1** | **67.0** |

CroPS outperforms the strongest baseline (TriSampler) by 9.3% on CT and FS-LR by 7.1% on QR—both substantial margins. The large gain on QR indicates that users are more likely to find desired content on the first search attempt, reducing the need for reformulation.

### Table 2: Online A/B Test Results

| Model Type | CTR Gain | LPR Gain | RQR Reduction |
|:---:|:---:|:---:|:---:|
| Dense Model | +0.869% | +0.483% | -0.646% |
| Sparse Model | +0.783% | +0.423% | -0.614% |

In online A/B tests on Kuaishou Search, CroPS achieves a 0.869% increase in click-through rate, a 0.483% increase in long-play rate, and a 0.646% decrease in query reformulation rate on the Dense Model. These gains are substantial for an industrial-scale system. Consistent improvements on the Sparse Model further validate the architecture-agnostic nature of the approach.

## Highlights & Insights

1. **Precise Problem Identification**: Attributing the root cause of the filter bubble effect to the restricted positive sample space—rather than to negative sampling strategies—is a novel and incisive perspective. Prior work has been overly focused on negative sampling, while positive sample enrichment as "low-hanging fruit" has long been neglected.

2. **Well-Motivated Three-Perspective Design**: Query reformulation (capturing intent continuity), cross-system data (breaking information silos), and LLM world knowledge (introducing external semantics) each address a distinct dimension of the filter bubble. Ablation studies confirm that their gains are additive.

3. **Insightful HLA Design**: Assigning the highest weight to query reformulation positives reflects a deep understanding—such samples capture users' most authentic information needs, and their high training weight guides the model to proactively reduce user reformulation behavior, creating a positive feedback loop.

4. **Industrial Deployment Friendliness**: H-InfoNCE training speed is comparable to standard InfoNCE (88h vs. 178h or faster). CroPS introduces no additional inference overhead, is architecture-agnostic, and has been fully deployed serving hundreds of millions of users.

## Limitations & Future Work

1. **Dependence on the Semantic Discriminator**: The quality of query-level and system-level positive samples is highly dependent on the accuracy of the lightweight discriminator $\theta(\cdot)$. The choice of threshold $\alpha = 0.6$ lacks in-depth analysis, and the impact of different threshold values on noise introduction is insufficiently discussed.

2. **Quality of LLM-Synthesized Samples**: The quality control pipeline for the 35 million synthetic positive samples is not described in detail. LLM hallucinations may introduce incorrect semantic associations, and the filtering and quality evaluation strategies for synthetic samples are not sufficiently transparent.

3. **Non-Public Dataset**: The CPSQA dataset is constructed from Kuaishou's internal data, making experiments non-reproducible and fair comparison by external researchers infeasible.

4. **Generalizability of Label Level Design**: The HLA level partitioning (0–5) and specific assignments are empirically determined for the Kuaishou search scenario and may require redesign when transferred to other search contexts.

5. **Text-Only Modality Evaluation**: The document encoder relies solely on textual information (titles, captions, etc.) from videos, without leveraging visual or audio modalities, which may create semantic expression bottlenecks for certain query types.

## Related Work & Insights

- **DPR / ANCE / ADORE+STAR**: Represent the evolution of dense retrieval in negative sampling strategies; CroPS is complementary to these methods from the positive sample perspective.
- **FS-LR (Zheng et al., 2024)**: Introduces multi-level negative labels and serves as a precursor to the HLA concept on the negative side; CroPS extends the hierarchical idea to a unified positive-negative framework.
- **ColBERT / Poly-encoder**: Structure-enhanced methods, but late interaction is difficult to integrate with ANN indexing; CroPS's architecture-agnostic nature is a clear advantage.
- **Hierarchical/Weighted Strategies in Contrastive Learning**: Methods such as RINCE explore graded contrastive learning; CroPS's H-InfoNCE provides a more systematic hierarchical contrastive framework.

**Insights**: This work offers a paradigm-level solution to systematic data-level bias in retrieval and recommendation systems. The central insight is that when model performance bottlenecks stem from bias in the training data itself, optimizing model architecture or loss functions is merely symptomatic treatment; introducing multi-perspective signals at the data source is the fundamental cure. CroPS's approach of bridging search and recommendation data across system boundaries carries transferable value for any industrial system with multiple data silos.

## Rating

**4/5 ⭐**

A solid industrial systems paper. The problem is precisely identified, the method is systematically designed, and online deployment validation is thorough. Points are deducted primarily because the non-public dataset precludes reproducibility, and the core discriminator and label design lack sufficient sensitivity analysis. The hierarchical contrastive learning framework of HLA + H-InfoNCE makes a clear methodological contribution to the dense retrieval field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Semi-Supervised Synthetic Data Generation with Fine-Grained Relevance Control for Short Video Search Relevance Modeling](semi-supervised_synthetic_data_generation_with_fine-grained_relevance_control_fo.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)
- [\[ICLR 2026\] Search Arena: Analyzing Search-Augmented LLMs](../../ICLR2026/recommender/search_arena_analyzing_search-augmented_llms.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)

</div>

<!-- RELATED:END -->
