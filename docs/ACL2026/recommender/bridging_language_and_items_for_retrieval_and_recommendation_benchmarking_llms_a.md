---
title: >-
  [Paper Note] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders
description: >-
  [ACL 2026][Recommender Systems][BLaIR] This paper introduces the Amazon Reviews 2023 large-scale dataset (570M reviews / 48M items) and constructs the BLaIR benchmark. Covering Sequential Recommendation, Collaborative Filtering, and Item Search (short and complex queries), the study benchmarks 11 top-tier LLMs as semantic encoders. It reveals that model ran
tags:
  - ACL 2026
  - Recommender Systems
  - BLaIR
  - Amazon Reviews 2023
date: 2026-05-08
content_hash: caabd8ee1a3ac137
---
# Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders

**Conference**: ACL 2026  
**arXiv**: [2403.03952](https://arxiv.org/abs/2403.03952)  
**Code**: https://github.com/hyp1231/BLaIR-Bench  
**Area**: Recommender Systems / LLM Semantic Encoding / Information Retrieval  
**Keywords**: BLaIR, Amazon Reviews 2023, Semantic Encoders, Complex Query Search, MTEB Correlation  

## TL;DR
This paper introduces the Amazon Reviews 2023 large-scale dataset (570M reviews / 48M items) and constructs the BLaIR benchmark. Covering Sequential Recommendation, Collaborative Filtering, and Item Search (short and complex queries), the study benchmarks 11 top-tier LLMs as semantic encoders. It reveals that model rankings on BLaIR are almost uncorrelated with MTEB (Spearman -0.476), highlighting the unique requirements of recommendation scenarios for semantic encoders.

## Background & Motivation

**Background**: Recommender systems have long relied on manual feature engineering. Although textual features (item titles/descriptions) are semantically rich, they are difficult to integrate directly into models. Recently, LLM-based semantic encoders (e.g., UniSRec, AlphaRec, EasyRec) have emerged as a trend, encoding item text into dense vectors for downstream recommendation models. However, the selection of LLMs typically follows MTEB (general text embedding benchmark) rankings, lacking evaluation specifically designed for recommendation.

**Limitations of Prior Work**: Two fundamental mismatches exist between MTEB and recommendation scenarios: (1) MTEB treats embeddings as the final product (direct similarity retrieval or simple classification), whereas in recommendation, embeddings serve as inputs for downstream Transformers or linear layers; (2) MTEB evaluates well-formed sentences/paragraphs, while recommendation item text consists of short, noisy titles requiring world knowledge for disambiguation.

**Key Challenge**: "Strong general embedding capability $\neq$ strong capability in recommendation scenarios." Models ranking high on MTEB may perform poorly in recommendation tasks, and vice versa. Current academic research still selects models based on MTEB, which constitutes a mismatch.

**Goal**: (1) Build a newer, cleaner, and larger Amazon Reviews dataset (replacing the 2018 version); (2) Design a unified benchmark covering three major recommendation scenarios; (3) Introduce a new sub-task, "Complex Query Item Search," to reflect real user behavior in the ChatGPT-buy / Rufus era; (4) Systematically verify the correlation between MTEB and recommendation scenarios using 11 SOTA LLMs.

**Key Insight**: Starting from "evaluation mismatch," the paper develops the data and benchmark first, then provides empirical evidence via a horizontal benchmark of 11 models, using Spearman coefficients to disprove the implicit assumption that MTEB is a sufficient proxy for recommendation task performance.

**Core Idea**: "Semantic encoders for recommendation $\neq$ semantic encoders for general NLP." The paper provides systematic evidence through rigorous multi-task and multi-dataset benchmarking + Borda Count cumulative rankings, while offering a new dataset and toolkit for community reproducibility and expansion.

## Method

### Overall Architecture
The BLaIR benchmark consists of three layers:

- **Data Layer**: Amazon Reviews 2023 (self-collected, 570M reviews / 48M items / 30.1B tokens / cleaned metadata / millisecond timestamps) + public datasets (ML-1M, Yelp, Book-Crossing, ESCI, Reddit-Movie).
- **Task Layer**: (1) Sequential Recommendation—UniSRec architecture $P(v_t | S_{t-1}) \propto \text{Trm}(\bm{e}'_{v_1}, ..., \bm{e}'_{v_{t-1}}) \cdot \bm{e}'_{v_t}$, across 5 datasets; (2) Collaborative Filtering—AlphaRec architecture $P(v|u) \propto \cos(W\bm{e}'_u, W\bm{e}'_v)/\tau$, across 6 datasets; (3) Item Search—zero-shot $\text{score}(q, v) = \bm{e}_q \cdot \bm{e}_v$, divided into short query (ESCI) and complex query (Amazon-C4 + Reddit-Movie) sub-tasks.
- **Model Layer**: 11 LLMs in three tiers—Small Open-source (<1B: RoBERTa, SimCSE, Sentence-T5, Qwen3-Emb-0.6B), Large Open-source (≥1B: Qwen3-Emb-4B/8B, SFR-Mistral, E5-Mistral, GritLM), and Closed-source (Gemini-Emb, text-emb-3-large).

To ensure fair comparison, an adaptor layer (PCA whitening by default) is added in sequential recommendation and collaborative filtering to project embeddings from different LLMs to a unified $d'$ dimension, ensuring identical parameter counts for downstream models.

### Key Designs

**1. Amazon Reviews 2023 Dataset: Larger, Newer, Cleaner Metadata, and Millisecond Timestamps**

The day-level timestamps in the 2018 Amazon dataset are unsuitable for time-split sequential recommendation, and the metadata contains significant noise and missing fields. The authors built a user-centric crawler to collect user-review pairs and re-parsed raw HTML metadata into structured JSON (including descriptions, features, multi-resolution images, and videos), with timestamps accurate to the millisecond. This covers 33 categories, 570M reviews, 48M items, and 54M users—a significant increase over the 2018 version. Millisecond timestamps allow for a global shared cutoff (e.g., 1628643414042), enabling a clean 8:1:1 split by timestamp rather than random per-user splits.

**2. Complex Query Item Search: Addressing Neglected Long-Sentence Descriptions**

Existing search benchmarks use short keyword queries, failing to reflect modern behaviors like those in ChatGPT-buy or Amazon Rufus where users describe needs in full paragraphs. Since real user chat histories are private, the authors used two approaches: Amazon-C4, which uses ChatGPT to rewrite 5-star reviews (≥100 characters) from Amazon Reviews 2023 into first-person queries; and Reddit-Movie, which uses real forum posts ($\text{is\_seeker}=\text{True}$) as queries and upvoted recommendations as ground-truth. The high Pearson correlation (0.94, p<0.01) between these two datasets on NDCG@100 validates semi-synthetic data as a reliable proxy for real user intent.

**3. Unified Adaptor + Borda Count Comprehensive Ranking: Isolating Encoder Performance**

Different LLMs have varying embedding dimensions, and datasets use different metric scales. Directly comparing raw embeddings would confound encoder strength with downstream decoder capacity. The authors use an adaptor layer (PCA whitening) to project all embeddings to a fixed $d'$ dimension for consistent downstream parameters. Comprehensive ranking is performed using Borda Count to avoid dominance by high-variance datasets, supplemented by Average (Overall) and Average (Task) scores to mitigate metric scale bias.

### Loss & Training
Sequential recommendation uses cross-entropy to train the Transformer; Collaborative filtering uses InfoNCE + in-batch negatives to train the AlphaRec linear layer; Search tasks remain zero-shot. Hyperparameter grid: lr ∈ {1e-3, 3e-4, 1e-4}, selected based on validation NDCG.

## Key Experimental Results

### Main Results: Comprehensive Performance of 11 LLMs across 4 Scenarios

| Model | Rank (Borda↓) | Avg.(Overall) | Avg.(Task) | Seq.Rec | Col.Fil | Short | Complex |
|-------|---------------|---------------|------------|---------|---------|-------|---------|
| FacebookAI/roberta-large | 11 (15.0) | 0.0263 | 0.0190 | 0.0393 | 0.0269 | 0.0096 | 0.0001 |
| Qwen3-Emb-0.6B | 10 (35.5) | 0.0507 | 0.0829 | 0.0415 | 0.0274 | 0.1876 | 0.0750 |
| Sentence-T5-large | 8 (42.5) | 0.0513 | 0.0801 | 0.0418 | 0.0304 | 0.1691 | 0.0790 |
| Qwen3-Emb-4B | 7 (69.5) | 0.0620 | 0.1036 | 0.0416 | 0.0350 | 0.2258 | 0.1120 |
| Qwen3-Emb-8B | 6 (54.0) | 0.0637 | 0.1069 | 0.0415 | 0.0362 | 0.2328 | 0.1172 |
| Gemini-Emb-001 | 5 (96.5) | 0.0629 | 0.1040 | 0.0434 | 0.0355 | 0.2233 | 0.1140 |
| SFR-Embedding-Mistral | 4 (98) | 0.0679 | 0.1160 | 0.0433 | 0.0372 | 0.2560 | 0.1273 |
| E5-Mistral-7B | 3 (101) | 0.0666 | 0.1120 | 0.0434 | 0.0377 | 0.2437 | 0.1232 |
| **GritLM-7B** | **2 (105)** | **0.0685** | **0.1161** | 0.0434 | 0.0385 | 0.2537 | 0.1290 |
| **text-emb-3-large (OpenAI)** | **1 (116)** | 0.0665 | 0.1112 | **0.0440** | 0.0366 | 0.2366 | 0.1278 |

"text-emb-3-large" ranks only 42nd on MTEB English v2 but takes the top spot in BLaIR Borda Count, strongly supporting the core argument that "MTEB $\neq$ recommendation."

### MTEB-BLaIR Correlation

| Metric | Value |
|------|------|
| Spearman correlation (BLaIR avg per task vs MTEB eng v2) | **-0.476** (p=0.233) |
| Pearson correlation (Amazon-C4 vs Reddit-Movie NDCG@100) | **0.94** (p<0.01) |

The Spearman correlation shows almost no relationship (slight negative correlation) between MTEB and BLaIR rankings. Conversely, high Pearson correlation confirms semi-synthetic data as a valid proxy for real complex queries.

### Ablation Study: Adaptor Design (PCA vs MRL)

| Model | Adaptor | Seq.Rec | Col.Fil |
|-------|---------|---------|---------|
| Qwen3-Emb-8B | PCA | **0.0415** | 0.0362 |
| Qwen3-Emb-8B | MRL | 0.0359 | **0.0392** |
| Gemini-Emb-001 | PCA | **0.0434** | **0.0355** |
| Gemini-Emb-001 | MRL | 0.0384 | 0.0313 |
| text-emb-3-large | PCA | **0.0440** | 0.0366 |
| text-emb-3-large | MRL | 0.0383 | **0.0379** |

PCA performs better in complex downstream tasks (Transformer SeqRec) as whitening encourages discriminability, while MRL is superior in simpler downstream tasks (Linear CF) as it preserves task-relevant information at low dimensions.

### Key Findings
- **MTEB $\neq$ Recommendation**: The Spearman -0.476 indicates MTEB rankings offer little reference value for recommendation LLM selection.
- **Scaling Varies by Downstream Complexity**: Large encoders show clear gains in Collaborative Filtering (single linear layer), yet benefits are marginal in Sequential Recommendation (Transformer decoder), suggesting that downstream capacity may dilute scaling gains from the encoder.
- **Title-only vs Title+Description**: Including descriptions provides inconsistent improvements, as LLM world knowledge often captures description information from the title alone.
- **Amazon-C4 is a Reliable Proxy**: High correlation with real-world Reddit-Movie data suggests semi-synthetic datasets can effectively evaluate or even train models for complex queries.
- **Case Study on Capability Caps**: Even the strongest models fail on specific complex intents (e.g., "Over the Top Bonkers Action Movies"), showing significant room for improvement in complex query processing.

## Highlights & Insights
- **Empirical Falsification of "MTEB $\approx$ Recommendation"**: Using a Spearman coefficient of -0.476 across 11 SOTA models, the paper corrects a commonly accepted but flawed assumption.
- **Substantial Dataset Contribution**: The Amazon Reviews 2023 dataset provides a massive 3x expansion, millisecond timestamps, and cleaner metadata, serving as a long-term foundation for the community.
- **Semi-Synthetic Validation Paradigm**: Demonstrating a 0.94 Pearson correlation between synthetic (Amazon-C4) and real (Reddit-Movie) data provides a robust methodology for low-cost, high-fidelity data construction.
- **Scaling Law Hypothesis in Two-Stage Systems**: The conjecture that late-stage module capacity limits the marginal utility of early-stage encoder scaling is a significant open research question.
- **Context-Dependent Adaptor Selection**: The finding that PCA suits complex downstream tasks while MRL suits simpler ones highlights that adaptors are not neutral choices.

## Limitations & Future Work
- **Language Scope**: Currently limited to English. Cross-lingual recommendation and search remain unexplored.
- **Model and Category Coverage**: Limited to 11 LLMs and specific Amazon categories due to computational constraints.
- **Performance Ceiling in Complex Queries**: Low absolute performance (NDCG@100 < 0.2) indicates a need for better intent modeling and reasoning.
- **Domain Specialization**: Reddit-Movie focuses only on film; performance in other verticals like apparel or travel is unknown.
- **Causal Analysis of Scaling Diminishment**: Quantitative verification of how task complexity regulates encoder scaling gains is still needed.

## Related Work & Insights
- **MTEB / MMTEB (Muennighoff 2023, Enevoldsen 2025)**: BLaIR emphasizes the unique requirements of downstream model integration and disambiguation of noisy/short text.
- **BEIR (Thakur 2021)**: BLaIR extends beyond IR to include collaborative filtering and sequential recommendation.
- **UniSRec / AlphaRec / EasyRec**: Rather than proposing a new encoding method, BLaIR provides a systematic framework for selecting the optimal LLM backbone.
- **ShoppingBench / Shopping MMLU**: While those evaluate LLMs as shopping agents, BLaIR evaluates them specifically as backbone semantic encoders.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](../../AAAI2026/recommender/inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[ACL 2026\] HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment](hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g.md)
- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)

</div>

<!-- RELATED:END -->
