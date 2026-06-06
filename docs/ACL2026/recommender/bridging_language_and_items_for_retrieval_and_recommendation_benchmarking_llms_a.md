---
title: >-
  [Paper Note] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders
description: >-
  [ACL 2026][Recommender Systems][BLaIR] This paper introduces the Amazon Reviews 2023 large-scale dataset (570M reviews / 48M items) and constructs the BLaIR benchmark covering sequential recommendation…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "BLaIR"
  - "Amazon Reviews 2023"
  - "Semantic Encoder"
  - "Complex Query Search"
  - "MTEB Correlation"
date: 2026-05-08
content_hash: 3021ae5843a0bd16
---

# Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders

**Conference**: ACL 2026  
**arXiv**: [2403.03952](https://arxiv.org/abs/2403.03952)  
**Code**: https://github.com/hyp1231/BLaIR-Bench  
**Area**: Recommendation Systems / LLM Semantic Encoding / Information Retrieval  
**Keywords**: BLaIR, Amazon Reviews 2023, Semantic Encoder, Complex Query Search, MTEB Correlation  

## TL;DR
This paper introduces the Amazon Reviews 2023 large-scale dataset (570M reviews / 48M items) and constructs the BLaIR benchmark covering sequential recommendation, collaborative filtering, and item search (short and complex queries). Benchmarking 11 state-of-the-art LLMs as semantic encoders reveals that their rankings on BLaIR are nearly uncorrelated with MTEB (Spearman -0.476), highlighting that recommendation scenarios impose unique requirements on semantic encoders.

## Background & Motivation

**Background**: Recommendation systems have long relied on manual feature engineering. While text features (item title/description) are semantically rich, they are difficult to integrate directly. Recently, LLM-based semantic encoders (UniSRec / AlphaRec / EasyRec) have become a trend—encoding item text into dense vectors for downstream models. However, LLM selection often follows MTEB (generic text embedding benchmark) rankings, lacking evaluations specialized for recommendations.

**Limitations of Prior Work**: There are two fundamental mismatches between MTEB and recommendation scenarios: (1) MTEB treats embeddings as the final product (direct similarity search or simple classification), whereas in recommendations, embeddings are inputs to downstream Transformers or linear layers; (2) MTEB tests well-formed sentences/paragraphs, while recommendation item text consists of short, noisy titles requiring world knowledge for disambiguation.

**Key Challenge**: "Strong general embedding capability $\neq$ strong capability in recommendation scenarios." Models ranking high on MTEB may perform poorly in recommendation tasks and vice versa. Despite this, the academic community continues to select models based on MTEB, leading to a significant mismatch.

**Goal**: (1) Construct a newer, cleaner, and larger Amazon Reviews dataset (replacing the 2018 version); (2) Design a unified benchmark covering three major recommendation scenarios; (3) Introduce a new sub-task "Complex Query Item Search" to reflect real user behavior in the ChatGPT-buy / Rufus era; (4) Systematically verify the correlation between MTEB and recommendation scenarios using 11 SOTA LLMs.

**Key Insight**: Starting from the "evaluation mismatch," the authors first build the data, then the benchmark, and finally use 11 models for a horizontal benchmark to provide a rigorous Spearman coefficient that disproves the implicit assumption that "MTEB serves as a proxy for recommendation LLM selection."

**Core Idea**: "Semantic encoders for recommendation $\neq$ semantic encoders for general NLP." Systematic evidence is provided through a rigorous multi-task, multi-dataset benchmark and Borda Count consolidated rankings, alongside a new dataset and toolkit for community reproducibility and extensibility.

## Method

### Overall Architecture
The BLaIR benchmark consists of three layers:

- **Data Layer**: Amazon Reviews 2023 (self-collected, 570M reviews / 48M items / 30.1B tokens / cleaned metadata / millisecond timestamps) + Public datasets (ML-1M / Yelp / Book-Crossing / ESCI / Reddit-Movie).
- **Task Layer**: (1) Sequential Recommendation—UniSRec architecture $P(v_t | S_{t-1}) \propto \text{Trm}(\bm{e}'_{v_1}, ..., \bm{e}'_{v_{t-1}}) \cdot \bm{e}'_{v_t}$, 5 datasets; (2) Collaborative Filtering—AlphaRec architecture $P(v|u) \propto \cos(W\bm{e}'_u, W\bm{e}'_v)/\tau$, 6 datasets; (3) Item Search—zero-shot $\text{score}(q, v) = \bm{e}_q \cdot \bm{e}_v$, divided into short query (ESCI) and complex query (Amazon-C4 + Reddit-Movie) sub-tasks.
- **Model Layer**: 11 LLMs categorized into three tiers—Small Open Source (<1B: RoBERTa / SimCSE / Sentence-T5 / Qwen3-Emb-0.6B), Large Open Source (≥1B: Qwen3-Emb-4B/8B / SFR-Mistral / E5-Mistral / GritLM), and Closed Source (Gemini-Emb / text-emb-3-large).

To ensure fair comparison, an adaptor layer (PCA whitening by default) is added in sequential recommendation and collaborative filtering to project different LLM embedding dimensions into a unified $d'$, ensuring identical parameter counts for downstream models.

### Key Designs

1. **Amazon Reviews 2023 Dataset + Cleaned Metadata + Millisecond Timestamps**:
    - **Function**: Replaces the 2018 version as a larger, newer recommendation research infrastructure with cleaner metadata.
    - **Mechanism**: A self-built user-centric crawler collects user→reviews from public Amazon pages, re-parsing raw HTML metadata into structured JSON (containing description/features/multi-resolution images/videos) with millisecond precision. Final scale: 33 categories / 570M reviews / 48M items / 54M users, representing ×3.18 items and ×2.58 tokens compared to the 2018 version.
    - **Design Motivation**: Day-level timestamps in the old version were unsuitable for time-split sequential recommendation; old metadata was noisy and missing fields. Millisecond timestamps allow for a global shared cutoff (1628643414042 / 1658002729837) to achieve an 8:1:1 split by timestamp.

2. **New Complex Query Item Search Sub-task (Amazon-C4 + Reddit-Movie)**:
    - **Function**: Fills the gap in existing search benchmarks that use only short keyword queries, reflecting the trend of users describing needs in natural, long sentences in the ChatGPT-buy / Amazon Rufus era.
    - **Mechanism**: (1) Amazon-C4—Sampling 5-star reviews with ≥100 characters from Amazon Reviews 2023, and using ChatGPT to rewrite them into first-person queries (removing information that might leak the target item), resulting in ~20k pairs; (2) Reddit-Movie—Extracting real forum posts from reddit_movie_large_v1 where is_seeker=True and replies have upvotes >20 as queries, with the recommended movies as ground-truth.
    - **Design Motivation**: Real user complex chat histories are private; reverse-synthesizing from reviews serves as a "semi-synthetic but grounded in real user intent" compromise. Reddit data validates the representativeness of Amazon-C4—the Pearson correlation of NDCG@100 between the two is 0.94 (p<0.01), proving semi-synthetic data is a reliable proxy.

3. **Unified Adaptor + Borda Count Consolidated Ranking**:
    - **Function**: Addresses the fairness challenges of "varying LLM embedding dimensions" and "varying metric scales across datasets."
    - **Mechanism**: (a) Using PCA whitening to project all LLM embeddings to a fixed $d'$ dimension; (b) Using Borda Count (same as MTEB) for cross-dataset consolidated ranking to avoid dominance by high-variance datasets, while providing Avg.(Overall) and Avg.(Task) as supplementary metrics.
    - **Design Motivation**: Directly comparing models with different embedding dimensions changes downstream model parameters, failing to isolate "encoder capability" from "decoder capacity." The adaptor locks the comparison to the encoder level. Borda Count mitigates metric scale bias.

### Loss & Training
Sequential recommendation uses cross-entropy to train the Transformer; collaborative filtering uses InfoNCE with in-batch negatives to train the AlphaRec linear layer; the search task is zero-shot. Hyperparameter grid: lr ∈ {1e-3, 3e-4, 1e-4}, selected based on optimal val NDCG.

## Key Experimental Results

### Main Results: Consolidated Performance of 11 LLMs across 4 Scenarios

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

While text-emb-3-large ranks only 42nd on MTEB English v2, it takes the lead in BLaIR Borda Count, strongly supporting the core argument that "MTEB $\neq$ recommendation."

### MTEB-BLaIR Correlation

| Metric | Value |
|------|------|
| Spearman correlation (BLaIR avg per task vs MTEB eng v2) | **-0.476** (p=0.233) |
| Pearson correlation (Amazon-C4 vs Reddit-Movie NDCG@100) | **0.94** (p<0.01) |

The first row shows that MTEB rankings are nearly uncorrelated, and even slightly negatively correlated, with BLaIR rankings. The second row confirms that semi-synthetic and real-world complex queries are highly consistent.

### Adaptor Design Comparison (PCA vs MRL)

| Model | Adaptor | Seq.Rec | Col.Fil |
|-------|---------|---------|---------|
| Qwen3-Emb-8B | PCA | **0.0415** | 0.0362 |
| Qwen3-Emb-8B | MRL | 0.0359 | **0.0392** |
| Gemini-Emb-001 | PCA | **0.0434** | **0.0355** |
| Gemini-Emb-001 | MRL | 0.0384 | 0.0313 |
| text-emb-3-large | PCA | **0.0440** | 0.0366 |
| text-emb-3-large | MRL | 0.0383 | **0.0379** |

PCA performs better in complex downstream tasks (Transformer sequential recommendation) due to whitening; MRL performs better in simple downstream tasks (linear layer CF).

### Key Findings
- **MTEB $\neq$ Recommendation**: A Spearman correlation of -0.476 indicates that MTEB rankings have almost no reference value for recommendation LLM selection.
- **Scaling is effective in simple downstream tasks but weakens in complex ones**: In collaborative filtering (single linear layer), large encoders show clear gains (Qwen3-Emb-0.6B → 8B improves from 0.0274 to 0.0362); in sequential recommendation (Transformer decoder), performance is nearly identical (0.0415 vs 0.0415), suggesting that later modules in a "two-stage neural system" may dilute the scaling benefits of earlier modules.
- **Title-only vs Title+Description**: Adding descriptions inconsistently improves performance, indicating that long text introduces noise and LLM world knowledge already captures description information.
- **Amazon-C4 Is a Reliable Evaluation Proxy**: High consistency (Pearson 0.94) with Reddit-Movie real data proves semi-synthetic datasets can effectively evaluate or even train complex query models at low cost.
- **Failure Cases (GritLM-7B)**: Even the strongest model achieves an NDCG@100 of only 0.0734 on Reddit-Movie, highlighting significant room for improvement in parsing queries like "Over the Top Bonkers Action Movies."

## Highlights & Insights
- **Empirical Falsification of "MTEB $\neq$ Recommendation"**: Using a Spearman correlation of -0.476 across 11 SOTA models provides a rigorous counter-example to a commonly accepted assumption.
- **Amazon Reviews 2023 as a Major Contribution**: Scaling up by 3 times with millisecond timestamps and cleaner metadata provides the recommendation community with a new, high-value infrastructure.
- **Cross-Validation of Semi-Synthetic Data**: Proving the reliability of Amazon-C4 through cross-domain Pearson correlation with real Reddit data provides a blueprint for data construction in other fields.
- **Hypothesis of Weakened Scaling in "Two-Stage Neural Systems"**: The observation that marginal gains from scaling the encoder diminish when the later module (Transformer decoder) is sufficiently powerful is an open question for future study.
- **Adaptor Choice Depends on Downstream Complexity**: The finding that PCA suits complex downstream tasks while MRL suits simpler ones reminds researchers that the choice of adaptor is not neutral.

## Limitations & Future Work
- **English Coverage Only**: Multilingual recommendation and search were not tested; Amazon data is also primarily English.
- **Model and Category Constraints**: Limited by computational budgets, future work should expand model and product category coverage.
- **Low Ceiling for Complex Query Tasks**: Strongest models achieve low NDCG scores, indicating a need for better intent modeling, explicit reasoning, or ranking signals.
- **Reddit-Movie Domain Specificity**: Performance of complex queries in other vertical domains (books, apparel, travel) remains untested.
- **Lack of Causal Analysis for Scaling Weakness**: The authors provide a hypothesis but lack controlled experiments to quantitatively verify how "task complexity" moderates encoder scaling gains.

## Related Work & Insights
- **vs MTEB / MMTEB (Muennighoff 2023, Enevoldsen 2025)**: General text embedding benchmarks; BLaIR is specialized for recommendation, emphasizing downstream integration and noisy text disambiguation.
- **vs BEIR (Thakur 2021)**: IR-only benchmark; BLaIR includes sequential recommendation and collaborative filtering tasks.
- **vs BRIGHT (Su 2025)**: Reasoning-intensive retrieval benchmark; partially overlaps with the complex query sub-task but does not focus on recommendation settings.
- **vs UniSRec / AlphaRec / EasyRec**: Methods for encoding items with LLMs; this work provides a systematic answer for model selection rather than an individual method.
- **vs ShoppingBench / Shopping MMLU**: Evaluates LLMs as shopping agents; this work evaluates LLMs as backbone encoders.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of dataset, benchmark, and new sub-tasks is original; systematic validation of MTEB mismatch.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 models × 4 tasks × 14 datasets + adaptor ablation + metadata ablation + scaling analysis + failure studies.
- Writing Quality: ⭐⭐⭐⭐ Tight coupling between arguments and evidence; detailed per-dataset tables in the appendix.
- Value: ⭐⭐⭐⭐⭐ Amazon Reviews 2023 dataset and toolkit provide long-term infrastructure for the RecSys-NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation](../../NeurIPS2025/recommender/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommendation.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[ACL 2026\] HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment](hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g.md)
- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](../../AAAI2026/recommender/inductive_generative_recommendation_via_retrieval-based_speculation.md)

</div>

<!-- RELATED:END -->
