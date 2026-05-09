---
title: >-
  [Paper Note] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling
description: >-
  [ACL 2026][Recommender Systems][Sequential Recommendation] This paper presents HORIZON, the first fully open-source large-scale cross-domain long-term recommendation benchmark. Built by merging all categories of Amazon Reviews into a unified interaction history covering 54M users and 35M items, HORIZON introduces a four-quadrant evaluation protocol that orthogonally decouples the temporal and user axes. The benchmark reveals that models such as BERT4Rec perform strongly in-distribution but degrade significantly under temporal extrapolation and unseen-user settings, and that LLMs do not consistently outperform dedicated architectures for user behaviour modeling.
tags:
  - ACL 2026
  - Recommender Systems
  - Sequential Recommendation
  - Cross-Domain User Modeling
  - Long-Term Behaviour Prediction
  - Temporal Generalization
  - LLM-based Recommendation
date: 2026-05-08
content_hash: d54d3d2f383c2e43
---

# HORIZON: A Benchmark for in-the-wild User Behaviour Modeling

**Conference**: ACL 2026
**arXiv**: [2604.17259](https://arxiv.org/abs/2604.17259)
**Code**: [https://github.com/microsoft/horizon-benchmark](https://github.com/microsoft/horizon-benchmark)
**Area**: Recommender Systems / User Behaviour Modeling
**Keywords**: Sequential Recommendation, Cross-Domain User Modeling, Long-Term Behaviour Prediction, Temporal Generalization, LLM-based Recommendation

## TL;DR

This paper presents HORIZON, the first fully open-source large-scale cross-domain long-term recommendation benchmark. Built by merging all categories of Amazon Reviews into a unified interaction history covering 54M users and 35M items, HORIZON introduces a four-quadrant evaluation protocol that orthogonally decouples the temporal and user axes. The benchmark reveals that models such as BERT4Rec perform strongly in-distribution but degrade significantly under temporal extrapolation and unseen-user settings, and that LLMs do not consistently outperform dedicated architectures for user behaviour modeling.

## Background & Motivation

**State of the Field**: Sequential recommendation is central to personalized systems. Mainstream methods (SASRec, BERT4Rec, etc.) have achieved notable progress on single-domain, short-sequence benchmarks such as MovieLens and Amazon Reviews. In practice, user behaviour spans multiple domains and platforms, and preferences evolve continuously over time.

**Limitations of Prior Work**: (1) Existing benchmarks predominantly focus on single-domain next-item prediction — Amazon Reviews spans multiple categories but is evaluated per category, failing to capture cross-category transfer behaviour. (2) Leave-One-Out and Ratio-Based evaluation protocols risk temporal leakage, as training interactions may temporally follow test interactions from other users. (3) No publicly available benchmark simultaneously supports cross-domain, long time-span, and unseen-user generalization evaluation. (4) PinnerFormer and USE, while well-designed, rely on proprietary data and are not reproducible.

**Root Cause**: Existing evaluation protocols conflate all generalization dimensions — temporal generalization (can the model predict behaviour in future time periods?), user generalization (can the model handle unseen users?), and cross-domain generalization (can the model leverage cross-category signals?) — are evaluated indiscriminately, preventing accurate diagnosis of specific model weaknesses.

**Paper Goals**: To construct a large-scale, cross-domain, temporally continuous public benchmark and design an evaluation protocol that orthogonally decouples the temporal and user dimensions, enabling systematic assessment of recommender model generalization across each axis.

**Starting Point**: All category-level interactions from Amazon Reviews 2023 are merged into a unified user history. A global temporal cutoff $\tau$=2020 is established, and four-quadrant evaluation is constructed along the axes of "seen/unseen users × pre-/post-cutoff time."

**Core Idea**: Generalization ability is not one-dimensional — the same model may excel in-distribution yet collapse under temporal extrapolation, or perform well on seen users but poorly on unseen ones. Decoupled evaluation is the key to diagnosing such failure modes.

## Method

### Overall Architecture

HORIZON is constructed from Amazon Reviews 2023 by merging interactions across all product categories into a unified dataset (54M users, 35M items, 486M interactions). With $\tau$=2020 as the global temporal cutoff, three tasks are defined: Task 1 is conventional next-item recommendation (with four-quadrant evaluation); Task 2 is LLM query rewriting for recommendation (converting user history into search queries for retrieval); Task 3 is LLM long-term user modeling (generating natural-language descriptions of future items for retrieval).

### Key Designs

1. **Four-Quadrant Decoupled Evaluation Protocol (Task 1)**:

    - **Function**: Orthogonally separates temporal generalization and user generalization.
    - **Mechanism**: (1a) In-distribution + temporally aligned — Leave-One-Out on seen users before the cutoff, i.e., the standard evaluation setting; (1b) In-distribution + temporal extrapolation — all interactions of the same users after the cutoff; (1c) Unseen users + temporally aligned — Leave-One-Out on entirely new users before the cutoff; (1d) Unseen users + temporal extrapolation — the most challenging setting, predicting interactions of new users after the cutoff. A single trained model is evaluated across all four settings; only (1a) is used during training.
    - **Design Motivation**: Traditional Leave-One-Out evaluation covers only (1a), while Ratio-Based evaluation conflates multiple dimensions. The four-quadrant design surfaces findings hidden by conventional protocols — for instance, BERT4Rec performs best under (1a) but degrades severely under (1c).

2. **LLM Query Rewriting for Recommendation (Task 2)**:

    - **Function**: Evaluates the ability of LLMs to translate user behaviour history into semantic search intent.
    - **Mechanism**: Given a user's interaction history, an LLM generates 10 diverse search queries $Q = \{q_1,...,q_{10}\}$. A pretrained BLAIR encoder maps both queries and items into a shared embedding space, and Top-K candidates are retrieved via an ANN index. Recall@K and Precision@K are used for evaluation.
    - **Design Motivation**: LLMs are naturally suited for semantic understanding. Query rewriting converts user behaviour into interpretable search intent, serving as a semantic complement to traditional ID-based approaches.

3. **LLM Long-Term User Modeling (Task 3)**:

    - **Function**: Evaluates the ability of LLMs to capture long-term preference evolution.
    - **Mechanism**: Given a user's pre-cutoff history, an LLM generates natural-language descriptions of 10 items the user may interact with in the future. These descriptions are matched against the product catalog through the same retrieval pipeline. Unlike Task 2, this setting requires predicting long-term evolution (multiple targets vs. a single target), and the evaluation window spans the entire post-cutoff period.
    - **Design Motivation**: Real-world recommendation requires anticipating long-term user needs — for proactive recommendation and inventory planning — rather than merely predicting the next click. This dimension is almost entirely absent from existing benchmarks.

### Loss & Training

In Task 1, conventional models are trained using the standard RecBole framework. In Tasks 2 and 3, LLMs are applied in a zero-shot manner; LoRA fine-tuning and full fine-tuning are also included as comparison baselines.

## Key Experimental Results

### Main Results

**Task 1: Four-Quadrant Evaluation Results (NDCG@10 / Recall@10)**

| Model | (1a) In-dist. Aligned | (1b) In-dist. Extrapolation | (1c) Unseen Aligned | (1d) Unseen Extrapolation |
|-------|-----------------------|-----------------------------|---------------------|--------------------------|
| BERT4Rec | 26.4 / 33.9 | 1.1 / 2.8 | 11.8 / 17.8 | 1.1 / 2.8 |
| SASRec | 25.2 / 34.1 | 2.9 / 6.2 | 17.8 / 26.2 | 3.1 / 6.7 |
| CORE | 8.5 / 12.1 | 0.09 / 0.26 | 5.9 / 11.1 | 0.10 / 0.32 |
| GRU4Rec | 0.08 / 0.14 | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 |

**Task 2: LLM Query Rewriting (Zero-Shot)**

| Model | Recall@10 | Recall@100 | Precision@10 |
|-------|-----------|------------|-------------|
| Qwen3-8B | 2.06 | 3.50 | 0.25 |
| LLaMA-3.1-8B | 1.62 | 2.84 | 0.20 |
| Gemma2-9B | 1.45 | 2.66 | 0.16 |

### Ablation Study

| Analysis Dimension | Finding | Details |
|--------------------|---------|---------|
| Temporal vs. User Generalization | Temporal extrapolation degrades more severely | BERT4Rec NDCG@10: 26.4→1.1 (−96%) |
| Seen vs. Unseen Users | SASRec degrades more robustly | SASRec maintains NDCG=17.8 under (1c) vs. BERT4Rec's 11.8 |
| LLM Scale Effect | Qwen3-235B ≈ Qwen3-8B | R@100: 3.40 vs. 3.50; scale and reasoning yield no significant gain |
| LLM Fine-tuning vs. Zero-Shot | Fine-tuning offers limited benefit | Zero-shot is superior in terms of scalability |
| Non-attention Models | GRU4Rec largely fails | Complex cross-domain settings require flexible contextual modeling |

### Key Findings

- BERT4Rec achieves the strongest performance under the standard (1a) setting (NDCG@10=26.4) but degrades severely on unseen users in (1c) (dropping to 11.8), while SASRec (17.8) proves more robust — a critical difference masked by conventional evaluation protocols.
- Temporal distribution shift is more damaging than user distribution shift: all models suffer a 90%+ performance collapse under (1b)/(1d), as ID-based models cannot handle entirely new items.
- LLMs do not exhibit overwhelming advantages on recommendation tasks — absolute Recall values are very low (<4%@100), indicating that LLMs' world knowledge does not readily translate into precise understanding of user preferences.
- Qwen3-235B in reasoning mode performs slightly worse than in non-reasoning mode (R@100: 2.96 vs. 3.40), suggesting that model scale and chain-of-thought reasoning offer limited benefit for recommendation tasks.

## Highlights & Insights

- The four-quadrant evaluation design is the paper's most significant methodological contribution — evaluating a single trained model across four orthogonal settings reveals generalization deficiencies systematically concealed by conventional protocols. This evaluation paradigm can be directly transferred to other settings requiring generalization assessment, such as dialogue systems and search ranking.
- Merging cross-domain user histories (rather than splitting by category) is a simple but powerful data processing strategy — average user history length increases from 3.86 to 9.07, unlocking richer cross-domain signals.
- The LLM recommendation paradigm of "query rewriting → retrieval" yields limited performance but provides interpretable intermediate representations (search queries), making it more amenable to debugging and analysis than black-box recommender models.

## Limitations & Future Work

- The benchmark is limited to English-language e-commerce data; multilingual settings and other domains (news, social media, video) are not covered.
- Only text modality is used; multimodal information such as product images is not incorporated.
- Due to computational constraints, models are trained on a 100K-user subset, leaving the full 54M-user dataset underutilized.
- LLM evaluation in Tasks 2 and 3 is conducted only on out-of-distribution users, without in-distribution comparison.

## Related Work & Insights

- **vs. Amazon Reviews**: Shares the same data source but adopts a fundamentally different evaluation approach — Amazon Reviews splits by category, whereas HORIZON merges all categories into a unified cross-domain history.
- **vs. PinnerFormer**: Pinterest's large-scale multi-year user modeling benchmark, but relies on proprietary data and is not reproducible. HORIZON is the first fully open-source benchmark of equivalent scope.
- **vs. MIND**: Microsoft's news recommendation dataset, covering only two weeks of single-domain history; HORIZON spans multiple years of cross-domain interactions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The four-quadrant decoupled evaluation is an important methodological contribution to the recommendation field; the cross-domain merging strategy is simple yet powerful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coverage of both conventional models and LLM baselines is comprehensive, though computational constraints prevent full utilization of the complete dataset.
- **Writing Quality**: ⭐⭐⭐⭐ The evaluation protocol design is clearly presented, and findings are communicated in a well-organized manner.
- **Value**: ⭐⭐⭐⭐⭐ Provides a much-needed standardized generalization testing framework for recommender system evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](../../AAAI2026/recommender/length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)

<!-- RELATED:END -->
