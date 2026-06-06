---
title: >-
  [Paper Note] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling
description: >-
  [ACL 2026][Recommender Systems][Sequential Recommendation] This paper proposes HORIZON, the first fully open-source large-scale cross-domain long-term recommendation benchmark. By merging all categories from Amazon Revie…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Sequential Recommendation"
  - "Cross-domain User Modeling"
  - "Long-term Behavior Prediction"
  - "Temporal Generalization"
  - "LLM Recommendation"
date: 2026-05-08
content_hash: d5ea3ef13081b2da
---

# HORIZON: A Benchmark for in-the-wild User Behaviour Modeling

**Conference**: ACL 2026  
**arXiv**: [2604.17259](https://arxiv.org/abs/2604.17259)  
**Code**: [https://github.com/microsoft/horizon-benchmark](https://github.com/microsoft/horizon-benchmark)  
**Area**: Recommender Systems / User Behavior Modeling  
**Keywords**: Sequential Recommendation, Cross-domain User Modeling, Long-term Behavior Prediction, Temporal Generalization, LLM Recommendation

## TL;DR

This paper proposes HORIZON, the first fully open-source large-scale cross-domain long-term recommendation benchmark. By merging all categories from Amazon Reviews, it constructs a unified interaction history containing 54M users and 35M items. It introduces a four-quadrant evaluation protocol decoupled along the temporal axis and user dimension, revealing that models such as BERT4Rec perform strongly in-distribution but suffer significant degradation in temporal extrapolation and unseen user scenarios. Furthermore, it demonstrates that LLMs do not consistently outperform specialized architectures in user behavior modeling.

## Background & Motivation

**Background**: Sequential recommendation is the core of personalized systems. Mainstream methods (e.g., SASRec, BERT4Rec) have made significant progress on single-domain short-sequence benchmarks like MovieLens and Amazon Reviews. In reality, user behavior spans multiple domains and platforms, and preferences continuously evolve over time.

**Limitations of Prior Work**: (1) Existing benchmarks primarily focus on single-domain next-item prediction; while Amazon Reviews covers multiple categories, it is usually partitioned by category during evaluation, failing to capture cross-category transition behaviors. (2) Leave-One-Out and Ratio-Based evaluations carry temporal leakage risks, as user interactions in the training set might occur chronologically later than interactions of other users in the test set. (3) No public benchmark simultaneously supports generalization evaluation across domains, long time spans, and unseen users. (4) While PinnerFormer and USE are well-designed, they rely on private data and are not reproducible.

**Key Challenge**: Existing evaluation protocols conflate all dimensions of generalization—temporal generalization (whether a model can predict behavior in future periods), user generalization (whether a model can handle unseen users), and cross-domain generalization (whether a model can utilize cross-category signals) are evaluated indiscriminately, making it impossible to accurately diagnose specific model weaknesses.

**Goal**: To build a large-scale, cross-domain, temporally continuous public benchmark and design an evaluation protocol that orthogonally decouples the temporal axis and user dimensions to systematically assess the generalization capabilities of recommendation models across various dimensions.

**Key Insight**: All category interactions from Amazon Reviews 2023 are merged into a unified user history. A global temporal cutoff point $\tau$=2020 is set to construct a four-quadrant evaluation based on "Seen/Unseen Users × Pre-cutoff/Post-cutoff Time."

**Core Idea**: Generalization capability is not a single dimension; the same model might perform excellently in-distribution but collapse during temporal extrapolation, or be strong for seen users but weak for unseen ones. Decoupled evaluation is critical for diagnosing these issues.

## Method

### Overall Architecture

Starting from Amazon Reviews 2023, HORIZON merges user interactions across all categories to construct a unified dataset (54M users, 35M items, 486M interactions). With $\tau$=2020 as the global temporal cutoff, three tasks are defined: Task 1: Traditional next-item recommendation (four-quadrant evaluation); Task 2: LLM query rewriting for recommendation (transforming user history into search queries for retrieval); Task 3: LLM long-term user modeling (generating natural language descriptions of the next 10 items for retrieval).

### Key Designs

1.  **Four-Quadrant Decoupled Evaluation Protocol (Task 1)**:
    - **Function**: Orthogonally separates the two dimensions of temporal generalization and user generalization.
    - **Mechanism**: (1a) In-distribution + Temporal Alignment: Leave-One-Out for seen users before the cutoff (standard evaluation); (1b) In-distribution + Temporal Extrapolation: All interactions for the same users after the cutoff; (1c) Unseen Users + Temporal Alignment: Leave-One-Out for completely new users before the cutoff; (1d) Unseen Users + Temporal Extrapolation: The most difficult setting, predictions for new users after the cutoff. A single trained model is evaluated across all four settings, with only (1a) used for training.
    - **Design Motivation**: Traditional Leave-One-Out evaluation only covers (1a), while Ratio-Based methods mix multiple dimensions. The four-quadrant design reveals critical findings obscured by traditional protocols, such as BERT4Rec performing best in (1a) but degrading severely in (1c).

2.  **LLM Query Rewriting for Recommendation (Task 2)**:
    - **Function**: Evaluates the LLM's ability to transform user behavior history into semantic search intents.
    - **Mechanism**: Given a user interaction history, the LLM generates 10 diverse search queries $Q = \{q_1,...,q_{10}\}$. A pre-trained BLAIR encoder maps queries and items into the same embedding space, and Top-K candidates are retrieved via an ANN index. Performance is evaluated using Recall@K and Precision@K.
    - **Design Motivation**: LLMs are naturally skilled at semantic understanding. Query rewriting can transform user behavior into interpretable search intents, serving as a semantic complement to traditional ID-based methods.

3.  **LLM Long-term User Modeling (Task 3)**:
    - **Function**: Evaluates the LLM's ability to capture the evolution of long-term preferences.
    - **Mechanism**: Given user history before the cutoff, the LLM generates natural language descriptions of the 10 items likely to be interacted with in the future, matching them against the item catalog through the same retrieval pipeline. This differs from Task 2 by requiring the prediction of long-term evolution (multiple targets vs. a single target) with an evaluation window covering the entire post-cutoff period.
    - **Design Motivation**: Real-world recommendation needs to foresee long-term user needs (e.g., proactive recommendation, inventory planning) rather than just predicting the next click. This is a dimension rarely evaluated in existing benchmarks.

### Loss & Training

In Task 1, traditional models are trained using the standard RecBole framework. In Tasks 2 and 3, LLMs are used in a zero-shot manner, with LoRA fine-tuning and full-parameter fine-tuning provided as comparative baselines.

## Key Experimental Results

### Main Results

**Task 1: Four-Quadrant Evaluation Results (NDCG@10 / Recall@10)**

| Model | (1a) In-dist. Aligned | (1b) In-dist. Extrapol. | (1c) Unseen Aligned | (1d) Unseen Extrapol. |
| :--- | :--- | :--- | :--- | :--- |
| BERT4Rec | 26.4 / 33.9 | 1.1 / 2.8 | 11.8 / 17.8 | 1.1 / 2.8 |
| SASRec | 25.2 / 34.1 | 2.9 / 6.2 | 17.8 / 26.2 | 3.1 / 6.7 |
| CORE | 8.5 / 12.1 | 0.09 / 0.26 | 5.9 / 11.1 | 0.10 / 0.32 |
| GRU4Rec | 0.08 / 0.14 | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 |

**Task 2: LLM Query Rewriting (Zero-shot)**

| Model | Recall@10 | Recall@100 | Precision@10 |
| :--- | :--- | :--- | :--- |
| Qwen3-8B | 2.06 | 3.50 | 0.25 |
| LLaMA-3.1-8B | 1.62 | 2.84 | 0.20 |
| Gemma2-9B | 1.45 | 2.66 | 0.16 |

### Ablation Study

| Analysis Dimension | Finding | Description |
| :--- | :--- | :--- |
| Temporal vs. User Generalization | Temporal extrapolation degrades more severely | BERT4Rec NDCG@10: 26.4 $\rightarrow$ 1.1 (-96%) |
| Seen vs. Unseen Users | SASRec is more robust to degradation | SASRec maintains NDCG=17.8 in (1c) vs. BERT4Rec's 11.8 |
| LLM Scaling Effects | Qwen3-235B $\approx$ Qwen3-8B | R@100: 3.40 vs 3.50; no significant gain from scale/reasoning |
| LLM Fine-tuning vs. Zero-shot | Limited effect of fine-tuning | Zero-shot is superior in terms of scalability |
| Non-attention Models | GRU4Rec largely fails | Complex cross-domain environments require flexible context modeling |

### Key Findings

- BERT4Rec is strongest in the standard (1a) setting (NDCG@10=26.4), but degrades heavily for unseen users (1c) (dropping to 11.8), whereas SASRec (17.8) is more robust—traditional evaluations mask this critical difference.
- Temporal distribution shift is more fatal than user distribution shift: all models experience a 90%+ drop in performance on (1b)/(1d) because ID-based models cannot handle entirely new items.
- LLMs do not demonstrate an overwhelming advantage in recommendation tasks—absolute Recall values are very low (<4%@100), suggesting that LLM world knowledge is difficult to translate directly into precise user preference understanding.
- Qwen3-235B in reasoning mode is slightly worse than in non-reasoning mode (R@100: 2.96 vs 3.40), indicating that model scale and chain-of-thought provide limited help for recommendation tasks.

## Highlights & Insights

- The four-quadrant evaluation design is the primary methodological contribution of this work. By evaluating a single trained model across four orthogonal settings, it reveals generalization flaws systematically hidden by traditional protocols. This evaluation paradigm can be directly transferred to any scenario requiring generalization assessment, such as dialogue systems or search ranking.
- Merging user history across domains (rather than partitioning by category) is a simple but powerful data processing idea. The average user history length increases from 3.86 to 9.07, unlocking more cross-domain signals.
- While the "query rewriting $\rightarrow$ retrieval" paradigm for LLM recommendation shows limited current effectiveness, it provides interpretable intermediate representations (search queries), making it more suitable for debugging and analysis than black-box recommendation models.

## Limitations & Future Work

- Limited to English e-commerce data; multilingual and other scenarios (news, social media, video) are not covered.
- Only utilizes the text modality; multimodal information like product images is not integrated.
- Due to computational constraints, models were only trained on a 100K user subset, failing to fully utilize the total 54M user data.
- LLM evaluation for Tasks 2/3 was conducted only on OOD users, lacking in-distribution comparisons.

## Related Work & Insights

- **vs Amazon Reviews**: Same data source but fundamentally different evaluation. Amazon Reviews is partitioned by category, while HORIZON merges them into a cross-domain unified history.
- **vs PinnerFormer**: Scale-wise similar to Pinterest’s multi-year user modeling, but PinnerFormer uses private data. HORIZON is the first open-source benchmark with equivalent positioning.
- **vs MIND**: Microsoft News Dataset covers only two weeks of history in a single domain, whereas HORIZON covers multi-year cross-domain interactions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The four-quadrant decoupled evaluation is a significant methodological contribution to the recommendation field; the cross-domain merger is simple yet powerful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage of traditional models and LLM baselines, though full data utilization was limited by computational resources.
- Writing Quality: ⭐⭐⭐⭐ The evaluation protocol is clearly designed, and findings are presented in an organized manner.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed standardized generalization testing framework for evaluating recommendation systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)
- [\[AAAI 2026\] Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction](../../AAAI2026/recommender/length-adaptive_interest_network_for_balancing_long_and_short_sequence_modeling_.md)

</div>

<!-- RELATED:END -->
