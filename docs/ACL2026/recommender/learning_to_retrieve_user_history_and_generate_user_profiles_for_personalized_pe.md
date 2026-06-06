---
title: >-
  [Paper Note] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction
description: >-
  [ACL 2026][Recommender Systems][Persuasiveness Prediction] This paper proposes the ReCAP framework, which significantly enhances personalized persuasiveness prediction by utilizing a trainable query generator and user pr…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Persuasiveness Prediction"
  - "User Profiling"
  - "Retrieval-Augmented Generation"
  - "DPO Training"
  - "Personalization"
date: 2026-05-08
content_hash: 6f7d9951bdaa4f26
---

# Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction

**Conference**: ACL 2026  
**arXiv**: [2601.05654](https://arxiv.org/abs/2601.05654)  
**Code**: [GitHub](https://github.com/holi-lab/ReCAP)  
**Area**: Personalized Recommendation and User Modeling  
**Keywords**: Persuasiveness Prediction, User Profiling, Retrieval-Augmented Generation, DPO Training, Personalization

## TL;DR

This paper proposes the ReCAP framework, which significantly enhances personalized persuasiveness prediction by utilizing a trainable query generator and user profile generator to retrieve persuasion-relevant information from user history and construct context-aware user profiles.

## Background & Motivation

**Background**: LLMs are increasingly deployed in decision-support applications to evaluate and generate persuasive messages, such as health coaching, educational tutoring, and targeted marketing. These scenarios require systems to predict the persuasive effect of a specific message on a specific user, known as **persuasiveness prediction**. Persuasion is inherently personalized—the same argument may be effective for one user but ineffective for another, depending on factors such as beliefs, values, experiences, and reasoning styles.

**Limitations of Prior Work**: Existing methods face three key limitations: (1) reliance on predefined explicit user attributes (e.g., ideology, demographics) that are often unavailable and fail to capture deep persuasion-related factors; (2) use of heuristic retrieval methods (e.g., most recent records or random sampling) to select history, failing to adapt dynamically to the current persuasion context; and (3) general profiling techniques (e.g., demographic extraction) have limited or even detrimental effects on persuasion prediction.

**Key Challenge**: Persuasiveness prediction requires **context-dependent** user information—which historical records are valuable for the current persuasion depends on the topic and stance of the post. However, there is a lack of a systematic framework to optimize the utilization of user history.

**Goal**: Design a trainable user profiling framework to learn "what to retrieve" and "how to summarize." **Key Insight**: Decompose user profile construction into two learnable modules—query generation and profile summarization—using downstream task performance as the supervisory signal. **Core Idea**: Effective user profiles are context-dependent and predictor-specific, rather than static attributes.

## Method

### Overall Architecture

ReCAP consists of a three-stage inference pipeline: (1) Retrieval stage—a trainable query generator $\phi^{\text{query}}$ generates user-focused retrieval queries to retrieve the top-k records from the user history $R_u$; (2) Profiling stage—a trainable profile generator $\phi^{\text{prof}}$ summarizes the retrieved records into a textual user profile $P_i$; (3) Prediction stage—a predictor $\mathcal{M}^{\text{pred}}$ predicts opinion change based on the post, comments, and profile. Training is conducted in three steps: DPO training for the profiler, record-level persuasion utility scoring, and DPO training for the query generator.

### Key Designs

1.  **DPO-based Profiler Training**:
    -   **Function**: Train the profile generator to produce user profiles most helpful for persuasiveness prediction.
    -   **Mechanism**: Since ground-truth profile annotations are unavailable, a weak supervision approach is used. Multiple sets of historical records are randomly sampled to generate candidate profiles. The downstream prediction F1 score is used as the quality metric for the profiles to construct preference pairs (high F1 profile vs. low F1 profile), and the profiler is trained via DPO to favor high-quality profiles.
    -   **Design Motivation**: To transform the open question of "what constitutes a good user profile" into an optimizable preference learning objective.

2.  **Record-level Persuasion Utility Scoring**:
    -   **Function**: Estimate the contribution of each historical record to persuasiveness prediction.
    -   **Mechanism**: Records are randomly grouped (5 per group), and the grouping process is repeated 3 times. For each group, the trained profiler generates 3 profiles (temperature 0.7). The F1 scores of all profiles containing a specific record are aggregated as its utility score.
    -   **Design Motivation**: Since direct labels for "which record is most useful for persuasion" are unavailable, supervisory signals are obtained indirectly via marginal contribution estimation.

3.  **Persuasion-aware Query Generator**:
    -   **Function**: Generate retrieval queries targeting user attributes rather than simply using the post text as the query.
    -   **Mechanism**: Two-stage training: first, the model is prompted to generate user-focused questions (e.g., "What are the user's core values regarding government intervention in personal choices?"); then, the model is trained to generate retrieval queries based on the post and questions. Preference pairs are constructed based on NDCG@5 (derived from utility scores) for DPO training.
    -   **Design Motivation**: Post text often lacks implicit user attribute information (values, experiences, etc.); using the post directly as a query fails to retrieve records truly relevant to persuasion.

### Loss & Training

Both the profiler and the query generator are trained using DPO, with Llama-3.1-8B-Instruct as the backbone model. The predictor remains frozen (Llama-3.1-8B, Llama-3.3-70B, GPT-4o-mini) to ensure the framework's universality. The training requires no ground-truth annotations and relies entirely on performance signals from the downstream task.

## Key Experimental Results

### Main Results

| Method | Llama-8B F1 | Llama-70B F1 | GPT-4o-mini F1 |
| :--- | :--- | :--- | :--- |
| No Personalization | 0.346 | 0.328 | 0.253 |
| PAG | 0.257 | 0.314 | 0.083 |
| RecurSumm | 0.313 | 0.414 | 0.105 |
| HSumm | 0.324 | 0.406 | 0.113 |
| Retrieval-only | 0.295 | 0.418 | 0.132 |
| **ReCAP (Ours)** | **0.400** | **0.466** | **0.279** |

### Ablation Study

| Configuration | Llama-70B F1 | Description |
| :--- | :--- | :--- |
| Our retrieval + Our profiler | **0.466** | Full model |
| BGE retrieval + Our profiler | 0.445 | Swapping with semantic retrieval |
| Our retrieval + Base profiler | 0.393 | Swapping with untrained profiler |
| Our retrieval + Demographic | 0.384 | Swapping with demographic profiling |
| HyDE retrieval + Our profiler | 0.451 | Swapping with HyDE retrieval |

### Key Findings
- Existing personalization frameworks (PAG, HSumm, RecurSumm) transfer poorly to persuasiveness prediction tasks; general profiles may even degrade performance.
- The trained profiler consistently outperforms demographic profiling and untrained profilers across all predictors.
- Effective profiling dimensions vary with post topics and predictor models—political posts rely on group identity, while socio-moral posts depend on personal values.
- Different predictors prefer different user records (top-5 overlap is only 0.24-0.28), indicating that profiles should be predictor-specific.
- Inference cost is only 1/6 to 1/13 of full-history summarization methods.

## Highlights & Insights
- Successfully transforms "what is a good user profile" into DPO preference learning without explicit labels, offering a concise and effective methodology.
- The three-step training workflow (Profiler → Utility Scoring → Query Generator) is elegantly designed with clear optimization goals for each step.
- The empirical finding that "profiles should be context-dependent and predictor-specific" provides broad insights, challenging the assumption of static user profiles.
- Generalization experiments on OpinionQA and PRISM demonstrate that the framework is not limited to the Reddit context.

## Limitations & Future Work
- Primarily evaluated on the CMV Reddit dataset; expansion to short-text dialogues or real-time recommendations requires additional validation.
- The record-level scoring stage consumes approximately 1.4B tokens (GPT-4o-mini), making training costs non-negligible.
- The dynamic evolution of user profiles over time has not been explored.
- Future work could extend the framework to multimodal interaction histories and real-time personalization scenarios.

## Related Work & Insights
- **vs PAG (Richardson et al., 2023)**: PAG uses fixed methods for retrieval and summarization, whereas ReCAP adapts dynamically through trainable queries and profilers.
- **vs HSumm/RecurSumm**: Full-history summarization methods are costly and perform poorly on persuasion tasks; retrieval-based methods are more efficient.
- **vs Zhang et al. (2025)**: Fine-tuning predictors to encode user information requires retraining as new data arrives; ReCAP keeps the predictor frozen, offering better scalability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic trainable user profiling framework for persuasiveness prediction, driven by DPO-based label-free training.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple predictors × Multiple retrieval strategies × Multiple profiling methods × Cross-dataset generalization × Efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear framework diagrams, rich levels of analysis, and rigorous experimental design.
- **Value**: ⭐⭐⭐⭐ The philosophy of "context-aware + predictor-specific" user profiling is transferable to other personalized scenarios like recommendation and dialogue.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation](from_past_to_path_masked_history_learning_for_next-item_prediction_in_generative.md)
- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)
- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)

</div>

<!-- RELATED:END -->
