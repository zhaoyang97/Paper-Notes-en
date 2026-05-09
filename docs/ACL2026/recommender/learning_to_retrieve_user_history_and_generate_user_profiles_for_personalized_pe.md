---
title: >-
  [Paper Note] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction
description: >-
  [ACL 2026][Recommender Systems][persuasiveness prediction] This paper proposes ReCAP, a framework featuring a trainable query generator and a user profile generator that retrieves persuasion-relevant information from user history and constructs context-aware user profiles, significantly improving personalized persuasiveness prediction.
tags:
  - ACL 2026
  - Recommender Systems
  - persuasiveness prediction
  - user profiling
  - retrieval augmentation
  - DPO training
  - personalization
date: 2026-05-08
content_hash: 8ef5a3658feb2808
---

# Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction

**Conference**: ACL 2026
**arXiv**: [2601.05654](https://arxiv.org/abs/2601.05654)
**Code**: [GitHub](https://github.com/holi-lab/ReCAP)
**Area**: Personalized Recommendation & User Modeling
**Keywords**: persuasiveness prediction, user profiling, retrieval augmentation, DPO training, personalization

## TL;DR

This paper proposes ReCAP, a framework featuring a trainable query generator and a user profile generator that retrieves persuasion-relevant information from user history and constructs context-aware user profiles, significantly improving personalized persuasiveness prediction.

## Background & Motivation

**State of the Field**: LLMs are increasingly applied in decision-support scenarios to evaluate and generate persuasive messages, including health coaching, educational tutoring, and targeted marketing. Such applications require systems capable of predicting how effectively a given message persuades a specific user — i.e., **persuasiveness prediction**. Persuasion is inherently personalized: the same argument may be effective for one user but not another, depending on beliefs, values, experiences, and reasoning styles.

**Limitations of Prior Work**: Existing approaches suffer from three critical limitations: (1) reliance on predefined explicit user attributes (e.g., ideology, demographics) that are often unavailable and fail to capture deeper persuasion-relevant factors; (2) use of heuristic retrieval strategies (e.g., most recent records or random sampling) that cannot dynamically adapt to the current persuasion context; and (3) general-purpose profiling techniques (e.g., demographic extraction) that provide limited benefit or even harm persuasiveness prediction.

**Root Cause**: Persuasiveness prediction requires **context-dependent** user information — which historical records are relevant depends on the topic and stance of the target post. Yet no systematic framework exists to optimize the use of user history for this purpose.

**Paper Goals**: To design a trainable user profiling framework that learns *what to retrieve* and *how to summarize*. **Starting Point**: Decompose user profile construction into two learnable modules — query generation and profile summarization — supervised by downstream task performance. **Core Idea**: Effective user profiles are context-dependent and predictor-specific, rather than static attribute collections.

## Method

### Overall Architecture

ReCAP comprises a three-stage inference pipeline: (1) **Retrieval** — a trainable query generator $\phi^{\text{query}}$ produces user-focused retrieval queries to fetch top-$k$ records from user history $R_u$; (2) **Profiling** — a trainable profile generator $\phi^{\text{prof}}$ summarizes the retrieved records into a textual user profile $P_i$; (3) **Prediction** — a predictor $\mathcal{M}^{\text{pred}}$ estimates opinion change based on the post, comment, and profile. Training proceeds in three steps: profiler DPO training, record-level persuasion utility scoring, and query generator DPO training.

### Key Designs

1. **DPO-Based Profiler Training**:

    - **Function**: Train the profile generator to produce user profiles most beneficial for persuasiveness prediction.
    - **Mechanism**: In the absence of ground-truth profile annotations, a weak supervision strategy is adopted — multiple sets of historical records are randomly sampled to generate candidate profiles; downstream prediction F1 serves as the profile quality measure; preference pairs (high-F1 profile vs. low-F1 profile) are constructed and used for DPO training.
    - **Design Motivation**: Reformulates the open-ended question of "what constitutes a good user profile" into an optimizable preference learning objective.

2. **Record-Level Persuasion Utility Scoring**:

    - **Function**: Estimate the contribution of each user history record to persuasiveness prediction.
    - **Mechanism**: Records are randomly partitioned into groups of five, repeated three times; the trained profiler generates three profiles per group (temperature 0.7); the F1 scores of all profiles containing a given record are aggregated as its utility score.
    - **Design Motivation**: Direct annotation of "which records are most persuasion-relevant" is unavailable; marginal contribution estimation provides an indirect supervision signal.

3. **Persuasion-Aware Query Generator**:

    - **Function**: Generate retrieval queries targeting user attributes rather than naively using post text as the query.
    - **Mechanism**: Two-stage training — the model first generates user-focused questions (e.g., "What are the user's core values regarding government intervention in personal choices?"), then learns to generate retrieval queries conditioned on the post and these questions; NDCG@5 (based on utility scores) is used to construct preference pairs for DPO training.
    - **Design Motivation**: Post text lacks implicit user attribute information (values, experiences, etc.); using posts directly as queries fails to retrieve records genuinely relevant to persuasion.

### Loss & Training

Both the profiler and query generator are trained with DPO using Llama-3.1-8B-Instruct as the backbone. The predictor is kept frozen (Llama-3.1-8B, Llama-3.3-70B, GPT-4o-mini), ensuring the generality of the profiling framework. Training requires no ground-truth annotations and relies entirely on downstream task performance signals.

## Key Experimental Results

### Main Results

| Method | Llama-8B F1 | Llama-70B F1 | GPT-4o-mini F1 |
|--------|-------------|--------------|----------------|
| No Personalization | 0.346 | 0.328 | 0.253 |
| PAG | 0.257 | 0.314 | 0.083 |
| RecurSumm | 0.313 | 0.414 | 0.105 |
| HSumm | 0.324 | 0.406 | 0.113 |
| Retrieval-only | 0.295 | 0.418 | 0.132 |
| **ReCAP (Ours)** | **0.400** | **0.466** | **0.279** |

### Ablation Study

| Configuration | Llama-70B F1 | Note |
|---------------|--------------|------|
| Our retrieval + Our profiler | **0.466** | Full model |
| BGE retrieval + Our profiler | 0.445 | Replaced with semantic retrieval |
| Our retrieval + Base profiler | 0.393 | Replaced with untrained profiler |
| Our retrieval + Demographic | 0.384 | Replaced with demographic profiling |
| HyDE retrieval + Our profiler | 0.451 | Replaced with HyDE retrieval |

### Key Findings

- Existing personalization frameworks (PAG, HSumm, RecurSumm) transfer poorly to persuasiveness prediction; general-purpose profiling can even degrade performance.
- The trained profiler consistently outperforms demographic profiling and untrained profilers across all predictors.
- Effective profile dimensions vary with post topic and predictor model — political posts rely on group identity, while social-moral posts rely on personal values.
- Different predictors favor different user records (top-5 overlap of only 0.24–0.28), indicating that profiles should be predictor-specific.
- Inference cost is only 1/6 to 1/13 that of full-history summarization methods.

## Highlights & Insights

- Reformulating "what constitutes a good user profile" as DPO preference learning — without explicit annotations — is methodologically elegant and effective.
- The three-step training pipeline (profiler → utility scoring → query generator) is carefully designed, with each step having a clear optimization objective.
- The empirical finding that "profiles should be context-dependent and predictor-specific" carries broad implications, challenging the static user profile assumption.
- Generalization experiments on OpinionQA and PRISM demonstrate that the framework is not limited to Reddit-based scenarios.

## Limitations & Future Work

- Evaluation is primarily conducted on the CMV Reddit dataset; extension to short-text dialogue or real-time recommendation requires additional validation.
- The record-level scoring stage consumes approximately 1.4B tokens (GPT-4o-mini), imposing non-trivial training costs.
- Dynamic modeling of user profiles evolving over time is not explored.
- Future work may extend the framework to multimodal interaction histories and real-time personalization scenarios.

## Related Work & Insights

- **vs. PAG (Richardson et al., 2023)**: PAG retrieves and summarizes with fixed strategies; ReCAP dynamically adapts via trainable query and profile generators.
- **vs. HSumm/RecurSumm**: Full-history summarization methods are costly and underperform on persuasion tasks; retrieval-based approaches are more efficient.
- **vs. Zhang et al. (2025)**: Fine-tuning the predictor to encode user information requires retraining as new data arrives; ReCAP keeps the predictor frozen, offering greater scalability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First systematic trainable user profiling framework for persuasiveness prediction; annotation-free training driven by DPO.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple predictors × retrieval strategies × profiling methods × cross-dataset generalization × efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear architectural diagrams, rich analytical depth, rigorous experimental design.
- **Value**: ⭐⭐⭐⭐ The concept of context-aware, predictor-specific user profiling is transferable to personalization in recommendation, dialogue, and beyond.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[ICLR 2026\] ProPerSim: Developing Proactive and Personalized AI Assistants through User-Assistant Simulation](../../ICLR2026/recommender/propersim_developing_proactive_and_personalized_ai_assistants_through_user-assis.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents](decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_unstruc.md)
- [\[NeurIPS 2025\] VisualLens: Personalization through Task-Agnostic Visual History](../../NeurIPS2025/recommender/visuallens_personalization_through_task-agnostic_visual_history.md)

<!-- RELATED:END -->
