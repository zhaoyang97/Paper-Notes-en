---
title: >-
  [Paper Note] Personalized Benchmarking: Evaluating LLMs by Individual Preferences
description: >-
  [ACL 2026][Recommender Systems][Personalized benchmarking] This paper analyzes personalized rankings for 115 active Chatbot Arena users and finds that the average Spearman correlation between Bradley-Terry personalized r…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Personalized benchmarking"
  - "LLM ranking"
  - "user preference heterogeneity"
  - "Bradley-Terry model"
  - "topic and style analysis"
date: 2026-05-08
content_hash: d35e66f4d41094b4
---

# Personalized Benchmarking: Evaluating LLMs by Individual Preferences

**Conference**: ACL 2026
**arXiv**: [2604.18943](https://arxiv.org/abs/2604.18943)
**Code**: None
**Area**: LLM Evaluation / Personalized Recommendation
**Keywords**: Personalized benchmarking, LLM ranking, user preference heterogeneity, Bradley-Terry model, topic and style analysis

## TL;DR

This paper analyzes personalized rankings for 115 active Chatbot Arena users and finds that the average Spearman correlation between Bradley-Terry personalized rankings and the global ranking is only $\rho=0.04$ (with 57% of users exhibiting near-zero or negative correlation), demonstrating that aggregated benchmarks fail to reflect individual user preferences. Topic and style features are shown to successfully predict user-specific model rankings.

## Background & Motivation

**State of the Field**: Benchmarks such as Chatbot Arena, AlpacaEval, and MT-Bench aggregate preference votes across all users to construct global model rankings, implicitly assuming that user preferences are homogeneous. These rankings are widely used to guide model selection and development directions.

**Limitations of Prior Work**: (1) User needs vary considerably — software developers favor concise and precise technical responses, while creative writers prefer imaginative and expressive ones, making aggregated rankings suboptimal for both groups; (2) as LLMs are deployed to increasingly diverse user populations, aggregate metrics may recommend a model that is mediocre for everyone rather than optimal for specific user groups; (3) quantitative evidence characterizing how far individual preferences deviate from global consensus has been lacking.

**Root Cause**: A one-size-fits-all model ranking versus the fundamental heterogeneity of user preferences — users do not cluster around a shared ordering with minor deviations, but instead hold model preferences that are substantially or even diametrically opposed to the global ranking.

**Paper Goals**: (1) Compute personalized model rankings for individual users and quantify their deviation from the global ranking; (2) analyze the topic and stylistic heterogeneity of user queries; (3) validate whether topic and style features can predict user-specific model rankings.

**Starting Point**: The paper leverages existing pairwise comparison data from Chatbot Arena to compute personalized rankings using both ELO and Bradley-Terry scoring systems, then characterizes user heterogeneity via topic modeling (FastTopic) and style analysis (LISA).

**Core Idea**: Personalized benchmarking — rather than pursuing a single global ranking, the goal is to provide different model ranking recommendations for different types of users, using topic and style features as a bridge connecting user characteristics to model preferences.

## Method

### Overall Architecture

The approach proceeds in three stages: (1) compute personalized model rankings for 115 active users using both ELO and Bradley-Terry scoring systems, and measure Spearman correlation against the global ranking; (2) characterize the heterogeneity of user queries via FastTopic topic modeling and LISA style embeddings; (3) train a regression model on topic and style features to predict user-specific model ranking vectors.

### Key Designs

1. **Personalized Ranking via Dual Scoring Systems (ELO + Bradley-Terry)**:

    - Function: Quantify the deviation of personalized rankings from the global ranking from two complementary perspectives.
    - Mechanism: ELO maintains per-user model scores via incremental updates $ELO_u(m_a) \leftarrow ELO_u(m_a) + K(1 - E_a)$ ($K=32$); Bradley-Terry estimates user-specific model strength parameters $\beta_{u,m}$ via maximum likelihood, with preference probability $P(m_a \succ_u m_b) = \frac{\beta_{u,m_a}}{\beta_{u,m_a} + \beta_{u,m_b}}$. Correlation is computed only over models actually evaluated by each user.
    - Design Motivation: ELO's incremental update mechanism tends to smooth preference signals, potentially overestimating agreement with the global ranking; Bradley-Terry's probabilistic framework is more sensitive to individual preference variation and captures finer-grained pattern differences. The contrast between the two is itself a key finding.

2. **Multi-Dimensional User Heterogeneity Characterization (FastTopic + LISA + HypoGeniC)**:

    - Function: Characterize systematic differences in user queries along two interpretable dimensions — topic and style.
    - Mechanism: For topics, a global FastTopic model (10 topics) is trained on the union of all user queries; each user's topic profile is the mean topic distribution over their queries $\mathbf{t}_{u_i} \in \mathbb{R}^{10}$. For style, LISA generates 768-dimensional style embeddings, which are compressed via LDA into 6 meta-styles (Theatrical, Academic, Fervent, Hostile, Inquisitive, Fragmented); HypoGeniC is further used to generate natural-language style hypotheses.
    - Design Motivation: Topic and style are orthogonal yet complementary dimensions — topic captures *what* users ask, while style captures *how* they ask it. A shared global topic space ensures direct comparability across users.

3. **Topic- and Style-Driven Ranking Prediction**:

    - Function: Validate whether user features can predict personalized model rankings, providing a practical pathway toward deployable personalized benchmarks.
    - Mechanism: Each user's topic profile and LISA style embedding are concatenated into a 778-dimensional input $\mathbf{x}_{u_i} = [\mathbf{t}_{u_i}; \mathbf{s}_{u_i}]$, with the regression target being a 20-dimensional model score vector. ELO prediction uses an ensemble of 50 MLPs; BT prediction uses a single MLP with dropout.
    - Design Motivation: If topic and style features can effectively predict rankings, it implies that personalized benchmarking is achievable by inferring user profiles from a small number of queries, without requiring extensive preference elicitation.

### Loss & Training

Regression models use the Adam optimizer with standardized features and targets. The ELO model employs an ensemble of 50 MLPs with early stopping; the BT model uses a single MLP with dropout.

## Key Experimental Results

### Main Results

**Personalized vs. Global Ranking Correlation**

| Scoring System | Mean $\rho$ | Std. Dev. | Median | Users with Near-Zero/Negative Correlation |
|----------------|-------------|-----------|--------|------------------------------------------|
| ELO | 0.432 | 0.257 | 0.442 | 70% ($\rho < 0.5$) |
| Bradley-Terry | 0.043 | 0.283 | 0.011 | 57% ($\rho < 0.1$) |

### Ablation Study

**Ranking Prediction MAE**

| Model | ELO MAE | BT MAE |
|-------|---------|--------|
| Mean-Predictor (global mean) | 0.688 | 0.510 |
| Topic + Style (Ours) | 0.450 (↓35%) | 0.450 (↓12%) |

### Key Findings

- The mean BT personalized ranking correlation of $\rho=0.043$ is statistically indistinguishable from zero ($p=0.165$), meaning that for the majority of users, the personalized BT ranking is essentially a random ordering relative to the global ranking.
- The difference between ELO and BT results is itself statistically significant (paired Wilcoxon $p < 10^{-13}$), indicating that the two systems capture fundamentally different signals.
- Topic diversity varies substantially across users — from queries concentrated in as few as 4 topics to coverage of more than 20 diverse topics.
- The 6 meta-styles (Theatrical, Academic, etc.) effectively differentiate user groups, yielding 3 interpretable style clusters via k-means clustering.

## Highlights & Insights

- The BT model reveals preference divergence more sensitively than ELO — this is a feature rather than a limitation, since ELO's incremental update mechanism inherently smooths preference signals. This alerts the community that the choice of ranking algorithm affects the *visibility of personalization*.
- The predictive power of topic and style features demonstrates that personalized benchmarking is achievable in the near term — matching users to models requires only inferring user profiles from a small number of queries, without complex preference elicitation pipelines.
- User preferences do not represent minor perturbations around the global ranking but rather fundamentally different orderings — a finding that challenges the basic paradigm of current LLM evaluation.

## Limitations & Future Work

- Only 115 active users (≥25 votes) are analyzed, limiting sample size.
- Only English queries are covered; cross-lingual heterogeneity remains unexplored.
- The analysis is correlational rather than causal — whether topic/style differences directly cause preference differences requires further experimentation.
- The framework could be extended to real-time personalized recommendation on platforms such as Chatbot Arena.

## Related Work & Insights

- **vs. Chatbot Arena**: Aggregates all user preferences into a global ranking; this paper demonstrates that such rankings are effectively misleading for 57% of users.
- **vs. HyPerAlign**: Focuses on interpretable personalized alignment; this paper provides a quantitative framework for measuring preference divergence.
- **vs. RLHF**: Treats human preferences as a single aggregated signal; this paper argues that individual differences should be explicitly modeled.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic quantification of the gap between personalized and global rankings; findings are highly impactful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual scoring systems + topic/style analysis + regression prediction, though sample size is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Narrative is fluent, arguments build progressively, and quantitative evidence is well-presented.
- Value: ⭐⭐⭐⭐⭐ Poses a fundamental challenge to the LLM evaluation paradigm with a clear practical path forward.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)
- [\[AAAI 2026\] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios](../../AAAI2026/recommender/evaluating_llms_for_police_decision-making_a_framework_based_on_police_action_sc.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[ACL 2026\] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)
- [\[ACL 2026\] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters](icebreaker_for_conversational_agents_breaking_the_first-message_barrier_with_per.md)

</div>

<!-- RELATED:END -->
