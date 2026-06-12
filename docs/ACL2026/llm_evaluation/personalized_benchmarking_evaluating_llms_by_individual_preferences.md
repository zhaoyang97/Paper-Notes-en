---
title: >-
  [Paper Note] Personalized Benchmarking: Evaluating LLMs by Individual Preferences
description: >-
  [ACL 2026][LLM Evaluation][Personalized benchmarking] This paper analyzes personalized rankings for 115 active users of Chatbot Arena and finds that the average Spearman correlation between personalized Bradley-Terry ran…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Personalized benchmarking"
  - "LLM ranking"
  - "user preference heterogeneity"
  - "Bradley-Terry model"
  - "topic and style analysis"
date: 2026-05-08
content_hash: 9226d6ee631595d5
---

# Personalized Benchmarking: Evaluating LLMs by Individual Preferences

**Conference**: ACL 2026  
**arXiv**: [2604.18943](https://arxiv.org/abs/2604.18943)  
**Code**: None  
**Area**: LLM Evaluation / Personalized Recommendation  
**Keywords**: Personalized benchmarking, LLM ranking, user preference heterogeneity, Bradley-Terry model, topic and style analysis

## TL;DR

This paper analyzes personalized rankings for 115 active users of Chatbot Arena and finds that the average Spearman correlation between personalized Bradley-Terry rankings and global rankings is only $\rho=0.04$ (with 57% of users showing near-zero or negative correlation). This demonstrates that aggregated benchmarks fail to reflect the individual preferences of most users. Furthermore, the study successfully predicts user-specific model rankings using topic and style features.

## Background & Motivation

**Background**: Benchmarks such as Chatbot Arena, AlpacaEval, and MT-Bench establish global model rankings by aggregating preference votes from all users, implicitly assuming that user preferences are homogeneous. These rankings are widely used to guide model selection and development directions.

**Limitations of Prior Work**: (1) User needs vary significantly—software developers may prefer concise and precise technical answers, while creative writers may prefer imaginative ones; aggregated rankings may be suboptimal for both. (2) As LLMs are deployed to increasingly diverse user groups, aggregated metrics might recommend a "mediocre" model for everyone rather than finding the "best" model for a specific user group. (3) There is a lack of quantitative evidence regarding how far individual preferences actually deviate from the global consensus.

**Key Challenge**: The contradiction between "one-size-fits-all" model rankings and the fundamental heterogeneity of user preferences—users do not just have minor deviations around a common ordering, but often possess model preferences that are completely different or even opposite to the global ranking.

**Goal**: (1) Compute personalized model rankings for each user and quantify their deviation from global rankings; (2) Analyze the heterogeneity of user queries in terms of topic and style; (3) Verify whether user-specific model rankings can be predicted using topic and style features.

**Key Insight**: Utilizing existing pairwise comparison data from Chatbot Arena, the study calculates personalized rankings using both ELO and Bradley-Terry scoring systems. It then characterizes user heterogeneity through topic modeling (FastTopic) and style analysis (LISA).

**Core Idea**: Personalized Benchmarking—instead of pursuing a single global ranking, the goal is to provide different model ranking recommendations for different types of users, using topic and style features as a bridge to connect user characteristics with model preferences.

## Method

### Overall Architecture

The method consists of three steps: (1) Calculating personalized model rankings for 115 active users using ELO and Bradley-Terry scoring systems and performing Spearman correlation analysis against global rankings; (2) Characterizing the heterogeneity of user queries using FastTopic modeling and LISA style embeddings; (3) Training regression models with topic and style features to predict user-specific model ranking vectors.

### Key Designs

1.  **Personalized Ranking with Dual Scoring (ELO + Bradley-Terry)**:
    - **Function**: To quantify the deviation of personalized rankings from global rankings from two complementary perspectives.
    - **Mechanism**: ELO maintains scores for each model via incremental updates: $ELO_u(m_a) \leftarrow ELO_u(m_a) + K(1 - E_a)$ (with $K=32$); Bradley-Terry estimates user-specific model strength parameters $\beta_{u,m}$ via maximum likelihood, where the preference probability is $P(m_a \succ_u m_b) = \frac{\beta_{u,m_a}}{\beta_{u,m_a} + \beta_{u,m_b}}$. Crucially, correlations are calculated only for models actually evaluated by the user.
    - **Design Motivation**: The incremental update mechanism of ELO tends to smooth preference signals and may overestimate alignment with the global consensus. The Bradley-Terry probabilistic framework is more sensitive to individual preference variance and can capture finer pattern differences. The comparison between the two is itself a key finding.

2.  **Multidimensional User Heterogeneity Characterization (FastTopic + LISA + HypoGeniC)**:
    - **Function**: To represent systematic differences in user queries across interpretable dimensions of topic and style.
    - **Mechanism**: For topics, a global FastTopic model (10 topics) is trained on the collection of all user queries; each user's topic profile is the mean of their query topic distributions: $\mathbf{t}_{u_i} \in \mathbb{R}^{10}$. For style, LISA generates 768-dimensional style embeddings, which are compressed into 6 meta-styles (Theatrical, Academic, Fervent, Hostile, Inquisitive, Fragmented) via LDA. HypoGeniC is then used to generate natural language style hypotheses.
    - **Design Motivation**: Topic and style are orthogonal but complementary dimensions—topics capture "what a user asks," while style captures "how a user asks." A global topic space ensures direct comparability between users.

3.  **Topic + Style Feature-Driven Ranking Prediction**:
    - **Function**: To verify if user features can predict personalized model rankings, providing a path toward practical personalized benchmarks.
    - **Mechanism**: Topic profiles and LISA style embeddings for each user are concatenated into a 778-dimensional input $\mathbf{x}_{u_i} = [\mathbf{t}_{u_i}; \mathbf{s}_{u_i}]$. The regression target is a 20-dimensional model score vector. ELO predictions use an ensemble of 50 MLPs, and BT predictions use a single MLP with dropout.
    - **Design Motivation**: If topic and style features can effectively predict rankings, it implies that personalized benchmarks can be implemented by inferring user profiles from a small number of queries without requiring extensive preference collection.

### Loss & Training

The regression models use the Adam optimizer, with both features and targets standardized. The ELO model utilizes an ensemble of 50 MLPs with early stopping, while the BT model uses a single MLP with dropout.

## Key Experimental Results

### Main Results

**Correlation between Personalized and Global Rankings**

| Scoring System | Mean $\rho$ | Std Dev | Median | Users with Near-Zero/Neg Correlation |
| :--- | :--- | :--- | :--- | :--- |
| ELO | 0.432 | 0.257 | 0.442 | 70% ($\rho < 0.5$) |
| Bradley-Terry | 0.043 | 0.283 | 0.011 | 57% ($\rho < 0.1$) |

### Ablation Study

**Ranking Prediction MAE**

| Model | ELO MAE | BT MAE |
| :--- | :--- | :--- |
| Mean-Predictor (Global Average) | 0.688 | 0.510 |
| Topic + Style (Ours) | 0.450 ($\downarrow 35\%$) | 0.450 ($\downarrow 12\%$) |

### Key Findings

- The mean $\rho=0.043$ for BT personalized rankings is statistically indistinguishable from zero ($p=0.165$), meaning that for most users,personalized BT rankings are no better than random relative to the global ranking.
- The difference between ELO and BT is statistically significant (paired Wilcoxon $p < 10^{-13}$), suggesting they capture fundamentally different signals.
- User topic diversity varies greatly—ranging from users concentrated on 4 topics to those covering over 20 diverse topics.
- The 6 meta-styles (Theatrical, Academic, etc.) effectively distinguish user groups, forming 3 interpretable style clusters via k-means.

## Highlights & Insights

- The BT model reveals preference divergence more sensitively than ELO; this is a strength, as ELO’s incremental mechanism naturally smooths preference signals. This reminds the community that the choice of ranking algorithm affects the "visibility of personalization."
- The predictive power of topic and style features proves that personalized benchmarking is achievable in the near term—matching models by inferring user profiles from a few queries is possible without complex preference acquisition workflows.
- User preferences are not "minor perturbations" around a global ranking but represent "fundamentally different orderings," challenging the basic paradigm of current LLM evaluation.

## Limitations & Future Work

- Limited sample size of only 115 active users ($\ge 25$ votes).
- Only covers English queries; cross-lingual heterogeneity remains unknown.
- The analysis is correlational rather than causal—whether topic/style differences directly cause preference differences requires further experimentation.
- The approach could be extended to real-time personalized recommendations on platforms like Chatbot Arena.

## Related Work & Insights

- **vs Chatbot Arena**: Arena aggregates all user preferences for a global ranking; this study proves this is misleading for 57% of users.
- **vs HyPerAlign**: While HyPerAlign focuses on interpretable personalized alignment, this study provides a framework for quantifying preference divergence.
- **vs RLHF**: RLHF often treats human preference as a single aggregated signal; this study demonstrates the necessity of modeling individual differences.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic quantification of personalized vs. global ranking divergence; impactful findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual scoring systems + topic/style analysis + regression prediction, though limited by sample size.
- Writing Quality: ⭐⭐⭐⭐⭐ Fluent narrative, well-structured arguments, and sufficient quantitative evidence.
- Value: ⭐⭐⭐⭐⭐ Fundamentally challenges the LLM evaluation paradigm with a clear practical path.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[ACL 2026\] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)

</div>

<!-- RELATED:END -->
