---
title: >-
  [Paper Note] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework
description: >-
  [ICLR 2026][LLM Evaluation] This paper proposes the HUMAINE framework, which conducts multi-dimensional evaluations of 28 models with 23,404 demographically stratified participants, revealing that age is the greatest axis of divergence in human preference and that a single leaderboard obscures critical differences.
tags:
  - ICLR 2026
  - LLM Evaluation
  - human preference
  - demographic bias
  - Bradley-Terry model
  - multi-dimensional evaluation
date: 2026-05-08
content_hash: 0c704ea4f45dbcc9
---

# Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework

**Conference**: ICLR 2026
**arXiv**: [2603.04409](https://arxiv.org/abs/2603.04409)
**Code**: [https://huggingface.co/spaces/ProlificAI/humaine-leaderboard](https://huggingface.co/spaces/ProlificAI/humaine-leaderboard) (leaderboard + dataset publicly available)
**Area**: LLM Evaluation
**Keywords**: LLM evaluation, human preference, demographic bias, Bradley-Terry model, multi-dimensional evaluation

## TL;DR

This paper proposes the HUMAINE framework, which conducts multi-dimensional evaluations of 28 models with 23,404 demographically stratified participants, revealing that age is the greatest axis of divergence in human preference and that a single leaderboard obscures critical differences.

## Background & Motivation

LLM evaluation faces an "evaluation gap" — both technical benchmarks and human preference evaluations suffer from fundamental flaws.

**Problems with technical benchmarks**: MMLU, HELM, and BIG-Bench only measure "what the model knows," failing to capture "how the model performs in human collaboration." Over-optimizing benchmarks leads to Goodhart's Law — the metric itself becomes the target rather than genuine improvement in user experience.

**Problems with existing human preference evaluations** (exemplified by Chatbot Arena):

- **Non-representative sampling**: Self-selected anonymous user populations exhibit severe bias, skewing toward tech enthusiasts.
- **Insufficient evaluation depth**: Judgments based on minimal interactions fail to capture deep interaction quality.
- **Single-metric reductionism**: Binary preference voting compresses multi-dimensional interaction quality into a single score.

The authors' core argument: when a model's ranking can shift by ±2.8 positions depending on user age, an "average leaderboard" is far from sufficient.

## Method

### Overall Architecture

HUMAINE is grounded in psychometric principles and comprises four layers:

1. **Demographically stratified recruitment**: Participants are recruited via the Prolific platform and cross-stratified into 22 demographic groups by geography (UK/US), age (18–34 / 35–54 / 55+), ethnicity, and political orientation.
2. **Multi-turn naturalistic dialogue data collection**: Anonymous side-by-side comparison of two models; participants choose their own topics and engage in at least 3 dialogue turns.
3. **Five-dimensional evaluation**: After each conversation, participants provide preference judgments along five dimensions for both models.
4. **Hierarchical Bayesian BTD model**: Pairwise comparisons are converted into continuous skill scores, with post-stratification calibration applied using demographic weights.

### Key Designs

**Data collection design**:
- 28 SOTA models accessed via openrouter.ai.
- Participants send identical messages to both models, ensuring fully consistent conversational context.
- TrueSkill adaptive sampling dynamically selects model pairings that maximize information gain.
- gpt-4o-mini monitors conversation quality in real time; participants receive up to three warnings before removal (affecting <1.6%).

**Five evaluation dimensions** (identified via preliminary factor analysis):
- **Core Task Performance & Reasoning**: Effectiveness in task completion and reasoning.
- **Communication Style & Presentation**: Appropriateness of language, tone, and level of detail.
- **Interaction Fluidity & Adaptability**: Dialogue flow management and responsiveness to user input.
- **Trust, Ethics & Safety**: Reliability, transparency, and ethical soundness of outputs.
- **Overall Winner**: Holistic preference integrating all dimensions.

**Hierarchical Bayesian BTD model**:
- Core: learns a global skill parameter $\theta$ per model–dimension combination, plus demographic-specific adjustments $u$.
- BTD formula handling ties: $p_A = e^\eta/Z$, $p_T = \nu_k/Z$, $p_B = e^{-\eta}/Z$.
- Demographic effects decomposed as additive adjustments along three axes: age, ethnicity, and political orientation.
- Post-stratification: census weights are used to generalize participant effects to the true population.

### Loss & Training

The model employs hierarchical Bayesian inference (rather than conventional training). Key design choices:
- Zero-sum constraint on skill parameters: $\sum_i \theta_{i,k} = 0$.
- Partial pooling on demographic adjustment parameters to prevent overfitting.
- Priors: $u_{\text{raw}} \sim N(0,1)$, $\tau \sim \text{Exponential}(\lambda=12)$.
- A scaling factor of $1/\sqrt{3}$ on demographic effects ensures that combined effects across three axes remain commensurable with single-axis magnitudes.

## Key Experimental Results

### Main Results

**Overall leaderboard** (Overall Winner, Winshare score, max 27):

| Rank | Model | Key Statistics |
|------|-------|---------------|
| 1 | google/gemini-2.5-pro | P(best)=95.6% |
| 2 | deepseek/deepseek-chat-v3-0324 | Notably behind 1st |
| 3–5 | magistral-medium / grok-4 / grok-3 | Overlapping CIs, statistically indistinguishable |

**Demographic heterogeneity in preference** (core findings):

| Demographic Axis | Mean Rank Shift | Significance |
|-----------------|----------------|--------------|
| **Age** | **±2.8 positions** | **Largest divergence factor** |
| Ethnicity | ±1.3 positions | Moderate |
| Political orientation | ±1.5 positions | Moderate |

Illustrative case: magistral-medium ranks 1st–2nd among the 18–34 age group but drops to 5th–10th among the 55+ group; gemini-2.5-pro's ranking improves with increasing user age.

### Ablation Study (Dimension Discriminability Analysis)

| Evaluation Dimension | Tie Rate | Discriminability |
|----------------------|----------|-----------------|
| Overall Winner | 10% | Strongest |
| Communication Style | 17–20% | Strong |
| Core Task Performance | 32–39% | Moderate |
| Interaction Fluidity | ~30% | Moderate |
| **Trust, Ethics & Safety** | **65%** | **Weakest** |

Tie rates increase with age: 9.7% (18–34) → 12.5% (55+), a 29% increase.

### Key Findings

1. **Age is the greatest axis of preference divergence**: Its effect far exceeds that of ethnicity and political orientation; model rankings can differ by ±2.8 positions between younger and older users.
2. **"Best" is a contextual illusion**: gemini-2.5-pro's success stems from **consistency across dimensions** rather than extremes on any single one — it ranks only 13th on HELM yet 1st in human preference.
3. **Trust/safety dimension is nearly indiscriminate**: A 65% tie rate suggests that models converge on safety behavior in open-ended dialogue, or that this dimension is inherently difficult to assess in brief interactions.
4. **Overall judgment is most decisive**: Even when individual dimensions are ambiguous, users form clear holistic preferences.
5. **Conversation content distribution**: 71.5% are information queries spanning 41 domains; median complexity is 4/5; 92.6% of interactions achieve their stated goal.

## Highlights & Insights

1. **Methodological contribution is the standout**: This is not "yet another leaderboard" but a complete methodology — from sampling strategy and evaluation framework to statistical modeling — that can be reused by future evaluation research.
2. **The age-bias finding carries policy implications**: If LLMs are primarily optimized based on feedback from young, tech-savvy users, the needs of older user populations may be systematically neglected.
3. **Living leaderboard design**: Regular updates as a "living benchmark" address the rapid obsolescence of static leaderboards.
4. **Appropriate use of LLM judges**: LLM judgments are strictly confined to post-hoc interpretive analysis and not used to generate competitive rankings, thereby avoiding the biases inherent in LLM-as-judge approaches.

## Limitations & Future Work

1. **Geographic limitation**: Coverage is restricted to the US and UK; cultural background may have an even larger influence on preference.
2. **Incomplete demographic dimensions**: Gender, education level, and socioeconomic status are absent.
3. **Short-interaction constraints**: Multi-turn but relatively brief conversations cannot assess long-term persona consistency or performance degradation.
4. **Uncontrolled task complexity**: Open topic selection prevents precise control of task difficulty.
5. **Text-only interactions**: The capabilities of multimodal models are not evaluated.
6. **Trust/safety dimension requires dedicated design**: Reliable assessment in open-ended dialogue is difficult; targeted scenario design is needed.

## Related Work & Insights

- **Chatbot Arena** (Zheng et al., 2023): The evaluation paradigm this paper directly responds to, identifying its sampling and methodological shortcomings.
- **Santurkar et al. (2023)**: Provides empirical evidence that demographics influence LLM preferences.
- **Bradley-Terry model** (1952): The classical statistical model for pairwise comparisons, extended here to a hierarchical Bayesian formulation.
- Implication: Future RLHF training should account for demographic diversity to avoid overfitting to the preferences of any single population group.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4.0 |
| Theoretical Depth | 4.0 |
| Experimental Thoroughness | 4.5 |
| Writing Quality | 4.5 |
| Value | 4.5 |
| Overall | 4.3 |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis](talk_evaluate_diagnose_user-aware_agent_evaluation_with_automated_error_analysis.md)
- [\[ICLR 2026\] Subliminal Signals in Preference Labels](subliminal_signals_in_preference_labels.md)
- [\[ICLR 2026\] Human-LLM Collaborative Feature Engineering for Tabular Learning](human-llm_collaborative_feature_engineering_for_tabular_data.md)
- [\[ICLR 2026\] Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)
- [\[ICLR 2026\] Benchmarking Overton Pluralism in LLMs](benchmarking_overton_pluralism_in_llms.md)

<!-- RELATED:END -->
