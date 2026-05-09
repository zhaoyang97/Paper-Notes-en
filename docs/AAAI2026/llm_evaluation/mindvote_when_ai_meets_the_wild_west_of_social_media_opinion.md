---
title: >-
  [Paper Note] MindVote: When AI Meets the Wild West of Social Media Opinion
description: >-
  [AAAI 2026][LLM Evaluation][Social media public opinion] MindVote is a bilingual benchmark (3,918 naturally occurring Reddit/Weibo polls × 23 topics) for evaluating the opinion prediction capabilities of 15 LLMs. A key finding is that survey-specialized fine-tuned models underperform general-purpose models (the "specialization trap").
tags:
  - AAAI 2026
  - LLM Evaluation
  - Social media public opinion
  - bilingual polling
date: 2026-05-08
content_hash: 32323f13cc9dac38
---

# MindVote: When AI Meets the Wild West of Social Media Opinion

**Conference**: AAAI 2026
**arXiv**: [2505.14422](https://arxiv.org/abs/2505.14422)
**Code**: Available
**Area**: LLM Evaluation
**Keywords**: Opinion prediction, social media, LLM benchmark, cultural bias, context dependency

## TL;DR
This paper introduces MindVote — the first LLM opinion prediction benchmark grounded in real social media poll data, comprising 3,918 naturally occurring polls (across 23 topics) collected from Reddit and Weibo, enriched with platform- and topic-level context. Evaluation of 15 LLMs reveals: the best model (o3-medium) achieves a 1-Wasserstein score of only 0.892 versus an upper bound of 0.972; survey-specialized fine-tuned models underperform general-purpose models (the "survey specialization trap"); and models exhibit strong cultural alignment — Western models excel on Reddit while Chinese models excel on Weibo.

## Background & Motivation

### State of the Field

**Background**: LLMs are increasingly used as scalable substitutes for surveys — predicting public opinion distributions before costly surveys are deployed. However, existing evaluations are based on traditional structured questionnaires.

### Limitations of Prior Work

**Limitations of Prior Work**: Traditional surveys lack the contextual signals inherent to social media (platform norms, community discourse, cultural factors).

### Root Cause

**Key Challenge**: Survey data is decoupled from real social discussions — questionnaires strip away the social environment in which opinions are actually formed.

### Proposed Direction

**Proposed Direction**: Existing benchmarks are topically narrow, culturally homogeneous, and lack contextual metadata.

**Key Challenge**: LLMs are deployed in social media settings in practice, yet are evaluated on structured surveys that differ fundamentally in style from social media content.

**Goal**: Construct an opinion prediction benchmark grounded in real social media discussions.

**Key Insight**: Collect authentic poll data (not artificially constructed) from Reddit and Weibo, accompanied by platform- and topic-level context.

**Core Idea**: Real social polls + dual-platform cross-cultural design + rich contextual metadata = ecologically valid opinion evaluation.

## Method

### Overall Architecture
Collect 3,918 naturally occurring polls from Reddit/Weibo → filter across 23 topics via whitelist → annotate platform context (user demographics, technical orientation) and topic context (current news, industry data) → zero-shot evaluation of 15 LLMs.

### Key Designs

1. **Dual-Platform Cross-Cultural Design**:

    - Function: Covers distinct community norms and cultures across the Western and Chinese spheres.
    - Reddit (English/Western users) and Weibo (Chinese/Chinese users), with translation augmentation (BLEU > 35).
    - Design Motivation: Enables direct comparison of model performance on the same topics across different cultural contexts.

2. **Structured Context Annotation**:

    - Function: Provides each poll with context that influences opinion formation.
    - Includes platform context + topic context.
    - Ablation finding: Removing all context degrades performance by 5.91% — context is a critical signal.

3. **Four-Metric Evaluation**: 1-Wasserstein / KL divergence / Spearman correlation / Accuracy.

## Key Experimental Results

### Main Results

| Model | 1-Wass↑ | Spearman↑ | Acc |
|-------|---------|-----------|-----|
| o3-medium | **0.892** | **0.756** | 58.1% |
| DeepSeek-R1 | 0.876 | 0.739 | 55.8% |
| SubPop-Llama (survey fine-tuned) | 0.774 | - | - |
| Upper Bound | 0.972 | - | - |

### Ablation Study: Effect of Context

| Configuration | 1-Wass Change |
|---------------|---------------|
| Remove all context | -5.91% |
| Remove platform context only | -5.12% |
| Remove topic context only | -4.52% |

### Key Findings
- **Survey Specialization Trap**: Survey-specialized fine-tuned models underperform general-purpose models.
- **Strong Cultural Alignment**: Western models excel on Reddit; Chinese models excel on Weibo.
- **Significant Gap from Upper Bound** (0.892 vs. 0.972).

## Highlights & Insights
- The core insight that **"surveys ≠ social media"** has broad implications for evaluation methodology.
- The **cultural alignment phenomenon** reveals that LLMs reason in a culturally centered manner consistent with their training data.
- The context ablation design is clean and convincing.

## Limitations & Future Work
- Limited to two platforms: Reddit and Weibo.
- Machine translation may fail to fully capture cultural nuances.
- Excluding categorical preference polls reduces topical diversity.

## Related Work & Insights
- **vs. OpinionQA / SubPop**: Both rely on general social surveys. MindVote uses real social media data and thus has greater ecological validity.
- **vs. GlobalOpinionQA**: Multi-country in scope but still survey-format.
- Provides guidance for the design of multicultural AI systems.

## Rating
- Novelty: ⭐⭐⭐⭐ First benchmark grounded in real social media opinion polling; the "survey specialization trap" is a significant finding.
- Experimental Thoroughness: ⭐⭐⭐⭐ 15 models, 3,918 polls, dual platforms, context ablation.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is precise.
- Value: ⭐⭐⭐⭐ Methodological contribution to opinion prediction evaluation.

**Area**: NLP Understanding / Opinion Analysis
**Keywords**: Social media public opinion, bilingual polling, LLM evaluation

## TL;DR
MindVote is a bilingual benchmark (3,918 naturally occurring Reddit/Weibo polls × 23 topics) for evaluating the opinion prediction capabilities of 15 LLMs. A key finding is that survey-specialized fine-tuned models underperform general-purpose models (the "specialization trap").

## Method

### Key Designs
1. **3,918 naturally occurring polls** (bilingual)
2. **Evaluation of 15 LLMs**
3. **Multi-metric evaluation**: Wasserstein / KL divergence / rank correlation

## Highlights & Insights
- **Specialization Trap**: Survey fine-tuned models < general-purpose models — a counterintuitive and important finding.

## Rating
- Novelty: ⭐⭐⭐⭐ First benchmark grounded in real social media polls.
- Experimental Thoroughness: ⭐⭐⭐⭐ 15 models, bilingual.
- Value: ⭐⭐⭐⭐ Practical value for social media analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AutoReproduce: Automatic AI Experiment Reproduction with Paper Lineage](../../ACL2026/llm_evaluation/autoreproduce_automatic_ai_experiment_reproduction_with_paper_lineage.md)
- [\[ACL 2026\] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues](../../ACL2026/llm_evaluation/are_they_lovers_or_friends_evaluating_llms39_social_reasoning_in_english_and_kor.md)
- [\[ICLR 2026\] AstaBench: Rigorous Benchmarking of AI Agents with a Scientific Research Suite](../../ICLR2026/llm_evaluation/astabench_benchmarking_ai_agents.md)
- [\[ICLR 2026\] When Priors Backfire: On the Vulnerability of Unlearnable Examples to Pretraining](../../ICLR2026/llm_evaluation/when_priors_backfire_on_the_vulnerability_of_unlearnable_examples_to_pretraining.md)
- [\[ICCV 2025\] InterSyn: Interleaved Learning for Dynamic Motion Synthesis in the Wild](../../ICCV2025/llm_evaluation/intersyn_interleaved_learning_for_dynamic_motion_synthesis_in_the_wild.md)

</div>

<!-- RELATED:END -->
