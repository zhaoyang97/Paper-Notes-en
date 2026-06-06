---
title: >-
  [Paper Note] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues
description: >-
  [ACL 2026][LLM Evaluation][Social relationship reasoning] Ours proposes the SCRIPTS benchmark, containing 1.1K English and Korean movie dialogues…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Social relationship reasoning"
  - "cross-linguistic analysis"
  - "movie dialogue"
  - "cultural dependency"
date: 2026-05-08
content_hash: 2668a8f37af84339
---

# Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues

**Conference**: ACL 2026  
**arXiv**: [2510.19028](https://arxiv.org/abs/2510.19028)  
**Code**: [https://github.com/rladmstn1714/SCRIPTS](https://github.com/rladmstn1714/SCRIPTS)  
**Area**: LLM Evaluation  
**Keywords**: Social relationship reasoning, LLM evaluation, cross-linguistic analysis, movie dialogue, cultural dependency

## TL;DR

Ours proposes the SCRIPTS benchmark, containing 1.1K English and Korean movie dialogues, to evaluate the social relationship reasoning capabilities of 9 LLMs using a three-tier probabilistic labeling scheme (HIGHLY LIKELY / LESS LIKELY / UNLIKELY). Findings indicate that models achieve only $75-80\%$ accuracy in English and $58-69\%$ in Korean, with CoT and thinking models providing negligible benefits for social reasoning.

## Background & Motivation

**Background**: As LLM agents become increasingly prevalent in multi-party interaction scenarios (e.g., ChatGPT group chats), LLMs must correctly identify social relationships between participants (e.g., lovers, friends, parent-child). Incorrect relationship inference can lead to safety risks such as inappropriate responses and privacy leaks.

**Limitations of Prior Work**: Previous studies evaluating the social relationship reasoning of LLMs suffer from simplified setups: (1) excessive use of multi-choice classification, which limits reasoning granularity; (2) limited relationship taxonomies; (3) a focus on simple dyadic dialogues; and (4) single correct labels that fail to capture the inherent uncertainty of social relationships.

**Key Challenge**: Social relationship reasoning is inherently ambiguous and context-dependent—the same phrase "You never listen to me" could be a serious complaint between a couple or a joke between friends—yet existing evaluation frameworks cannot capture this ambiguity.

**Goal**: To construct a cross-lingual social relationship reasoning benchmark that supports uncertainty-aware evaluation, comprehensively assessing the social reasoning capabilities and failure modes of current LLMs in English and Korean.

**Key Insight**: Using movie scripts as data sources approximating real conversations, introducing a three-tier probabilistic labeling scheme, including triadic dialogue scenarios, and analyzing cultural specificity from a cross-lingual perspective.

**Core Idea**: Social relationship reasoning should not be evaluated with a single correct label; instead, it requires distinguishing between "Highly Likely," "Less Likely," and "Unlikely" relationship inferences to measure the social intelligence of models with fine granularity.

## Method

### Overall Architecture

SCRIPTS extracts multi-turn dialogues from American and Korean movie scripts. Native speakers perform a three-stage annotation process: (1) labeling UNLIKELY relationships (majority vote, $\ge 2/3$ agreement); (2) open-ended labeling of HIGHLY LIKELY relationships (union of labels, allowing multi-labeling); and (3) deriving LESS LIKELY relationships (remainder of the predefined set after excluding the first two categories). During evaluation, each model runs 5 times per dialogue to obtain a majority response, calculating HIGHLY LIKELY accuracy and UNLIKELY error rates.

### Key Designs

1.  **Three-Tier Probabilistic Labeling Scheme**:
    - **Function**: Capture inherent ambiguity in social relationship reasoning.
    - **Mechanism**: HIGHLY LIKELY (relationships strongly supported by dialogue, union of open-ended annotations), LESS LIKELY (possible but not prominent), and UNLIKELY (relations explicitly contradicting the dialogue, $\ge 2/3$ agreement). Each dialogue averages $3.67$ HIGHLY LIKELY and $20.79$ UNLIKELY relationships.
    - **Design Motivation**: Traditional single-label evaluations cannot distinguish between "absurd inferences" and "suboptimal but reasonable inferences." The three-tier scheme allows penalties for nonsensical predictions (UNLIKELY metric) while rewarding the ability to identify the most prominent relationships (HIGHLY LIKELY metric).

2.  **Cross-Lingual Bilingual Dataset Design**:
    - **Function**: Evaluate the social reasoning capabilities of LLMs across different cultural contexts.
    - **Mechanism**: 580 English dialogues (28 US movies) + 567 Korean dialogues (32 Korean movies), including $41.8\%$ triadic dialogues to increase reasoning complexity. Dialogues average approximately 10 turns, featuring $230$ (English) and $617$ (Korean) unique HIGHLY LIKELY relationship types.
    - **Design Motivation**: Social relationship reasoning is highly dependent on linguistic and cultural backgrounds. The grammaticalized honorific system in Korean (power dynamics, verb morphology) encodes relationship information absent in English, making cross-lingual evaluation necessary to understand the cultural dependency of social reasoning.

3.  **Dialogue-Level Labeling Strategy**:
    - **Function**: Avoid misleading static character labels.
    - **Mechanism**: Instead of using global character labels from movie metadata, relationships are annotated independently for each specific dialogue. Comparisons reveal that $19\%$ of movie-level labels are irrelevant in specific dialogues, and each dialogue contains an average of more than 3 HIGHLY LIKELY relationships.
    - **Design Motivation**: Social relationships are dynamic; speakers can switch roles across contexts and hold multiple relationships simultaneously. Global labels fail to reflect this context-dependent complexity.

### Evaluation Setup

Open-ended generation is used instead of fixed-choice classification. Prompts include 27 example relationship types as references but allow the model to generate freely. Each model runs 5 times for majority voting, with GPT-4o serving as the evaluator (achieving $92\%$ human verification accuracy).

## Key Experimental Results

### Main Results

**Multilingual Model Performance Comparison (Majority Vote of 5 Runs)**

| Model | English HIGHLY LIKELY $\uparrow$ | English UNLIKELY $\downarrow$ | Korean HIGHLY LIKELY $\uparrow$ | Korean UNLIKELY $\downarrow$ |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 0.767 | 0.116 | 0.642 | 0.215 |
| o3 (thinking) | 0.807 | 0.086 | 0.742 | 0.152 |
| Gemini-2.5-Flash | 0.759 | 0.154 | 0.582 | 0.318 |
| Qwen-3-14B | 0.623 | 0.164 | 0.455 | 0.444 |
| Llama-3.1-8B | 0.413 | 0.319 | 0.321 | — |
| A.X-4.0-Light (KR-Specific) | 0.589 | — | 0.467 | — |

### Ablation Study

**Thinking Mode Ablation (Single Run)**

| Model | Thinking | English HL $\uparrow$ | English UL $\downarrow$ | Korean HL $\uparrow$ | Korean UL $\downarrow$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini-2.5-Flash | ✗ | 0.759 | 0.154 | 0.582 | 0.318 |
| Gemini-2.5-Flash | ✓ | 0.776 | 0.138 | 0.538 | 0.239 |
| Qwen-3-14B | ✗ | 0.623 | 0.164 | 0.455 | 0.444 |
| Qwen-3-14B | ✓ | 0.673 | 0.107 | 0.467 | 0.443 |

**Impact of Auxiliary Social Information (GPT-4o, Human-Annotated Labels)**

Providing auxiliary social information (age/gender, relationship dimensions, etc.) reduces the proportion of UNLIKELY predictions but does not consistently improve HIGHLY LIKELY accuracy. Auxiliary information inferred by the models themselves fails to help due to lack of accuracy (age/gender $<60\%$, relationship dimensions $<75\%$).

### Key Findings

- All models perform significantly worse in Korean than in English (HL gap of $7-19\%p$), with UNLIKELY proportions increasing by $7-17\%p$.
- CoT prompting provides no consistent benefit for social reasoning and occasionally amplifies social bias (e.g., Llama increases Korean UNLIKELY by $3.1\%p$).
- A Korean-specific model (A.X-4.0-Light) ranks first across both languages, yet its Korean HL remains only $0.467$.
- The effectiveness of thinking modes is statistically insignificant (bootstrap test, $p > 0.05$).
- Auxiliary social information helps models avoid unreasonable inferences, but current LLMs lack the capability to autonomously infer this information accurately.

## Highlights & Insights

- The analysis of four failure modes is highly insightful: (1) Confusing appellatives with referentials—models misinterpret references to third parties (e.g., "that's my Dad") as addressing the listener; (2) Inability to aggregate multiple clues; (3) Failure to recognize atypical relationships—e.g., predicting "siblings" when parents and children talk like peers; (4) Lack of understanding of Korean/cultural features (accounting for $46\%$ of Korean errors)—misinterpreting honorifics and kinship titles.
- The distribution of failure modes for GPT-4o differs sharply between English and Korean: English is dominated by "failure to aggregate multiple clues" ($36.7\%$), while Korean is dominated by "failure to understand cultural features" ($46\%$).
- The evaluation paradigm of open-ended generation combined with probabilistic labeling is extensible to other NLP tasks requiring uncertainty awareness.

## Limitations & Future Work

- Covers only English and Korean, which may not generalize to other cultures.
- Movie scripts may not fully reflect real-world conversations.
- CoT traces may represent post-hoc rationalizations rather than actual reasoning processes.
- Future work should extend to more languages and more authentic conversation sources (e.g., privacy-preserved real dialogues, human-AI dialogue logs).

## Related Work & Insights

- **vs DDRel (Jia et al., 2021)**: DDRel uses multi-choice classification and limited relationship types; SCRIPTS adopts open-ended generation and three-tier probabilistic labels with a much richer relationship set ($230+/617+$ unique types).
- **vs PRIDE (Tigunova et al., 2021)**: PRIDE collects annotations from movie summaries rather than dialogues, failing to reflect relationships as they actually manifest in conversation.
- **vs Jurgens et al. (2023)**: Uses single-turn utterances rather than multi-turn dialogues and lacks non-English languages like Korean.

## Rating

- Novelty: ⭐⭐⭐⭐ Three-tier probabilistic labeling and cross-lingual social reasoning evaluation represent meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage with 9 models, bilingual datasets, CoT ablation, auxiliary information ablation, and failure mode analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, vivid case studies, and excellent failure mode categorization.
- Value: ⭐⭐⭐⭐ Reveals critical blind spots in LLM social reasoning, providing guidance for safe deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](../../AAAI2026/llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)
- [\[ACL 2026\] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics](evaluating_legal_reasoning_traces_with_legal_issue_tree_rubrics.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)

</div>

<!-- RELATED:END -->
