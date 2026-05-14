---
title: >-
  [Paper Note] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues
description: >-
  [ACL 2026][LLM Evaluation][Social Relationship Reasoning] This paper introduces SCRIPTS, a benchmark comprising 1.1K English and Korean movie dialogues…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Social Relationship Reasoning"
  - "Cross-lingual Analysis"
  - "Movie Dialogues"
  - "Cultural Dependency"
date: 2026-05-08
content_hash: bb4dab13b4325840
---

# Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues

**Conference**: ACL 2026
**arXiv**: [2510.19028](https://arxiv.org/abs/2510.19028)
**Code**: [https://github.com/rladmstn1714/SCRIPTS](https://github.com/rladmstn1714/SCRIPTS)
**Area**: LLM Evaluation
**Keywords**: Social Relationship Reasoning, LLM Evaluation, Cross-lingual Analysis, Movie Dialogues, Cultural Dependency

## TL;DR

This paper introduces SCRIPTS, a benchmark comprising 1.1K English and Korean movie dialogues, evaluating the social relationship reasoning capabilities of 9 LLMs via a three-tier probabilistic labeling scheme (HIGHLY LIKELY / LESS LIKELY / UNLIKELY). Results show that models achieve only 75–80% accuracy on English and 58–69% on Korean, with Chain-of-Thought prompting and reasoning models providing little to no benefit for social reasoning.

## Background & Motivation

**Background**: As LLM agents become increasingly prevalent in multi-party interaction scenarios (e.g., ChatGPT group chats), LLMs must correctly identify social relationships among dialogue participants (e.g., lovers, friends, parent–child). Incorrect relationship inference may lead to inappropriate responses and privacy-related safety risks.

**Limitations of Prior Work**: Prior studies evaluating LLMs' social relationship reasoning suffer from oversimplified settings: (1) most rely on multiple-choice classification, limiting reasoning granularity; (2) relationship taxonomies are restricted; (3) most focus on simple two-party dialogues; (4) single correct labels fail to capture the inherent ambiguity of social relationships.

**Key Challenge**: Social relationship reasoning is inherently ambiguous and context-dependent — the same utterance "You never listen to me" may be a serious complaint between lovers or a playful remark between friends — yet existing evaluation frameworks cannot capture this ambiguity.

**Goal**: To construct a cross-lingual social relationship reasoning benchmark that supports uncertainty-aware evaluation, and to comprehensively assess the social reasoning capabilities and failure modes of current LLMs in both English and Korean.

**Key Insight**: Movie scripts are used as a data source approximating authentic dialogue. A three-tier probabilistic labeling scheme is introduced, three-party dialogue scenarios are included, and cultural specificity is analyzed from a cross-lingual perspective.

**Core Idea**: Social relationship reasoning cannot be evaluated with a single correct label; instead, it requires distinguishing "highly likely," "less likely," and "unlikely" relationship inferences to measure models' social intelligence at a fine-grained level.

## Method

### Overall Architecture

SCRIPTS extracts multi-turn dialogues from American and Korean movie scripts, which are then annotated by native-speaker annotators through a three-stage process: (1) marking UNLIKELY relationships (majority vote, ≥2/3 annotator agreement); (2) open-ended annotation of HIGHLY LIKELY relationships (union of labels, allowing multi-label); (3) deriving LESS LIKELY relationships (the remaining relationships from a predefined set, excluding the previous two categories). During evaluation, each model runs 5 times per dialogue and takes the majority answer, with HIGHLY LIKELY accuracy and UNLIKELY error rate computed as metrics.

### Key Designs

1. **Three-Tier Probabilistic Labeling Scheme**:

    - Function: Captures the inherent ambiguity in social relationship reasoning.
    - Mechanism: HIGHLY LIKELY (relationships strongly supported by the dialogue, annotated by taking the union of open-ended labels), LESS LIKELY (plausible but non-salient relationships), and UNLIKELY (relationships explicitly contradicted by the dialogue, ≥2/3 annotator agreement). On average, each dialogue has 3.67 HIGHLY LIKELY and 20.79 UNLIKELY relationships.
    - Design Motivation: Traditional single-label evaluation cannot distinguish between "absurd inferences" and "suboptimal but reasonable inferences." The three-tier scheme enables penalizing meaningless predictions (via the UNLIKELY metric) while rewarding the identification of the most salient relationships (via the HIGHLY LIKELY metric).

2. **Cross-lingual Bilingual Dataset Design**:

    - Function: Evaluates LLMs' social reasoning ability across different cultural contexts.
    - Mechanism: 580 English dialogues (28 American films) + 567 Korean dialogues (32 Korean films), with 41.8% three-party dialogues to increase reasoning complexity. Each dialogue averages approximately 10 turns, with 230 (English) and 617 (Korean) unique HIGHLY LIKELY relationship types.
    - Design Motivation: Social relationship reasoning is highly dependent on language and cultural background — Korean's grammaticalized honorific system (hierarchical relationships, verb morphology variations) encodes relational information absent in English, making cross-lingual evaluation essential for understanding the cultural dependency of social reasoning.

3. **Dialogue-Level Labeling Strategy**:

    - Function: Avoids the misleading nature of static character-level labels.
    - Mechanism: Rather than using global character labels from film metadata, each specific dialogue is independently annotated for relationships. Comparison reveals that 19% of film-level labels are irrelevant in specific dialogues, and each dialogue contains on average more than 3 HIGHLY LIKELY relationships.
    - Design Motivation: Social relationships are dynamic — speakers can switch roles across contexts and simultaneously hold multiple relationships. Global labels fail to reflect this context-dependent complexity.

### Evaluation Setup

Open-ended generation is used instead of fixed-option classification. Prompts include 27 example relationship types as reference, while allowing the model to generate freely. Each model runs 5 times with majority voting, using GPT-4o as the evaluator (human-verified accuracy: 92%).

## Key Experimental Results

### Main Results

**Multilingual Model Performance Comparison (Majority Vote over 5 Runs)**

| Model | EN HIGHLY LIKELY ↑ | EN UNLIKELY ↓ | KO HIGHLY LIKELY ↑ | KO UNLIKELY ↓ |
|------|---------------------|----------------|---------------------|----------------|
| GPT-4o | 0.767 | 0.116 | 0.642 | 0.215 |
| o3 (thinking) | 0.807 | 0.086 | 0.742 | 0.152 |
| Gemini-2.5-Flash | 0.759 | 0.154 | 0.582 | 0.318 |
| Qwen-3-14B | 0.623 | 0.164 | 0.455 | 0.444 |
| Llama-3.1-8B | 0.413 | 0.319 | 0.321 | — |
| A.X-4.0-Light (Korean-specific) | 0.589 | — | 0.467 | — |

### Ablation Study

**Thinking Mode Ablation (Single Run)**

| Model | Thinking | EN HL ↑ | EN UL ↓ | KO HL ↑ | KO UL ↓ |
|------|---------|----------|----------|----------|----------|
| Gemini-2.5-Flash | ✗ | 0.759 | 0.154 | 0.582 | 0.318 |
| Gemini-2.5-Flash | ✓ | 0.776 | 0.138 | 0.538 | 0.239 |
| Qwen-3-14B | ✗ | 0.623 | 0.164 | 0.455 | 0.444 |
| Qwen-3-14B | ✓ | 0.673 | 0.107 | 0.467 | 0.443 |

**Effect of Auxiliary Social Information (GPT-4o, Human-Annotated Labels)**

Providing auxiliary social information (age/gender, relationship dimensions, etc.) reduces the proportion of UNLIKELY predictions but does not consistently improve HIGHLY LIKELY accuracy. Auxiliary information autonomously inferred by models is insufficiently accurate (age/gender <60%, relationship dimensions <75%) to be helpful.

### Key Findings

- All models perform significantly worse on Korean than English (HL gap: 7–19 percentage points), with UNLIKELY proportions increasing by 7–17 percentage points.
- CoT prompting provides no consistent benefit for social reasoning and occasionally amplifies social biases (e.g., Llama's UNLIKELY rate on Korean increases by 3.1 percentage points).
- The Korean-specific model (A.X-4.0-Light) ranks first in both languages, yet its Korean HL score remains only 0.467.
- The effect of thinking mode is statistically non-significant (bootstrap test, $p > 0.05$).
- Auxiliary social information helps models avoid implausible inferences, but current LLMs' ability to autonomously infer such information is insufficient.

## Highlights & Insights

- The analysis of four failure modes is particularly insightful: (1) Confusing address terms with referential terms — models misinterpret "that's my Dad" as addressing the listener rather than referring to a third party; (2) Failure to aggregate multiple cues — multiple cues are detected but incorrectly weighted; (3) Failure to recognize atypical relationships — predicting "siblings" when parents and children converse as peers; (4) Failure to understand Korean language/cultural features (accounting for 46% of Korean errors) — misinterpreting honorifics and kinship terms.
- GPT-4o's failure mode distribution differs markedly between English and Korean: English errors are dominated by "failure to aggregate multiple cues" (36.7%), while Korean errors are dominated by "failure to understand cultural features" (46%).
- The evaluation paradigm combining open-ended generation with probabilistic labeling is generalizable to other NLP tasks requiring uncertainty-aware assessment.

## Limitations & Future Work

- Only English and Korean are covered, which may limit generalizability to other cultures.
- Movie scripts may not fully reflect authentic conversations.
- CoT traces may represent post-hoc rationalization rather than genuine reasoning processes.
- Future work should extend to more languages and more authentic dialogue sources (e.g., privacy-preserving real conversations, human–AI conversation logs).

## Related Work & Insights

- **vs. DDRel (Jia et al., 2021)**: DDRel uses multiple-choice classification and a limited set of relationship types; SCRIPTS adopts open-ended generation and three-tier probabilistic labeling with a richer relationship type space (230+/617+ unique types).
- **vs. PRIDE (Tigunova et al., 2021)**: PRIDE collects annotations from movie summaries rather than dialogues, failing to reflect relationships as actually manifested in conversation.
- **vs. Jurgens et al. (2023)**: Uses single-turn utterances rather than multi-turn dialogues and lacks non-English languages such as Korean.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-tier probabilistic labeling and cross-lingual social reasoning evaluation represent meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 9 models, bilingual settings, CoT ablations, auxiliary information ablations, and failure mode analysis comprehensively.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, case analyses are vivid, and failure mode categorization is excellent.
- Value: ⭐⭐⭐⭐ Reveals important blind spots in LLMs' social reasoning with meaningful implications for safe deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](../../AAAI2026/llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)
- [\[ACL 2026\] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity](roleconflictbench_a_benchmark_of_role_conflict_scenarios_for_evaluating_llms39_c.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)
- [\[ACL 2026\] Do LLMs Overthink Basic Math Reasoning? Benchmarking the Accuracy-Efficiency Tradeoff](do_llms_overthink_basic_math_reasoning_benchmarking_the_accuracy-efficiency_trad.md)
- [\[NeurIPS 2025\] On Evaluating LLM Alignment by Evaluating LLMs as Judges](../../NeurIPS2025/llm_evaluation/on_evaluating_llm_alignment_by_evaluating_llms_as_judges.md)

</div>

<!-- RELATED:END -->
