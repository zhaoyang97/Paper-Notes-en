---
title: >-
  [Paper Note] Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues
description: >-
  [ACL 2026][LLM Evaluation][Paper Note] This paper proposes the SCRIPTS benchmark, containing 1.1K English and Korean movie dialogues, to evaluate the social relation reasoning capabilities of 9 LLMs through three-tier probabilistic labels (HIGHLY LIKELY / LESS LIKELY / UNLIKELY). The study finds that models achieve only 75-80% accuracy in English and 58-69%
tags:
  - ACL 2026
  - LLM Evaluation
date: 2026-05-08
content_hash: 62b82878d0a92f44
---
# Are They Lovers or Friends? Evaluating LLMs' Social Reasoning in English and Korean Dialogues

**Conference**: ACL 2026  
**arXiv**: [2510.19028](https://arxiv.org/abs/2510.19028)  
**Code**: [https://github.com/rladmstn1714/SCRIPTS](https://github.com/rladmstn1714/SCRIPTS)  
**Area**: LLM Evaluation  
**Keywords**: Social relation reasoning, LLM evaluation, cross-lingual analysis, movie dialogues, cultural dependency

## TL;DR

This paper proposes the SCRIPTS benchmark, containing 1.1K English and Korean movie dialogues, to evaluate the social relation reasoning capabilities of 9 LLMs through three-tier probabilistic labels (HIGHLY LIKELY / LESS LIKELY / UNLIKELY). The study finds that models achieve only 75-80% accuracy in English and 58-69% in Korean, and CoT or reasoning-based models provide almost no benefit for social reasoning.

## Background & Motivation

**Background**: As LLM agents become increasingly prevalent in multi-party interaction scenarios (such as ChatGPT group chats), they must correctly identify social relations (e.g., lovers, friends, parent-child) between dialogue participants. Incorrect relational inference may lead to safety risks such as inappropriate responses or privacy leaks.

**Limitations of Prior Work**: Previous studies evaluating LLM social reasoning suffer from simplified settings: (1) excessive use of multi-choice classification, which limits reasoning granularity; (2) limited taxonomies for relation types; (3) a focus on simple dyadic dialogues; and (4) the use of a single "correct" label that fails to capture the inherent uncertainty of social relations.

**Key Challenge**: Social relation reasoning is inherently ambiguous and context-dependent—the same phrase "You never listen to me" could be a serious complaint between a couple or a joke between friends—but existing evaluation frameworks cannot capture this ambiguity.

**Goal**: To build a cross-lingual social relation reasoning benchmark that supports uncertainty-aware evaluation, comprehensively assessing the social reasoning capabilities and failure modes of current LLMs in English and Korean.

**Key Insight**: Movie scripts serve as a data source close to real-life dialogue. By introducing a three-tier probabilistic labeling scheme, including triadic dialogue scenes, and analyzing cultural specificity from a cross-lingual perspective, the complexity of social intelligence can be better measured.

**Core Idea**: Social relation reasoning should not be evaluated with a single correct label; instead, it requires distinguishing between "Highly Likely," "Less Likely," and "Unlikely" relational inferences to measure the model's social intelligence with fine granularity.

## Method

### Overall Architecture

SCRIPTS examines whether LLMs can infer social relations between speakers from multi-turn dialogues and explicitly distinguishes which relations are "highly likely," "less likely," or "altogether unlikely." The benchmark extracts 1.1K dialogues from American and Korean movie scripts, annotated by native speakers in three stages: first, UNLIKELY relations are identified via majority vote ($\ge 2/3$ agreement); second, HIGHLY LIKELY relations are identified via open-ended annotation and taking the union (allowing multi-labels); finally, LESS LIKELY relations are derived by excluding the first two categories from a pre-defined set. During evaluation, each model generates five open-ended responses per dialogue to obtain a majority-vote answer. GPT-4o serves as the evaluator (with a 92% human verification agreement rate), reporting HIGHLY LIKELY accuracy and UNLIKELY error rates separately.

### Key Designs

**1. Three-tier Probabilistic Labeling Scheme: Characterizing Inherent Ambiguity**

Traditional single-label evaluations cannot distinguish between an "absurd inference" and a "suboptimal but reasonable inference." However, social relations are naturally fuzzy. SCRIPTS categorizes relations into three tiers: HIGHLY LIKELY for relations strongly supported by the dialogue (union of open-ended annotations), LESS LIKELY for possible but not prominent relations, and UNLIKELY for relations clearly contradicted by the dialogue ($\ge 2/3$ agreement among annotators). On average, each dialogue has 3.67 HIGHLY LIKELY and 20.79 UNLIKELY relations. This allows the UNLIKELY metric to penalize nonsensical predictions while the HIGHLY LIKELY metric rewards the ability to identify the most prominent relations.

**2. Cross-lingual Bilingual Dataset Design: Exposing Cultural Dependency**

The dataset contains 580 English dialogues (from 28 US movies) and 567 Korean dialogues (from 32 Korean movies), where 41.8% are triadic dialogues to increase reasoning complexity. Dialogues average approximately 10 turns. There are 230 and 617 unique HIGHLY LIKELY relation types for English and Korean, respectively. The bilingual comparison is crucial because social reasoning relies heavily on linguistic and cultural contexts: the grammaticalized honorific system in Korean (encoding hierarchy and verb morphology) provides relational information absent in English.

**3. Dialogue-level Annotation Strategy: Independent Annotation per Dialogue Segment**

Social relations are dynamic—the same speaker can switch roles or hold multiple relations depending on the context. SCRIPTS avoids using global character labels from movie metadata, independently annotating each specific dialogue segment instead. Comparison reveals that 19% of movie-level labels are irrelevant in specific dialogues, and dialogues average over 3 HIGHLY LIKELY relations, indicating that static global labels would seriously mislead evaluation. Although evaluation uses open-ended generation, 27 example relation types are provided for reference in the prompt.

## Key Experimental Results

### Main Results

**Multi-lingual Model Performance Comparison (Majority Vote of 5 Runs)**

| Model | EN HIGHLY LIKELY ↑ | EN UNLIKELY ↓ | KO HIGHLY LIKELY ↑ | KO UNLIKELY ↓ |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 0.767 | 0.116 | 0.642 | 0.215 |
| o3 (thinking) | 0.807 | 0.086 | 0.742 | 0.152 |
| Gemini-2.5-Flash | 0.759 | 0.154 | 0.582 | 0.318 |
| Qwen-3-14B | 0.623 | 0.164 | 0.455 | 0.444 |
| Llama-3.1-8B | 0.413 | 0.319 | 0.321 | — |
| A.X-4.0-Light (KO specialized) | 0.589 | — | 0.467 | — |

### Ablation Study

**Thinking Mode Ablation (Single Run)**

| Model | Thinking | EN HL ↑ | EN UL ↓ | KO HL ↑ | KO UL ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini-2.5-Flash | ✗ | 0.759 | 0.154 | 0.582 | 0.318 |
| Gemini-2.5-Flash | ✓ | 0.776 | 0.138 | 0.538 | 0.239 |
| Qwen-3-14B | ✗ | 0.623 | 0.164 | 0.455 | 0.444 |
| Qwen-3-14B | ✓ | 0.673 | 0.107 | 0.467 | 0.443 |

**Impact of Auxiliary Social Information (GPT-4o, Human-annotated Labels)**

Providing auxiliary social information (age/gender, relationship dimensions, etc.) reduces the proportion of UNLIKELY predictions but does not consistently improve HIGHLY LIKELY accuracy. Auxiliary information inferred by the models themselves fails to help because it is insufficiently accurate (age/gender <60%, relationship dimensions <75%).

### Key Findings

- All models performed significantly worse in Korean than in English (HL gap of 7-19%p), with UNLIKELY proportions increasing by 7-17%p.
- CoT prompting provides no consistent help for social reasoning and occasionally amplifies social bias (e.g., Llama's UNLIKELY rate in Korean increased by 3.1%p).
- The Korean-specialized model (A.X-4.0-Light) performed well across languages relative to its size, yet its Korean HL accuracy remained low at 0.467.
- The effect of thinking modes was not statistically significant ($p > 0.05$ via bootstrap test).
- Auxiliary social information helps models avoid unreasonable inferences, but current LLMs lack the capability to infer this information accurately on their own.

## Highlights & Insights

- The analysis of four failure modes is insightful: (1) Confusion between vocative and referential terms—models misinterpret a reference to a third party (e.g., "that's my Dad") as a direct address to the listener; (2) Inability to aggregate multi-clues—detecting multiple clues but assigning weights incorrectly; (3) Failure to recognize atypical relations—predicting "siblings" when parents and children talk like peers; (4) Failure to understand Korean/cultural features (46% of Korean errors)—misinterpreting honorifics and kinship terms.
- The distribution of failure modes for GPT-4o differs drastically between English and Korean: English failures are dominated by "Inability to aggregate multi-clues" (36.7%), while Korean failures are dominated by "Cultural feature understanding failure" (46%).
- The evaluation paradigm of open-ended generation combined with probabilistic labels is generalizable to other NLP tasks requiring uncertainty awareness.

## Limitations & Future Work

- The study covers only English and Korean, which may not generalize to other cultures.
- Movie scripts may not perfectly reflect real-world conversations.
- CoT traces may represent post-hoc rationalization rather than the actual reasoning process.
- Future work should expand to more languages and more authentic dialogue sources (e.g., privacy-protected real conversations, human-AI dialogue logs).

## Related Work & Insights

- **vs DDRel (Jia et al., 2021)**: DDRel uses multiple-choice classification and limited relation types; SCRIPTS employs open-ended generation and three-tier probabilistic labels with a richer set of relations (230+/617+ unique types).
- **vs PRIDE (Tigunova et al., 2021)**: PRIDE collects annotations from movie summaries rather than dialogues, failing to reflect the relations actually presented in discourse.
- **vs Jurgens et al. (2023)**: Uses single-turn utterances rather than multi-turn dialogues and lacks non-English languages like Korean.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-tier probabilistic labeling and cross-lingual social reasoning evaluation are meaningful innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ 9 models, bilingual coverage, CoT ablation, auxiliary information ablation, and failure mode analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, vivid case studies, and excellent failure mode categorization.
- Value: ⭐⭐⭐⭐ Highlights major blind spots in LLM social reasoning, providing guidance for safe deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[AAAI 2026\] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning](../../AAAI2026/llm_evaluation/where_norms_and_references_collide_evaluating_llms_on_normative_reasoning.md)
- [\[ACL 2026\] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics](evaluating_legal_reasoning_traces_with_legal_issue_tree_rubrics.md)
- [\[ACL 2026\] Evaluating Reasoning Models for Queries with Presuppositions](evaluating_reasoning_models_for_queries_with_presuppositions.md)
- [\[ACL 2026\] Can LLMs Act as Historians? Evaluating Historical Research Capabilities of LLMs via the Chinese Imperial Examination](can_llms_act_as_historians_evaluating_historical_research_capabilities_of_llms_v.md)
- [\[ACL 2026\] ReTraceQA: Evaluating Reasoning Traces of Small Language Models in Commonsense Question Answering](retraceqa_evaluating_reasoning_traces_of_small_language_models_in_commonsense_qu.md)

</div>

<!-- RELATED:END -->
