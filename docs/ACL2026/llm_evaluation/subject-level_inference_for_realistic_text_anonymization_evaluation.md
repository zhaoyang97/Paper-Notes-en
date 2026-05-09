---
title: >-
  [Paper Note] Subject-level Inference for Realistic Text Anonymization Evaluation
description: >-
  [ACL 2026][LLM Evaluation][Text Anonymization] SPIA introduces the first subject-level PII inference evaluation benchmark (675 documents, 1,712 subjects, 7,040 PII instances), revealing that even when 90%+ of PII spans are redacted, the subject-level inference protection rate can be as low as 33%, and that anonymization focused on a single target subject leads to greater exposure of non-target subjects.
tags:
  - ACL 2026
  - LLM Evaluation
  - Text Anonymization
  - Privacy Evaluation
  - Subject-level Inference
  - PII Reasoning
  - Multi-subject Protection
date: 2026-05-08
content_hash: 6e4e1f995aa38b02
---

# Subject-level Inference for Realistic Text Anonymization Evaluation

**Conference**: ACL 2026
**arXiv**: [2604.21211](https://arxiv.org/abs/2604.21211)
**Code**: [https://github.com/maisonOP/spia.git](https://github.com/maisonOP/spia.git)
**Area**: LLM Evaluation
**Keywords**: Text Anonymization, Privacy Evaluation, Subject-level Inference, PII Reasoning, Multi-subject Protection

## TL;DR

SPIA introduces the first subject-level PII inference evaluation benchmark (675 documents, 1,712 subjects, 7,040 PII instances), revealing that even when 90%+ of PII spans are redacted, the subject-level inference protection rate can be as low as 33%, and that anonymization focused on a single target subject leads to greater exposure of non-target subjects.

## Background & Motivation

**Background**: Text anonymization protects individuals from identification by modifying text, and is a core requirement under privacy regulations such as GDPR. Existing evaluation methods are dominated by span-based metrics such as Token Recall and Entity Recall, which measure whether explicit PII mentions are redacted. Established benchmarks include i2b2/UTHealth (medical), TAB (legal), and WikiPII (Wikipedia).

**Limitations of Prior Work**: Two critical flaws exist. First, span-based metrics fail to capture inference risk — Staab et al. (2025) demonstrate that even after NER-based anonymization, 66.3% of personal attributes remain inferable from context. Second, existing methods assume documents have a single data subject, whereas real-world texts (legal judgments, medical records, online posts) typically involve multiple individuals. Current techniques primarily protect one main subject while providing insufficient protection for other mentioned individuals.

**Key Challenge**: Redacting all explicit PII mentions (high span recall) does not equate to protecting all individuals (high inference protection). LLMs can infer redacted personal information from contextual cues, and the protection of non-target subjects in multi-subject documents is systematically neglected. This represents a fundamental error in the unit of evaluation — the field should shift from text spans to individual persons.

**Goal**: To shift the unit of anonymization evaluation from text spans to individuals, construct an inference-based evaluation benchmark covering multiple subjects and domains, and design new subject-level protection metrics.

**Key Insight**: A "subject" is defined as any identifiable individual in a document, and each subject's PII is independently assessed for recoverability by an adversarial LLM from the anonymized text.

**Core Idea**: The unit of evaluation = individual persons (not text spans); the protection metric = proportion of PII still inferable after anonymization (not the redaction rate).

## Method

### Overall Architecture

SPIA comprises two components: benchmark construction and an evaluation framework. For benchmark construction, 675 documents are selected from TAB (legal judgments) and PANORAMA (online text), with 1,712 subjects and 7,040 PII instances across 15 categories annotated via a combination of human and LLM annotation. The evaluation framework follows a three-stage pipeline: (1) an anonymization method processes the original text; (2) an adversarial LLM (Claude-Sonnet-4.5) performs two-stage inference (subject identification + PII inference) on the anonymized text; (3) subject matching, PII scoring, and computation of CPR/IPR metrics.

### Key Designs

1. **Two-stage Subject-level Inference Framework**:

    - Function: Identifies all recognizable subjects from anonymized text and infers PII for each subject separately.
    - Mechanism: Stage A identifies all subjects in the document and provides distinguishing descriptions (name, role, etc.). Stage B infers 15 categories of PII for each identified subject in two separate batches — CODE types (ID numbers, phone numbers, email addresses, driver's licenses, passports: 5 categories) and NON-CODE types (name, gender, age, location, nationality, education, relationships, occupation, affiliation, title: 10 categories). The separation avoids the model processing 15 categories simultaneously, reduces prompt length, and allows type-specific handling.
    - Design Motivation: Extends the single-author profiling approach of Staab et al. (2024) to multi-subject settings. Validation across 11 LLMs confirms that Claude-Sonnet-4.5 achieves the best performance on subject matching (96%) and inference accuracy (91%).

2. **CPR and IPR Protection Metrics**:

    - Function: Quantifies the protection effectiveness of anonymization from both collective and individual perspectives.
    - Mechanism: CPR (Collective Protection Rate) $= 1 - \sum A_i / \sum O_i$, weighted by PII count such that subjects with more PII contribute more. IPR (Individual Protection Rate) $= \frac{1}{N}\sum(1 - A_i/O_i)$, averaging equally across all subjects. Here $O_i$ denotes the number of PII items for subject $i$ in the original text, and $A_i$ denotes the number of PII items still inferable by the adversary from the anonymized text. Both metrics equal 1 for complete protection and 0 for complete exposure.
    - Design Motivation: CPR focuses on overall PII leakage volume, while IPR addresses whether each individual is equally protected — a high CPR may coexist with certain subjects being fully exposed.

3. **PII Taxonomy (CODE + NON-CODE)**:

    - Function: Covers 15 PII categories, classified by structural characteristics to support different detection and evaluation strategies.
    - Mechanism: CODE types have fixed-format patterns (ID numbers, driver's licenses, phone numbers, passports, email addresses); NON-CODE types are free-form text (name, gender, age, location, nationality, education, relationships, occupation, affiliation, title). CODE types are included in inference evaluation because pattern-based NER may miss unseen formats.
    - Design Motivation: Classifying by structural characteristics is more stable than the conventional direct-identifier/quasi-identifier taxonomy, which is highly context-dependent.

### Loss & Training
This paper presents an evaluation benchmark and framework and does not involve model training. The adversarial LLM used is Claude-Sonnet-4.5. PII scoring follows a three-level scheme: 1.0 for exact match, 0.5 for partial match, and 0.0 for no match.

## Key Experimental Results

### Main Results (TAB Legal Dataset, Selected Anonymization Methods × Best Backbone)

| Method | Token Recall | Entity Recall (di) | CPR | IPR | Utility |
|------|-------------|-------------------|-----|-----|---------|
| Longformer | 0.940 | 0.997 | 0.330 | 0.325 | 0.874 |
| DeID-GPT (GPT-4.1) | 0.990 | 1.000 | 0.674 | 0.665 | 0.754 |
| DP-Prompt (Claude-Sonnet) | 0.789 | 0.450 | 0.452 | 0.446 | 0.764 |
| Adversarial (GPT-4.1) | 0.894 | 1.000 | 0.359 | 0.365 | 0.857 |

### Span-based vs. Inference-based Discrepancy

| Dataset | Highest Token Recall | Corresponding CPR | Gap |
|--------|------------------|----------|------|
| TAB | 0.990 | 0.674 | 31.6%p |
| TAB (Longformer) | 0.940 | 0.330 | 61.0%p |
| PANORAMA | 0.984 | 0.799 | 18.5%p |

### Key Findings
- **Span-based metrics severely overestimate protection**: Longformer achieves an Entity Recall of 99.7%, yet its CPR is only 33.0%, meaning that even when nearly all PII spans are redacted, two-thirds of personal information remains inferable from context.
- **Target-focused anonymization (Adversarial) exposes non-target subjects**: On TAB, 1-AAC (target subject protection) is substantially higher than CPR (overall protection), indicating that adversarial anonymization protects the applicant while neglecting non-target subjects such as witnesses and judges.
- **The gap is larger on TAB (long legal documents) than on PANORAMA (short online text)**: Legal documents are contextually rich, providing greater room for inference.
- Even under the best configuration (DeID-GPT + GPT-4.1), CPR on TAB reaches only 67.4%, leaving nearly one-third of PII inferable.
- Substituting adversary models (GPT-4.1, Claude-Haiku-4.5) yields Spearman $\rho > 0.98$, confirming evaluation robustness.

## Highlights & Insights
- **Shifting the unit of evaluation from spans to individuals** is the paper's most significant contribution. This seemingly simple but profound observation reframes the logical foundation of anonymization evaluation, exposing a systematic blind spot created by span-based metrics across the field.
- The finding on **multi-subject exposure disparity** has direct practical relevance: adversarial anonymization protects the target subject while neglecting others, posing a serious compliance risk under GDPR, which requires protection for all identifiable individuals.
- The two-stage inference framework is transferable to other privacy-related tasks, such as privacy auditing of anonymized text and PII detection in LLM training data.

## Limitations & Future Work
- Only English documents are included; PII inference difficulty may vary across languages and cultural contexts.
- The benchmark scale is relatively small (675 documents), with TAB comprising only 144 documents.
- More advanced anonymization methods (e.g., generative approaches incorporating differential privacy) are not evaluated.
- CPR/IPR treat all PII categories uniformly — the privacy risk of leaking a name is clearly different from leaking an age.
- Future work could extend to multilingual, larger-scale document collections and incorporate category-specific PII weights.

## Related Work & Insights
- **vs. TAB**: TAB provides comprehensive PII coverage but lacks inference evaluation; SPIA adds an inference layer on top of TAB data.
- **vs. PersonalReddit**: Supports inference evaluation but targets only single authors; SPIA extends to multi-subject settings.
- **vs. PII-Bench**: Distinguishes subjects but remains at span-level evaluation; SPIA supports both multi-subject and inference-based evaluation.
- **vs. Staab et al. (2024) AAC**: AAC measures protection only for the target subject, whereas SPIA's CPR/IPR measure protection across all subjects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm shift in evaluation (spans → individuals) is an impactful contribution; the multi-subject perspective directly addresses the core requirements of GDPR.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four anonymization methods × six backbones × two datasets, with adversary substitution validating robustness.
- Writing Quality: ⭐⭐⭐⭐ Figure 1's comparison of three evaluation paradigms is highly intuitive, with clearly structured conceptual layers.
- Value: ⭐⭐⭐⭐⭐ The finding that "90% redaction yet 67% still inferable" has direct implications for privacy protection practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Accessible, Realistic, and Fair Evaluation of Positive-Unlabeled Learning Algorithms](../../ICLR2026/llm_evaluation/accessible_realistic_and_fair_evaluation_of_positive-unlabeled_learning_algorith.md)
- [\[ACL 2026\] Text-to-Distribution Prediction with Quantile Tokens and Neighbor Context](text-to-distribution_prediction_with_quantile_tokens_and_neighbor_context.md)
- [\[ACL 2026\] CAST: Achieving Stable LLM-based Text Analysis for Data Analytics](cast_achieving_stable_llm-based_text_analysis_for_data_analytics.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](piarena_a_platform_for_prompt_injection_evaluation.md)

</div>

<!-- RELATED:END -->
