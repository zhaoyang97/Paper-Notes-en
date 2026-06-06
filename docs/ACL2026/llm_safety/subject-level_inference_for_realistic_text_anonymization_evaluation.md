---
title: >-
  [Paper Note] Subject-level Inference for Realistic Text Anonymization Evaluation
description: >-
  [ACL 2026][LLM Safety][Text Anonymization] SPIA proposes the first subject-level PII inference evaluation benchmark (675 documents, 1,712 subjects, 7,040 PII), revealing that even when 90%+ of PII spans are masked…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Text Anonymization"
  - "Privacy Evaluation"
  - "Subject-level Inference"
  - "PII Reasoning"
  - "Multi-subject Protection"
date: 2026-05-08
content_hash: 0210999b74480bdd
---

# Subject-level Inference for Realistic Text Anonymization Evaluation

**Conference**: ACL 2026  
**arXiv**: [2604.21211](https://arxiv.org/abs/2604.21211)  
**Code**: [https://github.com/maisonOP/spia.git](https://github.com/maisonOP/spia.git)  
**Area**: LLM Evaluation  
**Keywords**: Text Anonymization, Privacy Evaluation, Subject-level Inference, PII Reasoning, Multi-subject Protection

## TL;DR

SPIA proposes the first subject-level PII inference evaluation benchmark (675 documents, 1,712 subjects, 7,040 PII), revealing that even when 90%+ of PII spans are masked, the subject-level inference protection rate can be as low as 33%, and focusing anonymization on a single target subject leads to higher exposure of non-target subjects.

## Background & Motivation

**Background**: Text anonymization modifies text to prevent personal identification and is a core requirement of privacy regulations like GDPR. Existing evaluation methods primarily rely on span-based metrics such as Token Recall and Entity Recall to measure whether explicit PII mentions are masked. Existing benchmarks include i2b2/UTHealth (medical), TAB (legal), and WikiPII (Wikipedia).

**Limitations of Prior Work**: There are two critical flaws. First, span-based metrics fail to capture inference risks—Staab et al. (2025) demonstrated that even after NER-based anonymization, 66.3% of personal attributes can still be inferred from the context. Second, existing methods assume a document contains only a single data subject, whereas real-world texts (legal judgments, medical records, online posts) often involve multiple individuals. Current techniques primarily protect a single main subject, leaving other mentioned individuals inadequately protected.

**Key Challenge**: Masking all explicit PII mentions (high span recall) does not equate to protecting all individuals (high inference protection). LLMs can infer masked personal information from contextual clues, and protection for non-target subjects in multi-subject documents is systematically neglected. This represents a fundamental error in the evaluation unit—it should shift from text spans to individual persons.

**Goal**: To shift the unit of anonymization evaluation from text spans to individuals, construct a multi-subject, multi-domain inference-based evaluation benchmark, and design new subject-level protection metrics.

**Key Insight**: Define a "subject" as any identifiable individual within a document and independently evaluate whether each subject's PII can be inferred from the anonymized text by an adversarial LLM.

**Core Idea**: Evaluation Unit = Individual People (instead of text spans); Protection Metric = Proportion of identifiable PII remaining after inference (instead of masking rate).

## Method

### Overall Architecture

SPIA consists of a benchmark construction phase and an evaluation framework. Benchmark construction: 675 documents were filtered from TAB (legal judgments) and PANORAMA (online text), with 1,712 subjects and 7,040 PII (across 15 categories) annotated by humans and LLMs. Evaluation framework: A three-stage pipeline—(1) Processing original text with an anonymization method; (2) An adversarial LLM (Claude-Sonnet-4.5) performing two-stage inference (subject identification + PII inference) on the anonymized text; (3) Subject matching + PII scoring + calculation of CPR/IPR metrics.

### Key Designs

1.  **Two-stage Subject-level Inference Framework**:
    - **Function**: Identifies all identifiable subjects from anonymized text and infers PII for each subject separately.
    - **Mechanism**: Stage A identifies all subjects in the document and provides distinguishing descriptions (names, roles, etc.). Stage B infers 15 categories of PII for each identified subject, split into CODE (5 categories like ID numbers, phones, emails) and NON-CODE (10 categories like name, age, occupation) for independent inference. The separation prevents the model from processing 15 categories simultaneously, reduces prompt length, and allows for type-specific handling.
    - **Design Motivation**: Extends the single-author profiling method of Staab et al. (2024) to multi-subject scenarios. Validated across 11 LLMs, Claude-Sonnet-4.5 performed best in subject matching (96%) and inference accuracy (91%).

2.  **CPR and IPR Protection Metrics**:
    - **Function**: Quantifies the protection effect of anonymization from both collective and individual perspectives.
    - **Mechanism**: Collective Protection Rate (CPR) = $1 - \sum A_i / \sum O_i$, weighted by PII count where subjects with more PII contribute more. Individual Protection Rate (IPR) = $\frac{1}{N}\sum(1 - A_i/O_i)$, an equal-weighted average across all subjects. $O_i$ is the number of PII for subject $i$ in the original text, and $A_i$ is the number of PII an adversary can still infer from the anonymized text. A value of 1 indicates total protection; 0 indicates total exposure.
    - **Design Motivation**: CPR focuses on total PII leakage, while IPR focuses on whether each individual is protected equally—it is possible to have high CPR while some subjects remain completely exposed.

3.  **PII Taxonomy (CODE + NON-CODE)**:
    - **Function**: Covers 15 categories of PII, classified by structural features to support different detection and evaluation strategies.
    - **Mechanism**: CODE types have fixed format patterns (ID, Driver's License, Phone, Passport, Email). NON-CODE types are free text (Name, Gender, Age, Location, Nationality, Education, Relationship, Occupation, Affiliation, Position). CODE types are included in inference evaluation because pattern-based NER may miss unseen formats.
    - **Design Motivation**: Unlike traditional direct/quasi-identifier classifications (which are context-dependent), classification by structural features is more stable.

### Loss & Training
This work presents an evaluation benchmark and framework and does not involve model training. The adversarial LLM used is Claude-Sonnet-4.5. PII scoring uses a three-tier system: 1.0 for exact match, 0.5 for partial match, and 0.0 for no match.

## Key Experimental Results

### Main Results (TAB Legal Dataset, selected methods × optimal backbone)

| Method | Token Recall | Entity Recall (di) | CPR | IPR | Utility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Longformer | 0.940 | 0.997 | 0.330 | 0.325 | 0.874 |
| DeID-GPT (GPT-4.1) | 0.990 | 1.000 | 0.674 | 0.665 | 0.754 |
| DP-Prompt (Claude-Sonnet) | 0.789 | 0.450 | 0.452 | 0.446 | 0.764 |
| Adversarial (GPT-4.1) | 0.894 | 1.000 | 0.359 | 0.365 | 0.857 |

### Span-based vs Inference-based Discrepancy

| Dataset | Highest Token Recall | Corresponding CPR | Gap |
| :--- | :--- | :--- | :--- |
| TAB | 0.990 | 0.674 | 31.6%p |
| TAB (Longformer) | 0.940 | 0.330 | 61.0%p |
| PANORAMA | 0.984 | 0.799 | 18.5%p |

### Key Findings
- **Span metrics significantly overestimate protection levels**: Longformer's Entity Recall reaches as high as 99.7%, but its CPR is only 33.0%, meaning that even when nearly all PII spans are masked, 2/3 of personal information can still be inferred via context.
- **Anonymization focusing on target subjects (Adversarial) exposes non-target subjects**: On TAB, 1-AAC (target subject protection) is significantly higher than CPR (overall protection), indicating that adversarial anonymization overlooks witnesses, judges, and other non-target subjects while protecting applicants.
- **The gap is larger in TAB (long legal docs) than in PANORAMA (short online text)**: Legal documents provide rich context, offering more space for inference.
- Even in the best configuration (DeID-GPT + GPT-4.1), CPR on TAB is only 67.4%, with nearly 1/3 of PII still inferable.
- Evaluation results remain robust after changing adversary models (GPT-4.1, Claude-Haiku-4.5) with Spearman $\rho > 0.98$.

## Highlights & Insights
- **The shift of the evaluation unit from spans to individuals** is the most significant contribution. This simple yet profound observation changes the logical foundation of anonymization evaluation, exposing a blind spot where the field was misled by span metrics.
- The finding of **differential exposure in multi-subject scenarios** is highly practical: adversarial anonymization protects the target but neglects others, posing a serious compliance risk under GDPR, which requires protection for all identifiable individuals.
- The two-stage inference framework can be transferred to other privacy-related tasks, such as privacy auditing of anonymized texts and PII detection in LLM training data.

## Limitations & Future Work
- Includes only English documents; PII inference difficulty may vary across languages and cultures.
- The benchmark scale is relatively small (675 docs), particularly for TAB (144 docs).
- More advanced anonymization methods (e.g., generative methods combined with Differential Privacy) were not evaluated.
- CPR/IPR does not distinguish between PII categories—the privacy risk of leaking a name is clearly different from leaking an age.
- Future work could expand to multilingual and larger document collections and introduce weights for PII categories.

## Related Work & Insights
- **vs TAB**: TAB provides comprehensive PII coverage but lacks inference evaluation; SPIA adds an inference layer to TAB data.
- **vs PersonalReddit**: Supports inference evaluation but only for a single author; SPIA extends this to multiple subjects.
- **vs PII-Bench**: Distinguishes subjects but remains at the span evaluation level; SPIA supports both multi-subject and inference evaluation.
- **vs Staab et al. (2024) AAC**: AAC only measures target subject protection; SPIA's CPR/IPR measures all subjects.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The shift in evaluation paradigm (span → individual) is an influential contribution; the multi-subject perspective targets the core requirements of GDPR.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 anonymization methods × 6 backbones × 2 datasets, with robustness verification using different adversaries.
- Writing Quality: ⭐⭐⭐⭐ The comparison of the three evaluation methods in Figure 1 is very intuitive, with clear conceptual hierarchies.
- Value: ⭐⭐⭐⭐⭐ The finding of "90% masking but 67% inferable" has a direct impact on privacy protection practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[ACL 2026\] Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization](look_twice_before_you_leap_a_rational_framework_for_localized_adversarial_anonym.md)
- [\[ACL 2026\] Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks](do_multimodal_rag_systems_leak_data_a_comprehensive_evaluation_of_membership_inf.md)
- [\[ICML 2026\] AliMark: Enhancing Robustness of Sentence-Level Watermarking Against Text Paraphrasing](../../ICML2026/llm_safety/alimark_enhancing_robustness_of_sentence-level_watermarking_against_text_paraphr.md)
- [\[ACL 2026\] De-Anonymization at Scale via Tournament-Style Attribution](de-anonymization_at_scale_via_tournament-style_attribution.md)

</div>

<!-- RELATED:END -->
