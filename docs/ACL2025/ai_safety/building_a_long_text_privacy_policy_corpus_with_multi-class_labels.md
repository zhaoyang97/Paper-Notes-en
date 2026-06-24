---
title: >-
  [Paper Note] Building a Long Text Privacy Policy Corpus with Multi-Class Labels
description: >-
  [ACL 2025][AI Safety][Privacy Policy] This paper constructs a multi-dimensional annotated corpus (64 annotation dimensions) containing the privacy policies of 149 companies, covering contentious clauses and legal rules in EU and US privacy regulations, and establishes classification benchmarks using current large language models (LLMs).
tags:
  - "ACL 2025"
  - "AI Safety"
  - "Privacy Policy"
  - "Legal Text"
  - "Multi-Class Labeling"
  - "Long Text Classification"
  - "Corpus Construction"
date: 2026-05-08
content_hash: eba5300e79af25b5
---

# Building a Long Text Privacy Policy Corpus with Multi-Class Labels

**Conference**: ACL 2025  
**Link**: [ACL Anthology](https://aclanthology.org/2025.acl-long.401/)  
**Code**: None  
**Area**: AI Safety  
**Keywords**: Privacy Policy, Legal Text, Multi-Class Labeling, Long Text Classification, Corpus Construction

## TL;DR

This paper constructs a multi-dimensional annotated corpus (64 annotation dimensions) containing the privacy policies of 149 companies, covering contentious clauses and legal rules in EU and US privacy regulations, and establishes classification benchmarks using current large language models (LLMs).

## Background & Motivation

**Background**: Privacy policies are legal contracts between internet companies and users, present on almost all websites and applications. Automated privacy policy analysis is a crucial application of NLP in the legal domain, involving tasks such as clause classification, compliance checking, and risk flagging.

**Limitations of Prior Work**: Legal texts pose unique challenges for NLP: (1) The legal meaning of terminology may depend on omissions, cross-references, or even "silence" (the omission of a topic can hold legal significance); (2) Legal texts are inherently open to multiple interpretations—"a good lawyer's answer is always 'it depends'"; (3) Existing privacy policy datasets (e.g., OPP-115, Polisis) have limited annotation dimensions and do not fully account for legal interpretation ambiguities or cross-document dependencies.

**Key Challenge**: Truly important legal issues in privacy policies (such as whether data sharing requires explicit consent or the scope of user deletion rights) often involve complex legal judgments and multi-dimensional annotations. Simple binary classification schemes fail to capture this complexity.

**Goal**: (1) Construct a corpus covering the complete privacy policies of 149 companies; (2) Design a multi-class annotation scheme encompassing 64 legal dimensions; (3) Address unique challenges such as legal text ambiguity, inter-clause dependencies, and meaningful "silence".

**Key Insight**: Drawing from a legal background (NYU School of Law), the authors design the annotation scheme from a practical legal perspective, ensuring that the annotation dimensions reflect real regulatory concerns and litigation disputes.

**Core Idea**: Employ legal experts instead of crowdsourced annotators to label privacy policies, design 64 dimensions covering core disputes in US and EU privacy regulations, and construct a "legal-grade" high-precision privacy policy corpus.

## Method

### Overall Architecture

The entire work is divided into three stages: corpus collection (gathering the full-text privacy policies of 149 companies, including referenced documents), annotation scheme design and execution (manual annotation across 64 dimensions), and benchmark experiments (evaluation of classification performance using LLMs).

### Key Designs

1. **Legal-Oriented Multi-Dimensional Annotation Scheme**:
    - **Function**: Annotates each privacy policy across 64 legal dimensions.
    - **Mechanism**: Annotation dimensions are derived from (a) core clauses in EU GDPR and US state-level privacy laws; (b) common disputes in privacy litigation; and (c) key issues of concern to legal scholars and practitioners. Each dimension uses multi-class labels (instead of simple binary classification); for instance, a "data sharing clause" might be labeled as "explicitly prohibited", "consent required", "silent", "vague", etc. A "silence" label is specially designed to handle cases where important issues are omitted in the privacy policy but carry legal meaning.
    - **Design Motivation**: The precision requirement for legal text analysis is much higher than that of general NLP tasks; binary classification schemes lose a significant amount of legally meaningful information.

2. **Long Text Processing and Cross-Referencing**:
    - **Function**: Annotates privacy policies and their referenced collateral documents as a single entity.
    - **Mechanism**: Real-world privacy policies often link to other documents (e.g., Cookie Policies, Data Processing Agreements) through "incorporation by reference." This dataset includes all referenced documents within the annotation scope, requiring annotators to synthesise all relevant documents to make a judgment. Consequently, the average length of each annotated sample far exceeds that of common NLP datasets.
    - **Design Motivation**: Analyzing only the main document while ignoring referenced documents leads to severe information loss, which is a prevalent issue in existing datasets.

3. **LLM Benchmark Evaluation Framework**:
    - **Function**: Evaluates the capability of current LLMs in multi-dimensional classification of privacy policies.
    - **Mechanism**: Evaluates the classification performance of LLMs such as GPT-4 and Claude across 64 dimensions under zero-shot and few-shot settings. To address the long-text challenge, two strategies are evaluated: full-text input and chunked input. Evaluation metrics include exact match rate and class-level F1 score.
    - **Design Motivation**: To establish reproducible baseline results and quantify the performance gap between current technology and legal expert annotations.

### Annotation Strategy

Employed law research assistants for labeling (instead of crowdsourcing). Annotators performed the task after receiving formal legal training. Multi-annotator labeling was conducted on controversial dimensions to measure consistency.

## Key Experimental Results

### Main Results

| Dimension Category | Model | Exact Match Rate | Weighted F1 | Note |
|----------|------|-----------|--------|------|
| Data Collection | GPT-4 Zero-shot | ~55-65% | ~60-70% | Moderate performance on relatively clear clauses |
| Data Sharing | GPT-4 Zero-shot | ~40-55% | ~50-60% | Difficult dimensions involving legal interpretation |
| User Rights | GPT-4 Zero-shot | ~45-60% | ~55-65% | "Silence" category is the most difficult to identify |
| Overall 64 Dimensions | GPT-4 Zero-shot | ~50% | ~58% | Significant gap remains compared to expert level |
| Overall 64 Dimensions | Claude Zero-shot | ~48% | ~56% | Comparable to GPT-4 |

### Ablation Study

| Configuration | Average F1 | Description |
|------|--------|------|
| Full Document Input | Superior | Long context helps capture cross-references |
| Chunked Input | Decreased | Loses cross-paragraph relationship information |
| With Referenced Documents | Superior | Referenced documents contain key information |
| Main Document Only | Drops 5-10% | Many dimension answers resolve in referenced documents |
| Excluding "Silence" Category | Gains 10%+ | "silence" is themost challenging category |

### Key Findings
- The "silence" category (where a privacy policy does not mention a specific topic) is the most difficult for LLMs to handle. Models tend to "find some related text" rather than correctly identifying that "the topic is not addressed at all."
- Ambiguity in legal interpretations (e.g., phrasing such as "reasonable efforts" or "may") causes the model to hesitate between multiple categories.
- Cross-document reasoning (main document + referenced documents) is critical for accurate annotations, though it significantly increases the input length.
- The performance of the best current LLMs on this task still falls significantly short of legal experts.

## Highlights & Insights
- **"Silence" as an annotation category** is a unique contribution to legal NLP. In general NLP tasks, omitted information is usually not labeled, but it holds vital significance in legal scenarios. This design philosophy is transferable to other compliance checking tasks.
- **The legal expert perspective in the annotation design** is more practically valuable than pure NLP-centric schemes, as the 64 dimensions map directly onto actual regulatory requirements and litigation scenarios.
- The dataset construction itself serves as an exemplary case of interdisciplinary (Law + NLP) collaborative research.

## Limitations & Future Work
- The sample size of 149 companies is statistically limited and primarily covers US and European corporations.
- Jurisdiction differences affect legal applicability; the annotation scheme may require adjustments for other jurisdictions (e.g., China's Personal Information Protection Law).
- Annotation costs are exceptionally high (requiring legal professionals), making it difficult to scale up.
- Future work could explore combining legal knowledge graphs to assist LLMs in long-text legal document understanding.

## Related Work & Insights
- **vs OPP-115**: OPP-115 is an early privacy policy dataset with only 10 annotation categories, primarily at the paragraph level. This work’s 64 dimensions and document-level annotations match practical legal requirements much more closely.
- **vs Polisis**: Polisis automatically analyzes privacy policies via classifiers but features fewer annotation dimensions and does not handle "silence" or cross-referencing.
- **vs LegalBench**: LegalBench is a general legal NLP benchmark with broad coverage, but its depth in the privacy policy domain cannot match that of this work.

## Rating
- Novelty: ⭐⭐⭐⭐ The 64-dimensional annotation scheme designed by legal experts and the handling of the "silence" category are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ LLM baseline experiments cover multiple settings with deep analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-balanced interdisciplinary writing covering both legal and NLP domains.
- Value: ⭐⭐⭐⭐ Provides a high-quality resource for legal NLP research, though the application domain is specialized.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](../../NeurIPS2025/ai_safety/multi-class_support_vector_machine_with_differential_privacy.md)
- [\[ACL 2025\] CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](centaur_bridging_the_impossible_trinity_of.md)
- [\[ICML 2026\] Demystifying the Optimal Fair Classifier in Multi-Class Classification](../../ICML2026/ai_safety/demystifying_the_optimal_fair_classifier_in_multi-class_classification.md)
- [\[ICML 2025\] Retraining with Predicted Hard Labels Provably Increases Model Accuracy](../../ICML2025/ai_safety/retraining_with_predicted_hard_labels_provably_increases_model_accuracy.md)
- [\[ACL 2025\] PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance](privacibench_evaluating_privacy_with_contextual_integrity.md)

</div>

<!-- RELATED:END -->
