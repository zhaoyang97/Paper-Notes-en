---
title: >-
  [Paper Note] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation
description: >-
  [ACL 2026][Multilingual & Machine Translation][Machine Translation Evaluation] Ours proposes LQM (Linguistically Motivated Multidimensional Quality Metrics)…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Machine Translation Evaluation"
  - "Taxonomy of Errors"
  - "Arabic Dialects"
  - "Multidimensional Quality Metrics"
  - "Linguistically Motivated"
date: 2026-05-08
content_hash: a41d976aa35866fb
---

# LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation

**Conference**: ACL 2026  
**arXiv**: [2604.18490](https://arxiv.org/abs/2604.18490)  
**Code**: [GitHub](https://github.com/UBC-NLP/LQM_MT)  
**Area**: Multilingual Translation  
**Keywords**: Machine Translation Evaluation, Taxonomy of Errors, Arabic Dialects, Multidimensional Quality Metrics, Linguistically Motivated

## TL;DR

Ours proposes LQM (Linguistically Motivated Multidimensional Quality Metrics), a six-level linguistically driven MT error taxonomy (Sociolinguistics → Pragmatics → Semantics → Morphosyntax → Orthography → Graphetics). By constructing a bidirectional parallel corpus of 3,850 sentences across 7 Arabic dialects and expert-annotating 6,113 error spans, the study reveals systematic deficiencies in current MT systems regarding dialectal and culture-aware translation.

## Background & Motivation

**Background**: Existing MT evaluation frameworks—including automatic metrics (e.g., BLEU, COMET) and human evaluation schemes (e.g., MQM)—are largely designed to be language-agnostic and targeted at general translation quality assessment.

**Limitations of Prior Work**: For diglossic languages (e.g., Arabic), standard evaluation frameworks fail to capture dialect- and culture-specific translation errors. In such languages, translation failures often arise not from surface-level formal errors, but from language variety mismatches (e.g., Modern Standard Arabic vs. dialects), improper content coverage, and pragmatic appropriateness issues.

**Key Challenge**: While existing standards like MQM provide hierarchical error taxonomies, their classification dimensions primarily target surface linguistic features (e.g., fluency, accuracy). They lack systematic modeling of deep linguistic layers (e.g., sociolinguistics, pragmatics), resulting in an inability to capture and quantify core error types in dialectal translation.

**Goal**: To design a linguistically motivated multidimensional error taxonomy capable of diagnosing MT errors across layers from sociolinguistics to graphetics, and to systematically validate it on Arabic dialect translation.

**Key Insight**: A hierarchical error taxonomy is constructed based on six fundamental linguistic levels: sociolinguistics, pragmatics, semantics, morphosyntax, orthography, and graphetics.

**Core Idea**: MT quality assessment should transcend surface forms to perform systematic diagnostics across all linguistic levels. Although validated with Arabic, LQM is a language-agnostic framework adaptable to any language.

## Method

### Overall Architecture

LQM is a hierarchical MT error taxonomy comprising six linguistic levels, ranging from macro socio-cultural factors to micro character representations, with specific error types subdivided under each level. Alongside this taxonomy, a bidirectional parallel corpus covering 7 Arabic dialects was constructed, followed by zero-shot LLM translation evaluation and expert human annotation.

### Key Designs

1. **Six-layer Linguistic Error Taxonomy (LQM Taxonomy)**:
    - **Function**: Provides a systematic MT error diagnostic framework.
    - **Mechanism**: A six-layer design—(1) Sociolinguistic layer: choice of dialect vs. standard language, register appropriateness, and cultural sensitivity; (2) Pragmatic layer: translation of illocutionary force, politeness strategies, and conversational implicatures; (3) Semantic layer: accuracy of word meanings, collocations, and metaphors; (4) Morphosyntactic layer: correctness of inflectional forms and syntactic structures; (5) Orthographic layer: standardization of spelling and punctuation; (6) Graphetic layer: correctness of character encoding and display.
    - **Design Motivation**: Existing MQM mainly focuses on semantic and morphosyntactic levels, lacking modeling for deeper dimensions like sociolinguistics and pragmatics. For diglossic languages like Arabic, dialect choice and cultural appropriateness are often critical to translation success.

2. **Seven-Dialect Bidirectional Parallel Corpus Construction**:
    - **Function**: Provides multi-dialectal, culturally rich translation evaluation data.
    - **Mechanism**: Constructs a bidirectional parallel corpus covering 7 Arabic dialects (Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, Yemeni), totaling 3,850 sentences (550 per dialect) sourced from conversational and culturally rich content.
    - **Design Motivation**: Existing Arabic translation evaluation datasets mainly focus on Modern Standard Arabic (MSA), neglecting dialectal translation, which is a more challenging and practical scenario.

3. **Zero-Shot LLM Evaluation and Expert Span Annotation**:
    - **Function**: Evaluates current LLM performance on dialectal translation and quantifies specific errors.
    - **Mechanism**: Evaluates 6 LLMs in a zero-shot setting, followed by expert human annotation using the LQM system at the span level, producing 6,113 labeled error spans across 3,495 unique erroneous sentences, accompanied by severity-weighted quality scores.
    - **Design Motivation**: The zero-shot setting reflects the out-of-the-box translation capabilities of LLMs, while expert annotation ensures the precision and linguistic soundness of error diagnostics.

## Key Experimental Results

### Main Results

| Dimension | Quantity | Notes |
|------|--------|------|
| Number of Dialects | 7 Arabic Dialects | Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, Yemeni |
| Parallel Sentence Pairs | 3,850 sentences | 550 per dialect |
| Number of Evaluated LLMs | 6 | Zero-shot setting |
| Annotated Error Spans | 6,113 | Expert-level span annotation |
| Erroneous Sentences | 3,495 | Unique erroneous sentences |
| Translation Direction | Bidirectional | Dialect ↔ English |

### Ablation Study

| Analysis Dimension | Key Findings | Notes |
|---------|---------|------|
| Automatic Metrics vs. Human | Contrast between spBLEU and LQM quality scores | Automatic metrics struggle to capture deep linguistic errors |
| By-Dialect Analysis | Significant differences in error distribution across dialects | Translation quality is notably lower for low-resource dialects |
| By-Error Layer Analysis | High proportion of sociolinguistic and pragmatic errors | Confirms the necessity of evaluation beyond surface levels |
| Severity Weighting | Distribution of error severity differs across linguistic levels | Sociolinguistic errors tend to be the most severe |

### Key Findings

- Errors in dialectal translation within existing LLMs are not limited to lexical and grammatical levels; a significant number of errors occur at the sociolinguistic (dialect choice, cultural appropriateness) and pragmatic (illocutionary force, politeness strategies) levels.
- The standard MQM framework cannot systematically capture these deep errors; the six-layer LQM taxonomy fills this gap.
- Performance of 6 LLMs varies significantly across 7 dialects, with noticeably poorer translation quality for low-resource dialects (e.g., Mauritanian).
- A significant deviation exists between automatic metrics like spBLEU and LQM expert scores, particularly in dimensions involving cultural and pragmatic appropriateness.

## Highlights & Insights

- **Linguistic Depth**: Constructing an error taxonomy from six fundamental linguistic levels provides significantly more diagnostic power than the "Accuracy/Fluency" dichotomy of existing MQM.
- **Dialectal Diversity**: Covering 7 Arabic dialects is among the largest studies of its kind, selecting representative dialects across various sub-regions (Maghreb, Mashriq, Gulf, and Yemen).
- **Framework Generality**: Although validated on Arabic, LQM is designed as a language-agnostic framework adaptable to other diglossic or multi-dialectal languages (e.g., Chinese dialects, Hindi-Urdu).
- **Data Quality**: Expert-level span annotation (6,113 error spans) provides more granular error diagnostic information than sentence-level scoring.

## Limitations & Future Work

- Validation is limited to Arabic dialects; applicability to other languages, especially those with significantly different morphological systems, requires further verification.
- Although the data scale (3,850 sentences) is substantial for human annotation, it may be insufficient for training LQM-based automated evaluation models.
- Specific performance variations among the six LLMs were not elaborated in detail in the summary.
- Integration of the LQM system into automatic MT evaluation metrics to achieve end-to-end automated assessment has not yet been explored.
- Future work could extend LQM to speech translation and multimodal translation evaluation.

## Related Work & Insights

- **vs MQM**: LQM adds sociolinguistic and pragmatic layers to MQM, enabling the capture of dialect- and culture-related errors typically missed by MQM.
- **vs BLEU/COMET**: Automatic metrics focus on n-gram matching or semantic similarity, failing to diagnose specific error types or capture failures in the sociolinguistic dimension.
- **vs Arabic MT Research**: Prior research mostly focuses on MSA translation; LQM provides the first systematic evaluation of multi-dialectal translation quality.

## Rating

- Novelty: ⭐⭐⭐⭐ The six-layer linguistic error taxonomy is well-designed and deep; incorporating sociolinguistics and pragmatics into MT evaluation is a significant innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 dialects, 6 LLMs, and 6,113 error annotations constitute a substantial scale.
- Writing Quality: ⭐⭐⭐⭐ The linguistic framework is clearly articulated with a well-structured taxonomy.
- Value: ⭐⭐⭐⭐ Significantly advances dialect- and culture-aware MT evaluation; the framework's generality ensures broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] Evaluating the Impact of Verbal Multiword Expressions on Machine Translation](evaluating_the_impact_of_verbal_multiword_expressions_on_machine_translation.md)
- [\[ACL 2026\] NiuTrans.LMT: Toward Inclusive and Scalable Multilingual Machine Translation with LLMs](niutranslmt_toward_inclusive_and_scalable_multilingual_machine_translation_with_.md)

</div>

<!-- RELATED:END -->
