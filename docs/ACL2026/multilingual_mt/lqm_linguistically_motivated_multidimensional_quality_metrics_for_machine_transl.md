---
title: >-
  [Paper Note] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation
description: >-
  [ACL 2026][Multilingual & Machine Translation][Machine translation evaluation] This paper proposes LQM (Linguistically Motivated Multidimensional Quality Metrics)…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Machine translation evaluation"
  - "error taxonomy"
  - "Arabic dialects"
  - "multidimensional quality metrics"
  - "linguistically motivated"
date: 2026-05-08
content_hash: 200f159b6a42cdad
---

# LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation

**Conference**: ACL 2026
**arXiv**: [2604.18490](https://arxiv.org/abs/2604.18490)  
**Code**: [GitHub](https://github.com/UBC-NLP/LQM_MT)  
**Area**: Multilingual Translation
**Keywords**: Machine translation evaluation, error taxonomy, Arabic dialects, multidimensional quality metrics, linguistically motivated

## TL;DR

This paper proposes LQM (Linguistically Motivated Multidimensional Quality Metrics), a six-tier linguistically motivated MT error taxonomy spanning sociolinguistics → pragmatics → semantics → morphosyntax → orthography → graphetics. A bidirectional parallel corpus of 3,850 sentences across seven Arabic dialects is constructed, and 6,113 expert-annotated error spans are produced to reveal systematic deficiencies of existing MT systems in dialect-aware and culturally sensitive translation.

## Background & Motivation

**Background**: Existing MT evaluation frameworks—including automatic metrics (e.g., BLEU, COMET) and human evaluation schemes (e.g., MQM)—are largely designed to be language-agnostic, targeting general-purpose translation quality assessment.

**Limitations of Prior Work**: For diglossic languages such as Arabic, standard evaluation frameworks fail to capture dialect- and culture-specific translation errors. In such languages, translation failures often stem not from surface-form errors but from language variety mismatches (e.g., Modern Standard Arabic vs. dialects), inadequate content coverage, and pragmatic appropriateness issues.

**Key Challenge**: Although existing standards such as MQM provide hierarchical error taxonomies, their dimensions primarily target surface linguistic features (e.g., fluency, accuracy) and lack systematic modeling of deeper linguistic dimensions such as sociolinguistics and pragmatics, rendering many core error types in dialectal translation uncapturable and unquantifiable.

**Goal**: To design a linguistically motivated multidimensional error taxonomy capable of diagnosing MT errors across levels ranging from sociolinguistics to graphetics, and to systematically validate it on Arabic dialect translation.

**Key Insight**: The framework is grounded in six fundamental linguistic strata—sociolinguistics, pragmatics, semantics, morphosyntax, orthography, and graphetics—from which a hierarchical error taxonomy is constructed.

**Core Idea**: MT quality evaluation should go beyond surface form and perform systematic diagnosis at every linguistic level. Although validated on Arabic, LQM is designed as a language-agnostic framework adaptable to any language.

## Method

### Overall Architecture

LQM is a hierarchical MT error taxonomy comprising six linguistic tiers, ranging from macro-level sociocultural factors to micro-level character representation, each further subdivided into concrete error types. Complementing this taxonomy, a bidirectional parallel corpus covering seven Arabic dialects is constructed, and zero-shot LLM translation evaluation together with expert human annotation is conducted.

### Key Designs

1. **Six-Tier Linguistic Error Taxonomy (LQM Taxonomy)**:
    - Function: Provides a systematic diagnostic framework for MT errors.
    - Mechanism: Six tiers are defined—(1) Sociolinguistics: dialect vs. standard language selection, register appropriateness, cultural sensitivity; (2) Pragmatics: implicature, politeness strategies, translation of presuppositions; (3) Semantics: accuracy of word meaning, collocation, and metaphor; (4) Morphosyntax: correctness of inflectional morphology and syntactic structures; (5) Orthography: spelling and punctuation norms; (6) Graphetics: correctness of character encoding and rendering.
    - Design Motivation: Existing MQM primarily operates at the semantic and morphosyntactic levels and lacks modeling of deeper dimensions such as sociolinguistics and pragmatics. For diglossic languages such as Arabic, dialect selection and cultural appropriateness are often decisive factors in translation success or failure.

2. **Seven-Dialect Bidirectional Parallel Corpus Construction**:
    - Function: Provides multi-dialect, culturally rich translation evaluation data.
    - Mechanism: A bidirectional parallel corpus covering seven Arabic dialects (Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, and Yemeni) is constructed, comprising 3,850 sentences (550 per dialect) sourced from conversational and culturally rich content.
    - Design Motivation: Existing Arabic translation evaluation datasets focus primarily on Modern Standard Arabic (MSA), neglecting dialectal translation—a scenario that is both more challenging and more practically relevant.

3. **Zero-Shot LLM Evaluation and Expert Span Annotation**:
    - Function: Assesses current LLM performance on dialectal translation and quantifies specific errors.
    - Mechanism: Six LLMs are evaluated in a zero-shot setting, after which linguistic experts perform span-level manual annotation using the LQM taxonomy, producing 6,113 labeled error spans covering 3,495 unique erroneous sentences, accompanied by severity-weighted quality scores.
    - Design Motivation: The zero-shot setting reflects the out-of-the-box translation capability of LLMs, while expert annotation ensures precision and linguistic validity in error diagnosis.

## Key Experimental Results

### Main Results

| Dimension | Scale | Notes |
|-----------|-------|-------|
| Number of dialects | 7 Arabic dialects | Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, Yemeni |
| Parallel sentence pairs | 3,850 | 550 per dialect |
| LLMs evaluated | 6 | Zero-shot setting |
| Annotated error spans | 6,113 | Expert-level span annotation |
| Erroneous sentences | 3,495 | Unique erroneous sentences |
| Translation direction | Bidirectional | Dialect ↔ English |

### Ablation Study

| Analysis Dimension | Key Finding | Notes |
|-------------------|------------|-------|
| Automatic metrics vs. human | spBLEU vs. LQM quality scores compared | Automatic metrics fail to capture deep linguistic errors |
| Per-dialect analysis | Error distribution varies significantly across dialects | Lower-resource dialects yield worse translation quality |
| Per-tier analysis | Sociolinguistic and pragmatic errors account for a high proportion | Confirms the necessity of evaluation beyond surface form |
| Severity weighting | Severity distributions differ across error tiers | Sociolinguistic errors tend to be the most severe |

### Key Findings

- Errors made by existing LLMs in dialectal translation extend beyond the lexical and grammatical levels; a substantial proportion occurs at the sociolinguistic (dialect selection, cultural appropriateness) and pragmatic (implicature, politeness strategies) levels.
- The standard MQM framework cannot systematically capture these deep-level errors; the six-tier LQM taxonomy fills this gap.
- Performance of the six LLMs varies markedly across the seven dialects, with lower-resource dialects (e.g., Mauritanian Arabic) exhibiting noticeably inferior translation quality.
- Automatic metrics such as spBLEU diverge considerably from LQM expert scores, particularly on dimensions involving cultural and pragmatic appropriateness.

## Highlights & Insights

- **Linguistic Depth**: Constructing an error taxonomy grounded in six fundamental linguistic tiers offers far greater diagnostic power than the accuracy/fluency dichotomy of existing MQM frameworks.
- **Dialectal Diversity**: Coverage of seven Arabic dialects represents the largest scale among comparable studies, with representative dialects spanning Maghrebi, Mashreqi, Gulf, and Yemeni sub-regions.
- **Framework Generalizability**: Although validated on Arabic, LQM is designed as a language-agnostic framework adaptable to other diglossic or multi-dialectal languages (e.g., Chinese dialects, Hindi–Urdu).
- **Data Quality**: Expert-level span annotation (6,113 error spans) provides considerably more fine-grained diagnostic information than sentence-level scoring.

## Limitations & Future Work

- Validation is limited to Arabic dialects; applicability to other languages—particularly those with substantially different morphological systems—requires further investigation.
- The dataset size (3,850 sentences), while substantial for human annotation, may be insufficient to support training of automated evaluation models based on LQM.
- Detailed performance differences among the six LLMs are not elaborated in the abstract.
- Integration of the LQM taxonomy into automatic MT evaluation metrics for end-to-end automated assessment remains unexplored.
- Future work may extend LQM to speech translation and multimodal translation evaluation.

## Related Work & Insights

- **vs. MQM**: LQM augments MQM with sociolinguistic and pragmatic tiers, enabling the capture of dialect- and culture-related errors that MQM misses.
- **vs. BLEU/COMET**: Automatic metrics focus solely on n-gram overlap or semantic similarity and are incapable of diagnosing specific error types, let alone capturing translation failures at the sociolinguistic level.
- **vs. Arabic MT Research**: Existing work primarily targets MSA translation; LQM provides the first systematic evaluation of multi-dialect translation quality.

## Rating

- Novelty: ⭐⭐⭐⭐ The six-tier linguistic error taxonomy is well-motivated and substantive; incorporating sociolinguistics and pragmatics into MT evaluation represents an important contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven dialects, six LLMs, and 6,113 annotated error spans constitute a considerable scale.
- Writing Quality: ⭐⭐⭐⭐ The linguistic framework is clearly articulated and the taxonomy tiers are well-structured.
- Value: ⭐⭐⭐⭐ The work meaningfully advances dialect-aware and culturally sensitive MT evaluation; the generalizability of the framework broadens its applicability.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] Beyond Literal Mapping: Benchmarking and Improving Non-Literal Translation Evaluation](beyond_literal_mapping_benchmarking_and_improving_non-literal_translation_evalua.md)

</div>

<!-- RELATED:END -->
