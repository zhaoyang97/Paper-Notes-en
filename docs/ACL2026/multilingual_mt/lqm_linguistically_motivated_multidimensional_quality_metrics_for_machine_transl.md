---
title: >-
  [Paper Note] LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation
description: >-
  [ACL 2026][Multilingual & Machine Translation][Machine Translation Evaluation] Ours proposes LQM (Linguistically Motivated Multidimensional Quality Metrics), a six-level linguistically motivated MT error typology (sociolinguistics → pragmatics → semantics → morphosyntax → orthography → graphetics), and constructs a bidirectional parallel corpus of 3,850 sentences across 7 Arabic dialects. Through expert annotation of 6,113 error spans, the study reveals systematic deficiencie…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Machine Translation Evaluation"
  - "Error Typology"
  - "Arabic Dialects"
  - "Multidimensional Quality Metrics"
  - "Linguistically Motivated"
date: 2026-05-08
content_hash: 3aee07dd181b0490
---

# LQM: Linguistically Motivated Multidimensional Quality Metrics for Machine Translation

**Conference**: ACL 2026  
**arXiv**: [2604.18490](https://arxiv.org/abs/2604.18490)  
**Code**: [GitHub](https://github.com/UBC-NLP/LQM_MT)  
**Area**: Multilingual Translation  
**Keywords**: Machine Translation Evaluation, Error Typology, Arabic Dialects, Multidimensional Quality Metrics, Linguistically Motivated

## TL;DR

Ours proposes LQM (Linguistically Motivated Multidimensional Quality Metrics), a six-level linguistically motivated MT error typology (sociolinguistics → pragmatics → semantics → morphosyntax → orthography → graphetics), and constructs a bidirectional parallel corpus of 3,850 sentences across 7 Arabic dialects. Through expert annotation of 6,113 error spans, the study reveals systematic deficiencies in existing MT systems regarding dialectal and culture-aware translation.

## Background & Motivation

**Background**: Existing MT evaluation frameworks—including automatic metrics (e.g., BLEU, COMET) and human evaluation schemes (e.g., MQM)—are largely designed to be language-agnostic, targeting general translation quality assessment.

**Limitations of Prior Work**: For diglossic languages (e.g., Arabic), standard evaluation frameworks fail to capture dialect- and culture-specific translation errors. In such languages, translation failures often stem not from surface-form errors but from language variety mismatches (e.g., Modern Standard Arabic vs. dialects), improper content coverage, and pragmatic appropriateness issues.

**Key Challenge**: While existing standards like MQM provide hierarchical error typologies, their classification dimensions primarily address surface linguistic features (e.g., fluency, accuracy) and lack systematic modeling of deep linguistic dimensions (e.g., sociolinguistics, pragmatics), resulting in an inability to capture and quantify core error types in dialectal translation.

**Goal**: To design a linguistically motivated multidimensional error typology capable of diagnosing MT errors across levels ranging from sociolinguistics to graphetics, and to perform systematic verification on Arabic dialect translation.

**Key Insight**: A hierarchical error typology is constructed based on six fundamental linguistic levels: sociolinguistics, pragmatics, semantics, morphosyntax, orthography, and graphetics.

**Core Idea**: MT quality evaluation should transcend surface forms to perform systematic diagnosis at various linguistic levels. Although verified on Arabic, LQM is designed as a language-agnostic framework adaptable to any language.

## Method

### Overall Architecture

LQM is a hierarchical MT error typology comprising six linguistic levels, from macro socio-cultural factors to micro character representations, with specific error types further subdivided under each level. Alongside this typology, the authors construct a bidirectional parallel corpus covering 7 Arabic dialects and conduct zero-shot LLM translation evaluations and expert manual annotations.

### Key Designs

**1. Six-layer Linguistic Error Typology (LQM Taxonomy): Shifting MT error diagnosis from "accuracy/fluency" dichotomy to six linguistic levels**

Dimensions in existing MQM frameworks primarily reside at the semantic and morphosyntactic levels. For diglossic languages like Arabic, the dialect selection and cultural appropriateness that determine translation success often fall outside this scope. LQM spans six layers along linguistic depth: the sociolinguistic layer manages dialect vs. standard selection, register appropriateness, and cultural sensitivity; the pragmatic layer handles illocutionary meaning, politeness strategies, and translation of implicit assumptions; the semantic layer manages accuracy of word meanings, collocations, and metaphors; the morphosyntactic layer handles inflection and syntactic structures; the orthographic layer manages spelling and punctuation standards; and the graphetic layer handles character encoding and display correctness. The higher levels address "soft" socio-cultural factors while lower levels address "hard" surface forms, filling the gap in MQM for deep-level dimensions.

**2. Seven-Dialect Bidirectional Parallel Corpus Construction: Moving evaluation from MSA to challenging dialects**

Existing Arabic translation evaluation datasets predominantly focus on Modern Standard Arabic (MSA), leaving a gap in data for dialectal translation—a more difficult and realistic scenario. The authors construct a bidirectional parallel corpus covering 7 Arabic dialects (Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, and Yemeni), totaling 3,850 sentences (550 per dialect). Content is intentionally sourced from conversational and culturally rich material to elicit sociolinguistic and pragmatic errors.

**3. Zero-shot LLM Evaluation and Expert Span Annotation: Out-of-the-box translation capabilities paired with fine-grained diagnosis**

The zero-shot setting reflects the true translation capabilities of LLMs without dialect adaptation; thus, 6 LLMs are evaluated under this setting. Beyond scoring, linguistic experts use the LQM framework for span-level manual annotation, producing 6,113 labeled error spans across 3,495 unique erroneous sentences, providing severity-weighted quality scores for each span. The span-level granularity allows for precise analysis of error location, level, and severity.

## Key Experimental Results

### Main Results

| Dimension | Quantity | Remarks |
|-----------|----------|---------|
| Number of Dialects | 7 Arabic dialects | Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, Yemeni |
| Parallel Sentence Pairs | 3,850 | 550 per dialect |
| Evaluated LLMs | 6 | Zero-shot setting |
| Annotated Error Spans | 6,113 | Expert-level span annotation |
| Erroneous Sentences | 3,495 | Unique erroneous sentences |
| Translation Direction | Bidirectional | Dialect ↔ English |

### Ablation Study

| Analysis Dimension | Key Finding | Remarks |
|--------------------|-------------|---------|
| Auto Metrics vs. Human | Comparison of spBLEU and LQM scores | Auto metrics struggle to capture deep linguistic errors |
| Dialectal Analysis | Significant variance in error distribution | Lower-resource dialects exhibit poorer translation quality |
| Error Level Analysis | High proportion of sociolinguistic and pragmatic errors | Confirms necessity of beyond-surface evaluation |
| Severity Weighting | Severity distribution varies across levels | Sociolinguistic errors are often the most severe |

### Key Findings

- Errors in LLM dialectal translation are not limited to lexical and grammatical levels; a significant number of errors occur at sociolinguistic (dialect selection, cultural appropriateness) and pragmatic (illocutionary meaning, politeness strategies) levels.
- The standard MQM framework cannot systematically capture these deep-level errors, a gap filled by the LQM six-layer typology.
- Performance of the 6 LLMs varies significantly across the 7 dialects, with low-resource dialects (e.g., Mauritanian) showing markedly lower quality.
- Significant deviations exist between automatic metrics like spBLEU and LQM expert scores, particularly in dimensions involving cultural and pragmatic appropriateness.

## Highlights & Insights

- **Linguistic Depth**: Constructing an error typology from six fundamental linguistic levels provides greater diagnostic power than the traditional "accuracy/fluency" dichotomy of MQM.
- **Dialectal Diversity**: Covering 7 Arabic dialects represents the largest scale in similar studies, including representative sub-regions like Maghreb, Mashriq, Gulf, and Yemen.
- **Framework Generality**: Although verified on Arabic, LQM is designed to be language-agnostic and adaptable to other diglossic or multi-dialectal languages (e.g., Chinese dialects, Hindi-Urdu).
- **Data Quality**: Expert-level span annotation (6,113 spans) provides more granular diagnostic information than sentence-level scoring.

## Limitations & Future Work

- Verification is limited to Arabic dialects; applicability to other languages (especially those with significantly different morphological systems) requires further validation.
- The data scale (3,850 sentences), while substantial for manual annotation, may be insufficient for training automated evaluation models based on LQM.
- Specific performance differences among the 6 LLMs were not detailed in the summary.
- Integration of the LQM typology into automatic MT evaluation metrics for end-to-end evaluation was not explored.
- Future work may extend LQM to speech translation and multimodal translation evaluation.

## Related Work & Insights

- **vs. MQM**: LQM adds sociolinguistic and pragmatic layers to MQM, capturing dialect- and culture-related errors missed by the latter.
- **vs. BLEU/COMET**: Automatic metrics focusing on n-gram matching or semantic similarity cannot diagnose specific error types or capture sociolinguistic translation failures.
- **vs. Arabic MT Research**: While existing research focuses on MSA, LQM provides the first systematic evaluation of multi-dialectal translation quality.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of the six-layer linguistic error typology is deep and well-structured; incorporating sociolinguistics and pragmatics into MT evaluation is a significant innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Substantial scale with 7 dialects, 6 LLMs, and 6,113 error annotations.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of the linguistic framework and hierarchical classification.
- Value: ⭐⭐⭐⭐ Provides an important foundation for dialectal and culture-aware MT evaluation with a highly generalizable framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] XQ-MEval: A Dataset with Cross-lingual Parallel Quality for Benchmarking Translation Metrics](xq-meval_a_dataset_with_cross-lingual_parallel_quality_for_benchmarking_translat.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ACL 2025\] M-MAD: Multidimensional Multi-Agent Debate for Advanced Machine Translation Evaluation](../../ACL2025/multilingual_mt/m-mad_multidimensional_multi-agent_debate_for_advanced_machine_translation_evalu.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models](laobench_a_large-scale_multidimensional_lao_benchmark_for_large_language_models.md)

</div>

<!-- RELATED:END -->
