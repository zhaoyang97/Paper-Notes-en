---
title: >-
  [Paper Note] KoGEM: Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean
description: >-
  [ACL 2025][LLM (Other)][linguistic competence] Proposes KoGEM (Korean Grammar Evaluation Benchmark), which contains 1,524 multiple-choice questions based on theoretical linguistics classification, covering 16 subcategories across 5 major domains: phonology, morphology, syntax, semantics, and prescriptive grammar. It evaluates 27 LLMs under a zero-shot setting and compares them with humans, revealing that LLMs perform significantly worse than humans on linguistic subcategories…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "linguistic competence"
  - "Korean grammar"
  - "benchmark"
  - "phonology"
  - "experiential knowledge"
  - "LLM evaluation"
date: 2026-05-08
content_hash: 348487788a3749ad
---

# KoGEM: Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean

**Conference**: ACL 2025  
**arXiv**: [2506.01237](https://arxiv.org/abs/2506.01237)  
**Code**: [https://github.com/SungHo3268/KoGEM](https://github.com/SungHo3268/KoGEM)  
**Area**: LLM Evaluation / Linguistics  
**Keywords**: linguistic competence, Korean grammar, benchmark, phonology, experiential knowledge, LLM evaluation

## TL;DR

Proposes KoGEM (Korean Grammar Evaluation Benchmark), which contains 1,524 multiple-choice questions based on theoretical linguistics classification, covering 16 subcategories across 5 major domains: phonology, morphology, syntax, semantics, and prescriptive grammar. It evaluates 27 LLMs under a zero-shot setting and compares them with humans, revealing that LLMs perform significantly worse than humans on linguistic subcategories requiring experiential knowledge (e.g., pronunciation rules and phonological changes), while explicitly supplementing experiential knowledge (e.g., pronunciation text, morphemic decomposition) can lead to substantial improvements.

## Background & Motivation

**LLM language competence evaluation is biased towards English**: Existing language competence benchmarks mainly focus on English morphology and syntax, paying insufficient attention to unique linguistic phenomena in non-English languages.

**Korean as an agglutinative language presents unique challenges**: Korean features rich morphological variations (particle agglutination) and a unique writing system (Hangul) that triggers specific phonological rules (consonant assimilation, vowel harmony), requiring an independent evaluation framework.

**Lack of fine-grained grammar benchmarks**: Existing Korean evaluations (such as vocabulary knowledge tests and spelling error detection) have narrow coverage and lack systematic classification and fine-grained subcategory analysis based on theoretical linguistics.

**Overall scores mask real differences in capability**: Looking only at the total scores across the five major domains, o1-preview comprehensively outperforms humans, but this masks its significant disadvantages in specific subcategories—demanding a more fine-grained analysis.

The fundamental issue of **"linguistic competence" vs. "statistical pattern matching"**: The excellent performance of LLMs may stem from large-scale training data rather than genuine language understanding (linguistic competence in the Chomskyan sense).

**Experiential knowledge dimension ignored**: Linguistic knowledge naturally acquired by humans through daily pronunciation experiences (subvocalization, morphemic decomposition intuition) is difficult for LLMs to obtain, but has not been systematically studied before.

## Method

### Framework Overview

Extracts Korean grammar questions from four types of official Korean examinations (College Scholastic Ability Test CSAT, National United Achievement Test NUAT, High School Graduation Equivalency Examination HSQE, Civil Service Examination CSE), classifies them into 16 subcategories under 5 major domains based on theoretical linguistics, and constructs a multiple-choice QA benchmark. It evaluates 27 LLMs of different sizes and types under a zero-shot setting and collects human performance through public statistics and crowdsourcing.

### Key Designs

1. **Based on Theoretical Linguistics Taxonomy**

    - **Function**: Classifies 1,524 questions into 16 subcategories: phonology (phonological system / phonological change), morphology (parts of speech / morpheme / word formation), syntax (sentence structure / syntactic features), semantics (vocabulary / lexical semantics / pragmatics), and prescriptive grammar (orthography / standard language / standard pronunciation / loanword spelling / romanization / cross-category).
    - **Mechanism**: Uses prescriptive grammar as a measurable proxy for linguistic competence, where each subcategory corresponds to a core subfield of linguistics.
    - **Design Motivation**: Fine-grained classification can reveal the heterogeneous performance of LLMs across different linguistic dimensions—a high overall score does not mean strength in every dimension.

2. **Multi-Source Authoritative Data Construction**

    - **Function**: Extracts questions via OCR, manually proofreads, and formats them in HTML from four types of official examinations aimed at native Korean speakers, excluding image-dependent questions.
    - **Mechanism**: Three native Korean major annotators independently classify the questions, with a majority vote determining the final category, and questions spanning more than three categories are removed to ensure the purity of the subcategory evaluation.
    - **Design Motivation**: Official examination questions ensure the authority of difficulty and quality, and exams for native speakers reflect realistic language competence expectations.

3. **Experiential Knowledge Enhancement Experiments**

    - **Function**: Attaches phonetic text generated by g2pK (simulating human subvocalization) to the phonological change subcategory, and morphemic decomposition text from the Kiwi tool to the morpheme subcategory.
    - **Mechanism**: Explicitly converts the "implicit" experiential knowledge used by humans into text inputs to test whether LLMs can leverage this extra information to improve performance.
    - **Design Motivation**: Validates the core hypothesis of "whether the performance gap of LLMs in weak dimensions stems from lack of knowledge or lack of reasoning capability."

## Key Experimental Results

### Table 1: Zero-shot Accuracy in Major Categories (Selected Models)

| Model | Type | Phonology | Morphology | Syntax | Semantics | Prescriptive | Average |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| o1-preview | EN/Closed | 71.83 | 79.48 | 80.14 | 89.35 | 79.09 | **81.04** |
| Claude-3.5-Sonnet | EN/Closed | 47.42 | 52.61 | 64.38 | 74.55 | 46.82 | 59.97 |
| GPT-4o | EN/Closed | 44.60 | 51.49 | 55.48 | 71.95 | 58.64 | 57.87 |
| EXAONE-3.5-32B | KO/Open | 27.23 | 37.31 | 36.30 | 50.65 | 37.27 | 38.98 |
| HyperCLOVA-HCX-003 | KO/Closed | 32.39 | 41.79 | 41.10 | 55.32 | 48.18 | 44.62 |
| LLM Average | - | 31.33 | 37.08 | 39.00 | 51.36 | 35.71 | 40.24 |
| **Human** | - | **66.70** | **56.95** | **64.75** | **70.84** | **54.34** | **63.04** |

### Table 2: Quality Evaluation of Explanations Generated by LLMs

| Model | Faithfulness | Coherence | Fluency | Relevance |
|------|:---:|:---:|:---:|:---:|
| HyperCLOVA-HCX-003 | 0.80 | 0.86 | 0.98 | 0.92 |
| Claude-3.5-Sonnet | 0.92 | 0.96 | 1.00 | 1.00 |
| GPT-4o | 0.86 | 0.94 | 1.00 | 1.00 |

### Table 3: Effects of Experiential Knowledge Enhancement

| Subcategory | Enhancement Method | Gain |
|------|:---:|:---:|
| Phonological Change | +Pronunciation Text (g2pK) | +3.1% ~ +26.3% |
| Morpheme | +Morphemic Decomposition (Kiwi) | +7.1% ~ +20.0% |

## Key Findings

- **o1-preview is the only LLM that comprehensively outperforms humans** (by 18% on average), but it is only 5.13% higher in the phonology subcategory, far below its advantage in other dimensions—phonology is a "hidden weakness" of LLMs.
- **Phonological change subcategory has the largest gap**: Humans outperform the LLM average by 35%+, which is the largest performance gap among all 16 subcategories.
- **Experiential knowledge is a key bottleneck**: After supplementing pronunciation text, the performance gain of LLMs on phonological change can reach up to 26.3%, and with morphemic decomposition, the gain can reach up to 20.0%—proving that the gap mainly stems from a lack of knowledge rather than insufficient reasoning ability.
- **s1-32B thinking time analysis**: The model spends significantly longer thinking in three subcategories—phonological change, morpheme, and pragmatics—which happen to be the subcategories with the largest gaps compared to humans. This indicates that these subcategories are inherently more difficult for LLMs.
- **Korean-centric vs. English-centric**: The English-centric s1-32B and DeepSeek-R1 series outperform all Korean-centric models due to multilingual training, demonstrating the effectiveness of multilingual training and test-time scaling.
- **Test-time scaling is effective**: The performance of the o1-preview, s1-32B, and DeepSeek-R1 series proves that increasing the computation budget during inference can significantly improve language competence.

## Highlights & Insights

- **Fine-grained analysis of 16 subcategories** reveals the hidden weaknesses of "seemingly omnipotent" LLMs—something that cannot be discovered through overall scores alone and requires deep diving into subcategories.
- **Proposal and validation of the "experiential knowledge" concept**: Pronunciation rules, morphemic decomposition, and other knowledge naturally acquired by humans through senses and usage are blind spots for LLMs, which can be compensated for via explicit textualization.
- **Thinking time as a difficulty indicator**: Using the thinking time of s1-32B to indirectly validate which subcategories are inherently more difficult for LLMs, aligning highly with performance gaps.
- **Official examination questions as the data source** ensure the authority of difficulty calibration and quality control.

## Limitations & Future Work

- Covers only Korean; transferring to other agglutinative languages (Japanese, Turkish, Finnish) requires additional linguistic adaptation.
- The scale of 1,524 questions can be further expanded, and some subcategories (e.g., 42 questions for romanization) have a relatively small sample size.
- Based on prescriptive grammar, not covering descriptive grammar (language variations in actual use).
- The influence of training data contamination cannot be completely ruled out, although the consistent trend across 27 models mitigates this risk.
- Experiential knowledge enhancement experiments were conducted on only two subcategories; enhancement schemes for other weak subcategories such as pragmatics remain to be explored.

## Related Work & Insights

- **Linguistic probing**: Work such as Conneau et al. (2018) and Hewitt & Manning (2019) evaluates linguistic information in hidden representations of models via probing methods, but is limited to English morphology/syntax and does not directly evaluate systematic grammatical competence.
- **Korean knowledge evaluation**: Son et al. (2024) focus on lexical knowledge, Kim et al. (2024a) offer a general Korean grammar evaluation but without clear classification, while Koo et al. (2022) and Yoon et al. (2023) focus only on specific error types. KoGEM comprehensively surpasses them in coverage and systematicity.
- **Cross-lingual NLP**: Test-time scaling and multilingual training have been proven effective for non-English languages (the Qwen series supports over 29 languages), but unique phenomena such as Korean phonology still require specialized evaluation.

## Rating

- Novelty: ⭐⭐⭐⭐ The first fine-grained Korean grammar benchmark based on theoretical linguistics, with a novel concept of "experiential knowledge"
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 27 models + human comparison + experiential knowledge enhancement + thinking time analysis + explanation quality evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear taxonomy, in-depth subcategory-by-subcategory analysis of all 16 subcategories, complete motivational chain
- Value: ⭐⭐⭐⭐ Crucial implications for non-English language competence evaluation; findings on the "lack of experiential knowledge" have theoretical value for understanding LLM capabilities

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Psycholinguistic Word Features: A New Approach for the Evaluation of LLMs Alignment with Humans](psycholinguistic_word_features_a_new_approach_for_the_evaluation_of_llms_alignme.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Leveraging In-Context Learning for Political Bias Testing of LLMs](leveraging_in-context_learning_for_political_bias_testing_of_llms.md)
- [\[ACL 2025\] How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](conceptual_knowledge_org.md)
- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)

</div>

<!-- RELATED:END -->
