---
title: >-
  [Paper Note] CoAM: Corpus of All-Type Multiword Expressions
description: >-
  [ACL 2025][Multiword Expressions] Constructed CoAM (1.3K sentences), a high-quality, all-type Multiword Expression (MWE) identification dataset. Through a multi-step quality assurance pipeline, this work addresses the annotation inconsistency issues in existing datasets. It also demonstrates that fine-tuning Large Language Models (LLMs) significantly outperforms the previous SOTA method, MWEasWSD, on the MWE identification task.
tags:
  - "ACL 2025"
  - "Multiword Expressions"
  - "MWE Identification"
  - "Dataset Construction"
  - "Sequence Labeling"
  - "LLM Fine-tuning"
date: 2026-05-08
content_hash: 196623c9ef712ac0
---

# CoAM: Corpus of All-Type Multiword Expressions

**Conference**: ACL 2025  
**arXiv**: [2412.18151](https://arxiv.org/abs/2412.18151)  
**Code**: [https://github.com/Yusuke196/CAIGen](https://github.com/Yusuke196/CAIGen)  
**Area**: Others  
**Keywords**: Multiword Expressions, MWE Identification, Dataset Construction, Sequence Labeling, LLM Fine-tuning

## TL;DR

Constructed CoAM (1.3K sentences), a high-quality, all-type Multiword Expression (MWE) identification dataset. Through a multi-step quality assurance pipeline, this work addresses the annotation inconsistency issues in existing datasets. It also demonstrates that fine-tuning Large Language Models (LLMs) significantly outperforms the previous SOTA method, MWEasWSD, on the MWE identification task.

## Background & Motivation

Multiword Expressions (MWEs) refer to idiomatic multi-word sequences whose meanings cannot be directly derived from their individual components, such as "under the weather" and "real estate." MWE Identification (MWEI) is crucial for downstream tasks like machine translation and reading assistance—Briakou et al. (2024) demonstrated that identifying MWEs in the initial step of translation can improve translation quality.

However, existing MWEI datasets suffer from three major issues:

**Annotation Inconsistency**: Over 80% of false positives in the DiMSUM dataset are actually caused by annotation inconsistencies (the same MWE is annotated in one instance but omitted in another).

**Limited Types**: The high-quality PARSEME corpus only focuses on verbal MWEs.

**Restricted Scale**: ID10M provides only 200 evaluation sentences per language and fails to annotate discontinuous MWEs.

These limitations prevent reliable evaluation of MWEI systems. The motivation behind CoAM is to construct an annotation-consistent, all-type, and quality-assured dataset.

## Method

### Overall Architecture

The construction of CoAM adopts a four-stage pipeline: data selection and preprocessing $\rightarrow$ double annotation $\rightarrow$ human review $\rightarrow$ automatic consistency check. In addition, a novel annotation interface generator, CAIGen, was developed to support any form of MWE annotation (including discontinuous and overlapping MWEs).

### Key Designs

1. **Diverse Data Sources**: Four types of text sources were selected: news (EMM NewsBrief), news commentary (WMT23), TED talk transcripts, and Universal Dependencies web corpora. This covers both written text and spoken transcripts, ensuring standard English with minimal grammatical errors.

2. **Strict Definition of MWE**: An MWE must meet three conditions: (a) contain at least two words and always be realized by the same lexemes; (b) exhibit semantic, lexical, or syntactic idiomaticity (excluding transparent collocations such as "stuck at"); and (c) not be a proper noun. This definition is more precise than that of DiMSUM (which includes multi-word proper nouns).

3. **CAIGen Annotation Interface**: A checkbox-based annotation interface built on Google Sheets. Compared to brat and FLAT, CAIGen supports discontinuous/overlapping MWE annotation, requires no server management, and is intuitive and highly customizable.

4. **Double Annotation + Review Mechanism**: Each sentence is assigned to two annotators (one hired annotator + one author), ensuring at least one native English speaker. After annotation, all labels are reviewed by two native English-speaking authors to resolve discrepancies.

5. **Automatic Consistency Checking**: A rule-based pipeline is utilized to identify potentially missed MWEs, followed by human validation. In CoAM, 147 inconsistencies were found and fully corrected; a comparative experiment showed that 118 inconsistencies remained uncorrected in a similarly-sized subset of DiMSUM.

6. **MWE Type Annotation**: MWEs are categorized into 5 types: Noun, Verb, Mod/Conn (Modifier/Connective), Clause, and Other. They are automatically annotated based on dependency parsed structures, followed by manual correction of the test set (yielding an error rate of 7.8%).

### MWEI Approach

Two approaches are evaluated:

- **MWEasWSD (MaW)**: Uses the WordNet dictionary + a rule-based pipeline to identify candidate MWEs, followed by bi-encoder filtering. Based on BERT-base-uncased.
- **LLM Fine-tuning**: Conducts QLoRA fine-tuning on the Llama-3.1 and Qwen-2.5 series, utilizing a `tsv_to_tsv` input-output format.

## Key Experimental Results

### Main Results

| Method | F1 | Precision | Recall | Description |
|------|-----|-----------|--------|------|
| MaW Rule-only | 32.4 | 27.9 | 38.6 | Rule-only baseline |
| MaW Rule+DCA | 41.9 | 49.0 | 36.7 | Prev. SOTA |
| FT Llama-8B | 24.9 | 92.0 | 14.4 | High precision but extremely low recall |
| FT Qwen-7B | 48.1 | 60.9 | 39.7 | Improvements at medium scale |
| **FT Qwen-72B** | **57.8** | **63.8** | **52.8** | Best overall |

### Ablation Study

| Configuration | F1 (Qwen-72B) | Description |
|------|---------------|------|
| Fine-tuning | 57.8 | Main results |
| Few-shot (5-shot) | 12.9 | Significant drop |
| Zero-shot | 1.4 | Hardly able to complete the task |

### Recall Analysis by Type

| MWE Type | Qwen-72B Recall | MaW Rule+DCA Recall |
|----------|---------------|-------------------|
| Verb (139) | 59.7% | 45.3% |
| Noun (121) | 42.1% | 29.8% |
| Mod/Conn (111) | 57.7% | 36.6% |
| Clause (7) | 28.6% | 0.0% |

### Key Findings

- **Fine-tuning LLMs significantly outperforms rule-based methods**: Qwen-72B's F1 is 15.9 percentage points higher than the best MaW system.
- **Recall is low across all methods**: Even the best model (Qwen-72B) achieves only 52.8% recall, leaving nearly half of the MWEs undetected.
- **Discontinuous MWEs present a major challenge**: Recall for continuous MWEs is 57.3%, whereas it drops to 17.1% for discontinuous ones.
- **MWEs absent from WordNet are exceptionally difficult to identify**: Recall for MWEs listed in WordNet is 62.7%, compared to only 45.5% for those missing from it.
- **The average recall of human annotators is also only around 51%**, comparable to Qwen-72B, highlighting MWEI as an extremely challenging task in itself.

## Highlights & Insights

- CoAM is the first all-type MWE dataset enriched with MWE type labels, facilitating fine-grained error analysis.
- The consistency-checking method is both effective and versatile, easily adaptable to ensure quality control in other annotation tasks.
- The performance analysis of human annotators reveals a critical insight: top annotators (with a linguistics background) achieve an F1 score of 72%, while the average is only 56.5%. This highlights that MWE identification heavily relies on specialized training.
- This work is the first to apply LLM fine-tuning to MWEI and achieve SOTA results, proving that LLMs indeed acquire MWE knowledge during pre-training.

## Limitations & Future Work

- The dataset size is relatively small (1.3K sentences); while sufficient for evaluation, it limits training scalability.
- The corpus covers English only, making multilingual expansion a vital future direction.
- Discontinuous MWE identification remains a grand challenge (recall of only 17%), which warrants dedicated modeling efforts.
- BERT-level sequence labeling methods (such as BIO tagging + fine-tuning) was not explored in this work.
- The CAIGen interface has usability issues—annotators might occasionally forget to use separate rows for different MWEs.

## Related Work & Insights

- MWEasWSD's approach of leveraging WordNet as external knowledge is noteworthy, though it is constrained by dictionary coverage.
- The methodology for consistency checking (collecting all annotated MWEs first, then searching for missed instances throughout the corpus) can be generalized to other NLP annotation tasks.
- The effectiveness of LLM fine-tuning in sequence-labeling-like tasks is further validated.

## Rating

- **Novelty**: ⭐⭐⭐ — The core contribution lies in dataset construction. Although the technical methodology is not entirely new, the combination of all-type MWEs and consistency checking is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The multi-dimensional analysis (covering types, continuity, WordNet membership, and seen/unseen categories) is exhaustive and detailed.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured, with an elaborate description of the dataset construction process.
- **Value**: ⭐⭐⭐⭐ — It provides a reliable evaluation benchmark for the MWE identification community, and the consistency-checking methodology holds generalizable value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)
- [\[ACL 2025\] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)
- [\[ACL 2025\] All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)
- [\[NeurIPS 2025\] Ultrametric Cluster Hierarchies: I Want 'em All!](../../NeurIPS2025/others/ultrametric_cluster_hierarchies_i_want_em_all.md)
- [\[ACL 2025\] What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma](what_is_stigma_attributed_to_a_theory-grounded_expert-annotated_interview_corpus.md)

</div>

<!-- RELATED:END -->
