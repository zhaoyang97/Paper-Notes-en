---
title: >-
  [Paper Note] An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual Corpora] This paper introduces HPLT v2, a large-scale multilingual dataset extracted from 4.5 PB of Internet Archive and Common Crawl data. It contains 8 trillion tokens of monolingual data covering 193 languages and 380 million parallel sentence pairs covering 51 languages, achieving significantly improved data quality through an enhanced data processing pipeline.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual Corpora"
  - "Web Crawling"
  - "Data Pipeline"
  - "Machine Translation"
  - "Language Model Pre-training"
date: 2026-05-08
content_hash: 46b1e0cfec2e7abf
---

# An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)

**Conference**: ACL 2025  
**arXiv**: [2503.10267](https://arxiv.org/abs/2503.10267)  
**Code**: [github](https://github.com/hplt-project/HPLT-textpipes)  
**Area**: Others (Multilingual Datasets/Corpus Construction)  
**Keywords**: Multilingual Corpora, Web Crawling, Data Pipeline, Machine Translation, Language Model Pre-training

## TL;DR

This paper introduces HPLT v2, a large-scale multilingual dataset extracted from 4.5 PB of Internet Archive and Common Crawl data. It contains 8 trillion tokens of monolingual data covering 193 languages and 380 million parallel sentence pairs covering 51 languages, achieving significantly improved data quality through an enhanced data processing pipeline.

## Background & Motivation

Training state-of-the-art large language models requires vast amounts of clean and diverse text data, but constructing suitable multilingual datasets remains a challenge. While English-centric LLMs have demonstrated impressive multilingual capabilities, the research community is increasingly focusing on the construction of explicitly multilingual corpora.

Existing multilingual datasets (such as OSCAR, CC-100, mC4, CulturaX, and MADLAD-400) primarily originate from Common Crawl. The uniqueness of HPLT v2 lies in its heavy utilization of web crawl data from the Internet Archive, making it a complementary source to these existing datasets. Furthermore, effective NLP research requires open training data so that results can be replicated and verified.

HPLT v2 is a direct successor to HPLT v1.2, offering improvements in several aspects: the data source scale is expanded by 2.5 times (to 4.5 PB), the text extraction tool is upgraded from warc2text to the more efficient Trafilatura, language identification is transitioned from CLD2 to a modified version of OpenLID (expanding coverage from 75 to 193 languages), and new features like robots.txt compliance labeling and PII annotations are introduced.

## Method

### Overall Architecture

The data construction pipeline is split into three main stages:

1. **HTML Extraction**: Extracting HTML and metadata from web crawl data in WARC format.
2. **Monolingual Text Processing**: Deduplication, cleaning, and quality filtering.
3. **Parallel Data Extraction**: Extracting bilingual aligned sentence pairs from the monolingual data.

### Key Designs

**Text Extraction Stage:**

- The total data source scale reaches 4.5 PB: 3.7 PB from the Internet Archive (2012–2020) and 0.8 PB from Common Crawl (2014–2022).
- The warc2text tool is used to extract HTML and metadata from WARC files.
- Trafilatura 1.8.0 is employed for de-boilerplateing (with settings `include_comments=False`, `include_tables=False`, `no_fallback=False`).
- A modified version of the OpenLID model is used for language identification (merging Arabic dialects, improving preprocessing).
- The data size is reduced from 4.5 PB to 62 TB after extraction.

**Monolingual Text Cleaning:**

- Documents with language label prediction confidence lower than 0.5 are filtered out.
- Crawl-level deduplication is performed using MinHash (240 hashes, Jaccard threshold of 0.8).
- Robots.txt rules are respected, and disallowed documents are removed.
- Document quality scores are calculated using the Web Docs Scorer (WDS), filtering out documents with scores below 5.
- Documents with fewer than 500 characters or an average of fewer than 5 words per paragraph are discarded (10 characters for Chinese, Japanese, and Korean).
- URLs from the UT1 blacklists (adult content) are filtered.
- PII (Personally Identifiable Information) metadata annotations are added.

**Parallel Data Extraction:**

- Extracted from the cleaned monolingual HPLT v2 using an improved version of the Bitextor pipeline.
- Loomchild (an SRX-based sentence segmenter) is used to support more languages.
- Bicleaner AI is leveraged for translation quality filtering (multilingual models can handle unseen language pairs).
- The final output includes 380 million sentence pairs matching 50 languages with English.
- Additionally, a multi-way parallel corpus MultiHPLT v2 (1275 language pairs, 16.7 billion sentence pairs) is created using English as a pivot.

### Loss & Training

The core contribution of this paper is the dataset construction rather than model training. However, several models were trained during the evaluation phase:
- **MLM**: Masked language models were trained on 52 languages using the LTG-BERT architecture.
- **Generative LM**: A decoder-only language model with 1.7B parameters was trained (on 100B tokens for English and 30B tokens for Norwegian).
- **Machine Translation**: Models were trained using the Transformer-base architecture and the Marian NMT toolkit.

## Key Experimental Results

### Main Results

**MLM Evaluation (52 languages):**

On four tasks—POS tagging, lemmatization, dependency parsing, and named entity recognition—models trained on HPLT v2 show significantly higher win rates compared to mBERT, XLM-R, and HPLT v1.2. Only in the lemmatization task do XLM-R and HPLT v1.2 provide competitive results (with differences of less than 1%).

**Generative LM Evaluation:**

- English: The model trained on HPLT v2 (cleaned) achieves downstream performance similar to FineWeb, significantly outperforming HPLT v1.2.
- Norwegian: HPLT v2 performs comparably to FineWeb, CulturaX, and mC4, all of which outperform HPLT v1.2. Performance plateaus after 16B tokens.

**Machine Translation Evaluation:**

| Comparison | BLEU (xx→en) | COMET (xx→en) | BLEU (en→xx) | COMET (en→xx) |
|------|-------------|---------------|-------------|---------------|
| HPLT v1.2 | 28.5 | 0.7943 | 24.4 | 0.7623 |
| **HPLT v2** | **32.7** | **0.8343** | **27.9** | **0.8137** |

HPLT v2 shows massive gains over v1.2. Combining it with OPUS data further improves BLEU and COMET, indicating that HPLT v2 contains content that does not overlap with existing OPUS corpora.

### Ablation Study

**Data Quality Analysis:**

- The deduplicated version is 21 TB, and the cleaned version is 15 TB.
- Contrast before and after cleaning: The proportion of unique paragraphs increased from 22.2% (v1.2) to 40.9% (v2).
- The proportion of long documents (>25 paragraphs) decreased from 90.8% to 23.2% (due to better de-boilerplateing).
- Paragraphs matching the document language improved from 58.6% to 81.5%.
- 80% of the parallel sentence pairs have translation likelihood scores between 0.8 and 1.0.

**Human Annotation (22 languages, 200 documents each):**

- Pornographic content and non-target language proportions are around 0-3% in most languages.
- Unnatural text accounts for about 10% (up to 30% in some languages).
- Common Crawl data after 2017 shows higher quality (the probability of unnatural text is about half of that from other sources).

### Key Findings

1. Although CC crawl data constitutes less than 20% of the input data, it contributes approximately 60% of the final text because CC is more focused on text content, whereas the Internet Archive contains substantial multimedia content.
2. Smaller language datasets tend to contain more Wikipedia and religious content.
3. European languages have the highest proportion of geographic top-level domains (ccTLDs), while African languages are dominated by generic top-level domains (gTLDs).
4. Punctuation in Chinese (and likely Korean and Japanese) was erroneously normalized to Latin equivalents, leading to performance degradation (to be fixed in the next release).
5. The Internet Archive provides more text for certain languages (e.g., Chinese, Persian) compared to Common Crawl.

## Highlights & Insights

1. **Breakthrough in Scale and Coverage**: 8 trillion tokens of monolingual data covering 193 languages, paired with 380 million parallel sentence pairs, make it one of the largest open multilingual datasets to date.

2. **Unique Value of the Internet Archive**: As one of the few projects utilizing IA data at scale, HPLT v2 complements major CC-based datasets, offering the research community a highly diverse source of data.

3. **Complete Reproducibility**: The entire data pipeline code is publicly available, and the data is released under the CC0 license, demonstrating a strong commitment to open science.

4. **Register Genre Labeling**: Using register classifiers in 16 languages, genre labeling was conducted for data in 100 languages, assisting users in making more informed data sampling decisions.

5. **Document-level Parallel Data**: DocHPLT is provided, containing document-level parallel data with sentence- and paragraph-alignment annotations, which is highly valuable for document-level translation research.

## Limitations & Future Work

1. The data is predominantly Indo-European (especially English), and parallel data is English-centric. Enhancing the quantity of underrepresented languages remains critical future work.
2. Language identification errors, boilerplate residues (particularly from Wikipedia and blog platforms), and other residual cleaning errors still exist.
3. There is only limited support for detecting and removing machine-generated content (such as machine translations and LLM outputs).
4. The punctuation normalization issue for Chinese, Japanese, and Korean needs to be addressed.
5. Evaluation only covers a subset of languages in HPLT v2, limited by available evaluation resources.
6. The computational cost is immense: approximately 4.4 million CPU hours and 106,000 GPU hours in total.

## Related Work & Insights

HPLT v2 continues the ongoing trend of expanding scale and coverage in multilingual corpora, progressing from early OSCAR and CC-100 to CulturaX and MADLAD-400. The pipeline design of this work (particularly the combination of Trafilatura for de-boilerplateing, OpenLID for language identification, and WDS for document quality scoring) serves as an excellent reference for other large-scale corpus construction initiatives.

The multi-layer filtering strategy during data cleaning (LID confidence + MinHash deduplication + robots.txt compliance + document quality scoring + length filtering + adult content filtering) provides a practical pipeline template. The quality evaluation methodology incorporating multilingual generative LM training is also highly informative.

## Rating

- **Novelty**: ★★★☆☆ — Mainly pipeline improvements, without methodological breakthroughs
- **Value**: ★★★★★ — Extremely high value to the multilingual NLP community
- **Experimental Thoroughness**: ★★★★★ — Three evaluation tracks (MLM, Generative LM, MT) along with human evaluation for 22 languages
- **Writing Quality**: ★★★★☆ — Clearly structured and highly detailed, though slightly verbose

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LangMark: A Multilingual Dataset for Automatic Post-Editing](langmark_a_multilingual_dataset_for_automatic_post-editing.md)
- [\[ACL 2025\] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)](towards_global_ai_inclusivity_a_large-scale_multilingual_terminology_dataset_gis.md)
- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline](llms_can_achieve_high-quality_simultaneous_machine_translation_as_efficiently_as.md)
- [\[NeurIPS 2025\] Zero-Shot Performance Prediction for Probabilistic Scaling Laws](../../NeurIPS2025/multilingual_mt/zero-shot_performance_prediction_for_probabilistic_scaling_laws.md)

</div>

<!-- RELATED:END -->
