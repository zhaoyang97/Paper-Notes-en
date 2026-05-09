---
title: >-
  [Paper Note] Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training
description: >-
  [ICLR 2026 Oral][LLM Pretraining][pre-training data] This work constructs Common Corpus — the largest legally licensed LLM pre-training dataset at approximately 2 trillion tokens — spanning 6 major collections (government, culture, science, code, web, and semantic), covering multiple languages including low-resource ones. All data originates from copyright-free or permissively licensed sources, accompanied by complete data provenance and a multi-stage filtering pipeline. The dataset has been adopted by industry leaders including Anthropic.
tags:
  - ICLR 2026 Oral
  - LLM Pretraining
  - pre-training data
  - ethical data
  - open data
  - multilingual
  - data curation
  - copyright
  - AI legislation
date: 2026-05-08
content_hash: 30af317004b785fb
---

# Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training

**Conference**: ICLR 2026 Oral
**arXiv**: [2506.01732](https://arxiv.org/abs/2506.01732)
**Code**: [HuggingFace](https://huggingface.co/datasets/PleIAs/common_corpus)
**Area**: LLM Pre-training Data / Data Engineering / AI Compliance
**Keywords**: pre-training data, ethical data, open data, multilingual, data curation, copyright, AI legislation

## TL;DR
This work constructs Common Corpus — the largest legally licensed LLM pre-training dataset at approximately 2 trillion tokens — spanning 6 major collections (government, culture, science, code, web, and semantic), covering multiple languages including low-resource ones. All data originates from copyright-free or permissively licensed sources, accompanied by complete data provenance and a multi-stage filtering pipeline. The dataset has been adopted by industry leaders including Anthropic.

## Background & Motivation
**Background**: LLM pre-training requires trillion-token-scale data (recent models such as DeepSeek v3 and Llama 4 use 14–36T tokens), yet mainstream datasets (The Pile, RefinedWeb, C4) extensively incorporate copyrighted content.

**Limitations of Prior Work**:
   - **Escalating legal risk**: The NYT lawsuit against OpenAI, EU AI Act legislation, and the fact that 45% of C4 content is now restricted from crawling under ToS.
   - **Damage to open science**: Key resources such as Books3, LAION, and the MATH benchmark have been successively taken down due to DMCA challenges and litigation — rendering prior research irreproducible.
   - **Insufficient compliant datasets**: C4C (228B tokens, English only), KL3M (1.2T tokens, US administrative text only), Common Pile (1T tokens, English only) — all limited in scale or language coverage.

**Key Challenge**: Training powerful LLMs requires massive data, yet compliant data falls far short in scale; compliant data for multilingual and low-resource languages is even more scarce.

**Core Idea**: Systematically collect and filter approximately 2T tokens from copyright-free or permissively licensed sources (government documents, public-domain literature, open-access scientific papers, open-source code, Creative Commons web content) to establish open scientific infrastructure for AI training data.

## Method

### Overall Architecture
Source identification (public-domain literature, government documents, scientific papers, licensed code, CC web) → License verification (per-document confirmation of copyright/license status) → Multi-stage data filtering (deduplication, language detection, quality scoring, toxicity filtering, PII removal) → Domain classification → Data provenance annotation → Release as 10,000 Parquet files

### Six Data Collections

| Collection | Documents | Tokens | Sources |
|------|-----------|--------|------|
| Open Government | 74.7M | 406.6B | Government documents, legal texts, parliamentary records from multiple countries |
| Open Culture | 93.2M | 886.0B | Public-domain books, historical documents, digitized library collections |
| Open Science | 19.2M | 281.2B | Open-access papers, preprints (arXiv, etc.) |
| Open Code | 202.8M | 283.2B | Permissively licensed open-source code (MIT/Apache/BSD, etc.) |
| Open Web | 96.2M | 73.2B | Creative Commons licensed web content |
| Open Semantic | 30.1M | 68.0B | Structured knowledge (Wikipedia, etc.) |
| **Total** | **517.0M** | **~2.0T** | |

### Key Designs

1. **License Compliance Verification**

    - Function: Per-document verification of copyright status and license type, ensuring all data is "usable without permission."
    - Source strategy: Follows the Open Source Initiative's definition of "maximally open" — not merely accessible, but unrestricted in use.
    - Public-domain confirmation: Based on national copyright law (e.g., pre-1928 US publications, post-70-year European copyright terms).
    - Code license filtering: Retains only attribution-free permissive licenses (MIT, Apache 2.0, BSD, etc.).

2. **Multilingual Coverage**

    - Function: Systematic collection of data not only in major languages but also in low-resource languages.
    - Language distribution: English 968.8B tokens (48.5%), French 275.4B, German 166.3B, …, covering 50+ languages.
    - 11% of documents are multilingual.
    - **All multilingual data consists of native text, not machine translation.**
    - Design Motivation: Existing compliant datasets are almost exclusively English, limiting compliant training of multilingual LLMs.

3. **Multi-Stage Data Filtering Pipeline**

    - Deduplication: Fuzzy and exact deduplication.
    - Language detection: Ensuring accurate language labeling.
    - Quality scoring: Filtering low-quality documents via perplexity and content quality models.
    - Toxicity filtering: Removing harmful content.
    - PII removal: Eliminating personally identifiable information.
    - OCR correction: Repairing OCR quality in historically digitized documents.

4. **Provenance Transparency**

    - Each document includes: source URL, license type, language tag, collection/domain classification, and additional metadata.
    - Supports complete AI auditing — from model back to training data and original sources.

## Key Experimental Results

### Dataset Scale Comparison

| Dataset | Scale | Language | Compliance |
|--------|------|------|--------|
| C4 | 156B | Primarily English | Partially restricted |
| RefinedWeb | 5T | English | Copyright disputed |
| C4C | 228B | English | Compliant |
| KL3M | 1.2T | English | Compliant (US administrative text) |
| Common Pile | 1T | English | Compliant |
| **Common Corpus** | **~2.0T** | **50+ languages** | **Fully compliant** |

Common Corpus is the only dataset that simultaneously satisfies "trillion-scale + multilingual + fully compliant."

### Data Diversity

| Dimension | Characteristics |
|------|------|
| Temporal span | Ancient to contemporary (historical public-domain documents to recent CC web pages) |
| Domain coverage | Law, science, literature, code, encyclopedias, community content |
| Linguistic diversity | 50+ languages, including African and Asian low-resource languages |

### Community Impact
- Adopted by Anthropic for model training.
- Utilized by multiple LLM training projects.
- Derivative multimodal datasets, classifiers, synthetic datasets, and benchmarks built upon Common Corpus.

### Key Findings
- **"Open data paradox"**: A large volume of public-domain and openly licensed content has low visibility online and is absent from mainstream pre-training data sources — active mining is required rather than reliance on Common Crawl alone.
- Documents digitized by government and cultural institutions represent an undervalued data source — high quality, copyright-free, and multilingual.
- Even under the most permissive licenses, gaps in diversity and quality of compliant data remain — requiring greater community effort.

## Highlights & Insights
- **A milestone for AI compliance infrastructure**: Against the backdrop of the EU AI Act and waves of copyright litigation, Common Corpus demonstrates that "compliant + large-scale" is achievable, providing a compliance data "commons" for the entire industry.
- **A model for open science practice**: Complete data provenance, license verification, and processing tools are all open-sourced — the entire pipeline is reusable by other projects.
- **Unique semantic coverage**: Substantially different from web-crawled data — encompassing historical documents, legal texts, and scientific papers — likely yielding a knowledge distribution distinct from web text.

## Limitations & Future Work
- **Scale ceiling**: ~2T tokens is far smaller than non-compliant datasets (RefinedWeb 5T+), which under scaling laws may be insufficient for training the largest models.
- **Quality gap**: Public-domain literature (OCR quality, archaic language styles) may exhibit distributional differences from modern web text; the impact on model performance has not been quantified.
- **Absence of training validation**: No performance comparison is reported between models trained on Common Corpus versus non-compliant data — a critical omission.
- **Code data**: 283B tokens is far less than dedicated code datasets such as The Stack.
- **Low-resource languages**: Although covered, data volumes may still be insufficient for training high-quality models.

## Related Work & Insights
- **vs. The Pile**: A pioneering pre-training dataset, but contains copyrighted content such as Books3 and has been partially taken down.
- **vs. RefinedWeb**: 5T scale but entirely sourced from Common Crawl with no license filtering.
- **vs. KL3M**: Compliant but restricted to US administrative text (narrow domain coverage).
- **Implications for the field**: Common Corpus demonstrates that compliant data collection is an ongoing infrastructure investment rather than a one-time effort. Community collaboration models (e.g., BLOOM's ROOTS) may represent a sustainable path forward.

## Rating
- Novelty: ⭐⭐⭐ Primarily a data engineering contribution; methodological innovation is limited.
- Experimental Thoroughness: ⭐⭐⭐ Dataset description is thorough, but a critical model training performance comparison is absent.
- Writing Quality: ⭐⭐⭐⭐ Data sources and pipeline are described clearly, following best-practice guidelines.
- Value: ⭐⭐⭐⭐⭐ ICLR Oral is well deserved — the work significantly advances AI industry compliance and represents an important contribution to open scientific infrastructure.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Pre-training LLM without Learning Rate Decay Enhances Supervised Fine-Tuning](pre-training_llm_without_learning_rate_decay_enhances_supervised_fine-tuning.md)
- [\[ICLR 2026\] Token-level Data Selection for Safe LLM Fine-tuning](token-level_data_selection_for_safe_llm_fine-tuning.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)

<!-- RELATED:END -->
