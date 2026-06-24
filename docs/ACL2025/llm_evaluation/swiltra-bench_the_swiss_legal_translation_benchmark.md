---
title: >-
  [Paper Note] SwiLTra-Bench: The Swiss Legal Translation Benchmark
description: >-
  [ACL 2025][LLM Evaluation][Legal Translation] Constructs SwiLTra-Bench, a large-scale multilingual benchmark featuring over 180,000 aligned Swiss legal translation pairs (covering laws, headnotes, and press releases across German, French, Italian, Romansh, and English). It systematically evaluates the performance of frontier LLMs and fine-tuned open-source SLMs in legal translation, and proposes the SwiLTra-Judge automatic evaluation methodology.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Legal Translation"
  - "Multilingual Benchmark"
  - "Swiss Law"
  - "LLM Translation Evaluation"
  - "Fine-Tuning"
date: 2026-05-08
content_hash: cc4b28199322a94d
---

# SwiLTra-Bench: The Swiss Legal Translation Benchmark

**Conference**: ACL 2025  
**arXiv**: [2503.01372](https://arxiv.org/abs/2503.01372)  
**Code**: Yes (The paper provides links to Datasets and Code)  
**Area**: LLM Evaluation  
**Keywords**: Legal Translation, Multilingual Benchmark, Swiss Law, LLM Translation Evaluation, Fine-Tuning

## TL;DR

Constructs SwiLTra-Bench, a large-scale multilingual benchmark featuring over 180,000 aligned Swiss legal translation pairs (covering laws, headnotes, and press releases across German, French, Italian, Romansh, and English). It systematically evaluates the performance of frontier LLMs and fine-tuned open-source SLMs in legal translation, and proposes the SwiLTra-Judge automatic evaluation methodology.

## Background & Motivation

Switzerland has four official languages (German, French, Italian, and Romansh), requiring legal documents to be translated into multiple languages. Traditional legal translation relies heavily on professionals proficient in both law and translation, creating a severe translation bottleneck and hindering equitable access to justice.

Existing neural machine translation (NMT) systems exhibit limited performance on legal texts due to:
1. Unique discourse structures and professional terminology in legal language
2. Lack of high-quality multilingual parallel legal corpora
3. Particular difficulty in translation coverage for low-resource languages (e.g., Romansh)

Although preliminary explorations exist, the actual performance of current LLMs on large-scale Swiss legal translation benchmarks, under both zero-shot and fine-tuned settings, remains unclear.

## Method

### Overall Architecture

This work presents three core contributions:
1. **SwiLTra-Bench Dataset**: A large-scale multilingual legal translation benchmark
2. **Comprehensive Model Evaluation**: The first large-scale comparison between frontier LLMs and fine-tuned SLMs
3. **SwiLTra-Judge**: An LLM evaluation methodology aligned with human expert assessments

### Key Designs

1. **Three Sub-Datasets of Legal Texts**: 
    - CH-Law-Trans (Swiss Law Translation): Contains law-, article-, and paragraph-level translations, covering 5 languages (German, French, Italian, Romansh, and English), with over 150,000 paragraph-level training pairs.
    - CH-Headnote-Trans (Case Law Headnote Translation): Sourced from iconic precedents of the Federal Supreme Court of Switzerland, covering three levels (BGE/Regest/Text), with over 26,000 training pairs.
    - CH-Press-Trans (Press Release Translation): National court press releases, featuring 867 training pairs.
    - All datasets are aligned with high quality leveraging official governmental HTML structures rather than noisy automatic sentence alignment.

2. **Comprehensive Evaluation of Five Model Categories**: Systematically compares translation-specific models (MADLAD-400, Tower-Instruct), frontier models (Claude-3.5-Sonnet, GPT-4o, Gemini-1.5-Pro, etc.), reasoning models (o1), open-source SLMs, and fine-tuned models, covering both zero-shot and fine-tuned settings. Evaluation metrics include vocabulary-level metrics (BLEU, ChrF, METEOR) and model-based metrics (BERTScore, BLEURT, XCOMET, GEMBA-MQM).

3. **SwiLTra-Judge Evaluation System**: Designs a specialized LLM-based evaluation system to automatically evaluate translation quality. Validated via comparison with human expert annotations, SwiLTra-Judge achieves the highest alignment with expert evaluations, offering a reliable automatic evaluation framework for legal translation.

### Loss & Training

Fine-tuning Setup:
- 4-bit quantization + 8-bit AdamW optimizer
- Rank Stabilized LoRA (rank=16, alpha=16)
- Sequence length 512 (covering over 99% of training data)
- Packing technology with a batch size of 128
- Linear learning rate schedule, 1000 warmup steps, learning rate 1e-4
- Early stopping (patience=3), with most models reaching minimum validation loss in 1 epoch
- Fine-tuned 13 open-source models (including Gemma, Llama, Phi, and Qwen series)

## Key Experimental Results

### Main Results

Comparison of translation models (average scores, higher is better):

| Model | Size | GEMBA-MQM | XCOMET | METEOR | ChrF |
|------|------|-----------|--------|--------|------|
| Google Translate | N/A | 53.20 | 64.61 | 41.15 | 47.81 |
| MADLAD-400-7B | 7B | 62.66 | 87.40 | 43.70 | 51.67 |
| Tower-Instruct-13B | 13B | 57.38 | 75.94 | 43.95 | 48.46 |
| Claude-3.5-Sonnet | large | 80.66 | 90.70 | 56.71 | 65.87 |
| GPT-4o | large | 80.27 | 80.96 | 55.56 | 63.27 |
| Gemini-1.5-Pro | large | 81.88 | 87.13 | 57.92 | 70.07 |
| o1 | large | **85.81** | **91.35** | **58.91** | **70.11** |
| GPT-4o-mini | small | 82.59 | 87.90 | 54.03 | 59.86 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Translation Models vs Frontier Models | MADLAD-400 outperforms GPT-4o on XCOMET | Translation-specific models exhibit competitive performance on legal text |
| Zero-shot vs Fine-tuned SLMs | Fine-tuning significantly improves quality but still trails frontier zero-shot models | The performance gap narrows post-fine-tuning but is not fully bridged |
| Laws vs Case Law Headnotes | Translation models perform strongly on laws but weakly on headnotes | Text type significantly impacts model performance |
| Cross-lingual Performance | Translation quality is relatively uniform across different languages | Multilingual coverage is well-balanced |

### Key Findings

- **The o1 reasoning model achieves the highest overall score** (GEMBA-MQM of 85.81), but its cost is vastly superior to Claude-3.5-Sonnet, which presents the optimal cost-efficiency.
- **MADLAD-400 is surprisingly strong at legal translation**, outperforming GPT-4o in XCOMET (87.40 vs 80.96).
- **Fine-tuning open-source SLMs significantly enhances quality** but still lags behind the best zero-shot frontier models.
- **Google Translate performs surprisingly poorly** (GEMBA-MQM score of only 53.20).
- **Human experts exhibit higher agreement on law translations than on case law headnotes**, reflecting the higher degree of standardization in statutory texts.
- The cost-effectiveness of smaller frontier models, such as Claude-3.5-Haiku, highlights potential practical value.

## Highlights & Insights

- High dataset quality: Leverages official governmental HTML structures for alignment, avoiding traditional noisy automatic sentence alignment.
- Comprehensive evaluation: Features a complete evaluation matrix of five model categories $\times$ seven metrics $\times$ three text types.
- High practical value: Directly addresses the Swiss government's legal translation needs and advances equitable access to justice.
- Special significance for the coverage of low-resource languages (e.g., Romansh).

## Limitations & Future Work

- Limited coverage of Romansh and English in the legal dataset (roughly only 20,000 and 30,000 paragraph-level samples, respectively).
- Fine-tuning utilizes only LoRA and limits the sequence length to 512, which may be insufficient for lengthy legal texts.
- Lack of evaluation regarding the end-to-end utility of translation systems in real-world legal workflows.
- The SwiLTra-Judge evaluation itself relies on GPT-4o (GEMBA-MQM), carrying a risk of circular dependency.

## Related Work & Insights

- Translation benchmarks in the legal NLP domain are scarce; this work fills an important gap in the Swiss legal context.
- The robust performance of MADLAD-400 as a translation-specific model highlights the value of domain-specific fine-tuning.
- Insights: For multilingual legal domains, the zero-shot capabilities of frontier LLMs are already highly functional, although verification by legal experts remains necessary.

## Rating

- Novelty: ⭐⭐⭐ — The primary contributions are the dataset and evaluation, with no significant innovation in methodology.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extremely broad model coverage and comprehensive evaluation metrics, including validation by human experts.
- Writing Quality: ⭐⭐⭐⭐ — Clearly structured with detailed statistical data.
- Value: ⭐⭐⭐⭐ — The dataset and evaluation framework offer direct practical utility for legal translation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] KITAB-Bench: A Comprehensive Multi-Domain Benchmark for Arabic OCR and Document Understanding](kitab-bench_a_comprehensive_multi-domain_benchmark_for_arabic_ocr_and_document_u.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](../../NeurIPS2025/llm_evaluation/parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[ACL 2026\] Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics](../../ACL2026/llm_evaluation/evaluating_legal_reasoning_traces_with_legal_issue_tree_rubrics.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages](tumlu_a_unified_and_native_language_understanding_benchmark_for_turkic_languages.md)

</div>

<!-- RELATED:END -->
