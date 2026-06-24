---
title: >-
  [Paper Note] WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models
description: >-
  [ACL 2025][LLM Evaluation][benchmark] Proposes WXImpactBench, the first LLM evaluation benchmark for extreme weather impact understanding. It features a four-stage data construction pipeline and two evaluation tasks (multi-label classification and ranking-based question answering) to systematically evaluate the capabilities of multiple LLMs in the domain of climate adaptation.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "benchmark"
  - "weather impact"
  - "multi-label classification"
  - "question answering"
date: 2026-05-08
content_hash: f0943065322b5d4e
---

# WXImpactBench: A Disruptive Weather Impact Understanding Benchmark for Evaluating Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2505.20249](https://arxiv.org/abs/2505.20249)  
**Code**: [GitHub](https://github.com/Michaelyya/WXImpactBench)  
**Area**: LLM Evaluation  
**Keywords**: benchmark, weather impact, LLM evaluation, multi-label classification, question answering  

## TL;DR

Proposes WXImpactBench, the first LLM evaluation benchmark for extreme weather impact understanding. It features a four-stage data construction pipeline and two evaluation tasks (multi-label classification and ranking-based question answering) to systematically evaluate the capabilities of multiple LLMs in the domain of climate adaptation.

## Background & Motivation

- **Problem Definition**: Climate change adaptation requires understanding the societal impacts of extreme weather, yet the effectiveness of LLMs in this domain remains unexplored.
- **Limitations of Prior Work**: Existing climate-related data mostly originates from structured meteorological records, which suffer from a lack of daily impact narratives. Furthermore, these data may have already been included in LLMs' pre-training corpora, leading to evaluation bias.
- **Key Challenge**: Climate terminology in historical newspapers is often polysemous (e.g., "blizzard" can refer to both a snowstorm and a sports team), and text noise from OCR digitization severely degrades downstream tasks.
- **Ours**: Builds a high-quality extreme weather impact dataset from historical newspapers and designs the WXImpactBench benchmark to evaluate LLMs using multi-label classification and ranking-based QA tasks.

## Method

### Overall Architecture

Four-stage data construction pipeline + two-task evaluation framework:
1. **Corpus Collection**: Digitized newspaper texts from two historical periods were acquired from proprietary archives.
2. **Post-OCR Error Correction**: GPT-4o was utilized for OCR text error correction, achieving BLEU/ROUGE scores highly consistent with human annotations.
3. **Topic-Aware Article Selection**: Filtered from 53,521 articles through LDA topic modeling, yielding 350 high-quality samples after manual review by three domain experts.
4. **Human Label Annotation**: Six categories of vulnerability-related impacts were defined (infrastructure, political, financial, ecological, agricultural, and human health), with multi-label binary annotations conducted by three annotators.

### Key Designs

- **Multi-Label Classification Task**: Evaluates LLMs' ability to distinguish among six categories of weather impacts, utilizing row-wise accuracy as a strict metric (requiring correct classification across all six labels simultaneously).
- **Ranking-Based QA Task**: Generates pseudo-questions for each article and constructs a candidate article pool of 100 articles (1 positive + 99 negatives) to evaluate the retrieval and ranking capabilities of LLMs, laying the foundation for climate RAG system development.
- **Hybrid Context Version**: Segmented long texts into passages of approximately 250 tokens and annotated them independently, creating 1,386 samples to evaluate the impact of long-context understanding.

### Loss & Evaluation Metrics

- Classification Task: $\mathcal{L}(\hat{\mathcal{Y}}_t, \mathcal{Y}_t) = -\sum_{i=1}^{6} y_i \log \hat{y}_i$
- Classification Metrics: F1-score, Accuracy, Row-wise Accuracy
- Ranking Task Metrics: Hit@1, nDCG@5, Recall@5, MRR

## Experiments

### Main Results

| Model | Infrastructure | Political | Financial | Ecological | Agricultural | Human Health | Average |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GPT-4o | **80.94** | **58.46** | **65.82** | 46.81 | 70.33 | 73.23 | **65.93** |
| DeepSeek-V3-671B | 81.87 | 44.44 | 60.91 | 36.00 | 61.74 | 65.20 | 58.03 |
| Mistral-24B-IT | 79.12 | 47.18 | 59.64 | **44.90** | **67.74** | 66.88 | 60.91 |
| Gemma-2-9b-IT | 77.42 | 43.33 | 54.60 | 42.16 | 55.60 | 61.82 | 55.82 |

> Zero-shot F1-score (hybrid context version), with ↑ indicating the improvement over the long context version.

### Ablation Study

| Setup | Impact |
|------|------|
| Long Context vs Hybrid Context | The hybrid context version achieves an average improvement of 2.38 F1, indicating that LLMs perform better on shorter texts. |
| Zero-shot vs One-shot | One-shot generally improves performance, though some models (e.g., Mixtral) exhibit instability. |
| Historical vs Modern Text | Performance on modern text is generally better, as historical narrative styles increase the difficulty of understanding. |

### Key Findings

- GPT-4o performs the best in most categories, yet all models show weak performance in identifying ecological and political impacts.
- Model size is not the decisive factor: DeepSeek-V3 (671B) underperforms Mistral-24B in certain categories.
- The hybrid context version generally outperforms the long context version, indicating that current LLMs still have room for improvement in long text understanding.
- Row-wise accuracy is extremely low (the highest is only ~30%), demonstrating that accurately classifying all six categories of impact simultaneously is highly challenging.

## Highlights & Insights

- The first LLM evaluation benchmark for extreme weather impact understanding, filling a critical gap in climate NLP.
- The four-stage data construction pipeline is elegantly designed, combining OCR error correction, LDA topic modeling, and domain-expert annotation.
- The evaluation tasks are designed to encompass both classification and retrieval application scenarios, establishing a foundation for the development of climate RAG systems.

## Limitations & Future Work

- The relatively small dataset size (350 articles) may limit the statistical significance of the evaluation.
- Only English newspapers are covered, lacking cross-lingual evaluation.
- The pseudo-questions for the ranking-based QA task are generated by an LLM, which might introduce bias.
- The taxonomy of six impact categories may not cover all types of weather impacts.

## Related Work & Insights

- **Climate Text Processing**: Mallick et al. (2024), Xie et al. (2024) focus on extreme weather event extraction.
- **Climate Benchmarks**: CLLMate (Li et al., 2024) focuses on weather forecasting rather than impact understanding.
- **OCR Correction**: Neural OCR correction models by Drobac & Lindén (2020).
- **Disaster NLP**: Disaster text classification by Purohit et al. (2013), Imran et al. (2016).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Batayan: A Filipino NLP Benchmark for Evaluating Large Language Models](batayan_a_filipino_nlp_benchmark_for_evaluating_large_language_models.md)
- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ACL 2025\] CodeMEnv: Benchmarking Large Language Models on Code Migration](codemenv_benchmarking_large_language_models_on_code_migration.md)
- [\[ACL 2025\] Com2: A Causal-Guided Benchmark for Exploring Complex Commonsense Reasoning in Large Language Models](com2_causal_commonsense.md)

</div>

<!-- RELATED:END -->
