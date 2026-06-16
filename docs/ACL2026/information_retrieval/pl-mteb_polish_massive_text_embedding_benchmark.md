---
title: >-
  [Paper Note] PL-MTEB: Polish Massive Text Embedding Benchmark
description: >-
  [ACL 2026][Information Retrieval & RAG][Polish NLP] PL-MTEB constructs a 30-task evaluation suite for Polish text embeddings covering classification, clustering, pair classification, retrieval, and semantic similarity. A systematic evaluation of 30 Polish and multilingual embedding models shows that while large models lead overall, conclusions are significantly influenc
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - Polish NLP
  - Text Embedding
  - MTEB
  - Information Retrieval
date: 2026-05-08
content_hash: de186aee14588c52
---
# PL-MTEB: Polish Massive Text Embedding Benchmark

**Conference**: ACL2026 Findings  
**arXiv**: [2405.10138](https://arxiv.org/abs/2405.10138)  
**Code**: https://github.com/rafalposwiata/pl-mteb  
**Area**: Information Retrieval/RAG  
**Keywords**: Polish NLP, Text Embedding, MTEB, Information Retrieval, Benchmark Evaluation

## TL;DR
PL-MTEB constructs a 30-task evaluation suite for Polish text embeddings covering classification, clustering, pair classification, retrieval, and semantic similarity. A systematic evaluation of 30 Polish and multilingual embedding models shows that while large models lead overall, conclusions are significantly influenced by task types, training data leakage, and model scale.

## Background & Motivation
**Background**: Text embeddings are fundamental components for retrieval, clustering, classification, Q&A, and semantic matching systems. MTEB provides a unified evaluation framework for English and some multilingual tasks. Recently, language-specific extensions have emerged for Chinese, French, Persian, Dutch, Russian, Vietnamese, Turkish, Arabic, and African languages.

**Limitations of Prior Work**: Polish models have long lacked an embedding benchmark with sufficient breadth in task types. Existing Polish evaluations usually cover only a single or few tasks, such as sentiment classification, semantic correlation, or BEIR-PL retrieval, failing to indicate whether a model is stable across different application scenarios.

**Key Challenge**: Multilingual models may be usable for Polish, but their performance is heavily influenced by training corpora, task types, and model scale. Without a unified, public, diverse, and quality-controlled benchmark, it is difficult for users to select models for practical systems or to fairly compare Polish-specific models with general multilingual models.

**Goal**: The authors aim to build a Polish version of MTEB by reusing existing public Polish tasks and filling gaps in missing task types, especially clustering. Simultaneously, they collect results from 30 public embedding models to analyze the impact of task types, model size, and zero-shot coverage on evaluation conclusions.

**Key Insight**: Instead of merely translating English tasks, this paper integrates existing Polish data, BEIR-PL retrieval tasks, KLEJ/LEPISZCZE-related data, and newly constructed PLSC and Wikinews-PL clustering data into the MTEB framework, while releasing the code, data, and leaderboard.

**Core Idea**: Build a 30-task, multi-type, reproducible embedding benchmark for Polish using the unified evaluation interface of MTEB. Additionally, similarity with training data is marked as a zero-shot column to remind readers to distinguish between true generalization and gains from training set similarity.

## Method
The methodology of PL-MTEB focuses on benchmark construction and evaluation protocols rather than proposing a new embedding model. The authors performed three tasks: defining the task set, cleaning and constructing new data, and evaluating models with analysis by task type/model scale.

### Overall Architecture
The benchmark includes 30 sub-tasks across 5 categories. Classification tasks evaluate the linear separability of embeddings using few-shot logistic regression; clustering tasks use mini-batch k-means and v-measure; pair classification uses average precision under similarity thresholds; retrieval tasks use nDCG@10; STS uses Spearman correlation under cosine similarity.

Task sources are divided into three groups. The first group consists of Polish sub-tasks extractable from existing MTEB or multilingual MTEB, mainly comprising BEIR-PL retrieval tasks. The second group consists of tasks newly added by the authors based on existing Polish public data, mostly from human-annotated resources like KLEJ. The third group consists of two newly constructed datasets, PLSC and Wikinews-PL, generating four clustering tasks.

Evaluated models include 30 public dense embedding models, covering small, base, large, and 1B+ parameter models. These include multilingual E5, SBERT, Arctic-Embed, Qwen3-Embedding, BGE-Multilingual-Gemma2, as well as Polish-related models like MMLW, Stella-PL, and Silver Retriever. Each model is run as recommended by developers, and the proportion of tasks where it is considered zero-shot is recorded.

### Key Designs

**1. Unified protocol for five task categories: Integrating classification, clustering, pair classification, retrieval, and STS into a single evaluation interface.**

Embedding models often perform strongly on one type of task but fail on others; relying on a single average score masks such imbalances. PL-MTEB thus equips each category with a lightweight, reproducible evaluator: classification uses 8-shot logistic regression per class (averaged over 10 runs) to check linear separability; clustering uses mini-batch k-means with $k$ equal to the number of labels (averaged over 10 runs) scored by v-measure; retrieval focuses on nDCG@10; STS uses Spearman correlation with cosine similarity; pair classification uses average precision with cosine similarity thresholds. All protocols share the same embedding output, allowing for horizontal comparison—readers can view both the 30-task mean Avg(30) and the Avg(by type) to select models based on specific needs like retrieval or clustering.

**2. PLSC and Wikinews-PL clustering enhancement: Filling the missing clustering dimension in Polish with two new datasets.**

Most existing Polish evaluations revolve around classification, retrieval, and STS, leaving clustering tasks almost empty. Clustering is a strong probe for checking whether the "representation space is stable" as it relies on global semantic structure and is least affected by supervised classifiers or retrieval training data. The authors created two new datasets: PLSC, derived from Polish Library of Science metadata with ~160K records and hierarchical labels for 8 scientific fields and 44 disciplines; and Wikinews-PL, categorized into politics, economy, disasters, culture, science, law, sports, society, and technology. Both generate S2S (sentence-to-sentence/title level) and P2P (paragraph-to-paragraph) clustering tasks, truncated to 2,048 entries per task for efficiency.

**3. Data quality and zero-shot labeling: Cleaning for leakage and exposing training data similarity.**

Embedding benchmarks are prone to contamination, especially retrieval and STS tasks that might have appeared in a model's training corpora. The authors perform hard cleaning: removing empty text and samples with fewer than 3 words, verifying labels/scores, removing near-duplicates with conflicting labels or score differences > 0.5, and deduplicating at the split level to prevent test-train leakage. Furthermore, they include a zero-shot column in the evaluation table, indicating the percentage of tasks where the model's training data did not contain similar tasks. This provides a defensive layer for readers: a high retrieval score in a model with only 80% zero-shot proportion should be scrutinized for training data overlap rather than assumed as pure generalization.

### Loss & Training
PL-MTEB does not train new models and has no unified training loss. Evaluation only involves training lightweight downstream evaluators: logistic regression for classification and k-means for clustering. Other tasks directly use embedding similarity or retrieval ranking. All models are loaded according to their original release; evaluation code is based on the MTEB framework, with results and data publicly available on GitHub and Hugging Face.

## Key Experimental Results

### Main Results
The benchmark contains 30 tasks: 7 classification, 5 clustering, 4 pair classification, 11 retrieval, and 3 STS. Retrieval tasks include ArguAna-PL, FiQA-PL, MSMARCO-PLHardNeg, and others.

| Task Type | Task Count | Main Metric | Repr. Task / Data Source | Design Points |
|------|------|------|------|------|
| Classification | 7 | Accuracy | CBD, PolEmo2.0, AllegroReviews, PAC, MassiveIntent/Scenario | 8-shot logistic regression per class, 10 repeats |
| Clustering | 5 | V-measure | EightTags, PLSC, Wikinews-PL | mini-batch k-means, hierarchical tasks use layer average |
| Pair Classification | 4 | Cosine AP | SICK-E-PL, CDSC-E, PSC, PPC | Evaluate similarity separability of sentence pairs |
| Retrieval | 11 | nDCG@10 | BEIR-PL series | Mostly query-corpus retrieval; some HardNeg limit corpus size |
| STS | 3 | Cosine Spearman | SICK-R-PL, CDSC-R, STSBenchmarkMultilingual | Measure semantic similarity ranking correlation |

| Model | Params | Zero-shot % | Classification | Clustering | PairClass | Retrieval | STS | Avg(30) | Avg(by type) |
|------|--------|----------------|----------------|------------|-----------|-----------|-----|---------|--------------|
| mmlw-roberta-base | 124M | 96 | 62.53 | 48.00 | 88.16 | 53.60 | 85.20 | 62.52 | 67.50 |
| multilingual-e5-base | 278M | 90 | 55.36 | 44.10 | 82.08 | 47.63 | 79.13 | 56.59 | 61.66 |
| mmlw-retrieval-roberta-large | 435M | 93 | 63.90 | 45.18 | 88.48 | 57.23 | 84.71 | 63.69 | 67.90 |
| Qwen3-Embedding-0.6B | 596M | 90 | 69.66 | 56.65 | 81.31 | 48.59 | 78.45 | 62.20 | 66.93 |
| stella-pl | 1.5B | 80 | 66.94 | 38.08 | 89.20 | 60.82 | 86.87 | 64.85 | 68.38 |
| stella-pl-retrieval-8k | 1.5B | 80 | 68.14 | 35.42 | 89.56 | 61.59 | 86.56 | 64.98 | 68.25 |
| Qwen3-Embedding-4B | 4.0B | 90 | 79.30 | 59.90 | 86.68 | 56.65 | 85.55 | 69.37 | 73.62 |
| Qwen3-Embedding-8B | 7.6B | 90 | 79.87 | 58.64 | 87.61 | 59.21 | 86.72 | 70.47 | 74.41 |
| BGE-Multilingual-Gemma2 | 9.2B | 83 | 77.77 | 58.15 | 89.75 | 58.93 | 83.97 | 69.81 | 73.71 |

### Ablation Study
As a benchmark paper, there is no traditional module ablation; the analysis focuses on task types, model scale, and training data similarity.

| Analysis Dimension | Observation | Insight |
|------|------|------|
| Task Type Winners | Qwen3-8B best in classification, Qwen3-4B best in clustering, BGE-Gemma2 best in PairClass, stella-pl-retrieval-8k best in retrieval, stella-pl best in STS | No model dominates all tasks; average scores cannot replace task-level selection |
| Model Scale | Models >1B lead overall; Qwen3-8B leads with Avg(30)=70.47 | LLMs have clear advantages, but not all tasks scale monotonically |
| Small Models | mmlw-roberta-base leads <150M group significantly, Avg(30)=62.52 | Polish-specifically trained small models are highly competitive in resource-constrained scenarios |
| Base Model Group | snowflake-arctic-embed-m-v2.0 Avg(30)=57.06, multilingual-e5-base Avg(by type)=61.66 | No clear dominator among mid-sized multilingual models; depends on task type |
| Retrieval Tasks | stella-pl series strongest, but zero-shot ratio is only 80% | High retrieval scores might be influenced by similar training data; must interpret with zero-shot column |
| P2P vs S2S Clustering | P2P usually outperforms S2S in PLSC/Wikinews | Longer texts contain more clustering info; title-level embeddings are more challenging |

### Key Findings
- Qwen3-Embedding-8B is the strongest overall model (Avg(30)=70.47, Avg(by type)=74.41), but it primarily excels in classification and is not first in every category.
- BGE-Multilingual-Gemma2 has the highest average in pair classification, suggesting ultra-large multilingual models remain strong in semantic matching.
- The Polish-specific stella-pl-retrieval-8k performs best in retrieval (nDCG@10 avg 61.59), though its training data has higher similarity to the retrieval tasks.
- mmlw-roberta-base, with only 124M parameters, achieved Avg(30)=62.52, outperforming many base/large multilingual models, proving value in language-specific distillation.
- The benchmark's primary contribution is extending Polish embedding evaluation to 5 types and 30 tasks while incorporating data quality and zero-shot perspectives.

## Highlights & Insights
- PL-MTEB is highly suitable as a tool for "practical embedding model selection" rather than just a leaderboard, as it reports both task-type and overall averages.
- New PLSC and Wikinews-PL clustering tasks are critical. Many models perform well in supervised retrieval but fail to show stable semantic space structures in clustering tests.
- Inclusion of the zero-shot column is an excellent evaluation practice. Multilingual model training data is complex; explicit labeling of similarity helps reduce misinterpretation of scores.
- Results remind us that ultra-large multilingual models and small language-specific models are not simple replacements. If resources permit, Qwen3/BGE classes are strong; if deployment is restricted, MMLW/Stella-PL have significant value.

## Limitations & Future Work
- Although covering 30 tasks, many retrieval tasks are derived from machine-translated BEIR-PL, which might introduce translation artifacts or bias from original English structures.
- Zero-shot judgment depends on available training data documentation. Since many models are not fully open-source, contamination can only be approximated.
- Classification uses 8-shot logistic regression, which is suitable for evaluating separability but may not represent full-data fine-tuning performance.
- The benchmark primarily evaluates dense embeddings and does not deeply compare combinations of sparse retrieval, hybrid retrieval, rerankers, or instruction embeddings.

## Related Work & Insights
- **vs Original MTEB**: MTEB provides the framework, but English and few multilingual tasks dominate; PL-MTEB grounds task and quality control in Polish for local applications.
- **vs BEIR-PL / PIRB**: These focus mainly on retrieval; PL-MTEB adds classification, clustering, pair classification, and STS for general embedding evaluation.
- **vs KLEJ / LEPISZCZE**: These are more oriented toward NLU/classification; PL-MTEB focuses on representation quality without task-specific deep models.
- **vs MMTEB**: MMTEB is a large-scale community expansion; PL-MTEB is a refined subset for Polish with added data curation and local model analysis.

## Rating
- Novelty: ⭐⭐⭐☆☆ Algorithmic innovation is limited, but the systematic construction of a language-specific benchmark is high-value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 30 tasks, 30 models, and solid analysis across task types and model scales.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and informative tables.
- Value: ⭐⭐⭐⭐⭐ Highly practical for Polish NLP and multilingual embedding selection, providing a reusable paradigm for other languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](../../ICLR2026/information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ICLR 2026\] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)
- [\[ACL 2026\] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)
- [\[ACL 2025\] A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens](../../ACL2025/information_retrieval/a_text_is_worth_several_tokens_text_embedding_from_llms_secretly_aligns_well_wit.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
