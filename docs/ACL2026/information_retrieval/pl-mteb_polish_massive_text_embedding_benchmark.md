---
title: >-
  [Paper Note] PL-MTEB: Polish Massive Text Embedding Benchmark
description: >-
  [ACL2026][Information Retrieval & RAG][Polish NLP] PL-MTEB constructs an evaluation suite for Polish text embeddings consisting of 30 tasks across classification, clustering, pair classification, retrieval…
tags:
  - "ACL2026"
  - "Information Retrieval & RAG"
  - "Polish NLP"
  - "Text Embedding"
  - "MTEB"
  - "Information Retrieval"
  - "Benchmark Evaluation"
date: 2026-05-08
content_hash: 31e4db45f6d109b2
---

# PL-MTEB: Polish Massive Text Embedding Benchmark

**Conference**: ACL2026  
**arXiv**: [2405.10138](https://arxiv.org/abs/2405.10138)  
**Code**: https://github.com/rafalposwiata/pl-mteb  
**Area**: Information Retrieval/RAG  
**Keywords**: Polish NLP, Text Embedding, MTEB, Information Retrieval, Benchmark Evaluation

## TL;DR
PL-MTEB constructs an evaluation suite for Polish text embeddings consisting of 30 tasks across classification, clustering, pair classification, retrieval, and semantic similarity. Systematic evaluation of 30 Polish and multilingual embedding models indicates that while Large Language Models (LLMs) lead overall, conclusions are significantly influenced by task types, training data leakage, and model scale.

## Background & Motivation
**Background**: Text embeddings serve as foundational components for retrieval, clustering, classification, question answering, and semantic matching systems. MTEB provides a unified evaluation framework for English and some multilingual tasks. Recently, language-specific extensions have emerged for Chinese, French, Persian, Dutch, Russian, Vietnamese, Turkish, Arabic, and African languages.

**Limitations of Prior Work**: Polish models have long lacked an embedding benchmark with sufficient task coverage. Existing Polish evaluations typically cover only single or limited tasks, such as sentiment classification, semantic relevance, or BEIR-PL retrieval, failing to determine whether a model remains stable across different application scenarios.

**Key Challenge**: Multilingual models may be functional for Polish, but their performance is heavily influenced by training corpora, task types, and model scale. Without a unified, public, diverse, and quality-controlled benchmark, it is difficult for users to select models for practical systems or to fairly compare Polish-specific models with general multilingual models.

**Goal**: The authors aim to construct a Polish version of MTEB by reusing existing public Polish tasks and filling gaps in missing task types, particularly clustering. Additionally, they collect results from 30 public embedding models to analyze the impact of task types, model size, and zero-shot coverage on evaluation conclusions.

**Key Insight**: Instead of merely translating English tasks, the paper integrates existing Polish data, BEIR-PL retrieval tasks, KLEJ/LEPISZCZE-related data, and newly constructed PLSC and Wikinews-PL clustering data into the MTEB framework, while making the code, data, and leaderboard public.

**Core Idea**: Establish a 30-task, multi-type, reproducible embedding benchmark for Polish using the unified evaluation interface of MTEB. Simultaneously, training data similarity is marked as zero-shot columns to remind readers to distinguish between genuine generalization and gains from training set similarity.

## Method
The methodology of PL-MTEB focuses on benchmark construction and evaluation protocols rather than proposing new embedding models. The authors performed three core actions: defining task sets, cleaning and constructing new data, and evaluating models with analysis by task type and scale.

### Overall Architecture
The benchmark contains 30 sub-tasks across 5 categories. Classification tasks evaluate linear separability using few-shot logistic regression; clustering tasks utilize mini-batch k-means and v-measure; pair classification uses average precision under similarity thresholds; retrieval tasks use nDCG@10; and STS tasks use Spearman correlation under cosine similarity.

The task sources are divided into three groups. The first group extracts Polish sub-tasks from existing MTEB or multilingual MTEB, primarily including BEIR-PL retrieval tasks. The second group consists of new tasks added by the authors based on existing public Polish data, mostly from human-annotated resources like KLEJ. The third group includes two newly constructed datasets, PLSC and Wikinews-PL, which generate four clustering tasks.

Evaluation includes 30 public dense embedding models, covering small, base, large, and 1B+ parameter models. These include multilingual E5, SBERT, Arctic-Embed, Qwen3-Embedding, and BGE-Multilingual-Gemma2, as well as Polish-related models like MMLW, Stella-PL, and Silver Retriever. Each model is run according to developer-recommended configurations, and its zero-shot status is recorded for each task.

### Key Designs
1.  **Unified Protocol for Five Task Categories**:
    - **Function**: Allows the same embedding model to be evaluated across classification, clustering, pair classification, retrieval, and STS using a unified interface.
    - **Mechanism**: Classification uses 8 training samples per class to train logistic regression (averaged over 10 runs); clustering uses mini-batch k-means where $k$ equals the number of labels (10 runs); retrieval uses nDCG@10 as the primary metric; STS uses cosine Spearman correlation; pair classification uses cosine average precision.
    - **Design Motivation**: Embedding models are often strong in one task but weak in another. Splitting results by task type provides better guidance for practical model selection than a single average score.

2.  **PLSC and Wikinews-PL Clustering Reinforcement**:
    - **Function**: Addresses the deficiency of clustering tasks in Polish benchmarks.
    - **Mechanism**: PLSC originates from the Polish Library of Science metadata, containing approximately 160K Polish paper records with hierarchical labels (8 scientific fields and 44 disciplines); Wikinews-PL originates from Polish Wikinews, with articles categorized into politics, economy, disasters, culture, science, law, sports, society, and technology. Both generate S2S and P2P clustering tasks, limited to 2,048 entries per task to align with MTEB efficiency assumptions.
    - **Design Motivation**: Clustering relies more on the global structure of the embedding space rather than supervised classifiers. New clustering tasks better distinguish general semantic representation capabilities.

3.  **Data Quality and Zero-shot Labeling**:
    - **Function**: Reduces the interference of duplicates, leakage, and training data similarity on results.
    - **Mechanism**: Cleaning empty text and samples with fewer than 3 words; verifying labels and scores; removing near-duplicates with label conflicts or score differences exceeding 0.5; de-duplication at the split level; and verifying test-train leakage. The evaluation table also records the zero-shot ratio for each model, representing the proportion of tasks where no similar data was present in the model's training set.
    - **Design Motivation**: Embedding benchmarks are prone to contamination by training data, especially in retrieval and STS. The zero-shot column helps readers identify if high scores result from training data similarity.

### Loss & Training
PL-MTEB does not train new models and thus lacks a unified training loss. Only lightweight downstream evaluators are trained during evaluation: logistic regression for classification and k-means for clustering. Other tasks directly use embedding similarity or retrieval ranking. All models are loaded via original methods, and the evaluation code is based on the MTEB framework.

## Key Experimental Results

### Main Results
The benchmark comprises 30 tasks: 7 classification, 5 clustering, 4 pair classification, 11 retrieval, and 3 STS. Retrieval tasks include ArguAna-PL, DBPedia-PLHardNeg, FiQA-PL, etc.

| Task Category | No. of Tasks | Main Metric | Representative Task / Data Source | Key Design Point |
|------|------|------|------|------|
| Classification | 7 | Accuracy | CBD, PolEmo2.0, AllegroReviews, PAC, Massive | 8-shot logistic regression per class, 10-run average |
| Clustering | 5 | V-measure | EightTags, PLSC, Wikinews-PL | mini-batch k-means, hierarchical tasks averaged |
| Pair Classification | 4 | Cosine AP | SICK-E-PL, CDSC-E, PSC, PPC | Evaluate similarity separability of sentence pairs |
| Retrieval | 11 | nDCG@10 | BEIR-PL series | Mostly query-corpus retrieval, some with limited corpus scale |
| STS | 3 | Cosine Spearman | SICK-R-PL, CDSC-R, STS-B Multilingual | Measure semantic similarity ranking correlation |

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
As this is a benchmark paper, there is no traditional module ablation. The analysis focuses on task type, model scale, and training data similarity.

| Analysis Dimension | Observation | Insight |
|------|------|------|
| Task Type Winners | Qwen3-8B (Class.), Qwen3-4B (Clust.), BGE-Gemma2 (PairClass.), stella-pl-retrieval (Retr.), stella-pl (STS) | No single model dominates all tasks; average scores cannot replace task-level selection. |
| Model Scale | 1B+ models are highest overall; Qwen3-8B leads with Avg(30)=70.47 | LLMs have clear advantages, but performance is not monotonic across all tasks. |
| Small Models | mmlw-roberta-base leads significantly in the <150M group (Avg=62.52) | Polish-specific small models remain competitive in resource-constrained scenarios. |
| Base Group | snowflake-arctic-embed-m-v2.0 Avg=57.06 vs multilingual-e5-base Avg(type)=61.66 | No clear dominator among mid-scale multilingual models; depends on task type. |
| Retrieval Task | stella-pl series is strongest but with only 80% zero-shot ratio | High retrieval scores may be influenced by similar training data; must be interpreted with zero-shot metrics. |
| P2P vs S2S Cluster | P2P generally outperforms S2S in new tasks | Longer text contains more clustering information; title-level embedding is more challenging. |

### Key Findings
- Qwen3-Embedding-8B is the overall strongest model (Avg=70.47), but it excels primarily in classification and is not first in every category.
- BGE-Multilingual-Gemma2 has the highest average in pair classification, indicating that ultra-large multilingual models remain powerful in semantic matching.
- Polish-specific stella-pl-retrieval-8k performs best in retrieval (avg 61.59), but caution is needed as its training data shows high similarity to retrieval tasks.
- mmlw-roberta-base, with only 124M parameters, achieves an Avg(30) of 62.52, outperforming several base/large multilingual models, validating the value of language-specific distillation.
- The primary contribution of the benchmark is extending Polish evaluation from scattered tasks to a comprehensive 5-category 30-task suite with data quality and zero-shot perspectives.

## Highlights & Insights
- PL-MTEB is highly suitable as a tool for "practical embedding model selection" rather than just a leaderboard, as it reports both task-type averages and total averages.
- The addition of PLSC and Wikinews-PL clustering tasks is critical. While models may perform well on supervised retrieval, clustering better exposes the stability of the semantic space structure.
- The zero-shot column is an excellent evaluation practice. Given the complexity of multilingual training data, explicit labeling of similarity reduces misinterpretation of performance.
- Results remind us that ultra-large multilingual models and language-specific small models are not simple substitutes. While LLMs are strong overall, customized models like MMLW or Stella-PL retain significant value for constrained deployments.

## Limitations & Future Work
- While PL-MTEB covers 30 tasks, many retrieval tasks are from machine-translated BEIR-PL, which may introduce translation bias and structural artifacts from English.
- Zero-shot determination relies on collected training data descriptions. Since many models are not fully open-source regarding corpora, contamination can only be approximately estimated.
- Classification tasks use 8-shot logistic regression, which is suitable for evaluating separability but may not represent performance in full-data fine-tuning scenarios.
- The benchmark focuses on dense embeddings and does not compare the combined effects of sparse retrieval, hybrid retrieval, rerankers, or instruction embeddings in specific business scenarios.
- Subsequent versions could include more native Polish retrieval data, long-document tasks, cross-lingual retrieval, and domain-specific tasks.

## Related Work & Insights
- **vs Original MTEB**: While MTEB provides the framework, PL-MTEB grounds task and data quality control specifically for Polish, making conclusions more applicable to local language needs.
- **vs BEIR-PL / PIRB**: These focused primarily on retrieval; PL-MTEB offers a broader evaluation including clustering and classification.
- **vs KLEJ / LEPISZCZE**: These benchmarks lean toward NLU and classification understanding; PL-MTEB focuses on representation quality without task-specific deep models.
- **vs MMTEB**: PL-MTEB acts as a refined Polish subset of the Massive Multilingual MTEB, adding data curation, task explanations, and local model analysis.

## Rating
- Novelty: ⭐⭐⭐☆☆ Algorithm novelty is limited, but the systematic construction of a language-specific benchmark is of clear value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 30 tasks, 30 models, and robust analysis across task types and scales.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with informative tables; some appendix tables are extensive.
- Value: ⭐⭐⭐⭐⭐ Highly practical for Polish NLP and multilingual embedding selection; provides a reproducible paradigm for other mid-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](../../ICLR2026/information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ICLR 2026\] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)
- [\[ACL 2026\] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)
- [\[ACL 2026\] Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings](why_mean_pooling_works_quantifying_second-order_collapse_in_text_embeddings.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
