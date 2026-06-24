---
title: >-
  [Paper Note] PL-MTEB: Polish Massive Text Embedding Benchmark
description: >-
  [ACL2026 Findings][Information Retrieval & RAG][Polish NLP] PL-MTEB constructs a 30-task evaluation set for Polish text embeddings covering classification, clustering, pair classification, retrieval, and semantic similarity. It systematically evaluates 30 Polish and multilingual embedding models, showing that while large models generally lead, factors such as task type, training data leakage, and model scale significantly impact the conclusions.
tags:
  - "ACL2026 Findings"
  - "Information Retrieval & RAG"
  - "Polish NLP"
  - "Text Embedding"
  - "MTEB"
  - "Information Retrieval"
  - "Benchmark Evaluation"
date: 2026-05-08
content_hash: a69679529c4a8d92
---

# PL-MTEB: Polish Massive Text Embedding Benchmark

**Conference**: ACL2026 Findings  
**arXiv**: [2405.10138](https://arxiv.org/abs/2405.10138)  
**Code**: https://github.com/rafalposwiata/pl-mteb  
**Area**: Information Retrieval/RAG  
**Keywords**: Polish NLP, Text Embedding, MTEB, Information Retrieval, Benchmark Evaluation

## TL;DR
PL-MTEB constructs a 30-task evaluation set for Polish text embeddings covering classification, clustering, pair classification, retrieval, and semantic similarity. It systematically evaluates 30 Polish and multilingual embedding models, showing that while large models generally lead, factors such as task type, training data leakage, and model scale significantly impact the conclusions.

## Background & Motivation
**Background**: Text embedding is a fundamental component of retrieval, clustering, classification, Q&A, and semantic matching systems. MTEB provides a unified evaluation framework for English and some multilingual tasks. In recent years, language-specific extensions have emerged for Chinese, French, Persian, Dutch, Russian, Vietnamese, Turkish, Arabic, African languages, etc.

**Limitations of Prior Work**: Polish models have long lacked an embedding benchmark with sufficiently broad task coverage. Existing Polish evaluations usually cover only a single task or a few tasks, such as sentiment classification, semantic correlation, or BEIR-PL retrieval, failing to answer whether a model remains stable across different application scenarios.

**Key Challenge**: Multilingual models may be usable for Polish, but their performance is heavily influenced by training corpora, task types, and model scale. Without a unified, public, diverse benchmark with controllable annotation quality, it is difficult for users to select models for practical systems or to fairly compare Polish-specific models with general-purpose multilingual models.

**Goal**: The authors aim to construct a Polish version of MTEB that reuses existing public Polish tasks and fills in missing task types, especially clustering. Simultaneously, they collect results from 30 public embedding models and analyze the impact of task type, model size, and zero-shot coverage on the evaluation conclusions.

**Key Insight**: Instead of merely translating English tasks, the paper integrates existing Polish data, BEIR-PL retrieval tasks, KLEJ/LEPISZCZE-related data, and newly constructed PLSC and Wikinews-PL clustering data into the MTEB framework, making the code, data, and leaderboard public.

**Core Idea**: Build a 30-task, multi-type, reproducible embedding benchmark for Polish using the unified evaluation interface of MTEB. Additionally, label training data similarity as a zero-shot column to remind readers to distinguish between true generalization and gains from training set similarity.

## Method
The methodology of PL-MTEB focuses on benchmark construction and evaluation protocols rather than proposing a new embedding model. The authors performed three tasks: defining the task set, cleaning and constructing new data, and evaluating models with analysis by task type and model scale.

### Overall Architecture
The benchmark contains 30 sub-tasks across 5 categories. Classification tasks use few-shot logistic regression to assess the linear separability of embeddings. Clustering tasks use mini-batch k-means and v-measure. Pair classification uses average precision under similarity thresholds. Retrieval tasks use nDCG@10. STS uses Spearman correlation under cosine similarity.

Task sources are divided into three groups. The first group consists of Polish sub-tasks extractable from existing MTEB or multilingual MTEB, primarily BEIR-PL retrieval tasks. The second group includes tasks newly added by the authors based on existing Polish public data, mostly from human-annotated resources like KLEJ. The third group includes two newly constructed datasets, PLSC and Wikinews-PL, which generate four clustering tasks.

Evaluation models include 30 public dense embedding models, covering small, base, large, and 1B+ parameter models. These include multilingual E5, SBERT, Arctic-Embed, Qwen3-Embedding, and BGE-Multilingual-Gemma2, as well as Polish-specific models like MMLW, Stella-PL, and Silver Retriever. Each model is run following the developer's recommended configuration, and the proportion of tasks where it is considered zero-shot is recorded.

### Key Designs

**1. Unified Protocol for Five Task Categories: Integrating Classification, Clustering, Pair Classification, Retrieval, and STS into a Single Evaluation Interface**

Embedding models are often strong in one category but weak in another; viewing only a total average score can mask such imbalances. PL-MTEB therefore provides a lightweight, reproducible evaluator for each of the five categories. Classification tasks use only 8 samples per category to train a logistic regression, repeated 10 times to check linear separability. Clustering uses mini-batch k-means where the number of clusters $k$ equals the number of labels, also repeated 10 times, scored with v-measure. Retrieval uses nDCG@10 as the primary metric. STS uses Spearman correlation under cosine similarity. Pair classification uses average precision under cosine similarity thresholds. All five protocols share the same embedding output, allowing results to be compared horizontally—readers can examine both the 30-task overall mean Avg(30) and the Avg(by type) to select models based on specific needs like retrieval or clustering.

**2. PLSC and Wikinews-PL Clustering Reinforcement: Filling the Missing Dimension of Polish Clustering with Two New Datasets**

Most existing Polish evaluations revolve around classification, retrieval, and STS, leaving clustering tasks almost entirely neglected. Yet clustering relies most on the global semantic structure of embeddings and benefits least from supervised classifiers or retrieval training data, making it a good probe for testing the stability of the representation space. To address this, the authors created two new datasets: PLSC, taken from Polish Library of Science metadata with approximately 160K Polish paper records labeled by 8 scientific fields and 44 disciplines; and Wikinews-PL, taken from Polish Wikinews and labeled by categories such as politics, economy, disasters, culture, science, law, sports, society, and technology. Each dataset constructs S2S (sentence-to-sentence/title-level) and P2P (paragraph-to-paragraph) clustering tasks, with each task truncated to 2,048 entries to align with the efficiency assumptions of MTEB. Adding clustering allows the benchmark to expose models that score high on supervised tasks but have loose semantic spaces.

**3. Data Quality and Zero-shot Labeling: Cleaning Leakage and Surfacing Training Data Similarity**

Embedding benchmarks are highly susceptible to training data contamination—retrieval tasks and common STS data are particularly likely to have appeared in model training corpora, meaning high scores do not necessarily indicate true generalization. The authors perform hard cleaning: removing empty texts and samples with fewer than 3 words, checking labels and scores, removing near-duplicates with conflicting labels or score differences exceeding 0.5, and performing deduplication at the split level to verify test-train leakage. Furthermore, they include a zero-shot ratio column for each model in the evaluation table, representing the percentage of tasks where similar tasks were not present in the model's training data. This column provides a defense for readers: a high retrieval score in a model with a low zero-shot ratio suggests the model may have benefited from similar training data rather than purely generalization ability.

### Loss & Training
PL-MTEB itself does not train new models and has no unified training loss. Only lightweight downstream evaluators are trained during evaluation: logistic regression for classification and k-means for clustering. Other tasks directly use embedding similarity or retrieval ranking. All models are loaded as originally released. The evaluation code is based on the MTEB framework, with results and data shared on GitHub and Hugging Face.

## Key Experimental Results

### Main Results
The benchmark consists of 30 tasks: 7 classification, 5 clustering, 4 pair classification, 11 retrieval, and 3 STS. Retrieval tasks include ArguAna-PL, DBPedia-PLHardNeg, FiQA-PL, HotpotQA-PLHardNeg, MSMARCO-PLHardNeg, NFCorpus-PL, NQ-PLHardNeg, Quora-PLHardNeg, SCIDOCS-PL, SciFact-PL, and TRECCOVID-PL.

| Task Type | Task Count | Main Metric | Representative Tasks / Data Sources | Design Key Points |
|------|------|------|------|------|
| Classification | 7 | Accuracy | CBD, PolEmo2.0, AllegroReviews, PAC, MassiveIntent/Scenario | 8-shot logistic regression per class, 10 repetitions |
| Clustering | 5 | V-measure | EightTags, PLSC, Wikinews-PL | mini-batch k-means, hierarchical tasks averaged |
| Pair Classification | 4 | Cosine AP | SICK-E-PL, CDSC-E, PSC, PPC | Evaluate similarity separability of sentence pairs |
| Retrieval | 11 | nDCG@10 | BEIR-PL series tasks | Primarily query-corpus retrieval; some HardNeg limit corpus size |
| STS | 3 | Cosine Spearman | SICK-R-PL, CDSC-R, STSBenchmarkMultilingual | Measure semantic similarity rank correlation |

| Model | Params | Zero-shot Ratio | Classification | Clustering | PairClass | Retrieval | STS | Avg(30) | Avg(by type) |
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
As this is a benchmark paper, there is no traditional ablation of model modules; the analysis dimensions are task type, model scale, and training data similarity.

| Analysis Dimension | Observation | Insight |
|------|------|------|
| Task Type Winners | Qwen3-8B best for classification; Qwen3-4B best for clustering; BGE-Gemma2 best for PairClass; stella-pl-retrieval-8k best for Retrieval; stella-pl best for STS. | No single model dominates all tasks; average scores cannot replace task-level selection. |
| Model Scale | Models above 1B generally perform best; Qwen3-8B leads with Avg(30)=70.47. | Large models have a clear advantage, but performance is not always monotonic with scale across all tasks. |
| Small Models | mmlw-roberta-base leads significantly in the <150M group with Avg(30)=62.52. | Polish-specific small models remain highly competitive in resource-constrained scenarios. |
| Base Model Group | snowflake-arctic-embed-m-v2.0 Avg(30)=57.06; multilingual-e5-base Avg(by type)=61.66. | No clear dominant model among mid-sized multilingual models; depends on task type. |
| Retrieval Tasks | stella-pl-retrieval-8k and stella-pl are strongest but have lower zero-shot ratios (80). | High retrieval scores may be affected by similar training data; interpret alongside the zero-shot column. |
| P2P vs S2S Clustering | In new PLSC/Wikinews tasks, P2P usually outperforms S2S. | Longer texts contain more information for clustering; title-level embedding is more challenging. |

### Key Findings
- Qwen3-Embedding-8B is the best overall model (Avg(30)=70.47, Avg(by type)=74.41), but its dominance is mostly in classification; it is not first in every category.
- BGE-Multilingual-Gemma2 has the highest average score for pair classification, showing that ultra-large multilingual models remain very strong in semantic matching tasks.
- The Polish-specific stella-pl-retrieval-8k is best for retrieval with an nDCG@10 type average of 61.59, though its scores should be interpreted carefully given high similarity between training and evaluation data.
- mmlw-roberta-base, with only 124M parameters, achieved Avg(30)=62.52 in the small model group, outperforming several base/large multilingual models, showing the continued value of language-specific distillation.
- The primary contribution at the benchmark level is not any specific model ranking but the expansion of Polish embedding evaluation from scattered tasks to a 5-type, 30-task suite with data quality and zero-shot perspectives.

## Highlights & Insights
- PL-MTEB is highly suitable as a tool for "practical embedding model selection" rather than just a leaderboard, as it reports both task-type averages and overall averages.
- The addition of PLSC and Wikinews-PL clustering tasks is critical. Many models perform well on supervised retrieval or STS, but clustering tests reveal if the semantic space structure is truly stable.
- The zero-shot column is an excellent evaluation practice. Multilingual embedding training corpora are complex; explicit labeling of similar training data reduces misinterpretation.
- Results remind us that ultra-large multilingual models and language-specific small models are not simply substitutes. Large models like Qwen3/BGE are overall stronger if resources permit, but language-specific models like MMLW/Stella-PL have real-world value for constrained deployments.

## Limitations & Future Work
- Although PL-MTEB covers 30 tasks, several retrieval tasks originate from machine-translated BEIR-PL, which may introduce biases in translation style and structure.
- Zero-shot determination depends on the availability of training data descriptions. Since many models do not fully disclose training corpora, data contamination can only be estimated approximately.
- Classification tasks use 8-shot logistic regression, which is suitable for evaluating embedding separability but may not represent performance in full-data fine-tuned downstream systems.
- The benchmark primarily evaluates dense embeddings and does not deeply compare sparse retrieval, hybrid retrieval, rerankers, or instruction embeddings in specific business scenarios.
- Future work could include more native Polish retrieval data, long-document tasks, cross-lingual retrieval, domain-specific tasks, and continuous leaderboard updates.

## Related Work & Insights
- **vs Original MTEB**: MTEB provides the framework, but English and a few multilingual tasks dominate; PL-MTEB brings task and quality control to Polish, making conclusions applicable to local language use.
- **vs BEIR-PL / PIRB**: These focus mainly on retrieval; PL-MTEB covers classification, clustering, pair classification, retrieval, and STS for general embedding evaluation.
- **vs KLEJ / LEPISZCZE**: These are focused more on NLU and classification; PL-MTEB focuses on vector representation quality without task-specific deep models.
- **vs MMTEB**: MMTEB is a large-scale multilingual extension; PL-MTEB is a fine-grained Polish subset with enhanced data curation and local model analysis.

## Rating
- Novelty: ⭐⭐⭐☆☆ The algorithm novelty is modest, but the value of a systematically constructed language-specific benchmark is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 30 tasks, 30 models, and solid analysis of task types and model scales with public code/data.
- Writing Quality: ⭐⭐⭐⭐☆ Structure is clear with informative tables, though some long appendix tables require effort to interpret.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for Polish NLP and multilingual embedding selection; provides a reusable paradigm for building MTEB extensions for other medium-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SkMTEB: Slovak Massive Text Embedding Benchmark and Model Adaptation](skmteb_slovak_massive_text_embedding_benchmark_and_model_adaptation.md)
- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](../../ICLR2026/information_retrieval/hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)
- [\[ACL 2026\] REZE: Representation Regularization for Domain-adaptive Text Embedding Pre-finetuning](reze_representation_regularization_for_domain-adaptive_text_embedding_pre-finetu.md)
- [\[ICLR 2026\] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs](../../ICLR2026/information_retrieval/btzsc_a_benchmark_for_zero-shot_text_classification_across_cross-encoders_embedd.md)
- [\[ICLR 2026\] Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement](../../ICLR2026/information_retrieval/let_llms_speak_embedding_languages_generative_text_embeddings_via_iterative_cont.md)

</div>

<!-- RELATED:END -->
