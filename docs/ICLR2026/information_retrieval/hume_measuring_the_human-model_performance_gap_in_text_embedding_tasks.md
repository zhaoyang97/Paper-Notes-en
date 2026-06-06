---
title: >-
  [Paper Note] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks
description: >-
  [ICLR 2026][Information Retrieval & RAG][text embedding] This paper proposes HUME, a human evaluation framework that systematically measures human performance on 16 MTEB datasets spanning reranking, classification…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "text embedding"
  - "human baseline"
  - "MTEB"
  - "evaluation framework"
  - "multilingual"
date: 2026-05-08
content_hash: 1ff0014a0a451d74
---

# HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks

**Conference**: ICLR 2026  
**arXiv**: [2510.10062](https://arxiv.org/abs/2510.10062)  
**Code**: [GitHub](https://github.com/embeddings-benchmark/mteb)  
**Area**: Information Retrieval  
**Keywords**: text embedding, human baseline, MTEB, evaluation framework, multilingual

## TL;DR

This paper proposes HUME, a human evaluation framework that systematically measures human performance on 16 MTEB datasets spanning reranking, classification, clustering, and STS tasks. Humans rank 4th overall (77.6 vs. the best model score of 80.1). The study reveals that cases where models surpass human performance tend to occur on tasks with the lowest human agreement, and evaluates 9 LLMs as potential annotation proxies.

## Background & Motivation

**Background**: Embedding models are central to search, recommendation, and semantic analysis. MTEB provides the most comprehensive model evaluation suite, covering dozens of task types and hundreds of datasets. However, the interpretability of MTEB scores is severely limited — if a model achieves MAP=0.85 on a reranking task, is that good or bad? Without a human performance reference, there is no basis for judgment. **Limitations of Prior Work**: Current evaluations use the theoretical maximum (e.g., MAP=1.0) as the reference point, but inherent ambiguity and annotator disagreement in NLP tasks (plank-2022-problem) render perfect scores both unrealistic and meaningless. This creates a risk of "blind optimization," where models may be replicating annotation artifacts rather than achieving genuine semantic progress. **Key Challenge**: Generative NLP tasks (translation, summarization, dialogue) have a well-established tradition of human evaluation, whereas human baselines for embedding tasks are nearly absent — GLUE/SuperGLUE include human baselines but only for classification and inference, and STS benchmarks report inter-annotator agreement without converting it into comparable performance scores. **Goal**: To establish a systematic and reproducible human performance baseline for MTEB, treating human performance as a diagnostic signal (rather than an upper bound) for understanding dataset quality. **Key Insight**: Multi-annotator experiments are conducted across 4 task types, 5 languages (including low-resource), and 16 datasets, with an additional evaluation of whether 9 LLMs can substitute for human annotation. **Core Idea**: A standardized framework is used to measure human performance on embedding tasks, revealing that model "superhuman" performance frequently reflects dataset quality issues rather than genuine model superiority.

## Method

### Overall Architecture

HUME builds upon MTEB and comprises four core components: (1) task-specific annotation interfaces (built on Argilla), (2) principled dataset sampling guided by five selection criteria, (3) standardized result formats, and (4) evaluation metrics fully aligned with model evaluation protocols. Humans and models are evaluated on identical downsampled instances using the same metrics and procedures.

### Key Designs

1. **Dataset Selection Strategy (5-Dimensional Coverage)**:
    - Function: Ensure that findings generalize broadly
    - Mechanism: Language diversity (English/Arabic/Russian/Norwegian/Danish, covering resource levels 1–5), domain diversity (news/social media/encyclopedic/scientific/forum), construction method (human-annotated and synthetic), task relevance, and variation in complexity
    - Design Motivation: Avoid drawing skewed conclusions from English-only, high-resource datasets

2. **Annotation Protocol and Alignment Design**:
    - Function: Ensure fairness in human–model comparisons
    - Mechanism: Annotation conducted via Argilla; 20–50 samples per task; English tasks assigned 2 annotators to compute agreement; multilingual tasks annotated by native speakers; annotation instructions strictly match original task definitions (same label sets, same 1–5 rating scale)
    - Design Motivation: Eliminate incomparability arising from differences in annotation instructions

3. **Reranking Proxy Design**:
    - Function: Make information retrieval tasks feasible for human annotators
    - Mechanism: Direct retrieval requires evaluating thousands of candidate documents, which is infeasible for humans; reranking (evaluating top-$k$ candidates) serves as a semantically equivalent proxy
    - Design Motivation: Preserve the core semantic challenge while making human evaluation operationally tractable

### Loss & Training

As an evaluation framework, HUME does not train any models. Statistical significance is assessed using Wilson Score Intervals (for classification accuracy) and Fisher $z$-transformation (for correlation-based metrics) to compute 95% confidence intervals. Models fall outside the human confidence interval on 14 of 26 tasks ($p<0.05$).

## Key Experimental Results

### Main Results

**Humans vs. 13 embedding models** (16 datasets, 4 task types × 5 languages, 26 task–language pairs):

| Evaluated Entity | Classification (eng) | Clustering (eng) | Reranking (eng) | STS (eng) | Overall |
|---------|----------|----------|-----------|---------|---------|
| Human | 70.3 | 67.4 | 87.2 | 83.1 | **77.6** |
| jasper (best) | 87.1 | 83.2 | 95.8 | 88.1 | **80.1** |
| e5-mistral-7b | 70.0 | 82.7 | 96.4 | 85.9 | 78.2 |
| SFR-Embedding | 69.8 | 85.1 | 96.3 | 86.4 | 78.3 |
| gte-Qwen2-1.5B | 76.5 | 75.9 | 95.3 | 84.0 | 77.5 |

Humans rank 4th overall (77.6), achieving the highest score on 5 of 14 aggregated task–language pairs. Human advantages are pronounced in non-English tasks: Arabic sentiment 95.0 vs. 77.5 (+17.5), Russian sentiment 92.5 vs. 81.2, Arabic STS 67.5 vs. 40.9 (+26.6).

### Ablation Study

**LLM-as-Annotator** evaluation (9 LLMs, 19 task–language pairs, clustering excluded):

| Category | Human | GPT-5 | GPT-4.1-mini | Mistral-Small | Qwen3-32B | Best Emb. |
|-----|------|-------|-------------|--------------|----------|-----------|
| Classification | 79.1 | 78.9 | 76.1 | 73.8 | 73.0 | 80.3 |
| Reranking | 88.3 | 75.1 | 77.2 | 78.0 | 74.8 | 94.8 |
| STS | 76.5 | 73.0 | 74.9 | 75.0 | 68.6 | 77.1 |
| **Average** | **81.2** | 75.8 | **76.1** | 75.5 | 72.2 | — |

Human and LLM difficulty rankings show moderate positive correlation ($\rho=0.52, p<0.05$), but the gap is largest on reranking tasks (88.3 vs. 78.0) — precisely the task type where human agreement is highest ($\rho=0.64$–$0.85$).

### Key Findings

1. **Model "superhuman" performance often signals dataset quality issues**: Humans achieve only 45.8% on emotion classification ($\kappa=0.39$), yet the best model reaches 87.1%; on ArXiv clustering, humans score 49.2% (ARI=−0.001) vs. 84.6% for models — these superhuman results occur on tasks with the lowest human agreement.
2. **Identification of high-quality benchmarks**: Reranking ($\rho=0.64$–$0.85$) and toxicity classification ($\kappa=0.55$) exhibit high human agreement and represent reliable evaluation targets.
3. **Cross-lingual gap**: Humans hold a clear advantage on non-English tasks (win rate 29% vs. 0% on English), with the largest advantage on Arabic (67% win rate vs. best model).
4. **LLMs cannot substitute for human annotation**: The best LLM (GPT-4.1-mini, 76.1%) trails humans by 5.1 points, with the largest gap on high-agreement reranking tasks.

## Highlights & Insights

- **Filling a critical gap**: MTEB obtains its first systematic human performance baseline, already integrated into the MTEB package.
- **Core insight — diagnostic signal, not upper bound**: Variance in human performance reflects task quality more than human limitations; low human scores typically indicate task design problems (ambiguous annotation guidelines, subjectivity) rather than targets for models to "surpass."
- **Counterintuitive finding**: Model "superhuman" performance consistently appears on tasks with the lowest human agreement — suggesting that models replicate consistent label patterns rather than demonstrating genuine semantic understanding.
- **Practical recommendation**: Benchmark leaderboards should report human agreement metrics alongside scores — 85% on emotion classification (human 45.8%, $\kappa=0.39$) and 85% on reranking (human 87.2%, $\rho=0.75$) represent fundamentally different levels of achievement.

## Limitations & Future Work

- Sample sizes per task are small (20–50 instances), prioritizing breadth (16 datasets) over depth.
- Annotators are predominantly male and aged 20–35, limiting demographic diversity.
- Most tasks use only 2 annotators, constraining the completeness of agreement analysis.
- The underlying causes of the human–model performance gap (e.g., training data coverage, cultural bias) are not deeply investigated.
- The relationship between task design characteristics and human agreement is not systematically studied.

## Related Work & Insights

- **vs. GLUE/SuperGLUE**: These benchmarks report human baselines but only for classification and inference tasks, not for embedding tasks.
- **vs. STS series**: These works report inter-annotator agreement but do not convert it into performance scores comparable to model outputs.
- **vs. TREC**: Human relevance judgments serve as gold standards rather than human performance baselines under model metrics.
- **Insight**: The quality of evaluation benchmarks (annotation agreement) is as important as model capability; "superhuman" scores may signal evaluation noise rather than genuine progress.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic human baseline for embedding tasks with a well-defined problem formulation, though the methodology is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 16 datasets × 13 models × 9 LLMs, though per-task sample sizes are small.
- Writing Quality: ⭐⭐⭐⭐⭐ Analysis is thorough and insightful, with well-supported conclusions.
- Value: ⭐⭐⭐⭐⭐ Directly changes how MTEB leaderboard scores are interpreted; already integrated into the open-source framework.
5. **Full coverage**: 13 models × 16 datasets × 5 languages

## Limitations & Future Work

1. **Limited annotator diversity**: NLP practitioners aged 20–35, predominantly male.
2. **Small sample sizes**: 20–50 samples per task.
3. **Retrieval absent**: Proxied by reranking.
4. **Single annotator (multilingual tasks)**.
5. **Incomplete coverage**: 16 datasets represent only a small fraction of MTEB.
6. **LLM clustering absent**.

## Related Work & Insights

- **MTEB** [Muennighoff et al., 2022]: No human baseline — HUME provides this complement.
- **GLUE/SuperGLUE**: Include human baselines — do not cover embedding tasks.
- **STS Benchmark**: Reports agreement — not converted to comparable scores.
- **TREC**: Human judgments — used as standards, not baselines.
- **Chatbot Arena**: Human preference rankings — HUME extends this paradigm to embedding tasks.

## Rating

| Dimension | Rating |
|------|------|
| Theoretical Depth | ⭐⭐⭐ |
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PL-MTEB: Polish Massive Text Embedding Benchmark](../../ACL2026/information_retrieval/pl-mteb_polish_massive_text_embedding_benchmark.md)
- [\[ACL 2026\] FLARE: Task-Agnostic Embedding Model Evaluation via Normalizing Flows](../../ACL2026/information_retrieval/flare_task-agnostic_embedding_model_evaluation_through_a_normalization_process.md)
- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)
- [\[ICLR 2026\] Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement](judges_verdict_a_comprehensive_analysis_of_llm_judge_capability_through_human_ag.md)
- [\[ICLR 2026\] Embedding-Based Context-Aware Reranker](embedding-based_context-aware_reranker.md)

</div>

<!-- RELATED:END -->
