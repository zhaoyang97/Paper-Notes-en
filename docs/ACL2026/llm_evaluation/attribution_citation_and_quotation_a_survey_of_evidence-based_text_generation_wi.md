---
title: >-
  [Paper Note] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][RAG] This paper systematically reviews 134 papers on evidence-based text generation for LLMs. It proposes the first unified taxonomy (Attribution Mechanism × Citation Features × Task), analyzes 300 evaluation metrics categorized into seven dimensions and six methods, and provides a panoramic reference framework for this fra
tags:
  - ACL 2026
  - LLM Evaluation
  - RAG
date: 2026-05-08
content_hash: 0044649ee0547393
---
# Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2508.15396](https://arxiv.org/abs/2508.15396)  
**Code**: [https://github.com/faerber-lab/AttributeCiteQuote](https://github.com/faerber-lab/AttributeCiteQuote)  
**Area**: Survey/NLP  
**Keywords**: Evidence-based text generation, citation attribution, LLM trustworthiness, evaluation framework, RAG

## TL;DR

This paper systematically reviews 134 papers on evidence-based text generation for LLMs. It proposes the first unified taxonomy (Attribution Mechanism × Citation Features × Task), analyzes 300 evaluation metrics categorized into seven dimensions and six methods, and provides a panoramic reference framework for this fragmented field.

## Background & Motivation

**Background**: LLMs face trustworthiness challenges such as hallucinations and knowledge limitations. Increasing research focuses on "evidence-based text generation"—ensuring LLM outputs are traceable to supporting evidence. However, the field is highly fragmented: various terms are used, such as "citation" (75% of papers), "attribution" (62%), or "quotation" (13%), with isolated evaluation practices.

**Limitations of Prior Work**: (1) Lack of unified terminology and classification systems, making it difficult for researchers to position their work; (2) Inconsistent evaluation standards—among 300 metrics, only two frameworks (ALCE, G-Eval) are reused across multiple papers; (3) Although RAG is popular, it is only one of seven relevant methods, and over-focusing on RAG misses other critical approaches.

**Key Challenge**: Rapidly growing research interest (the number of papers in 2024 is 3.4x that of 2023, with 75% published after 2023) vs. the lack of a unified perspective to integrate and compare different methods.

**Goal**: To provide the first systematic survey specifically targeting LLM evidence-based text generation, establish a unified taxonomy, analyze evaluation practices, and identify research trends and future directions.

**Key Insight**: A systematic mapping study was conducted following the PRISMA protocol, screening 134 relevant papers from 805 deduplicated candidates, using a faceted classification method to construct the taxonomy.

**Core Idea**: Unify "citation," "attribution," and "quotation" into an "evidence-based text generation" paradigm, providing a systematic perspective through a three-dimensional taxonomy and a seven-dimensional evaluation framework.

## Method

### Overall Architecture

A systematic mapping study was performed per the PRISMA protocol: 134 relevant works were screened from 805 deduplicated papers, followed by manual encoding using a faceted classification method. The core output is a 3D independent taxonomy—combining "Attribution Mechanism (how content relates to evidence) × Citation Features (the form of the evidence) × Task (application scenario)." Any work in evidence-based text generation can be positioned within this cube. A cross-cutting perspective of LLM integration (Training vs. Prompting) is overlaid to answer how models acquire attribution capabilities.

### Key Designs

**1. Attribution Mechanism: Dichotomy and Subdivision of Parametric vs. Non-parametric.** This dimension characterizes the fundamental path through which LLMs associate generated content with supporting evidence. **Parametric** (25 papers) involves evidence entering model weights, subdivided into Pure LLM (leveraging existing capabilities, 72% of parametric), Model-centric (architecture/training changes), and Data-centric (data curation/augmentation). **Non-parametric** (126 papers) keeps evidence outside the weights, categorized by retrieval timing: Post-retrieval (58%, e.g., RAG), Post-generation (18%, generating then finding evidence), In-generation (4%, dynamic decision to retrieve), and In-context (20%, evidence directly in the prompt). This three-way split is finer than the simple "RAG / Non-RAG" distinction and reveals field imbalances: parametric methods are largely neglected, while in-generation attribution (e.g., Self-RAG), though only 4%, represents the frontier of tight coupling.

**2. Citation Features: Five facets characterizing evidence appearance.** The same evidence can be presented differently. These facets include Citation Modality (96% text, with charts/visuals nearly blank), Evidence Level (Document 43%, Paragraph 40%, Sentence 12%, Token 2%), Citation Style (In-line 62%, plus reports, snippets, narrative, highlight gradients, quotations, etc.), Visibility (Final response 91% vs. intermediate text), and Frequency (Multiple 64% vs. Single). These reveal two structural gaps: non-text modalities are severely underdeveloped (4%), while fine-grained evidence (sentence/token level) is the fastest-growing segment, indicating a trend toward precise traceability.

**3. Task: Mapping six application scenarios.** The third dimension maps tasks, finding QA and Grounded Text Generation as dominant, with Summarization and Fact-checking in the middle, and Citation Text Generation or Related Work Generation as emerging tasks. This reveals path dependency in evaluation: existing metrics are largely designed for QA and might not apply to emerging tasks that require reasoning about "why" a citation was chosen.

Additionally, **LLM Integration** methods are categorized into Training (45% of papers, primarily SFT to improve attribution behavior) and Prompting (78%, primarily zero/few-shot, with specialized strategies like chain-of-citation or conflict-aware prompting).

## Key Experimental Results

### Literature Analysis/Coverage

### Evaluation Metric System

**300 metrics classified into seven evaluation dimensions**

| Evaluation Dimension | When to Use | Primary Method | Representative Metrics (Reuse Count) |
| :--- | :--- | :--- | :--- |
| Attribution | No ground-truth evidence | NLI-based | Citation NLI P/R/F1 (33/33/16), Auto-AIS (11), FActScore (7) |
| Citation | Ground-truth evidence exists | Retrieval-based | Citation Retrieval P/R/F1 (6/6/5), Citation Accuracy (2) |
| Correctness | Always required | Lexical overlap/NLI | Exact Match (12), BLEU-N (5), Claim Recall (17) |
| Language Quality | When model is modified | LLM-as-Judge | G-Eval Fluency (4), MAUVE (21), Perplexity (4) |
| Preservation | Post-generation attribution | Lexical overlap | Preservation-Levenshtein (3), F1-AP (2) |
| Relevance | User scenarios | LLM-as-Judge | G-Eval Relevance (3), RAGAS (2) |
| Retrieval | Non-parametric attribution | Retrieval metrics | P@k (4), R@k (4), MRR (3) |

### Evaluation Guidelines

| Dimension Category | When to Evaluate | Description |
| :--- | :--- | :--- |
| **Core Dimensions** | Attribution or Citation + Correctness | Correctness should always be evaluated; Attribution vs. Citation depends on evidence availability. |
| **Contextual Dimensions** | Language Quality, Preservation, Relevance, Retrieval | Depends on task design and system architecture. |

### Key Findings

- Only two frameworks (ALCE, G-Eval) and two benchmarks are reused across multiple papers, indicating severe lack of standardization.
- Identified 19 frameworks, 11 benchmarks, and 231 datasets among 134 papers.
- Text accounts for 96% of citation modalities; multimodal evidence is almost nonexistent.
- Parametric attribution is crucial for understanding internal knowledge and data provenance but is significantly neglected.
- Human evaluation still dominates the correctness dimension, reflecting the limitations of automated metrics in capturing semantic errors.

## Highlights & Insights

- Unifying "citation," "attribution," and "quotation" as "evidence-based text generation" is a significant conceptual contribution, resolving long-standing terminological confusion.
- The seven-dimension evaluation guide (Table 1) provides clear metric selection advice for practitioners—the distinction between core and contextual dimensions is highly practical.
- The three-way split for parametric attribution (Pure LLM / Model-centric / Data-centric) is more refined than previous binary classifications.
- In-generation attribution is identified as a promising but undervalued direction (4%), representing the trend of tighter integration between retrieval and generation.
- Notes that citation behavior may exhibit biases similar to human authors, calling for research into the interpretability of LLM citation reasoning.

## Limitations & Future Work

- A single search string may miss some relevant studies (sensitivity analysis showed only 4% additional findings).
- Only covers English papers, potentially underestimating non-English research.
- Manual screening and classification inevitably introduce some subjectivity.
- **Four Future Directions**: (1) Deep research into parametric and hybrid attribution; (2) Standardized evaluation frameworks (only 2 of 300 metrics/frameworks are reused); (3) Explainable citation reasoning—understanding why LLMs select specific sources; (4) Multimodal evidence support—extending from 96% text to charts, tables, and images.

## Related Work & Insights

- **vs. Li et al. (2023a)**: The only prior relevant survey, but largely outdated (75%+ papers published after it), and does not cover the full paradigm.
- **vs. Huang & Chang (2024)**: A position paper that emphasizes citation importance but lacks a systematic survey.
- **vs. RAG surveys**: RAG surveys only cover post-retrieval; this paper covers seven attribution mechanisms.
- **vs. Hallucination/Grounding surveys**: Focuses on different aspects; this paper concentrates on evidence generation rather than detection.

## Rating

- Novelty: ⭐⭐⭐⭐ First comprehensive unified classification for evidence-based text generation; 3D taxonomy is well-designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive coverage of 134 papers, 300 metrics, 19 frameworks, 231 datasets, and 11 benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, balanced multi-dimensional analysis, and concise "Takeaways" for each section.
- Value: ⭐⭐⭐⭐⭐ A panoramic synthesis of a rapidly growing but fragmented field, providing significant reference value for researchers and practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ICML 2026\] Authority, Truth, and Citation Bias: A Large-Scale Multi-Domain Benchmark for Studying Epistemic Susceptibility in Large Language Models](../../ICML2026/llm_evaluation/authority_truth_and_citation_bias_a_large-scale_multi-domain_benchmark_for_study.md)

</div>

<!-- RELATED:END -->
