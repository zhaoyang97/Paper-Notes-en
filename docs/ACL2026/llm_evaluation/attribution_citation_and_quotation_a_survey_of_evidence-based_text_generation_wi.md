---
title: >-
  [Paper Note] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Evidence-based text generation] This paper presents a systematic survey of 134 papers on evidence-based text generation with LLMs, proposing for the first time a unified taxonomy (attribution approach × citation characteristics × task), analyzing 300 evaluation metrics organized into seven dimensions and six method categories, and providing a panoramic reference framework for this fragmented field.
tags:
  - ACL 2026
  - LLM Evaluation
  - Evidence-based text generation
  - citation attribution
  - LLM trustworthiness
  - evaluation framework
  - RAG
date: 2026-05-08
content_hash: ca9711b5c7f594cc
---

# Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models

**Conference**: ACL 2026
**arXiv**: [2508.15396](https://arxiv.org/abs/2508.15396)
**Code**: [https://github.com/faerber-lab/AttributeCiteQuote](https://github.com/faerber-lab/AttributeCiteQuote)
**Area**: Survey / NLP
**Keywords**: Evidence-based text generation, citation attribution, LLM trustworthiness, evaluation framework, RAG

## TL;DR

This paper presents a systematic survey of 134 papers on evidence-based text generation with LLMs, proposing for the first time a unified taxonomy (attribution approach × citation characteristics × task), analyzing 300 evaluation metrics organized into seven dimensions and six method categories, and providing a panoramic reference framework for this fragmented field.

## Background & Motivation

**Background**: LLMs face trustworthiness challenges such as hallucination and knowledge limitations, driving increasing research interest in "evidence-based text generation"—enabling LLM outputs to be traceable to supporting evidence. However, the field is highly fragmented: some work uses the term "citation" (75% of papers), others use "attribution" (62%), and still others use "quotation" (13%), with each subcommunity maintaining isolated evaluation practices.

**Limitations of Prior Work**: (1) The absence of unified terminology and a classification system makes it difficult for researchers to situate their own work; (2) evaluation standards are inconsistent—300 metrics exist, yet only 2 frameworks (ALCE and G-Eval) are reused across multiple papers; (3) RAG, though popular, represents only one of seven relevant methods, and excessive focus on RAG risks overlooking other important approaches.

**Key Challenge**: Rapidly growing research interest (the number of papers in 2024 is 3.4× that of 2023, with 75% published after 2023) versus the lack of a unified perspective for integrating and comparing different methods.

**Goal**: To provide the first systematic survey dedicated to evidence-based text generation with LLMs, establish a unified taxonomy, analyze evaluation practices, and identify research trends and future directions.

**Key Insight**: A PRISMA-protocol systematic mapping study, screening 134 relevant papers from 805 deduplicated papers, with a faceted classification approach used to construct the taxonomy.

**Core Idea**: Unifying "citation," "attribution," and "quotation" under the paradigm of "evidence-based text generation," and providing a systematic perspective on this fragmented field through a three-dimensional taxonomy and a seven-dimensional evaluation framework.

## Survey Framework / Taxonomy

### Overall Architecture

The paper proposes a three-dimensional, orthogonal taxonomy capturing core design choices in evidence-based text generation:

- **Dimension 1: Attribution Approach** — how content is linked to evidence (parametric vs. non-parametric)
- **Dimension 2: Citation Characteristics** — the form and presentation of evidence (modality, granularity, style, visibility, frequency)
- **Dimension 3: Task** — application scenarios (six task categories including QA, grounded generation, summarization, etc.)

### Key Designs

1. **Attribution Approach**:

    - Function: Describes how LLMs associate generated content with supporting evidence.
    - Core taxonomy: **Parametric** (25 papers)—pure LLM (leveraging existing capabilities, 72% of parametric), model-centric (modifying architecture/training), data-centric (curating/augmenting data). **Non-parametric** (126 papers)—post-retrieval (58%, RAG being representative), post-generation (18%, generating first then retrieving evidence), in-generation (4%, dynamically deciding whether retrieval is needed), in-context (20%, user directly provides evidence).
    - Key Findings: Parametric attribution is severely underexplored, with model-centric and data-centric approaches receiving particularly little attention. Among non-parametric methods, post-retrieval dominates, but in-generation attribution (e.g., Self-RAG) represents a promising yet undervalued emerging direction.

2. **Citation Characteristics**:

    - Function: Describes the specific presentation of evidence.
    - Core taxonomy: **Citation modality**—text 96%, figures, tables, visuals. **Evidence granularity**—document-level 43%, paragraph-level 40%, sentence-level 12%, token-level 2%. **Citation style**—inline citation 62%, citation report, paragraph display, narrative citation, highlight gradient, quotation. **Visibility**—final response 91% vs. intermediate text. **Frequency**—multiple citations 64% vs. single citation.
    - Key Findings: Non-textual evidence modalities (figures, tables, images) remain severely underexplored (only 4%); fine-grained evidence (sentence-level, token-level), though a small share, is growing faster.

3. **Task Distribution**:

    - Function: Maps the application landscape of the field.
    - Core taxonomy: QA and grounded text generation are dominant tasks; summarization and fact verification occupy a middle tier; citation text generation and related work generation are emerging tasks.
    - Key Findings: Evaluation practices have been developed primarily around QA tasks and may not be appropriate for emerging tasks (e.g., citation text generation requires more evaluation of citation selection reasoning).

### LLM Integration

**Training** (used in 45% of papers): Dominated by supervised fine-tuning, primarily to improve attribution behavior. Pre-training is less common.
**Prompting** (used in 78% of papers): Dominated by zero/few-shot prompting. Strategies specifically targeting citation behavior include chain-of-citation, chain-of-quote, and conflict-aware prompting.

## Literature Analysis / Coverage

### Evaluation Metric Framework

**300 metrics organized into seven evaluation dimensions**

| Evaluation Dimension | When to Use | Primary Methods | Representative Metrics (reuse count) |
|---|---|---|---|
| Attribution | When annotated evidence is unavailable | Primarily NLI | Citation NLI P/R/F1 (33/33/16), Auto-AIS (11), FActScore (7) |
| Citation | When annotated evidence is available | Primarily retrieval-based | Citation Retrieval P/R/F1 (6/6/5), Citation Accuracy (2) |
| Correctness | Always required | Lexical overlap / NLI | Exact Match (12), BLEU-N (5), Claim Recall (17) |
| Language Quality | When model is modified | LLM-as-Judge | G-Eval Fluency (4), MAUVE (21), Perplexity (4) |
| Preservation | For post-generation attribution | Lexical overlap | Preservation-Levenshtein (3), F1-AP (2) |
| Relevance | User-facing scenarios | LLM-as-Judge | G-Eval Relevance (3), RAGAS (2) |
| Retrieval | For non-parametric attribution | Retrieval metrics | P@k (4), R@k (4), MRR (3) |

### Evaluation Guidelines

| Dimension Category | When to Evaluate | Notes |
|---|---|---|
| **Core Dimensions** | Attribution or Citation + Correctness | Correctness should always be evaluated; attribution and citation are chosen based on evidence availability |
| **Contextual Dimensions** | Language Quality, Preservation, Relevance, Retrieval | Depends on task design and system architecture |

### Key Findings

- Only 2 frameworks (ALCE and G-Eval) and 2 benchmarks are reused across multiple papers, indicating a severe lack of evaluation standardization.
- 19 frameworks, 11 benchmarks, and 231 datasets are identified across the 134 papers.
- Text accounts for 96% of citation modalities; multimodal evidence remains virtually unexplored.
- Parametric attribution, despite being critical for understanding models' internal knowledge and data provenance, is severely neglected.
- Human evaluation still dominates the correctness dimension, reflecting the limitations of automatic metrics in capturing semantic errors.

## Highlights & Insights

- Unifying "citation," "attribution," and "quotation" under "evidence-based text generation" constitutes an important conceptual contribution, resolving longstanding terminological confusion.
- The seven-dimensional evaluation guideline (Table 1) provides practitioners with clear metric selection recommendations—the distinction between core and contextual dimensions is highly practical.
- The tripartite classification of parametric attribution (pure LLM / model-centric / data-centric) is more nuanced than previous binary distinctions.
- In-generation attribution is identified as a promising yet undervalued direction—accounting for only 4% of papers but representing a trend toward tighter integration of retrieval and generation.
- The paper notes that LLM citation behavior may exhibit biases analogous to those of human authors, calling for research into the interpretability of LLM citation reasoning.

## Limitations & Future Work

- A single search string may miss some relevant studies (sensitivity analysis shows only 4% additional papers found).
- Coverage is limited to English-language papers, potentially underrepresenting non-English research.
- Manual screening and classification inevitably introduce some degree of subjectivity.
- **Four major future directions**: (1) deeper investigation of parametric and hybrid attribution; (2) standardized evaluation frameworks (currently only 2 frameworks are reused among 300 metrics); (3) interpretable citation reasoning—understanding why LLMs select particular sources; (4) multimodal evidence support—extending beyond the current 96% text dominance to figures, tables, and images.

## Related Work & Insights

- **vs. Li et al. (2023a)**: The only prior related survey, but now severely outdated (75%+ of papers were published after it) and does not cover the full paradigm.
- **vs. Huang & Chang (2024)**: A position paper that only emphasizes the importance of citation without providing a systematic survey.
- **vs. RAG surveys**: RAG surveys cover only post-retrieval, one of the seven attribution approaches covered in this paper.
- **vs. hallucination/grounding surveys**: Those surveys focus on different aspects; this paper is specifically concerned with evidence generation rather than detection.

## Rating

- Novelty: ⭐⭐⭐⭐ First comprehensive unified taxonomy for evidence-based text generation; the three-dimensional taxonomy is well-designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage of 134 papers, 300 metrics, 19 frameworks, 231 datasets, and 11 benchmarks is highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, balanced multi-dimensional analysis, with concise "Takeaways" summarizing each section.
- Value: ⭐⭐⭐⭐⭐ A panoramic synthesis of a rapidly growing yet fragmented field, providing important reference value for both researchers and practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] AnchorMem: Anchored Facts with Associative Contexts for Building Memory in Large Language Models](anchormem_anchored_facts_with_associative_contexts_for_building_memory_in_large_.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)

</div>

<!-- RELATED:END -->
