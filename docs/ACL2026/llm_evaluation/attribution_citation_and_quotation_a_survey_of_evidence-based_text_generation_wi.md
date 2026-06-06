---
title: >-
  [Paper Note] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][Evidence-based text generation] This paper systematically reviews 134 papers on evidence-based text generation with LLMs. It proposes a unified taxonomy (Attribution Approach × Citation Charact…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Evidence-based text generation"
  - "Citation attribution"
  - "LLM trustworthiness"
  - "Evaluation framework"
  - "RAG"
date: 2026-05-08
content_hash: 7ddca41f7c947ff5
---

# Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2508.15396](https://arxiv.org/abs/2508.15396)  
**Code**: [https://github.com/faerber-lab/AttributeCiteQuote](https://github.com/faerber-lab/AttributeCiteQuote)  
**Area**: Survey/NLP  
**Keywords**: Evidence-based text generation, Citation attribution, LLM trustworthiness, Evaluation framework, RAG

## TL;DR

This paper systematically reviews 134 papers on evidence-based text generation with LLMs. It proposes a unified taxonomy (Attribution Approach × Citation Characteristics × Task) for the first time, analyzes 300 evaluation metrics categorized into seven dimensions and six methods, and provides a panoramic reference framework for this fragmented field.

## Background & Motivation

**Background**: LLMs face trustworthiness challenges such as hallucinations and knowledge limitations. Increasing research focus has shifted toward "evidence-based text generation"—enabling LLM outputs to be traceable to supporting evidence. However, the field is highly fragmented: some refer to it as "citation" (used by 75% of papers), others as "attribution" (62%), or "quotation" (13%), with isolated evaluation practices.

**Limitations of Prior Work**: (1) Absence of unified terminology and classification systems, making it difficult for researchers to position their work; (2) Inconsistent evaluation standards—among 300 identified metrics, only 2 frameworks (ALCE, G-Eval) are reused across multiple papers; (3) While RAG is popular, it is only one of seven relevant methodologies; over-focusing on RAG neglects other critical approaches.

**Key Challenge**: Rapidly growing research interest (the number of papers in 2024 is 3.4 times that of 2023, with 75% published after 2023) versus the lack of a unified perspective to integrate and compare different methods.

**Goal**: To provide the first systematic survey specifically targeting LLM evidence-based text generation, establish a unified taxonomy, analyze evaluation practices, and identify research trends and future directions.

**Key Insight**: A systematic mapping study was conducted according to the PRISMA protocol, filtering 134 relevant papers from 805 deduplicated candidates, using a multi-faceted classification approach to construct the taxonomy.

**Core Idea**: Unify "citation," "attribution," and "quotation" into an "evidence-based text generation" paradigm. By providing a three-dimensional taxonomy and a seven-dimension evaluation framework, the paper offers a systematic perspective for this fragmented domain.

## Method

### Overall Architecture

This paper proposes a three-dimensional independent taxonomy to capture the core design choices in evidence-based text generation:

- **Dimension 1: Attribution Approach** — How content is associated with evidence (Parametric vs. Non-parametric).
- **Dimension 2: Citation Characteristics** — The form and presentation of evidence (Modality, Granularity, Style, Visibility, Frequency).
- **Dimension 3: Task** — Application scenarios (six categories including QA, Grounded Generation, Summarization, etc.).

### Key Designs

1. **Attribution Approach**:
    - **Function**: Describes how the LLM associates generated content with supporting evidence.
    - **Core Idea**: **Parametric** (25 papers) — Pure LLM (leveraging existing capabilities, 72% of parametric), Model-centric (modifying architecture/training), Data-centric (curating/augmenting data). **Non-parametric** (126 papers) — Post-retrieval (58%, represented by RAG), Post-generation (18%, generating then retrieving evidence), In-generation (4%, dynamically deciding if retrieval is needed), In-context (20%, evidence provided directly by the user).
    - **Key Findings**: Parametric attribution is significantly under-researched, especially model-centric and data-centric methods. Post-retrieval dominates the non-parametric category, but in-generation attribution (e.g., Self-RAG) is a promising but undervalued direction.

2. **Citation Characteristics**:
    - **Function**: Describes the specific presentation of evidence.
    - **Core Idea**: **Citation Modality** — Text 96%, Charts, Tables, Visual. **Evidence Level** — Document-level 43%, Paragraph-level 40%, Sentence-level 12%, Token-level 2%. **Citation Style** — Inline citation 62%, Citation reports, Paragraph display, Narrative citation, Highlight gradients, Quotation. **Visibility** — Final response 91% vs. Intermediate text. **Frequency** — Multiple citations 64% vs. Single citation.
    - **Key Findings**: Non-text evidence modalities (charts, tables, images) are severely under-explored (only 4%); fine-grained evidence (sentence or token level) accounts for a small percentage but is growing rapidly.

3. **Task Landscape**:
    - **Function**: Maps the application scenarios of the field.
    - **Core Idea**: QA and Grounded Text Generation are the dominant tasks. Summarization and Fact Verification are moderate, while Citation Text Generation and Related Work Generation are emerging tasks.
    - **Key Findings**: Evaluation practices are primarily developed around QA tasks and may not be suitable for emerging tasks (e.g., Citation Text Generation requires more evaluation regarding the reasoning behind citation selection).

### LLM Integration

**Training** (Used in 45% of papers): Primarily Supervised Fine-Tuning (SFT) used to improve attribution behavior. Pre-training is rarely used.
**Prompting** (Used in 78% of papers): Primarily Zero/Few-shot prompting. Specialized strategies for citation behavior include chain-of-citation, chain-of-quote, and conflict-aware prompting.

## Key Experimental Results

### Evaluation Metrics System

**300 metrics categorized into seven evaluation dimensions**

| Evaluation Dimension | When to Use | Primary Method | Representative Metrics (Reuse Count) |
|---------|---------|---------|-------------------|
| Attribution | No labeled evidence | NLI-based | Citation NLI P/R/F1 (33/33/16), Auto-AIS (11), FActScore (7) |
| Citation | Labeled evidence exists | Retrieval-based | Citation Retrieval P/R/F1 (6/6/5), Citation Accuracy (2) |
| Correctness | Always required | Lexical overlap/NLI | Exact Match (12), BLEU-N (5), Claim Recall (17) |
| Language Quality | When model is modified | LLM-as-Judge | G-Eval Fluency (4), MAUVE (21), Perplexity (4) |
| Preservation | For post-generation attribution | Lexical overlap | Preservation-Levenshtein (3), F1-AP (2) |
| Relevance | User scenarios | LLM-as-Judge | G-Eval Relevance (3), RAGAS (2) |
| Retrieval | Non-parametric attribution | Retrieval metrics | P@k (4), R@k (4), MRR (3) |

### Evaluation Guidelines

| Dimension Category | When to Evaluate | Description |
|---------|---------|------|
| **Core Dimensions** | Attribution or Citation + Correctness | Correctness should always be evaluated; choose between Attribution or Citation based on evidence availability. |
| **Contextual Dimensions** | Language Quality, Preservation, Relevance, Retrieval | Depends on the task design and system architecture. |

### Key Findings

- Only 2 frameworks (ALCE, G-Eval) and 2 benchmarks are reused across multiple papers, indicating a severe lack of evaluation standardization.
- Identified 19 frameworks, 11 benchmarks, and 231 datasets across 134 papers.
- Text accounts for 96% of citation modalities; multimodal evidence is nearly a blank space.
- Parametric attribution is critically neglected despite being essential for understanding internal model knowledge and data provenance.
- Human evaluation still dominates the Correctness dimension, reflecting the limitations of automatic metrics in capturing semantic errors.

## Highlights & Insights

- Unifying "citation," "attribution," and "quotation" into "evidence-based text generation" is a significant conceptual contribution that resolves long-standing terminological confusion.
- The seven-dimension evaluation guide (Table 1) provides clear recommendations for practitioners—the distinction between core and contextual dimensions is highly practical.
- The tripartite classification of parametric attribution (Pure LLM / Model-centric / Data-centric) is more granular than previous binary classifications.
- Identifies in-generation attribution as a promising but undervalued direction—representing only 4% but suggesting a trend toward tighter integration of retrieval and generation.
- Points out that citation behavior may exhibit biases similar to human authors, calling for research into the interpretability of LLM citation reasoning.

## Limitations & Future Work

- A single search string may have missed some relevant studies (sensitivity analysis indicated only 4% additional findings).
- Coverage is limited to English papers, which may underestimate non-English research.
- Manual screening and classification inevitably introduce some subjectivity.
- **Four Future Directions**: (1) In-depth research on parametric and hybrid attribution; (2) Standardized evaluation frameworks (currently 300 metrics but only 2 frameworks reused); (3) Interpretable citation reasoning—understanding why LLMs select specific sources; (4) Support for multimodal evidence—expanding from 96% text to include charts, tables, and images.

## Related Work & Insights

- **vs. Li et al. (2023a)**: The only prior relevant survey, but it is now significantly outdated (75%+ of papers were published after it) and does not cover the complete paradigm.
- **vs. Huang & Chang (2024)**: A position paper that emphasizes the importance of citation but is not a systematic survey.
- **vs. RAG surveys**: RAG surveys only cover post-retrieval methods, whereas this paper covers seven distinct attribution approaches.
- **vs. Hallucination/Grounding surveys**: These focus on different facets; this paper focuses on evidence generation rather than detection.

## Rating

- Novelty: ⭐⭐⭐⭐ First comprehensive and unified classification of evidence-based text generation; well-designed three-dimensional taxonomy.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive coverage of 134 papers, 300 metrics, 19 frameworks, 231 datasets, and 11 benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with balanced multi-dimensional analysis; each section includes a refined "Takeaways" summary.
- Value: ⭐⭐⭐⭐⭐ A panoramic synthesis of a fast-growing but fragmented field, serving as a vital reference for researchers and practitioners.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/llm_evaluation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)

</div>

<!-- RELATED:END -->
