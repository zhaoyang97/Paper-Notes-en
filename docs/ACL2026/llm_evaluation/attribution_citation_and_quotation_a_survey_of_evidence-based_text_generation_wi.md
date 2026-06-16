---
title: >-
  [Paper Note] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models
description: >-
  [ACL 2026][LLM Evaluation][RAG] This paper systematically reviews 134 papers on evidence-based text generation with LLMs, proposing a unified taxonomy (attribution mode $\times$ citation features $\times$ task) for the first time. It analyzes 300 evaluation metrics categorized into seven dimensions and six methods, providing a comprehensive reference
tags:
  - ACL 2026
  - LLM Evaluation
  - RAG
date: 2026-05-08
content_hash: 821f61d043dfedd7
---
# Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2508.15396](https://arxiv.org/abs/2508.15396)  
**Code**: [https://github.com/faerber-lab/AttributeCiteQuote](https://github.com/faerber-lab/AttributeCiteQuote)  
**Area**: Survey/NLP  
**Keywords**: Evidence-based text generation, citation attribution, LLM trustworthiness, evaluation frameworks, RAG

## TL;DR

This paper systematically reviews 134 papers on evidence-based text generation with LLMs, proposing a unified taxonomy (attribution mode $\times$ citation features $\times$ task) for the first time. It analyzes 300 evaluation metrics categorized into seven dimensions and six methods, providing a comprehensive reference framework for this fragmented field.

## Background & Motivation

**Background**: LLMs face reliability challenges such as hallucinations and knowledge limitations. An increasing number of studies focus on "evidence-based text generation"—ensuring LLM outputs are traceable to supporting evidence. However, the field is highly fragmented: terms like "citation" (used by 75% of papers), "attribution" (62%), and "quotation" (13%) are used interchangeably, and evaluation practices remain isolated.

**Limitations of Prior Work**: (1) Lack of unified terminology and taxonomy, making it difficult for researchers to position their work; (2) Inconsistent evaluation standards—among 300 metrics, only two frameworks (ALCE, G-Eval) are reused across multiple papers; (3) While RAG is popular, it is only one of seven relevant methods, and overfocusing on RAG overlooks other critical approaches.

**Key Challenge**: The rapid growth in research interest (2024 paper count is 3.4x that of 2023, with 75% published after 2023) vs. the lack of a unified perspective to integrate and compare different methods.

**Goal**: To provide the first systematic survey specifically targeting LLM evidence-based text generation, establishing a unified taxonomy, analyzing evaluation practices, and identifying research trends and future directions.

**Key Insight**: Conduct a systematic mapping study following the PRISMA protocol, filtering 134 relevant papers from 805 unique records and constructing a taxonomy using a multi-faceted classification approach.

**Core Idea**: Unify "citation," "attribution," and "quotation" into an "evidence-based text generation" paradigm. Provide a systematic perspective for this fragmented field through a three-dimensional taxonomy and a seven-dimension evaluation framework.

## Method

### Overall Architecture

The paper follows the PRISMA protocol for a systematic mapping study: it filters 134 relevant works from 805 deduplicated papers and encodes each using a faceted classification method. The core output is a 3D independent taxonomy that combines "Attribution Mode (how content relates to evidence) $\times$ Citation Features (how evidence is presented) $\times$ Task (application scenario)." Any evidence-based text generation work can be positioned within this cube. An additional cross-cutting perspective of "LLM Integration Mode" (training vs. prompting) is layered on top to answer how models acquire attribution capabilities.

### Key Designs

**1. Attribution Mode: Fine-grained dichotomy of Parametric vs. Non-parametric.** This dimension describes the fundamental path an LLM takes to associate generated content with supporting evidence. **Parametric** (25 papers) integrates evidence into model weights, further subdivided into Pure LLM (leveraging existing capabilities, 72% of parametric), Model-centric (arhcitecture/training changes), and Data-centric (data curation/augmentation). **Non-parametric** (126 papers) keeps evidence outside the weights, categorized by retrieval timing: Post-retrieval (58%, e.g., RAG), Post-generation (18%, generating first then finding evidence), In-generation (4%, dynamically deciding when to retrieve), and In-context (20%, evidence provided directly in the prompt). This classification reveals a significant neglect of parametric methods, while in-generation attribution (e.g., Self-RAG), though only 4%, represents a frontier for tighter retrieval-generation coupling.

**2. Citation Features: Five facets characterizing evidence appearance.** The same evidence can be presented differently. These facets include: Citation Modality (96% text; graphs/tables/vision are nearly blank), Evidence Level (Document 43%, Paragraph 40%, Sentence 12%, Token 2%), Citation Style (Inline 62%, and reports, snippets, narrative, etc.), Visibility (Final response 91% vs. intermediate text), and Frequency (Multiple 64% vs. single). These facets highlight structural gaps: non-text modalities are severely underdeveloped (4%), and fine-grained evidence (sentence/token level) is the fastest-growing trend.

**3. Task: Mapping six application scenarios.** The third dimension maps tasks, finding that QA and grounded text generation dominate. Summary and fact-checking are intermediate, while citation text generation and related work generation are emerging. This reveals a path dependency in evaluation: existing metrics are largely designed for QA and may not suit emerging tasks that require reasoning about "why" a citation was chosen.

Additionally, the **LLM Integration Mode** cross-cuts these dimensions: Training (45% of papers, primarily SFT for attribution behavior) and Prompting (78% of papers, primarily zero/few-shot, with specialized strategies like chain-of-citation or conflict-aware prompting).

## Key Experimental Results

### Evaluation Metric System

**300 metrics categorized into seven evaluation dimensions**

| Evaluation Dimension | When to Use | Main Method | Representative Metrics (Reuse Count) |
| :--- | :--- | :--- | :--- |
| Attribution | No labeled evidence | NLI-based | Citation NLI P/R/F1 (33/33/16), Auto-AIS (11), FActScore (7) |
| Citation | Labeled evidence exists | Retrieval-based | Citation Retrieval P/R/F1 (6/6/5), Citation Accuracy (2) |
| Correctness | Always required | Lexical/NLI | Exact Match (12), BLEU-N (5), Claim Recall (17) |
| Language Quality | Model is modified | LLM-as-Judge | G-Eval Fluency (4), MAUVE (21), Perplexity (4) |
| Preservation | Post-generation attribution | Lexical overlap | Preservation-Levenshtein (3), F1-AP (2) |
| Relevance | User scenarios | LLM-as-Judge | G-Eval Relevance (3), RAGAS (2) |
| Retrieval | Non-parametric | Retrieval metrics | P@k (4), R@k (4), MRR (3) |

### Evaluation Guidelines

| Dimension Category | When to Evaluate | Description |
| :--- | :--- | :--- |
| **Core Dimensions** | Attribution/Citation + Correctness | Correctness should always be evaluated; choose Attribution or Citation based on evidence availability. |
| **Context Dimensions** | Quality, Preservation, Relevance, Retrieval | Depends on task design and system architecture. |

### Key Findings

- Evaluation standardization is severely lacking; only 2 frameworks (ALCE, G-Eval) and 2 benchmarks are reused across multiple papers.
- Identified 19 frameworks, 11 benchmarks, and 231 datasets across 134 papers.
- Text accounts for 96% of citation modalities; multimodal evidence is nearly absent.
- Parametric attribution is crucial for understanding internal knowledge and data provenance but is heavily neglected.
- Human evaluation still dominates the "Correctness" dimension, reflecting limitations of automated metrics in capturing semantic errors.

## Highlights & Insights

- Unifying "Citation," "Attribution," and "Quotation" into "Evidence-based Text Generation" is a major conceptual contribution that resolves long-standing terminological confusion.
- The 7-dimension evaluation guide (Table 1) provides practitioners with clear recommendations for metric selection—the distinction between core and context dimensions is highly practical.
- The tripartite division of parametric attribution (Pure LLM / Model-centric / Data-centric) is more granular than previous binary classifications.
- Identified in-generation attribution as a promising but undervalued direction (only 4% of papers), representing the trend toward integrated retrieval-generation.
- Noted that citation behaviors may exhibit biases similar to human authors, calling for research into the interpretability of LLM citation reasoning.

## Limitations & Future Work

- A single search string might miss some relevant studies (sensitivity analysis showed only 4% additional findings).
- Coverage is limited to English papers, potentially underestimating non-English research.
- Manual screening and classification inevitably introduce some subjectivity.
- **Four Future Directions**: (1) In-depth study of parametric and hybrid attribution; (2) Standardized evaluation frameworks; (3) Explainable citation reasoning—understanding why LLMs select specific sources; (4) Multimodal evidence support—expanding from 96% text to charts, tables, and images.

## Related Work & Insights

- **vs. Li et al. (2023a)**: The only prior relevant survey, now outdated (75%+ papers published after it) and covering an incomplete paradigm.
- **vs. Huang & Chang (2024)**: A position paper emphasizing citation importance without a systematic review.
- **vs. RAG surveys**: RAG surveys only cover post-retrieval methods; this paper covers seven attribution modes.
- **vs. Hallucination/Grounding surveys**: These focus on different aspects; this paper focuses on evidence generation rather than detection.

## Rating

- Novelty: ⭐⭐⭐⭐ First comprehensive unification and taxonomy for evidence-based text generation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive coverage of 134 papers, 300 metrics, 19 frameworks, and 231 datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with balanced multi-dimensional analysis and concise "Takeaways" for each section.
- Value: ⭐⭐⭐⭐⭐ Provides a panoramic overview and reference for a rapidly growing but fragmented field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dynamic Infilling Anchors for Format-Constrained Generation in Diffusion Large Language Models](dynamic_infilling_anchors_for_format-constrained_generation_in_diffusion_large_l.md)
- [\[ACL 2026\] Comprehensiveness Metrics for Automatic Evaluation of Factual Recall in Text Generation](comprehensiveness_metrics_for_automatic_evaluation_of_factual_recall_in_text_gen.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task](e2edev_benchmarking_large_language_models_in_end-to-end_software_development_tas.md)

</div>

<!-- RELATED:END -->
