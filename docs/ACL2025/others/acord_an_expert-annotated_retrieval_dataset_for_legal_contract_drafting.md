---
title: >-
  [Paper Note] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval
description: >-
  [ACL 2025][Contract Clause Retrieval] Builds the first expert-annotated clause retrieval benchmark for contract drafting, ACORD (114 queries, 126K+ pairs, 1-5 star ratings). Evaluating 20 retrieval methods reveals that BM25 + GPT-4o pointwise reranking performs best (NDCG@5 = 76.9%), but the accuracy for high-quality clauses is extremely low (5-star precision@5 is only 17.2%), highlighting a significant gap between models and the actual needs of lawyers.
tags:
  - "ACL 2025"
  - "Contract Clause Retrieval"
  - "Legal NLP"
  - "Expert Annotation"
  - "Information Retrieval"
  - "RAG"
date: 2026-05-08
content_hash: 261d87ca09612820
---

# ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval

**Conference**: ACL 2025  
**arXiv**: [2501.06582](https://arxiv.org/abs/2501.06582)  
**Code**: [GitHub](https://github.com/wang-steven-h/ACORD)  
**Area**: Others  
**Keywords**: Contract Clause Retrieval, Legal NLP, Expert Annotation, Information Retrieval, RAG

## TL;DR

Builds the first expert-annotated clause retrieval benchmark for contract drafting, ACORD (114 queries, 126K+ pairs, 1-5 star ratings). Evaluating 20 retrieval methods reveals that BM25 + GPT-4o pointwise reranking performs best (NDCG@5 = 76.9%), but the accuracy for high-quality clauses is extremely low (5-star precision@5 is only 17.2%), highlighting a significant gap between models and the actual needs of lawyers.

## Background & Motivation

**Background**: Contracts are the foundation of modern business—43% of corporate legal departments spend over half of their time drafting, editing, and negotiating contracts. Lawyers rarely draft contracts from scratch; instead, they retrieve and adapt precedent clauses. Clause retrieval is the core task of contract drafting.

**Limitations of Prior Work**: (1) The quality of contracts drafted directly by LLMs is unreliable—Table 1 in the paper clearly showcases draft deficiencies annotated by lawyers: inter-clause conflicts (e.g., "Notwithstanding" causing an exception clause to invalidate a limitation clause), non-standard language ("Application of Limitations" is rare in commercial contracts), and missing key concepts (using "paid" instead of "paid and payable"). (2) Clause retrieval faces unique challenges: multi-layered structures (sections, subsections, paragraphs, exceptions, cross-references that can span several pages), semantic complexity (relevance judgments are highly subjective, with a 21% inter-annotator disagreement rate), and high professional barriers. (3) Lack of domain benchmarks—most existing legal NLP datasets are in QA or classification formats (e.g., CUAD, LegalBench), without a dedicated benchmark for contract clause retrieval.

**Key Challenge**: Retrieval-Augmented Generation (RAG) is the key pathway to address the unreliability of LLM contract drafting, but RAG presupposes high-quality retrieval—yet there is currently a lack of benchmark datasets to evaluate the quality of contract clause retrieval.

**Goal**: Build the first expert-annotated clause retrieval benchmark for contract drafting to fill the gap in legal IR evaluation.

**Key Insight**: Collaborate deeply with lawyers to define retrieval tasks using real contract drafting workflows—queries are written by experienced lawyers, clauses are extracted from public SEC EDGAR contracts, and relevance is rated by a legal team.

**Core Idea**: By constructing the high-quality expert-annotated benchmark ACORD, provide a systematic evaluation tool for contract clause retrieval, a core task for lawyers.

## Method

### Overall Architecture

The ACORD construction pipeline consists of four stages: (1) lawyers author 114 retrieval queries covering 9 major clause categories; (2) a clause corpus is extracted from SEC EDGAR contracts and Fortune 500 Terms of Service (ToS); (3) annotators retrieve relevant clauses, which are then rated by three people (two lawyers and one annotator), with disagreements resolved by a 3-6 person legal committee; (4) the CUAD dataset is used to supplement 1-star irrelevant clauses to eliminate false negatives. The evaluation covers 20 methods, including BM25, bi-encoders, cross-encoders, and LLM rerankers.

### Key Designs

1. **Query Design and Clause Classification Taxonomy**:

    - Function: Ensure the benchmark covers the most common and complex clause types in contract drafting.
    - Mechanism: The 114 queries were authored by senior lawyers, covering 9 major categories—Limitation of Liability, Indemnification, Affirmative Covenants, Restrictive Covenants, Term & Termination, Governing Law, Change of Control, Intellectual Property (IP), and Most-Favored Nation (MFN). Each query corresponds to a specific drafting scenario, such as "liability cap is based on purchase price".
    - Design Motivation: Focus on "complex and heavily negotiated clauses" rather than simple boilerplate clauses, as these are the exact scenarios where lawyers most need to retrieve precedents.

2. **Multi-tiered Quality Control Annotation Process**:

    - Function: Ensure the annotation quality and consistency of 126K+ query-clause pairs.
    - Mechanism: A five-step annotation pipeline: (1) Extraction: annotators extract various clauses from the contract corpus; (2) Retrieval: retrieve 10 relevant clauses (3-5 stars) and 20 irrelevant ones (2 stars) for each query; (3) Rating: three people independently rate each pair (1-5 stars) using a detailed 4-page rating rubric; (4) Reconciliation: when the rating difference is > 2 stars or relevance is inconsistent, a 3-6 person legal committee reviews and reconciles; (5) Extension: supplement 1-star clauses with CUAD to eliminate false negatives. The overall annotation cost is estimated to exceed $1,000,000, calculated at a lawyer hourly rate of $550+ and non-lawyer rate of $150.
    - Design Motivation: Legal clause relevance judgment is inherently highly subjective (21% disagreement rate), and multi-tiered quality control guarantees data quality to the greatest extent.

3. **Task-Specific Evaluation Indicator Design**:

    - Function: Provide evaluation metrics that better reflect the actual needs of lawyers than NDCG.
    - Mechanism: In addition to standard NDCG@5/10, normalized x-star precision@5 is introduced—5-star precision@5 measures the proportion of top-quality clauses in the top-5 results (since most queries have < 5 five-star clauses, normalized to 0-1). The paper points out that NDCG is overly lenient—NDCG@5 = 76.9% seems usable, but 5-star precision@5 is only 17.2%, meaning there are almost no highest-quality clauses in the top-5.
    - Design Motivation: In contract drafting scenarios, lawyers do not need "roughly relevant" 3-star clauses, but rather "directly usable" 4-5-star clauses—measuring this requires more fine-grained metrics.

## Key Experimental Results

### Main Results: Comparison of Retrieval Methods

| Retriever | Reranker | NDCG@5 | NDCG@10 | 3-star prec@5 | 4-star prec@5 | 5-star prec@5 |
|--------|---------|--------|---------|----------|----------|----------|
| OpenAI Embed (large) | None | 62.1% | 64.1% | 58.6% | 38.9% | 11.0% |
| BM25 | None | 52.5% | 54.0% | 50.9% | 38.9% | 9.0% |
| BM25 | MiniLM Cross-Encoder | 59.3% | 60.9% | 60.0% | 43.5% | 6.2% |
| BM25 | GPT-4o | **76.9%** | **79.7%** | **81.1%** | **60.0%** | **17.2%** |
| BM25 | GPT-4o-mini | 75.2% | 78.2% | 78.6% | 58.2% | 18.6% |
| BM25 | Llama-3B | 62.6% | 65.3% | 63.9% | 48.1% | 9.7% |
| BM25 | Llama-1B | 13.8% | 14.4% | 13.0% | 10.5% | 4.1% |

### Ablation Study: Model Size and Reranking Strategy

| Dimension | Finding | Impact |
|---------|------|------|
| Model Scale | Llama 1B → 3B: Over 40%+ improvement in most metrics | Small models are entirely incapable of performing legal retrieval |
| Fine-tuning Effect | After fine-tuning MiniLM, NDCG@5 increased by 2.0%, and 5-star precision increased by 5.1% | Domain fine-tuning is effective but yields limited gains |
| Pointwise vs Pairwise | Pointwise outperforms Pairwise across all methods except Llama-1B | A finding contrary to conclusions in existing literature |
| Query Length | Long queries (with more context) significantly outperform short queries | Legal terms require contextual explanation |

### Key Findings

- **NDCG is a misleading metric**: NDCG@5 = 76.9% looks good, but 4-star precision@5 is only 60%, and 5-star precision@5 is only 17.2%—40% of the queries have a 4-star precision under 50%. In contract drafting, 3-star clauses may lead to contract quality issues.
- **Models are strong at understanding legal terminology but weak at ranking**: For "change of control", they can retrieve semantically equivalent concepts like "ownership changes" and "sale of substantially all assets", but fail to rank the highest-quality clauses at the very top.
- **Legal terms fail to retrieve without context**: The NDCG@5 for "as-is clause" is 0, but improves significantly when explanations are added.
- **Pointwise outperforms Pairwise contrary to existing literature**: This is likely because absolute quality judgments of legal clauses are more suited for a pointwise manner than relative pairwise comparisons.

## Highlights & Insights

- **Lawyer annotations of LLM drafting defects are highly convincing**: Table 1 does not present academic quantitative metrics, but actual issues annotated verbatim by lawyers (clause conflicts, non-standard language, missing key concepts), intuitively demonstrating why "retrieve-then-draft" is more reliable than "direct generation".
- **An investment of >$1M in annotation costs guarantees data quality**: In the era of fast-and-cheap AI, this "slow and meticulous" approach to building a benchmark sets a standard for AI evaluation in high-stakes domains.
- **The design of the 5-star precision@5 metric is a core innovation**: It exposes the true gap masked by NDCG—models can find "roughly relevant" clauses but fail to find "directly usable" high-quality clauses.

## Limitations & Future Work

- **Only covers English contracts**: Other legal systems/languages (such as civil law contracts in Germany or France) are not covered.
- **Clauses are pre-extracted from contracts**: This simplifies real-world scenarios—in practice, models must first extract relevant clauses from the entire contract and then rank them.
- **Limited coverage with 9 clause categories**: Common types like Representations & Warranties and payment clauses are not included.
- **Public SEC EDGAR contracts may not represent enterprise private contract repositories**: Although clause types are standardized, the depth and complexity of negotiation may differ.
- **21% inter-annotator disagreement rate**: Legal relevance is inherently subjective, but this also means some of the model's "errors" might be marginal cases where annotators did not reach a consensus.

## Related Work & Insights

- **vs CUAD (Hendrycks et al., 2021)**: CUAD is a large-scale, expert-annotated contract dataset (2000+ clauses, 41 categories), but is designed for clause classification and extraction (NER-style tasks) without evaluation on retrieval and ranking; ACORD fills the gap in the retrieval dimension.
- **vs BEIR (Thakur et al., 2021)**: BEIR is a general IR benchmark covering multiple domains; ACORD provides a specialized benchmark for legal contracts, with both task definition and evaluation metrics tailored to lawyers' needs.
- **vs BigLaw Bench (Harvey AI, 2024)**: BigLaw covers various legal tasks but has a much smaller scale of annotations; ACORD focuses purely on clause retrieval but is far more in-depth.

## Rating

- Novelty: ⭐⭐⭐⭐ The first expert-annotated benchmark for contract clause retrieval, filling an important gap in legal IR.
- Experimental Thoroughness: ⭐⭐⭐⭐ A comprehensive comparison of 20 methods, introducing task-specific metrics and in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The comparison between LLMs and lawyers is cleverly presented, and the motivation is convincingly argued.
- Value: ⭐⭐⭐⭐ Possesses direct practical value for legal AI practitioners; the open-source dataset can drive the development of the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)
- [\[ACL 2025\] MIR: Methodology Inspiration Retrieval for Scientific Research Problems](mir_methodology_inspiration_retrieval_for_scientific_research_problems.md)
- [\[ACL 2025\] What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma](what_is_stigma_attributed_to_a_theory-grounded_expert-annotated_interview_corpus.md)
- [\[ACL 2025\] Hard Negative Mining for Domain-Specific Retrieval in Enterprise Systems](hard_negative_mining_for_domain-specific_retrieval_in_enterprise_systems.md)

</div>

<!-- RELATED:END -->
