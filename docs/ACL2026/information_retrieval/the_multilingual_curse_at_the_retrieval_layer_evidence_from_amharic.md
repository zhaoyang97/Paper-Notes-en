---
title: >-
  [Paper Note] The Dilemma of Low-Resource Languages in Multilingual Retrieval: Evidence from Amharic
description: >-
  [ACL 2026][Information Retrieval & RAG][Multilingual Retrieval] This paper uses Amharic as a diagnostic case to reveal that powerful multilingual retrieval models fail to transfer effectively to morphologically rich low-…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Multilingual Retrieval"
  - "Low-Resource Languages"
  - "Amharic"
  - "Zero-Shot Transfer"
  - "Neural Information Retrieval"
date: 2026-05-08
content_hash: b6da2c744652ce99
---

# The Dilemma of Low-Resource Languages in Multilingual Retrieval: Evidence from Amharic

**Conference**: ACL 2026  
**arXiv**: [2605.24556](https://arxiv.org/abs/2605.24556)  
**Code**: https://github.com/rasyosef/amharic-neural-ir  
**Area**: Information Retrieval / Multilingual NLP  
**Keywords**: Multilingual Retrieval, Low-Resource Languages, Amharic, Zero-Shot Transfer, Neural Information Retrieval

## TL;DR

This paper uses Amharic as a diagnostic case to reveal that powerful multilingual retrieval models fail to transfer effectively to morphologically rich low-resource languages in zero-shot settings, exhibiting a 23% relative decline in MRR@10 performance. While language-specific fine-tuning improves performance by 32-60%, it still falls short of monolingual retrievers, indicating that multilingual retrieval is insufficient to guarantee equitable access to information for low-resource languages.

## Background & Motivation

**Background**: Multilingual retrieval has become a core component of cross-lingual Question Answering (QA) and Retrieval-Augmented Generation (RAG). Latest multilingual embedding models (e.g., E5, Arctic Embed) achieve strong zero-shot performance on multilingual benchmarks, leading researchers to generally believe these models can reliably transfer across languages.

**Limitations of Prior Work**: However, this assumption does not hold for morphologically rich, resource-scarce languages. Despite having 58 million speakers, Amharic performs poorly in multilingual retrieval. Multilingual tokenizers mismanage the root-pattern morphology, complex affixes, and non-Latin script (Ge'ez characters) of Amharic, leading to fragmented representations. The key problem is that aggregate high scores in multilingual benchmarks mask severe degradation at the monolingual level.

**Key Challenge**: While zero-shot multilingual retrieval performs well on average, there may be a hidden quality ceiling at the retrieval layer for low-resource languages. If RAG systems are built on zero-shot multilingual retrievers, they will inherit this ceiling, affecting downstream generation quality.

**Goal**: (1) Quantify the performance gap of zero-shot multilingual retrieval in low-resource languages; (2) Evaluate the upper bound of gains from language-specific fine-tuning; (3) Establish a unified evaluation framework to compare four mainstream retrieval paradigms.

**Key Insight**: Amharic is chosen as a diagnostic case not only because of its large speaker population but also because it simultaneously possesses multiple features that cause retrieval difficulties—non-Latin script, complex morphology, and limited multilingual pre-training coverage—features common to hundreds of languages globally.

**Core Idea**: By comparing three categories of retrievers (zero-shot multilingual, fine-tuned multilingual, and monolingual) under a rigorous shared protocol, the study quantifies the true gap in retrieval performance for low-resource languages and argues that aggregated multilingual scores cannot replace in-depth language-level evaluation.

## Method

### Overall Architecture

The paper adopts a "three-layer contrastive" architecture: first, the Amharic Dataset V2 is established (68K query-document pairs from four sources: news, summaries, Wikipedia, and QA). Then, the three categories of retrievers are evaluated under a unified protocol. Finally, a systematic comparison is conducted across four retrieval paradigms (dense bi-encoder, late interaction, learned sparse, cross-encoder). The core evaluation pipeline consists of: (1) Stage 1: Initial ranking using various retrievers to calculate Recall@k, MRR@10, and NDCG@10; (2) Stage 2: Reranking the top 50 candidates using a cross-encoder to observe the gains from joint scoring.

### Key Designs

1.  **Multi-source Dataset Construction and Weak Supervision Labels**:
    *   Function: Constructs the Amharic Passage Retrieval Dataset V2, containing 68K query-document pairs derived from AMNEWS (headlines to body), XL-SUM (summary data), Amharic Wikipedia, and AmQA (QA pairs), with MD5 deduplication.
    *   Mechanism: Uses source-aligned weak supervision labels where each query has only one labeled positive document. This single-positive evaluation is conservative and reflects real-world scenarios of incomplete labeling. Binary relevance judgment ensures the monotonicity of evaluation.
    *   Design Motivation: Multi-source fusion diversifies the data and avoids bias from a single news source; weak supervision reduces labeling costs, though reliability requires careful interpretation.

2.  **Unified Evaluation Protocol for Four Paradigms**:
    *   Function: Compares dense bi-encoders, late interaction (ColBERT), learned sparse (SPLADE), and cross-encoders under shared data and metrics. Each paradigm includes Medium and Base variants.
    *   Mechanism: Dense bi-encoders use MultipleNegativesRankingLoss and Matryoshka representation learning; late interaction utilizes token-level MaxSim scoring (query limit 32 tokens, document limit 256 tokens); sparse retrieval uses SPLADE via vocabulary pooling combined with sparsity regularization; cross-encoders jointly encode query-document pairs with weighted BCE loss (positive weight 7) to handle class imbalance.
    *   Design Motivation: The four paradigms represent the evolution from sparse (BM25) to dense to hybrid retrieval. A unified framework enables fair comparison and avoids false differences caused by disparate training setups.

3.  **Controlled Experiments: Zero-shot vs. Fine-tuning**:
    *   Function: Maintains consistent fine-tuning data (68K training set) and methods (SentenceTransformers + MNR + Matryoshka) while fine-tuning two recent multilingual models (EmbeddingGemma, Harrier) on Amharic to isolate variables.
    *   Mechanism: Fine-tuning uses a learning rate of $4\times 10^{-5}$, warmup of 0.025, mixed precision BF16, 6 epochs, and a batch size of 128. From pre-mined negatives, the two most similar and two least similar samples are selected. This strategy ensures the model prioritizes learning difficult samples under resource constraints.
    *   Design Motivation: By fixing supervision data and optimization settings, the specific contribution of "multilingual initialization" relative to "monolingual initialization" is isolated, accurately quantifying the differences in the initialization space.

### Loss & Training

Monolingual Amharic models: Dense bi-encoders use MultipleNegativesRankingLoss (with 4 pre-mined negatives) + Matryoshka Loss, with early stopping based on validation NDCG@10. Late interaction and sparse retrieval are trained for 4-6 fixed epochs. Cross-encoders use weighted BCE with a positive sample weight of 7. Evaluation metrics: Recall@5/10, MRR@10, and NDCG@10.

## Key Experimental Results

### Main Results: Stage 1 Retrieval Comparison

| Model | Params (M) | R@5 | R@10 | MRR@10 | NDCG@10 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BM25 | – | 0.734 | 0.789 | 0.612 | 0.655 |
| embedding-gemma-300m (Zero-shot) | 300 | 0.558 | 0.621 | 0.448 | 0.489 |
| harrier-oss-v1-270m (Zero-shot) | 270 | 0.697 | 0.753 | 0.576 | 0.619 |
| multilingual-e5-large (Zero-shot) | 560 | 0.736 | 0.791 | 0.603 | 0.648 |
| snowflake-arctic (Zero-shot, Best) | 568 | 0.795 | 0.848 | 0.653 | 0.701 |
| embedding-gemma + Finetune | 300 | 0.813 | 0.862 | 0.718 | 0.753 |
| harrier + Finetune | 270 | 0.860 | 0.903 | 0.760 | 0.795 |
| ColBERT-Base-Amharic (Monolingual, Best) | 110 | 0.902 | 0.930 | **0.803** | **0.835** |
| Embed-Base-Amharic (Monolingual) | 110 | 0.870 | 0.907 | 0.774 | 0.807 |
| SPLADE-Base-Amharic (Monolingual) | 110 | 0.871 | 0.906 | 0.754 | 0.792 |

**Key Findings**: (1) Gap between zero-shot multilingual and monolingual: The strongest zero-shot model, Arctic, reaches an MRR@10 of 0.653, while the monolingual ColBERT reaches 0.803, representing a **relative decline of 23.0%**; (2) Parameters are not the solution: Arctic has 568M parameters but is outperformed by a 110M monolingual model; (3) Limited fine-tuning gains: Gemma improved from 0.448→0.718 (+60.3%) and Harrier from 0.576→0.760 (+32.0%), but fine-tuned Harrier is still 5.4% lower than the monolingual model; (4) Consistency across paradigms: Monolingual models for dense, late interaction, and sparse paradigms all surpass the strongest zero-shot multilingual model.

### Ablation Study: Two-Stage Reranking Results

| Model | MRR@10 | NDCG@10 | Gain |
| :--- | :--- | :--- | :--- |
| Embed-Base-Amharic (Stage 1) | 0.774 | 0.807 | – |
| + Re-rank-Medium-Amharic | 0.805 | 0.835 | +4.0% / +3.5% |
| + Re-rank-Base-Amharic | 0.830 | 0.856 | +7.2% / +6.1% |

Cross-encoder reranking achieved the highest overall score (MRR@10: 0.830), indicating that joint query-document encoding can capture subtle ranking distinctions missed by independent bi-encoders.

### Key Findings & Analysis

*   **Hidden Failure of Zero-shot Multilingualism**: While multilingual models perform well on average, significant degradation can occur for individual low-resource languages. The Amharic case shows that a 23% relative performance drop occurs in the top-10 MRR@10 region, directly impacting the quality of retrieval results seen by users.
*   **Necessity of Language-Specific Modeling**: Monolingual models across different retrieval paradigms outperform zero-shot multilingual models, suggesting the issue is not the choice of algorithm but rather the inability of shared multilingual representation spaces to fit the morphological features of Amharic.
*   **Partial Compensation via Fine-tuning**: Fine-tuning does unlock the learning potential of multilingual models for Amharic (Gemma gained most), but it cannot completely bridge the gap with monolingual models.
*   **Supplementary Role of Architecture Optimization**: Cross-encoder reranking provides a 7.2% improvement over dense bi-encoders, showing that once a sufficiently good initial ranking is obtained, more complex joint scoring methods can provide further improvements.

## Highlights & Insights

*   **Diagnostic Case Study**: The paper cleverly uses Amharic, which shares universal features (non-Latin script, complex morphology, resource scarcity) despite having millions of speakers, to transform a local problem into a systemic insight.
*   **Value of Shared Evaluation Protocols**: Comparing four paradigms under a unified framework demonstrates that the root of the problem lies in representation space rather than architecture. This avoids the pitfall of comparing independently trained models from different papers.
*   **Refined Fine-tuning Design**: By fixing supervision data and training strategies, the study minimizes fine-tuning variance, making the comparative conclusions more robust.
*   **Real-world Significance for RAG**: The paper explicitly points out that retrieval layer failures become a quality ceiling for downstream generation, which aggregate scores of final answer quality might obscure.

## Limitations & Future Work

*   **Limited Monolingual Evidence**: The study focuses only on Amharic. While its features are common, the exact magnitude of the performance gap may vary by language.
*   **Weak Supervision Constraints**: The dataset uses source-aligned single-positive labeling, which might miss other relevant documents.
*   **Scope of Fine-tuning**: Only two recent multilingual models were fine-tuned. Multilingual models with larger scales or different instruction-tuning approaches might perform differently.
*   **Lack of End-to-End RAG Evaluation**: The paper infers RAG quality degradation based on retrieval metrics but does not verify this on actual generation tasks.

**Future Directions**: (1) Increase data weighting for low-resource, morphologically rich languages during multilingual pre-training; (2) Design morphology-aware tokenizers; (3) Maintain public retrieval benchmarks for all languages, not just high-resource ones; (4) Embed language-specific retrieval adapter layers in LLM applications.

## Related Work & Insights

*   **vs. Multilingual Benchmarks (MIRACL/mMARCO)**: These provide aggregate evaluations but mask language-level differences. The key innovation here is the shift to language-specific deep evaluation.
*   **vs. Mekonnen et al. (2025)**: Prior work showed monolingual Amharic models outperform zero-shot multilingual ones. This paper extends those observations—validating on larger datasets, adding learned sparse and cross-encoder paradigms, and systematically studying the upper bounds of multilingual fine-tuning.
*   **vs. Multilingual Pre-training Work**: Papers for E5, Arctic, etc., often report average multilingual scores. This paper reveals significant disparities hidden behind those averages through monolingual deep dives.
*   **Insights**: Multilingual system evaluations must drill down to the monolingual level. Issues in underserved languages are often not about algorithm choice, but about initialization and representation space. Parameter scale and high benchmark scores are no substitute for language-specific stress testing.

## Rating

*   Novelty: ⭐⭐⭐⭐ The choice of a diagnostic case study is clever. The "hidden failure" phenomenon in multilingual retrieval is easily overlooked, though the experimental design represents incremental improvement rather than fundamental innovation.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systemic comparison across four paradigms, three model types, and a two-stage process, combined with clear ablation and analysis, provides compelling evidence.
*   Writing Quality: ⭐⭐⭐⭐⭐ Logical flow, with a strong progression from problem definition to design, results, and implications.
*   Value: ⭐⭐⭐⭐ Important warnings and guidance for multilingual RAG/QA system deployment and multilingual model evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG](all_languages_matter_understanding_and_mitigating_language_bias_in_multilingual_.md)
- [\[ACL 2026\] Reliable Evaluation Protocol for Low-Precision Retrieval](reliable_evaluation_protocol_for_low-precision_retrieval.md)
- [\[ACL 2026\] CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](coral_adaptive_retrieval_loop_for_culturally-aligned_multilingual_rag.md)
- [\[ACL 2026\] GLIER: Generative Legal Inference and Evidence Ranking for Legal Case Retrieval](glier_generative_legal_inference_and_evidence_ranking_for_legal_case_retrieval.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)

</div>

<!-- RELATED:END -->
