---
title: >-
  [Paper Note] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG
description: >-
  [ACL 2026][Information Retrieval & RAG][Multilingual RAG] This paper systematically reveals severe language bias (favoring English and the query language) in the reranking stage of multilingual RAG systems…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Multilingual RAG"
  - "reranking bias"
  - "language fairness"
  - "evidence selection"
  - "cross-lingual retrieval"
date: 2026-05-08
content_hash: cf2a8615d951e221
---

# All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG

**Conference**: ACL 2026
**arXiv**: [2604.20199](https://arxiv.org/abs/2604.20199)  
**Code**: None  
**Area**: Information Retrieval / Multilingual NLP
**Keywords**: Multilingual RAG, reranking bias, language fairness, evidence selection, cross-lingual retrieval

## TL;DR
This paper systematically reveals severe language bias (favoring English and the query language) in the reranking stage of multilingual RAG systems, and proposes the LAURA framework, which aligns the reranker via supervision signals driven by downstream generation quality, effectively mitigating bias and improving generation performance.

## Background & Motivation

**Background**: Multilingual RAG enhances the global knowledge coverage of LLMs by leveraging cross-lingual evidence. In the real world, much knowledge is documented exclusively in specific languages (e.g., regional policies, cultural contexts), so an ideal mRAG system should select the most informative documents across languages.

**Limitations of Prior Work**: Current mRAG systems exhibit pronounced language preference bias in the reranking stage. Taking the BGE reranker as an example, averaged across 13 languages, more than 70% of top-5 documents come from English and the query language. This means that even when high-quality evidence in other languages exists in the candidate pool, it is systematically suppressed in the rankings.

**Key Challenge**: The root of the problem is not a lack of relevant information in the candidate pool—Oracle experiments show that selecting the correct documents from already-retrieved candidates alone improves performance by 12.9–20 percentage points. The true bottleneck is that the reranker's language preference prevents it from identifying critical evidence in non-English/non-query languages. Furthermore, the Pearson correlation between reranking scores and downstream generation quality is below 0.2.

**Goal**: (1) Quantify and diagnose language bias in mRAG; (2) Design a method to align document selection by the reranker with downstream generation quality, rather than relying solely on semantic relevance signals.

**Key Insight**: An Oracle evidence estimation method is proposed—documents are independently reranked by language group and answers are generated per group, with the performance of the best language group serving as an upper bound, thereby precisely quantifying the performance loss caused by bias.

**Core Idea**: Replace semantic relevance with answer utility as the training signal for the reranker, eliminating the influence of language priors.

## Method

### Overall Architecture
LAURA consists of a two-stage data construction pipeline and listwise reranker fine-tuning. Stage 1 employs language-grouped debiased sampling; Stage 2 uses document-level utility estimation to select positive samples. The reranker is ultimately fine-tuned with a softmax cross-entropy loss.

### Key Designs

1. **Oracle Evidence Estimation Analysis Framework**:

    - Function: Quantify the theoretical performance upper bound of existing rerankers and the degree of bias.
    - Mechanism: For each query, candidate documents are grouped by language; within each group, the top-5 documents are independently reranked and used to generate answers. The score of the best-performing language group serves as the Oracle upper bound. Comparing the Oracle distribution with the actual distribution reveals that Oracle evidence is spread across multiple languages rather than concentrated in the query language.
    - Design Motivation: Disentangle two sources of bias—information being inherently richer in certain languages vs. insufficient cross-lingual capability of the reranker. Experiments confirm the latter.

2. **Language-Debiased Subset Selection (Stage 1)**:

    - Function: Construct a cross-lingually balanced candidate positive sample set.
    - Mechanism: Retrieved documents are grouped by language; within each group, the reranker selects the top-5, ensuring equal exposure opportunity for each language subset. Multiple generators (Qwen, Llama, DeepSeek, and one additional model, totaling 4) independently evaluate the generation quality of each document, and the average is taken as the utility score.
    - Design Motivation: Estimating utility directly on the global top-k would amplify existing language bias. Language grouping ensures the candidate pool is not dominated by high-resource languages.

3. **Document-Level Utility Estimation and Listwise Fine-Tuning (Stage 2)**:

    - Function: Precisely select genuinely useful documents and train the reranker.
    - Mechanism: Documents retained from Stage 1 are evaluated individually for generation utility. An absolute threshold $\theta=0.8$ is applied for filtering, retaining only documents that demonstrably assist in generating correct answers as positive samples. Listwise fine-tuning is performed with a softmax cross-entropy loss, encouraging the reranker to assign the highest scores to positive samples.
    - Design Motivation: An absolute threshold (rather than relative ranking) avoids introducing implicit language bias, ensuring positive sample selection is based entirely on answer quality.

### Loss & Training
A softmax cross-entropy loss is adopted: $\mathcal{L} = -s(q, d_{pos}) + \log \sum_{d \in \mathcal{D}_q} \exp(s(q,d))$. The BGE reranker uses 1 negative sample per query; the Qwen reranker uses 7. The AdamW optimizer is used with a learning rate of $6 \times 10^{-6}$ for 5 training epochs.

## Key Experimental Results

### Main Results

| Reranker | Setting | Avg 3-gram Recall (Llama) | Avg 3-gram Recall (Qwen) |
|---------|------|--------------------------|--------------------------|
| BGE | Original | 48.9 | 46.7 |
| BGE | + LAURA | **49.9** | **47.7** |
| Qwen3 | Original | 47.1 | 44.9 |
| Qwen3 | + LAURA | **49.2** | **46.7** |
| - | Oracle Upper Bound | 63.6 | 61.3 |

### Ablation Study

| Configuration | Llama 3-gram | Pearson Corr. | Notes |
|------|-------------|-------------|------|
| BGE Original | 48.9 | 0.198 | Baseline |
| Self-Training | 48.9 | 0.188 | Pseudo-labels reinforce existing bias |
| mMARCO Fine-tuning | 48.7 | 0.132 | General data cannot resolve specific distribution mismatch |
| LAURA | 49.9 | 0.236 | Utility-driven approach effectively improves alignment |

### Key Findings
- A gap of approximately 15 percentage points exists between the Oracle and actual performance, indicating that sufficiently strong evidence already exists in the candidate pool but is overlooked by the reranker.
- After LAURA training, the Pearson correlation of the Qwen reranker improves from 0.127 to 0.264 (+108%), significantly enhancing alignment between reranking scores and generation quality.
- The language distribution of reranker outputs after training more closely approximates the Oracle distribution, with JS divergence reduced from 0.203 to 0.090 (BGE).
- The proportion of English and query-language documents decreases, affording other languages fairer ranking opportunities.

## Highlights & Insights
- The Oracle evidence estimation framework is an elegant diagnostic tool that cleanly separates the factors of "lacking information" and "failing to select correctly." This analytical approach can be transferred to any scenario involving multi-source information selection.
- Using the average generation quality across multiple models as a document utility signal both reduces model-specific bias and avoids the cost of manual annotation.
- The low correlation (<0.2) between reranking scores and generation quality is an important finding, demonstrating that "semantic relevance" and "answer utility" are fundamentally distinct concepts for current rerankers.

## Limitations & Future Work
- The Oracle upper bound remains an estimate rather than a true upper bound, potentially under- or overestimating achievable performance.
- Validation is conducted solely on the MKQA dataset, limiting coverage of languages and domains.
- LAURA requires multiple generative models for utility estimation, resulting in relatively high data construction costs.
- Future work may explore lightweight utility estimation methods or directly optimize the reranker online via reinforcement learning.

## Related Work & Insights
- **vs. Traditional Multilingual Retrieval**: Traditional approaches focus on retrieval recall, whereas this paper identifies reranking as the true bottleneck.
- **vs. mMARCO Fine-tuning**: General ranking data cannot resolve the specific language bias present in mRAG.
- **vs. Translation-based mRAG**: Translation strategies sidestep the problem of multilingual evidence selection; this paper directly optimizes the selection mechanism.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the Oracle analysis framework and utility-driven alignment represent meaningful novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 13 languages, multiple rerankers, multiple generators, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem analysis proceeds in a layered, progressive manner with a clear diagnosis-to-treatment logic.
- Value: ⭐⭐⭐⭐ Reveals the overlooked reranking bias problem in mRAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
