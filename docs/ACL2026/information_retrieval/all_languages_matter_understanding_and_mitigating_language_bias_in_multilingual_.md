---
title: >-
  [Paper Note] All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG
description: >-
  [ACL 2026][Information Retrieval & RAG][Multilingual RAG] This paper systematically reveals that multilingual RAG systems suffer from severe language bias during the re-ranking stage (preferring English and the query lan…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Multilingual RAG"
  - "Re-ranking bias"
  - "Language Fairness"
  - "Evidence Selection"
  - "Cross-lingual Retrieval"
date: 2026-05-08
content_hash: 9b93c19caa5f93b4
---

# All Languages Matter: Understanding and Mitigating Language Bias in Multilingual RAG

**Conference**: ACL 2026  
**arXiv**: [2604.20199](https://arxiv.org/abs/2604.20199)  
**Code**: None  
**Area**: Information Retrieval / Multilingual NLP  
**Keywords**: Multilingual RAG, Re-ranking bias, Language Fairness, Evidence Selection, Cross-lingual Retrieval

## TL;DR
This paper systematically reveals that multilingual RAG systems suffer from severe language bias during the re-ranking stage (preferring English and the query language). It proposes the LAURA framework, which aligns the reranker using supervision signals driven by downstream generation quality, effectively mitigating bias and improving generation performance.

## Background & Motivation

**Background**: Multilingual RAG (mRAG) enhances the global knowledge coverage of LLMs through cross-lingual evidence. In real-world scenarios, much knowledge is only recorded in specific languages (e.g., regional policies, cultural backgrounds). Therefore, an ideal mRAG system should select the most informative documents across languages.

**Limitations of Prior Work**: Current mRAG systems exhibit significant language preference bias during re-ranking. Using the BGE reranker as an example, across 13 languages, over 70% of the top-5 documents originate from English or the query language. This implies that even when high-quality evidence in other languages exists in the candidate pool, it is systematically ranked lower.

**Key Challenge**: The root cause is not a lack of relevant information in the candidate pool—Oracle experiments demonstrate that simply selecting the correct document from already retrieved candidates can improve performance by 12.9-20 percentage points. The true bottleneck is the reranker's language bias, which prevents it from identifying critical non-English/non-query evidence. Furthermore, the Pearson correlation between re-ranking scores and downstream generation quality is less than 0.2.

**Goal**: (1) Quantify and diagnose language bias in mRAG; (2) Design a method to align reranker document selection with downstream generation quality rather than relying solely on semantic relevance signals.

**Key Insight**: The authors propose an Oracle evidence estimation method—grouping candidates by language for independent re-ranking and answer generation, then using the best-performing language group as the upper bound to precisely quantify the performance loss caused by bias.

**Core Idea**: Replace semantic relevance with "answer utility" as the training signal for the reranker to eliminate the influence of language priors.

## Method

### Overall Architecture
LAURA consists of a two-stage data construction pipeline and listwise re-ranking fine-tuning. Stage 1 utilizes language-grouped debiased sampling, and Stage 2 employs document-level utility estimation to filter positive samples. Finally, the reranker is fine-tuned using a softmax cross-entropy loss.

### Key Designs

1.  **Oracle Evidence Estimation Analysis Framework**:
    - **Function**: Quantifies the theoretical performance upper bound and the degree of bias in existing rerankers.
    - **Mechanism**: For each query, candidate documents are grouped by language. Independent re-ranking and top-5 generation are performed within each group. The highest score among groups serves as the Oracle upper bound. Comparing the Oracle distribution with the actual distribution reveals that Oracle evidence is dispersed across multiple languages rather than concentrated in the query language.
    - **Design Motivation**: Distinguish between two sources of bias—inherent information richness in certain languages vs. insufficient cross-lingual capabilities of the reranker. Experiments confirm the latter.

2.  **Language-Debiased Subset Selection (Stage 1)**:
    - **Function**: Constructs a cross-lingually balanced set of candidate positive samples.
    - **Mechanism**: Retrieved documents are grouped by language, and the reranker selects the top-5 within each group to ensure equal exposure for all languages. Then, multiple generators (4 models including Qwen, Llama, and DeepSeek) independently evaluate the generation quality of each document. The average is taken as the utility score.
    - **Design Motivation**: Estimating utility directly on global top-k results would amplify existing language bias. Language grouping ensures the candidate pool is not dominated by high-resource languages.

3.  **Document-level Utility Estimation and Listwise Fine-tuning (Stage 2)**:
    - **Function**: Finely filters truly useful documents and trains the reranker.
    - **Mechanism**: Documents retained in Stage 1 are individually evaluated for generation utility. An absolute threshold $\theta=0.8$ is used to filter and retain only those documents that genuinely assist in generating correct answers as positive samples. Listwise fine-tuning is performed using softmax cross-entropy loss to encourage the reranker to assign the highest scores to positive samples.
    - **Design Motivation**: Using an absolute threshold (rather than relative ranking) avoids introducing implicit language bias and ensures positive sample selection is based purely on answer quality.

### Loss & Training
The model is trained using a softmax cross-entropy loss: 
$$\mathcal{L} = -s(q, d_{pos}) + \log \sum_{d \in \mathcal{D}_q} \exp(s(q,d))$$
For the BGE reranker, 1 negative sample per query is used, while the Qwen reranker uses 7. The AdamW optimizer is employed with a learning rate of $6 \times 10^{-6}$ for 5 epochs.

## Key Experimental Results

### Main Results

| Reranker | Setting | Avg 3-gram Recall (Llama) | Avg 3-gram Recall (Qwen) |
| :--- | :--- | :--- | :--- |
| BGE | Original | 48.9 | 46.7 |
| BGE | + LAURA | **49.9** | **47.7** |
| Qwen3 | Original | 47.1 | 44.9 |
| Qwen3 | + LAURA | **49.2** | **46.7** |
| - | Oracle Upper Bound | 63.6 | 61.3 |

### Ablation Study

| Configuration | Llama 3-gram | Pearson Correlation | Description |
| :--- | :--- | :--- | :--- |
| BGE Original | 48.9 | 0.198 | Baseline |
| Self-Training | 48.9 | 0.188 | Pseudo-labels reinforce existing bias |
| mMARCO FT | 48.7 | 0.132 | Generic data fails to solve specific distribution mismatch |
| LAURA | 49.9 | 0.236 | Utility-driven approach is effective |

### Key Findings
- A massive gap of ~15 percentage points exists between the Oracle and actual performance, indicating that sufficient evidence exists in the candidate pool but is ignored by the reranker.
- After LAURA training, the Pearson correlation coefficient for the Qwen reranker increased from 0.127 to 0.264 (+108%), significantly enhancing the alignment between re-ranking scores and generation quality.
- The language distribution of the reranker output after training is much closer to the Oracle distribution, with JS divergence dropping from 0.203 to 0.090 (BGE).
- The proportion of English and query-language documents decreased, allowing other languages fairer ranking opportunities.

## Highlights & Insights
- The Oracle evidence estimation framework is an elegant diagnostic tool that clearly separates "lack of information" from "incorrect selection." This analytical method can be transferred to any scenario involving multi-source information selection.
- Using the average generation quality of multiple models as a document utility signal reduces model bias and avoids manual annotation costs.
- The low correlation (<0.2) between re-ranking scores and generation quality is a major finding, indicating that "semantic relevance" and "answer utility" in current rerankers are distinct concepts.

## Limitations & Future Work
- The Oracle upper bound is still an estimate rather than a true upper bound and may under- or over-estimate achievable performance.
- Validated only on the MKQA dataset; coverage of languages and domains is limited.
- LAURA requires multiple generative models to estimate utility, leading to high data construction costs.
- Future work could explore lightweight utility estimation methods or direct online optimization of rerankers using RL.

## Related Work & Insights
- **vs. Traditional Multilingual Retrieval**: Traditional methods focus on retrieval recall; this paper reveals that re-ranking is the true bottleneck.
- **vs. mMARCO Fine-tuning**: General ranking data cannot solve the specific language bias issues in mRAG.
- **vs. Translation-based mRAG**: Translation strategies bypass the multilingual evidence selection problem, whereas this paper optimizes the selection mechanism directly.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The Oracle analysis framework and utility-driven alignment are significant new contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 13 languages, multiple rerankers, multiple generators, and thorough ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear, step-by-step problem analysis with a logical progression from diagnosis to treatment.
- **Value**: ⭐⭐⭐⭐ Highlights the overlooked issue of re-ranking bias in mRAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)
- [\[ACL 2026\] The Dilemma of Low-Resource Languages in Multilingual Retrieval: Evidence from Amharic](the_multilingual_curse_at_the_retrieval_layer_evidence_from_amharic.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](coral_adaptive_retrieval_loop_for_culturally-aligned_multilingual_rag.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)

</div>

<!-- RELATED:END -->
