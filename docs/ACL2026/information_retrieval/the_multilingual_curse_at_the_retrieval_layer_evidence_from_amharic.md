---
title: >-
  [Paper Note] 多语言检索中的低资源语言困境：Amharic 语言证据
description: >-
  [ACL 2026][Information Retrieval & RAG][Amharic] Using Amharic as a diagnostic case, this paper reveals that powerful multilingual retrieval models fail to migrate effectively to morphologically rich low-resource languages in zero-shot settings, with a 23% relative drop in MRR@10 performance. While language-specific fine-tuning provides 32-60% improvements, it still
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - Amharic
date: 2026-05-08
content_hash: 19a9068b746dbd9e
---
# Low-Resource Language Dilemma in Multilingual Retrieval: Evidence from Amharic

**Conference**: ACL 2026  
**arXiv**: [2605.24556](https://arxiv.org/abs/2605.24556)  
**Code**: https://github.com/rasyosef/amharic-neural-ir  
**Area**: Information Retrieval / Multilingual NLP  
**Keywords**: Multilingual Retrieval, Low-Resource Language, Amharic, Zero-shot Transfer, Neural IR

## TL;DR

Using Amharic as a diagnostic case, this paper reveals that powerful multilingual retrieval models fail to migrate effectively to morphologically rich low-resource languages in zero-shot settings, with a 23% relative drop in MRR@10 performance. While language-specific fine-tuning provides 32-60% improvements, it still fails to reach the level of monolingual retrievers, indicating that multilingual retrieval is insufficient to guarantee equitable information access for low-resource languages.

## Background & Motivation

**Background**: Multilingual retrieval has become a core component of cross-lingual question answering and Retrieval-Augmented Generation (RAG). Latest multilingual embedding models (e.g., E5, Arctic Embed) have achieved strong zero-shot performance on multilingual benchmarks, leading researchers to believe these models can reliably transfer across languages.

**Limitations of Prior Work**: However, this assumption does not hold for morphologically rich, resource-scarce languages. Despite Amharic having 58 million speakers, its performance in multilingual retrieval is severely deficient. Multilingual tokenizers handle Amharic's root-pattern morphology, complex affixes, and non-Latin script (Ge'ez characters) inappropriately, resulting in fragmented representations. The Core Problem is that high aggregate scores in multilingual benchmarks mask serious degradation at the individual language level.

**Key Challenge**: While zero-shot multilingual retrieval performs well on average, there may be a hidden retrieval quality ceiling for low-resource languages. If RAG systems are built on zero-shot multilingual retrievers, they will inherit this ceiling, affecting downstream generation quality.

**Goal**: (1) Quantify the performance gap of zero-shot multilingual retrieval in low-resource languages; (2) Evaluate the upper bound of gains from language-specific fine-tuning; (3) Establish a unified evaluation framework to compare four mainstream retrieval paradigms.

**Key Insight**: Amharic is chosen as a diagnostic case not only because of its large speaker population but also because it simultaneously possesses several features that make retrieval difficult (non-Latin script, complex morphology, limited multilingual pre-training coverage)—features common to hundreds of languages globally.

**Core Idea**: By rigorously comparing three types of retrievers (zero-shot multilingual, multilingual fine-tuned, and monolingual) under a shared protocol, the paper quantifies the real gap in retrieval performance for low-resource languages, arguing that aggregate multilingual scores are no substitute for in-depth language-level evaluation.

## Method

### Overall Architecture

This is a diagnostic paper focused on rigorous evaluation: it seeks to answer "how many points strong multilingual retrievers actually lose on low-resource languages and how much fine-tuning can recover," rather than proposing a new model. For this purpose, the authors built a controlled experimental pipeline—first aggregating four sources into the Amharic Passage Retrieval Dataset V2 (68K query-document pairs), then comparing three categories of retrievers (zero-shot multilingual, multilingual fine-tuned, monolingual) across four paradigms (dense bi-encoder, late interaction, learned sparse, cross-encoder) under the same data and metrics. The evaluation is conducted in two stages: first, initial ranking with Recall@k, MRR@10, and NDCG@10; second, re-ranking the top 50 candidates using a cross-encoder to observe additional gains from joint scoring. By fixing variables (data, supervision signals, optimization settings), differences can be cleanly attributed to the "language initialization space."

### Key Designs

**1. Multi-source Dataset + Source-aligned Weak Supervision: Approximating Real Deployment with Conservative Evaluation**

To avoid bias from a single news source, V2 integrates four sources: AMNEWS (headlines to body text), XL-SUM (summaries), Amharic Wikipedia, and AmQA (Q&A pairs), totaling 68K pairs after MD5 deduplication. Labeling uses source-aligned weak supervision—one target document per query with binary relevance. This single-positive setting is intentionally conservative: it closer approximates real-world scenarios with incomplete labeling and ensures metric monotonicity, with the trade-off of potentially missing other relevant documents. Thus, absolute scores should be interpreted cautiously, though they are fair for cross-model comparisons.

**2. Unified Evaluation Protocol for Four Paradigms: Separating "Algorithmic Differences" from "Training Setups"**

The four paradigms cover the evolution from sparse to dense to hybrid retrieval, each with Medium and Base variants. All are trained and evaluated on the same data and metrics to avoid false differences caused by comparing models from different papers. Specifically: dense bi-encoders use MultipleNegativesRankingLoss with Matryoshka representation learning; late interaction (ColBERT) uses token-level MaxSim scoring (query 32 tokens, doc 256); learned sparse (SPLADE) uses vocab pooling with sparsity regularization; cross-encoders jointly encode query-doc pairs using weighted BCE (weight 7 for positives) to handle class imbalance. The protocol ensures that if monolingual models across different paradigms consistently outperform zero-shot multilingual models, the issue lies in the shared representation space.

**3. Zero-shot vs. Fine-tuning Control: Converging Variables on "Initialization Space"**

To quantify the gap between "multilingual initialization" and "monolingual initialization," all factors except initialization are kept identical. The authors fixed the 68K training set and the same SentenceTransformers + MNR + Matryoshka fine-tuning method for two state-of-the-art multilingual models (EmbeddingGemma, Harrier). Hyperparameters were locked: learning rate $4\times10^{-5}$, warmup 0.025, BF16 mixed precision, 6 epochs, batch size 128, and a strategy of selecting the most and least similar from pre-mined negatives. By fixing supervised data and optimization, the results cleanly reflect the differences in the initial space—showing that fine-tuning recovers a significant portion (Gemma +60.3%, Harrier +32.0%), though Harrier still lags 5.4% behind the monolingual model.

### Loss & Training

Monolingual Amharic models: Dense bi-encoders use MultipleNegativesRankingLoss (4 pre-mined negatives) + Matryoshka Loss, with early stopping based on validation NDCG@10. Late interaction and sparse retrieval are fixed at 4–6 epochs. Cross-encoders use weighted BCE with a positive weight of 7. Evaluation metrics are standardized as Recall@5/10, MRR@10, and NDCG@10.

## Key Experimental Results

### Main Results: Stage 1 Retrieval Results Comparison

| Model | Params (M) | R@5 | R@10 | MRR@10 | NDCG@10 |
|------|--------|-----|------|--------|---------|
| BM25 | – | 0.734 | 0.789 | 0.612 | 0.655 |
| embedding-gemma-300m (Zero-shot) | 300 | 0.558 | 0.621 | 0.448 | 0.489 |
| harrier-oss-v1-270m (Zero-shot) | 270 | 0.697 | 0.753 | 0.576 | 0.619 |
| multilingual-e5-large (Zero-shot) | 560 | 0.736 | 0.791 | 0.603 | 0.648 |
| snowflake-arctic (Zero-shot, Best) | 568 | 0.795 | 0.848 | 0.653 | 0.701 |
| embedding-gemma + Fine-tuned | 300 | 0.813 | 0.862 | 0.718 | 0.753 |
| harrier + Fine-tuned | 270 | 0.860 | 0.903 | 0.760 | 0.795 |
| ColBERT-Base-Amharic (Monolingual, Best) | 110 | 0.902 | 0.930 | **0.803** | **0.835** |
| Embed-Base-Amharic (Monolingual) | 110 | 0.870 | 0.907 | 0.774 | 0.807 |
| SPLADE-Base-Amharic (Monolingual) | 110 | 0.871 | 0.906 | 0.754 | 0.792 |

**Key Findings**: (1) Gap between Zero-shot Multilingual and Monolingual: the strongest zero-shot model Arctic reached MRR@10 of 0.653, while monolingual ColBERT reached 0.803, a **relative drop of 23.0%**; (2) Parameters are not the solution: Arctic has 568M parameters but loses to the 110M monolingual model; (3) Limited fine-tuning gains: Gemma improved from 0.448→0.718 (+60.3%) and Harrier from 0.576→0.760 (+32.0%), but Harrier fine-tuned still lags 5.4% behind the monolingual model; (4) Consistency across paradigms: monolingual models in dense, late interaction, and sparse paradigms all outperformed the strongest zero-shot multilingual model.

### Ablation Study: Stage 2 Re-ranking Results

| Model | MRR@10 | NDCG@10 | Gain |
|------|--------|---------|------|
| Embed-Base-Amharic (Stage 1) | 0.774 | 0.807 | – |
| + Re-rank-Medium-Amharic | 0.805 | 0.835 | +4.0% / +3.5% |
| + Re-rank-Base-Amharic | 0.830 | 0.856 | +7.2% / +6.1% |

Cross-encoder re-ranking achieved the highest score of MRR@10 0.830, indicating that joint query-document encoding captures subtle ranking nuances missed by independent bi-encoders.

### Key Findings & Analysis

- **Hidden failure of zero-shot multilingual models**: While multilingual models perform well on average, significant degradation can occur in specific low-resource languages. The Amharic case shows a 23% relative performance drop in the MRR@10 top-10 region, directly impacting the quality of retrieval results seen by users.
- **Necessity of language-specific modeling**: Monolingual models across various retrieval paradigms consistently outperform zero-shot multilingual models, suggesting the issue is not algorithmic choice but the inability of shared representation spaces to fit Amharic morphological features.
- **Partial compensation via fine-tuning**: Fine-tuning unlocks learning potential (Gemma gained most), but cannot fully eliminate the gap with monolingual models.
- **Complementary role of architecture optimization**: Cross-encoder re-ranking provided an additional 7.2% boost over dense bi-encoders, indicating that more complex joint scoring can further improve results once an adequate initial ranking is obtained.

## Highlights & Insights

- **Diagnostic Case Study**: The paper cleverly uses Amharic—a language with universal features (non-Latin script, complex morphology, resource scarcity) but high usage (58 million speakers)—to transform a local issue into systemic insight.
- **Value of Shared Evaluation Protocol**: Contrast across four paradigms in a unified framework effectively demonstrates that the root problem is the representation space rather than the architecture. It avoids the trap of comparing independently trained models.
- **Precise Fine-tuning Design**: By fixing supervised data and training strategies, the authors minimized fine-tuning variance, making comparative conclusions robust.
- **Real-world Significance for RAG**: The paper explicitly points out that retrieval layer failure becomes a quality ceiling for downstream generation, which cannot be masked by aggregate scores of final answer quality.

## Limitations & Future Work

- **Limited Monolingual Evidence**: The paper only uses Amharic as a case study. While its features are common, the specific performance gap may vary by language.
- **Weak Supervision Limitations**: The single-positive labels from source-aligned data might miss other relevant documents.
- **Scope of Fine-tuning Study**: Fine-tuning was only performed on two recent multilingual models. Larger models or those with different instruction-tuning approaches might behave differently.
- **Lack of End-to-end RAG Evaluation**: The paper infers RAG quality degradation based on retrieval metrics but does not verify this on actual generation tasks.

**Future Directions**: (1) Increasing data weight for low-resource, morphologically rich languages during multilingual pre-training; (2) Designing morphology-aware tokenizers; (3) Maintaining public retrieval benchmarks for all languages, not just dominant ones; (4) Embedding language-specific retrieval adapter layers in LLM applications.

## Related Work & Insights

- **vs MIRACL/mMARCO**: These works provide aggregate multilingual evaluation but hide language-level differences. Ours innovates by shifting to language-specific deep evaluation.
- **vs Mekonnen et al. (2025)**: Previous work showed monolingual Amharic models outperform zero-shot multilingual ones. Ours extends this by validating on larger datasets, adding learned sparse and cross-encoder paradigms, and systematically studying fine-tuning upper bounds.
- **vs Multilingual Pre-training Works (E5, Arctic)**: High average scores often reported in these papers are challenged here by deep-diving into individual languages.
- **Insight**: Evaluation of multilingual systems must descend to the individual language level; problems in under-served languages are often issues of initialization and representation space rather than algorithm choice; parameter scale and high benchmark scores are no substitute for language-specific stress tests.

## Rating

- Novelty: ⭐⭐⭐⭐ The selection of a diagnostic case study is clever, highlighting the "hidden failure" phenomenon often overlooked, though the experimental design is incremental improvement rather than fundamental innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Systematic comparison across four paradigms, three model types, and two stages, supported by clear ablation and analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic is clear, with a powerful progressive argument from problem positioning to design to results and implications.
- Value: ⭐⭐⭐⭐ Significant warning and guidance for deploying multilingual RAG/QA systems and evaluating multilingual models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](../../ACL2025/information_retrieval/optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2026\] Code-Switching Information Retrieval: Benchmarks, Analysis, and the Limits of Current Retrievers](code-switching_information_retrieval_benchmarks_analysis_and_the_limits_of_curre.md)
- [\[ACL 2026\] AuthorityBench: Benchmarking LLM Authority Perception for Reliable Retrieval-Augmented Generation](authoritybench_benchmarking_llm_authority_perception_for_reliable_retrieval-augm.md)
- [\[ACL 2026\] eTracer: Towards Traceable Text Generation via Claim-Level Grounding](etracer_towards_traceable_text_generation_via_claim-level_grounding.md)
- [\[ACL 2026\] MAB-DQA: Addressing Query Aspect Importance in Document Question Answering with Multi-Armed Bandits](mab-dqa_addressing_query_aspect_importance_in_document_question_answering_with_m.md)

</div>

<!-- RELATED:END -->
