---
title: >-
  [Paper Note] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs
description: >-
  [ICLR2026][Information Retrieval & RAG][zero-shot classification] This paper proposes BTZSC, a benchmark comprising 22 datasets, which for the first time systematically compares four model families — NLI cross-encoders…
tags:
  - "ICLR2026"
  - "Information Retrieval & RAG"
  - "zero-shot classification"
  - "benchmark"
  - "reranker"
  - "embedding model"
  - "NLI"
date: 2026-05-08
content_hash: 6f142d10a53021f6
---

# BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs

**Conference**: ICLR2026
**arXiv**: [2603.11991](https://arxiv.org/abs/2603.11991)  
**Code**: [GitHub](https://github.com/IliasAarab/btzsc)  
**Area**: Information Retrieval
**Keywords**: zero-shot classification, benchmark, reranker, embedding model, NLI

## TL;DR
This paper proposes BTZSC, a benchmark comprising 22 datasets, which for the first time systematically compares four model families — NLI cross-encoders, embedding models, rerankers, and instruction-tuned LLMs (38 models in total) — under a unified zero-shot protocol. Qwen3-Reranker-8B achieves a new SOTA with macro F1 = 0.72, while embedding models demonstrate the best accuracy–latency trade-off.

## Background & Motivation

**Background**: Zero-shot text classification (ZSC) matches text directly against human-readable label descriptions, eliminating the need for costly annotation. Early mainstream approaches reformulated classification as a natural language inference (NLI) task, employing cross-encoders to assess entailment relations between text and labels.

**Limitations of Prior Work**: Embedding models, rerankers, and instruction-tuned LLMs have advanced rapidly in recent years, yet no benchmark exists that fairly compares all four model families under a unified zero-shot protocol. Although MTEB offers broad coverage, its classification evaluation relies on supervised linear probes rather than genuine zero-shot inference.

**Key Challenge**: Existing evaluations either cover only a single model family or incorporate supervised signals, making it impossible to accurately assess the capability gap and applicable scenarios of each model family under truly zero-shot conditions.

**Goal**: To construct a standardized zero-shot text classification benchmark spanning diverse task types, domains, and label-set sizes, enabling systematic comparison of the four model families in terms of performance, scaling behavior, and efficiency.

**Key Insight**: Twenty-two public datasets are selected to cover four major task categories — sentiment, topic, intent, and emotion detection — using a unified label verbalization and inference protocol to ensure all models are evaluated under identical conditions.

**Core Idea**: Construct BTZSC, the first benchmark to uniformly evaluate NLI cross-encoders, embedding models, rerankers, and LLMs for zero-shot text classification.

## Method

### Overall Architecture
BTZSC is an evaluation benchmark rather than a novel model. The input consists of text samples and a set of candidate labels (converted into natural language descriptions via label verbalization), and the output is each model's score or prediction over the candidate labels. The benchmark comprises 22 English datasets across four task categories (sentiment, topic, intent, and emotion detection), spanning domains such as news, social media, product reviews, encyclopedias, and politics. A total of 38 models are evaluated, with macro F1 as the primary metric.

### Key Designs

1. **Dataset Design**:

    - Function: Provide a task collection with multi-dimensional coverage.
    - Mechanism: Datasets are selected along four dimensions — task diversity (4 categories), class granularity (2 to 77 classes), domain diversity, and document length (8–293 tokens) — with weighted Jaccard similarity used to verify lexical diversity across datasets.
    - Design Motivation: Avoid evaluation bias caused by single-task or single-domain datasets, ensuring conclusions generalize broadly.

2. **Unified Inference Protocol**:

    - Function: Standardize the inference procedure across all four model families.
    - Mechanism: NLI cross-encoders select the label with the highest entailment logit; embedding models compute cosine similarity between text and label representations and select the maximum; rerankers treat the text as a query and label descriptions as documents for ranking; LLMs use a multiple-choice prompt and select the option with the highest conditional probability.
    - Design Motivation: The inference mechanisms of different model families differ substantially; each family must operate in its most natural inference mode while maintaining zero-shot conditions.

3. **Label Verbalization**:

    - Function: Convert class labels into semantically rich natural language descriptions.
    - Mechanism: For example, the positive class in Amazon Polarity is verbalized as "The overall sentiment within the Amazon product review is positive."
    - Design Motivation: Raw labels (e.g., "positive") carry insufficient information; verbalization enables models to better leverage their semantic understanding capabilities.

### Evaluation Metrics
The primary metric is macro F1 (equal weight per class), with micro accuracy reported as a supplementary metric. AUROC is used to evaluate NLI performance to avoid issues related to threshold selection and probability calibration.

## Key Experimental Results

### Main Results

| Model Family | Representative Model | Topic F1 | Sentiment F1 | Intent F1 | Emotion F1 | Avg. F1 | Avg. Acc |
|---|---|---|---|---|---|---|---|
| Base Encoder | bert-large | 0.34 | 0.38 | 0.15 | 0.08 | 0.30 | 0.40 |
| NLI Cross-Enc | deberta-v3-large-nli-triplet | 0.50 | 0.90 | 0.45 | 0.42 | **0.60** | 0.62 |
| Reranker | Qwen3-Reranker-8B | — | — | — | — | **0.72** | 0.76 |
| Embedding | gte-large-en-v1.5 | — | — | — | — | 0.62 | 0.65 |
| LLM | Mistral-Nemo-12B | — | — | — | — | 0.67 | 0.71 |

### Scaling and Efficiency Analysis

| Analysis Dimension | Key Finding |
|---|---|
| Reranker Scaling | Monotonically improves with parameter count; 8B achieves F1 = 0.72 |
| Embedding Scaling | Saturates at 0.60–0.62 F1 beyond a few hundred million parameters |
| LLM Scaling | Steepest improvement in the 3B→8B range, matching the best embedding models |
| Accuracy–Latency Trade-off | Embedding models occupy the Pareto frontier (high accuracy + low latency) |
| NLI→ZSC Transfer | Linear positive correlation for NLI cross-encoders; no correlation for embedding models |

### Key Findings
- The reranker (Qwen3-Reranker-8B) sets a new ZSC SOTA at macro F1 = 0.72, surpassing the best NLI cross-encoder by +12 F1 points.
- Embedding models (gte-large-en-v1.5) achieve F1 = 0.62 without cross-attention, approaching cross-encoder performance while being substantially faster.
- NLI cross-encoder performance saturates as the backbone scales; deberta-v3 remains the strongest backbone.
- LLMs are particularly strong on topic classification (F1 up to 0.69) but weaker on emotion detection.
- Even the compact Qwen3-Reranker-0.6B outperforms all NLI cross-encoders.
- NLI capability is not a reliable predictor of ZSC performance for embedding models — the structure of the embedding space is the critical factor.

## Highlights & Insights
- **First unified zero-shot evaluation across four model families**: Fills the gap left by MTEB in zero-shot classification and reveals that rerankers are a severely underestimated model family for ZSC.
- **Unexpected rise of rerankers**: Repositions rerankers from information retrieval to the top choice for ZSC, providing actionable deployment guidance — use rerankers when accuracy is the priority, and embedding models when low latency is required.
- **NLI-to-ZSC transferability analysis**: Reveals that the relationship between NLI capability and ZSC performance is model-family-dependent, implying that improving NLI performance does not straightforwardly translate into better zero-shot classification for embedding models.
- **Reusable design**: The label verbalization strategy and unified evaluation protocol are directly transferable to multilingual or other classification tasks.

## Limitations & Future Work
- Only English datasets are covered; multilingual zero-shot classification capability is not evaluated, limiting applicability in cross-lingual settings.
- Pretraining data for some models may overlap with benchmark datasets (data leakage risk); while the authors conducted checks, contamination cannot be fully excluded.
- Evaluation does not include larger LLMs (>12B), leaving open whether LLMs at greater scale can surpass rerankers.
- The quality of label verbalization substantially affects results, yet no verbalization sensitivity analysis is conducted.
- Only single-label classification is evaluated; multi-label classification scenarios are not considered.

## Related Work & Insights
- **vs. MTEB**: MTEB evaluates classification via supervised linear probes rather than true zero-shot inference; BTZSC is purely zero-shot and better reflects models' intrinsic semantic understanding.
- **vs. Yin et al. (2019)**: The earliest NLI-ZSC benchmark covers only 3 datasets and exclusively cross-encoders; BTZSC extends to 22 datasets × 4 model families.
- **vs. TTC23**: Evaluates only topic classification using prompt-based methods, excluding embedding and reranker models.
- **vs. Lepagnol et al. (2024)**: Covers only small models (100M–1B) and lacks comparison with embedding and reranker models.
- For practical applications requiring zero-shot classification, this paper provides a clear model selection guide: choose rerankers when accuracy is paramount and embedding models when latency is critical.

## Rating
- Novelty: ⭐⭐⭐⭐ First unified evaluation across four model families, though the contribution is a benchmark rather than an algorithmic innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 datasets × 38 models, with analyses covering scaling behavior, efficiency, and NLI transferability.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, well-defined conclusions, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Provides important empirical evidence for model selection in zero-shot text classification.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PL-MTEB: Polish Massive Text Embedding Benchmark](../../ACL2026/information_retrieval/pl-mteb_polish_massive_text_embedding_benchmark.md)
- [\[CVPR 2026\] Explaining CLIP Zero-shot Predictions Through Concepts](../../CVPR2026/information_retrieval/explaining_clip_zero-shot_predictions_through_concepts.md)
- [\[AAAI 2026\] OAD-Promoter: Enhancing Zero-shot VQA using Large Language Models with Object Attribute Description](../../AAAI2026/information_retrieval/oad-promoter_enhancing_zero-shot_vqa_using_large_language_models_with_object_att.md)
- [\[ICML 2026\] BlitzRank: Principled Zero-shot Ranking Agents with Tournament Graphs](../../ICML2026/information_retrieval/blitzrank_principled_zero-shot_ranking_agents_with_tournament_graphs.md)
- [\[ICLR 2026\] HUME: Measuring the Human-Model Performance Gap in Text Embedding Tasks](hume_measuring_the_human-model_performance_gap_in_text_embedding_tasks.md)

</div>

<!-- RELATED:END -->
