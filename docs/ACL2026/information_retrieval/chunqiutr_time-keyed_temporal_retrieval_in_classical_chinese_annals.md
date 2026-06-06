---
title: >-
  [Paper Note] ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals
description: >-
  [ACL 2026][Information Retrieval & RAG][Temporal Retrieval] Ours proposes ChunQiuTR, the first time-keyed retrieval benchmark based on non-Gregorian calendars derived from the *Spring and Autumn Annals* (*Chunqiu*) and i…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Temporal Retrieval"
  - "Classical Chinese"
  - "Calendar Encoding"
  - "Bi-encoder"
  - "RAG"
date: 2026-05-08
content_hash: a1cacaefd1883037
---

# ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals

**Conference**: ACL 2026  
**arXiv**: [2604.06997](https://arxiv.org/abs/2604.06997)  
**Code**: [https://github.com/xbdxwyh/ChunQiuTR](https://github.com/xbdxwyh/ChunQiuTR)  
**Area**: Information Retrieval / Temporal Retrieval  
**Keywords**: Temporal Retrieval, Classical Chinese, Calendar Encoding, Bi-encoder, RAG

## TL;DR
Ours proposes ChunQiuTR, the first time-keyed retrieval benchmark based on non-Gregorian calendars derived from the *Spring and Autumn Annals* (*Chunqiu*) and its commentary traditions. Ours designs CTD (Calendar-Temporal Dual-encoder), which achieves time-aware retrieval through Fourier absolute calendar contexts and relative offset biases, significantly outperforming pure semantic baselines.

## Background & Motivation

**Background**: Retrieval serves as a critical interface for LLMs to acquire and localize knowledge in RAG systems. In historical research, retrieval targets are not arbitrary relevant passages but precise records of specific years and months—temporal consistency is as essential as thematic relevance.

**Limitations of Prior Work**: Classical Chinese annals use concise, implicit non-Gregorian regnal year expressions (e.g., "First Year, Spring," "Summer, Fifth Month"). Temporal information often omits the absolute year, requiring inference from context. Semantically similar passages may be temporally incorrect—for instance, a query for "Twelfth Month, Second Year of Duke Zhuang" might retrieve commentary on the same date phrase (repeated date but no event answer) or highly similar events from adjacent months.

**Key Challenge**: Semantic similarity does not equate to temporal consistency. Existing neural retrieval methods model relevance as semantic similarity, failing to distinguish "temporal proximity confounders"—records with highly similar phrasing but occurring in different months.

**Goal**: To achieve temporally consistent retrieval under non-Gregorian, regnal calendar systems as a key prerequisite for downstream historical RAG.

**Key Insight**: Leveraging the multi-layered structure of the *Chunqiu* and its Three Commentaries (*Zuo Zhuan*, *Gongyang Zhuan*, *Guliang Zhuan*)—where all layers share the same regnal timeline but describe the same events in different phrasing—naturally produces "near-duplicate" hard negatives.

**Core Idea**: Introducing calendar position awareness on top of semantic matching—learning a continuous calendar axis, injecting absolute calendar context, and adding relative temporal biases.

## Method

### Overall Architecture
ChunQiuTR consists of benchmark construction and methodology. The benchmark aligns *Chunqiu* records to month-level time keys $\tau = (gong, year, month)$, designing three types of queries: point, gap, and window, and extracts counterfactual temporal proximity hard negatives from later historical texts. The CTD method adds a calendar temporal head and bias module to a standard bi-encoder.

### Key Designs

1. **Time-Key Alignment and Counterfactual Negatives**:
    - **Function**: Constructing a high-quality temporal retrieval benchmark.
    - **Mechanism**: Aligning chronological records to month-level time keys, comprising 20,172 records and 16,226 queries. Paraphrases of the same events are extracted from later historical books (e.g., Gu Donggao's *Da Shi Biao*) as temporal proximity counterfactual hard negatives—they share the time key and have highly similar phrasing with the target record but are not the correct retrieval targets.
    - **Design Motivation**: Real-world historical retrieval failure modes involve these temporal proximity confounders; the benchmark must include such hard negatives.

2. **Latent Calendar Scalars**:
    - **Function**: Establishing continuous positions for text on a unified timeline.
    - **Mechanism**: Attaching three lightweight prediction heads (Duke/Year/Month) to embeddings from a shared Transformer encoder. The expectation of the output probability distribution yields soft coordinates $g_x, y_x, m_x$, linearized as $u_x = \frac{g_x \cdot (Y \cdot M) + y_x \cdot M + m_x}{G \cdot Y \cdot M - 1} \in [0,1]$.
    - **Design Motivation**: Regnal years are discrete identifiers that do not directly provide position metrics or cross-reign distances; learning a continuous axis makes temporal relationships quantifiable.

3. **Absolute + Relative Temporal Augmentation**:
    - **Function**: Injecting temporal consistency constraints into semantic matching.
    - **Mechanism**: Absolute part—mapping soft predictions to temporal context vectors using a Fourier codebook, injected into embeddings via gated residuals $\tilde{h}_x = h_x + \gamma c_x$. Relative part—calculating query-record temporal offsets $\Delta u_{ij}$, generating an additive bias $b_{ij}^{time}$ via Fourier features and an MLP. Final score $s_{ij}^{CTD} = s_{ij}^{abs} + b_{ij}^{time}$.
    - **Design Motivation**: Absolute context lets the embedding "know" the text's position in the calendar; relative bias penalizes matches with large temporal distances, even if semantically similar.

### Loss & Training
An interval-overlapping multi-positive InfoNCE loss is used: temporal interval overlap serves as weak supervision to label in-batch positive examples. Auxiliary losses train the temporal prediction heads (cross-entropy for Duke/Year/Month classification + temporal label smoothing).

## Key Experimental Results

### Main Results

| Method | P-Time R@1 | G-Time R@1 | W-Time R@1 | Average |
|------|-----------|-----------|-----------|------|
| BM25 | Baseline | Baseline | Baseline | - |
| DPR | Semantic Baseline | Semantic Baseline | Semantic Baseline | - |
| CTD (Ours) | **Best** | **Best** | **Best** | Significant Gain |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Semantic only | Baseline | No temporal awareness |
| + Absolute context | Gain | Embeddings carry calendar position information |
| + Relative bias | Further Gain | Penalizes matches with large temporal distances |
| + Multi-positive | **Best** | Interval overlap supervision enhances temporal generalization |

### Key Findings
- Temporal proximity confusion is the primary failure mode of pure semantic retrieval—records from adjacent months with highly similar phrasing are frequently misretrieved.
- CTD shows the most significant improvement in temporal proximity and adjacent-month confounder scenarios.
- Absolute and relative temporal signals are complementary—using either alone provides gains, but the combination works best.

## Highlights & Insights
- **Precise Problem Definition**: Decoupling "temporal consistency" from "semantic relevance" reveals the core failure mode of RAG systems in historical texts.
- **Fourier Calendar Encoding**: The design can be generalized to any non-standard temporal system (e.g., Lunar, Islamic, Japanese era names), not limited to the *Chunqiu*.
- **Benchmark Methodology**: The combination of LLM-assisted proposals and manual verification has strong extensibility in the field of cultural heritage digitization.

## Limitations & Future Work
- Validated only on *Chunqiu* corpora; generalizability to other annals (e.g., *Zizhi Tongjian*) is unknown.
- Month-level is the finest granularity; day-level temporal information in the *Chunqiu* is too sparse to be systematized.
- Evaluated retrieval quality but did not further verify improvement in faithfulness for downstream RAG generation.

## Related Work & Insights
- **vs. Standard TIR**: Standard temporal information retrieval assumes modern timestamps and open retrieval; Ours handles non-Gregorian fine-grained annals, presenting entirely different challenges.
- **vs. BM25/DPR**: Pure semantic methods fail systematically when faced with temporal proximity confounders.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First non-Gregorian time-keyed retrieval benchmark; highly unique problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorous benchmark construction and sufficient ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Excellent integration of historical background and technical methodology.
- Value: ⭐⭐⭐⭐ Unique value for digital humanities and historical RAG.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Benchmarking and Enabling Efficient Chinese Medical Retrieval via Asymmetric Encoders](benchmarking_and_enabling_efficient_chinese_medical_retrieval_via_asymmetric_enc.md)
- [\[ACL 2026\] Test-Time Training for Zero-Resource Dense Retrieval Reranking](test-time_training_for_zero-resource_dense_retrieval_reranking.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[AAAI 2026\] Towards Inference-Time Scaling for Continuous Space Reasoning](../../AAAI2026/information_retrieval/towards_inference-time_scaling_for_continuous_space_reasoning.md)
- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](../../NeurIPS2025/information_retrieval/retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)

</div>

<!-- RELATED:END -->
