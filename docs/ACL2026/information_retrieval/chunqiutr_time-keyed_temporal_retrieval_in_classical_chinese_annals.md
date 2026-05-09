---
title: >-
  [Paper Note] ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals
description: >-
  [ACL 2026][Temporal Retrieval] This paper proposes ChunQiuTR, the first temporal retrieval benchmark built upon a non-Gregorian calendar system, constructed from the *Spring and Autumn Annals* and its exegetical traditions. It further introduces CTD (Calendrical Temporal Dual-encoder), which achieves temporally-aware retrieval via Fourier-based absolute calendrical context and relative temporal offset biases, substantially outperforming pure semantic baselines.
tags:
  - ACL 2026
  - Temporal Retrieval
  - Classical Chinese
  - Calendar Encoding
  - Bi-Encoder
  - RAG
date: 2026-05-08
content_hash: b8a11a6afcf904a6
---

# ChunQiuTR: Time-Keyed Temporal Retrieval in Classical Chinese Annals

**Conference**: ACL 2026
**arXiv**: [2604.06997](https://arxiv.org/abs/2604.06997)
**Code**: [https://github.com/xbdxwyh/ChunQiuTR](https://github.com/xbdxwyh/ChunQiuTR)
**Area**: Information Retrieval / Temporal Retrieval
**Keywords**: Temporal Retrieval, Classical Chinese, Calendar Encoding, Bi-Encoder, RAG

## TL;DR
This paper proposes ChunQiuTR, the first temporal retrieval benchmark built upon a non-Gregorian calendar system, constructed from the *Spring and Autumn Annals* and its exegetical traditions. It further introduces CTD (Calendrical Temporal Dual-encoder), which achieves temporally-aware retrieval via Fourier-based absolute calendrical context and relative temporal offset biases, substantially outperforming pure semantic baselines.

## Background & Motivation

**State of the Field**: Retrieval serves as the critical interface through which LLMs access and locate knowledge in RAG systems. In historical research, retrieval targets are not arbitrary relevant passages but precise records associated with specific regnal months — temporal consistency is as important as topical relevance.

**Limitations of Prior Work**: Classical Chinese annalistic texts employ concise, implicit non-Gregorian reign-era expressions (e.g., "first year, spring" or "fifth month of summer"), where absolute year information is omitted and must be inferred from context. Semantically similar passages may be entirely misaligned in time — for instance, a query for "the twelfth month of Duke Zhuang's second year" may retrieve exegetical commentary sharing the same date phrase (repeating the date but not answering the event), or highly similar events from adjacent months.

**Root Cause**: Semantic similarity does not imply temporal consistency. Existing neural retrieval methods model relevance as semantic similarity and cannot distinguish "temporal neighbor confounders" — records that are lexically near-identical but occur in different months.

**Paper Goals**: To achieve temporally consistent retrieval under a non-Gregorian, dynastic reign-era system, as an essential prerequisite for downstream historical RAG.

**Starting Point**: The multi-layered structure of the *Spring and Autumn Annals* and its three commentaries (*Zuozhuan*, *Gongyang*, *Guliang*) — all sharing the same annalistic timeline while describing the same events in different phrasing — naturally generates near-duplicate hard negatives.

**Core Idea**: Introduce calendrical position awareness on top of semantic matching — learning a continuous calendrical axis, injecting absolute calendrical context, and adding relative temporal biases.

## Method

### Overall Architecture
ChunQiuTR comprises two components: benchmark construction and the retrieval method. The benchmark aligns *Spring and Autumn* records to month-level temporal keys $\tau = (gong, year, month)$, defines three query types (point / gap / window queries), and extracts temporal-neighbor counterfactual hard negatives from later historical sources. The CTD method augments a standard bi-encoder with a calendrical temporal head and a bias module.

### Key Designs

1. **Temporal Key Alignment and Counterfactual Hard Negatives**:

    - Function: Construct a high-quality temporal retrieval benchmark.
    - Mechanism: Annalistic records are aligned to month-level temporal keys, yielding 20,172 records and 16,226 queries. Paraphrases of the same events from later historical works (e.g., Gu Dongao's *Dashibiao*) are extracted as temporal-neighbor counterfactual hard negatives — they share the temporal key with the target record and are lexically highly similar, yet are not the correct retrieval targets.
    - Design Motivation: Temporal-neighbor confusion is precisely the real-world failure mode in historical retrieval; the benchmark must incorporate such hard negatives.

2. **Latent Calendrical Scalar**:

    - Function: Establish a continuous positional representation for texts on a unified temporal axis.
    - Mechanism: Three lightweight prediction heads (reign / year / month) are appended to embeddings from a shared Transformer encoder, outputting probability distributions whose expectations yield soft coordinates $g_x, y_x, m_x$, linearized as $u_x = \frac{g_x \cdot (Y \cdot M) + y_x \cdot M + m_x}{G \cdot Y \cdot M - 1} \in [0,1]$.
    - Design Motivation: Dynastic reign eras are discrete identifiers that provide no direct positional metric or cross-dynasty distances; a learned continuous axis is required to make temporal relationships quantifiable.

3. **Absolute + Relative Temporal Augmentation**:

    - Function: Inject temporal consistency constraints into semantic matching.
    - Mechanism: For the absolute component, soft predictions are mapped to temporal context vectors via a Fourier encoding dictionary and injected into embeddings through a gated residual connection: $\tilde{h}_x = h_x + \gamma c_x$. For the relative component, the query–record temporal offset $\Delta u_{ij}$ is computed, passed through Fourier features and an MLP to produce an additive bias $b_{ij}^{time}$. The final score is $s_{ij}^{CTD} = s_{ij}^{abs} + b_{ij}^{time}$.
    - Design Motivation: Absolute context enables embeddings to encode a text's calendrical position; relative bias penalizes temporally distant matches even when they are semantically similar.

### Loss & Training
An interval-overlap multi-positive InfoNCE loss is employed: temporal interval overlap serves as weak supervision to label in-batch positives. Auxiliary losses train the temporal prediction heads (cross-entropy classification for reign / year / month, with temporal label smoothing).

## Key Experimental Results

### Main Results

| Method | P-Time R@1 | G-Time R@1 | W-Time R@1 | Avg. |
|--------|-----------|-----------|-----------|------|
| BM25 | Baseline | Baseline | Baseline | — |
| DPR | Semantic baseline | Semantic baseline | Semantic baseline | — |
| CTD (ours) | **Best** | **Best** | **Best** | Significant gain |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| Semantic only | Baseline | No temporal awareness |
| + Absolute context | Improved | Embeddings carry calendrical position |
| + Relative bias | Further improved | Penalizes temporally distant matches |
| + Multi-positive | **Best** | Interval-overlap supervision enhances temporal generalization |

### Key Findings
- Temporal-neighbor confusion is the dominant failure mode of pure semantic retrieval — records from adjacent months with highly similar phrasing are frequently retrieved erroneously.
- CTD yields the most pronounced improvements in scenarios involving temporal-neighbor and adjacent-month confounders.
- Absolute and relative temporal signals are complementary — each individually improves performance, and their combination yields superior results.

## Highlights & Insights
- **Precise problem formulation**: The paper cleanly separates "temporal consistency" from "semantic relevance," revealing the core failure mode of RAG systems on historical texts.
- **Fourier calendrical encoding** is generalizable to any non-standard temporal system (e.g., the lunisolar calendar, the Islamic calendar, Japanese imperial eras), not limited to the *Spring and Autumn Annals*.
- **Benchmark construction methodology** (LLM-assisted proposal + human validation) demonstrates strong transferability to the digital humanities and cultural heritage domains.

## Limitations & Future Work
- Validation is limited to the *Spring and Autumn* corpus; generalizability to other annalistic works (e.g., *Zizhi Tongjian*) remains unexplored.
- Month-level granularity is the finest achievable; day-level temporal information in the *Spring and Autumn Annals* is too sparse for systematic treatment.
- Retrieval quality is evaluated but the downstream improvement in RAG generation faithfulness is not further verified.

## Related Work & Insights
- **vs. Standard TIR**: Standard temporal information retrieval assumes modern timestamps and open retrieval; this paper addresses non-Gregorian fine-grained annalistic texts, posing fundamentally different challenges.
- **vs. BM25/DPR**: Pure semantic methods systematically fail in the presence of temporal-neighbor confounders.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First temporal retrieval benchmark under a non-Gregorian calendar; the problem is highly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Rigorous benchmark construction with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Historical background and technical methodology are integrated exceptionally well.
- Value: ⭐⭐⭐⭐ Offers unique contributions to digital humanities and historical RAG.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] TaxPraBen: A Scalable Benchmark for Structured Evaluation of LLMs in Chinese Real-World Tax Practice](taxpraben_a_scalable_benchmark_for_structured_evaluation_of_llms_in_chinese_real.md)
- [\[ACL 2026\] CounterRefine: Answer-Conditioned Counterevidence Retrieval for Inference-Time Knowledge Repair in Factual Question Answering](counterrefine_answer-conditioned_counterevidence_retrieval_for_inference-time_kn.md)
- [\[ACL 2026\] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge](curate_continual_unlearning_in_real_time_with_ensured_preservation_of_llm_knowle.md)
- [\[NeurIPS 2025\] Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](../../NeurIPS2025/information_retrieval/retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)
- [\[AAAI 2026\] Towards Inference-Time Scaling for Continuous Space Reasoning](../../AAAI2026/information_retrieval/towards_inference-time_scaling_for_continuous_space_reasoning.md)

<!-- RELATED:END -->
