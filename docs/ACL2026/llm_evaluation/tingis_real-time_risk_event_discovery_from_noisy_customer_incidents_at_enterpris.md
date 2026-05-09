---
title: >-
  [Paper Note] TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale
description: >-
  [ACL 2026][LLM Evaluation][risk event discovery] TingIS is an end-to-end risk event discovery system deployed on a fintech platform. It employs a five-module architecture—semantic distillation, cascaded routing, event linking engine, state management, and multi-dimensional denoising—to extract actionable risk events from massive noisy customer complaints in real time, achieving a P90 alert latency of 3.5 minutes and a 95% high-priority event discovery rate.
tags:
  - ACL 2026
  - LLM Evaluation
  - risk event discovery
  - customer complaint mining
  - incident linking
  - stream processing
  - SNR optimization
date: 2026-05-08
content_hash: 9dbc0a0a11343fec
---

# TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale

**Conference**: ACL 2026
**arXiv**: [2604.21889](https://arxiv.org/abs/2604.21889)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: risk event discovery, customer complaint mining, incident linking, stream processing, SNR optimization

## TL;DR

TingIS is an end-to-end risk event discovery system deployed on a fintech platform. It employs a five-module architecture—semantic distillation, cascaded routing, event linking engine, state management, and multi-dimensional denoising—to extract actionable risk events from massive noisy customer complaints in real time, achieving a P90 alert latency of 3.5 minutes and a 95% high-priority event discovery rate.

## Background & Motivation

**Background**: Large-scale online platforms rely on complex microservices and cloud-native architectures, where even minor failures can rapidly propagate into large-scale incidents. Internal observability systems (metrics/logs/traces) serve as the first line of defense, yet they are not infallible.

**Limitations of Prior Work**: Customer complaint data represents a critical signal for uncovering monitoring blind spots, but poses extreme challenges due to high noise, high throughput, and semantic complexity. Detecting a systemic failure from only 3 complaints out of a 2,000-per-minute stream presents a severe signal-to-noise ratio (SNR) problem. Low-SNR systems generate excessive false positives, leading to alert fatigue in operations teams.

**Key Challenge**: A fundamental tension exists among achieving high noise tolerance, minimal latency, and near-zero missed detections simultaneously—satisfying real-time requirements (minute-level), high discovery rate (>95%), and low false-positive rate all at once.

**Goal**: To build an enterprise-scale system capable of discovering risk events in real time from 300,000+ customer complaints per day.

**Key Insight**: The problem is decomposed into five orthogonal modules, each addressing a core sub-problem. A hybrid intelligence strategy of "lightweight rule pre-filtering + LLM deep judgment" balances precision and cost.

**Core Idea**: A multi-stage event linking engine (LSH high-speed clustering → LLM purity checking → cross-batch historical association → time-decay weighted scoring) achieves semantic convergence and identity persistence, complemented by cascaded routing and multi-dimensional denoising to maintain overall system SNR.

## Method

### Overall Architecture

TingIS consists of three layers (data observation, semantic engine, and long-term memory) and five orthogonal modules: M1 Semantic Distillation, M2 Cascaded Routing, M3 Event Linking Engine, M4 Event State Management, and M5 Multi-dimensional Denoising. The design follows three core principles: semantic convergence with identity persistence, hybrid intelligence collaboration, and multi-constraint SNR balancing.

### Key Designs

**1. Semantic Distillation Module (M1)**

- **Function**: Transforms raw customer complaint text into unambiguous semantic units.
- **Mechanism**: Qwen3-8B generates initial summaries in a "subject + problem" format (e.g., "credit card online payment + discount error"), filtering emotional expressions, PII, and irrelevant details; BGE-M3 then converts these summaries into vectors.
- **Design Motivation**: Raw complaint text is noisy, colloquial, and highly diverse. High-density semantic representations must be created at controllable computational cost.

**2. Cascaded Routing Module (M2)**

- **Function**: Accurately attributes complaints to the corresponding business domain (`biz_code`).
- **Mechanism**: A two-stage strategy—keyword matching (high precision, for explicit complaints) → vector retrieval + BGE-Reranker re-ranking (high recall, for ambiguous complaints). The reranker operates within a Top-10 candidate pool to satisfy latency constraints.
- **Design Motivation**: Semantic differences across business domains are large; accurate routing is a prerequisite for subsequent event discovery.

**3. Multi-stage Event Linking Engine (M3)**

- **Function**: Determines whether multiple complaints point to the same underlying risk event.
- **Mechanism**: Two steps—(a) *Intra-batch efficient aggregation*: partition by `biz_code` → LSH high-speed clustering → LLM purity checking (impure clusters are split and titled); (b) *Cross-batch historical association*: match against historical events using time-decay weighted semantic similarity $s^* = s \cdot e^{-k\Delta t}$, with LLM as the final arbiter for merge/create decisions when the score exceeds the threshold.
- **Design Motivation**: This is the system's core challenge—accurately determining "event identity" across complaints from different times and phrasings. LSH ensures efficiency; LLM ensures precision; time decay prevents historical inertia from incorrectly absorbing new complaints.

## Key Experimental Results

### Main Results (One Month of Online Deployment)

| Metric | Value |
|---|---|
| Daily complaints processed | 300,000+ |
| Peak throughput | 2,000 complaints/min |
| P90 alert latency | 3.5 minutes |
| High-priority event discovery rate | 95% |

### Ablation Study

| Evaluation Dimension | Baseline | TingIS |
|---|---|---|
| Routing accuracy | Baseline method | Significantly outperforms baseline |
| Clustering quality | Baseline method | Significantly outperforms baseline |
| Signal-to-noise ratio (SNR) | Baseline method | Significantly outperforms baseline |

### Key Findings

1. **Alert triggered from as few as 3 complaints**: The system can detect a potential risk event from only 3 correlated complaints, which is critical for early warning.
2. **Hybrid intelligence strategy effectively reduces LLM cost**: Rule-based pre-filtering substantially reduces input volume; LSH and similarity thresholds gate expensive LLM calls; persistent historical state yields incremental efficiency gains.
3. **Alert penetration mechanism balances fatigue prevention and emergency response**: A 2-hour silence period prevents alert fatigue under normal conditions, but the silence window is automatically bypassed when an explosive growth pattern is detected.
4. **Modular design supports low-cost maintenance**: The five orthogonal modules can be upgraded independently (e.g., replacing a stronger LLM or a faster embedding model).

## Highlights & Insights

1. **Complete industrial-grade system design**: This goes beyond algorithmic innovation, presenting a full engineering solution spanning data observation → semantic processing → event management → denoising → alerting.
2. **Elegantly designed three-layer data model**: The state layer (real-time decision-making), audit layer (immutable evidence chain), and snapshot layer (historical baseline) decouple data requirements across different dimensions.
3. **Time-decay semantic association mechanism**: $s^* = s \cdot e^{-k\Delta t}$ concisely fuses semantic similarity with temporal proximity, preventing old events from incorrectly absorbing new complaints.
4. **Hybrid intelligence paradigm worth emulating**: The progressive "rule → retrieval → LLM" invocation strategy maintains precision while controlling computational cost.

## Limitations & Future Work

1. **Insufficient presentation of experimental details**: The HTML version of the paper is truncated in the experimental section; detailed offline benchmark data is not fully presented.
2. **Strong domain specificity**: The system is designed for fintech customer complaints; transferring it to other domains requires adapting the routing knowledge base and denoising strategies.
3. **LLM dependency**: Core purity checking and merge arbitration rely on an LLM (Kimi-K2), which may face latency variability and cost concerns.
4. **Cold-start problem**: System performance may degrade for new business domains that lack historical events and keyword knowledge bases.
5. **Multilingual scenarios not discussed**: Although complaints may span multiple languages, the paper does not address multilingual support.

## Related Work & Insights

1. **BGE-M3 (2024)**: The embedding and reranker model adopted in this work, providing high-quality multilingual semantic representations.
2. **Qwen3-8B (2025)**: The LLM used for semantic distillation, balancing output quality and inference cost.
3. **Kimi-K2 (2025)**: Used for purity checking and merge arbitration in the event linking engine, providing high-quality reasoning.
4. **LSH (Locality-Sensitive Hashing)**: A classical approximate nearest-neighbor algorithm, used here for high-speed pre-clustering.

## Rating

- **Novelty**: ⭐⭐⭐ — The system design integrates multiple existing techniques; innovation is reflected more in engineering integration and module synergy than in algorithmic novelty.
- **Experimental Thoroughness**: ⭐⭐⭐ — Online deployment data demonstrates system viability, but offline comparison experiment details are insufficiently complete.
- **Writing Quality**: ⭐⭐⭐⭐ — System architecture is described clearly, inter-module relationships are articulated precisely, and real-world cases enhance readability.
- **Value**: ⭐⭐⭐⭐ — Offers strong reference value for building enterprise-grade LLM application systems, demonstrating engineering practice for LLMs in real-time stream processing scenarios.

## Highlights & Insights
To be supplemented after a thorough reading of the paper.

## Limitations & Future Work
To be supplemented after a thorough reading of the paper.

## Related Work & Insights
To be supplemented after a thorough reading of the paper.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Finch: Benchmarking Finance & Accounting across Spreadsheet-Centric Enterprise Workflows](finch_benchmarking_finance_amp_accounting_across_spreadsheet-centric_enterprise_.md)
- [\[NeurIPS 2025\] Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking](../../NeurIPS2025/llm_evaluation/time_travel_is_cheating_going_live_with_deepfund_for_real-time_fund_investment_b.md)
- [\[ACL 2026\] ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition](researchbench_benchmarking_llms_in_scientific_discovery_via_inspiration-based_ta.md)
- [\[CVPR 2026\] Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](../../CVPR2026/llm_evaluation/cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)
- [\[AAAI 2026\] GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](../../AAAI2026/llm_evaluation/goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)

<!-- RELATED:END -->
