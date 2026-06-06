---
title: >-
  [Paper Note] TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale
description: >-
  [ACL 2026][LLM/NLP][Risk Event Discovery] TingIS is an end-to-end risk event discovery system deployed on a fintech platform. Utilizing a five-module architecture (semantic distillation, cascaded routing…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Risk Event Discovery"
  - "Customer Complaint Mining"
  - "Incident Linking"
  - "Stream Processing"
  - "SNR Optimization"
date: 2026-05-08
content_hash: 726ccf42b4cccecb
---

# TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale

**Conference**: ACL 2026  
**arXiv**: [2604.21889](https://arxiv.org/abs/2604.21889)  
**Code**: None  
**Area**: LLM Evaluation  
**Keywords**: Risk Event Discovery, Customer Complaint Mining, Incident Linking, Stream Processing, SNR Optimization

## TL;DR

TingIS is an end-to-end risk event discovery system deployed on a fintech platform. Utilizing a five-module architecture (semantic distillation, cascaded routing, incident linking engine, state management, and multi-dimensional noise reduction), it extracts actionable risk events from massive, noisy customer complaints in real-time, achieving a P90 alert latency of 3.5 minutes and a 95% discovery rate for high-priority incidents.

## Background & Motivation

**Background**: Large-scale online platforms rely on complex microservice and cloud-native architectures where even minor failures can propagate rapidly into large-scale incidents. Internal observability systems (metrics/logs/traces) serve as the first line of defense but are not infallible.

**Limitations of Prior Work**: While customer complaint data is a vital signal for discovering monitoring blind spots, it presents challenges such as extreme noise, high throughput, and semantic complexity. Identifying systemic failures from a flow of 2,000 complaints/minute based on as few as 3 reports faces severe signal-to-noise ratio (SNR) issues. Low SNR systems trigger numerous false positives, leading to alert fatigue for operations teams.

**Key Challenge**: A balance must be struck between extremely high noise, minimal latency, and low miss rates—simultaneously satisfying real-time requirements (minute-level), high discovery rates (>95%), and low false alarm rates.

**Goal**: Build an enterprise-grade system capable of real-time risk event discovery from an average of 300,000 daily customer complaints.

**Key Insight**: Decompose the problem into five orthogonal modules, each addressing a core sub-problem. A hybrid intelligence strategy of "lightweight rule pre-filtering + LLM deep judgment" balances precision and cost.

**Core Idea**: Achieve semantic convergence and identity persistence through a multi-stage incident linking engine (LSH high-speed clustering → LLM purity check → cross-batch historical association → time-decay weighting), supported by cascaded routing and multi-dimensional noise reduction to ensure overall system SNR.

## Method

### Overall Architecture

TingIS consists of three layers (data observation, semantic engine, and long-term memory) and five orthogonal modules: M1 Semantic Distillation, M2 Cascaded Routing, M3 Incident Linking Engine, M4 Incident State Management, and M5 Multi-dimensional Noise Reduction. The design follows three core insights: semantic convergence and identity persistence, hybrid intelligence synergy, and multi-constraint SNR balance.

### Key Designs

**1. Semantic Distillation Module (M1)**

- **Function**: Transform raw customer complaint text into unambiguous semantic units.
- **Mechanism**: Use Qwen3-8B to generate initial summaries in a "subject + problem" format (e.g., "credit card online payment + discount error"), filtering emotional expressions, PII, and irrelevant details, then convert them into vectors using the BGE-M3 model.
- **Design Motivation**: Raw complaint text is noisy, colloquial, and highly diverse; high-density semantic representations are needed under controllable computational costs.

**2. Cascaded Routing Module (M2)**

- **Function**: Accurately attribute complaints to corresponding business domains (biz_code).
- **Mechanism**: A two-stage strategy—keyword matching (high precision for explicit complaints) followed by vector retrieval + BGE-Reranker (high recall for vague complaints). The Reranker is limited to a Top-10 candidate pool to meet latency constraints.
- **Design Motivation**: Significant semantic differences exist between business domains; accurate routing is a prerequisite for subsequent event discovery.

**3. Multi-stage Incident Linking Engine (M3)**

- **Function**: Determine whether multiple complaints point to the same underlying risk event.
- **Mechanism**: Executed in two steps: (a) In-batch efficient aggregation: partition by biz_code → LSH high-speed clustering → LLM purity check (split and generate titles if impure); (b) Cross-batch historical association: use time-decay weighted semantic similarity $s^* = s \cdot e^{-k\Delta t}$ to match historical incidents; an LLM makes the final decision to merge or create if thresholds are exceeded.
- **Design Motivation**: This is the core challenge—accurately determining "incident identity" from varied descriptions over time. LSH ensures efficiency, LLM ensures precision, and time decay prevents "historical inertia."

## Key Experimental Results

### Main Results (Online deployment for one month)

| Metric | Value |
|---|---|
| Avg. Daily Complaints Processed | 300,000+ |
| Peak Throughput | 2,000/min |
| P90 Alert Latency | 3.5 min |
| High-Priority Event Discovery Rate | 95% |

### Ablation Study

| Evaluation Dimension | Comparison Method | TingIS Performance |
|---|---|---|
| Routing Accuracy | Baseline Methods | Significantly outperforms |
| Clustering Quality | Baseline Methods | Significantly outperforms |
| Signal-to-Noise Ratio (SNR) | Baseline Methods | Significantly outperforms |

### Key Findings

1. **3 complaints can trigger alerts**: The system can identify potential risk events from only 3 related complaints, which is critical for early warning.
2. **Hybrid intelligence strategy effectively reduces LLM costs**: Rule pre-filtering significantly reduces input volume; LSH and similarity thresholding gate expensive LLM calls; historical state persistence brings progressive efficiency gains.
3. **Alert penetration mechanism balances fatigue and urgency**: A standard 2-hour silence period prevents alert fatigue, but the system automatically penetrates this window upon detecting explosive growth.
4. **Modular design supports low-cost maintenance**: The five orthogonal modules can be upgraded independently (e.g., replacing with a more powerful LLM or a faster embedding model).

## Highlights & Insights

1. **Complete industrial-grade system design**: This is not just an algorithmic innovation but a full engineering solution covering data observation → semantic processing → incident management → noise reduction → alerting.
2. **Sophisticated three-layer data model**: The state layer (real-time decision), audit layer (immutable evidence chain), and snapshot layer (historical baseline) decouple data requirements across different dimensions.
3. **Time-decay semantic association mechanism**: The formula $s^* = s \cdot e^{-k\Delta t}$ succinctly integrates semantic similarity and temporal proximity, preventing old incidents from incorrectly absorbing new complaints.
4. **Valuable hybrid intelligence paradigm**: The progressive strategy of "rules → retrieval → LLM" maintains precision while controlling computational costs, serving as a model for real-time LLM applications.

## Limitations & Future Work

1. **Specific experimental data not fully presented**: The HTML version of the paper is truncated in the experimental section, and detailed offline benchmark data is missing.
2. **Strong domain specificity**: The system is tailored for fintech customer complaints; migration to other domains would require adjusting routing knowledge bases and noise reduction strategies.
3. **Dependence on LLMs**: Core purity checks and merging decisions rely on LLMs (Kimi-K2), which may face latency fluctuations and cost issues.
4. **Cold start issues**: System performance may degrade in new business domains lacking historical incident data and keyword knowledge bases.
5. **Lack of multilingual discussion**: While complaints may involve multiple languages, the paper does not explicitly discuss multilingual support.

## Related Work & Insights

1. **BGE-M3 (2024)**: The embedding and reranker model used in this study, providing high-quality multilingual semantic representations.
2. **Qwen3-8B (2025)**: The LLM used for semantic distillation, balancing quality and inference cost.
3. **Kimi-K2 (2025)**: Used for purity checks and merging decisions within the linking engine, providing high-quality reasoning.
4. **LSH (Locality-Sensitive Hashing)**: A classic approximate nearest neighbor algorithm used here for high-speed pre-clustering.

## Rating

- **Novelty**: ⭐⭐⭐ — The design integrates several existing technologies; innovation lies primarily in engineering integration and module synergy.
- **Experimental Thoroughness**: ⭐⭐⭐ — Online deployment data proves feasibility, but offline comparative details are incomplete.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture and module relationships are clearly described, with practical cases enhancing readability.
- **Value**: ⭐⭐⭐⭐ — Highly valuable for building enterprise-grade LLM applications, demonstrating engineering practices for LLMs in real-time stream processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Trapped by simplicity: When Transformers fail to learn from noisy features](../../ICLR2026/llm_nlp/trapped_by_simplicity_when_transformers_fail_to_learn_from_noisy_features.md)
- [\[ICLR 2026\] How Catastrophic is Your LLM? Certifying Risk in Conversation](../../ICLR2026/llm_nlp/how_catastrophic_is_your_llm_certifying_risk_in_conversation.md)
- [\[ICML 2026\] Rare Event Analysis of Large Language Models](../../ICML2026/llm_nlp/rare_event_analysis_of_large_language_models.md)
- [\[ICLR 2026\] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery](../../ICLR2026/llm_nlp/llema_evolutionary_search_with_llms_for_multi-objective_material_design.md)
- [\[ICLR 2026\] Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerabilities](../../ICLR2026/llm_nlp/toward_safer_diffusion_language_models_discovery_and_mitigation_of_priming_vulne.md)

</div>

<!-- RELATED:END -->
