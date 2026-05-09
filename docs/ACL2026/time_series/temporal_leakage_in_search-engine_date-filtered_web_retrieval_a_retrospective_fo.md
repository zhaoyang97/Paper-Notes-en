---
title: >-
  [Paper Note] Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study
description: >-
  [ACL 2026][Time Series][temporal leakage] This paper presents a systematic audit of date filters in Google and DuckDuckGo, revealing that search engine date filtering critically fails in retrospective forecasting evaluation — 71% (Google) and 81% (DuckDuckGo) of questions have at least one page containing significant post-cutoff information leakage, causing the prediction Brier score to drop artificially from 0.24 to 0.10.
tags:
  - ACL 2026
  - Time Series
  - temporal leakage
  - date filtering
  - retrospective forecasting
  - search engine audit
  - evaluation reliability
date: 2026-05-08
content_hash: ab0c54428f69efb5
---

# Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study

**Conference**: ACL 2026
**arXiv**: [2602.00758](https://arxiv.org/abs/2602.00758)
**Code**: [GitHub](https://github.com/theolivecode/WebDataLeakageAudit)
**Area**: Time Series
**Keywords**: temporal leakage, date filtering, retrospective forecasting, search engine audit, evaluation reliability

## TL;DR

This paper presents a systematic audit of date filters in Google and DuckDuckGo, revealing that search engine date filtering critically fails in retrospective forecasting evaluation — 71% (Google) and 81% (DuckDuckGo) of questions have at least one page containing significant post-cutoff information leakage, causing the prediction Brier score to drop artificially from 0.24 to 0.10.

## Background & Motivation

**State of the Field**: Retrospective Forecasting (RF) is the dominant approach for evaluating LLM forecasting ability — backtesting on questions with known answers while requiring retrieved evidence to be strictly limited to before the question's publication date. In practice, nearly all RF systems rely on search engine date filters to enforce information cutoffs.

**Limitations of Prior Work**: (1) Prior work mentions date filtering unreliability only through a small number of manual cases, lacking systematic quantitative study; (2) it is unclear whether temporal leakage is a rare edge case or a systemic issue; (3) the actual impact of leakage on downstream prediction accuracy has not been quantified.

**Root Cause**: The validity of the entire RF evaluation paradigm rests on the assumption that date filtering excludes post-cutoff information — if this assumption does not hold, all RF evaluation results based on date-filtered search are untrustworthy.

**Paper Goals**: To systematically audit date filters of two major search engines, quantify the prevalence and mechanisms of temporal leakage, and measure its practical impact on prediction accuracy.

**Starting Point**: Using 393 resolved forecasting questions from Metaculus, approximately 100 URLs are retrieved per question, and an LLM-as-Judge assigns each page a leakage severity score on a 0–4 scale.

**Core Idea**: Search engine date filtering is unreliable for temporally constrained retrieval — it is systematically undermined by four leakage mechanisms: page updates, related-content modules, unreliable metadata, and absence signals.

## Method

### Overall Architecture

The study proceeds at three levels: (1) leakage audit — per-page leakage scoring of approximately 39K (Google) and approximately 35K (DuckDuckGo) retrieved URLs; (2) downstream impact — comparing LLM prediction accuracy when using leaked versus non-leaked documents; (3) mechanism analysis — categorizing and documenting four temporal leakage pathways.

### Key Designs

1. **Leakage Severity Scoring System (0–4 Scale)**:

    - Function: Quantifies the severity of post-cutoff information present in each retrieved page.
    - Mechanism: 0 = no post-cutoff information or irrelevant to the question; 1 = topically related but uninformative; 2 = weak directional signal; 3 = significant signal supporting strong inference or decisive for part of the answer; 4 = directly reveals the answer. Absence signals (a key source failing to mention expected information) are capped at 3 to avoid over-interpreting omissions.
    - Design Motivation: A binary leaked/not-leaked classification is insufficient; distinguishing degrees of leakage is necessary to assess their differential impact on prediction decisions.

2. **LLM-as-Judge Audit System**:

    - Function: Large-scale automated leakage detection.
    - Mechanism: Each request includes the question title, background, resolution criteria, resolved answer, cutoff date, page content, and scoring rubric with examples. GPT-oss-120b (temperature=0.5) outputs leakage assessments in JSON format. Human validation achieves 76.1% exact-match accuracy (merging scores 0–1), 0.85 quadratic weighted Kappa, and F1=0.82 for direct leakage (score 4).
    - Design Motivation: The scale of approximately 73K URLs makes manual auditing infeasible; LLM-as-Judge provides a scalable solution.

3. **Prediction Experiment Design (Downstream Impact Quantification)**:

    - Function: Directly measures the causal impact of leakage on prediction accuracy.
    - Mechanism: Binary questions opened in 2025 (after LLM knowledge cutoffs) are selected; documents grouped by leakage level are provided to GPT-oss-120b for prediction, and Brier scores are compared. Using 2025 questions ensures fairness for the control condition (no retrieval).
    - Design Motivation: To demonstrate that leakage not only exists but also systematically inflates prediction accuracy.

### Loss & Training

No model training is involved. For documents exceeding 7,680 tokens, MMR is used to extract the most relevant passages (256-token chunks, up to 30 chunks, Qwen-0.6B embedding model, $\lambda=0.7$).

## Key Experimental Results

### Main Results

**Leakage Prevalence**

| Metric | Google | DuckDuckGo |
|--------|--------|------------|
| Questions evaluated | 393 | 389 |
| Total URLs retrieved | 38,879 | 34,454 |
| URLs with post-cutoff information | 33.2% | 34.5% |
| Question-level: ≥1 (topically relevant) | 98.5% | 98.2% |
| Question-level: ≥3 (significant signal) | **71.0%** | **81.2%** |
| Question-level: 4 (directly reveals answer) | **41.0%** | **54.8%** |

**Impact on Prediction Accuracy (93 binary questions from 2025)**

| Retrieval Condition | Avg. Sources | Mean Brier | Median Brier |
|---------------------|-------------|-----------|-------------|
| No retrieval (baseline) | — | 0.244 | 0.090 |
| Score 0 (no post-cutoff info) | 73.5 | 0.242 | 0.102 |
| Scores 2–4 (weak to full leakage) | 9.6 | 0.128 | 0.023 |
| **Scores 3–4 (strong to full leakage)** | **4.8** | **0.108** | **0.014** |
| Score 4 only (full leakage) | 2.6 | 0.129 | 0.014 |

### Ablation Study

**Leakage Rate by Cutoff Year**

| Cutoff Year | Google Leakage Rate | DuckDuckGo Leakage Rate |
|-------------|---------------------|------------------------|
| 2021 | 46.3% | 47.1% |
| 2022 | 46.5% | 48.0% |
| 2023 | 34.5% | 31.4% |
| 2025 | 26.6% | 27.7% |

### Key Findings

- Leakage is systemic rather than incidental — nearly all questions (98%+) have at least one topically relevant post-cutoff information source.
- The Brier score for non-leaked documents (0.242) is nearly identical to the no-retrieval baseline (0.244), indicating that "clean" date-filtered retrieval provides almost no useful information.
- Leakage at scores 3–4 (Brier: 0.108) outperforms score-4-only leakage (Brier: 0.129), because score-3 documents provide context that helps the model more reliably interpret evidence.
- Earlier cutoff dates (2021–2022) exhibit the highest leakage rates (>46%), decreasing for more recent dates (2025: ~27%), as older pages have had more time to accumulate updates.
- Four leakage mechanisms are identified: **direct page updates** (most common), **related-content sidebars** (main text clean but sidebar leaked), **absence signals** (omission in a comprehensive source implies the answer), and **unreliable metadata** (self-reported timestamps incorrect).

## Highlights & Insights

- This work poses a fundamental challenge to the RF evaluation methodology as a whole — nearly all LLM forecasting systems claiming to approach human-level forecasting ability rely on date-filtered search, and their reported performance may be systematically overestimated.
- The absence-signal leakage mechanism is particularly subtle — a timeline covering up to 2025 that fails to mention an expected event implicitly suggests the answer, yet cannot be excluded by any metadata-based filter.
- Comparing non-leaked retrieval against no retrieval yields nearly identical Brier scores, suggesting that even if date filtering worked perfectly, historical documents would provide minimal benefit for prediction.

## Limitations & Future Work

- Only Google and DuckDuckGo are audited; leakage patterns for other search engines may differ.
- The same model (GPT-oss-120b) is used for both leakage detection and prediction experiments, potentially introducing shared interpretive bias.
- MMR-based document processing may miss leakage signals dispersed across excluded passages.
- The paper diagnoses the problem but does not experimentally evaluate mitigation strategies (e.g., Wayback Machine retrieval or frozen snapshot databases).

## Related Work & Insights

- **vs. FutureSearch (2025)**: The latter proposes a frozen web snapshot approach but still uses active Google search for ranking — this paper provides empirical support for abandoning active date-filtered search.
- **vs. Paleka et al. (2026)**: The latter qualitatively raises concerns about date filtering unreliability; this paper is the first to systematically quantify the issue — confirming on approximately 73K URLs that leakage is indeed systemic.
- **vs. ForecastBench (Karger et al., 2025)**: The latter avoids leakage through a prospective benchmark design but iterates slowly — the two approaches are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic quantification of temporal leakage in search engine date filtering, filling an important methodological gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ ~73K URL audit, dual-engine comparison, downstream impact quantification, human validation, and temporal dimension analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is clear, experimental design is rigorous, and leakage mechanism taxonomy is concrete and URL-evidenced.
- Value: ⭐⭐⭐⭐⭐ Has direct and far-reaching implications for RF evaluation methodology; all systems using date-filtered search warrant reexamination.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](../../NeurIPS2025/time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[AAAI 2026\] Task-Aware Retrieval Augmentation for Dynamic Recommendation](../../AAAI2026/time_series/task-aware_retrieval_augmentation_for_dynamic_recommendation.md)
- [\[ACL 2026\] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)
- [\[AAAI 2026\] Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing](../../AAAI2026/time_series/towards_non-stationary_time_series_forecasting_with_temporal_stabilization_and_f.md)

<!-- RELATED:END -->
