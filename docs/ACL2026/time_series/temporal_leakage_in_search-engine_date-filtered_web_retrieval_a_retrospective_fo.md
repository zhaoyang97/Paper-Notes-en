---
title: >-
  [Paper Note] Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study
description: >-
  [ACL 2026][Time Series][Temporal Leakage] This paper provides a systematic audit of date filters on Google and DuckDuckGo, finding that search engine date filtering fail significantly in retrospective forecasting evaluat…
tags:
  - "ACL 2026"
  - "Time Series"
  - "Temporal Leakage"
  - "Date Filtering"
  - "Retrospective Forecasting"
  - "Search Engine Audit"
  - "Evaluation Reliability"
date: 2026-05-08
content_hash: 3d9493ef3fa59a2a
---

# Temporal Leakage in Search-Engine Date-Filtered Web Retrieval: A Retrospective Forecasting Case Study

**Conference**: ACL 2026  
**arXiv**: [2602.00758](https://arxiv.org/abs/2602.00758)  
**Code**: [GitHub](https://github.com/theolivecode/WebDataLeakageAudit)  
**Area**: Time Series  
**Keywords**: Temporal Leakage, Date Filtering, Retrospective Forecasting, Search Engine Audit, Evaluation Reliability

## TL;DR

This paper provides a systematic audit of date filters on Google and DuckDuckGo, finding that search engine date filtering fail significantly in retrospective forecasting evaluations—71% (Google) and 81% (DuckDuckGo) of questions contain at least one page with substantial post-cutoff information leakage, causing the predicted Brier score to artificially drop from 0.24 to 0.10.

## Background & Motivation

**Background**: Retrospective Forecasting (RF) is a mainstream method for evaluating the predictive capabilities of LLMs—backtesting on questions with known answers, requiring retrieved evidence to be strictly limited to dates prior to the question's public disclosure. In practice, almost all RF systems rely on search engine date filters to enforce information cutoffs.

**Limitations of Prior Work**: (1) Previous work only mentioned the possible unreliability of date filtering through a few manual cases, lacking systematic quantitative research; (2) It is unclear whether temporal leakage is a rare edge case or a systemic problem; (3) The actual impact of leakage on downstream prediction accuracy has not been quantified.

**Key Challenge**: The validity of the entire RF evaluation paradigm is built on the assumption that "date filtering can exclude post-cutoff information"—if this assumption does not hold, all RF evaluation results based on date-filtered searches are untrustworthy.

**Goal**: Systematically audit the date filters of two major search engines, quantify the prevalence and mechanisms of temporal leakage, and measure its actual impact on prediction accuracy.

**Key Insight**: Using 393 resolved forecasting questions from the Metaculus platform, approximately 100 URLs were retrieved for each question, and an LLM-as-Judge was used to score the leakage severity of each page on a scale of 0-4.

**Core Idea**: Search engine date filtering is unreliable in temporal back-retrieval—it is systematically undermined by four leakage mechanisms: page updates, related content modules, unreliable metadata, and missing signals.

## Method

### Overall Architecture

The research is conducted across three levels: (1) Leakage Audit—performing page-by-page leakage scoring for approximately 39K (Google) and 35K (DuckDuckGo) retrieved URLs; (2) Downstream Impact—comparing the difference in LLM prediction accuracy when using leaked vs. non-leaked documents; (3) Mechanism Analysis—categorizing and documenting the four pathways of temporal leakage.

### Key Designs

1.  **Leakage Severity Scoring System (Levels 0-4)**:
    *   **Function**: Quantify the severity of post-cutoff information in each retrieved page.
    *   **Mechanism**: 0 = No post-cutoff info or irrelevant to the question; 1 = Topic-related but uninformative; 2 = Weak directional signal; 3 = Significant signal supporting strong reasoning or decisive for partial answers; 4 = Directly reveals the answer. The score for "missing signals" (where a key source fails to mention expected info) is capped at 3 to avoid over-interpreting omissions.
    *   **Design Motivation**: Needs to move beyond simple "leakage/no leakage" binary classification to distinguish the impact of different levels of leakage on forecasting decisions.

2.  **LLM-as-Judge Audit System**:
    *   **Function**: Large-scale automated leakage detection.
    *   **Mechanism**: Each request includes the question title, background, resolution criteria, resolved answer, cutoff date, page content, and scoring rubrics with examples. gpt-oss-120b (temperature=0.5) is used to output leakage evaluations in JSON format. Manual verification achieved 76.1% exact match accuracy (merging grades 0-1), 0.85 quadratic weighted Kappa, and F1=0.82 for direct leakage (grade 4).
    *   **Design Motivation**: The scale of ~73K URLs makes manual auditing infeasible; LLM-as-Judge provides a scalable solution.

3.  **Forecasting Experimental Design (Quantifying Downstream Impact)**:
    *   **Function**: Directly measure the causal impact of leakage on prediction accuracy.
    *   **Mechanism**: Select binary questions opening in 2025 (after the LLM knowledge cutoff), provide documents grouped by leakage level to gpt-oss-120b for prediction, and compare Brier scores. Using 2025 questions ensures a fair comparison for the control group (no retrieval).
    *   **Design Motivation**: To prove that leakage not only exists but indeed systematically inflates prediction accuracy.

### Loss & Training

No model training is involved. For long documents exceeding 7680 tokens, MMR is used to extract the most relevant passages (256-token chunks, maximum 30 chunks, Qwen-0.6B embedding model, $\lambda=0.7$).

## Key Experimental Results

### Main Results

**Prevalence of Leakage**

| Metric | Google | DuckDuckGo |
| :--- | :--- | :--- |
| Evaluated Questions | 393 | 389 |
| Total URLs Retrieved | 38,879 | 34,454 |
| % URLs with Post-cutoff Info | 33.2% | 34.5% |
| Question-level: ≥ Grade 1 (Topic Related) | 98.5% | 98.2% |
| Question-level: ≥ Grade 3 (Significant Signal) | **71.0%** | **81.2%** |
| Question-level: Grade 4 (Direct Reveal) | **41.0%** | **54.8%** |

**Impact on Prediction Accuracy (93 Binary Questions from 2025)**

| Retrieval Condition | Avg. Sources | Mean Brier | Median Brier |
| :--- | :--- | :--- | :--- |
| No Retrieval (Baseline) | — | 0.244 | 0.090 |
| Grade 0 (No Leakage) | 73.5 | 0.242 | 0.102 |
| Grades 2-4 (Weak to Full Leakage) | 9.6 | 0.128 | 0.023 |
| **Grades 3-4 (Strong to Full Leakage)** | **4.8** | **0.108** | **0.014** |
| Grade 4 Only (Full Leakage) | 2.6 | 0.129 | 0.014 |

### Ablation Study

**Change in Leakage Rate by Cutoff Year**

| Cutoff Year | Google Leakage Rate | DuckDuckGo Leakage Rate |
| :--- | :--- | :--- |
| 2021 | 46.3% | 47.1% |
| 2022 | 46.5% | 48.0% |
| 2023 | 34.5% | 31.4% |
| 2025 | 26.6% | 27.7% |

### Key Findings

*   Leakage is systemic rather than incidental—nearly all questions (98%+) have at least one topic-related post-cutoff information point.
*   The Brier score for non-leaked documents (0.242) is almost identical to the no-retrieval baseline (0.244)—indicating that "clean" date-filtered retrieval provides almost no useful information.
*   The Brier score for Grade 3-4 leakage (0.108) is lower than Grade 4 alone (0.129), because Grade 3 documents provide context that helps the model interpret evidence more reliably.
*   Early cutoff dates (2021-2022) have the highest leakage rates (>46%), while more recent dates are lower (2025: ~27%)—because older pages have had more time to accumulate updates.
*   Four leakage mechanisms: **Direct Page Updates** (most common), **Related Content Sidebars** (main text is clean but sidebar leaks), **Missing Signals** (omissions in comprehensive sources imply the answer), and **Unreliable Metadata** (errors in self-reported timestamps).

## Highlights & Insights

*   This work presents a fundamental challenge to the entire RF evaluation methodology—almost all LLM forecasting systems claiming "near-human predictive capability" rely on date-filtered searches, and their performance may be systematically overestimated.
*   The "Missing Signal" leakage mechanism is particularly subtle—a timeline covering up to 2025 that fails to mention an expected event inherently suggests the answer, yet cannot be excluded by any metadata filtering.
*   Comparing non-leaked retrieval with no retrieval reveals almost identical Brier scores, suggesting that even if date filtering worked perfectly, historical documents provide extremely limited help for forecasting.

## Limitations & Future Work

*   Only Google and DuckDuckGo were audited; leakage patterns in other engines might differ.
*   Leakage detection and forecasting experiments used the same model (gpt-oss-120b), potentially introducing shared interpretation bias.
*   MMR document processing might miss leakage signals scattered within excluded passages.
*   The study diagnosed the problem but did not experimentally evaluate mitigation strategies (such as Wayback Machine retrieval or frozen snapshot databases).

## Related Work & Insights

*   **vs FutureSearch (2025)**: The latter proposed frozen webpage snapshots but still used active Google search for ranking—this paper provides empirical support for moving away from active date-filtered searches.
*   **vs Paleka et al. (2026)**: The latter qualitatively raised concerns about date filter unreliability; this paper provides the first systematic quantification—confirming leakage as a systemic issue across ~73K URLs.
*   **vs ForecastBench (Karger et al., 2025)**: The latter used prospective benchmarks to avoid leakage, but at a slower iteration speed—the two approaches are complementary.

## Rating

*   Novelty: ⭐⭐⭐⭐ First study to systematically quantify search engine date filtering leakage, filling a significant methodological gap.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Audit of ~73K URLs, comparison of two engines, quantification of downstream impact, manual verification, and temporal dimension analysis.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous experimental design, and specific categorization of leakage mechanisms with URL evidence.
*   Value: ⭐⭐⭐⭐⭐ Direct and profound impact on RF evaluation methodology; all systems using date-filtered search need to be re-examined.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval](../../ICLR2026/time_series/enhancing_multivariate_time_series_forecasting_with_global_temporal_retrieval.md)
- [\[NeurIPS 2025\] StRap: Spatio-Temporal Pattern Retrieval for Out-of-Distribution Generalization](../../NeurIPS2025/time_series/strap_spatio-temporal_pattern_retrieval_for_out-of-distribution_generalization.md)
- [\[AAAI 2026\] Task-Aware Retrieval Augmentation for Dynamic Recommendation](../../AAAI2026/time_series/task-aware_retrieval_augmentation_for_dynamic_recommendation.md)
- [\[ACL 2026\] STK-Adapter: Incorporating Evolving Graph and Event Chain for Temporal Knowledge Graph Extrapolation](stk-adapter_incorporating_evolving_graph_and_event_chain_for_temporal_knowledge_.md)
- [\[ICML 2026\] Nested Spatio-Temporal Time Series Forecasting](../../ICML2026/time_series/nested_spatio-temporal_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
