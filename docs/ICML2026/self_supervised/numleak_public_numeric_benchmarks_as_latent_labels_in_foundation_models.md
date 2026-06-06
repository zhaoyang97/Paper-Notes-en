---
title: >-
  [Paper Note] NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models
description: >-
  [ICML 2026][Self-Supervised Learning][Numeric benchmark memorization] NumLeak detects and quantifies the degree of memorization of public numerical benchmarks (financial factors, macroeconomic data…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Numeric benchmark memorization"
  - "Foundation model contamination"
  - "Evaluation credibility"
  - "Fama-French factors"
date: 2026-05-08
content_hash: 2159350fdec2aba7
---

# NumLeak: Public Numeric Benchmarks as Latent Labels in Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2605.30393](https://arxiv.org/abs/2605.30393)  
**Code**: To be confirmed  
**Area**: LLM Evaluation / Model Safety / Data Contamination  
**Keywords**: Numeric benchmark memorization, Foundation model contamination, Evaluation credibility, Fama-French factors

## TL;DR
NumLeak detects and quantifies the degree of memorization of public numerical benchmarks (financial factors, macroeconomic data, climate data) in foundation models through a **four-layer diagnostic protocol**. It reveals how such contamination leaks into downstream financial signals and evaluates risk mitigation via system prompt defenses; Opus 4.7 achieves a within-25 bps accuracy of 0.60 and Pearson $r = 0.99$ on the Mkt-RF factor.

## Background & Motivation

**Background**: Foundation model evaluations typically assume that models "learn" during inference rather than "recalled" from training. However, public datasets (e.g., Fama-French factor libraries, unemployment rates, CPI) are widely mirrored on the internet and are highly likely to have entered pre-training corpora.

**Limitations of Prior Work**: Existing memorization detection methods (such as verbatim string extraction by Carlini et al.) are designed for text sequences and cannot capture contamination in the form of **date-indexed continuous numerical sequences**. Furthermore, diagnosing closed-source models is limited by APIs, making it difficult to distinguish between memorized recall and general numerical fluency.

**Key Challenge**: Financial applications of frontier LLMs claim to "generate alpha," yet it is impossible to distinguish whether this alpha stems from genuine reasoning or leakage of pre-training data.

**Goal**: To develop a feasible detection protocol for closed-source models; to quantify the degree of memorization across various domains (finance, macroeconomics, climate); to establish white-box controlled experiments to verify diagnostic signals; and to test the effectiveness of deployment-level defenses.

**Key Insight**: If a model has indeed memorized date-to-value mappings, it should exhibit characteristics that are **factor-selective, rejectable, and decoupled from ranking capabilities** across relevant series.

**Core Idea**: Differentiate true memorization from false positives using a **joint diagnostic signature** (high scores across four indicators simultaneously vs. high scores in only some); cross-validate through four independent diagnostic dimensions.

## Method

### Overall Architecture
The study consists of three progressive stages: (1) **Identification Protocol**: Executing a four-layer diagnostic (factor specificity, temporal control, fictitious probes, and ranking/numerical probes) at the API boundary of production models; (2) **White-box Controlled Verification**: Performing LoRA fine-tuning on Qwen-2.5-1.5B with known contamination levels as the independent variable; (3) **Mitigation Stress Testing**: Evaluating four system prompt defense strategies under six adversarial suffix attacks.

### Key Designs

1.  **Four-metric Joint Diagnostic Signature**:
    - **Function**: Differentiates genuine memorization from false positives.
    - **Mechanism**: $r$ (Pearson correlation) + MAE (Mean Absolute Error in percentage points) + accuracy within 25 bps + sign accuracy + parsing rate.
    - **Design Motivation**: Relying on a single accuracy metric of whether the model outputs "plausible numbers" confuses memorization with fluency. The joint pattern of these four indicators forms a clear "fingerprint": memorization yields high scores across all four, calibrated fluency yields high sign and $r$ but low MAE/25bps, and fabrication yields high sign accuracy only.

2.  **Four-layer Independent Diagnostic Probes**:
    - **Function**: Cross-validates the existence of memorization through controlled experiments across different dimensions.
    - **Mechanism**: (i) **Factor Specificity**: Comparing high scores for Mkt-RF vs. low scores for SMB/HML vs. a zero-baseline for shuffled factors; (ii) **Temporal Control**: Stratifying results by model cutoff dates; (iii) **Fictitious Probes**: Analyzing outputs for non-existent factor names under the same query format; (iv) **Ranking/Numerical Decoupling**: Observing that the model achieves only 52.5% accuracy in a two-month ranking task (vs. a numerical $r = 0.98$), indicating the target is date-indexed continuous values rather than text fragments.
    - **Design Motivation**: Consistency across multiple diagnostic layers is required to exclude confounding variables and establish the credibility of causal inference.

3.  **White-box Dose-Response Verification + Logprob Ranking**:
    - **Function**: Reproduces diagnostic signals on synthetic data with known contamination levels; discovers that greedy decoding underestimates memorization, necessitating supplementation with logprob ranking.
    - **Mechanism**: A synthetic series SMR-A (480 Gaussian values) is integrated into the fine-tuning corpus at frequencies of $0\times / 1\times / 5\times / 20\times$; 4 seeds $\times$ 8 epochs of LoRA. **Dose-Response**: Logprob top-1 accuracy increases from 0.10 at $0\times$ $\rightarrow$ $0.67 \pm 0.26$ at $5\times$ $\rightarrow$ 0.93 at $20\times$. **Rank/Value Decoupling**: The strongest $5\times$ seed ranks the ground truth first in logprob for 29/30 months, yet greedy decoding outputs the ground truth for only 5/30 months.
    - **Design Motivation**: Closed-source models lack token-level probability access, and sampling from open endpoints cannot distinguish cases where the "ground truth ranks first but another value is sampled." Logprob ranking captures information accessibility from a ranking perspective.

## Key Experimental Results

### Main Results: Memorization Levels Across Models and Factors

| Model | Mkt-RF $n$ | within-25 bps | Sign Accuracy | Pearson $r$ |
| :--- | :--- | :--- | :--- | :--- |
| Opus 4.7 | 120 | 0.60 | 0.97 | 0.99 |
| Sonnet 4.6 | 117 | 0.35 | 0.94 | 0.97 |
| Haiku 4.5 | 120 | 0.12 | 0.73 | 0.57 |
| GPT-5.4 | 120 | 0.48 | 0.89 | 0.94 |

**Key Findings**: Stronger model capabilities correlate with stronger memorization. For factors like SMB/HML, within-25 bps accuracy is consistently $\le 0.15$; the shuffled factor baseline is approximately 1/19th of the observed value for Sonnet $\times$ Mkt-RF.

### Cross-Domain Replication

| Data Source | Opus $r$ | Sonnet $r$ |
| :--- | :--- | :--- |
| S&P 500 | 1.000 | 0.970 |
| U.S. Unemployment Rate | $\ge 0.995$ | $\ge 0.995$ |
| CPI YoY Inflation | $\ge 0.995$ | $\ge 0.995$ |
| NOAA Avg Monthly Temp | — | $\ge 0.995$ |

### White-box Dose-Response

| Fine-tuning Frequency | Logprob Top-1 Accuracy | Greedy Generation $r$ | Ground Truth Mean Rank |
| :--- | :--- | :--- | :--- |
| $0\times$ | 0.10 | $\approx 0$ | 3.33 |
| $5\times$ | $0.67 \pm 0.26$ | $0.035 \pm 0.262$ | 1.27 |
| $20\times$ | 0.93 | 1.000 | 1.07 |

**Key Findings**
- The dose-response relationship is monotonically increasing; lack of logprob access in closed-source models likely underestimates contamination.
- Defense effectiveness is highly consistent: three defense methods blocked over 99.8% of 960 adversarial attacks.
- Utility costs are concentrated on proximal values: "Soft Admonition" has near-zero cost, while only the "Retrieval Defense" showed a 33% performance drop on proximal values.

## Highlights & Insights
- **Ingenious Four-Metric "Fingerprint"**: While single accuracy metrics can be contaminated by fabricated generation, the joint signature creates a clear identification space based on the intuition that memorization should score high across all dimensions.
- **Decoupling Rank/Value**: The failure in ranking tasks elegantly proves that the target of memorization is continuous value mapping rather than text segments.
- **Profound logprob Supplementation**: This exposes the "black-box underestimation" phenomenon in closed-source API evaluations—endpoints with only sampling capabilities may severely underestimate information leakage.
- **Practical Defense Design**: Soft Admonition can block all adversarial attempts with a single line of instruction at near-zero cost; although Retrieval Defense is the most aggressive, it sacrifices estimation accuracy for proximal values.

## Limitations & Future Work
- The defenses in §5 are deployment-side patches rather than root-cause cures.
- Claims regarding production models are observational and specific to query dates and model versions.
- White-box experiments differ fundamentally from actual pre-training mechanisms, proving "pathway feasibility" rather than "actual occurrence mechanisms."
- The adversarial attack set only tests non-adaptive single-turn suffixes.

## Related Work & Insights
- **vs. Carlini et al. (Text Extraction)**: Carlini focuses on verbatim string recovery; this work focuses on date-indexed continuous value mapping—the latter's unreliable ranking capability serves as a key differentiator.
- **vs. Financial LLM Literature**: Prior works noted contamination but did not precisely pinpoint the mechanism; NumLeak provides the diagnostic framework.
- **Inspiration for Transfer**: The four-metric joint diagnosis can be applied to other forms of data contamination detection (e.g., Wikipedia dumps, GitHub extractions).

## Rating
- Novelty: ⭐⭐⭐⭐⭐  First complete pipeline establishing rigorous identification, verification, and mitigation for date-indexed numerical contamination.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐  Multi-dimensional cross-validation + white-box dose-response + defense stress testing.
- Writing Quality: ⭐⭐⭐⭐  Clear logical hierarchy with detailed appendices.
- Value: ⭐⭐⭐⭐⭐  Directly addresses core risks in financial LLMs, providing a plug-and-play evaluation framework and defense templates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](../../ICLR2026/self_supervised/regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[ICML 2026\] Understanding Self-Supervised Learning via Latent Distribution Matching](understanding_self-supervised_learning_via_latent_distribution_matching.md)
- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)

</div>

<!-- RELATED:END -->
