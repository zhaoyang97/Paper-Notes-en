---
title: >-
  [Paper Note] A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting
description: >-
  [ACL 2026][Time Series][credit risk prediction] This paper proposes FinLangNet, a dual-module framework comprising DeepFM for static feature processing and a Transformer with a dual-granularity prompting mechanism for sequential behavior modeling, enabling multi-scale credit risk prediction. Upon deployment on the Didi Finance platform, the system achieves a 6.3 pp improvement in KS and a 9.9% reduction in bad debt rate.
tags:
  - ACL 2026
  - Time Series
  - credit risk prediction
  - heterogeneous financial data
  - dual-granularity prompting
  - multi-scale forecasting
  - industrial deployment
date: 2026-05-08
content_hash: be15a8143cc9a69c
---

# A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting

**Conference**: ACL 2026
**arXiv**: [2404.13004](https://arxiv.org/abs/2404.13004)
**Code**: [GitHub](https://github.com/didiglobal-fintech-credit-risk/FinLangNet)
**Area**: Time Series / Financial NLP
**Keywords**: credit risk prediction, heterogeneous financial data, dual-granularity prompting, multi-scale forecasting, industrial deployment

## TL;DR

This paper proposes FinLangNet, a dual-module framework comprising DeepFM for static feature processing and a Transformer with a dual-granularity prompting mechanism for sequential behavior modeling, enabling multi-scale credit risk prediction. Upon deployment on the Didi Finance platform, the system achieves a 6.3 pp improvement in KS and a 9.9% reduction in bad debt rate.

## Background & Motivation

**State of the Field**: Industrial credit scoring systems remain heavily reliant on statistical learning methods such as XGBoost, requiring extensive manual feature engineering, while deep learning approaches have yet to consistently outperform traditional methods in this domain.

**Limitations of Prior Work**: (1) XGBoost demands time-consuming feature engineering and domain expertise; (2) static models fail to capture temporal dependencies in user behavior; (3) existing methods perform only point-in-time prediction, unable to model the evolution of creditworthiness across different time windows.

**Root Cause**: User credit risk is inherently dynamic—a borrower may appear low-risk in the short term yet present elevated risk over longer horizons—rendering single-point predictions insufficient for comprehensive risk management decisions.

**Paper Goals**: To reformulate credit scoring from a static binary classification task into a multi-scale sequence learning problem that jointly handles heterogeneous financial data (static attributes and multi-source temporal behaviors).

**Starting Point**: Drawing on Transformer architectures and prompting techniques from NLP, the paper designs a dual-granularity prompting mechanism tailored to the specific characteristics of financial time-series data.

**Core Idea**: Feature-level prompts capture channel-specific temporal patterns, while user-level prompts aggregate an overall user profile, yielding a complete risk representation spanning fine-grained to coarse-grained granularities.

## Method

### Overall Architecture

FinLangNet consists of two complementary modules: (1) a non-sequential module based on DeepFM that processes static user profiles and extracts high-order feature interactions; and (2) a Sequential Representation Generator (SRG) that handles multi-source temporal behaviors via a dual-granularity prompting mechanism. The outputs of both modules are fused and passed through multi-task prediction heads to estimate default probabilities over multiple time windows.

### Key Designs

1. **Non-Sequential Module (DeepFM-based)**:

    - **Function**: Extracts complex interactions from the static feature vector $m \in \mathbb{R}^M$.
    - **Mechanism**: The FM component captures second-order feature interactions via $y_{FM} = \langle w, m \rangle + \sum_{j_1}\sum_{j_2} \langle V_{j_1}, V_{j_2} \rangle m_{j_1} m_{j_2}$; the DNN component models high-order nonlinear relationships. Their combination produces the static embedding $O_m$.
    - **Design Motivation**: Risk signals in static features typically arise from multi-feature combinations (e.g., age × occupation × income), necessitating explicit feature interaction modeling.

2. **Dual-Granularity Prompting Mechanism in the SRG**:

    - **Function**: Extracts user behavior representations from multi-source heterogeneous time-series data.
    - **Mechanism**: Continuous financial signals are first discretized into tokens to enhance robustness. Two levels of prompts are then introduced—Feature-level Prompts $\widetilde{\phi}_c$ append learnable aggregation tokens to each channel sequence to capture channel-specific global patterns; User-level Prompts $P_s$ aggregate information across all channels to capture the overall user behavioral profile.
    - **Design Motivation**: Financial sequences differ fundamentally from natural language—they are multi-source, highly sparse, and noisy. Modeling at both channel and user granularities is necessary to obtain a complete representation.

3. **Dynamically Weighted Hybrid Loss**:

    - **Function**: Addresses class imbalance and sample difficulty variance.
    - **Mechanism**: Weighted log-loss (WLL) assigns higher penalties to the minority class; dynamic hard sample mining computes per-sample weights $\omega_i$ based on gradient norms $g_i = |\partial \mathcal{L}_i / \partial y'_i|$, automatically up-weighting samples that the model finds difficult. The total loss balances regression and classification objectives.
    - **Design Motivation**: Credit risk data suffer from severe class imbalance (defaults constitute the minority class), and sample difficulty varies substantially across instances.

### Loss & Training

The overall objective is $\mathcal{L}_{total} = \frac{1}{n} \sum_{i=1}^{n} \omega_i [\beta(y'_i - y_i)^2 + (1-\beta) \mathcal{L}_{WLL,i}]$, where $\beta$ balances regression smoothness against classification stability, and $\omega_i$ denotes the dynamic sample weight. Multi-scale prediction is performed using independent task heads that estimate default probabilities over six distinct time windows.

## Key Experimental Results

### Main Results

| Model | y1 AUC | y1 KS | y2 AUC | y2 KS | y3 AUC | y3 KS |
|-------|--------|-------|--------|-------|--------|-------|
| XGBoost | 72.78 | 32.85 | 75.76 | 37.42 | 70.89 | 30.00 |
| Transformer | 72.54 | 32.62 | 75.95 | 37.98 | 70.97 | 30.12 |
| TimesNet | 72.49 | 32.54 | 75.90 | 37.98 | 70.83 | 29.99 |
| GPT-4.1 (zero-shot) | 55.90 | 10.85 | 56.80 | 12.50 | 55.15 | 9.30 |
| **FinLangNet** | **73.55** | **34.08** | **76.96** | **39.46** | **71.92** | **31.60** |

### Ablation Study

| Configuration | Key Metric | Observation |
|---------------|-----------|-------------|
| Remove Feature-level Prompt | KS decreases | Channel-level patterns are critical for fine-grained risk characterization |
| Remove User-level Prompt | KS decreases | User-level aggregation is essential for holistic profiling |
| Remove multi-scale prediction | Long-term prediction degrades | Multi-scale heads provide mutual gradient signals |
| Industrial deployment | KS +6.3 pp, bad debt rate −9.9% | Significantly outperforms the incumbent XGBoost system |

### Key Findings
- FinLangNet consistently outperforms both XGBoost and deep learning baselines across all six time scales.
- Zero-shot LLM-based credit scoring performs poorly (AUC of only 55–56), confirming that this task requires domain-specific models.
- Both granularity levels of the dual-granularity prompting mechanism contribute uniquely and irreplaceably.
- The industrial deployment results are substantial: a 6.3 pp KS gain carries significant commercial value in the financial domain.

## Highlights & Insights
- The transfer of the prompting paradigm from NLP to financial time-series processing creatively addresses the unified representation of heterogeneous multi-source data.
- The reframing of credit scoring from classification to multi-scale temporal prediction is a particularly insightful problem reformulation.
- The industrial deployment results compellingly validate the practical value of the approach—a 6.3 pp KS improvement and a 9.9% reduction in bad debt rate translate to substantial economic benefits.

## Limitations & Future Work
- Validation is currently limited to the Didi Finance setting; generalization to other financial domains (e.g., banking, insurance) requires further investigation.
- Model interpretability is a hard requirement in financial applications, yet this aspect receives insufficient discussion in the paper.
- The strategy for selecting hyperparameters of the dynamically weighted loss ($\alpha$, $\beta$) is not thoroughly explained.
- Future work could explore leveraging LLM knowledge to assist feature engineering, or investigate privacy-preserving approaches within federated learning frameworks.

## Related Work & Insights
- **vs. XGBoost**: The proposed framework retains the advantages of XGBoost for static feature processing (via DeepFM) while augmenting it with temporal modeling capability.
- **vs. General time-series models (TimesNet, etc.)**: The dual-granularity prompting mechanism better accommodates the multi-source heterogeneous nature of financial data.
- **vs. Zero-shot LLM methods**: The results demonstrate that credit scoring requires purpose-built models, and general-purpose LLMs are inadequate for this task.

## Rating
- Novelty: ⭐⭐⭐⭐ — Both the dual-granularity prompting mechanism and the problem reformulation are genuinely innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Validated on both public benchmarks and industrial deployment.
- Writing Quality: ⭐⭐⭐⭐ — Method descriptions are clear and industrial results are persuasive.
- Value: ⭐⭐⭐⭐⭐ — Strong industrial deployment results; an important reference for financial AI.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](../../ICLR2026/time_series/delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](../../ICLR2026/time_series/edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](../../ICLR2026/time_series/reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)

<!-- RELATED:END -->
