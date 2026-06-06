---
title: >-
  [Paper Note] A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting
description: >-
  [ACL 2026][Time Series][Credit Risk Prediction] Ours proposes the FinLangNet framework, which achieves multi-scale credit risk prediction through a dual-module architecture (DeepFM for static features + Transformer with…
tags:
  - "ACL 2026"
  - "Time Series"
  - "Credit Risk Prediction"
  - "Heterogeneous Financial Data"
  - "Dual-Granularity Prompting"
  - "Multi-Scale Prediction"
  - "Industrial Deployment"
date: 2026-05-08
content_hash: cfe4b29407474bac
---

# A Unified Framework for Modeling Heterogeneous Financial Data via Dual-Granularity Prompting

**Conference**: ACL 2026  
**arXiv**: [2404.13004](https://arxiv.org/abs/2404.13004)  
**Code**: [GitHub](https://github.com/didiglobal-fintech-credit-risk/FinLangNet)  
**Area**: Time Series / Financial NLP  
**Keywords**: Credit Risk Prediction, Heterogeneous Financial Data, Dual-Granularity Prompting, Multi-Scale Prediction, Industrial Deployment

## TL;DR

Ours proposes the FinLangNet framework, which achieves multi-scale credit risk prediction through a dual-module architecture (DeepFM for static features + Transformer with a dual-granularity prompting mechanism for temporal behavior). Deployment on the Didi Finance platform resulted in a 6.3pp KS gain and a 9.9% reduction in bad debt rates.

## Background & Motivation

**Background**: Industrial credit scoring systems still rely heavily on statistical learning methods like XGBoost, which require extensive manual feature engineering. Deep learning methods have yet to consistently outperform traditional methods in this domain.

**Limitations of Prior Work**: (1) XGBoost requires time-consuming feature engineering and domain expertise; (2) static models cannot capture the temporal dependencies of user behavior; (3) existing methods only perform point-in-time predictions and cannot model the evolution of creditworthiness across different time windows.

**Key Challenge**: User credit risk is dynamic—users may be safe in the short term but risky in the long term—making a single prediction point insufficient for comprehensive risk management decisions.

**Goal**: Redefine credit scoring from a static binary classification problem to a multi-scale sequence learning problem while processing heterogeneous financial data (static attributes + multi-source temporal behaviors).

**Key Insight**: Drawing inspiration from Transformer and prompting techniques in NLP, this work designs a dual-granularity prompting mechanism to handle the specificities of financial time series data.

**Core Idea**: Use feature-level prompts to capture channel-specific temporal patterns and user-level prompts to aggregate the overall user profile, achieving a complete risk representation from fine-grained to coarse-grained levels.

## Method

### Overall Architecture

FinLangNet consists of two complementary modules: (1) a non-sequential module (DeepFM) to process static user profiles and extract high-order feature interactions; (2) a sequential representation generator (SRG) to process multi-source temporal behavior through a dual-granularity prompting mechanism. The outputs of both modules are fused and passed through multi-task heads to predict default probabilities for different time windows.

### Key Designs

1.  **Non-sequential Module (DeepFM-based)**:
    *   **Function**: Extracts complex interactions from static feature vectors $m \in \mathbb{R}^M$.
    *   **Mechanism**: The FM component captures second-order feature interactions $y_{FM} = \langle w, m \rangle + \sum_{j_1}\sum_{j_2} \langle V_{j_1}, V_{j_2} \rangle m_{j_1} m_{j_2}$, while the DNN component models high-order non-linear relationships. The combination yields the static embedding $O_m$.
    *   **Design Motivation**: Risk signals in static features often emerge from combinations of multiple features (e.g., age × occupation × income), necessitating explicit modeling of feature interactions.

2.  **Sequential Representation Generator (SRG) with Dual-Granularity Prompting**:
    *   **Function**: Extracts user behavior representations from multi-source heterogeneous temporal data.
    *   **Mechanism**: Continuous financial signals are first discretized into tokens to enhance robustness. Then, two levels of prompts are introduced: Feature-level Prompt $\widetilde{\phi}_c$ adds learnable aggregation tokens to each channel sequence to capture channel-specific global patterns; User-level Prompt $P_s$ aggregates across all channels to capture the overall user behavior profile.
    *   **Design Motivation**: Financial sequences differ from natural language—they are multi-source, highly sparse, and noisy—requiring modeling at both channel and user granularities to obtain a complete representation.

3.  **Dynamic Weighted Hybrid Loss Function**:
    *   **Function**: Handles class imbalance and variations in sample difficulty.
    *   **Mechanism**: Weighted Log Loss (WLL) assigns higher penalties to the minority class; dynamic hard sample mining calculates sample weights $\omega_i$ based on the gradient norm $g_i = |\partial \mathcal{L}_i / \partial y'_i|$, automatically increasing weights for samples difficult for the model. The total loss balances regression and classification objectives.
    *   **Design Motivation**: Credit risk data is severely imbalanced (defaults are rare), and the difficulty varies significantly across samples.

### Loss & Training

The total objective function is $\mathcal{L}_{total} = \frac{1}{n} \sum_{i=1}^{n} \omega_i [\beta(y'_i - y_i)^2 + (1-\beta) \mathcal{L}_{WLL,i}]$, where $\beta$ balances regression smoothness and classification stability, and $\omega_i$ is the dynamic weight. Multi-scale prediction uses independent task heads to predict default probabilities for six different time windows.

## Key Experimental Results

### Main Results

| Model | y1 AUC | y1 KS | y2 AUC | y2 KS | y3 AUC | y3 KS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| XGBoost | 72.78 | 32.85 | 75.76 | 37.42 | 70.89 | 30.00 |
| Transformer | 72.54 | 32.62 | 75.95 | 37.98 | 70.97 | 30.12 |
| TimesNet | 72.49 | 32.54 | 75.90 | 37.98 | 70.83 | 29.99 |
| GPT-4.1 (Zero-shot) | 55.90 | 10.85 | 56.80 | 12.50 | 55.15 | 9.30 |
| **FinLangNet (Ours)** | **73.55** | **34.08** | **76.96** | **39.46** | **71.92** | **31.60** |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Remove Feature-level Prompt | KS Gain decreases | Channel-level patterns are important for fine-grained risk characterization |
| Remove User-level Prompt | KS Gain decreases | User-level aggregation is necessary for the overall profile |
| Remove Multi-scale Prediction | Long-term prediction deteriorates | Multiple scales provide reciprocal gradient signals |
| Industrial Deployment | KS +6.3pp, Bad Debt -9.9% | Significantly outperforms the original XGBoost system |

### Key Findings

*   FinLangNet outperforms XGBoost and deep learning baselines across all six time scales.
*   Zero-shot LLM credit scoring performs poorly (AUC only 55-56), proving that this task requires specialized models.
*   Both granularities of the dual-granularity prompting mechanism play indispensable roles.
*   Industrial deployment results are significant; a 6.3pp KS gain provides substantial business value in the financial sector.

## Highlights & Insights

*   Adapts the concept of prompts from NLP to financial time series data, creatively solving the unified representation problem of heterogeneous multi-source data.
*   The redefinition of the problem from "credit scoring as classification" to "credit scoring as multi-scale temporal prediction" is highly insightful.
*   Industrial deployment data strongly validates the practical value of the method—a 6.3pp KS gain and a 9.9% reduction in bad debt rates signify immense economic benefits.

## Limitations & Future Work

*   The method is currently validated only in the Didi Finance scenario; generalization to other financial contexts (e.g., banking, insurance) requires further testing.
*   Interpretability is a critical requirement for financial models, but it is insufficiently discussed in this work.
*   The strategy for choosing hyperparameters ($\alpha, \beta$) for the dynamic weighted loss is not fully explained.
*   Future work could combine LLM knowledge to assist in feature engineering or explore privacy protection under federated learning frameworks.

## Related Work & Insights

*   **vs XGBoost**: Retains XGBoost's advantage in processing static features (via DeepFM) while adding temporal modeling capabilities.
*   **vs Generic Time Series Models (TimesNet, etc.)**: Better fits the multi-source heterogeneous nature of financial data through the dual-granularity prompting mechanism.
*   **vs LLM Zero-shot methods**: Proves that credit scoring requires specialized models; general LLMs are insufficient for this task.

## Rating

*   Novelty: ⭐⭐⭐⭐ Both the dual-granularity prompting mechanism and the problem redefinition are innovative.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Double validation with public datasets and industrial deployment.
*   Writing Quality: ⭐⭐⭐⭐ Clear methodological description and persuasive industrial data.
*   Value: ⭐⭐⭐⭐⭐ Significant industrial impact, providing important reference value for financial AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework](../../ICLR2026/time_series/towards_robust_real-world_multivariate_time_series_forecasting_a_unified_framewo.md)
- [\[ICLR 2026\] Delta-XAI: A Unified Framework for Explaining Prediction Changes in Online Time Series Monitoring](../../ICLR2026/time_series/delta-xai_a_unified_framework_for_explaining_prediction_changes_in_online_time_s.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](../../ICLR2026/time_series/edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICLR 2026\] Reasoning on Time-Series for Financial Technical Analysis](../../ICLR2026/time_series/reasoning_on_time-series_for_financial_technical_analysis.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)

</div>

<!-- RELATED:END -->
