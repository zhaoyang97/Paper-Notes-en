---
title: >-
  [Paper Note] Reasoning on Time-Series for Financial Technical Analysis
description: >-
  [ICLR2026][Time Series][Time series reasoning] This paper proposes the Verbal Technical Analysis (VTA) framework, which combines the linguistic reasoning capabilities of LLMs with the pattern-capturing capacity of time-s…
tags:
  - "ICLR2026"
  - "Time Series"
  - "Time series reasoning"
  - "financial technical analysis"
  - "reinforcement learning"
  - "LLM fine-tuning"
  - "interpretable forecasting"
date: 2026-05-08
content_hash: 8f58b8cc468a3c75
---

# Reasoning on Time-Series for Financial Technical Analysis

**Conference**: ICLR2026
**arXiv**: [2511.08616](https://arxiv.org/abs/2511.08616)  
**Code**: [chen-jan/VTA](https://github.com/chen-jan/VTA)  
**Area**: Time Series
**Keywords**: Time series reasoning, financial technical analysis, reinforcement learning, LLM fine-tuning, interpretable forecasting
**Authors**: Kelvin J.L. Koa, Jan Chen, Yunshan Ma, Huanhuan Zheng, Tat-Seng Chua (NUS, TUM, SMU, CityU HK)

---

## TL;DR

This paper proposes the Verbal Technical Analysis (VTA) framework, which combines the linguistic reasoning capabilities of LLMs with the pattern-capturing capacity of time-series models. Time-GRPO reinforcement learning is employed to optimize reasoning chains, and inferred attributes are used to condition time-series forecasting, achieving financial time-series prediction that is both accurate and interpretable.

---

## Background & Motivation

**Limitations of LLMs in finance**: Existing financial LLMs primarily analyze textual reports (earnings Q&A, sentiment analysis) while neglecting interpretable analysis of historical price data — i.e., Technical Analysis (TA) — which is critically important for trading practitioners.

**LLMs are poor at time-series reasoning**: Prior work (Merrill et al., 2024) has shown that LLMs perform "remarkably bad" at zero-shot time-series reasoning, and feeding raw time-series data directly yields poor results.

**Time-series LLMs sacrifice interpretability**: Methods such as Time-LLM and CALF modify the embedding space to produce time-series forecasts, but in doing so LLMs lose their natural language reasoning capability and cannot provide interpretable analysis.

**Inadequacy of existing interpretable approaches**: The most closely related work, TimeCAP, produces only classification label predictions rather than full time-series trajectories, and its reasoning relies on external auxiliary data rather than endogenous signals.

**Cross-domain challenge**: The task requires switching between two domains — the time-series domain (stock prices) for inputs and outputs, and the natural language domain for the reasoning process — which increases modeling difficulty.

**Inherently interpretable signals in financial time series**: Unlike general time series, financial data contains a wealth of expert-studied technical indicators (MACD, RSI, Bollinger Bands, etc.), which provide a natural anchor for language-based reasoning.

---

## Method

### Overall Architecture

The VTA framework consists of three core components:

- **Time-Series Reasoning**: Teaches the LLM to perform linguistic reasoning over time-series inputs.
- **Time-Series Forecasting**: Uses a backbone time-series model to capture underlying complex patterns.
- **Joint Conditional Training**: Injects reasoning attributes as conditions into the time-series forecasting process.

### Problem Formulation

Given a historical input of $T$ trading days $\mathbf{X} = \{\mathbf{x}_{t-T+1}, \ldots, \mathbf{x}_t\}$, where $\mathbf{x}_t = [o_t, h_t, l_t, v_t, c_t, p_t]$ (open, high, low, volume, close, and adjusted close price), the goal is to generate:

- A linguistic reasoning trace $\mathbf{v}$
- Price forecasts $\mathbf{y} = \{p_{t+1}, \ldots, p_{t+T'}\}$ for the next $T'$ trading days

Experiments use $T = T' = 10$ (short-term forecasting).

### Time-Series Reasoning (Time-GRPO)

**Textual annotation**: Raw time-series data is converted into a textual annotation $\mathbf{X'} = \mathbf{f}(\mathbf{X})$, including statistical summaries (mean/min/max) and financial technical indicators (moving averages, momentum, MACD, RSI, Bollinger Bands, etc.).

**Time-GRPO objective**: Adapted from GRPO (Shao et al., 2024), with the core formulation:

$$\mathcal{L}_{\text{time-grpo}}(\theta) = \mathbb{E}_{\mathbf{q} \sim \mathcal{Q}} \frac{1}{G} \sum_{i=1}^{G} \left( \min\left(\frac{\pi_\theta(\mathbf{o_i}|\mathbf{q})}{\pi_{\theta_{\text{old}}}(\mathbf{o_i}|\mathbf{q})} A_i, \text{clip}(\cdot, 1{-}\epsilon, 1{+}\epsilon) A_i \right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right)$$

**Inverse MSE reward**: Encourages reasoning chains to maximize prediction accuracy:

$$r_{\text{MSE}}(\theta) = \frac{1}{\lambda \cdot \|\hat{\mathbf{y}}_\theta - \mathbf{y}\|_2^2}$$

The inverse MSE formulation is used because the reward must be maximized (smaller MSE yields a larger reward).

**Multi-stage training pipeline**:

1. **Cold-Start stage**: Time-GRPO is applied to generate initial training samples; performance gains are limited at this stage but the generated data serves as the foundation for subsequent stages.
2. **Rejection Sampling + SFT stage**: Reasoning chains in the top 10% MSE within each bucket are retained for supervised fine-tuning.
3. **RL Optimization stage**: Time-GRPO is applied again on the model that has already learned to reason, to search for the optimal reasoning strategy.

### Time-Series Forecasting Backbone

Based on GPT-2, adapted via cross-modal fine-tuning:

- Time-series inputs are processed through Embedding + Multi-head Attention to produce projected time tokens $\mathbf{X}_{\text{time}}$.
- PCA is applied to LLM word embeddings to obtain principal component word embeddings $\hat{\mathbf{D}}$.
- Multi-head Cross-Attention aligns time tokens with word embeddings:

$$\mathbf{X}_{\text{text}} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{C}}\right)\mathbf{V}$$

- Layer-wise feature regularization aligns the temporal and text branches:

$$\mathcal{L}_{\text{feature}} = \sum_{n=1}^{N} \gamma^{(N-n)} \text{sim}\left(\phi_{\text{text}}^n(\mathbf{F}_{\text{text}}^n), \phi_{\text{time}}^n(\mathbf{F}_{\text{time}}^n)\right)$$

### Joint Conditional Training

Descriptive attribute classes $\mathbf{c}$ (maximum/minimum/mean values) are extracted from the reasoning output and used to condition the time-series forecasting:

$$\mathcal{L}_{\text{forecast}}(\phi) = \mathbb{E}_{\mathbf{X}, \mathbf{y}, \mathbf{c}} \left[\|\hat{\mathbf{y}}_\psi(\mathbf{X}, \tilde{\mathbf{c}}) - \mathbf{y}\|^2\right]$$

With probability $p_{\text{uncond}}=0.3$, $\mathbf{c}$ is randomly set to null (analogous to Classifier-Free Guidance), jointly training both conditional and unconditional paths. At inference:

$$\hat{\mathbf{y}} = s \cdot \hat{\mathbf{y}}_\psi(\mathbf{X}, \mathbf{c}) + (1-s) \cdot \hat{\mathbf{y}}_\theta(\mathbf{X})$$

where the guidance scale $s=0.1$.

---

## Key Experimental Results

### Main Results: Forecasting Performance Comparison

**Datasets**: ACL18 StockNet (88 US stocks, 2012–2017) + Dow Jones / China A50 / EURO STOXX 50 (2024)

| Model | StockNet MSE | StockNet MAE | All MSE | All MAE |
|------|-------------|-------------|---------|---------|
| GPT-4.1 mini | 0.0846 | 0.1827 | 0.2014 | 0.2376 |
| DeepSeek-R1 | 0.0788 | 0.1853 | 0.1428 | 0.2323 |
| TimesNet | 0.0708 | 0.1789 | 0.1286 | 0.2229 |
| TimeLLM | 0.0704 | 0.1780 | 0.1262 | 0.2210 |
| CALF | 0.0674 | 0.1738 | 0.1235 | 0.2180 |
| **VTA (Ours)** | **0.0659** | **0.1701** | **0.1178** | **0.2122** |

VTA achieves the best MSE and MAE across all four datasets, with an overall MSE improvement of 4.6% and MAE improvement of 2.7%.

### Ablation Study: Contribution of the Multi-Stage Pipeline

| Stage | Llama-3.1-8B MSE | Qwen-2.5-3B MSE | Qwen-2.5-7B MSE |
|------|-----------------|-----------------|-----------------|
| Base Model | 0.1482 | 0.1707 | 0.0949 |
| Cold Start (RL) | 0.1475 | 0.1648 | 0.0941 |
| SFT for Reasoning | 0.1168 | 0.1032 | 0.0893 |
| RL for Reasoning | 0.0955 | 0.0832 | 0.0686 |
| + Conditioning (VTA) | 0.0667 | 0.0672 | **0.0659** |

**Key Findings**:
- Cold-Start RL yields only a 1.6% average improvement but generates the data essential for subsequent stages.
- Rejection sampling + SFT followed by RL achieves a 20.3% improvement, validating the effectiveness of the multi-stage pipeline.
- Conditioning the backbone model further reduces error, demonstrating the complementary benefit of combining external reasoning with internal pattern modeling.
- Qwen-2.5-7B performs best as the reasoning model, but with conditioning training the 3B model achieves comparable performance.

### Reasoning Quality Evaluation

25 financial industry experts from JPMorgan, UBS, Evercore, Allianz, and other institutions conducted blind evaluations (1–5 scale) of reasoning chains from VTA, GPT-4.1 mini, and DeepSeek-R1:

- **Depth, Accuracy, Relevance**: VTA leads significantly, reflecting its effective use of technical indicators and reasoning capacity.
- **Coherence, Clarity**: Gaps are smaller, as general-purpose LLMs retain inherent advantages in text fluency.

### Portfolio Evaluation

| Model | Returns | Volatility | Max Drawdown | Sharpe Ratio |
|------|---------|-----------|-------------|-------------|
| TimeLLM | 0.2185 | 0.1193 | -0.1040 | 1.5230 |
| CALF | 0.2019 | 0.1247 | -0.0981 | 1.4566 |
| **VTA (Ours)** | **0.2409** | **0.1185** | **-0.0883** | **1.7190** |

VTA substantially outperforms baselines on Sharpe Ratio (1.7190 vs. the second-best 1.5230), confirming its practical value in real investment scenarios.

### Reasoning Perturbation Experiments

- Removing technical indicators leads to a noticeable drop in forecasting performance, indicating that the reasoning chain provides genuinely useful guidance signals.
- Adding adversarial noise degrades performance but with inconsistent trends, possibly because the model learns during joint training to rely more heavily on the time-series backbone when reasoning is unreliable.

---

## Highlights & Insights

1. **Elegant cross-domain bridging**: Financial technical indicators serve as a natural bridge between the time-series and language domains, addressing the known weakness of LLMs in processing raw time-series data directly.
2. **Time-GRPO design**: Using inverse MSE as the RL reward directly drives reasoning chain optimization toward prediction accuracy, requiring no manual annotation of reasoning data.
3. **Multi-stage training pipeline**: The progressive design of Cold-Start → Rejection Sampling SFT → RL yields more stable and efficient training.
4. **Transfer of Classifier-Free Guidance**: The conditional guidance technique from diffusion models is adapted for time-series forecasting, jointly training conditional and unconditional paths.
5. **Comprehensive evaluation**: Evaluation covers not only prediction accuracy but also expert reasoning quality scores and Markowitz portfolio validation.

---

## Limitations & Future Work

1. **Restricted to financial time series**: Cross-domain experiments (medical/energy) indicate that VTA's reasoning advantage relies on the inherently interpretable signals of financial technical indicators; on general time-series data, the approach degrades to simple trend extrapolation.
2. **Short-horizon forecasting**: $T=T'=10$ covers only short-term trading scenarios; the effectiveness of long-horizon forecasting remains unvalidated.
3. **Alignment between reasoning and forecasting**: Conditioning uses only simple attributes (max/min/mean); richer information within the reasoning chain (trend direction, indicator signals) is not fully exploited.
4. **Fixed guidance scale**: $s=0.1$ indicates that the model relies primarily on the backbone in practice, and the actual contribution of reasoning guidance is relatively small.
5. **Computational cost**: The multi-stage RL + LLM reasoning + time-series backbone joint training pipeline entails considerable resource consumption.
6. **Base model selection**: Only three LLM bases are tested (Llama-3.1-8B, Qwen-2.5-3B/7B); larger-scale models remain unexplored.

---

## Related Work & Insights

| Direction | Representative Work | Distinction from VTA |
|------|---------|-------------|
| Financial LLMs | Fin-R1, FinMem, SEP | Analyze textual reports/news; do not process price time series |
| Time-series LLMs | Time-LLM, CALF | Modify embedding space; lose language reasoning capability |
| Time-series reasoning | TimeCAP | Relies on external auxiliary data; produces only classification labels |
| LLM time-series reasoning | Merrill et al. | Identifies poor zero-shot TS reasoning in LLMs; VTA addresses this via RL fine-tuning |
| Reasoning optimization | DeepSeek-R1, GRPO | VTA adapts GRPO into Time-GRPO with an inverse MSE reward |

---

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Combining RL-based reasoning optimization (GRPO) with time-series forecasting, and transferring Classifier-Free Guidance to time-series conditioning, are genuinely novel contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, 14+ baselines, ablation studies, expert evaluations, portfolio validation, and cross-domain generalization analysis constitute an exceptionally comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, with well-motivated problem formulation and rich figures and tables.
- **Value**: ⭐⭐⭐⭐ — Interpretable financial forecasting offers direct practical value to practitioners; Sharpe Ratio validation confirms real investment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models](timeomni-1_incentivizing_complex_reasoning_with_time_series_in_large_language_mo.md)
- [\[ICLR 2026\] EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements](edinet-bench_evaluating_llms_on_complex_financial_tasks_using_japanese_financial.md)
- [\[ICML 2026\] Adaptive Time Series Reasoning via Segment Selection](../../ICML2026/time_series/adaptive_time_series_reasoning_via_segment_selection.md)
- [\[AAAI 2026\] A Theoretical Analysis of Detecting Large Model-Generated Time Series](../../AAAI2026/time_series/a_theoretical_analysis_of_detecting_large_model-generated_time_series.md)
- [\[ACL 2026\] Time-RA: Towards Time Series Reasoning for Anomaly Diagnosis with LLM Feedback](../../ACL2026/time_series/time-ra_towards_time_series_reasoning_for_anomaly_diagnosis_with_llm_feedback.md)

</div>

<!-- RELATED:END -->
