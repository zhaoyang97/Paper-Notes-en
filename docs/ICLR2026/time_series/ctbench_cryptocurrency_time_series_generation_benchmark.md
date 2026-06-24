---
title: >-
  [Paper Note] CTBench: Cryptocurrency Time Series Generation Benchmark
description: >-
  [ICLR2026][Time Series][Cryptocurrency] CTBench is the first benchmark specifically designed for time series generation (TSG) in cryptocurrency markets. Utilizing hourly data from 452 coins, 13 financial metrics, and a "Predictive Utility + Statistical Arbitrage" dual-task evaluation framework, it systematically benchmarks 8 SOTA generative models across 5 families. The study reveals a core trade-off: "high statistical fidelity $\neq$ actual profitability…
tags:
  - "ICLR2026"
  - "Time Series"
  - "Cryptocurrency"
  - "Time Series Generation"
  - "Dual-Task Evaluation"
  - "Statistical Arbitrage"
  - "Financial Metrics"
date: 2026-05-08
content_hash: 0df85dea17f57d8e
---

# CTBench: Cryptocurrency Time Series Generation Benchmark

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=RzT2sombPD](https://openreview.net/forum?id=RzT2sombPD)  
**Code**: https://github.com/MilleXi/CTBench/  
**Area**: Time Series Generation / Quantitative Finance / Benchmark  
**Keywords**: Cryptocurrency, Time Series Generation, Dual-Task Evaluation, Statistical Arbitrage, Financial Metrics

## TL;DR
CTBench is the first benchmark specifically designed for time series generation (TSG) in cryptocurrency markets. Utilizing hourly data from 452 coins, 13 financial metrics, and a "Predictive Utility + Statistical Arbitrage" dual-task evaluation framework, it systematically benchmarks 8 SOTA generative models across 5 families. The study reveals a core trade-off: "high statistical fidelity $\neq$ actual profitability," and provides a practical guide for model selection based on market conditions.

## Background & Motivation
**Background**: Time Series Generation (TSG) has become a fundamental tool for downstream tasks such as data augmentation, anomaly detection, privacy protection, and domain adaptation. The core objective is to synthesize sequences that preserve the temporal dependencies and structural characteristics of real data. However, the vast majority of existing TSG benchmarks and methods target non-financial domains like healthcare, mobility trajectories, and sensors. Even when addressing finance, they primarily focus on stock data and often operate under traditional market assumptions such as "regular trading hours, stable macro signals, and overall stationarity."

**Limitations of Prior Work**: Cryptocurrencies represent an extreme class of financial time series—24/7 continuous trading, lack of intrinsic valuation anchors, and violent fluctuations driven by speculation and fragmented liquidity, with rapid market regime shifts (bull/crash/consolidation). These properties directly violate the built-in assumptions of recent financial benchmarks like FinTSB and FinTSBridge. The authors summarize the shortcomings of existing work into three points: (1) **Narrow domain generalization**—covering only traditional assets like stock indices (SPX, CSI300) with low volatility and limited trading hours, with almost no support for crypto; (2) **Narrow task scope**—most focus only on classification and prediction, ignoring generation itself and crypto-specific trading tasks like arbitrage and market neutrality; (3) **Lack of crypto-specific evaluation**—TSGBench focuses only on statistical fidelity, while FinTSB has limited financial metrics; neither considers continuous trading, heavy-tail risks, or executable signal quality.

**Key Challenge**: A generative model having "low reconstruction error/distributional similarity" in a statistical sense does not mean the data it generates can yield profits when used for trading. A systematic trade-off exists between statistical fidelity and tradability, yet existing benchmarks only measure the former, failing to guide real-world deployment.

**Goal**: Construct a crypto-specific benchmark capable of simultaneously measuring the "predictive fidelity of synthetic data" and its "tradability," linking generation quality with financial utility.

**Key Insight**: The authors' key observation is that the most direct way to verify the utility of synthetic data is not to check its distributional resemblance, but to feed it through two real trading pipelines: one that trains a predictor on synthetic data to trade in the real market, and another that reconstructs residuals to perform mean-reversion arbitrage. A model is truly useful only if its generated data enables downstream strategies to be profitable.

**Core Idea**: A evaluation framework utilizing "Dual-Tasks (Predictive Utility + Statistical Arbitrage) + 13 Financial Metrics" to benchmark TSG models in real trading scenarios, measuring generation quality through economic returns rather than reconstruction error.

## Method

### Overall Architecture
CTBench is not a new generative model, but rather an evaluation pipeline connecting "Cryptocurrency Data $\rightarrow$ TSG Models $\rightarrow$ Dual Trading Tasks $\rightarrow$ Financial Metrics." It unifies five modules: (1) **Crypto-specific datasets** (hourly OHLC + quantitative factor features for 452 coins); (2) **Dual-task evaluation** (Predictive Utility and Statistical Arbitrage); (3) **Trading strategies** (Cross-sectional momentum, Long top quantile, and Proportional weight); (4) **Financial metric suite** (13 metrics across 6 categories); (5) **TSG model library** (8 models across 5 families—GAN/VAE/Diffusion/Flow/Hybrid + classic baselines).

The entire pipeline adopts a walk-forward backtesting protocol to simulate real trading: given a training window size $w$ and test step $s$, with split offsets $\tau \in \{w, w+s, \dots\}$, each $\tau$ defines a training segment $R^{(\tau)}_{\text{train}}=[r_{\tau-w+1},\dots,r_\tau]$ and a test segment $R^{(\tau)}_{\text{test}}=[r_{\tau+1},\dots,r_{\tau+s}]$, with models retrained before each window. Each TSG model $g$ is used in two modes: **Generation mode** samples synthetic sequences from Gaussian noise $R_{\text{gen}}=g(z),\ z\sim\mathcal{N}(0,I)$; **Reconstruction mode** reconstructs real sequences $\hat{R}=g(R)$. These two modes correspond to the two tasks below. The portfolio value evolves as $V_t=V_{t-1}\times(\eta_t\cdot r_t)$, where $\eta_t$ is the hourly weight vector across assets.

### Key Designs

**1. Crypto-specific Dataset: Standardizing the "Real Crypto Market" for Evaluation**

Addressing the "lack of crypto in existing benchmarks," CTBench scrapes historical hourly data for all USDT-quoted spot pairs from Binance, spanning January 2020 to December 2024, intentionally covering various regimes like bull markets, crashes, and consolidations. After cleaning by removing missing records and keeping only USDT pairs, **452 coins** are retained. Each asset at each hour is represented by four standard fields $x_{i,t}=[O_{i,t},H_{i,t},L_{i,t},C_{i,t}]$ (OHLC), stacked into a multi-asset array $D\in\mathbb{R}^{n\times(l+1)\times 4}$, with hourly log returns defined by closing prices $r_{i,t}=\log\frac{C_{i,t}}{C_{i,t-1}}$, resulting in a return matrix $R\in\mathbb{R}^{n\times l}$. Furthermore, $d$-dimensional features commonly used in quantitative trading—Alpha101 factors, Bollinger Bands, RSI, moving averages, etc.—are extracted. The same feature pipeline is applied to both real and synthetic data $\Phi(R)\in\mathbb{R}^{n\times l\times d}$ to ensure comparability. The value of this design lies in organizing the high-volatility, 24/7, highly cross-sectionally dispersed crypto market into "analysis-ready" standardized matrices.

**2. Dual-Task Evaluation Framework: Translating "Generation Quality" into "Economic Returns"**

This is the core innovation of CTBench, directly addressing the "fidelity $\neq$ profitability" contradiction. It designs two complementary tasks to examine the ability of synthetic data to preserve "predictive signals" and "tradable structures."

**Predictive Utility** utilizes the generation mode: The TSG model generates synthetic returns $R_{\text{gen}}=g(z)$, features $\Phi(R_{\text{gen}})$ are extracted to train a predictor $f$ (XGBoost by default), and the trained $f$ is deployed on **real** test data $R_{\text{test}}$ to generate signals, constructing a dollar-neutral long-short portfolio rebalanced hourly. It tests whether the predictor trained on synthetic data can generate economically valuable signals in the real market—i.e., whether the synthetic data preserves transferable predictive structures.

**Statistical Arbitrage** utilizes the reconstruction mode: Model $g$ is trained on real returns $R_{\text{train}}$ and reconstructs $\hat{R}_{\text{train}}$. Residuals $\rho_{i,t}=r_{i,t}-\hat{r}_{i,t}$ are assumed to follow an Ornstein–Uhlenbeck (OU) mean-reverting process, with parameters $(\mu_i,\theta_i,\sigma_i)$ estimated per asset. In the test data, new residuals $\epsilon_{i,t}$ are mapped to standardized s-scores:

$$s_{i,t}=\frac{\epsilon_{i,t}-\mu_i}{\sigma_i/\sqrt{2\theta_i}}$$

Trading is triggered by thresholds ($\gamma=2$), with normalized weights and hourly rebalancing. It tests whether the model can reconstruct market structures and isolate tradable mean-reverting (market-neutral) alpha from the residuals. One task measures "generation" and the other "reconstruction," together bridging the gap between statistical fidelity and actual trading utility.

**3. Financial Metric Suite: 13 Metrics Focused on Practitioner Concerns**

Statistical similarity like MSE is insufficient for financial purposes. CTBench defines **13 core metrics in 6 categories**, each answering a question critical to traders: **Error metrics** (MSE emphasizes volatility mismatch, MAE is robust to outliers) ask if synthetic returns are numerically similar; **Ranking metrics** (Information Coefficient IC, Information Ratio IR) ask if the relative ranking between assets is preserved—crucial for cross-sectional strategies; **Trading Performance** (CAGR, Sharpe Ratio SR) asks if signals actually make money; **Risk Assessment** (MDD, VaR at 95% confidence, Expected Shortage ES) asks about liquidation risk and fat-tail resilience; **Efficiency** (Training time, Inference time) asks if real-time deployment is supported; **Visualization** ($10{,}000$ initial capital equity curves, ranking plots across regimes) asks if results are contextually realistic.

**4. Strategy-Agnostic + Multi-Family Model Library: Ensuring Robustness**

CTBench ensures coverage across two dimensions. It is **strategy-agnostic**, calculating profitability and risk across three common crypto paradigms: Cross-Sectional Momentum (CSM), Long Top Quantile (LOTQ), and Proportional Weight (PW). The model library benchmarks **8 SOTA models across 5 families**—GAN (Quant-GAN, COSCI-GAN), VAE (TimeVAE, KoVAE), Diffusion (Diffusion-TS, FIDE), Flow (Fourier-Flow), and Hybrid (LS4), alongside classic baselines (ARMA-GARCH, Bootstrap) and a statistical arbitrage baseline (PCA).

## Key Experimental Results

### Main Results
Experiments were conducted on 452 USDT pairs using a walk-forward protocol: each period involves 500 days of training and 30 days of testing (Predictive Utility) or 15 days (Statistical Arbitrage), with models retrained before each window.

| Task | Top Performing Models | Key Observations |
|------|---------------|---------|
| Predictive Utility: Error | Diffusion-TS | Lowest MSE/MAE, but **hardly translates to trading returns**. |
| Predictive Utility: Returns | TimeVAE, COSCI-GAN | Preserving structural noise leads to more executable signals; TimeVAE excels in stationary regimes, COSCI-GAN in trending regimes. |
| Predictive Utility: All-weather | Fourier-Flow | Frequency-based structure yields stable mid-to-high performance across metrics. |
| Statistical Arbitrage: Returns | KoVAE, LS4 | Outstanding CAGR/Sharpe, but risk dimensions (MDD, tails) collapse; KoVAE drops significantly after transaction fees. |
| Statistical Arbitrage: Risk | FIDE | Lowest VaR/ES/MDD, but returns are near zero or negative (over-regularization suppresses tradable variance). |
| Classic Baselines | ARMA-GARCH | Top-tier CAGR from 2021-2023, but tail risks (VaR/ES) are consistently the worst. |

Key Finding: **No single model dominates across all metrics**. Fidelity, tradability, and robustness constitute a systematic trade-off.

### Ablation Study
The selection of the predictor is a key variable in the Predictive Utility task. The authors compared 5 predictors to verify the stability of TSG model rankings:

| Predictor | Prediction Error (MSE/MAE) | Ranking Fidelity (IC/IR) | Conclusion |
|--------|------|------|------|
| **XGBoost** (Default) | Low | High | Low error + strong rank correlation, most sensitive to TSG model differences. |
| Linear Regression | High | Near zero | Weak predictive power. |
| Random Forest | Relatively Low | Poor | Reduced error but poor cross-sectional ranking. |
| MLP | High | Unstable | Flexible but inconsistent across regimes. |
| Transformer | Medium | Unstable | Captures long-range patterns but lacks stable ranking. |

Ablation shows that XGBoost provides the most discriminative and trading-relevant evaluation signals.

### Key Findings
- **Low reconstruction/prediction error does not guarantee trading success**: Over-regularization (FIDE) suppresses alpha-bearing variance, whereas models that retain structural noise (TimeVAE, COSCI-GAN) generate more executable signals.
- **Transaction fees compress ranking gaps**: High-turnover models (KoVAE) drop several Sharpe ranks after fees, while models with smoother residuals and lower turnover (TimeVAE, Diffusion-TS) maintain their rankings.
- **Regime dependency is strong**: KoVAE excels in volatile periods but lags in calm ones; LS4 performed exceptionally in the 2023 consolidation but suffered high drawdowns in 2022.
- **Efficiency varies significantly**: VAE-based models are the fastest (TimeVAE training < 1 min), while Diffusion models are the slowest due to iterative denoising.

## Highlights & Insights
- **Redefining "Generation Quality" as "Economic Utility"**: Shifting from distributional distance to evaluating synth data through real trading pipelines using CAGR/Sharpe/MDD.
- **Dual-Task Design**: Covering both "Generation" and "Reconstruction" usage modes, avoiding biased conclusions from testing only one side.
- **Empirical Evidence of "Low Error $\neq$ High Returns"**: The failure of Diffusion-TS in trading despite its high accuracy serves as a counter-example to error-driven TSG evaluation paradigms.
- **Actionable Selection Guide**: Providing a recommendation table based on market regimes (Trending/Mean-reverting/Fee-sensitive/Risk-averse), converting benchmark results into decision support.

## Limitations & Future Work
- **Known Limitations**: Currently uses a limited set of predictors (XGBoost) and residual modeling (OU). Future plans include more advanced predictors, richer residual processes, more coins, and exogenous signals like volume.
- **Single Data Source**: Data is sourced solely from Binance spot USDT pairs; generalization to other exchanges or derivatives/perpetuals is unverified.
- **Idealized Backtesting**: Defaulting to zero fees (with one 0.03% test) does not fully model slippage, liquidity impact, or borrowing costs. 
- **Improvement Directions**: Integrating more realistic execution models and automating "regime diagnosis $\rightarrow$ model selection" using a meta-selector.

## Related Work & Insights
- **vs TSGBench**: TSGBench focuses on statistical fidelity. CTBench argues fidelity $\neq$ tradability and introduces dual tasks to translate quality into economic returns.
- **vs FinTSB / FinTSBridge**: These benchmarks operate on traditional market assumptions. CTBench re-engineers data and metrics for the 24/7, heavy-tailed crypto market.
- **vs Traditional Financial TSG**: Previous works generate under simplified assumptions; CTBench stress-tests them in real crypto long-short and mean-reversion scenarios, highlighting significant differences in actual tradability.

## Rating
- Novelty: ⭐⭐⭐⭐ First crypto TSG benchmark + utility-based evaluation framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 452 coins, 4 years of data, 8 SOTA models + 3 baselines, across multiple strategies and predictors.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and useful practical guidance.
- Value: ⭐⭐⭐⭐⭐ Open-source data and code provide direct value for quantitative practitioners and researchers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SciTS: Scientific Time Series Understanding and Generation with LLMs](scits_scientific_time_series_understanding_and_generation_with_llms.md)
- [\[ICLR 2026\] Latent-to-Data Cascaded Diffusion Models for Unconditional Time Series Generation](latent-to-data_cascaded_diffusion_models_for_unconditional_time_series_generatio.md)
- [\[ACL 2026\] TSAQA: Time Series Analysis Question And Answering Benchmark](../../ACL2026/time_series/tsaqa_time_series_analysis_question_and_answering_benchmark.md)
- [\[ICML 2026\] It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks](../../ICML2026/time_series/its_time_towards_the_next_generation_of_time_series_forecasting_benchmarks.md)
- [\[ICML 2026\] TimeOmni-VL: Unified Models for Time Series Understanding and Generation](../../ICML2026/time_series/timeomni-vl_unified_models_for_time_series_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
