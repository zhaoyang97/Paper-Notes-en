---
title: >-
  [Paper Note] DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables
description: >-
  [ICML 2026][Time Series][Covariate Forecasting] For time series forecasting with known future covariates (TSF-X), DAG designs a dual-pathway network: one pathway captures "historical exogenous → future exogenous" attenti…
tags:
  - "ICML 2026"
  - "Time Series"
  - "Covariate Forecasting"
  - "Future Exogenous Variables"
  - "Temporal Correlation"
  - "Channel Correlation"
  - "Attention Injection"
date: 2026-05-08
content_hash: a0117e8bb97da15a
---

# DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables

**Conference**: ICML 2026  
**arXiv**: [2509.14933](https://arxiv.org/abs/2509.14933)  
**Code**: https://github.com/decisionintelligence/DAG  
**Area**: Time Series Forecasting / Exogenous Variable Modeling / Attention Mechanism  
**Keywords**: Covariate Forecasting, Future Exogenous Variables, Temporal Correlation, Channel Correlation, Attention Injection

## TL;DR
For time series forecasting with known future covariates (TSF-X), DAG designs a dual-pathway network: one pathway captures "historical exogenous → future exogenous" attention patterns along the temporal dimension and injects them into "historical endogenous → future endogenous" prediction; the other captures "historical exogenous → historical endogenous" patterns along the channel dimension and injects them into "future exogenous → future endogenous" prediction. On 12 public/new TSF-X datasets, DAG achieves the best MSE in 10/10 cases, significantly outperforming TimeXer, TFT, TiDE, CrossLinear, PatchTST, etc.

## Background & Motivation
**Background**: Time series forecasting has evolved from early ARIMA/ETS to recent Transformer (Informer, Autoformer, PatchTST) and MLP (DLinear, CycleNet, DUET) models, most of which only use endogenous variables (historical values of the target variable). However, in many real-world scenarios, covariates (weather, holidays, grid load) are not only known historically but also for a future period (e.g., weather forecasts, scheduled promotion calendars); such "known future covariates" are among the most powerful levers for improving forecast accuracy.

**Limitations of Prior Work**: (1) PatchTST/DUET completely ignore covariates; (2) TimeXer/CrossLinear use only historical covariates, wasting known future information; (3) TFT/TiDE use both historical and future covariates, but simply concatenate or apply cross-attention, without modeling the two types of correlation structures—temporal and channel—between covariates and target variables, making it easy to learn spurious correlations; (4) Most TSF-X datasets are small and covariate descriptions are inconsistent, making the benchmark itself insufficient.

**Key Challenge**: "Known future covariates" can inject two types of information—temporal: "evolution patterns from historical to future covariates" ≈ "evolution from historical to future endogenous" (Granger causality intuition); channel: "how covariates affect endogenous variables" is stable between history and future (Pearson correlation intuition). Existing methods only treat them as "features" and do not explicitly transfer these two relationships.

**Goal**: (1) Design a predictor that can utilize both historical and future covariates, (2) explicitly model and inject both temporal and channel correlations into the main prediction path, (3) provide a higher-quality TSF-X benchmark.

**Key Insight**: Decompose "covariate information influencing endogenous variable prediction" into two structurally symmetric subtasks—temporal: attention weights from $X^{exo} \to Y^{exo}$ can be borrowed for $X^{endo} \to Y^{endo}$; channel: attention weights from $X^{exo} \to X^{endo}$ can be borrowed for $Y^{exo} \to Y^{endo}$.

**Core Idea**: Extract "covariate → endogenous" correlations as learnable $Q, K$ matrices, then fuse them with original attention via gating, injecting the correlation into the main prediction path, allowing the Transformer to carry extra structured signals in both directions.

## Method

### Overall Architecture
Given $X^{endo} \in \mathbb{R}^{N \times T}$, $X^{exo} \in \mathbb{R}^{D \times T}$, $Y^{exo} \in \mathbb{R}^{D \times F}$, the goal is to predict $\hat Y^{endo} \in \mathbb{R}^{N \times F}$. DAG consists of two symmetric modules:
- **Temporal Correlation Module (TCM)**: A pair of submodules $\mathcal{F}_{\theta_1}, \mathcal{G}_{\theta_2}$—Discovery uses patchify + Transformer to predict $\hat Y^{exo}$ (auxiliary task), outputting attention $W_q', W_k'$; Injection applies the same patchify to $X^{endo}$ and uses the borrowed $W_q', W_k'$ plus its own $W_q, W_k, W_v$ for dual-attention fusion, predicting $\ddot Y^{endo}$.
- **Channel Correlation Module (CCM)**: Mirror design; Discovery uses series-wise embedding + Transformer to predict $\hat X^{endo}$ (auxiliary task, inferring historical endogenous from historical exogenous), outputting $\mathcal{W}_{q'}, \mathcal{W}_{k'}$; Injection injects these into the Transformer for $Y^{exo}$, predicting $\dot Y^{endo}$.
- Final prediction is $\hat Y^{endo} = \lambda_1 \cdot \ddot Y^{endo} + (1-\lambda_1) \cdot \dot Y^{endo}$, with total loss $L_{total} = L_f + \lambda_2 (L_t + L_c)$ jointly optimizing the main and two discovery auxiliary tasks.

### Key Designs
1. **Temporal Correlation Module: Cross-task Transfer of $W_q', W_k'$**:

    - **Function**: Transfers the token-level attention patterns learned from "historical exogenous → future exogenous" directly to "historical endogenous → future endogenous" prediction.
    - **Mechanism**: Discovery patchifies $X^{exo}$ into $M$ tokens, uses standard Transformer to compute $S_i' = \text{softmax}(Q K^\top / \sqrt d) V$ to predict $\hat Y^{exo}$, but only extracts the learnable $W_q', W_k'$ matrices (not attention scores, making it sample-independent and more robust). Injection patchifies $X^{endo}$, computes $(Q, K, V)$ with its own $W_q, W_k, W_v$, and $(Q', K')$ with borrowed $W_q', W_k'$, then $S_{\text{fused}} = \alpha \cdot \sigma(QK^\top/\sqrt d) + (1-\alpha) \sigma(Q' K'^\top / \sqrt d)$, $P_i' = \sigma(S_{\text{fused}}) V$.
    - **Design Motivation**: Directly transferring attention scores would over-couple the two tasks and make them sample-dependent; transferring learnable parameters $W_q', W_k'$ provides a task-level inductive bias, which is more robust and interpretable. Transformer attention essentially models "which token influences which token," a pattern naturally transferable between exogenous and endogenous variables.

2. **Channel Correlation Module: Cross-channel Correlation Transfer**:

    - **Function**: Transfers inter-channel correlations (e.g., temperature affecting sales) from "historical exogenous → historical endogenous" to "future exogenous → future endogenous" prediction.
    - **Mechanism**: Discovery uses series-wise embedding (each sequence becomes a token) to encode each $X^{exo}_i$ into $u_i \in \mathbb{R}^d$, forming $U \in \mathbb{R}^{D \times d}$, runs Transformer to predict $\hat X^{endo}$ (self-supervised auxiliary task), and extracts $\mathcal{W}_{q'}, \mathcal{W}_{k'}$. Injection series-embeds $Y^{exo}$ into $O$, runs the same dual-attention fusion + gating to obtain $\dot Y^{endo}$.
    - **Design Motivation**: Channel correlations are usually stable (e.g., the effect of weather on sales is approximately unchanged between history and future), so direct transfer is a reasonable approximation; series-wise embedding allows attention to operate at the variable level, capturing Pearson correlation structures, consistent with Figure 2 "Pearson correlation for channels" in the paper.

3. **Sample-level Gating $\alpha$ for Adaptive Fusion**:

    - **Function**: Allows each sample to determine the weight between "original attention" and "borrowed attention," addressing the reality that correlation strength varies by sample.
    - **Mechanism**: $\alpha = \text{MLP}(X^{exo}_i)^\top \cdot \text{MLP}(X^{endo}_i)$, where both inputs are passed through MLPs and their dot product yields a scalar weight, which is then used to fuse the two attention scores. TCM and CCM each use a separate gating.
    - **Design Motivation**: If exogenous and endogenous variables are nearly uncorrelated for a sample, $\alpha$ approaches 1, reducing to a standard Transformer; if highly correlated, $\alpha$ approaches 0, letting borrowed attention dominate. This "dynamic balance" avoids overfitting from hard transfer and maintains robustness for low-correlation data.

### Loss & Training
Three losses are summed: $L_t = \|Y^{exo} - \hat Y^{exo}\|_1$ (temporal auxiliary task), $L_c = \|X^{endo} - \hat X^{endo}\|_1$ (channel auxiliary task), $L_f = \|Y^{endo} - \hat Y^{endo}\|_1$ (main prediction). The total loss is $L_{total} = L_f + \lambda_2 (L_t + L_c)$, with $\lambda_1, \lambda_2$ as hyperparameters. All tasks are trained jointly, end-to-end.

## Key Experimental Results

### Main Results
12 TSF-X datasets, some newly released: NP / PJM / BE / FR / DE (5 European electricity prices), Energy / Sdwpfm1/2 / Sdwpfh1/2 / Colbun / Rapel. Baselines include GCGNet (graph correlation), TimeXer / CrossLinear (historical exogenous), TFT / TiDE (historical + future exogenous), DUET / PatchTST / Amplifier / TimeKAN (endogenous only).

| Dataset | DAG MSE | GCGNet | TimeXer | TFT | TiDE | PatchTST | Relative Best Gain |
|---------|---------|--------|---------|-----|------|----------|--------------------|
| NP      | **0.362** | 0.370 | 0.418 | 0.379 | 0.443 | 0.390 | −2.2% vs GCGNet |
| PJM     | **0.093** | 0.095 | 0.108 | 0.114 | 0.142 | 0.133 | −2.1% |
| BE      | **0.423** | 0.431 | 0.452 | 0.454 | 0.498 | 0.577 | −1.9% |
| Energy  | **0.124** | 0.131 | 0.163 | 0.130 | 0.153 | 0.226 | −4.6% |
| Colbun  | **0.098** | 0.107 | 0.145 | 0.238 | 0.164 | 0.239 | −8.4% |
| Rapel   | **0.230** | 0.306 | 0.344 | 0.305 | 0.320 | 0.269 | −14.5% |
| **1st Count** | **10/12** | 2/12 | 0/12 | 0/12 | 0/12 | 0/12 | — |

DAG achieves the best MSE on 10 out of 12 datasets, GCGNet on 2, and TimeXer/TFT/TiDE on none. The advantage is especially pronounced on newly released hydrological/wind datasets such as Colbun, Rapel, and Sdwpfh.

### Ablation Study (Historical Exogenous Only)
Comparison under the "future exogenous unavailable" scenario, verifying that DAG still leads even without future covariates (input only $X^{endo} + X^{exo}$):

| Dataset | DAG MSE | TimeXer | CrossLinear | DUET | Note |
|---------|---------|---------|-------------|------|------|
| NP      | **0.419** | 0.440 | 0.451 | 0.444 | Historical exogenous mode |
| PJM     | **0.126** | 0.141 | 0.147 | 0.140 | DAG still optimal |
| FR      | **0.435** | 0.454 | 0.476 | 0.468 | Channel correlation significant |
| DE      | **0.603** | 0.659 | 0.635 | 0.660 | Weakened but still leading |

| Configuration | Avg. MSE Change | Note |
|---------------|-----------------|------|
| Full DAG      | baseline        | Complete dual-pathway |
| TCM only      | ↑ ~5%           | Lacks channel correlation, weaker future covariate utilization |
| CCM only      | ↑ ~4%           | Lacks temporal correlation, weaker temporal evolution modeling |
| No correlation loss ($\lambda_2 = 0$) | ↑ ~3% | Auxiliary task supervision provides regularization |

### Key Findings
- The core benefit of DAG comes from the combination of "future exogenous + dual correlation injection": in settings without future exogenous (Table 3), the improvement narrows but still leads; with future exogenous (Table 2), the gain increases to 8–15% (especially on Colbun/Rapel).
- Channel correlation contributes more on datasets with many covariates (Sdwpfh 6D, Weather 20D), while temporal correlation is more significant on long sequences (NP/PJM ~52k points); the two are complementary.
- Extracting attention $W_q', W_k'$ instead of directly transferring attention scores yields a stable 1–2% improvement in ablation on 4 datasets, confirming that "task-level parameter transfer is more robust than sample-level score transfer."
- For methods like PatchTST, DUET that do not use covariates, even on datasets with strong covariate signals (electricity prices), they cannot outperform the historical-exogenous-only DAG—reinforcing the claim that "covariates are the biggest lever for TSF accuracy."

## Highlights & Insights
- **"$Q, K$ matrix transfer" is a reusable transfer trick**: Sharing learnable attention projection matrices between two related tasks is more robust and flexible than transferring attention scores; this idea can be applied to any "two related Transformer paths" scenario, such as multimodal or auxiliary task learning.
- **Auxiliary tasks as regularization, dual benefit**: $L_t$ and $L_c$ are ostensibly auxiliary prediction tasks, but in practice provide gradient signals to the main prediction and supervision for $W_q', W_k'$, adding an implicit inductive bias beyond standard multi-task learning.
- **Sample-level gating softens transfer constraints**: $\alpha$ is computed via MLP dot product, ensuring gradient flow and allowing dynamic fallback to standard Transformer; naturally robust to low-correlation samples, this "soft switch" design is worth adopting in other transfer frameworks.
- **The new TSF-X benchmark is a significant community contribution**: Hydrological (Colbun, Rapel) and wind (Sdwpfh, Sdwpfm) datasets fill the gap for covariate-rich datasets; accompanying open-source code and data are hidden values of the paper.

## Limitations & Future Work
- The model consists of 4 Transformer submodules, with parameter count and computational cost significantly higher than PatchTST/DLinear; inference latency and parameter comparisons are not provided in the paper, making deployment less friendly.
- The assumption of "known future covariates" does not hold in many scenarios (e.g., finance, sudden event forecasting); although Table 3 shows "works without future exogenous," the DAG design philosophy fundamentally relies on this assumption, and robustness to noisy future covariates is not analyzed.
- Only validated on 1D covariates; does not address missing/asynchronous covariates (CrossLinear and ExoTST explicitly handle these), which are practical engineering issues.
- $\lambda_1, \lambda_2, \alpha$ involve hyperparameters or additional learnable parameters, as does the gating MLP; the overall hyperparameter space is larger than pure Transformer, and the paper does not provide systematic tuning advice.
- The channel correlation module's series-wise embedding assumes sufficient variable-level tokens; for 1000+ channels (Traffic 861, Electricity 320), attention complexity $O(D^2)$ remains a bottleneck.

## Related Work & Insights
- **vs TimeXer / CrossLinear**: Both use only historical covariates and cannot leverage known future covariates; DAG consistently outperforms TimeXer on all 12 datasets, with MSE 5–15% lower on average, demonstrating the dual advantage of "future covariates + explicit transfer."
- **vs TFT / TiDE**: Both use historical + future covariates, but only via attention/concat, without modeling "temporal × channel" dual correlation; DAG achieves 0.093 vs 0.114 (−18%) on PJM compared to TFT, showing the substantive difference of structured correlation injection.
- **vs GCGNet**: GCGNet models correlation via graph structure, without distinguishing historical/future or endogenous/exogenous; DAG wins on 10/12 datasets, indicating the necessity of distinguishing these roles.
- **vs PatchTST / DUET / DLinear**: Completely ignore covariates; even SOTA DUET lags significantly on the TSF-X benchmark, reinforcing that "univariate-only models are outdated in covariate-rich scenarios."

## Rating
- Novelty: ⭐⭐⭐⭐ "Learnable $Q,K$ matrix cross-task transfer" + dual-pathway structure is a clear original design, though correlation modeling itself has precedents in multi-task learning literature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 datasets + 9 baselines + main experiments + historical-exogenous-only ablation + parameter sensitivity + long lookback + visualization, plus new dataset release, making the experiments extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Figure 1 clearly categorizes the TSF-X landscape, Figure 3 architecture is intuitive; although there are many formulas, sub-sectioning aids readability.
- Value: ⭐⭐⭐⭐ Provides a new SOTA, new benchmark, and open-source code for "known future covariate" scenarios, making it plug-and-play for industry (power, retail, energy scheduling).

## Related Papers

- [\[AAAI 2026\] DeepBooTS: Dual-Stream Residual Boosting for Drift-Resilient Time-Series Forecasting](../../AAAI2026/time_series/deepboots_dual-stream_residual_boosting_for_drift-resilient_time-series_forecast.md)
- [\[AAAI 2026\] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](../../AAAI2026/time_series/xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)
- [\[AAAI 2026\] Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting](../../AAAI2026/time_series/sonnet_spectral_operator_neural_network_for_multivariable_time_series_forecastin.md)
- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](../../ICML2025/time_series/hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [\[ICML 2025\] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](../../ICML2025/time_series/temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting](../../AAAI2026/time_series/sonnet_spectral_operator_neural_network_for_multivariable_time_series_forecastin.md)
- [\[AAAI 2026\] XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs](../../AAAI2026/time_series/xlinear_a_lightweight_and_accurate_mlp-based_model_for_long-term_time_series_for.md)
- [\[AAAI 2026\] DeepBooTS: Dual-Stream Residual Boosting for Drift-Resilient Time-Series Forecasting](../../AAAI2026/time_series/deepboots_dual-stream_residual_boosting_for_drift-resilient_time-series_forecast.md)
- [\[ICML 2026\] Ellipsoidal Time Series Forecasting](ellipsoidal_time_series_forecasting.md)
- [\[ICCV 2025\] VA-MoE: Variables-Adaptive Mixture of Experts for Incremental Weather Forecasting](../../ICCV2025/time_series/va-moe_variables-adaptive_mixture_of_experts_for_incremental_weather_forecasting.md)

</div>

<!-- RELATED:END -->
