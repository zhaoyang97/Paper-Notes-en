---
title: >-
  [Paper Note] Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring
description: >-
  [ICLR 2026][Anomaly Detection][Paper Note] The authors propose W1-ACAS: a post-hoc, tuning-free adaptive conformal anomaly detection framework. It maps prediction errors from pre-trained Time Series Foundation Models (TSFMs) into anomaly scores directly interpretable as false positive rates (p-values) and learns weights online by minimizing the Wasserstein dist
tags:
  - ICLR 2026
  - Anomaly Detection
date: 2026-05-08
content_hash: 279abdb4e5f745e8
---
# Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring

**Conference**: ICLR 2026  
**Code**: [github.com/ibm-granite/granite-tsfm](https://github.com/ibm-granite/granite-tsfm/tree/main/notebooks/hfdemo/adaptive_conformal_tsad)  
**Area**: Time Series Anomaly Detection / Conformal Prediction  
**Keywords**: Time Series Foundation Models, Conformal Prediction, Adaptive Anomaly Detection, Signal Monitoring, False Positive Rate Control  

## TL;DR
The authors propose W1-ACAS: a post-hoc, tuning-free adaptive conformal anomaly detection framework. It maps prediction errors from pre-trained Time Series Foundation Models (TSFMs) into anomaly scores directly interpretable as false positive rates (p-values) and learns weights online by minimizing the Wasserstein distance to maintain stable false positive control under non-stationary data.

## Background & Motivation
In industrial predictive maintenance and signal monitoring scenarios, there is often a lack of historical data, clean data pipelines, and training expertise, making anomaly detectors that rely on large-scale training difficult to deploy. TSFMs (e.g., Chronos, TTM, TiRex) provide zero-shot forecasting capabilities and can offer reasonable expectations for "normal signals" even in data-scarce environments. Consequently, using prediction residuals for anomaly detection has become a natural path.

**Limitations of Prior Work**: While residual-based methods are intuitive, residual thresholds typically depend on global statistics and assume fixed distributions. These methods are neither suitable for online streaming use nor do they provide probabilistic meaning—users see an anomaly score without knowing the corresponding false positive probability. While conformal prediction can provide distribution-free uncertainty quantification with finite-sample guarantees, standard conformal methods rely on the **exchangeability** assumption, which time series naturally violate due to temporal dependencies and distribution shifts.

**Key Challenge**: The goal is to make anomaly scores interpretable (directly equal to the false alarm rate), robust to complex error distributions, and calibrated under streaming settings where distributions drift over time and samples are non-exchangeable. Existing conformal anomaly detection methods either assume exchangeability with fixed thresholds or optimize for a single false alarm rate, failing to satisfy all these requirements simultaneously.

Anomalies themselves manifest in various forms: point anomalies (single points deviating significantly) and contextual anomalies (anomalous only within a specific temporal context). Effective detection requires models to capture temporal dependencies and adapt to non-stationary distributions—further increasing the risk of failure for "fixed threshold" approaches.

**Goal**: Construct a conformal anomaly score that is calibrated for any false alarm rate $\alpha$ and can adapt to distribution shifts online without fine-tuning the TSFM. **Core Idea**: Define the anomaly score as a weighted conformal p-value and learn the weight vector of historical residuals online by minimizing the 1-Wasserstein distance between the p-value distribution and a uniform distribution across all $\alpha$. The framework is named **W1-ACAS** (1-Wasserstein Adaptive Conformal Anomaly Score).

## Method

### Overall Architecture
W1-ACAS serves as a wrapper for any pre-trained forecaster (TSFM). For each time step and each prediction horizon $d$, the prediction residual $S_{t+1}^d=|Y_{t+1}-\hat Y_{t+1}^d|$ is taken as the nonconformity score. Historical residuals and a set of weights $w$ are used through a weighted conformal quantile function to map the residual into an anomaly score in the form of a p-value. The weights of this mapping are updated online by minimizing the Wasserstein distance between the p-value distribution and a uniform distribution. Finally, p-values from multiple prediction horizons are aggregated using the median to produce the final anomaly score; an alarm is triggered if it falls below the threshold.

```mermaid
flowchart LR
    A[Pre-trained TSFM<br/>Multi-step Forecast] --> B[Residual S_t+1^d<br/>= |Y - Ŷ^d|]
    B --> C[Weighted Conformal Quantile<br/>p-value mapping φ_w]
    C --> D[Online Weight Update w<br/>min W1 · Uniform Dist]
    D --> C
    C --> E[Multi-horizon p-values<br/>Median Aggregation]
    E --> F{p̄ < α ?}
    F -->|Yes| G[Alarm]
```

### Key Designs

**1. Normalizing residuals into "FPR-interpretable" conformal p-values:** The practical difficulty in anomaly detection is that the residual $S$ lacks probabilistic meaning and its distribution drifts. The authors define a weighted conformal quantile $Q_{1-\alpha}(s,w)$ (assigning weight $w_i\in[0,1]$ to each historical sample $i$) and define the anomaly score as $\phi_w(S_{t+1})=\sup\{\alpha\in[0,1]:S_{t+1}\le Q_{1-\alpha}(s,w)\}$. This $\phi_w$ is the weighted conformalized p-value: it maps any real-valued residual into $[0,1]$ and allows direct comparison with any $\alpha$ threshold—the user-defined $\alpha$ directly represents the expected false positive rate. Proposition 4.1 proves that a detector based on $\mathbb{1}[\phi_w(S_{t+1})<\alpha]$ is equivalent to the original detector based on weighted quantiles, thus inheriting the false positive rate guarantees.

**2. Leveraging non-exchangeable conformal bounds to determine optimal weights:** By applying the non-exchangeable conformal bounds from Barber et al. (2023) (Proposition 3.1), the false alarm rate is bounded near $\alpha + \sum_i \frac{w_i}{\|w\|_1+1}d_{TV}(s,s_i)$, where $d_{TV}(s,s_i)$ is the total variation distance between two sequences after swapping the test point with the $i$-th historical point. This provides clear engineering intuition: higher weights should be assigned to historical residuals that are **approximately exchangeable** with the test sample, while outliers should be downweighted. Simultaneously, the lower bound encourages $\|w\|_1$ to be as large as possible to maintain effective sample size. This transforms "adaptive weighting" from a heuristic into a principled optimization problem.

**3. Using Wasserstein distance to formulate "All-FPR Calibration" as a differentiable objective:** Ideally, on normal data, $\phi_w(S_{t+1})\sim U[0,1]$, meaning $P(\phi_w\le\alpha)=\alpha$ for all $\alpha$. Accordingly, the authors minimize the 1-Wasserstein distance between the CDF of the p-values and the CDF of a uniform distribution: $\min_w W_1(F_{\phi_{t+1}(w)},F_U)$, subject to an effective sample size constraint $\|w\|>1/\alpha_c-1$ (where $\alpha_c$ is the critical false alarm rate). The key observation is the dual form $W_1=\mathbb{E}_{\alpha\sim U}|P(\phi_{t+1}\le\alpha)-\alpha|$, meaning that minimizing the Wasserstein distance is **equivalent to minimizing the calibration error consistently across all false alarm rates**—perfectly aligning with the desire for "calibration for any $\alpha$".

**4. Online streaming optimization via closed-form gradients and PGD:** Under the empirical CDF, the $W_1$ objective is the sum of integrals of piecewise linear functions, making it differentiable with respect to each $\phi_{t+i}(w)$, and consequently with respect to $w$ with closed-form expressions (Eq. 13, 14). The algorithm maintains an online buffer of size $n_b$, performs Projected Gradient Descent (using ADAM) per Eq. 12, and projects weights back to the feasible region $\{w\in[0,1]^n,\|w\|>n_c\}$. The gradient intuition is straightforward: $\partial W_1/\partial\phi_{t+i}$ pushes the normalized score into its empirical quantile bin, and $\partial\phi_{t+i}/\partial w_k$ suggests that decreasing the weight of historical residuals larger than the current score will increase the p-value. For multi-horizon forecasts, $D$ instances are run in parallel, and the median $\bar\phi_{t+1}=\mathrm{median}_d\phi_{t+1}^d$ is used, requiring more than half of the horizons to trigger an anomaly for a final alarm, thereby reducing false positives.

## Key Experimental Results

### Main Results (univariate, mean±std across 7 datasets)

| Forecaster | AD Method | PA-F1↑ | Affiliation-F↑ | FPR↓ | CalErr↓ | AUC-PR↑ | VUS-PR↑ |
|---|---|---|---|---|---|---|---|
| TiRex | **W1-ACAS** | **0.925** | **0.897** | 0.084 | **0.025** | 0.344 | 0.438 |
| TiRex | Conformal | 0.878 | 0.890 | 0.107 | 0.038 | 0.308 | 0.429 |
| TiRex | Gaussian | 0.714 | 0.837 | 0.119 | 0.090 | 0.270 | 0.432 |
| Chronos | W1-ACAS | 0.912 | 0.893 | 0.077 | 0.025 | 0.355 | 0.440 |
| TTM | W1-ACAS | 0.889 | 0.886 | 0.082 | 0.029 | 0.342 | 0.449 |
| - | CNN* | 0.858 | 0.881 | 0.083 | 0.643 | 0.269 | 0.423 |
| - | MOMENT ZS | 0.596 | 0.867 | 0.261 | 0.417 | 0.110 | 0.461 |
| - | KShape | 0.533 | 0.789 | 0.508 | 0.176 | 0.125 | 0.303 |
| - | USAD* | 0.498 | 0.809 | 0.425 | 0.324 | 0.088 | 0.398 |

(* denotes semi-supervised deep methods requiring training on normal data)

### Ablation Study
- **Across three TSFMs (Chronos / TTM / TiRex)**: W1-ACAS consistently outperforms Gaussian and Conformal offline baselines on every backbone (improving PA-F1 by ~5-20 points and reducing CalErr by over half).
- **Consistency across backbones**: The appendix shows that prediction errors of the three TSFMs are similar across datasets, leading to similar AD performance, which suggests the gains primarily stem from the conformal post-processing rather than a specific backbone.
- **Hyperparameter Sensitivity**: Parameters include context length 52, horizon $D=15$, $\alpha_c=0.01$, $n_b=10$, $\eta=0.001$. Performance is stable with small learning rates and batch sizes; $D=15$ balances performance and sample efficiency.
- **Multi-horizon Aggregation**: The appendix confirms that aggregation (median) further reduces false positives compared to single-horizon detection.
- **Synthetic Experiments**: Validation on synthetic data where true p-values are available demonstrates that the method remains calibrated under both gradual and abrupt distribution shifts.

### Key Findings
- W1-ACAS leads significantly in **threshold-dependent metrics (PA-F1, Affiliation-F)**, even surpassing trained semi-supervised methods (CNN/USAD/OmniAnomaly), while remaining competitive in threshold-independent metrics (AUC, VUS).
- **Calibration Error (CalErr) is extremely low (0.025)**, far superior to deep baselines (CNN reaches 0.643). In low FPR regions, its threshold is the most conservative with the lowest variance, making threshold selection more reliable in practice.
- Classic distance/density methods (KShape, Sub-KNN, SAND), while performing well on some benchmarks, are generally non-causal, require full data access, and have high FPR (0.4-0.5), making them unsuitable for streaming—W1-ACAS outperforms them significantly.
- Learned weights automatically capture temporal patterns like periodicity in errors and adapt quickly to new distributions near anomalous regions, reducing the total number of alarms in end-to-end systems.

## Highlights & Insights
- **Interpretability as a First-class Citizen**: Anomaly scores are not black-box values but directly equate to the false positive rate (p-value). Industrial users can set $\alpha$ to define the false alarm rate, making decisions transparent and actionable.
- **Strong Alignment between Theory and Objective**: The derivation follows a logical loop: non-exchangeable conformal bounds (which samples to weight) $\rightarrow$ Wasserstein calibration objective (minimizing it equals all-$\alpha$ calibration) $\rightarrow$ closed-form differentiable gradients.
- **Completely Post-hoc + Model-agnostic**: It does not modify TSFM parameters and can be attached to any forecaster or any anomaly score, making it ideal for resource-constrained "instant inference" industrial deployments.
- **Statistical Post-processing can Outperform Heavy Models**: On calibration metrics like CalErr, lightweight conformal post-processing far exceeds trained deep semi-supervised methods, suggesting that "Foundation Models + Well-calibrated Statistical Layers" is an undervalued, high-ROI approach.

## Limitations & Future Work
- **Dependency on initial normal data**: Each dataset requires an initial anomaly-free segment to learn weights, limiting performance in pure cold-start scenarios.
- **Multivariate aggregation via p-values**: Multivariate anomalies are handled through horizon-based aggregation without explicitly modeling cross-variable correlations; complex coupled anomalies might be missed.
- **Weights based only on residual distribution similarity**: The authors suggest incorporating contextual features to refine conformal weighting in the future; the current adaptation to "semantically similar but numerically different" contextual anomalies is relatively weak.
- **Manual setting of critical FPR $\alpha_c$**: A smaller $\alpha_c$ requires more historical normal samples $n_c$, which may increase memory and latency overhead in long-term monitoring.

## Related Work & Insights
- **Conformal Prediction Lineage**: Evolution from split conformal (exchangeability assumption) $\rightarrow$ online adaptive conformal (Gibbs & Candès, handling drift but targeting a single FPR) $\rightarrow$ weighted conformal (re-weighting by similarity) $\rightarrow$ this paper's use of Wasserstein to achieve "universal calibration across all $\alpha$".
- **Non-exchangeable Conformal Bounds (Barber 2023)**: This serves as the theoretical pillar for the weighting criteria, grounding "high weights for approximately exchangeable samples" into provable FPR bounds.
- **TSFM for Anomaly Detection**: Compared to zero-shot generic anomaly scoring like MOMENT, this work proves that "TSFM prediction + adaptive conformal post-processing" is more reliable for calibration and false alarm control.
- **Streaming AD Evaluation**: The experiments follow the curated benchmark and oracle threshold strategy of Liu & Paparrizos (2024), reporting both point-level (AUC, PA-F1) and interval-level (VUS, Affiliation-F) metrics, serving as a reference for fair evaluation in time series anomaly detection.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Elevating the goal of weighted conformal from "single FPR coverage" to "all-$\alpha$ uniform calibration" via Wasserstein duality is novel and theoretically sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 7 univariate and multiple multivariate datasets across three TSFMs, comparing against classic, semi-supervised, and zero-shot baselines. Includes sensitivity and synthetic calibration analysis. Lacks large-scale industrial deployment proof.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear progression from motivation to theory to algorithm. Includes complete propositions and gradient derivations. High formula density may be challenging for readers without a conformal background.
- **Value**: ⭐⭐⭐⭐ — Tuning-free, interpretable, and false-alarm-controllable. Directly addresses industrial pain points of data scarcity and has open-source code; high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization](low_rank_transformer_for_multivariate_time_series_anomaly_detection_and_localiza.md)
- [\[ICLR 2026\] Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors](foundation_visual_encoders_are_secretly_few-shot_anomaly_detectors.md)
- [\[ICLR 2026\] MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval](mrad_zero-shot_anomaly_detection_with_memory-driven_retrieval.md)
- [\[ICLR 2026\] ReTabAD: A Benchmark for Restoring Semantic Context in Tabular Anomaly Detection](retabad_a_benchmark_for_restoring_semantic_context_in_tabular_anomaly_detection.md)
- [\[ICLR 2026\] UniOD: A Universal Model for Outlier Detection across Diverse Domains](uniod_a_universal_model_for_outlier_detection_across_diverse_domains.md)

</div>

<!-- RELATED:END -->
