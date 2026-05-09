---
title: >-
  [Paper Note] Contextual and Seasonal LSTMs for Time Series Anomaly Detection
description: >-
  [ICLR 2026][Time Series][time series anomaly detection] To address "small-magnitude point anomalies" and "slowly rising anomalies" that existing methods struggle to detect in univariate time series, this paper proposes CS-LSTMs, a dual-branch architecture in which S-LSTM models periodic evolution in the frequency domain and C-LSTM captures local trends in the time domain. Combined with a wavelet-based noise decomposition strategy, the method comprehensively outperforms state-of-the-art approaches on four benchmarks while improving inference speed by 40%.
tags:
  - ICLR 2026
  - Time Series
  - time series anomaly detection
  - LSTM
  - frequency domain
  - noise decomposition
  - univariate time series
date: 2026-05-08
content_hash: e8d73e561e287d9f
---

# Contextual and Seasonal LSTMs for Time Series Anomaly Detection

**Conference**: ICLR 2026
**arXiv**: [2602.09690](https://arxiv.org/abs/2602.09690)
**Code**: [https://github.com/NESA-Lab/Contextual-and-Seasonal-LSTMs-for-TSAD](https://github.com/NESA-Lab/Contextual-and-Seasonal-LSTMs-for-TSAD)
**Area**: AI Safety / Time Series Anomaly Detection
**Keywords**: time series anomaly detection, LSTM, frequency domain, noise decomposition, univariate time series

## TL;DR
To address "small-magnitude point anomalies" and "slowly rising anomalies" that existing methods struggle to detect in univariate time series, this paper proposes CS-LSTMs, a dual-branch architecture in which S-LSTM models periodic evolution in the frequency domain and C-LSTM captures local trends in the time domain. Combined with a wavelet-based noise decomposition strategy, the method comprehensively outperforms state-of-the-art approaches on four benchmarks while improving inference speed by 40%.

## Background & Motivation

**Background**: Univariate time series (UTS) anomaly detection is a core task in cloud services, IoT, and system monitoring. Mainstream methods are broadly categorized as reconstruction-based (e.g., VAE) and prediction-based (e.g., Transformer/LSTM).

**Limitations of Prior Work**: Through reproduction of state-of-the-art methods including FCVAE, KAN-AD, and TFAD, two anomaly types are found to be particularly difficult to detect: (1) *small-magnitude point anomalies*, where brief minor spikes appear normal within longer windows; and (2) *slowly rising anomalies*, where segment-level deviations gradually diverge from periodic patterns.

**Key Challenge**: Anomaly judgment depends on local context rather than absolute values; a change of the same magnitude may be normal or anomalous depending on context. Existing methods either focus solely on frequency components (ignoring local dependencies) or rely exclusively on temporal information (ignoring periodic evolution).

**Goal**: Three challenges are addressed: (1) capturing local trends rather than absolute values; (2) modeling the dynamic evolution of periodicity rather than treating it as static; and (3) learning normal patterns from data containing anomalies and noise.

**Key Insight**: Combining time-domain and frequency-domain representations, with a dual-branch LSTM architecture to separately handle periodic evolution and local trend variation.

**Core Idea**: Jointly model periodic evolution and local trends via a dual-branch LSTM operating in both the time and frequency domains, augmented by wavelet noise decomposition, to achieve precise detection of subtle anomalies.

## Method

### Overall Architecture
Given a univariate time series $x_{0:t}$, wavelet noise decomposition is first applied to remove noise, yielding $\hat{x}$. The denoised signal is then fed into a dual-branch network: S-LSTM (seasonal branch) segments the historical series into windows, applies FFT to each, and learns periodic evolution; C-LSTM (contextual branch) uses overlapping windows to capture local trends. Each branch independently predicts the mean and variance of future values, and both are jointly trained via an NLL loss. At inference time, anomalies are detected by comparing predicted values against actual observations.

### Key Designs

1. **Wavelet Noise Decomposition**:

   - *Function*: Filters noise and anomalous points prior to training, retaining trend and periodic components.
   - *Mechanism*: The signal is decomposed via wavelet transform into approximation coefficients $c_A$ and detail coefficients $c_D^{(i)}$ at each level. The noise level is estimated using MAD: $\sigma_i = \frac{\text{median}(|c_D^{(i)}|)}{\Phi^{-1}(0.75)}$. A universal threshold $\lambda_i = \sigma_i \sqrt{2\log n}$ is computed, soft-thresholding is applied to the detail coefficients, and the signal is reconstructed.
   - *Design Motivation*: More precise than the pooling-based decomposition in DLinear and more efficient than STL decomposition. Crucially, only denoising is performed—trend/periodic decomposition is not applied—so the complete signal is preserved for the downstream branches.

2. **S-LSTM (Seasonal Branch)**:

   - *Function*: Learns the evolutionary trend of periodic patterns in the historical series.
   - *Mechanism*: The historical series preceding the detection point is partitioned into equal-length, non-overlapping windows. FFT is applied to each window to obtain frequency-domain vectors $z_s \in \mathbb{R}^{n \times w_s}$, which are concatenated with the raw time-domain values as covariates and fed into a single-layer LSTM to predict future periodic patterns.
   - *Design Motivation*: Periodicity evolves dynamically (period length and frequency vary over time), so examining only adjacent cycles is insufficient; modeling evolutionary trends across multiple cycles is necessary.

3. **C-LSTM (Contextual Branch)**:

   - *Function*: Captures short-term local trends and distributional shifts.
   - *Mechanism*: The short historical series preceding the detection point is divided into overlapping windows (with a relatively small window size $w_c$). FFT is applied to each window, and the results are concatenated into $z_c \in \mathbb{R}^{n \times w_c}$, which is fed into a single-layer LSTM to predict future values.
   - *Design Motivation*: Each time point in a UTS carries only a single value, making information scarce. Overlapping windows convert point-level learning into segment-level learning, alleviating the insufficiency of single-point information.

### Loss & Training
A noise-decomposition-aware negative log-likelihood (NLL) loss is employed, jointly predicting mean $\mu$ and variance $\sigma^2$:
$$\mathcal{D}(\mu, \sigma, x, \hat{x}) = \log \sigma^2 + \frac{(x \odot \text{mask} + \hat{x} \odot \tilde{\text{mask}} - \mu)^2}{\sigma^2}$$
where the mask identifies normal regions that use the original value $x$, while anomalous regions use the denoised value $\hat{x}$ as the reference. The total loss is $L = L_s + L_c$.

## Key Experimental Results

### Main Results

| Dataset | Metric | CS-LSTMs | FCVAE (Prev. SOTA) | Gain |
|---------|--------|----------|--------------------|------|
| Yahoo | Best F1 | 0.885 | 0.854 | +3.1% |
| KPI | Best F1 | 0.936 | 0.924 | +1.2% |
| WSD | Best F1 | 0.910 | 0.805 | +10.5% |
| NAB | Best F1 | 0.996 | 0.972 | +0.6% |
| Yahoo | Delay F1 | 0.878 | 0.839 | +3.9% |
| KPI | Delay F1 | 0.879 | 0.851 | +2.8% |
| WSD | Delay F1 | 0.857 | 0.696 | +16.1% |

### Ablation Study

| Configuration | Yahoo Best F1 | KPI Best F1 | WSD Best F1 | Note |
|---------------|---------------|-------------|-------------|------|
| Full CS-LSTMs | 0.885 | 0.936 | 0.910 | Complete model |
| w/o C-LSTM | 0.864 | 0.923 | 0.856 | Removing contextual branch: −5.4% (WSD) |
| w/o S-LSTM | 0.717 | 0.904 | 0.762 | Removing seasonal branch: −16.8% (Yahoo) |
| w/o Covariate | 0.826 | 0.925 | 0.840 | Removing covariates: −7.0% (WSD) |
| w/o Noise Decomp | 0.868 | 0.913 | 0.858 | Removing denoising: −5.2% (WSD) |

### Key Findings
- **S-LSTM contributes most**: Removing S-LSTM causes an 16.8% drop on Yahoo, indicating that modeling periodic evolution is critical for datasets with strong periodicity.
- **WSD shows the largest improvement** (+10.5% / +16.1%): The WSD dataset contains abundant slowly varying segment anomalies, precisely the target scenario for which CS-LSTMs is optimized.
- **Efficiency advantage is prominent**: Only 600K parameters (less than half of the 1.4M used by SOTA methods), with an inference time of 4.62 ms (GPU), representing approximately a 40% reduction.
- **Strong transferability**: In cross-domain evaluation (Yahoo→KPI/WSD), F1 scores of 0.929 and 0.883 are achieved, substantially outperforming other methods.

## Highlights & Insights
- **The time-frequency dual-domain complementary design is elegant**: The frequency domain captures periodicity while the time domain captures local trends; the two LSTM branches are both specialized and complementary, yielding a simple yet effective solution. This dual-perspective paradigm is transferable to other time series tasks.
- **MAD rather than mean/standard deviation for noise decomposition**: This choice provides greater robustness to outliers, since anomalous values do not affect the median. This trick is reusable in any scenario requiring robust statistical estimation.
- **Overlapping windows mitigate information scarcity in UTS**: Reformulating a "point-level" problem as a "segment-level" problem is a general technique for handling univariate data.

## Limitations & Future Work
- **Restricted to UTS**: Multivariate time series scenarios require modeling inter-variable dependencies, which the current architecture does not address.
- **Window sizes require manual tuning**: The selection of $w_s$ and $w_c$ influences results; adaptive window sizing may be preferable.
- **LSTM rather than Transformer**: Although more parameter-efficient and faster, the model's capacity to handle very long sequences may be limited.
- **Evaluation protocol controversy**: The point adjustment strategy (counting a segment as detected if any point within it is identified) may overestimate detection precision in real deployment scenarios.

## Related Work & Insights
- **vs. FCVAE**: FCVAE emphasizes frequency-domain reconstruction but neglects local dependencies; CS-LSTMs addresses this gap through its dual-branch design.
- **vs. Anomaly-Transformer**: Anomaly-Transformer employs a minimax strategy for robustness but incurs high parameter counts and slow inference; CS-LSTMs achieves superior performance at substantially lower cost.
- **vs. KAN-AD**: KAN-AD leverages temporal information for prediction while overlooking frequency-domain details, making it directly complementary to the present approach.

## Rating
- Novelty: ⭐⭐⭐ Dual-branch time-frequency joint modeling is not an entirely new concept, but the specific design and noise decomposition strategy are original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, ten baselines, and complete ablation, transfer, and efficiency experiments.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly analyzed; the method is systematically described.
- Value: ⭐⭐⭐⭐ Highly practical; the lightweight and efficient design is well-suited for industrial deployment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)
- [\[NeurIPS 2025\] ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection](../../NeurIPS2025/time_series/scatterad_temporal-topological_scattering_mechanism_for_time_series_anomaly_dete.md)
- [\[NeurIPS 2025\] Structured Temporal Causality for Interpretable Multivariate Time Series Anomaly Detection](../../NeurIPS2025/time_series/structured_temporal_causality_for_interpretable_multivariate_time_series_anomaly.md)
- [\[AAAI 2026\] CometNet: Contextual Motif-guided Long-term Time Series Forecasting](../../AAAI2026/time_series/cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)

<!-- RELATED:END -->
