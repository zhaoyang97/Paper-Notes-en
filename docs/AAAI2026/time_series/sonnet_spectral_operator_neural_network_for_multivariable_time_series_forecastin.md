---
title: >-
  [Paper Note] Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting
description: >-
  [AAAI 2026 (Oral)][Time Series][Multivariate time series] This paper proposes Sonnet, which maps inputs to the time-frequency domain via learnable wavelet transforms…
tags:
  - "AAAI 2026 (Oral)"
  - "Time Series"
  - "Multivariate time series"
  - "spectral analysis"
  - "wavelet transform"
  - "Koopman operator"
  - "spectral coherence attention"
  - "exogenous variables"
date: 2026-05-08
content_hash: 92743da824260a92
---

# Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting

**Conference**: AAAI 2026 (Oral)  
**arXiv**: [2505.15312](https://arxiv.org/abs/2505.15312)  
**Code**: [https://github.com/ClaudiaShu/Sonnet](https://github.com/ClaudiaShu/Sonnet)  
**Area**: Time Series Forecasting  
**Keywords**: Multivariate time series, spectral analysis, wavelet transform, Koopman operator, spectral coherence attention, exogenous variables

## TL;DR
This paper proposes Sonnet, which maps inputs to the time-frequency domain via learnable wavelet transforms, introduces multivariate coherence-based attention (MVCA) to model inter-variable dependencies, and employs a Koopman operator for stable temporal evolution forecasting. Sonnet achieves state-of-the-art performance on 34 out of 47 forecasting tasks, reducing average MAE by 2.2%.

## Background & Motivation

**Background**: Multivariate time series (MTS) forecasting leverages exogenous variables to predict a single target variable, with broad applications in meteorology, influenza forecasting, and electricity consumption. While Transformer architectures can capture long-range dependencies, they fall short in modeling complex inter-variable relationships.

**Limitations of Prior Work**:  
   - Methods such as iTransformer and Samformer embed sequences along the temporal dimension before applying inter-variable attention, thereby destroying temporal information;  
   - Crossformer and ModernTCN attempt dual-dimension dependency modeling but incur significant GPU overhead;  
   - TimeXer and DeformTime capture inter-variable dependencies only within the receptive field of convolutional kernels, limiting their scope;  
   - Frequency-domain methods (FEDformer, FiLM) focus on intra-variable seasonality modeling, neglecting spectral correlations across variables.

**Key Challenge**: Existing attention mechanisms are not consistently effective for MTS tasks — experiments reveal that removing attention modules from certain models does not significantly degrade performance, indicating that vanilla attention fails to capture inter-variable information.

**Goal**: To design a novel architecture that models inter-variable dependencies in the spectral domain, while simultaneously preserving temporal information and capturing cross-variable correlations.

**Key Insight**: Spectral coherence is a classical signal-processing tool for measuring the frequency-wise correlation between two signals. Incorporating it into attention mechanisms enables quantification of inter-variable frequency-domain dependencies.

**Core Idea**: Learnable wavelets for time-frequency decomposition → spectral coherence attention for inter-variable frequency-domain dependency modeling → Koopman operator for linearized temporal evolution → inverse wavelet transform for reconstruction.

## Method

### Overall Architecture
Sonnet chains four core modules: **joint embedding** → **learnable wavelet transform** → **multivariate spectral coherence attention (MVCA)** → **Koopman temporal evolution** → **inverse wavelet transform reconstruction** → **convolutional decoder**. The input consists of an exogenous variable matrix $\mathbf{X} \in \mathbb{R}^{L \times C}$ and an endogenous variable $\mathbf{y} \in \mathbb{R}^{L}$; a hyperparameter $\alpha$ governs the dimensional allocation between the two in the embedding. The final output is the predicted sequence $\hat{\mathbf{y}} \in \mathbb{R}^{H}$.

### Key Design 1: Learnable Wavelet Transform
- Defines $K$ learnable wavelet atoms, each parameterized as a Gaussian envelope multiplied by a frequency-modulated cosine: $\mathbf{M}_k = \exp(-\mathbf{w}_\alpha \mathbf{t}^2) \times \cos(\mathbf{w}_\beta \mathbf{t} + \mathbf{w}_\gamma \mathbf{t}^2)$
- Three sets of learnable weights $\mathbf{w}_\alpha, \mathbf{w}_\beta, \mathbf{w}_\gamma$ control the Gaussian window width, linear frequency modulation, and quadratic frequency modulation, respectively
- The embedding is projected into the wavelet space via element-wise multiplication: $\mathbf{P}_k = \mathbf{E} \odot \mathbf{M}_k^\top$
- **Advantage**: Unlike fixed mother wavelets, the learnable parameters allow atoms to adapt to the local time-frequency structure of the data; the $K$ atoms naturally correspond to $K$ attention heads

### Key Design 2: Multivariate Spectral Coherence Attention (MVCA)
- For each attention head, Q/K/V linear projections are applied, followed by FFT along the **variable dimension** to transform representations into the frequency domain
- The cross-spectral density $\mathbf{P}_{qk} = \mathbf{Q}_f \odot \mathbf{K}_f^*$ and power spectral densities $\mathbf{P}_{qq}, \mathbf{P}_{kk}$ are computed
- Spectral coherence is obtained via normalization: $\mathbf{C}_{qk} = |\bar{\mathbf{P}}_{qk}|^2 / (\bar{\mathbf{P}}_{qq} \cdot \bar{\mathbf{P}}_{kk} + \epsilon)$
- The spectral coherence values are normalized via Softmax to serve as attention weights, element-wise multiplied with the Value, and then passed through a 2-layer MLP with residual connection
- **Core Idea**: Spectral coherence measures the linear correlation between Q and K across multiple frequencies; higher coherence indicates stronger inter-variable dependency. This captures frequency-domain inter-variable associations more effectively than dot-product attention

### Key Design 3: Koopman Temporal Evolution
- Initializes a learnable complex matrix $\mathbf{S}$; at each forward pass, QR decomposition is applied to retain the unitary matrix $\mathbf{U}$ (ensuring $\mathbf{U}^\dagger \mathbf{U} = \mathbf{I}$)
- A phase vector $\mathbf{p}$ is learned to construct the diagonal matrix $\mathbf{D} = \text{diag}(e^{ip_k})$
- The Koopman operator $\mathbf{K} = \mathbf{U} \mathbf{D} \mathbf{U}^\dagger$ is applied to the complex-valued MVCA output
- **Advantage**: The unitary matrix guarantees no amplification or distortion of the data; a single global projection replaces step-by-step recursion, reducing error accumulation

### Loss & Training
Mean squared error (MSE) is computed over all time steps $\{t+1, \ldots, t+H\}$, optimized with Adam using linear learning rate decay.

## Key Experimental Results

### Main Results

| Task | H | Sonnet MAE | DeformTime MAE | ModernTCN MAE | PatchTST MAE | iTransformer MAE |
|------|---|-----------|----------------|---------------|-------------|-----------------|
| ELEC | 12 | **0.1040** | 0.1162 | 0.1596 | 0.1419 | 0.1468 |
| ELEC | 36 | **0.1389** | 0.1729 | 0.2065 | 0.1659 | 0.1791 |
| ILI-ENG | 7 | **1.4791** | 1.6417 | 1.9489 | 2.3115 | 2.3084 |
| ILI-ENG | 28 | **2.7481** | 2.7228 | 3.3611 | 4.9964 | 4.8125 |
| ILI-US2 | 7 | **0.3806** | 0.4122 | 0.4398 | 0.7097 | 0.6507 |
| WEA-HK | 4 | **0.6389** | 0.6804 | 0.7004 | 1.1488 | 0.8048 |
| WEA-LD | 4 | **1.7231** | 1.8753 | 1.9456 | 2.7602 | 2.1509 |

- Sonnet achieves best performance on 34 out of 47 forecasting tasks, and second-best on 9
- Overall average MAE decreases by 2.2% (vs. DeformTime), statistically significant ($p = 5 \times 10^{-4}$)
- On challenging tasks such as ILI and WEA, MAE is reduced by 3.5% (ILI) and 2.0% (WEA)

### MVCA Substitution Experiments

| Attention | PatchTST ILI-ENG H=7 | PatchTST ILI-US2 H=7 | PatchTST ILI-US9 H=7 |
|--------|----------------------|----------------------|----------------------|
| Naïve (original) | 2.3115 | 0.7097 | 0.4116 |
| ¬ Attn | 2.4723 | 0.7702 | 0.4585 |
| FED | 3.6158 | 0.9292 | 0.5614 |
| VDAB | 2.0799 | 0.5925 | 0.3820 |
| **MVCA** | **2.0054** | **0.5824** | **0.3705** |

- Replacing vanilla attention with MVCA reduces average MAE by **10.7%** across 3 base models on ILI tasks
- PatchTST benefits most (15.1%), as it originally does not model inter-variable dependencies

### Ablation Study

| Ablation Variant | ILI Avg. Δ% | WEA Avg. Δ% |
|---------|------------|------------|
| ¬ MVCA (full module) | +6.3% | +2.4% |
| ¬ Spectral coherence | +5.2% | +1.7% |
| ¬ MLP residual | +4.1% | +1.5% |
| ¬ Joint embedding | +4.0% | +2.3% |
| ¬ Koopman | +3.2% | +1.5% |

### Key Findings
1. MVCA contributes the most to overall performance; as the forecasting horizon increases, the importance of spectral coherence grows substantially (short-horizon +2% → long-horizon +13.6%)
2. Models that preserve temporal ordering (DeformTime, ModernTCN, Crossformer) consistently outperform those that embed along the temporal dimension (iTransformer, Samformer)
3. Models that do not capture inter-variable dependencies (DLinear, PatchTST) fail to surpass the persistence baseline on ILI tasks
4. When data exhibit strong seasonality (e.g., ELEC), setting $\alpha=0$ (using only historical target values) yields better results, suggesting that exogenous variables are not always beneficial

## Highlights & Insights
- **Plug-and-play MVCA**: Can replace vanilla attention in any model, achieving an average improvement of 10.7%
- **Theoretical intuition of spectral coherence attention**: Grounded in signal-processing frequency-domain correlation measures, it is better suited than dot-product attention for characterizing inter-variable dependencies in time series
- **Stability guarantee of the Koopman operator**: QR decomposition maintains the unitary matrix; unitary transformations do not amplify signals, preventing numerical collapse
- **Comprehensive evaluation protocol**: 12 datasets, 47 tasks, multiple test seasons, and diverse evaluation metrics — far exceeding the typical 4–5 dataset standard

## Limitations & Future Work
1. **Insufficient interpretability**: Prediction outcomes cannot be directly attributed to specific input variables
2. **Limited effectiveness** when exogenous variables are few and training spans are short (e.g., ETT datasets with only 6 exogenous variables and 2 years of training)
3. The morphology of learnable wavelet atoms depends on initialization, and there is no theoretically grounded criterion for selecting the optimal number of atoms $K$
4. The effectiveness of the method is validated empirically only, without theoretical convergence or generalization analysis

## Related Work & Insights
- **Frequency-domain methods**: FEDformer (frequency-enhanced decomposition), FiLM (frequency-domain MLP), AdaWaveNet (wavelet-based seasonal decomposition)
- **Inter-variable modeling**: iTransformer, Crossformer (dual-dimension attention), ModernTCN (time-varying convolution), TimeXer (cross-attention), DeformTime (deformable attention)
- **Koopman-based methods**: Koopa (learning the Koopman operator for temporal evolution), Lange et al. (Fourier + Koopman)

## Rating
⭐⭐⭐⭐ — Spectral coherence attention is a novel and convincing contribution; the plug-and-play nature of MVCA is highly practical; the evaluation protocol substantially exceeds comparable work (12 datasets × 47 tasks). Limitations include insufficient interpretability and marginal advantages in low-dimensional exogenous variable settings. The AAAI Oral designation is well deserved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Graph Neural Network Approach for Localized and High-Resolution Temperature Forecasting](../../NeurIPS2025/time_series/a_graph_neural_network_approach_for_localized_and_high-resolution_temperature_fo.md)
- [\[NeurIPS 2025\] Simple and Efficient Heterogeneous Temporal Graph Neural Network](../../NeurIPS2025/time_series/simple_and_efficient_heterogeneous_temporal_graph_neural_network.md)
- [\[ICML 2026\] DAG: A Dual Correlation Network for Time Series Forecasting with Exogenous Variables](../../ICML2026/time_series/dag_a_dual_correlation_network_for_time_series_forecasting_with_exogenous_variab.md)
- [\[ICLR 2026\] Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition](../../ICLR2026/time_series/routing_channel-patch_dependencies_in_time_series_forecasting_with_graph_spectra.md)
- [\[NeurIPS 2025\] Fern: Chaining Spectral Pearls — Ellipsoidal Forecasting Beyond Trajectories for Time Series](../../NeurIPS2025/time_series/friren_beyond_trajectories_--_a_spectral_lens_on_time.md)

</div>

<!-- RELATED:END -->
