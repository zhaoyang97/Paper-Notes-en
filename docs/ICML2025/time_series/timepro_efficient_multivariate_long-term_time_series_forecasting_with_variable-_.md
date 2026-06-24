---
title: >-
  [Paper Note] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state
description: >-
  [ICML 2025][Time Series][Time Series Forecasting] This paper proposes the Mamba-based TimePro model. By constructing variable- and time-aware hyper-states, it adaptively selects key time steps to modulate the hidden states of variable dimensions, achieving efficient multivariate long-term time series forecasting with linear complexity.
tags:
  - "ICML 2025"
  - "Time Series"
  - "Time Series Forecasting"
  - "Mamba"
  - "State Space Models"
  - "Multivariate Modeling"
  - "Multi-delay Issue"
date: 2026-05-08
content_hash: 7abab75b7f55343d
---

# TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware Hyper-state

**Conference**: ICML 2025  
**arXiv**: [2505.20774](https://arxiv.org/abs/2505.20774)  
**Code**: [Available](https://github.com/xwmaxwma/TimePro)  
**Area**: Time Series  
**Keywords**: Time Series Forecasting, Mamba, State Space Models, Multivariate Modeling, Multi-delay Issue

## TL;DR

This paper proposes the Mamba-based TimePro model. By constructing variable- and time-aware hyper-states, it adaptively selects key time steps to modulate the hidden states of variable dimensions, achieving efficient multivariate long-term time series forecasting with linear complexity.

## Background & Motivation

**Background**: Long-term time series forecasting (LTSF) is a critical task in machine learning. Currently, mainstream approaches include models based on Transformers (iTransformer, PatchTST), MLPs (DLinear, TimeMixer), and Mamba (S-Mamba, TimeMachine, Bi-Mamba+). Mamba demonstrates significant advantages in time series forecasting due to its linear complexity and efficient capability in capturing long-range dependencies.

**Limitations of Prior Work—Multi-delay Issue**: In multivariate time series, the impacts of different variables on the target variable exhibit distinct time lags. For instance, a temperature change might affect electricity consumption after a few hours, whereas a humidity change might take longer to show an effect. This discrepancy in time lag is termed the multi-delay issue.

**Limitations of Prior Work**:
- **Transformer-based models**: iTransformer focuses on modeling variable correlations but performs a uniform coarse-grained linear projection for different time steps; PatchTST captures global temporal dependencies in a channel-independent manner but treats all variables equally. Both suffer from quadratic complexity bottlenecks.
- **Mamba-based models**: Existing Mamba approaches (S-Mamba, Bi-Mamba+, TimeMachine) achieve linear complexity but only transfer plain states when scanning the variable dimension. They neglect the varying time-lag characteristics among variables and fail to capture complex temporal variations within variables.

**Ours**: This paper proposes TimePro, which innovatively introduces a **time-tune strategy**. During Mamba's variable scanning process, it adaptively selects critical time steps to modulate hidden states, constructing **hyper-states** that simultaneously perceive variable correlations and salient temporal information to effectively address the multi-delay issue while maintaining linear complexity.

## Method

### Overall Architecture

TimePro adopts a Transformer-like encoder-only architecture consisting of the following components:

1. **Reversible Instance Normalization (RevIN)**: Normalizes input sequences to zero mean and unit variance to mitigate distribution shifts between training and testing data. Denormalization is performed after forecasting.
2. **Time-Variable Preserving Embedding**: Splits each univariate time series $\mathbf{X}_{i,:} \in \mathbb{R}^L$ into overlapping patches, preserving the variable dimension to obtain an embedding $\mathcal{E}_0 \in \mathbb{R}^{N \times P \times D}$, where $P$ is the number of patches and $D$ is the feature dimension.
3. **Multi-layer ProBlock**: $\mathcal{E}_{i+1} = \text{ProBlock}(\mathcal{E}_i)$, where each layer consists of HyperMamba (for inter-variable interaction) and TimeFFN (for capturing intra-variable temporal variations).
4. **Linear Projection**: Flattens the embedding of each variable and linearly projects it to obtain forecasting results.

### Key Designs

#### 1. **HyperMamba Module**: Efficient Modeling of Variable Dependencies

HyperMamba is the core component of TimePro, with targeted modifications to the vanilla Mamba:
- **Replace Selective Scan with Hyper-Scan**: Introduces the time-tune strategy to build hyper-states.
- **Remove Depthwise Separable Convolution**: Locality does not exist across variable dimensions, making convolution unnecessary.
- **Remove Post-Scan Linear Projection**: Replaced by TimeFFN to avoid redundancy.
- **Scanning Inception**: Splits along the channel dimension into two halves, scanning in forward ($1 \to N$) and backward ($N \to 1$) variable directions, respectively, to enhance global variable dependency capture.

Specific pipeline: The input $\mathcal{E} \in \mathbb{R}^{N \times (P \times D)}$ is projected by two linear layers to obtain $\mathcal{E}_t$ and $\mathcal{E}_z$. After bidirectional Hyper-Scan on $\mathcal{E}_t$, the concatenated enhanced embedding $\hat{\mathcal{E}}_t$ is generated, and the final output is obtained via a gating mechanism:

$$\hat{\mathcal{E}} = \hat{\mathcal{E}}_t \cdot \text{SiLU}(\mathcal{E}_z)$$

#### 2. **Hyper-Scan: Core Implementation of the Time-Tune Strategy**

This is the most critical innovation of this paper. When vanilla Mamba scans along the variable dimension, the hidden states only contain variable information, making them unaware of temporal variations within the variables. Hyper-Scan constructs time-variable aware hyper-states through the following steps:

**Step 1 — Obtain Initial States**: Scan the embedding $\mathcal{E}_t \in \mathbb{R}^{N \times (P \times D)}$ along the variable dimension to obtain initial hidden states $h \in \mathbb{R}^{N \times (P \times D)}$ (completed inside the GPU SRAM to reduce HBM read/write operations).

**Step 2 — Generate Offsets**: Reshape the hidden states to restore fine-grained temporal dimensions, and generate initial offsets $\delta_h = \text{Conv}(h)$ via convolution.

**Step 3 — Adaptive Sampling**: Add reference points to learnable offsets to get sampling point coordinates, and extract key time steps from states via differentiable bilinear interpolation $\psi$:

$$h_{samp} = h_{ref} + \delta_h$$
$$\hat{h} = h_{ref} + \psi(h; h_{samp})$$

where $\hat{h} \in \mathbb{R}^{N \times P \times D \times M}$, and $M$ is the number of sampling time steps (default of 9).

**Step 4 — Fusion for Hyper-States**: Fuse the sampled time steps via a linear mapping to get the hyper-state $h_o = \text{Linear}(\hat{h})$, which is then multiplied by the parameter matrix $\mathbf{C}$ to obtain the output.

The core idea of this design borrows from Deformable Convolution, allowing the model to adaptively focus on the most important time steps for each variable rather than processing all time steps uniformly.

#### 3. **Hardware-Aware Implementation**

Fully exploit GPU memory hierarchy:
- Initial state acquisition is completed on GPU **SRAM** (following the original Mamba implementation) to avoid frequent HBM read/write.
- The remaining operations (reshape, offset generation, sampling, linear mapping) are executed on **HBM**.
- Maintains computational efficiency comparable to vanilla Mamba.

### Loss & Training

- Loss function: Standard MSE loss
- Optimizer: Adam
- Training setups follow the general configurations of iTransformer.
- Lookback window $L = 96$, prediction length $H \in \{96, 192, 336, 720\}$.
- Hardware: 4 Tesla V100 GPUs.

### Complexity Analysis

The overall complexity of HyperMamba is $O(NL)$ (linear complexity), where $N$ is the number of variables and $L$ is the sequence length. This is because:
- Two linear projections: $O(NP^2D^2)$
- SSM parameter matrix calculation: $O(NPD)$ (state dimension set to 1)
- Variable dimension scanning: $O(NPD)$
- Time-tuning (convolution + linear mapping): $O(NPMD)$, where $D$ and $M$ are constants and can be neglected

| Model | Complexity |
|------|--------|
| **TimePro** | $O(NL)$ |
| iTransformer | $O(N^2 + NL)$ |
| PatchTST | $O(NL^2)$ |
| Transformer | $O(NL + L^2)$ |

## Key Experimental Results

### Main Results

Evaluated on 8 real-world datasets (ETTh1/h2, ETTm1/m2, ECL, Exchange, Weather, Solar-Energy) with a lookback window $L=96$ and prediction length $H \in \{96, 192, 336, 720\}$:

| Dataset | Metric | TimePro | iTransformer | SOFTS | S-Mamba | PatchTST |
|--------|------|---------|-------------|-------|---------|----------|
| ECL | MSE | **0.169** | 0.178 | 0.174 | 0.170 | 0.189 |
| Weather | MSE | **0.251** | 0.258 | 0.255 | 0.251 | 0.256 |
| ETTh1 | MSE | **0.438** | 0.454 | 0.449 | 0.455 | 0.453 |
| ETTm1 | MSE | **0.391** | 0.407 | 0.393 | 0.398 | 0.396 |
| ETTm2 | MSE | **0.281** | 0.288 | 0.287 | 0.288 | 0.287 |
| Exchange | MSE | **0.352** | 0.360 | 0.361 | 0.367 | 0.367 |
| ETTh2 | MSE | 0.377 | 0.383 | **0.373** | 0.381 | 0.385 |
| Solar | MSE | 0.232 | 0.233 | **0.229** | 0.240 | 0.236 |

Achieved **12 first places and 2 second places** out of 16 settings.

**Efficiency Comparison** (ECL dataset, $L=96, H=720$, V100):
- Fewest parameters and FLOPs, with only 67% parameters and 78% GFLOPs of S-Mamba.
- Inference speed is 2.7x faster than PatchTST and 14.4x faster than TimesNet.
- Training/inference time is comparable to S-Mamba.

### Ablation Study

**Ablation of Time-Tune Strategy** (average over Exchange / ETTh1 datasets):

| Configuration | Exchange MSE | ETTh1 MSE | Description |
|------|-------------|-----------|------|
| Non-Adaptive (Linear Projection) | 0.360 | 0.451 | Uniformly processes all time steps |
| **Adaptive (TimePro)** | **0.352** | **0.438** | Adaptively selects key time steps |

**Ablation of HyperMamba Architecture**:

| Configuration | Exchange MSE | ETTh1 MSE | Description |
|------|-------------|-----------|------|
| Mamba + Hyper-Scan | 0.358 | 0.449 | Keeps all components of vanilla Mamba |
| − DWConv | 0.358 | 0.447 | Removes depthwise separable convolution |
| − Linear | 0.356 | 0.447 | Further removes linear projection |
| **HyperMamba** | **0.352** | **0.438** | Complete design |

### Key Findings

1. **Visual Verification of Multi-Delay Issue**: Through Pearson correlation coefficient visualization, the variable correlation matrix processed by HyperMamba is significantly closer to the correlation matrix of the ground truth sequence, verifying that the time-tune strategy effectively mitigates the multi-delay issue.
2. **Hyperparameter Sensitivity**: The feature dimension $D=48$ achieves the optimal performance on most datasets; 2-4 encoder layers are preferred; a patch length of 16-32 is a reasonable starting point.
3. **Lookback Window Robustness**: TimePro consistently outperforms baselines under varying lookback window lengths (48/96/192/336). Particularly when the lookback window is short (48), TimePro shows a significant advantage over SOFTS, demonstrating its competence in effectively capturing key temporal information from limited data.

## Highlights & Insights

1. **Precise Problem Definition**: The multi-delay issue is a genuine challenge in multivariate time series forecasting. Through clear formal definitions and visual validation, the paper demonstrates the importance of this problem and the effectiveness of the solution.
2. **Ingenious Technical Adaptation**: Adapting the adaptive sampling idea from deformable convolutions into the hidden state space of SSMs—using learnable offsets for the adaptive selection of key time steps—is a highly inspiring cross-domain technical transfer.
3. **Subtractive Design Philosophy**: Removing both depthwise separable convolution and post-scan linear projection from Mamba improved performance rather than degrading it, reflecting a deep understanding of the model's architecture.
4. **Excellent Balance of Efficiency and Performance**: While achieving state-of-the-art performance, it maintains the fewest parameters and FLOPs alongside a linear complexity, making it well-suited for resource-constrained deployments.

## Limitations & Future Work

1. **Limited Dataset Scale**: Evaluated only on 8 relatively conventional time series forecasting benchmarks, lacking validation in large-scale, high-dimensional, or more complex real-world scenarios.
2. **Fixed Number of Sampling Points $M=9$**: Currently uses a fixed number of sampling points for all datasets, without exploring the possibility of adaptive adjustment of $M$.
3. **Focus on Forecasting Only**: The generalization capability of TimePro in other time series analysis tasks like classification, anomaly detection, and imputation remains unexplored.
4. **Single Forecasting Loss**: Uses only MSE as the training objective without exploring multi-task learning or auxiliary losses to improve performance.
5. **Variable Relationship Assumption**: Implicitly assumes that time-lag relationships between variables can be learned in a data-driven manner. However, introducing domain-specific prior knowledge (e.g., physical constraints) might be more effective in certain areas.

## Related Work & Insights

- **iTransformer** (Liu et al., 2024b): The inverted design treating variables as tokens and time steps as features inspired TimePro's priority in modeling the variable dimension.
- **SOFTS** (Han et al., 2024): The Series-core fusion method, which is the primary competitor of TimePro across multiple datasets.
- **S-Mamba** (Wang et al., 2025): The first work applying Mamba to multivariate time series forecasting, though using sequence embedding limits fine-grained temporal modeling.
- **Deformable ConvNets v4** (Xiong et al., 2024): The adaptive sampling design in deformable convolutions directly inspired the time-tune strategy of TimePro.
- **RevIN** (Kim et al., 2022): Reversible instance normalization has become a standard component in time series forecasting.

## Rating

- Novelty: ⭐⭐⭐⭐ The time-tune strategy is novel. Incorporating deformable sampling into SSM hidden states is a valuable innovation, but the overall framework is still a Mamba variant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quite comprehensive, with 8 datasets, multiple ablation studies, efficiency comparisons, and visual analyses, though detailed analysis over different prediction lengths is lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, systematic method description, and good illustrations, though some equation formatting and notation consistency could be improved.
- Value: ⭐⭐⭐⭐ Achieves an excellent balance between efficiency and performance. Its linear complexity and state-of-the-art results are highly valuable for real-world deployment, posing it as a promising backbone for time series foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] A Generalizable Physics-Enhanced State Space Model for Long-Term Dynamics Forecasting in Complex Environments](a_generalizable_physics-enhanced_state_space_model_for_long-term_dynamics_foreca.md)
- [\[ICML 2025\] TQNet: Temporal Query Network for Efficient Multivariate Time Series Forecasting](temporal_query_network_for_efficient_multivariate_time_series_forecasting.md)
- [\[AAAI 2026\] CometNet: Contextual Motif-guided Long-term Time Series Forecasting](../../AAAI2026/time_series/cometnet_contextual_motif-guided_long-term_time_series_forecasting.md)
- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [\[ICML 2025\] WAVE: Weighted Autoregressive Varying Gate for Time Series Forecasting](wave_weighted_autoregressive_varying_gate_for_time_series_forecasting.md)

</div>

<!-- RELATED:END -->
