---
title: >-
  [Paper Note] IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction
description: >-
  [ICML 2025][Time Series][Irregular Multivariate Time Series] The VIMTS framework is proposed, which converts irregular multivariate time series (IMTS) into an image-like time × channel patch structure. By leveraging the sparse multi-channel modeling capability of a visual MAE pre-trained on large-scale RGB images, combined with GCN-based cross-channel imputation and a coarse-to-fine prediction strategy, VIMTS achieves SOTA performance and strong few-shot capabilities on IMTS…
tags:
  - "ICML 2025"
  - "Time Series"
  - "Irregular Multivariate Time Series"
  - "Visual Masked Autoencoder"
  - "Self-Supervised Learning"
  - "Graph Convolutional Network"
  - "Coarse-to-Fine Prediction"
date: 2026-05-08
content_hash: ef7381923b1edf39
---

# IMTS is Worth Time × Channel Patches: Visual Masked Autoencoders for Irregular Multivariate Time Series Prediction

**Conference**: ICML 2025  
**arXiv**: [2505.22815](https://arxiv.org/abs/2505.22815)  
**Code**: [https://github.com/WHU-HZY/VIMTS](https://github.com/WHU-HZY/VIMTS)  
**Area**: Time Series Analysis  
**Keywords**: Irregular Multivariate Time Series, Visual Masked Autoencoder, Self-Supervised Learning, Graph Convolutional Network, Coarse-to-Fine Prediction

## TL;DR

The VIMTS framework is proposed, which converts irregular multivariate time series (IMTS) into an image-like time × channel patch structure. By leveraging the sparse multi-channel modeling capability of a visual MAE pre-trained on large-scale RGB images, combined with GCN-based cross-channel imputation and a coarse-to-fine prediction strategy, VIMTS achieves SOTA performance and strong few-shot capabilities on IMTS prediction tasks.

## Background & Motivation

**Problem Definition**: Irregular multivariate time series (IMTS) prediction faces two core challenges—unaligned timestamps across multi-channel signals and the presence of substantial missing values. Such data widely exists in scenarios like finance, healthcare, transportation, and meteorology.

**Limitations of Prior Work**:

- **Statistical Imputation Methods** (e.g., linear interpolation): Require an in-depth understanding of system dynamics and discard information contained in missing points.
- **GCN-based Methods**: Alternately model temporal and channel information, which leads to severe cumulative errors due to data sparsity.
- **Neural-ODE-based Methods**: Pose difficulties in building accurate models from highly sparse single channels, and suffer from high computational overhead.
- **Pre-trained Foundation Models** (e.g., VisionTS): Although performing well on RTS (Regular Time Series), they are only applicable to regularly sampled time series.

**Core Motivation**: Visual MAEs possess a powerful capability to model sparse semantic multi-channel information in image pre-training, whereas IMTS data itself has image-like multi-channel sparseness. VisionTS has demonstrated pattern similarity between visual MAEs and time series. Therefore, this paper proposes reconstructing IMTS into an image-like time × channel structure to fully exploit the pre-trained capabilities of visual MAEs.

## Method

### Overall Architecture

VIMTS consists of three core modules:

1. **Time × Channel Patchify**: Converts unstructured IMTS data into regular patches and mitigates the impact of missing values through cross-channel information completion.
2. **Time-wise Reconstruction**: Utilizes a pre-trained visual MAE to model temporal dependencies among intra-channel patches.
3. **Patch2Point Prediction**: Generates precise point-level predictions from patch-level interval representations through a coarse-to-fine strategy.

### Key Designs

#### 1. Time-wise Dividing and Embedding

Divide the IMTS data along the time axis into $P$ equally spaced time windows (window size $s$), where each window generates patches for all channels. A learnable time embedding is adopted to capture aperiodic and periodic temporal patterns:

$$\phi(t)[d] = \begin{cases} \omega_0 \cdot t + \alpha_0, & \text{if } d=0 \\ \sin(\omega_d \cdot t + \alpha_d), & \text{if } 0 < d < D_{te} \end{cases}$$

where $\omega_d$ and $\alpha_d$ are learnable parameters, with the linear term capturing trends and the sinusoidal term capturing periodic patterns.

#### 2. Transformable Time-aware Convolutional Network (TTCN)

TTCN processes variable-length sequences within each time window using adaptive convolutional filters to generate patches aligned in shape and semantics. The detailed steps are as follows:

- Concatenate the time embedding $\phi(t_i^n)$ with the observed value $x_i^n$ within each channel's time interval.
- Generate adaptive softmax-normalized convolutional kernels via a meta-filter (MLP).
- Perform temporal convolution to obtain a fixed-dimension feature patch $h_p^{n'}$.
- Concatenate a binary mask indicating whether there are observed values in the patch ($m_p^n=1$ indicates observations, $m_p^n=0$ indicates an empty patch).
- Add a channel-specific learnable embedding $e_n$ to distinguish heterogeneous channels.

This design cleverly transforms variable-length irregular inputs into regular feature patches while preserving missingness information.

#### 3. Cross-channel Information Interaction

Since substantial missing values in IMTS lead to insufficient information in single-channel patches, GCN is adopted to model bidirectional channel dependencies:

- **Mixed Graph Embedding**: Two sets of learnable static embedding dictionaries $\mathbf{E}_1^s$ and $\mathbf{E}_2^s$ (representing in/out nodes) are maintained and fused with dynamic patch features through a gating mechanism:
$$\mathbf{E}_{p,k} = \mathbf{E}_k^s + g_{p,k} \odot \mathbf{H}_p \mathbf{W}_k^d$$

- **Adaptive Adjacency Matrix**: Dynamically captures directional dependencies among channels:
$$\mathbf{A}_p = \text{Softmax}(\text{ReLU}(\mathbf{E}_{p,1} \mathbf{E}_{p,2}^\top))$$

- **Graph Convolution + Residual Connection**: Exchange channel information while preserving original representations:
$$\mathbf{H}_p^{gcn} = \text{ReLU}\left(\sum_{m=0}^{M} (\mathbf{A}_p)^m \mathbf{H}_p \mathbf{W}_m^{gcn}\right) + \mathbf{H}_p$$

- Finally, concatenate the original and GCN-enhanced representations: $\mathbf{H}_p^{in} = [\mathbf{H}_p \| \mathbf{H}_p^{gcn}] \in \mathbb{R}^{N \times 2D}$

#### 4. Time-wise Reconstruction

Model the temporal dependency of the patch sequence using a pre-trained visual MAE:

- **Input Projection**: A linear layer compresses $2D$-dimensional features to the MAE encoder dimension $D_e$.
- **Temporal Periodic Position Embedding (TPE)**: Initialized with 2D sine-cosine encoding to treat the patch sequence as a $T \times 1$ patch sequence, leveraging the pre-trained spatial understanding of MAE.
- **Encoding**: The MAE encoder encodes the visible patches.
- **Reconstruction**: Append learnable mask tokens and corresponding TPEs, and reconstruct the target interval patch representations using the MAE decoder.

#### 5. Patch2Point

A two-stage strategy is adopted to generate predictions from the patch level to the point level:

- **Coarse Stage**: The MAE reconstructs target interval patches.
- **Fine Stage**: Given a query timestamp $t_q$, generate query embedding $\phi(t_q)$, locate the corresponding patch, and generate the point-level prediction via a 2-layer MLP:
$$\hat{x}_q = \mathcal{F}(\phi(t_q), \hat{z}_{i_q}^m)$$

This strategy enables flexible and accurate prediction on continuous timestamps while filtering out irrelevant temporal-channel context.

### Loss & Training

A **two-stage training strategy** is adopted:

**Stage 1: Self-Supervised Learning (SSL)**

- Randomly mask a proportion $r$ of the patches, encode only visible patches, and then reconstruct the masked patches using the decoder.
- The loss is the MSE reconstruction error of the time points corresponding to the masked patches:
$$\mathcal{L}_{ssl} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{\mathcal{H}_n} \sum_{h=1}^{\mathcal{H}_n} \|\mathcal{F}(\phi(t_h^n), \hat{z}_{i_h}^{m,n}) - x_h^n\|_2^2$$
- This stage optimizes all parameters to adapt the capabilities of the visual MAE to the IMTS data.

**Stage 2: Supervised Fine-Tuning (Fine-tuning)**

- Encode all historical patches, append mask tokens to reconstruct target future interval patches, and predict future time points.
- The loss is the MSE between the predicted and ground-truth values:
$$\mathcal{L}_{ft} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{\mathcal{Q}_n} \sum_{q=1}^{\mathcal{Q}_n} \|\mathcal{F}(\phi(t_q^n), \hat{z}_{i_q}^{m,n}) - x_q^n\|_2^2$$
- **Selective Freezing**: For USHCN/PhysioNet/Human Activity, freeze GCN and MAE (retaining LayerNorm as trainable); for MIMIC, freeze only MAE (retaining LayerNorm, position embedding, and patch projection layers as trainable).

## Key Experimental Results

### Main Results

VIMTS is compared against 19 baseline methods (including RTS methods, GNN methods, IMTS-specific methods, and pre-trained methods) across 4 real-world datasets.

| Dataset | Metric | VIMTS (100%) | T-PatchGNN (Prev. SOTA) | Gain |
|--------|------|-------------|----------------------|------|
| PhysioNet | MSE (×10⁻³) | **4.81±0.07** | 4.98±0.08 | 3.4% |
| PhysioNet | MAE (×10⁻²) | **3.54±0.04** | 3.72±0.03 | 4.8% |
| Human Activity | MSE (×10⁻³) | **2.65±0.01** | 2.66±0.03 | 0.4% |
| Human Activity | MAE (×10⁻²) | **3.08±0.01** | 3.15±0.02 | 2.2% |
| USHCN | MSE (×10⁻³) | **4.86±0.02** | 5.00±0.04 | 2.8% |
| MIMIC | MSE (×10⁻²) | **1.36±0.02** | 1.36±0.02 | Comparable |
| MIMIC | MAE (×10⁻²) | **6.40±0.17** | 6.56±0.11 | 2.4% |

Few-shot capability: VIMTS using only 20% of the training data already approaches or outperforms T-PatchGNN using 100% of the data.

### Ablation Study

| Configuration | PhysioNet MSE (×10⁻³) | Human Activity MSE (×10⁻³) | MIMIC MSE (×10⁻²) | Description |
|------|----------------------|---------------------------|-------------------|------|
| Complete | **4.81±0.07** | **2.65±0.01** | **1.36±0.02** | Full Model |
| w/o Pre | 5.13±0.04 | 2.73±0.02 | 1.39±0.02 | Remove visual pre-trained weights |
| w/o SSL | 5.46±0.30 | 2.76±0.08 | 1.41±0.03 | Remove self-supervised learning stage |
| w/o Pre & SSL | 5.70±0.42 | 2.84±0.06 | 1.45±0.05 | Remove both pre-training and SSL |
| w/o GCN | 4.94±0.03 | 2.66±0.01 | 2.25±0.02 | Remove cross-channel GCN (largest impact on MIMIC) |
| rp Transformer | 5.57±0.34 | 2.84±0.07 | 1.40±0.04 | Replace MAE with Transformer |

### Key Findings

1. **Both Pre-training and SSL are Indispensable**: Removing pre-training increases PhysioNet MSE from 4.81 to 5.13 (+6.7%), removing SSL increases it to 5.46 (+13.5%), and removing both increases it to 5.70 (+18.5%).
2. **GCN Cross-Channel Imputation is Crucial for Data with High Missing Rates**: For MIMIC (96.7% missing rate), removing the GCN causes the MSE to soar from 1.36 to 2.25 (+65.4%).
3. **MAE Outperforms Standard Transformer**: Replacing MAE with a standard Transformer degrades performance across all datasets, suggesting that sparse data modeling capability brought by visual pre-training is critical.
4. **Effectiveness of the Patch2Point Strategy**: Utilizing Patch2Point in both stages yields the optimal results, verifying the value of the coarse-to-fine prediction strategy.
5. **Strong Few-shot Capability**: On Human Activity, VIMTS with only 10% data (MSE=2.87) significantly outperforms T-PatchGNN (MSE=3.21).

## Highlights & Insights

1. **Bridge Between Vision and Time Series**: A profound insight that the sparse multi-channel characteristic of IMTS is highly similar to masked image patch structures, which allows for the elegant transfer of visual pre-training capabilities to the time-series domain.
2. **2D Reconstruction of time × channel**: The formulation of reconstructing 1D time series into a 2D image-like structure is elegant, enabling the natural reuse of the 2D positional embeddings of the visual MAE.
3. **Sound Two-Stage Training Design**: The SSL stage adapts the model to perform patch reconstruction on IMTS data (domain adaptation), and the FT stage focuses on the prediction task (task adaptation), ensuring a clear division of labor.
4. **Selective Freezing Strategy**: Differentiative freezing of different modules depending on dataset characteristics balances knowledge retention with task adaptation.
5. **Handling Continuous Timestamp Queries**: The Patch2Point mechanism supports predictions at arbitrary time points, without being restricted to fixed steps.

## Limitations & Future Work

1. **Validation on Only Four Datasets**: Although covering clinical, human activity, and climatological domains, high-frequency scenarios like finance are not yet validated.
2. **Insufficient Discussion on Computational Efficiency**: The GCN + MAE pipeline entails a considerable computational footprint, and the efficiency comparison with Neural-ODE methods lacks detail.
3. **Scalability of Number of Channels**: MIMIC has 96 channels and a 96.7% missing rate, where VIMTS only performs comparably to T-PatchGNN, indicating room for improvement in scenarios with high channels and high missing rates.
4. **Fixed Patch Size**: A uniform time window size $s$ is used for all channels, whereas sampling frequencies of different channels can vary greatly; adaptive patch strategies are worth exploring.
5. **Fixed MAE Backbone (MAE-base)**: The performance of larger pre-trained vision models (e.g., MAE-large/huge) has not been explored.

## Related Work & Insights

- **VisionTS** (Chen et al., 2025): Proves that visual MAEs can be adapted to RTS prediction, serving as the direct inspiration for this paper.
- **T-PatchGNN** (Zhang et al., 2024a): The previous IMTS SOTA, combining GNN and patch designs.
- **MAE** (He et al., 2022): A milestone work in visual self-supervised learning, providing powerful sparse reconstruction capabilities.
- **Insights**: The transfer potential of other visual pre-trained models (e.g., DINOv2, SAM) to time-series tasks could be further explored. The paradigm of cross-modal pre-training $\rightarrow$ domain adaptation $\rightarrow$ task fine-tuning is highly worth generalizing to more non-image domains.

## Rating

| Dimension | Score (1-10) | Description |
|------|:-----------:|------|
| Novelty | 8 | The transfer route of visual MAE $\rightarrow$ IMTS is novel, and the time×channel patchify design is highly ingenious. |
| Technical Depth | 8 | The multi-module collaborative design of TTCN + GCN + MAE + Patch2Point is comprehensive. |
| Experimental Thoroughness | 8 | 19 baseline comparisons, various ablation studies, and few-shot experiments, although the number of datasets is relatively small. |
| Writing Quality | 7 | Clearly structured, but math equations are dense and method description is slightly wordy. |
| Value | 7 | Open-source code, but deployment complexity is high (requires visual MAE pre-trained weights + GCN). |
| **Overall Score** | **7.8** | Pioneering work transferring visual pre-training to IMTS, with a complete and effective methodological design. |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters](visionts_visual_masked_autoencoders_are_free-lunch_zero-shot_time_series_forecas.md)
- [\[ICML 2025\] HyperIMTS: Hypergraph Neural Network for Irregular Multivariate Time Series Forecasting](hyperimts_hypergraph_neural_network_for_irregular_multivariate_time_series_forec.md)
- [\[ICML 2025\] Channel Normalization for Time Series Channel Identification](channel_normalization_for_time_series_channel_identification.md)
- [\[NeurIPS 2025\] Rotary Masked Autoencoders are Versatile Learners](../../NeurIPS2025/time_series/rotary_masked_autoencoders_are_versatile_learners.md)
- [\[NeurIPS 2025\] Channel Matters: Estimating Channel Influence for Multivariate Time Series](../../NeurIPS2025/time_series/channel_matters_estimating_channel_influence_for_multivariate_time_series.md)

</div>

<!-- RELATED:END -->
