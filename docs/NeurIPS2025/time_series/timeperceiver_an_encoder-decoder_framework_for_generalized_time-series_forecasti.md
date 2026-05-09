---
title: >-
  [Paper Note] TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting
description: >-
  [NeurIPS 2025][Time Series][Time series forecasting] TimePerceiver proposes a unified encoder-decoder framework that generalizes the forecasting task (encompassing extrapolation, interpolation, and imputation) and employs a latent bottleneck encoder with a query-based decoder, achieving comprehensive state-of-the-art performance across 8 standard benchmarks.
tags:
  - NeurIPS 2025
  - Time Series
  - Time series forecasting
  - encoder-decoder
  - latent bottleneck
  - generalized forecasting formulation
  - cross-attention
date: 2026-05-08
content_hash: d3ecafc55c3efcef
---

# TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting

**Conference**: NeurIPS 2025
**arXiv**: [2512.22550](https://arxiv.org/abs/2512.22550)
**Code**: [GitHub](https://github.com/efficient-learning-lab/TimePerceiver)
**Area**: Time Series
**Keywords**: Time series forecasting, encoder-decoder, latent bottleneck, generalized forecasting formulation, cross-attention

## TL;DR

TimePerceiver proposes a unified encoder-decoder framework that generalizes the forecasting task (encompassing extrapolation, interpolation, and imputation) and employs a latent bottleneck encoder with a query-based decoder, achieving comprehensive state-of-the-art performance across 8 standard benchmarks.

## Background & Motivation

Recent years have witnessed a proliferation of novel architectures for time series forecasting (Transformers, CNNs, MLPs, SSMs); however, these works **overemphasize encoder design** while neglecting two equally important aspects:

**Coarse decoding strategies**: Most methods directly apply linear projections from encoded representations to future values, making it difficult to capture complex temporal structures.

**Misalignment between training strategy and architecture**: BERT-inspired two-stage training (mask-and-reconstruct pretraining → forecasting fine-tuning) suffers from objective misalignment—the pretraining objective is reconstruction, whereas the ultimate goal is prediction.

Furthermore, channel-independent models (e.g., PatchTST) are simple and robust but ignore cross-channel interactions, while channel-dependent models (e.g., iTransformer, CARD) model such interactions at the cost of high computational overhead and inconsistent performance.

**Core Innovation**: The standard forecasting task (predicting future contiguous values from past contiguous observations) is **generalized** to predict at arbitrary positions along the time axis (extrapolation + interpolation + imputation), naturally aligning the training objective with the forecasting goal and eliminating the need for two-stage training.

## Method

### Overall Architecture

TimePerceiver consists of three components: (1) patch-based embedding construction; (2) an encoder with a latent bottleneck that jointly models temporal and cross-channel dependencies; and (3) a query-based decoder that selectively retrieves relevant information conditioned on target timestamps.

### Key Designs

1. **Generalized Forecasting Formulation**: The standard forecasting objective $f_\theta(\mathbf{X}_{\text{past}}) \to \mathbf{X}_{\text{future}}$ is generalized to predict over arbitrary subsets of time indices. Given input indices $\mathcal{I}$ and target indices $\mathcal{J} = \{1,...,T\} \setminus \mathcal{I}$:

    $\hat{\mathbf{X}}_{\mathcal{J}} = g_\theta(\mathbf{X}_{\mathcal{I}}, \mathcal{I}, \mathcal{J})$

   During training, input–target splits are sampled randomly, covering extrapolation (future prediction), interpolation (missing intermediate values), and imputation (missing past values). Standard forecasting is a special case of this formulation. This enables the model to learn deep temporal dynamics within a single end-to-end training stage.

2. **Latent Bottleneck Encoder**: $M$ learnable latent tokens $\mathbf{Z}^{(0)} \in \mathbb{R}^{M \times D}$ (with $M \ll C|\mathcal{I}_{\text{patch}}|$) are introduced, encoding the input through a three-step bottleneck process:

    - **Compression**: Latent tokens aggregate contextual information from input tokens via cross-attention:
      $\mathbf{Z}^{(1)} = \text{AttnBlock}(\mathbf{Z}^{(0)}, \mathbf{H}^{(0)})$
    - **Refinement**: $K$ self-attention layers allow interactions within the latent space:
      $\mathbf{Z}^{(k+1)} = \text{AttnBlock}(\mathbf{Z}^{(k)}, \mathbf{Z}^{(k)})$
    - **Back-projection**: Updated latent tokens enhance the input tokens in return:
      $\mathbf{H}^{(1)} = \text{AttnBlock}(\mathbf{H}^{(0)}, \mathbf{Z}^{(K+1)})$

   Complexity is reduced from $\mathcal{O}(N^2)$ for full attention to $\mathcal{O}(NM)$, while selectively retaining key temporal and cross-channel patterns.

3. **Query-Based Decoder**: Queries $\mathbf{Q}^{(0)}$ are constructed from positional embeddings (temporal + channel positions) corresponding to target patches, and relevant information is retrieved from encoder outputs via cross-attention:

    $\mathbf{Q}^{(1)} = \text{AttnBlock}(\mathbf{Q}^{(0)}, \mathbf{H}^{(1)})$

   Predictions are generated via a linear projection $\hat{\mathbf{X}}_{\mathcal{P}_j, c} = \mathbf{Q}^{(1)}_{c,j} \mathbf{W}_{\text{output}}$. This design naturally accommodates the generalized forecasting formulation—regardless of the target position, the decoder retrieves the appropriate context through positional queries.

### Loss & Training

End-to-end training with MSE loss:

$$\mathcal{L} = \frac{1}{|\mathcal{J}|C} \sum_{j \in \mathcal{J}} \|\hat{\mathbf{x}}_j - \mathbf{x}_j\|_2^2$$

Input–target index splits are randomly sampled during training, requiring no pretraining stage. Input lengths are varied from $\{96, 384, 768\}$ to enhance generalization.

## Key Experimental Results

### Main Results (8 datasets, averaged MSE, averaged over $L \in \{96, 384, 768\}$)

| Dataset | TimePerceiver | DeformableTST | CARD | PatchTST | iTransformer | Gain (vs. 2nd best) |
|---------|--------------|--------------|------|----------|-------------|-------------------|
| Weather | **0.227** | 0.233 | 0.247 | 0.236 | 0.244 | -2.6% |
| Solar | **0.198** | 0.199 | 0.228 | 0.234 | 0.214 | -0.5% |
| Electricity | **0.161** | 0.169 | 0.174 | 0.177 | 0.175 | -4.7% |
| Traffic | **0.407** | 0.410 | 0.426 | 0.430 | 0.424 | -0.7% |
| ETTh1 | **0.410** | 0.413 | 0.430 | 0.438 | 0.461 | -0.7% |
| ETTh2 | **0.344** | 0.336 | 0.355 | 0.356 | 0.390 | — |
| ETTm1 | **0.347** | 0.358 | 0.368 | 0.365 | 0.386 | -3.1% |
| ETTm2 | **0.261** | 0.267 | 0.268 | 0.273 | 0.281 | -2.2% |
| **Rank** | **1.375** | 2.525 | 4.975 | 5.450 | 6.475 | — |

***55 best and 17 second-best results out of 80 metrics.***

### Ablation Study

| Formulation / Encoder / PE Strategy | ETTh1 MSE | ETTm1 MSE | Solar MSE | ECL MSE |
|-------------------------------------|----------|----------|----------|---------|
| Standard formulation + Latent bottleneck | 0.420 | 0.355 | 0.194 | 0.169 |
| **Generalized formulation + Latent bottleneck** | **0.404** | **0.338** | **0.182** | **0.157** |
| Generalized formulation + Full self-attention | 0.425 | 0.353 | 0.192 | 0.161 |
| Generalized formulation + Decoupled self-attention | 0.423 | 0.356 | 0.189 | 0.158 |
| Generalized formulation + Non-shared PE | 0.423 | 0.342 | 0.193 | 0.163 |

### Key Findings

1. **The generalized formulation consistently outperforms the standard formulation**: Average MSE improves by 5.0% and MAE by 3.4%, indicating that exposure to more diverse temporal reasoning tasks improves generalization.
2. **The latent bottleneck outperforms full attention**: The bottleneck is not only computationally efficient but also forces the model to learn more essential patterns through information compression, acting as a form of regularization.
3. **The generalized formulation is broadly applicable**: Applying it to a PatchTST encoder combined with a query-based decoder also yields improvements (ETTh1 MSE: 0.423 → 0.415).
4. **Channel-shared PE outperforms non-shared PE**: Shared positional encodings enable the model to better leverage positional information across channels.

## Highlights & Insights

- **Perspective shift**: Rather than solely pursuing better encoders, this work systematically rethinks the forecasting problem from the perspective of training objectives and decoder design.
- **Elegance of the generalized formulation**: By randomly sampling input–target splits, pretraining and forecasting training are unified into a single process, eliminating the complexity of two-stage training.
- **Dual role of the bottleneck mechanism**: It simultaneously reduces computational cost and acts as a regularizer, improving generalization.

## Limitations & Future Work

1. The random sampling strategy in the generalized formulation may require more training epochs to converge.
2. The query-based decoder introduces additional cross-attention computation, making it slower than pure linear projection.
3. Evaluation is currently limited to fixed patch size settings; adaptive patch strategies remain unexplored.
4. When the number of channels is very large (e.g., 862 channels in Traffic), the choice of bottleneck size has a notable impact on performance.

## Related Work & Insights

TimePerceiver's name and architectural inspiration derive from the Perceiver series (DeepMind), adapting the latent bottleneck concept to the temporal domain. The generalized forecasting formulation can be viewed as a unification of BERT-style pretraining and forecasting, offering a new training paradigm for time series foundation models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The unified framework combining generalized forecasting formulation, bottleneck encoder, and query-based decoder is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 8 datasets, multiple input lengths, extensive ablations, and comprehensive comparison against 9 baselines.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is well articulated, formulations are clear, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new state of the art in the highly competitive time series forecasting landscape, with ideas that are broadly transferable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Multi-Scale Finetuning for Encoder-based Time Series Foundation Models](multi-scale_finetuning_for_encoder-based_time_series_foundation_models.md)
- [\[NeurIPS 2025\] IonCast: A Deep Learning Framework for Forecasting Ionospheric Dynamics](ioncast_a_deep_learning_framework_for_forecasting_ionospheric_dynamics.md)
- [\[NeurIPS 2025\] Selective Learning for Deep Time Series Forecasting](selective_learning_for_deep_time_series_forecasting.md)
- [\[NeurIPS 2025\] AERO: A Redirection-Based Optimization Framework Inspired by Judo for Robust Probabilistic Forecasting](aero_a_redirection-based_optimization_framework_inspired_by_judo_for_robust_prob.md)
- [\[NeurIPS 2025\] Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)

<!-- RELATED:END -->
