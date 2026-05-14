---
title: >-
  [Paper Note] Rotary Masked Autoencoders are Versatile Learners
description: >-
  [NeurIPS 2025][Time Series][Masked Autoencoder] This paper proposes RoMAE, which extends Rotary Position Embedding (RoPE) to continuous positions and integrates it with Masked Autoencoders (MAE). Without any time-series-…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "Masked Autoencoder"
  - "RoPE"
  - "Irregular Time Series"
  - "Multimodality"
  - "Self-Supervised Pretraining"
date: 2026-05-08
content_hash: e6a5c358a87ee65c
---

# Rotary Masked Autoencoders are Versatile Learners

**Conference**: NeurIPS 2025
**arXiv**: [2505.20535](https://arxiv.org/abs/2505.20535)
**Code**: [GitHub](https://chromeilion.github.io/RoMAE-Website/)
**Area**: Time Series
**Keywords**: Masked Autoencoder, RoPE, Irregular Time Series, Multimodality, Self-Supervised Pretraining

## TL;DR

This paper proposes RoMAE, which extends Rotary Position Embedding (RoPE) to continuous positions and integrates it with Masked Autoencoders (MAE). Without any time-series-specific architectural modifications, RoMAE matches or surpasses specialized models across diverse modalities including irregular time series, images, and audio.

## Background & Motivation

Transformers have achieved remarkable success in vision and NLP, yet applying them to irregularly sampled time series faces a fundamental limitation: standard Transformers support only discrete integer position encodings and cannot handle continuous timestamps from non-uniformly sampled data.

Existing solutions fall into two categories:

**Modifying internal Transformer architecture**: e.g., altering feedforward layers or using Neural ODEs as position encodings, which increases computational overhead and method complexity.

**Using State Space Models**: e.g., Mamba and S5 natively support multiple modalities but depart from the Transformer ecosystem.

**Core Insight**: Although RoPE was originally designed for discrete text positions, the position $m$ in its rotation matrix formulation $R(\theta, m)$ can naturally take arbitrary real values. Exploiting this property enables handling of continuous positional information without modifying any Transformer architecture, thereby inheriting all optimizations and advances within the Transformer/MAE ecosystem.

## Method

### Overall Architecture

RoMAE follows the asymmetric encoder–decoder structure of MAE (large encoder + small decoder) and introduces three innovations: (1) N-dimensional patchification for arbitrary modality inputs; (2) continuous axial RoPE for positional encoding; and (3) a p-RoPE truncation strategy for improved robustness.

### Key Designs

1. **Continuous Axial RoPE**: Extends standard RoPE from discrete integer positions to continuous real-valued positions. For $D$-dimensional inputs, axial RoPE partitions the embedding space into $D$ subspaces, each encoding the continuous position along one dimension. The RoPE rotation formula is:

    $\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} x_m^{(i)}$

   where $m \in \mathbb{R}$ can be an arbitrary real number (e.g., a timestamp) and $\theta_i = 10000^{-2(i-1)/d_x}$. The paper adopts $p$-RoPE ($p=0.75$), retaining only the 75% smallest $\theta_i$ values and reserving a portion of the embedding space as unmodified data channels, enhancing robustness to variable-length sequences.

2. **N-Dimensional Patchification**: Defines patch sizes $(p_1, \ldots, p_D)$ and partitions inputs into non-overlapping patches along each dimension. A key constraint is that for any irregular dimension $d_i$, the corresponding patch size $p_i$ must equal 1, since the number of points within a patch is not fixed under irregular sampling. All patches are flattened into a single sequence, enabling the model to jointly model across all dimensions.

3. **[CLS] Token and Absolute Position Recovery**: Since RoPE is a relative position encoding, the model is inherently translation-invariant. The authors theoretically prove that when a learnable [CLS] token is included, the model can recover absolute positional information (with [CLS] serving as an anchor); without it, only relative positions are available, making pretraining harder but potentially beneficial for translation-invariant tasks.

### Loss & Training

- **Pretraining**: Masked autoencoding objective with uniform random masking of 75% of patches; the decoder predicts original values of masked patches.
- **Image Pretraining**: Loss is computed on normalized patch values (following MAE).
- **Architectural Details**: SiLU activations and RMSNorm are used (following LLaMA), which are more efficient than standard LayerNorm.
- **Fine-tuning**: The decoder is removed and a task-specific head is attached to the encoder output.

## Key Experimental Results

### Main Results (Irregular Time Series Classification — ELAsTiCC)

| Method | F-score | Notes |
|--------|---------|-------|
| Transformer | 0.526 | Standard architecture |
| ATAT (specialized) | 0.627 | Designed specifically for ELAsTiCC |
| RoMAE-tiny-shallow | 0.711 | Comparable parameters to ATAT |
| **RoMAE-tiny** | **0.803** | +0.18 advantage |

### Ablation Study (Multimodal Performance Summary)

| Task / Dataset | RoMAE | Prev. SOTA | Prev. SOTA Name | Notes |
|----------------|-------|-----------|-----------------|-------|
| Tiny ImageNet classification | 0.500 (no CLS) | 0.479 (abs. PE) | MAE | RoPE ≥ absolute position |
| ESC-50 audio (AudioSet-20k) | **84.7%** | 82.2% | SSAST | Outperforms SSAST under same conditions |
| Pendulum regression MSE×10⁻³ | **3.32** | 3.41 (S5), 4.63 (ContiFormer) | S5/ContiFormer | Surpasses without pretraining |
| PhysioNet interpolation MSE | **0.467** | 0.562 | HeTVAE | More balanced across sparse channels |
| Spirals interpolation RMSE | **0.018** | 0.49 | ContiFormer | Order-of-magnitude improvement |

### Key Findings

1. **Cross-modal generality**: A single RoMAE architecture achieves competitive or state-of-the-art performance on images (ImageNet), audio (ESC-50), irregular time series (ELAsTiCC), and interpolation tasks.
2. **No architectural specialization required**: No time-series-specific modifications are needed; standard Transformer components suffice.
3. **[CLS] token breaks translation invariance**: Experiments confirm that with [CLS], position reconstruction MSE drops to 0.003; without [CLS], it rises to 200.33 (completely unrecoverable).
4. **MAE pretraining is particularly effective for irregular time series**: Achieves an 18% improvement over ATAT without pretraining on ELAsTiCC.
5. **Data efficiency**: Strong performance is maintained even on UEA datasets with only hundreds of samples.

## Highlights & Insights

- **Extreme simplicity**: No new architecture is invented; only continuous-position RoPE and the standard MAE framework are combined, yet strong cross-modal capability is achieved.
- **Deep theoretical insight**: The theoretical proof and experimental validation that [CLS] tokens recover absolute position information are particularly illuminating.
- **High practical value**: Demonstrates that standard tools within the Transformer ecosystem are sufficient to handle irregular time series, without switching to new paradigms such as SSMs.

## Limitations & Future Work

1. Continuous-position RoPE incurs additional computational overhead when positions change at every forward pass.
2. The $O(n^2)$ memory complexity of standard attention limits the processing of long sequences.
3. Extrapolation capability is limited.
4. Token count grows exponentially with dimensionality under N-dimensional patchification; current experiments only use up to 3 dimensions.

## Related Work & Insights

This paper cleverly combines RoPE (RoFormer) and MAE (He et al.), two independently successful directions, demonstrating that "ingenious combination of standard tools" can sometimes outperform "entirely novel architecture design." Future work integrating RoPE with linear attention could overcome the long-sequence bottleneck.

## Rating

- **Overall**: Achieves maximum modality coverage with minimal architectural modifications — a textbook example of the "less is more" philosophy.
- **Novelty**: ⭐⭐⭐⭐ The continuous extension of RoPE is simple yet insightful, supported by rigorous theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 5 tasks/modalities with extensive baseline comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theory and experiments are well-integrated with clear exposition.
- **Value**: ⭐⭐⭐⭐⭐ Provides a general and elegant solution for learning from irregular time series.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dissecting Chronos: Sparse Autoencoders Reveal Causal Feature Hierarchies in Time Series Foundation Models](../../ICLR2026/time_series/dissecting_chronos_sparse_autoencoders_reveal_causal_feature_hierarchies_in_time.md)
- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[NeurIPS 2025\] Time-O1: Time-Series Forecasting Needs Transformed Label Alignment](time-o1_time-series_forecasting_needs_transformed_label_alignment.md)
- [\[NeurIPS 2025\] Less is More: Unlocking Specialization of Time Series Foundation Models via Structured Pruning](less_is_more_unlocking_specialization_of_time_series_foundation_models_via_struc.md)

</div>

<!-- RELATED:END -->
