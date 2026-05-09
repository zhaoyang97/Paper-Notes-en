---
title: >-
  [Paper Note] The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface
description: >-
  [NeurIPS 2025][Earth science foundation model] This paper proposes Transparent Earth, a Transformer-based multimodal foundation model that fuses 8 heterogeneous geophysical observation modalities via positional encoding and text-derived modality embeddings, enabling zero-shot inference and in-context learning for Earth subsurface property prediction.
tags:
  - NeurIPS 2025
  - Earth science foundation model
  - multimodal fusion
  - Transformer
  - subsurface property reconstruction
  - in-context learning
date: 2026-05-08
content_hash: 0e39bae03c5a4bb8
---

# The Transparent Earth: A Multimodal Foundation Model for the Earth's Subsurface

**Conference**: NeurIPS 2025
**arXiv**: [2509.02783](https://arxiv.org/abs/2509.02783)
**Code**: Not available
**Area**: Information Retrieval
**Keywords**: Earth science foundation model, multimodal fusion, Transformer, subsurface property reconstruction, in-context learning

## TL;DR

This paper proposes Transparent Earth, a Transformer-based multimodal foundation model that fuses 8 heterogeneous geophysical observation modalities via positional encoding and text-derived modality embeddings, enabling zero-shot inference and in-context learning for Earth subsurface property prediction.

## Background & Motivation

Machine learning models in the geosciences have long suffered from a "disciplinary fragmentation" problem:

**Data heterogeneity**: Geophysical observations span diverse modalities (stress angles, strain angles, plate types, fault types, basin types/ages, sediment thickness, mantle temperature), with substantial differences in spatial resolution, sparsity, and data type across modalities.

**Domain specialization**: Existing foundation models are confined to individual sub-disciplines (seismology, climate prediction, oceanography) and cannot leverage cross-domain information.

**Data sparsity**: Direct observational data are scarce and costly in many regions, requiring models to generalize from limited observations.

The key insight is that **these modalities share intrinsic physical relationships** — the type, age, and thickness of sedimentary basins correlate with fault type and location, which in turn relate to seismic events and the present-day stress field. A unified model capable of jointly learning these correlations can produce more accurate predictions in data-sparse regions.

## Method

### Overall Architecture

Transparent Earth adopts an encoder-decoder Transformer architecture. The core pipeline consists of: (1) input processing for each modality's observations (features + positional encoding + modality embeddings); (2) concatenation of all modalities along the sequence dimension; (3) encoding into a shared latent space via cross-attention and self-attention; and (4) a decoder that generates predictions conditioned on query positions and task embeddings.

### Key Designs

1. **Input Processing and Modality Fusion**

   Each observation is transformed into a token embedding encoding both content and positional information. For each modality $\mathcal{M}_i$, $k_i \sim \mathcal{U}(1, k_{\max})$ observation points are randomly sampled; a feature vector $f_i$ and spatial coordinates $c_i$ are extracted, and three components are concatenated:

   $x_i = [f_i \| p_i \| m_i]$

   where $p_i = \text{PosEnc}(c_i)$ is a sinusoidal positional encoding (with depth dimension extension), and $m_i = \text{ModEmbed}(\mathcal{M}_i)$ is a modality embedding generated from modality name descriptions using the multilingual E5 text embedding model.

   Frequency selection for positional encoding follows the Nyquist–Shannon sampling theorem: for a target resolution of 0.5°×0.5°, $f_{\max}^{lat} = 36$ and $f_{\max}^{lon} = 72$:

   $\mathbf{e} = [\sin(\pi\phi \cdot \mathbf{f}_\phi), \cos(\pi\phi \cdot \mathbf{f}_\phi), \sin(\pi\lambda \cdot \mathbf{f}_\lambda), \cos(\pi\lambda \cdot \mathbf{f}_\lambda), z] \in \mathbb{R}^{4F+1}$

   Surface modalities set $z=0$; depth-dependent modalities such as mantle temperature use normalized depth values. The **scalability advantage** of text embeddings is that new modalities can be incorporated by providing a textual description alone, without retraining.

2. **Encoder Design**

   The fused sequence first passes through a cross-attention layer, where learnable latent query vectors serve as queries and input tokens serve as keys/values, compressing information into a fixed-size latent space. This is followed by 3 self-attention + MLP layers that capture higher-order cross-modal interactions. The encoder's core role is to learn implicit physical correlations among modalities.

3. **Decoder Design**

   The query-driven decoder supports prediction at arbitrary locations for arbitrary modalities. Each query point is formed by concatenating positional encoding $p_i$ and task embedding $e_i$: $Q_i = [p_i \| e_i]$. The decoder extracts information from the encoder's latent representations via multi-head cross-attention, followed by a 4-layer MLP to produce predictions.

### Loss & Training

Modality-specific loss functions are employed:
- **Angular quantities (stress/strain angles)**: Periodic angular loss $\mathcal{L}_{angular} = \frac{1}{N}\sum_{i=1}^N\left(\frac{(\hat\theta_i - \theta_i + R/2) \bmod R - R/2}{R/2}\right)^2$
- **Classification tasks (plate types, etc.)**: Cross-entropy loss
- **Regression tasks (sediment thickness, etc.)**: Mean squared error

The total loss is a uniform average over per-modality losses. During training, observation counts and available modalities are randomly sampled, functioning as modality-level dropout to encourage generalization.

## Key Experimental Results

### Main Results — Model Scaling Performance

| Model Scale | Stress Angle MAE↓ | Strain Angle MAE↓ | Plate Cls. Acc↑ | Fault Cls. Acc↑ | Basin Type Acc↑ |
|-------------|-------------------|-------------------|-----------------|-----------------|-----------------|
| 3M (baseline) | ~33° | ~35° | >95% | ~75% | ~88% |
| 30M | Improved | Improved | >95% | ~82% | ~92% |
| 243M | Best | Best | >97% | ~88% | >95% |

### In-Context Learning — Global Inference

| Observation Configuration | Stress Angle MAE | Strain Angle MAE |
|---------------------------|------------------|------------------|
| No input (prior only) | ~33° | ~35° |
| 2 same-modality observations | ~25° | ~28° |
| 8 same-modality observations | ~20° | ~22° |
| 2 × 8 modalities (16 total) | **~9°** | **~13°** |

Multimodal fusion reduces stress angle error by more than a factor of three.

### Key Findings

1. **Scaling laws hold**: From 3M to 243M parameters, MAE decreases and classification accuracy improves consistently across all regression tasks.
2. **Multimodal fusion outperforms unimodal baselines**: A model trained on the strain modality alone struggles to reduce MAE even with many observations, whereas the full multimodal baseline achieves substantial improvement with only a small number of observations.
3. **In-context learning capability**: The model adaptively adjusts prediction accuracy based on the number and modality of input observations, smoothly transitioning from "no-input degradation to a global prior" to "accurate prediction with multiple observations."
4. Plate classification maintains >95% accuracy across all model scales, indicating this task is comparatively simple.

## Highlights & Insights

1. **Unifying multiple disciplines**: This is the first attempt to incorporate 8 geophysical modalities of varying resolutions and types into a single model, breaking disciplinary boundaries.
2. **Text embeddings enable scalability**: Using a pretrained text model to generate modality embeddings allows new modalities to be added with only a textual description, requiring no architectural modification.
3. **Random sampling training strategy**: Randomizing observation counts and available modalities naturally induces robustness to missing data.
4. **Significant cross-modal information gain**: Error reduction from multimodal fusion substantially exceeds that from adding more observations within a single modality.

## Limitations & Future Work

- Only 8 modalities are currently included, falling short of comprehensive subsurface coverage.
- The spatial resolution of certain modalities is coarse (e.g., mantle temperature at 5°×5°), potentially limiting fine-grained prediction.
- Data quality in sparse modalities (e.g., stress angles) directly affects model performance.
- No systematic comparison with classical geostatistical methods (e.g., kriging interpolation) is provided.
- Depth-dependent modalities are represented only by normalized depth values, which may be insufficiently expressive.

## Related Work & Insights

- **Aurora/Pangu**: Climate/weather foundation models, but limited to the atmosphere.
- **Seismic foundation models**: Unimodal models that process seismic waveform data only.
- **Perceiver IO**: The cross-attention design in the encoder draws inspiration from this architecture.

## Rating

- Novelty: ⭐⭐⭐⭐☆ — Multimodal subsurface modeling constitutes a novel problem formulation.
- Experimental Thoroughness: ⭐⭐⭐☆☆ — Scaling experiments and in-context learning results are convincing, but comparisons with traditional methods are absent.
- Writing Quality: ⭐⭐⭐⭐☆ — Space-limited as a workshop paper, but well-structured.
- Value: ⭐⭐⭐⭐☆ — Opens a multimodal direction for geoscience foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)
- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[NeurIPS 2025\] MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)
- [\[NeurIPS 2025\] Windsock is Dancing: Adaptive Multimodal Retrieval-Augmented Generation](windsock_is_dancing_adaptive_multimodal_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](benchmarking_retrievalaugmented_multimodal_generation_for_do.md)

</div>

<!-- RELATED:END -->
