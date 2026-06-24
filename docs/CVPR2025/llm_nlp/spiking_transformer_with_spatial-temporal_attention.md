---
title: >-
  [Paper Note] Spiking Transformer with Spatial-Temporal Attention
description: >-
  [CVPR 2025][LLM (Other)][Spiking Transformer] Spatially-temporally decoupled attention is integrated into the Spiking Transformer architecture. By combining spatial-temporal decoupled attention designs with a spike-driven self-attention mechanism, the approach bridges the performance gap with ANNs while preserving the energy efficiency advantages of SNNs, achieving SOTA performance on multiple vision benchmarks.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Spiking Transformer"
  - "Spatial-Temporal Attention"
  - "SNN"
  - "Energy-Efficient Inference"
  - "Surrogate Gradient"
date: 2026-05-08
content_hash: d64484ed9de6f47c
---

# Spiking Transformer with Spatial-Temporal Attention

**Conference**: CVPR 2025  
**arXiv**: [2409.19764](https://arxiv.org/abs/2409.19764)  
**Code**: None  
**Area**: Spiking Neural Networks / Efficient Inference  
**Keywords**: Spiking Transformer, Spatial-Temporal Attention, SNN, Energy-Efficient Inference, Surrogate Gradient

## TL;DR
Spatially-temporally decoupled attention is integrated into the Spiking Transformer architecture. By combining spatial-temporal decoupled attention designs with a spike-driven self-attention mechanism, the approach bridges the performance gap with ANNs while preserving the energy efficiency advantages of SNNs, achieving SOTA performance on multiple vision benchmarks.

## Background & Motivation

### Background

**Background**: SNNs have attracted significant attention due to their low power consumption and biological interpretability, but a considerable accuracy gap remains compared to ANNs. Recently, Spikformer and Spike-driven Transformers have introduced attention mechanisms into SNNs, achieving notable progress.

**Key Challenge**: The softmax operation and floating-point multiplications in standard self-attention are incompatible with the binary spiking nature of SNNs. Direct transplanting leads to a loss of energy efficiency, while simplified attention mechanisms suffer from accuracy degradation.

**Mechanism**: The attention mechanism is decoupled into spatial attention (capturing relationships between patches) and temporal attention (capturing dynamics across timesteps), both implemented using spike-compatible operations.

### Proposed Solution

**Goal**: ### Overall Architecture
Image → Spike Encoding → Multi-layer Spiking Transformer (Alternating Spatial Attention + Temporal Attention) → Classification Output.


## Method

### Overall Architecture
Image → Spike Encoding → Multi-layer Spiking Transformer (Alternating Spatial Attention + Temporal Attention) → Classification Output.

### Key Designs

1. **Spike Spatial Attention**: Uses a linear attention approximation that replaces multiplication with addition. Q, K, and V are generated using spiking convolutions, avoiding softmax.
2. **Spike Temporal Attention**: Models the interaction between spike tokens across timesteps to capture temporal evolution patterns.
3. **Adaptive Membrane Potential**: Learnable LIF parameters across different layers are introduced to simulate the heterogeneity of biological neurons.
4. **Surrogate Gradient Training**: Approximates the gradient of the step function using a rectangular window function.

### Loss & Training
Cross-entropy loss + spike sparsity regularization, optimized by SGD, with 4 timesteps.

## Key Experimental Results

### Main Results

| Dataset | Architecture | T | Ours | Prev. SOTA | Gain |
|--------|------|---|------|--------|------|
| CIFAR-10 | ResNet-19 | 4 | **96.8%** | 96.5% | +0.3% |
| CIFAR-100 | ResNet-19 | 4 | **81.5%** | 80.1% | +1.4% |
| ImageNet | ResNet-34 | 4 | **69.8%** | 67.7% | +2.1% |

### Ablation Study

| Configuration | CIFAR-100 acc | Description |
|------|-------------|------|
| Baseline SNN | 78.2% | Without attention |
| + Spatial Attention | 79.5% | +1.3% |
| + Temporal Attention | 80.3% | +2.1% |
| Full Model | **81.5%** | +3.3% |

### Key Findings
- Temporal attention delivers a higher performance contribution than spatial attention.
- The model achieves optimal performance at 4 timesteps, with significantly lower energy consumption than ANNs.
- An improvement of 2.1% is achieved on ImageNet, indicating a more pronounced advancement in large-scale tasks.

## Highlights & Insights
- Decoupling spatial-temporal attention modularizes the design, enabling independent validation of each component.
- Spike-compatible attention preserves the energy efficiency advantages of SNNs.
- Adaptive membrane parameters enhance biological fidelity.

## Limitations & Future Work
- An accuracy gap with ANNs still exists (approximately 6-7% on ImageNet).
- Whether the additional computational cost of attention is fully offset by the energy efficiency of the SNN requires more rigorous analysis.
- The effectiveness of temporal attention on video or event-driven data requires further validation.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of spatially-temporally decoupled attention in SNNs is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough validations across multiple datasets alongside detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured.
- Value: ⭐⭐⭐⭐ Facilitates bridging the gap between SNNs and ANNs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] STAA-SNN: Spatial-Temporal Attention Aggregator for Spiking Neural Networks](staa-snn_spatial-temporal_attention_aggregator_for_spiking_neural_networks.md)
- [\[CVPR 2025\] Spiking Transformer: Introducing Accurate Addition-Only Spiking Self-Attention for Transformer](spiking_transformer_introducing_accurate_addition-only_spiking_self-attention_fo.md)
- [\[CVPR 2025\] Rethinking Spiking Self-Attention Mechanism: Implementing a-XNOR Similarity Calculation in Spiking Transformers](rethinking_spiking_self-attention_mechanism_implementing_a-xnor_similarity_calcu.md)
- [\[NeurIPS 2025\] Spectral Conditioning of Attention Improves Transformer Performance](../../NeurIPS2025/llm_nlp/spectral_conditioning_of_attention_improves_transformer_performance.md)
- [\[CVPR 2025\] Exposure-slot: Exposure-centric Representations Learning with Slot-in-Slot Attention](exposure-slot_exposure-centric_representations_learning_with_slot-in-slot_attent.md)

</div>

<!-- RELATED:END -->
