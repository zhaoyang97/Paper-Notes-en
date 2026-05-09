---
title: >-
  [Paper Note] MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition
description: >-
  [NeurIPS 2025][Motion Primitives] MoPFormer is proposed to decompose wearable sensor signals into sequences of motion primitives and model their temporal dependencies via a Transformer, surpassing state-of-the-art methods on multiple HAR benchmarks while maintaining a lightweight architecture.
tags:
  - NeurIPS 2025
  - Motion Primitives
  - Transformer
  - Wearable Sensors
  - Activity Recognition
  - Temporal Decomposition
date: 2026-05-08
content_hash: a2b884a8f29f0d6b
---

# MoPFormer: Motion-Primitive Transformer for Wearable-Sensor Activity Recognition

**Conference**: NeurIPS 2025
**arXiv**: [2505.20744](https://arxiv.org/abs/2505.20744)
**Code**: Available
**Area**: Interpretability
**Keywords**: Motion Primitives, Transformer, Wearable Sensors, Activity Recognition, Temporal Decomposition

## TL;DR

MoPFormer is proposed to decompose wearable sensor signals into sequences of motion primitives and model their temporal dependencies via a Transformer, surpassing state-of-the-art methods on multiple HAR benchmarks while maintaining a lightweight architecture.

## Background & Motivation

**Background**: Human activity recognition (HAR) using wearable sensors is widely applied in health monitoring and sports analysis; mainstream approaches employ CNNs or RNNs to process raw signals directly.

**Limitations of Prior Work**: (1) Raw signals contain substantial noise, and varying sampling rates hinder generalization; (2) CNNs struggle to capture long-range temporal dependencies; (3) standard Transformers adopt unreasonable tokenization strategies for HAR signals.

**Key Challenge**: The continuous nature of sensor signals conflicts with the discrete token format required by Transformers.

**Key Insight**: Human activities can be naturally decomposed into motion primitives (e.g., "raise arm," "take a step"), which serve as more semantically meaningful tokens for the Transformer.

## Method

### Overall Architecture

Input IMU signals → Motion primitive extraction (learned segmentation + encoding) → Primitive-sequence Transformer → Activity classification.

### Key Designs

1. **Motion Primitive Extraction**

   - **Function**: Automatically decomposes continuous sensor signals into discrete primitive sequences.
   - **Mechanism**: A learned segmentation network identifies primitive boundaries in the signal; an encoder maps each segment to a fixed-length primitive embedding.
   - **Design Motivation**: Primitives constitute more natural and stable representation units that are invariant to sampling rate variations.

2. **Primitive Transformer**

   - **Function**: Models temporal dependencies among motion primitives.
   - **Mechanism**: Standard Transformer encoder with positional encoding, multi-head self-attention, and feed-forward layers.
   - **Design Motivation**: The primitive sequence is far shorter than the raw signal (10–20 primitives vs. hundreds of sample points), yielding high computational efficiency.

3. **Lightweight Design**

   - **Function**: Keeps the model compact for deployment on edge devices.
   - **Mechanism**: Small embedding dimensions (64–128), shallow Transformer (2–4 layers), and parameter sharing.
   - **Design Motivation**: Wearable devices have limited computational resources.

### Loss & Training

Cross-entropy classification loss combined with an auxiliary primitive segmentation loss. The model is trained end-to-end.

## Key Experimental Results

### Main Results

| Method | UCI-HAR Acc↑ | PAMAP2 F1↑ | Opportunity F1↑ | Params↓ |
|--------|-------------|-----------|----------------|---------|
| DeepConvLSTM | 93.2% | 89.5% | 85.3% | 2.1M |
| InceptionTime | 94.1% | 90.8% | 86.7% | 3.5M |
| HAR-Transformer | 94.5% | 91.2% | 87.1% | 4.2M |
| **MoPFormer** | **95.8%** | **92.7%** | **89.3%** | **0.8M** |

### Ablation Study

| Configuration | UCI-HAR Acc | Note |
|---------------|-----------|------|
| Fixed-window tokenization | 93.8% | Standard sliding-window segmentation |
| Learned segmentation, no Transformer | 94.2% | Primitives + MLP |
| Learned segmentation + Transformer | **95.8%** | Full model |

### Key Findings

- MoPFormer outperforms all baselines on every benchmark while achieving the smallest parameter count (0.8M vs. 4.2M).
- Motion primitive decomposition contributes most significantly — even replacing the Transformer with an MLP yields better results than standard methods.
- Strong cross-sampling-rate generalization — performance drops only 1.2% when transferring from 50 Hz to 25 Hz, compared to 5%+ for standard methods.

## Highlights & Insights

- **Naturalness of Motion Primitives**: Human motion is inherently composed of elemental primitives, making this representation more aligned with kinematic reality. The approach is transferable to robot action recognition, sign language recognition, and related domains.
- **Efficiency Advantage**: Primitive sequences are substantially shorter than raw signals, drastically reducing the computational cost of the Transformer while simultaneously improving accuracy.
- **Robustness**: Resilience to sampling rate variation is a critical practical advantage for real-world deployment.

## Limitations & Future Work

- The interpretability of learned primitive segmentations remains to be validated — it is unclear whether the discovered primitives correspond to genuine motion units.
- Evaluation is limited to IMU data; extension to other sensor modalities (EMG, pressure) is necessary.
- Primitive decomposition may be inaccurate for complex transitional activities (e.g., "sitting down immediately followed by standing up").

## Related Work & Insights

- **vs. HAR-Transformer**: Applies fixed-window tokenization directly to raw signals, ignoring the natural structure of motion; MoPFormer's learned segmentation provides a more principled alternative.
- **vs. DeepConvLSTM**: A CNN-RNN hybrid with a large parameter count and insufficient capacity to capture long-range dependencies.

## Rating
- Novelty: ⭐⭐⭐⭐ The use of motion primitives as tokens is intuitive and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear and well-structured presentation.
- Value: ⭐⭐⭐⭐ High practical value for real-world wearable HAR applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)
- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[CVPR 2026\] Geometry-Guided Camera Motion Understanding in VideoLLMs](../../CVPR2026/interpretability/geometryguided_camera_motion_understanding_in_vide.md)
- [\[NeurIPS 2025\] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework](discovering_transformer_circuits_via_a_hybrid_attribution_and_pruning_framework.md)
- [\[NeurIPS 2025\] Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models](bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)

</div>

<!-- RELATED:END -->
