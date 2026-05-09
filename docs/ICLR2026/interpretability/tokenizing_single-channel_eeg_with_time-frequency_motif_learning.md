---
title: >-
  [Paper Note] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning
description: >-
  [ICLR 2026][EEG signal analysis] This paper proposes TFM-Tokenizer, the first framework that learns a time-frequency motif vocabulary from single-channel EEG and encodes it into discrete tokens. It consistently improves performance on tasks such as event classification and seizure detection, and can serve as a plug-and-play component to enhance existing EEG foundation models.
tags:
  - ICLR 2026
  - EEG signal analysis
  - discrete tokenization
  - time-frequency motif
  - vector quantization
  - foundation models
date: 2026-05-08
content_hash: 6bff4484138b420b
---

# Tokenizing Single-Channel EEG with Time-Frequency Motif Learning

**Conference**: ICLR 2026
**arXiv**: [2502.16060](https://arxiv.org/abs/2502.16060)
**Code**: [https://github.com/Jathurshan0330/TFM-Tokenizer](https://github.com/Jathurshan0330/TFM-Tokenizer)
**Area**: Interpretability
**Keywords**: EEG signal analysis, discrete tokenization, time-frequency motif, vector quantization, foundation models

## TL;DR

This paper proposes TFM-Tokenizer, the first framework that learns a time-frequency motif vocabulary from single-channel EEG and encodes it into discrete tokens. It consistently improves performance on tasks such as event classification and seizure detection, and can serve as a plug-and-play component to enhance existing EEG foundation models.

## Background & Motivation

- **State of the Field**: Inspired by NLP, EEG analysis is shifting toward task-agnostic foundation model paradigms.
- **Limitations of Prior Work**: Tokenization is central to NLP, yet existing EEG foundation models simply segment continuous signals into short time windows without data-driven vocabulary learning. Although LaBraM proposes a neural tokenizer, it is used only as a training objective and discarded during downstream inference.
- **Root Cause — Three Core Challenges**:
  1. **Tokenization granularity**: Operation must be performed at the single-channel level to achieve device independence.
  2. **Token resolution**: Tokens should represent underlying motifs (short recurring patterns) rather than plain time segments.
  3. **Learning objective**: Time-frequency information must be explicitly incorporated, as the time domain alone cannot capture important frequency patterns.

## Method

### Overall Architecture

A two-stage design:
1. **TFM-Tokenizer pre-training**: Unsupervised learning of a time-frequency motif vocabulary from single-channel EEG.
2. **Downstream Transformer training**: Masked pre-training and fine-tuning using discrete token sequences.

### Key Designs 1: Dual-Path Time-Frequency Encoding

**Localized Spectral Window Encoder**:
- Divides the spectrogram into $P$ non-overlapping patches along the frequency axis.
- Each patch is independently projected: $e_{(i,p)} = \text{GroupNorm}(\text{GeLU}(\mathbf{W}_p \mathbf{S}_{(i,p)}))$
- A frequency Transformer models cross-band dependencies.
- **Gated per-patch aggregation**: A sigmoid gate selectively emphasizes informative frequency patches:

$$\mathbf{E}_i^F = \text{Concat}\left[\sigma(\mathbf{W}_{g1} \mathbf{e}_{(i,p)}) \mathbf{W}_{g2} \mathbf{e}_{(i,p)}\right]$$

**Temporal Encoder**: Linearly projects raw EEG patches followed by GELU and GroupNorm.

**Temporal Transformer**: Models long-range dependencies by concatenating frequency embeddings $\mathbf{E}_i^F$ and temporal embeddings $\mathbf{E}_i^T$.

### Key Designs 2: VQ Vocabulary Learning

Vector quantization (VQ-VAE) maps fused embeddings to a discrete codebook:

$$q(\mathbf{z}_i) = \arg\min_{\mathbf{v}_k \in \mathcal{V}} \|\mathbf{z}_i - \mathbf{v}_k\|_2^2$$

### Key Designs 3: Time-Frequency Masked Prediction

A joint frequency–time masking strategy:
- Random grouped masking $M_F$ along the frequency axis and random masking $M_T$ along the time axis.
- Symmetric masking is applied for data augmentation.

Overall loss:

$$\mathcal{L}_{\text{token}} = \sum_{(f,t)} \|\mathbf{S}(f,t) - \hat{\mathbf{S}}(f,t)\|_2^2 + \alpha \sum_i \|\text{sg}[E_i] - v_i\|_2^2 + \beta \sum_i \|E_i - \text{sg}[v_i]\|_2^2$$

- Reconstruction loss + codebook update (commitment loss + exponential moving average).
- No positional encoding is used, given the non-stationary and potentially chaotic nature of EEG.

### Downstream Transformer

- Initializes the token embedding lookup table with the VQ codebook.
- Linear-attention Transformer (~0.7M parameters).
- Incorporates channel embeddings and positional embeddings across channels.
- Pre-trained with masked token prediction, then fine-tuned on downstream tasks.

## Key Experimental Results

### Main Results: TUEV Event Classification

| Model | Parameters | Cohen's Kappa (Single Dataset) | Cohen's Kappa (Multi-Dataset) |
|-------|------------|-------------------------------|-------------------------------|
| SPaRCNet | 0.79M | 0.4233 | - |
| BIOT | 3.2M | 0.4482 | - |
| BIOT⋆ | 3.2M | 0.4890 | - |
| LaBraM⋆ | ~6M | - | 0.5588 |
| **TFM-Tokenizer** | **~0.7M** | **~0.53** | **0.6189 (+11%)** |

### IIIC Seizure Classification

| Model | Cohen's Kappa (Multi-Dataset) |
|-------|-------------------------------|
| LaBraM | 0.3658 |
| CBraMod | 0.4792 |
| **TFM-Tokenizer** | **0.4979 (+36% vs LaBraM)** |

### Cross-Device Scalability: Ear-EEG Sleep Staging

| Setting | TFM-Tokenizer vs. Baseline |
|---------|---------------------------|
| Ear-EEG (non-standard 10-20 system) | **+14%** |

### Integration with Existing Foundation Models

| Foundation Model | Original | + TFM-Tokenizer |
|-----------------|----------|----------------|
| BIOT | baseline | **+~4% (TUEV)** |
| LaBraM | baseline | **+~4% (TUEV)** |

### Key Findings

- TFM-Tokenizer achieves state-of-the-art performance with 3× fewer parameters than LaBraM and 1.5× fewer than BIOT.
- As a plug-and-play component, it consistently improves existing foundation models such as BIOT and LaBraM.
- Cross-device experiments on ear-EEG demonstrate that single-channel tokenization generalizes well across devices.
- Token analysis shows that learned tokens are class-discriminative, frequency-aware, and consistent.
- The gated aggregation mechanism effectively focuses on task-relevant frequency bands.

## Highlights & Insights

- **First genuine EEG tokenization**: Learns a discrete motif vocabulary used directly as downstream model input, rather than solely as a training objective.
- **Device-agnostic design**: Single-channel operation allows the tokenizer to adapt to arbitrary channel configurations and devices.
- **Extremely lightweight**: A downstream Transformer with ~0.7M parameters achieves state-of-the-art performance.
- **Interpretability**: Discrete tokens correspond to specific neurophysiological events and support timestamp-level retrieval.

## Limitations & Future Work

- The VQ codebook size $K$ must be predefined and may require adjustment for different EEG types.
- Validation is currently limited to classification tasks; generative tasks (e.g., EEG reconstruction, cross-modal translation) remain unexplored.
- The frequency patch size and band-splitting strategy in gated aggregation may need tuning for different sampling rates.
- The scale of multi-dataset pre-training remains far smaller than NLP corpora, leaving the upper-bound potential of the tokenizer underexplored.
- The ear-EEG experiment involves only 10 subjects, limiting statistical power.

## Related Work & Insights

- **EEG Foundation Models**: BIOT (segment-level continuous tokenization), LaBraM (VQ tokenizer used only as training objective), BRANT, MMM.
- **VQ Tokenizers**: Applications of VQ-VAE in images (VQGAN) and EEG (LaBraM).
- **EEG Motif Learning**: Only a few prior works (Schäfer & Leser 2022) address time-domain motifs; joint time-frequency motif learning is introduced here for the first time.
- **Signal Tokenization**: Design philosophy draws from NLP tokenization methods (BPE/WordPiece) and applies them to continuous signals.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★★ |
| Theoretical Depth | ★★★☆☆ |
| Experimental Thoroughness | ★★★★☆ |
| Value | ★★★★☆ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning](uni-ntfm_a_unified_foundation_model_for_eeg_signal_representation_learning.md)
- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](../../NeurIPS2025/interpretability/fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[NeurIPS 2025\] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex](../../NeurIPS2025/interpretability/time-evolving_dynamical_system_for_learning_latent_representations_of_mouse_visu.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)

<!-- RELATED:END -->
