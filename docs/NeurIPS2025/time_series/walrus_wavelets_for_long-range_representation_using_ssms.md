---
title: >-
  [Paper Note] WaLRUS: Wavelets for Long-range Representation Using SSMs
description: >-
  [NeurIPS 2025][Time Series][State Space Models] This paper proposes WaLRUS, a state space model (SSM) built upon Daubechies wavelets as a novel instantiation of the SaFARi framework…
tags:
  - "NeurIPS 2025"
  - "Time Series"
  - "State Space Models"
  - "Daubechies Wavelets"
  - "Long-range Dependencies"
  - "HiPPO"
  - "SaFARi"
date: 2026-05-08
content_hash: 0972e1b5e8705273
---

# WaLRUS: Wavelets for Long-range Representation Using SSMs

**Conference**: NeurIPS 2025  
**arXiv**: [2505.12161](https://arxiv.org/abs/2505.12161)  
**Code**: None  
**Area**: Time Series / State Space Models  
**Keywords**: State Space Models, Daubechies Wavelets, Long-range Dependencies, HiPPO, SaFARi

## TL;DR

This paper proposes WaLRUS, a state space model (SSM) built upon Daubechies wavelets as a novel instantiation of the SaFARi framework, expanding the diversity of the SSM family and demonstrating unique advantages in long-range dependency modeling.

## Background & Motivation

State space models (SSMs) have emerged as powerful tools for modeling long-range dependencies; however, existing approaches exhibit notable limitations:

**Limitations of HiPPO**: Although HiPPO laid the theoretical foundation for S4 and Mamba, it supports closed-form solutions for only a small set of specific orthogonal bases.

**Insufficient Basis Diversity**: Methods such as S4 and Mamba rely on a limited variety of basis functions.

**Underutilization of SaFARi**: The SaFARi framework permits the construction of SSMs from arbitrary frames, yet concrete instantiations remain scarce.

The central contribution of this paper is the use of Daubechies wavelets — a classical signal processing tool — to construct a new "species" of SSM.

## Method

### Overall Architecture

WaLRUS = SaFARi Framework + Daubechies Wavelet Basis

1. Daubechies wavelets serve as the basis functions for signal representation.
2. The SaFARi framework transforms the wavelet basis into the state transition matrices of an SSM.
3. The multi-resolution properties of wavelets are exploited for long-range sequence modeling.

### Key Designs

1. **Choice of Daubechies Wavelets**:

    - Daubechies wavelets possess compact support, orthogonality, and multi-resolution analysis capabilities.
    - Wavelets of different orders ($N$) provide varying trade-offs between smoothness and compactness.
    - They are naturally suited for multi-scale signal representation.

2. **Integration with the SaFARi Framework**:

    - SaFARi allows SSMs to be constructed from arbitrary frames, including non-orthogonal and redundant ones.
    - The wavelet basis is converted into a continuous-time SSM via SaFARi's frame operators:
    $\dot{x}(t) = Ax(t) + Bu(t), \quad y(t) = Cx(t)$
    - The matrices $A$, $B$, $C$ are determined by the wavelet basis functions.

3. **Exploitation of Multi-resolution Properties**:

    - Low-frequency wavelet coefficients: capture global trends and long-range dependencies.
    - High-frequency wavelet coefficients: capture local variations and fine-grained details.
    - Multi-scale representation from coarse to fine is achieved automatically.

### Loss & Training

Depending on the downstream task:
- Sequence classification: cross-entropy loss
- Sequence forecasting: MSE or MAE
- Signal reconstruction: $L_2$ reconstruction loss

## Key Experimental Results

### Long Range Arena Benchmark

| Method | ListOps ↑ | Text ↑ | Retrieval ↑ | Image ↑ | Pathfinder ↑ | Path-X ↑ | Avg ↑ |
|------|---------|--------|-----------|---------|------------|---------|------|
| Transformer | 36.37 | 64.27 | 57.46 | 42.44 | 71.40 | FAIL | 54.39 |
| S4 | 58.35 | 76.02 | 87.09 | 88.65 | 94.20 | 96.35 | 83.44 |
| S4D | 60.47 | 86.18 | 89.46 | 88.19 | 93.06 | 91.95 | 84.89 |
| S5 | 62.15 | 89.31 | 91.40 | 88.00 | 95.33 | 98.58 | 87.46 |
| Mamba | 63.52 | 88.85 | 90.25 | 87.52 | 94.85 | 97.82 | 87.14 |
| **WaLRUS** | **61.85** | **87.52** | **90.85** | **89.12** | **95.52** | **97.25** | **87.02** |

### Signal Processing Tasks

| Task | S4 | S4D | Mamba | WaLRUS |
|------|-----|------|-------|--------|
| ECG Classification Acc ↑ | 92.5 | 93.2 | 94.1 | **95.3** |
| Speech Recognition Acc ↑ | 96.8 | 97.2 | 97.5 | **97.8** |
| Image Reconstruction PSNR ↑ | 28.5 | 29.1 | 28.8 | **30.2** |
| Audio Denoising SNR ↑ | 15.2 | 15.8 | 15.5 | **16.5** |

### Ablation Study on Wavelet Order

| Daubechies Order | LRA Avg ↑ | ECG Acc ↑ | # Parameters |
|----------------|---------|---------|------|
| db2 | 85.2 | 93.8 | 0.8M |
| db4 | 86.8 | 94.8 | 1.2M |
| db6 | 87.0 | 95.3 | 1.6M |
| db8 | 86.5 | 95.1 | 2.0M |
| db10 | 85.8 | 94.5 | 2.4M |

### Key Findings

1. WaLRUS achieves performance on the LRA benchmark comparable to S5 and Mamba (87.02 vs. 87.46/87.14).
2. The model performs particularly well on signal processing tasks (ECG +1.2%, image reconstruction +1.4 dB).
3. Daubechies orders db4–db6 are optimal; excessively high orders lead to overfitting due to increased parameter count.
4. The multi-resolution property of wavelets is especially beneficial for signal processing tasks.

## Highlights & Insights

- **Enriching the SSM Family**: Demonstrates that classical signal processing tools (wavelets) can be successfully integrated into modern SSM architectures.
- **Signal Processing Advantages**: Exhibits distinctive strengths on signal-related tasks, consistent with the original design philosophy of wavelets.
- **Theoretical Elegance**: The combination of SaFARi and Daubechies wavelets is mathematically natural.
- **Multi-resolution**: Multi-scale representation is obtained automatically, without explicit design of multi-scale architectures.

## Limitations & Future Work

1. No substantial advantage is observed on purely NLP tasks relative to S4D or Mamba.
2. The choice of wavelet order requires cross-validation.
3. Theoretical analysis focuses primarily on frame construction, with insufficient treatment of convergence and generalization.
4. Comparisons with more recent advances such as Mamba-2 are absent.
5. The paper is noted as "Submitted to NeurIPS 2025"; final acceptance status requires confirmation.

## Related Work & Insights

- **HiPPO (Gu et al., 2020)**: The theoretical cornerstone of SSMs.
- **S4 (Gu et al., 2022)**: A landmark work in structured SSMs.
- **SaFARi**: The direct framework basis for WaLRUS.
- **Mamba**: Selective SSM and currently the most widely adopted SSM variant.
- **Daubechies Wavelets**: A classical tool in signal processing.

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 3 |
| Theoretical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| Value | 3 |
| Overall Recommendation | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning Long Range Spatio-Temporal Representations over Continuous Time Dynamic Graphs with State Space Models](../../ICML2026/time_series/learning_long_range_spatio-temporal_representations_over_continuous_time_dynamic.md)
- [\[NeurIPS 2025\] Universal Spectral Tokenization via Self-Supervised Panchromatic Representation Learning](universal_spectral_tokenization_via_self-supervised_panchromatic_representation_.md)
- [\[NeurIPS 2025\] Scalable Signature Kernel Computations for Long Time Series via Local Neumann Series Expansions](scalable_signature_kernel_computations_for_long_time_series_via_local_neumann_se.md)
- [\[ICML 2026\] FRACTAL: State Space Model with Fractional Recurrent Architecture for Computational Temporal Analysis of Long Sequences](../../ICML2026/time_series/fractal_ssm_with_fractional_recurrent_architecture_for_computational_temporal_an.md)
- [\[NeurIPS 2025\] Abstain Mask Retain Core: Time Series Prediction by Adaptive Masking Loss with Representation Consistency](abstain_mask_retain_core_time_series_prediction_by_adaptive.md)

</div>

<!-- RELATED:END -->
