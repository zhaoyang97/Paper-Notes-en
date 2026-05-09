---
title: >-
  [Paper Note] PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation
description: >-
  [NeurIPS 2025][Medical Imaging][Wavelet transform] This paper proposes PhysioWave, a multi-scale Transformer architecture based on learnable wavelet decomposition and frequency-guided masking. It establishes, for the first time, large-scale pretrained foundation models for EMG and ECG, and achieves state-of-the-art performance on both unimodal and multimodal physiological signal tasks through a multimodal fusion framework.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Wavelet transform
  - physiological signals
  - self-supervised learning
  - multimodal fusion
  - foundation model
date: 2026-05-08
content_hash: e7ddf2e5524ad6df
---

# PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation

**Conference**: NeurIPS 2025
**arXiv**: [2506.10351](https://arxiv.org/abs/2506.10351)
**Code**: [Available](https://github.com/ForeverBlue816/PhysioWave)
**Area**: Medical Imaging / Biosignal Processing
**Keywords**: Wavelet transform, physiological signals, self-supervised learning, multimodal fusion, foundation model

## TL;DR

This paper proposes PhysioWave, a multi-scale Transformer architecture based on learnable wavelet decomposition and frequency-guided masking. It establishes, for the first time, large-scale pretrained foundation models for EMG and ECG, and achieves state-of-the-art performance on both unimodal and multimodal physiological signal tasks through a multimodal fusion framework.

## Background & Motivation

Physiological signals (EEG, EMG, ECG) are core data sources for health monitoring, clinical diagnosis, and brain-computer interfaces, yet they present three major challenges: (1) **low signal-to-noise ratio**: severe interference from motion artifacts, baseline drift, etc.; (2) **strong non-stationarity**: signals contain spikes and abrupt transitions that conventional time-domain or fixed-window Fourier methods cannot effectively capture; (3) **cross-modal heterogeneity**: significant differences in sampling rates and dimensionality across modalities.

Although pretrained models such as LaBraM and EEGPT exist for EEG, foundation models for EMG and ECG remain absent. Existing NLP-inspired self-supervised methods (e.g., random token masking) are ill-suited for physiological signals—raw signal segments do not correspond to meaningful units like words, and random dropping may remove critical events or mask redundant regions.

## Method

### Overall Architecture

The pretraining pipeline of PhysioWave consists of four stages: (1) learnable wavelet decomposition decomposes raw multi-channel signals into multi-scale frequency-band representations; (2) frequency-guided masking (FgM) selectively masks high-information patches based on FFT energy; (3) a Transformer encoder processes the token sequence; (4) a lightweight decoder reconstructs the masked patches.

### Key Designs

1. **Adaptive Wavelet Selector**: Maintains $M$ candidate wavelet bases $\{(k_w^{\text{low}}, k_w^{\text{high}})\}_{w=1}^M$ and computes selection weights via MLP + Softmax: $\alpha = \text{Softmax}(\text{MLP}(\text{AvgPool}(x)))$, adaptively combining filters as $k^{\text{low}} = \sum_w \alpha_w k_w^{\text{low}}$. The **design motivation** is that different signal characteristics require different wavelet bases, and manual selection cannot accommodate diverse signals.

2. **Soft-Gated Multi-Resolution Analysis**: After each decomposition layer, the signal is upsampled back to the original length, and multi-head attention estimates an adaptive gate $G_c^{(\ell)} \in [0,1]$ that dynamically weights the current-layer signal and the upsampled signal: $\hat{a}_c^{(\ell)}[n] = G_c^{(\ell)} a_c^{(\ell)} + (1-G_c^{(\ell)}) \tilde{a}_c^{(\ell+1)}$. Compared to U-Net's hard skip connections, soft gating enables per-channel adjustment of frequency content emphasis, reducing aliasing and ringing artifacts.

3. **Cross-Scale Channel-Aggregating Feed-Forward Network (Cross-Scale CAFFN)**: Wavelet features at each layer are fused across scales via channel aggregation and multi-head attention, where the current-layer features serve as queries and shallower features serve as keys/values: $Y^{(\ell)} = U^{(\ell)} + \beta \cdot \text{Attention}(U^{(\ell)}, \{Y^{(i)}\}_{i<\ell})$, enabling fine-grained subband features to incorporate long-range patterns from coarser resolutions.

4. **Frequency-guided Masking (FgM)**: The FFT spectral energy of each patch is computed and mixed with random noise to yield an importance score $s_n = \alpha \cdot e_n + (1-\alpha) \cdot z_n$, prioritizing high-energy patches for masking. This forces the model to infer critical information from context, producing richer discriminative features compared to random time-domain masking.

### Loss & Training

- **Pretraining loss**: Smooth-L1 reconstruction loss computed only on masked patches: $\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{n \in \mathcal{M}} \text{SmoothL1}(\hat{p}_n, p_n)$
- **Unimodal downstream**: End-to-end fine-tuning; mean pooling followed by a two-layer MLP classifier.
- **Multimodal downstream**: Pretrained encoders for each modality are frozen; only the classification head and softmax-constrained fusion weights are trained: $z_{\text{fused}} = \sum_{m \in \mathcal{M}} \alpha_m z_m$

## Key Experimental Results

### Main Results

**ECG Rhythm Classification (PTB-XL)**

| Method | Params | F1 (%) | AUROC (%) |
|--------|--------|--------|-----------|
| ECG-Chat (2024) | 13B | 55.9 | 94.1 |
| MaeFE (2023) | 9M | 64.7 | 88.6 |
| **PhysioWave-Large** | **37M** | **66.7** | **94.6** |

**EMG Gesture Recognition (EPN-612)**

| Method | Params | Acc (%) | F1 (%) |
|--------|--------|---------|--------|
| Moment (2024) | 385M | 90.87 | 90.16 |
| OTiS (2024) | 45M | 87.55 | 88.03 |
| **PhysioWave-Large** | **37M** | **94.50** | **94.56** |

### Ablation Study

| Configuration | Training Loss | Acc (%) | F1 (%) |
|---------------|--------------|---------|--------|
| w/o FgM (random masking) | 0.24 | 92.48 | 92.85 |
| w/o pretraining | 0.27 | 91.67 | 91.57 |
| **Full model** | **0.22** | **93.12** | **93.67** |

### Key Findings

- PhysioWave-Large achieves 66.7% F1 on PTB-XL, surpassing the 13B-parameter ECG-Chat.
- The EMG model outperforms Moment across three benchmarks using fewer than one-tenth of its parameters.
- Multimodal fusion yields a +7.3% accuracy improvement on DEAP emotion recognition (81.3%→88.6%).
- FgM improves F1 by 0.8% over random masking; pretraining contributes an additional 2.1% F1 gain.

## Highlights & Insights

- **The first large-scale pretrained foundation models for EMG and ECG**, trained on 823 GB of EMG and 182 GB of ECG data respectively, filling a critical gap in the field.
- Learnable wavelet decomposition elegantly addresses the multi-scale and non-stationary characteristics of physiological signals, proving highly effective as a general-purpose front-end.
- The FgM strategy upgrades masking from "random dropout" to "information-guided selection," aligning well with the non-uniform information distribution inherent in physiological signals.
- The multimodal framework adopts a frozen-encoder plus lightweight-fusion design, mitigating overfitting risks on small downstream datasets.

## Limitations & Future Work

- Other modalities such as optical biosensors are not yet supported.
- The multimodal fusion relies on simple weighted summation; more sophisticated attention-based fusion may yield further improvements.
- Pretraining data are biased toward Western populations, and generalization across different ethnic and age groups remains to be validated.
- Real-time inference efficiency is not thoroughly discussed, which may pose challenges for edge device deployment.

## Related Work & Insights

- **Relation to LaBraM / EEGPT**: PhysioWave extends the pretraining paradigm from EEG to EMG and ECG, providing a unified multimodal framework.
- **Comparison with Moment / OTiS**: General-purpose time-series foundation models underperform domain-specific models on physiological signals, underscoring the importance of domain-specific design choices such as the wavelet front-end.
- **Insight**: The combination of learnable wavelet decomposition and frequency-guided masking may generalize to other non-stationary signal analysis tasks, such as seismic or acoustic signal processing.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of learnable wavelets and FgM is novel, though individual components have prior precedents.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Three modalities, 6+ datasets, comprehensive ablations.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure with detailed formulations.)
- Value: ⭐⭐⭐⭐⭐ (Fills a critical gap in EMG/ECG foundation models with high practical applicability.)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Unpaired Image-to-Image Translation for Segmentation and Signal Unmixing](unpaired_image-to-image_translation_for_segmentation_and_signal_unmixing.md)
- [\[NeurIPS 2025\] STARC-9: A Large-scale Dataset for Multi-Class Tissue Classification for CRC Histopathology](starc-9_a_large-scale_dataset_for_multi-class_tissue_classification_for_crc_hist.md)
- [\[NeurIPS 2025\] LoMix: Learnable Weighted Multi-Scale Logits Mixing for Medical Image Segmentation](lomix_learnable_weighted_multi-scale_logits_mixing_for_medical_image_segmentatio.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[NeurIPS 2025\] Domain-Adaptive Transformer for Data-Efficient Glioma Segmentation in Sub-Saharan MRI](domain-adaptive_transformer_for_data-efficient_glioma_segmentation_in_sub-sahara.md)

<!-- RELATED:END -->
