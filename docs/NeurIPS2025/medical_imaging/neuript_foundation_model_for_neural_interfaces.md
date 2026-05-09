---
title: >-
  [Paper Note] NeurIPT: Foundation Model for Neural Interfaces
description: >-
  [NeurIPS 2025][Medical Imaging][EEG foundation model] NeurIPT is an EEG foundation model for diverse brain–computer interface (BCI) applications. Through four key innovations—Amplitude-Aware Masking Pre-training (AAMP), Progressive Mixture-of-Experts (PMoE) architecture, 3D electrode spatial encoding, and Intra- and Inter-Lobe Pooling (IILP)—it achieves state-of-the-art performance across eight downstream BCI tasks.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - EEG foundation model
  - self-supervised pre-training
  - Mixture-of-Experts
  - EEG signals
  - brain–computer interface
date: 2026-05-08
content_hash: 9fa95a5ceabdaf07
---

# NeurIPT: Foundation Model for Neural Interfaces

**Conference**: NeurIPS 2025
**arXiv**: [2510.16548](https://arxiv.org/abs/2510.16548)
**Code**: [https://github.com/](https://github.com/) (available; project page provided)
**Area**: Medical Imaging / Brain–Computer Interfaces
**Keywords**: EEG foundation model, self-supervised pre-training, Mixture-of-Experts, EEG signals, brain–computer interface

## TL;DR
NeurIPT is an EEG foundation model for diverse brain–computer interface (BCI) applications. Through four key innovations—Amplitude-Aware Masking Pre-training (AAMP), Progressive Mixture-of-Experts (PMoE) architecture, 3D electrode spatial encoding, and Intra- and Inter-Lobe Pooling (IILP)—it achieves state-of-the-art performance across eight downstream BCI tasks.

## Background & Motivation

Electroencephalography (EEG) is widely used in clinical diagnosis and brain–computer interfaces owing to its non-invasiveness, portability, and high temporal resolution. As large-scale EEG datasets continue to emerge, researchers seek to establish EEG foundation models (FMs) analogous to those in NLP and CV, enabling generalization across datasets and tasks.

Existing EEG foundation model approaches, however, face several critical challenges:

**Deficiencies in spatial encoding**: Current positional encodings treat electrode channels as interchangeable, ignoring the true three-dimensional physical arrangement of electrodes in space, which severely impairs cross-dataset transferability.

**Limitations of masked pre-training**: BERT-style random contiguous-segment masking strategies tend to encourage "local interpolation" rather than the learning of meaningful global representations—the model need only interpolate from neighboring points adjacent to the masked region.

**Inadequate downstream fine-tuning**: Conventional fully connected layers or global pooling mechanisms cannot explicitly exploit regional features from distinct cortical areas.

**Signal heterogeneity**: EEG signal patterns are highly diverse, ranging from slow-wave oscillations during sleep to rapid spikes during epileptic seizures; a single feed-forward network struggles to adaptively capture such heterogeneous temporal dynamics.

The central approach is to design an EEG foundation model that simultaneously addresses spatio-temporal heterogeneity, proposing targeted solutions along both the temporal and spatial dimensions.

## Method

### Overall Architecture

NeurIPT adopts an encoder–decoder architecture built upon Crossformer's hierarchical attention module, comprising a pre-training stage and a fine-tuning stage. During pre-training, the model learns robust representations via self-supervised learning (without any labeled data) on over 2,000 hours of EEG data. During fine-tuning, classification tasks are performed on eight downstream BCI datasets.

### Key Designs

1. **3D-Aligned Spatial Encoding**:

    - The three-dimensional physical coordinates $(x_d, y_d, z_d)$ of EEG electrodes in the international 10–20 system are used. Each spatial coordinate is encoded with sinusoidal functions and the results are concatenated: $PE^{(s)}_d = \text{Concat}(PE_x(x_d), PE_y(y_d), PE_z(z_d))$
    - Point-level embedding (pixel-level embedding) is adopted to preserve temporal detail; each data point simultaneously encodes both temporal and spatial information: $\mathbf{s}_{t,d} = \mathbf{E}\mathbf{x} + PE^{(t)} + PE^{(s)}$
    - **Design Motivation**: This natively supports variation in temporal length $T$ and spatial dimension $D$, requiring no additional convolution or padding operations, and seamlessly accommodates different electrode placement standards such as 10–05 and 10–20.

2. **Amplitude-Aware Masking Pre-training (AAMP)**:

    - **Core Idea**: Rather than randomly masking contiguous temporal segments, masking is performed based on signal amplitude. For each channel, a random percentile $\xi_d \sim \mathcal{U}(0,1)$ is sampled, and the interval centered at that percentile covering $T \cdot \mathcal{P}$ points in the amplitude-sorted sequence is masked.
    - Masking formula: $\mathcal{M} = \{\mathbb{1}\{x_{t,d} \in [\mathcal{L}_d, \mathcal{U}_d]\}\}$
    - **Design Motivation**: Amplitude serves as a proxy for signal energy, compelling the model to learn underlying EEG patterns rather than performing simple local interpolation. Because the masked points are non-contiguous in the time axis (samples within the same amplitude range are distributed throughout the entire time series), the model must understand global signal structure in order to reconstruct.
    - Reconstruction loss: $\mathcal{L}_{AAMP} = \frac{1}{n}(\sum_{i=1}^{n}\|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\|^p)^{1/p}$

3. **Progressive Mixture-of-Experts (PMoE)**:

    - EEG signals contain complex heterogeneous information (different frequency bands, transient events, artifacts), which a single FFN cannot handle adequately. PMoE progressively introduces more expert sub-networks in deeper layers: shallower layers employ fewer experts (e.g., [0, 0, 2, 4, 4, 6]) while deeper layers employ more.
    - Output at each layer: $\text{PMoE}^{(l)}(\hat{\mathbf{Z}}^l) = \sum_{e=1}^{E_l} g_e^l \odot Y_e^l + \text{FFN}_{shared}^{(l)}(\hat{\mathbf{Z}}^l)$
    - Shared experts capture general patterns, while specialist experts introduced layer by layer handle increasingly task-specific signal features.
    - TopKSoftmax sparse activation is used to reduce computation; an auxiliary loss $\mathcal{L}_{aux}$ ensures balanced expert utilization.

4. **Intra- and Inter-Lobe Pooling (IILP)**:

    - A two-step pooling strategy used during fine-tuning. First, average pooling is applied along the temporal axis: $\widetilde{\mathbf{V}}_d^l = \frac{1}{T}\sum_{t=1}^{T}\mathbf{Z}_{t,d}^{enc,l}$
    - **Intra-lobe pooling**: EEG channels are partitioned by functional lobe (e.g., frontal, occipital), and the average is taken within each lobe: $V_k^l = \frac{1}{|P_k|}\sum_{d \in P_k}\widetilde{\mathbf{V}}_d^l$
    - **Inter-lobe concatenation**: Lobe embeddings are concatenated, then stacked across all encoder layers to form the final representation.
    - **Design Motivation**: This explicitly leverages functional differences across cortical regions, as epilepsy detection and depression classification rely on region-specific signal variations in distinct brain areas.

### Loss & Training
- Pre-training: $\ell_p$-norm reconstruction loss + MoE auxiliary load-balancing loss.
- Pre-training uses the AdamW optimizer with a OneCycle learning rate schedule for approximately 400K steps on 8× RTX 4090 GPUs with bfloat16 mixed precision.
- Fine-tuning: cross-entropy classification loss applied separately on each of the eight downstream datasets.

## Key Experimental Results

### Main Results

| Dataset | Metric (Balanced Acc) | NeurIPT | Prev. SOTA (CBraMod) | Gain |
|--------|------|------|----------|------|
| MentalArithmetic | Balanced Acc | 86.46 | 72.56 | +13.90 |
| Mumtaz2016 | Balanced Acc | 98.03 | 95.60 | +2.43 |
| PhysioP300 | Balanced Acc | 67.31 | 65.02 (EEGPT) | +2.29 |
| Sleep-EDFx | Balanced Acc | 70.47 | 69.17 (EEGPT) | +1.30 |
| BCIC-IV-2A | Balanced Acc | 55.04 | 51.38 | +3.66 |
| TUEV | Balanced Acc | 67.61 | 66.71 | +0.90 |

### Ablation Study

| Configuration | TUEV | MentalArith | Mumtaz | Notes |
|------|---------|------|------|------|
| No components | 51.80 | 73.36 | 91.83 | Baseline |
| 3D PE only | 59.64 | 73.61 | 86.07 | Spatial encoding helps TUEV +7.8 |
| PMoE only | 52.79 | 74.65 | 85.58 | MoE alone provides limited gain |
| IILP only | 59.10 | 73.96 | 91.55 | Pooling helps TUEV +7.3 |
| All combined | 68.94 | 75.69 | 97.07 | Synergistic effect is optimal |

**Pooling strategy comparison** (BCIC-IV-2A dataset):

| Strategy | Balanced Acc | Notes |
|------|---------|------|
| No pooling | 45.14 | — |
| Mean pooling | 37.24 | Regional information is lost |
| IILP | 55.04 | Explicitly leverages brain-region features; +17.8 vs. mean pooling |

### Key Findings
- PMoE's progressive allocation strategy outperforms uniform and decreasing allocation, and is robust to the specific distribution scheme.
- IILP yields especially pronounced improvements on tasks requiring cross-regional differential analysis (epilepsy/depression).
- Motor imagery tasks such as BCIC-IV-2A are highly sensitive to spatial information (performance degrades significantly when 3D PE is removed).
- Different task categories activate different numbers of MoE experts, reflecting the adaptive nature of PMoE.

## Highlights & Insights
- Amplitude-aware masking is a particularly elegant design: by using EEG signal amplitude as the masking criterion, it produces a non-contiguous masking pattern along the time axis, forcing the model to learn global rather than local patterns.
- 3D electrode encoding makes the model natively compatible with different electrode systems, resolving a core bottleneck in cross-dataset transfer.
- Pre-training on 2,000+ hours of data across 8× RTX 4090 GPUs demonstrates the scalability of EEG foundation models.
- Attention score visualizations confirm that the model has learned meaningful inter-regional interaction patterns (e.g., contralateral activation for hand movement tasks).

## Limitations & Future Work
- On the TUAB dataset, NeurIPT does not surpass CBraMod in Cohen's Kappa and AUROC, possibly due to dataset-specific characteristics.
- The pre-training data scale (2,000 hours) remains limited compared to NLP/CV; larger-scale data may yield further improvements.
- The lobe partitioning in IILP is currently fixed; data-driven adaptive brain region segmentation may be more effective.
- Only classification tasks have been evaluated; performance on regression tasks (e.g., emotion scoring) remains unknown.

## Related Work & Insights
- Prior EEG foundation models including BENDR, LaBraM, and CBraMod provided iterative inspiration for the present design.
- The masked reconstruction paradigm from MAE is creatively adapted into the amplitude-aware variant AAMP.
- The success of MoE in large language models inspired the design of PMoE.
- The lobe-based pooling concept is generalizable to other multi-channel physiological signal modalities that require regional feature aggregation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ All four components are innovative, with AAMP being particularly creative; overall, however, the work represents a clever combination of existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across eight datasets with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and rich visualizations, though some details are deferred to the appendix.
- **Value**: ⭐⭐⭐⭐ Advances the state of the art in EEG foundation models and provides a practical approach for the BCI community.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model](janusdna_a_powerful_bi-directional_hybrid_dna_foundation_model.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[NeurIPS 2025\] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)

<!-- RELATED:END -->
