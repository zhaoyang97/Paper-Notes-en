---
title: >-
  [Paper Note] EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping
description: >-
  [ICML 2025][Medical Imaging][EEG-language models] This paper pioneers the EEG-Language Model (ELM). Trained on 15,000 EEG recordings and clinical reports, ELM integrates time-series cropping, text segmentation, and multi-instance learning strategies. It achieves zero-shot EEG classification and cross-modal retrieval for the first time, significantly outperforming EEG-only self-supervised methods in low-label scenarios.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "EEG-language models"
  - "multimodal alignment"
  - "multi-instance learning"
  - "clinical phenotyping"
  - "zero-shot classification"
  - "pathology detection"
  - "self-supervised learning"
date: 2026-05-08
content_hash: 7b71f6e5c622d229
---

# EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping

**Conference**: ICML 2025  
**arXiv**: [2409.07480](https://arxiv.org/abs/2409.07480)  
**Code**: Not released  
**Area**: EEG Signal Analysis / Multimodal Learning  
**Keywords**: EEG-language models, multimodal alignment, multi-instance learning, clinical phenotyping, zero-shot classification, pathology detection, self-supervised learning

## TL;DR

This paper pioneers the EEG-Language Model (ELM). Trained on 15,000 EEG recordings and clinical reports, ELM integrates time-series cropping, text segmentation, and multi-instance learning strategies. It achieves zero-shot EEG classification and cross-modal retrieval for the first time, significantly outperforming EEG-only self-supervised methods in low-label scenarios.

## Background & Motivation

The field of medical neuroimaging (particularly electroencephalography, EEG) severely lags behind other domains in leveraging deep learning. Although EEG is widely used in pathological detection such as epilepsy and sleep disorders, available labeled data is extremely scarce.

Existing challenges include:

**Scarce Labeled Data**: Self-supervised learning (SSL) can scale up training using unlabeled data, but existing methods are limited by the difficulty of designing data augmentations and low signal-to-noise ratios.

**Unexploited Cross-Modal Information**: Computer vision and radiology have demonstrated that natural language can significantly enhance representation learning, but EEG-language pretraining remains unexplored.

**Heterogeneity of EEG Reports**: Clinical reports typically span multiple paragraphs, contain information irrelevant to downstream tasks, and lack temporal annotations.

**Inter-Modal Misalignment**: EEG is a long-duration time series, while reports are multi-paragraph texts, leading to a granularity mismatch between the two modalities.

Core motivation: To introduce natural language as a pretraining signal for EEG representation learning, while simultaneously addressing the heterogeneity and misalignment of EEG-text pairs.

## Method

### Overall Architecture

The ELM framework consists of the following core components (Figure 1):

- **EEG Encoder $f_e$**: A residual convolutional neural network that projects EEG segments into low-dimensional vectors
- **Language Encoder $f_l$**: A pretrained MedCPT model (with frozen weights) that encodes clinical texts
- **Projectors $g_e, g_l$**: Projecting both modalities into a shared latent space

### Key Designs

#### 1. Subunit Alignment Strategy

To address the challenges of long time series and multi-paragraph reports, the authors propose:
- **Time-Series Cropping**: Cropping EEG recordings into multiple non-overlapping segments (60 seconds / 20 seconds)
- **Text Segmentation**: Using regular expressions to split reports into four categories based on headings: clinical history, recording description, medication, and clinical interpretation

#### 2. Three Alignment Strategies

- **ELM$_{e,l}$** (CLIP-like): EEG and text are mapped to a new shared space, optimized using the InfoNCE loss
- **ELM$_l$** (M-FLAG-like): EEG is projected directly into the output space of the language model, and the loss includes an alignment loss and an orthogonality loss:

$$\mathcal{L}_{total} = \mathcal{L}_{align} + \mathcal{L}_{orth}$$

Where $\mathcal{L}_{align} = \|\hat{\mathbf{e}} - \hat{\mathbf{l}}\|_2^2$ minimizes embedding discrepancy, and $\mathcal{L}_{orth}$ promotes independence among dimensions of the EEG embeddings.

#### 3. Multi-Instance Learning Extension (ELM-MIL)

Core innovation — relaxing the assumption of strict alignment for each EEG-text pair:

- For each text sample, multiple positive EEG segments are sampled to approximate the $P(e|l)$ distribution
- For each EEG segment, multiple text paragraphs are sampled to approximate the $P(l|e)$ distribution
- Bidirectional alignment is used to approximate $P(e,l)$

Extending the InfoNCE loss to a multi-instance formulation:

$$\mathcal{L}^{e|l} = -\frac{1}{B_l}\sum_{k=1}^{B_l} \log \frac{\frac{1}{|P_k|}\sum_{j \in P_k} \exp(s^{e2l}_{j,k}/\tau)}{\sum_{j=1}^{B_e} \exp(s^{e2l}_{j,k}/\tau)}$$

The temperature parameter is set to $\tau = 0.3$, sampling at most $N=32$ EEG segments and $M=8$ text sections per subject.

#### 4. Text Preprocessing

- Reports are categorized into four sections by headings: clinical history, record description, medication information, and clinical interpretation/impression
- Irrelevant information (e.g., EEG system info, technical issues, disclaimers) is filtered out
- Local Llama-3 8B is used to generate offline single-sentence summaries as a supplement

### Loss & Training

ELM-MIL final loss:

$$\mathcal{L}^{e,l} = \frac{1}{2}(\mathcal{L}^{e|l} + \mathcal{L}^{l|e})$$

## Key Experimental Results

### Main Results: TUAB Pathology Detection (Linear Probing)

| Method | ZS(BAcc) | 1%(BAcc) | 10%(BAcc) | 100%(BAcc) | 1%(AUROC) | 100%(AUROC) |
|------|----------|----------|-----------|------------|-----------|-------------|
| Supervised | - | 71.36 | 81.06 | 84.13 | 79.87 | 91.83 |
| TS (EEG-only) | - | 74.99 | 82.16 | 84.10 | 82.51 | 91.50 |
| **ELM-MIL e,l** | **84.31** | 83.10 | 84.21 | **87.11** | **91.56** | **93.91** |
| ELM-MIL e\|l | 79.10 | **83.71** | 84.37 | 85.65 | 92.37 | 93.65 |

Key Findings:
- **Zero-shot Classification**: ELM-MIL $e,l$ achieves 84.31% balanced accuracy, marking the first realization of zero-shot classification in this field
- **Label-Efficiency Advantage**: With only 1% of labeled data, the AUROC reaches 91.56%, a gain of approximately 9 percentage points over the best EEG-only method (TS 82.51%)
- **100% Labeling**: AUROC of 93.91%, significantly surpassing the fully supervised method (91.83%)

### Cross-Dataset Generalization (NMT Dataset)

| Method | 1%(AUROC) | 10%(AUROC) | 100%(AUROC) |
|------|-----------|------------|-------------|
| TS | 64.90 | 81.36 | 87.08 |
| **ELM-MIL e,l** | **76.10** | **88.98** | **90.25** |

On the NMT dataset, which originates from Pakistan and features different acquisition equipment, ELM-MIL using only 1% of the annotations outperforms TS using 10% of the annotations.

### Retrieval Performance

For EEG $\leftrightarrow$ report retrieval across 437 patients, ELM-MIL significantly outperforms alternative methods in terms of Top-K retrieval accuracy, demonstrating successful cross-modal alignment generalization.

## Highlights & Insights

1. **Pioneering EEG-Language Pretraining**: Bridges the gap in cross-modal representation learning between functional brain data and natural language text.
2. **Elegant Design of Multi-Instance Learning**: Relaxes the strict alignment assumption, enabling better handling of inconsistent correlations within EEG-text pairs.
3. **Importance of Text Types**: Integrating multiple text clusters (history + description + interpretation + medication) yields the best performance, indicating that diverse information sources provide complementary knowledge.
4. **Additional Benefits of Subunit Alignment**: Even when reports are randomly shuffled, the subunit alignment strategy still facilitates inter-subject information encoding, demonstrating its intrinsic advantages.
5. **Clinical Practicality**: No clinical reports are required during downstream training and inference phases, making ELM fully compatible with standard clinical EEG practice.

## Limitations & Future Work

1. The dataset scale is still much smaller than those in radiology and computer vision.
2. The report texts exhibit high heterogeneity, and the information quality varies significantly.
3. Freezing the language encoder may limit the potential upper bound of cross-modal alignment.
4. Evaluation is primarily focused on pathology detection and has not yet been extended to fine-grained clinical tasks.

## Related Work & Insights

- **EEG Self-Supervised Learning**: BYOL, VICReg, ContraWR, RP, TS, CPC
- **Medical Multimodal Language Modeling**: M-FLAG, CLIP, BiomedCLIP
- **Multi-Instance Learning**: MIL-NCE (video-text alignment)

## Rating

⭐⭐⭐⭐ — Pioneering introduction of multimodal language modeling to the EEG domain, with an elegant and highly effective MIL expansion design. The experiments comprehensively cover zero-shot, low-label, and cross-dataset scenarios. It holds significant clinical utility, especially with massive performance gains in extremely low-labeled scenarios. The limitations lie in the dataset scale and the limited diversity of downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining](from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)
- [\[ICLR 2026\] CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical EEG Interpretation](../../ICLR2026/medical_imaging/cerebragloss_instruction-tuning_a_large_vision-language_model_for_fine-grained_c.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[CVPR 2026\] From Panel to Pixel: Zoom-In Vision-Language Pretraining from Biomedical Scientific Literature](../../CVPR2026/medical_imaging/from_panel_to_pixel_zoom-in_vision-language_pretraining_from_biomedical_scientif.md)
- [\[CVPR 2025\] A Semi-Supervised Framework for Breast Ultrasound Segmentation with Training-Free Pseudo-Label Generation and Label Refinement](../../CVPR2025/medical_imaging/a_semi-supervised_framework_for_breast_ultrasound_segmentation_with_training-fre.md)

</div>

<!-- RELATED:END -->
