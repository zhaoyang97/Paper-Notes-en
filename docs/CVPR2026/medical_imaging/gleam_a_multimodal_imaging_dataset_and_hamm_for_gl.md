---
title: >-
  [Paper Note] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification
description: >-
  [CVPR 2026][Medical Imaging][Glaucoma classification] This paper presents GLEAM, the first publicly available trimodal glaucoma dataset (SLO fundus photography + peripapillary OCT + visual field deviation maps, 1…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Glaucoma classification"
  - "multimodal fusion"
  - "masked autoencoder"
  - "trimodal dataset"
  - "graph attention"
date: 2026-05-08
content_hash: a9d597d5d94febf7
---

# GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification

**Conference**: CVPR 2026
**arXiv**: [2603.12800](https://arxiv.org/abs/2603.12800)  
**Code**: [Kaggle Dataset](https://kaggle.com/datasets/zhangyiyinge/gleam-dataset)  
**Area**: Medical Imaging / Multimodal Learning / Ophthalmic Imaging
**Keywords**: Glaucoma classification, multimodal fusion, masked autoencoder, trimodal dataset, graph attention

## TL;DR

This paper presents GLEAM, the first publicly available trimodal glaucoma dataset (SLO fundus photography + peripapillary OCT + visual field deviation maps, 1,200 cases, four-stage annotation), along with HAMM, a CNN-based hierarchical attention masked modeling framework. HAMM achieves cross-modal fusion via clinically inspired multi-head modality gating and relational graph attention, attaining a four-class classification accuracy of 81.08%.

## Background & Motivation

**Background**: Glaucoma is one of the leading causes of irreversible blindness worldwide, affecting approximately 70 million individuals. Clinical diagnosis relies on the integrated interpretation of multiple examinations: fundus imaging for optic disc morphology, OCT for retinal nerve fiber layer (RNFL) thickness measurement, and visual field testing for functional impairment assessment. Computer-aided diagnosis (CAD) systems have made steady progress over the past decade.

**Limitations of Prior Work**: Existing public datasets suffer from three key deficiencies: (1) most are unimodal (fundus or OCT only), lacking modality diversity; (2) classification granularity is coarse, typically limited to binary normal/glaucoma labels, insufficient to support staging-based treatment; (3) sample sizes are limited or datasets are not publicly released. Existing multimodal datasets such as GAMMA contain only 200 bimodal cases.

**Key Challenge**: Clinicians routinely integrate findings from three distinct examinations for cross-validation and holistic judgment, yet there is a lack of corresponding datasets and fusion frameworks to support automated diagnostic research.

**Goal**: (1) Construct the first publicly available trimodal, four-stage annotated, high-quality glaucoma dataset; (2) design an effective self-supervised multimodal fusion framework to fully exploit complementary inter-modal information.

**Key Insight**: Emulating the clinical reasoning of ophthalmologists—first assessing the quality and reliability of each modality, then cross-validating structural-functional consistency.

**Core Idea**: Multi-head gating mechanisms simulate clinician assessment of modality reliability; relational graph attention simulates cross-modal cross-validation; both are embedded within a CNN masked autoencoder for self-supervised pretraining.

## Method

### Overall Architecture

HAMM adopts a two-stage training strategy. **Stage 1 (Pretraining)**: Inputs from three modalities are randomly masked (masking ratio 0.7) and processed by three parallel ResNet-50 encoders—each with MCGA modules embedded at every layer for hierarchical cross-modal fusion—to extract features. Lightweight depthwise separable convolutional decoders reconstruct the masked regions, with MSE reconstruction loss as the training objective. **Stage 2 (Fine-tuning)**: The decoders are discarded; the pretrained encoders are retained, and trimodal features are concatenated and passed through a GAP + two-layer fully connected classification head for four-class prediction, trained with cross-entropy loss.

### Key Designs

1. **Multimodal Channel Graph Attention (MCGA) Module**:

    - **Function**: Enables hierarchical cross-modal information interaction at each downsampling layer of the encoder.
    - **Mechanism**: Operates in three steps: (a) GAP, GMP, and GeM pooling are applied to each modality's feature maps and concatenated, then projected through a fully connected layer to produce modality embeddings $v_k$; (b) a multi-head gating mechanism $\hat{v}_k = v_k \odot \frac{1}{H}\sum_{h=1}^{H} g^{(h)}(v_k)$ assigns adaptive reliability weights to each modality, simulating multiple ophthalmologists independently assessing modality quality; (c) a relational graph attention network captures inter-modal dependencies and models structural-functional consistency via relation-type embeddings $R_{r_{ij}}^{(h)}$.
    - **Design Motivation**: Mimics the clinical reasoning of ophthalmologists—first evaluating the reliability of each examination result, then cross-validating consistency across modalities. Hierarchical fusion (applied at every layer) outperforms late fusion, with experimental results showing accuracy improving from 78.50% to 79.17%.

2. **CNN Masked Autoencoder Pretraining**:

    - **Function**: Learns robust cross-modal representations by reconstructing masked regions.
    - **Mechanism**: For each modality, 70% of pixel regions are randomly masked; the encoder infers masked content from visible regions and information from other modalities. The decoder adopts a lightweight design (depthwise separable convolution + bilinear interpolation upsampling) with skip connections fusing features from each encoder layer. Training loss is MSE computed exclusively over masked pixels: $\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}\sum_{k \in K}\sum_{p=1}^{P}(s_i^k(p) - \hat{s}_i^k(p))^2$
    - **Design Motivation**: Ophthalmic images frequently suffer from information loss due to artifacts, blur, and anatomical occlusion; masked modeling naturally simulates these scenarios. CNN architectures (vs. Transformer-based MAE) are better suited to small-sample medical data due to visual inductive biases and reduced susceptibility to overfitting.

3. **GLEAM Trimodal Dataset**:

    - **Function**: Establishes the first publicly available trimodal, four-stage annotated glaucoma dataset.
    - **Mechanism**: Retrospectively collected 1,200 paired cases (841 patients, aged 8–90 years, mean 55.4±16.7) from Shenyang Fourth People's Hospital, comprising SLO fundus images (Optos ultra-widefield), peripapillary OCT (Heidelberg Spectralis), and visual field PD maps (Zeiss perimeter). Four stages are annotated: normal (NG, 600 cases), early (EaG, 200 cases), intermediate (InG, 200 cases), and advanced (AdG, 200 cases), stratified based on EMR diagnoses and MD values (early: MD > −6 dB; intermediate: −12 dB ≤ MD ≤ −6 dB; advanced: MD < −12 dB).
    - **Design Motivation**: Fills a critical gap in the field—existing datasets are either uni/bimodal or restricted to binary classification, precluding multimodal staging research. Three senior ophthalmologists independently annotated all cases with consensus review; inter-annotator Cohen's Kappa > 95.5% and intra-annotator Kappa > 97.4%.

### Loss & Training

- **Pretraining**: MSE reconstruction loss (computed only on masked pixels), 20 epochs, learning rate $1 \times 10^{-5}$, batch size 8.
- **Fine-tuning**: Cross-entropy classification loss, learning rate $3 \times 10^{-6}$, batch size 16, early stopping (10 epochs without improvement in validation loss).
- **Data Augmentation**: SLO (random cropping / color jitter / vertical flip), OCT (color jitter), VF (vertical flip); synchronized horizontal flipping across all three modalities to preserve anatomical consistency.
- Results averaged over five independent training runs to ensure statistical reliability.

## Key Experimental Results

### Main Results

| Method | Pretraining | Acc (%) | F1 (%) | AUROC (%) | Kappa |
|--------|-------------|---------|--------|-----------|-------|
| ResNet50 | — | 76.75±1.47 | 66.84±2.60 | 89.95±0.27 | 85.88 |
| ResNet50 | TL | 77.67±0.86 | 70.19±0.93 | 92.14±1.81 | 87.00 |
| ViT-S | TL | 77.75±1.52 | 69.62±3.79 | 91.79±0.48 | 88.03 |
| ConvNeXt-T | TL | 79.00±0.76 | 71.58±1.32 | 91.87±0.77 | 87.83 |
| MHCA | TL | 78.16±0.63 | 69.97±3.20 | 92.28±0.27 | 87.14 |
| DRIFA-Net | TL | 77.83±0.86 | 69.70±1.96 | 92.42±0.10 | 86.75 |
| Corolla | SCL | 78.67±0.74 | 72.87±1.21 | 92.39±0.55 | 88.50 |
| ETSCL | SCL | 79.08±0.80 | 72.52±2.11 | 92.73±0.32 | 87.31 |
| MultiMAE | SSL | 78.00±0.18 | 69.02±2.18 | 90.64±0.26 | 86.98 |
| UrFound | SSL | 78.67±0.35 | 70.67±1.46 | 92.49±0.44 | 87.86 |
| **HAMM (Ours)** | **SSL** | **81.08±0.63** | **75.90±0.80** | **93.03±0.26** | **90.07** |

HAMM outperforms the strongest baseline (ETSCL) by: Acc +2.00%, F1 +3.38%, AUROC +0.30%, Kappa +2.76.

### Ablation Study

| MCGA | Pretraining | Acc (%) | F1 (%) | AUROC (%) | Kappa |
|------|-------------|---------|--------|-----------|-------|
| ✗ | ✗ | 77.67 | 70.19 | 92.14 | 87.00 |
| ✓ | ✗ | 79.17 | 71.93 | 92.89 | 89.52 |
| ✗ | ✓ | 79.67 | 73.68 | 92.83 | 89.57 |
| ✓ | ✓ | **81.08** | **75.90** | **93.03** | **90.07** |

Modality combination ablation:

| Modality | Acc (%) | F1 (%) | AUROC (%) | Acc-EaG (%) |
|----------|---------|--------|-----------|-------------|
| SLO | 60.25 | 37.25 | 74.72 | 3.00 |
| OCT | 61.75 | 42.39 | 76.70 | 8.00 |
| VF | 74.25 | 59.85 | 90.42 | 6.00 |
| SLO+OCT | 64.42 | 46.47 | 67.22 | 11.00 |
| SLO+VF | 77.67 | 68.36 | 91.87 | 26.00 |
| OCT+VF | 77.08 | 67.38 | 92.24 | 22.50 |
| **SLO+OCT+VF** | **81.08** | **75.90** | **93.03** | **51.50** |

External validation (GAMMA dataset):

| Method | Ensemble | Kappa |
|--------|----------|-------|
| SmartDSP | ✓ | 85.49 |
| COROLLA | ✓ | 85.50 |
| GeCoM-Net | ✓ | 88.10 |
| ETSCL (+ extra modality) | ✗ | 88.44 |
| **HAMM (Ours)** | ✗ | **87.59** |
| **HAMM (Ours)** | ✓ | **89.35** |

### Key Findings

- Early glaucoma (EaG) classification is the most challenging: single modalities are nearly incapable of detection (VF achieves only 6.0%), while trimodal fusion raises this to 51.50%, demonstrating that multimodal complementarity is essential for early diagnosis.
- VF is the most discriminative single modality (Acc 74.25%), yet fails on early-stage cases; SLO and OCT individually perform poorly but contribute substantially to early- and intermediate-stage classification when combined.
- A masking ratio of 0.7 is the optimal configuration (with 20 pretraining epochs); at this ratio, the model is compelled to rely more heavily on cross-modal information for inference.
- HAMM also outperforms comparison methods under missing-modality conditions (Acc 74.59% vs. UrFound 72.48%), demonstrating robustness.

## Highlights & Insights

- The dataset contribution itself is of significant importance: GLEAM is the first publicly available trimodal, four-stage annotated glaucoma dataset with exceptionally high annotation quality (Kappa > 95.5%).
- The clinically inspired design of the MCGA module is particularly elegant—multi-head gating simulates independent assessment of modality reliability by multiple clinicians, while graph attention simulates cross-modal cross-validation.
- CNN-based MAE is applied to multimodal medical tasks for the first time, demonstrating greater suitability for small-sample scenarios compared to Transformer-based MAE.
- Trimodal fusion completely eliminates cross-class misclassification between NG and AdG (as verified by the confusion matrix), which has important implications for clinical safety.

## Limitations & Future Work

- Data are sourced from a single center (Shenyang Fourth People's Hospital); generalizability requires multi-center validation.
- Glaucoma subtypes (primary open-angle, normal-tension, angle-closure, etc.) are not distinguished, despite differing pathological features and spatial damage patterns across subtypes.
- The current framework addresses four-class classification; continuous severity estimation (e.g., predicting MD values) may be more clinically granular and practical, potentially benefiting from ordinal regression losses.
- Early-stage accuracy of 51.50%, while superior to baselines, leaves room for improvement; larger-scale data or dedicated class imbalance handling strategies may be warranted.
- Longitudinal follow-up data analysis (disease progression prediction) is not addressed.

## Related Work & Insights

- **vs. RETFound / EyeCLIP**: These are unimodal (fundus-only) self-supervised pretraining approaches that do not cover OCT or visual field data; HAMM explicitly models trimodal interactions.
- **vs. MultiMAE**: A Transformer-based multimodal masked modeling approach prone to overfitting on small-scale medical data (Acc 78.00% on GLEAM); HAMM employs a CNN architecture with MCGA, better suited to limited data regimes.
- **vs. MHCA / DRIFA-Net**: MHCA and DRIFA-Net have parameter counts of 248M and 931M, respectively; HAMM achieves 237M parameters with only 12.68G FLOPs (vs. DRIFA-Net's 88.48G), offering higher efficiency alongside superior performance.
- **vs. GAMMA dataset**: GAMMA contains only 200 bimodal cases with two/three-class labels; GLEAM offers 1,200 trimodal cases with four-class labels, representing a significant improvement in both scale and annotation granularity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First trimodal glaucoma dataset + clinically inspired MCGA module design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive coverage including main experiments, modality ablation, component ablation, masking ratio analysis, external validation, missing-modality robustness, and reliability analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear methodological exposition, systematic experimental design, and well-articulated clinical motivation.
- **Value**: ⭐⭐⭐⭐ The dataset fills a critical gap in the field and directly advances ophthalmic AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EI: Early Intervention for Multimodal Imaging based Disease Recognition](ei_early_intervention_for_multimodal_imaging_based_disease_recognition.md)
- [\[CVPR 2026\] Multimodal Classification of Radiation-Induced Contrast Enhancements and Tumor Recurrence Using Deep Learning](multimodal_classification_of_radiation-induced_contrast_enhancements_and_tumor_r.md)
- [\[CVPR 2026\] Transformer-Based Multi-Region Segmentation and Radiomic Analysis of HR-pQCT Imaging for Osteoporosis Classification](transformer-based_multi-region_segmentation_and_radiomic_analysis_of_hr-pqct_ima.md)
- [\[AAAI 2026\] EgoEMS: A High-Fidelity Multimodal Egocentric Dataset for Cognitive Assistance in Emergency Medical Services](../../AAAI2026/medical_imaging/egoems_a_high-fidelity_multimodal_egocentric_dataset_for_cognitive_assistance_in.md)
- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](../../ICLR2026/medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)

</div>

<!-- RELATED:END -->
