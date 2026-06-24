---
title: >-
  [Paper Note] CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos
description: >-
  [ECCV2024][Medical Imaging][echocardiogram] A reconstruction-based cardiac disease assessment framework, CardiacNet, is proposed. By utilizing a Consistency Deformation Codebook (CDC) and a Consistency Deformation Discriminator (CDD), the model learns structural and motion discrepancies between normal and abnormal echocardiogram videos, achieving state-of-the-art (SOTA) performance in ejection fraction prediction, pulmonary arterial hypertension classification…
tags:
  - "ECCV2024"
  - "Medical Imaging"
  - "echocardiogram"
  - "cardiac disease assessment"
  - "video reconstruction"
  - "vector quantization"
  - "optimal transport"
date: 2026-05-08
content_hash: 58f712fae9d0a192
---

# CardiacNet: Learning to Reconstruct Abnormalities for Cardiac Disease Assessment from Echocardiogram Videos

**Conference**: ECCV2024  
**arXiv**: [2410.20769](https://arxiv.org/abs/2410.20769)  
**Code**: [xmed-lab/CardiacNet](https://github.com/xmed-lab/CardiacNet)  
**Area**: Medical Image  
**Keywords**: echocardiogram, cardiac disease assessment, video reconstruction, vector quantization, optimal transport

## TL;DR

A reconstruction-based cardiac disease assessment framework, CardiacNet, is proposed. By utilizing a Consistency Deformation Codebook (CDC) and a Consistency Deformation Discriminator (CDD), the model learns structural and motion discrepancies between normal and abnormal echocardiogram videos, achieving state-of-the-art (SOTA) performance in ejection fraction prediction, pulmonary arterial hypertension classification, and atrial septal defect classification.

## Background & Motivation

Echocardiogram videos are the most widely used imaging modality in cardiology to assess various cardiac diseases. Existing methods (e.g., EchoNet) primarily leverage global spatio-temporal features for classification/regression, neglecting the periodic motion characteristics of local cardiac structures. The authors observe two types of abnormal patterns in cardiac diseases:

1. **Local structural abnormalities**: Prominent morphological abnormalities visible in a single frame, such as the defect in the atrial septum in atrial septal defect (ASD).
2. **Cardiac motion abnormalities**: Subtle anomalies that are difficult to notice in a single frame but can be detected through abnormal motion of local cardiac structures in videos, such as in pulmonary arterial hypertension (PAH).

Existing classification-based methods focus on global information and struggle to capture local representations. Existing reconstruction-based methods are mainly designed for CT/MRI/X-ray modalities and perform poorly when processing complex spatio-temporal abnormalities in echocardiograms.

## Core Problem

How to simultaneously learn representations of local cardiac structural and motion abnormalities from echocardiogram videos for accurate assessment of various cardiac diseases? 
Core Hypothesis: If a model can accurately reconstruct abnormal samples from normal ones (and vice versa), it can better understand disease characteristics regarding local structural and motion variations.

## Method

### Overall Architecture: Bidirectional Reconstruction Network

CardiacNet comprises two independent networks, $\phi^A(\cdot)$ and $\phi^B(\cdot)$, which are responsible for the reconstruction processes of "normal $\rightarrow$ abnormal" and "abnormal $\rightarrow$ normal", respectively. An input ultrasound video is first partitioned into patches and randomly masked, processed through an encoder-codebook-decoder pipeline to reconstruct a target-class video, and finally reconstructed back to the original class using a reverse network. The model is supervised using an L1 reconstruction loss:

$$\mathcal{L}_{\text{recon}}(X, X^R) = \|X - X^R\|_1$$

### Consistency Deformation Codebook (CDC)

The core idea of CDC is to discretize continuous features via vector quantization to model the deformation patterns of cardiac structures in a regionalized manner:

- **Consistency Deformation Coding**: After the encoder extracts feature $F$, vector quantization is performed using a learnable codebook $\mathcal{Z}=\{Z_k\}_{k=1}^K$. To maintain temporal consistency, learnable positional encodings $\mathcal{P}$ are added along the temporal dimension, and the quantization process utilizes nearest-neighbor matching.
- **Optimal Transport Distance Optimization**: To distinguish the normal and abnormal distributions learned by the two networks, memory banks $\mathcal{M}^A$ and $\mathcal{M}^B$ are constructed to store encoded features. The Wasserstein distance (via Sinkhorn iterations) is used to maximize the transport distance $\mathcal{L}_{\text{OT}}$ between the two distributions, while minimizing the distance $\mathcal{L}_{\text{dis}}$ between quantized features and their corresponding memory bank centroids.
- The codebook is updated via EMA with an update weight of $\omega=0.01$.

### Consistency Deformation Discriminator (CDD)

CDD consists of two discriminators to ensure the spatio-temporal consistency of reconstruction results:

- **Temporal Discriminator** $\eta^T(\cdot)$: Takes the entire video as input to evaluate global temporal consistency.
- **Spatial Discriminator** $\eta^S(\cdot)$: Evaluates spatial consistency frame-by-frame.
- **Local Region Discrimination**: Splits the video into non-overlapping patches and applies the temporal discriminator to each local patch region to ensure reconstruction quality across different cardiac areas.

The total loss is: $\mathcal{L}_{all} = \mathcal{L}_{\text{CDC}} + \mathcal{L}_{\text{CDD}} + \mathcal{L}_{\text{recon}}(X, X^R) + \mathcal{L}_{\text{recon}}(Y, Y^R)$

### Inference Stage

The feature extractor parameters of $\phi^A$ are frozen. The features are flattened and fed into a single-layer Linear network for classification or regression fine-tuning. The input consists of 16 frames with a sampling interval of 4.

## Key Experimental Results

### Datasets

| Dataset | Videos | Task |
|--------|--------|------|
| CardiacNet-PAH (Self-built) | 496 | PAH Classification |
| CardiacNet-ASD (Self-built) | 231 | ASD Classification |
| CAMUS | 500 | EF Regression/Classification |
| EchoNet-Dynamic | 10,300 | EF Regression/Classification |

### PAH Classification (CardiacNet-PAH)

| Method | AUC-ROC | ACC | FID |
|------|---------|-----|-----|
| HiFuse | 84.11% | 83.67% | - |
| EchoNet | 81.63% | 80.95% | - |
| CyTran | 72.69% | 69.38% | 16.40 |
| **CardiacNet** | **89.32%** | **85.71%** | **14.73** |

### ASD Classification (CardiacNet-ASD)

| Method | AUC-ROC | ACC | DICE |
|------|---------|-----|------|
| DeepGuide | 85.02% | 84.79% | - |
| CyTran | 74.35% | 72.41% | 70.21% |
| **CardiacNet** | **91.24%** | **89.63%** | **73.52%** |

### EF Prediction (CAMUS / EchoNet)

| Method | CAMUS MAE | EchoNet MAE |
|------|-----------|-------------|
| HiFuse | 6.34 | 4.08 |
| EchoNet | 6.30 | 4.22 |
| **CardiacNet** | **5.97** | **3.83** |

### Ablation Study (CardiacNet-PAH)

- Enabling CDC alone: AUC increased from 52.37% to 80.27%, FID decreased from 18.90 to 16.82.
- Enabling CDD alone: AUC was only 52.46%, indicating that CDD requires coordination with CDC.
- Contribution of positional encoding in CDC: Classification accuracy improved by ~20%, FID improved by 1.34.
- Contribution of optimal transport in CDC: Classification accuracy improved by ~30%, FID improved by 0.84.
- The combination of global and local discriminators in CDD achieved the best performance.

## Highlights & Insights

1. **Reconstruction-Driven Disease Assessment Paradigm**: Instead of direct classification, the model learns disease representations via normal $\leftrightarrow$ abnormal bidirectional reconstruction, providing a novel perspective.
2. **Exquisite CDC Design**: The combination of vector quantization, temporal positional encoding, and optimal transport distance effectively captures local cardiac structural and motion patterns.
3. **Self-built Benchmark Datasets**: Introduces two new benchmarks, PAH and ASD, filling the vacancy of disease assessment datasets in echocardiogram videos.
4. **Cross-Task Generalization**: Achieves SOTA across three different tasks: classification (PAH, ASD) and regression (EF).

## Limitations & Future Work

1. **Inference Efficiency**: The inference time is 4.523s; although significantly faster than diffusion-based methods, there is still room for optimization.
2. **Limited Data Scale**: Self-built datasets contain only 496 and 231 videos from 4 hospitals, requiring further validation for generalizability.
3. **Computational Overhead of Bidirectional Reconstruction**: Training two independent networks doubles the number of parameters and computational cost.
4. **Lack of Validation on More Disease Types**: Only three cardiac diseases were validated, leaving it unknown whether the method generalizes to more complex multi-class diagnostic scenarios.
5. **Resolution Constraints**: Training inputs are resized to $144 \times 144$ and cropped to $112 \times 112$, which may lead to a loss of fine-grained structural details.

## Related Work & Insights

- **vs EchoNet (R2+1D)**: EchoNet utilizes global spatio-temporal features, neglecting local cardiac structures; CardiacNet learns local abnormal representations via reconstruction, boosting PAH classification AUC by ~8%.
- **vs Existing Reconstruction Methods (CyTran, Wolleb)**: These lack cardiac prior constraints, resulting in poor quality in directly reconstructed ultrasound images; CardiacNet's CDC+CDD introduces structural consistency constraints, achieving a significantly superior FID.
- **vs DiffMIC (Diffusion Model)**: Diffusion models require 1000 denoising steps (1182s) for inference, whereas CardiacNet requires only 4.5s while outperforming them.
- **vs Attention Methods (AGXNet)**: Attention mechanisms rely on classification backbones, which are prone to noise and yield insufficient precision in localizing anomalous regions.

## Insights & Connections

- The bidirectional reconstruction + codebook paradigm can be generalized to other medical imaging tasks requiring the capturing of discrepancies between normal and abnormal distributions.
- Optimizing codebook distributions with optimal transport distance provides valuable baseline concepts for contrastive learning and domain adaptation scenarios.
- The local + global discriminator design serves as a reference for maintaining spatio-temporal consistency in video generation/translation tasks.

## Rating
- **Novelty**: 8/10 — The combination of reconstruction-driven evaluation, CDC, and optimal transport is relatively novel.
- **Experimental Thoroughness**: 8/10 — Four datasets, three tasks, detailed ablations, and visualization analyses.
- **Writing Quality**: 7/10 — Overall clear, though with heavy mathematical notation and some verbose descriptions.
- **Value**: 8/10 — Provides new benchmark datasets and an effective method, holding practical significance for AI-assisted echocardiogram diagnosis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Brain Netflix: Scaling Data to Reconstruct Videos from Brain Signals](brain_netflix_scaling_data_to_reconstruct_videos_from_brain_signals.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](../../NeurIPS2025/medical_imaging/care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)
- [\[CVPR 2025\] Deep Learning-based Assessment of the Relation Between the Third Molar and Mandibular Canal on Panoramic Radiographs using Local, Centralized, and Federated Learning](../../CVPR2025/medical_imaging/deep_learning-based_assessment_of_the_relation_between_the_third_molar_and_mandi.md)
- [\[AAAI 2026\] From Policy to Logic for Efficient and Interpretable Coverage Assessment](../../AAAI2026/medical_imaging/from_policy_to_logic_for_efficient_and_interpretable_coverage_assessment.md)
- [\[ECCV 2024\] Improving Medical Multi-modal Contrastive Learning with Expert Annotations](improving_medical_multi-modal_contrastive_learning_with_expert_annotations.md)

</div>

<!-- RELATED:END -->
