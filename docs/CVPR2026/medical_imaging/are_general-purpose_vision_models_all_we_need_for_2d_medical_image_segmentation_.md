---
title: >-
  [Paper Note] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study
description: >-
  [CVPR 2026][Medical Imaging][medical image segmentation] Through standardized comparative experiments on 11 architectures across three heterogeneous medical datasets, it is demonstrated that General-Purpose Vision Models (GP-VMs) can outperform most Specialized Medical segmentation Architectures (SMAs). Furthermore, XAI analysis indicates that GP-VMs capture clinically relev
tags:
  - CVPR 2026
  - Medical Imaging
  - medical image segmentation
  - general-purpose vision models
  - empirical study
  - benchmarking
  - Grad-CAM
date: 2026-05-08
content_hash: 36c487ae3c74cc90
---
# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study

**Conference**: CVPR 2026  
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)  
**Code**: [GitHub](https://github.com/)  
**Area**: Medical Image Segmentation  
**Keywords**: medical image segmentation, general-purpose vision models, empirical study, benchmarking, Grad-CAM

## TL;DR

Through standardized comparative experiments on 11 architectures across three heterogeneous medical datasets, it is demonstrated that General-Purpose Vision Models (GP-VMs) can outperform most Specialized Medical segmentation Architectures (SMAs). Furthermore, XAI analysis indicates that GP-VMs capture clinically relevant structures without the need for domain-specific designs.

## Background & Motivation

Medical Image Segmentation (MIS) is a fundamental component of computer-aided diagnosis and clinical decision support systems. Over the past decade, numerous specialized architectures targeting specific medical imaging challenges (low contrast, small anatomical structures, limited annotated data) have emerged, such as HiFormer, MISSFormer, U-KAN, and Swin-UMamba. Simultaneously, General-Purpose Vision Models (GP-VMs) have made significant progress on natural images.

However, existing research lacks **standardized controlled experiments** to fairly compare the two types of models. Different papers use varying datasets, preprocessing pipelines, augmentation strategies, and evaluation protocols, leading to performance differences that may stem from experimental design rather than the merits of the architectures themselves.

The core problem of this paper is: **Does the medical segmentation task truly require specialized architectures, or can general-purpose vision models achieve or even exceed their performance?**

## Method

### Overall Architecture

This paper does not propose a new architecture but instead establishes a standardized benchmarking framework to address whether 2D medical image segmentation requires specialized architectures. The approach involves conducting a fair head-to-head comparison of 11 architectures (5 SMAs + 6 GP-VMs) across three medical datasets with distinct imaging modalities under a unified preprocessing, training, and evaluation protocol. Additionally, Grad-CAM is utilized to examine whether the regions of focus for both model types are clinically relevant.

### Key Designs

**1. Model Selection: Competing at Comparable Scales**

To fairly answer "Specialized vs. General," the compared models must be equivalent in scale and visibility. For SMAs, the selection includes U-Net (31M, CNN), HiFormer-B (26M, CNN-ViT hybrid), MISSFormer (42M, Transformer), Swin-UMamba (60M, State Space hybrid), and U-KAN-L (25M, CNN+KAN). For GP-VMs, the selection includes SegFormer-B3, SegNeXt-L, VWFormer (with MiT-B3 and ConvNeXt-S backbones), InternImage-T+UPerHead, and TransNeXt-Tiny+UPerHead (all in the 47–58M range). Selection criteria focused on architectural diversity, comparable computational scale, academic visibility, and code availability to avoid outdated or mismatched models.

**2. Heterogeneous Datasets: Covering Diverse Imaging Scenarios**

To ensure conclusions are not biased by a single modality, three datasets with significant differences were selected: ISIC'18 (RGB dermoscopy, binary lesion segmentation, 3565 images); BKAI-IGH NeoPolyp Small (RGB endoscopy, three-class polyp segmentation, 945 images); and CAMUS (grayscale echocardiography, four-class cardiac region segmentation, 1996 images). Variability in modality (RGB/grayscale), task (binary/multi-class), and data volume tests the cross-scenario robustness of the conclusions.

**3. Standardized Training Protocol: Controlling Confounding Factors**

Since performance differences in literature often arise from preprocessing and training settings rather than architectures, this study enforces a single protocol: ImageNet-pretrained encoders, $512 \times 512$ input, AdamW + REX scheduler, batch size 8, learning rate search within $\{10^{-4}, 5 \times 10^{-5}, 10^{-5}\}$, 5-fold cross-validation, 150 epochs, and unified early stopping. Data augmentation is customized by dataset but remains consistent for all models. This ensures final performance differences are cleanly attributed to the architectures themselves.

### Loss & Training

- ISIC'18 uses Binary Cross-Entropy loss.
- Multi-class tasks (NeoPolyp, CAMUS) use Cross-Entropy loss.
- Mixed-precision training (except for SegNeXt due to instability).
- Evaluation metrics: mIoU, mDSC, mRec, mPrec (excluding background class, global micro-average).
- Interpretability analysis performed via Grad-CAM visualization.

## Key Experimental Results

### Main Results

| Model | Type | NeoPolyp mDSC | CAMUS mDSC | ISIC'18 mDSC | Avg mDSC |
|------|------|:---:|:---:|:---:|:---:|
| VW-MiT | GP-VM | 89.7 | 91.4 | 91.8 | **91.0** |
| VW-Conv | GP-VM | 89.6 | 91.3 | 91.8 | **90.9** |
| TransNeXt | GP-VM | 89.4 | 91.7 | 91.7 | **90.9** |
| InternImage | GP-VM | 89.6 | 91.2 | 91.5 | **90.8** |
| SegFormer | GP-VM | 89.1 | 91.4 | 91.7 | **90.7** |
| SegNeXt | GP-VM | 89.2 | 91.3 | 91.5 | **90.7** |
| SU-Mamba | SMA | 88.9 | 91.3 | 91.3 | 90.5 |
| HiFormer | SMA | 84.6 | 90.8 | 91.0 | 88.8 |
| U-KAN | SMA | 82.5 | 90.5 | 90.6 | 87.9 |
| MISSFormer | SMA | 82.9 | 90.4 | 90.3 | 87.9 |
| U-Net | SMA | 83.3 | 89.1 | 89.3 | 87.2 |

### Ablation Study

| Metric Dimension | GP-VMs Performance | SMAs Performance | Gap Analysis |
|----------|:---:|:---:|------|
| NeoPolyp Non-neoplastic (C1) | 62.4-66.1% | 34.9-59.2% | Largest lead for GP-VMs (7+ pp) |
| CAMUS LV Wall (C2) | 87.7-88.4% | 85.6-88.3% | Smaller gap (≈1-2 pp) |
| ISIC'18 Overall | 91.5-91.8% | 89.3-91.3% | Stable advantage for GP-VMs |
| Grad-CAM Interpretability | Captures clinical regions | Similar performance | GP-VMs perform well without specific designs |

### Key Findings

- **GP-VMs Lead Overall**: In terms of average mDSC ranking, the top 6 models are all GP-VMs.
- **Dataset-Dependent Performance Gaps**: The largest gap occurs in NeoPolyp (up to 7+ pp between GP-VMs and SMAs), while the gap in CAMUS is smaller (≈1-2 pp).
- **SU-Mamba is the Strongest SMA**: It consistently outperforms other SMAs but remains slightly below the best GP-VMs.
- **GP-VMs Exhibit Good XAI**: Grad-CAM analysis shows that GP-VMs can capture clinically relevant structures without domain-specific design modifications.

## Highlights & Insights

- **Rigorous Experimental Design**: Comparing models under identical preprocessing, augmentation, and optimization settings eliminates confounding factors, which is relatively rare in the MIS field.
- **Practical Implications**: Before introducing new specialized architectures, researchers should systematically evaluate the performance of existing GP-VMs.
- **Resource Efficiency Perspective**: Given that GP-VMs are already competitive, research efforts could shift toward data curation, optimization of training protocols, and OOD generalization assessments.

## Limitations & Future Work

- Only three 2D datasets were covered, which may not fully represent the diversity of clinical imaging (e.g., 3D, CT, MRI).
- Parameter counts for some models are not perfectly comparable (e.g., U-KAN lacks ImageNet pretraining).
- OOD generalization was not evaluated (e.g., testing NeoPolyp-trained models on Kvasir-SEG).
- 3D medical image segmentation and semi-supervised/few-shot scenarios were not addressed.

## Related Work & Insights

- **SAM in Medical Imaging**: While SAM has been adapted for medical contexts, this paper notes that even without SAM-scale foundation models, standard GP-VMs + fine-tuning are highly effective.
- **Comparison with Moglia et al. (2026) Review**: That review also found general-purpose models perform excellently but relied on original results from various papers. This study provides more reliable evidence through a standardized protocol.
- **Inspiration for the Medical Imaging Community**: When resources are limited, priority should be given to mature GP-VMs (e.g., InternImage, VWFormer) rather than designing specialized architectures from scratch.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Evaluation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?](../../CVPR2025/medical_imaging/are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[CVPR 2026\] SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation](sd_fsmis_adapting_stable_diffusion_for_few_shot_medical_image_segmentation.md)
- [\[CVPR 2026\] Delving Aleatoric Uncertainty in Medical Image Segmentation via Vision Foundation Models](delving_aleatoric_uncertainty_in_medical_image_segmentation_via_vision_foundatio.md)
- [\[CVPR 2026\] Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification](revisiting_2d_foundation_models_for_scalable_3d_medical_image_classification.md)
- [\[CVPR 2026\] Building Robust Vision Encoders for Cross-Dataset Evaluation in Immunofluorescent Microscopy](building_robust_vision_encoders_for_cross-dataset_evaluation_in_immunofluorescen.md)

</div>

<!-- RELATED:END -->
