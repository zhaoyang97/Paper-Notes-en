---
title: >-
  [Paper Note] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study
description: >-
  [CVPR 2026][Medical Imaging][general-purpose vision models] By evaluating 11 models on three heterogeneous medical datasets under a unified training protocol…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "general-purpose vision models"
  - "medical image segmentation"
  - "empirical study"
  - "cross-dataset evaluation"
  - "Grad-CAM"
date: 2026-05-08
content_hash: d14c218dbd2dff5c
---

# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study

**Conference**: CVPR 2026
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)  
**Code**: Available (GitHub)  
**Area**: Medical Image Segmentation
**Keywords**: general-purpose vision models, medical image segmentation, empirical study, cross-dataset evaluation, Grad-CAM

## TL;DR

By evaluating 11 models on three heterogeneous medical datasets under a unified training protocol, this study demonstrates that general-purpose vision models (GP-VMs) systematically outperform most specialized medical segmentation architectures (SMAs) under standardized conditions, challenging the prevailing assumption that medical image segmentation inherently requires domain-specific architectures.

## Background & Motivation

Medical image segmentation (MIS) is a core component of computer-aided diagnosis. Over the past decade, numerous specialized architectures tailored to medical imaging challenges—such as low contrast, small anatomical structures, and limited annotations—have been proposed (e.g., U-Net and its variants). Meanwhile, general-purpose vision models (e.g., SegFormer, InternImage) have demonstrated strong generalization on natural images.

However, a fundamental question remains underexplored: **does medical image segmentation truly require specialized architectures, or are general-purpose vision models already sufficient?** Existing comparative studies lack standardized evaluation protocols; differences in datasets, preprocessing, augmentation strategies, and training configurations across papers mean that observed performance gaps may reflect experimental design choices rather than architectural merits.

This paper provides the first systematic answer to this question through a strictly controlled cross-dataset empirical study.

## Method

### Overall Architecture

Rather than proposing a new architecture, this paper constructs a **standardized benchmarking framework** that fairly compares two broad model categories under unified conditions:

- **Specialized Medical Segmentation Architectures (SMAs)**: 5 models, including U-Net, HiFormer-B, MISSFormer, U-KAN-L, and Swin-UMamba
- **General-Purpose Vision Models (GP-VMs)**: 6 models, divided into semantic segmentation architectures (SegFormer-B3, SegNeXt-L, VWFormer×2) and vision backbones with UPerHead decoder (InternImage-T, TransNeXt-Tiny)

### Key Designs

1. **Standardized Training Protocol**: All models share a unified configuration—ImageNet-pretrained encoders, $512 \times 512$ input resolution, AdamW optimizer with REX learning rate scheduler, batch size 8, identical augmentation pipelines per dataset. Learning rates are selected from $\{10^{-4}, 5 \times 10^{-5}, 10^{-5}\}$, with 150-epoch training and a consistent early stopping criterion.

2. **Heterogeneous Dataset Coverage**: Three medical datasets are selected to span diverse imaging modalities, label structures, and data characteristics:
    - **ISIC'18**: Dermoscopy RGB images, binary lesion segmentation (3,565 images), characterized by irregular boundaries
    - **NeoPolyp**: Endoscopy RGB images, three-class polyp segmentation (945 images), characterized by high intra-class variability
    - **CAMUS**: Echocardiography grayscale images, four-class cardiac region segmentation (1,996 images), characterized by heavy noise

3. **Rigorous Data Leakage Prevention**: Duplicate and near-duplicate images are filtered using raw-byte and perceptual hash similarity; CAMUS uses patient-level splits to prevent information leakage; five-fold cross-validation is applied to all datasets.

### Loss & Training

- Binary Cross-Entropy loss is used for ISIC'18
- Cross-Entropy loss is used for NeoPolyp and CAMUS
- Mixed-precision training on NVIDIA A100 GPUs is applied uniformly, except for SegNeXt due to training instability
- Special handling: MISSFormer and HiFormer operate internally at $224 \times 224$ resolution; U-KAN lacks ImageNet pretraining and is trained from scratch

## Key Experimental Results

### Main Results

| Model | Type | NeoPolyp mDSC | CAMUS mDSC | ISIC'18 mDSC | Avg. mDSC |
|-------|------|:---:|:---:|:---:|:---:|
| VW-MiT | GP-VM | 89.7 | 91.4 | 91.8 | **91.0** |
| VW-Conv | GP-VM | 89.6 | 91.3 | 91.8 | **90.9** |
| TransNeXt | GP-VM | 89.4 | 91.7 | 91.7 | **90.9** |
| InternImage | GP-VM | 89.6 | 91.2 | 91.5 | **90.8** |
| SegFormer | GP-VM | 89.1 | 91.4 | 91.7 | **90.7** |
| SegNeXt | GP-VM | 89.2 | 91.3 | 91.5 | **90.7** |
| Swin-UMamba | SMA | 88.9 | 91.3 | 91.3 | 90.5 |
| HiFormer | SMA | 84.6 | 90.8 | 91.0 | 88.8 |
| U-KAN | SMA | 82.5 | 90.5 | 90.6 | 87.9 |
| MISSFormer | SMA | 82.9 | 90.4 | 90.3 | 87.9 |
| U-Net | SMA | 83.3 | 89.1 | 89.3 | 87.2 |

### Ablation Study / Per-Class Analysis

| Model | NeoPolyp C1 (Non-neoplastic) | NeoPolyp C2 |
|-------|:---:|:---:|
| VW-MiT (GP-VM) | **66.1±4.3** | 92.7±0.9 |
| InternImage (GP-VM) | 66.0±5.7 | 92.9±0.7 |
| Swin-UMamba (SMA) | 59.2±3.8 | **92.5±0.6** |
| HiFormer (SMA) | 52.7±4.9 | 88.9±0.7 |
| U-KAN (SMA) | 36.9±12.2 | 87.1±0.9 |

On the most challenging NeoPolyp C1 category, GP-VMs lead the best-performing SMA (Swin-UMamba) by up to **7 percentage points**.

### Key Findings

- **GP-VMs systematically outperform most SMAs**: Ranked by average mDSC across three datasets, the top six positions are occupied entirely by GP-VMs (91.0%–90.7%); the best SMA (Swin-UMamba) ranks seventh (90.5%)
- **Performance gaps vary by dataset**: The largest gap appears on NeoPolyp (GP-VMs lead SMAs by 4–7 pp), while gaps narrow to 1–2% on ISIC'18 and CAMUS
- **Swin-UMamba is the only SMA competitive with GP-VMs**: All other SMAs trail by more than 4 percentage points on NeoPolyp
- **XAI analysis**: Grad-CAM visualizations indicate that GP-VMs capture clinically relevant structures without any explicit domain-specific design

## Highlights & Insights

- **Core Insight**: Under standardized conditions, the purported performance advantage of specialized architectures largely disappears—advantages reported in prior literature likely stem from differences in experimental setup rather than architectural design
- **Practical Value**: For new medical segmentation tasks, GP-VMs (especially VWFormer, InternImage, etc.) should be evaluated as strong baselines before investing in bespoke architecture design
- **Paradigm Implication**: The MIS community should shift greater attention toward data curation, training protocol optimization, and out-of-distribution (OOD) generalization evaluation, rather than pursuing new architectures as a primary research direction

## Limitations & Future Work

- Coverage is limited to three 2D datasets; 3D volumetric data (e.g., CT/MRI) and additional imaging modalities are not evaluated
- The 11 models surveyed are not exhaustive, and parameter count differences (25M–60M) may introduce confounds
- OOD generalization is not assessed (e.g., training on NeoPolyp and testing on Kvasir-SEG)
- Fine-tuned versions of recent foundation models (e.g., SAM2, BiomedParse) are not included
- Computational efficiency (FLOPs, inference latency) is not systematically compared

## Related Work & Insights

- **U-Net [MICCAI 2015]**: The foundational work in medical image segmentation; one of the baselines in this study
- **SAM [ICCV 2023]**: A general-purpose segmentation foundation model with multiple medical adaptations
- **VWFormer [ICLR 2024]**: A multi-scale window attention semantic segmentation model; the top-performing GP-VM in this study
- **Moglia et al. [2026]**: A survey of general-purpose models in MIS, though relying on non-standardized metrics reported in original papers
- Future work could extend this benchmarking framework to additional modalities and incorporate more rigorous OOD evaluation designs

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)

</div>

<!-- RELATED:END -->
