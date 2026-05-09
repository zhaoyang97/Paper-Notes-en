---
title: >-
  [Paper Note] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study
description: >-
  [CVPR 2026][Medical Imaging][Medical image segmentation] Under a unified training and evaluation protocol, this study compares 11 models — 5 specialized medical segmentation architectures (SMAs) and 6 general-purpose vision models (GP-VMs) — across 3 heterogeneous medical datasets. GP-VMs systematically outperform most SMAs on all datasets (average mDSC: VW-MiT 91.0% vs. best SMA SU-Mamba 90.5%), and Grad-CAM analysis demonstrates that GP-VMs capture clinically relevant structures.
tags:
  - CVPR 2026
  - Medical Imaging
  - Medical image segmentation
  - general-purpose vision models
  - empirical comparison
  - interpretability
  - Grad-CAM
date: 2026-05-08
content_hash: 880dc8874927d6ff
---

# Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study

**Conference**: CVPR 2026
**arXiv**: [2603.13044](https://arxiv.org/abs/2603.13044)
**Code**: GitHub (see paper for specific link)
**Area**: Medical Imaging
**Keywords**: Medical image segmentation, general-purpose vision models, empirical comparison, interpretability, Grad-CAM

## TL;DR

Under a unified training and evaluation protocol, this study compares 11 models — 5 specialized medical segmentation architectures (SMAs) and 6 general-purpose vision models (GP-VMs) — across 3 heterogeneous medical datasets. GP-VMs systematically outperform most SMAs on all datasets (average mDSC: VW-MiT 91.0% vs. best SMA SU-Mamba 90.5%), and Grad-CAM analysis demonstrates that GP-VMs capture clinically relevant structures.

## Background & Motivation

**Background**: Since U-Net, the medical image segmentation (MIS) community has produced a proliferation of specialized architectures, including HiFormer (CNN-ViT hybrid), MISSFormer (pure Transformer), Swin-UMamba (state space model), and U-KAN (KAN-integrated), each incorporating domain-specific mechanisms to address challenges such as low contrast, small structures, and annotation scarcity. Concurrently, general-purpose vision models such as SegFormer, SegNeXt, ConvNeXt, InternImage, and TransNeXt have achieved strong performance on natural image semantic segmentation benchmarks.

**Limitations of Prior Work**: Performance comparisons in the literature are heavily confounded — different papers employ different datasets, preprocessing pipelines, data augmentation strategies, optimizer configurations, and evaluation protocols. Consequently, observed performance differences may reflect experimental design choices rather than genuine architectural superiority, leaving the fundamental question of whether specialized architectures truly outperform general-purpose models without a reliable answer.

**Key Challenge**: The medical imaging community's default assumption is that "the unique characteristics of medical images necessitate specialized architectural design." However, general-purpose models benefit from large-scale ImageNet pretraining and optimization validated on millions of images; the general visual representations they learn may already be sufficiently powerful. This tension between the two paradigms has not been subjected to controlled empirical examination.

**Goal**: (1) Under a unified protocol that eliminates all confounding factors, do specialized medical segmentation architectures hold a systematic advantage over general-purpose vision models? (2) Beyond accuracy, are the decision-making patterns of general-purpose models consistent with clinical knowledge, as validated through explainable AI (XAI)? (3) Do conclusions generalize across different imaging modalities and task settings?

**Key Insight**: A rigorously controlled empirical methodology is adopted — uniform ImageNet pretraining, $512\times512$ input resolution, AdamW with REX scheduling, identical data augmentation and early stopping — with per model-dataset learning rate search and 5-fold cross-validation to reduce stochasticity. Three datasets are selected to provide heterogeneity in modality (dermoscopy RGB / endoscopy RGB / ultrasound grayscale), class structure (binary / multi-class), and data characteristics.

**Core Idea**: Under a strictly controlled benchmarking framework, empirical evidence is used to demonstrate that general-purpose vision models can already substitute for most specialized medical segmentation architectures, suggesting that the marginal benefit of domain-specific design may have been overestimated.

## Method

### Overall Architecture

This is a purely empirical comparison study that proposes no new method. The research framework operates at three levels: (1) **Model selection** — representative architectures are chosen from both the SMA and GP-VM families, covering CNN, ViT, hybrid, state space, and KAN paradigms; (2) **Unified training and evaluation** — all models are trained under the same protocol on ISIC'18 (binary skin lesion), NeoPolyp (multi-class polyp), and CAMUS (multi-class cardiac ultrasound); (3) **Multi-dimensional analysis** — in addition to segmentation accuracy, Grad-CAM interpretability analysis is employed to reveal model attention patterns.

### Key Designs

1. **Unified Benchmarking Protocol (Eliminating Confounding Factors)**

   - **Function**: Ensure fair and credible comparisons across architectures.
   - **Mechanism**: All models use ImageNet-pretrained encoders, $512\times512$ input resolution, AdamW optimizer with REX learning rate scheduling, and batch size 8. For each model-dataset pair, the optimal learning rate is selected from $\{10^{-4}, 5\times10^{-5}, 10^{-5}\}$; models are trained for 100 epochs to identify the best configuration, then evaluated via 5-fold cross-validation over 150 epochs. Dataset-specific loss functions (BCE for ISIC'18, CE for multi-class datasets) and data augmentation are held constant across all models on a given dataset.
   - **Design Motivation**: Many comparisons in the literature are unreliable due to inconsistent training setups — differences in augmentation strategy, learning rate, and number of epochs can cause performance fluctuations of 2–5%, on the same order of magnitude as architectural differences.

2. **Heterogeneous Dataset Coverage (Validating Generalizability)**

   - **Function**: Select three datasets with high heterogeneity in modality, class structure, and data characteristics.
   - **ISIC'18**: Dermoscopic RGB images, 3,565 samples, binary (lesion / background), characterized by irregular boundaries.
   - **NeoPolyp (BKAI-IGH)**: Endoscopic RGB images, 945 samples, 3 classes (non-neoplastic / neoplastic polyp + background), characterized by large inter-subtype variation.
   - **CAMUS**: Cardiac ultrasound grayscale images, 1,996 samples, 4 classes (left ventricle / left ventricular wall / left atrium + background), characterized by substantial ultrasound noise.
   - **Design Motivation**: Covering RGB/grayscale, binary/multi-class, and dermatology/gastrointestinal/cardiac settings avoids bias from any single dataset. Duplicate and near-duplicate images are filtered, and CAMUS uses patient-level grouping to prevent data leakage.

3. **Grad-CAM Interpretability Analysis (Evaluation Beyond Accuracy)**

   - **Function**: Analyze whether model attention regions correspond to clinically relevant structures.
   - **Mechanism**: Grad-CAM is implemented via M3d-CAM, with automatic selection of the appropriate final layer to generate attention heatmaps. Attention distributions are visualized for each model on the 50 worst-performing samples per fold, assessing whether models "attend to the right regions."
   - **Design Motivation**: In safety-critical medical applications, a high-accuracy model attending to incorrect regions poses genuine risk. XAI analysis provides additional evidence for the reliability of GP-VMs.

### Loss & Training

Binary cross-entropy (BCE) is used for ISIC'18; multi-class cross-entropy (CE) is used for NeoPolyp and CAMUS. All models are optimized with AdamW and REX learning rate scheduling. Evaluation metrics include mDSC (primary), mIoU, mRecall, and mPrecision, all computed via global micro-averaging with the background class excluded. Training is conducted on 2× NVIDIA A100 GPUs with mixed-precision training.

## Key Experimental Results

### Main Results

5-fold cross-validation results across three datasets (mDSC%, mean ± std; **bold** denotes best within family):

| Model | Type | Params | ISIC'18 mDSC | NeoPolyp mDSC | CAMUS mDSC | Avg. |
|---|---|---|---|---|---|---|
| VW-MiT | GP-VM | 51M | 91.7±0.5 | **89.7±0.8** | 91.6±0.1 | **91.0** |
| VW-Conv | GP-VM | 57M | 91.5±0.4 | 89.6±1.3 | 91.4±0.1 | 90.9 |
| TransNeXt | GP-VM | 58M | **91.9±0.7** | 89.4±0.7 | 91.5±0.1 | 90.9 |
| InternImage | GP-VM | 58M | 91.3±0.4 | 89.6±1.1 | 91.4±0.2 | 90.8 |
| SegNeXt | GP-VM | 49M | 91.4±0.6 | 89.2±0.7 | **91.6±0.1** | 90.7 |
| SegFormer | GP-VM | 47M | 91.3±0.8 | 89.1±1.3 | 91.5±0.1 | 90.7 |
| **SU-Mamba** | **SMA** | **60M** | **91.3±0.5** | **88.9±0.6** | **91.3±0.3** | **90.5** |
| HiFormer | SMA | 26M | 91.0±0.6 | 84.6±0.9 | 90.8±0.2 | 88.8 |
| U-KAN | SMA | 25M | 89.2±1.1 | 82.5±1.7 | 90.5±0.2 | 87.4 |
| MISSFormer | SMA | 42M | 90.3±0.8 | 82.9±1.6 | 90.4±0.1 | 87.9 |
| U-Net | SMA | 31M | 89.0±0.9 | 83.3±1.1 | 89.1±0.3 | 87.1 |

### Ablation Study

Per-class mDSC on NeoPolyp (the most challenging dataset with the largest performance spread):

| Model | C1 Non-neoplastic mDSC | C2 Neoplastic mDSC | Overall mDSC |
|---|---|---|---|
| VW-MiT (GP-VM) | **66.1±4.3** | 92.7±0.9 | 89.7 |
| InternImage (GP-VM) | 66.0±5.7 | 92.9±0.7 | 89.6 |
| SU-Mamba (SMA) | 59.2±3.8 | **92.5±0.6** | 88.9 |
| HiFormer (SMA) | 52.7±4.9 | 88.9±0.7 | 84.6 |
| U-KAN (SMA) | 36.9±12.2 | 87.1±0.9 | 82.5 |
| U-Net (SMA) | 34.9±18.2 | 88.1±0.6 | 83.3 |

### Key Findings

- **GP-VMs systematically outperform SMAs**: Ranked by average mDSC across three datasets, the top six positions are occupied entirely by GP-VMs (91.0–90.7%), while the best SMA, SU-Mamba, achieves 90.5%. Although the margin is small (~0.5%), it is stable and reproducible under the unified protocol.
- **Largest gap on NeoPolyp**: GP-VMs show the most pronounced advantage on multi-class polyp segmentation — VW-MiT (89.7%) vs. SU-Mamba (88.9%) vs. HiFormer (84.6%). The key factor is the extremely challenging non-neoplastic polyp class (C1): GP-VMs achieve ~66% C1 mDSC while most SMAs attain only 35–53%, a gap of up to 7 percentage points.
- **Smaller gaps on ISIC'18 and CAMUS**: On skin lesion and cardiac ultrasound datasets, the difference between GP-VMs and the best SMA narrows to approximately 1–2%, indicating that architectural differences are attenuated in simple binary classification or large-data settings.
- **Grad-CAM confirms clinical plausibility of GP-VMs**: GP-VMs not only achieve higher accuracy but their attention heatmaps more precisely focus on clinically relevant structures. On ISIC'18 cases, GP-VMs attend more accurately to lesion regions than several SMAs; on CAMUS, GP-VMs detect more true positives for the difficult left atrium class (C3).
- **SU-Mamba is the only SMA approaching GP-VM performance**: Its Mamba state space model's capacity for long-range dependency modeling sets it clearly apart within the SMA family, yet it still falls short of GP-VMs.

## Highlights & Insights

- **Counter-intuitive empirical findings carry corrective value**: This work challenges the community consensus that "medical images require specialized architectures." Under controlled comparison, the advantages conferred by large-scale pretraining in general-purpose models outweigh the domain knowledge embedded in specialized designs, suggesting that researchers should benchmark general-purpose models before proposing new architectures.
- **The benchmarking methodology is itself a contribution**: The three-tier evaluation framework — unified protocol + heterogeneous datasets + XAI analysis — is more comprehensive and credible than conventional comparisons based solely on segmentation accuracy. The methodology for eliminating confounding factors is transferable to other medical AI subfields.
- **Practical implications for resource allocation**: If GP-VMs are already sufficient, research effort should shift from architectural innovation toward data curation, training protocol optimization, and out-of-distribution generalization evaluation — factors of greater practical consequence for clinical deployment.

## Limitations & Future Work

- **Only three 2D datasets**: 3D imaging modalities such as CT and MRI are not covered; specialized architectures (e.g., nnU-Net) may hold greater advantages in 3D segmentation.
- **Pretraining data imbalance**: GP-VMs benefit from large-scale ImageNet pretraining, whereas U-KAN has no pretrained weights and must be trained from scratch, introducing a degree of bias.
- **Limited computational fairness**: Model parameter counts range from 25M to 60M, and FLOPs/inference speed are not strictly controlled.
- **Foundation models not included**: SAM-based medical segmentation methods such as SAM-Med2D and MedSAM are not included in the comparison.
- **OOD generalization not assessed**: All evaluations are in-domain; robustness to out-of-distribution data — critical for clinical deployment — is not examined.

## Related Work & Insights

- **vs. nnU-Net**: nnU-Net achieves strong performance through an adaptive framework rather than fixed architectural design, which aligns with this paper's finding that training strategy matters more than architecture. However, nnU-Net is not directly compared in this study.
- **vs. SAM/MedSAM**: The SAM family represents an alternative paradigm — foundation model with prompting for medical segmentation. This paper focuses on the standard fine-tuning regime; the zero-shot/few-shot capabilities of SAM-based methods are a complementary research direction.
- **vs. TransUNet/SwinUNet**: These early Transformer+U-Net hybrid architectures aimed to introduce global attention, but MISSFormer (a pure Transformer SMA) performs modestly (87.9%) in this study, suggesting that the Transformer architecture per se is not the decisive factor — pretraining quality and scale are.

## Rating

- **Novelty**: ⭐⭐⭐ — No new method is proposed; however, the study surfaces an important finding long overlooked by the community.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 11 models × 3 datasets × 5-fold CV is rigorously designed, though 3D and OOD evaluations are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Research questions are clearly stated, experimental design is rigorous, and limitations are objectively acknowledged.
- **Value**: ⭐⭐⭐⭐ — Provides direct guidance for model selection strategies and resource allocation in medical segmentation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)

<!-- RELATED:END -->
