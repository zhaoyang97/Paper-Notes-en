---
title: >-
  [Paper Note] An OpenMind for 3D Medical Vision Self-Supervised Learning
description: >-
  [ICCV 2025][Medical Imaging][3D medical imaging] This work releases OpenMind, the largest publicly available 3D medical imaging pre-training dataset (114k brain MRIs), and systematically compares 7+ SSL methods across two architectures — a CNN (ResEnc-L) and a Transformer (Primus-M) — on 15 downstream datasets. Key findings: MAE pre-training yields the best segmentation performance, contrastive learning excels at classification, and for the first time, a pre-trained Transformer is shown to outperform a randomly initialized CNN on select datasets.
tags:
  - ICCV 2025
  - Medical Imaging
  - 3D medical imaging
  - self-supervised pre-training
  - benchmark
  - brain MRI
  - foundation model
date: 2026-05-08
content_hash: 06cb3039207f91a4
---

# An OpenMind for 3D Medical Vision Self-Supervised Learning

**Conference**: ICCV 2025
**arXiv**: [2412.17041](https://arxiv.org/abs/2412.17041)
**Code**: [OpenMind](https://github.com/MIC-DKFZ/openmind)
**Area**: Medical Imaging / Self-Supervised Learning
**Keywords**: 3D medical imaging, self-supervised pre-training, benchmark, brain MRI, foundation model

## TL;DR

This work releases OpenMind, the largest publicly available 3D medical imaging pre-training dataset (114k brain MRIs), and systematically compares 7+ SSL methods across two architectures — a CNN (ResEnc-L) and a Transformer (Primus-M) — on 15 downstream datasets. Key findings: MAE pre-training yields the best segmentation performance, contrastive learning excels at classification, and for the first time, a pre-trained Transformer is shown to outperform a randomly initialized CNN on select datasets.

## Background & Motivation

The 3D medical image self-supervised learning (SSL) field suffers from severe fragmentation, making it practically impossible to identify state-of-the-art methods. Three root causes are identified:

**Lack of large-scale open datasets**: Large-scale datasets (e.g., UK-BioBank >100k, ABCD >40k) require formal application and impose strict data use agreements (DUAs), such as mandating the dataset name in paper titles or requiring internal review. Existing public SSL methods are either developed on restricted large datasets or validated only on small public ones.

**Lack of comparability**: Different methods use different pre-training data, different network architectures (CNN / ViT / Swin / hybrid), and are evaluated on different downstream tasks, making fair comparison nearly impossible.

**Neglected fine-tuning strategies**: Fine-tuning strategy has a substantial impact on downstream performance of pre-trained models, yet systematic comparisons are largely absent in prior work.

**Key Challenge**: How can the true capabilities of 3D medical image SSL methods be fairly evaluated within a unified framework?

**Key Insight**: Address the problem from three dimensions — data, benchmark, and code — by providing the largest open dataset, a unified architecture/training/evaluation framework, and fully open-sourced pre-trained weights.

## Method

### Overall Architecture

The primary contribution of this work is systematic engineering rather than a novel algorithm:

1. **OpenMind Dataset**: 114k 3D head-and-neck MRIs, 23 modalities, CC-BY license
2. **OpenMind Benchmark**: Comparison of 7+ SSL methods × 2 architectures × 15 downstream datasets under unified conditions
3. **Open-source Ecosystem**: Pre-training/fine-tuning code + model weights

### Key Designs

#### 1. OpenMind Dataset Construction

- **Data source**: 800 independent studies from the OpenNeuro platform
- **Raw data**: 71k 3D MRIs + 15k 4D diffusion-weighted images (DWI)
- **DWI preprocessing**: 4D DWI volumes are converted into three types of 3D derived images (MD maps, FA maps, T2-weighted), yielding an additional 43k volumes
- **Defacing**: Anonymization masks and anatomical masks are generated to prevent reconstruction-based SSL methods from being penalized on defaced regions
- **Metadata standardization**: Participant demographics and scan parameters are harmonized
- **Image Quality Score (IQS)**: Two independent reviewers score each modality of each dataset on a 1–5 scale

#### 2. Benchmark Design

**Unified setup**:
- All methods are pre-trained on OpenMind for 1,000 epochs × 250 steps/epoch
- Distributed training with 4×A100 GPUs (DDP)
- Two architectures: ResEnc-L (CNN) and Primus-M (Transformer)

**Downstream evaluation**:
- 4 development datasets (for hyperparameter tuning) + 8 test segmentation datasets + 3 test classification datasets
- All datasets split 50/50 into train/test
- Segmentation fine-tuned via nnU-Net; classification via an independent framework

**SSL methods**: VoCo, SimCLR, VolumeFusion, Models Genesis, MAE, S3D, SimMIM, SwinUNETR pre-training scheme

#### 3. Fine-tuning Strategy Comparison

Five fine-tuning strategies are designed to balance pre-trained representation preservation with downstream adaptation:
- **Default**: Polynomial decay LR, initial rates of 1e-2 / 1e-3
- **Frozen**: Only the decoder is trained
- **Warm-Up**: Linear warmup followed by the default strategy
- **Valley**: Decoder-only training (linear LR decay) → linear warmup → default
- **Sawtooth**: Two-phase warmup (decoder only with frozen encoder → full network) → default

### Loss & Training

- Segmentation downstream: NFL loss (nnU-Net default), polynomial LR decay
- Classification downstream: 200-epoch fine-tuning
- Pre-training: Each SSL method retains its original loss design

## Key Experimental Results

### Main Results (Segmentation, DSC%)

| Pre-training Method | Architecture | ID Mean | OOD Mean | Overall Mean |
|---------------------|--------------|---------|----------|--------------|
| Scratch 1k | ResEnc-L | 64.15 | 89.43 | 70.47 |
| MAE | ResEnc-L | **65.11** | **88.30** | **70.91** |
| S3D | ResEnc-L | 64.46 | 88.06 | 70.36 |
| MG | ResEnc-L | 64.37 | 88.09 | 70.30 |
| Scratch 1k | Primus-M | 60.05 | 87.90 | 67.01 |
| MAE | Primus-M | **64.34** | **88.69** | **70.42** |
| VoCo | Primus-M | 52.00 | 74.43 | 57.61 |

MAE pre-trained CNN surpasses the 1,000-epoch from-scratch baseline with only 150 fine-tuning epochs; MAE pre-trained Transformer gains ~3.5% DSC, approaching the from-scratch CNN performance level.

### Ablation Study (Fine-tuning Strategy Comparison, CNN)

| Strategy | VoCo | SimCLR | VF | MG | MAE | S3D | Mean |
|----------|------|--------|----|----|-----|-----|------|
| Default | 77.09 | 74.48 | 76.70 | 76.65 | 76.49 | 76.65 | 76.15 |
| Frozen | 58.28 | 61.73 | 34.73 | 59.48 | 57.18 | 59.40 | 55.25 |
| Warm-Up | 75.47 | 76.30 | 76.60 | 77.36 | 77.75 | 76.40 | 76.33 |
| Sawtooth | **75.96** | **76.23** | **77.68** | **76.66** | **77.50** | **76.87** | **76.60** |

The Sawtooth strategy achieves the best overall performance. The large performance drop under the Frozen strategy indicates that current SSL representations still lack sufficient generalization capacity.

### Key Findings

1. **Reconstruction-based methods (MAE) are best for segmentation**, while contrastive methods (VoCo, SimCLR) **excel at classification** — no single method dominates both tasks simultaneously.
2. **Transformers benefit more from pre-training**: MAE pre-trained Primus-M already outperforms the best pre-trained CNN on the ATL, COS, and ACD datasets.
3. **Fine-tuning strategy is critical**: An inappropriate strategy can eliminate all gains from pre-training.
4. **Data quality filtering is effective but limited**: Removing the lowest-quality data yields modest improvements, but aggressive filtering (retaining only 33%) degrades performance.
5. **Accounting for defaced regions improves MAE representation quality**: Excluding anonymized regions from the reconstruction loss improves both MAE and S3D performance.

## Highlights & Insights

1. **Landmark resource contribution**: 114k 3D MRIs under a CC-BY license represents a significant milestone for the 3D medical SSL community.
2. **First demonstration that pre-trained Transformers can approach or surpass from-scratch CNNs**: This challenges the prevailing assumption that Transformers consistently underperform CNNs in 3D medical imaging.
3. **Deep insight into global vs. local representations**: Contrastive learning captures global features suited to classification; MAE captures local features suited to segmentation; SwinUNETR's hybrid training objective biases representations toward classification.
4. **Complete reproducibility ecosystem**: Pre-trained checkpoints + nnU-Net-integrated fine-tuning code.

## Limitations & Future Work

- Classification results are less reliable due to a less mature classification framework compared to nnU-Net.
- Pre-training is limited to 1,000 epochs; longer training may alter method rankings.
- Evidence for data-centric approaches remains preliminary, relying only on simple filtering strategies.
- Parameter-efficient fine-tuning (PEFT) methods are not explored.
- The dataset covers only head-and-neck MRI; abdominal and thoracic regions remain out-of-domain.

## Related Work & Insights

- The nnU-Net framework provides a reliable basis for segmentation evaluation, lending high credibility to the segmentation conclusions of this work.
- The success of DINOv2 and MAE in natural image domains motivates the systematic evaluation of their 3D counterparts in this work.
- Data quality filtering research (e.g., DataComp) suggests that the OpenMind dataset has the potential to support more sophisticated data-centric approaches.

## Rating

- **Novelty**: ⭐⭐⭐ — Primary contributions are resources and benchmarks rather than novel methods.
- **Technical Depth**: ⭐⭐⭐⭐ — Benchmark design is rigorous and findings are insightful.
- **Practical Value**: ⭐⭐⭐⭐⭐ — Highly impactful for the 3D medical SSL community.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with comprehensive and thorough experiments.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](../../CVPR2026/medical_imaging/learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[ICCV 2025\] UKBOB: One Billion MRI Labeled Masks for Generalizable 3D Medical Image Segmentation](ukbob_one_billion_mri_labeled_masks_for_generalizable_3d_medical_image_segmentat.md)
- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)

<!-- RELATED:END -->
