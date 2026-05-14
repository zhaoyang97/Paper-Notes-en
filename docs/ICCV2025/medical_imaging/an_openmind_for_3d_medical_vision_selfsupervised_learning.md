---
title: >-
  [Paper Note] An OpenMind for 3D Medical Vision Self-supervised Learning
description: >-
  [ICCV 2025][Medical Imaging][Self-supervised learning] This work releases OpenMind, the largest publicly available 3D medical imaging pretraining dataset (114k brain MRI volumes)…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "Self-supervised learning"
  - "3D medical imaging"
  - "pretraining dataset"
  - "brain MRI"
  - "benchmark"
date: 2026-05-08
content_hash: 4d7eac864fee8a62
---

# An OpenMind for 3D Medical Vision Self-supervised Learning

**Conference**: ICCV 2025
**arXiv**: [2412.17041](https://arxiv.org/abs/2412.17041)
**Code**: [https://github.com/MIC-DKFZ/nnssl](https://github.com/MIC-DKFZ/nnssl)
**Area**: Medical Imaging
**Keywords**: Self-supervised learning, 3D medical imaging, pretraining dataset, brain MRI, benchmark

## TL;DR
This work releases OpenMind, the largest publicly available 3D medical imaging pretraining dataset (114k brain MRI volumes), and conducts a systematic benchmark of existing 3D SSL methods on this dataset using state-of-the-art CNN (ResEnc-L) and Transformer (Primus-M) architectures, establishing the current SOTA for 3D medical image SSL.

## Background & Motivation
The 3D medical imaging SSL field lacks consistency and standardization, making fair comparison among existing methods impossible due to three key issues:
1. **Absence of large-scale public pretraining datasets**: Large datasets (e.g., UK-BioBank >100k, ABCD >40k) are restricted by data use agreements (DUAs) requiring institutional review and mandatory authorship, impeding reproducible research. Most SSL methods are developed on small-scale or private data.
2. **Inconsistent architectural choices**: Different methods adopt CNNs, ViTs, Swin Transformers, or hybrid architectures, precluding direct comparison of the methods themselves.
3. **Inconsistent downstream evaluation**: Varying choices of evaluation datasets and small evaluation set sizes undermine result reliability.

## Core Problem
How to establish a standardized benchmark for 3D medical image SSL—with unified pretraining data, architecture choices, and evaluation protocols—so as to determine the true value of SSL pretraining and identify the current SOTA?

## Method

### Overall Architecture
Rather than proposing a new SSL method, this paper makes three key contributions:

**a) OpenMind Dataset**
- Source: 800 public studies from the OpenNeuro platform following the BIDS format
- Scale: **114k** 3D brain MRI volumes (34,191 subjects), covering 23 MRI modalities
- Includes 71k direct 3D MRI scans + 15k 4D DWI preprocessed into 43k 3D images (MD maps, FA maps, T2-weighted images)
- CC-BY-4.0 license with no access restrictions
- Accompanied by anonymization masks, anatomical masks, unified metadata, and image quality scores (IQS, 1–5)
- Released on HuggingFace

**b) SSL Benchmark**
- Two SOTA architectures: ResEnc-L (CNN) and Primus-M (Transformer)
- Seven SSL methods: VoCo, SwinUNETR pretraining, SimCLR, VolumeFusion (VF), ModelsGenesis (MG), MAE, S3D/SimMIM
- 15 downstream datasets: 12 segmentation + 3 classification
- Split into 4 development sets (for hyperparameter tuning) + 8 segmentation test sets + 3 classification test sets

**c) Open-source Release**
- Pretraining and fine-tuning framework code
- All pretrained model checkpoints
- Integration into the nnU-Net framework

### Key Designs

**Pretraining Configuration**:
- All methods uniformly pretrained on OpenMind for 1000 epochs × 250 steps/epoch
- Trained with 4×40GB A100 GPUs using DDP
- Unified spacing: 1mm³ isotropic; z-score normalization

**Five Fine-tuning Strategies** (one of the core contributions):
1. **Default**: Polynomial lr decay, initial lr decreasing from 1e-2 to 1e-3 (transfer learning setting)
2. **Frozen**: Encoder frozen; only the decoder is trained
3. **Warm-Up**: Linear lr increase followed by transition to the default schedule
4. **Valley**: Decreasing lr for decoder training → linear warm-up for the full network → default
5. **Sawtooth**: Two-stage warm-up: frozen encoder with increasing lr for decoder → increasing lr warm-up for the full network → default

**Optimal for CNN**: Sawtooth; **Optimal for Transformer**: Warm-Up

**Data Filtering Experiments** (data-centric):
- Filtering low-quality images by IQS (three thresholds)
- Modality filtering (retaining only T1w, T2w, FLAIR)
- Whether anonymized regions are included in the reconstruction loss

### Loss & Training
Each SSL method uses its standard loss (no new method is proposed in this paper):
- **MAE/S3D/SimMIM**: L2 reconstruction loss (applied only to masked regions)
- **SimCLR**: NT-Xent contrastive loss
- **VoCo**: Cosine similarity + regularization
- **VF**: Cross-entropy segmentation loss (pseudo-segmentation task)
- **MG**: Denoising + masked reconstruction
- **SwinUNETR**: Inpainting + rotation prediction + contrastive learning (equal-weight aggregation)

Fine-tuning is performed for 150 or 1000 epochs, batch size=2, using nnU-Net's polynomial lr schedule.

## Key Experimental Results

### Segmentation Results (DSC %, 150-epoch fine-tuning, averaged over 12 datasets)

| Method | Architecture | All Mean | ID Mean | OOD Mean | vs. Scratch 150ep |
|--------|-------------|----------|---------|----------|-------------------|
| Scratch 1k | ResEnc-L | 70.47 | 64.15 | 89.43 | - |
| Scratch | ResEnc-L | 68.44 | 62.23 | 87.08 | - |
| **MAE** | **ResEnc-L** | **70.91** | **65.11** | **88.30** | **+2.47** |
| S3D | ResEnc-L | 70.36 | 64.46 | 88.06 | +1.92 |
| MG | ResEnc-L | 70.30 | 64.37 | 88.09 | +1.86 |
| SimCLR | ResEnc-L | 69.44 | 63.40 | 87.56 | +1.00 |
| VoCo | ResEnc-L | 68.50 | 62.14 | 87.58 | +0.06 |
| Scratch 1k | Primus-M | 67.01 | 60.05 | 87.90 | - |
| Scratch | Primus-M | 63.62 | 57.29 | 82.61 | - |
| **MAE** | **Primus-M** | **70.42** | **64.34** | **88.69** | **+6.80** |
| SimMIM | Primus-M | 69.18 | 62.85 | 88.16 | +5.56 |
| VF | Primus-M | 68.19 | 61.75 | 87.51 | +4.57 |

**Key Findings**:
- MAE-pretrained ResEnc-L with 150-epoch fine-tuning surpasses the 1000-epoch from-scratch baseline (70.91 vs. 70.47)
- Transformers (Primus-M) benefit substantially more from pretraining than CNNs (+6.80 vs. +2.47)
- MAE-pretrained Primus-M nearly matches ResEnc-L (70.42 vs. 70.91), and even surpasses it on several datasets (ATL, COS, ACD)

### Classification Results
- Contrastive learning methods (VoCo, SwinUNETR, SimCLR) perform best on classification
- MAE performs worst on classification
- This indicates that global features (contrastive learning) suit classification, while local features (reconstruction) suit segmentation
- No single SSL method achieves top performance on both segmentation and classification simultaneously

### Ablation Study
1. **Fine-tuning strategy**: Sawtooth (CNN) and Warm-Up (Transformer) are optimal; the Frozen strategy degrades performance significantly, suggesting that current SSL representations have insufficient generalizability
2. **Data filtering**: Removing the lowest-quality images (retaining ~57%) yields a slight improvement (+0.15 DSC); however, reducing modality diversity (retaining only T1w/T2w/FLAIR, 62% of data) hurts performance (−0.43 DSC)
3. **Anonymization awareness**: Excluding anonymized regions from the reconstruction loss improves MAE and S3D performance (MAE All Mean: 70.91→71.29)
4. **Extended fine-tuning**: 1000-epoch fine-tuning benefits OOD datasets but may cause degradation (overfitting) on datasets where pretraining is already effective

## Highlights & Insights
1. **Major dataset contribution**: 114k 3D brain MRIs constitute the largest publicly available 3D medical imaging dataset under a CC-BY license, substantially lowering the barrier to SSL research
2. **First large-scale evidence that pretrained Transformers can match CNNs in 3D medical segmentation**: MAE-pretrained Primus-M surpasses the strongest ResEnc-L on several datasets
3. **Systematic benchmark**: Unified data, architecture, and evaluation across 7 methods × 2 architectures × 15 downstream tasks yield reliable conclusions
4. **Critical role of fine-tuning strategies**: Fine-tuning strategy is found to have a major impact on pretraining efficacy; Sawtooth/Warm-Up substantially outperform naive fine-tuning
5. **Complete open-source ecosystem**: Dataset, code framework, all checkpoints, and nnU-Net integration provide strong practical value
6. **Image quality metadata (IQS)**: First exploration of data-centric approaches in 3D medical SSL

## Limitations & Future Work
1. **Restricted to brain MRI**: All pretraining data are head-and-neck MRI; transferability to CT, chest, or abdominal imaging remains to be validated
2. **Classification experiments are less reliable**: The classification pipeline is less mature than nnU-Net; some datasets approach chance performance (ABI ~50% balanced accuracy)
3. **Limited gains from data filtering**: Simple IQS-based filtering yields only marginal improvements; the potential of data-centric methods remains underexplored
4. **Only 1000 pretraining epochs**: Computational constraints preclude longer pretraining, which may reveal different trends
5. **PEFT methods not explored**: Although freezing the encoder yields poor results, parameter-efficient fine-tuning approaches such as LoRA may mitigate this
6. **Metadata-aware SSL not explored**: The dataset provides rich metadata that is not leveraged during pretraining

## Related Work & Insights

| Aspect | Ours | Prior Work |
|--------|------|------------|
| Pretraining data | 114k public 3D MRI | Small-scale or private data (e.g., ABCD requires institutional approval) |
| Architecture comparison | Unified comparison of CNN + Transformer | Each method uses its own architecture |
| Downstream evaluation | 15 datasets, development/test split | Few datasets, unstable results |
| Reproducibility | Fully open-source (data + code + weights) | Mostly non-reproducible |

OpenMind is complementary to CT-Rate (50k CT): OpenMind focuses on MRI, is larger in scale (114k), and has a more permissive license (CC-BY vs. CC-BY-NC).

**Broader implications**:
- **Evidence for Transformers in medical imaging**: First large-scale demonstration that pretraining can close the performance gap between Transformers and CNNs
- **Fine-tuning strategy as a research direction**: Fine-tuning strategy selection deserves greater attention in other SSL + medical imaging tasks
- **Data-centric SSL**: Although simple filtering has limited effect, the dataset is large enough to support more sophisticated data curation approaches (e.g., semantic deduplication, curriculum learning)
- **Cross-modality pretraining**: Future work can combine CT datasets (e.g., CT-Rate) for multi-modal SSL
- **Urgent need for PEFT methods**: Given that frozen encoder underperforms and extended fine-tuning causes overfitting, LoRA/Adapter-style methods hold great promise for 3D SSL fine-tuning

## Rating
- Novelty: ⭐⭐⭐⭐ [Not a methodological contribution, but the systematic dataset + benchmark contribution is highly valuable and establishes the first standard for 3D medical SSL]
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ [7 methods × 2 architectures × 15 datasets × 5 fine-tuning strategies + data filtering + anonymization ablations — extremely comprehensive]
- Writing Quality: ⭐⭐⭐⭐⭐ [Clear structure, rigorous experimental design, well-summarized findings]
- Value: ⭐⭐⭐⭐⭐ [Dataset + benchmark + open-source framework are of tremendous community value; this will serve as the standard reference for 3D medical SSL]

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](../../CVPR2026/medical_imaging/learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[ICCV 2025\] Vector Contrastive Learning for Pixel-wise Pretraining in Medical Vision](vector_contrastive_learning_for_pixel-wise_pretraining_in_medical_vision.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](../../NeurIPS2025/medical_imaging/self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](../../NeurIPS2025/medical_imaging/ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data](../../NeurIPS2025/medical_imaging/self-supervised_learning_via_flow-guided_neural_operator_on_time-series_data.md)

</div>

<!-- RELATED:END -->
