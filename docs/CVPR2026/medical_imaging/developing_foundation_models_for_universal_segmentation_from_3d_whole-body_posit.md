---
title: >-
  [Paper Note] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography
description: >-
  [CVPR2026][Medical Imaging][PET segmentation] This work constructs PETWB-Seg11K, the largest whole-body PET segmentation dataset to date (11,041 3D PET scans + 59,831 masks)…
tags:
  - "CVPR2026"
  - "Medical Imaging"
  - "PET segmentation"
  - "foundation model"
  - "universal segmentation"
  - "interactive segmentation"
  - "whole-body PET"
  - "SAM"
date: 2026-05-08
content_hash: a6ab4b236192561a
---

# Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

**Conference**: CVPR2026
**arXiv**: [2603.11627](https://arxiv.org/abs/2603.11627)
**Code**: [YichiZhang98/SegAnyPET](https://github.com/YichiZhang98/SegAnyPET)
**Area**: Medical Imaging
**Keywords**: PET segmentation, foundation model, universal segmentation, interactive segmentation, whole-body PET, SAM

## TL;DR

This work constructs PETWB-Seg11K, the largest whole-body PET segmentation dataset to date (11,041 3D PET scans + 59,831 masks), and proposes SegAnyPET — the first 3D promptable segmentation foundation model tailored for functional PET imaging — achieving strong zero-shot generalization across multi-center, multi-tracer, and multi-disease scenarios.

## Background & Motivation

**Clinical indispensability of PET**: PET is the only nuclear medicine imaging modality capable of visualizing metabolic processes in vivo, playing a critical role in early diagnosis in oncology and neurology. However, PET images inherently lack anatomical contrast, making segmentation substantially more challenging than CT/MRI.

**Data scarcity and high annotation cost**: PET image acquisition and annotation are expensive. Existing public PET segmentation datasets are limited to specific tumor tasks and lack large-scale annotations covering whole-body multi-organ and multi-lesion scenarios.

**Poor generalization of task-specific models**: Conventional deep learning models can only segment fixed categories seen during training. Segmenting new target organs or lesions requires additional annotated data and retraining, which cannot accommodate the diverse and unpredictable clinical demands of whole-body PET interpretation.

**Cross-modality failure of existing foundation models**: 3D medical segmentation foundation models such as SAM-Med3D, SegVol, and SAT are primarily trained on structural CT/MRI images. The anatomical priors and appearance statistics they learn exhibit a large domain gap with functional PET images, resulting in poor direct transfer performance.

**Complete failure of text-prompted models on PET**: Text-prompted models (e.g., SAT) achieve DSC near zero on PET organ segmentation, demonstrating that their cross-modal alignment is severely overfitted to structural imaging.

**Inefficiency and inconsistency of manual delineation**: Manual annotation of PET volumes in clinical practice is time-consuming, exhibits high inter-observer variability, and suffers from poor reproducibility, underscoring the urgent need for automated and interactive high-precision segmentation tools.

## Method

### Overall Architecture

SegAnyPET follows SAM's three-component design of "image encoder + prompt encoder + mask decoder," extending all 2D Transformer components to a **fully 3D architecture** that directly captures inter-slice spatial relationships on volumetric inputs:

- **Image Encoder $\mathcal{E}_I$**: Takes a raw PET volume $X \in \mathbb{R}^{H \times W \times D}$ as input and extracts high-dimensional feature embeddings with 3D absolute positional encoding.
- **Prompt Encoder $\mathcal{E}_P$**: Supports sparse prompts (click points encoded via 3D positional encoding) and dense prompts (coarse masks aligned to the image latent space via 3D convolution + LayerNorm + GELU).
- **Mask Decoder $\mathcal{D}$**: Fuses image and prompt embeddings via 3D Transformer blocks, progressively upsamples via 3D transposed convolutions, and outputs volumetric segmentation masks.

### Key Designs

1. **Iterative Refinement Loop (Human-in-the-Loop)**: After each prediction, $\hat{Y}^{(t)}$ is automatically compared against the ground truth. Positive points are sampled from false-negative regions and negative points from false-positive regions; accumulated prompts progressively refine segmentation according to $\hat{Y}^{(t+1)} = \mathcal{D}(\mathcal{E}_I(X), \mathcal{E}_P(p^{(t)}, \hat{Y}^{(t)}))$.
2. **Dual-Variant Design**: A general model SegAnyPET (trained on the full dataset) and a specialized model SegAnyPET-Lesion (fine-tuned on lesion data for 200 epochs to improve sensitivity to small heterogeneous lesions).
3. **Interaction Simulation During Training**: At each iteration, 1–20 click points are randomly sampled to improve model adaptability across diverse interaction scenarios.
4. **Patch-Based Inference**: Patches of size $128^3$ are cropped; a sliding window strategy with 50% overlap is applied when predicted regions reach the boundary.

### Training Details

- **Hardware**: 8× NVIDIA A100 80 GB GPUs, DDP + NCCL backend.
- **Optimizer**: AdamW; lr = 8e-5 for the image encoder, lr = 8e-6 for the prompt encoder and mask decoder; weight decay = 0.1.
- **Training**: 500 epochs; input patch size $128^3$; global batch size = 96.
- **SegAnyPET-Lesion**: Additional fine-tuning on lesion data for 200 epochs.
- **Learning Rate Schedule**: MultiStepLR with 10× decay at epochs 120 and 180.
- **Data Augmentation**: Random flipping along three axes + adaptive cropping/padding.

### Loss & Training

Dice Loss + Cross-Entropy Loss (with sigmoid activation and squared predictions), jointly optimizing the volumetric segmentation task.

## Experiments

### Datasets

| Split | Source | Tracer | Scans | Masks |
|-------|--------|--------|-------|-------|
| Development set (C1–C4) | AutoPET + UDPET + two private cohorts | FDG | 11,041 | 59,831 |
| Internal validation | Independent in-center samples (organ/lymphoma/lung cancer) | FDG | 886 | 9,828 |
| External validation | Independent-center PET/CT + PET/MRI + PSMA-PET | FDG/PSMA | 1,551 | 34,579 |

### Comparison with Task-Specific Models (Internal Evaluation, DSC on 5 Organs)

| Model | Liver | Kidney-L | Kidney-R | Heart | Spleen |
|-------|-------|----------|----------|-------|--------|
| nnUNet | 0.938 | 0.903 | 0.870 | 0.912 | 0.887 |
| SegResNet | 0.936 | 0.907 | 0.859 | 0.903 | 0.894 |
| STUNet | 0.935 | 0.903 | 0.877 | 0.918 | 0.880 |
| SwinUNETR | 0.935 | 0.899 | 0.846 | 0.890 | 0.886 |
| SegAnyPET (1p) | 0.926 | 0.875 | 0.870 | 0.892 | 0.876 |
| SegAnyPET (3p) | 0.939 | 0.887 | 0.888 | 0.905 | 0.891 |
| **SegAnyPET (5p)** | **0.949** | **0.898** | **0.898** | **0.916** | **0.905** |

With only 5 point prompts, SegAnyPET surpasses all four fully supervised task-specific models — including nnUNet — across all 5 organs; with a single point, performance already matches that of the specialized models.

### Comparison with Other Foundation Models

2D models (SAM, MedSAM) employ a slice-by-slice processing strategy, yielding extremely low DSC (0.17–0.39) on PET and requiring several times more inference time than 3D models. Among 3D foundation models:

- **SAT (text prompts)**: DSC ≈ 0; text–visual alignment is severely overfitted to structural images and entirely fails to interpret PET metabolic signals.
- **SAM-Med3D / nnInteractive (point prompts)**: Spatial anchors aid localization but the underlying features cannot accurately delineate PET boundaries.
- **SegVol / VISTA3D**: Show some improvement but remain overall insufficient.

SegAnyPET consistently and significantly outperforms all baselines on every evaluation task (organs + lesions), attributed to domain-specific representations learned from large-scale PET data.

### Ablation Study & Key Findings

- **Incremental prompt benefit**: DSC improves by approximately 2–3 percentage points with each step from 1 → 3 → 5 points.
- **Zero-shot generalization**: Robust segmentation performance is maintained on completely unseen external data (different centers, PET/MRI, PSMA tracer).
- **Clinical utility**: The SegAnyPET-assisted interactive workflow reduces annotation time by **82.4%** and **83.0%** (two expert annotators).
- **Downstream applications**: Whole-body metabolic covariance networks constructed from segmentation outputs exhibit high biological fidelity, validating the clinical usability of the model outputs.

## Highlights & Insights

- **First PET segmentation foundation model**: Fills a critical gap in foundation model research for functional PET imaging.
- **Largest PET segmentation dataset**: PETWB-Seg11K substantially surpasses existing datasets with 11K+ scans and 60K masks.
- **One model replacing multiple specialized networks**: A single model reaches or exceeds specialized model performance on both organ and lesion segmentation.
- **Strong zero-shot cross-domain generalization**: Robust performance across centers, tracers, and modalities (PET/CT → PET/MRI).
- **Clinically deployable**: Supports human-in-the-loop collaboration with annotation efficiency gains exceeding 80%.

## Limitations & Future Work

- **Insufficient coverage of rare diseases and tracers**: Although large, PETWB-Seg11K still underrepresents certain rare diseases and anatomical regions.
- **Limited efficiency for discrete lesion segmentation**: Multi-focal discrete lesions such as whole-body lymphoma require individual click prompts, and a single click cannot capture all lesions.
- **Lesion segmentation accuracy has room for improvement**: Quantitative metrics indicate that tumor segmentation has yet to reach parity with organ segmentation.
- **Absence of text prompts**: The current system supports only point/mask prompts; future multimodal vision–language directions incorporating radiology reports may offer greater efficiency.

## Related Work & Insights

- **SAM / MedSAM**: 2D general/medical segmentation foundation models; slice-by-slice processing of 3D data is inefficient and spatially discontinuous.
- **SAM-Med3D / SegVol / SAT / VISTA3D / nnInteractive**: 3D medical segmentation foundation models primarily trained on CT/MRI data, with poor generalization to PET.
- **nnUNet**: A self-configuring task-specific segmentation framework that remains highly competitive under full supervision but cannot generalize to categories outside its training set.
- **TotalSegmentator / AbdomenAtlas**: Large-scale CT segmentation datasets and models; neither addresses the PET modality.

## Rating

- Novelty: ⭐⭐⭐⭐ — First 3D segmentation foundation model specifically designed for functional PET imaging, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Large data scale with comprehensive internal, external, cross-modality, and cross-tracer validation, including clinical utility evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated problem setting, rich figures and tables.
- Value: ⭐⭐⭐⭐ — Directly advances PET quantitative analysis and clinical workflows; both dataset and code are open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](../../ICCV2025/medical_imaging/seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)
- [\[CVPR 2026\] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)
- [\[AAAI 2026\] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling](../../AAAI2026/medical_imaging/propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)
- [\[CVPR 2026\] MUSE: Harnessing Precise and Diverse Semantics for Few-Shot Whole Slide Image Classification](muse_harnessing_precise_and_diverse_semantics_for_few-shot_whole_slide_image_cla.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)

</div>

<!-- RELATED:END -->
