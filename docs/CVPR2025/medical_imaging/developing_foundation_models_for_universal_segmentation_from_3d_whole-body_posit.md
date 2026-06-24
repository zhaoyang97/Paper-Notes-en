---
title: >-
  [Paper Note] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography
description: >-
  [CVPR2025][Medical Imaging][PET segmentation] Constructed the largest PET segmentation dataset PETWB-Seg11K (11,041 whole-body PET cases + 59,831 segmentation masks) and proposed SegAnyPET—a foundation model for universal PET segmentation based on a 3D architecture + prompt engineering. It demonstrates strong zero-shot generalization capabilities across multi-center, multi-tracer, and multi-disease scenarios.
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "PET segmentation"
  - "foundation models"
  - "universal segmentation"
  - "prompt-driven"
  - "3D medical imaging"
date: 2026-05-08
content_hash: 81e0e07eccf0b7c5
---

# Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

**Conference**: CVPR2025  
**arXiv**: [2603.11627](https://arxiv.org/abs/2603.11627)  
**Code**: To be confirmed  
**Area**: Medical Imaging  
**Keywords**: PET segmentation, foundation models, universal segmentation, prompt-driven, 3D medical imaging

## TL;DR

Constructed the largest PET segmentation dataset PETWB-Seg11K (11,041 whole-body PET cases + 59,831 segmentation masks) and proposed SegAnyPET—a foundation model for universal PET segmentation based on a 3D architecture + prompt engineering. It demonstrates strong zero-shot generalization capabilities across multi-center, multi-tracer, and multi-disease scenarios.

## Background & Motivation

- PET is a crucial imaging modality in nuclear medicine, revealing metabolic processes in the body through radioactive tracer distribution, and is irreplaceable in oncology and neurology.
- Accurate organ/lesion segmentation in PET is essential for quantitative analysis, but PET inherently lacks anatomical boundary contrast, making manual annotation time-consuming and inconsistent.
- Breakthroughs in deep learning-based segmentation have mainly focused on CT/MRI, while PET lags far behind due to the high cost of data acquisition and annotation.
- Existing public PET segmentation datasets are limited to specific tumor tasks and lack whole-body multi-organ coverage; task-specific models fail to generalize to new targets.
- The adaptation of foundation models like SAM to the medical field primarily targets CT/MRI/microscopic images, ignoring the diffuse metabolic signals and low-resolution characteristics of PET, leading to poor direct migration performance.
- **Design Motivation**: The PET field urgently requires a dedicated large-scale dataset and a foundation model to achieve universal, promptable organ/lesion segmentation.

## Method

### Dataset: PETWB-Seg11K
- A total of 11,041 whole-body 3D PET scans + 59,831 segmentation masks, curated from 2 public datasets (AutoPET, UDPET) + 3 private cohorts.
- Covers multi-center, multi-scanner, and multi-disease types, exhibiting real-world variability in scanner manufacturers, acquisition protocols, slice thicknesses, etc.
- Validation set design: internal validation (unseen cases from the same centers) + external validation (independent centers + unseen cancer types + different tracers like PSMA-PET).

### Model Architecture: SegAnyPET
Adopts a SAM-style universal promptable segmentation design, extended to a fully 3D architecture:

1. **Image Encoder**: Extract discrete 3D feature embeddings from the input PET volume.
2. **Prompt Encoder**: Transform user inputs (sparse prompts like clicks, dense prompts like coarse masks) into prompt embeddings via fixed positional encoding + adaptive embedding layers.
3. **Mask Decoder**: Fuse image and prompt features, upsample, and generate the final segmentation output via an MLP.

### Key Designs
- **Point prompts**: Fast and efficient 3D interaction; **mask prompts**: Support iterative refinement, enabling a human-in-the-loop clinical workflow.
- **Two variants**: SegAnyPET (general-purpose, trained on the full dataset) and SegAnyPET-Lesion (lesion-specialized, fine-tuned on lesion data).
- Essential difference from task-specific models: No need for re-labeling and re-training for new targets; region of interest is dynamically specified via prompts.

## Loss & Training
- The paper does not provide detailed loss formulations, but follows the standard segmentation losses of the SAM series (Dice + BCE).
- Trained on the large-scale heterogeneous PETWB-Seg11K dataset, covering multi-manufacturer, multi-protocol, and multi-disease distributions.
- SegAnyPET-Lesion is obtained by fine-tuning SegAnyPET on lesion-centric data, improving the sensitivity and boundary accuracy of small, heterogeneous lesions.
- All task-specific baselines are implemented within the standardized nnUNet framework, using its automatic preprocessing, resampling, and augmentation pipelines to ensure a fair comparison.

## Key Experimental Results

### Comparison with Task-Specific Models (Internal Validation, Organ Segmentation)
- Without task-specific training, SegAnyPET achieves comparable or superior performance compared to dedicated models such as nnUNet, STUNet, SwinUNETR, and SegResNet.
- nnUNet remains the strongest baseline, but a single SegAnyPET model can replace multiple task-specific networks.

### Comparison with Segmentation Foundation Models
- Universal foundation models such as SAM-Med3D, SegVol, SAT, nnInteractive, and VISTA3D perform poorly on PET.
- Text prompt models (e.g., SAT) yield DSC close to zero on PET organ segmentation—cross-modal alignment is severely overfitted to CT anatomical structures.
- Point prompt models (such as SAM-Med3D) are slightly better but still insufficient.
- SegAnyPET consistently outperforms state-of-the-art (SOTA) foundation models across all evaluated tasks.

### Generalization Capability (External Validation)
- Unseen cancer types / independent centers: Robust generalization.
- PET/MRI (different attenuation correction physics): Maintains reliable segmentation.
- PSMA-PET (entirely new tracer): Successful cross-tracer generalization.

### Clinical Utility
- Annotation efficiency: Compared to purely manual delineation, the SegAnyPET-assisted interactive workflow saves **82.37%** and **82.95%** of annotation time for two experts, respectively.
- Whole-body metabolic covariance network: Segmentation results can be directly used for downstream inter-organ metabolic network analysis, showing high biological fidelity.

## Highlights & Insights

- **First PET-dedicated foundation model**: Fills the gap in segmentation foundation models for functional imaging.
- **Largest PET segmentation dataset**: Over 11K whole-body PET scans, far exceeding existing datasets in both scale and diversity.
- **Strong zero-shot generalization**: Effective across different centers, tracers, and disease types.
- **Clinical practicality**: >82% annotation time savings + human-in-the-loop iterative refinement design.
- **Scalability**: Incorporates new organs/lesions into the analysis pipeline via prompting without requiring retraining.
- **Two-variant design**: General-purpose SegAnyPET offers broad coverage, while lesion-specific SegAnyPET-Lesion is optimized for small, heterogeneous lesions, allowing users to choose according to their needs.

## Limitations & Future Work

- Data for rare diseases, uncommon tracers, and specific anatomical regions remain insufficient.
- There is still room for improvement in quantitative metrics for lesion segmentation, especially for small and disseminated lesions.
- The point prompt paradigm has limited efficiency for diffuse lesions (e.g., multiple lymphoma lesions), requiring clicks on individual lesions.
- Lack of text prompt support—multimodal vision-language PET foundation models represent a future direction.
- Private data accounts for a large proportion, limiting external reproducibility.
- Current evaluations rely mainly on volumetric metrics like DSC, lacking boundary accuracy assessments such as surface distance.

## Related Work & Insights

- **CT/MRI segmentation foundation models**: SAM $\rightarrow$ MedSAM $\rightarrow$ SAM-Med3D / SegVol / VISTA3D; however, these are trained on structural imaging and generalized poorly to PET.
- **PET segmentation datasets**: Public datasets like AutoPET and UDPET have limited scale and focus on single tasks.
- **Task-specific PET segmentation**: Frameworks like nnUNet remain strong baselines but are limited by a fixed label space.
- **Universal segmentation**: The success of SAM in natural images inspired adaptations in the medical domain, but the functional signal characteristics of PET cause direct transfer to fail.
- **PET quantitative analysis**: Traditional workflows rely on manual ROI delineation, which suffers from poor inter-observer consistency, highlighting an urgent need for automation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First-of-its-kind PET-dedicated foundation model, representing a milestone in dataset scale)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Internal and external validation + comparisons with multiple baselines + ablation studies + downstream applications + clinical annotation experiments)
- Writing Quality: ⭐⭐⭐⭐ (Clear and well-structured, with rich figures and tables, although some content is repetitive)
- Value: ⭐⭐⭐⭐⭐ (Significant driving force for the field of PET imaging AI)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](../../ICCV2025/medical_imaging/seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)
- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [\[CVPR 2025\] Show and Segment: Universal Medical Image Segmentation via In-Context Learning](show_and_segment_universal_medical_image_segmentation_via_in-context_learning.md)
- [\[CVPR 2025\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilistic_models_through_large-scale_.md)
- [\[CVPR 2025\] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging](vista3d_a_unified_segmentation_foundation_model_for_3d_medical_imaging.md)

</div>

<!-- RELATED:END -->
