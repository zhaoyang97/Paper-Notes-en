---
title: >-
  [Paper Note] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images
description: >-
  [ICCV 2025][Medical Imaging][PET segmentation] This paper introduces PETS-5k, the largest PET segmentation dataset to date (5,731 3D whole-body PET scans, over 1.3 million 2D slices), and proposes SegAnyPET — the first 3D promptable segmentation foundation model tailored for PET imaging. Through a Cross-Prompt Confidence Learning (CPCL) strategy to handle inconsistent annotation quality, SegAnyPET substantially outperforms existing foundation models and task-specific models on both seen and unseen targets.
tags:
  - ICCV 2025
  - Medical Imaging
  - PET segmentation
  - foundation model
  - 3D segmentation
  - noisy label learning
  - SAM adaptation
date: 2026-05-08
content_hash: a64e2ec001411990
---

# SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images

**Conference**: ICCV 2025
**arXiv**: [2502.14351](https://arxiv.org/abs/2502.14351)
**Code**: [https://github.com/](https://github.com/) (project page to be released)
**Area**: Medical Imaging
**Keywords**: PET segmentation, foundation model, 3D segmentation, noisy label learning, SAM adaptation

## TL;DR
This paper introduces PETS-5k, the largest PET segmentation dataset to date (5,731 3D whole-body PET scans, over 1.3 million 2D slices), and proposes SegAnyPET — the first 3D promptable segmentation foundation model tailored for PET imaging. Through a Cross-Prompt Confidence Learning (CPCL) strategy to handle inconsistent annotation quality, SegAnyPET substantially outperforms existing foundation models and task-specific models on both seen and unseen targets.

## Background & Motivation

**Background**: Positron emission tomography (PET) is a critical molecular imaging modality that reveals in vivo metabolic processes via radiotracers (e.g., 18F-FDG), and is widely used for tumor detection, treatment response assessment, and efficacy monitoring. In recent years, segmentation foundation models exemplified by SAM have demonstrated powerful general-purpose segmentation capabilities on natural images.

**Limitations of Prior Work**: PET images differ fundamentally from structural modalities such as CT and MRI — they suffer from low resolution, severe partial volume effects, poor contrast, and blurred boundaries. Directly applying SAM to PET images yields poor results. Existing medical SAM adaptation works (e.g., MedSAM, SAMed) almost exclusively focus on CT and MRI, entirely neglecting PET as a molecular imaging modality.

**Key Challenge**: (a) Large-scale annotated PET segmentation datasets are lacking; existing datasets are small in scale and limited in target coverage. (b) Low PET image quality leads to inconsistent annotation quality, with high-quality and noisy annotations intermixed. (c) 3D volumetric information is critical for PET segmentation, yet SAM is a 2D architecture.

**Goal**: (a) Construct a large-scale PET segmentation dataset; (b) design a 3D foundation model architecture tailored to PET characteristics; (c) achieve robust training under inconsistent annotation quality.

**Key Insight**: The authors observe that the distinctive properties of PET images (low contrast, weak boundaries, 3D volumetric structure) necessitate a modality-specific foundation model design rather than simple fine-tuning of general-purpose models. Meanwhile, identifying noisy annotations via confidence learning and applying self-correction enables effective utilization of both high-quality and low-quality data.

**Core Idea**: Construct the largest PET dataset + 3D SAM architecture + Cross-Prompt Confidence Learning to build the first universal promptable segmentation foundation model for PET imaging.

## Method

### Overall Architecture
The SegAnyPET pipeline takes a 3D PET volume and positional prompts (points or bounding boxes) as input. A 3D image encoder extracts features, which are combined with prompt encodings from a prompt encoder and passed through a mask decoder to produce 3D segmentation masks. During training, the CPCL strategy partitions data into high-quality and low-quality (noisy) annotation sets. Two models with different prompt strategies cross-supervise each other, and an uncertainty-guided self-correction mechanism is applied to refine noisy labels.

### Key Designs

1. **PETS-5k Large-Scale Dataset Construction**:

    - **Function**: Curate the largest PET segmentation dataset to date.
    - **Mechanism**: Multiple public PET datasets are aggregated and standardized into a unified benchmark, comprising 5,731 3D whole-body PET scans covering diverse organ and lesion targets across more than 1.3 million 2D slices.
    - **Design Motivation**: Prior PET datasets are too small in scale (e.g., AutoPET contains only ~500 cases) and cover a limited range of segmentation targets, making them insufficient for training foundation models.

2. **3D SAM Architecture Reconstruction**:

    - **Function**: Extend SAM's 2D architecture to 3D to fully exploit inter-slice contextual information from volumetric PET data.
    - **Mechanism**: The image encoder is upgraded from 2D convolutions/attention to their 3D counterparts, enabling the model to capture spatial relationships across adjacent slices. The prompt encoder is similarly adapted to support 3D point/box prompts.
    - **Design Motivation**: PET images are inherently 3D volumetric data, and inter-slice context is critical for accurate segmentation.

3. **Cross-Prompt Confidence Learning (CPCL)**:

    - **Function**: Enable robust training on a mixture of high-quality and low-quality noisy annotations.
    - **Mechanism**: Two model replicas are maintained, each using a different prompt strategy (e.g., point prompts vs. bounding box prompts), and they cross-supervise each other. The consistency between the two models' predictions is used to assess annotation reliability. For noisy-annotation samples, when the two models produce highly consistent predictions that diverge from the given annotation, that annotation is flagged as noisy.
    - **Design Motivation**: Blurred boundaries in PET images cause significant inter-annotator disagreement. CPCL leverages a cross-validation principle, using two models with different inductive biases to mutually verify each other.

4. **Uncertainty-Guided Self-Correction**:

    - **Function**: Automatically correct identified noisy annotations.
    - **Mechanism**: Prediction uncertainty is used to weight the self-correction process. Annotations in high-uncertainty regions are retained or down-weighted, while in low-uncertainty regions the model's predictions replace the original noisy annotations.
    - **Design Motivation**: Simply discarding noisy-annotation samples wastes a substantial amount of training data; self-correction allows the useful information in these samples to be fully exploited.

### Loss & Training
- High-quality data are supervised with standard Dice Loss + BCE Loss.
- For low-quality data, the loss function incorporates CPCL confidence weights to reduce the contribution of noisy-annotation samples.
- Training proceeds in two stages: warm-up on high-quality data, followed by joint training with low-quality data.

## Key Experimental Results

### Main Results

| Method | Seen Target Dice | Unseen Target Dice | Prompt Type |
|---|---|---|---|
| SAM (vanilla) | ~60% | ~45% | 1-point |
| MedSAM | ~72% | ~55% | 1-point |
| SAM-Med3D | ~75% | ~58% | 1-point |
| **SegAnyPET** | **~85%** | **~78%** | **1-point** |

### Ablation Study

| Configuration | Seen Target Dice | Unseen Target Dice | Note |
|---|---|---|---|
| Full model (CPCL + self-correction) | ~85% | ~78% | Complete model |
| w/o CPCL | ~80% | ~70% | Noisy annotations significantly degrade performance |
| w/o 3D architecture (2D) | ~78% | ~68% | 2D variant lacks inter-slice information |
| w/o uncertainty self-correction | ~82% | ~74% | Self-correction is critical for noisy data utilization |
| Training on high-quality data only | ~83% | ~72% | Reduced data volume hurts generalization |

### Key Findings
- The CPCL strategy contributes the most; removing it causes ~8-point drop on unseen targets, underscoring the importance of noise-robust training for PET foundation models.
- The 3D architecture yields approximately 7% gain over the 2D variant, validating the critical role of inter-slice context for PET segmentation.
- With only a single prompt point, SegAnyPET matches or surpasses fully supervised task-specific models.
- Strong generalization to unseen segmentation targets demonstrates the universality of the proposed foundation model.

## Highlights & Insights
- **First PET Foundation Model**: This work fills the gap in segmentation foundation models for molecular imaging modalities, establishing a foundation for downstream PET applications.
- **CPCL Noise-Robust Learning**: The cross-prompt confidence learning strategy requires no architectural modifications and can be directly transferred to other promptable foundation models. The principle of using two models with different prompt strategies for mutual verification is applicable to any scenario with inconsistent annotation quality.
- **Large-Scale Dataset Contribution**: PETS-5k, as the largest PET segmentation dataset, holds significant value for the broader PET analysis community.

## Limitations & Future Work
- The paper primarily focuses on FDG-PET; generalization to other tracers (e.g., PSMA-PET, FLT-PET) remains to be validated.
- CPCL requires maintaining two model replicas, doubling training cost.
- The 3D architecture incurs substantial computational overhead; efficiency optimization is needed for practical deployment.
- Multi-modal PET-CT fusion segmentation is not explored; joint PET+CT input may further improve performance.
- Self-correction relies on model quality; when the model is insufficiently trained in early stages, incorrect pseudo-labels may be introduced.

## Related Work & Insights
- **vs. MedSAM**: MedSAM fine-tunes SAM on CT/MRI but does not account for PET-specific characteristics and operates in 2D. SegAnyPET is purpose-designed for PET with a 3D architecture.
- **vs. SAM-Med3D**: Although also 3D, SAM-Med3D does not address annotation noise and is primarily trained on CT data.
- **vs. SAMed**: SAMed fine-tunes SAM with LoRA but remains a 2D architecture and lacks large-scale PET data.

## Rating
- Novelty: ⭐⭐⭐⭐ Pioneering as the first PET foundation model with CPCL strategy, though the overall technical design is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale dataset validation across diverse settings with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and well-organized structure.
- Value: ⭐⭐⭐⭐⭐ The dataset and model offer substantial value to the PET analysis community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](../../CVPR2026/medical_imaging/developing_foundation_models_for_universal_segmentation_from_3d_whole-body_posit.md)
- [\[ICCV 2025\] Coordinate-based Speed of Sound Recovery for Aberration-Corrected Photoacoustic Computed Tomography](coordinate-based_speed_of_sound_recovery_for_aberration-corrected_photoacoustic_.md)
- [\[ICCV 2025\] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images](aczerots_active_learning_for_zeroshot_tissue_segmentation_in.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-Supervised Learning](an_openmind_for_3d_medical_vision_self-supervised_learning.md)
- [\[ICCV 2025\] MRGen: Segmentation Data Engine for Underrepresented MRI Modalities](mrgen_segmentation_data_engine_for_underrepresented_mri_modalities.md)

<!-- RELATED:END -->
