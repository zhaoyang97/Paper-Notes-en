---
title: >-
  [Paper Note] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography
description: >-
  [CVPR 2026][Medical Imaging][Foundation Models] This work constructs PETWB-Seg11K, the largest whole-body PET segmentation dataset to date (11,041 3D PET scans and 59,831 segmentation masks), and proposes SegAnyPET, a foundation model enabling prompt-driven universal volumetric segmentation of organs and lesions in PET imaging. The model demonstrates strong performance in zero-shot cross-center and cross-tracer settings.
tags:
  - CVPR 2026
  - Medical Imaging
  - Foundation Models
  - PET Imaging
  - Universal Segmentation
  - 3D Segmentation
  - Promptable Segmentation
date: 2026-05-08
content_hash: 655623217ac06c94
---

# Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography

**Conference**: CVPR 2026
**arXiv**: [2603.11627](https://arxiv.org/abs/2603.11627)
**Code**: None
**Area**: Medical Image Segmentation
**Keywords**: Foundation Models, PET Imaging, Universal Segmentation, 3D Segmentation, Promptable Segmentation

## TL;DR

This work constructs PETWB-Seg11K, the largest whole-body PET segmentation dataset to date (11,041 3D PET scans and 59,831 segmentation masks), and proposes SegAnyPET, a foundation model enabling prompt-driven universal volumetric segmentation of organs and lesions in PET imaging. The model demonstrates strong performance in zero-shot cross-center and cross-tracer settings.

## Background & Motivation

PET (Positron Emission Tomography) is a critical modality in nuclear medicine that visualizes in vivo metabolic processes via radioactive tracers, and is indispensable in oncology and neurology. However, PET image segmentation faces multiple challenges:

1. **Inherent difficulties**: PET lacks high-contrast anatomical boundary information, suffers from low signal-to-noise ratio and limited spatial resolution, making organ/lesion delineation significantly harder than CT/MRI.
2. **Data scarcity**: PET data acquisition and annotation are prohibitively costly, and public datasets are scarce and narrow in scope (restricted to specific tumor tasks).
3. **Failure of existing foundation models**: General medical segmentation foundation models such as SAM-Med3D, SegVol, and SAT are primarily trained on CT/MRI; direct transfer to PET yields poor results (SAT achieves DSC near zero).
4. **Limitations of task-specific models**: Conventional deep learning models can only segment fixed categories seen during training, requiring re-annotation and retraining for new organs or lesions.

## Method

### Overall Architecture

SegAnyPET adopts a SAM-like 3D promptable segmentation architecture consisting of three core components:

- **Image Encoder**: Extracts discrete 3D feature embeddings from the input PET volume.
- **Prompt Encoder**: Converts user-provided sparse prompts (e.g., points) or dense prompts (e.g., coarse masks) into compact prompt embeddings via fixed positional encodings and adaptive embedding layers.
- **Mask Decoder**: Fuses image features and prompt embeddings, generating the final segmentation output via upsampling and MLPs.

### Key Designs

1. **PETWB-Seg11K Dataset Construction**:
   - Integrates two public datasets (AutoPET, UDPET) and three private cohorts, totaling 11,041 whole-body 3D PET scans and 59,831 segmentation masks.
   - Covers real-world variations across multiple centers (global clinical sites), devices, tracers (FDG/PSMA), and disease types.
   - Carefully partitioned into internal validation sets (in-distribution) and external validation sets (different centers, cancer types, and tracers) for rigorous evaluation.

2. **3D Volumetric Architecture**:
   - Unlike 2D models that segment slice-by-slice and stack results, SegAnyPET operates directly on 3D volumes, fully exploiting inter-slice contextual information in PET volumes.
   - Point prompts enable efficient 3D interaction; mask prompts support iterative refinement.

3. **Dual-Variant Strategy**:
   - **SegAnyPET**: A universal segmentation foundation model trained on the full dataset, providing broad organ and lesion coverage with strong generalizability.
   - **SegAnyPET-Lesion**: A specialized variant fine-tuned on lesion-centric data, improving sensitivity and boundary precision for small, heterogeneous lesions.
   - Clinicians can select the general model or the lesion-specialized variant based on clinical needs.

### Loss & Training

- Training is conducted on PETWB-Seg11K using prompt engineering strategies for mask generation.
- Supports a human-in-the-loop interactive workflow: radiologists can iteratively refine segmentation results by appending positive/negative point prompts.
- SegAnyPET-Lesion is obtained by fine-tuning SegAnyPET on lesion-centric data.

## Key Experimental Results

### Main Results: Comparison with Task-Specific Models

SegAnyPET, as a universal model, is compared against several task-specific models trained exclusively on organ annotations:

| Model | Type | Training | Organ Segmentation | New Target Capability |
|---|---|---|---|---|
| nnU-Net | Task-specific | Fully supervised | Strong | ❌ Requires retraining |
| STUNet | Task-specific | Fully supervised | Moderate | ❌ Requires retraining |
| SwinUNETR | Task-specific | Fully supervised | Moderate | ❌ Requires retraining |
| SegResNet | Task-specific | Fully supervised | Moderate | ❌ Requires retraining |
| **SegAnyPET** | Universal Foundation | Prompt-based | **Comparable/Superior** | ✅ Zero-shot |

Key finding: Despite being a general-purpose model not specifically trained for organ segmentation, SegAnyPET achieves performance comparable to or exceeding nnU-Net on five seen target organs, without requiring task-specific retraining.

### Comparison with Segmentation Foundation Models

| Model | Prompt Type | PET Organ Segmentation | PET Lesion Segmentation |
|---|---|---|---|
| SAM-Med3D | Point prompt | Poor | Poor |
| SegVol | Point prompt | Poor | Poor |
| SAT | Text prompt | DSC≈0 | DSC≈0 |
| nnInteractive | Point prompt | Poor | Poor |
| VISTA3D | Point prompt | Poor | Poor |
| **SegAnyPET** | Point/Mask prompt | **Best** | **Best** |

Key finding: Text-prompt models (e.g., SAT) completely fail on PET (DSC≈0), indicating that their text-visual alignment is severely overfitted to anatomical features of structural imaging.

### Ablation & Generalization Experiments

| Validation Scenario | Distribution Shift | SegAnyPET Performance |
|---|---|---|
| Internal validation | In-distribution | Consistent and reliable |
| External – new cancer type | Unseen disease types | Robust generalization |
| External – PET/MRI | Scanner architecture change | Robust generalization |
| External – PSMA-PET | New tracer | Robust generalization |

Clinical utility validation: In lymphoma and lung cancer scenarios, the SegAnyPET-assisted annotation workflow reduced annotation time by **82.37%** and **82.95%** for two expert annotators, respectively.

### Key Findings

1. Existing general medical segmentation foundation models (e.g., SAM-Med3D) perform poorly on PET, exposing a large domain gap between structural and functional imaging.
2. Large-scale PET-specific training is essential — SegAnyPET learns domain-robust metabolic representations.
3. A single SegAnyPET model can effectively replace multiple task-specific networks.
4. Prompt-driven interaction enables the model to handle targets beyond the training label space, which is critical for whole-body PET analysis.

## Highlights & Insights

- **Outstanding data contribution**: PETWB-Seg11K is the largest and most comprehensive whole-body PET segmentation dataset to date, far surpassing existing PET datasets in scale.
- **First PET foundation model**: SegAnyPET is the first promptable segmentation foundation model specifically designed for PET imaging, filling a gap in functional imaging.
- **Zero-shot generalization**: The model demonstrates robustness under strict distribution shifts including cross-center, cross-tracer (FDG→PSMA), and cross-scanner (PET/CT→PET/MRI) settings.
- **Demonstrated clinical value**: Beyond segmentation accuracy, the work validates annotation efficiency gains (>82% time savings) and downstream applications in whole-body metabolic network analysis.
- **Insightful experimental observations**: The complete failure of text-prompt models on PET is attributed to cross-modal alignment being overfitted to structural imaging anatomy.

## Limitations & Future Work

1. **Interaction efficiency for diffuse lesions**: For multifocal lesions distributed throughout the body (e.g., lymphoma), point prompts require per-lesion interaction, limiting efficiency.
2. **Underrepresentation of rare diseases/tracers**: Certain rare diseases and tracers remain underrepresented in the dataset.
3. **Room for improvement in lesion segmentation**: Quantitative metrics indicate significant room for improvement in lesion segmentation accuracy.
4. **Absence of text prompt support**: Multimodal vision-language PET foundation models are an important future direction, enabling simultaneous identification of all diffuse lesions via semantic descriptions.
5. **Inference efficiency**: The computational overhead of 3D volumetric processing is not sufficiently discussed.

## Related Work & Insights

- **SAM (Segment Anything)** inspired the promptable segmentation paradigm, but fundamental architectural and data adaptations are required to transfer from 2D natural images to 3D PET.
- **nnU-Net** remains competitive under sufficient task-specific supervision, suggesting that the advantage of foundation models lies in flexibility rather than absolute superiority on any single task.
- **AutoPET** and **UDPET** are existing PET segmentation datasets, but their scope is too narrow; PETWB-Seg11K substantially expands both scale and diversity.
- **Insight**: A fundamental domain gap exists between functional imaging (PET/SPECT) and structural imaging (CT/MRI); general foundation models cannot be naively transferred, and modality-specific large-scale training is necessary.

## Rating

| Dimension | Score |
|---|---|
| Novelty | ⭐⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](../../ICCV2025/medical_imaging/seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)
- [\[CVPR 2026\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](mil-pf_multiple_instance_learning_on_precomputed_features_for_mammography_classi.md)
- [\[CVPR 2026\] Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning](act_like_a_pathologist_tissue-aware_whole_slide_image_reasoning.md)
- [\[CVPR 2026\] NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization](neuroseg_meets_dinov3_transferring_2d_self-supervised_visual_priors_to_3d_neuron.md)
- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)

</div>

<!-- RELATED:END -->
