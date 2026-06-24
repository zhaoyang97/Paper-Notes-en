---
title: >-
  [Paper Note] Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation
description: >-
  [CVPR 2025][Medical Imaging][SAM] Proposes a semi-supervised medical image segmentation framework that enhances SAM. By utilizing CLIP and VQA, it unsupervisedly generates efficient prompts containing semantic, location, and shape information (without requiring expert annotations). It then employs Direct Preference Optimization (DPO) combined with a virtual annotator (replacing human annotators to provide rankings/scores) to train the optimal segmentation policy…
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "SAM"
  - "Semi-Supervised Segmentation"
  - "DPO Preference Optimization"
  - "Unsupervised Prompt Generation"
  - "Virtual Annotator"
date: 2026-05-08
content_hash: e8d8c92b4fb17651
---

# Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-Supervised Medical Image Segmentation

**Conference**: CVPR 2025  
**Code**: To be confirmed  
**Area**: Medical Images / RLHF Alignment  
**Keywords**: SAM, Semi-Supervised Segmentation, DPO Preference Optimization, Unsupervised Prompt Generation, Virtual Annotator

## TL;DR

Proposes a semi-supervised medical image segmentation framework that enhances SAM. By utilizing CLIP and VQA, it unsupervisedly generates efficient prompts containing semantic, location, and shape information (without requiring expert annotations). It then employs Direct Preference Optimization (DPO) combined with a virtual annotator (replacing human annotators to provide rankings/scores) to train the optimal segmentation policy, achieving SOTA performance on multi-modal tasks including lung, breast tumor, and organ segmentation.

## Background & Motivation

**Background**: The Segment Anything Model (SAM), as a vision foundation model, has demonstrated strong potential in medical image segmentation, theoretically generalizing to various downstream segmentation tasks. However, SAM is inherently a supervised model, and its powerful performance relies on large-scale annotated data or high-quality prompts (such as points/boxes/mask prompts) provided by domain experts.

**Limitations of Prior Work**: Medical image annotation is extremely costly, requiring pixel-wise annotations by specialists like radiologists, and annotation protocols vary across different modalities (X-ray/Ultrasound/CT), making scaling difficult. Traditional mitigation strategies, such as active learning, reduce annotation volume but still require continuous human involvement to refine labels or establish ground truth rewards, resulting in limited scope and complex workflows. More critically, SAM's prompting mechanism itself requires expert intervention to provide spatial priors, creating a paradox where using a foundation model still necessitates human experts.

**Key Challenge**: SAM’s strong generalization capabilities must be activated by precise prompts. However, in medical scenarios, the cost of obtaining precise prompts is almost as high as obtaining full annotations. Automatically generating effective SAM prompts under unsupervised or semi-supervised settings while maintaining segmentation quality remains the key challenge.

**Goal**: (1) How to generate effective segmentation prompts for SAM without relying on expert annotations? (2) How to optimize the segmentation policy of SAM in scenes with scarce annotations to produce high-fidelity segmentation results?

**Key Insight**: Leveraging the vision-language alignment capability of CLIP and the question-answering ability of VQA to automatically extract semantic, location, and shape information from medical images as SAM prompts, and drawing inspiration from DPO (Direct Preference Optimization) in LLMs, the framework utilizes simple rankings/scores provided by a virtual annotator to optimize the segmentation policy, completely bypassing traditional pixel-level annotation requirements.

**Core Idea**: Employing CLIP+VQA to generate SAM prompts automatically and utilizing DPO+virtual annotators for preference optimization, thereby achieving high-quality semi-supervised medical image segmentation without expert annotation.

## Method

### Overall Architecture

Input medical image $\rightarrow$ CLIP provides semantic-level features (identifying semantic information of the target region) + VQA module obtains location and shape priors through question-answering $\rightarrow$ Fusion to generate SAM prompts (point/box prompts or dense prompts) $\rightarrow$ SAM generates multiple candidate segmentations $\rightarrow$ Virtual Annotator ranks/scores candidate segmentations $\rightarrow$ DPO preference optimization trains the optimal segmentation policy. The entire pipeline requires zero human annotation intervention.

### Key Designs

1. **Unsupervised Prompt Generation Module (CLIP + VQA)**:

    - **Function**: Automatically generates efficient prompts containing semantic, location, and shape information for SAM in the absence of manual annotation.
    - **Mechanism**: Leverages the pre-trained contrastive learning capability of CLIP for semantic-level region localization by aligning medical terminology (e.g., "lung", "tumor") with image regions to find the most probable locations containing the target. Simultaneously, a VQA model answers questions regarding target shape, size, and boundaries to supplement spatial priors. The fusion of both generates prompt formats compatible with SAM (such as bounding boxes or point annotations).
    - **Design Motivation**: Traditional SAM prompts must be manually provided by experts, making them costly and non-scalable. Since both CLIP and VQA are pre-trained models that do not require task-specific annotation data, they are naturally suited for unsupervised scenarios.

2. **DPO Preference Optimization Strategy**:

    - **Function**: Optimizes the segmentation output quality of SAM through preference learning rather than traditional supervised learning.
    - **Mechanism**: Drawing on DPO (Direct Preference Optimization) from LLMs, the segmentation task is reframed as a preference learning problem. Given two candidate segmentation results of the same image, a virtual annotator determines which one is superior (instead of providing pixel-level labels). The model learns an optimal policy to ensure its outputs align with the preference ranking. The core advantage of DPO is that it learns directly from preference pairs without requiring an explicit reward model.
    - **Design Motivation**: The cost of obtaining ground truth annotations for medical segmentation is high, whereas relative judgments regarding "which segmentation is better" are significantly simpler than pixel-level precision labeling. DPO downgrades the annotation requirement from absolute judgments to relative rankings.

3. **Virtual Annotator**:

    - **Function**: Simulates the human annotation process by providing scores or rankings for candidate segmentation results.
    - **Mechanism**: An automated evaluator is designed to automatically rank multiple candidate segmentations based on metrics like connectivity of segmentation results, boundary smoothness, and consistency with CLIP semantic features. This virtual annotator replaces the human annotation loop in traditional RLHF, making the entire pipeline fully automated.
    - **Design Motivation**: DPO requires preference data (win/lose pairs), but human annotators are costly and inconsistent. The virtual annotator provides consistent and scalable preference signals through calculable quality metrics.

### Loss & Training

A DPO loss function is employed to train the segmentation policy: given a preference pair $(y_w, y_l)$ consisting of a win segmentation and a lose segmentation, the DPO loss encourages the model to generate segmentations closer to $y_w$. Training is split into two phases: first, unsupervised prompts are used to generate initial candidate segmentations and collect preference data, after which DPO is applied to optimize the model policy.

## Key Experimental Results

### Main Results

| Task | Modality | Metrics | Ours |
|------|------|------|------------|
| Lung Segmentation | X-ray | Dice / IoU | SOTA |
| Breast Tumor Segmentation | Ultrasound | Dice / IoU | SOTA |
| Organ Segmentation | Abdominal CT | Dice / IoU | SOTA |

The method achieves state-of-the-art performance across all three different modalities of medical segmentation tasks, demonstrating the cross-modal generalization capability of the framework.

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| Full model (CLIP + VQA + DPO) | Optimal | Complete framework |
| w/o DPO (unsupervised prompting only) | Decreased | Preference optimization is critical for improving segmentation quality |
| w/o VQA (CLIP prompt only) | Decreased | Shape/location priors provided by VQA contribute positively |
| w/o Virtual Annotator (using random preferences) | Significantly decreased | Quality judgment of the virtual annotator is key to the effectiveness of DPO |

### Key Findings

- The quality of unsupervised prompt generation is unexpectedly high, validating that CLIP's cross-modal semantic alignment capabilities remain effective on medical images.
- Compared to traditional semi-supervised methods, the main advantage of DPO preference optimization is that it bypasses the need for precise pseudo-labels, requiring only a relative judgment of "which is better."
- The design of the virtual annotator is a crucial component for the method's effectiveness—random preference pairs yield almost no training benefit.
- Consistent improvements across the three modalities indicate strong generalizability of the framework, which does not rely on modality-specific prior knowledge.

## Highlights & Insights

- **Introducing DPO to medical segmentation represents a novel cross-domain transfer**: DPO, originally used for LLM alignment, is resourcefully adapted here for segmentation tasks, downgrading the annotation requirement from "pixel-level labels" to "rankings/scores," which significantly reduces annotation costs.
- The concept of using **CLIP + VQA for unsupervised prompt generation** can be transferred to other domains requiring SAM where annotations are scarce (e.g., remote sensing, industrial inspection).
- The design of **replacing human annotators with a virtual annotator** further reduces the cost of RLHF to zero human labor, presenting a valuable exploration into automated alignment.

## Limitations & Future Work

- The quality upper bound of the virtual annotator is constrained by its heuristic-designed metrics; if these metrics fail to accurately reflect segmentation quality, the direction of DPO optimization may drift.
- CLIP’s capacity to understand medical images is limited—generic CLIP models may not localize highly specialized medical terminology and rare lesions precisely, suggesting that medical-specific pre-trained models like BiomedCLIP could be considered.
- The gap between this semi-supervised setup and fully supervised methods has not been quantitatively clarified; it is necessary to study performance curves under varying annotation ratios.
- 3D medical segmentation scenarios (e.g., 3D CT/MRI volumes) have not been explored, with current validation limited to 2D slices.

## Related Work & Insights

- **vs MedSAM**: MedSAM fine-tunes SAM with large-scale medical data and still relies heavily on annotations. In contrast, the proposed method optimizes prompts and policies instead of fine-tuning SAM itself.
- **vs SAMed**: SAMed utilizes LoRA to adapt SAM to the medical domain, which is a supervised method. The innovation of this work lies in combining semi-supervised learning with DPO.
- **vs Traditional Semi-Supervised Segmentation (e.g., Mean Teacher, FixMatch)**: Traditional methods rely on consistency regularization of pseudo-labels, whereas this work alternates it with preference optimization, avoiding the accumulation of pseudo-label noise.
- The DPO + virtual annotator paradigm in this framework could inspire applications of other vision foundation models (such as DINO, GroundingDINO) in scenarios with scarce annotations.

## Rating

- Novelty: ⭐⭐⭐⭐ Integrating DPO preference optimization into medical segmentation is an ingenious cross-domain transfer.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three modalities and three tasks extensively, though concrete numerical tables are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and intuitive methodology pipeline.
- Value: ⭐⭐⭐⭐ Offers practical value for low-annotation medical segmentation; the DPO + virtual annotator paradigm is highly transferrable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)
- [\[CVPR 2025\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2025\] Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline](interactive_medical_image_segmentation_a_benchmark_dataset_and_baseline.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Noise-Consistent Siamese-Diffusion for Medical Image Synthesis and Segmentation](noise-consistent_siamese-diffusion_for_medical_image_synthesis_and_segmentation.md)

</div>

<!-- RELATED:END -->
