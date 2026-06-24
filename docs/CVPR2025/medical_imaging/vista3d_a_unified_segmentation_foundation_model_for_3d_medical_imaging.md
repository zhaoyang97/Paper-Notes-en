---
title: >-
  [Paper Note] VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging
description: >-
  [CVPR 2025][Medical Imaging][3D Medical Segmentation] This paper proposes VISTA3D, the first unified 3D medical image segmentation foundation model. It simultaneously supports automatic segmentation of 127 classes, 3D interactive editing, and zero-shot segmentation. By utilizing a 3D supervoxel technology distilled from SAM, VISTA3D achieves state-of-the-art (SOTA) zero-shot performance and matches or exceeds specialized expert models on 14 datasets.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "3D Medical Segmentation"
  - "Foundation Model"
  - "Interactive Segmentation"
  - "Zero-Shot Segmentation"
  - "Supervoxel Distillation"
date: 2026-05-08
content_hash: 9ebc8f5a07a42062
---

# VISTA3D: A Unified Segmentation Foundation Model For 3D Medical Imaging

**Conference**: CVPR 2025  
**arXiv**: [2406.05285](https://arxiv.org/abs/2406.05285)  
**Code**: [github](https://github.com/Project-MONAI/VISTA)  
**Area**: Medical Imaging / Segmentation  
**Keywords**: 3D Medical Segmentation, Foundation Model, Interactive Segmentation, Zero-Shot Segmentation, Supervoxel Distillation

## TL;DR

This paper proposes VISTA3D, the first unified 3D medical image segmentation foundation model. It simultaneously supports automatic segmentation of 127 classes, 3D interactive editing, and zero-shot segmentation. By utilizing a 3D supervoxel technology distilled from SAM, VISTA3D achieves state-of-the-art (SOTA) zero-shot performance and matches or exceeds specialized expert models on 14 datasets.

## Background & Motivation

1. **Background**: There are strong automatic segmentation models for 3D medical imaging (such as TotalSegmentator which protects 117+ classes), as well as interactive segmentation methods based on SAM. However, existing solutions either support only automatic segmentation (lacking zero-shot capability) or only interactive segmentation (with lower accuracy than expert models), lacking a unified solution.

2. **Limitations of Prior Work**: (a) Automatic models like TotalSegmentator lack zero-shot capabilities and are helpless when encountering unseen classes (such as rare lesions or animal data); (b) Directly applying SAM/SAM2 to 3D medical images yields poor results because the requirement for spatial consistency across 3D slices is fundamentally different from temporal tracking across video frames; (c) Slice-by-slice 2D annotation is extremely time-consuming for 3D volume data; (d) In-context learning/open-vocabulary segmentation accuracy is significantly lower than that of expert models.

3. **Key Challenge**: A foundation model for 3D medical segmentation needs to achieve both "high-precision automatic segmentation" (requiring heavy supervised training) and "zero-shot generalization" (requiring 2D pre-trained knowledge). However, the training strategies for these two goals are intrinsically in conflict—3D convolutional networks excel at the former, while 2D ViT-based models excel at the latter.

4. **Goal**: To build a unified model that simultaneously achieves SOTA status in 3D automatic segmentation (127 classes), 3D interactive segmentation, and 3D zero-shot segmentation.

5. **Key Insight**: Build the backbone based on a mature 3D segmentation pipeline (SegResNet + sliding window inference), distill 2D SAM knowledge into the 3D space via an innovative 3D supervoxel method, and design a dual-branch architecture to handle automatic and interactive segmentation separately.

6. **Core Idea**: A dual-branch shared encoder architecture (automatic branch uses class embeddings as prompts, while the interactive branch uses 3D clicks for editing) + 3D supervoxels generated from SAM feature maps to train the interactive branch and obtain zero-shot capabilities.

## Method

### Overall Architecture

The input is a 3D CT volume image. Based on the user-provided class index (automatic segmentation) or 3D click coordinates (interactive segmentation), the model outputs binary segmentation results. The model consists of a shared SegResNet encoder and two independent decoder branches. A four-stage training strategy is adopted: (1) train the interactive branch -> (2) fine-tune the interactive branch -> (3) train the automatic branch -> (4) fine-tune the automatic branch. The dataset contains 11,454 CT scans, including manual annotations, pseudo-labels, and SAM supervoxels.

### Key Designs

1. **Dual-Branch Shared Encoder Architecture**:
    - **Function**: To support both high-precision automatic segmentation and flexible interactive segmentation within a single model.
    - **Mechanism**: The SegResNet encoder is shared between the two branches. The automatic branch utilizes learnable $N \times C$ class embeddings $E_c$. Given a class index $i$, the output is computed as $sigmoid(M(E_c[i]) \times F)$, where $F$ is the output feature of the decoder and $M$ is an MLP. The interactive branch is based on SAM's point encoder, taking 3D click coordinates and positive/negative labels, and utilizing a cross-attention transformer to generate the output. Zero-shot embeddings and ambiguity resolution embeddings are also introduced to handle special cases.
    - **Design Motivation**: Promptable automatic segmentation (outputting a binary mask for only one class at a time) significantly reduces memory consumption compared to multi-class softmax outputs and naturally resolves the training issue of partially labeled datasets. The shared encoder ensures both branches benefit from the same strong feature representations.

2. **3D Supervoxel Generation (SAM Feature Distillation)**:
    - **Function**: Distill SAM's 2D image understanding capability into 3D space to provide large-scale diverse training data for the interactive branch.
    - **Mechanism**: For each 3D CT scan, the outputs of the SAM ViT encoder and mask decoder's scaling layers are extracted slice-by-slice along three orientations: axial, coronal, and sagittal, obtaining upsampled 2D SAM feature maps. The features from these three directions are accumulated into a 3D feature volume $F_{3D} = F_A + F_C + F_S$. Then, the SLIC superpixel algorithm is run on this 3D feature space to generate supervoxels (100 segments, sigma=3). Supervoxels are generated for all 11,454 CT scans for training.
    - **Design Motivation**: SAM’s zero-shot capability stems from training on 11 million fully annotated images—an annotated scale that is impossible to obtain for 3D medical images. The supervoxel distillation method bypasses this limitation, utilizing SAM's pre-learned priors on "what constitutes an object" to generate 3D zero-shot training data without fine-tuning SAM. Experiments show that this approach is significantly better than graph-cut-based low-level feature supervoxels (such as SegVol).

3. **Interactive Refinement Algorithm (Alg. 1)**:
    - **Function**: Refine the automatic segmentation results using interactive clicks without destroying correctly segmented regions.
    - **Mechanism**: The difference regions between the automatic and interactive results are computed and decomposed into 3D connected components. Addition or subtraction operations are performed exclusively on the connected components containing the user's clicks, leaving other regions unaffected. The added connected components corresponding to positive clicks are integrated into the final results, while the deleted connected components corresponding to negative clicks are removed.
    - **Design Motivation**: Directly overwriting automatic results with interactive outputs can destroy previously correct regions (a limitation identified in FocalClick). The refinement strategy based on connected components limits modifications locally and precisely.

### Loss & Training

A four-stage training strategy is employed:
- **Stage 1** (Interactive Branch Training): Generous iterations, mixing manual/pseudo annotations and supervoxel training, using SAM-style iterative training (5 iterations sampling false positive/negative regions).
- **Stage 2** (Interactive Branch Fine-tuning): Over-sampling datasets to address class imbalance, omitting supervoxels and unlabeled data.
- **Stage 3** (Automatic Branch Training): Freezing the encoder, training only the automatic branch's decoder and class heads.
- **Stage 4** (Automatic Branch Fine-tuning): Leveraging MAISI synthesized data to augment rare categories (tumors/lesions).

Training uses 128-cubic patches and sliding-window inference, optimized with standard BCE and Dice losses.

## Key Experimental Results

### Main Results

14 datasets, 127-class automatic segmentation (average Dice):

| Method | MSD09 Spleen | BTCV Abdomen | Bone Lesion | Lung Tumor | No. of Classes |
|------|---------|---------|--------|--------|-------|
| nnUNet (Expert) | 0.967 | 0.807 | 0.396 | 0.554 | Task-by-task |
| Auto3DSeg (Expert) | 0.965 | 0.807 | 0.343 | 0.562 | Task-by-task |
| TotalSegmentator | 0.966 | - | - | - | 117 |
| **VISTA3D auto** | 0.952 | ~0.80 | **0.491** | **0.613** | **127** |
| **VISTA3D auto+point** | **0.954** | ~0.82 | **0.585** | **0.719** | **127** |

### Ablation Study

| Configuration | Zero-shot Dice↑ | Explanation |
|------|-----------|------|
| Without supervoxel training | ~0.30 | Zero-shot capability is almost non-existent |
| Graph-cut supervoxels (SegVol-style) | ~0.38 | Low-level feature supervoxels |
| **SAM feature supervoxels (VISTA3D)** | **~0.57** | SAM semantic feature supervoxels, +50% improvement |

### Key Findings

- VISTA3D is the only model that achieves automatic segmentation accuracy comparable to nnUNet/TotalSegmentator while supporting interactive and zero-shot segmentation.
- SAM feature-distilled 3D supervoxels are critical for zero-shot capabilities—improving performance by more than 50% compared to graph-cut supervoxels.
- With only 1 click of correction, VISTA3D auto+point outperforms expert models on several tasks.
- Randomly initialized class embeddings slightly outperform CLIP embeddings, indicating that text embeddings provide no advantage for closed-vocabulary automatic segmentation.
- Inference speed is significantly faster than TotalSegmentator (which uses a 5-model ensemble).

## Highlights & Insights

- The **phased training strategy** is highly practical: training the encoder and interactive branch first to obtain general features -> freezing the encoder and training the automatic branch to prevent catastrophic forgetting -> fine-tuning to resolve long-tail classes. This strategy holds general value for multi-task foundation models.
- Using **supervoxels as the medium for SAM knowledge distillation** is a core innovation: instead of fine-tuning SAM weights, its feature maps are used to perform 3D clustering -> yielding training labels -> training a native 3D model. This avoids the performance bottleneck of 2D -> 3D adaptors.
- The finding that **learnable class embeddings outperform CLIP embeddings** is significant—suggesting that task-specific learning is more effective than pre-trained semantic representations in closed-set automatic segmentation scenarios.

## Limitations & Future Work

- Only CT modality is supported; it has not been expanded to other 3D medical modalities such as MRI.
- While 127 classes are extensive, they still represent a closed set, and true open-vocabulary segmentation has not been achieved.
- The quality of supervoxel training data is constrained by SAM's performance on medical images.
- The four-stage training process is relatively complex and requires careful hyperparameter tuning for each stage.
- Future work could explore incorporating open-vocabulary capabilities and MRI support.

## Related Work & Insights

- **vs TotalSegmentator**: Both are 3D automatic segmentation foundation models. However, VISTA3D additionally supports interactive editing and zero-shot capabilities, using a single model (vs. a 5-model ensemble).
- **vs MedSAM**: MedSAM fine-tunes SAM for 2D medical segmentation, lacking 3D support and automatic segmentation capabilities.
- **vs SegVol**: SegVol also supports 3D semantic and interactive segmentation, but its automatic segmentation performance lags behind, and its graph-cut supervoxels are inferior to SAM feature-based supervoxels.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The strategy of distilling SAM features to generate 3D supervoxels is highly creative, and the dual-branch unified architecture is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Thorough evaluation across 14 datasets and 127 classes against 3 strong baselines with clear ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Workflow diagrams and architectural designs are clear, and the motivations behind the four-stage training are well-justified.
- **Value**: ⭐⭐⭐⭐⭐ Immense practical contribution to the 3D medical image segmentation community, open-sourced and deployment-ready.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation](vesselfm_a_foundation_model_for_universal_3d_blood_vessel_segmentation.md)
- [\[CVPR 2025\] Revisiting MAE Pre-Training for 3D Medical Image Segmentation](revisiting_mae_pre-training_for_3d_medical_image_segmentation.md)
- [\[NeurIPS 2025\] BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals](../../NeurIPS2025/medical_imaging/brainomni_a_brain_foundation_model_for_unified_eeg_and_meg_signals.md)
- [\[CVPR 2025\] UNIStainNet: Foundation-Model-Guided Virtual Staining of H&E to IHC](unistainnet_foundation-model-guided_virtual_staining_of_he_to_ihc.md)
- [\[CVPR 2025\] OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection](openmibood_open_medical_imaging_benchmarks_for_out-of-distribution_detection.md)

</div>

<!-- RELATED:END -->
