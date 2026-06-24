---
title: >-
  [Paper Note] Show and Segment: Universal Medical Image Segmentation via In-Context Learning
description: >-
  [CVPR 2025][Medical Imaging][Medical Image Segmentation] Iris is proposed as a framework that extracts task embeddings from reference image-label pairs using a lightweight task encoding module to guide target image segmentation. It adapts to new tasks without fine-tuning, achieving or exceeding the performance of task-specific models across 12 datasets, while demonstrating excellent generalization capabilities on 7 unseen datasets.
tags:
  - "CVPR 2025"
  - "Medical Imaging"
  - "Medical Image Segmentation"
  - "In-Context Learning"
  - "Universal Segmentation"
  - "Task Encoding"
  - "Few-Shot Segmentation"
date: 2026-05-08
content_hash: 751d17521b2447ea
---

# Show and Segment: Universal Medical Image Segmentation via In-Context Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.19359](https://arxiv.org/abs/2503.19359)  
**Code**: None  
**Area**: Medical Image  
**Keywords**: Medical Image Segmentation, In-Context Learning, Universal Segmentation, Task Encoding, Few-Shot Segmentation

## TL;DR

Iris is proposed as a framework that extracts task embeddings from reference image-label pairs using a lightweight task encoding module to guide target image segmentation. It adapts to new tasks without fine-tuning, achieving or exceeding the performance of task-specific models across 12 datasets, while demonstrating excellent generalization capabilities on 7 unseen datasets.

## Background & Motivation

**Background**: Medical image segmentation methods are generally categorized into four types: task-specific models (e.g., nnUNet), which perform well but are not scalable; multi-task universal models (e.g., UniSeg), which can handle various tasks but require fine-tuning for unseen classes; interactive models (e.g., SAM-like), which require multiple manual prompts; and current in-context learning (ICL) methods (e.g., UniverSeg, Tyche), which exhibit suboptimal performance compared to dedicated models and suffer from low computational efficiency.

**Limitations of Prior Work**: (1) Task-specific models cannot handle unseen classes; (2) Universal models still require fine-tuning; (3) SAM requires multiple interactions, especially for 3D structures; (4) Existing ICL methods re-encode the context for every inference, leading to low efficiency and poor performance.

**Key Challenge**: The contradiction between flexibility and performance—methods capable of adapting to new tasks perform worse than dedicated models, whereas high-performing methods cannot generalize.

**Goal**: (1) Achieve the performance of task-specific models on the training distribution; (2) Maintain strong generalization capabilities on unseen classes/out-of-distribution (OOD) data; (3) Realize efficient automated inference.

**Key Insight**: Decouple task encoding from inference—first extract a compact task embedding from the reference pairs (one-time overhead), and then reuse it across any number of query images.

**Core Idea**: Design a lightweight task encoding module that encodes both foreground and context information from reference image-label pairs into task embedding tokens. These tokens guide 3D query image segmentation through cross-attention, supporting multi-class single-forward inference.

## Method

### Overall Architecture

Iris consists of three components: a 3D UNet encoder (trained from scratch), a task encoding module, and a mask decoding module. The task encoding module extracts task embeddings from reference image-label pairs, and the decoding module generates segmentation masks for query images using the task embeddings through bidirectional cross-attention.

### Key Designs

1. **Foreground Feature Encoding**:

    - **Function**: Extract precise features from the region corresponding to the reference label.
    - **Mechanism**: Upsample the encoder feature $\mathbf{F}_s$ to the original resolution, multiply it element-wise with the high-resolution binary mask $\mathbf{y}_s$, and then perform pooling to obtain the foreground embedding $\mathbf{T}_f \in \mathbb{R}^{1 \times C}$. High-resolution masks are used instead of downsampled masks because many structures in medical images occupy only a few voxels.
    - **Design Motivation**: Downsampled masking loses fine boundary and small-structure details. Masking after upsampling ensures accurate ROI feature extraction.

2. **Context Feature Encoding**:

    - **Function**: Capture global context information to complement the foreground encoding.
    - **Mechanism**: PixelShuffle is utilized to expand features to high resolution while reducing channel dimensions, followed by concatenation with the mask, convolution, and PixelUnshuffle to return to the original resolution. The fused features interact with $m$ learnable query tokens via cross/self-attention, yielding the context embedding $\mathbf{T}_c \in \mathbb{R}^{m \times C}$. The final embedding is constructed as $\mathbf{T} = [\mathbf{T}_f; \mathbf{T}_c]$.
    - **Design Motivation**: PixelShuffle enables memory-efficient high-resolution feature-mask fusion.

3. **Mask Decoding Module**:

    - **Function**: Utilize task embeddings to guide multi-class segmentation.
    - **Mechanism**: Multi-class task embeddings are concatenated into $\mathbf{T} \in \mathbb{R}^{K(m+1) \times C}$. They interact with query features via bidirectional cross-attention to output $K$-class segmentations in a single forward pass.
    - **Design Motivation**: Improves efficiency by $K$ times compared to class-by-class inference.

### Loss & Training

A combination of Dice and cross-entropy loss is used. Episode training is adopted to simulate ICL scenarios. Training uses the Lamb optimizer with lr=$2\times10^{-3}$, 80K iterations, a batch size of 32, and a volume size of $128^3$. Data augmentation includes random cropping, affine transformation, intensity adjustment, and random perturbation.

## Key Experimental Results

### Main Results

| Method Category | Method | Average Dice(%) |
|---------|------|------------|
| Task-Specific | nnUNet | 83.18 |
| Multi-Task Universal | Multi-Talent | 84.47 |
| Interactive | SAM-Med3D | 68.42 |
| ICL | Tyche-IS | 61.20 |
| **ICL** | **Iris** | **84.52** |

### Ablation Study

| Dataset(OOD) | nnUNet-gen | UniSeg | Tyche | Iris |
|------------|-----------|--------|-------|------|
| ACDC | 82.06 | 84.98 | 74.91 | **86.45** |
| SegTHOR | 76.92 | 78.56 | 56.75 | **82.77** |
| MSD Pancreas (unseen) | — | — | 11.97 | **28.28** |
| Pelvic (unseen) | — | — | 61.92 | **69.03** |

### Key Findings

- Iris matches or exceeds the performance of task-specific and multi-task universal models on in-distribution datasets for the first time (84.52% vs 84.47%).
- 3D architecture is crucial—existing 2D ICL methods lag far behind (61.20% vs 84.52%).
- Object-level context retrieval outperforms image-level retrieval.
- Task embeddings can automatically reveal anatomical relationships across datasets.

## Highlights & Insights

- **Decoupling task encoding from inference** is the core design—task embeddings only need to be extracted once and can then be reused, which is far more efficient than re-encoding reference images during every inference step.
- **High-resolution foreground encoding** is crucial for small structures in medical images.
- **Multi-class single-forward pass** is achieved through task embedding concatenation, significantly improving efficiency.

## Limitations & Future Work

- Segmentation capability on unseen classes still falls short of nnUNet models trained directly on those target classes.
- The choice of reference images has a significant impact on performance, and the optimal selection strategy remains an open question.
- The 3D UNet is trained from scratch, without utilizing pre-trained vision foundation models.

## Related Work & Insights

- **vs nnUNet**: Almost on par in-distribution but demonstrates stronger out-of-distribution generalization, and does not require retraining.
- **vs UniverSeg**: Outperforms it through 3D architecture, task-inference decoupling, and multi-class single-forward inference.
- **vs SAM-Med3D**: Iris automatically defines tasks through reference pairs, making it more suitable for fully automated workflows.

## Rating

- Novelty: ⭐⭐⭐⭐ Clever design of decoupled task encoding and inference.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 19 datasets with four categories of methods.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure and high-quality charts.
- Value: ⭐⭐⭐⭐⭐ First ICL method to match dedicated models in-distribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] K-Prism: A Knowledge-Guided and Prompt Integrated Universal Medical Image Segmentation Model](../../ICLR2026/medical_imaging/k-prism_a_knowledge-guided_and_prompt_integrated_universal_medical_image_segment.md)
- [\[ECCV 2024\] I-MedSAM: Implicit Medical Image Segmentation with Segment Anything](../../ECCV2024/medical_imaging/i-medsam_implicit_medical_image_segmentation_with_segment_anything.md)
- [\[CVPR 2025\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2025\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation?](are_general-purpose_vision_models_all_we_need_for_2d_medical_image_segmentation_.md)
- [\[CVPR 2025\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](semantic_class_distribution_learning_for_debiasing_semi-supervised_medical_image.md)

</div>

<!-- RELATED:END -->
