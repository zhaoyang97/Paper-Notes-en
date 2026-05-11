---
title: >-
  [Paper Note] ZIM: Zero-Shot Image Matting for Anything
description: >-
  [ICCV 2025][Segmentation][Image Matting] This paper proposes ZIM, a zero-shot image matting model that constructs the SA1B-Matte dataset by converting SA1B segmentation labels into fine-grained matting labels via a label…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Image Matting"
  - "Zero-Shot"
  - "SAM"
  - "Label Conversion"
  - "Hierarchical Decoder"
date: 2026-05-08
content_hash: d38c858eeb7358ae
---

# ZIM: Zero-Shot Image Matting for Anything

**Conference**: ICCV 2025
**arXiv**: [2411.00626](https://arxiv.org/abs/2411.00626)
**Code**: [https://naver-ai.github.io/ZIM](https://naver-ai.github.io/ZIM)
**Area**: Image Segmentation
**Keywords**: Image Matting, Zero-Shot, SAM, Label Conversion, Hierarchical Decoder

## TL;DR

This paper proposes ZIM, a zero-shot image matting model that constructs the SA1B-Matte dataset by converting SA1B segmentation labels into fine-grained matting labels via a label converter. A hierarchical pixel decoder and a prompt-aware masked attention mechanism are further introduced to achieve micro-level fine-grained matting while preserving zero-shot generalization capability.

## Background & Motivation

SAM has achieved remarkable progress in zero-shot segmentation; however, its generated masks lack fine edge quality (e.g., hair strands, branches). Existing methods that extend SAM to matting (Matte-Any, Matting-Any) rely on fine-tuning with public matting datasets, which **contain only macro-level labels** (e.g., full human portraits). This causes fine-tuned models to lose zero-shot capability at the micro level—yielding macro-level outputs even when given micro-level prompts (catastrophic forgetting).

Root Cause:
- **SAM**: Strong zero-shot generalization, but coarse masks with checkerboard artifacts.
- **Existing matting models**: High-fidelity outputs, but poor zero-shot generalization.
- **Data bottleneck**: Large-scale micro-level matting annotation is prohibitively expensive.

The key insight of this paper is that a **label converter** can automatically transform the large-scale micro-level segmentation labels in SA1B into matting labels, enabling large-scale training data acquisition without manual annotation.

## Method

### Overall Architecture

ZIM is built upon the SAM architecture, comprising four components: (1) an image encoder (ViT-B, stride 16); (2) a prompt encoder; (3) a Transformer decoder (token-to-image and image-to-token cross-attention); and (4) an **improved hierarchical pixel decoder**. The two core contributions correspond to **data construction** and **network architecture**, respectively.

### Key Designs

1. **Label Converter**: Built upon MGMatting with a Hiera-base-plus backbone. It takes an image and a segmentation label as input and outputs a fine-grained matting label. Training data are sourced from six public matting datasets (20,591 natural images + 118,749 synthetic images). Coarse segmentation labels are derived from matting labels via thresholding, downsampling, Gaussian blurring, and dilation/erosion operations.

   Two key strategies address training challenges:
   - **Spatial Generalization Augmentation (SGA)**: Randomly crops identical regions from segmentation and matting labels, forcing the converter to handle incomplete or irregular input patterns, thereby improving generalization to micro-level segmentation labels.
   - **Selective Transformation Learning (STL)**: Not all objects require fine-grained matting (e.g., cars, tables). Coarse-grained object masks from ADE20K (187,063 masks) are incorporated, where the ground-truth matte equals the original segmentation label (i.e., no conversion), teaching the model to **selectively** refine only objects that require fine-grained processing.

   Training loss: $L = L_{l1} + \lambda L_{grad}$, where $L_{grad}$ denotes the gradient loss.

2. **Hierarchical Pixel Decoder**: The original SAM decoder contains only two transposed convolution layers (stride 4), which are prone to checkerboard artifacts. The proposed decoder adopts a multi-resolution feature pyramid design, generating feature maps at stride 2/4/8 from the input image. The image embeddings are progressively upsampled and concatenated with features at the corresponding resolution, yielding a high-resolution feature map at stride 2. **Only 10 ms of additional inference latency is incurred.**

3. **Prompt-Aware Masked Attention**:
   - Box prompt: Generates a binary attention mask $\mathcal{M}^b \in \{0, -\infty\}$ to constrain model attention within the box region.
   - Point prompt: Generates a soft attention mask $\mathcal{M}^p \in [0,1]$ based on a 2D Gaussian distribution (standard deviation $\sigma=21$).
   - Applied exclusively to token-to-image cross-attention layers (experiments show that applying it to image-to-token attention disrupts global feature capture).

### Loss & Training

- SA1B-Matte dataset: ~2.2M labels from SA1B (1% subset) are converted for training.
- Training loss follows the label converter: $L = L_{l1} + \lambda L_{grad}$ ($\lambda=10$).
- AdamW optimizer, lr=1e-5, cosine decay, 500K iterations.
- Fine-tuned from SAM pre-trained weights.

## Key Experimental Results

### Main Results (MicroMat-3K Benchmark, Box Prompt)

| Method | Prompt | Fine-grained SAD↓ | Fine-grained MSE↓ | Fine-grained Grad↓ | Coarse SAD↓ | Coarse MSE↓ |
|------|------|------|------|------|------|------|
| SAM | box | 36.086 | 11.057 | 14.867 | 3.516 | 1.044 |
| HQ-SAM | box | 124.262 | 42.457 | 13.673 | 8.458 | 2.733 |
| Matte-Any | box | 34.661 | 9.746 | 7.021 | 6.950 | 1.983 |
| Matting-Any | box | 246.214 | 68.372 | 19.185 | 109.639 | 23.780 |
| **ZIM (ours)** | **box** | **9.961** | **1.893** | **4.813** | **1.860** | **0.448** |

*ZIM outperforms all baselines by a substantial margin across all metrics, reducing MSE by 83% relative to SAM and 81% relative to Matte-Any.*

### Ablation Study (Component Analysis, Box Prompt)

| Attention | Decoder | Fine-grained SAD↓ | Fine MSE↓ | Fine Grad↓ | Coarse SAD↓ | Coarse MSE↓ |
|--------|--------|------|------|------|------|------|
| ✗ | ✗ | 13.623 | 2.718 | 6.516 | 2.071 | 0.474 |
| ✓ | ✗ | 13.198 | 2.504 | 6.445 | 2.049 | 0.471 |
| ✗ | ✓ | 11.074 | 2.094 | 5.401 | 2.069 | 0.487 |
| **✓** | **✓** | **9.961** | **1.893** | **4.813** | **1.860** | **0.448** |

*The two components are complementary: the hierarchical decoder primarily reduces Grad error (suppressing artifacts), while the attention mechanism further improves overall accuracy.*

### Key Findings

- **Downstream transferability**: Replacing SAM with ZIM in multiple downstream frameworks—including Matte-Any, HQ-SAM, Inpainting Anything, medical image segmentation, and 3D segmentation—consistently yields significant improvements.
- **Training data impact**: Training ZIM on public matting data causes MSE on MicroMat-3K to surge from 1.893 to 38.332, confirming the necessity of micro-level training data.
- **Domain shift**: ZIM performs poorly on traditional matting benchmarks (full-body portraits) under box prompts due to SAM's inherent prompt ambiguity; however, it surpasses all existing methods when dense multi-point prompts are used.

## Highlights & Insights

- **Data engineering innovation**: The label converter approach of converting segmentation labels into matting labels is both elegant and effective; the SGA and STL strategies are carefully designed.
- **Lightweight improvement with large gains**: The hierarchical decoder adds only 10 ms of latency while resolving the long-standing checkerboard artifact issue in SAM.
- **MicroMat-3K benchmark**: 3,000 high-quality micro-level matting annotations that fill a critical gap in zero-shot matting evaluation.
- **Strong practicality**: Supports multiple prompt modalities, including points, boxes, text, and strokes.

## Limitations & Future Work

- Inherits SAM's ambiguity under vague prompts (whole-object vs. part-level ambiguity).
- Cannot handle transparency estimation for transparent objects (e.g., glass, fire).
- The quality ceiling of the label converter is bounded by the coverage of source matting datasets.
- The potential of larger backbones (ViT-H) remains unexplored.

## Related Work & Insights

- The label conversion paradigm can be generalized to other dense prediction tasks (e.g., upgrading coarse annotations to fine-grained labels).
- The Selective Transformation Learning (STL) idea is applicable to any scenario requiring discrimination between objects that do and do not need refinement.
- The hierarchical decoder design serves as a plug-and-play improvement for the SAM family of models.

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel label conversion framework and dataset construction pipeline)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Zero-shot evaluation on 23 datasets, multiple downstream tasks, and extensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (High practical utility, filling the gap in zero-shot matting)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation](../../CVPR2026/segmentation/discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](../../CVPR2026/segmentation/discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](object-level_correlation_for_few-shot_segmentation.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](../../AAAI2026/segmentation/rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)

</div>

<!-- RELATED:END -->
