---
title: >-
  [Paper Note] OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation
description: >-
  [NeurIPS 2025][Segmentation][Multi-modal pretraining] OmniSegmentor constructs a large-scale ImageNeXt dataset encompassing 5 visual modalities (1.2M samples), proposes an efficient pretraining strategy that randomly selects one supplementary modality to align with RGB per iteration, and establishes the first flexible multi-modal pretrain-finetune pipeline, achieving state-of-the-art results on 6 multi-modal semantic segmentation benchmarks.
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Multi-modal pretraining"
  - "ImageNeXt"
  - "Semantic Segmentation"
  - "Modality Alignment"
  - "Pretrain-Finetune"
date: 2026-05-08
content_hash: d97e3f68a79183ba
---

# OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation

**Conference**: NeurIPS 2025  
**arXiv**: [2509.15096](https://arxiv.org/abs/2509.15096)  
**Code**: [https://github.com/VCIP-RGBD/DFormer](https://github.com/VCIP-RGBD/DFormer)  
**Area**: Image Segmentation  
**Keywords**: Multi-modal pretraining, ImageNeXt, Semantic Segmentation, Modality Alignment, Pretrain-Finetune

## TL;DR
OmniSegmentor constructs a large-scale ImageNeXt dataset encompassing 5 visual modalities (1.2M samples), proposes an efficient pretraining strategy that randomly selects one supplementary modality to align with RGB per iteration, and establishes the first flexible multi-modal pretrain-finetune pipeline, achieving state-of-the-art results on 6 multi-modal semantic segmentation benchmarks.

## Background & Motivation

**Background**: Multi-modal semantic segmentation (RGB + depth/thermal/LiDAR/event camera) has become a key direction for robust scene understanding. Existing methods such as CMX and CMNeXt integrate multi-modal information through cross-modal interaction and fusion.

**Limitations of Prior Work**: Most methods rely on RGB-pretrained or randomly initialized weights for supplementary modalities, resulting in a modality mismatch between the pretraining and fine-tuning stages. Although DFormer attempts to address this via RGB-D pretraining, it is modality-specific and cannot generalize to additional modalities.

**Key Challenge**: There is no large-scale multi-modal dataset to support general multi-modal pretraining, and simultaneously pretraining on all modalities is computationally prohibitive and optimization-unfriendly — experiments show that joint pretraining leads to convergence difficulties, with ImageNet Top-1 accuracy dropping from 81.4% to 79.9%.

**Goal**: (a) How to construct a large-scale pretraining dataset covering multiple visual modalities? (b) How to conduct multi-modal pretraining efficiently? (c) How to flexibly deploy pretrained weights to diverse downstream multi-modal tasks?

**Key Insight**: The authors observe that inputting all modalities simultaneously introduces optimization conflicts, while RGB remains the most critical modality. They therefore propose randomly selecting one supplementary modality per iteration to pair with RGB during pretraining, rather than processing all modalities jointly.

**Core Idea**: Pretraining on ImageNeXt (a 5-modality synthetic dataset) by randomly pairing RGB with one supplementary modality, followed by flexible fine-tuning via modality-specific MLPs that support arbitrary modality combinations.

## Method

### Overall Architecture
OmniSegmentor introduces two main contributions: (1) the **ImageNeXt** dataset — synthesizing depth, thermal, LiDAR, and event camera modalities for every ImageNet-1K image, yielding 1.2M training samples across 5 modalities; and (2) an **efficient pretrain-finetune pipeline** — during pretraining, the model receives RGB paired with one randomly selected supplementary modality; during fine-tuning, pretrained weights are loaded and extended into a multi-modal encoder with a lightweight decode head.

### Key Designs

1. **ImageNeXt Dataset Construction**:

    - **Function**: Synthesize 4 supplementary modality data for each RGB image in ImageNet.
    - **Mechanism**: Depth maps are generated using the Omnidata estimation model; event data are sourced from N-ImageNet (event camera recordings of ImageNet images displayed on a monitor); LiDAR data are produced via pseudo-LiDAR methods from synthetic depth maps and converted to range-view format; thermal images are synthesized using a thermal estimation model based on AdaBins, trained on RGB-T datasets.
    - **Design Motivation**: To resolve the fundamental bottleneck of lacking large-scale multi-modal data for representation learning, given that existing datasets such as SUNRGBD (10K samples) and KITTI-360 (60K samples) are limited in both scale and modality coverage.

2. **Efficient Multi-Modal Pretraining Strategy (ImageNeXt Pretraining)**:

    - **Function**: At each training iteration, only RGB and one randomly selected supplementary modality are fed as input for feature alignment.
    - **Mechanism**: Building upon DFormer's block design, a fusion module aggregates RGB and the selected modality features, with independent MLPs encoding each modality. The fusion module weights are shared across modalities, while each modality has its own stem layer and MLP.
    - **Design Motivation**: Simultaneously pretraining all modalities increases parameters from 39.0M to 48.7M, FLOPs from 14.7G to 21.8G, training time from 78.9h to 180.5h, and degrades Top-1 accuracy to 79.9%. The random selection strategy allows each supplementary modality to participate in pretraining while avoiding inter-modality optimization conflicts, achieving a Top-1 of 83.0%.

3. **Flexible Multi-Modal Fine-Tuning**:

    - **Function**: Deploy pretrained weights to downstream segmentation tasks involving arbitrary modality combinations.
    - **Mechanism**: During fine-tuning, each supplementary modality is assigned an independent stem layer and MLP (initialized from pretrained supplementary modality weights). Features from all modalities are aggregated (simple addition + LayerNorm) before fusion with RGB features, followed by a Ham decode head for segmentation prediction.
    - **Design Motivation**: Modality-specific encoding (separate MLPs) outperforms shared MLPs by 0.9% mIoU on EventScape RGB-D-E (67.6 vs. 66.7), with only a 2.9M parameter increase and no additional FLOPs. Experiments further show that simple fusion achieves performance on par with complex fusion (SQ-Hub) after ImageNeXt pretraining, as pretraining aligns the feature distributions.

### Loss & Training
- **Pretraining**: Cross-entropy classification loss, AdamW optimizer, initial learning rate 6e-5, polynomial decay schedule.
- **Fine-tuning**: Cross-entropy segmentation loss; data augmentation includes random resizing (0.5–1.75), random horizontal flipping, and random cropping.
- Multi-scale inference is applied on selected datasets (NYU Depth V2, SUNRGBD).

## Key Experimental Results

### Main Results

| Dataset | Modality | Backbone | mIoU (%) | vs. Prev. SOTA |
|--------|------|----------|----------|-----------|
| NYU Depth V2 | RGB-D | DFormer-L | **57.6** | DFormer 57.2 (+0.4) |
| SUNRGBD | RGB-D | DFormer-L | **52.8** | DFormer 52.5 (+0.3) |
| MFNet | RGB-T | DFormer-L | **60.6** | CMNeXt 59.9 (+0.7) |
| KITTI-360 | RGB-L | DFormer-L | **69.2** | DFormer 66.3 (+2.9) |
| EventScape | RGB-D-E | DFormer-L | **67.6** | CMNeXt 63.9 (+3.7) |
| DeLiVER | RGB-D-E-L | DFormer-L | **68.0** | CMNeXt 66.3 (+1.7) |

### Ablation Study

| Pretraining Strategy | Params | FLOPs | Top-1 (%) | Training Time (h) |
|-----------|--------|-------|-----------|-------------|
| RGB-only | 39.0M | 14.7G | 81.4 | 69.5 |
| All modalities simultaneously | 48.7M | 21.8G | 79.9 | 180.5 |
| Ours (random selection) | 39.0M | 14.7G | **83.0** | 78.9 |

| Missing Modality in Pretraining | NYU V2 (RGB-D) | MFNet (RGB-T) | KITTI (RGB-L) | EventScape (RGB-E) |
|---------------|----------------|---------------|---------------|-------------------|
| All 5 modalities | 54.3 | 57.6 | 64.6 | 61.8 |
| w/o Depth | 52.2 | 57.5 | 64.6 | 61.6 |
| w/o Event | 54.2 | 57.6 | 64.5 | **60.5** |
| w/o LiDAR | 54.3 | 57.7 | **61.2** | 61.9 |
| RGB only | 50.9 | 55.6 | 60.1 | 58.7 |

### Key Findings
- Omitting a specific modality during pretraining leads to the most notable performance drop on the corresponding downstream task (e.g., removing Event pretraining degrades RGB-E from 61.8 to 60.5), confirming that ImageNeXt pretraining gains are directly attributable to the corresponding modality data.
- OmniSegmentor's advantage grows with the number of modalities: gains on EventScape increase from RGB-E (+0.7) to RGB-D-E (+2.6).
- The largest improvement is observed on KITTI-360 (+2.9 mIoU), indicating that the LiDAR modality benefits most from pretraining.
- Using the same MiT-B2 backbone, OmniSegmentor even surpasses competing methods that use the larger MiT-B4 backbone.

## Highlights & Insights
- **Practical data synthesis strategy for ImageNeXt**: Leveraging existing estimation models (Omnidata, N-ImageNet) to generate multi-modal data from ImageNet avoids costly real-world multi-modal data collection. This "synthesize supplementary modalities" paradigm is transferable to other multi-modal learning scenarios.
- **Random selection outperforms joint input**: This counterintuitive finding reveals optimization conflicts among modalities during multi-modal pretraining — "less is more," as aligning two modalities per step is more stable than aligning five simultaneously.
- **Simple fusion ≈ complex fusion (post-alignment)**: Once pretraining aligns the feature distributions, fine-tuning no longer requires sophisticated attention-based fusion; simple addition suffices. This demonstrates that strong pretraining can simplify downstream architecture design.

## Limitations & Future Work
- All supplementary modality data are synthetically generated (estimated/simulated), introducing a domain gap relative to real sensor data; the paper does not evaluate the impact of this gap.
- The thermal estimation model is trained on only four small-scale RGB-T datasets, which may limit synthesis quality.
- Only the supervised classification pretraining → segmentation fine-tuning paradigm is explored; self-supervised or unsupervised pretraining approaches are not investigated.
- Existing evaluation benchmarks cover limited modality combinations, and no real-world dataset simultaneously encompasses all 5 modalities.

## Related Work & Insights
- **vs. DFormer**: DFormer performs modality-specific RGB-D pretraining; OmniSegmentor extends this to a modality-agnostic multi-modal pretraining framework while directly reusing DFormer's building block design.
- **vs. CMNeXt**: CMNeXt is a general multi-modal segmentation framework but relies on RGB-pretrained weights; OmniSegmentor surpasses it across all settings through ImageNeXt pretraining.
- **vs. MultiMAE**: MultiMAE also performs multi-modal pretraining (RGB-D) but adopts a MAE-based self-supervised paradigm and is limited to dual-modality input.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First unified pretraining framework across 5 modalities, though the core architecture reuses DFormer's design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation on 6 benchmarks with extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-motivated design choices, rich figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides a reusable pretraining paradigm and a large-scale dataset for multi-modal segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation](../../CVPR2026/segmentation/towards_robust_multi-modal_semantic_segmentation_with_teacher-student_framework_.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)
- [\[NeurIPS 2025\] Torch-Uncertainty: A Deep Learning Framework for Uncertainty Quantification](torch-uncertainty_a_deep_learning_framework_for_uncertainty_quantification.md)
- [\[NeurIPS 2025\] Diffusion-Driven Two-Stage Active Learning for Low-Budget Semantic Segmentation](diffusion-driven_two-stage_active_learning_for_low-budget_semantic_segmentation.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)

</div>

<!-- RELATED:END -->
