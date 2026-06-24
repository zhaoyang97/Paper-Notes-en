---
title: >-
  [Paper Note] DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs
description: >-
  [ECCV 2024][Segmentation][DenseNet] Revisiting the dense concatenation shortcut of DenseNet, this paper proposes RDNet (Revitalized DenseNet) through a systematic modernization (wider and shallower architecture, modernized blocks, enlarged intermediate dimensions, more transition layers, etc.). RDNet outperforms Swin Transformer, ConvNeXt, and DeiT-III on ImageNet-1K, proving the powerful potential of concatenation as an underestimated paradigm.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "DenseNet"
  - "Dense connection"
  - "Feature concatenation"
  - "ConvNeXt"
  - "Modernization"
date: 2026-05-08
content_hash: b0f4ac3780a19a59
---

# DenseNets Reloaded: Paradigm Shift Beyond ResNets and ViTs

**Conference**: ECCV 2024  
**arXiv**: [2403.19588](https://arxiv.org/abs/2403.19588)  
**Code**: [https://github.com/naver-ai/rdnet](https://github.com/naver-ai/rdnet)  
**Area**: Network Architecture Design  
**Keywords**: DenseNet, Dense connection, Feature concatenation, ConvNeXt, Modernization  

## TL;DR
Revisiting the dense concatenation shortcut of DenseNet, this paper proposes RDNet (Revitalized DenseNet) through a systematic modernization (wider and shallower architecture, modernized blocks, enlarged intermediate dimensions, more transition layers, etc.). RDNet outperforms Swin Transformer, ConvNeXt, and DeiT-III on ImageNet-1K, proving the powerful potential of concatenation as an underestimated paradigm.

## Background & Motivation
- The additive shortcut of ResNet has become the standard in modern vision architectures (Swin, ConvNeXt, ViT, etc.), while the concatenation shortcut of DenseNet has been gradually forgotten.
- DenseNet initially outperformed ResNet, but lost momentum due to memory issues (feature dimension expansion caused by concatenation) and difficulties in width scaling.
- The authors' core hypothesis: **Concatenation is a more effective way to increase the matrix rank**. For $f(XW)$, the rank of the concatenation $[X, f(XW)]$ is $\text{rank}(X) + \text{rank}(f(XW))$, whereas the rank of the addition $X + f(XW)$ does not exceed $\text{rank}(X)$.
- Pilot study: After sampling 15k+ networks on Tiny-ImageNet for comparison, the concatenation shortcut achieved an average accuracy of **56.8±3.9** vs **55.9±4.1** for the additive shortcut.

## Core Problem
Is DenseNet's dense concatenation shortcut truly inferior to the ResNet-style additive shortcut? Can DenseNet return to the SOTA through modernized design?

## Method

### Overall Architecture
Starting from DenseNet-201, a series of 7 modernization stages are applied step-by-step to obtain the RDNet model family. Each stage consists of several mixing blocks, where each mixing block contains 3 feature mixers and 1 transition layer. The dense concatenation shortcut is retained as a core principle.

### Key Designs

1. **Wider & Shallower Architecture and Dense Connection Refactoring**:
    - Function: Redesign the macro-architecture of DenseNet to balance the feature reuse advantage of dense connections with computational efficiency.
    - Mechanism: Significantly increase the growth rate (GR) from 32 to 60+, and reduce the depth from (6,12,48,32) to (3,3,12,3). Decouple the expansion ratio (ER) from the GR by binding it to the input dimension, allowing the intermediate dimensions to grow naturally as the network goes deeper. Different stages use different GRs (e.g., 64, 104, 128, 224). Additionally, insert stride=1 transition layers after every 3 blocks within a stage for dimension reduction (without downsampling) to effectively control the channel expansion problem caused by dense concatenation.
    - Design Motivation: The original narrow-and-deep design of DenseNet led to slow training and high memory usage. After making it wider and shallower, the training speed improved by ~35% and memory reduced by 18% with almost no accuracy drop. Frequent transition layers further suppress channel explosion, allowing the use of larger GRs to enhance expressiveness.

2. **Modernized Block Design**:
    - Function: Upgrade DenseNet's feature mixing module to a modern convolutional network style.
    - Mechanism: Adopt a ConvNeXt-style design—Layer Norm instead of Batch Norm, post-activation, 7×7 depthwise separable convolutions, and fewer normalization/activation layers. The stem uses patchification with patch_size=4 and stride=4. Transition layers remove average pooling, replacing it with convolutions and LN. Integrate a channel re-scaling mechanism combining channel layer-scale and squeeze-and-excitation.
    - Design Motivation: The original BN + pre-activation design of DenseNet is outdated. Modernizing the block design brings ~1.3pp accuracy improvement, while the patchification stem accelerates computation without sacrificing accuracy.

### RDNet Model Family Configurations

| Model | GR | Blocks (B) |
|------|-----|-----------|
| RDNet-T | (64,104,128,224) | (3,3,12,3) |
| RDNet-S | (64,128,128,240) | (3,3,21,6) |
| RDNet-B | (96,128,168,336) | (3,3,21,6) |
| RDNet-L | (128,192,256,360) | (3,3,24,6) |

## Key Experimental Results

### ImageNet-1K Classification

| Model | Params (M) | FLOPs (G) | Top-1 Acc (%) |
|------|-----------|----------|--------------|
| ConvNeXt-T | 29 | 4.5 | 82.1 |
| Swin-T | 28 | 4.5 | 81.3 |
| DeiT-III-S | 22 | 4.6 | 81.4 |
| **RDNet-T** | ~25 | ~5.0 | **82.8** |
| ConvNeXt-S | 50 | 8.7 | 83.1 |
| Swin-S | 50 | 8.7 | 83.0 |
| **RDNet-S** | ~40 | ~8.7 | **83.7** |
| ConvNeXt-B | 89 | 15.4 | 83.8 |
| Swin-B | 88 | 15.4 | 83.5 |
| **RDNet-B** | ~87 | ~15.4 | **84.4** |
| **RDNet-L** | ~186 | ~34.7 | **84.8** |

### Progressive Modernization Effects (From DenseNet-201 to RDNet)

| Step | Top-1 (%) | Training Latency (ms) | Training Memory (GB) |
|------|-----------|-------------|------------|
| (a) DenseNet-201 baseline | 79.7 | 131 | 3.9 |
| (b) +Wider & shallower | 79.5 | 85 (-35%) | 3.2 (-18%) |
| (c) +Modernized blocks | 80.4 | - | - |
| (d) +Larger intermediate dims | 80.8 | - | - |
| (e) +More transition layers | 81.2 | - | - |
| (f) +Patchification | ~81.2 | - | - |
| (g) +Refined transition | ~81.4 | - | - |
| (h) +Channel re-scaling | ~81.6 | - | - |

### Downstream Tasks

- **ADE20K Semantic Segmentation**: Used as a backbone under the UperNet framework, RDNet outperforms ConvNeXt and Swin.
- **COCO Object Detection / Instance Segmentation**: As a backbone under the Mask R-CNN framework, RDNet shows strong competitiveness.

### Ablation Study
- **Width vs Depth**: Prioritizing width (GR) expansion is more efficient than increasing depth.
- **Expansion ratio**: Binding the ER to the input dimension (instead of GR) is the optimal choice; ER=4 yields the best performance.
- **Transition Interval**: Frequent transition layers (every 3 blocks) generally outperform using them only between stages.
- **Uniform vs Variable GR**: Using different GRs for different stages is superior to a uniform GR.
- **Transition ratio**: 0.5 (the original DenseNet setting) remains optimal; a higher ratio hurts accuracy.

## Highlights & Insights
- **Courage to challenge the mainstream paradigm**: In an era where additive residual connections have become a "consensus", this work proves that concatenation shortcuts are equally, if not more, powerful.
- Provides theoretical support for why concatenation outperforms addition based on matrix rank theory, backed by empirical validation from a 15k-network pilot study.
- A clear step-by-step modernization roadmap, where each modification is validated by ablation, offering high reproducibility.
- The final RDNet achieves near-SOTA performance on ImageNet-1K and exhibits excellent performance in multiple downstream tasks.

## Limitations & Future Work
- The memory overhead of concatenation connections remains higher than that of additive connections, which is mitigated by transition layers but not fundamentally solved.
- The design of frequent transition layers increases model complexity and hyperparameters.
- Not fully validated with large-scale pre-training (e.g., ImageNet-22K) and a wider range of downstream tasks.
- Performance under resource-constrained scenarios like mobile devices remains uncertain.

## Related Work & Insights
- **vs ConvNeXt**: Both are classic architectures subjected to "modernization." However, ConvNeXt modernizes ResNet (additive connection), whereas this work modernizes DenseNet (concatenation). RDNet achieves ~0.5-0.7pp higher accuracy under comparable FLOPs.
- **vs VoVNet**: VoVNet also utilizes concatenation but simplifies it (one-shot aggregation), while this work retains full dense connections and manages them via transition layers.
- **vs Swin Transformer**: A pure CNN architecture outperforms Swin, demonstrating that local attention is not the only effective way for information aggregation.

## Inspirations & Connections
- **Feature concatenation as a means of rank expansion**: This theoretical perspective is highly inspiring and can be extended to other scenarios requiring feature diversity preservation.
- DenseNet-style dense connections may offer more significant advantages in downstream tasks requiring multi-scale information, such as semantic segmentation.
- The design concept of frequent transition layers can be borrowed by other architectures that need to control feature dimensions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Challenges the mainstream perception that "additive connection is optimal", backed by both theory and experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers classification/segmentation/detection, but lacks large-scale pre-training experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative on the progressive modernization, and the pilot study is convincing.
- Value: ⭐⭐⭐⭐ Provides an overlooked architectural paradigm, offering valuable inspiration for network design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Token CropR: Faster ViTs for Quite a Few Tasks](../../CVPR2025/segmentation/token_cropr_faster_vits_for_quite_a_few_tasks.md)
- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](../../ICCV2025/segmentation/hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[CVPR 2026\] Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness](../../CVPR2026/segmentation/concept-guided_fine-tuning_steering_vits_away_from_spurious_correlations_to_impr.md)
- [\[CVPR 2026\] Beyond Appearance: Camouflaged Object Detection via Geometric Structure](../../CVPR2026/segmentation/beyond_appearance_camouflaged_object_detection_via_geometric_structure.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](../../CVPR2026/segmentation/seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)

</div>

<!-- RELATED:END -->
