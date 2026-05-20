---
title: >-
  [Paper Note] Robust Ego-Exo Correspondence with Long-Term Memory
description: >-
  [NeurIPS 2025][Segmentation][ego-exo correspondence] This paper proposes LM-EEC, a SAM 2-based cross-view video object segmentation framework for ego-exo correspondence. It introduces a Memory-View MoE (MV-MoE) module to…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "ego-exo correspondence"
  - "SAM2"
  - "MoE"
  - "video object segmentation"
  - "long-term memory"
date: 2026-05-08
content_hash: 97407353261775d6
---

# Robust Ego-Exo Correspondence with Long-Term Memory

**Conference**: NeurIPS 2025
**arXiv**: [2510.11417](https://arxiv.org/abs/2510.11417)  
**Code**: [GitHub](https://github.com/juneyeeHu/LM-EEC)  
**Area**: Image Segmentation
**Keywords**: ego-exo correspondence, SAM2, MoE, video object segmentation, long-term memory

## TL;DR
This paper proposes LM-EEC, a SAM 2-based cross-view video object segmentation framework for ego-exo correspondence. It introduces a Memory-View MoE (MV-MoE) module to adaptively fuse memory features with cross-view features, coupled with a dual memory bank compression strategy for retaining long-term information. LM-EEC substantially outperforms existing methods on the EgoExo4D benchmark (Ego2Exo IoU: 54.98 vs. 38.26).

## Background & Motivation

1. **Task Definition**: Given synchronized ego/exo video pairs and a target mask from one viewpoint, the goal is to segment the same target in the other viewpoint — a capability critical for AR and robotics.
2. **Limitations of Prior Work**: VOS methods such as XView-XMem fail to handle extreme viewpoint changes, occlusions, and small objects. Although SAM 2 generalizes well, naively adding memory features and prompts leads to insufficient cross-view fusion, and its FIFO memory management discards long-term information.
3. **Mechanism**: Adaptive fusion of two feature types via MoE, dual memory banks for ego/exo storage separately, and a temporal-redundancy-based compression strategy.

## Method

### Overall Architecture
Built upon SAM 2 with three core components: multi-view encoding, dual memory compression, and target mask prediction.

### Key Design 1: Memory-View MoE (MV-MoE)
- Treats memory-aware features $F_{mem}$ and cross-view features $F_{view}$ as two complementary "experts."
- **Channel Routing**: Concatenation → global pooling → two-branch MLP → sigmoid weights → residual modulation.
- **Spatial Routing**: Concatenation → two-branch Conv-ReLU-Conv-Sigmoid → spatial weights → residual modulation.
- The two refined features are summed to produce the fused feature $F_{tar}$.
- The design is lightweight, avoiding the network-level sparse complexity of conventional MoE.

### Key Design 2: Dual Memory Banks + Compression Strategy
- **Dual Memory Banks**: Ego and exo features are stored separately (rather than jointly) to fully exploit their complementary information.
- **Compression Strategy**: When memory exceeds capacity $M$, the Euclidean distance between adjacent frames is computed at each spatial position; the most similar adjacent frame pair is identified and merged by averaging — reducing redundancy while preserving long-term information.
- Compression is applied only during inference.

### Training Details
- Initialized from SAM 2 Base pretrained weights; trained on 8-frame clips with a memory bank size of 6.
- Jointly trained for 60 epochs on 8× A100 GPUs; input resized to $480\times480$.

## Key Experimental Results

### EgoExo4D Test Set

| Method | Ego2Exo IoU↑ | Ego2Exo LE↓ | Exo2Ego IoU↑ | Exo2Ego LE↓ |
|--------|-------------|-------------|-------------|-------------|
| XView-XMem+XSegTx | 34.90 | 0.038 | 25.00 | 0.117 |
| SimVOS | 38.26 | 0.090 | 40.67 | 0.099 |
| Cutie | 27.03 | 0.108 | 47.52 | 0.070 |
| Base model (SAM2+dual memory) | 52.13 | 0.024 | 57.27 | 0.047 |
| **LM-EEC** | **54.98** | **0.017** | **65.77** | **0.031** |

### Ablation Study (Ego2Exo Val)

| Component | IoU↑ |
|-----------|------|
| w/o cross-view prompt | 0.5691 |
| Simple addition (base) | 0.5673 |
| **MV-MoE** | **0.5925** |
| w/o ego memory | 0.5748 |
| w/o exo memory | 0.5420 |
| FIFO strategy | 0.5823 |
| **Ours (compression strategy)** | **0.5925** |

## Highlights & Insights
1. This work is the first to introduce MoE-based thinking into ego-exo cross-view feature fusion, with adaptive dual-path channel and spatial routing.
2. The dual memory bank design appropriately distinguishes the differing characteristics of ego and exo viewpoints.
3. The compression strategy is concise yet effective — relying solely on inter-frame Euclidean distance and mean merging.
4. On the Exo2Ego task, LM-EEC achieves an 18.25 IoU improvement over the second-best method.

## Limitations & Future Work
1. Inference speed is only 8.4 FPS (on V100), making real-time deployment infeasible.
2. The BA metric (object existence judgment) still lags behind XSegTx (64.22 vs. 66.31).
3. The compression strategy is relatively simple (mean merging), which may discard unique information from key frames.

## Related Work & Insights
- **vs. SAM 2**: Direct addition of prompt and memory features in SAM 2 causes distribution conflicts; LM-EEC's MoE routing adaptively reweights contributions.
- **vs. XMem/Cutie**: These VOS models do not account for ego/exo viewpoint differences and perform poorly when applied directly.
- **vs. XSegTx**: XSegTx is a co-segmentation method that holds an advantage on the BA metric but falls significantly short in IoU.

The MoE-based feature fusion paradigm is transferable to other multi-view and multimodal segmentation tasks. The dual memory bank with compression strategy provides broadly applicable insights for long-video understanding. Ego-exo correspondence remains a core problem in AR and robotics and warrants continued attention.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Combined design of MoE fusion and dual memory compression.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple baselines, comprehensive ablations, and analysis across object sizes.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear figures and captions.
- **Value**: ⭐⭐⭐⭐ Establishes a new state of the art on the ego-exo correspondence task.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAM2Long: Enhancing SAM 2 for Long Video Segmentation with a Training-Free Memory Tree](../../ICCV2025/segmentation/sam2long_enhancing_sam_2_for_long_video_segmentation_with_a.md)
- [\[CVPR 2026\] HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding](../../CVPR2026/segmentation/hippomm_hippocampal-inspired_multimodal_memory_for_long_audiovisual_event_unders.md)
- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[NeurIPS 2025\] Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective](towards_robust_pseudo-label_learning_in_semantic_segmentation_an_encoding_perspe.md)
- [\[NeurIPS 2025\] STEAD: Robust Provably Secure Linguistic Steganography with Diffusion Language Model](stead_robust_provably_secure_linguistic_steganography_with_diffusion_language_mo.md)

</div>

<!-- RELATED:END -->
