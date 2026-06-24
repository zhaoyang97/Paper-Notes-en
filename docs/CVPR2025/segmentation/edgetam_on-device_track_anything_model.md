---
title: >-
  [Paper Note] EdgeTAM: On-Device Track Anything Model
description: >-
  [CVPR 2025][Segmentation][SAM 2] Through a detailed latency analysis, EdgeTAM identifies that the bottleneck of SAM 2 lies in memory attention rather than the image encoder. To address this, it proposes a 2D Spatial Perceiver to compress frame-level memory from 64×64 dimensions to ~500 tokens (while preserving spatial structure). Coupled with a two-stage knowledge distillation pipeline, EdgeTAM achieves 16 FPS real-time Track Anything on an iPhone 15 Pro Max.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "SAM 2"
  - "On-Device Deployment"
  - "Memory Compression"
  - "2D Spatial Perceiver"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: c5363a5723a7b5c1
---

# EdgeTAM: On-Device Track Anything Model

**Conference**: CVPR 2025  
**arXiv**: [2501.07256](https://arxiv.org/abs/2501.07256)  
**Code**: Yes (Meta Reality Labs)  
**Area**: Segmentation / Video Understanding  
**Keywords**: SAM 2, On-Device Deployment, Memory Compression, 2D Spatial Perceiver, Knowledge Distillation

## TL;DR

Through a detailed latency analysis, EdgeTAM identifies that the bottleneck of SAM 2 lies in memory attention rather than the image encoder. To address this, it proposes a 2D Spatial Perceiver to compress frame-level memory from 64×64 dimensions to ~500 tokens (while preserving spatial structure). Coupled with a two-stage knowledge distillation pipeline, EdgeTAM achieves 16 FPS real-time Track Anything on an iPhone 15 Pro Max.

## Background & Motivation

**Background**: SAM 2 extends SAM to videos by introducing a memory bank mechanism. It encodes historical frames via a memory encoder and stores them in the memory bank, and then fuses the current frame features with the memory features via memory attention blocks, achieving cross-frame tracking and segmentation. SAM 2 achieves outstanding performance on video object segmentation (VOS) and promptable video segmentation (PVS), becoming a foundation model in this field.

**Limitations of Prior Work**: The smallest variant of SAM 2 runs at only about 1 FPS on an iPhone 15 Pro Max, preventing its use in real-time on-device applications. Existing efficiency optimization works for SAM (e.g., EdgeSAM, EfficientViT-SAM) focus entirely on compressing the image encoder, as the mask decoder in SAM v1 is extremely lightweight. However, directly applying these methods to SAM 2 yields suboptimal results.

**Key Challenge**: Through detailed iPhone latency benchmarking, the authors discovered that even when replacing the image encoder with extremely lightweight backbones like ViT-Tiny or RepViT, the overall latency improvement remains limited. The root cause is that the newly introduced **memory attention module** in SAM 2 is the actual latency bottleneck. Each memory feature map has a size of $64 \times 64 \times 64$, resulting in a computational complexity of $O(T \cdot C \cdot H^2 \cdot W^2)$ in the cross-attention over T memory frames—a massive matrix multiplication that the limited parallel computing capability of mobile devices cannot execute efficiently.

**Goal**: How to compress memory features to reduce the computational cost of memory attention without performance degradation in video segmentation?

**Key Insight**: Videos naturally possess information redundancy, where consecutive frames share highly repetitive contents. Thus, dense storage of frame-level memory can be compressed. The key is how to compress it—naive spatial pooling severely degrades accuracy because video segmentation is a dense prediction task that requires preserving spatial structural information.

**Core Idea**: Compress frame-level memory using a 2D Spatial Perceiver while preserving spatial structure, accelerating memory attention by 8x.

## Method

### Overall Architecture

EdgeTAM maintains the meta-architecture of SAM 2 unchanged and introduces three main modifications: (1) replacing the image encoder with RepViT-M1; (2) reducing the number of memory attention blocks from 4 to 2; and (3) inserting a **2D Spatial Perceiver** module before memory attention to compress the memory features of each frame. Additionally, a two-stage knowledge distillation pipeline is employed to transfer knowledge from the teacher SAM 2 model to the lightweight student.

### Key Designs

1. **2D Spatial Perceiver (Core Innovation)**:

    - **Function**: Compresses dense frame-level memory $M_t \in \mathbb{R}^{C \times H \times W}$ (4096 tokens) into $\sim$500 tokens while preserving spatial structure.
    - **Mechanism**: Consists of two sets of learnable queries:

        **Global Perceiver**: $N_g$ global queries, each globally attending to all memory tokens to output $N_g$ vectors as frame-level summaries. Redundancy exists between queries, but they can dynamically distribute across any position in the image.

        **2D Spatial Perceiver**: $N_l$ local queries. The memory feature map is partitioned into $N_l$ non-overlapping patches using window partition. Each query only attends to the tokens within its corresponding patch, outputting explicit spatial positions. Positional encodings are moved from the input to the output side (using 2D-RoPE) to maintain spatial structure.

        The two sets of outputs are flattened along the spatial dimension and concatenated, replacing the original memory tokens in memory attention. The memory attention complexity is reduced from $O(TCH^2W^2)$ to $O(TCHW(N_g + N_l))$, achieving roughly $T$-fold acceleration.

    - **Design Motivation**: A pure Global Perceiver discards spatial structural information, leading to severe performance degradation in dense prediction tasks. The 2D Spatial Perceiver explicitly preserves spatial information through local windows.

2. **Two-Stage Knowledge Distillation (Distillation Pipeline)**:

    - **Function**: Enhances the accuracy of the lightweight student model using a teacher SAM 2 model with zero inference overhead.
    - **Mechanism**:

        **Stage 1 - Image Segmentation Pre-training**: Trained on SA-1B using task loss paired with image encoder feature alignment (MSE loss). $\mathcal{L}_{sam} = \mathcal{L}_{task} + \gamma \mathcal{L}_{img}(F_{16}^t, F_{16}^s)$

        **Stage 2 - Video Segmentation Training**: Trained on SA-V and other datasets. In addition to image encoder alignment, the memory attention output features are also aligned. $\mathcal{L}_{sam2} = \mathcal{L}_{task} + \alpha \mathcal{L}_{img} + \beta \mathcal{L}_{mem}(F_M^t, F_M^s)$

    - **Design Motivation**: Stage 1 encoder distillation helps the student learn better representations, while Stage 2 memory distillation allows the student's memory module to also receive supervision signals from the teacher, compensating for information loss caused by compression.

3. **Progressive Fine-Tuning on Long Sequences (Progressive Fine-tuning)**:

    - **Function**: Improves tracking stability in long video scenarios.
    - **Mechanism**: Trains with 8 frames first, then fine-tunes with 16 frames, and finally fine-tunes with 32 frames. The sequence is extended each time while keeping the memory bank size constant. In the latter two stages, the image encoder is frozen and distillation is omitted.
    - **Design Motivation**: EdgeTAM consumes significantly less GPU memory than SAM 2, allowing for longer training sequences. SAM 2.1 also adopts a similar strategy.

### Loss & Training

- Task loss: dice loss (weight 20) + focal loss (weight 1) + IoU loss (weight 1) + occlusion prediction BCE loss (Stage 2)
- Distillation loss: image encoder MSE (weight 1) + memory attention output MSE (weight 1)
- Teacher model: SAM2-HieraB+; Student backbone: RepViT-M1
- Default configuration: 2 memory attention blocks, 256 queries each for Global and 2D Spatial Perceiver

## Key Experimental Results

### Main Results

Video Object Segmentation (VOS) $\mathcal{J}\&\mathcal{F}$:

| Method | DAVIS 2017 | MOSE | SA-V val | SA-V test | iPhone FPS |
|------|-----------|------|---------|---------|-----------|
| SAM 2-B+ | 90.9 | 75.8 | 73.6 | 74.1 | 0.7 |
| SAM 2.1-B+ | 90.2 | 76.6 | 76.8 | 77.0 | 0.7 |
| Cutie-base+ | 88.1 | 71.7 | 61.3 | 62.8 | - |
| XMem | 86.0 | 59.6 | 60.1 | 62.3 | - |
| **EdgeTAM** | **87.7** | **70.0** | **72.3** | **71.7** | **15.7** |

Segment Anything (SA-23 Benchmark, 1-click mIoU):

| Method | SA-23 All | SA-23 Image | SA-23 Video | iPhone FPS |
|------|----------|------------|------------|-----------|
| SAM 2 | 61.4 | 63.1 | 59.1 | 1.3 |
| SAM 2.1 | 61.9 | 63.3 | 60.1 | 1.3 |
| **EdgeTAM** | **55.5** | **56.0** | **54.8** | **40.4** |

### Ablation Study

2D Spatial Perceiver Ablation (RepViT-M1, 2 blocks):

| Configuration | DAVIS | MOSE | SA-V val | iPhone FPS |
|------|-------|------|---------|-----------|
| No compression (baseline) | 86.2 | 66.1 | 71.4 | 2.5 |
| Spatial pooling 4× | 83.1 | 60.2 | 64.5 | 11.3 |
| Global Perceiver only | 84.5 | 63.8 | 67.2 | 14.8 |
| **2D Spatial Perceiver** | **86.8** | **67.0** | **72.3** | **15.7** |

Distillation Ablation (SA-V val/test $\mathcal{J}\&\mathcal{F}$):

| Configuration | SA-V val | SA-V test | Description |
|------|---------|---------|------|
| No distillation | 71.0 | 68.4 | Baseline |
| + Image encoder distillation | 71.8 | 70.2 | +0.8/+1.8 |
| + Memory distillation | **72.3** | **71.7** | +1.3/+3.3 |

### Key Findings

- Memory attention is the latency bottleneck: reducing memory attention blocks from 4 to 2 almost linearly reduces decoding latency; removing cross-attention within blocks yields the most significant speedup.
- Spatial pooling leads to severe performance degradation (SA-V val -6.9), whereas the 2D Spatial Perceiver not only recovers but surpasses the baseline (+0.9), indicating that preserving spatial structure is crucial.
- Knowledge distillation brings a +3.3 gain on SA-V test, with Stage 2 memory distillation contributing an additional +0.5 / +1.5.
- EdgeTAM achieves 150.9 FPS on an A100 and 15.7 FPS on an iPhone, making it the first unified segmentation-tracking model to run in real-time on mobile devices.
- Progressive long sequence fine-tuning (8 $\rightarrow$ 16 $\rightarrow$ 32 frames) brings an additional ~1 point gain.

## Highlights & Insights

- **Deep latency analysis shifts the optimization paradigm**: It reveals the non-intuitive observation that "SAM 2's bottleneck lies in memory attention rather than the image encoder," correcting the prior practice of blindly compressing encoders. This profiling-first optimization methodology is highly instructive.
- **Elegant design of 2D Spatial Perceiver**: Global queries capture frame-level summaries, while local queries preserve spatial structure, complementing each other. The local queries are implemented using window partition, with 2D-RoPE positional encodings added to the output. This design is simple and plug-and-play.
- **Extending distillation to the video domain**: Aligning not only image encoder features but also memory attention outputs allows the memory module to receive supervision from the teacher. This is the first work to extend video segmentation distillation to memory modules.

## Limitations & Future Work

- A significant gap still exists with SAM 2 on the SA-23 image segmentation benchmark (55.5 vs 61.4), indicating that the representation capability of the RepViT-M1 backbone is limited.
- The memory bank size (7 frames + 16 pointers) is fixed, with adaptive memory management strategies remaining unexplored.
- Only RepViT-M1 and ViT-Tiny backbones are validated, leaving other efficient architectures (e.g., MobileViT, FastViT) unexplored.
- In high frame-rate video scenarios, the compression ratio of the 2D Spatial Perceiver might be insufficient (as higher temporal redundancy allows for more aggressive compression).
- Distillation requires forward propagation of the teacher model, increasing training costs.

## Related Work & Insights

- **vs EdgeSAM/EfficientViT-SAM**: These methods only compress the image encoder of SAM v1. EdgeTAM additionally addresses the memory attention bottleneck of SAM 2, representing the first complete paradigm for efficient SAM 2.
- **vs Cutie/XMem**: Although traditional VOS methods show decent efficiency, their accuracy on large-scale multi-granularity datasets (like SA-V) is far inferior to the SAM 2 series. EdgeTAM achieves on-device deployment while maintaining SAM 2-level accuracy.
- **vs Perceiver/Perceiver IO**: The original Perceiver discards spatial structure during compression, which is unsuitable for dense prediction. The 2D Spatial Perceiver solves this issue via mixed local-global queries.

## Rating

- Novelty: ⭐⭐⭐⭐ The 2D Spatial Perceiver represents a strong combination of engineering and design, and the latency analysis yields vital insights.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 20+ datasets across three task types (PVS/SA/VOS), with latency tests on both iPhone and A100 platforms.
- Writing Quality: ⭐⭐⭐⭐ The latency analysis charts are clear, and the method has a logically complete motivational chain.
- Value: ⭐⭐⭐⭐⭐ The first on-device real-time Track Anything model, with direct value for mobile AR/MR applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SAM2-LOVE: Segment Anything Model 2 in Language-Aided Audio-Visual Scenes](sam2-love_segment_anything_model_2_in_language-aided_audio-visual_scenes.md)
- [\[ICML 2025\] InfoSAM: Fine-Tuning the Segment Anything Model from An Information-Theoretic Perspective](../../ICML2025/segmentation/infosam_fine-tuning_the_segment_anything_model_from_an_information-theoretic_per.md)
- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](../../AAAI2026/segmentation/segment_and_matte_anything_in_a_unified_model.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [\[CVPR 2025\] GroupMamba: Efficient Group-Based Visual State Space Model](groupmamba_efficient_group-based_visual_state_space_model.md)

</div>

<!-- RELATED:END -->
