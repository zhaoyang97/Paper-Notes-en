---
title: >-
  [Paper Note] 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification
description: >-
  [CVPR 2025][Segmentation][2D SSM] Proposes 2DMamba, the first **native 2D selective State Space Model** with an efficient parallel algorithm. By maintaining 2D spatial continuity (rather than flattening into a 1D sequence) to model inter-patch relationships in WSIs, it comprehensively outperforms 1D Mamba methods across 10 public pathology datasets, while also achieving improvements on ImageNet classification and ADE20K segmentation.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "2D SSM"
  - "Mamba"
  - "Whole Slide Image"
  - "MIL"
  - "Hardware-Aware"
date: 2026-05-08
content_hash: e14fe44e96341d1c
---

# 2DMamba: Efficient State Space Model for Image Representation with Applications on Giga-Pixel Whole Slide Image Classification

**Conference**: CVPR 2025  
**arXiv**: [2412.00678](https://arxiv.org/abs/2412.00678)  
**Code**: [https://github.com/AtlasAnalyticsLab/2DMamba](https://github.com/AtlasAnalyticsLab/2DMamba)  
**Area**: Image Segmentation  
**Keywords**: 2D SSM, Mamba, Whole Slide Image, MIL, Hardware-Aware  

## TL;DR
Proposes 2DMamba, the first **native 2D selective State Space Model** with an efficient parallel algorithm. By maintaining 2D spatial continuity (rather than flattening into a 1D sequence) to model inter-patch relationships in WSIs, it comprehensively outperforms 1D Mamba methods across 10 public pathology datasets, while also achieving improvements on ImageNet classification and ADE20K segmentation.

## Background & Motivation
Giga-pixel Whole Slide Image (WSI) analysis is a core task in computational pathology. Traditional MIL methods partition the WSI into patches and aggregate them independently, ignoring spatial relationships. Transformers introduce inter-patch interactions but suffer from quadratic complexity. Mamba (Selective SSM) has emerged as a powerful alternative due to its linear complexity and high parallelism. However, existing Vision Mamba methods flatten 2D images into 1D sequences, causing a "spatial discontinuity" problem—adjacent patches in 2D may end up far apart in the 1D sequence, resulting in information being forgotten during propagation (due to the high-order decay of $\bar{A}$). On the other hand, although existing 2D SSMs preserve the 2D structures, they lack efficient parallel algorithms, leading to extremely slow training.

## Core Problem
How to design a selective State Space Model that **both maintains 2D spatial continuity and features an efficient parallel implementation**? Existing methods either flatten the input into 1D (fast but loses spatial structure) or use native 2D recursion (preserves structure but is too slow to be practical).

## Method

### Overall Architecture
Pipeline of 2DMambaMIL: Input WSI $\to$ segment into patches $\to$ extract patch features using a pre-trained feature extractor (UNI, ViT-L/16) $\to$ pad non-tissue regions with a learnable token to form a 2D feature map $\to$ feed into $U$ layers of 2DMamba blocks $\to$ attention aggregator $\to$ classification/survival analysis output.

### Key Designs
1. **2D Selective SSM Architecture**: The core innovation is extending the 1D selective scan to a true 2D scan. It first performs a horizontal scan on each row independently (equivalent to 1D Mamba) to obtain $h^{hor}_{i,j}$, and then performs a vertical scan on the results of the horizontal scan to get the final state $h_{i,j}$. Expanding this yields $h_{i,j} = \sum_{i'\leq i}\sum_{j'\leq j} \bar{A}^{(i-i'+j-j')} \bar{B} x_{i',j'}$, where the exponent of $\bar{A}$ is the Manhattan distance rather than the 1D flattened distance, thereby preserving 2D spatial continuity. For instance, in a 3×3 feature map, the distance from (1,1) to (3,3) is 4 in 2D ($\bar{A}^4$), but 8 in 1D flattening ($\bar{A}^8$), which suffers from more severe forgetting.

2. **Hardware-Aware 2D Parallel Scan Operator**: Naive 2D scanning requires explicitly storing feature maps of $N$ intermediate state dimensions on the High Bandwidth Memory (HBM), which has a memory complexity of $O(NL)$. 2DMamba employs a 2D tiling strategy: slicing the feature map into 2D grid blocks, loading each block into SRAM to complete horizontal and vertical scans, and performing reduction of the $N$ state dimensions within SRAM, only writing the aggregated results back to HBM. The overall memory access complexity remains $O(L)$, identical to 1D Mamba.

3. **SegmentedBlockScan Algorithm**: NVIDIA CUB's BlockScan only supports 1D sequences and requires sizes to be multiples of 32. 2DMamba proposes SegmentedBlockScan, which distributes GPU threads across multiple rows and columns (requiring only $H \times W$ to be a multiple of 32 rather than each individually), significantly reducing padding waste (e.g., from 56% to ~1% for a 14×14 feature map).

4. **Learnable Non-Tissue Region Padding**: Large areas in WSIs consist of non-tissue background. 2DMamba replaces fixed zero-padding with a learnable token $p$ to represent these regions, enabling the model to adaptively learn the background representation during training.

### Loss & Training
- Uses the AdamW optimizer with an initial learning rate of 0.0001 and cosine annealing.
- Trained for 20 epochs with a batch size of 1.
- Patch size: 512×512 at 20x magnification.
- Feature extractor: UNI (frozen), SSM dimension: 128, state dimension: $N=16$.

## Key Experimental Results

### WSI Classification (Average of 5 Datasets)

| Dataset | Metric | 2DMambaMIL | MambaMIL | Best Non-Mamba |
|--------|------|------------|----------|------------|
| PANDA | Acc | **0.5075** | 0.4679 | 0.5047 (S4-MIL) |
| PANDA | AUC | **0.8184** | 0.7781 | 0.7986 (S4-MIL) |
| TCGA-BRCA | Acc | **0.9458** | 0.9333 | 0.9375 (DSMIL) |
| TCGA-BRCA | AUC | **0.9782** | 0.9657 | 0.9770 (S4-MIL) |
| BRACS | F1 | **0.7045** | 0.6832 | 0.6131 (DTFD) |

### WSI Survival Analysis (C-index)

| Dataset | 2DMambaMIL | MambaMIL | Best Baseline |
|--------|------------|----------|-------------|
| KIRC | **0.7311** | 0.7096 | 0.7271 (DTFD) |
| LUAD | **0.6198** | 0.5952 | 0.6157 (ABMIL) |
| STAD | **0.6428** | 0.6244 | 0.6244 (MambaMIL) |

### Natural Images

| Task | Method | Metric |
|------|------|------|
| ImageNet-1K | 2DVMamba-T | **82.8%** (vs VMamba-T 82.6%) |
| ADE20K SS | 2DVMamba-T | **48.6 mIoU** (vs VMamba-T 47.9) |
| ADE20K MS | 2DVMamba-T | **49.3 mIoU** (vs VMamba-T 48.8) |

### Ablation Study Key Points
- **Learnable Padding vs. Zero Padding**: Learnable padding improves Acc by 1.56% and AUC by 0.62% on PANDA.
- **Multi-directional 1D Scan vs. 2D Scan**: 4-way cross scan (the best variant of MambaMIL) achieves Acc=0.4939, AUC=0.8006; 2DMamba achieves Acc=0.5075, AUC=0.8184, indicating that multi-directional 1D is ultimately inferior to native 2D.
- **Positional Encoding**: Adding PE degrades 2DMamba's performance (Acc -1.0%, AUC -1.0%), suggesting that the 2D structure already implicitly encodes positional information.
- **Scan Order**: The difference between horizontal-vertical and vertical-horizontal is small ($\pm0.5\%$), showing low sensitivity.
- **Number of Blocks $U$**: 1 layer is optimal; performance decreases slightly with more layers.
- **Efficiency**: CUDA 2D scan throughput is about 70%-90% of 1D, and GPU memory is comparable to 1D (linear), which is far superior to the Python implementation (90% memory savings).

## Highlights & Insights
- **First Efficiently Parallel Native 2D Mamba**: Through mathematical derivation, the 2D scan is decomposed into a cascade of row scans and column scans, elegantly maintaining the same parallelism as the 1D scan.
- **Hardware-Aware Design**: 2D tiling + in-SRAM reduction avoids storing intermediate states on HBM, solving the efficiency problem at the operator level.
- **Mathematical Proof of Spatial Continuity**: The exponent of $\bar{A}$ equals the Manhattan distance instead of the 1D sequence distance, elegantly explaining why the 2D scan outperforms 1D.
- **Generality**: Not only applicable to WSIs but can also be seamlessly integrated into VMamba to improve performance on natural image tasks.

## Limitations & Future Work
- The 2D scan only models the information flow from the top-left to the current position; supplementary backward scans are required to capture the full context.
- There is still a 10-30% loss in throughput compared to 1D Mamba, which may still pose a bottleneck for ultra-large-scale feature maps (e.g., high-resolution remote sensing).
- Only Tiny and Small scales of VMamba integration are validated; the effectiveness of larger scales (Base/Large) remains unknown.
- The theoretical optimality of 2D SSMs (e.g., scan order, $\bar{A}$ sharing strategy) lacks theoretical guarantees.

## Related Work & Insights
- **MambaMIL / S4-MIL**: Both are 1D SSMs that flatten WSIs into sequences. 2DMamba avoids spatial fracturing via native 2D processing, consistently outperforming them across all datasets.
- **2D-SSM (Baron et al.)**: An early 2D SSM work, but it lacks a selective mechanism and an efficient parallel algorithm, making it practically unusable. 2DMamba addresses both selectivity and efficiency.
- **VMamba**: Uses 4-way 1D scans to approximate 2D structures. Replacing its scanning module with native 2D in 2DMamba improves performance in segmentation tasks by 0.5-0.7 mIoU, indicating that multi-directional 1D still has limitations.

### Insights & Connections
- The hardware-aware design of 2D SSMs can be generalized to 3D volumetric data (CT/MRI), similarly decomposing into a cascade of scans in three directions.

## Rating
- Novelty: ⭐⭐⭐⭐ First efficiently parallel native 2D selective SSM, though the core idea (row scan + column scan decomposition) is somewhat intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 datasets + natural images + rich ablation studies + efficiency analysis + visualization, highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and mathematical derivation; the hardware section is relatively dense but necessary.
- Value: ⭐⭐⭐⭐ Solves real problems, code is open-sourced, and highly beneficial for the computational pathology community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GroupMamba: Efficient Group-Based Visual State Space Model](groupmamba_efficient_group-based_visual_state_space_model.md)
- [\[CVPR 2025\] DefMamba: Deformable Visual State Space Model](defmamba_deformable_visual_state_space_model.md)
- [\[CVPR 2025\] Exploiting Temporal State Space Sharing for Video Semantic Segmentation](exploiting_temporal_state_space_sharing_for_video_semantic_segmentation.md)
- [\[CVPR 2025\] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation](mv-ssm_multi-view_state_space_modeling_for_3d_human_pose_estimation.md)
- [\[CVPR 2025\] Your ViT is Secretly an Image Segmentation Model](your_vit_is_secretly_an_image_segmentation_model.md)

</div>

<!-- RELATED:END -->
