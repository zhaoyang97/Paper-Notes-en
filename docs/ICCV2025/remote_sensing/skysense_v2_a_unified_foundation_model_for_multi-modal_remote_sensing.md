---
title: >-
  [Paper Note] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing
description: >-
  [ICCV 2025][Remote Sensing][Remote sensing foundation model] This paper proposes SkySense V2, which employs a single unified Transformer backbone to process three remote sensing modalities — high-resolution optical, multispectral, and SAR imagery — and introduces Adaptive Patch Merging (APM), modality-specific prompt tokens, and Query-based Semantic Aggregation Contrastive Learning (QSACL) for pre-training. With only 665M parameters (vs. 1.26B in the predecessor SkySense), SkySense V2 achieves an average improvement of 1.8 points across 7 tasks on 16 datasets.
tags:
  - ICCV 2025
  - Remote Sensing
  - Remote sensing foundation model
  - multi-modal learning
  - unified Transformer
  - self-supervised learning
  - Mixture of Experts
date: 2026-05-08
content_hash: e2f377905e3e3ebf
---

# SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing

**Conference**: ICCV 2025
**arXiv**: [2507.13812](https://arxiv.org/abs/2507.13812)
**Code**: N/A
**Area**: 3D Vision / Remote Sensing
**Keywords**: Remote sensing foundation model, multi-modal learning, unified Transformer, self-supervised learning, Mixture of Experts

## TL;DR
This paper proposes SkySense V2, which employs a single unified Transformer backbone to process three remote sensing modalities — high-resolution optical, multispectral, and SAR imagery — and introduces Adaptive Patch Merging (APM), modality-specific prompt tokens, and Query-based Semantic Aggregation Contrastive Learning (QSACL) for pre-training. With only 665M parameters (vs. 1.26B in the predecessor SkySense), SkySense V2 achieves an average improvement of 1.8 points across 7 tasks on 16 datasets.

## Background & Motivation

Multi-modal remote sensing foundation models (MM-RSFMs) play a critical role in Earth observation tasks such as urban planning, environmental monitoring, and natural disaster management. The predecessor SkySense is the largest MM-RSFM to date and demonstrates strong generalization capability, yet it suffers from two core issues:

**Parameter redundancy**: SkySense employs separate backbone networks for different modalities — Swin-H for high-resolution optical images, ViT-L for multispectral data, and ViT-L for SAR data — resulting in a total of 1.26B parameters with low parameter utilization efficiency.

**Ill-suited SSL for remote sensing**: SkySense primarily relies on DINOv2 for self-supervised pre-training. However, remote sensing images differ fundamentally from natural images: natural images typically focus on a single subject (e.g., a cat or dog), whereas remote sensing images contain diverse semantic objects across different regions (buildings, forests, ponds, land, etc.). Applying conventional SSL by directly contrasting different cropped views can lead to **semantic misalignment**, as two views may capture entirely different subjects.

**Root Cause**: *How can a single unified backbone handle multi-modal remote sensing data of varying resolutions, while adopting an SSL strategy that accounts for the distributional characteristics of remote sensing imagery?*

The paper addresses this through: (1) a unified Transformer backbone with adaptive resolution handling; and (2) QSACL to resolve the multi-semantic challenge in remote sensing contrastive learning.

## Method

### Overall Architecture

SkySense V2 adopts a teacher–student architecture for pre-training. The student network extracts multi-modal features via the backbone, while the teacher network is updated via EMA. Inputs consist of geographically aligned HR optical images (2048×2048), Sentinel-2 multispectral sequences (64×64), and Sentinel-1 SAR sequences (64×64). Training uses approximately 21 million multi-modal remote sensing samples.

### Key Designs

1. **Unified Transformer Backbone**:

    - **Function**: A single four-stage hierarchical encoder processes data from all three modalities.
    - **Mechanism**: The first two stages employ Swin Transformer V2 blocks (window size 8), leveraging locality and translation-invariance priors while reducing computational complexity; the latter two stages use standard Transformer blocks with global self-attention to learn global feature representations. All three modalities share identical parameters but have separate tokenizers.
    - **Design Motivation**: Full parameter sharing substantially improves parameter efficiency (665M vs. 1.26B), while the local–global attention combination balances efficiency and expressiveness.

2. **Adaptive Patch Merging (APM)**:

    - **Function**: Selectively reduces feature resolution after each stage based on the ground sampling distance (GSD) of each modality.
    - **Mechanism**: For high-resolution optical images, APM merges 2×2 neighboring patch features at each stage (reducing spatial resolution by 4× and doubling channel dimensions); for medium-resolution MS/SAR data, APM preserves the spatial resolution (applying only a linear projection with averaged weights).
    - **Design Motivation**: Geographically aligned multi-modal data have different spatial resolutions, necessitating adaptive processing to maintain feature spatial alignment.

3. **Modality-Specific Prompt Tokens**:

    - **Function**: A small number of learnable prompt tokens are introduced per modality in the last two stages (4 tokens per modality per stage).
    - **Mechanism**: Modality prompt tokens are concatenated with input tokens before being fed into the Transformer block; the prompt token outputs are discarded after the last block of each stage:
    $[E_{drop}, E_i^4] = \mathcal{F}_3([P_i^3, E_i^3])$
    - **Design Motivation**: Full parameter sharing may reduce feature diversity; a small number of modality-specific parameters can capture unique characteristics of each modality while preserving parameter efficiency.

4. **Query-based Semantic Aggregation Contrastive Learning (QSACL)**:

    - **Function**: Multiple learnable queries aggregate semantically similar features from different views via cross-attention, and contrastive learning is then applied to the aggregated features.
    - **Mechanism**: Given features from global and local views, $m$ learnable queries perform cross-attention with each view's features to produce semantically aggregated representations $z_i^g$ and $z_i^l$. The contrastive loss is computed over aggregated feature pairs sharing the same query:
    $\mathcal{L}_{QSACL} = \frac{1}{2m}\sum_{i=1}^m (\mathcal{L}_{CL}(z_i^g, z_i^{l'}) + \mathcal{L}_{CL}(z_i^l, z_i^{g'}))$
    - **Design Motivation**: Different cropped views of remote sensing images may contain different semantic objects, causing semantic misalignment in direct contrastive learning. Query-based semantic aggregation ensures that contrastive learning is performed over consistent semantics.

5. **Mixture of Experts (MoE) Extension**:

    - **Function**: In the last $L=6$ Transformer blocks, the FFN is replaced by an MoE module ($M=8$ experts, top-$k=1$).
    - **Mechanism**: $MOE(x) = \sum_{i \in \mathcal{T}} \mathcal{G}_i(x) \cdot \mathcal{E}_i(x)$, where $\mathcal{G}$ is a linear gate followed by softmax.
    - **Design Motivation**: The parameter budget saved by the unified backbone design can be reinvested in MoE expansion, achieving greater model capacity via sparse feed-forward layers without a proportional increase in computation.

### Loss & Training

The total training loss is a weighted sum of three components:
$$\mathcal{L} = \lambda_1 \mathcal{L}_{MGCL} + \lambda_2 \mathcal{L}_{ITA} + \lambda_3 \mathcal{L}_{QSACL}$$
- $\mathcal{L}_{MGCL}$: Multi-granularity contrastive learning (pixel, object, and image levels)
- $\mathcal{L}_{ITA}$: Dense image–text alignment based on OpenStreetMap labels
- $\mathcal{L}_{QSACL}$: Query-based Semantic Aggregation Contrastive Learning

Training configuration: batch size 1024, 128 H20 GPUs, 600K iterations, AdamW optimizer, initial learning rate $2 \times 10^{-4}$ with cosine decay to $1 \times 10^{-6}$, token dimension $C=352$.

## Key Experimental Results

### Main Results (Scene Classification)

| Model | AID (20%/50%) OA | RESISC-45 (10%/20%) OA | BEN-S2 (10%/100%) mAP | fMoW-S2 Top-1/5 |
|------|------------------|----------------------|---------------------|----------------|
| SatMAE | 95.02/96.94 | 91.72/94.10 | 86.18/89.50 | 63.84/- |
| Scale-MAE | 96.44/97.58 | 92.63/95.04 | - | - |
| SkySense | 97.68/98.60 | 94.85/96.32 | 88.67/92.09 | 64.38/87.27 |
| **SkySense V2** | **98.34/99.05** | **96.42/97.24** | **89.13/93.78** | **66.65/89.32** |

### Ablation Study (Semantic Segmentation)

| Model | Dyna.-Pla. (5%/10%) mIoU | iSAID mIoU | Potsdam mF1 |
|------|--------------------------|-----------|-------------|
| SkySense | 39.7/46.5 | 70.91 | 93.99 |
| **SkySense V2** | **41.2/47.6** | **71.87** | **95.86** |

### Key Findings

- SkySense V2 achieves state-of-the-art or near state-of-the-art performance across all 16 datasets and 7 task types.
- Parameter count is reduced from 1.26B to 665M (a 47% reduction), while average performance improves by 1.8 points.
- QSACL enables different queries to aggregate consistent semantic features (e.g., buildings, vegetation), effectively addressing multi-semantic contrastive learning in remote sensing images.
- Performance gains are especially pronounced under low training ratio (low TR) settings, demonstrating stronger feature representations.
- The introduction of MoE further improves performance at a modest additional computational cost.

## Highlights & Insights

1. **Efficiency advantage of the unified backbone**: Replacing three separate backbones (1.26B) with a single 665M-parameter backbone not only avoids performance degradation but yields improvements, demonstrating the viability of multi-modal parameter sharing.
2. **Elegant design of APM**: A simple conditional branch (merge or preserve) resolves the challenge of differing resolutions across modalities, without requiring complex alignment modules.
3. **QSACL accurately targets remote sensing data characteristics**: The multi-semantic distribution of remote sensing images is one of the most fundamental distinctions from natural images; QSACL directly addresses this via query-based aggregation.
4. **Impressive engineering scale**: 21 million training samples × 128 H20 GPUs × 600K iterations reflects industrial-scale pre-training capability.

## Limitations & Future Work

- Only three modalities are supported (optical RGB, multispectral, and SAR); other important remote sensing data sources such as hyperspectral imagery and LiDAR are not covered.
- The massive pre-training data scale (21 million samples) limits reproducibility within the academic community.
- MoE is applied only to the last 6 blocks; whether earlier-stage incorporation would be beneficial remains unexplored.
- The unified backbone may be disadvantageous for lightweight deployment, as 665M parameters remain substantial.
- The analysis of how the number of QSACL queries ($m$) affects performance is insufficiently thorough.

## Related Work & Insights

- SkySense [Guo et al., CVPR 2024] is the direct predecessor; this paper addresses its parameter redundancy and ill-suited SSL.
- Meta-Transformer [Zhang et al.] explored processing multiple modalities with a single Transformer; this paper instantiates that idea in the remote sensing domain with critical adaptations.
- AnySat [ECCV 2024] also adopts a unified architecture for multi-modal remote sensing data, but differs fundamentally from SkySense V2 in backbone design and SSL strategy.
- Key takeaway: In remote sensing foundation models, **SSL strategies tailored to data characteristics** (such as QSACL) are more important than simply borrowing SSL methods designed for natural images.

## Rating
- **Novelty**: ⭐⭐⭐⭐ APM and QSACL are well-motivated innovations tailored to remote sensing characteristics, though the unified backbone design itself is not novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Large-scale evaluation across 16 datasets and 7 task types with broad coverage.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, though placing some technical details in the appendix compromises completeness.
- **Value**: ⭐⭐⭐⭐⭐ As a grand unified foundation model for remote sensing, it has direct applicability to real-world Earth observation tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)

<!-- RELATED:END -->
