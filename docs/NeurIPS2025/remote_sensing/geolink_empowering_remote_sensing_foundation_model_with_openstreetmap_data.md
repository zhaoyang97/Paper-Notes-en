---
title: >-
  [Paper Note] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data
description: >-
  [NeurIPS 2025][Remote Sensing][Remote sensing foundation model] GeoLink directly integrates OpenStreetMap vector data into remote sensing foundation model pretraining by encoding OSM data with a heterogeneous GNN and designing multi-granularity cross-modal learning objectives (region–image-level contrastive + object–patch-level fusion). Pretrained efficiently on 1.27 million sample pairs, GeoLink surpasses existing RS FMs across 7 classification and 4 segmentation/change detection benchmarks.
tags:
  - NeurIPS 2025
  - Remote Sensing
  - Remote sensing foundation model
  - OpenStreetMap
  - multimodal pretraining
  - heterogeneous graph neural network
  - cross-modal alignment
date: 2026-05-08
content_hash: 355dab5e9628432a
---

# GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data

**Conference**: NeurIPS 2025
**arXiv**: [2509.26016](https://arxiv.org/abs/2509.26016)
**Code**: [GitHub](https://github.com/bailubin/GeoLink_NeurIPS2025)
**Area**: Remote Sensing
**Keywords**: Remote sensing foundation model, OpenStreetMap, multimodal pretraining, heterogeneous graph neural network, cross-modal alignment

## TL;DR

GeoLink directly integrates OpenStreetMap vector data into remote sensing foundation model pretraining by encoding OSM data with a heterogeneous GNN and designing multi-granularity cross-modal learning objectives (region–image-level contrastive + object–patch-level fusion). Pretrained efficiently on 1.27 million sample pairs, GeoLink surpasses existing RS FMs across 7 classification and 4 segmentation/change detection benchmarks.

## Background & Motivation

**Background**: RS foundation models have advanced in multi-scale, multi-temporal, and multi-sensor directions, yet the integration of ground-level geospatial data remains insufficient.

**Limitations of Prior Work**: Existing methods that incorporate OSM into RS tasks mostly adopt indirect strategies (converting to labels, knowledge graphs, or synthetic text), which are labor-intensive, task-specific, and discard spatial information.

**Key Challenge**: A substantial modality gap exists between RS imagery and OSM data (differing data structures, content, and spatial granularity), yet the location semantics, structured knowledge, and socioeconomic information provided by OSM are inaccessible to purely visual analysis.

**Goal**: Design an explicitly geospatial approach that directly leverages raw OSM vector elements to inject geographic context into RS FMs.

**Key Insight**: Model OSM as a heterogeneous graph, encode it via GNN, and enable multi-granularity interaction with an RS ViT encoder.

**Core Idea**: Use the heterogeneous graph structure of OSM as multi-granularity supervision signals for RS self-supervised pretraining, while supporting mask-efficient training and multimodal downstream fusion.

## Method

### Overall Architecture

GeoLink comprises three encoders: (1) a ViT-L RS image encoder producing patch embeddings; (2) a GATConv-based heterogeneous GNN OSM encoder producing node embeddings (three types: point/polyline/polygon); and (3) a Two-way Transformer fusion encoder generating mixed embeddings. During pretraining, both modalities are masked and jointly optimized via three SSL objectives.

### Key Designs

1. **Heterogeneous OSM Graph Construction and Encoding**:
    - Function: Models OSM vector maps as a heterogeneous graph, with nodes representing points/polylines/polygons and edges encoding topological-spatial relationships.
    - Mechanism: BERT encodes OSM tag key-value pairs; features are averaged with global-frequency weighting $\sigma_V = \sum w_i h_i / \sum w_i$; edges are constructed via Delaunay triangulation and other topological relations.
    - Design Motivation: OSM's free-form tagging system requires a language model to handle unseen values; topological relations are more robust than distance-based ones.

2. **Region–Image-Level Contrastive Alignment**:
    - Function: Aligns RS and OSM representations at the global level.
    - Mechanism: Set2Set aggregates three node types separately → type-wise attention weighting → OSM region embedding $\varepsilon_G$; RS mean pooling → $\varepsilon_I$; InfoNCE contrastive loss $\mathcal{L}_{cont}$.
    - Design Motivation: Contrastive learning transfers structured OSM semantics to the image encoder.

3. **Object–Patch Fusion + Spatial Consistency Constraint**:
    - Function: Enables fine-grained cross-modal associative learning.
    - Mechanism: Two-way Transformer with sinusoidal positional embeddings resolves spatial ambiguity; consistency loss $\mathcal{L}_{cst} = \frac{1}{N}\sum\|\varepsilon_{OR}^m - \sigma_V^m\|^2$ enforces agreement between fused representations of masked nodes and their original features.
    - Design Motivation: Grounded in Tobler's First Law of Geography — spatial context is strongly correlated with masked object attributes.

### Loss & Training

$\mathcal{L} = \mathcal{L}_{rec} + 0.01\mathcal{L}_{cont} + 0.01\mathcal{L}_{cst}$. RS images use 75% masking with MAE reconstruction; OSM applies 20% node masking. Pretraining converges in only 60 epochs (vs. 800 for Scale-MAE) on 4×RTX6000 with batch size 2640.

## Key Experimental Results

### Main Results

| Task | Dataset | Metric | GeoLink | Prev. SOTA | Gain |
|------|---------|--------|---------|------------|------|
| kNN Classification | RESISC-45 | Top-1 | **87.33%** | 85.42% (Scale-MAE) | +1.91% |
| Fine-tuning | EuroSAT | Top-1 | 98.30% | 98.27% (MMEarth) | ≈0 |
| Segmentation (FT) | AI4SmallFarms | mIoU | **47.29%** | 45.98% (Scale-MAE) | +1.31% |
| Change Detection (FT) | SpaceNet7 | mIoU | **64.07%** | 63.22% (Scale-MAE) | +0.85% |
| UV Recognition (Multimodal) | UV Dataset | mIoU | **81.68%** | 80.09% (Scale-MAE+OSM) | +1.59% |

### Ablation Study

| Configuration | Effect | Note |
|---------------|--------|------|
| w/o OSM pretraining | Drop in both classification and segmentation | OSM pretraining substantially enhances the RS encoder |
| w/o contrastive loss | Degraded classification performance | Region-level cross-modal alignment is critical |
| w/o consistency loss | Degraded fusion quality | Fine-grained spatial constraint is necessary |

### Key Findings
- Advantages are most pronounced under the kNN protocol, indicating a well-structured RS representation space.
- Gains are amplified under limited training data, reflecting high data efficiency.
- Multimodal fusion substantially improves separability of confusable UFZ categories.
- Spatial correlation plays a key role in multimodal geospatial data fusion.

## Highlights & Insights
- The first framework to directly leverage raw OSM vector data for RS FM pretraining.
- The heterogeneous GNN design for OSM encoding is elegant — three node types, topological edges, and BERT-based tag encoding.
- Convergence in only 60 pretraining epochs demonstrates exceptional training efficiency.
- The multi-granularity learning objectives are well-motivated: global contrastive and local position-aware fusion are complementary.

## Limitations & Future Work
- Pretraining data may exhibit geographic bias due to uneven OSM annotation coverage.
- Only RGB bands are used; extension to multispectral or SAR imagery is not explored.
- The fusion encoder introduces additional computational overhead.

## Related Work & Insights
- **vs. Scale-MAE/CROMA**: These models focus on multi-scale or multi-sensor aspects but neglect ground-level geographic knowledge; GeoLink fills this gap.
- **vs. Indirect OSM Utilization**: Conventional methods discard spatial information; GeoLink's direct graph encoding preserves it in full.

## Rating
- Novelty: ⭐⭐⭐⭐ First to directly integrate OSM into RS FM pretraining.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7+4 benchmarks evaluated under multiple protocols.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with high-quality figures.
- Value: ⭐⭐⭐⭐ Opens a new direction for multimodal RS FMs; code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/remote_sensing/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](../../ICCV2025/remote_sensing/rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](../../ICCV2025/remote_sensing/smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)

</div>

<!-- RELATED:END -->
