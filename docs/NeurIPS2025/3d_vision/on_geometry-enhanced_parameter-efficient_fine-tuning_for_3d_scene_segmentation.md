---
title: >-
  [Paper Note] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation
description: >-
  [NeurIPS 2025][3D Vision][Parameter-Efficient Fine-Tuning] This paper proposes the Geometry Encoding Mixer (GEM), a geometry-aware PEFT module designed for 3D point cloud Transformers. It captures fine-grained local geom…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Parameter-Efficient Fine-Tuning"
  - "Point Cloud Segmentation"
  - "Geometry Encoding"
  - "3D Scene Understanding"
  - "Transformer"
date: 2026-05-08
content_hash: 39163f55e1c5efe2
---

# On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2505.22444](https://arxiv.org/abs/2505.22444)  
**Code**: [https://github.com/LiyaoTang/GEM](https://github.com/LiyaoTang/GEM)  
**Area**: 3D Vision
**Keywords**: Parameter-Efficient Fine-Tuning, Point Cloud Segmentation, Geometry Encoding, 3D Scene Understanding, Transformer

## TL;DR

This paper proposes the Geometry Encoding Mixer (GEM), a geometry-aware PEFT module designed for 3D point cloud Transformers. It captures fine-grained local geometric details via a Spatial Adapter and injects global scene context via a Context Adapter, achieving performance on par with or exceeding full fine-tuning while updating only 1.6% of parameters.

## Background & Motivation

Large-scale pretrained point cloud models (e.g., Sonata, PTv3) have achieved notable progress in 3D scene understanding. However, adapting these models to downstream tasks typically requires full fine-tuning, incurring substantial computational and storage costs. While PEFT methods from NLP and 2D vision (LoRA, Adapter, Prompt Tuning, etc.) are well established, they transfer poorly to 3D point clouds.

The root cause lies in the nature of point clouds: they are unordered sets of 3D coordinates with strong irregularity, sparsity, and structural variability, resulting in significant geometric and spatial distribution shifts between pretraining and downstream domains. Existing PEFT methods either adapt at the point-wise feature level (Adapter, LoRA), ignoring spatial structure, or insert fixed global tokens (Prompt Tuning), failing to capture scene-specific context. Critically, modern 3D Transformers predominantly employ local attention mechanisms, which inherently limit global context modeling.

The paper's starting point is that effective 3D PEFT must simultaneously and explicitly model fine-grained local spatial patterns and global geometric context — neither alone is sufficient.

## Method

### Overall Architecture

GEM is inserted as a lightweight module into each layer of a pretrained point cloud Transformer. It comprises two complementary components: a Spatial Adapter for local geometry and a Context Adapter for global context. Both follow a residual bottleneck design, operating along the spatial dimension rather than the channel dimension. The overall pipeline first enhances positional encodings via the Spatial Adapter, then supplements local attention outputs with global information via the Context Adapter.

### Key Designs

1. **Spatial Adapter**: Applies a lightweight 3D convolutional bottleneck over each point's neighborhood to enhance pretrained positional encodings. Concretely, for each point, neighboring voxels in a 3D grid are considered as local neighbors; a bottleneck of dimensionality-reduction projection → locally-weighted convolution → dimensionality-expansion projection is used to learn fine-grained local spatial details. With kernel size $k=3$, each point attends to at most $k^3=27$ neighbors; the additional parameter count is $2rd + k^3r^2$ and computational complexity is $O(nd)$, making it highly efficient. In essence, this serves as an efficient convolutional positional encoding operating in parallel with the pretrained positional encoding.

2. **Context Adapter**: Introduces $m$ learnable latent tokens as global context vectors that interact with the entire point cloud via two-step attention: the latent tokens first serve as queries to aggregate global information over all points (complexity $O(nm)$), after which all points retrieve global context from these latent tokens via attention. A key innovation is that latent tokens are shared across layers via residual updates ($L \leftarrow L + L_c$), forming dynamic, scene-specific prompts rather than the static prefix tokens used in Prompt Tuning.

3. **Residual Bottleneck Structure**: Both components adopt a reduce–process–expand bottleneck design with rank $r=32$ and $m=4$ latent tokens, ensuring minimal parameter overhead (only 1.6%). The Spatial Adapter is added to the original positional encoding via a residual connection; the Context Adapter is added to the local attention output via a residual connection.

### Loss & Training

During training, the pretrained backbone weights are frozen and only the GEM module parameters are updated. Standard fine-tuning settings are followed, with cross-entropy loss for semantic segmentation supervision. All PEFT baselines use their official implementations with best-validated configurations.

## Key Experimental Results

### Main Results

| Dataset | Metric (mIoU) | GEM | Full FT (Sonata ft.) | LoRA | Adapter | Prompt |
|--------|---------------|-----|----------------------|------|---------|--------|
| ScanNet Val | mIoU | 78.3 | 78.3 | 76.7 | 77.0 | 74.3 |
| ScanNet200 Val | mIoU | 35.6 | 37.3 | 33.6 | 33.6 | 31.4 |
| ScanNet++ Val | mIoU | 46.6 | 49.8 | 44.2 | 42.6 | 41.2 |
| S3DIS Area5 | mIoU | 75.1 | 72.4 | 74.5 | 73.8 | 73.4 |
| S3DIS 6-fold | mIoU | 77.9 | 79.5 | 77.4 | 76.4 | 73.7 |

Using only 1.6% of parameters (1.8M), GEM matches full fine-tuning (108.5M, 100%) on most datasets and surpasses it by 2.7 mIoU on S3DIS Area5.

### Ablation Study

| Configuration | ScanNet mIoU | Notes |
|---------------|-------------|-------|
| Linear Probing | 72.5 | Train classification head only |
| + Spatial Adapter only | ~76 | Local geometry modeling |
| + Context Adapter only | ~76 | Global context modeling |
| + GEM (SA + CA) | 78.3 | Complementary; best performance |

In data efficiency experiments, GEM achieves 47.5 mIoU with only 1% annotated scenes, outperforming Sonata full. (45.3) and Sonata ft. (44.4), with especially pronounced advantages under extremely low-data regimes.

### Key Findings

- GEM surpasses full fine-tuning on ScanNet++, which features sub-millimeter resolution and highly diverse scenes with a large distributional gap from pretraining, demonstrating the value of explicit geometry modeling under large domain shifts.
- LoRA and Adapter perform comparably, suggesting that when geometry is not explicitly modeled, the choice of adaptation target is secondary.
- Prompt Tuning on S3DIS 6-fold even underperforms linear probing, revealing the penalty for ignoring spatial structure.
- On a supervised pretrained backbone (PTv3-PPT), existing PEFT methods may cause performance degradation (negative transfer), whereas GEM still improves performance to 79.1 mIoU.

## Highlights & Insights

- This work provides the first systematic exploration and validation of PEFT methods for large-scale 3D scene understanding, filling an important gap in the literature.
- The dual-path local+global design is both conceptually clean and empirically effective, with the Spatial Adapter and Context Adapter each addressing a clearly defined problem.
- The cross-layer residual update of latent tokens in the Context Adapter is an elegant design choice that transforms static prompts into dynamic, scene-aware context representations.
- The experimental coverage is comprehensive: indoor and outdoor scenes, self-supervised and supervised pretraining, with and without decoders, and data efficiency across multiple dimensions.

## Limitations & Future Work

- Transferring indoor-pretrained models to outdoor settings (SemanticKITTI) still yields a substantial performance gap; cross-domain PEFT remains a future direction.
- The number of latent tokens in the Context Adapter is fixed at $m=4$, which may be insufficient for very large scenes.
- Evaluation is currently limited to semantic segmentation; generalization to instance segmentation, object detection, and other tasks remains to be explored.
- Validation is conducted solely on PTv3-family backbones; applicability to other 3D backbone architectures has not been fully verified.

## Related Work & Insights

- Compared to other 3D PEFT works such as PointLoRA and STAG, GEM is the first to target large-scale scene-level inputs rather than object-level inputs.
- The design of the Spatial Adapter draws inspiration from Conditional Positional Encodings (CPE), embedding them within a PEFT framework.
- The Context Adapter resembles the latent bottleneck design of Perceiver, but achieves dynamic behavior through cross-layer residual updates.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](../../AAAI2026/3d_vision/parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[NeurIPS 2025\] Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Prediction](object-centric_representation_learning_for_enhanced_3d_semantic_scene_graph_pred.md)
- [\[NeurIPS 2025\] EUGens: Efficient, Unified, and General Dense Layers](eugens_efficient_unified_and_general_dense_layers.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)

</div>

<!-- RELATED:END -->
