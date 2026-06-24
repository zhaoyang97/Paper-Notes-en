---
title: >-
  [Paper Note] PerLA: Perceptive 3D Language Assistant
description: >-
  [CVPR 2025][3D Vision][3D Language Assistant] Proposes PerLA, a perceptive 3D language assistant that achieves parallel capture of high-resolution local details via Hilbert curve partitioning, and aggregates local information with low-resolution global context using cross-attention and graph convolutional networks. This significantly improves fine-grained perception in 3D scene understanding without increasing the number of LLM input tokens.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Language Assistant"
  - "Point Cloud Understanding"
  - "Local-Global Fusion"
  - "Hilbert Curve"
  - "Graph Neural Networks"
date: 2026-05-08
content_hash: 5e7ef1d3fee18b37
---

# PerLA: Perceptive 3D Language Assistant

**Conference**: CVPR 2025  
**arXiv**: [2411.19774](https://arxiv.org/abs/2411.19774)  
**Code**: [Project Page](https://gfmei.github.io/PerLA)  
**Area**: 3D Vision/Multimodal  
**Keywords**: 3D Language Assistant, Point Cloud Understanding, Local-Global Fusion, Hilbert Curve, Graph Neural Networks

## TL;DR

Proposes PerLA, a perceptive 3D language assistant that achieves parallel capture of high-resolution local details via Hilbert curve partitioning, and aggregates local information with low-resolution global context using cross-attention and graph convolutional networks. This significantly improves fine-grained perception in 3D scene understanding without increasing the number of LLM input tokens.

## Background & Motivation

- **Development of 3D Language Assistants**: 3DLA aims to jointly process natural language and 3D data to achieve scene understanding (e.g., 3D QA, 3D dense captioning). The core challenge lies in efficiently converting 3D scene information into token representations that can be processed by LLMs.
- **Detail Loss Caused by Downsampling**: Existing methods (such as LL3DA) downsample point clouds to generate superpoints to control computational costs, which leads to the loss of critical local details—making them unable to distinguish, for example, a "black monitor" from a "black suitcase."
- **Simply Increasing Tokens is Ineffective**: Intuitively, increasing the number of visual tokens could preserve more information, but experiments show this has limited effect on capturing scene details while instead increasing the computational burden.
- **Successful Experience of 2D Multi-granularity**: 2D multimodal models such as Mini-Gemini and LLaVA-Next process high and low resolutions through dual branches or view partitioning, demonstrating that combining local and global views outperforms a single global view. However, this idea has not yet been explored in the 3D point cloud domain.
- **Unique Challenges of Point Cloud Partitioning**: Unlike grid pixels in images, point clouds are unordered point sets. How to partition while preserving locality and efficiently aggregate local-global information remains a key challenge.

## Method

### Overall Architecture

PerLA takes a point cloud $\mathcal{P}$, a textual prompt, and a visual prompt as inputs. Its core is a perceptive scene encoder: it first uses a Hilbert curve to serialize the point cloud and divides it into $L$ equal-sized partitions. A pre-trained 3D encoder is used to encode the full scene (low-resolution) and each partition (high-resolution) separately. Then, Hilbert k-NN, cross-attention, and a GCN are utilized to aggregate local-global information, generating enhanced point-level representations that are fed into the LLM (via a Q-former and a linear projection).

### Key Designs

**Design 1: Hilbert Curve Scene Partitioning and Parallel Encoding**
- **Function**: Partition the point cloud into equal-sized segments while preserving spatial locality.
- **Mechanism**: A Hilbert curve is used to serialize the 3D point cloud, yielding a one-dimensional arrangement that preserves spatial proximity, which is then evenly divided into $L$ parts (each containing $\lfloor N/L \rfloor$ points). This equal-cardinality strategy naturally allows semantically dense regions to receive spatially smaller partitions (higher sampling density) and sparse regions to receive larger ones. The same pre-trained encoder $\phi$ is applied to encode the full scene and the $L$ partitions separately. Each partition is downsampled to $M$ superpoints, meaning the local representations $\mathcal{F}^l$ are encoded at a higher resolution.
- **Design Motivation**: The Hilbert curve is the best locality-preserving spatial-filling curve, ensuring that points adjacent in the serialized sequence are also close in 3D space. Equal-cardinality partitioning naturally achieves adaptive density sampling.

**Design 2: Hilbert k-NN + Localized Cross-Attention**
- **Function**: Efficiently locate global-local superpoint correspondences and fuse information.
- **Mechanism**: Jointly serialize the global superpoints $\mathcal{P}^g$ and local superpoints $\mathcal{P}^l$, utilizing geometric labels to ensure that points of the same instance have indices within a contiguous range, thereby achieving $O(1)$ nearest neighbor search. For each global superpoint $p_i^g$ and its $k$ local nearest neighbors, 3D relative position encodings $\mathcal{R}_{ij} = \text{pos}((p_i^g - p_j^l)/\sigma)$ (3D Fourier encoding) are computed and aggregated through cross-attention: $\hat{f}_i^g = f_i^g + w_i(W_v(\mathcal{F}_{\mathcal{K}_i}^l + \mathcal{R}_i))$.
- **Design Motivation**: Conventional k-NN is computationally heavy on large-scale point clouds, whereas Hilbert serialization allows fast index-based lookups. Restricting cross-attention to a localized neighborhood both reduces computational complexity and ensures that the aggregated points likely belong to the same object.

**Design 3: GCN Message Passing and Local Representation Consistency Loss**
- **Function**: Refine the aggregated representations and stabilize training.
- **Mechanism**: GCN message passing is employed to further propagate the aggregated information among global superpoints, enhancing spatial context modeling. Concurrently, a local representation consistency loss is introduced to regularize local representations in overlapping areas, addressing representation divergence during local-global aggregation.
- **Design Motivation**: Cross-attention only handles local-global correspondence but lacks information exchange among global superpoints. The consistency loss ensures that different partitions generate compatible representations in overlapping or adjacent regions.

### Loss & Training

Training involves the standard next-token prediction loss for language models, combined with a local representation consistency regularization loss (smoothing loss + regularization loss). The latter constrains representations in overlapping areas encoded by different partitions to remain consistent, promoting training stability.

## Key Experimental Results

### Main Results

| Method | ScanQA CiDEr ↑ | ScanRefer C ↑ | Nr3D C ↑ |
|------|---------------|--------------|---------|
| 3D-LLM | 58.0 | - | - |
| LL3DA | 63.2 | 62.35 | 61.50 |
| Chat-Scene | 67.1 | 63.87 | 64.78 |
| **PerLA** | **+1.34 vs SOTA** | **+4.22 vs SOTA** | **+3.88 vs SOTA** |

### Ablation Study

| Configuration | ScanQA CiDEr | Effect |
|------|-------------|------|
| Increasing token count (no partitioning) | Slight improvement | Massive computational increase |
| Partitioning + Global (PerLA) | Significant improvement | Controllable computational increase |
| Global-only encoding | baseline | - |
| Local-only encoding | Performance drop | Lacks global context |

### Key Findings

1. PerLA achieves a SOTA performance boost of +1.34 CiDEr on ScanQA, and +4.22 and +3.88 on ScanRefer and Nr3D, respectively.
2. Simply increasing the token number for global encoding yields limited gains, whereas the local-global aggregation strategy performs significantly better under the same token budget.
3. Removing GCN message passing or consistency loss leads to a performance drop, verifying their complementary nature.
4. Hilbert curve partitioning outperforms random partitioning and spatial grid partitioning, better preserving spatial locality.

## Highlights & Insights

- **Detail-aware perception improvement without increasing LLM input tokens**: Solves the problem by better encoding (aggregating high-resolution local + low-resolution global) rather than greedily increasing token counts.
- **巧妙的 application of Hilbert curve**: Kills two birds with one stone by using it for locality-preserving partitioning and efficient $O(1)$ k-NN search.
- **Idea migration from 2D to 3D**: Effectively migrates successful multi-granularity strategies from 2D multimodal models (such as view partitioning in LLaVA-Next) to the 3D point cloud domain for the first time.

## Limitations & Future Work

- The partition count $L$ is a fixed hyperparameter, which may require tuning for scenes of different scales.
- Currently validated only on indoor scenes (ScanNet series); generalization to large-scale outdoor point clouds remains to be verified.
- The representation capability of the pre-trained 3D encoder itself (e.g., PointBERT) remains a bottleneck.
- GCN message passing adds extra computational overhead, which could become a bottleneck in ultra-large-scale scenes.

## Related Work & Insights

- The multi-granularity perspective of PerLA can be generalized to other point cloud tasks (e.g., 3D object detection, point cloud segmentation).
- The application of Hilbert curves in point cloud processing (such as PTv3) is increasingly popular; this work further utilizes it for k-NN acceleration.
- The design of local consistency loss can serve as a reference for other local-global fusion frameworks.

## Rating

⭐⭐⭐⭐ — Cleverly migrates the 2D multi-granularity perception concept to a 3D language assistant. The dual utilization of the Hilbert curve (partitioning + k-NN) is elegantly designed. The experiments are thorough with significant improvements, providing valuable insights to the 3D scene understanding field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Empowering Large Language Models with 3D Situation Awareness](empowering_large_language_models_with_3d_situation_awareness.md)
- [\[CVPR 2025\] Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](grounding_3d_object_affordance_with_language_instructions_visual_observations_an.md)
- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [\[CVPR 2025\] Dr. Splat: Directly Referring 3D Gaussian Splatting via Direct Language Embedding Registration](dr_splat_directly_referring_3d_gaussian_splatting_via_direct_language_embedding_.md)
- [\[CVPR 2025\] Perception Tokens Enhance Visual Reasoning in Multimodal Language Models](perception_tokens_enhance_visual_reasoning_in_multimodal_language_models.md)

</div>

<!-- RELATED:END -->
