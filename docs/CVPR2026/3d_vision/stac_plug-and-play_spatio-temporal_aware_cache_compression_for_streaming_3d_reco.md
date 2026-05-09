---
title: >-
  [Paper Note] STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction
description: >-
  [CVPR 2026][3D Vision][Streaming 3D Reconstruction] This paper proposes STAC, a framework that exploits spatio-temporal sparsity in the KV cache of causal Transformers. Through three modules—working temporal token caching, long-term spatial token caching, and chunk-based multi-frame optimization—STAC reduces memory consumption by approximately 10× and improves inference speed by 4× for streaming 3D reconstruction, without any additional training and with negligible degradation in reconstruction quality.
tags:
  - CVPR 2026
  - 3D Vision
  - Streaming 3D Reconstruction
  - KV Cache Compression
  - Spatio-Temporal Sparsity
  - Causal Transformer
  - Voxelized Storage
date: 2026-05-08
content_hash: 12159542b8732ce2
---

# STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.20284](https://arxiv.org/abs/2603.20284)
**Code**: [https://stac-3r.github.io/](https://stac-3r.github.io/) (Project Page)
**Area**: 3D Vision
**Keywords**: Streaming 3D Reconstruction, KV Cache Compression, Spatio-Temporal Sparsity, Causal Transformer, Voxelized Storage

## TL;DR

This paper proposes STAC, a framework that exploits spatio-temporal sparsity in the KV cache of causal Transformers. Through three modules—working temporal token caching, long-term spatial token caching, and chunk-based multi-frame optimization—STAC reduces memory consumption by approximately 10× and improves inference speed by 4× for streaming 3D reconstruction, without any additional training and with negligible degradation in reconstruction quality.

## Background & Motivation

1. **Background**: Transformer-based 3D reconstruction methods (e.g., VGGT) jointly infer camera parameters, depth maps, and point maps via feed-forward networks, achieving state-of-the-art performance. To support streaming input, causal variants (STream3R, StreamVGGT) replace global self-attention with causal self-attention and maintain historical frame context via KV caches.
2. **Limitations of Prior Work**: The KV cache grows linearly with the number of frames, causing severe memory bottlenecks in long-sequence scenarios. For instance, processing 200–300 frames can push the cache to nearly 20 GB. Early eviction strategies under memory constraints significantly degrade reconstruction quality and temporal consistency.
3. **Key Challenge**: Existing methods treat all tokens uniformly, ignoring the structured sparse patterns inherent in the cache—some tokens are relevant only at specific spatial locations (spatial sparsity), while others remain persistently important across frames (temporal sparsity). Uniform eviction/caching strategies cause valuable tokens to be discarded prematurely and redundant tokens to be retained unnecessarily.
4. **Goal**: (1) Maintain long-sequence 3D reconstruction quality under limited memory; (2) exploit spatio-temporal sparsity for KV cache compression; (3) achieve plug-and-play acceleration without additional training.
5. **Key Insight**: By analyzing attention maps of causal Transformers, the authors identify that different attention heads assume differentiated roles—some focus on spatial reasoning (attending to spatially adjacent regions) while others focus on temporal consistency (persistently attending to the first frame, landmark frames, or camera tokens). This intrinsic spatio-temporal sparsity provides a theoretical basis for structured compression.
6. **Core Idea**: Inspired by human memory mechanisms, the paper replaces the linearly growing KV cache with a dual-cache architecture consisting of working memory (short-term, high-fidelity) and long-term memory (spatially compressed storage).

## Method

### Overall Architecture

The input is a streaming sequence of video frames; each frame is encoded by a ViT to produce visual tokens. STAC maintains two complementary caches at each decoder layer of Causal-VGGT: (1) a working temporal cache $\mathcal{M}^{\text{temp}}$ that stores high-fidelity representations of recent frames and anchor tokens; and (2) a long-term spatial cache $\mathcal{M}^{\text{spat}}$ that stores evicted tokens in a voxel grid indexed by 3D position and performs online merging. During decoding, queries from the current frame attend to key-values in both caches. Additionally, chunk-based multi-frame optimization groups consecutive frames into chunks for joint processing, improving GPU utilization and temporal consistency.

### Key Designs

1. **Working Temporal Token Caching**:

    - **Function**: Maintains a compact short-term memory, preserving recent observations and globally important anchor tokens.
    - **Mechanism**: The cache consists of three token types: (a) global reference tokens $\mathcal{M}^{\text{refer}}$—all tokens from the first frame are permanently retained as global coordinate references; (b) sliding-window tokens $\mathcal{M}^{\text{window}}$—tokens from the most recent $s$ frames are retained to capture short-term motion continuity; (c) anchor tokens $\mathcal{M}^{\text{anchor}}$—persistently important tokens dynamically selected via decayed cumulative attention scores. Each token maintains an importance score $s_t^i = \gamma s_{t-1}^i + \sum_j \alpha_t^{j,i}$, where $\gamma \in (0,1)$ is a decay factor. Tokens evicted from the sliding window compete with existing anchor tokens, and the Top-K highest-scoring tokens are retained as new anchors.
    - **Design Motivation**: Attention analysis reveals that certain tokens (first frame, landmark frames, camera tokens) receive persistent attention across frames. A simple sliding window discards this global information; the exponentially decayed cumulative score effectively identifies and retains these temporal anchor points.

2. **Long-term Spatial Token Caching**:

    - **Function**: Organizes evicted tokens in a voxel grid indexed by 3D coordinates and performs online merging to achieve memory-efficient spatial reuse.
    - **Mechanism**: Each voxel maintains a dual buffer—a short-term buffer $\mathcal{E}_u$ that temporarily stores newly evicted tokens, and a long-term set $\mathcal{G}_u$ that stores merged representative tokens. Token merging comprises two operations: (a) one-to-one merging—when the cosine similarity between an evicted token and a long-term representative exceeds threshold $\lambda$, they are fused with similarity-weighted averaging: $\hat{m}^p \leftarrow \frac{Z(\hat{m}^p)\hat{m}^p + \omega(m^e,\hat{m}^p)m^e}{Z(\hat{m}^p) + \omega(m^e,\hat{m}^p)}$; (b) many-to-one aggregation—when the buffer is full, the highest-scoring token serves as a pivot and all buffered tokens are aggregated into a single new representative inserted into the long-term set. When a voxel is full, re-merging is triggered, fusing the least important representative into its nearest neighbor.
    - **Design Motivation**: In 3D reconstruction, spatially adjacent tokens are typically highly redundant (observations of the same object from different viewpoints), making voxelized organization a natural fit for spatial locality. The dual-buffer design prevents premature merging of potentially unique evidence while maintaining feature diversity.

3. **Chunk-based Multi-frame Optimization**:

    - **Function**: Groups adjacent frames for joint processing, improving GPU utilization and reconstruction consistency.
    - **Mechanism**: Consecutively arriving frames are grouped into temporal chunks of size 4. Bidirectional attention is applied within each chunk to enable limited information exchange, while the streaming constraint is maintained at chunk boundaries (no future frames are used). Morton encoding maps 3D voxel coordinates to linear indices, enabling batched token selection, merging, and retrieval operations.
    - **Design Motivation**: Frame-level processing incurs substantial kernel launch overhead; chunk-level execution amortizes this cost. Meanwhile, intra-chunk bidirectional attention allows neighboring frames to mutually correct each other, improving local geometric consistency.

### Loss & Training

STAC is completely training-free and can be applied directly as a plug-and-play module on top of pretrained Causal-VGGT. Only hyperparameters need to be configured: decay factor $\gamma=0.9$, voxel resolution 0.05, and merging threshold $\lambda=0.8$. A custom CUDA kernel is implemented to support merge-aware attention computation (with $\log n$ bias compensation).

## Key Experimental Results

### Main Results

Point cloud reconstruction results on NRGBD and 7-Scenes datasets (sampling stride 5, 200–300 frames/sequence):

| Method | Type | NRGBD Acc↓ | NRGBD Comp↓ | NRGBD NC↑ | 7-Scenes Acc↓ | 7-Scenes NC↑ | Memory (GB)↓ | FPS↑ |
|--------|------|-----------|------------|----------|--------------|-------------|-------------|------|
| VGGT | Offline | 0.017 | 0.012 | 0.740 | 0.022 | 0.602 | – | <1 |
| STream3R | Online | 0.053 | 0.013 | 0.703 | 0.044 | 0.606 | 19.75 | 2.52 |
| STream3R-W8 | Online | 0.078 | 0.015 | 0.687 | 0.107 | 0.587 | 0.86 | 6.19 |
| **STream3R-STAC** | **Online** | **0.065** | **0.014** | **0.700** | **0.047** | **0.606** | **2.20 (0.86)** | **10.53** |
| StreamVGGT | Online | 0.134 | 0.059 | 0.651 | 0.046 | 0.595 | 19.75 | 2.48 |
| StreamVGGT-STAC | Online | 0.126 | 0.047 | 0.682 | 0.056 | 0.596 | 2.57 (0.86) | 10.49 |

STAC reduces STream3R's memory from 19.75 GB to 2.20 GB (0.86 GB at runtime) and increases FPS from 2.52 to 10.53, with virtually no loss in reconstruction quality.

### Ablation Study

Using STream3R-W8 as the baseline on the NRGBD dataset (averaged over 7-Scenes metrics):

| Configuration | Acc↓ | Comp↓ | NC↑ | Memory (GB) | Runtime (ms) |
|---------------|------|-------|-----|-------------|-------------|
| Baseline (W8) | 0.0776 | 0.0150 | 0.6865 | 0.858 | 92.56 |
| w/o Anchor Tokens | 0.0725 | 0.0209 | 0.6991 | 1.901 | 61.36 |
| w/o Spatial Cache | 0.0713 | 0.0199 | 0.6939 | 0.572 | 39.08 |
| w/o Count Bias | 0.0666 | 0.0175 | 0.6973 | 2.063 | 56.08 |
| w/o Chunk Optimization | 0.0673 | 0.0156 | 0.6948 | 1.805 | 138.12 |
| **Full Model** | **0.0648** | **0.0142** | **0.6995** | **2.210** | **71.18** |

### Key Findings
- Chunk optimization has the largest impact on runtime: removing it nearly doubles latency from 71 ms to 138 ms while also degrading quality.
- The spatial cache increases memory consumption but significantly improves the Completion metric, indicating that long-term spatial information is critical for completeness.
- Anchor tokens have the greatest impact on the Comp metric (0.0142 → 0.0209), confirming the importance of persistent tokens for temporal consistency.
- Under the same memory budget, STAC consistently outperforms simple sliding-window extensions (W22/W26), validating the superiority of structured compression.

## Highlights & Insights
- **Training-free plug-and-play**: STAC requires no additional training and can be directly applied to any Causal-VGGT architecture (validated on both STream3R and StreamVGGT), substantially lowering the barrier to practical deployment.
- **Precise analogy to human memory**: The design of working memory (short-term, high-fidelity) plus long-term memory (compressed, retrievable storage) elegantly mirrors the dual-process model of human cognition. Its concrete realization at the token level in 3D reconstruction—where decayed cumulative scores identify anchor points—closely resembles the memory consolidation process.
- **Natural fit of voxelized organization**: By leveraging the fact that 3D reconstruction inherently requires 3D point outputs, the spatial cache is organized in a voxel grid enabling O(1) retrieval. This structural prior is not available to KV cache compression methods in other domains (e.g., LLMs).

## Limitations & Future Work
- The voxel grid resolution is fixed; in large-scale open scenes the number of active voxels grows continuously—CPU offloading or adaptive resolution strategies may be needed.
- In highly dynamic scenes, fast motion may introduce inconsistent token representations, affecting cache stability.
- Evaluation is currently limited to indoor datasets; generalization to large-scale outdoor scenarios (e.g., autonomous driving) remains to be explored.
- The sensitivity of hyperparameters such as the merging threshold $\lambda$ and voxel resolution to different scene types has not been systematically analyzed.

## Related Work & Insights
- **vs. StreamVGGT/STream3R**: These serve as the underlying architectures for STAC. The linear growth of the KV cache is the core bottleneck; STAC addresses this through structured compression without modifying training.
- **vs. H2O/StreamLLM (LLM KV cache compression)**: These methods target 1D text sequences and do not account for spatial structure. STAC introduces a voxelized spatial cache exploiting the 3D structural prior of reconstruction, representing the first work to incorporate spatio-temporal structured sparsity into KV cache compression for 3D vision models.
- **vs. Spann3R/CUT3R**: These methods rely on implicit or latent memory and suffer from drift and forgetting over long sequences. STAC's explicit dual-cache mechanism better preserves long-term consistency.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First systematic study of spatio-temporal sparsity in KV caches of causal Transformers for 3D vision; the overall design is elegant, though individual components are technically moderate.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple datasets, multiple baselines, detailed ablations, and comprehensive evaluation of memory, speed, and quality.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — In-depth motivation analysis, clear figures, and a complete logical chain from observations to design.
- **Value**: ⭐⭐⭐⭐ — High practical value (10× memory reduction + 4× speedup); the training-free property enables easy integration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](towards_spatio-temporal_world_scene_graph_generation_from_monocular_videos.md)
- [\[CVPR 2026\] STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding](sts_mixer_4d_point_cloud.md)
- [\[CVPR 2026\] LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](longstream_long-sequence_streaming_autoregressive_visual_geometry.md)
- [\[ICCV 2025\] LONG3R: Long Sequence Streaming 3D Reconstruction](../../ICCV2025/3d_vision/long3r_long_sequence_streaming_3d_reconstruction.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)

</div>

<!-- RELATED:END -->
