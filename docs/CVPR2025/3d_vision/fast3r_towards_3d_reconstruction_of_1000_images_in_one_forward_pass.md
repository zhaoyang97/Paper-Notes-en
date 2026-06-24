---
title: >-
  [Paper Note] Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass
description: >-
  [CVPR 2025][3D Vision][3D reconstruction] Proposes Fast3R, which generalizes the pairwise pointmap regression of DUSt3R to multi-view scenarios. By employing all-to-all self-attention in a Transformer, it processes $N$ unposed and unordered images in a single forward pass, completely eliminating the $O(N^2)$ pairwise inference and global alignment optimization.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D reconstruction"
  - "multi-view"
  - "Transformer"
  - "pointmap regression"
  - "DUSt3R"
  - "scalability"
date: 2026-05-08
content_hash: 946158f8ff432740
---

# Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass

**Conference**: CVPR 2025  
**arXiv**: [2501.13928](https://arxiv.org/abs/2501.13928)  
**Code**: [https://fast3r-3d.github.io](https://fast3r-3d.github.io)  
**Area**: 3D Vision  
**Keywords**: 3D reconstruction, multi-view, Transformer, pointmap regression, DUSt3R, scalability

## TL;DR

Proposes Fast3R, which generalizes the pairwise pointmap regression of DUSt3R to multi-view scenarios. By employing all-to-all self-attention in a Transformer, it processes $N$ unposed and unordered images in a single forward pass, completely eliminating the $O(N^2)$ pairwise inference and global alignment optimization.

## Background & Motivation

**Background**: DUSt3R pioneered the end-to-end 3D reconstruction paradigm by simplifying multi-view geometry into pointmap regression. However, its core limitation is that it can only handle image pairs, requiring $O(N^2)$ pairwise inferences followed by global alignment optimization for $N$ images.

**Limitations of Prior Work**:
- DUSt3R suffers from OOM (on an A100 GPU) with just 48 views, making it unable to handle large-scale scenes.
- The pairwise processing paradigm limits the model's context, failing to utilize global information across multiple views and leading to accumulated errors.
- Traditional SfM/MVS pipelines (e.g., COLMAP) require complex multi-stage engineering and exhibit a >40% catastrophic failure rate on datasets like ETH-3D.
- Spann3R scales to a larger number of views via a sliding window and spatial memory, but its sequential processing cannot correct early errors.

**Key Challenge**: DUSt3R demonstrates the effectiveness of end-to-end pointmap regression, but its pairwise assumption creates a scalability bottleneck in terms of both computational complexity and information utilization efficiency.

**Goal**: To achieve a 3D reconstruction model capable of processing 1000+ unordered images in a single forward pass.

**Key Insight**: Replace pairwise processing with the all-to-all self-attention of Transformers, allowing all views to simultaneously attend to each other and output pointmaps in a global coordinate system in one go.

**Core Idea**: Simplify the "pairwise regression + global alignment" pipeline of DUSt3R into "one-shot multi-view regression", achieving globally consistent 3D reconstruction using the parallel attention mechanism of Transformers.

## Method

### Overall Architecture

1. **Image Encoder**: Independently encodes $N$ images into patch features (CroCo ViT-L).
2. **Fusion Transformer**: A 24-layer ViT-L that performs all-to-all self-attention over patch tokens from all views, incorporating one-dimensional image index positional embeddings.
3. **Pointmap Decoding Heads**: Two DPT heads output the local pointmap $\mathbf{X}_L$ and global pointmap $\mathbf{X}_G$, respectively, along with their respective confidence maps.

### Key Designs

**1. Image Index Positional Embedding + Position Interpolation**
- **Function**: Adds a 1D index positional embedding to the patch tokens of each image to help the Transformer distinguish different images; during training, $N=20$ indices are randomly sampled from a pool of $N'=1000$.
- **Mechanism**: Leverages Position Interpolation from LLMs—training with 20 views but randomly sampling indices from 1000 candidate positions (equivalent to image masking), allowing direct scaling to 1000 images during inference. The first image always uses $p_1$ (defining the global coordinate system).
- **Design Motivation**: Naive continuous index embeddings suffer from severe performance drops when the number of testing views exceeds the training range. The random sampling strategy exposes the model to sparse index distributions during training, enabling seamless generalization from 20 to 1000+ views.

**2. Dual Pointmaps + Confidence-Weighted Loss**
- **Function**: Predicts a local pointmap (in individual camera coordinate systems) and a global pointmap (in the first camera's coordinate system), each accompanied by a confidence map.
- **Mechanism**: The total loss is defined as $\mathcal{L}_{total} = \mathcal{L}_{\mathbf{X}_G} + \mathcal{L}_{\mathbf{X}_L}$, where each loss uses normalized 3D regression + confidence weighting: $\mathcal{L}_\mathbf{X} = \frac{1}{|\mathbf{X}|}\sum \hat{\Sigma}_+ \cdot \ell_{regr} + \alpha \log(\hat{\Sigma}_+)$.
- **Design Motivation**: Confidence weighting helps the model cope with label noise (e.g., errors in glass or thin structures from laser scans), while $\Sigma_+ = 1 + \exp(\hat{\Sigma})$ ensures positive values.

**3. Memory-Efficient Inference (Tensor Parallelism)**
- **Function**: Replicates DPT heads across multiple GPUs during inference; after running the ViT encoder and fusion transformer on GPU 0, the outputs are distributed to $K$ GPUs for parallel decoding.
- **Mechanism**: Analysis reveals that the DPT head consumes >60% of inference VRAM (due to upsampling 1024 tokens to 512×512 images), constituting the primary memory bottleneck.
- **Design Motivation**: Utilizing DeepSpeed ZeRO stage 2 + FlashAttention during training enables model training with a batch size of 128 and $N=20$ on 128 A100 GPUs.

### Loss & Training

- Normalized 3D pointwise regression loss (normalizing predictions and ground truth separately to eliminate scale ambiguity)
- Confidence-weighted loss to handle label noise
- AdamW, lr=0.0001, cosine annealing, 174K steps
- Training data: CO3D, ScanNet++, ARKitScenes, Habitat, BlendedMVS, MegaDepth (6 out of 9 datasets used in DUSt3R)
- 6.13 days × 128 A100-80GB

## Key Experimental Results

### Main Results (Camera Pose Estimation, CO3Dv2 10-view)

| Method | RRA@15°↑ | RRA@5°↑ | FPS |
|---|---|---|---|
| DUSt3R | 96.2 | - | 0.78 |
| MASt3R | 94.6 | 93.2 | 0.23 |
| Fast3R-no-outdoor | **99.7** | **97.4** | **251.1** |
| Fast3R | 96.2 | 90.2 | **251.1** |

### Scalability Comparison

| Number of Views | Fast3R Time (s) | Fast3R Memory (GiB) | DUSt3R Time (s) | DUSt3R Memory (GiB) |
|---|---|---|---|---|
| 8 | 0.122 | 6.33 | 8.386 | 24.59 |
| 32 | 0.509 | 13.25 | 129.0 | 67.61 |
| 48 | 0.84 | 20.8 | OOM | OOM |
| 320 | 15.94 | 41.90 | OOM | OOM |
| 1000 | 137.62 | 63.01 | OOM | OOM |

### 3D Reconstruction Quality (7-scenes)

| Method | FPS | Acc↓ | Comp↓ |
|---|---|---|---|
| DUSt3R | 0.78 | 1.23 | 0.91 |
| Spann3R | 65.4 | 1.48 | 0.85 |
| Fast3R | **251.1** | — | — |

### Key Findings

1. **Performance scales with number of views**: The model's accuracy increases when trained on more views, and increasing the number of views during inference continues to improve reconstruction quality. This demonstrates generalization to scales far exceeding the training views.
2. **Orders of magnitude speedup**: Compared to DUSt3R, the model is 68× faster at 8 views and 253× faster at 32 views, whereas DUSt3R experiences OOM beyond 48 views.
3. **All-to-all attention outperforms pairwise**: By eliminating the error accumulation inherent in pairwise processing, it reduces the 15° error by 14× compared to DUSt3R's global alignment on CO3Dv2.
4. **Critical role of Position Interpolation**: Without the PI strategy, model performance drops sharply when the test view count exceeds the training range.

## Highlights & Insights

- Fundamentally solves the scalability bottleneck of DUSt3R, reducing computational complexity from $O(N^2)$ pairwise processing to a single forward pass.
- Showcases an innovative application of Position Interpolation by migrating it from LLMs to 3D reconstruction, achieving cross-domain generalization over the number of views.
- Leverages standard Transformer infrastructure (FlashAttention, DeepSpeed) to continuously benefit from system-level optimizations.
- Achieves comparable or superior accuracy while utilizing only 6 out of 9 datasets from DUSt3R.

## Limitations & Future Work

- The DPT head remains the bottleneck for inference memory as each image requires upsampling to high resolutions.
- Although capable of handling 1000+ views, the quadratic complexity of all-to-all attention remains a long-term bottleneck.
- Does not currently integrate the local feature matching capability of MASt3R.
- Scale has only been explored up to ViT-L; larger models may yield further improvements.
- Performance on dynamic scenes remains unverified.
- High training cost (128 GPUs × 6 days).

## Related Work & Insights

- DUSt3R pioneered the pointmap regression paradigm; Fast3R naturally generalizes it to multi-view scenarios.
- Spann3R scales the number of views via sequential processing plus spatial memory but cannot correct early errors; Fast3R's all-to-all strategy is superior.
- The dual success of Position Interpolation in LLMs (RoPE scaling) and 3D reconstruction indicates the universality of this technique.
- Insight: The maturation of large-scale Transformer infrastructure is poised to drive end-to-end revolutions in a wider range of traditional computer vision tasks.

## Rating

⭐⭐⭐⭐⭐ — Exceptionally solves the scalability problem of multi-view 3D reconstruction, achieving orders of magnitude speedup without sacrificing accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MUSt3R: Multi-view Network for Stereo 3D Reconstruction](must3r_multi-view_network_for_stereo_3d_reconstruction.md)
- [\[CVPR 2025\] HandOS: 3D Hand Reconstruction in One Stage](handos_3d_hand_reconstruction_in_one_stage.md)
- [\[CVPR 2026\] UniSH: Unifying Scene and Human Reconstruction in a Feed-Forward Pass](../../CVPR2026/3d_vision/unish_unifying_scene_and_human_reconstruction_in_a_feed-forward_pass.md)
- [\[CVPR 2026\] Omni-3DEdit: Generalized Versatile 3D Editing in One-Pass](../../CVPR2026/3d_vision/omni-3dedit_generalized_versatile_3d_editing_in_one-pass.md)
- [\[CVPR 2025\] Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors](pow3r_empowering_unconstrained_3d_reconstruction_with_camera_and_scene_priors.md)

</div>

<!-- RELATED:END -->
