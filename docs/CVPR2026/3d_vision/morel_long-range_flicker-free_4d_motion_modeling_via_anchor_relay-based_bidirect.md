---
title: >-
  [Paper Note] MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification
description: >-
  [CVPR 2026][3D Vision][4D Gaussian Splatting] To address the challenges of memory explosion, temporal flickering, and occlusion handling in 4D Gaussian Splatting for long-video dynamic scene modeling…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "4D Gaussian Splatting"
  - "dynamic scene reconstruction"
  - "long video modeling"
  - "temporal consistency"
  - "memory efficiency"
date: 2026-05-08
content_hash: 62dd7048a0396cc8
---

# MoRel: Long-Range Flicker-Free 4D Motion Modeling via Anchor Relay-based Bidirectional Blending with Hierarchical Densification

**Conference**: CVPR 2026
**arXiv**: [2512.09270](https://arxiv.org/abs/2512.09270)
**Code**: [https://cmlab-korea.github.io/MoRel/](https://cmlab-korea.github.io/MoRel/)
**Area**: 3D Vision
**Keywords**: 4D Gaussian Splatting, dynamic scene reconstruction, long video modeling, temporal consistency, memory efficiency

## TL;DR
To address the challenges of memory explosion, temporal flickering, and occlusion handling in 4D Gaussian Splatting for long-video dynamic scene modeling, this paper proposes MoRel, a framework based on Anchor Relay-based Bidirectional Blending (ARBB). Through progressive construction of keyframe anchors and learnable temporal opacity control, MoRel achieves flicker-free, memory-bounded long-range 4D motion reconstruction.

## Background & Motivation

1. **Background**: 3D Gaussian Splatting (3DGS) has become the dominant paradigm for novel view synthesis and has been naturally extended to 4D dynamic scenes. Existing 4DGS methods are broadly categorized into two groups: "global one-shot training" and "chunk-based training."

2. **Limitations of Prior Work**:
    - **Global one-shot training** (e.g., 4DGS, MoDec-GS): All frames are optimized jointly, ensuring global temporal consistency, but GPU memory explodes for long videos as the number of high-dimensional Gaussians grows with sequence length.
    - **Chunk-based training** (e.g., GIFStream): Long videos are split into short segments trained independently, reducing memory overhead but introducing temporal discontinuities and appearance discontinuities at segment boundaries, i.e., "flickering" artifacts.
    - Sliding-window strategies only provide local patches and cannot guarantee global consistency; temporal Gaussian hierarchies achieve near-constant memory but at the cost of high system complexity.

3. **Key Challenge**: A fundamental tension in long-video modeling between global temporal consistency and bounded memory usage — smooth transitions must be maintained across thousands of frames without allowing memory to grow linearly with frame count.

4. **Goal**: (a) bounded-memory long-range 4D modeling; (b) flicker-free temporal consistency; (c) efficient random temporal access; (d) no reliance on external cues such as optical flow.

5. **Key Insight**: Inspired by the "keyframe + GOP" paradigm in video coding, keyframe anchors (KfA) are placed periodically along the time axis, enabling smooth transitions via bidirectional deformation and adaptive blending.

6. **Core Idea**: Replace global or chunk-based strategies with a keyframe anchor relay mechanism. Bidirectional deformations are learned between anchors and smooth blending is achieved via learnable opacity control, yielding memory-bounded, flicker-free long-range 4D reconstruction.

## Method

### Overall Architecture
MoRel adopts an anchor-based 3DGS representation in which anchors on a sparse voxel grid define the canonical space. Training proceeds in two phases comprising four sequential steps: the **Anchor Relay Phase** (GCA training → KfA training) and the **Bidirectional Blending Phase** (PWD training → IFB training). The inputs are multi-view long video sequences; the output is a 4D Gaussian representation supporting real-time rendering.

### Key Designs

1. **Global Canonical Anchors (GCA) + Keyframe Anchors (KfA)**:

    - **Function**: GCA sweeps over the entire video to train global anchors $\mathbf{A}^{\text{Global}}$, providing a globally consistent initialization for all subsequent KfAs; KfAs are placed periodically along the time axis and are fine-tuned within their respective temporal neighborhoods.
    - **Mechanism**: GCA is initialized from a single point cloud (rather than a temporally dense one) and coarsely trained on all frames. After training, anchors are assigned hierarchical levels according to their feature variance. All KfAs are initialized from GCA rather than trained from scratch; each KfA covers the temporal range $[t_n - \text{GOP}, t_n + \text{GOP}]$ and incorporates a temporal tolerance $\epsilon$ for robustness. Periodically placed KfAs are analogous to keyframes in video coding, providing random-access points and bounded memory.
    - **Design Motivation**: This ensures global appearance consistency while capturing fine-grained motion through local canonical spaces; on-demand loading and unloading of KfAs keeps memory bounded.

2. **Progressive Window Deformation (PWD) + Intermediate Frame Blending (IFB)**:

    - **Function**: PWD independently learns forward and backward deformation fields within each KfA's bidirectional deformation window; IFB trains learnable temporal opacity control between adjacent KfAs for smooth blending.
    - **Mechanism**: In PWD, each KfA is trained independently via dynamic loading and unloading, preventing "backward contamination" — the phenomenon where training on a later chunk modifies anchors depended upon by earlier chunks. Deformation fields use normalized relative time $\tau_n \in [-1, 1]$. In the IFB stage, all anchor attributes and deformation fields are frozen; only the temporal offset $o_{n,k}^{\text{dir}}$ and decay speed $d_{n,k}^{\text{dir}}$ are trained. Temporal opacity is defined as $w_{n,k}^{\text{dir}} = \exp[-\lambda_{\text{decay}} \cdot d_{n,k}^{\text{dir}} \cdot |\tau_n - o_{n,k}^{\text{dir}}|]$.
    - **Design Motivation**: PWD fundamentally prevents inter-chunk interference (the core flaw of chunk-wise training); IFB adaptively handles irregular motion such as occlusions via learnable parameters, outperforming simple linear interpolation.

3. **Feature-variance-guided Hierarchical Densification (FHD)**:

    - **Function**: Controls the densification schedule based on anchor feature variance, balancing memory usage and high-frequency detail preservation.
    - **Mechanism**: After GCA training, anchors are partitioned into three levels (low / mid / high frequency) according to feature variance $\sigma_k^2 = \text{Var}(\hat{f}_k)$. Early in training, low-frequency anchors receive weight 1 while high-frequency anchors receive a lower weight $\lambda_L$; as training proceeds, high-frequency weights are gradually increased via linear interpolation $\eta_t$. Gradient statistics are weighted per level as $g_L^{j_n^S} = g^{j_n^S} \cdot w_L^{j_n^S}$ and serve as the densification criterion.
    - **Design Motivation**: High-frequency regions are unstable early in training; premature densification produces redundant anchors. Delaying high-frequency densification controls memory while preserving final detail quality.

### Loss & Training
Standard 3DGS reconstruction loss (L1 + SSIM) is employed. The four training stages are executed sequentially. The critical mechanism is on-demand loading and unloading: at any point during training at most one to two KfAs and their associated deformation fields are loaded, ensuring bounded training and rendering memory.

## Key Experimental Results

### Main Results
The SelfCap_LR dataset is constructed (5 scenes, >3500 frames) with larger average motion magnitude and wider spatial extent.

| Method | Type | Avg PSNR↑ | Avg SSIM↑ | Avg LPIPS↓ | tOF↓ | Training Memory↓ |
|--------|------|-----------|-----------|------------|------|-----------------|
| 4DGS | Global | 18.95 | 0.648 | 0.402 | 0.222 | ~18,000MB |
| MoDec-GS | Global | 19.61 | 0.643 | 0.391 | 0.249 | ~22,000MB |
| LocalDyGS | Global | 20.64 | 0.652 | 0.371 | 0.215 | ~12,000MB |
| GIFStream | Chunk | 19.02 | 0.653 | 0.405 | 0.539 | ~9,000MB |
| 4DGS_chunk | Chunk | 19.31 | 0.656 | 0.389 | 0.680 | ~4,500MB |
| **MoRel** | **Ours** | **21.00** | **0.664** | **0.355** | **0.203** | **~6,000MB** |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Training Mem. | Rendering Mem. |
|---------------|-------|-------|--------|--------------|----------------|
| (a) GCA + unidirectional deformation | 19.71 | 0.654 | 0.386 | ~12,000 | 156 |
| (b) + KfA | 19.90 | 0.647 | 0.364 | ~4,500 | 94 |
| (c) + PWD + linear blending | 20.66 | 0.656 | 0.358 | ~6,500 | 138 |
| (d) + PWD + IFB | 21.07 | 0.672 | 0.342 | ~6,500 | 144 |
| (e) + FHD (full MoRel) | 21.20 | 0.672 | 0.348 | ~6,000 | 126 |

### Key Findings
- Introducing KfA reduces training memory from ~12,000MB to ~4,500MB (62.5%↓) while simultaneously improving LPIPS from 0.386 to 0.364.
- IFB improves PSNR by 0.41 dB over linear blending; learnable opacity control is critical for handling irregular motion.
- FHD reduces rendering memory from 144MB to 126MB while maintaining quality.
- MoRel achieves the best tOF of 0.203, far surpassing chunk-based methods at 0.539 and 0.680.

## Highlights & Insights
- **Elegant anchor relay concept**: The keyframe-in-video-coding analogy simultaneously resolves memory constraints and naturally provides random temporal access, which is practically valuable for streaming systems.
- **PWD eliminates "backward contamination"**: The fundamental flaw of chunk-based training is that later chunks corrupt earlier representations; PWD avoids this entirely through independent bidirectional deformation window training.
- **Transferable design philosophy of FHD**: Using feature variance as a proxy for frequency complexity to govern the densification schedule is applicable to static 3DGS scene reconstruction.

## Limitations & Future Work
- The four-stage sequential training pipeline incurs a long total training time; parallelizing certain stages warrants exploration.
- GOP size is fixed; adaptive adjustment based on motion complexity is a potential direction.
- Evaluation is primarily conducted on a self-collected dataset; generalization to other benchmarks requires further validation.
- The paper does not discuss handling of extreme motions such as objects that appear or disappear instantaneously.

## Related Work & Insights
- **vs. 4DGS (global)**: PSNR is 2.05 dB higher, tOF is 8.6% lower, with no memory explosion.
- **vs. GIFStream (chunk-based)**: IFB fundamentally resolves boundary flickering, reducing tOF from 0.539 to 0.203.
- **vs. TGH (hierarchical)**: Lower system complexity; no CPU–GPU streaming is required.

## Rating
- Novelty: ⭐⭐⭐⭐ The anchor relay and PWD strategies are novel, though the underlying intuition draws from video coding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are thorough; the primary dataset is self-collected.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures are clear and the logical flow is well-structured.
- Value: ⭐⭐⭐⭐ Practical value for long-video 4D reconstruction applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?](can_natural_image_autoencoders_compactly_tokenize_fmri_volumes_for_long-range_dy.md)
- [\[CVPR 2026\] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](long_scope_fully_sparse_long_range_cooperative_3d_perception.md)
- [\[CVPR 2026\] 4C4D: 4 Camera 4D Gaussian Splatting](4c4d_4_camera_4d_gaussian_splatting.md)
- [\[CVPR 2026\] GP-4DGS: Probabilistic 4D Gaussian Splatting from Monocular Video via Variational Gaussian Processes](gp-4dgs_probabilistic_4d_gaussian_splatting_from_monocular_video_via_variational.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)

</div>

<!-- RELATED:END -->
