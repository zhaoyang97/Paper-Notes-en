---
title: >-
  [Paper Note] CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video
description: >-
  [CVPR 2026][Video Generation][360° video generation] CubeComposer decomposes 360° video into a cubemap six-face representation and generates each face autoregressively in a spatio-temporal order…
tags:
  - "CVPR 2026"
  - "Video Generation"
  - "360° video generation"
  - "cubemap projection"
  - "spatio-temporal autoregression"
  - "diffusion models"
  - "native 4K generation"
date: 2026-05-08
content_hash: 66a914408ba9e95a
---

# CubeComposer: Spatio-Temporal Autoregressive 4K 360° Video Generation from Perspective Video

**Conference**: CVPR 2026
**arXiv**: [2603.04291](https://arxiv.org/abs/2603.04291)
**Code**: [Project Page](https://lg-li.github.io/project/cubecomposer)
**Area**: Video Generation / 360° Panoramic Video
**Keywords**: 360° video generation, cubemap projection, spatio-temporal autoregression, diffusion models, native 4K generation

## TL;DR

CubeComposer decomposes 360° video into a cubemap six-face representation and generates each face autoregressively in a spatio-temporal order, achieving for the first time native 4K (3840×1920) 360° panoramic video generation from perspective video without post-hoc super-resolution.

## Background & Motivation

Immersive VR applications demand high-quality 360° panoramic video, yet existing 360° video generation methods are constrained by the computational overhead of vanilla diffusion models:

- Existing methods achieve a native resolution of at most ≤1K (roughly 1024×512) and rely on external super-resolution modules for upscaling.
- External upsampling lacks intrinsic generative reasoning capability, frequently introducing cascading errors that yield high resolution with insufficient fine-grained detail.
- The memory and computational cost of full-attention diffusion models renders native high-resolution generation infeasible.

**Core Problem:** How can native 4K 360° video generation be achieved under tractable memory constraints?

## Method

### Overall Architecture

The input is a perspective video $\{I_t^{\mathrm{pers}}\}_{t=1}^N$ with known camera rotations. It is first projected into equirectangular format and then converted into a cubemap six-face representation (F/R/B/L/U/D) to form masked conditional inputs. The model partitions the temporal dimension into $L$ windows of length $T_{\mathrm{win}}$; within each window, faces are generated sequentially in descending order of coverage. Each step generates one face segment at a time, and the results are assembled into a 4K equirectangular output. The model is trained on top of the Wan 2.2 5B video foundation model.

### Key Designs

1. **Spatio-Temporal Autoregressive Generation Order Planning**: The temporal dimension follows a causal generation order. The spatial dimension orders faces by descending per-face coverage of the perspective video, computed as $c_{f,w} = \frac{1}{T_{\mathrm{win}}} \sum_{t=s_w}^{e_w-1} \langle M_{f,t} \rangle_{(i,j)}$. Generating faces with more conditional information first reduces early uncertainty and effectively propagates geometric, appearance, and motion cues to subsequent faces, mitigating error accumulation.

2. **Context Management and Sparse Attention**: The context $\mathbf{u}_{w,f}$ at each generation step comprises three components: (a) *historical tokens* — content already generated in the preceding $H$ windows; (b) *current-window tokens* — perspective conditions for both generated and ungenerated faces in the current window; (c) *future segment tokens* — dynamically selected from spatially adjacent future faces, retaining the nearest temporal segments whose coverage exceeds threshold $r$. To improve efficiency, a sparse context attention mechanism is designed: the generation sequence (length $G$) performs full self-attention, while the context sequence (length $C$) attends fully to the generation sequence but attends to itself only via a diagonal band mask of bandwidth $K$, reducing context self-attention complexity from $O(C^2)$ to $O(C \cdot K)$, i.e., linear complexity.

3. **Continuity-Aware Design**: To resolve seam artifacts at face boundaries after autoregressive assembly: (a) *Cube-aware positional encoding* — remaps the spatial indices of RoPE according to the unfolded cubemap topology (the top of the U face starts at 0, the F face at $R$, the D face at $2R$), encoding inter-face topological relationships; (b) *Cube-aware padding and blending* — pads the current face's latent with topologically aligned strips from neighboring faces during generation, then blends overlapping regions via weighted averaging in pixel space after generation to ensure smooth transitions.

### Loss & Training

- The model is trained with a flow-matching objective on the velocity field: $\mathcal{L} = \mathbb{E}_{t,\mathbf{z}_0}\left[\|\mathbf{v}_\theta(\mathbf{z}_t, t; \mathbf{u}_{w,f}, y) - \mathbf{v}_t\|^2\right]$
- During training, the autoregressive process is simulated on ground-truth 360° videos with randomly sampled windows and faces.
- The model supports a global prompt and optional per-face prompt conditioning; per-face captions are used randomly during training.
- The 4K360Vid dataset comprises 11,832 high-quality 4K 360° video clips, captioned by Qwen3-VL with low-quality content filtered out.

## Key Experimental Results

### Main Results

| Method | Resolution | LPIPS↓ | CLIP↑ | FID↓ | FVD↓ | Aesthetic Quality↑ | Imaging Quality↑ |
|--------|-----------|--------|-------|------|------|-------------------|-----------------|
| Argus | 1K | 0.407 | 0.886 | 141.2 | 4.08 | 0.372 | 0.427 |
| Argus+VEnhancer | 2K | 0.469 | 0.858 | 169.0 | 6.13 | 0.360 | 0.429 |
| CubeComposer | 2K | **0.370** | **0.923** | **119.1** | 3.90 | 0.398 | **0.521** |
| CubeComposer | 4K | 0.383 | 0.911 | 130.9 | **2.22** | **0.405** | **0.562** |

CubeComposer significantly outperforms all baselines on both the 4K360Vid and ODV360 datasets without relying on any super-resolution post-processing.

### Ablation Study

| Configuration | FVD↓ | FID↓ | LPIPS↓ | CLIP↑ |
|--------------|------|------|--------|-------|
| Full model | **4.26** | **125.6** | **0.425** | **0.891** |
| w/o future tokens | 6.04 | 128.3 | 0.452 | 0.888 |
| Full-token context | 5.23 | 116.6 | 0.416 | 0.896 |
| w/o cube positional encoding | 4.47 | 201.4 | 0.550 | 0.855 |
| w/o padding & blending | 4.37 | 190.3 | 0.560 | 0.841 |

### Key Findings

- Future segment tokens are critical for temporal consistency (FVD degrades from 4.26 to 6.04 without them).
- The full model even surpasses the full-token context model on FVD (4.26 vs. 5.23), demonstrating that selective context is more effective than exhaustive context.
- Both continuity-aware components are indispensable; removing either introduces severe seam artifacts.

## Highlights & Insights

- The paper elegantly reformulates 360° video generation as a spatio-temporal autoregressive problem over cubemap faces, circumventing the memory bottleneck of native high-resolution generation.
- Coverage-guided spatial ordering is the core innovation — generating the most constrained faces first naturally propagates information to subsequent faces.
- The sparse context attention design is concise and efficient; linear complexity makes long-context generation practical.
- The 4K360Vid dataset with 11K+ captioned videos is itself a standalone contribution.

## Limitations & Future Work

- The total inference latency of face-by-face autoregressive generation is high; reducing diffusion steps or streaming generation warrants exploration.
- The cubemap representation still introduces some distortion near the poles.
- Temporal consistency for fast-motion scenes may be suboptimal.
- The current approach assumes known camera rotations; scenes without rotation estimates require additional processing.

## Related Work & Insights

- Compared with existing 360° video generation methods such as Argus, Imagine360, and ViewPoint, CubeComposer is the first to surpass the 1K resolution barrier.
- Relative to temporally autoregressive video generation methods (e.g., StreamDiffusion), CubeComposer introduces an additional spatial autoregressive dimension.
- The sparse attention design is transferable to other video generation tasks requiring long-context modeling.

## Rating

- Novelty: ⭐⭐⭐⭐ — The cubemap spatio-temporal autoregressive framework is novel; coverage-guided ordering and sparse attention are well-motivated design choices.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, detailed ablations, and comparisons against multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ — Clear figures and tables; the method is described systematically and formally.
- Value: ⭐⭐⭐⭐ — First native 4K 360° video generation with practical applicability to VR content creation.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](../../ICLR2026/video_generation/lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[ICLR 2026\] JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization](../../ICLR2026/video_generation/javisdit_joint_audio-video_diffusion_transformer_with_hierarchical_spatio-tempor.md)
- [\[CVPR 2026\] VideoCoF: Unified Video Editing with Temporal Reasoner](videocof_unified_video_editing_with_temporal_reasoner.md)
- [\[CVPR 2026\] I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)
- [\[CVPR 2026\] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)

</div>

<!-- RELATED:END -->
