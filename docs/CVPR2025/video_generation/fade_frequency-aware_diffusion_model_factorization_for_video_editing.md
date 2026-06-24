---
title: >-
  [Paper Note] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing
description: >-
  [CVPR 2025][Video Generation][video editing] Proposes FADE, a training-free video editing method. By analyzing the frequency roles (sketching vs. sharpening) of each transformer block in T2V models, it leverages spectrum-guided modulation to separate preserved and edited content in the frequency domain, achieving high-quality appearance and motion editing.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "video editing"
  - "diffusion model"
  - "frequency-aware factorization"
  - "spectrum-guided modulation"
  - "training-free"
date: 2026-05-08
content_hash: b42e0318a603b39e
---

# FADE: Frequency-Aware Diffusion Model Factorization for Video Editing

**Conference**: CVPR 2025  
**arXiv**: [2506.05934](https://arxiv.org/abs/2506.05934)  
**Code**: [https://github.com/EternalEvan/FADE](https://github.com/EternalEvan/FADE)  
**Area**: Video Generation  
**Keywords**: video editing, diffusion model, frequency-aware factorization, spectrum-guided modulation, training-free

## TL;DR

Proposes FADE, a training-free video editing method. By analyzing the frequency roles (sketching vs. sharpening) of each transformer block in T2V models, it leverages spectrum-guided modulation to separate preserved and edited content in the frequency domain, achieving high-quality appearance and motion editing.

## Background & Motivation

**Background**: Diffusion models have elevated video editing capabilities to high levels of fidelity and strong text alignment, but traditional methods based on T2I models perform poorly when handling video dynamics (especially motion editing).

**Limitations of Prior Work**:
- T2I models lack video priors, leading to temporal inconsistency and limited motion editing capabilities.
- Null-text inversion methods require extensive iterative calculations of null-text embeddings, which is highly time-consuming.
- Attention feature injection methods have heavy memory requirements and limit editing flexibility.
- Directly adding noise on video diffusion models (e.g., CogVideoX-V2V) fails to fully exploit video priors.

**Key Challenge**: Video diffusion models (T2V) possess rich spatiotemporal prior knowledge, but their massive computational demands (full-attention across dozens of transformer blocks) make direct migration of prior T2I editing techniques unfeasible.

**Goal**: Design an efficient and flexible video editing strategy that can fully utilize the video priors in pre-trained T2V models, supporting both appearance and motion editing.

**Key Insight**: Analyze the functional division of each block in the T2V model from a frequency-domain perspective. This reveals that early blocks are responsible for outlining low-frequency spatial layout and temporal dynamics (sketching blocks), while later blocks refine high-frequency details (sharpening blocks). The role factorization is based on this discovery.

**Core Idea**: Through frequency-aware block factorization + spectrum-guided modulation, using only a few sketching blocks to provide low-frequency structural guidance can reduce computational overhead while unleashing the editing potential of video diffusion priors.

## Method

### Overall Architecture

1. Use a pre-trained T2V model (CogVideoX, 48-layer DiT) to perform DDIM inversion on the input video to obtain the noise $\boldsymbol{z}_T^*$ and the inversion trajectory $\{\boldsymbol{z}_t^*\}_{t=0}^T$.
2. During the denoising sampling process, extract the full-attention outputs $\boldsymbol{F}_t^*$ and $\boldsymbol{F}_t$ for the source and target videos from the first 4 sketching blocks.
3. Apply 3D DFT transform to the attention outputs, separating low-frequency structural information via a low-pass filter.
4. Calculate the spectrum guidance term $\mathcal{G}_t$, and modulate the DDIM sampling trajectory using its gradient.

### Key Designs

**1. Frequency-Aware Factorization**
- **Function**: Divides the 48 transformer blocks of the T2V model into sketching blocks (first 4 layers) and sharpening blocks (remaining 44 layers).
- **Mechanism**: Visual analysis shows that the attention maps in early blocks are densely aligned along the diagonals (main diagonal = intra-frame space, sub-diagonals = inter-frame temporal correspondence), with the spectrum concentrated in low frequencies and outputs being blurry—they sketch out the basic layout and motion. Late blocks have sparser and more uniform attention distributions, processing high-frequency textures, colors, and other details.
- **Design Motivation**: Leveraging this functional division, editing only requires manipulating the sketching blocks for structural reconstruction guidance. The sharpening blocks are left to freely generate details, which is both efficient (reducing computation) and flexible (no restrictions on high-frequency editing).

**2. Spectrum-Guided Modulation**
- **Function**: Transforms the attention outputs of sketching blocks into the frequency domain, extracts low-frequency components using a low-pass filter, and calculates the low-frequency difference between the source and target videos as a guidance signal.
- **Mechanism**: Performs 3D DFT (spatial + temporal dimensions) on the attention output $\boldsymbol{F}_t$ to obtain $\mathcal{F}_t$; after low-pass filtering, calculates spectrum guidance $\mathcal{G}_t = \|\text{LP}(\mathcal{F}_t) - \text{LP}(\mathcal{F}_t^*)\|_2^2$; modulates the sampling trajectory using the gradient of $\mathcal{G}_t$ with respect to $\boldsymbol{z}_t$: $\boldsymbol{z}_{t-1} = \text{DDIM}(\boldsymbol{\epsilon}_\theta, \boldsymbol{z}_t, t, \boldsymbol{y}_{tgt}) - \lambda \text{Norm}(\nabla_{\boldsymbol{z}_t} \mathcal{G}_t)$.
- **Design Motivation**: Guiding in the frequency domain instead of the feature domain elegantly avoids information leakage (inappropriate preservation of source video's high-frequency details) caused by direct attention feature injection. It retains only the low-frequency structure (basic spatial layout + temporal motion), leaving space for high-frequency detail editing.

**3. Dual Branch Strategy**
- **Function**: In each denoising step, simultaneously runs the sketching blocks on the source video inversion trajectory $\boldsymbol{z}_t^*$ and target video $\boldsymbol{z}_t$, using the same source prompt $\boldsymbol{y}_{src}$ to calculate attention outputs.
- **Mechanism**: The source branch provides reference structural information, while the target branch undergoes full denoising using the editing prompt $\boldsymbol{y}_{tgt}$. The difference between the sketching block outputs of the two branches drives the spectrum guidance.
- **Design Motivation**: Avoids the iterative overhead of null-text optimization and does not require direct swapping or blending of attention maps, providing a more flexible guidance mechanism.

### Loss & Training

- **Training-free method**, requiring no optimization or fine-tuning.
- Uses $T=50$ steps of DDIM sampling, with the guidance interval set to $[0, 0.6T]$.
- The guidance weight $\lambda$ is between 10-15, adjusted based on the editing task.
- Uses multimodal language models like BLIP to automatically generate text descriptions of the source video.
- The low-pass filter retains approximately 2/3 of the frequency components.

## Key Experimental Results

### Main Results

| Method | CLIP↑ | M.PSNR↑ | LPIPS↓ | OSV↓ | Human Preference↑ |
|---|---|---|---|---|---|
| **Appearance Editing** | | | | | |
| Tune-A-Video | 0.3522 | 19.86 | 0.4625 | 35.01 | 0.12 |
| FateZero | 0.3562 | 20.65 | 0.3057 | 33.23 | 0.29 |
| CogVideoX-V2V | 0.3754 | 18.96 | 0.4811 | 31.45 | 0.09 |
| **FADE (Ours)** | **0.3762** | **20.69** | **0.3085** | **31.36** | **0.35** |
| **Motion Editing** | | | | | |
| Tune-A-Video | 0.3281 | 18.68 | 0.4637 | 35.85 | 0.10 |
| FateZero | 0.3259 | 19.02 | 0.3712 | 34.47 | 0.13 |
| CogVideoX-V2V | 0.3678 | 18.17 | 0.4928 | 35.52 | 0.19 |
| **FADE (Ours)** | **0.3683** | **19.26** | **0.3692** | **32.28** | **0.43** |

### Ablation Study

| Configuration | CLIP↑ | M.PSNR↑ | LPIPS↓ | OSV↓ | Time |
|---|---|---|---|---|---|
| Symm. blocks | 0.3659 | 20.73 | 0.3367 | 32.61 | 5 min |
| W/o factorization | 0.3691 | 20.94 | 0.3328 | 32.05 | 12 min |
| W/o filter | 0.3612 | 20.89 | 0.3364 | 32.28 | 3 min |
| **FADE (Ours)** | **0.3728** | 20.87 | 0.3352 | **31.77** | **3 min** |

### Key Findings

1. **Sketching blocks are sufficient**: Using only the first 4 blocks (out of 48) achieves the best editing quality. Adding sharpening blocks instead misleads the model and reduces editing performance.
2. **Critical role of low-pass filtering**: Removing the low-pass filter causes high-frequency information leakage, leading to the target object retaining too many source characteristics and a decrease in text alignment (CLIP drops from 0.3728 to 0.3612).
3. **Significant advantage in motion editing**: FADE's human preference score (0.43) in motion editing far exceeds other methods, thanks to the full utilization of video priors.
4. **Efficiency boost**: FADE completes editing in 3 minutes, whereas traditional methods require 15 minutes or more.

## Highlights & Insights

- Reveals the functional division of blocks within T2V models (sketching vs. sharpening) from a frequency-domain perspective, a discovery that holds independent value.
- The design of performing guidance in the frequency domain instead of the feature domain cleverly avoids information leakage issues.
- The training-free design makes the method highly practical and directly applicable to various T2V models.
- Provides a unified framework that supports both appearance and motion editing.
- Counter-intuitive discovery: Using fewer blocks for guidance actually leads to better editing results.

## Limitations & Future Work

- Editing performance depends on the generative capability of the underlying T2V model.
- In severe occlusion scenarios, advanced temporal reasoning capabilities are required, which current models fail to adequately address.
- Only one T2V model (CogVideoX) was explored, leaving generalization across other architectures unverified.
- The frequency truncation ratio (2/3) of the low-pass filter requires empirical tuning.
- The complexity of motion editing is limited, making it difficult to handle large-scale motion changes.

## Related Work & Insights

- Compared to attention injection methods like FateZero, frequency-domain guidance is more flexible and is not constrained by attention map swapping.
- Significantly outweighs CogVideoX-V2V while using the same T2V model, proving the value of the factorization strategy.
- The frequency-aware paradigm can be extended to other diffusion model tasks (such as 3D generation, audio editing, etc.).
- The block function analysis methodology can be used to understand other large-scale Transformer architectures.

## Rating

⭐⭐⭐⭐ — Innovative frequency-domain analysis perspective + highly practical training-free design, but limited to specific T2V models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Identity-Preserving Text-to-Video Generation by Frequency Decomposition](identity-preserving_text-to-video_generation_by_frequency_decomposition.md)
- [\[CVPR 2025\] Improved Video VAE for Latent Video Diffusion Model](improved_video_vae_for_latent_video_diffusion_model.md)
- [\[CVPR 2025\] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion](posetraj_pose-aware_trajectory_control_in_video_diffusion.md)
- [\[CVPR 2025\] SketchVideo: Sketch-Based Video Generation and Editing](sketchvideo_sketch-based_video_generation_and_editing.md)
- [\[CVPR 2025\] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](timestep_embedding_tells_its_time_to_cache_for_video_diffusion_model.md)

</div>

<!-- RELATED:END -->
