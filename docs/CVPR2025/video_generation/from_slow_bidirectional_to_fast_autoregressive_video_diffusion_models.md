---
title: >-
  [Paper Note] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
description: >-
  [CVPR 2025][Video Generation][Autoregressive Diffusion] CausVid distills a pre-trained bidirectional video diffusion Transformer into an autoregressive 4-step causal generator through asymmetric distillation. Combined with ODE initialization and KV caching, it achieves streaming video generation at 9.4 FPS (160× faster than CogVideoX) and ranks first on the VBench-Long benchmark with a score of 84.27.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Autoregressive Diffusion"
  - "Distillation"
  - "KV Cache"
  - "Real-time Video"
date: 2026-05-08
content_hash: 55514475bec32a1d
---

# From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2412.07772](https://arxiv.org/abs/2412.07772)  
**Code**: [https://causvid.github.io/](https://causvid.github.io/) (Project page + Code)  
**Area**: Diffusion Models  
**Keywords**: Video Generation, Autoregressive Diffusion, Distillation, KV Cache, Real-time Video

## TL;DR
CausVid distills a pre-trained bidirectional video diffusion Transformer into an autoregressive 4-step causal generator through asymmetric distillation. Combined with ODE initialization and KV caching, it achieves streaming video generation at 9.4 FPS (160× faster than CogVideoX) and ranks first on the VBench-Long benchmark with a score of 84.27.

## Background & Motivation

**Background**: Current state-of-the-art video diffusion models (such as CogVideoX, MovieGen) are based on the Diffusion Transformer (DiT) architecture, which utilizes bidirectional attention to establish dependencies among all frames. While these models deliver outstanding generation quality, they suffer from severe latency issues—generating even a single frame requires processing the entire video sequence and demands dozens of denoising iterations.

**Limitations of Prior Work**: (1) **High Latency**: Bidirectional dependencies require waiting for the entire video to finish generating before any output can be viewed (CogVideoX takes 208 seconds for 128 frames); (2) **No Interaction**: Generating the current frame depends on future frames as conditioning inputs, preventing response to real-time user inputs; (3) **Length Limitation**: Computational and memory costs scale quadratically with the number of frames, making long video generation extremely expensive.

**Key Challenge**: Autoregressive models can resolve latency and interaction issues but face severe error accumulation—each frame is generated based on potentially flawed preceding frames, causing errors to amplify over time. Furthermore, the quality of existing autoregressive video models lags significantly behind bidirectional counterparts.

**Goal**: How to transfer the quality advantages of bidirectional video diffusion models to an autoregressive architecture while achieving fast streaming generation and resilience to error accumulation?

**Key Insight**: The authors observe that they can exploit the flexibility of Distribution Matching Distillation (DMD)—since DMD's supervision occurs at the distribution level rather than the trajectory level, it allows the teacher and student to use different architectures. Consequently, they can supervise a causal student with a bidirectional teacher, enabling the student to simultaneously acquire the quality of the bidirectional model and the efficiency of the causal model.

**Core Idea**: Use asymmetric DMD distillation to distill a multi-step bidirectional video diffusion model into a 4-step causal autoregressive generator, stabilizing training with ODE initialization and achieving streaming inference via KV caching.

## Method

### Overall Architecture
The pipeline of CausVid: (1) A 3D VAE compresses the video into the latent space (every 16 frames $\rightarrow$ 5 latent frames as a chunk); (2) In the latent space, a causal diffusion Transformer generates chunks autoregressively—bidirectional attention is used within the current chunk (preserving local temporal consistency), while causal attention is used between chunks (preventing dependence on future frames); (3) Training consists of two stages: first initializing the student with the teacher's ODE trajectory, then distilling via asymmetric DMD loss; (4) During inference, KV caching is employed for efficient streaming generation.

### Key Designs

1. **Block-wise Causal Attention**:

    - **Function**: Adapts the bidirectional DiT into a causal architecture that supports autoregressive generation while maintaining temporal consistency within each chunk.
    - **Mechanism**: Defines an attention mask $M_{i,j} = 1$ when $\lfloor j/k \rfloor \leq \lfloor i/k \rfloor$, where $k$ is the chunk size. That is, frames within the same chunk can attend to each other (bidirectional), but cannot attend to frames in future chunks. Similar to a decoder-only LLM, in each iteration, it can leverage supervisory signals from all input frames. Diffusion Forcing is adopted during training, where each chunk has an independent noise timestep $t^i$.
    - **Design Motivation**: Pure frame-level causal attention would sacrifice local temporal consistency, and a 3D VAE requires an entire chunk of latent frames to decode pixels; block-wise causal does not introduce additional latency. Initializing weights from a pre-trained bidirectional model accelerates convergence.

2. **Asymmetric Distillation**:

    - **Function**: Distills a multi-step bidirectional teacher model into a 4-step causal student model, equipping the student with teacher-level quality and effectively suppressing error accumulation.
    - **Mechanism**: Based on the DMD2 framework, the teacher $s_{data}$ uses bidirectional attention, while the student $G_\phi$ uses causal attention. During training, the student makes 4-step denoising predictions on noisy video frames, and then a DMD loss aligns the student's output distribution with the data distribution. The core gradient formula is: $\nabla_\phi \mathcal{L}_{DMD} \approx -\mathbb{E}_t[(s_{data} - s_{gen,\xi}) \cdot \frac{dG_\phi}{d\phi}]$, where $s_{gen,\xi}$ is an online-trained score function on the student's output. A two time-scale update rule (ratio of 5) is utilized to alternately update the student and the generator score function.
    - **Design Motivation**: Direct distillation from a causal teacher would inherit the causal model's defects (low quality, error accumulation). DMD's distribution-level supervision allows the teacher and student to use different architectures—the bidirectional teacher provides a higher-quality distribution target. Experiments show that the asymmetric-distilled causal student even outperforms multi-step causal models in quality.

3. **ODE-based Student Initialization**:

    - **Function**: Pre-trains the student using the teacher's ODE trajectory before distillation to stabilize the convergence of subsequent DMD training.
    - **Mechanism**: First, 1000 pairs of ODE trajectories (full paths from pure noise to clean videos) are generated using the bidirectional teacher, and a subset matching the student's inference timesteps is selected. The student is pre-trained on these trajectory pairs for 3000 iterations using a regression loss: $\mathcal{L}_{init} = \mathbb{E}[\|G_\phi(\{x_{t^i}\}, \{t^i\}) - \{x_0^i\}\|^2]$.
    - **Design Motivation**: Due to architectural discrepancies (bidirectional vs. causal), training directly with DMD loss is unstable. ODE initialization provides a reasonable starting point for the student—it already roughly knows how to map noise to clean videos, and subsequent DMD training only needs to fine-tune the distribution match based on this.

### Loss & Training
Two-stage training: (1) In the ODE initialization stage, MSE regression loss is used to train for 3000 steps with a learning rate of $5 \times 10^{-6}$. (2) In the asymmetric DMD distillation stage, the model is trained with DMD loss + an online score function for 6000 steps with a learning rate of $2 \times 10^{-6}$ and a guidance scale of 3.5. All training is completed on 64 H100 GPUs in about 2 days. During inference, uniform timesteps [999, 748, 502, 247] are used for 4-step denoising.

## Key Experimental Results

### Main Results

| Method | Video Length | Temporal Quality↑ | Frame Quality↑ | Text Alignment↑ | Latency (s)↓ | Throughput (FPS)↑ |
|------|---------|-------------------|----------------|-----------------|---------|-----------|
| CogVideoX-5B | 6s | 89.9 | 59.8 | 29.1 | 208.6 | 0.6 |
| MovieGen | 10s | 91.5 | 61.1 | 28.8 | - | - |
| Pyramid Flow | 10s | 89.6 | 55.9 | 27.1 | 6.7 | 2.5 |
| **CausVid (Ours)** | **10s** | **94.7** | **64.4** | **30.1** | **1.3** | **9.4** |

Long videos (30s):

| Method | Temporal Quality↑ | Frame Quality↑ | Text Alignment↑ |
|------|-------------------|----------------|-----------------|
| FIFO-Diffusion | 93.1 | 57.9 | 29.9 |
| Pyramid Flow | 89.0 | 48.3 | 24.4 |
| **CausVid (Ours)** | **94.9** | **63.4** | **28.9** |

### Ablation Study

| Configuration | Causal? | #Steps | Temporal↑ | Frame↑ | Text↑ |
|------|---------|-----|-----------|--------|-------|
| Bidirectional Teacher | ✗ | 100 | 94.6 | 62.7 | 29.6 |
| Causal Fine-tuning | ✓ | 100 | 92.4 | 60.1 | 28.5 |
| ODE init + No Teacher Distillation | ✓ | 4 | 92.9 | 48.1 | 25.3 |
| ODE init + Causal Teacher | ✓ | 4 | 91.9 | 61.7 | 28.2 |
| **ODE init + Bidirectional Teacher** | ✓ | **4** | **94.7** | **64.4** | **30.1** |

### Key Findings
- **Asymmetric distillation is the key breakthrough**: Causal teacher distillation (91.9) is far inferior to bidirectional teacher distillation (94.7). The 4-step causal student's Temporal Quality even surpasses that of the 100-step bidirectional teacher (94.7 vs 94.6).
- **DMD effectively suppresses error accumulation**: The 100-step causal model severely degrades in 30s videos (Fig. 8, orange line), whereas the DMD-distilled 4-step causal student maintains stable quality (blue line).
- **ODE initialization is indispensable**: Directly training with DMD without ODE initialization is unstable, and with ODE initialization, the Frame Quality increases from 48.1 to 64.4.
- Latency reduced by **160×** (208.6s $\rightarrow$ 1.3s), throughput increased by **16×** (0.6 $\rightarrow$ 9.4 FPS).
- In human preference studies, CausVid consistently outperforms MovieGen, CogVideoX, and Pyramid Flow (win rate >50%).

## Highlights & Insights
- The core insight of **asymmetric distillation**—DMD's distribution-level supervision allows the teacher and student to use different architectures. This breaks the conventional assumption that "distillation requires architectural consistency." The bidirectional teacher $\rightarrow$ causal student pathway enables the student to harvest the quality advantage of the teacher and its own efficiency advantage, which can be generalized to any distillation scenario where inference characteristics need to be altered.
- **Distillation can conversely solve error accumulation**: This is a counter-intuitive finding—the 4-step distilled student exhibits less error accumulation than the 100-step causal teacher. This is because DMD aligns at the distribution level rather than performing frame-by-frame regression, and the global consistency knowledge of the bidirectional teacher is transferred to the student during distillation.
- The combination of **KV caching + block-wise causal attention** allows video diffusion models to realize an LLM-like streaming generation paradigm for the first time.

## Limitations & Future Work
- Quality still degrades for extremely long videos (>10 minutes), and error accumulation strategies require further improvements.
- Constrained by the 3D VAE design, 5 latent frames must be generated before decoding pixels; a frame-level VAE could further reduce latency.
- The DMD objective based on reverse KL may reduce output diversity; alternative solutions like EM-Distillation could be considered.
- The current resolution of 352×640 is relatively low; scaling to higher resolutions requires more engineering optimization.
- Video-to-video translation and image-to-video functionalities are zero-shot; dedicated fine-tuning may further improve quality.

## Related Work & Insights
- **vs CogVideoX**: Both use the DiT architecture, but CogVideoX is bidirectional and multi-step, whereas CausVid is causal and 4-step. CausVid surpasses CogVideoX in quality (94.7 vs 89.9 Temporal Quality) while being 160× faster.
- **vs Pyramid Flow**: Pyramid Flow also supports autoregressoring but still requires multi-step denoising and degrades severely in long videos (Frame Quality 48.3). CausVid effectively addresses the degradation issue via asymmetric distillation.
- **vs FIFO-Diffusion**: FIFO also achieves streaming video generation (Temporal 93.1) but still requires multi-step denoising and is not truly autoregressive. CausVid is superior in both quality and efficiency.
- **vs DMD/DMD2**: CausVid extends DMD to the video domain and introduces a new paradigm of teacher-student architectural asymmetry.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Asymmetric distillation (bidirectional $\rightarrow$ causal) is a brand-new paradigm, and the finding that distillation solves error accumulation is counter-intuitive and important.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive; includes full VBench evaluation, human preference, long videos, ablations, and multiple application scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete algorithm pseudocode, and well-organized ablation studies.
- Value: ⭐⭐⭐⭐⭐ The first autoregressive video generation method to match the quality of bidirectional models, ranking 1st in VBench-Long with 160× acceleration, which has immense practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](../../ICCV2025/video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)
- [\[CVPR 2025\] Parallelized Autoregressive Visual Generation](parallelized_autoregressive_visual_generation.md)
- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2025\] InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
