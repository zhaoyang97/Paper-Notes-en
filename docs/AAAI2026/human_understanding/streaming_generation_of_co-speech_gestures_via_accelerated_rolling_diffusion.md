---
title: >-
  [Paper Note] Streaming Generation of Co-Speech Gestures via Accelerated Rolling Diffusion
description: >-
  [AAAI2026][Human Understanding][co-speech gestures] This paper proposes a streaming co-speech gesture generation framework based on Rolling Diffusion…
tags:
  - "AAAI2026"
  - "Human Understanding"
  - "co-speech gestures"
  - "rolling diffusion"
  - "streaming generation"
  - "real-time"
  - "noise scheduling"
  - "motion synthesis"
date: 2026-05-08
content_hash: df56be8dc1d4e665
---

# Streaming Generation of Co-Speech Gestures via Accelerated Rolling Diffusion

**Conference**: AAAI2026
**arXiv**: [2503.10488](https://arxiv.org/abs/2503.10488)  
**Code**: [GitHub](https://github.com/andrewbo29/co-speech-gestures-rolling-diffusion)  
**Area**: Human Understanding
**Keywords**: co-speech gestures, rolling diffusion, streaming generation, real-time, noise scheduling, motion synthesis

## TL;DR
This paper proposes a streaming co-speech gesture generation framework based on Rolling Diffusion, which converts arbitrary diffusion models into streaming gesture generators via a structured progressive noise schedule. It further introduces Rolling Diffusion Ladder Acceleration (RDLA) to achieve up to 4× speedup (200 FPS), comprehensively outperforming baselines on the ZEGGS and BEAT benchmarks.

## Background & Motivation
- Co-speech gestures are critical for virtual assistants, video conferencing, gaming, and embodied AI, where **real-time streaming generation** is a hard requirement for interactive scenarios.
- Core challenges faced by existing diffusion-based methods:
    - **Chunk-wise stitching**: Methods such as PersonaGestor and DiffSHEG segment long sequences into fixed-length clips for independent generation followed by concatenation, causing **visual discontinuities** and **post-processing latency**.
    - **Seed-frame conditioning**: DiffuseStyleGesture and Taming rely on preceding frames as conditions to improve continuity but incur **substantial computational overhead**.
    - **Outpainting strategy**: DiffSHEG employs incremental outpainting, which still requires additional post-processing steps.
- Rolling Diffusion Models (RDMs) are a promising alternative that converts diffusion models into autoregressive processes to improve temporal consistency; however, their autoregressive loops are **computationally expensive** and have not been successfully applied to real-time co-speech gesture generation.

## Core Problem
How to effectively adapt the Rolling Diffusion framework to co-speech gesture generation, enabling seamless streaming synthesis of arbitrary length, while substantially reducing inference latency to meet real-time requirements?

## Method

### Overall Architecture
A unified streaming framework based on Rolling Diffusion Models (RDMs), whose core mechanism imposes a **progressive noise schedule** along the temporal axis:

- A rolling window of size $N$ is maintained as $\mathbf{x}_j^{t_0} = \{x_{j+n}^{t_n}\}_{n=0}^{N-1}$.
- Noise levels of frames within the window increase linearly from front to back: $t_n = t_0 + n \cdot s$, where $s = T/N$.
- After every $s$ denoising steps, the first frame is fully denoised and output, the window shifts one frame to the right, and a new Gaussian noise frame is appended at the tail.
- This process generates **arbitrarily long** continuous gesture sequences without post-processing.

### Key Design 1: Model Adaptation Strategy
- **Generality**: The framework is **decoupled** from specific diffusion model architectures; only the time embedding injection needs to be modified.
- In the original model, all frames share a single time embedding; in this method, each frame within the window is assigned an **independent time embedding** reflecting its individual noise level.
- Conditional inputs remain unchanged: audio features $U = \{u_k\}$ are extracted by a pretrained WavLM, with optional style or speaker ID conditioning.
- The model input is the concatenation of context frames and the rolling window: $[\mathbf{x}_j^{cont}, \mathbf{x}_j^t]$.

### Key Design 2: Context Frame Regularization
- $n^{cont}$ previously generated frames are prepended to the rolling window as context.
- Key finding: applying **minimal noise** $t=1$ ($\sigma_1^2 = 0.00004$) to context frames as regularization significantly improves model robustness and generalization, preventing overfitting.

### Key Design 3: Rolling Diffusion Ladder Acceleration (RDLA)
Standard Rolling Diffusion fully denoises only one frame every $s$ steps, creating a **sequential bottleneck**. RDLA enables simultaneous multi-frame denoising via a ladder-shaped noise schedule:

- The original linear noise schedule is transformed into a **ladder-shaped** schedule with step size $l$:
  $$t_i^l = t_0^l + (k+1) \cdot l - 1, \quad kl \le i < (k+1)l - 1$$
- Frames within the same ladder step share the same noise level and can be **jointly denoised** across $l$ frames.
- $l=1$ degenerates to standard Rolling Diffusion; $l=2$ yields 2× speedup; $l=4$ yields 4× speedup.
- The noise level of the last ladder step is guaranteed to equal $T$, preserving the zero-SNR starting point.

### Loss & Training
1. **Standard training**: Window start positions $j$ and initial noise levels $t_0$ are sampled uniformly; uniform weights $a(t_n)=1$ are used (rather than SNR weighting) for simplicity and stability.
2. **Rolling-phase-only training**: The initial descent phase is omitted from training to simplify the procedure.
3. **Progressive RDLA fine-tuning**: The ladder step size is gradually increased as $l=2, 4, \ldots$, with each stage initialized from the previous stage's weights.
4. **Inertial Loss**:
   $$\mathcal{L}_{RDLA} = \sum_{n} \|x_{j+n}^0 - \hat{x}_{j+n}\|^2 - 2\lambda \sum_{n} \langle x_{j+n}^0 - \hat{x}_{j+n}, x_{j+n+1}^0 - \hat{x}_{j+n+1} \rangle$$
   The second term penalizes abrupt changes between denoising results of adjacent frames, suppressing motion jitter.
5. **On-the-Fly Smoothing (OFS)**: At inference time, cosine similarity thresholds are used to decide whether to apply mean smoothing to frames at boundaries between adjacent denoised blocks.

## Key Experimental Results

### Quantitative Results on ZEGGS

| Method | Div_g ↑ | Div_k ↑ | FD_g ↓ | FD_k ↓ |
|--------|---------|---------|--------|--------|
| GT | 272.34 | 213.97 | – | – |
| DSG orig. | 239.37 | 161.07 | 6393.99 | 14.24 |
| **DSG roll. (Ours)** | **251.35** | **175.12** | **3831.35** | **8.08** |
| DSG RDLA 2× | 222.25 | 173.76 | 5772.40 | 13.65 |
| Taming orig. | 154.70 | 80.70 | 10784.86 | 418.85 |
| **Taming roll. (Ours)** | **190.09** | **124.42** | **9064.00** | **353.62** |
| PersGestor orig. | 230.11 | 165.17 | 4060.36 | 11.12 |
| **PersGestor roll. (Ours)** | **242.14** | **189.31** | **3936.75** | **9.14** |

### Inference Speed Comparison

| Method | Ladder step $l$ | Denoising steps | FPS | Latency (s) |
|--------|----------------|-----------------|-----|-------------|
| DSG orig. (baseline) | – | 1000 | 10 | 8.0 |
| DSG roll. (Ours) | 1 | 1000 | 10 | 0.06 |
| DSG roll. (Ours) | 1 | 100 | 70 | 0.006 |
| DSG RDLA (Ours) | 2 | 100 | 120 | 0.003 |
| **DSG RDLA (Ours)** | **4** | **100** | **200** | **0.002** |

- Latency is reduced from the baseline's 8 seconds to **0.002 seconds**, achieving truly real-time generation.
- User study: 48.4% of participants preferred the rolling version vs. 36.3% for the original DSG (Wilcoxon test $p<0.05$).
- RDLA $l=2$ vs. rolling: 48.2% vs. 45.7%, indicating negligible quality cost for the speedup.

### Key Findings on BEAT
- RDLA $l=2$ achieves FD_g of **17309.63** on BEAT (vs. 21441.91 for rolling) and FD_k of **56.24** (vs. 69.23 for rolling), where acceleration actually **improves** quality.
- Reason: BEAT gestures are smoother, and the smoothing effect introduced by the ladder schedule is beneficial for this dataset.

## Highlights & Insights
- **Plug-and-play**: The framework is decoupled from specific diffusion architectures and has been successfully applied to four distinct baselines — DSG, Taming, PersonaGestor, and DiffSHEG — yielding improvements in all cases.
- **First successful application of Rolling Diffusion to a practical task**, demonstrating its utility for streaming generation scenarios.
- **RDLA ladder acceleration** is a novel contribution: by transforming a linear noise schedule into a ladder shape, it enables joint multi-frame denoising and is orthogonal to methods such as DDIM.
- The **minimal-noise regularization on context frames** is a simple yet effective trick worth generalizing to other sequential diffusion tasks.
- The 200 FPS inference speed greatly exceeds real-time requirements (30 FPS), leaving ample headroom for downstream applications.

## Limitations & Future Work
- RDLA $l=4$ shows notable quality degradation on the expressive ZEGGS dataset (FD_g increases from 3831 to 16791); the speed–quality trade-off is dataset-dependent.
- Validation is limited to skeletal motion data in BVH format; extension to 3D mesh or video-driven gesture generation has not been explored.
- Evaluation metrics (FD, Div) are based on distributional matching and may not fully capture semantic alignment quality.
- The user study is limited in scale (22 evaluators) and conducted against only a single baseline (DSG).
- No comparison with non-diffusion streaming methods (e.g., GAN-based or flow-based approaches) is provided.

## Related Work & Insights

| Dimension | Ours (Rolling Diffusion) | Chunk-stitching | Outpainting (DiffSHEG) | Seed-frame conditioning |
|-----------|--------------------------|-----------------|------------------------|------------------------|
| Streaming generation | ✓ Native support | ✗ Requires post-processing | Partial support | ✗ Non-streaming |
| Temporal continuity | Guaranteed by progressive noise | Stitching artifacts | Incremental extension | Seed-frame constraint |
| Real-time speed | 200 FPS (RDLA 4×) | Limited by post-processing | Limited by outpainting | ~10 FPS |
| Architectural generality | Plug-and-play | Model-specific | Model-specific | Model-specific |
| Sequence length | Arbitrary | Fixed window | Incremental | Fixed window |

## Rating
- Novelty: ⭐⭐⭐⭐ — First application of Rolling Diffusion to co-speech gesture generation; the RDLA ladder acceleration scheme is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets × four baselines cross-validation + user study + ablation study.
- Writing Quality: ⭐⭐⭐⭐ — Framework description is clear, mathematical derivations are complete, and figures are intuitive.
- Value: ⭐⭐⭐⭐ — Provides a general-purpose streaming gesture generation solution with outstanding 200 FPS real-time performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](../../ICCV2025/human_understanding/gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](../../ICCV2025/human_understanding/semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[CVPR 2026\] LASER: Layer-wise Scale Alignment for Training-Free Streaming 4D Reconstruction](../../CVPR2026/human_understanding/laser_layer-wise_scale_alignment_for_training-free_streaming_4d_reconstruction.md)
- [\[CVPR 2026\] Talking Together: Synthesizing Co-Located 3D Conversations from Audio](../../CVPR2026/human_understanding/talking_together_synthesizing_co-located_3d_conversations_from_audio.md)

</div>

<!-- RELATED:END -->
