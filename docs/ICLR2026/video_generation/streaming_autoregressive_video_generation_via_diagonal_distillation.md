---
title: >-
  [Paper Note] Streaming Autoregressive Video Generation via Diagonal Distillation
description: >-
  [ICLR 2026][Video Generation][autoregressive generation] This paper proposes Diagonal Distillation (DiagDistill), which achieves 277.3× acceleration and 31 FPS real-time streaming autoregressive video generation via a di…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "autoregressive generation"
  - "distillation"
  - "streaming generation"
  - "real-time video"
date: 2026-05-08
content_hash: 278c633c5cfaf054
---

# Streaming Autoregressive Video Generation via Diagonal Distillation

**Conference**: ICLR 2026
**arXiv**: [2603.09488](https://arxiv.org/abs/2603.09488)  
**Code**: [Project Page](https://SphereLab.ai/diagdistill)  
**Area**: Video Generation
**Keywords**: video generation, autoregressive generation, distillation, streaming generation, real-time video

## TL;DR

This paper proposes Diagonal Distillation (DiagDistill), which achieves 277.3× acceleration and 31 FPS real-time streaming autoregressive video generation via a diagonal denoising strategy (more steps for early chunks, fewer for later chunks) and a flow distribution matching loss.

## Background & Motivation

1. **Background**: Diffusion models have achieved remarkable progress in video generation quality, but global bidirectional attention requires generating the entire video at once, making it unsuitable for streaming or real-time scenarios. Autoregressive models are naturally suited for streaming generation but require multi-step denoising to maintain quality.

2. **Limitations of Prior Work**: Existing video distillation methods (e.g., CausVid, Self-Forcing) are largely adapted from image distillation techniques and overlook the special nature of the temporal dimension. Reducing denoising steps leads to degraded motion coherence, error accumulation over long sequences, and oversaturation.

3. **Key Challenge**: In autoregressive video generation, predicting the next chunk implicitly entails predicting the next noise level. This introduces exposure bias—training conditions on clean frames while inference conditions on generated frames—causing quality to progressively degrade over time. Furthermore, if early chunks have already established structural priors, later chunks should require fewer denoising steps, yet existing methods do not exploit this property.

4. **Goal**: Substantially reduce latency in streaming video generation while preserving video quality.

5. **Key Insight**: Exploit the temporal structure of autoregressive generation—structural priors established by early chunks can be "relayed" to subsequent chunks—motivating a non-uniform denoising step allocation strategy of "more steps early, fewer steps late."

6. **Core Idea**: By employing a diagonal denoising trajectory (more steps for early chunks, gradually decreasing to 2 steps for later chunks) together with a flow distribution matching loss, the method jointly optimizes across the temporal and denoising-step dimensions to achieve an optimal quality–efficiency trade-off.

## Method

### Overall Architecture

DiagDistill is built upon Wan2.1-T2V-1.3B and incorporated into the DMD (Distribution Matching Distillation) framework. Three core innovations are introduced: (1) a diagonal denoising strategy—the first three chunks use 5/4/3 steps, respectively, and all subsequent chunks use a fixed 2 steps; (2) a Diagonal Forcing training mechanism—noisy frames rather than clean frames are used as KV cache conditioning; and (3) a flow distribution matching loss—explicitly aligning the motion distributions of the teacher and student during distillation to prevent motion degradation.

### Key Designs

**1. Diagonal Denoising**

- **Function**: Adaptively allocates denoising steps according to temporal position, balancing quality and efficiency.
- **Mechanism**: The first three chunks are generated with distilled models using 5/4/3 steps, respectively, and all chunks from the fourth onward use a fixed 2-step denoising schedule. Later chunks can inherit rich appearance information from the thoroughly processed early chunks. Key insight: the structural priors established by early chunks allow later chunks to generate clear frames with fewer denoising steps.
- **Design Motivation**: Uniform step allocation is suboptimal—early chunks require high quality to establish the visual foundation, while later chunks can benefit from the priors already in place.

**2. Diagonal Forcing**

- **Function**: Mitigates error accumulation and oversaturation in long sequences.
- **Mechanism**: The clean output of the previous chunk $\mathbf{X}_{k-1}$ is injected with controlled noise as $\tilde{\mathbf{X}}_{k-1} = \sqrt{\alpha_{k-1}}\cdot\mathbf{X}_{k-1} + \sqrt{1-\alpha_{k-1}}\cdot\bm{\epsilon}$, which is then used as the KV cache conditioning for the current chunk. The optimal noise timestep is 100 (where 1000 is fully noisy and 0 is a clean frame).
- **Design Motivation**: Next-chunk prediction in autoregressive generation implicitly involves predicting the next noise level. Conditioning on clean frames (timestep 0) causes the model to over-denoise subsequent chunks (oversaturation); conditioning on moderately noisy frames aligns with the actual conditions encountered during inference and slows error propagation.

**3. Flow Distribution Matching**

- **Function**: Preserves motion magnitude and temporal consistency after step compression.
- **Mechanism**: A flow distribution matching loss $\nabla_\phi\mathcal{L}_{\text{DMD}}^{\text{flow}}$ is defined to align the distributions of the teacher and student over the motion flow field $\mathcal{F}(\mathbf{x})$. A lightweight learnable motion feature extraction module (convolution + MLP applied to latent differences) is used, avoiding reliance on an external optical flow estimator.
- **Design Motivation**: Few-step denoising tends to attenuate motion magnitude—the regression loss in standard DMD ensures per-frame quality but neglects temporal dynamics.

### Loss & Training

Total loss: $\mathcal{L}_{\text{Total}} = \lambda_{\text{spatial}}\mathcal{L}_{\text{DMD}} + \mathcal{L}_{\text{reg}} + \gamma(\lambda_{\text{flow}}\mathcal{L}_{\text{DMD}}^{\text{flow}} + \mathcal{L}_{\text{reg}}^{\text{flow}})$

where $\lambda_{\text{spatial}}=4$ and $\lambda_{\text{flow}}=4$. Inference employs a rolling KV cache (chunk size: 3 frames) with a fixed memory footprint of 17.5 GB.

## Key Experimental Results

### Main Results

VBench evaluation (5-second video generation, single H100 GPU):

| Method | Throughput (FPS)↑ | First-Frame Latency↓ | Speedup | Total↑ | Quality↑ | Semantics↑ |
|---|---|---|---|---|---|---|
| Wan2.1 | 0.78 | 103s | 1× | 84.26 | 85.30 | 80.09 |
| CausVid | 17.0 | 0.69s | 149.3× | 81.20 | 84.05 | 69.80 |
| Self-Forcing | 17.0 | 0.69s | 149.3× | 84.31 | 85.07 | 81.28 |
| **DiagDistill** | **31.0** | **0.37s** | **277.3×** | **84.48** | **85.26** | **81.73** |

### Ablation Study

| Configuration | Temporal Quality↑ | Frame Quality↑ | Text Alignment↑ | Total↑ |
|---|---|---|---|---|
| w/o Diagonal Forcing | 92.1 | 60.1 | 26.9 | 83.58 |
| w/o Flow Loss | 92.5 | 60.8 | 27.8 | 84.18 |
| w/o Diagonal Denoising | 95.1 | 63.2 | 28.6 | 84.46 |
| **Full Method** | 94.9 | 63.4 | 28.9 | **84.48** |

### Key Findings

- DiagDistill achieves a further **1.88× speedup** over Self-Forcing (277.3× vs. 149.3×) with no degradation in quality.
- The optimal noise timestep for Diagonal Forcing is 100—excessive noise blurs structural priors, while insufficient noise leads to oversaturation.
- The Flow Loss contributes primarily in the few-step denoising regime, with limited gains in multi-step settings.
- In 45-second long video generation, DiagDistill clearly outperforms CausVid and Self-Forcing, both of which exhibit saturation artifacts.

## Highlights & Insights

- **"More steps early, fewer steps late" is an intuitive yet effective principle**: It exploits the temporal structure of autoregressive generation, establishing the visual foundation early and saving computation later.
- **A novel solution to exposure bias**: Conditioning on moderately noisy frames bridges the gap between training and inference conditions.
- **Flow distribution matching**: For the first time, motion distribution alignment is explicitly addressed in video distillation.
- **High practical value**: At 31 FPS, the method exceeds the 16 FPS playback threshold, enabling genuinely real-time generation.

## Limitations & Future Work

- The method is built on Wan2.1-1.3B; its effectiveness on larger models remains to be verified.
- The fixed step-reduction schedule (5/4/3/2/2/…) may not be optimal across all scenarios.
- The learnable motion feature extraction module may be less precise than dedicated optical flow models.
- Adaptive step allocation—dynamically determining the number of steps per chunk based on scene complexity—is a promising direction for future exploration.

## Related Work & Insights

- CausVid and Self-Forcing lay the groundwork for streaming video generation; DiagDistill further accelerates upon them.
- The DMD framework provides the theoretical foundation for distillation; flow distribution matching is a natural extension to its temporal dimension.
- Insight: Distillation for video generation must account for temporal structure explicitly and cannot simply transplant image distillation techniques.

## Rating

- Novelty: ⭐⭐⭐⭐ The diagonal denoising strategy is novel, and flow distribution matching represents a first-of-its-kind contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive VBench evaluation, detailed ablations, and convincing long-video comparisons.
- Writing Quality: ⭐⭐⭐⭐ Figures are clear and intuitive explanations are well articulated.
- Value: ⭐⭐⭐⭐⭐ Exceptionally high practical value; 31 FPS real-time generation marks a milestone.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation](../../ICML2026/video_generation/aad-1_asymmetric_adversarial_distillation_for_one-step_autoregressive_video_gene.md)
- [\[ICLR 2026\] Lumos-1: On Autoregressive Video Generation with Discrete Diffusion from a Unified Model Perspective](lumos-1_on_autoregressive_video_generation_with_discrete_diffusion_from_a_unifie.md)
- [\[CVPR 2026\] StreamDiT: Real-Time Streaming Text-to-Video Generation](../../CVPR2026/video_generation/streamdit_real-time_streaming_text-to-video_generation.md)
- [\[CVPR 2026\] LottieGPT: Tokenizing Vector Animation for Autoregressive Generation](../../CVPR2026/video_generation/lottiegpt_vector_animation_generation.md)
- [\[ICCV 2025\] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](../../ICCV2025/video_generation/dollar_fewstep_video_generation_via_distillation_and_latent.md)

</div>

<!-- RELATED:END -->
