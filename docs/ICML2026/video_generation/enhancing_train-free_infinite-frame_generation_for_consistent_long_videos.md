---
title: >-
  [Paper Note] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos
description: >-
  [ICML 2026][Video Generation][Long video generation] MIGA enables base video models to generate **infinite-length** and **highly temporally consistent** videos without training through two core mechanisms: **Two-stage Tr…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Long video generation"
  - "training-free extension"
  - "temporal consistency"
  - "autoregressive generation"
date: 2026-05-08
content_hash: bd00cb37a5bf7993
---

# Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos

**Conference**: ICML 2026  
**arXiv**: [2605.18233](https://arxiv.org/abs/2605.18233)  
**Code**: To be confirmed  
**Area**: Video Generation / Long Video  
**Keywords**: Long video generation, training-free extension, temporal consistency, autoregressive generation

## TL;DR
MIGA enables base video models to generate **infinite-length** and **highly temporally consistent** videos without training through two core mechanisms: **Two-stage Training-inference Alignment** (TTA) and **Dual Consistency Enhancement** (DCE: Self-Reflection + Long-range Frame Guidance). Its comprehensive VBench score improves by 2.8% compared to FIFO-Diffusion (97.82 vs 95.02).

## Background & Motivation

**Background**: Current video generation models perform excellently on short videos but are constrained by training length. To meet long video demands in film production/game development, researchers explore two directions: training specialized long video models (Self-Forcing), which requires massive computation; or training-free extension (FreeNoise, FreeLong, FreePCA, FIFO-Diffusion), which operates directly on pre-trained base models.

**Limitations of Prior Work**: (1) Fixed-length extension methods suffer from memory consumption that grows linearly with the number of generated frames, making minute-level videos difficult to achieve. (2) FIFO-Diffusion achieves fixed memory consumption and infinite frame generation via frame-level autoregression, but the model processes latent variables of a single noise level during training while handling multiple different noise levels during inference—this **training-inference mismatch** leads to content drift and visual artifacts, and the method lacks explicit modeling of long-term temporal consistency.

**Key Challenge**: How to retain the advantages of the autoregressive framework (fixed memory, infinite frames) while narrowing the noise space discrepancy between training and inference and explicitly enhancing temporal consistency throughout the generation process?

**Goal**: (1) Proactively reduce the noise span during inference to closer match training conditions in a lightweight manner. (2) Efficiently detect and correct consistency anomalies in early high-noise frames without external evaluators. (3) Enable interaction between distant frames to improve global temporal consistency.

**Key Insight**: (1) The noise queue maintained in autoregressive frameworks inevitably covers multiple noise levels, but the span can be narrowed by slowing the rate of noise change. (2) Similarity in VAE latent space directly reflects inter-frame differences without requiring external models. (3) The queue structure naturally distinguishes early and late frames, allowing for targeted application of different consistency enhancement strategies.

**Core Idea**: A two-stage design solves two major problems simultaneously: Stage 1 uses a zigzag queue structure to slow the noise change rate (reducing training-inference mismatch), and Stage 2 unifies noise levels for standard denoising. This is combined with Self-Reflection (detecting and correcting anomalies) and Long-range Frame Guidance (utilizing generated clean frames) to form a two-pronged consistency enhancement.

## Method

### Overall Architecture
The method is built upon frame-level autoregressive generation. The standard workflow maintains a latent variable queue of length $L$, containing frames corresponding to $T$ denoising timesteps. MIGA introduces two-stage alignment and dual consistency enhancement: (1) Two-stage Training-inference Alignment (TTA); (2) Self-Reflection mechanism; (3) Long-range Frame Guidance.

### Key Designs

1.  **Two-stage Training-inference Alignment (TTA)**:
    - **Function**: Gradually reduces the noise span of model inputs through staged processing, bringing inference conditions closer to the single-noise-level condition used in training.
    - **Mechanism**: Stage 1 adopts a zigzag queue structure where the noise level changes every $L_{\text{zig}}$ frames (instead of every frame). The noise span in the $f_0$ frames input to the model is reduced from the original $f_0$ levels to approximately $\lceil f_0 / L_{\text{zig}} \rceil$. After $n$ iterations, $n L_{\text{zig}}$ frames occupy the same noise level $\tau_{e-1}$. Stage 2 then performs standard denoising at this unified noise level, matching training conditions.
    - **Design Motivation**: Training-inference mismatch is the fundamental bottleneck of autoregressive frameworks; countering it directly is theoretically difficult, but can be mitigated by progressively "smoothing" input noise; the zigzag structure does not change the total denoising steps, only the noise diversity seen by the model, with minimal computational cost.

2.  **Self-Reflection**:
    - **Function**: Detects consistency anomalies in real-time at the end of the queue (early high-noise frames) and corrects them by generating multiple candidates via expanded search, without requiring external consistency evaluators.
    - **Mechanism**: Defines a consistency score $C_{\text{score}} = \text{mean}_1(\text{mean}_2(q'_{\text{eval}} (q'_{\text{ref}})^\top))$ by calculating the cosine similarity matrix between the frame to be evaluated and reference frames. It was discovered that consistency scores of high-noise frames correlate with those of final clean frames, allowing for early detection. When the score drops below a threshold $\delta_{\text{adju}}$, an expanded search is triggered: $n_{\text{samp}}$ Gaussian-initialized candidate frames are sampled and denoised using prefix guidance, and the candidate with the highest score replaces the original frame.
    - **Design Motivation**: The earlier an anomaly is found and corrected, the less impact it has on subsequent generation; unlike methods using external models like DINO, Self-Reflection works directly in the latent space to avoid repeated VAE decoding overhead.

3.  **Long-Range Frame Guidance**:
    - **Function**: Uniformly samples $m_{\text{guid}}$ frames from the head of the queue (already generated low-noise frames) during sliding window denoising and concatenates them with the current processing window to allow distant frames to interact.
    - **Mechanism**: The standard sliding window $q_{\text{input}} = [z^l, \ldots, z^{l + f_0 - 1}]$ only considers local neighbors. Long-range guidance expands this to $q_{\text{input}} = [z^1, \ldots, z^{m_{\text{guid}}}, z^l, \ldots, z^{l + f_0 - m_{\text{guid}} - 1}]$ (when $l > m_{\text{guid}}$), adding clean generated frame prefixes to every denoising step.
    - **Design Motivation**: Temporal consistency inherently requires global information; local windows are memory-efficient but prone to "quality drift"; including verified frames in each window maintains fixed memory while providing global perspective constraints.

## Key Experimental Results

### Main Results (VBench Benchmark)

| Method | Infinite Frames | Subject Cons. | Background Cons. | Motion Smoothness | Temp. Flickering | Overall Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| VideoCrafter2-FreePCA | ✗ | 93.57 | 95.24 | 93.73 | 91.27 | 93.45 |
| VideoCrafter2-FreeLong | ✗ | 95.72 | 96.42 | 98.38 | 97.28 | 96.95 |
| VideoCrafter2-FIFO-Diffusion | ✓ | 92.92 | 95.01 | 97.19 | 94.94 | 95.02 |
| VideoCrafter2-ScalingNoise | ✓ | 94.29 | 95.52 | 97.86 | 96.12 | 95.95 |
| **VideoCrafter2-MIGA** | ✓ | **97.66** | **96.99** | **98.60** | **98.03** | **97.82** |
| Wan2.1-FIFO-Diffusion | ✓ | 92.67 | 93.37 | 98.03 | 97.09 | 95.29 |
| **Wan2.1-MIGA** | ✓ | **96.46** | **95.50** | **98.85** | **98.14** | **97.24** |

Compared to FIFO-Diffusion, MIGA achieves a +4.74% boost in Subject Consistency on VideoCrafter2.

### Ablation Study

| TTA | DCE | S.C. | B.C. | M.S. | T.F. | O.S. |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ✗ | ✗ | 92.92 | 95.01 | 97.19 | 94.94 | 95.02 |
| ✓ | ✗ | 96.74 | 96.75 | 97.57 | 97.12 | 97.05 |
| ✗ | ✓ | 96.10 | 96.47 | 97.88 | 96.56 | 96.75 |
| ✓ | ✓ | 97.66 | 96.99 | 98.60 | 98.03 | 97.82 |

The two core mechanisms independently contribute approximately 2% to the overall score improvement (TTA +2.03%, DCE +1.73%).

### Key Findings
- TTA provides the largest individual benefit—the two-stage alignment is the most significant contribution, improving the baseline by 2% alone, indicating that training-inference mismatch is indeed the primary bottleneck of autoregressive frameworks.
- DCE is highly complementary—combining it with TTA produces a synergistic effect, with a total improvement exceeding the sum of the two (4.8% > 2.0% + 1.7%).
- Cross-model consistency—both base models (animation vs realistic) benefit from MIGA.
- Significant performance advantage on NarrLV—on complex narrative tasks (scene changes, object attribute transitions), MIGA’s advantage over FIFO-Diffusion is even more pronounced (+2.3-12.5%).

## Highlights & Insights
- **Ingenious Progressive Smoothing of Noise Space**: Mitigating training-inference mismatch by changing the input queue structure without altering computation graphs or requiring retraining is a "lightweight adaptation" approach that could inspire other autoregressive tasks (text, images).
- **Self-consistency Scoring Avoids Computational Traps**: Leveraging the correlation between high-noise and clean frames in latent space avoids frequent VAE decoding and external evaluator calls—a reusable engineering trick.
- **Complementary Design of Dual Mechanisms**: Self-reflection focuses on timely correction of early anomalies (local constraint), while long-range guidance ensures global temporal smoothness (global constraint). This local-global combination is transferable to other tasks requiring long-term correlation (multimodal generation, long-text translation).

## Limitations & Future Work
- Different hyperparameters were used for the two models; general hyperparameter recommendation rules remain to be explored.
- Self-Reflection detects consistency by comparing adjacent frames and may not be sensitive enough for rapid content changes (intense action or scene cuts).
- The choice of $m_{\text{guid}}$ in long-range guidance lacks a principled design and currently relies on empirical values.
- Future Work: Adaptive hyperparameter strategies; extending Self-Reflection to multi-scale anomaly detection; investigating mechanisms to dynamically determine $m_{\text{guid}}$.

## Related Work & Insights
- **vs FIFO-Diffusion**: Both use frame-level autoregression + fixed memory architecture, but MIGA improves on two key fronts: actively narrowing the noise span mismatch and introducing explicit temporal consistency modeling.
- **vs ScalingNoise**: Both attempt search optimization at inference, but ScalingNoise's per-step assessment is computationally expensive; MIGA’s Self-Reflection only triggers search upon anomaly detection and works entirely in latent space, whereas ScalingNoise relies on external DINO models.
- **vs FreeLong / FreePCA**: These are finite-frame extensions that cannot generate minute-level videos; MIGA's advantage as an infinite-frame method is constant memory and no upper limit on frame count.

## Rating
- Novelty: ⭐⭐⭐⭐ The two-stage alignment is intuitive, while the combination of latent space consistency scoring and long-range guidance is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers two mainstream base models, two authoritative benchmarks (VBench + NarrLV), and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed methodology, and intuitive visual comparisons.
- Value: ⭐⭐⭐⭐⭐ Solves a practical problem (training-free long video generation) with a lightweight method easy to integrate into different models. High practical value for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Explainable Forensics of Manipulated Segments in Untrimmed Long Videos](explainable_forensics_of_manipulated_segments_in_untrimmed_long_videos.md)
- [\[CVPR 2026\] NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos](../../CVPR2026/video_generation/neoverse_enhancing_4d_world_model_with_in-the-wild_monocular_videos.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](../../CVPR2026/video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)
- [\[AAAI 2026\] FilmWeaver: Weaving Consistent Multi-Shot Videos with Cache-Guided Autoregressive Diffusion](../../AAAI2026/video_generation/filmweaver_weaving_consistent_multi-shot_videos_with_cache-guided_autoregressive.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)

</div>

<!-- RELATED:END -->
