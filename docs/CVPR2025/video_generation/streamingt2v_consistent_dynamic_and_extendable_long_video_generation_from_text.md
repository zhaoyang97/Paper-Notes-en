---
title: >-
  [Paper Note] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text
description: >-
  [CVPR 2025][Video Generation][long video generation] Proposes StreamingT2V, an autoregressive text-to-long video generation method, which achieves seamless, highly dynamic video generation of over 2 minutes ($1200+$ frames) through a Conditional Attention Module (CAM) for short-term memory and an Appearance Preservation Module (APM) for long-term memory.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "long video generation"
  - "autoregressive video synthesis"
  - "temporal consistency"
  - "diffusion models"
  - "text-to-video"
date: 2026-05-08
content_hash: 4cc8007a8d6342ab
---

# StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text

**Conference**: CVPR 2025  
**arXiv**: [2403.14773](https://arxiv.org/abs/2403.14773)  
**Code**: [GitHub](https://github.com/Picsart-AI-Research/StreamingT2V)  
**Area**: Video Generation  
**Keywords**: long video generation, autoregressive video synthesis, temporal consistency, diffusion models, text-to-video

## TL;DR

Proposes StreamingT2V, an autoregressive text-to-long video generation method, which achieves seamless, highly dynamic video generation of over 2 minutes ($1200+$ frames) through a Conditional Attention Module (CAM) for short-term memory and an Appearance Preservation Module (APM) for long-term memory.

## Background & Motivation

Text-to-video diffusion models have made remarkable progress in short video generation ($\le 16$ frames), but they face severe challenges when generating long videos. Directly training models on long videos is computationally infeasible (requiring 260K training steps and a 4.5K batch size for just 16 frames). Existing autoregressive extension schemes primarily suffer from the following issues:

1. **Frame concatenation schemes**: Concatenating the noisy latents of the last few frames of the previous segment with the current segment provides a conditioning signal that is too weak, leading to inconsistency between segments.
2. **CLIP image embedding schemes**: Models like SVD and I2VGen-XL extract features from the previous frame using a CLIP image encoder, but CLIP discards crucial fine-grained details required for reconstruction.
3. **Sparse encoder schemes**: Methods like SparseCtrl require padding zero frames after conditional frames, and the resulting input inconsistency causes output inconsistency.

More critically, all existing methods when applied autoregressively suffer from **video stagnation**—the background freezes, motion disappears, and the video degrades into almost static images. Furthermore, as the autoregressive process progresses, object appearance drift and video quality degradation also present severe issues.

The Core Idea of StreamingT2V is to introduce **short-term memory** (conditioning the current generation on multi-frame features of the previous segment via an attention mechanism) and **long-term memory** (extracting-high level features from the first frame to prevent forgetting the initial scene), while maintaining rich motion and consistency across segments.

## Method

### Overall Architecture

The pipeline of StreamingT2V consists of three stages:
1. **Initialization Stage**: Generating the initial 16-frame chunk using an off-the-shelf text-to-video model (such as ModelScope).
2. **Streaming T2V Stage**: Autoregressively generating subsequent frames through CAM and APM, generating 16 frames at each step with an 8-frame overlap with the previous chunk.
3. **Streaming Refinement Stage**: Utilizing a high-resolution video enhancement model (e.g., MS-Vid2Vid-XL) to autoregressively enhance the long video to $720 \times 720$ resolution using a randomized blending method.

### Key Designs

1. **Conditional Attention Module (CAM) — Short-term Memory**:
    - **Function**: Conditioning the generation of the current chunk using features from the last $F_{\text{cond}}=8$ frames of the previous chunk.
    - **Mechanism**: Composed of a feature extractor (weights cloned from the frame encoder and UNet encoder layers) and a feature injector. During injection, at each skip connection of the UNet, the features of the current frames attend to the features extracted by CAM via Temporal Multi-Head Cross-Attention (T-MHA). The query $Q$ comes from the UNet skip connection features, and key/value $K/V$ come from the CAM features.
    - **Design Motivation**: Compared to concatenation, the attention mechanism allows more effective borrowing of content information from the previous chunk without restricting the motion of the current frames. Zero-initialization of the output projection layer ensures that it does not affect the base model during the early stages of training.

2. **Appearance Preservation Module (APM) — Long-term Memory**:
    - **Function**: Extracting high-level scene and object features from the anchor frame of the first chunk to maintain appearance consistency throughout the entire autoregressive process.
    - **Mechanism**: (i) The CLIP image tokens of the anchor frame are expanded into $k=16$ tokens via an MLP, which are then concatenated with text CLIP tokens and passed through a projection block to obtain the mixed encoding $x_{\text{mixed}}$; (ii) Learnable weights $\alpha_l$ (initialized to 0) are introduced in each cross-attention layer to weightedly combine image and text information via $\text{SiLU}(\alpha_l) \cdot x_{\text{mixed}} + x_{\text{text}}$.
    - **Design Motivation**: Methods relying solely on the previous frame gradually forget the initial scene features, leading to appearance drift and quality degradation. Using a fixed anchor frame provides a global consistency prior and avoids error accumulation.

3. **Randomized Blending**:
    - **Function**: Ensuring smooth transitions between adjacent chunks when enhancing long videos using a high-resolution model.
    - **Mechanism**: The long video is divided into chunks of 24 frames (with an 8-frame overlap). During each denoising step, adjacent chunks share the noise in the overlapping region. Then, a cut-off frame index $f_{\text{thr}}$ is randomly sampled within the overlap region, and the $1:F-f_{\text{thr}}$ frames of the previous chunk are concatenated with the $f_{\text{thr}}+1:F$ frames of the current chunk. For each frame in the overlap region, the latent from the previous chunk is used with a probability of $1-f/(O+1)$.
    - **Design Motivation**: Naively enhancing each chunk independently leads to inconsistent transitions; simply sharing noise still exhibits alignment issues. Randomized blending effectively eliminates inconsistencies between chunks by probabilistically blending the latents in the overlap region.

### Loss & Training

- The CAM and APM are trained using the standard diffusion model denoising loss.
- The output projection layer $P_{\text{out}}$ of CAM is zero-initialized, and the weights $\alpha_l$ of APM are initialized to 0, ensuring no disruption to the base model at the start of training.
- The streaming refinement stage requires no additional training and is directly performed on the pre-trained high-resolution model using the SDEdit method.

## Key Experimental Results

### Main Results

Evaluation on 240-frame videos generated from a test set of 50 text prompts:

| Method | MAWE↓ | SCuts↓ | CLIP↑ |
|------|-------|--------|-------|
| SparseCtrl | 6069.7 | 5.48 | 29.32 |
| I2VGenXL | 2846.4 | 0.4 | 27.28 |
| DynamiCrafterXL | 176.7 | 1.3 | 27.79 |
| SEINE | 718.9 | 0.28 | 30.13 |
| SVD | 857.2 | 1.1 | 23.95 |
| FreeNoise | 1298.4 | 0 | 31.55 |
| OpenSora | 1165.7 | 0.16 | 31.54 |
| OpenSoraPlan | 72.9 | 0.24 | 29.34 |
| **StreamingT2V** | **52.3** | **0.04** | **31.73** |

### Ablation Study

| Configuration | Key Performance | Description |
|------|---------|------|
| CAM Only | Smooth chunk transitions | Attention mechanism outperforms the concatenation approach |
| CAM + APM | Maintains appearance consistency | Anchor frame prevents scene forgetting |
| Naive Concatenation Refinement | Obvious transition inconsistency | X-T slice visualization shows severe fractures |
| Shared Noise Refinement | Slight improvement but still has alignment issues | Insufficient to solve the issue on its own |
| Randomized Blending Refinement | Smooth and seamless transitions | Probabilistic blending effectively eliminates inconsistencies |

### Key Findings

1. The MAWE metric significantly outperforms all competing methods (approximately 30% lower than the runner-up, OpenSoraPlan).
2. Although FreeNoise achieves $\text{SCuts}=0$, it generates nearly static videos with almost zero motion.
3. Methods utilizing CLIP image encoders (SVD, DynamiCrafterXL, I2VGenXL) yield very low CLIP scores in the autoregressive setting, likely because the CLIP image encoder exhibits a domain shift on generated images.
4. The metrics of StreamingT2V remain stable over time (MAWE: 43-46, CLIP: 31.79-32.45 in the 120-220 frame range).
5. The proposed method can be generalized to DiT architectures (e.g., OpenSora).

## Highlights & Insights

1. **Complementary Design of Short-term and Long-term Memory**: CAM ensures smooth transitions between adjacent chunks, while APM prevents long-range appearance drift. Together, they address the core challenge of long video generation.
2. **Advantages of Attention vs. Concatenation**: The cross-attention design of CAM does not require padding condition frames with zeros to match the target frame length, avoiding the input inconsistency issues found in methods like SparseCtrl.
3. **Ingenuity of Randomized Blending**: It turns deterministic fusion into probabilistic blending, switching at different locations in each denoising step, making the transitions natural and robust.
4. **Unveiling the Video Stagnation Problem**: Systematically identifying and demonstrating that all existing autoregressive methods lead to video stagnation.

## Limitations & Future Work

1. The frame rate and resolution of the generated videos are still limited by the capabilities of the base model.
2. It relies heavily on the quality of the initial generation—if the first frame is of poor quality, APM will propagate undesirable features to all subsequent frames.
3. Autoregressive generation is relatively slow, as each chunk requires a full denoising process.
4. Since MAWE is a newly proposed metric, its comprehensiveness and correlation with human perception still require further validation.
5. Future work can explore deeper integration with DiT architectures and end-to-end training of the enhancement pipeline.

## Related Work & Insights

- **vs. FreeNoise**: FreeNoise achieves inter-frame consistency by reusing noise vectors, but at the cost of almost completely sacrificing motion; StreamingT2V maintains motion while preserving consistency through an attention mechanism.
- **vs. SparseCtrl**: SparseCtrl uses a sparse encoder similar to ControlNet, which requires zero-padding and leads to input inconsistency; CAM naturally handles varying frame counts via cross-attention.
- **vs. SVD/DynamiCrafter**: These methods use CLIP image embeddings for conditioning, but CLIP discards crucial reconstruction details; CAM performs attention interaction directly in the feature space, retaining richer information.
- **vs. OpenSora/OpenSoraPlan**: While Transformer-based methods can generate longer videos (384 frames), their motion is limited, and chunk transition issues still persist.

## Rating

- Novelty: ⭐⭐⭐⭐ The autoregressive framework with short-term + long-term memory is novel, and the randomized blending method is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated against 9 baseline methods, introduced new metrics (MAWE/SCuts), and included ablation and stability analyses.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with deep problem analysis and solid motivation.
- Value: ⭐⭐⭐⭐ Long video generation is a major challenge; this method is practical, open-sourced, and provides highly valuable references for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Enhancing Train-Free Infinite-Frame Generation for Consistent Long Videos](../../ICML2026/video_generation/enhancing_train-free_infinite-frame_generation_for_consistent_long_videos.md)
- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [\[CVPR 2025\] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation](moviebench_a_hierarchical_movie_level_dataset_for_long_video_generation.md)
- [\[CVPR 2025\] World-Consistent Video Diffusion with Explicit 3D Modeling](world-consistent_video_diffusion_with_explicit_3d_modeling.md)
- [\[CVPR 2025\] AnimateAnything: Consistent and Controllable Animation for Video Generation](animateanything_consistent_and_controllable_animation_for_video_generation.md)

</div>

<!-- RELATED:END -->
