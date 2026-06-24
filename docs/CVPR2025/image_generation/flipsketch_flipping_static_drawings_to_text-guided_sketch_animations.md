---
title: >-
  [Paper Note] FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations
description: >-
  [Image Generation] FlipSketch is the first to achieve unconstrained raster sketch animation generation from a single static sketch and text prompt. Through three key innovations—LoRA fine-tuning on a T2V diffusion model, a DDIM-inversion-based reference frame mechanism, and a dual-attention composition—it generates smooth, dynamic animation sequences while maintaining sketch identity.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: b50c58649f9d3aa7
---

# FlipSketch: Flipping Static Drawings to Text-Guided Sketch Animations

## TL;DR

FlipSketch is the first to achieve unconstrained raster sketch animation generation from a single static sketch and text prompt. Through three key innovations—LoRA fine-tuning on a T2V diffusion model, a DDIM-inversion-based reference frame mechanism, and a dual-attention composition—it generates smooth, dynamic animation sequences while maintaining sketch identity.

## Background & Motivation

- **Appeal and pain points of sketch animation**: Flip-book animation is the most classic form of animation, but traditional animation requires a large number of professional artists to draw keyframes and in-betweens.
- **Limitations of prior work**:
    - **Vector-based animation methods** (Live-Sketch): Achieve animation via control point coordinate transformations, but are limited by: (1) only being able to displace/scale existing strokes without adding/deleting them, (2) 2D sketches representing only local perspectives of 3D objects and failing to express perspective transformations, and (3) extremely time- and computation-consuming SDS optimization.
    - **I2V methods** (SVD, DynamiCrafter): Suffer from a sketch-to-photo domain gap, making it difficult to preserve sketch identity in the generated results.
    - **Skeleton-based methods**: Require human-like inputs and are not applicable to general objects.
- **Key Challenge**:
  1. How to make video generation models generate sketch-style frames.
  2. How to maintain the visual integrity (identity consistency) of the input sketch.
  3. How to support unconstrained motion (beyond stroke displacement).

## Method

### Overall Architecture

Based on the ModelScope T2V diffusion model, the pipeline is divided into three parts:
1. **LoRA Fine-tuning**: Adapts the T2V model to the sketch style using synthetic sketch animations.
2. **Reference Frame Mechanism**: Constructs the reference noise via DDIM inversion followed by iterative frame alignment.
3. **Dual Attention Composition**: Injects reference frame information during spatial and temporal attention to guide denoising.

### Key Designs

#### 1. LoRA Fine-Tuning for Sketch-Style Adaptation

- Employs synthetic vector animations from Live-Sketch as training data.
- Trains LoRA (rank=4) on the 3D U-Net of ModelScope T2V with only 2500 iteration steps.
- The fine-tuned model can generate sketch-style frame sequences from text prompts.
- Extremely small parameter count ($< 0.01\%$), preserving the strong motion priors of the T2V model.

#### 2. Reference Frame Mechanism (Reference Frame via DDIM Inversion)

- **Setup**: Encodes the input sketch $I_s$ and performs DDIM inversion (null-text inversion) to obtain the reference noise $x_T^r$.
- The first frame uses reference noise $x_T^r$, while the remaining $M-1$ frames are sampled from a standard normal distribution $\{f_T^i\}_{i=2}^M \sim \mathcal{N}(0, \mathbf{I})$.
- **Iterative Frame Alignment**:
    - For each timestep $t \in [T, \tau_1]$:
    - Denoise the reference frame independently: $\eta_1 = \epsilon_\theta(x_t^r, t, \mathcal{P}_{null})$ as the GT feature.
    - Jointly denoise all frames: $[\eta'_i] = \epsilon_\theta([x_t^r, f_t^{train}], t, \mathcal{P}_{input})$.
    - Calculate alignment loss: $\mathcal{L}_{align} = \|\eta'_1 - \eta_1\|_2^2$.
    - Backpropagate to optimize $f_t^{train}$, aligning the first frame in joint denoising with the independently denoised reference.
    - Performed only at early timesteps ($\tau_1 = 2T/5$), as coarse structures are determined in the early stages of diffusion.

#### 3. Dual Attention Composition

At timestep $t \in [T, \tau_2]$ ($\tau_2 = 3T/5$), two-stream denoising is performed simultaneously:
- (i) Jointly denoise all frames: $\epsilon_\theta([x_t^r, f_t^i], t, \mathcal{P}_{input})$
- (ii) Denoise the reference frame only: $\epsilon_\theta([x_t^r], t, \mathcal{P}_{null})$

**Spatial Attention Composition $\mathcal{C}^S$**:
- Performs cross-attention between reference frame query $q_t^r$ and joint frame key $k_t^g$, replacing part of the self-attention.
- Repeats the reference frame $N$ times ($N$ linearly decays from $M$ to 1) to prevent the generated frames from degenerating into static frames.
- Effect: Injects spatial features (stroke positions, structures) of the reference frame into the generated frames.

**Temporal Attention Composition $\mathcal{C}^T$**:
- Directly replaces the first frame's key in temporal self-attention with the reference frame key $k_t^r$.
- Controls the influence weight of the first frame on other frames.
- Supports a motion-fidelity trade-off parameter $\lambda$: $k_t^r = k_t^r \cdot (1 + \lambda \cdot 2e^{-2})$, where high $\lambda$ enhances stability and low $\lambda$ increases motion magnitude.

### Loss & Training

- LoRA training: Standard diffusion denoising loss.
- Inference-time frame alignment: $\mathcal{L}_{align} = \|\eta'_1 - \eta_1\|_2^2$ (optimizes only the sampling noise without updating model parameters).

## Key Experimental Results

### Quantitative Comparison (Tab. 1 — CLIP Metrics)

| Method | S2V Consistency↑ | T2V Alignment↑ |
|------|-----------------|---------------|
| SVD | 0.917 | - |
| DynamiCrafter | 0.780 | 0.127 |
| Live-Sketch | **0.965** | 0.142 |
| **FlipSketch** | 0.956 | **0.172** |
| FlipSketch (λ=1) | 0.968 | 0.170 |

### Ablation Study

| Configuration | S2V Consistency↑ | T2V Alignment↑ |
|------|-----------------|---------------|
| FlipSketch (Full) | 0.956 | 0.172 |
| w/o frame alignment | 0.952 | 0.171 |
| w/o $\mathcal{C}^T$ & $\mathcal{C}^S$ | 0.876 | 0.168 |
| λ=0 (max motion) | 0.949 | 0.174 |
| λ=1 (max fidelity) | 0.968 | 0.170 |

### User Study (Tab. 2)

Users rated FlipSketch higher than Live-Sketch and the ablated versions in both text fidelity and sketch consistency.

### Key Findings

1. Removing the dual attention composition ($\mathcal{C}^T$ & $\mathcal{C}^S$) causes S2V consistency to plunge from 0.956 to 0.876, proving its critical role in identity preservation.
2. FlipSketch significantly outperforms Live-Sketch in text-video alignment (0.172 vs 0.142), showcasing richer motion.
3. Live-Sketch slightly outperforms in S2V consistency (0.965 vs 0.956) because vector-based methods naturally constrain strokes.
4. Computational efficiency: FlipSketch generates a 10-frame animation in approximately a few seconds, whereas Live-Sketch requires hours of SDS optimization.
5. Frame extrapolation can smoothly concatenate animations by using the last frame as the input sketch for the next segment.

## Highlights & Insights

1. **Raster vs. Vector Paradigm Shift**: Abandoning vector-level constraints in favor of raster-level degrees of freedom allows the animation to depict stroke additions/deletions and perspective transformations that vector-based methods cannot achieve.
2. **Ingenious Utilization of DDIM Inversion**: Using the inversion noise of the input sketch as the reference frame naturally guarantees accurate reconstruction after denoising, elegantly solving the identity preservation problem.
3. **Inference-Time vs. Training-Time Optimization**: Frame alignment is achieved in inference time by optimizing noise (rather than model parameters), ensuring manageable overhead.
4. **Explicit Motion-Fidelity Control**: The $\lambda$ parameter behaves as a user-adjustable knob to meet different creative demands.
5. **Minimalist LoRA Adaptation**: Standard training with only rank=4 and 2500 steps successfully adapts the T2V model to the sketch domain.

## Limitations & Future Work

1. **10-Frame Limitation**: Generates roughly 10 frames per run, needing frame extrapolation to splice longer animations, which might accumulate drift.
2. **Sketch-Motion Consistency**: For complex 3D motion (e.g., rotation), raster frames may exhibit unnatural distortions.
3. **Depth of Text Understanding**: Performance is heavily dependent on the T2V model's text comprehension, yielding limited fidelity to precise motion descriptions.
4. **Post-Processing Constraints**: Output frames are forced into black strokes on a white background via post-processing, potentially discarding grayscale details.

## Related Work & Insights

- **Live-Sketch (NeurIPS'24)**: Vector-based animation optimized via SDS $\rightarrow$ This work demonstrates that direct raster generation is faster and more flexible.
- **ModelScope T2V**: Open-source T2V model $\rightarrow$ Provides foundational motion priors.
- **TF-ICON**: Attention composition in diffusion models $\rightarrow$ Inspired the design of the dual attention composition.
- **Insights**: Performing lightweight adaptation (LoRA) on diffusion models combined with delicate inference-time control (attention manipulation and noise optimization) can unlock novel generation capabilities at a very low cost.

## Rating

⭐⭐⭐⭐ — Highly creative work that elegantly combines the simple experience of flip-book animations with modern T2V techniques. The three core innovations (LoRA, reference frame, and dual attention) fulfill their respective roles and coordinate well to form a practical and entertaining system.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)
- [\[CVPR 2025\] FineLIP: Extending CLIP's Reach via Fine-Grained Alignment with Longer Text Inputs](finelip_extending_clips_reach_via_fine-grained_alignment_with_longer_text_inputs.md)
- [\[CVPR 2025\] Dynamic Motion Blending for Versatile Motion Editing (MotionReFit)](dynamic_motion_blending_for_versatile_motion_editing.md)
- [\[CVPR 2025\] Dual Prompting Image Restoration with Diffusion Transformers (DPIR)](dual_prompting_image_restoration_with_diffusion_transformers.md)
- [\[CVPR 2025\] FilmComposer: LLM-Driven Music Production for Silent Film Clips](filmcomposer_llm-driven_music_production_for_silent_film_clips.md)

</div>

<!-- RELATED:END -->
