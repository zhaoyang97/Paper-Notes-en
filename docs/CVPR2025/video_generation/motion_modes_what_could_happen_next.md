---
title: >-
  [Paper Note] Motion Modes: What Could Happen Next?
description: >-
  [CVPR 2025][Video Generation][Motion Prediction] Motion Modes is proposed, a training-free method that explores the latent distribution of a pretrained image-to-video generator by designing four guidance energy functions, discovering multiple plausible and diverse motion modes of objects from a single image while decoupling object motion from camera motion.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Motion Prediction"
  - "Training-free Method"
  - "Diffusion Guidance"
  - "Object Motion Diversity"
  - "Image-to-Video"
date: 2026-05-08
content_hash: 113d53b7e0cff738
---

# Motion Modes: What Could Happen Next?

**Conference**: CVPR 2025  
**arXiv**: [2412.00148](https://arxiv.org/abs/2412.00148)  
**Code**: [https://motionmodes.github.io](https://motionmodes.github.io)  
**Area**: Video Generation  
**Keywords**: Motion Prediction, Training-free Method, Diffusion Guidance, Object Motion Diversity, Image-to-Video

## TL;DR

Motion Modes is proposed, a training-free method that explores the latent distribution of a pretrained image-to-video generator by designing four guidance energy functions, discovering multiple plausible and diverse motion modes of objects from a single image while decoupling object motion from camera motion.

## Background & Motivation

Predicting multiple possible motions of an object from a single static image remains an open challenge. Limitations of prior work: (1) video generation models typically couple object motion with camera motion and scene changes; (2) trajectory-based methods (such as Motion-I2V) can predict specific motions but rely heavily on synthetic training data and predefined motions, struggling with complex scenes (e.g., crashing waves); (3) more critically, these methods require users to **provide** motion instructions rather than **automatically discovering** multiple possible motions.

Key Insight: Pretrained image-to-video generators have already encoded rich motion distributions over massive amounts of data. The key question is whether this latent distribution can be explored to discover diverse motions of an object. Direct random sampling is inefficient and often introduces excessive camera motion. Motion Modes achieves efficient exploration of this distribution through carefully designed guidance energy functions.

## Method

### Overall Architecture

Given an input image $\mathbf{y}$ and an object mask $\mathbf{m}$, the objective is to discover a set of diverse possible object motions $\mathcal{X} = \{\mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \ldots\}$. Motion is represented as a time-dependent 2D vector field $\mathbf{x} \in \mathbb{R}^{F \times H \times W \times 2}$. Built upon Motion-I2V as the backbone model—which separates motion generation from appearance, naturally decoupling motion from other scene changes—the motion set is constructed via iterative sampling combined with a stopping criterion.

### Key Designs

1. **Four Guidance Energy Functions**: Four energy guidances are injected during the denoising process at inference time, without requiring any training or fine-tuning:
    - **Static Camera Guidance $E_c$**: Penalizes the average motion magnitude outside the object mask, promoting a static camera.
    - **Object Motion Guidance $E_o$**: Encourages a difference in motion magnitude inside versus outside the mask, ensuring the object undergoes action.
    - **Diversity Guidance $E_d$**: Inspired by particle guidance, this exerts a repulsive force against each already generated motion in the set $\mathcal{X}$, encouraging the new motion to differ from existing ones in both direction ($w_{\text{angle}}=0.75$) and magnitude ($w_{\text{mag}}=0.25$).
    - **Smoothness Guidance $E_s$**: Regularizes motion variations across consecutive frames to prevent jitter.

   The total energy is $E = \lambda_d E_d + \lambda_c E_c + \lambda_o E_o + \lambda_s E_s$. During denoising, the noise trajectory is modified via gradient descent: $\mathbf{x}_t' = \mathbf{x}_t - \nabla_{\mathbf{x}_t} E(x_\theta^0(\mathbf{x}_t; t, \mathbf{y}), \mathbf{m}, \mathcal{X})$.

2. **Iterative Sampling and Stopping Criterion**: Motions are sampled sequentially and added to the set $\mathcal{X}$, with a maximum budget of 6 motions. If the final denoised motion's guidance energy exceeds a threshold $\rho=5.0$, it is discarded and resampled. Sampling terminates if two consecutive motions are discarded. This automatically adapts to the number of permissible motions in different scenes (e.g., a drawer can only open/close, whereas a flag can wave in many ways).

3. **Distance Metric Design**: The distance between motion vectors jointly considers direction and magnitude: $d(\mathbf{a}, \mathbf{b}) = w_{\text{mag}}(|\|\mathbf{a}\| - \|\mathbf{b}\||) + w_{\text{angle}}(1 - \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\|\|\mathbf{b}\|})$. Diversity guidance focuses on directional differences, while smoothness guidance focuses on amplitude stability.

### Loss & Training

Motion Modes is a fully **training-free** approach—it does not modify any parameters of the pretrained model, only altering the denoising trajectory through energy guidance during inference.

Guidance energy weight settings: $\lambda_d=3.0$, $\lambda_c=0.2$, $\lambda_o=0.025$, $\lambda_s=0.1$. The activation function is defined as $\phi(a) = \text{softplus}((a+e)^{-1} - \tau)$ (a soft inverse function), where $\tau$ is set to 40 for object motion guidance and 1 for diversity guidance.

## Key Experimental Results

### Main Results

| Method | Diversity $\bar{E}_d$ ↓ | Focus $\bar{E}_f$ ↓ | Camera Stillness $\bar{E}_c$ ↓ | Object Motion $\bar{E}_o$ ↓ |
|------|---------------------|---------------------|---------------------|---------------------|
| Prompt Generation (GPT-4o) | 1.28 | 1.71 | 1.11 | 2.31 |
| ControlNet | 1.75 | 1.14 | 0.07 | 2.22 |
| Random Arrows | 1.77 | 1.17 | 0.07 | 2.27 |
| Random Noise | 1.27 | 2.20 | 1.36 | 3.05 |
| FPS Noise | 1.21 | 1.98 | 1.23 | 2.74|
| **Motion Modes (Ours)** | **1.04** | **0.07** | **0.09** | **0.05** |

### Ablation Study

| Configuration | $\bar{E}_d$ ↓ | $\bar{E}_f$ ↓ | $\bar{E}_c$ ↓ | $\bar{E}_o$ ↓ | $\bar{E}$ ↓ |
|------|--------------|--------------|--------------|--------------|-------------|
| Without $E_c$ | 1.02 | 0.64 | 1.29 | 0.00 | 0.83 |
| Without $E_o$ | 1.03 | 0.91 | 0.06 | 1.75 | 0.97 |
| Without $E_d$ | 1.36 | 0.08 | 0.13 | 0.04 | 0.72 |
| FPS replacing $E_d$ | 1.49 | 0.10 | 0.11 | 0.08 | 0.79 |
| ControlNet replacing $E_c$+$E_o$ | 0.96 | 0.80 | 0.15 | 1.45 | 0.88 |
| **Motion Modes (Full)** | **1.04** | **0.07** | **0.09** | **0.05** | **0.55** |

### Key Findings

1. **User Study I (32 participants, 320 comparative sets)**: Motion Modes comprehensively outperforms all baselines across three dimensions: plausibility, diversity, and alignment with expectations. The Prompt Generation baseline is the closest in diversity but exhibits poor focus.
2. **User Study II (12 participants)**: 96% of the generated motions are rated as plausible, 92% align with user expectations, and 19% offer highly novel yet plausible motions that exceed expectations—demonstrating that the method is both accurate and inspiring.
3. While ControlNet and Random Arrows achieve good focus metrics, this is actually because object motion is also suppressed (yielding static scenes).
4. FPS noise sampling outperforms random sampling but is significantly inferior to energy guidance, proving that simple noise-space strategies are insufficient for ensuring motion diversity.
5. Each guidance energy term is indispensable; only the full combination achieves the optimal balance between diversity and focus.

## Highlights & Insights

- **Training-free** is the most prominent highlight—it directly leverages the implicit motion priors of pretrained models, offering strong environmental adaptability without being constrained by training data.
- Clever guidance energy design: Static camera + Object motion = Decoupling; Diversity repulsive force = Distribution exploration; Smoothness = Motion quality.
- Iterative sampling + stopping criterion adaptively handles scene complexity, preventing the forced generation of implausible motions in simple scenes.
- The motion completion application demonstrates practical value: a user's crude arrow can be automatically matched to the nearest detailed motion, resolving ambiguities in drag-based editing.

## Limitations & Future Work

- Inheritance of pretrained model data biases—unable to generate motion types not present in the training data.
- Continuous motion space is only sampled discretely (e.g., a laptop can move in any direction, but only a limited number of directions can be sampled).
- The number of forward passes equals the number of motions, meaning computational cost scales linearly with the number of motions.
- Only handles 2D motion fields, without extending to 3D motion.
- Does not support scenes with moving cameras (e.g., action tracking shots).

## Related Work & Insights

- The repulsive energy concept from particle guidance [6] is transferred to motion diversity in video generation, while cleverly bypassing memory limits (via iterative rather than parallel sampling).
- The motion/appearance separation architecture of Motion-I2V is fundamental to this method, illustrating that decoupled representations are crucial for downstream control.
- The concept of guidance energy can be extended to controllable diverse sampling in other generative tasks.
- The motion completion application provides a new paradigm for drag-based image editing: coarse input $\rightarrow$ detailed, plausible motion.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The first training-free method to discover diverse object motions, with exquisitely designed guidance energy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes quantitative metrics, two user studies, and ablations, but with a relatively small number of images (28).
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, comprehensive mathematical derivations, and high-quality visualizations.
- Value: ⭐⭐⭐⭐ Establishes the new research direction of object motion mode discovery, showing broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Motion Prompting: Controlling Video Generation with Motion Trajectories](motion_prompting_controlling_video_generation_with_motion_trajectories.md)
- [\[ICLR 2026\] Video-GPT via Next Clip Diffusion](../../ICLR2026/video_generation/video-gpt_via_next_clip_diffusion.md)
- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2025\] DiTFlow: Video Motion Transfer with Diffusion Transformers](video_motion_transfer_with_diffusion_transformers.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)

</div>

<!-- RELATED:END -->
