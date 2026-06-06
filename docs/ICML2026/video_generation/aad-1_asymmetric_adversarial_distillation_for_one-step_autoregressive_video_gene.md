---
title: >-
  [Paper Note] AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation
description: >-
  [ICML 2026][Video Generation][Autoregressive Diffusion] AAD-1 employs asymmetric adversarial distillation using a "causal generator + bidirectional video-level discriminator" alongside DMD warmup to compress autoregressi…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "Autoregressive Diffusion"
  - "One-step Distillation"
  - "Adversarial Distillation"
  - "Long Video Consistency"
date: 2026-05-08
content_hash: 0fc5bf4345e6f78f
---

# AAD-1: Asymmetric Adversarial Distillation for One-Step Autoregressive Video Generation

**Conference**: ICML 2026  
**arXiv**: [2606.03972](https://arxiv.org/abs/2606.03972)  
**Code**: https://aad-1.github.io/  
**Area**: Video Generation  
**Keywords**: Video Generation, Autoregressive Diffusion, One-step Distillation, Adversarial Distillation, Long Video Consistency  

## TL;DR
AAD-1 employs asymmetric adversarial distillation using a "causal generator + bidirectional video-level discriminator" alongside DMD warmup to compress autoregressive image-to-video generation into a single sampling step per chunk, while mitigating motion collapse and long-range drift.

## Background & Motivation
**Background**: Video diffusion models typically generate short clips, but fixed lengths and multi-step sampling limit real-time streaming applications. Autoregressive video diffusion supports longer videos by generating chunk-by-chunk and reusing context and KV cache, making it suitable for games, world models, and online generation.

**Limitations of Prior Work**: Compressing autoregressive models to few-step or one-step generation is extremely difficult. Existing methods often perform causal adaptation, autoregressive rollout, and sampling step distillation simultaneously, imposing a heavy optimization burden. While adversarial distillation is suitable for one-step generation, it often causes the video to remain static near the initial frame, leading to motion collapse.

**Key Challenge**: Specifically, during deployment, the generator must be strictly causal and cannot observe future frames. However, if the discriminator is also restricted to causal observation during training, it becomes difficult to detect drift and static replication that accumulate across the entire video. The generation side requires causality, whereas the supervision side requires a global temporal perspective.

**Goal**: The authors aim to train a one-step autoregressive I2V model that maintains streaming generation capabilities while ensuring the training signal can penalize long-range drift and global motion failures.

**Key Insight**: This paper breaks the structural symmetry between the generator and discriminator: the generator maintains a causal structure, while the discriminator utilizes bidirectional spatiotemporal context during training and outputs a video-level realism score for the entire segment.

**Core Idea**: Use asymmetric adversarial distillation to allow the discriminator a global view while the generator remains causal, combined with ODE initialization and DMD warmup to bring the one-step generator close to the stable distribution.

## Method
The AAD-1 method can be understood as a three-stage training recipe. The first stage transforms a pre-trained bidirectional video model into a causal student; the second stage uses distribution matching to bring the one-step student closer to the teacher; the third stage performs adversarial refinement, using a bidirectional video-level discriminator to provide global temporal supervision.

### Overall Architecture
During deployment, the generator $G_\theta$ generates the video chunk-by-chunk. In each step, it only observes the initial sink frames and the most recent sliding-window context to output the current chunk. During training, the model autoregressively rolls out a complete segment, which is then fed into the discriminator. The discriminator is initialized from the Wan 2.1 T2V backbone, with cross-attention heads inserted into several transformer layers. Learnable query tokens aggregate complete spatiotemporal features to output a single video-level logit.

Training consists of three steps. Stage I uses Diffusion Forcing and an ODE teacher trajectory to replace the full attention of the bidirectional model with block-wise causal attention, and learns downstream few-step target timesteps. Stage II utilizes Self-Forcing DMD to match the teacher and student distributions under an autoregressive context, ensuring the one-step output does not deviate from the data manifold. Stage III performs asymmetric adversarial distillation using a logistic GAN loss and approximate R1/R2 regularization.

### Key Designs
1. **Asymmetric Generator-Discriminator Architecture**:

	- **Function**: Simultaneously satisfies streaming causal generation and global temporal quality supervision.
	- **Mechanism**: The generator can only access sink frames and recent historical frames to ensure autoregressive inference. During discriminator training, the discriminator accesses the full video, where bidirectional attention can compare any past and future frames to judge global authenticity via a video-level logit.
	- **Design Motivation**: Motion collapse is a global temporal failure. Single-frame or causal discriminators are easily deceived by realistic-looking static frames. A bidirectional video-level discriminator can detect "static segments" or "gradual drift."

2. **DMD warmup before GAN**:

	- **Function**: Allows the one-step generator to approach the teacher distribution before adversarial training.
	- **Mechanism**: Following the student's autoregressive rollout, Gaussian noise is added to the entire generated sequence. The difference between the real score model and the fake score model provides the DMD gradient, pushing the student distribution toward the teacher.
	- **Design Motivation**: When cold-starting a GAN, one-step predictions are too far from the data distribution, leading to unstable discriminator gradients. DMD warmup provides a trainable starting point.

3. **Noised Discriminator Inputs & Regularization**:

	- **Function**: Stabilizes adversarial training for 14B-parameter models.
	- **Mechanism**: Both real and generated videos are fed into the discriminator with Gaussian noise at random timesteps. Simultaneously, approximate R1/R2 regularization is used to penalize discriminator hypersensitivity to small perturbations, with the regularization weight set to $\lambda=20$.
	- **Design Motivation**: Asymmetric GANs are prone to instability. Noise and regularization help the discriminator provide smooth gradients, preventing the generator from collapsing quickly or producing grid artifacts.

### Loss & Training
Stage I uses ODE trajectory regression with a target such as $\|G_\theta(z_t,\tilde{x}_{ctx,t},c)-S^{ODE}_\phi(z_t,\tilde{x}_{ctx,t},c)\|_2^2$. Stage II uses DMD gradients, essentially the difference between the real score and the fake score multiplied by the gradient of the generated sequence relative to the parameters. Stage III uses standard logistic GAN: the discriminator maximizes the score of real segments and minimizes the score of generated segments, while the generator maximizes the discriminator's assessment of generated segments. Implementation uses the Wan 2.1 14B backbone. Stage I is trained for 2,000 steps, Stage II DMD generator is trained for roughly 100 steps with early stopping, and Stage III generator is trained for 200 steps.

## Key Experimental Results

### Main Results
The main experiment compares one-step AAD-1 with multi-step autoregressive baselines on VBench-I2V, using Wan 2.1 I2V with 100 NFE as a bidirectional reference.

| Method | NFE | Subject Cons.↑ | Background Cons.↑ | Dynamic Degree↑ | Imaging Quality↑ | I2V Subject↑ | I2V Background↑ |
|--------|------|------|------|------|------|------|------|
| Wan 2.1 I2V | 100 | 93.88 | 94.86 | 51.09 | 70.12 | 96.80 | 98.59 |
| CausVid | 4 | 83.45 | 89.37 | 33.80 | 70.60 | 92.91 | 83.34 |
| Self Forcing | 4 | 91.77 | 93.41 | 34.93 | 71.50 | 95.79 | 91.18 |
| **Ours** Stage-II | 1 | 92.14 | 92.13 | 50.30 | 69.37 | 96.56 | 95.12 |
| **Ours** Stage-III | 1 | 94.34 | 95.08 | 41.46 | 71.49 | 98.65 | 97.83 |

### Ablation Study
| Configuration | Key Metrics | Description |
|------|---------|------|
| w/o DMD warmup | Aesthetic 53.63, Imaging 62.81 | Initial one-step distribution is too far; GAN refinement is unstable. |
| w/ DMD warmup | Aesthetic 58.64, Imaging 69.37 | Warmup significantly improves base quality before the adversarial stage. |
| Causal DiT + frame-wise logit | Dynamic Degree 1.08 | Degenerates into static video, typical motion collapse. |
| Causal DiT + video-wise logit | Drift 7.10, Dynamics 42.07 | Some motion exists but long-range drift is severe. |
| Bidirectional DiT + frame-wise logit | Drift 4.38, Dynamics 39.04 | Bidirectional context significantly reduces drift. |
| Bidirectional DiT + video-wise logit | Drift 4.02, Dynamics 39.29 | Best drift control, default configuration for this work. |
| 14B 1 NFE inference | Latency 1.134s, Throughput 14.33 FPS | Much faster than 2.822s / 5.71 FPS for 4 NFE. |

### Key Findings
- Stage-III adversarial refinement improves subject/background consistency and I2V faithfulness, but sacrifices some motion magnitude; Stage-II exhibits a higher Dynamic Degree.
- The visibility of the discriminator is more critical than logit granularity: a causal backbone accumulates errors, whereas a bidirectional backbone provides a future-anchored critique.
- DMD warmup is not just a trick but a prerequisite for stable one-step GAN training; without it, the generator degenerates if it enters the adversarial stage too early.
- The regularization coefficient has a narrow window: $\lambda=0$ results in collapse, $\lambda=50$ introduces grid artifacts, and $\lambda=20$ achieves the best balance.

## Highlights & Insights
- The most clever aspect is the asymmetry: inference constraints only require the generator to be causal, while training the discriminator can fully leverage future frames. This separation decouples the "deployment architecture" from the "supervision architecture."
- The video-level logit directly addresses the root cause of motion collapse. Frame-wise discrimination only considers marginal image distributions, where copying the previous frame looks like a real image; video-level discrimination penalizes sequences without motion.
- Three-stage training decomposes a difficult problem: learning causalization first, then the one-step distribution, and finally performing perceptual refinement. This is more stable than merging all objectives into a single joint loss.

## Limitations & Future Work
- One-step chunk-wise generation is still prone to blurring or structural deformation in fast-motion scenes, as large displacements are compressed into a single denoising step.
- Complex local structures such as human faces and hands require high multi-frame synchronization within a chunk; detail preservation remains weaker than multi-step or single-frame refinement.
- Adversarial refinement is mainly trained on 5-second clips, and long video extrapolation still accumulates errors with autoregressive rollout.
- Training costs are high: full training takes approximately 3.5 days on 64 H20 GPUs, with Stage III peak memory reaching about 1040GB, indicating high entry barriers despite fast inference.

## Related Work & Insights
- **vs Self Forcing / Diffusion Forcing**: These address the autoregressive train-test gap; AAD-1 inherits the self-rollout idea and pushes it to one-step.
- **vs APT2**: APT2 uses a causal frame-wise discriminator, whereas AAD-1 switches to a bidirectional video-level discriminator and provides large-scale controlled ablations proving its necessity.
- **vs Wan 2.1 I2V**: Wan is a strong bidirectional multi-step teacher/reference; AAD-1 utilizes its backbone and distribution knowledge to obtain a real-time friendly autoregressive model.
- **Insight**: In many generative tasks, the critic during training does not need to obey the causal constraints of inference; as long as the generator maintains the deployment constraints, the critic can perform more global and expensive quality reviews.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The asymmetric design of the causal generator and bidirectional video-level discriminator captures the key contradiction in one-step autoregressive video.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers VBench, user preferences, warmup, discriminator, and efficiency; benchmarks for longer real videos could be further strengthened.
- Writing Quality: ⭐⭐⭐⭐☆ The method is clearly outlined with sufficient formulas and training details; some cost information is placed in the appendix.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for real-time video generation and world model streaming inference, especially regarding the critic design logic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Phased One-Step Adversarial Equilibrium for Video Diffusion Models](../../AAAI2026/video_generation/phased_one-step_adversarial_equilibrium_for_video_diffusion_models.md)
- [\[ICML 2026\] SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion](sgmd_score_gradient_matching_distillation_for_few-step_video_diffusion_distillat.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)
- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[ICCV 2025\] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](../../ICCV2025/video_generation/dollar_fewstep_video_generation_via_distillation_and_latent.md)

</div>

<!-- RELATED:END -->
