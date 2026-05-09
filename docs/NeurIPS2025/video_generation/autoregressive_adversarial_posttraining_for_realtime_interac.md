---
title: >-
  [Paper Note] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation
description: >-
  [NeurIPS 2025][Video Generation][adversarial training] This paper proposes AAPT (Autoregressive Adversarial Post-Training), which converts a pretrained video diffusion model into an autoregressive real-time video generator via adversarial training. The model requires only one forward pass per frame (1NFE), employs student-forcing training to reduce error accumulation, and achieves real-time streaming generation at 736×416 resolution and 24fps on a single H100 GPU, supporting videos up to one minute in length (1440 frames).
tags:
  - NeurIPS 2025
  - Video Generation
  - adversarial training
  - autoregressive video generation
  - real-time interaction
  - one-step generation
  - KV cache
date: 2026-05-08
content_hash: 3cce3fee8553232f
---

# Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation

**Conference**: NeurIPS 2025
**arXiv**: [2506.09350](https://arxiv.org/abs/2506.09350)
**Code**: [https://seaweed-apt.com/2](https://seaweed-apt.com/2)
**Area**: Diffusion Models / Video Generation
**Keywords**: adversarial training, autoregressive video generation, real-time interaction, one-step generation, KV cache

## TL;DR

This paper proposes AAPT (Autoregressive Adversarial Post-Training), which converts a pretrained video diffusion model into an autoregressive real-time video generator via adversarial training. The model requires only one forward pass per frame (1NFE), employs student-forcing training to reduce error accumulation, and achieves real-time streaming generation at 736×416 resolution and 24fps on a single H100 GPU, supporting videos up to one minute in length (1440 frames).

## Background & Motivation

**Background**: Foundation models for video generation (e.g., Wan2.1, HunyuanVideo, Seaweed) can already produce high-quality short videos, but at prohibitive computational cost. Interactive video generation—as required by game engines and world simulators—demands real-time response to user inputs with continuous coherent output, imposing extremely stringent latency and throughput requirements.

**Limitations of Prior Work**: (1) Diffusion models require multi-step denoising; even when distilled to 4–8 steps, they remain too slow. (2) Diffusion Forcing introduces causal attention and KV caching but is inefficient in one-step generation settings—each autoregressive step must process two frames (the current frame and a noisy frame). (3) Existing methods are trained on short windows (typically 5 seconds), and errors accumulate rapidly during long video generation. (4) CausVid (current SOTA) achieves only 640×352 resolution at 9.4fps on a single H100.

**Key Challenge**: Real-time interaction demands extremely low latency and high throughput, yet high-quality video generation is inherently compute-intensive. Teacher-forcing training induces a train-inference distribution mismatch, causing rapid error accumulation in autoregressive settings. Long continuous-shot training data (>10 seconds) is exceedingly rare in most datasets.

**Goal**: (1) Achieve single-step, frame-by-frame real-time video generation. (2) Support minute-long streaming video generation without collapse. (3) Support interactive control (pose, camera).

**Key Insight**: Adversarial training is naturally suited to one-step generation (no paired targets required), and student-forcing is naturally realized within adversarial training—the generator's actual outputs are directly fed back as inputs for the next step, while the discriminator evaluates the entire generated sequence.

**Core Idea**: Transform a video diffusion model into an autoregressive one-step generator through a three-stage pipeline—diffusion adaptation → consistency distillation → adversarial training—combined with student-forcing and long-video training techniques to address error accumulation and data scarcity.

## Method

### Overall Architecture

Given a user-provided first frame and text prompt, the model autoregressively generates video frame by frame. At each step: (1) the previously generated frame (or the first frame) is concatenated channel-wise as input, together with noise and text conditioning; (2) a single forward pass generates all tokens of the next frame via block causal attention and KV caching; (3) the generated frame is decoded by a causal VAE and streamed to the user. A sliding window caps the KV cache size at $N=30$ frames, ensuring constant speed and memory usage.

### Key Designs

1. **Causal Autoregressive Architecture**:

    - **Function**: Converts a bidirectional video diffusion model into an efficient causal autoregressive generator.
    - **Mechanism**: Full attention is replaced with block causal attention (text tokens attend only to themselves; visual tokens attend only to the current and preceding frames). The key innovation is *result recycling*—the previous step's generated output is channel-concatenated and reused as input for the next step, rather than requiring two-frame input as in Diffusion Forcing. This limits each step to processing a single frame's computation, yielding a 2× speedup over Diffusion Forcing. A sliding window of $N=30$ frames (5 seconds) constrains the attention range, preventing unbounded KV cache growth.
    - **Design Motivation**: The LLM-style autoregressive paradigm is naturally compatible with KV caching. Unlike LLMs that emit one token per step, this model outputs all tokens of an entire frame per step, maximizing parallelism.

2. **Student-Forcing Adversarial Training**:

    - **Function**: Eliminates train-inference distribution mismatch and reduces error accumulation in long video generation.
    - **Mechanism**: During adversarial training, the generator operates under student-forcing—only the ground-truth first frame is used; thereafter, each step recycles its own actual generated output as the next input, fully simulating inference behavior. The discriminator evaluates all generated frames in parallel, outputting per-frame logits. Gradients are backpropagated through the KV cache to update all parameters (with frame inputs detached for training stability). The R3GAN relative loss $\mathcal{L} = f(D(G(\epsilon, c), c) - D(x_0, c))$ is used together with approximate R1/R2 regularization.
    - **Design Motivation**: Teacher-forcing adversarial training fails entirely in experiments—content drifts severely within a few frames, as small errors in continuous latents accumulate rapidly. Student-forcing fundamentally eliminates the train-inference distribution gap.

3. **Long-Video Training Technique**:

    - **Function**: Enables minute-level video generation capability in the absence of long-duration training data.
    - **Mechanism**: The generator produces long videos (e.g., 60 seconds), which are segmented into short clips (e.g., 10 seconds with 1-second overlap); each clip is evaluated independently by the discriminator. The discriminator is trained to distinguish generated clips from real short videos. Crucially, the discriminator requires no paired ground truth—it only needs to learn to distinguish real from fake—and can therefore learn from arbitrary short videos. After generating each segment, the KV cache is detached, gradients are backpropagated, and losses are accumulated.
    - **Design Motivation**: Continuous single shots in datasets average only 8 seconds; continuous shots of 30–60 seconds are extremely rare. Supervised methods require long-video ground truth, whereas adversarial training elegantly circumvents this constraint.

### Loss & Training

Three-stage training: (1) Diffusion adaptation (flow-matching loss, teacher-forcing, 30K iterations); (2) Consistency distillation (32 fixed steps, 5K iterations); (3) Adversarial training (R3GAN + R1/R2 regularization, student-forcing, 500+500 generator updates, long-video extension to 55 seconds). The discriminator shares the same causal architecture as the generator (8B parameters) and is initialized from diffusion weights. Total training: approximately 7 days on 256 H100 GPUs.

## Key Experimental Results

### Main Results

| Benchmark / Model | Frames | Temporal Quality | Frame Quality | I2V Subject | I2V Background |
|---|---|---|---|---|---|
| AAPT (Ours) | 120 | 89.31 | **67.18** | **98.20** | **99.38** |
| CausVid | 120 | **92.00** | 65.00 | N/A | N/A |
| Wan 2.1 | 120 | 87.95 | 66.58 | 96.82 | 98.57 |
| Hunyuan | 120 | 89.80 | 64.18 | 97.71 | 97.97 |
| AAPT (Ours) | 1440 | **89.79** | **62.16** | **96.11** | **97.52** |
| SkyReel-V2 | 1440 | 86.51 | 52.58 | 95.28 | 97.85 |
| MAGI-1 | 1440 | 88.90 | 54.76 | 96.70 | 98.61 |

| Task | Method | Params | GPU | Resolution | NFE | Latency | FPS |
|---|---|---|---|---|---|---|---|
| Streaming | CausVid | 5B | 1×H100 | 640×352 | 4 | 1.30s | 9.4 |
| Streaming | **AAPT** | 8B | 1×H100 | 736×416 | **1** | **0.16s** | **24.8** |
| Streaming | MAGI-1 | 24B | 8×H100 | 736×416 | 8 | 7.00s | 3.43 |
| Streaming | **AAPT** | 8B | 8×H100 | 1280×720 | **1** | **0.17s** | **24.2** |

### Ablation Study

| Configuration | Result | Note |
|---|---|---|
| Teacher-forcing adversarial training | Severe drift after a few frames | Distribution mismatch causes rapid error accumulation |
| Student-forcing | Stable generation | Training and inference behavior are aligned |
| Long-video training 10s | Temporal 85.86, Frame 57.92 | Poor quality on 1-minute generation |
| Long-video training 20s | Temporal 85.60, Frame 65.69 | Quality improves |
| Long-video training 60s | Temporal **89.79**, Frame 62.16 | Quality on 1-minute generation substantially improved |
| Without result recycling | Cannot generate large motions | Lack of previous frame leads to temporal incoherence |

### Key Findings

- Adversarial training is better suited to one-step autoregressive generation than Diffusion Forcing + step distillation, achieving 2× efficiency gains with superior quality.
- Student-forcing is the key to success—teacher-forcing adversarial training fails completely.
- Long-video training improves the model from "collapsing after 10 seconds" to "remaining stable at 60 seconds"; competing baselines (SkyReel-V2, MAGI-1) exhibit noticeable degradation after 20–30 seconds.
- Pose control (AKD=2.740) approaches SOTA OmniHuman-1 (2.136); camera control surpasses CameraCtrl2 on FVD and translation error.

## Highlights & Insights

- The adversarial training paradigm is innovatively applied not only to improve generation quality, but also to address two critical engineering challenges—student-forcing eliminates error accumulation, and adversarial training enables long-video training without paired data.
- The synergy between architecture design and training strategy is notable: result recycling + block causal attention + 1NFE work in concert so that each step processes only one frame, doubling efficiency over the two-frame design in Diffusion Forcing.
- Achieving 24fps real-time video generation on a single H100 represents a significant practical milestone, substantially lowering the deployment barrier for interactive applications.

## Limitations & Future Work

- **Consistency**: Subjects and scenes may drift over long videos; the sliding window limits the maintenance of global consistency.
- **One-step generation quality**: Artifacts occasionally appear and, once introduced, propagate persistently through the temporal sequence.
- Zero-shot 5-minute generation tests still exhibit artifacts; ultra-long video generation requires further improvement.
- Training cost is high (256 H100 GPUs × 7 days); the discriminator is of equal scale to the generator (16B parameters combined).

## Related Work & Insights

- **vs. CausVid (Yin et al., 2024)**: CausVid converts a bidirectional diffusion model into a causal Diffusion Forcing model distilled to 4 steps. AAPT goes further—reducing to 1 step and replacing step distillation with adversarial training—achieving a 2.6× speedup (24fps vs. 9.4fps).
- **vs. APT (Lin et al., 2025)**: AAPT extends APT (adversarial post-training for images) to the autoregressive video setting, adding key components including student-forcing, a per-frame discriminator, and long-video training.
- **vs. MAGI-1 / SkyReel-V2**: These models are trained from scratch with Diffusion Forcing and require 8–24 steps and 24+ GPUs for real-time generation. AAPT achieves one-step generation by post-training an existing model, resulting in substantially lower deployment cost.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of adversarial training to autoregressive video generation; the combination of student-forcing and long-video training is highly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive VBench evaluation on short and long videos, two interactive control applications (pose and camera), and thorough speed comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, intuitive method description, and detailed appendix.
- **Value**: ⭐⭐⭐⭐⭐ Real-time video generation is a major practical breakthrough, directly enabling interactive gaming, virtual avatar, and related applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MotionStream: Real-Time Video Generation with Interactive Motion Controls](../../ICLR2026/video_generation/motionstream_real-time_video_generation_with_interactive_motion_controls.md)
- [\[NeurIPS 2025\] Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](self_forcing_bridging_the_train-test_gap_in_autoregressive_video_diffusion.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](../../ICCV2025/video_generation/adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[NeurIPS 2025\] Training-Free Efficient Video Generation via Dynamic Token Carving](training-free_efficient_video_generation_via_dynamic_token_carving.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](../../ICCV2025/video_generation/realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)

</div>

<!-- RELATED:END -->
