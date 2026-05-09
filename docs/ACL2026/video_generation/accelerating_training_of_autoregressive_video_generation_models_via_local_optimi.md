---
title: >-
  [Paper Note] Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity
description: >-
  [ACL 2026][Video Generation][Autoregressive Video Generation] This paper proposes a Local Optimization + Representation Continuity (ReCo) training strategy that optimizes within local windows while constraining smooth transitions of hidden states, achieving 2× training speedup for autoregressive video generation models without sacrificing generation quality.
tags:
  - ACL 2026
  - Video Generation
  - Autoregressive Video Generation
  - Training Acceleration
  - Local Optimization
  - Representation Continuity
  - Lipschitz Continuity
date: 2025-04-17
content_hash: b088897d33c681f7
---

# Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity

**Conference**: ACL 2026  
**arXiv**: [2604.07402](https://arxiv.org/abs/2604.07402)  
**Code**: N/A  
**Area**: Video Generation  
**Keywords**: Autoregressive Video Generation, Training Acceleration, Local Optimization, Representation Continuity, Lipschitz Continuity

## TL;DR
This paper proposes a Local Optimization + Representation Continuity (ReCo) training strategy that optimizes within local windows while constraining smooth transitions of hidden states, achieving 2× training speedup for autoregressive video generation models without sacrificing generation quality.

## Background & Motivation

**Background**: Autoregressive models have demonstrated superior inference speed and performance over diffusion models in image generation, but in video generation, the extremely long video token sequences make training costs prohibitively high (requiring full-sequence autoregressive modeling across complete video frame sequences).

**Limitations of Prior Work**: Intuitively, training can be accelerated by reducing the number of training frames (Fewer-Frames method) — training on short sequences then iteratively generating during inference. However, experiments reveal this causes severe error accumulation and temporal inconsistency, as each block at inference is generated based only on the previous (potentially erroneous) block without global context information, leading to exponential error amplification.

**Key Challenge**: A trade-off exists between training efficiency and generation consistency. Reducing training frames decreases computation but disrupts temporal coherence between video frames, causing severe FVD degradation (e.g., FFS from 73.65 to 229.32).

**Goal**: Halve training cost while maintaining baseline-level video quality and temporal consistency.

**Key Insight**: The authors approach from two levels: (1) Training strategy: replace full-sequence optimization with local window optimization, using out-of-window context as frozen conditional input; (2) Representation space: constrain hidden state variation magnitude between adjacent timesteps based on Lipschitz continuity to suppress error propagation.

**Core Idea**: Optimize autoregressive loss within randomly sampled local windows (Local Opt.) while using representation continuity loss (ReCo) to constrain smooth hidden state transitions, thereby substantially reducing computation during training while maintaining full-sequence generation consistency during inference.

## Method

### Overall Architecture
Input video is first encoded into discrete token sequences via VQ-VAE (OmniTokenizer), then modeled with an autoregressive Transformer. During training, loss is not computed on the complete sequence but on a randomly sampled local window, with preceding tokens outside the window serving as frozen context (stop-gradient). Continuity constraints are simultaneously applied to hidden states within the window. Inference uses standard full-sequence autoregressive generation.

### Key Designs

1. **Local Optimization**:

    - Function: Compute autoregressive loss within randomly sampled local windows, substantially reducing per-step training computation
    - Mechanism: Given the complete token sequence $\mathbf{E}$, randomly sample starting position $s$ and window length $W$, computing cross-entropy loss only within window $\mathbf{E}_\mathcal{W} = (\mathbf{e}_s, ..., \mathbf{e}_{s+W-1})$. Tokens before the window $\mathbf{E}_{<s}$ serve as frozen context (no gradient backpropagation). Overlapping windows are created using stride $S < W$, enabling tokens to be optimized multiple times under different contexts
    - Design Motivation: Addresses two core problems of the Fewer-Frames method: (1) always conditioning on ground-truth context, avoiding exposure bias; (2) overlapping windows force the model to learn more robust representations. Inference still uses standard full-sequence generation without affecting inference speed

2. **First-Frame Balanced Sampling**:

    - Function: Resolve training-generation distribution mismatch by increasing sampling proportion of windows containing the first frame
    - Mechanism: Analysis reveals that the Local Opt. model's loss distribution on generated samples differs significantly from training samples, with particularly high first-frame loss. The sampling probability of windows containing the first frame is increased to 0.5, enabling more optimization of the video beginning
    - Design Motivation: First-frame quality directly impacts all subsequent frame generation. Experiments show balanced sampling reduces FVD from 190.46 to 127.11 while further increasing training speed to 2.0×

3. **Representation Continuity (ReCo)**:

    - Function: Constrain hidden state variation magnitude between adjacent timesteps, enhancing temporal smoothness
    - Mechanism: Viewing the autoregressive model as a discrete-time dynamical system, inspired by Lipschitz continuity, a continuity loss is applied to adjacent hidden states within the window: $\mathcal{L}_{ReCo} = \frac{1}{W-1}\sum_{i=s}^{s+W-2}\|\mathbf{h}_{i+1} - \mathbf{h}_i\|_2^2$. Total loss is $\mathcal{L}_{Total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{ReCo}$
    - Design Motivation: Local Opt. focusing on independent windows may produce abrupt changes in representation space. By constraining small local Lipschitz constants, error propagation is bounded to linear growth $\|\epsilon_{t+1}\| \leq L \cdot \|\epsilon_t\| + \delta_t$ rather than exponential amplification

### Loss & Training
Total loss consists of two parts: (1) standard cross-entropy loss $\mathcal{L}_{CE}$ within the window; (2) representation continuity regularizer $\mathcal{L}_{ReCo}$ with weight $\lambda=0.1$. First-frame window sampling probability is set to 0.5. Training for 300 epochs with learning rate $1\times10^{-4}$.

## Key Experimental Results

### Main Results

| Dataset | Metric | ReCo★ | Baseline★ | Gain |
|--------|------|-------|-----------|------|
| FFS | FVD↓ | 42.5 | 46.1 | -7.8% |
| SKY | FVD↓ | 58.8 | 62.7 | -6.2% |
| UCF101 | FVD↓ | 251.4 | 254.5 | -1.2% |
| Taichi | FVD↓ | 98.3 | 105.5 | -6.8% |

Training speed: ReCo is approximately 2× faster than Baseline.

### Ablation Study

| Config | FFS FVD↓ | SKY FVD↓ | Training Speed |
|------|----------|----------|----------|
| Baseline | 73.65 | 89.09 | 1.0× |
| Fewer-Frames | 229.32 | 292.41 | 2.5× |
| Local-Opt. | 190.46 | 256.94 | 1.7× |
| Local-Opt. (w/ first frame) | 134.73 | 186.63 | 1.7× |
| Local-Opt. (w/ balanced) | 127.11 | 179.84 | 2.0× |
| ReCo (full method) | 72.6 | 87.5 | 2.0× |

### Key Findings
- Fewer-Frames method achieves 2.5× training speedup but FVD degrades by over 3×, confirming the theoretical error accumulation analysis
- Local Opt.'s first-frame balanced sampling strategy contributes substantially, reducing FVD from 190 to 127
- ReCo further reduces FVD from 127 to 72.6, matching or even surpassing the Baseline (73.7), validating the effectiveness of Lipschitz regularization
- On MSR-VTT text-to-video tasks, ReCo* achieves comparable CLIP Score and FVD to the 7B baseline at 50% training cost

## Highlights & Insights
- **Dynamical systems perspective**: Viewing the autoregressive model as a discrete dynamical system and using Lipschitz continuity theory to guide regularization design provides a new tool for understanding and improving autoregressive generation
- **Training-inference decoupled design**: Local Opt. only modifies the training procedure (local window optimization) while maintaining standard full-sequence generation during inference — this "training trick without affecting inference" design philosophy is worth adopting
- **Loss distribution analysis-driven improvement**: Discovering the first-frame bottleneck by comparing loss distributions between training/generated samples, then designing balanced sampling, represents a data-driven improvement approach transferable to other sequence generation tasks

## Limitations & Future Work
- Experiments are mainly on small-scale models (110M-770M) and short videos (17 frames); not tested on commercial large models
- ReCo's $\lambda$ hyperparameter may require tuning for different datasets and resolutions
- Text-to-video experiments are limited to zero-shot evaluation on MSR-VTT, lacking validation on more T2V benchmarks
- Combination effects of ReCo with other acceleration techniques (e.g., KV-cache compression, quantization) are unexplored

## Related Work & Insights
- **vs Fewer-Frames**: Fewer-Frames only reduces training frame count, causing severe error accumulation during iterative inference generation; ReCo solves the problem on the training side through Local Opt. + continuity constraints while maintaining full-sequence generation at inference
- **vs LARP**: LARP improves video quality through a better tokenizer; ReCo is orthogonal and complementary — ReCo♠ (combined with LARP) achieves 56.1 FVD on UCF (LARP original: 57.0)

## Rating
- Novelty: ⭐⭐⭐⭐ The dynamical systems perspective + Lipschitz regularization in autoregressive video generation is relatively novel, though the core ideas (local optimization + smoothness constraints) have precedents in NLP sequence modeling
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets + 2 model scales + text-to-video extension experiments + detailed ablation, but lacks large-scale validation
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem analysis → theoretical proof → method design → experimental validation is very clear; figures are intuitive and effective

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[CVPR 2026\] Infinity-RoPE: Action-Controllable Infinite Video Generation Emerges From Autoregressive Self-Rollout](../../CVPR2026/video_generation/infinity-rope_action-controllable_infinite_video_generation_emerges_from_autoreg.md)

<!-- RELATED:END -->
