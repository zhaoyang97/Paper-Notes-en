---
title: >-
  [Paper Note] Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity
description: >-
  [ACL 2026][Video Generation][Autoregressive Video Generation] The authors propose a training strategy combining Local Optimization and Representation Continuity (ReCo). By optimizing within local windows and constraining…
tags:
  - "ACL 2026"
  - "Video Generation"
  - "Autoregressive Video Generation"
  - "Training Acceleration"
  - "Local Optimization"
  - "Representation Continuity"
  - "Lipschitz Continuity"
date: 2026-05-08
content_hash: 836892bacc750451
---

# Accelerating Training of Autoregressive Video Generation Models via Local Optimization with Representation Continuity

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.07402](https://arxiv.org/abs/2604.07402)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Autoregressive Video Generation, Training Acceleration, Local Optimization, Representation Continuity, Lipschitz Continuity

## TL;DR
The authors propose a training strategy combining Local Optimization and Representation Continuity (ReCo). By optimizing within local windows and constraining the smooth transition of hidden states, it achieves a 2x acceleration in training for autoregressive video generation models without compromising generation quality.

## Background & Motivation

**Background**: Autoregressive models have demonstrated superior inference speed and performance compared to diffusion models in image generation. However, in video generation, training costs are extremely high because video token sequences are significantly longer than those of images, requiring full-sequence autoregressive modeling.

**Limitations of Prior Work**: Intuitively, training could be accelerated by reducing the number of frames during training (Fewer-Frames method). However, experiments show that this leads to severe error accumulation and temporal inconsistency. During inference, each block is generated based on a preceding block that might contain errors; without global context, these errors amplify exponentially.

**Key Challenge**: There is a trade-off between training efficiency and generation consistency. While reducing training frames lowers the computational load, it disrupts temporal coherence, causing significant deterioration in FVD (e.g., FFS increases from 73.65 to 229.32).

**Goal**: To halve the training cost while maintaining baseline-level video quality and temporal consistency.

**Key Insight**: The authors approach this from two levels: (1) Training strategy: replacing full-sequence optimization with local window optimization, using context outside the window as frozen conditions; (2) Representation space: employing Lipschitz continuity to constrain the magnitude of hidden state changes between adjacent time steps to suppress error propagation.

**Core Idea**: Autoregressive loss is optimized within randomly sampled local windows (Local Opt.), while a representation continuity loss (ReCo) constrains smooth transitions of hidden states. This significantly reduces computation during training while maintaining full-sequence generation consistency during inference.

## Method

### Overall Architecture
Input videos are first encoded into discrete token sequences by a VQ-VAE (OmniTokenizer) and then modeled by an autoregressive Transformer. Instead of calculating loss over the complete sequence, the model randomly samples a local window for optimization. Tokens preceding the window serve as frozen context (stop-gradient). Continuity constraints are applied to hidden states within the window. Standard full-sequence autoregressive generation is still used for inference.

### Key Designs

1.  **Local Optimization (Local Opt.)**:
    - **Function**: Calculates autoregressive loss within a randomly sampled local window to significantly reduce the computational load per training step.
    - **Mechanism**: Given a complete token sequence $\mathbf{E}$, a starting position $s$ and window length $W$ are randomly sampled. The cross-entropy loss is calculated only within the window $\mathbf{E}_\mathcal{W} = (\mathbf{e}_s, ..., \mathbf{e}_{s+W-1})$. Preceding tokens $\mathbf{E}_{<s}$ act as frozen context. Overlapping windows with stride $S < W$ are used so that tokens are optimized multiple times under different contexts.
    - **Design Motivation**: To address two core issues of the Fewer-Frames method: (1) Always conditioning on ground-truth context to avoid exposure bias; (2) Forcing the model to learn more robust representations via overlapping windows. This does not affect inference speed as standard generation is used.

2.  **First-Frame Balanced Sampling**:
    - **Function**: Resolves the training-generation distribution mismatch by increasing the sampling ratio of windows containing the first frame.
    - **Mechanism**: Analysis revealed that the loss distribution of the Local Opt. model on generated samples differs significantly from training samples, particularly with higher loss on the first frame. The sampling probability for windows containing the first frame is increased to 0.5.
    - **Design Motivation**: The quality of the first frame directly impacts all subsequent frames. Experiments showed that balanced sampling reduced FVD from 190.46 to 127.11 and further improved training speed to 2.0x.

3.  **Representation Continuity (ReCo)**:
    - **Function**: Constrains the magnitude of hidden state changes between adjacent time steps to enhance temporal smoothness.
    - **Mechanism**: Viewing the autoregressive model as a discrete-time dynamical system and inspired by Lipschitz continuity, a continuity loss is applied within the window: $\mathcal{L}_{ReCo} = \frac{1}{W-1}\sum_{i=s}^{s+W-2}\|\mathbf{h}_{i+1} - \mathbf{h}_s\|_2^2$. The total loss is $\mathcal{L}_{Total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{ReCo}$.
    - **Design Motivation**: Local Opt. focused on independent windows might produce abrupt changes in representation space. By constraining a small local Lipschitz constant, error propagation is restricted to a linear growth range $\|\epsilon_{t+1}\| \leq L \cdot \|\epsilon_t\| + \delta_t$ rather than exponential amplification.

### Loss & Training
The total loss consists of: (1) standard cross-entropy loss $\mathcal{L}_{CE}$ within the window; (2) representation continuity regularization $\mathcal{L}_{ReCo}$ with weight $\lambda=0.1$. The first-frame window sampling probability is set to 0.5. The model is trained for 300 epochs with a learning rate of $1\times10^{-4}$.

## Key Experimental Results

### Main Results

| Dataset | Metric | ReCo★ | Baseline★ | Gain |
| :--- | :--- | :--- | :--- | :--- |
| FFS | FVD↓ | 42.5 | 46.1 | -7.8% |
| SKY | FVD↓ | 58.8 | 62.7 | -6.2% |
| UCF101 | FVD↓ | 251.4 | 254.5 | -1.2% |
| Taichi | FVD↓ | 98.3 | 105.5 | -6.8% |

Training Speed: ReCo is approximately 2x faster than Baseline.

### Ablation Study

| Config | FFS FVD↓ | SKY FVD↓ | Training Speed |
| :--- | :--- | :--- | :--- |
| Baseline | 73.65 | 89.09 | 1.0× |
| Fewer-Frames | 229.32 | 292.41 | 2.5× |
| Local-Opt. | 190.46 | 256.94 | 1.7× |
| Local-Opt. (w/ first frame) | 134.73 | 186.63 | 1.7× |
| Local-Opt. (w/ balanced) | 127.11 | 179.84 | 2.0× |
| ReCo (Full method) | 72.6 | 87.5 | 2.0× |

### Key Findings
- The Fewer-Frames method, while 2.5x faster, deteriorates FVD by over 3x, confirming the theoretical analysis of error accumulation.
- The first-frame balanced sampling strategy in Local Opt. contributes significantly, reducing FVD from 190 to 127.
- ReCo further reduces FVD from 127 to 72.6, performing equal to or better than the Baseline (73.7), validating the effectiveness of Lipschitz regularization.
- On the MSR-VTT text-to-video task, ReCo* achieves CLIP Scores and FVD comparable to a 7B baseline at 50% of the training cost.

## Highlights & Insights
- **Inspiration from Dynamical Systems**: Treating the autoregressive model as a discrete dynamical system and using Lipschitz continuity theory to guide regularization is an innovative perspective for understanding and improving sequences.
- **Decoupled Training-Inference Design**: Local Opt. modifies only the training process (local window optimization). Inference remains standard full-sequence generation. This "training trick without inference overhead" philosophy is highly valuable.
- **Improvements Driven by Loss Distribution Analysis**: Identifying the first-frame bottleneck by comparing loss distributions between training and generated samples led to the balanced sampling strategy. This data-driven approach is transferable to other sequence tasks.

## Limitations & Future Work
- Experiments were primarily conducted on small-scale models (110M-770M) and short videos (17 frames), without testing on commercial large-scale models.
- The $\lambda$ hyperparameter for ReCo might require tuning for different datasets and resolutions.
- Text-to-video experiments were limited to zero-shot evaluation on MSR-VTT, lacking verification on more benchmarks.
- The combination of ReCo with other acceleration techniques (e.g., KV-cache compression, quantization) has not been explored.

## Related Work & Insights
- **vs Fewer-Frames**: Fewer-Frames only reduces frames for training and generates iteratively, leading to severe error accumulation. ReCo solves this on the training side via Local Opt. and continuity constraints, while maintaining full-sequence generation at inference.
- **vs LARP**: LARP improves video quality through a better tokenizer. ReCo is complementary to LARP—ReCo♠ (combined with LARP) achieved 56.1 FVD on UCF (vs 57.0 for LARP alone).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The application of dynamical system perspectives and Lipschitz regularization to autoregressive video generation is quite novel, though core concepts (local optimization + smoothing) have precedents in NLP.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 datasets, 2 model scales, text-to-video extensions, and detailed ablations, though lacks large-scale validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical chain from problem analysis to theoretical proof, method design, and experimental verification is very clear. The visualizations are intuitive and effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Light Forcing: Accelerating Autoregressive Video Diffusion via Sparse Attention](../../ICML2026/video_generation/light_forcing_accelerating_autoregressive_video_diffusion_via_sparse_attention.md)
- [\[CVPR 2026\] PhysVid: Physics Aware Local Conditioning for Generative Video Models](../../CVPR2026/video_generation/physvid_physics_aware_local_conditioning_for_generative_video_models.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[ICCV 2025\] VPO: Aligning Text-to-Video Generation Models with Prompt Optimization](../../ICCV2025/video_generation/vpo_aligning_text-to-video_generation_models_with_prompt_optimization.md)
- [\[ICML 2026\] WorldCache: Accelerating World Models for Free via Heterogeneous Token Caching](../../ICML2026/video_generation/worldcache_accelerating_world_models_for_free_via_heterogeneous_token_caching.md)

</div>

<!-- RELATED:END -->
