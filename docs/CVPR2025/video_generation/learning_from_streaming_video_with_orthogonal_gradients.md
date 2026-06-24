---
title: >-
  [Paper Note] Learning from Streaming Video with Orthogonal Gradients
description: >-
  [CVPR 2025][Video Generation][Streaming Video Learning] Addressing the issue of gradient redundancy and model collapse caused by highly correlated continuous frames in streaming video learning, an Orthogonal Optimizer is proposed. By projecting the current gradient onto the orthogonal component of historical gradients for decorrelation, it can be seamlessly integrated into SGD/AdamW. It significantly recovers the performance loss of transitioning from shuffled training to seq…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Streaming Video Learning"
  - "Orthogonal Gradients"
  - "Optimizer Design"
  - "Temporal Correlation"
  - "Non-IID Training"
date: 2026-05-08
content_hash: 7f66590c736ce5a0
---

# Learning from Streaming Video with Orthogonal Gradients

**Conference**: CVPR 2025  
**arXiv**: [2504.01961](https://arxiv.org/abs/2504.01961)  
**Code**: None  
**Area**: Video Generation  
**Keywords**: Streaming Video Learning, Orthogonal Gradients, Optimizer Design, Temporal Correlation, Non-IID Training

## TL;DR
Addressing the issue of gradient redundancy and model collapse caused by highly correlated continuous frames in streaming video learning, an Orthogonal Optimizer is proposed. By projecting the current gradient onto the orthogonal component of historical gradients for decorrelation, it can be seamlessly integrated into SGD/AdamW. It significantly recovers the performance loss of transitioning from shuffled training to sequential training across three scenarios: DoRA, VideoMAE, and future prediction.

## Background & Motivation

1. **Background**: The current standard practice in video representation learning is to partition long videos into short clips and randomly shuffle them for training, satisfying the IID (independent and identically distributed) assumption of optimizers like SGD.
2. **Limitations of Prior Work**: When videos can only be acquired as a continuous stream (e.g., online learning in robotics, privacy-preserving scenarios where video is not stored), the IID assumption is violated, causing a sharp drop in model performance or even collapse. The authors demonstrate on DoRA that sequential training with AdamW causes the ImageNet kNN accuracy to plummet from 74.4% to 1.8%.
3. **Key Challenge**: The changes between continuous video frames are extremely slow, leading to highly similar gradients in adjacent batches (cosine similarity near 1). This causes the optimizer to over-update in the same direction, failing to learn new information.
4. **Goal**: How to learn high-quality visual representations from sequential video streams without storing or shuffling videos?
5. **Key Insight**: Since the core problem lies in gradient correlation, decorrelation can be performed at the optimizer level—retaining only the orthogonal component of the current gradient relative to the historical gradients for updates.
6. **Core Idea**: Eliminate redundant gradient information using orthogonal projection, updating the model only with "new information" to bridge the gap between sequential training and IID training.

## Method

### Overall Architecture
The input is a continuous stream of long video, and the model sequentially processes video clips along the timeline. In each training step, after calculating the gradient of the current batch, instead of directly using it to update the parameters, it is first projected onto the orthogonal component of the historical gradient direction. This decorrelated gradient is then passed into the standard optimizer pipeline. This modification is a plug-and-play geometric transformation that can be applied to any optimizer.

### Key Designs

1. **Orthogonal Gradient Projection**:

    - **Function**: Eliminate redundant components in the current gradient that point in the same direction as the historical gradient, retaining only new information.
    - **Mechanism**: Given the current gradient $g_t$ and historical gradient $g_{t-1}$, the orthogonal component is $u_t = g_t - \text{proj}_{g_{t-1}}(g_t)$. The projection is computed using cosine distance and vector norms: $\text{proj}_{g_{t-1}}(g_t) = \frac{g_t \cdot g_{t-1}}{g_{t-1} \cdot g_{t-1}} g_{t-1}$. When data is approximately IID, $\cos(g_{t-1}, g_t) \approx 0$, so the orthogonal gradient is close to the original gradient, leaving normal training unaffected. When data is highly correlated, $\cos(g_{t-1}, g_t) \approx 1$, resulting in a very small orthogonal component, which avoids over-optimization in the same direction.
    - **Design Motivation**: Address the gradient redundancy problem directly from a geometric perspective with minimal computational overhead (requiring only vector dot products and norms) and no side effects on IID scenarios.

2. **EMA Momentum for Robust Orthogonalization**:

    - **Function**: Smooth historical gradients using exponential moving average (EMA) to reduce the interference of single-step noise on orthogonal projection.
    - **Mechanism**: Maintain an EMA of the original gradient: $c_t = \beta c_{t-1} + (1-\beta) g_t$ (default $\beta=0.9$), and then use $c_{t-1}$ instead of $g_{t-1}$ for orthogonal projection: $u_t = g_t - \text{proj}_{c_{t-1}}(g_t)$. Note that EMA is calculated on the original gradient $g_t$ rather than the orthogonal component $u_t$, so that the EMA preserves complete gradient direction information.
    - **Design Motivation**: Directly using single-step gradients for orthogonal projection is sensitive to noise. Analogous to "momentum" in standard optimizers, using EMA smoothing can robustly capture the main direction of the gradient.

3. **Multi-optimizer Adaptation (Orthogonal-SGD / Orthogonal-AdamW)**:

    - **Function**: Integrate orthogonal gradient techniques into SGD and AdamW, two mainstream optimizers.
    - **Mechanism**: Insert two lines into standard optimizer algorithms: (1) calculate the orthogonal gradient $u_t$; (2) update the EMA $c_t$. Then, replace $g_t$ with $u_t$ and feed it into subsequent momentum/second-moment estimation steps. For AdamW, the orthogonal gradient $u_t$ serves as the input to update the first moment $m_t$ and second moment $v_t$, with the rest of the pipeline remaining unchanged.
    - **Design Motivation**: The orthogonal modification is orthogonal to the specific design of the optimizer. In theory, it is applicable to any gradient-based optimizer, and has been empirically validated on the two most common optimizers: SGD and AdamW.

### Loss & Training
The orthogonal optimizer does not alter the loss function since it operates at the optimizer level. The three experimental scenarios utilize their respective original self-supervised loss functions (DINO-style teacher-student distillation loss for DoRA, masked autoencoder reconstruction loss for VideoMAE, and pixel-level prediction loss for future prediction).

## Key Experimental Results

### Main Results

**DoRA Single-Video Pre-training (WTvenice → ImageNet):**

| Initialization | Optimizer | Linear Probe Top1 | kNN Top1 |
|--------|--------|-------------------|----------|
| DINO_ImageNet | AdamW | 6.1 | 1.8 |
| DINO_ImageNet | **Orth-AdamW** | **64.5** | **51.8** |
| Random | AdamW | 3.5 | 0.8 |
| Random | **Orth-AdamW** | **8.2** | **3.1** |

**VideoMAE Multi-Video Pre-training (SSV2 → SSV2):**

| Video Processing Method | Optimizer | Linear-probe Top1 | Attn-probe Top1 |
|-------------|--------|-------------------|-----------------|
| Shuffled clips | AdamW | 19.0 | 54.9 |
| Shuffled clips | Orth-AdamW | 21.0 | 54.7 |
| Sequential (batch-along-time) | AdamW | 16.4 | 46.1 |
| Sequential (batch-along-time) | **Orth-AdamW** | **18.4** | **48.0** |
| Sequential (batch-along-video) | AdamW | 9.5 | 30.3 |
| Sequential (batch-along-video) | **Orth-AdamW** | **10.4** | **32.6** |

### Ablation Study

| Configuration | kNN Top1 (DINO init) | Description |
|------|---------------------|------|
| IID training (shuffled)| 74.4 | Upper-bound reference |
| Sequential + AdamW | 1.8 | Sequential training collapses |
| Sequential + Orth-AdamW | 51.8 | Orthogonal gradients recover performance significantly |
| Sequential + Orth-SGD | ~Lower | SGD converges slower, but still outperforms standard SGD |

### Key Findings
- **Orthogonal gradients are most effective in extreme scenarios**: With DINO initialization and sequential training, standard AdamW collapses directly (1.8%), while Orth-AdamW recovers performance to 51.8%, a gap of 50 percentage points.
- **Harmless to IID training**: When data is shuffled, the orthogonal optimizer yields performance close to standard optimizers, showing good compatibility.
- **batch-along-video is harder than batch-along-time**: Because sample frames within a batch come from adjacent frames of different videos, the temporal correlation between batches is extremely high.
- **Visualization of gradient cosine similarity** intuitively validates the effectiveness of the method: during Orthogonal Optimizer training, the cosine similarity of consecutive gradients gradually decreases from near 1 to near 0, approaching the distribution of IID training.

## Highlights & Insights
- **Minimalist yet highly effective optimizer modification**: By adding only two lines of code (orthogonal projection + EMA update) into any optimizer, it incurs negligible overhead but yields impressive results. The approach of "resolving data distribution shifts at the optimizer level" is highly elegant.
- **IID compatibility design**: In IID scenarios, the orthogonal gradient automatically degrades to the original gradient without manual switching, demonstrating a sound "no-harm" design principle.
- **Strong transferability**: This technique is not only applicable to video streaming but can also be transferred to other non-IID scenarios—such as continuous learning, online learning, and federated learning with non-uniform data distributions.

## Limitations & Future Work
- **Performance has not been fully recovered**: Although it recovers from 1.8% to 51.8% (DINO init), there is still a noticeable gap to the 74.4% of IID training, suggesting that optimizer-based decorrelation alone might be insufficient.
- **Only considers single-step history**: The orthogonal projection is computed relative to the EMA (which approximates a single-step historical direction), which might not suffice for longer-range gradient correlation patterns.
- **Strong alternatives like conjugate gradient are unexplored**: The authors mention that the conjugate gradient method might theoretically be superior, but it was not explored deeply due to higher computational costs.
- **Insufficient evaluation of computational overhead**: Although claimed to be low-cost, the specific overhead involved in maintaining an extra EMA state and performing vector projections is not quantified.

## Related Work & Insights
- **vs. OGD (Orthogonal Gradient Descent in continual learning) [Farajtabar et al.]**: While OGD performs orthogonal projection after task switching to prevent catastrophic forgetting, this work extends it to single-task streaming video, conducting orthogonal projections at every training step with EMA smoothing.
- **vs. Baby Learning [Orhan et al.]**: They study training on streaming videos but primarily focus on comparing different optimizers and replay buffers without explicitly addressing gradient correlation. This work resolves the issue directly at the gradient level, which is more fundamental.
- **vs. Replay Buffer Approaches**: Replay buffers approximate IID by storing historical data, which demands additional storage and access overhead. The Orthogonal Optimizer requires no data storage, making it more suitable for privacy-preserving scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Extends orthogonal gradients from continual learning to streaming video learning; the idea is clear and concise.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across three scenarios, multiple initializations, and various batching strategies, complemented by intuitive gradient visualization.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem is clearly defined, illustrations are intuitive, and the algorithm description is mathematically formal.
- Value: ⭐⭐⭐⭐ Highly practical optimizer improvement; holds broad reference value for online learning and stream processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [\[CVPR 2025\] Learning Temporally Consistent Video Depth from Video Diffusion Priors](learning_temporally_consistent_video_depth_from_video_diffusion_priors.md)
- [\[CVPR 2025\] 4Real-Video: Learning Generalizable Photo-Realistic 4D Video Diffusion](4real-video_learning_generalizable_photo-realistic_4d_video_diffusion.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)
- [\[ICCV 2025\] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning](../../ICCV2025/video_generation/disentangled_world_models_learning_to_transfer_semantic_knowledge_from_distracti.md)

</div>

<!-- RELATED:END -->
