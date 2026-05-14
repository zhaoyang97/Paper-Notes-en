---
title: >-
  [Paper Note] LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation
description: >-
  [CVPR2026][Video Generation][linear attention] This paper proposes LinVideo, a data-free post-training framework that selectively replaces quadratic attention with linear attention in video diffusion models…
tags:
  - "CVPR2026"
  - "Video Generation"
  - "linear attention"
  - "video diffusion"
  - "post-training"
  - "efficient inference"
  - "distribution matching"
date: 2026-05-08
content_hash: 3cc00d5dde139a42
---

# LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation

**Conference**: CVPR2026
**arXiv**: [2510.08318](https://arxiv.org/abs/2510.08318)
**Code**: None
**Area**: Video Generation
**Keywords**: linear attention, video diffusion, post-training, efficient inference, distribution matching

## TL;DR

This paper proposes LinVideo, a data-free post-training framework that selectively replaces quadratic attention with linear attention in video diffusion models, achieving 1.43–1.71× speedup. Combined with distillation, the speedup reaches 15.9–20.9× while maintaining generation quality.

## Background & Motivation

Video diffusion models (e.g., Wan, CogVideoX, Sora) have achieved remarkable generation quality, but their self-attention has $\mathcal{O}(n^2)$ computational complexity. When video sequence lengths $n$ are large (10-second videos often exceed 50K tokens), inference cost becomes a deployment bottleneck.

Existing acceleration approaches fall into two categories:

**Attention sparsification** (SVG, XAttention, etc.): Skips redundant computations, but at moderate sequence lengths it is difficult to achieve high sparsity, and in practice still retains >50% of dense attention computation.

**Linear attention** (SANA-Video, LinGen, etc.): Reduces complexity to $\mathcal{O}(n)$, but full replacement requires costly pretraining from scratch.

The root cause lies in the significant representation gap between linear attention and softmax attention, compounded by the complexity of spatiotemporal modeling in video generation, making cheap post-training difficult to apply effectively. The core problem this paper addresses is: **Can efficient post-training replace as many quadratic attention layers as possible with linear attention, achieving significant speedup without sacrificing quality?**

## Method

### Overall Architecture

LinVideo is a **data-free post-training framework** consisting of three stages:

1. **Data preparation**: Samples from the pretrained model itself, collecting 50K input-output pairs $(x_t, u_t)$ as training data — no external video datasets required.
2. **Selective Transfer**: Automatically selects which layers to replace with linear attention via learnable parameters.
3. **Arbitrary-timestep Distribution Matching (ADM)**: An optimization objective that aligns the distribution of the linearized model with the original model at every timestep along the sampling trajectory.

### Key Design 1: Selective Transfer

A key observation motivates this design: **the replaceability of different layers varies substantially**.

- Shallow layers (e.g., layers 2–11) recover accuracy more easily after replacement, possibly because subsequent layers can compensate for early-layer errors.
- Certain specific layers (e.g., layer 1) suffer irreversible performance degradation upon replacement.

Based on this, layer selection is formulated as a **binary classification problem**. A learnable scalar $r \in [0,1]$ is introduced per layer, with a mixed attention formulation:

$$o_i = r \cdot \text{SoftmaxAttn}(q_i, K, V) + (1-r) \cdot \text{LinearAttn}(q_i, K, V)$$

$r=1$ retains softmax attention; $r=0$ uses linear attention. After training, values are rounded to determine the final selection.

To ensure the target number of replacements, a **constraint loss** is designed:

$$\mathcal{L}_{\text{con}} = \left(\sum_{l=1}^{N} \lceil r^{(l)} \rfloor - \text{target}\right)^2$$

To prevent $r$ from oscillating near 0.5 and causing rounding errors, a **regularization loss** is designed:

$$\mathcal{L}_{\text{reg}} = \sum_{l=1}^{N} (1 - |2r^{(l)} - 1|^\alpha)$$

where $\alpha$ is annealed from large to small, allowing free exploration early on and forcing $r$ toward 0 or 1 in later stages. Ablations show that removing $\mathcal{L}_{\text{reg}}$ causes catastrophic performance collapse.

The Hedgehog kernel function is adopted for linear attention:

$$\phi(q) = \text{softmax}(q\widetilde{W}_q) \oplus \text{softmax}(-q\widetilde{W}_q)$$

### Key Design 2: Arbitrary-timestep Distribution Matching (ADM)

A naive MSE loss ($\mathcal{L}_{\text{mse}} = \|u_t - \hat{u}_\theta(x_t, t)\|^2$) introduces temporal artifacts (flickering, jitter), as it does not preserve the joint distribution across frames and harms generalization.

Existing distribution matching for few-step distillation (e.g., DMD) only matches the distribution at $t=0$, ignoring intermediate timesteps, which performs poorly in this setting. Moreover, it requires training an additional model to estimate the score function, incurring significant overhead (5–10× training cost).

The core idea of ADM is to **match distributions at arbitrary timesteps $t$ along the sampling trajectory**, minimizing the KL divergence between the linearized model distribution $q_t$ and the original model distribution $p_t$:

$$\mathcal{L}_{\text{ADM}} = \mathbb{E}_{\hat{x}_t \sim q_t}\left[\log \frac{q_t(\hat{x}_t)}{p_t(\hat{x}_t)}\right]$$

A key advantage: since LinVideo progressively transitions from softmax to linear attention, $\hat{u}_\theta$ can always be viewed as a flow model, enabling **self-estimation of the score function** $\hat{s}_t$ without training an additional model. The simplified score difference is:

$$s_t(\hat{x}_t) - \hat{s}_t(\hat{x}_t) = -\frac{1-t}{t}(u_\theta(\hat{x}_t) - \hat{u}_\theta(\hat{x}_t))$$

### Loss & Training

Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ADM}} + \lambda(\mathcal{L}_{\text{con}} + \mathcal{L}_{\text{reg}})$, with $\lambda = 0.01$.

- Wan 1.3B: 16/30 layers replaced; trained for 3K steps on 8×H100.
- Wan 14B: 22/40 layers replaced; trained for 3K steps on 32×H100.
- Optional: additional DMD2 distillation for 2K steps to enable 4-step generation.

## Key Experimental Results

### Main Results: VBench 8-Metric Comparison (Wan 1.3B, 480p)

| Method | Latency (s) | Speedup | Imaging Quality | Aesthetic Quality | Motion Smooth. | Dynamic Degree | BG Consist. | Subject Consist. |
|--------|-------------|---------|-----------------|-------------------|----------------|----------------|-------------|-----------------|
| FlashAttention2 | 97.32 | 1.00× | 66.25 | 59.49 | 98.42 | 59.72 | 96.57 | 95.28 |
| SVG | 74.52 | 1.31× | 65.78 | 59.16 | 97.32 | 58.87 | 95.79 | 93.94 |
| SVG2 | 84.91 | 1.15× | 66.03 | 59.31 | 98.07 | 59.44 | 96.61 | 94.95 |
| **LinVideo** | **68.26** | **1.43×** | **66.07** | **59.41** | **98.19** | **59.67** | **96.72** | **95.12** |
| LinVideo+DMD2 | 6.11 | **15.9×** | 65.62 | 57.74 | 97.32 | 61.26 | 95.47 | 93.74 |

On Wan 14B (720p), LinVideo achieves **1.71×** speedup (1127s vs. 1931s); combined with DMD2, it reaches **20.9×** speedup. VBench-2.0 overall score: LinVideo (56.74) = FA2 (56.74) > SVG2 (55.81).

### Ablation Study

| Ablation Dimension | Key Findings |
|-------------------|--------------|
| Number of targets | Speedup increases and quality decreases as target grows from 10 to 20; performance is stable at target ≤ 18, degrades significantly at ≥ 20 |
| Selection strategy | LinVideo (automatic) >> Manual (same-layer manual assignment) >> Heuristic (grid search) |
| $\mathcal{L}_{\text{reg}}$ | Removing it causes Imaging Quality to collapse from 66.07 to 18.62, confirming the indispensability of $r$ regularization |
| ADM vs. MSE | ADM (66.07) >> MSE (61.56) >> DMD (57.44); MSE introduces temporal artifacts |
| ADM self-estimated score | Self-estimated $\hat{s}_t$ (66.07) outperforms training an additional model (65.61), while being ~4.4× faster |
| $\lambda$ sensitivity | Performance varies by ~1% across $\lambda \in \{0.001, 0.01, 0.1\}$; not sensitive |

### Layer Selection Results

Automatically selected layers for replacement: $\{2\text{–}8, 10\text{–}13, 15\text{–}16, 23, 25, 30\}$, concentrated in shallow layers, consistent with the observation that shallow layers are more easily replaced.

## Highlights & Insights

1. **Data-free post-training paradigm**: Requires no external video data; training relies solely on the model's own sampled input-output pairs, avoiding data privacy and copyright concerns.
2. **Automated layer selection**: Formulating layer selection as a learnable binary classification problem offers a fundamental advantage over manual or heuristic approaches (Imaging Quality: 66.07 vs. 62.97 vs. 60.74).
3. **ADM training efficiency**: By using the model itself to estimate the score function, the need for an additional model is eliminated, improving training efficiency by ~4.4×.
4. **Orthogonal design**: LinVideo only changes the attention type (dense linear vs. dense quadratic), making it orthogonal to sparse attention methods and enabling future combination.
5. **Extreme speedup potential**: The 4-step distillation variant achieves 15.9–20.9× speedup with only ~1% quality loss, demonstrating practical deployment value.

## Limitations & Future Work

1. **No specialized kernels used**: The current linear attention implementation does not use custom CUDA kernels, leaving room for further speedup gains.
2. **Replacement ceiling**: Quality degrades significantly when target exceeds 18/30 layers, indicating that softmax attention in certain layers is irreplaceable.
3. **Validated only on the Wan series**: Generalizability to other architectures such as CogVideoX and HunyuanVideo has not been verified.
4. **Non-trivial training resources**: The 1.3B model requires 8×H100 for 3K steps; the 14B model requires 32×H100, posing a barrier for smaller research groups.
5. **Combination with sparse methods**: The authors note that LinVideo and methods such as SVG are orthogonal; combining them is a promising direction but has not yet been realized.

## Related Work & Insights

- **Linear attention pretraining**: SANA-Video, LinGen, Matten, and similar works require costly pretraining starting from image models; LinVideo provides a post-training alternative.
- **SLA** (concurrent work): Applies intra-layer mixed attention (mixing within a layer), whereas LinVideo performs inter-layer replacement (replacing entire layers); the two approaches are composable.
- **Few-step distillation**: DMD/DMD2 is used for final acceleration, but directly applying distillation on linear attention models fails catastrophically — LinVideo must precede distillation.
- **Hedgehog kernel**: The selected kernel design ensures non-negativity via softmax transformation.
- Inspiration: A similar progressive linearization + distribution matching strategy may prove effective for diffusion models in other modalities (e.g., audio, 3D).

## Rating

- Novelty: ⭐⭐⭐⭐ — Both core designs (Selective Transfer and ADM) are original; formulating layer selection as a binary classification problem is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Two model scales, two benchmarks, and comprehensive ablations covering target count, selection strategy, loss function, regularization, and training efficiency.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with a coherent motivation–method–experiment narrative.
- Value: ⭐⭐⭐⭐ — Provides a practical acceleration solution for video generation; introducing linear attention into video diffusion models via post-training is a meaningful research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[NeurIPS 2025\] Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](../../NeurIPS2025/video_generation/autoregressive_adversarial_posttraining_for_realtime_interac.md)
- [\[NeurIPS 2025\] VORTA: Efficient Video Diffusion via Routing Sparse Attention](../../NeurIPS2025/video_generation/vorta_efficient_video_diffusion_via_routing_sparse_attention.md)
- [\[CVPR 2026\] UniTalking: A Unified Audio-Video Framework for Talking Portrait Generation](unitalking_a_unified_audio-video_framework_for_talking_portrait_generation.md)

</div>

<!-- RELATED:END -->
