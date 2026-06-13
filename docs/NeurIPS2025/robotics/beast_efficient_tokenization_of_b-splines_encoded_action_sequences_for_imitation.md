---
title: >-
  [Paper Note] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning
description: >-
  [NeurIPS 2025][Robotics][Action Tokenizer] BEAST parameterizes action sequences via B-splines—estimating control points through ridge regression and uniformly quantizing them into fixed-length tokens—achieving 20× token…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Action Tokenizer"
  - "B-Spline"
  - "Parallel Decoding"
  - "Smooth Trajectory"
  - "Efficient Inference"
date: 2026-05-08
content_hash: cced0633b4104d63
---

# BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2506.06072](https://arxiv.org/abs/2506.06072)  
**Code**: [https://intuitive-robots.github.io/beast_website/](https://intuitive-robots.github.io/beast_website/)  
**Area**: Imitation Learning / Robotics
**Keywords**: Action Tokenizer, B-Spline, Parallel Decoding, Smooth Trajectory, Efficient Inference

## TL;DR
BEAST parameterizes action sequences via B-splines—estimating control points through ridge regression and uniformly quantizing them into fixed-length tokens—achieving 20× token compression (100 steps → 5 tokens), mathematically guaranteed $C^0$ continuity across action chunks, a top-1 success rate on LIBERO-Long (86.4%), and an inference throughput of 617 Hz (2.14× faster than π₀ and 101× faster than OpenVLA).

## Background & Motivation

**Background**: Action representations in imitation learning directly affect policy quality and inference efficiency. VQ-VAE requires separately trained codebooks; FAST uses BPE to produce variable-length sequences; per-step discretization (binning) yields token counts proportional to sequence length.

**Limitations of Prior Work**: (a) VQ-VAE codebook training is decoupled from policy training, leading to potential misalignment; (b) FAST's variable-length tokens are ill-suited for parallel decoding; (c) per-step binning offers low compression—100 steps require 100 tokens; (d) no existing method guarantees smooth transitions between action chunks (requiring temporal blending as post-processing).

**Key Challenge**: Simultaneously satisfying high compression (fewer tokens = faster decoding), fixed length (parallel decoding), smooth transitions (no discontinuities), and high accuracy is fundamentally challenging.

**Goal**: Design an action tokenizer that satisfies all of the above requirements.

**Key Insight**: B-splines naturally provide a continuous and smooth representation; the number of control points is fixed (equal to the token count) and independent of the number of sampled steps; they can be fitted rapidly via ridge regression; and clamping guarantees inter-chunk continuity.

**Core Idea**: Fit B-splines to action sequences → uniformly quantize control points into fixed-length tokens → clamp the starting point to guarantee $C^0$ continuity across chunks → achieve 20× compression with mathematically guaranteed smoothness.

## Method

### Overall Architecture
Action sequence $a_{1:T}$ ($T$ steps × $D$ degrees of freedom) → **B-spline fitting** (ridge regression $\mathbf{c} = (\Phi^T\Phi + \lambda I)^{-1}\Phi^T a$, Cox–de Boor basis functions) → $N$ control points $C \in \mathbb{R}^{D \times N}$ → **uniform quantization** to [0, 255] → **interleaved flattening** into fixed-length tokens → Transformer/VLM decoder generation → **dequantization + B-spline reconstruction** to recover continuous actions.

### Key Designs

1. **B-Spline Parameterization + Ridge Regression**:

    - Function: Compactly represent $T$-step action sequences using $N$ control points.
    - Mechanism: Select $N=10$ B-spline basis functions (degree 3), compute the basis matrix $\Phi$ via Cox–de Boor recursion, and solve for control points via ridge regression. Each degree of freedom is solved independently (parallelizable).
    - Design Motivation: $N \ll T$ (e.g., 10 vs. 100) achieves 10–20× compression. The local support property of B-splines ensures that modifying one control point affects only a local portion of the trajectory.

2. **Clamped B-Splines (Inter-Chunk Continuity)**:

    - Function: Mathematically guarantee discontinuity-free transitions between consecutive action chunks.
    - Mechanism: The first control point of the current chunk is fixed to the last action value of the previous chunk $c_0$; the residual $\hat{a} = a - c_0\Phi_0^P$ is computed by subtracting the contribution of $c_0$, and the remaining control points are fitted to the residual.
    - Design Motivation: Existing methods rely on temporal blending to smooth inter-chunk transitions—this is post-processing rather than a guarantee. Clamped B-splines provide mathematical $C^0$ continuity by construction.

3. **Uniform Quantization + Interleaved Flattening**:

    - Function: Convert control points into discrete tokens processable by a Transformer.
    - Mechanism: Control point values are normalized to [0, 255] (8-bit uniform quantization) and arranged in an interleaved order by basis function, so that adjacent tokens correspond to different degrees of freedom but the same temporal segment.
    - Design Motivation: Interleaved arrangement allows the Transformer to exploit dependencies among different degrees of freedom within the same temporal segment.

### Loss & Training
- Discrete tokens: cross-entropy loss; continuous variant (BEAST-CT): ELBO.
- Compatible with multiple architectures including decoder-only Transformers + CLIP, ACT (CVAE), and Florence-2 VLM.
- Supports both autoregressive and parallel decoding modes.

## Key Experimental Results

### Main Results

| Benchmark | Method | Success Rate | Rank |
|-----------|--------|-------------|------|
| LIBERO-Long | **BEAST** | **86.4%** | **#1** |
| LIBERO-Long | π₀ | 79.6% | #2 |
| LIBERO Average | BEAST | 92.5% | π₀ 94.2% (#1) |
| CALVIN ABC→D (5 tasks) | BEAST | 74.4% | Close to VPP 75.0% |
| ALOHA Bimanual | BEAST-ACT | **70%** | ACT 49% (+21%) |
| Franka Challenge | BEAST-D | **76.57%** | π₀ 53.43% |

### Efficiency Comparison

| Method | Throughput (Hz) | Latency (s) | vs. BEAST |
|--------|----------------|-------------|-----------|
| **BEAST-F** | **617.3** | **0.019** | 1× |
| π₀ | 288.1 | 0.103 | 0.47× |
| FAST | — | — | — |
| OpenVLA | 6.1 | 0.164 | **0.01×** |

### Ablation Study

| Variant | CALVIN Avg. Length | Notes |
|---------|-------------------|-------|
| BEAST-F (N=10) | 4.43 | Optimal |
| BEAST-F (N=5) | 3.88 | Too few basis functions (−12%) |
| BEAST-F (N=15) | 4.20 | Diminishing returns |
| Binning-F | 1.41 | 68% worse (no compression) |
| BEAST-CT (continuous) | 3.88 | Slightly below discrete |

### Key Findings
- 20× B-spline compression directly translates to inference speedup—617 Hz meets real-time control requirements.
- Clamped design yields +21% success rate on ALOHA bimanual tasks—eliminating failures caused by inter-chunk discontinuities.
- Ranks #1 on LIBERO-Long (longest sequences)—B-splines are particularly effective for long-horizon tasks.
- Training convergence is also faster—80% success rate reached at 20K steps (vs. ~20% for π₀ at the same point).
- 1D toy experiment: BEAST MSE 0.0004 vs. Binning 0.0215 (50× more accurate).

## Highlights & Insights
- **B-splines are an ideal action representation**: fixed length + continuous smoothness + high compression + fast fitting—all four requirements satisfied simultaneously, with no training required.
- **Mathematical elegance of the clamped design**: fixing the first control point guarantees inter-chunk continuity—a minimal constraint that yields a critical quality guarantee.
- **101× speedup** vs. OpenVLA underscores that action representation efficiency is critical for real-time deployment.

## Limitations & Future Work
- The number of basis functions $N$ must be selected manually, depending on trajectory smoothness and sampling rate.
- May underfit abrupt motions (e.g., collision responses)—B-splines are inherently biased toward smoothness.
- Uniform quantization may be less accurate than adaptive quantization for high-precision tasks.
- Real-world success rates (52–76%) still leave room for improvement.

## Related Work & Insights
- **vs. FAST (BPE)**: Variable-length tokens are unfavorable for parallel decoding; BEAST uses fixed-length tokens.
- **vs. VQ-VAE**: Requires separately trained codebooks; BEAST requires no training (purely analytic).
- **vs. ACT**: BEAST-ACT integrates B-splines into the ACT framework, improving success rate by 21%.
- **vs. RT-2/Octo/OpenVLA**: These methods use per-step binning; BEAST achieves 4–8× compression.
- **vs. π₀**: Flow matching generates continuous actions; BEAST achieves comparable effectiveness more simply via B-splines and discrete tokens.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ B-spline action tokenizer is original and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Simulation + real robot + efficiency comparison + ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Method derivation is clear; experiments are comprehensive.
- Value: ⭐⭐⭐⭐⭐ Likely to become a standard action representation for robot imitation learning.

| CALVIN ABC-D | BEAST-D | **74.4%** (5-task success rate) | SOTA |
| CALVIN ABCD-D | BEAST-D | **84.8%** (5-task success rate) | SOTA |
| LIBERO-LONG | BEAST-F (0.77B) | Competitive | vs. π₀ (3.3B) |
| Inference Speed | BEAST | **617 Hz** | vs. OpenVLA 6.1 Hz (**101×**) |
| Inference Latency | BEAST | **19 ms** | vs. π₀ 40 ms (**2.1×**) |

### Ablation Study

| Configuration | Key Finding | Notes |
|--------------|------------|-------|
| No. of control points N | 5–10 optimal | Too many → overfitting; too few → underfitting |
| B-spline degree P | P=3 optimal | Cubic spline is the standard choice |
| Compression ratio | 4–8× vs. per-step binning | 20× fewer tokens (toy task) |
| Parallel vs. autoregressive | Comparable accuracy, significantly faster | Action-level parallelism is feasible |
| Real-world | 52.86% (Franka), 70% (ALOHA) | Successful sim-to-real transfer |
| Training efficiency | 80% at 20K steps vs. π₀'s 20% | ~4× faster convergence |

### Key Findings
- 101× inference speedup is the most prominent result—arising from parallel decoding combined with compression.
- B-spline smoothness guarantees eliminate inter-chunk action discontinuities—particularly important for high-frequency control (100+ Hz).
- No tokenizer training is required—completely avoiding the problem of VQ tokenizers needing retraining on target domains.

## Highlights & Insights
- **"The best tokenizer is one that requires no training"**: B-spline fitting is purely mathematical (ridge regression), avoiding the difficulty of joint tokenizer–policy optimization.
- **Correctly exploiting the continuity prior of actions**: Prior work treats actions as discrete sequences, discarding the smoothness prior. B-splines encode it naturally.
- **Feasibility of parallel decoding**: Text generation must be autoregressive (each word depends on prior words), but action generation need not be—inter-control-point dependencies can be handled internally by the model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[NeurIPS 2025\] Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics](massively_parallel_imitation_learning_of_mouse_forelimb_musculoskeletal_reaching.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](reinforcement_learning_with_action_chunking.md)

</div>

<!-- RELATED:END -->
