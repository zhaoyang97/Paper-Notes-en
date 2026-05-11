---
title: >-
  [Paper Note] Anchoring and Rescaling Attention for Semantically Coherent Inbetweening
description: >-
  [CVPR 2026][LLM Evaluation][Generative inbetweening] This paper proposes KAB (Keyframe-Anchored Attention Bias) and ReTRo (Rescaled Temporal RoPE)…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "Generative inbetweening"
  - "attention anchoring"
  - "temporal RoPE rescaling"
  - "keyframe guidance"
  - "video diffusion models"
date: 2026-05-08
content_hash: 52640e8cf7e24530
---

# Anchoring and Rescaling Attention for Semantically Coherent Inbetweening

**Conference**: CVPR 2026
**arXiv**: [2603.17651](https://arxiv.org/abs/2603.17651)
**Code**: To be confirmed
**Area**: LLM Evaluation
**Keywords**: Generative inbetweening, attention anchoring, temporal RoPE rescaling, keyframe guidance, video diffusion models

## TL;DR

This paper proposes KAB (Keyframe-Anchored Attention Bias) and ReTRo (Rescaled Temporal RoPE), two training-free inference-time methods built upon the Wan2.1 video diffusion model. These methods address semantic infidelity, frame inconsistency, and temporal rhythm instability in generative inbetweening (GI) with sparse keyframes under large-motion conditions. The paper also introduces TGI-Bench, the first text-conditioned GI evaluation benchmark.

## Background & Motivation

Generative Inbetweening (GI) refers to the task of generating intermediate transition frames given a pair of start and end keyframes. Unlike traditional optical-flow-based frame interpolation, GI must "imagine" the intermediate process, which presents three core challenges in large-motion, long-horizon scenarios:

**Semantic Infidelity**: Intermediate frames contain objects or scene elements inconsistent with the keyframes.

**Frame Inconsistency**: Flickering or abrupt changes occur between adjacent frames.

**Temporal Rhythm Instability**: Motion speed is uneven, resulting in an unnatural temporal distribution.

Most existing methods adapt Image-to-Video (I2V) models—representative examples include TRF and SEINE—but their quality degrades sharply as the keyframe interval increases (e.g., 65 or 81 frames). The root causes are:

- Cross-attention mechanisms dilute focus on both endpoint keyframes in long sequences.
- Positional encodings in temporal attention do not account for the anchoring role of start and end frames.
- No unified evaluation benchmark exists for assessing the quality of text-conditioned GI.

The paper's starting point is to address these issues **without modifying model weights**, relying solely on attention manipulation at inference time.

## Method

### Overall Architecture

Built upon **Wan2.1** (a DiT-based First-Last-Frame-to-Video model), two complementary modules are introduced at inference time:

- **KAB**: Manipulates the logit distribution of cross-attention to inject semantic anchors from keyframes into intermediate frames.
- **ReTRo**: Adjusts the scaling coefficients of RoPE in temporal self-attention, treating edge frames and intermediate frames differently.

Both modules require **no additional training** and intervene directly during the denoising process.

### Key Designs

#### 1. KAB (Keyframe-Anchored Attention Bias)

Core Idea: Extract **semantic anchors** from the cross-attention maps of keyframes and guide the attention distribution of intermediate frames via logit biases.

**Step 1: Extract keyframe anchors**

For the first frame $I_{\text{first}}$ and last frame $I_{\text{last}}$, retrieve their attention distributions $A_{\text{first}}$ and $A_{\text{last}}$ from the cross-attention layers as keyframe anchors.

**Step 2: Generate per-frame target anchors via linear interpolation**

For frame $t$, interpolate linearly according to temporal position:

$$\bar{A}(t) = \frac{T - t}{T} \cdot A_{\text{first}} + \frac{t}{T} \cdot A_{\text{last}}$$

**Step 3: Compute and apply logit bias**

Define the attention bias as:

$$B(t) = \log(M(t) + \varepsilon) - \log(\bar{A}(t) + \varepsilon)$$

where $M(t)$ is the desired target mask and $\varepsilon$ prevents numerical overflow. This bias is added to the cross-attention logits before softmax, steering attention focus **without altering model parameters**.

**Triple Isolated Cross-Attention**:

To prevent information interference among the start frame, end frame, and text condition, the cross-attention for each is computed in complete isolation:

- Cross-attention for $I_{\text{first}}$ is computed independently.
- Cross-attention for $I_{\text{last}}$ is computed independently.
- Cross-attention for the text prompt is computed independently.

The three outputs are then fused via weighted combination, ensuring symmetric treatment of both endpoint keyframes.

#### 2. ReTRo (Rescaled Temporal RoPE)

The RoPE positional encoding in temporal self-attention governs the attention decay pattern across frames. ReTRo applies different scaling coefficients to frames at different positions:

- **Edge frames** (near the start/end keyframes): use $s_{\text{edge}} > 1$
    - Amplifying positional encoding frequency → sharpening local attention → better preserving keyframe details.
    - Intuition: frames close to keyframes should "resemble" the keyframes more closely.

- **Intermediate frames**: use $s_{\text{mid}} < 1$
    - Reducing positional encoding frequency → expanding the receptive field → promoting inter-frame consistency.
    - Intuition: frames far from keyframes need to "look further" to maintain coherence.

This non-uniform scaling produces a "U-shaped" distribution along the temporal axis—tight at both ends, loose in the middle—elegantly balancing keyframe fidelity with smooth intermediate transitions.

### Loss & Training

This method is **completely training-free**; all operations are performed at inference time:

- KAB only modifies cross-attention logits (by adding a bias).
- ReTRo only modifies the RoPE scaling coefficients.
- No additional parameters are introduced; no backpropagation is required.
- Computational overhead: limited to keyframe anchor extraction and bias computation, negligible relative to total inference time.

## Key Experimental Results

### TGI-Bench (New Benchmark)

The first text-conditioned generative inbetweening evaluation benchmark:

| Dimension | Scale |
|-----------|-------|
| Number of videos | 220 |
| Sequence length | 25 / 33 / 65 / 81 frames |
| Challenge categories | 4 (large motion / occlusion / appearance change / scene transition) |
| Evaluation metrics | PSNR, SSIM, FVD, VBench |

### Main Results

**Long-sequence (65/81 frames) performance comparison**:

| Method | Training Required | PSNR↑ | SSIM↑ | FVD↓ | VBench↑ |
|--------|------------------|-------|-------|------|---------|
| TRF | Yes | Medium | Medium | Medium | Medium |
| SEINE | Yes | Medium | Medium | Medium | Medium |
| Wan2.1 (baseline) | — | Medium | Medium | Medium | Medium |
| **KAB + ReTRo** | **No** | **Best** | **Best** | **Best** | **Best** |

Key observation: Performance gaps are modest on short sequences (25 frames), but as sequence length increases to 65/81 frames, the advantage of KAB+ReTRo becomes substantially more pronounced.

### Ablation Study

| Configuration | PSNR | SSIM | Notes |
|---------------|------|------|-------|
| Baseline (Wan2.1) | Base | Base | No intervention |
| + KAB only | ↑ | ↑ | Improved semantic consistency |
| + ReTRo only | ↑ | ↑ | Improved temporal stability |
| + KAB + ReTRo | ↑↑ | ↑↑ | Complementary; best overall |
| KAB w/o Triple Isolation | ↓ | ↓ | Degradation due to start/end frame interference |
| ReTRo uniform scaling ($s=1$) | → Base | → Base | Equivalent to no rescaling |
| $s_{\text{edge}}$ too large | ↓ | ↑ | Over-sharpening; loses smoothness |
| $s_{\text{mid}}$ too small | ↓ | ↓ | Receptive field too large; details blurred |

### Key Findings

1. **KAB and ReTRo address distinct problems**: KAB targets semantic fidelity; ReTRo targets temporal consistency; their combination yields the best results.
2. **Long-sequence advantage is pronounced**: The larger the gain as sequence length grows (65/81 frames), the more clearly the method addresses long-range dependency issues.
3. **Triple Isolation is indispensable**: Without isolating start/end frame attention, cross-contamination biases intermediate frames toward one endpoint.
4. **The U-shaped distribution of ReTRo is critical**: Uniform scaling has no effect; the edge-tight, middle-loose pattern is essential.

## Highlights & Insights

- The **training-free** design is highly practical: no paired data collection, no fine-tuning, plug-and-play deployment.
- KAB's logit bias approach is conceptually analogous to Classifier-Free Guidance, but operates in the spatial dimension (attention maps) rather than the class dimension.
- ReTRo's non-uniform RoPE scaling is a novel design that generalizes to other tasks requiring differentiated temporal modeling.
- The symmetric design of Triple Isolated Cross-Attention reflects careful consideration of equitable treatment of both endpoint keyframes.
- TGI-Bench fills the gap in text-conditioned GI evaluation; its design covering 4 challenge categories × 4 sequence lengths is scientifically comprehensive.
- The method offers strong interpretability: the physical meaning of each component is clear, and ablation experiments validate the independent contribution of each part.

## Limitations & Future Work

1. **Dependency on the Wan2.1 architecture**: KAB and ReTRo are tightly coupled with DiT + RoPE; adaptation to U-Net architectures requires non-trivial modification.
2. **Linear interpolation assumption**: The linear interpolation of target anchors assumes uniform motion; it may be suboptimal for nonlinear motion (acceleration/deceleration).
3. **Hyperparameter sensitivity**: $s_{\text{edge}}$ and $s_{\text{mid}}$ require manual tuning; no adaptive selection mechanism is provided.
4. **Computational cost not analyzed in detail**: Although overhead is claimed to be negligible, no concrete inference time comparisons are reported.
5. **Restricted to frame interpolation**: The method targets scenarios where both endpoint frames are known and cannot be directly extended to single-frame extrapolation or unconditional generation.
6. **Evaluation metric limitations**: PSNR/SSIM emphasize pixel-level fidelity and have limited coverage of perceptual quality; VBench offers broader coverage but lacks fine granularity.

## Related Work & Insights

- **vs. Wan2.1 (baseline)**: This paper builds directly on Wan2.1 FLF2V, manipulating attention without modifying weights; it can be viewed as an inference-time enhancement plugin.
- **vs. TRF / SEINE**: Prior inbetweening methods require training and degrade significantly on long sequences; KAB+ReTRo requires no training and exhibits greater advantages as sequence length increases.
- **vs. Classifier-Free Guidance**: CFG steers generation in the class dimension; KAB steers semantic focus in the spatial dimension (attention maps), serving as an attention-level analogue.
- **ReTRo's non-uniform RoPE scaling** generalizes to other tasks requiring differentiated temporal modeling (e.g., keyframe enhancement in long video understanding).
- Inference-time attention manipulation is a low-cost yet effective means of enhancing model capabilities, worthy of further exploration in video editing, video completion, and related tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The KAB + ReTRo combination is novel; the training-free design philosophy is distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ TGI-Bench as a new benchmark + comprehensive evaluation across 4 sequence lengths × 4 challenge categories.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play without training; the video generation community can benefit directly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](../../ICLR2026/llm_evaluation/spectral_attention_steering_for_prompt_highlighting.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](../../ICLR2026/llm_evaluation/breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)
- [\[NeurIPS 2025\] PaTH Attention: Position Encoding via Accumulating Householder Transformations](../../NeurIPS2025/llm_evaluation/path_attention_position_encoding_via_accumulating_householder_transformations.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/llm_evaluation/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[CVPR 2026\] SparseCam4D: Spatio-Temporally Consistent 4D Reconstruction from Sparse Cameras](sparsecam4d_spatio-temporally_consistent_4d_reconstruction_from_sparse_cameras.md)

</div>

<!-- RELATED:END -->
