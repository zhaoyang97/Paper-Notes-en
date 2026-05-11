---
title: >-
  [Paper Note] Dense Policy: Bidirectional Autoregressive Learning of Actions
description: >-
  [ICCV 2025][Image Generation][autoregressive policy] This paper proposes Dense Policy, a robot manipulation policy based on bidirectional autoregressive expansion…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "autoregressive policy"
  - "bidirectional expansion"
  - "coarse-to-fine generation"
  - "diffusion policy"
  - "imitation learning"
date: 2026-05-08
content_hash: 078763e0163bfeea
---

# Dense Policy: Bidirectional Autoregressive Learning of Actions

**Conference**: ICCV 2025
**arXiv**: [2503.13217](https://arxiv.org/abs/2503.13217)
**Code**: [https://selen-suyue.github.io/DspNet/](https://selen-suyue.github.io/DspNet/)
**Area**: Image Generation
**Keywords**: autoregressive policy, bidirectional expansion, coarse-to-fine generation, diffusion policy, imitation learning

## TL;DR

This paper proposes Dense Policy, a robot manipulation policy based on bidirectional autoregressive expansion, which achieves hierarchical coarse-to-fine action generation in logarithmic time and surpasses mainstream generative policies such as Diffusion Policy and ACT on both simulation and real-world tasks.

## Background & Motivation

Action generation paradigms in imitation learning fall into two categories:
- **Holistic generative policies** (ACT, Diffusion Policy): model the joint distribution of action sequences and generate all actions at once. Strong performance but high inference cost.
- **Autoregressive policies** (ICRT, ARP): generate actions incrementally token-by-token or chunk-by-chunk. While effective in language and vision, they underperform in action prediction.

Challenges of autoregressive policies for action generation:
1. Next-token prediction struggles to capture long-range temporal dependencies.
2. Next-chunk prediction has a limited attention span.
3. Existing multi-scale methods (CARP) rely on discretization and codebook construction, resulting in insufficient precision.

Core observation: humans do not reason through action trajectories step by step during manipulation; instead, they **first plan a few keyframes and then progressively refine them**. This is analogous to the concept of "receptive fields" in vision — a coarse-to-fine perception process.

## Method

### Overall Architecture

Dense Policy adopts an encoder-only architecture. Starting from an initial zero vector, it recursively expands the action sequence level by level via a "Dense Process":
$$P(A|O) = \prod_{i=1}^{n} P(A^i | A^{i-1}, A^{i-2}, ..., A^0, O)$$

The sequence length doubles at each level, reaching the target horizon $T$ after $\log_2 T$ recursive steps.

### Key Designs

1. **Bidirectional Expansion**: The core mechanism is the Dense Process. Given the sparse keyframe actions $A^n$ from the previous level (containing $2^n$ action points), the sequence is upsampled to $2^{n+1}$ points via linear interpolation:

    - Existing positions retain their original values.
    - New positions are set to the linear interpolation of their two neighbors: $\tilde{a}_{t+j}^n = \frac{1}{2}(a_{t+j-T/2^{n+1}} + a_{t+j+T/2^{n+1}})$
    - Boundary positions are filled by copying the nearest original point.

   The upsampled sequence is then processed by a 4-layer BERT Encoder with cross-attention to produce the next level $A^{n+1} = \text{Enc}(A_{up}^n, O)$.

   Key distinction from next-token and next-chunk prediction: bidirectional expansion simultaneously captures temporal dependencies in both directions, yielding more coherent action sequences. Inference complexity is $O(\log T)$ rather than $O(T)$.

2. **Encoder-Only Architecture**: A shared BERT Encoder handles all levels, integrating visual observation features into action representations via cross-attention. Advantages include:

    - Fewer parameters (compared to Diffusion Head and CVAE Head).
    - Faster inference (comparable to ACT, approximately 10× faster than Diffusion Policy).
    - Stable training (no variational inference from VAE or multi-step denoising from diffusion models).

3. **Flexible Visual Input**:

    - 2D: ResNet18 + GroupNorm (RGB images).
    - 3D: Sparse Convolutional Network (point clouds).
    - Can be seamlessly replaced with other visual backbones (e.g., RISE's sparse convolutional network).
    - Partial proprioception masking during training to prevent positional memorization bias.

### Loss & Training

- L2 loss supervises the deviation between predicted actions at each level and the ground truth.
- The initial action vector is set to $A^0 = \mathbf{0}$, providing an unbiased starting point.
- Random masking of partial end-effector poses during proprioception training improves generalization.
- The same number of training iterations and expert demonstrations as baseline methods are used.

## Key Experimental Results

### Main Results (Tables)

**Simulation (11 tasks, 3 benchmarks):**

| Method | Adroit-Door | Adroit-Pen | DexArt-Laptop | DexArt-Toilet | MetaWorld-BinPick | MetaWorld-BoxClose | MetaWorld-Hammer | MetaWorld-PegInsert | MetaWorld-Disassemble | MetaWorld-ShelfPlace | MetaWorld-Reach | **Avg** |
|--------|------------|-----------|---------------|---------------|-------------------|-------------------|-----------------|--------------------|-----------------------|---------------------|----------------|---------|
| DP3 | 62±4 | 43±6 | 81±2 | 71±3 | 34±30 | 42±3 | 76±4 | 69±7 | 69±4 | 17±10 | 24±1 | 53±7 |
| **3D Dense** | **72±3** | **61±0** | **85±4** | **74±3** | **47±10** | **69±8** | **100±0** | **82±4** | **98±1** | **77±4** | **31±3** | **72±4** |
| DP | 37±2 | 13±2 | 31±4 | 26±8 | 15±4 | 30±5 | 15±6 | 34±7 | 43±7 | 11±3 | 18±2 | 25±5 |
| **2D Dense** | **59±8** | **65±1** | 28±7 | **36±8** | **25±2** | **51±3** | **86±4** | **60±7** | **71±6** | **59±6** | **27±4** | **52±5** |

3D Dense Policy achieves an average success rate **19%** higher than DP3; 2D Dense Policy surpasses Diffusion Policy by **27%**.

**Real World (4 tasks):**

| Method | Put Bread | Open Drawer | Pour Balls (Complete) | Flower Arr. (Succ) | Flower (Avg Flowers) |
|--------|----------|------------|---------------------|-------------------|---------------------|
| ACT | 35% | 10% | 20% | - | - |
| Diffusion Policy | 40% | 20% | 20% | - | - |
| RISE | 75% | 40% | 25% | 50% | 0.6/3.0 |
| **2D Dense** | **55%** | **20%** | **25%** | - | - |
| **3D Dense** | **85%** | **45%** | **60%** | **70%** | **1.0/3.0** |

### Ablation Study (Tables)

**Comparison of autoregressive paradigms (learning efficiency and final performance):**

| Paradigm | Door | Bin Picking | Shelf Place | Box Close |
|----------|------|------------|-------------|-----------|
| Next-Token | Lower | Lower | Lower | Lower |
| Next-Chunk | Medium | Medium | Medium | Medium |
| **Bidirectional (Dense)** | **Highest** | **Highest** | **Highest** | **Highest** |

Bidirectional prediction demonstrates higher learning efficiency and a higher performance ceiling across all four challenging tasks.

**Inference time and parameter count comparison:**

| Policy | Parameters (Action Head) | Inference Time |
|--------|--------------------------|----------------|
| ACT | Larger | ~Same as Dense |
| DP (Diffusion) | Dense + 9.19M | ~10× Dense |
| **Dense Policy** | **Smallest** | **Fastest** |

### Key Findings

- **Bidirectional dependency is critical**: temporal steps in action sequences exhibit bidirectional dependencies that next-token prediction cannot capture.
- **Greater advantage on long-horizon tasks**: the relative advantage of Dense Policy is most pronounced in Flower Arrangement (long-horizon, multi-object), achieving 70% vs. 50% success rate.
- **Improved action precision**: strong performance on Peg Insert Side (contact-rich) and Pen Rotation (high-DoF dexterous manipulation).
- **More stable training**: avoids the training instability of ACT by eliminating variational inference and multi-step denoising.
- **Logarithmic-time inference**: $O(\log T)$ complexity, significantly faster than next-token's $O(T)$.

## Highlights & Insights

- **Coarse-to-fine action generation paradigm**: analogous to how humans plan keyframes before refining details, offering a novel perspective on action sequence modeling.
- **First application of bidirectional autoregression to action space**: challenges the assumption that autoregressive = unidirectional, demonstrating the effectiveness of bidirectional expansion in continuous action spaces.
- **Efficiency of the encoder-only architecture**: a shared encoder handles multi-level action representations, substantially reducing parameter count.
- **Logarithmic-complexity inference**: achieves extremely fast inference without sacrificing performance.

## Limitations & Future Work

- Extension of Dense Policy to a general-purpose VLA (Vision-Language-Action) model remains unexplored.
- Scalability and stability on large-scale foundation models have not been validated.
- 2D Dense Policy is still limited by its 2D representation on tasks requiring complex spatial reasoning.
- Occasional over-gripping issues arise in the ball-pouring task; adaptive error-correction capability warrants improvement.
- The horizon $T$ must be a power of two, limiting flexibility.

## Related Work & Insights

- BERT's bidirectional contextual learning → bidirectional action expansion in Dense Policy.
- Non-raster-order autoregressive image generation in MAR/SAR → bidirectional generation in action space.
- Complements rather than replaces Diffusion Policy and ACT, highlighting the untapped potential of autoregressive policies.
- Inspires the application of similar hierarchical expansion strategies to trajectory planning and motion generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Bidirectional autoregressive action generation is a genuinely new paradigm; the coarse-to-fine hierarchical design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 11 tasks across 3 simulation benchmarks plus 4 real-world tasks, with full 2D/3D coverage and thorough ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation is clear with well-articulated differentiation from prior work.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new direction for autoregressive policies in robotic manipulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping](inpaint4drag_repurposing_inpainting_models_for_drag-based_image_editing_via_bidi.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[AAAI 2026\] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](../../AAAI2026/image_generation/mp1_meanflow_tames_policy_learning_in_1-step_for_robotic_manipulation.md)
- [\[AAAI 2026\] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI](../../AAAI2026/image_generation/efficientflow_efficient_equivariant_flow_policy_learning_for_embodied_ai.md)

</div>

<!-- RELATED:END -->
