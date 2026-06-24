---
title: >-
  [Paper Note] One-Minute Video Generation with Test-Time Training
description: >-
  [CVPR 2025][Video Generation][Test-Time Training] This paper introduces Test-Time Training (TTT) layers into a pretrained Diffusion Transformer. By capitalizing on the high expressiveness of TTT layers, which employ neural networks as hidden states, the proposed method achieves the capability of generating coherent one-minute long videos from text storyboards, outperforming baselines like Mamba 2 and Gated DeltaNet by 34 Elo points in human evaluations.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Test-Time Training"
  - "Long Video"
  - "Diffusion Transformer"
  - "RNN layer"
date: 2026-05-08
content_hash: 79b26e8d88d6f302
---

# One-Minute Video Generation with Test-Time Training

**Conference**: CVPR 2025  
**arXiv**: [2504.05298](https://arxiv.org/abs/2504.05298)  
**Code**: [https://test-time-training.github.io/video-dit](https://test-time-training.github.io/video-dit)  
**Area**: Video Generation  
**Keywords**: Video Generation, Test-Time Training, Long Video, Diffusion Transformer, RNN layer

## TL;DR

This paper introduces Test-Time Training (TTT) layers into a pretrained Diffusion Transformer. By capitalizing on the high expressiveness of TTT layers, which employ neural networks as hidden states, the proposed method achieves the capability of generating coherent one-minute long videos from text storyboards, outperforming baselines like Mamba 2 and Gated DeltaNet by 34 Elo points in human evaluations.

## Background & Motivation

**Background**: Current state-of-the-art video generation models (e.g., Sora 20s, MovieGen 16s, Veo 2 8s) remain restricted to short clips and struggle to autonomously generate long videos with complex multi-scene narratives. The core bottleneck lies in the quadratic complexity of self-attention in Transformers relative to context length.

**Limitations of Prior Work**: Modern RNN layers represented by Mamba and DeltaNet exhibit linear complexity, but their hidden states are merely matrices (linear hidden states), limiting their expressiveness. Compressing hundreds of thousands of tokens into a rank-deficient matrix is extremely difficult, making it challenging for these RNN layers to memorize deep dependencies among distant tokens. As a result, the generated long videos lack complex narratives and dynamic motions.

**Key Challenge**: The contradiction between long-context requirements and computational efficiency—self-attention is highly expressive but computationally prohibitive, while linear RNNs are computationally efficient but lack sufficient expressiveness.

**Goal**: To discover an RNN layer with a more expressive hidden state while maintaining linear complexity, enabling a pretrained Diffusion Transformer to generate one-minute multi-scene complex narrative videos.

**Key Insight**: The authors observe that self-supervised learning can compress large-scale training sets into model weights. Thus, they propose representing the hidden state of an RNN as a neural network (a two-layer MLP) and updating these neural network weights on test sequences via gradient descent to compress historical context.

**Core Idea**: To replace traditional linear RNN layers with TTT layers (where the hidden state itself is a trainable neural network) to achieve stronger long-context memorization capabilities.

## Method

### Overall Architecture

Starting from a pretrained CogVideo-X 5B (which can only generate 3-second videos), the system inserts TTT layers with learnable gating after each attention layer. The input is a text storyboard (Format 3), decomposed into multiple text-video token pairs of 3-second segments. Self-attention layers are restricted to modeling local attention within each 3-second segment, whereas the TTT layers process the entire sequence globally to model long-range dependencies across segments. A one-minute video corresponds to over 300k tokens.

### Key Designs

1. **TTT-MLP Layer**:

    - **Function**: Acts as a new type of RNN layer, using a two-layer MLP as the hidden state to compress historical context.
    - **Mechanism**: The hidden state $W$ is itself a two-layer MLP (with a hidden dimension 4 times the input dimension and a GELU activation), which updates its weights on each input token via gradient descent using a self-supervised loss:
    
    $$\ell(W;x_t) = \|f(\theta_K x_t; W) - \theta_V x_t\|^2$$
    
    The output token is computed as $z_t = f(\theta_Q x_t; W_t)$, where $\theta_K, \theta_V, \theta_Q$ are analogous to the Key, Value, and Query matrices in self-attention, learned in the outer loop.
    - **Design Motivation**: Linear hidden states (matrices) have a limited rank and cannot effectively compress sequences exceeding 300k tokens. An MLP hidden state offers non-linearity and larger capacity, significantly outperforming the linear matrix hidden states of methods like Mamba.

2. **Gated + Bidirectional Mechanism**:

    - **Function**: Smoothly integrates randomly initialized TTT layers into the pretrained model and supports non-causal generation.
    - **Mechanism**: A learnable gating vector $\alpha$ is used to control the contribution of the TTT layer's output to the original features:
    
    $$\text{gate}(\text{TTT}, X; \alpha) = \tanh(\alpha) \otimes \text{TTT}(X) + X$$
    
    with $\alpha$ initialized to $0.1$ such that $\tanh(\alpha) \approx 0.1$. The bidirectional mechanism processes the sequence using both forward and backward TTT modules, which share kernel parameters but use different gating parameters.
    - **Design Motivation**: Direct insertion of randomly initialized layers severely disrupts the pretrained model; the gating mechanism ensures that the TTT layer contributes minimally during the initial stage. Additionally, diffusion models are non-causal and require bidirectional processing.

3. **On-Chip Tensor Parallel (On-Chip Tensor Parallel)**:

    - **Function**: Resolves the issue where the hidden state of TTT-MLP is too large to fit into the SMEM (Shared Memory) of a single SM (Streaming Multiprocessor).
    - **Mechanism**: Shards the two-layer weights $W^{(1)}, W^{(2)}$ of the MLP across multiple SMs, utilizing the DSMEM (Distributed Shared Memory) feature of NVIDIA Hopper GPUs for AllReduce operations among SMs. The entire hidden state update is completed on-chip, reading and writing to HBM (High Bandwidth Memory) only during initial loading and final output.
    - **Design Motivation**: The hidden state of TTT-MLP is much larger than that of Mamba, making it impossible to directly use a single-SM fused kernel like FlashAttention.

### Loss & Training

A five-stage context-extension strategy is adopted. Stage 1 fine-tunes the entire model on 3-second clips (with a higher learning rate for TTT layers). Stages 2-5 progressively fine-tune on 9/18/30/63-second videos, while freezing most parameters and only training the TTT layers, gating, and self-attention layers to preserve pretrained knowledge. The dataset is based on approximately 7 hours of "Tom and Jerry" animations, manually annotated with storyboards, and enhanced to 720×480 resolution using a video super-resolution model.

## Key Experimental Results

### Main Results

| Evaluation Metric | Mamba 2 | Gated DeltaNet | Sliding Window | TTT-MLP | TTT-MLP Gain |
|---|---|---|---|---|---|
| Text Following | 985 | 983 | 1016 | 1014 | - |
| Motion Naturalness | 976 | 984 | 1000 | 1039 | +39 vs 2nd |
| Aesthetics | 963 | 993 | 1006 | 1037 | +31 vs 2nd |
| Temporal Consistency | 988 | 1004 | 975 | 1042 | +38 vs 2nd |
| **Average** | 978 | 991 | 999 | **1033** | **+34 vs 2nd** |

### Ablation Study

| Configuration | 63s Avg Elo | 18s Avg Elo | Description |
|---|---|---|---|
| TTT-MLP | 1033 | 977 | Best on long videos, second best on short videos |
| Gated DeltaNet | 991 | 1005 | Best on short videos, second on long videos |
| Mamba 2 | 978 | 978 | Mediocre overall performance |
| TTT-Linear | Eliminated | Lower than TTT-MLP | Linear hidden state is insufficient |
| Local Attention | Eliminated | Worst | No cross-segment modeling capability |

### Key Findings

- TTT-MLP outperforms the runner-up by 34 Elo points on 63-second long videos, but underperforms Gated DeltaNet by 28 Elo points on 18-second short videos (~100k tokens), indicating that the advantage of non-linear hidden states only manifests in longer contexts.
- TTT-MLP inference is 1.4x slower than Gated DeltaNet, and its training is 2.1x slower. However, this is still a massive improvement compared to the 11x inference overhead of global attention.
- Video artifacts (unnatural motion, object morphing) remain a common issue across all methods, which likely stems from the inherent limitations of the base pretrained model, CogVideo-X 5B.

## Highlights & Insights

- **The insight of TTT layers is profound**—framing self-supervised learning as an RNN state compression mechanism transforms the observation "learned knowledge = compressed data" into a trainable sequence modeling layer, establishing an elegant theoretical framework.
- **The hybrid architecture of local attention + global TTT** achieves efficient long-context processing. This hierarchical strategy of "local-fine + global-coarse" can be transferred to other long-sequence tasks.
- **On-Chip Tensor Parallelism** is a clever system-level design. It applies the concept of cross-GPU tensor parallelism to cross-SM parallelism within a single GPU, opening up a new paradigm for the efficient implementation of large hidden state RNNs.

## Limitations & Future Work

- Evaluated only within the "Tom and Jerry" animation domain; not tested on real-world videos, leaving generalization capabilities questionable.
- Inference efficiency is still inferior to Mamba/DeltaNet, leaving significant room for kernel optimization (e.g., register spilling, asynchronous instruction scheduling).
- TTT-MLP underperforms Gated DeltaNet in short contexts (18 seconds / 100k tokens), suggesting that non-linear hidden states may be over-parameterized on short sequences.
- Future work could scale the hidden state into larger networks (e.g., Transformers) to support even longer video generation.

## Related Work & Insights

- **vs Mamba 2 / DeltaNet**: These methods use linear matrix hidden states, which perform well in short contexts but are limited in long contexts. TTT-MLP breaks this bottleneck by utilizing a non-linear MLP hidden state.
- **vs Sliding Window Attention**: Sliding window attention has a fixed receptive field (e.g., 8192 tokens ≈ 1.5 seconds) and cannot capture long-range dependencies across scenes.
- **vs Story Generation Methods (e.g., StoryDiffusion)**: Story generation requires additional components to maintain scene consistency, whereas TTT accomplishes end-to-end, single-pass generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First application of TTT layers in video generation, with a profound core contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Rigorously designed human evaluation (100 videos × 6 methods), but restricted to a single animation domain.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-written, with clear motivation and detailed explanation of system design.
- **Value**: ⭐⭐⭐⭐ A landmark achievement in long-video generation, though practical utility is currently limited by domain specificity and artifacts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LongDiff: Training-Free Long Video Generation in One Go](longdiff_training-free_long_video_generation_in_one_go.md)
- [\[CVPR 2025\] Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [\[ICML 2025\] Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [\[CVPR 2026\] Reasoning Diffusion for Unpaired Test Time Out-of-distribution Text-Image to Video Generation](../../CVPR2026/video_generation/reasoning_diffusion_for_unpaired_test_time_out-of-distribution_text-image_to_vid.md)
- [\[ICLR 2026\] TTOM: Test-Time Optimization and Memorization for Compositional Video Generation](../../ICLR2026/video_generation/ttom_test-time_optimization_and_memorization_for_compositional_video_generation.md)

</div>

<!-- RELATED:END -->
