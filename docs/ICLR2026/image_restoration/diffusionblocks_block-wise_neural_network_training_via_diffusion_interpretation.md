---
title: >-
  [Paper Note] DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation
description: >-
  [ICLR 2026][Image Restoration][block-wise training] This paper proposes DiffusionBlocks, which interprets the layer-wise updates of residual networks as discretization steps of a continuous-time diffusion process…
tags:
  - "ICLR 2026"
  - "Image Restoration"
  - "block-wise training"
  - "diffusion models"
  - "score matching"
  - "memory efficiency"
  - "residual networks"
date: 2026-05-08
content_hash: 405d037c788da9c8
---

# DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation

**Conference**: ICLR 2026
**arXiv**: [2506.14202](https://arxiv.org/abs/2506.14202)
**Code**: [SakanaAI/DiffusionBlocks](https://github.com/SakanaAI/DiffusionBlocks)
**Area**: Image Restoration
**Keywords**: block-wise training, diffusion models, score matching, memory efficiency, residual networks

## TL;DR

This paper proposes DiffusionBlocks, which interprets the layer-wise updates of residual networks as discretization steps of a continuous-time diffusion process, enabling the network to be partitioned into fully independently trainable blocks. This approach achieves competitive performance with end-to-end training while reducing training memory by a factor of $B$ (the number of blocks).

## Background & Motivation

- End-to-end backpropagation requires storing intermediate activations for all layers, causing memory to grow linearly with network depth and severely limiting model scale and practical deployment.
- Existing block-wise training methods (e.g., Forward-Forward, greedy layer-wise training) rely on ad hoc local objective functions, lack theoretical guarantees, and have been validated almost exclusively on classification tasks, making them difficult to generalize to generative tasks.
- The denoising objective of score-based diffusion models inherently allows independent optimization at each noise level—a property that provides the missing theoretical foundation for block-wise training.
- The residual update rule $\mathbf{z}_{\ell+1} = \mathbf{z}_\ell + f_{\theta_\ell}(\mathbf{z}_\ell)$, common in ResNets and Transformers, naturally corresponds to the Euler discretization of the probability flow ODE in diffusion processes.

## Core Problem

How to design a **theoretically grounded** block-wise training framework for Transformer-based networks such that:

1. Each block can be trained **completely independently** (without gradients or activations from other blocks);
2. Performance remains competitive with end-to-end training;
3. The framework generalizes across diverse tasks and architectures, including both classification and generation.

## Method

### Core Insight: Residual Connections = Discretized Diffusion Steps

Under the Variance Exploding (VE) diffusion framework, given noise levels $\sigma_0 > \sigma_1 > \cdots > \sigma_T$, the Euler discretization of the probability flow ODE yields:

$$\mathbf{z}_{\sigma_\ell} = \mathbf{z}_{\sigma_{\ell-1}} + \frac{\Delta\sigma_\ell}{\sigma_{\ell-1}}\left(\mathbf{z}_{\sigma_{\ell-1}} - D_\theta(\mathbf{z}_{\sigma_{\ell-1}}, \sigma_{\ell-1})\right)$$

This naturally corresponds to the residual network update rule $\mathbf{z}_\ell = \mathbf{z}_{\ell-1} + f_{\theta_\ell}(\mathbf{z}_{\ell-1})$.

### Three-Step Conversion Pipeline

**Step 1: Block Partitioning** — Partition an $L$-layer network into $B$ blocks $\mathcal{F}_1, \ldots, \mathcal{F}_B$, each consisting of a contiguous set of layers.

**Step 2: Noise Range Assignment** — Define a noise distribution $p_{\text{noise}}$ (log-normal recommended) and partition $[\sigma_{\min}, \sigma_{\max}]$ into $B$ intervals $\{[\sigma_b, \sigma_{b-1}]\}_{b=1}^B$, assigning each block to denoise within its corresponding range.

**Step 3: Noise-Conditioning Adaptation** — Extend each block's input to $\tilde{\mathbf{x}} = (\mathbf{x}, \mathbf{z}_\sigma)$, where $\mathbf{z}_\sigma = \mathbf{y} + \sigma\epsilon$; inject noise-level conditioning (e.g., AdaLN). Each block is trained independently to predict the target $\mathbf{y}$.

### Independent Training Objective

The loss for each block $b$ is:

$$\mathcal{L}_b(\theta_b) = \mathbb{E}_{(\mathbf{x},\mathbf{y}), \sigma\sim p_{\text{noise}}^{(b)}, \epsilon\sim\mathcal{N}(0,I)}\left[w(\sigma)\cdot\|f_{\theta_b|\sigma}(\mathbf{x}, \mathbf{y}+\sigma\epsilon) - \mathbf{y}\|_2^2\right]$$

Crucially, the $B$ blocks are optimized independently without any inter-block communication, yet together cover the full noise distribution.

### Equi-probability Partitioning

Rather than partitioning noise intervals uniformly (which wastes capacity at extreme noise levels), the method partitions by equal cumulative probability mass under the log-normal distribution:

$$\int_{\sigma_{b-1}}^{\sigma_b} p_{\text{noise}}(\sigma)\,d\sigma = 1/B$$

This ensures each block processes an equal share of the training distribution, allocating finer intervals to intermediate noise levels where denoising difficulty is greatest, thereby improving overall efficiency.

### Inference

At inference time, the blocks are invoked sequentially from high to low noise levels. For diffusion models, each denoising step requires loading only one block, yielding a $B$-fold inference speedup.

## Key Experimental Results

| Task / Architecture | Dataset | End-to-End Baseline | DiffusionBlocks | Blocks / Memory Reduction |
|---|---|---|---|---|
| ViT Classification | CIFAR-100 | 60.25% Acc | 59.30% Acc | B=3 / 3× |
| DiT Image Generation | CIFAR-10 | 32.84 FID | 30.59 FID | B=3 / 3× |
| DiT Image Generation | ImageNet 256 | 12.09 FID | 10.63 FID | B=3 / 3× |
| Masked Diffusion (Text) | text8 | 1.56 BPC | 1.45 BPC | B=3 / 3× |
| AR Transformer (Text) | LM1B | 0.50 MAUVE | 0.71 MAUVE | B=4 / 4× |
| AR Transformer (Text) | OpenWebText | 0.85 MAUVE | 0.82 MAUVE | B=4 / 4× |
| Huginn (recurrent-depth) | LM1B | 0.49 MAUVE | 0.70 MAUVE | Eliminates 32 iterations |

- Forward-Forward achieves only 7.85% accuracy on CIFAR-100, far below DiffusionBlocks.
- On ImageNet with B=2, FID=9.90 **outperforms** end-to-end training (12.09), suggesting that moderate block partitioning can yield performance gains.
- Equi-probability partitioning consistently outperforms uniform partitioning across all layer allocation configurations (CIFAR-10 FID: 38.03 vs. 43.53).

## Highlights & Insights

1. **Solid theoretical foundation**: The independent block training objective is derived naturally from the noise-level independence property of score matching, rather than being assembled heuristically.
2. **Strong generality**: A single three-step conversion pipeline applies uniformly to five distinct architecture families: ViT, DiT, AR Transformer, Masked Diffusion, and recurrent-depth models.
3. **Equi-probability partitioning** is an elegant yet critical design choice—ensuring each block handles an equal denoising load without manual tuning of layer assignments.
4. **Multiple efficiency gains**: $B$-fold training memory reduction; $B$-fold diffusion model inference speedup; elimination of BPTT for recurrent-depth models.
5. **Surpasses end-to-end training in select settings**: On ImageNet with B=2/3, FID improves over the end-to-end baseline, indicating that moderate specialization yields positive benefits.

## Limitations & Future Work

- Classification experiments are limited to CIFAR-100 (60.25→59.30); large-scale ImageNet classification has not been evaluated.
- Inference still requires sequential invocation of blocks and cannot be parallelized across denoising steps.
- Noise-conditioning adaptations (e.g., AdaLN) introduce a modest increase in parameter count and engineering complexity.
- Performance degrades at large $B$ (FID=14.43 at B=6), indicating a lower bound on viable block granularity.
- The framework primarily targets residual Transformer architectures; applicability to networks without residual connections is not discussed.

## Related Work & Insights

| Method | Theoretical Basis | Task Generality | Continuous Time | Block Independence |
|---|---|---|---|---|
| Forward-Forward | Contrastive objective | Classification only | ✗ | ✓ |
| NoProp | Diffusion-related | Classification only | ✓(CT) or ✗(DT) | ✗(CT) or ✓(DT) |
| DiffusionBlocks | Score matching | Classification + Generation | ✓ | ✓ |

- NoProp is tightly coupled to a custom CNN architecture and cannot be directly transferred to Transformers. DiffusionBlocks also outperforms all NoProp variants on the NoProp architecture (46.88 vs. 46.06/21.31/37.57).
- Unlike stage-specific diffusion models (e.g., eDiff-I), which employ joint training or fine-tuning from shared parameters, the blocks in DiffusionBlocks are **fully isolated** from one another.

The perspective that "residual connections ≈ discretized diffusion steps" is broadly extensible: any deep model with a residual structure may benefit from this form of partitioned independent training. The equi-probability partitioning principle is transferable to other settings requiring segmented handling of subtasks of varying difficulty (e.g., curriculum learning, multi-scale training). The ability to eliminate BPTT for recurrent-depth models is particularly noteworthy, as methods such as universal Transformers and Huginn continue to gain traction. Combined with model parallelism (placing each block on a separate GPU), this approach could enable more aggressive depth scaling.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Introducing diffusion-level independence into block-wise training constitutes a highly original theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Broad coverage across five architecture families, though classification experiments are limited in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear, and the three-step pipeline is presented intuitively.
- **Value**: ⭐⭐⭐⭐ — Offers a theoretically grounded new paradigm for addressing the memory bottleneck in large model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AdaBlock-dLLM: Semantic-Aware Diffusion LLM Inference via Adaptive Block Size](adablock-dllm_semantic-aware_diffusion_llm_inference_via_adaptive_block_size.md)
- [\[ICLR 2026\] Breaking Scale Anchoring: Frequency Representation Learning for Accurate High-Resolution Inference from Low-Resolution Training](breaking_scale_anchoring_frequency_representation_learning_for_accurate_high-res.md)
- [\[CVPR 2026\] The Surprising Effectiveness of Noise Pretraining for Implicit Neural Representations](../../CVPR2026/image_restoration/the_surprising_effectiveness_of_noise_pretraining_for_implicit_neural_representa.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](activation_steering_for_masked_diffusion_language_models.md)
- [\[ICLR 2026\] Horizon Imagination: Efficient On-Policy Rollout in Diffusion World Models](horizon_imagination_efficient_on-policy_rollout_in_diffusion_world_models.md)

</div>

<!-- RELATED:END -->
