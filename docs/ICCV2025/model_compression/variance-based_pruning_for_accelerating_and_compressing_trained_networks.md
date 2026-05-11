---
title: >-
  [Paper Note] Variance-Based Pruning for Accelerating and Compressing Trained Networks
description: >-
  [Model Compression] This paper proposes Variance-Based Pruning (VBP), a one-shot structured pruning method that removes neurons with the smallest activation variance in MLP hidden layers and compensates their mean activa…
tags:
  - "Model Compression"
date: 2026-05-08
content_hash: f8cf0570083141b9
---

# Variance-Based Pruning for Accelerating and Compressing Trained Networks

## Basic Information
- **Conference**: ICCV 2025
- **arXiv**: 2507.12988
- **Code**: [https://github.com/boschresearch/variance-based-pruning](https://github.com/boschresearch/variance-based-pruning)
- **Area**: Model Compression / Structured Pruning
- **Keywords**: Structured Pruning, Variance-Based Pruning, One-Shot Pruning, Mean-Shift Compensation, Vision Transformer

## TL;DR

This paper proposes Variance-Based Pruning (VBP), a one-shot structured pruning method that removes neurons with the smallest activation variance in MLP hidden layers and compensates their mean activations into the subsequent layer's bias. With only 10 epochs of fine-tuning, VBP recovers 99% of the original accuracy while reducing computation by 35% and parameters by 36%.

## Background & Motivation

- **Problem Definition**: Large pretrained models (e.g., ViT, Swin, ConvNeXt) face three challenges at deployment: high training cost, large storage overhead, and high inference latency. It is desirable to reuse already-trained models while reducing storage and inference costs.
- **Limitations of Prior Work**:
    - Unstructured pruning preserves accuracy but sparse matrices fail to achieve real-world speedups on modern hardware.
    - Structured pruning methods (e.g., NViT) require extensive retraining (300 epochs) to recover accuracy.
    - Dynamic pruning methods (e.g., Token Merging / ToMe) do not modify model structure and thus cannot reduce storage overhead.
- **Goal**: To design a simple structured pruning method that simultaneously addresses storage and inference costs, requiring only minimal fine-tuning to recover most of the original accuracy.

## Method

### Overall Architecture

VBP consists of three steps: (1) activation statistics computation; (2) variance-based neuron selection; and (3) mean-shift compensation.

### Step 1: Activation Statistics Computation

Pruning targets only the hidden layers of MLPs. Given an MLP that maps input $\mathbf{x} \in \mathbb{R}^{D_\text{in}}$ to output $\mathbf{y} \in \mathbb{R}^{D_\text{out}}$:

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

The mean $\boldsymbol{\mu}$ and variance $\boldsymbol{\sigma}^2$ of each neuron's activation are computed online using Welford's algorithm, which is numerically stable and efficient:

$$\boldsymbol{\mu}^{(j)} = \frac{j-1}{j}\boldsymbol{\mu}^{(j-1)} + \frac{1}{j}\mathbf{h}^{(j)}$$

$$\boldsymbol{\sigma}^2 = \frac{\mathbf{m}_2^{(N)}}{N-1}$$

### Step 2: Variance-Based Pruning

All hidden neurons across layers are ranked by variance $\sigma_i^2$ in ascending order, and the $p\%$ of neurons with the smallest variance are pruned.

**Optimality Argument**: Replacing a pruned neuron's activation with its mean $\mu_i$ introduces a reconstruction error whose expectation equals exactly $\sigma_i^2$. Therefore, pruning neurons with the smallest variance minimizes the reconstruction error.

### Step 3: Mean-Shift Compensation

The key innovation: rather than substituting the pruned neurons' activations with their means at inference time (which still requires the full matrix multiplication), the method exploits the linearity of the mapping to absorb the mean contribution directly into the next layer's bias:

$$\mathbf{b}_2' = \mathbf{b}_2 + \mathbf{W}_2 \Delta_\mu$$

where $\Delta_\mu$ takes the value $\mu_j$ at pruned indices and zero elsewhere. The corresponding rows of $\mathbf{W}_1$ and columns of $\mathbf{W}_2$ can then be removed entirely, reducing the size of both weight matrices.

### Loss & Training

After pruning, only 10 epochs of knowledge distillation fine-tuning are required: AdamW optimizer, lr=1.5e-5, cosine annealing, batch size=32, weight decay=0.01.

## Key Experimental Results

### Main Results

| Model | Pruning Ratio | MACs Reduction | Param Reduction | Accuracy Retention (post-pruning) | Final Accuracy | Speedup |
|-------|--------------|---------------|----------------|----------------------------------|---------------|---------|
| DeiT-B | 55% | 34.93% | 36.01% | 70.48% | 98.74% | 1.44× |
| DeiT-S | 50% | 30.37% | 32.15% | 80.85% | 98.64% | 1.34× |
| DeiT-T | 45% | 25.16% | 27.97% | 69.13% | 97.33% | 1.17× |
| Swin-B | 55% | 33.89% | 35.87% | 73.91% | 98.70% | 1.30× |
| Swin-S | 50% | 32.19% | 29.41% | 80.70% | 98.58% | 1.29× |
| DeiT-B | 20% | 12.68% | 13.09% | 98.98% | 100.07% | 1.11× |

At a pruning ratio of 20%, DeiT-B and Swin-B maintain 99% accuracy after pruning without any fine-tuning.

### Ablation Study

| Variance Pruning | Mean Compensation | Accuracy Retention | Final Accuracy |
|-----------------|------------------|--------------------|---------------|
| ✗ | ✓ | 55.19% | 80.23% |
| ✓ | ✗ | 26.04% | 80.62% |
| ✓ | ✓ | 66.40% | 80.99% |

Combining both components improves post-pruning accuracy retention by 11.21 percentage points and also yields the best fine-tuned accuracy.

### Comparison with Other Methods (50% Pruning Ratio, DeiT-B)

| Method | Accuracy Retention | Final Accuracy |
|--------|--------------------|---------------|
| Magnitude | 0.37% | 78.88% |
| SNIP | 53.24% | 80.40% |
| **VBP (Ours)** | **66.40%** | **80.99%** |

### Comparison with State of the Art

- vs. NViT (CVPR'23): VBP achieves 82.32% accuracy with 1-epoch pruning + 10-epoch fine-tuning, surpassing NViT's 82.18% obtained with 50-epoch pruning + 10-epoch fine-tuning.
- Hybrid approach combining VBP with ToMe: VBP+ToMe achieves 2.05× speedup while retaining 98% of original accuracy, demonstrating the orthogonality of the two methods.

### ConvNeXt Results

| Model | MACs Reduction | Param Reduction | Final Accuracy Retention | Speedup |
|-------|---------------|----------------|------------------------|---------|
| ConvNeXt-T | 33.8% | 55.9% | 98.1% | 1.28× |
| ConvNeXt-S | 41.3% | 53.2% | 97.9% | 1.42× |
| ConvNeXt-B | 42.1% | 53.4% | 97.6% | 1.49× |

Parameter reduction is more pronounced for ConvNeXt (>50%) due to the higher proportion of MLP parameters in these architectures.

## Highlights & Insights

1. **Minimalist yet Effective**: The combination of variance-based pruning and mean-shift compensation is simple and elegant, supported by a clear mathematical optimality argument.
2. **Post-Activation Statistics Are Superior**: In trained networks, post-activation variance better reflects neuron importance than pre-activation variance (accuracy retention: 66.40% vs. 0.43%).
3. **Orthogonal to ToMe**: VBP reduces parameters and model structure, while ToMe reduces the number of tokens; the two methods are complementary and can be stacked to achieve 2× speedup.
4. **Skewed Variance Distribution**: The 60% of neurons with the lowest variance contribute only 10% of total variance, explaining why high pruning ratios still preserve performance.

## Limitations & Future Work

- Only MLP layers are pruned; attention head pruning is not addressed.
- Accuracy retention after pruning is weaker for primarily convolutional architectures such as ConvNeXt.
- A certain amount of calibration data is required to compute activation statistics.
- Applicability to NLP tasks and large language models remains unexplored.

## Related Work & Insights

- NViT (CVPR'23): Performs full structural dimension pruning but requires 300 epochs of retraining; VBP avoids this overhead.
- ToMe (ICLR'23): Dynamic token merging, complementary to VBP.
- The online statistical computation via Welford's algorithm is generalizable to other scenarios requiring activation statistics.

## Rating

- **Novelty**: ⭐⭐⭐ (The combination of variance pruning and mean-shift compensation is simple yet effective, with rigorous mathematical derivation.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Covers multiple architectures, thorough ablations, and comprehensive comparison with state-of-the-art methods.)
- **Writing Quality**: ⭐⭐⭐⭐ (Well-structured with rigorous mathematical exposition.)
- **Value**: ⭐⭐⭐⭐ (Simple and practical; directly applicable to deployment optimization of existing pretrained models.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](../../ICLR2026/model_compression/rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[ICLR 2026\] UniFlow: A Unified Pixel Flow Tokenizer for Visual Understanding and Generation](../../ICLR2026/model_compression/uniflow_a_unified_pixel_flow_tokenizer_for_visual_understanding_and_generation.md)
- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[AAAI 2026\] StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs Through Knowledge-Reasoning Fusion](../../AAAI2026/model_compression/stepfun-formalizer_unlocking_the_autoformalization_potential_of_llms_through_kno.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)

</div>

<!-- RELATED:END -->
