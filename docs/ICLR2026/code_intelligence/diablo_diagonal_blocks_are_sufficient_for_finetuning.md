---
title: >-
  [Paper Note] DiaBlo: Diagonal Blocks Are Sufficient For Finetuning
description: >-
  [ICLR2026][PEFT] This paper proposes DiaBlo—a parameter-efficient fine-tuning method that replaces low-rank decomposition with diagonal block updates. The weight matrix is partitioned into $N \times N$ blocks, and only the diagonal blocks $\mathbf{D}_1, \ldots, \mathbf{D}_N$ are trained. This approach entirely bypasses the non-convex optimization, initialization sensitivity, and gradient instability introduced by the $\mathbf{AB}$ product in LoRA. Zero initialization suffices for convergence, and the method requires only a single `torch.einsum` batched matmul in PyTorch. Theoretical analysis proves that DiaBlo is strictly more expressive than LoRA under the same parameter budget. DiaBlo achieves state-of-the-art results across commonsense reasoning, arithmetic reasoning, code generation, and safety alignment, as well as 4-bit/2-bit quantization settings.
tags:
  - ICLR2026
  - PEFT
  - diagonal blocks
  - LoRA alternative
  - LLM fine-tuning
  - parameter efficiency
date: 2026-05-08
content_hash: 6f301567ab92225e
---

# DiaBlo: Diagonal Blocks Are Sufficient For Finetuning

**Conference**: ICLR2026
**arXiv**: [2506.03230](https://arxiv.org/abs/2506.03230)
**Code**: [ziyangjoy/DiaBlo](https://github.com/ziyangjoy/DiaBlo)
**Area**: Code Intelligence
**Keywords**: PEFT, diagonal blocks, LoRA alternative, LLM fine-tuning, parameter efficiency

## TL;DR

This paper proposes DiaBlo—a parameter-efficient fine-tuning method that replaces low-rank decomposition with diagonal block updates. The weight matrix is partitioned into $N \times N$ blocks, and only the diagonal blocks $\mathbf{D}_1, \ldots, \mathbf{D}_N$ are trained. This approach entirely bypasses the non-convex optimization, initialization sensitivity, and gradient instability introduced by the $\mathbf{AB}$ product in LoRA. Zero initialization suffices for convergence, and the method requires only a single `torch.einsum` batched matmul in PyTorch. Theoretical analysis proves that DiaBlo is strictly more expressive than LoRA under the same parameter budget. DiaBlo achieves state-of-the-art results across commonsense reasoning, arithmetic reasoning, code generation, and safety alignment, as well as 4-bit/2-bit quantization settings.

## Background & Motivation

**Background**: LoRA and its variants (DoRA, PiSSA, MiLoRA, LoRA-GA) constitute the dominant family of PEFT methods. They inject trainable low-rank matrix products $\Delta\mathbf{W} = \mathbf{AB}$ alongside frozen pretrained weights, substantially reducing the number of trainable parameters. Earlier approaches such as Prompt Tuning and Prefix Tuning are lightweight but lack expressiveness; Adapter-based methods require architectural modifications and introduce inference latency.

**Limitations of Prior Work**:

1. **Non-convex optimization**: The $\mathbf{AB}$ product in LoRA renders the objective non-convex with respect to $\mathbf{A}$ and $\mathbf{B}$. The gradients $\mathbf{g}_{\mathbf{A}} = \mathbf{g}_{\mathbf{W}} \mathbf{B}^\top$ and $\mathbf{g}_{\mathbf{B}} = \mathbf{A}^\top \mathbf{g}_{\mathbf{W}}$ are mutually dependent, causing extreme sensitivity to initialization and unstable convergence.
2. **Proliferating variant complexity**: DoRA decouples magnitude and direction; PiSSA initializes with large singular values; MiLoRA uses small singular values; LoRA-GA aligns the first-step gradient. These variants are essentially patches for the matrix-product structure, increasing both algorithmic and engineering complexity.
3. **Hardware-unfriendly sparse methods**: Fine-tuning based on unstructured sparsity (random masking or importance-based selection) avoids low-rank decomposition but induces irregular memory access patterns and poor GPU utilization.

**Core Insight**: The gradient of a diagonal block, $\mathbf{g}_{\mathbf{D}_i} = \mathbf{X}_i^\top \mathbf{g}_{\mathbf{Y}_i}$, equals exactly the gradient of the corresponding sub-block $\mathbf{W}_{ii}$ in full fine-tuning—without passing through any intermediate matrix-product variable. Consequently, zero initialization does not cause gradient vanishing, and the optimization landscape is far simpler than that of low-rank parameterization.

**Core Idea**: Rather than performing low-rank decomposition, DiaBlo directly updates $N$ diagonal blocks $\mathbf{D}_i \in \mathbb{R}^{d_1 \times d_2}$ of the weight matrix, implemented efficiently via batched matrix multiplication.

## Method

### Overall Architecture

For a linear layer $\mathbf{Y} = \mathbf{X}\mathbf{W}$, DiaBlo partitions the weight $\mathbf{W} \in \mathbb{R}^{m_1 \times m_2}$ into an $N \times N$ block matrix and introduces a block-diagonal adapter:

$$\mathbf{Y} = \mathbf{X}\mathbf{W}_0 + \mathbf{X}\mathbf{D}, \quad \mathbf{D} = \text{diag}(\mathbf{D}_1, \ldots, \mathbf{D}_N)$$

Each diagonal block satisfies $\mathbf{D}_i \in \mathbb{R}^{d_1 \times d_2}$ (where $d_1 = m_1/N$, $d_2 = m_2/N$) and is stored as a tensor $\mathcal{D} \in \mathbb{R}^{N \times d_1 \times d_2}$. All off-diagonal blocks are frozen; only the $N$ diagonal blocks are trained.

### Key Designs

1. **Structured sparsity without matrix products**: LoRA represents updates as the product of two matrices $\mathbf{AB}$, whereas DiaBlo directly updates diagonal sub-blocks of the original weight. Since no product is involved, the optimization problem is convex with respect to $\mathbf{D}_i$ in the linear case, eliminating the need for specialized initialization or optimization strategies—zero initialization is sufficient.
2. **GPU-friendly batched matmul**: The forward computation $\mathbf{X}\mathbf{D}$ is equivalent to reshaping $\mathbf{X}$ to $b \times N \times d_1$ and computing `torch.einsum("bNd1,Nd1d2->bNd2", X, D)`; the backward pass follows analogously, without reconstructing the sparse matrix $\mathbf{D}$.
3. **Theoretical guarantees in both linear and nonlinear settings**: Under linear least squares, the DiaBlo solution coincides with the full fine-tuning solution. With parameter count $Nd_1d_2 = m_1m_2/N \geq m_2 r$ versus LoRA's requirement of $(m_1+m_2)r$ parameters, DiaBlo is strictly more expressive under the same budget. In nonlinear settings, when the activations $\mathbf{X}$ and gradients $\mathbf{g}_{\mathbf{Y}}$ satisfy low-rank conditions (supported by prior empirical findings), the stationary points of DiaBlo coincide with those of full fine-tuning.

### Core Differences from LoRA

| Dimension | LoRA | DiaBlo |
|-----------|------|--------|
| Parameterization | Low-rank product $\mathbf{AB}$ | Direct diagonal block update $\mathbf{D}_i$ |
| Optimization landscape | Non-convex, initialization-sensitive | Convex (linear) / flatter (nonlinear) |
| Initialization | Requires Kaiming/SVD strategies | Zero initialization |
| Gradient computation | $\mathbf{g}_\mathbf{A}$ depends on $\mathbf{B}$ and vice versa | $\mathbf{g}_{\mathbf{D}_i} = \mathbf{X}_i^\top \mathbf{g}_{\mathbf{Y}_i}$, independent |
| Implementation complexity | Two parameter matrices + merge logic | Single tensor + einsum |
| FLOPs | $2bmr$ | $bNd^2$ (equal at the same parameter budget) |
| Training speed | Baseline | On par with LoRA; far faster than DoRA |

## Key Experimental Results

### Commonsense Reasoning (Commonsense Reasoning 170K, average over 8 subtasks)

| Model | Method | r/N | Trainable Params | Avg Acc (%) |
|-------|--------|-----|-----------------|-------------|
| LLaMA2-7B | Full FT | — | 100% | 83.5 |
| | LoRA | r=32 | 0.83% | 77.6 |
| | DoRA | r=16 | 0.42% | 80.5 |
| | MiLoRA | r=32 | 0.83% | 79.2 |
| | SMT(Best) | — | 4.91% | 83.4 |
| | **DiaBlo** | **N=128** | **0.52%** | **83.5** |
| LLaMA3-8B | Full FT | — | 100% | 87.5 |
| | LoRA | r=32 | 0.78% | 80.8 |
| | DoRA | r=32 | 0.78% | 85.2 |
| | SMT(Best) | — | 3.01% | 87.2 |
| | **DiaBlo** | **N=64** | **1.04%** | **87.3** |
| LLaMA-13B | DoRA | r=32 | 0.68% | 80.8 |
| | **DiaBlo** | **N=64** | **1.06%** | **84.9** |

### Arithmetic Reasoning (MetaMathQA → GSM8K + MATH, LLaMA2-7B)

| Method | r/N | Trainable Params | GSM8K | MATH | Avg |
|--------|-----|-----------------|-------|------|-----|
| Full FT | — | 100% | 66.5 | 19.8 | 43.2 |
| LoRA | r=64 | 1.67% | 60.6 | 16.9 | 38.7 |
| PiSSA | r=64 | 1.67% | 58.2 | 15.8 | 37.0 |
| MiLoRA | r=64 | 1.67% | 63.5 | 17.8 | 40.7 |
| **DiaBlo** | **N=32** | **2.09%** | **66.3** | **20.4** | **43.4** |

### Code Generation and Safety Alignment (LLaMA3-8B)

| Method | r/N | Trainable Params | Pass@1 | Pass@10 | HEx-PHI |
|--------|-----|-----------------|--------|---------|---------|
| LoRA | r=32 | 1.12% | 34.7 | 50.8 | 91.6 |
| DoRA | r=32 | 1.12% | 33.1 | 48.6 | 93.6 |
| LoRI | r=32 | 0.56% | 43.2 | 63.2 | 92.8 |
| **DiaBlo** | **N=64** | **1.51%** | **43.2** | **63.5** | **97.6** |

### Quantized Model Fine-tuning (Math10K, LLaMA2-7B, average over 4 tasks)

| Quantization | Method | Trainable Params | Avg Acc (%) |
|-------------|--------|-----------------|-------------|
| 4-bit | QLoRA (r=64) | 112M | 53.7 |
| | ApiQ-bw (r=64) | 112M | 53.5 |
| | **MagR-DiaBlo (N=64)** | **70M** | **54.8** |
| 2-bit | QLoRA (r=64) | 112M | 2.1 |
| | GPTQ-LoRA (r=64) | 112M | 39.9 |
| | ApiQ-bw (r=64) | 112M | 47.3 |
| | **MagR-DiaBlo (N=64)** | **70M** | **48.7** |

### Sparsity Pattern Comparison (GSM8K, LLaMA3-8B, Sparsity 1/64)

| Sparsity Pattern | Fine-tuned Acc (%) | Training Time (min) |
|-----------------|--------------------|---------------------|
| **DiaBlo (Diagonal Blocks)** | **67.68** | **17.26** |
| Random Entries | 65.35 | 26.51 |
| Random Block | 64.86 | 29.76 |
| Random Column | 65.19 | 17.01 |
| Random Row | 61.71 | 17.76 |

### Key Findings

- **Commonsense reasoning**: DiaBlo (N=128, 0.52% parameters) achieves 83.5% on LLaMA2-7B, matching Full FT and substantially outperforming LoRA (77.6%); SMT requires 4.91% parameters to barely reach parity.
- **Arithmetic reasoning**: DiaBlo (N=32) attains the highest MATH score of 20.4% among all methods, surpassing Full FT (19.8%).
- **Quantization robustness**: Under 2-bit quantization, QLoRA nearly collapses (2.1%), while DiaBlo maintains 48.7%—a gap of 46.6 percentage points.
- **Training efficiency**: DiaBlo matches LoRA in training speed (170 min/epoch), whereas DoRA requires 480 min/epoch (2.8× slower).
- **Structured sparsity advantage**: Diagonal blocks achieve the highest accuracy among all sparsity patterns and are 1.5–1.7× faster than unstructured alternatives.
- **Gradient stability**: The gradient norm variance of DiaBlo is consistently lower than that of LoRA; LoRA exhibits gradient vanishing in matrix $\mathbf{A}$ during early training.

## Highlights & Insights

- **Remarkably simple yet surprisingly effective**: Zero initialization combined with diagonal block updates requires no additional tricks.
- **Rigorous theoretical grounding**: DiaBlo is strictly superior to LoRA on linear problems—not merely approximately so.
- **Quantization-friendly**: The diagonal block structure is more robust than low-rank matrix products under low-bit quantization.
- **The optimization difficulty of LoRA is fundamental**: The low-rank matrix product is inherently a non-convex problem; DiaBlo circumvents this entirely.

## Limitations & Future Work

- The diagonal block assumption disregards cross-block interactions, which may limit performance on tasks requiring full-rank updates.
- The choice of $N$ must be tuned to match hardware constraints and parameter budgets.
- A systematic comparison with adapter-based methods has not been conducted.

## Related Work & Insights

- **vs. LoRA**: LoRA approximates $\Delta \mathbf{W}$ with a low-rank product $\mathbf{AB}$. DiaBlo employs structured sparsity (diagonal blocks), yielding greater stability and expressiveness.
- **vs. S²FT**: S²FT is also a structured sparse fine-tuning method. DiaBlo's diagonal blocks are more regular, leading to higher GPU efficiency.
- **vs. QLoRA**: QLoRA combines quantization with LoRA. The combination of DiaBlo and quantization appears more favorable, with a clear advantage at 2-bit precision.

## Rating

- Novelty: ⭐⭐⭐⭐ Conceptually minimalist yet effective, with solid theoretical support
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple tasks, precisions, and model scales
- Writing Quality: ⭐⭐⭐⭐ Theory and experiments are clearly presented with intuitive figures and tables
- Value: ⭐⭐⭐⭐⭐ Has the potential to replace LoRA as the default PEFT method

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Large Language Models to Reason in Parallel with Global Forking Tokens](training_large_language_models_to_reason_in_parallel_with_global_reflection.md)
- [\[ICLR 2026\] MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task](mathfimer_enhancing_mathematical_reasoning_by_expanding_reasoning_steps_through_.md)
- [\[ICLR 2026\] Improving Code Localization with Repository Memory](improving_code_localization_with_repository_memory.md)
- [\[ICLR 2026\] IMSE: Intrinsic Mixture of Spectral Experts Fine-tuning for Test-Time Adaptation](imse_intrinsic_mixture_of_spectral_experts_fine-tuning_for_test-time_adaptation.md)
- [\[ICLR 2026\] Execution-Grounded Credit Assignment for GRPO in Code Generation](execution-grounded_credit_assignment_for_grpo_in_code_generation.md)

</div>

<!-- RELATED:END -->
