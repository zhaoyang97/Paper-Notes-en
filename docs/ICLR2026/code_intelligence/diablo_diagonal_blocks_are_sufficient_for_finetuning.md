---
title: >-
  [Paper Note] Batched matmul for DiaBlo forward pass
description: >-
  [ICLR2026][Code Intelligence][PEFT] DiaBlo is proposed—a parameter-efficient fine-tuning method that replaces low-rank decomposition with diagonal block updates. By partitioning the weight matrix into $N \times N$ blocks and training only the diagonal blocks $\mathbf{D}_1, \ldots, \mathbf{D}_N$, it completely bypasses the non-convex optimization, initialization sensitivity, and gradient instability issues caused by the $\mathbf{AB}$ product in LoRA. It converges with zero ini…
tags:
  - "ICLR2026"
  - "Code Intelligence"
  - "PEFT"
  - "diagonal blocks"
  - "LoRA alternative"
  - "LLM fine-tuning"
  - "parameter efficiency"
date: 2026-05-08
content_hash: 931a841571489f8b
---

# DiaBlo: Diagonal Blocks Are Sufficient For Finetuning

**Conference**: ICLR2026  
**arXiv**: [2506.03230](https://arxiv.org/abs/2506.03230)  
**Code**: [ziyangjoy/DiaBlo](https://github.com/ziyangjoy/DiaBlo)  
**Area**: Code Intelligence  
**Keywords**: PEFT, diagonal blocks, LoRA alternative, LLM fine-tuning, parameter efficiency

## TL;DR

DiaBlo is proposed—a parameter-efficient fine-tuning method that replaces low-rank decomposition with diagonal block updates. By partitioning the weight matrix into $N \times N$ blocks and training only the diagonal blocks $\mathbf{D}_1, \ldots, \mathbf{D}_N$, it completely bypasses the non-convex optimization, initialization sensitivity, and gradient instability issues caused by the $\mathbf{AB}$ product in LoRA. It converges with zero initialization and is efficiently implemented using a single `torch.einsum` batched matmul in PyTorch. Theoretically, its expressivity is strictly superior to LoRA under the same parameter budget. It achieves state-of-the-art performance across four major tasks—commonsense reasoning, arithmetic reasoning, code generation, and security alignment—and in 4-bit/2-bit quantization scenarios.

## Background & Motivation

**Background**: LoRA and its variants (DoRA, PiSSA, MiLoRA, LoRA-GA) are currently the mainstream PEFT methods. They inject a trainable low-rank matrix product $\Delta\mathbf{W} = \mathbf{AB}$ alongside pre-trained weights to significantly reduce the number of trainable parameters. Early methods like Prompt Tuning or Prefix Tuning are lightweight but have limited expressivity, while Adapter methods require modifying the model architecture and introduce inference latency.

**Limitations of Prior Work**:

1.  **Challenging Non-convex Optimization**: The $\mathbf{AB}$ product in LoRA makes the objective function non-convex with respect to $\mathbf{A}$ and $\mathbf{B}$. The gradients $\mathbf{g}_{\mathbf{A}} = \mathbf{g}_{\mathbf{W}} \mathbf{B}^\top$ and $\mathbf{g}_{\mathbf{B}} = \mathbf{A}^\top \mathbf{g}_{\mathbf{W}}$ are interdependent, leading to extreme sensitivity to initialization and unstable convergence.
2.  **Complexity Bloat of Variants**: Variants like DoRA (decoupling magnitude/direction), PiSSA (using large singular values for initialization), MiLoRA (using small singular values), and LoRA-GA (aligning gradients of the first step) essentially patch the matrix product structure, increasing algorithmic and engineering complexity.
3.  **Hardware Unfriendliness of Sparse Methods**: Fine-tuning based on unstructured sparsity (random masking or importance selection) avoids low-rank decomposition but results in irregular memory access and low GPU utilization.

**Key Insight**: The gradient of the diagonal blocks $\mathbf{g}_{\mathbf{D}_i} = \mathbf{X}_i^\top \mathbf{g}_{\mathbf{Y}_i}$ is exactly equal to the gradient of the corresponding sub-blocks $\mathbf{W}_{ii}$ in full fine-tuning. Since it does not pass through any matrix product intermediate variables, zero initialization does not lead to vanishing gradients, and the optimization landscape is much simpler than low-rank parameterization.

**Core Idea**: Instead of low-rank decomposition, directly update $N$ diagonal blocks $\mathbf{D}_i \in \mathbb{R}^{d_1 \times d_2}$ of the weight matrix, efficiently implemented via batched matmul.

## Method

### Overall Architecture

DiaBlo partitions the weight matrix $\mathbf{W} \in \mathbb{R}^{m_1 \times m_2}$ of each linear layer into an $N \times N$ block matrix. It freezes the pre-trained weights $\mathbf{W}_0$ and trains only the $N$ blocks along the "diagonal." The forward pass becomes $\mathbf{Y} = \mathbf{X}\mathbf{W}_0 + \mathbf{X}\mathbf{D}$, where $\mathbf{D} = \text{diag}(\mathbf{D}_1, \ldots, \mathbf{D}_N)$ is a block-diagonal adaptation matrix. Each diagonal block $\mathbf{D}_i \in \mathbb{R}^{d_1 \times d_2}$ (where $d_1 = m_1/N$ and $d_2 = m_2/N$) is stored as a tensor $\mathcal{D} \in \mathbb{R}^{N \times d_1 \times d_2}$. The key to this design is that updates directly target sub-blocks of the original weights, unlike LoRA which requires the product of two matrices.

### Key Designs

**1. Direct Diagonal Block Updates: Bypassing Non-convex Optimization via Structured Sparsity**

LoRA formulates the update as $\Delta\mathbf{W} = \mathbf{AB}$, making the objective non-convex for $\mathbf{A}$ and $\mathbf{B}$. Their gradients are coupled, making it highly sensitive to initialization and prone to vanishing gradients for matrix $\mathbf{A}$ early in training. Variants like DoRA/PiSSA/MiLoRA essentially patch this product structure. DiaBlo avoids the product entirely by placing trainable parameters directly into the diagonal sub-blocks $\mathbf{D}_i$. The gradient $\mathbf{g}_{\mathbf{D}_i}$ matches the full fine-tuning sub-block gradient, avoiding intermediate product variables. Consequently, it achieves stable convergence from all-zero initialization without any specialized initialization tricks.

**2. Batched Matmul Implementation: Enabling Fast Structured Sparsity on GPUs**

While early sparse fine-tuning methods using random masks or importance selection avoided low-rank decomposition, their irregular sparsity patterns caused jumping memory access and poor GPU utilization. DiaBlo’s diagonal block structure is regular. The forward computation $\mathbf{X}\mathbf{D}$ does not require constructing the sparse matrix $\mathbf{D}$. Instead, $\mathbf{X}$ is reshaped to $b \times N \times d_1$, and a single line of code performs the batched matmul:

```python
# Batched matmul for DiaBlo forward pass
Y_delta = torch.einsum("bNd1,Nd1d2->bNd2", X, D)
```

The backward pass is handled similarly. With the same parameter count, its computational complexity $bNd^2$ is comparable to LoRA's $2bmr$, making training as fast as LoRA and approximately 2.8x faster than DoRA (170 vs 480 min/epoch).

**3. Theoretical Guarantee of Expressivity: Strictly Stronger than LoRA under Same Budget**

In a linear least-squares setting, the optimal solution for DiaBlo is the same as the full fine-tuning solution. The number of parameters required to achieve this is $Nd_1d_2 = m_1 m_2 / N \geq m_2 r$. Since LoRA requires at least $(m_1 + m_2)r$ parameters to represent a rank-$r$ update, DiaBlo's expressivity is strictly stronger (not just approximately stronger) under the same parameter budget. Extending this to non-linear networks, when activations $\mathbf{X}$ and output gradients $\mathbf{g}_{\mathbf{Y}}$ satisfy low-rank conditions (supported by existing empirical literature), DiaBlo's stationary points align with those of full fine-tuning, demonstrating that this diagonal block parameterization does not sacrifice reachable solutions in practice.

## Key Experimental Results

### Commonsense Reasoning (170K, Average of 8 sub-tasks)

| Model | Method | r/N | Trainable Params | Avg Acc (%) |
|------|------|-----|-----------|-------------|
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
|------|-----|-----------|-------|------|-----|
| Full FT | — | 100% | 66.5 | 19.8 | 43.2 |
| LoRA | r=64 | 1.67% | 60.6 | 16.9 | 38.7 |
| PiSSA | r=64 | 1.67% | 58.2 | 15.8 | 37.0 |
| MiLoRA | r=64 | 1.67% | 63.5 | 17.8 | 40.7 |
| **DiaBlo** | **N=32** | **2.09%** | **66.3** | **20.4** | **43.4** |

### Code Generation & Security Alignment (LLaMA3-8B)

| Method | r/N | Trainable Params | Pass@1 | Pass@10 | HEx-PHI |
|------|-----|-----------|--------|---------|---------|
| LoRA | r=32 | 1.12% | 34.7 | 50.8 | 91.6 |
| DoRA | r=32 | 1.12% | 33.1 | 48.6 | 93.6 |
| LoRI | r=32 | 0.56% | 43.2 | 63.2 | 92.8 |
| **DiaBlo** | **N=64** | **1.51%** | **43.2** | **63.5** | **97.6** |

### Quantized Model Fine-tuning (Math10K, LLaMA2-7B, Avg of 4 tasks)

| Bit-width | Method | Trainable Params | Avg Acc (%) |
|---------|------|-----------|-------------|
| 4-bit | QLoRA (r=64) | 112M | 53.7 |
| | ApiQ-bw (r=64) | 112M | 53.5 |
| | **MagR-DiaBlo (N=64)** | **70M** | **54.8** |
| 2-bit | QLoRA (r=64) | 112M | 2.1 |
| | GPTQ-LoRA (r=64) | 112M | 39.9 |
| | ApiQ-bw (r=64) | 112M | 47.3 |
| | **MagR-DiaBlo (N=64)** | **70M** | **48.7** |

### Ablation Study: Sparse Pattern Comparison (GSM8K, LLaMA3-8B, Sparsity 1/64)

| Sparse Pattern | Fine-tuning Acc (%) | Training Time (min) |
|----------|-------------|---------------|
| **DiaBlo (Diagonal Block)** | **67.68** | **17.26** |
| Random Entries | 65.35 | 26.51 |
| Random Block | 64.86 | 29.76 |
| Random Column | 65.19 | 17.01 |
| Random Row | 61.71 | 17.76 |

### Key Findings

- **Commonsense Reasoning**: DiaBlo (N=128, 0.52% params) reaches 83.5% on LLaMA2-7B, matching Full FT and significantly outperforming LoRA (77.6%); SMT requires 4.91% parameters to reach similar levels.
- **Arithmetic Reasoning**: DiaBlo (N=32) achieves the highest MATH score of 20.4% across all methods, exceeding Full FT (19.8%).
- **Quantization Robustness**: At 2-bit, QLoRA nearly collapses (2.1%), whereas DiaBlo maintains 48.7%—a gap of 46.6 percentage points.
- **Training Efficiency**: DiaBlo maintains the same training speed as LoRA (170 min/epoch), while DoRA requires 480 min/epoch (2.8x slower).
- **Structured Sparsity Advantage**: Diagonal blocks yield the highest accuracy among all sparse patterns and are 1.5-1.7x faster than unstructured methods.
- **Gradient Stability**: The variance of DiaBlo's gradient norm is consistently lower than LoRA's; LoRA exhibits vanishing gradients in matrix $\mathbf{A}$ during the early stages of training.

## Highlights & Insights
- **Extremely Simple yet Surprisingly Effective**: Zero initialization + diagonal block updates require no specialized tricks.
- **Theoretically Rigorous**: Strictly superior to LoRA in linear cases (not just approximately).
- **Quantization-friendly**: The diagonal block structure is more robust than low-rank products in low-bit scenarios.
- **Fundamental Optimization Issues in LoRA**: Low-rank matrix products are inherently non-convex; DiaBlo bypasses this entirely.

## Limitations & Future Work
- The diagonal block assumption ignores cross-block information—this might be a limitation in tasks requiring full-rank updates.
- The selection of $N$ needs to be tuned for specific hardware and parameter budgets.
- Lack of systematic comparison with adapter-based methods.

## Related Work & Insights
- **vs LoRA**: LoRA uses $\mathbf{AB}$ to approximate $\Delta \mathbf{W}$. DiaBlo uses structured sparsity (diagonal blocks)—resulting in higher stability and expressivity.
- **vs S²FT**: Also uses structured sparse fine-tuning. DiaBlo's diagonal blocks are more regular, leading to higher GPU efficiency.
- **vs QLoRA**: Quantization plus LoRA. The combination of DiaBlo and quantization appears superior (especially evident at 2-bit).

## Rating
- Novelty: ⭐⭐⭐⭐ Simple idea, robust theoretical support.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across diverse tasks, precisions, and models.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative for theory and experiments with intuitive visualizations.
- Value: ⭐⭐⭐⭐⭐ Potentially replaces LoRA as the new default choice for PEFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning](tikzilla_scaling_text-to-tikz_with_high-quality_data_and_reinforcement_learning.md)
- [\[ICLR 2026\] ATGen: Adversarial Reinforcement Learning for Test Case Generation](atgen_adversarial_reinforcement_learning_for_test_case_generation.md)
- [\[ICLR 2026\] SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification](spotit_evaluating_text-to-sql_evaluation_with_formal_verification.md)
- [\[ICLR 2026\] Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning](critique-coder_enhancing_coder_models_by_critique_reinforcement_learning.md)
- [\[ICLR 2026\] BOAD: Discovering Hierarchical Software Engineering Agents via Bandit Optimization](boad_discovering_hierarchical_software_engineering_agents_via_bandit_optimizatio.md)

</div>

<!-- RELATED:END -->
