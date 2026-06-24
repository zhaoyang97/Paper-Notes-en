---
title: >-
  [Paper Note] GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs
description: >-
  [ACL2025][LLM (Other)][PEFT] This paper proposes GIFT-SW, a novel parameter-efficient fine-tuning method. By updating only the "salient columns" of the weight matrices while injecting Gaussian noise into non-salient columns, GIFT-SW outperforms full-parameter fine-tuning and modern PEFT methods such as LoRA and DoRA under equivalent computational budgets.
tags:
  - "ACL2025"
  - "LLM (Other)"
  - "PEFT"
  - "Salient Weights"
  - "Gaussian Noise Injection"
  - "Quantization"
  - "LLM Fine-tuning"
date: 2026-05-08
content_hash: dd2d91bd5362aef1
---

# GIFT-SW: Gaussian Noise Injected Fine-Tuning of Salient Weights for LLMs

**Conference**: ACL2025  
**arXiv**: [2408.15300](https://arxiv.org/abs/2408.15300)  
**Code**: Yes (mentioned in paper)  
**Area**: LLM NLP / Parameter-Efficient Fine-Tuning  
**Keywords**: PEFT, Salient Weights, Gaussian Noise Injection, Quantization, LLM Fine-tuning  

## TL;DR

This paper proposes GIFT-SW, a novel parameter-efficient fine-tuning method. By updating only the "salient columns" of the weight matrices while injecting Gaussian noise into non-salient columns, GIFT-SW outperforms full-parameter fine-tuning and modern PEFT methods such as LoRA and DoRA under equivalent computational budgets.

## Background & Motivation

### Problem Definition
After pre-training, large language models (LLMs) typically require fine-tuning to enhance performance on specific downstream tasks or to recover capability after compression operations such as quantization or pruning. However, full-parameter fine-tuning introduces immense computational and memory overhead. Although existing parameter-efficient fine-tuning (PEFT) methods (e.g., LoRA) reduce resource requirements, they still struggle to match the accuracy of full-parameter fine-tuning.

### Core Observations

**Importance of Salient Weights**: Previous research has shown that a small portion of "salient weights" (weight outliers) exists in LLMs, which substantially affect post-training quantization (PTQ) and pruning performance. Dettmers et al. discovered that a very small fraction of outliers in input activations significantly impacts overall model performance.

**Regularization Effect of Noise Injection**: Perturbed Gradient Descent (PGD) stabilizes convergence and prevents overfitting by injecting noise before or after gradient steps, thereby helping the model escape saddle points and local optima.

**Equivalence of Quantization Noise and Gaussian Noise**: Mathematically, Quantization Noise Injection (QNI) in quantization-aware training (QAT) is equivalent to Gaussian Noise Injection (GNI). Consequently, GNI can also enhance the robustness of the model to subsequent quantization operations.

### Core Problem
The paper revolves around three core questions:
- Q1: Is updating only a subset of salient weights sufficient to fine-tune the model effectively?
- Q2: Does noise injection facilitate convergence?
- Q3: Does noise injection help improve robustness?

## Method

### Overall Architecture

The GIFT-SW method consists of three steps:
1. **Identify Salient Columns**: Using a sensitivity metric, select a fixed number of salient columns per layer (default: 128 columns) based on a small calibration set.
2. **Split Columns**: Divide the columns of the weight matrix into a salient column subset and a non-salient column subset.
3. **Training Process**: Update only the weights of the salient columns, while injecting Gaussian noise into the non-salient columns, which remain frozen.

### Key Designs

#### Key Design 1: Generalized Sensitivity Metrics

The paper presents a unified, generalized column sensitivity metric:

$$s_j = \|\mathbf{D}_j\|_\tau \|\mathbf{X}_j\|_\rho^\gamma$$

where:
- $\mathbf{D}_j$ is the weight perturbation metric (quantization error $\mathbf{W}_{:,j} - Q(\mathbf{W}_{:,j})$)
- $\mathbf{X}_j$ denotes the input features (computed via the calibration set)
- $\gamma \in \{1/2, 1, 2\}$, and $\tau, \rho$ are norm selection parameters.

This formulation unifies several metric definitions from prior work:
- QUIK/SmoothQuant uses $\|\mathbf{X}\|_\infty$
- OWQ uses $\lambda_j \|\mathbf{D}_j\|_2^2$ (corresponding to the OBD metric)
- Wanda uses an element-wise variant of $\|\mathbf{D}_j\|_1 \|\mathbf{X}_j\|_2$

The paper ultimately chooses the metric with $\gamma=1, \rho=\infty, \tau=\infty$. The rationale is that the $l_\infty$ norm can capture extreme outliers in activations and weight errors more effectively, preventing them from being averaged out by a large number of small values.

#### Key Design 2: Quantization Noise Injection

For non-salient columns, the noise injection operation is defined as:

$$\mho(\mathbf{W}) = \begin{cases} \mathbf{W}_{[:,\text{salient}]} \\ \mathbf{W}_{[:,\text{non-salient}]} + \frac{1}{2}\text{diag}(\mathbf{\Delta})\mathbf{\Omega} \end{cases}$$

where $\mathbf{\Omega} \sim \mathcal{N}(0, 1)$, and $\mathbf{\Delta}$ represents the quantization step size matrix.

Key details of the noise parameter design:
- **Row-wise Scaling Factor Calculation**: $\Delta_i = \frac{\alpha_i}{2^{b-1}-1}$, where $b$ is the bit-width.
- **Excluding Salient Columns**: Salient columns are excluded when calculating the scaling factor. This prevents them from generating excessively large quantization errors that could distort the row-wise scaling factor.
- **Forward Pass Injection**: Noise is injected during the forward pass (prior to the gradient step), serving as a regularizer.
- **Injection Only on Non-salient Columns**: This avoids perturbing sensitive weights, preventing model degradation.

### Loss & Training

A standard conditional language modeling objective (i.e., next token prediction) is employed, with key differences being:
- Only the weights of the salient columns participate in gradient updates.
- Non-salient columns are frozen but perturbed with noise during every forward pass.
- This is equivalent to a specialized variant of Perturbed Gradient Descent (PGD).

## Key Experimental Results

### Experimental Setup
- **Models**: LLaMA2-7B, LLaMA2-13B, LLaMA3-8B
- **Data**: TÜLU-V2-mix (main dataset), OpenOrca
- **Training**: 500 iterations, 4 GPUs (40GB), batch size 128(7B)/64(13B)
- **Number of Salient Columns**: Fixed at 128 columns (~3% of parameters)
- **Evaluation**: Zero-shot average accuracy across HellaSwag, BoolQ, WinoGrande, PiQA, ARC-easy, and ARC-challenge

### Main Results

| Method | LLaMA2-7B (TÜLU) | LLaMA2-13B (TÜLU) | LLaMA3-8B (TÜLU) |
|------|-------------------|---------------------|---------------------|
| Full FT| 71.97 | 75.09 | 76.13 |
| LoRA | 71.78 | 74.03 | 75.91 |
| DoRA | 72.03 | 73.97 | 75.89 |
| **GIFT-SW** | **73.33** | **75.93** | **76.37** |

**Key Findings**: Under full-precision settings, GIFT-SW outperforms full-parameter fine-tuning and LoRA/DoRA, achieving the best performance across all models and most datasets.

### Quantization Model Results

| Bit-width | Method | LLaMA2-7B | LLaMA2-13B | LLaMA3-8B |
|------|------|-----------|------------|-----------|
| 4-bit | STE | 72.43 | **75.29** | 74.84 |
| 4-bit | QUIK+LoRA | 63.99 | 71.08 | 74.27 |
| 4-bit | **GIFT-SW** | **72.53** | 74.50 | **75.46** |
| 3-bit | **GIFT-SW** | **71.00** | 74.34 | **73.27** |
| 2-bit | **GIFT-SW** | **61.09** | **67.61** | **58.89** |

**Key Findings**: In extreme low-precision settings (2-bit), the performance advantage of GIFT-SW is most pronounced, outperforming the runner-up by more than 5 percentage points.

### Comparison with TÜLU2

| Method | LLaMA2-7B Performance | Trainable Params / Iterations |
|------|---------------|-------------|
| TÜLU2 | 73.49 | 6.7B / 5K |
| TÜLU2-DPO | 73.80 | 6.7B / 5K |
| **GIFT-SW** | 73.33 | **174M / 500** |

With only 174M trainable parameters and 500 iterations, GIFT-SW achieves comparable performance to TÜLU2, which requires 6.7B parameters and 5K iterations.

### Ablation Study

**1. Impact of Noise Injection**

| Method | w/ Noise | w/o Noise |
|------|----------|-----------|
| Outlier FT (7B) | **73.33** | 73.16 |
| Outlier FT (13B) | **75.93** | 74.80 |
| Full FT (7B) | 71.64 | 71.97 |

Noise injection consistently benefits the fine-tuning of salient columns but may actually degrade performance for full-parameter fine-tuning.

**2. Comparison of Sensitivity Metrics**: $\|\mathbf{D}_j\|_\infty \|\mathbf{X}_j\|_\infty$ is the optimal metric across most settings, though no single metric holds an absolute advantage across all bit-widths and models.

**3. Pre- vs. Post-Quantization Training**:
- 4-bit: Training before quantization (Pre-GIFT-SW) is optimal.
- 2-bit: Quantization before training (Post-GIFT-SW) is optimal. This is because extremely low-bit quantization causes excessive weight shifts, disrupting the weight relationships established during pre-training.

### Data Scaling Stability
GIFT-SW exhibits stability comparable to full-parameter fine-tuning under different data budgets, whereas LoRA/DoRA display significant instability.

## Highlights & Insights

1. **Simple and Elegant Design**: Instead of introducing auxiliary parameters (such as low-rank matrices in LoRA), GIFT-SW directly updates a subset of the original weights, effectively utilizing noise injection as a regularization mechanism.
2. **Unified Framework**: It unifies salient weight identification metrics from both quantization (QUIK, OWQ, SmoothQuant) and pruning (Wanda) into a single, generalized mathematical formula.
3. **Dual Role of Noise**: Noise injection acts as a regularizer during training and naturally serves as an effective "pre-training" step for subsequent quantization.
4. **Superiority in Extreme Low-Bit Quantization**: Its advantages are particularly prominent in 2-bit quantization scenarios, demonstrating maximum value under extremely resource-constrained circumstances.
5. **High Computational Efficiency**: Achieving comparable performance to heavyweight methods like TÜLU2 requires only about 3% of the parameters and 1/10th of the computational resources.

## Limitations & Future Work

1. The method was validated only on LLaMA-family models; other LLM architectures remain untested.
2. Quantization experiments were restricted to GPTQ.
3. Definitive theoretical guidance for selecting sensitivity metrics is still lacking.
4. Noise parameters were determined solely via the QNI approach, whereas other noise distributions might yield better results.
5. An efficient CUDA kernel implementation has not been provided.
6. Experiments were conducted using only a limited set of instruction-tuning data.
7. The evaluation benchmark is constrained to only 6 zero-shot tasks.

## Related Work & Insights

- **PEFT Methods**: LoRA, DoRA, prompt tuning, etc.
- **Salient Weight Identification**: SparseGPT, Wanda, OWL (Pruning); QUIK, OWQ, SmoothQuant (Quantization)
- **Noise Injection**: Single-layer GNI by Orvieto et al., learnable layer-wise variance noise by Liu et al.
- **Quantization-Aware Training**: STE vs QNI, Défossez et al., Shin et al.

## Rating ⭐⭐⭐⭐

The method design is simple yet effective, and the experiments are comprehensive and convincing. The unified sensitivity metrics and the dual-role design of noise injection are particularly insightful. The method shows significant strengths in extreme low-bit quantization scenarios. However, model coverage is somewhat limited, and the depth of theoretical analysis could be further enhanced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Refining Salience-Aware Sparse Fine-Tuning Strategies for Language Models](refining_salience-aware_sparse_fine-tuning_strategies_for_language_models.md)
- [\[NeurIPS 2025\] SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models](../../NeurIPS2025/llm_nlp/space_noise_contrastive_estimation_stabilizes_self-play_fine-tuning_for_large_la.md)
- [\[ACL 2025\] GORP: Continual Gradient Low-Rank Projection Fine-Tuning for LLMs](gorp_continual_gradient_projection.md)
- [\[ACL 2025\] HFT: Half Fine-Tuning for Large Language Models](hft_half_fine-tuning_for_large_language_models.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)

</div>

<!-- RELATED:END -->
