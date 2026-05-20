---
title: >-
  [Paper Note] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation
description: >-
  [CVPR 2026][Model Compression][Test-Time Adaptation] This paper proposes FOZO, a forward-only zeroth-order prompt optimization paradigm that updates prompts via SPSA gradient estimation, a dynamic perturbation strategy…
tags:
  - "CVPR 2026"
  - "Model Compression"
  - "Test-Time Adaptation"
  - "Zeroth-Order Optimization"
  - "Visual Prompt"
  - "Forward Propagation"
  - "Quantized Model Deployment"
date: 2026-05-08
content_hash: 3f8de379cad2b9b4
---

# FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation

**Conference**: CVPR 2026
**arXiv**: [2603.04733](https://arxiv.org/abs/2603.04733)  
**Code**: [eVI-group-SCU/FOZO](https://github.com/eVI-group-SCU/FOZO)  
**Area**: Model Compression
**Keywords**: Test-Time Adaptation, Zeroth-Order Optimization, Visual Prompt, Forward Propagation, Quantized Model Deployment

## TL;DR

This paper proposes FOZO, a forward-only zeroth-order prompt optimization paradigm that updates prompts via SPSA gradient estimation, a dynamic perturbation strategy, and shallow–deep feature statistics alignment—without modifying model weights. FOZO achieves 59.52% accuracy on ImageNet-C, surpassing all forward-only methods including FOA (58.13%), and supports INT8 quantized models.

## Background & Motivation

**Distribution shift is ubiquitous**: Deep learning models frequently encounter train-test distribution shift in real-world deployment. TTA addresses this by dynamically adapting models using unlabeled test data at inference time.

**Backpropagation-based methods are resource-intensive**: Gradient-based TTA methods such as TENT, SAR, and EATA require backpropagation to update model weights, incurring high computational and memory overhead (e.g., TENT: 5,495 MiB vs. FOZO: 831 MiB), making them unsuitable for low-power edge devices.

**Gradient-free methods have limited capacity**: Methods such as AdaBN, T3A, and LAME do not construct explicit optimization objectives, resulting in limited learning capacity and suboptimal adaptation performance (e.g., LAME achieves only 54.16% on ImageNet-C).

**CMA-ES is inefficient in high-dimensional spaces**: The recent forward-only method FOA employs the CMA-ES evolutionary strategy to update prompts, but CMA-ES has $O(d^2)$ complexity and converges slowly in high-dimensional prompt spaces.

**ZOA modifies internal model parameters**: ZOA applies zeroth-order optimization to update normalization layer parameters, limiting its applicability in scenarios where model weights cannot be modified (e.g., hardware-encoded or quantized models).

**OOD data streams pose optimization challenges**: The continuously shifting data distribution in TTA makes zeroth-order gradient estimates prone to unreliability, necessitating specialized optimization strategies to ensure convergence stability.

## Method

### Overall Architecture

FOZO injects a small set of learnable prompts $\mathbf{P} = \{\mathbf{p}^k \in \mathbb{R}^d \mid 1 \leq k \leq p\}$ (default $p=3$) into the input layer of a pretrained ViT, with all model weights frozen. Upon each test batch, the prompts are updated via SPSA zeroth-order gradient estimation using only forward passes. The core pipeline is as follows:

1. Apply symmetric positive and negative perturbations to prompt $\mathbf{P}$: $\mathbf{P}_+ = \mathbf{P} + \epsilon_t \mathbf{Z}$, $\mathbf{P}_- = \mathbf{P} - \epsilon_t \mathbf{Z}$ (where $\mathbf{Z} \sim \mathcal{N}(0, I_d)$)
2. Compute losses $\ell_+$ and $\ell_-$ via separate forward passes
3. Estimate the projected gradient: $\hat{g} = \frac{\ell_+ - \ell_-}{2\epsilon_t} \mathbf{Z}$
4. Average gradient estimates over $n$ SPSA samples and update the prompts

### Key Designs: Dynamic Perturbation Strategy

Based on convergence analysis (Theorem 1), the bias term $C\eta\ell\epsilon_t^2 r$ requires $\epsilon_t \to 0$ for accurate convergence, yet large perturbations are needed early in training or after domain shifts to encourage exploration. FOZO introduces an adaptive decay mechanism:

$$\epsilon_t = \begin{cases} \epsilon_0 & \text{if } L_t > \tau \cdot \bar{L}_t \\ \max(\epsilon_{\min}, \epsilon_{t-1} \cdot \alpha) & \text{otherwise} \end{cases}$$

- When a loss spike is detected (domain shift or optimization stagnation), $\epsilon_t$ is reset to $\epsilon_0$
- Otherwise, it decays gradually with factor $\alpha = 0.9$
- Convergence rate is theoretically shown to depend on the effective Hessian rank $r$ rather than the parameter dimension $d$ (Theorem 2)

### Loss & Training

**Shallow–deep feature statistics alignment $\mathcal{L}_{stats}$**: The mean $\mu$ and standard deviation $\sigma$ of [CLS] token activations are collected from shallow ($1 \sim N/2$) and deep ($N/2+1 \sim N$) ViT layers, and aligned to source-domain statistics precomputed offline:

$$\mathcal{L}_{stats} = \sum_{k \in \{shallow, deep\}} (\|\mu_k^T - \mu_k^S\|_2 + \|\sigma_k^T - \sigma_k^S\|_2)$$

**Entropy minimization $\mathcal{L}_{ent}$**: Encourages high-confidence predictions on the target domain.

**Total loss**: $\mathcal{L} = \lambda \mathcal{L}_{stats} + \mathcal{L}_{ent}$, where $\lambda = 0.4$.

## Key Experimental Results

### Main Results

**Comparison with forward-only methods on ImageNet-C (5K, severity level 5, 2 forward passes):**

| Method | FP | Avg Acc (%) | Time (s) | Memory (MiB) | #Params |
|---|---|---|---|---|---|
| NoAdapt | 1 | 55.57 | 94 | 819 | 0 |
| LAME | 1 | 54.16 | 97 | 819 | 0 |
| T3A | 1 | 53.76 | 311 | 823 | 0 |
| FOA | 2 | 58.13 | 224 | 831 | 2304 |
| ZOA | 2 | 58.56 | 198 | 859 | 26145 |
| **FOZO** | **2** | **59.52** | **179** | **831** | **2304** |

**Comparison with backpropagation-based methods (28 forward passes):**

| Method | Avg Acc (%) | Time (s) | Memory (MiB) |
|---|---|---|---|
| TENT | 58.32 | 208 | 5495 |
| EATA | 61.35 | 218 | 5496 |
| SAR | 60.36 | 393 | 5495 |
| DEYO | 60.76 | 282 | 5499 |
| **FOZO (FP=26)** | **62.60** | **2102** | **831** |

### Ablation Study

| Configuration | Acc (%) | Δ |
|---|---|---|
| NoAdapt | 55.1 | — |
| Base FOZO (ZO + Entropy) | 57.3 | +2.2 |
| + Deep-Shallow Alignment | 60.1 | +2.8 |
| + Dynamic Perturbation (full) | 62.7 | +2.6 |

Shallow–deep feature alignment contributes the most (+2.8%), followed by the dynamic perturbation strategy (+2.6%).

### Key Findings

- **Strong compatibility with quantized models**: On INT8 PTQ4ViT, FOZO achieves 58.00% vs. FOA 57.07% and ZOA 56.91%, demonstrating the advantage of forward-only methods in quantized settings.
- **Exceptional memory efficiency**: FOZO requires only 831 MiB, on par with the no-adaptation baseline and approximately 15% of backpropagation-based methods (831 vs. 5,495 MiB).
- **Parameter efficiency**: Only 2,304 prompt parameters are updated, representing 8.8% of ZOA's 26,145 parameters.
- **Convergence speed**: FOZO reaches 65% accuracy in approximately 66% of the time required by FOA/ZOA.
- **Cross-dataset generalization**: FOZO outperforms all forward-only methods on ImageNet-R (64.1%) and ImageNet-Sketch (50.5%).

## Highlights & Insights

- **Solid theoretical foundation**: Convergence is rigorously proven under the SPSA framework and a local effective rank assumption; the convergence rate depends on the effective Hessian rank $r$ rather than the parameter dimension $d$.
- **Strong practical applicability**: Pure forward-pass inference, no model weight modification, and low memory footprint make FOZO directly deployable on edge devices and quantized models.
- **Elegant dynamic perturbation design**: Domain shifts are automatically detected to reset the perturbation scale, striking a balance between exploration and convergence.
- **Comprehensive evaluation**: Experiments cover full-precision and quantized models, multiple datasets, continual adaptation scenarios, clear ablation studies, and hyperparameter sensitivity analysis.

## Limitations & Future Work

- **High wall-clock time**: Under multiple forward passes (FP=26/28), adaptation time (2,102 s) is substantially higher than backpropagation-based methods (208–393 s); the memory-for-speed trade-off may be unfavorable in latency-sensitive scenarios.
- **Validated only on ViT architectures**: CNN or other architectures are not tested, and prompt injection relies on ViT's token concatenation mechanism.
- **Dependency on source-domain statistics**: Feature statistics must be precomputed from a source-domain validation set, which may be unavailable in fully black-box deployment settings.
- **Sensitivity to prompt count and batch size**: Although ablations show that 3 prompts and batch size 64 are robust configurations, performance degrades noticeably at small batch sizes (4/8).

## Related Work & Insights

- **FOA** (CVPR 2024): The first forward-only prompt optimization TTA method, which uses CMA-ES to update prompts. FOZO replaces CMA-ES with SPSA to address its $O(d^2)$ complexity.
- **ZOA** (ACM MM 2025): Applies zeroth-order optimization to TTA but updates normalization layer parameters (26,145 parameters); FOZO updates only prompts (2,304 parameters) without modifying the model.
- **TENT** (ICLR 2021): A landmark entropy-minimization TTA method that requires backpropagation to update batch normalization parameters; FOZO inherits the entropy loss formulation while eliminating backpropagation.
- **MeZO** (NeurIPS 2023): Proposes the local effective rank hypothesis to establish the feasibility of zeroth-order optimization; FOZO extends this theoretical framework to the TTA setting.
- **Visual Prompt Tuning** (ECCV 2022): The seminal method for injecting learnable prompts into the ViT input layer; FOZO adapts this technique to a backpropagation-free test-time setting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Replacing CMA-ES with SPSA zeroth-order estimation for TTA prompt optimization is a well-motivated and novel combination; the dynamic perturbation design is theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-dataset evaluation, quantized model testing, continual adaptation, detailed ablations, and hyperparameter analysis are provided, though non-ViT architectures are absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete theoretical derivations, and well-articulated motivation.
- **Value**: ⭐⭐⭐⭐ — Provides a strongly competitive solution for TTA on edge-deployed and quantized models in practical settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](../../ICLR2026/model_compression/fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[CVPR 2026\] TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery](talon_test-time_adaptive_learning_for_on-the-fly_category_discovery.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](towards_generalizable_ai-generated_image_detection_via_image-adaptive_prompt_lea.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](../../ACL2026/model_compression/training-free_test-time_contrastive_learning_for_large_language_models.md)
- [\[ACL 2026\] LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization](../../ACL2026/model_compression/llm_prompt_duel_optimizer_efficient_label-free_prompt_optimization.md)

</div>

<!-- RELATED:END -->
