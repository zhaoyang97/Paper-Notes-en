---
title: >-
  [Paper Note] COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics
description: >-
  [ICLR 2026][Optimization][Activation Steering] COLD-Steer is proposed as a training-free LLM activation steering method that approximates the representational change induced by one step of gradient descent on in-context examples, achieving 95% steering effectiveness with only 1/50 of the samples required by prior methods.
tags:
  - ICLR 2026
  - Optimization
  - Activation Steering
  - Learning Dynamics
  - Training-Free Inference
  - Sample Efficiency
  - Pluralistic Alignment
date: 2026-05-08
content_hash: 546f4494f798eace
---

# COLD-Steer: Steering Large Language Models via In-Context One-step Learning Dynamics

**Conference**: ICLR 2026
**arXiv**: [2603.06495](https://arxiv.org/abs/2603.06495)
**Code**: [https://github.com/Ksartik/cold-steer](https://github.com/Ksartik/cold-steer)
**Area**: Optimization
**Keywords**: Activation Steering, Learning Dynamics, Training-Free Inference, Sample Efficiency, Pluralistic Alignment

## TL;DR
COLD-Steer is proposed as a training-free LLM activation steering method that approximates the representational change induced by one step of gradient descent on in-context examples, achieving 95% steering effectiveness with only 1/50 of the samples required by prior methods.

## Background & Motivation

**State of the Field**: Activation steering controls LLM behavior at inference time without retraining, and falls into two categories — contrastive methods (DiffMean/CAA) construct steering vectors from activation differences of positive/negative pairs, while parameter-tuning methods (ReFT/BiPO) train steering parameters end-to-end.

**Limitations of Prior Work**: Contrastive methods are sample-efficient but rely solely on activation-level signals (without loss function information), limiting steering precision; parameter-tuning methods (e.g., ReFT) require 250–1000 training examples and multi-epoch tuning, incurring high cost.

**Root Cause**: There exists a fundamental trade-off between sample efficiency and steering precision — how can one achieve fine-tuning-level steering with few examples and no parameter updates?

**Paper Goals**: Design a training-free framework that steers LLM behavior effectively using only 10–50 examples.

**Starting Point**: The authors observe that representational changes during fine-tuning follow analytically tractable patterns (learning dynamics). The core insight is that the effect of gradient descent on representations can be *simulated* at inference time without actually updating parameters.

**Core Idea**: Reframe activation steering as "simulating the learning dynamics of a single gradient descent step" — compute how gradients from in-context examples would alter the target representation, and directly apply that change as a steering vector.

## Method

### Overall Architecture
Given $N$ examples $\{(\tilde{\mathbf{x}}_i, \tilde{\mathbf{y}}_i)\}$, COLD-Steer computes the representational change $\Delta\mathbf{Z}^*$ induced by one gradient descent step on these examples, then applies this change as an additive intervention to layer $l$ representations of new inputs:
$$\Delta\mathbf{Z}^*(\mathbf{x}) \approx -\frac{\eta}{N} \nabla_\theta \mathbf{Z}(\mathbf{x};\theta) \sum_i \nabla_\theta \mathcal{L}(\mathcal{M}(\tilde{\mathbf{x}}_i), \tilde{\mathbf{y}}_i)$$

### Key Designs

1. **COLD-Kernel-Steer (Kernel Approximation)**:

    - **Function**: Approximates the eNTK via a kernel function, avoiding backpropagation through new inputs.
    - **Mechanism**: Expands the gradient chain rule and introduces a kernel function $\kappa$:
      $\Delta\mathbf{Z}^{(\kappa)}(\mathbf{x}) = -\frac{\eta}{N} \sum_i \kappa(\mathbf{Z}(\mathbf{x}), \mathbf{Z}(\tilde{\mathbf{x}}_i)) \nabla_{\mathbf{Z}} \mathcal{L}|_{\mathbf{Z}(\tilde{\mathbf{x}}_i)}$.
      A unit kernel $\kappa = 1$ is used as the approximation, motivated by the linear representation hypothesis — gradients for the same concept are dominated by shared directions.
    - **Design Motivation**: Inference on new inputs requires only one forward pass plus $O(N \cdot d)$ kernel similarity computations, making it extremely efficient.

2. **COLD-FD-Steer (Finite Difference Approximation)**:

    - **Function**: Bypasses Jacobian computation via finite differences.
    - **Mechanism**: $\Delta\mathbf{Z}^{(fd)} = -\frac{\eta}{\varepsilon N} [\mathbf{Z}(\mathbf{x}; \theta + \varepsilon \sum_i \nabla_\theta \mathcal{L}_i) - \mathbf{Z}(\mathbf{x}; \theta)]$, with $\varepsilon = 10^{-6}$. Only two forward passes are required (original parameters + perturbed parameters), at the cost of storing gradients in $O(|\theta|)$.
    - **Design Motivation**: Completely avoids backpropagation through new inputs; computation cost is fixed at two forward passes.

3. **Unified Perspective — Contrastive Methods as Special Cases**:

    - DiffMean is equivalent to COLD-Kernel with a specific loss $\mathcal{L} = -\sum_i \|\mathbf{Z}(\tilde{\mathbf{x}}_i \oplus \tilde{\mathbf{y}}_i^+) - \mathbf{Z}(\tilde{\mathbf{x}}_i \oplus \tilde{\mathbf{y}}_i^-)\|^2$ and a unit kernel.
    - RepE/ICV are approximations of COLD-Kernel via PCA dimensionality reduction.

### Loss & Training
- Paired settings use the DPO loss; positive-only settings use cross-entropy loss.
- Hyperparameter search: $\eta \in \{0.1, 1, 2\}$, $l \in \{10, 15, 20, 30\}$.
- For open-ended generation, intervention is applied only at the first generated token to limit compounding steering effects.

## Key Experimental Results

### Main Results (CAA Dataset, Llama-2-7b-chat, Behavioral Choice Accuracy)

| Method | Corrigible-AIS | Correct-HH | Hallucination | Refusal | Sycophancy | Avg. Rank↓ |
|--------|---------------|------------|---------------|---------|------------|------------|
| Base | 0.28 | 0.62 | 0.70 | 0.62 | 0.80 | 5.14 |
| DiffMean | 0.52 | 0.82 | 0.86 | 0.74 | 0.80 | 4.00 |
| ReFT(vec) | 0.48 | 0.62 | 0.70 | 0.72 | 0.82 | 3.29 |
| **COLD-FD** | **0.90** | **0.86** | **0.96** | **0.98** | 0.86 | **2.00** |
| COLD-Kernel | 0.28 | 0.62 | 0.70 | 0.64 | 0.80 | 4.43 |

### Sample Efficiency Comparison

| Method | Samples Required | Avg. Steering Accuracy |
|--------|-----------------|----------------------|
| ReFT(mlp) | 250–1000 | ~70–80% |
| DiffMean | 50 | ~65–75% |
| **COLD-FD** | **10–50** | **~85–95%** |
| **COLD-Kernel** | **10–50** | **~75–85%** |

### Key Findings
- **COLD-FD achieves an average rank of 2.00 on CAA** (paired setting), significantly outperforming all baselines.
- Using only **1/50** of the samples, COLD-FD approaches ReFT-level performance.
- DiffMean is formally shown to be a special case of COLD-Kernel under a specific loss — unifying contrastive and gradient-based methods.
- The approach is also effective on the OpinionsQA pluralistic alignment task, supporting adaptation toward minority viewpoints.
- Cross-model validation: COLD-FD achieves up to 96% accuracy improvement on Qwen-2.5-7B-Instruct; results are consistent on Gemma-2-9B and Mistral-7B.

### Pluralistic Alignment (OpinionsQA, Llama-2-7b-chat)
- COLD-Kernel consistently achieves best performance across all demographic groups, reducing KL divergence for the Black group from 2.43 to 0.86 and for Republicans from 2.38 to 0.97.
- TV distances are all reduced below 0.4, indicating that the kernel method better preserves subgroup distributional fidelity.
- COLD-FD underperforms in distributional steering settings; the underlying reason remains an open question.

### Generation Quality (Judged by GPT-4o-mini)
- COLD-FD improves scores on the CAA hallucination task from 2.98 to 3.32 (Llama-2-7b-chat) and on survival-instinct from 5.26 to 6.20.
- COLD-Kernel is more conservative, largely maintaining Base-level behavior, making it suitable for scenarios where large behavioral shifts are undesirable.

## Highlights & Insights
- **Reframing activation steering as a simulation of learning dynamics** is an elegant conceptual contribution — rather than training a steering module, the method directly computes "what fine-tuning would have done."
- **Strong theoretical unification**: DiffMean, RepE, and ICV are all proven to be special cases of COLD-Kernel, providing a unified gradient-based perspective on existing methods.
- **COLD-FD's two-forward-pass scheme** completely avoids backpropagation through new inputs, offering high practical utility.

## Limitations & Future Work
- COLD-FD requires storing the full model gradient $O(|\theta|)$, imposing significant memory pressure for 70B+ models.
- The unit kernel approximation underperforms on certain tasks (e.g., COLD-Kernel shows no improvement on Llama-2 for some behaviors).
- Only single-layer intervention is evaluated; multi-layer coordinated steering may yield stronger results.
- The choice of finite difference step size $\varepsilon$ relies on empirical tuning.

## Related Work & Insights
- **vs. CAA/DiffMean**: COLD-Steer demonstrates that contrastive methods exploit only activation difference signals, leaving loss function information unutilized.
- **vs. ReFT**: ReFT trains an MLP requiring hundreds of samples and multiple epochs; COLD-Steer requires zero training and as few as 10 samples.
- **vs. Prompt Tuning**: COLD-Steer operates at the activation level, enabling finer-grained control without being constrained by context window size.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Redefining activation steering as a simulation of learning dynamics represents a significant theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 LLMs, multiple datasets, and pluralistic alignment; ablation studies could be more extensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear and the unified perspective is convincing.
- Value: ⭐⭐⭐⭐⭐ A 50× improvement in sample efficiency carries substantial practical value, especially for pluralistic alignment.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](../../NeurIPS2025/optimization/doubly_robust_alignment_for_large_language_models.md)
- [\[NeurIPS 2025\] Constrained Network Slice Assignment via Large Language Models](../../NeurIPS2025/optimization/constrained_network_slice_assignment_via_llms.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](../../NeurIPS2025/optimization/vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[ICLR 2026\] Provable and Practical In-Context Policy Optimization for Self-Improvement](provable_and_practical_in-context_policy_optimization_for_self-improvement.md)

<!-- RELATED:END -->
