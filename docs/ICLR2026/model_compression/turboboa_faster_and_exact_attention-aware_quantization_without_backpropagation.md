---
title: >-
  [Paper Note] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation
description: >-
  [ICLR 2026][Model Compression][post-training quantization] TurboBoA proposes a backpropagation-free post-training quantization method for LLMs that achieves over 3× speedup over BoA while retaining its accuracy advantage…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "post-training quantization"
  - "attention-aware"
  - "backpropagation-free"
  - "low-bit quantization"
  - "LLM compression"
date: 2026-05-08
content_hash: a2a41260f1624659
---

# TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation

**Conference**: ICLR 2026
**arXiv**: [2602.04929](https://arxiv.org/abs/2602.04929)  
**Code**: [GitHub](https://github.com/SamsungLabs/TurboBoA)  
**Area**: Model Compression / Quantization / LLM
**Keywords**: post-training quantization, attention-aware, backpropagation-free, low-bit quantization, LLM compression

## TL;DR

TurboBoA proposes a backpropagation-free post-training quantization method for LLMs that achieves over 3× speedup over BoA while retaining its accuracy advantages, through three innovations: joint multi-output-channel quantization, preceding-layer error compensation, and adaptive grid selection.

## Background & Motivation

The rapid growth of LLM scale has made post-training quantization (PTQ) a critical technique for reducing memory and computational costs. Backpropagation-free methods guided by Hessian-based error compensation (e.g., GPTQ) have attracted wide attention due to their efficiency.

A fundamental trade-off exists between two families of methods:
- **GPTQ**: Assumes inter-layer independence, leading to significant accuracy degradation at low bit-widths (e.g., INT2).
- **BoA**: Improves the Hessian approximation by exploiting cross-layer dependencies within attention modules, substantially boosting accuracy, but requires sequential per-output-channel quantization, making it far less efficient than GPTQ.

Core Problem: **Can BoA's accuracy be preserved while dramatically improving its efficiency?**

## Method

### Overall Architecture

TurboBoA introduces three key innovations:
1. **Joint multi-output-channel quantization** (Feature 1) — eliminates the sequential bottleneck.
2. **Preceding-layer quantization error compensation** (Feature 2) — mitigates error accumulation.
3. **Adaptive grid selection + coordinate descent refinement** (Feature 3) — maintains alignment.

### Key Design 1: Joint Multi-Output-Channel Quantization

BoA quantizes output channels one by one (e.g., 128 sequential operations). TurboBoA simultaneously quantizes $N$ output channels.

The error compensation problem is formulated as:

$$\min_{\Delta\mathbf{W}} \|\mathbf{G}\Delta\mathbf{W}\mathbf{X}\|_F^2, \quad \text{s.t. } \mathbf{e}_i^T \Delta\mathbf{W} = \mathbf{Q}_{i,:} - \mathbf{W}_{i,:} \; (0 \leq i < N)$$

**Proposition 3.1** (Closed-form solution):

$$[\Delta\mathbf{W}]_{N:,:} = -[\mathbf{U}_{out}^T]_{N:,B}[\mathbf{U}_{out}^T]_{B,B}^{-1}(\mathbf{W}_{B,:} - \mathbf{Q}_{B,:})$$

where $B = \{0, \ldots, N-1\}$ and $\mathbf{U}_{out} = \text{Chol}(\mathbf{H}_{out}^{-1})^T$.

At $N=16$, this yields over 3× speedup over BoA (128→8 sequential operations) with negligible accuracy loss.

### Key Design 2: Preceding-Layer Quantization Error Compensation

Quantization errors in preceding layers propagate to subsequent layers, a factor BoA does not account for. TurboBoA explicitly models the input deviation $\Delta\mathbf{X} = \mathbf{X} - \tilde{\mathbf{X}}$:

$$\mathbf{G}\mathbf{Q}\mathbf{X} - \mathbf{G}\mathbf{W}\tilde{\mathbf{X}} = \mathbf{G}\Delta\mathbf{W}\mathbf{X} + \mathbf{G}\mathbf{W}\Delta\mathbf{X}$$

**Proposition 3.2** (Closed-form solution with preceding-layer error):

$$[\Delta\mathbf{W}]_{N:,:} = -[\mathbf{U}_{out}^T]_{N:,B}[\mathbf{U}_{out}^T]_{B,B}^{-1}\left((\mathbf{W}_{B,:} - \mathbf{Q}_{B,:}) - \mathbf{W}_{B,:}\mathbf{R}\mathbf{H}_{in}^{-1}\right)$$

where $\mathbf{R} = \Delta\mathbf{X}\mathbf{X}^T$. In contrast to GPTAQ's vector-level optimization, TurboBoA handles the general dense $\mathbf{H}_{out}$.

### Key Design 3: Adaptive Grid + Coordinate Descent Refinement

- **Adaptive grid**: The quantization grid is computed on-the-fly before each quantization step to ensure alignment with updated weights.
- **Coordinate descent refinement**: The integer weights $\mathbf{W}_{int}$ are frozen, and only the scales are optimized:

$$\min_{\mathbf{s}} \|\mathbf{G}(\text{diag}(\mathbf{s})\mathbf{W}_{int} - \mathbf{W})\mathbf{X} + \mathbf{G}\mathbf{W}\Delta\mathbf{X}\|_F^2$$

**Proposition 3.3** (CD update rule):

$$s_j^* = s_j + \frac{[\mathbf{W}_{int}(\mathbf{H}_{in}(\mathbf{W}-\mathbf{Q})^T - \mathbf{R}^T\mathbf{W}^T)\mathbf{H}_{out}]_{j,j}}{[\mathbf{W}_{int}\mathbf{H}_{in}\mathbf{W}_{int}^T]_{j,j}[\mathbf{H}_{out}]_{j,j}}$$

### Loss & Training

The standard attention reconstruction error is employed, based on the Kronecker-structured Hessian $\mathbf{H} = \mathbf{H}_{in} \otimes \mathbf{H}_{out}$.

## Key Experimental Results

### Main Results: INT2 Quantization Speed

| Method | N | Llama3-8B Time | Wiki2 PPL |
|--------|---|----------------|-----------|
| BoA | 1 | 94.75 min | 15.20 |
| TurboBoA | 4 | 39.46 min | 15.27 |
| TurboBoA | 8 | 30.55 min | 15.30 |
| TurboBoA | **16** | **25.30 min** | **15.41** |
| TurboBoA | 32 | 22.95 min | 15.22 |

For 70B models: BoA requires 17 hours, while TurboBoA ($N=16$) requires only 5.6 hours, saving approximately 11 hours.

### Ablation Study: Three Features

| Method | F2 | F3 | Llama3-8B Wiki2↓ | C4↓ |
|--------|----|----|------------------|-----|
| BoA | - | - | 15.20 | 36.95 |
| TurboBoA (F1 only) | ✗ | ✗ | 15.41 | — |
| TurboBoA (F1+F2) | ✓ | ✗ | Improved | — |
| TurboBoA (All) | ✓ | ✓ | Best | Best |

### SOTA Results

When combined with outlier suppression techniques such as QuaRot:
- **Weight-only quantization**: Comprehensively outperforms GPTQ, BoA, and other methods at INT2.
- **Weight-activation quantization**: Also achieves state-of-the-art performance.

### Key Findings

1. Accuracy degradation remains negligible as $N$ increases to 64, indicating that the remaining output channels provide sufficient error compensation capacity.
2. Speedup gains diminish beyond $N > 16$, making $N=16$ the optimal trade-off point.
3. Preceding-layer error compensation and grid refinement each contribute independently and complementarily.

## Highlights & Insights

- All three Propositions provide closed-form solutions, resulting in theoretically elegant formulations.
- Over 3× speedup is achieved with accuracy on par with or better than BoA.
- The method is not tied to a specific Hessian formulation and can directly accommodate more advanced Hessian estimates.
- Over 11 hours of quantization time are saved for 70B models, demonstrating substantial practical value.

## Limitations & Future Work

- Validation is limited to the Llama model family; other architectures (e.g., Mixtral, Qwen) are not evaluated.
- Although the choice of $N$ is empirically robust, a theoretical error bound analysis is lacking.
- The stabilization coefficient $\alpha$ requires manual tuning from the set $\{0.05, 0.125, 0.25\}$.
- The method focuses exclusively on attention layer quantization; FFN layers rely on standard GPTQ.

## Related Work & Insights

- **Backpropagation-free quantization**: GPTQ (Frantar et al., 2023), BoA (Kim et al., 2025), GPTAQ (Li et al., 2025)
- **Transformation-based methods**: SmoothQuant (Xiao et al., 2023), QuaRot (Ashkboos et al., 2024)
- **Early PTQ works**: AdaRound (Nagel et al., 2020), BRECQ (Li et al., 2021)

## Rating

- Novelty: ⭐⭐⭐⭐ — The closed-form solution for joint quantization is the core contribution.
- Theoretical Depth: ⭐⭐⭐⭐⭐ — Three Propositions are complete and rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-scale model evaluation with comprehensive ablations.
- Value: ⭐⭐⭐⭐⭐ — Directly addresses the efficiency bottleneck of BoA.
- Writing Quality: ⭐⭐⭐⭐ — Notation is clear and mathematical derivations are detailed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compute-Optimal Quantization-Aware Training](compute-optimal_quantization-aware_training.md)
- [\[ICLR 2026\] FASA: Frequency-aware Sparse Attention](fasa_frequency-aware_sparse_attention.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[ICLR 2026\] ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](paroquant_pairwise_rotation_quantization_for_efficient_reasoning_llm_inference.md)

</div>

<!-- RELATED:END -->
