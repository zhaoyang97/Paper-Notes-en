---
title: >-
  [Paper Note] First-Order Error Matters: Accurate Compensation for Quantized Large Language Models
description: >-
  [AAAI 2026][Model Compression][Post-training quantization] This paper identifies a critical yet overlooked issue in LLM post-training quantization: the column-wise compensation process renders first-order gradient terms…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Post-training quantization"
  - "large language models"
  - "first-order error compensation"
  - "GPTQ"
  - "weight quantization"
date: 2026-05-08
content_hash: 69d383b331ea4bbc
---

# First-Order Error Matters: Accurate Compensation for Quantized Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2507.11017](https://arxiv.org/abs/2507.11017)  
**Code**: [https://github.com/Xingyu-Zheng/FOEM](https://github.com/Xingyu-Zheng/FOEM)  
**Area**: Model Compression
**Keywords**: Post-training quantization, large language models, first-order error compensation, GPTQ, weight quantization

## TL;DR

This paper identifies a critical yet overlooked issue in LLM post-training quantization: the column-wise compensation process renders first-order gradient terms non-negligible. The proposed FOEM method incorporates first-order terms into the error compensation formula, reducing the perplexity of Llama3-8B under 3-bit quantization by 17.3% with virtually no additional computational overhead.

## Background & Motivation

### State of the Field

Large language models (LLMs) have achieved remarkable success in language understanding, dialogue systems, and code generation, yet their massive parameter counts impose substantial memory and computational burdens. Quantization, a classical model compression technique, reduces memory usage and accelerates inference by converting high-precision floating-point parameters into low-bit fixed-point representations. Among various quantization approaches, post-training quantization (PTQ) is particularly well-suited for large models as it requires no gradient-based fine-tuning.

### Limitations of Prior Work

GPTQ is a representative method for LLM weight quantization that models quantization error using the second-order term of a Taylor expansion and performs column-wise weight quantization via the Hessian matrix, compensating for errors in subsequent columns. **However, these methods universally assume that the full-precision model is already sufficiently optimized, rendering the first-order term negligible.**

### Root Cause

This paper identifies a critical overlooked issue: during the column-wise quantize-and-compensate process, the compensation applied to previously quantized columns introduces a significant deviation between the latent weights $\mathbf{W}$ of subsequent unquantized columns and their original full-precision counterparts $\mathbb{W}$. This accumulated deviation means that subsequent columns carry non-negligible gradients $g = \frac{\partial E}{\partial w}$ at the time of quantization, making the assumption of ignorable first-order terms fundamentally flawed.

### Starting Point

Since first-order terms cannot be ignored, they should be explicitly incorporated into the quantization error compensation. However, directly computing gradients via backpropagation is prohibitively expensive for large models. FOEM cleverly approximates the gradient using the relationship between weight deviation and the Hessian, and upon substitution into the theoretical solution, the Hessian terms cancel out, leaving only a simple weight-difference computation.

## Method

### Overall Architecture

FOEM follows the column-wise quantization framework of GPTQ but introduces a first-order gradient correction into the compensation term. The overall pipeline is: compute Hessian matrix → Cholesky decomposition → column-wise quantization → **error compensation with first-order correction** → update subsequent column weights.

### Key Designs

#### 1. Theoretical Derivation of First-Order Terms

Retaining the first-order term in the Taylor expansion yields the following quantization error model:

$$\delta E = g \delta w^\top + \frac{1}{2} \delta w \mathbf{H} \delta w^\top$$

Solving the constrained optimization problem via Lagrange multipliers gives the optimal compensation:

$$\delta w = -\frac{w_q - \hat{w}_q - g H^{-1} e_q^\top}{\mathbf{T}_{qq}} \mathbf{T}_{q,q:} - g \mathbf{H}^{-1}$$

Compared to the GPTQ solution (which contains only the first term without gradient contributions in the numerator and omits the second term entirely), FOEM introduces additional gradient-dependent corrections.

- **Design Motivation**: Accumulated weight deviations during the compensation process produce non-negligible gradients; ignoring them leads to suboptimal compensation.

#### 2. Efficient Gradient Approximation

Directly computing gradients via backpropagation is impractical. FOEM approximates the gradient using a first-order Taylor expansion:

$$g(\mathbf{W}) \approx g(\mathbb{W}) + (\mathbf{W} - \mathbb{W})\mathbf{H}$$

Since the original full-precision weights are optimized to a local minimum, $g(\mathbb{W}) \approx 0$, and thus:

$$g(\mathbf{W}) \approx (\mathbf{W} - \mathbb{W})\mathbf{H}$$

A stabilization factor $\beta$ (default 0.1) is introduced:

$$g = \beta (\mathbf{W} - \mathbb{W}) \mathbf{H}$$

- **Mechanism**: The gradient is approximated by multiplying the weight deviation by the Hessian, eliminating the need for backpropagation.
- **Design Motivation**: $\beta$ controls the magnitude of the gradient correction, preventing amplification of errors due to overly large approximate gradients.

#### 3. Elimination of Hessian Terms

Substituting the gradient approximation into the theoretical solution yields the final compensation formula:

$$\delta w = -\frac{(w_q - \hat{w}_q) - \beta(w_q - \mathbb{W} e_q^\top)}{\mathbf{T}_{qq}} \mathbf{T}_{q,q:} - \beta(\mathbf{W} - \mathbb{W})$$

A key finding is that **the Hessian $\mathbf{H}$ and its inverse $\mathbf{H}^{-1}$ cancel completely upon substitution**. This implies:
- No explicit computation or inversion of the Hessian is required for the gradient term.
- Only a simple weight-difference computation ($\mathbf{W} - \mathbb{W}$) is needed.
- The existing Cholesky decomposition from GPTQ can be reused.
- Strategies such as GPTQ's lazy update mechanism remain fully compatible.

### Loss & Training

FOEM is a post-training quantization method and involves no training. Calibration uses 128 samples from the C4 dataset (sequence length 2048). The sole hyperparameter $\beta$ is fixed at 0.1 and demonstrates robust performance across all model architectures and quantization configurations.

## Key Experimental Results

### Main Results

**Llama3-8B 3-bit weight quantization (WikiText2 perplexity)**:

| Method | WikiText2 PPL↓ | C4 PPL↓ | 0-shot Avg↑ | MMLU↑ |
|--------|----------------|---------|-------------|-------|
| FP16 | 6.13 | 9.61 | 70.7 | 64.9 |
| RTN | 13.10 | 20.50 | 57.2 | 38.9 |
| GPTQ | 9.86 | 12.94 | 64.4 | 55.4 |
| GPTAQ | 8.92 | 12.82 | 63.3 | 53.8 |
| **FOEM** | **8.32** | **12.37** | **65.5** | **56.1** |

**Llama2-7B 3-bit weight quantization**:

| Method | WikiText2 PPL↓ | C4 PPL↓ | 0-shot Avg↑ | MMLU↑ |
|--------|----------------|---------|-------------|-------|
| FP16 | 5.48 | 6.90 | 66.9 | 45.8 |
| GPTQ | 6.38 | 7.85 | 63.5 | 40.3 |
| GPTAQ | 6.41 | 7.93 | 64.0 | 35.1 |
| **FOEM** | **6.27** | **7.81** | **64.5** | **42.0** |

**W4A4KV4 quantization (combined with SpinQuant)**:

| Model | Method | WikiText2↓ | C4↓ | 0-shot Avg↑ |
|-------|--------|-----------|-----|-------------|
| Llama3-8B | GPTQ | 8.55 | 13.24 | 64.1 |
| Llama3-8B | GPTAQ | 8.50 | 13.13 | 64.1 |
| Llama3-8B | **FOEM** | **8.35** | **12.94** | **64.3** |

### Ablation Study

**Sensitivity analysis of $\beta$ (Llama3-8B 3-bit)**:

| $\beta$ | WikiText2↓ | C4↓ | 0-shot Avg↑ | Note |
|---------|-----------|-----|-------------|------|
| 0 (GPTQ) | 9.86 | 12.94 | 64.4 | No first-order correction |
| 0.1 | **8.32** | **12.37** | 65.5 | Default setting |
| 0.2 | 8.87 | 12.90 | 65.7 | Still improved |
| 0.3 | 8.43 | 12.88 | **65.8** | Slight PPL increase but best Avg |
| 0.5 | 8.52 | 12.51 | 64.9 | Performance begins to decline |
| 0.8 | 10.09 | 13.09 | 61.5 | Significant degradation |
| 1.0 | 10.37 | 14.41 | 61.5 | Complete degradation |

**Efficiency analysis (Llama3-8B W4A4KV4)**:

| Method | Quantization Time (s) | WikiText2↓ |
|--------|----------------------|-----------|
| GPTQ | 825.5 | 8.55 |
| GPTAQ | 1112.2 | 8.50 |
| FOEM | 828.9 | 8.35 |

### Key Findings

1. **First-order terms are non-negligible**: Accumulated weight deviations during column-wise compensation introduce significant gradients in subsequent columns; ignoring them leads to suboptimal quantization.
2. **Stable and effective for $\beta < 0.5$**: Performance degrades significantly beyond 0.5, consistent with theoretical analysis — excessively large gradients amplify approximation errors.
3. **Negligible computational overhead**: Quantization time is nearly identical to GPTQ (828.9s vs. 825.5s), whereas GPTAQ requires 1112.2s.
4. **Cross-architecture generalization**: Effective across both Transformer architectures (Llama/Qwen/Mistral/Phi) and SSMs (Mamba).
5. **Complementary to SpinQuant**: Additional gains are achieved even under the aggressive W4A4KV4 quantization setting.

## Highlights & Insights

1. **Identifying an overlooked issue**: All compensation-based PTQ methods assume first-order terms are negligible, yet the compensation process itself violates this assumption — an elegant and precise observation.
2. **Mathematically elegant cancellation**: Approximating the gradient via Taylor expansion and substituting into the Lagrangian solution causes the Hessian terms to cancel exactly — a result of natural derivation rather than deliberate design.
3. **Significant gains from minimal modification**: Simply subtracting an additional weight-difference term in GPTQ's compensation formula yields substantial performance improvements.
4. **Rigorous theoretical grounding**: The paper forms a complete loop from mathematical derivation to experimental validation.

## Limitations & Future Work

1. **Fixed $\beta = 0.1$**: Although experiments show robustness across $\beta < 0.5$, adaptive mechanisms for selecting $\beta$ remain unexplored.
2. **Gradient approximation assumption**: The assumption $g(\mathbb{W}) \approx 0$ may not hold perfectly in practice, particularly for insufficiently pre-trained models.
3. **Higher-order terms beyond second order**: Only the first-order term is corrected; the cumulative effect of higher-order terms has not been considered.
4. **Unexplored ultra-low-bit (2-bit) scenarios**: Under more aggressive quantization, accumulated errors may be more pronounced, and first-order correction alone may be insufficient.

## Related Work & Insights

- **GPTQ** → The OBS-based column-wise quantization and compensation framework that serves as the foundation of this work.
- **GPTAQ** → An extension that incorporates true gradient information, but incurs significant computational overhead (+35% time); FOEM achieves superior results using approximate gradients.
- **SpinQuant** → A rotation-based activation quantization method that integrates seamlessly with FOEM.
- **Insight**: For any compensation-based iterative optimization method, the assumption that "first-order terms are negligible" deserves reexamination — compensation inherently reshapes the optimization landscape.

## Rating

- Novelty: ⭐⭐⭐⭐ (Precise observation, elegant mathematical derivation, though the improvement builds upon an existing framework)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers multiple model families, quantization configurations, efficiency analysis, sensitivity analysis, and cross-architecture validation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear mathematical derivations, complete logical chain)
- Value: ⭐⭐⭐⭐ (Plug-and-play improvement to quantization methods with high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](../../NeurIPS2025/model_compression/restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[AAAI 2026\] Failures to Surface Harmful Contents in Video Large Language Models](failures_to_surface_harmful_contents_in_video_large_language_models.md)
- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](../../CVPR2026/model_compression/quant_experts_token_aware_vlm_quantization.md)
- [\[AAAI 2026\] Error Correction in Radiology Reports: A Knowledge Distillation-Based Multi-Stage Framework](error_correction_in_radiology_reports_a_knowledge_distillation-based_multi-stage.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)

</div>

<!-- RELATED:END -->
