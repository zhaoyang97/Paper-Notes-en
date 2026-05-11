---
title: >-
  [Paper Note] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization
description: >-
  [NeurIPS 2025][Model Compression][post-training quantization] This paper identifies a critical bottleneck in existing layer-wise PTQ methods—namely…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "post-training quantization"
  - "LLM compression"
  - "quantization error propagation"
  - "layer-wise quantization"
  - "low-bit"
date: 2026-05-08
content_hash: b137364ee4071b3c
---

# Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization

**Conference**: NeurIPS 2025
**arXiv**: [2504.09629](https://arxiv.org/abs/2504.09629)
**Code**: [GitHub](https://github.com/FujitsuResearch/qep)
**Area**: Model Compression
**Keywords**: post-training quantization, LLM compression, quantization error propagation, layer-wise quantization, low-bit

## TL;DR

This paper identifies a critical bottleneck in existing layer-wise PTQ methods—namely, their neglect of cross-layer accumulation and growth of quantization errors—and proposes the QEP framework, which explicitly corrects accumulated errors via error propagation and compensation, achieving substantial performance gains under extremely low-bit settings (INT2/INT3).

## Background & Motivation

- **Background**: Layer-wise post-training quantization (PTQ) has become the dominant paradigm for LLM compression due to its simplicity and efficiency, as exemplified by GPTQ, AWQ, and QuIP. However, recent progress in this direction has begun to plateau.
- **Limitations of Prior Work**: Existing methods treat the quantization of each layer as an independent optimization problem (minimizing $\|W_l X_l - \hat{W}_l X_l\|_F^2$), neither accounting for quantization errors propagated from upstream layers nor correcting already-accumulated errors. Experiments show that quantization errors grow approximately exponentially across layers and continue to grow even in unquantized layers.
- **Key Challenge**: The fundamental issue lies in the layer-wise independent optimization formulation, which ignores inter-layer error dynamics.
- **Goal**: To revisit the core design of layer-wise PTQ and develop a general, lightweight framework that explicitly models and compensates for cross-layer quantization error accumulation.

## Method

### Overall Architecture

QEP is a general, lightweight framework that integrates seamlessly with any existing layer-wise PTQ method. The core idea is to reformulate the per-layer independent optimization into a joint optimization that accounts for error propagation, using a weight correction term to compensate for accumulated quantization errors from preceding layers.

### Key Designs

1. **Problem Reformulation**: The original objective $\min \|W_l X_l - \hat{W}_l X_l\|_F^2$ (sharing the same input $X_l$) is replaced by $\min \|W_l X_l - \hat{W}_l \hat{X}_l\|_F^2$, where $X_l$ is the full-precision input and $\hat{X}_l$ is the quantized input. Under this formulation, the quantized weights must not only approximate the full-precision weights but also compensate for upstream accumulated quantization errors. The key distinction is that the trivial optimal solution of the original objective is $\hat{W}_l = W_l$, whereas the optimal solution of the new objective generally satisfies $\hat{W}_l \neq W_l$, thereby achieving explicit error correction.

2. **Weight Correction (Proposition 5.1)**: The optimal solution after continuous relaxation is derived in closed form: $W_l^* = W_l + W_l \delta_l \hat{X}_l^T \hat{H}_l^{-1}$, where $\delta_l = X_l - \hat{X}_l$ is the accumulated quantization error and $\hat{H}_l = \hat{X}_l \hat{X}_l^T$ is the Hessian computed from quantized activations. The corrected weights preserve the same quadratic optimization structure as the original PTQ (Eq. 7), enabling direct reuse of existing Hessian-based acceleration methods.

3. **Controlling Propagation Strength**: A tunable parameter $\alpha_l \in [0,1]$ is introduced: $W_l^*(\alpha_l) = W_l + \alpha_l W_l \delta_l \hat{X}_l^T \hat{H}_l^{-1}$. Setting $\alpha_l=1$ recovers full correction, while $\alpha_l=0$ reduces to the original method. Proposition 5.3 establishes that $\alpha_l$ is equivalent to a regularization parameter, which effectively prevents overfitting—particularly important for MLP layers with large parameter counts. For large models (e.g., Llama-2 70B), setting $\alpha_l=0$ for MLP layers both reduces computational overhead (by approximately one-third to one-half) and provides implicit regularization.

### Loss & Training

QEP does not introduce any new training procedure. The core optimization objective (Eq. 7) is: $\min_{\hat{W}_l} \|W_l^* \hat{X}_l - \hat{W}_l \hat{X}_l\|_F^2$, which is structurally identical to the original layer-wise PTQ objective, with $W_l$ replaced by the corrected $W_l^*$. Theorem 5.2 provides a theoretical guarantee that the output quantization error of QEP satisfies $\|f_\theta(X) - f_{\hat{\theta}_{QEP}}(X)\|_F \leq \|f_\theta(X) - f_{\hat{\theta}_{BASE}}(X)\|_F$.

## Key Experimental Results

### Main Results

| Model | Method | INT4 PPL | INT3 PPL | INT2 PPL |
|-------|--------|----------|----------|----------|
| Llama-2-7B | QuIP | 8.434 | 12.048 | 65.593 |
| Llama-2-7B | QuIP+QEP | 5.753 | 6.154 | 11.972 |
| Llama-2-7B | GPTQ | 6.083 | 10.881 | 13051.5 |
| Llama-2-7B | GPTQ+QEP | 5.933 | 7.898 | 7214.3 |
| Llama-2-7B | AWQ | 5.831 | 15.299 | 199448.8 |
| Llama-2-7B | AWQ+QEP | 5.756 | 11.131 | - |
| Llama-3-8B | QuIP | 6.998 | 8.288 | 70.518 |
| Llama-3-8B | QuIP+QEP | 6.650 | 7.703 | 27.326 |
| Llama-2-7B | FP16 Baseline | 5.472 | 5.472 | 5.472 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| BASE (w/o QEP) | Standard PTQ error | Error grows approximately exponentially |
| With QEP | Error significantly reduced | Error growth effectively suppressed after the first 10 blocks |
| $\alpha_l=0$ (MLP) | Reduced compute + regularization | Particularly important for 70B models |
| $\alpha_l=1/2$ (default) | Balanced correction and regularization | Recommended for most models |
| RTN+QEP INT3 | 539.9 → 17.3 | Substantial gain even on the simplest baseline |

### Key Findings

- QEP yields consistent improvements across all tested methods (RTN/GPTQ/AWQ/QuIP) and all bit-widths (INT2/3/4), demonstrating strong generality.
- Gains are most pronounced in low-bit settings: QuIP INT2 perplexity drops from 65.6 to 12.0 on Llama-2-7B, approaching practical usability.
- GPTQ INT3 perplexity drops from 10.9 to 7.9 on Llama-2-7B, even surpassing the original AWQ INT3 result of 15.3.
- The quantization error propagation visualization (Figure 2) intuitively demonstrates that QEP effectively suppresses error accumulation and growth across both quantized and unquantized layers.
- Consistent improvements on zero-shot tasks confirm that perplexity gains translate to downstream task performance.

## Highlights & Insights

- **Unique Perspective**: The paper revisits the most fundamental optimization objective of layer-wise PTQ and uncovers a previously overlooked error accumulation problem.
- **Theory–Practice Unity**: Proposition 5.1 provides a closed-form correction solution, Theorem 5.2 offers theoretical guarantees, and the additional computational cost remains manageable.
- **Elegant $\alpha_l$ Design**: The parameter simultaneously prevents overfitting and controls computation; Proposition 5.3 reveals its equivalence to regularization.
- **Orthogonality**: QEP is orthogonal to all existing PTQ improvements (non-linear quantization, rotation matrices, etc.) and can be combined with them additively.

## Limitations & Future Work

- The current $\alpha_l$ strategy uses simple fixed values (1/2, or 0 for MLP layers); future work could develop adaptive, per-layer, data-aware tuning strategies.
- In INT2 settings, AWQ+QEP does not always outperform QuIP+QEP, indicating varying compatibility between different PTQ methods and QEP.
- The correction term computation depends on the quality and size of the calibration dataset; small calibration sets may lead to overfitting.
- The combination of QEP with block-wise PTQ or quantization-aware training (QAT) remains unexplored.

## Related Work & Insights

- The Hessian-based layer-wise quantization pioneered by GPTQ is the direct target of improvement in this work.
- AWQ's salience-based weight scaling and QuIP's rotation preprocessing are orthogonal techniques that can be stacked with QEP.
- The concept of error propagation bears resemblance to mechanisms in deep learning training (e.g., batch normalization mitigating gradient vanishing), but is applied to PTQ for the first time here.

## Supplementary Discussion

- The core insight of QEP can be analogized as follows: original layer-wise PTQ resembles editing a video frame by frame while ignoring inter-frame coherence; QEP introduces an inter-frame error compensation mechanism.
- Only one additional matrix multiplication $\delta_l \hat{X}_l^T$ is required to achieve significant quantization accuracy improvements, yielding an extremely favorable cost-benefit ratio.
- On Llama-3-8B INT4, GPTQ perplexity drops from 147.9 to 9.5, indicating severe error accumulation in the original GPTQ on this model.
- QuIP+QEP at INT2 (PPL 11.97) even outperforms GPTQ at INT3 (PPL 10.88), demonstrating the significant potential of ultra-low-bit quantization.
- The error propagation visualization in Figure 2 constitutes the most compelling evidence in the paper, intuitively showing how errors continue to grow in unquantized layers.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Identifies a previously overlooked problem by revisiting the most fundamental optimization objective; the solution is elegant yet had not been proposed before.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple model scales (7B–70B), multiple methods (4 PTQ approaches), and multiple bit-widths (INT2/3/4).
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain from problem identification to theoretical analysis to experimental validation is complete and coherent.
- **Value**: ⭐⭐⭐⭐⭐ As a general plug-in that substantially improves all layer-wise PTQ methods—especially under extremely low-bit settings—the practical value is significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](../../ICLR2026/model_compression/ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[NeurIPS 2025\] LittleBit: Ultra Low-Bit Quantization via Latent Factorization](littlebit_ultra_low-bit_quantization_via_latent_factorization.md)
- [\[NeurIPS 2025\] Binary Quadratic Quantization: Beyond First-Order Quantization for Real-Valued Matrix Compression](binary_quadratic_quantization_beyond_first-order_quantization_for_real-valued_ma.md)
- [\[NeurIPS 2025\] DP-LLM: Runtime Model Adaptation with Dynamic Layer-wise Precision Assignment](dp-llm_runtime_model_adaptation_with_dynamic_layer-wise_precision_assignment.md)

</div>

<!-- RELATED:END -->
