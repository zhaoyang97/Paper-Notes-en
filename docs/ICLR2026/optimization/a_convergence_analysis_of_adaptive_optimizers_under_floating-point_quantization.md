---
title: >-
  [Paper Note] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization
description: >-
  [ICLR 2026][Optimization][Low-precision training] This paper establishes the first theoretical framework for analyzing the convergence of adaptive optimizers under floating-point quantization. By imposing a relative error quantization model simultaneously on gradients, weights, and optimizer states (first and second moments), it proves that quantized Adam and Muon achieve the same $\tilde{O}(T^{-1/4})$ convergence rate as their full-precision counterparts when the mantissa length grows only logarithmically in the number of iterations. The analysis further reveals that Adam is highly sensitive to the quantization of weights and second moments, whereas Muon is theoretically more robust.
tags:
  - ICLR 2026
  - Optimization
  - Low-precision training
  - Adam
  - Muon
  - floating-point quantization
  - convergence analysis
date: 2026-05-08
content_hash: 3c0bcc04fd4157b2
---

# A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization

**Conference**: ICLR 2026
**arXiv**: [2510.21314](https://arxiv.org/abs/2510.21314)
**Code**: None
**Area**: Optimization
**Keywords**: Low-precision training, Adam, Muon, floating-point quantization, convergence analysis

## TL;DR
This paper establishes the first theoretical framework for analyzing the convergence of adaptive optimizers under floating-point quantization. By imposing a relative error quantization model simultaneously on gradients, weights, and optimizer states (first and second moments), it proves that quantized Adam and Muon achieve the same $\tilde{O}(T^{-1/4})$ convergence rate as their full-precision counterparts when the mantissa length grows only logarithmically in the number of iterations. The analysis further reveals that Adam is highly sensitive to the quantization of weights and second moments, whereas Muon is theoretically more robust.

## Background & Motivation
The rapid scaling of large language models (LLMs) has made low-precision training a critical technique for reducing memory footprint and improving efficiency. Low-precision formats such as BF16 and FP8 are already widely adopted in trillion-token-scale training runs (e.g., DeepSeek-V3, FP8-LM), with no significant accuracy degradation observed empirically.

However, **theoretical understanding lags far behind practice**. Existing convergence theories for quantized optimizers suffer from several critical gaps:

**Gradient quantization only**: Most theoretical work considers only gradient quantization in stochastic gradient descent (SGD), whereas modern low-precision training simultaneously quantizes weights, gradients, and optimizer states.

**Unrealistic assumptions**: Existing analyses either assume unbiased quantization or rely on error feedback mechanisms — the former does not reflect the behavior of floating-point quantization, and the latter is impractical at LLM scale due to memory overhead.

**Neglect of optimizer state quantization**: In practice, the first and second moments of Adam are also quantized to save memory (e.g., 8-bit Adam), yet this aspect is entirely absent from theoretical analyses.

**Absence of coverage for emerging optimizers**: Theoretical guarantees for matrix-based optimizers such as Muon under low precision are completely lacking.

**Core Problem**: Why do adaptive optimizers continue to converge effectively when all components are aggressively quantized?

## Method

### Overall Architecture
This paper proposes an **Analytical Low-Precision Training Framework** that explicitly models the following quantization operations:
- A master copy maintains full-precision weights $\mathbf{W}_t$, but transmits quantized versions $\mathbf{W}_t^Q$ to workers.
- Workers use $\mathbf{W}_t^Q$ for forward and backward passes, quantize the computed gradients, and send them back.
- The master dequantizes the gradients, updates the quantized optimizer states (first and second moments), applies the optimizer update, and requantizes for storage.

A defining feature of the framework is the use of a **Relative Error Model** in place of the conventional unbiased quantization assumption.

### Key Designs
1. **Relative Error Modeling of Floating-Point Quantization (Assumption 3.1)**: For any scalar $x$, the quantized value $x^Q$ satisfies $|x^Q - x| \leq q|x|$, where $q = \Theta(2^{-M})$ and $M$ is the mantissa length of the target floating-point format. **Design Motivation**: Floating-point quantization (e.g., FP32 → BF16) truncates mantissa bits while preserving the sign and exponent, so the quantization error is proportional to the magnitude of the value — a property precisely captured by the relative error model. This assumption is well satisfied in practice through per-tensor or per-channel scaling techniques.

2. **Component-wise Decomposition of Quantization Error**: The framework separately models and tracks the contribution of four quantization error terms to convergence:

    - $q_W$ (weight quantization error)
    - $q_G$ (gradient quantization error)
    - $q_M$ (first-moment/momentum quantization error)
    - $q_V$ (second-moment quantization error)

   This decomposition enables the theory to precisely characterize **the differential impact of quantizing each component on convergence**.

3. **Convergence Theorem for Quantized Adam (Theorem 4.5)**: Under standard assumptions (unbiased stochastic gradients, bounded gradients, $L$-smoothness), setting $\eta = \Theta(1/\sqrt{T})$, $1 - \beta_2 = \Theta(1/T)$, and quantization errors $q_G, q_M = O(1/T)$, $q_W, q_V = O(1/T^2)$, quantized Adam achieves a $\tilde{O}(T^{-1/4})$ convergence rate — matching the known optimal rate of full-precision Adam.

   **Key Finding**: Adam imposes stricter precision requirements on the second moment ($q_V$) and weights ($q_W$), which must satisfy $O(1/T^2)$, while the requirements on gradients and first moments are more lenient at $O(1/T)$. This asymmetry arises because, as $\beta_2 \to 1$, accumulated errors in the second moment are nonlinearly amplified through the inverse square root operation.

4. **Convergence Theorem for Quantized Muon (Theorem 4.6)**: For Muon, all components need only satisfy $q_G = q_W = q_M = O(T^{-1/2})$ to maintain the $O(T^{-1/4})$ convergence rate. This condition is **significantly weaker than that required by Adam** ($O(T^{-1/2})$ vs. $O(T^{-1})$ and $O(T^{-2})$). **Theoretical Explanation**: Muon employs an SVD-based sign operator, which avoids the amplification of quantization errors caused by the inverse square root of the second moment.

### Loss & Training
The theoretical analysis is conducted under the following standard assumptions:
- **Assumption 4.1**: Unbiased stochastic gradients.
- **Assumption 4.2**: Bounded gradients (Adam: $\ell_\infty$-bounded; Muon: bounded variance).
- **Assumption 4.3**: $L$-smooth objective function.
- **Assumption 4.4**: Bounded initialization.

Quantization is implemented in a simulated fashion: exponent and sign bits are preserved, mantissa bits are truncated to $M$ bits, and stochastic rounding is applied.

## Key Experimental Results

### Main Results (Synthetic Experiment — Rosenbrock Function)

| Optimizer | Mantissa length M | Convergence behavior | Gradient norm |
|-----------|-------------------|----------------------|---------------|
| Adam | M=23 (FP32) | Baseline, best convergence | Smallest |
| Adam | M=10 | Close to full precision | Slightly larger |
| Adam | M=7 (BF16) | Close to full precision | Slightly larger |
| Adam | M=3 | Slower convergence | Noticeably larger |
| Adam | M=1 | Severe degradation | Diverges |
| Muon | M=7 (BF16) | Close to full precision | Slightly larger |
| Muon | M=3 | Still converges | Minor degradation |
| Muon | M=2 | Begins to degrade | Noticeably larger |

### Real-Data Experiment (CIFAR-10, 4-layer Fully Connected Network)

| Optimizer | Mantissa length M | Gradient norm convergence | Comparison to full precision |
|-----------|-------------------|---------------------------|------------------------------|
| Adam | M≥7 | Close to full precision | Negligible gap |
| Adam | M=3 | Degraded | Visible gap |
| Adam | M=1–2 | Severely degraded | Unable to match |
| Muon | M≥3 | Close to full precision | Negligible gap |
| Muon | M=2 | Slight degradation | Small gap |

### Ablation Study

| Configuration | Key metric | Remarks |
|---------------|------------|---------|
| Gradient quantization only | Minimal impact | Gradients are most robust to quantization |
| Weight quantization only | Adam sensitive, Muon more robust | Validates the differential impact of $q_W$ |
| Second-moment quantization only | Adam most sensitive | $\beta_2 \to 1$ amplifies errors |
| First-moment quantization only | Moderate impact | Decay mechanism provides partial protection |
| Adam vs. Muon robustness | Muon more robust | Validates the theoretical prediction of $O(T^{-1/2})$ vs. $O(T^{-2})$ |

### Key Findings
- **Mantissa length needs only logarithmic growth**: $M = \Omega(\log T)$ is sufficient to guarantee full-precision convergence rates, which is fully consistent with the precision of existing hardware (BF16 with $M=7$, FP8 with $M=3$).
- **Adam's second moment and weights are the bottleneck**: $q_V$ and $q_W$ require $O(1/T^2)$ precision, while $q_G$ and $q_M$ require only $O(1/T)$ — validating the empirical observation in FP8-LM that the second moment benefits from slightly higher precision.
- **Muon requires weaker error control**: All components need only $O(T^{-1/2})$, providing a theoretical explanation for the empirical finding by Liu et al. (2025) that Muon performs better under low precision.
- **The relative error model is more principled than the unbiased assumption**: Floating-point quantization inherently satisfies the relative error property and requires no additional error feedback mechanism.

## Highlights & Insights
- **Fills an important theoretical gap**: This is the first work to provide convergence guarantees for adaptive optimizers — including both Adam and the emerging Muon — under a realistic floating-point quantization model.
- **Interpretable component-level sensitivity analysis**: The framework precisely quantifies the differential impact of each component on convergence, offering theoretical guidance for the design of mixed-precision training strategies (e.g., allocating higher precision to weights and second moments).
- **Quantitative comparison of Adam vs. Muon**: The theory provides a clear explanation of why Muon is more robust under low precision ($O(T^{-1/2})$ vs. $O(T^{-2})$), offering principled justification for optimizer selection.
- **Significant practical relevance**: The results directly establish the theoretical soundness of BF16 and FP8 training, providing formal backing for low-precision training practices in industry.
- **No reliance on error feedback**: Unlike prior theoretical frameworks that require per-parameter error feedback, the proposed framework more closely reflects actual large-scale training pipelines.

## Limitations & Future Work
- **Standard smoothness assumption**: The analysis assumes $L$-smoothness, whereas practical deep learning objectives may satisfy only the weaker $(L_0, L_1)$-smoothness condition, which the authors identify as a future direction.
- **Exact arithmetic assumption**: The analysis assumes that operations on quantized states are performed in exact arithmetic, without accounting for the additional errors introduced by low-precision operations such as FP8 matrix multiplication.
- **Communication efficiency not addressed**: Another important motivation for low-precision training is communication compression in distributed settings, which is not considered in this work.
- **Limited experimental scale**: Validation is conducted only on the Rosenbrock function and small networks on CIFAR-10; large-scale Transformer/LLM training experiments are absent.
- **The condition $q_W = O(1/T^2)$ may be overly strict**: The authors note that this condition arises from a worst-case treatment of unbounded weight norm growth in the proof, and can be relaxed to $O(1/T)$ in practical settings where the weight norm remains bounded.

## Related Work & Insights
- **Adaptive optimization theory**: The convergence analysis of full-precision Adam by Défossez et al. (2022) provides the theoretical backbone for this work.
- **Quantized SGD/SGDM**: QSGD by Alistarh et al. (2017) and subsequent error feedback methods (Karimireddy et al., 2019) address only the SGD setting.
- **Prior work on quantized Adam**: Chen et al. (2021) require error feedback; MicroAdam by Modoranu et al. (2024) ignores optimizer state quantization.
- **Low-precision training in practice**: Works such as DeepSeek-V3 (FP8), FP8-LM, and COAT demonstrate the practical feasibility of low-precision training.
- **Muon optimizer**: Proposed by Jordan et al. (2024) based on matrix SVD; full-precision convergence guarantees are provided by Shen et al. (2025).
- **Insights**: The relative error property of floating-point quantization constitutes a natural structural advantage over the absolute error of integer quantization, and should be fully exploited when designing quantization schemes. The $\beta_2 \to 1$ setting of Adam, while necessary for convergence, amplifies quantization errors — suggesting that particular attention must be paid to second-moment precision when designing quantization strategies for Adam.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)
- [\[ICLR 2026\] Convergence of Muon with Newton-Schulz](convergence_of_muon_with_newton-schulz.md)
- [\[AAAI 2026\] A Unified Convergence Analysis for Semi-Decentralized Learning: Sampled-to-Sampled vs. Sampled-to-All Communication](../../AAAI2026/optimization/a_unified_convergence_analysis_for_semi-decentralized_learni.md)
- [\[ICLR 2026\] Non-Asymptotic Analysis of Efficiency in Conformalized Regression](non-asymptotic_analysis_of_efficiency_in_conformalized_regression.md)
- [\[CVPR 2026\] Fed-ADE: Adaptive Learning Rate for Federated Post-adaptation under Distribution Shift](../../CVPR2026/optimization/fed-ade_adaptive_learning_rate_for_federated_post-adaptation_under_distribution_.md)

<!-- RELATED:END -->
