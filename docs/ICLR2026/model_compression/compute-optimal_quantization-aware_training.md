---
title: >-
  [Paper Note] Compute-Optimal Quantization-Aware Training
description: >-
  [ICLR 2026][Model Compression][Quantization-Aware Training] Through 757 QAT experiments spanning 86M–2.2B parameters and 1–6 bits, this paper demonstrates that the optimal QAT training fraction grows with total compute budget—contradicting the previously held belief that 10% is universally optimal—and proposes the tokens-per-parameter-byte statistic along with a new loss scaling law to accurately predict the optimal QAT allocation strategy and final loss across all configurations.
tags:
  - ICLR 2026
  - Model Compression
  - Quantization-Aware Training
  - Scaling Law
  - Compute-Optimal Allocation
  - tokens-per-parameter-byte
  - Low-Bit Quantization
date: 2026-05-08
content_hash: 3e3cfd5fe90183dc
---

# Compute-Optimal Quantization-Aware Training

**Conference**: ICLR 2026
**arXiv**: [2509.22935](https://arxiv.org/abs/2509.22935)
**Code**: None
**Area**: Model Compression / LLM Efficiency
**Keywords**: Quantization-Aware Training, Scaling Law, Compute-Optimal Allocation, tokens-per-parameter-byte, Low-Bit Quantization

## TL;DR
Through 757 QAT experiments spanning 86M–2.2B parameters and 1–6 bits, this paper demonstrates that the optimal QAT training fraction grows with total compute budget—contradicting the previously held belief that 10% is universally optimal—and proposes the tokens-per-parameter-byte statistic along with a new loss scaling law to accurately predict the optimal QAT allocation strategy and final loss across all configurations.

## Background & Motivation

**State of the Field**: QAT is the dominant approach for training high-quality quantized models, typically following a two-stage pipeline of full-precision (FP) pretraining followed by QAT fine-tuning. Liu et al. (2025) recommend allocating 10% of total training steps to the QAT phase.

**Limitations of Prior Work**:
   - The "10% optimum" conclusion was derived under limited compute budgets and has not been validated at larger scales
   - PTQ-induced quantization error grows with the amount of pretraining data (Kumar et al.), suggesting that the FP-to-QAT allocation should be scale-dependent
   - Existing QAT scaling laws (Chen et al.) assume $D_{fp}=0$ (i.e., QAT from scratch) and do not address the two-stage FP→QAT scenario
   - No unified scaling law exists across different bit-widths

**Root Cause**: Too few QAT steps leave the model unable to adapt to low-precision arithmetic; too many QAT steps compress the FP stage and extend training with noisy gradients. How does this trade-off shift as total compute grows?

**Paper Goals**:
   - How does the optimal QAT fraction vary with model size, total token count, and bit-width?
   - Can a single unified scaling law predict final loss across all configurations?
   - Can the training pipeline be further streamlined (e.g., by merging the cooldown and QAT phases)?

**Starting Point**: The paper introduces tokens-per-parameter-byte, $S = D/(N \cdot B/8)$, as a unified scaling variable that simultaneously encodes model size, data volume, and quantization precision.

**Core Idea**: The optimal time allocation for QAT is not a fixed 10% but rather a function that grows with tokens-per-parameter-byte, and can be accurately modeled by a unified scaling law.

## Method

### Overall Architecture
Two-stage training: FP phase ($D_{fp}$ tokens) → QAT phase ($D_{qat}$ tokens), totaling $D_{total} = D_{fp} + D_{qat}$. The central question is: given $N$, $D_{total}$, and $B$, what is the optimal fraction $f^* = D_{qat}^*/D_{total}$?

### Key Designs

1. **Tokens-per-Parameter-Byte Statistic**:

    - Function: Unifies the prediction of the optimal QAT allocation across different model sizes and bit-widths.
    - Mechanism: $S_{total} = D_{total}/(N \cdot B/8)$ normalizes the total token count by the quantized byte size of the model parameters. Larger models are easier to quantize ($N$ large → $S$ small); lower bit-widths are harder to quantize ($B$ small → $S$ large); longer training makes quantization harder ($D$ large → $S$ large).
    - Design Motivation: As shown in Figure 2, optimal points for different bit-widths are scattered when plotted against raw token counts, but collapse onto a single curve when plotted against tokens-per-parameter-byte.

2. **Optimal QAT Fraction Prediction**:

    - Function: Directly fits the relationship between the optimal QAT fraction and $S_{total}$.
    - Mechanism: $\hat{f}(D_{total}, N, B) = \frac{\exp(\log S_{total} - a/\log S_{total})}{S_{total}}$, where $a=6.7297$ is the sole fitted parameter.
    - Design Motivation: The relationship between $S_{total}$ and the optimal $S_{qat}$ is approximately linear in log-log space; a constraint $D_{qat} \leq D_{total}$ is imposed. The resulting MAE is only 0.091.

3. **Unified Loss Scaling Law**:

    - Function: Predicts final loss across all combinations of model size, token count, and bit-width.
    - Mechanism: $L = \text{Chinchilla-like} + \delta(N, D_{qat}, D_{fp}, B)$, where the QAT penalty term $\delta$ decomposes into three components:
        - **Irreducible QAT error** $\theta \cdot 2^{-\kappa B}$: a precision floor determined by bit-width
        - **Pure QAT penalty** $\frac{\phi \cdot 2^{-\chi B}}{N^\psi \cdot S_{qat}^\omega}$: error incurred when QAT steps are insufficient
        - **FP/QAT interaction term** $\frac{\lambda \cdot 2^{-\mu B}}{N^\nu \cdot S_{fp}^\xi \cdot S_{qat}^\rho}$: additional difficulty in quantization caused by an excessively long FP phase
    - Design Motivation: Prior scaling laws yield loss that diverges as $D \to \infty$, which is physically unreasonable. In the proposed formulation, all penalty terms decrease as $S$ grows, ensuring loss eventually converges. Fitted on 757 experimental data points, the law accurately predicts both the optimal fraction and the final loss.

4. **Cooldown + QAT Fusion**:

    - Function: Merges the learning rate decay phase with QAT, eliminating redundant FP updates.
    - Mechanism: QAT begins directly at the peak learning rate of the FP phase, with learning rate decay applied concurrently, rather than completing the FP cooldown before initiating QAT.
    - Design Motivation: In the standard FP + cooldown + QAT pipeline, the FP updates during cooldown contribute little to the final quantized model and can be safely omitted.

### Loss & Training
- QAT employs the straight-through estimator to handle the non-differentiability of quantization operations.
- The 757 experiments cover models from 86M to 2.2B parameters, 1/2/4/6-bit quantization, and 2.3B to 1.4T tokens.
- Scaling law parameters are fitted using Huber loss with gradient descent.

## Key Experimental Results

### Main Results: Optimal QAT Fraction Grows with Scale

| Model Size | Total Tokens | 2-bit Optimal $f^*$ | 4-bit Optimal $f^*$ | 6-bit Optimal $f^*$ |
|-----------|-------------|-------------------|-------------------|-------------------|
| 86M | Short | ~10% | ~8% | ~5% |
| 86M | Long | ~40% | ~25% | ~15% |
| 396M | Medium | ~25% | ~15% | ~10% |
| 759M | Long | ~30%+ | ~20% | ~12% |

### Scaling Law Prediction Accuracy

| Prediction Target | Error |
|------------------|-------|
| Optimal QAT fraction (direct fit) | MAE = 0.091 |
| Loss scaling law (757 experiments) | Accurately predicts all configurations |
| Optimal bit-width selection | Correctly predicted |
| Cooldown+QAT fusion vs. standard | Matches or exceeds standard pipeline |

### Key Findings
- **Refutes the "10% universality" claim**: Under large compute budgets, the optimal QAT fraction can reach 30–40% for low bit-widths.
- **Lower bit-widths require more QAT**: At the same scale, 2-bit quantization requires more QAT steps than 6-bit.
- **Larger models are easier to quantize**: Given the same $D_{total}$, larger models require a smaller optimal QAT fraction.
- **Cooldown fusion is effective**: The fused scheme matches or exceeds the standard FP+QAT pipeline without increasing the total token budget.
- **4-bit QAT offers the best cost-efficiency**: Under most memory constraints, 4-bit quantization achieves the optimal trade-off between loss and memory footprint.

## Highlights & Insights
- **Tokens-per-parameter-byte is an elegantly unified variable**: A single quantity simultaneously encodes model size, data volume, and quantization precision, revealing a consistent scaling behavior across diverse configurations. This methodology of identifying the right variable is broadly instructive.
- **Methodological value in overturning prior conclusions**: The paper demonstrates through large-scale systematic experimentation that earlier conclusions were valid only within a limited regime—a recurring and important contribution in scaling law research.
- **Engineering utility of the scaling law**: After fitting 757 experiments, the law directly answers the question of how many bits and what QAT fraction to use given a specific compute budget and memory constraint, providing actionable guidance for large-scale training planning.

## Limitations & Future Work
- Experiments are limited to models up to 2.2B parameters; validation on models with 7B+ parameters is absent.
- Only weight quantization is considered; activation quantization is not addressed.
- The scaling law has 15+ fitted parameters, raising potential concerns about overfitting.
- The learning rate schedule for the cooldown+QAT fusion strategy is relatively simple and may admit more optimal designs.
- Variables such as MoE architectures and data quality are not considered.

## Related Work & Insights
- **vs. Chen et al. (2025b)**: Their QAT scaling law only handles training from scratch ($D_{fp}=0$) and fits each bit-width separately; this paper provides a unified treatment of the FP→QAT pipeline across all bit-widths.
- **vs. Kumar et al. (2025)**: They show that PTQ error grows with the amount of pretraining data; this paper corroborates a similar trend in the QAT setting and offers a principled remedy.
- **vs. Chinchilla**: The proposed scaling law extends the Chinchilla framework with QAT-aware penalty terms, representing a natural generalization of Chinchilla to the quantization-aware training setting.

## Rating
- Novelty: ⭐⭐⭐⭐ Refutes established conclusions and proposes a unified scaling law, though the underlying methodology is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 757 experiments spanning multiple model sizes, bit-widths, and token counts; an enormous engineering undertaking.
- Writing Quality: ⭐⭐⭐⭐⭐ Beautifully designed figures, clear logical exposition, and well-motivated variable choices throughout.
- Value: ⭐⭐⭐⭐⭐ Provides direct, actionable guidance for practical LLM quantization training planning; characteristic of Apple's high-quality research output.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation](turboboa_faster_and_exact_attention-aware_quantization_without_backpropagation.md)
- [\[ICCV 2025\] Scheduling Weight Transitions for Quantization-Aware Training](../../ICCV2025/model_compression/scheduling_weight_transitions_for_quantization-aware_training.md)
- [\[ICLR 2026\] Dataset Distillation as Pushforward Optimal Quantization](dataset_distillation_as_pushforward_optimal_quantization.md)
- [\[ICLR 2026\] PTQ4ARVG: Post-Training Quantization for AutoRegressive Visual Generation Models](ptq4arvg_post-training_quantization_for_autoregressive_visual_generation_models.md)
- [\[ICLR 2026\] What Layers When: Learning to Skip Compute in LLMs with Residual Gates](what_layers_when_learning_to_skip_compute_in_llms_with_residual_gates.md)

<!-- RELATED:END -->
