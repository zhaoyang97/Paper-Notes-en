---
title: >-
  [Paper Note] Compute-Optimal Quantization-Aware Training
description: >-
  [ICLR 2026][Model Compression][Quantization-Aware Training] Based on 757 QAT experiments (86M-2.2B parameters, 1-6 bits), this paper discovers that the optimal QAT training fraction increases as total compute grows (contrary to the previous conclusion of a fixed 10%). It proposes a tokens-per-parameter-byte statistic and a new loss scaling law to accurately predict optimal QAT allocation strategies and final loss.
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Quantization-Aware Training"
  - "Scaling Law"
  - "Compute-Optimal Allocation"
  - "tokens-per-parameter-byte"
  - "Low-bit Quantization"
date: 2026-05-08
content_hash: 2f8462551da0e3bf
---

# Compute-Optimal Quantization-Aware Training

**Conference**: ICLR 2026  
**arXiv**: [2509.22935](https://arxiv.org/abs/2509.22935)  
**Code**: None  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: Quantization-Aware Training, Scaling Law, Compute-Optimal Allocation, tokens-per-parameter-byte, Low-bit Quantization

## TL;DR
Based on 757 QAT experiments (86M-2.2B parameters, 1-6 bits), this paper discovers that the optimal QAT training fraction increases as total compute grows (contrary to the previous conclusion of a fixed 10%). It proposes a tokens-per-parameter-byte statistic and a new loss scaling law to accurately predict optimal QAT allocation strategies and final loss.

## Background & Motivation

**Background**: QAT is the mainstream method for training high-quality quantized models, typically employing a two-stage "full-precision (FP) training → QAT fine-tuning" pipeline. Liu et al. (2025) previously suggested that the QAT stage should account for 10% of total training steps.

**Limitations of Prior Work**:
   - Previous conclusions regarding the "10% optimum" were derived under limited compute budgets and were not verified at larger scales.
   - Quantization error introduced by PTQ increases with the volume of pre-training data (Kumar et al.), implying that the FP-QAT allocation should be scale-dependent.
   - Existing QAT scaling laws (Chen et al.) assume $D_{fp}=0$ (QAT from scratch) and do not handle the "FP → QAT" two-stage scenario.
   - Lack of a unified scaling law across different bit-widths.

**Key Challenge**: Too few QAT steps prevent the model from adapting to low precision, while too many QAT steps compress the FP stage and lead to excessive training with noisy gradients. How does this equilibrium shift as total compute increases?

**Goal**:
   - How does the optimal QAT fraction vary with model size, total tokens, and bit-width?
   - Can a unified scaling law predict the final loss across all configurations?
   - Can the training pipeline be further optimized (e.g., merging cooldown and QAT)?

**Key Insight**: Introduce tokens-per-parameter-byte $S = D/(N \cdot B/8)$ as a unified scaling variable that simultaneously encodes model size, data volume, and quantization precision.

**Core Idea**: The optimal time allocation for QAT is not a fixed 10% but a function that grows with tokens-per-parameter-byte, which can be accurately modeled by a unified scaling law.

## Method

### Overall Architecture
Standard training for low-bit quantized models involves two stages: full-precision (FP) training ($D_{fp}$ tokens) followed by quantization-aware training (QAT, $D_{qat}$ tokens), with $D_{total} = D_{fp} + D_{qat}$. This work centers on one question: given model scale $N$, total token budget $D_{total}$, and bit-width $B$, what is the optimal QAT fraction $f^* = D_{qat}^*/D_{total}$ and how does it scale?

Instead of modifying training operators, the paper makes this allocation problem predictable: First, it identifies a 1D scaling variable, tokens-per-parameter-byte (Design 1), to fit a closed-form formula for the optimal fraction (Design 2). Second, it builds a unified scaling law for the entire loss curve across scales and bit-widths (Design 3) to infer both the optimal fraction and final loss. Finally, it provides a practical improvement to the training pipeline by fusing the learning rate cooldown with QAT (Design 4).

### Key Designs

**1. Tokens-per-Parameter-Byte Statistic: A Unified Scaling Variable**

The paper merges model size $N$, data volume $D$, and bit-width $B$ into $S_{total} = D_{total}/(N \cdot B/8)$, normalizing training tokens by the quantized parameter bytes. This variable captures three directions of "quantization difficulty": larger models are easier to quantize ($N$ increases → $S$ decreases), lower bits are harder ($B$ decreases → $S$ increases), and longer training is harder ($D$ increases → $S$ increases). In tokens-per-parameter-byte coordinates, optimal points for different bit-widths align on a single curve, identifying this as the true independent variable governing QAT allocation.

**2. Optimal QAT Fraction Prediction: A Single-Parameter Formula**

With $S_{total}$ as the correct variable, the paper observes an approximately linear relationship between $S_{total}$ and optimal $S_{qat}$ in log-log space. Incorporating the physical constraint $D_{qat} \leq D_{total}$ yields a closed-form prediction:

$$\hat{f}(D_{total}, N, B) = \frac{\exp(\log S_{total} - a/\log S_{total})}{S_{total}}$$

Only one parameter ($a=6.7297$) is needed to predict the optimal fraction across all configurations with an MAE of only 0.091.

**3. Unified Loss Scaling Law: Predicting Final Loss Across Configurations**

To predict the final loss, the paper adds a QAT-aware penalty term $\delta$ to the Chinchilla framework:

$$L = \text{Chinchilla-like} + \delta(N, D_{qat}, D_{fp}, B)$$

The penalty $\delta$ is decomposed into: **Irreducible QAT error** $\theta \cdot 2^{-\kappa B}$ (precision floor); **Pure QAT penalty** $\frac{\phi \cdot 2^{-\chi B}}{N^\psi \cdot S_{qat}^\omega}$ (residual error from insufficient QAT steps); and **FP/QAT interaction term** $\frac{\lambda \cdot 2^{-\mu B}}{N^\nu \cdot S_{fp}^\xi \cdot S_{qat}^\rho}$ (denoting that excessive FP training makes subsequent quantization harder). All terms decay as $S$ increases, ensuring the loss converges.

**4. Cooldown + QAT Fusion: Reducing Redundant Computation**

The standard process is FP training → cooldown → QAT. However, FP updates during cooldown offer little value to the final quantized model. This method switches directly to QAT from the peak learning rate of the FP stage, performing LR decay and QAT fine-tuning simultaneously. This eliminates redundant compute without increasing total tokens.

### Loss & Training
- QAT uses the straight-through estimator to handle non-differentiable quantization operations.
- 757 experiments covering 86M-2.2B models, 1-6 bits, and 2.3B-1.4T tokens.
- Huber loss and gradient descent are used to fit scaling law parameters.

## Key Experimental Results

### Main Results: Optimal QAT Fraction Increases with Scale

| Model Size | Total Tokens | 2-bit Opt f* | 4-bit Opt f* | 6-bit Opt f* |
| :--- | :--- | :--- | :--- | :--- |
| 86M | Short | ~10% | ~8% | ~5% |
| 86M | Long | ~40% | ~25% | ~15% |
| 396M | Medium | ~25% | ~15% | ~10% |
| 759M | Long | ~30%+ | ~20% | ~12% |

### Scaling Law Predictive Accuracy

| Prediction Target | Error |
| :--- | :--- |
| Optimal QAT fraction (Direct fit) | MAE = 0.091 |
| Loss scaling law (757 experiments) | Accurate across all configs |
| Cross bit-width optimal choice | Correctly predicted |
| Cooldown+QAT fusion vs. Standard | Meets or exceeds standard |

### Key Findings
- **Rejection of the "10% rule"**: At high compute budgets, the optimal QAT fraction can reach 30-40% for low bit-widths.
- **Low bits require more QAT**: 2-bit models require more QAT steps than 6-bit models at the same scale.
- **Large models are easier to quantize**: At the same $D_{total}$, larger models require a smaller QAT fraction.
- **Cooldown fusion is effective**: Achieving similar or better results than the standard pipeline without extra tokens.
- **4-bit QAT is the "sweet spot"**: It generally provides the best trade-off between loss and memory under most constraints.

## Highlights & Insights
- **Tokens-per-parameter-byte is an elegant unified variable**: It successfully maps model size, data volume, and precision to a 1D coordinate where universal laws emerge.
- **Methodological value in challenging consensus**: Systematically proving that previous conclusions were local optima by expanding the scale.
- **Practical Engineering Utility**: After fitting 757 experiments, the law directly answers how many bits and what QAT fraction to use given a specific compute and memory budget.

## Limitations & Future Work
- Only tested up to 2.2B parameters; verification for 7B+ models is needed.
- Focuses on weight quantization; does not address activation quantization.
- The scaling law has 15+ fittable parameters, risking potential over-fitting.
- Learning rate schedules in cooldown+QAT fusion could be further optimized.
- Does not account for MoE architectures or varying data quality.

## Related Work & Insights
- **vs. Chen et al. (2025b)**: Their law covers training from scratch ($D_{fp}=0$); this work handles FP→QAT and unifies bit-widths.
- **vs. Kumar et al. (2025)**: They identified PTQ error growth with data volume; this work validates this in QAT and provides a solution.
- **vs. Chinchilla**: This work is a natural extension of Chinchilla to the quantization training domain by adding QAT-aware terms.

## Rating
- Novelty: ⭐⭐⭐⭐ (Challenged existing consensus + unified scaling law)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (757 experimental points is a massive engineering effort)
- Writing Quality: ⭐⭐⭐⭐⭐ (Excellent visualizations and logical derivations)
- Value: ⭐⭐⭐⭐⭐ (Directly applicable to large-scale LLM training planning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Quantization-Aware Training for Ultra-Low-Bit Reasoning LLMs](towards_quantization-aware_training_for_ultra-low-bit_reasoning_llms.md)
- [\[ICLR 2026\] Post-Training Quantization for Video Matting](post-training_quantization_for_video_matting.md)
- [\[ICLR 2026\] SliderQuant: Accurate Post-Training Quantization for LLMs](sliderquant_accurate_post-training_quantization_for_llms.md)
- [\[ICLR 2026\] Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs](optimal_brain_restoration_for_joint_quantization_and_sparsification_of_llms.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](../../ACL2025/model_compression/efficientqat.md)

</div>

<!-- RELATED:END -->
