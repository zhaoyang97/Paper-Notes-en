---
title: >-
  [Paper Note] When to Restart? Exploring Escalating Restarts on Convergence
description: >-
  [ICLR 2026][Optimization][learning rate scheduling] This paper proposes SGD-ER (SGD with Escalating Restarts), a convergence-aware learning rate scheduling strategy that triggers restarts with linearly escalating learning rates upon detecting training stagnation, enabling the optimizer to escape sharp local minima and explore flatter loss landscape regions. SGD-ER achieves 0.5–4.5% test accuracy improvements on CIFAR-10/100 and TinyImageNet.
tags:
  - ICLR 2026
  - Optimization
  - learning rate scheduling
  - adaptive restarts
  - convergence-aware training
  - SGD optimization
  - deep learning training
date: 2026-05-08
content_hash: ae09fc1dac20ad9e
---

# When to Restart? Exploring Escalating Restarts on Convergence

**Conference**: ICLR 2026
**arXiv**: [2603.04117](https://arxiv.org/abs/2603.04117)
**Area**: Optimization
**Keywords**: learning rate scheduling, adaptive restarts, convergence-aware training, SGD optimization, deep learning training

## TL;DR

This paper proposes SGD-ER (SGD with Escalating Restarts), a convergence-aware learning rate scheduling strategy that triggers restarts with linearly escalating learning rates upon detecting training stagnation, enabling the optimizer to escape sharp local minima and explore flatter loss landscape regions. SGD-ER achieves 0.5–4.5% test accuracy improvements on CIFAR-10/100 and TinyImageNet.

## Background & Motivation

The learning rate is among the most critical hyperparameters in deep learning training, directly affecting convergence speed, stability, and generalization.

### Existing Learning Rate Schedules and Their Limitations

| Scheduler | Strategy | Limitation |
|--------|------|------|
| Exponential/Linear Decay | Monotonically decreasing | Cannot escape sharp minima or saddle points |
| Cosine Annealing (SGDR) | Periodic cosine decay + warm restarts | Restart timing is fixed, agnostic to training dynamics |
| Cyclical LR (CLR) | Smooth oscillation within predefined bounds | Fixed boundaries, non-adaptive |
| Warmup-Stable-Decay (WSD) | Three phases: warmup–stable–decay | Tied to a fixed computational budget |

**Core Problem**: Existing methods apply restarts or adjustments that are **predefined or periodic**, remaining entirely unaware of actual training dynamics such as stagnation or convergence behavior.

**Core Argument**: Restarts should be **adaptive**—triggered by convergence rather than a fixed schedule. When the model reaches a loss plateau, restarting with a larger learning rate can help escape the current local minimum.

## Method

### SGD-ER Algorithm

The core algorithmic logic proceeds as follows:

1. Begin training with initial learning rate $\eta_0$
2. Gradually reduce the learning rate using a decay strategy (exponential or linear)
3. Declare convergence when the validation loss shows no significant improvement within a **patience window**
4. Trigger a restart: linearly escalate the learning rate to $\eta_k = (k+1) \cdot \eta_0$, where $k$ denotes the restart count
5. Retain current model parameters and continue training
6. Termination condition: stop if the best loss after a restart does not improve upon the previous best, or if the maximum number of epochs is reached

### Learning Rate Update Rule

The SGD update at restart $k$:

$$\theta_{t+1} = \theta_t - \eta_k \nabla f(\theta_t), \quad \eta_k = (k+1)\eta_0$$

### Theoretical Analysis: Accelerated Saddle Point Escape

**Theorem 1**: Let $f$ be an $L$-smooth function and $\theta^*$ a strict saddle point with $\lambda_{\min}(\nabla^2 f(\theta^*)) = -\gamma < 0$. The number of iterations required to escape the $\delta$-neighborhood at restart $k$ satisfies:

$$T_k \geq \frac{\ln(\delta / |x_0|)}{\ln(1 + \eta_k \gamma)}$$

As $k \to \infty$, $T_k \to 0$—**a larger learning rate accelerates saddle point escape**. This provides a theoretical justification for learning rate escalation.

### Convergence Detection Criterion

A **plateau-based criterion** is employed: convergence is signaled when the validation loss exhibits no meaningful decrease within a predefined patience window. This is consistent with early stopping practice.

- CIFAR-100: patience = 50 epochs
- CIFAR-10: patience = 30 epochs

### Key Design Considerations

- **Parameter retention**: restarts modify only the learning rate without resetting model parameters, allowing continued exploration from learned representations
- **Linear escalation**: the learning rate increases by $\eta_0$ at each restart, providing a moderate but sustained increase in exploration intensity
- **Dual termination**: training stops if no improvement is observed after a restart, preventing unnecessary divergence

## Key Experimental Results

### Main Results: ResNet-18 Test Accuracy (%)

| Dataset | SGD_exp | SGD_lin | Adam | CosA | CLR | WSDS | **Ours_exp** | **Ours_lin** |
|--------|---------|---------|------|------|-----|------|:---:|:---:|
| CIFAR-10 | 90.86 | 91.93 | 91.34 | 92.59 | 92.15 | 93.05 | **93.83** | **93.83** |
| CIFAR-100 | 68.30 | 71.00 | 67.94 | 71.63 | 70.44 | 72.39 | **74.30** | **74.30** |
| TinyImageNet | 59.09 | 58.35 | 54.53 | 59.46 | 57.53 | 59.28 | 59.71 | **60.79** |

### Cross-Architecture Results: CIFAR-100 Test Accuracy (%, Exponential Decay)

| Architecture | SGD_exp | CosA | CLR | WSDS | **Ours** |
|------|---------|------|-----|------|:---:|
| ResNet-34 | 67.75 | 72.17 | 71.04 | 72.36 | **74.24** |
| ResNet-50 | 65.52 | 72.10 | 70.25 | 73.76 | **76.77** |
| VGG-16 | 65.17 | 67.35 | 67.23 | 68.08 | **68.56** |
| DenseNet-121 | 56.10 | 71.20 | 66.61 | 72.45 | **76.76** |

### Long-Training Results: CIFAR-100, 2000 Epochs

| SGD_exp | SGD_lin | Adam | CosA | CLR | WSDS | **Ours_exp** | **Ours_lin** |
|---------|---------|------|------|-----|------|:---:|:---:|
| 68.53 | 62.17 | 71.27 | 72.84 | 72.10 | 73.59 | **74.41** | **74.41** |

### Overfitting Analysis (CIFAR-100, Average over 3 Seeds)

| Method | Train Loss | Val Loss | Test Loss | Test Acc (%) |
|------|-----------|----------|-----------|:---:|
| CLR | **1.60e-05** | 0.00488 | 0.00496 | 70.65 |
| CosA | 1.75e-05 | 0.00466 | 0.00472 | 72.05 |
| WSDS | 1.64e-05 | 0.00462 | 0.00465 | 72.83 |
| **Ours_exp** | 2.40e-05 | 0.00434 | 0.00443 | 73.62 |
| **Ours_lin** | 2.16e-05 | **0.00427** | **0.00435** | **74.61** |

Note: CLR achieves the lowest training loss but the highest test loss—a classic overfitting pattern. SGD-ER exhibits slightly higher training loss yet substantially better generalization.

### Key Experimental Findings

1. SGD-ER achieves the highest test accuracy across **all** dataset–architecture combinations.
2. The largest gain is observed on DenseNet-121: 56.10% → 76.76% (+20.66%); standard SGD nearly fails to train DenseNet effectively.
3. A brief accuracy drop occurs immediately after each restart, but the model rapidly recovers and surpasses the previous best performance.
4. Under extended training (2000 epochs), SGD-ER continues to improve while competing methods saturate.
5. SGD-ER achieves better generalization at higher training loss, indicating convergence to flatter minima.

## Highlights & Insights

1. **Simplicity and effectiveness**: the method requires only a patience parameter and a linear increment, introduces no additional computational overhead, and can serve as a plug-and-play module for any SGD-based training pipeline.
2. **Convergence-aware vs. fixed-period scheduling**: the central principle—"let training dynamics determine when to restart"—is more principled than predefined periodic schedules.
3. **Higher training loss = better generalization**: this result directly reflects classical theory on flat vs. sharp minima; the minima found by SGD-ER are wider and generalize better.
4. **Greater benefit for weaker architectures**: DenseNet-121 nearly fails under standard SGD, yet SGD-ER restores it to a level competitive with ResNet variants.
5. **Theory–practice alignment**: Theorem 1 predicts faster saddle point escape with larger learning rates, which is empirically confirmed by the rapid recovery observed after each restart.

## Limitations & Future Work

1. **Methodological simplicity**: linear escalation may not be the optimal strategy; a systematic investigation of escalation magnitude and schedule is absent.
2. **Manual patience tuning**: different values are required for CIFAR-10 (30) and CIFAR-100 (50), necessitating task-specific adjustment.
3. **Post-restart accuracy fluctuation**: each restart induces a performance dip before recovery, resulting in a non-smooth training trajectory.
4. **Evaluation limited to image classification**: the method has not been validated on NLP, speech, or other task domains.
5. **Integration with Adam-family optimizers insufficiently explored**: the work primarily focuses on SGD; Adam variants are only briefly mentioned in the appendix.
6. **Theoretical analysis restricted to saddle point escape**: convergence guarantees for escaping local minima and bounds on convergence rates are not established.

## Rating

- **Novelty**: ⭐⭐⭐ — The idea is intuitive but not sophisticated; it belongs to the category of "simple yet effective" engineering improvements.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 3 datasets, 5 architectures, and multiple baselines with consistent and significant results.
- **Writing Quality**: ⭐⭐⭐⭐ — Figures and tables are clear; Fig. 1's learning rate curve comparison is particularly intuitive.
- **Value**: ⭐⭐⭐⭐ — Offers practical engineering value as a plug-and-play module, though theoretical depth is limited.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Exploring Diverse Generation Paths via Inference-time Stiefel Activation Steering](exploring_diverse_generation_paths_via_inference-time_stiefel_activation_steerin.md)
- [\[ICLR 2026\] Convergence of Muon with Newton-Schulz](convergence_of_muon_with_newton-schulz.md)
- [\[ICLR 2026\] Scaling Laws of SignSGD in Linear Regression: When Does It Outperform SGD?](scaling_laws_of_signsgd_in_linear_regression_when_does_it_outperform_sgd.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](../../ACL2026/optimization/clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)

<!-- RELATED:END -->
