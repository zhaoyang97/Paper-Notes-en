---
title: >-
  [Paper Note] Many Eyes, One Mind: Temporal Multi-Perspective and Progressive Distillation for Spiking Neural Networks
description: >-
  [ICLR 2026][Model Compression][Spiking Neural Network] To address two major pain points in Spiking Neural Network (SNN) distillation—"using a fixed ANN output to supervise all timesteps" and "information loss during truncated inference"—this paper proposes **masked re-weighting to generate diverse temporal teacher signals (Many Eyes)** and **cumulative average predictions
tags:
  - ICLR 2026
  - Model Compression
  - Spiking Neural Network
  - Knowledge Distillation
  - Temporal Dynamics
  - Truncated Inference
  - Multi-Teacher
date: 2026-05-08
content_hash: bc314889c6b49632
---
# Many Eyes, One Mind: Temporal Multi-Perspective and Progressive Distillation for Spiking Neural Networks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=NbdEDRRsCI](https://openreview.net/forum?id=NbdEDRRsCI)  
**Code**: [https://github.com/KaiSUN1/MEOM](https://github.com/KaiSUN1/MEOM)  
**Area**: Model Compression / Knowledge Distillation / Spiking Neural Networks  
**Keywords**: Spiking Neural Network, Knowledge Distillation, Temporal Dynamics, Truncated Inference, Multi-Teacher  

## TL;DR
To address two major pain points in Spiking Neural Network (SNN) distillation—"using a fixed ANN output to supervise all timesteps" and "information loss during truncated inference"—this paper proposes **masked re-weighting to generate diverse temporal teacher signals (Many Eyes)** and **cumulative average predictions to progressively align with full-length predictions (One Mind)**. It achieves SOTA on CIFAR/ImageNet and supports reliable inference at any timestep.

## Background & Motivation
- **Background**: SNNs are considered ideal for edge and neuromorphic hardware due to their event-driven nature and low energy consumption, but their accuracy has long lagged behind ANNs. The mainstream approach to bridge this gap is Knowledge Distillation (KD), transferring features/logits from an ANN teacher to an SNN student, requiring fewer timesteps than ANN-to-SNN conversion. Recent **Temporal Wise Distillation (TWD)** treats SNN outputs across timesteps as a temporal ensemble and supervises each timestep individually, improving both step-by-step and final accuracy.
- **Limitations of Prior Work**: TWD has two structural flaws. First, it uses the **same static ANN output** to supervise all timesteps, whereas SNNs evolve dynamically. Figure 1a in the paper shows that final logits change significantly over time; a fixed teacher target fails to capture this temporal diversity and instead **homogenizes** intermediate predictions. Second, neuromorphic deployment often requires **truncated inference** (only running initial steps) due to latency/energy constraints. Figure 1b shows membrane potentials concentrated at low values at $t=1$ and accumulating over time, meaning early information is naturally incomplete. While TWD improves single-step accuracy, it doesn't guarantee early predictions converge to the full-length prediction.
- **Key Challenge**: The advantage of **"refined stepwise supervision"** vs. the **loss of temporal diversity and truncated information** caused by a "single fixed teacher + isolated optimization of steps."
- **Goal**: Construct a unified KD framework that injects diverse temporal supervision and ensures predictions at any truncated length converge to the full-length prediction, achieving "reliable inference at any timestep."
- **Core Idea**: **[Many Eyes]** Uses lightweight time-indexed masks to perturb a single ANN teacher, simulating a "multi-teacher ensemble" to provide different perspectives for each timestep; **[One Mind]** Uses adjacent alignment of cumulative average predictions to backflow stable late-stage information to noisy early-stage predictions.

## Method

### Overall Architecture
MEOM (Many Eyes, One Mind) consists of two complementary modules: **TMPD (Temporal Multi-Perspective Distillation)** for "Many Eyes"—generating temporally diverse teacher supervision from a single ANN; and **TPD (Temporal Progressive Distillation)** for "One Mind"—forcing cumulative predictions at each timestep to progressively align with full-length predictions. Together with standard stepwise cross-entropy, they form a unified objective. The method requires only one ANN teacher and no multi-teacher training.

```mermaid
flowchart LR
    A[ANN Teacher Feature f^A] --> B[Time-indexed Mask m_t<br/>Stepwise Perturbation]
    B --> C[Masked Logits z̃^A_t<br/>+ Original Logits z^A Linear Combination]
    C --> D[Diverse Temporal Teacher p^A_t]
    E[SNN Student<br/>Stepwise Logits z^S_t] --> F[Cumulative Average z̄^S_≤t]
    D -->|TMPD: KL Stepwise Alignment| E
    F -->|TPD: Adjacent Cumulative Alignment| F
    E --> G[Lall = TWD-CE + TMPD-KL + TPD]
    F --> G
```

### Key Designs
**1. TMPD — Transforming "One Teacher" into "Many Eyes" via Mask Perturbation**: The core idea is to provide complementary supervision perspectives for different timesteps without needing multiple pre-trained teachers. For each timestep $t$, a lightweight time-indexed mask $m_t$ is introduced to perform a Hadamard product with the teacher features, yielding masked features $\tilde{f}^A_t = f^A \odot m_t$. Both original and masked features pass through the same classification head to get original logits $z^A$ and masked logits $\tilde{z}^A_t$. Since classification weights $W_c$ are shared, masked logits can be written as $\tilde{z}^A_t = z^A + \delta z^A_t$, where the perturbation $\delta z^A_t = W_c(f^A \odot m_t)$ deviates from the original while maintaining semantic consistency. These are linearly mixed $\hat{z}^A_t = (1-\lambda)z^A + \lambda\tilde{z}^A_t$ and passed through temperature softmax to get a stepwise teacher distribution $p^A_t$, which is aligned with the student distribution: $L_{\text{TMPD-KL}} = \frac{1}{T}\sum_{t=1}^{T}\mathrm{KL}(p^A_t \| p^S_t)$. This provides each timestep with "slightly perturbed but semantically consistent" teacher signals. Theoretically (Theorem 1/2), this is equivalent to implicit regularization, forcing the student to remain consistent within the teacher's local neighborhood, thereby achieving a lower error bound than TWD.

**2. TPD — Achieving "One Mind" via Progressive Cumulative Alignment**: To solve information loss in truncated inference, the most direct way is to align early predictions with full-length ones, but a large gap can cause training instability. TPD instead aligns **cumulative average predictions of adjacent timesteps**, providing smoother guidance. First, calculate the cumulative mean logits and probabilities up to step $t$: $\bar{z}^S_{\le t} = \frac{1}{t}\sum_{k=1}^{t} z^S_k$ and $\bar{p}^S_{\le t} = \mathrm{softmax}_\tau(\bar{z}^S_{\le t})$. Then, for each adjacent pair $(t, t+1)$, use cross-entropy to align the former to the latter: $L_{\text{TPD}} = \frac{1}{T-1}\sum_{t=1}^{T-1}\mathrm{CE}(\bar{p}^S_{\le t}, \bar{p}^S_{\le t+1})$. By treating the more stable late-stage cumulative average as the target, information "seeps" back to noisy early predictions. Theorem 3 provides a strict ordering for truncated accuracy degradation: $\Delta^{\text{TPD}} < \Delta^{\text{NSC}} < \Delta^{\text{NC}}$, indicating cumulative alignment is superior to simple adjacent step alignment (NSC) and lack of consistency constraints (NC).

**3. Unified Training Objective and Complementarity**: The total loss is $L_{\text{all}} = \alpha L_{\text{TWD-CE}} + \beta L_{\text{TMPD-KL}} + \gamma L_{\text{TPD}}$, with $\alpha=1, \beta=0.5, \gamma=0.3$. The two modules are complementary: TMPD improves full-length accuracy through richer temporal supervision and lower gradient variance, while TPD exponentially tightens truncated accuracy loss. The higher final accuracy provided by TMPD further tightens the TPD bound, enabling the framework to excel in both full-length and truncated scenarios.

## Key Experimental Results

### Main Results (CIFAR-10 / CIFAR-100 Top-1 %, ResNet-19)

| Method | T=2 | T=4 | T=6 |
|------|-----|-----|-----|
| TWSNN (TWD, 2025a) | 96.65 / 81.47 | 96.97 / 82.47 | 97.00 / 82.56 |
| HTA-KL (2025) | 96.68 / 80.51 | 96.76 / 81.03 | – |
| **MEOM (Ours)** | **96.65 / 81.82** | **97.13 / 82.85** | **97.08 / 83.22** |

On CIFAR-100, MEOM even **slightly outperforms the ANN teacher**; it remains competitive in low-latency settings (T=2).

### ImageNet (Top-1 %, T=4)

| Method | ResNet-34 | S-8-384 (Spiking Transformer) |
|------|-----------|-------------------------------|
| BKDSNN | 67.21 | 75.48 |
| TWSNN | 71.04 | – |
| **MEOM (Ours)** | **71.64** | **76.77** |

The method outperforms all SNN baselines across both convolutional and Spiking Transformer architectures, narrowing the gap with ANNs.

### Ablation Study (CIFAR-10/100 Top-1 %, and Temporal Variance Var@4 at T=4)

| TAD | TWD | TMPD | TPD | T=4 | Var@4 |
|-----|-----|------|-----|------|-------|
| ✗ | ✗ | ✗ | ✗ | 95.00 / 77.22 | 0.493 |
| ✗ | ✓ | ✗ | ✗ | 95.57 / 79.10 | 0.212 |
| ✗ | ✓ | ✓ | ✗ | 95.98 / 79.58 | 0.133 |
| ✗ | ✓ | ✗ | ✓ | 95.84 / 79.50 | 0.199 |
| ✗ | ✓ | ✓ | ✓ | **96.07 / 79.66** | 0.150 |

### Key Findings
- **TMPD significantly reduces temporal variance** (0.212→0.133), confirming its role in implicit regularization and stabilizing cross-step predictions; accuracy is highest when combined with TPD.
- **Truncated Inference**: Under $T=6$ training and $T=1 \sim 5$ evaluation, MEOM achieves the highest accuracy at all truncation points. CIFAR-100 ResNet-19 at $T=3$ even exceeds the ANN teacher, and even at $T=1$, it outperforms all baselines, verifying that TPD makes early predictions converge toward the full-length prediction.
- While TMPD and TPD provide gains individually, their **joint use is optimal**, consistent with the theoretical complementarity analysis.

## Highlights & Insights
- **"One teacher, many perspectives" is ingenious**: Using time-indexed masks and a shared classification head achieves the benefits of multi-teacher ensembles (complementary inductive biases) within a single ANN, avoiding the overhead of training multiple teachers.
- **Captures the essence of SNN contradictions**: It maps the temporal evolution of SNN outputs and the incomplete information of truncated inference—two temporal characteristics ignored by TWD—to the intuitive modules of Many Eyes and One Mind respectively.
- **Theoretical and Experimental Loop**: Three theorems (implicit regularization, error lower bound, truncated degradation order) support the modules, and the temporal variance metric directly validates the "stronger supervision $\rightarrow$ lower variance" argument.

## Limitations & Future Work
- The specific form of the mask $m_t$ (random vs. learnable) and the choice of mixing coefficient $\lambda$ are sensitive to the trade-off between "semantic consistency vs. diversity," which is not fully explored regarding parameter robustness.
- Experiments focus on classification (CIFAR/ImageNet) with ResNet/Spiking-Transformer, **lacking coverage of event camera datasets (DVS-CIFAR/Gesture) or more complex temporal tasks like detection/segmentation**, leaving transferability to be verified.
- TPD relies on cumulative averages, so gains for extremely short timestep budgets ($T=1$) are limited—essentially, early information loss is a physical bottleneck that distillation can alleviate but not eliminate.

## Related Work & Insights
- **SNN Knowledge Distillation**: KDSNN, LaSNN, BKDSNN (ANN $\rightarrow$ SNN feature/logits distillation), TSSD/Sparse-KD (SNN $\rightarrow$ SNN self-distillation), TWD/TWSNN (stepwise temporal supervision). This paper is a direct improvement over TWD.
- **Multi-teacher Ensemble Distillation** (ANN domain): Improving generalization by aggregating complementary inductive biases from teachers of different architectures; this paper "virtualizes" this principle into the temporal dimension via mask perturbation.
- **SNN Temporal Flexibility**: SEENN (confidence-based adaptive truncation), HSD/MTT/SSNN (strengthening early predictions during training), various consistency methods (adjacent step / all-to-all consistency). This paper fills the gap where early predictions are not guaranteed to approach full-length predictions using cumulative average alignment.
- **Insight**: Using "shared heads + lightweight input-side perturbations" to generate diverse teacher signals is a low-cost, generalizable idea for creating supervision diversity, applicable to other distillation scenarios where temporal/multi-perspective supervision is needed but teacher resources are scarce.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The "single teacher masked for multiple temporal perspectives" and "progressive cumulative average alignment" both offer insightful improvements to TWD with clear theoretical support.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers CIFAR/ImageNet, ResNet/Spiking-Transformer, full-length+truncated+ablation+variance+energy consumption, though missing event camera and downstream task validation.
- **Writing Quality**: ⭐⭐⭐⭐ — The "Many Eyes / One Mind" metaphor is consistently applied, and the chain of motivation-method-theory-experiment is coherent and readable.
- **Value**: ⭐⭐⭐⭐ — Truncated inference reliability is practically significant for neuromorphic/edge deployment; the method is lightweight, easy to reproduce, and open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[ICLR 2026\] Towards Lossless Memory-efficient Training of Spiking Neural Networks via Gradient Checkpointing and Spike Compression](towards_lossless_memory-efficient_training_of_spiking_neural_networks_via_gradie.md)
- [\[ICLR 2026\] Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis](why_attention_patterns_exist_a_unifying_temporal_perspective_analysis.md)
- [\[ICLR 2026\] Cannistraci-Hebb Training on Ultra-Sparse Spiking Neural Networks](cannistraci-hebb_training_on_ultra-sparse_spiking_neural_networks.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)

</div>

<!-- RELATED:END -->
