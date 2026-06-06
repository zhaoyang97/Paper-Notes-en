---
title: >-
  [Paper Note] TRACER: Robust Multimodal Fine-Tuning Proven by WMA Teacher + Geometric Decomposition
description: >-
  [ICML 2026][Self-Supervised Learning][CLIP Fine-tuning] TRACER utilizes closed-form theoretical analysis to geometrically decompose contrastive fine-tuning into a "task subspace" and "orthogonal preservation." It proves…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "CLIP Fine-tuning"
  - "OOD Robustness"
  - "Self-distillation"
  - "EMA teacher"
  - "WMA teacher"
date: 2026-05-08
content_hash: 2df376381fdc4c9c
---

# TRACER: Robust Multimodal Fine-Tuning Proven by WMA Teacher + Geometric Decomposition

**Conference**: ICML 2026  
**arXiv**: [2605.29380](https://arxiv.org/abs/2605.29380)  
**Code**: https://github.com/HesamAsad/TRACER  
**Area**: CLIP Fine-tuning / Robustness / Self-distillation  
**Keywords**: CLIP Fine-tuning, OOD Robustness, Self-distillation, EMA teacher, WMA teacher

## TL;DR
TRACER utilizes closed-form theoretical analysis to geometrically decompose contrastive fine-tuning into a "task subspace" and "orthogonal preservation." It proves that EMA teachers collapse and lose regularization power, and proposes the Weighted Moving Average (WMA) teacher to maintain finite-horizon constraint power with bias-free convergence to the task subspace. On CLIP ViT-B/16, the average ImageNet distribution shift performance improves to 64.07% vs. CaRot's 62.54%.

## Background & Motivation

**Background**: Zero-shot transfer in multimodal models like CLIP is strong, but downstream fine-tuning often damages OOD robustness (catastrophic forgetting). Existing mitigation methods fall into four categories: LP-FT (linear probe then full fine-tuning), FLYP (reusing the pre-trained text encoder as a head), WiSE-FT/Model Stock (weight interpolation), and L2-SP/Self-distillation (regularization).

**Limitations of Prior Work**: (1) Most methods are empirically designed, lacking a theoretical explanation for where and why forgetting occurs; (2) Self-distillation methods mostly use EMA teachers, which gradually align with the student, causing the teacher-student gap to converge to 0 and the regularization power to vanish—precisely during the late stages of training when OOD robustness is most vulnerable.

**Key Challenge**: Maintaining OOD robustness requires a persistent regularization anchor. EMA anchors automatically collapse toward the student, whereas static teachers (fixed at initial weights) do not collapse but introduce "anchor bias," preventing convergence to the task optimum.

**Goal**: (1) Provide a closed-form analytical framework for contrastive fine-tuning to clarify the geometric behavior of various fine-tuning strategies; (2) Design a teacher that maintains continuous regularization power while achieving bias-free convergence to the task optimum.

**Key Insight**: By employing linearized analysis (viewing the image encoder as a linear projection) and introducing a contrastive target matrix $\mathbf{Y}_{\mathrm{FT}} = \mathbf{W}_T^0 \mathbf{X}_T (n \mathbf{I}_n - \mathbf{J}_n)$, the contrastive loss is simplified to a matrix least-squares problem. This yields closed-form solutions for all fine-tuning strategies, geometrically decomposing them into "changes within the task subspace" versus "preservation in the orthogonal subspace."

**Core Idea**: The teacher is transitioned from EMA to WMA (Weighted Moving Average over the entire student trajectory, with a Beta(0.5, 0.5) U-shape kernel). It is proven that the WMA teacher converges to the minimum-norm task solution within the task subspace while preserving pre-trained knowledge in the orthogonal subspace, and the teacher-student gap does not vanish within a finite-horizon.

## Method

### Overall Architecture

The TRACER loss is defined as $\mathcal{L}_{\mathrm{TRACER}} = \mathcal{L}_{\mathrm{MMCL}} + \lambda_{\mathrm{SD}} \mathcal{L}_{\mathrm{SD-WMA}}$. The former is the standard CLIP InfoNCE plus a cross-Frobenius regularizer, while the latter is a multi-view distillation loss derived from the WMA teacher (encompassing feature distillation, contrastive relational distillation, interactive contrastive learning, and cross-knowledge distillation).

For each training step: (1) the student is updated via the MMCL gradient; (2) the WMA teacher is updated using $\mathbf{W}_{\mathrm{Teacher}}^t = (1-\omega_t) \mathbf{W}_{\mathrm{Teacher}}^{t-1} + \omega_t \mathbf{W}_I^t$, where $\omega_t = \kappa(\tau_t) / \sum_j \kappa(\tau_j)$ is the weight based on the Beta(0.5, 0.5) kernel; (3) the teacher provides four types of distillation signals to the student.

### Key Designs

1.  **Contrastive Target Matrix + Geometric Decomposition of Closed-Form Solutions**:
    - **Function**: Converts the non-linear optimization of contrastive fine-tuning into matrix least-squares, yielding closed-form solutions for each strategy to pinpoint where forgetting occurs.
    - **Mechanism**: Defining $\mathbf{Y}_{\mathrm{FT}} = \mathbf{W}_T^0 \mathbf{X}_T (n \mathbf{I}_n - \mathbf{J}_n)$ (frozen text encoder + centered contrast operator), it is proven that linearized MMCL loss is equivalent to $\min_{\mathbf{W}_I} \frac{1}{2} \|\mathbf{W}_I \mathbf{X}_I - \mathbf{Y}_{\mathrm{FT}}\|_F^2$. Theorem 3.2 provides solutions for three strategies—Direct FT: $\mathbf{W}_I^0 (I - \mathcal{P}_I) + \mathbf{Y}_{\mathrm{FT}} \mathbf{X}_I^\top (\mathbf{X}_I \mathbf{X}_I^\top)^+$ (preserves orthogonal, replaces parallel); L2-SP: blends all directions (no structural decomposition); Static SD: $\mathbf{W}_I^0 (I - \frac{1}{1+\lambda} \mathcal{P}_I) + \frac{1}{1+\lambda} \mathbf{Y}_{\mathrm{FT}} \mathbf{X}_I^\top (\mathbf{X}_I \mathbf{X}_I^\top)^+$ (orthogonal preservation + convex combination in task subspace).
    - **Design Motivation**: While previous explanations for SD were empirical, this work proves SD structurally preserves orthogonal knowledge while adapting to the task. L2 regularizers blend all directions, implying catastrophic forgetting in the orthogonal subspace as well. This provides a theoretical basis for choosing SD over L2.

2.  **WMA Teacher: U-shape Kernel + Bias-free Convergence Proof**:
    - **Function**: Addresses both the collapse of EMA teachers and the anchor bias of static SD.
    - **Mechanism**: The WMA teacher is a weighted average of the entire student trajectory. The kernel $\kappa(\tau)$ uses a Beta(0.5, 0.5) U-shape, assigning weights to both the initial checkpoint (to preserve robust priors) and the late-stage checkpoints (for task adaptation). Using $\tau_k = (k + 0.5) / (T + 1) \in (0, 1)$ avoids Beta divergence. Updates follow $\omega_t = \kappa(\tau_t) / \sum_{j=0}^t \kappa(\tau_j)$ and $\mathbf{W}_{\mathrm{Teacher}}^t = (1 - \omega_t) \mathbf{W}_{\mathrm{Teacher}}^{t-1} + \omega_t \mathbf{W}_I^t$. Theorem 3.4 proves the student converges to $\mathbf{W}_{\mathrm{FT}}^\star \mathcal{P}_I$ (minimum-norm task solution) in the task subspace while retaining the orthogonal component.
    - **Design Motivation**: Constant EMA weight $\omega_t$ leads to exponential convergence between teacher and student, eliminating the gap. Static SD ($\omega_t=0$ for $t>0$) never eliminates bias. The U-shape of WMA ensures "early anchors" and "recent anchors" both have non-zero weight, maintaining a meaningful teacher-student gap in a finite-horizon while enabling bias-free trajectory-weighted convergence.

3.  **Multi-view Distillation Loss $\mathcal{L}_{\mathrm{SD-WMA}}$**:
    - **Function**: Allows the student to learn from the WMA teacher via multiple perspectives, making it more robust than simple feature alignment.
    - **Mechanism**: Four sub-losses: (i) Feature Distillation: direct alignment of student/teacher embeddings; (ii) Contrastive Relational Distillation: matching batch-wise similarity distributions; (iii) Interactive Contrastive Learning: cross-modal student-teacher alignment; (iv) Cross Knowledge Distillation: cross-modal logits alignment.
    - **Design Motivation**: Single-view distillation (e.g., FD only) can lead to overfitting a specific representation dimension. The four views cover features, relations, cross-modal interactions, and logits, encompassing all facets of "preserving pre-trained knowledge."

### Toy Experiment: MNIST + ColoredMNIST

A multimodal CLIP-like model is pre-trained on MNIST and then fine-tuned on ColoredMNIST (spurious correlations where digits 0-4 are 95% red, and 5-9 are 95% blue). Direct FT learns the task but MNIST accuracy drops from 96.8% to 59.0% (37.9% forgetting); L2 Reg results in 13.6% forgetting; Static SD in 1.8%; and Dynamic SD (WMA) in 0.1%—validating the theoretically predicted relationship between geometric decomposition and forgetting rates.

## Key Experimental Results

### Main Results: CLIP ViT-B/16 on ImageNet + Distribution Shifts

| Method | IN | IN-V2 | IN-R | IN-A | IN-S | ObjNet | Average |
|------|-----|-------|------|------|------|--------|------|
| ZS (zero-shot) | 68.33 | 61.93 | 77.71 | 49.95 | 48.26 | 54.17 | 58.39 |
| LP-FT | 82.44 | 72.74 | 72.81 | 49.28 | 50.31 | 54.42 | 59.91 |
| FLYP | 82.72 | 72.76 | 71.32 | 48.49 | 49.87 | 54.83 | 59.45 |
| Lipsum-FT | 83.32 | 73.57 | 75.93 | 49.87 | 51.43 | 54.35 | 61.03 |
| CaRot | 83.15 | 74.08 | 77.74 | 51.57 | 52.68 | 56.63 | 62.54 |
| **TRACER** | 82.76 | **74.14** | **79.33** | **54.92** | **53.69** | **58.26** | **64.07** |

TRACER leads across all 5 OOD benchmarks, with an average of 64.07% vs. CaRot's 62.54% (+1.53). While ID (ImageNet) accuracy is slightly lower (82.76 vs. CaRot's 83.15), the trade-off favors OOD, and the gap is < 0.5. The most significant improvement occurs on IN-A (54.92 vs. 51.57, +3.35), validating the robustness of the WMA teacher to extreme OOD.

### Comparison with more baselines (ImageNet 5 columns)

| Method | IN | IN-V2 | IN-R | IN-A | IN-S | Avg. Shifts |
|------|-----|-------|------|------|------|-----|
| Direct FT | 82.83 | 72.57 | 68.53 | 39.23 | 47.97 | 57.08 |
| L2-SP | 82.87 | 72.63 | 68.77 | 39.73 | 48.23 | 57.34 |
| Static SD | 82.07 | 73.13 | 72.87 | 42.33 | 49.87 | 59.55 |
| LP-FT | 82.14 | 72.09 | 70.44 | 46.32 | 48.65 | 59.38 |
| FLYP | 82.72 | 72.76 | 71.32 | 48.49 | 49.87 | 60.61 |
| CAR-FT | 83.27 | 74.03 | 75.37 | 49.53 | 52.97 | 62.98 |
| Lipsum-FT | 83.33 | 73.57 | 75.93 | 49.87 | 51.43 | 62.70 |
| Model Stock | **84.07** | 74.83 | 71.77 | 51.23 | 51.77 | 62.40 |
| ARF | 82.73 | 72.77 | 75.63 | 50.27 | 51.83 | 62.63 |
| CaRot | 83.15 | 74.08 | 77.74 | 51.57 | 52.68 | 62.98 |
| **TRACER** | 82.76 | 74.14 | **79.33** | **54.92** | **53.69** | **64.07** |

### Key Findings

- **WMA teacher resolves EMA collapse**: At the end of training, the EMA teacher-student gap approaches 0. TRACER maintains this gap using WMA, leading to stable OOD performance improvements.
- **Support for geometric decomposition theory**: Direct FT performance on IN-A drops to 39.23 vs ZS 49.95, indicating that forgetting is more severe on harder OOD tasks. TRACER elevates IN-A to 54.92, proving the trajectory-weighted teacher preserves robustness.
- **Task subspace vs. orthogonal subspace**: From the closed-form SD solution, as long as $\lambda > 0$, the task subspace is biased towards $\mathbf{W}_I^0$, leading to underfitting. The WMA teacher allows the task subspace to converge to the minimum-norm solution without bias.
- **ColoredMNIST toy experiment matches theory**: The forgetting rates (Direct > L2 > Static SD > Dynamic SD) align perfectly with geometric ordering.
- **Consistency over SOTA**: TRACER outperforms recent methods like Lipsum-FT and CaRot by 1-2%, demonstrating WMA is a structural improvement rather than a marginal trick.

## Highlights & Insights

- **Geometric essence of fine-tuning via linearized analysis**: Converting contrastive loss to matrix least-squares provides a transparent look at the closed-form solutions of different strategies.
- **Serious discussion of EMA collapse**: Unlike prior literature that defaults to EMA, this work quantifies anchor failure in finite-horizon settings.
- **Elegant design of WMA + U-shape kernel**: The Beta(0.5, 0.5) kernel fulfills the dual requirement of preserving initial priors and task adaptation.
- **Mathematical guarantee of bias-free convergence**: Theorem 3.4 proves convergence to the minimum-norm solution in the task subspace, which static SD cannot achieve.
- **Comprehensive verification**: The rigorous verification chain from controllable toy experiments to industrial ImageNet benchmarks leaves few gaps for critique.
- **Purposeful multi-view distillation**: Each component of the loss relates to a specific aspect of "preserving pre-trained knowledge," as shown by ablation studies.

## Limitations & Future Work

- **Dependence on linearized encoders**: Real CLIP models are non-linear Transformers; the closed-form solution is a first-order approximation that may degrade after several epochs.
- **Computational cost of WMA**: Storing running averages and kernel weights increases computational overhead compared to EMA, potentially straining memory for models larger than 8B.
- **Heuristic Kernel Choice**: Beta(0.5, 0.5) was chosen empirically; other endpoint-aware kernels (e.g., arcsine) were not fully explored.
- **Scope of verification**: Effectiveness remains unknown for other multimodal models like DINO, BLIP-2, or SigLIP.
- **ID-OOD Trade-off**: TRACER exhibits lower ID performance compared to Model Stock, which may be a consideration for specific deployment scenarios.

## Related Work & Insights

- **vs. CaRot (Oh et al. 2024)**: CaRot uses EMA-based SD; TRACER improves OOD by +1.53 average by replacing it with a WMA teacher to avoid collapse.
- **vs. LP-FT / FLYP**: These address "initialization shift," whereas TRACER addresses "training dynamics shift"; they are orthogonal and combinable.
- **vs. WiSE-FT / Model Stock**: These are post-hoc weight averaging methods. TRACER is an in-loop regularization method that does not strictly require weight averaging.
- **vs. Mean Teacher (Tarvainen & Valpola 2017)**: TRACER serves as a finite-horizon correction for the classic EMA teacher approach.
- **vs. L2-SP**: This work proves that the uniform blending of L2-SP is geometrically inferior to the structured decomposition of SD.
- **Inspiration**: WMA can be a general fix for collapse issues in any self-distillation or EMA-teacher-based work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [\[ICML 2026\] The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)
- [\[AAAI 2026\] FineXtrol: Controllable Motion Generation via Fine-Grained Text](../../AAAI2026/self_supervised/finextrol_controllable_motion_generation_via_fine-grained_text.md)
- [\[CVPR 2026\] D2Dewarp: Dual Dimensions Geometric Representation Learning Based Document Image Dewarping](../../CVPR2026/self_supervised/d2dewarp_dual_dimensions_geometric_representation_learning_based_document_image_.md)
- [\[CVPR 2026\] VT-Intrinsic: Physics-Based Decomposition of Reflectance and Shading using a Single Visible-Thermal Image Pair](../../CVPR2026/self_supervised/vt-intrinsic_physics-based_decomposition_of_reflectance_and_shading_using_a_sing.md)

</div>

<!-- RELATED:END -->
