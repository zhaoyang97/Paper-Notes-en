---
title: >-
  [Paper Note] Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards
description: >-
  [ICML2026][Image Generation][Inference-time guidance] Addressing the off-manifold drift problem in flow models during inference-time guidance under multi-objective compositional rewards…
tags:
  - "ICML2026"
  - "Image Generation"
  - "Inference-time guidance"
  - "flow matching"
  - "compositional rewards"
  - "gradient conflict"
  - "off-manifold drift"
date: 2026-05-08
content_hash: 4819658ec6dcc6ec
---

# Conflict-Aware Additive Guidance for Flow Models under Compositional Rewards

**Conference**: ICML2026  
**arXiv**: [2605.20758](https://arxiv.org/abs/2605.20758)  
**Code**: https://github.com/yuxuehui/CAR-guidance  
**Area**: Image Generation  
**Keywords**: Inference-time guidance, flow matching, compositional rewards, gradient conflict, off-manifold drift

## TL;DR

Addressing the off-manifold drift problem in flow models during inference-time guidance under multi-objective compositional rewards, this paper proposes Conflict-Aware Additive Guidance (CAR). By detecting gradient conflicts and dynamically switching to a learnable value-gradient correction, it improves identity preservation by 25.4% and planning success rate by 38.75% with minimal additional computational cost.

## Background & Motivation

**Background**: Continuous-time flow models (Rectified Flow, Flow Matching) have become powerful generation paradigms. Inference-time guidance achieves controllable generation without fine-tuning by superimposing a gradient term $g_t(x_t,t)$ on the base velocity field.

**Limitations of Prior Work**: Approximate guidance methods (e.g., $g^{\text{cov-G}}$) are computationally efficient but prone to off-manifold drift when composing multiple reward functions—generated samples deviate from the data manifold into low-density regions, leading to image distortion or hallucinated jumps in planning trajectories. Exact guidance methods (Guidance Matching, GLASS-FKS) avoid drift, but the former requires ground-truth samples satisfying all constraints, while the latter incurs over $3\times$ the computational cost.

**Key Challenge**: When gradient directions of multiple reward functions conflict, the error of approximate guidance is sharply amplified with gradient misalignment $(1 - \cos\phi)$ and the number of reward functions $G$, forming "energy traps" that capture trajectories off the manifold.

**Goal**: Eliminate off-manifold drift in compositional reward scenarios at the computational scale of approximate guidance.

**Key Insight**: Starting from measure transport theory, the approximation error is decomposed into three attributable terms: coupling shift, gradient misalignment, and local approximation. Gradient misalignment is identified as the primary error source in compositional scenarios.

**Core Idea**: Use a conflict-aware gate to dynamically detect gradient conflict regions and activate lightweight learnable guidance for correction only in high-conflict areas.

## Method

### Overall Architecture

Given a pre-trained flow model $v_t(x_t,t)$ and compositional rewards $r(x_1) = \sum_{j=1}^G r_j(x_1)$, CAR modifies the base velocity field during inference to $v'_t = v_t + g^{\text{car}}$. The guidance term $g^{\text{car}}$ is a dynamic mixture of approximate guidance $g^{\text{approx}}$ and learnable guidance $g_\psi$: $g^{\text{car}} = (1 - w_t) g^{\text{approx}} + w_t g_\psi$. The weight $w_t$ is automatically determined by the degree of gradient conflict—switching to learned correction when conflict is high and retaining cheap approximation when conflict is low.

### Key Designs

1.  **Three-term Decomposition of Approximation Error (Theoretical Foundation)**:
    - **Function**: Reveals the sources of error in compositional guidance and guides method design.
    - **Mechanism**: The $W_2^2$ error between the exact target distribution and the approximate implementation is decomposed into three terms—(A) coupling shift error (assuming $\mathcal{P}(z) \approx 1$), (B) gradient misalignment error ($\propto G(G-1) \mu^2 (1-\cos\phi)$), and (C) local approximation error. Term (B) grows quadratically with reward count $G$ and is proportional to the gradient angular deviation.
    - **Design Motivation**: Provides theoretical proof that approximate guidance is sufficient for single rewards, but gradient misalignment is the primary error source in compositional scenarios, providing a theoretical basis for targeted correction.

2.  **Conflict-Aware Gating Mechanism**:
    - **Function**: Automatically determines whether to enable learned correction at each sampling step.
    - **Mechanism**: Calculates a raw conflict score $w_{\text{raw}} = 1 - \frac{2}{G(G-1)} \sum_{j<k} \frac{\langle g_j, g_k \rangle}{\|g_j\|\|g_k\| + \varepsilon}$ based on the average cosine similarity between all reward gradient pairs. This is mapped to $(0,1)$ as the mixing weight. When gradient directions are consistent, $w_t \approx 0$, using cheap approximation; when gradients conflict, $w_t \approx 1$, switching to learned guidance.
    - **Design Motivation**: Compared to projection-based de-confliction methods like PCGrad (which provide marginal improvements), the gating mechanism achieves "on-demand correction"—applying extra computation only in space-time regions where it is truly needed.

3.  **Terminal Value Regression (TVR)**:
    - **Function**: Stabilizes the training of the learnable guidance $g_\psi$.
    - **Mechanism**: Parametrizes a scalar value function $V_\psi(x_t, t)$ and defines $g_\psi = \nabla_{x_t} V_\psi$. Utilizing the determinacy of flow model ODE dynamics, it directly regresses the terminal reward $r(x_1)$ instead of bootstrapping. The loss is $\mathcal{L}(\psi) = \mathbb{E}[\mathbb{I}_t \cdot (r(x_1) - V_\psi(x_t,t))^2]$, where the mask $\mathbb{I}_t = \mathbf{1}(w(x_t) > \tau)$ restricts training to high-conflict regions.
    - **Design Motivation**: TVR avoids the "deadly triad" of Fitted Value Evaluation (function approximation, off-policy, bootstrapping) by eliminating bootstrapping to ensure stable convergence; masked training further saves computation.

## Key Experimental Results

### Main Results — Synthetic Dataset (2D Mixed Gaussian)

| Method | Posterior Coverage (PC) ↑ | Constraint Satisfaction (CS) ↑ | Inference Time (ms) ↓ | Training Data Size |
| :--- | :--- | :--- | :--- | :--- |
| $g^{\text{cov-G}}$ | 71.70% | 89.56% | 0.38 | — |
| PCGrad | 75.20% | 92.45% | 0.42 | — |
| GM | 84.50% | 99.99% | 2.81 | 10,240k |
| GLASS-FKS | 90.80% | 99.71% | 296 | — |
| **Ours (CAR)** | **93.80%** | **100.00%** | 4.20 | 1,574k |

> Under conflicting constraints [1,0], nearly 30% of $g^{\text{cov-G}}$ samples deviate from the manifold; CAR reduces the drift rate to 6.2%, with computational cost only 1/70 of GLASS-FKS.

### Ablation Study

| Task | Method | Violations ↓ | Success Rate ↑ | Key Gain |
| :--- | :--- | :--- | :--- | :--- |
| ManiSkill2 StackCube (Static) | $g^{\text{cov-G}}$ | 1.2 | 12% | — |
| ManiSkill2 StackCube (Static) | **Ours** | **0.1** | **72%** | Success Rate +60% |
| ManiSkill2 StackCube (Mixed) | $g^{\text{cov-G}}$ | 1.8 | 9% | — |
| ManiSkill2 StackCube (Mixed) | **Ours** | **0.4** | **61%** | Violation Rate -78% |
| Maze2D (Dynamic) | $g^{\text{cov-G}}$ | 0.9 | 42% | — |
| Maze2D (Dynamic) | **Ours** | **0.2** | **61%** | Success Rate +19% |
| CelebA-HQ Image Editing | $g^{\text{cov-G}}$ | ID=0.543 | — | — |
| CelebA-HQ Image Editing | **Ours** | **ID=0.681** | — | Identity Preservation +25.4% |

## Highlights & Insights

- Solid theoretical contribution: The three-term error decomposition clarifies the root cause of compositional guidance failure. The quadratic growth of $G(G-1)(1-\cos\phi)$ explains why multi-constraint scenarios are significantly harder than single-constraint ones.
- Pragmatic "on-demand correction": The conflict-aware gating ensures zero overhead when there is no conflict, only investing computation in conflict regions. Average inference time on synthetic data is only 1.65ms, far lower than exact methods.
- TVR ensures theoretical convergence by eliminating bootstrapping through direct terminal reward regression, leveraging the deterministic ODE property of flow models.
- Extensive cross-domain validation: Consistent effectiveness across 2D synthetic data, pixel-space image editing, and 3D point cloud robotic manipulation.

## Limitations & Future Work

- CLIP reward signals are not smooth; $g_\psi$ training in high-dimensional image editing can be unstable and may produce adversarial artifacts.
- Still requires online rollouts to collect training data (approx. 10 mins for Maze2D), not fully plug-and-play.
- Conflict threshold $\tau$ requires manual setting (0.20 or 0.50 used in experiments), necessitating hyperparameter tuning across different tasks.

## Related Work & Insights

- **Guidance Matching** (Feng et al., 2025): Precise guidance requiring ground-truth samples; CAR approaches its performance with significantly lower data requirements.
- **GLASS-FKS** (Holderrieth et al., 2026): Sampling-based exact method; suffers from high variance and high computational cost.
- **PCGrad** (Yu et al., 2020): Multi-task gradient projection for de-confliction; provides limited improvement in inference-time guidance scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ (Combining conflict detection with value-gradient correction in guided sampling is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive across 4 domains + theory + ablation + visualization)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivation and systematic experimental organization)
- Value: ⭐⭐⭐⭐ (Compositional constraints are a real pain point in practical deployment; high utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](../../NeurIPS2025/image_generation/entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[ICML 2026\] Adversarial Flow Models](adversarial_flow_models.md)
- [\[ICLR 2026\] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](../../ICLR2026/image_generation/uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)
- [\[ICLR 2026\] Translate Policy to Language: Flow Matching Generated Rewards for LLM Explanations](../../ICLR2026/image_generation/translate_policy_to_language_flow_matching_generated_rewards_for_llm_explanation.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](../../CVPR2026/image_generation/freqflow_frequency_aware_flow_matching.md)

</div>

<!-- RELATED:END -->
