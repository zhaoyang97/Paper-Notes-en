---
title: >-
  [Paper Note] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance
description: >-
  [CVPR 2026][Image Generation][Classifier-Free Guidance] This paper reinterprets Classifier-Free Guidance (CFG) as a feedback control process within flow matching diffusion models, proposes a unified framework termed CFG-Ctrl, and introduces SMC-CFG — a nonlinear feedback guidance mechanism grounded in sliding mode control (SMC) — which substantially improves semantic consistency and generation robustness at large guidance scales.
tags:
  - CVPR 2026
  - Image Generation
  - Classifier-Free Guidance
  - Control Theory
  - Sliding Mode Control
  - Flow Matching
  - Text-to-Image Generation
date: 2026-05-08
content_hash: 91565268762ab60a
---

# CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance

**Conference**: CVPR 2026
**arXiv**: [2603.03281](https://arxiv.org/abs/2603.03281)
**Code**: [Project Page](https://hanyang-21.github.io/CFG-Ctrl)
**Area**: Image Generation
**Keywords**: Classifier-Free Guidance, Control Theory, Sliding Mode Control, Flow Matching, Text-to-Image Generation

## TL;DR

This paper reinterprets Classifier-Free Guidance (CFG) as a feedback control process within flow matching diffusion models, proposes a unified framework termed CFG-Ctrl, and introduces SMC-CFG — a nonlinear feedback guidance mechanism grounded in sliding mode control (SMC) — which substantially improves semantic consistency and generation robustness at large guidance scales.

## Background & Motivation

**Central role of CFG**: CFG is a critical technique for enhancing semantic alignment in diffusion models, widely adopted in text-to-image and text-to-video tasks. However, its underlying mechanism has traditionally been treated as a simple linear extrapolation between conditional and unconditional predictions.

**Inherent limitations of linear extrapolation**: Standard CFG at large guidance scales tends to produce color oversaturation, structural distortion, and loss of fine details. Its extreme sensitivity to the guidance scale severely restricts the practically usable guidance range.

**Limitations of prior improvements**: Existing methods (Weight Scheduler, APG, CFG-Zero⋆, Rectified-CFG++, etc.) improve CFG from various perspectives but fundamentally remain linear control laws, unable to guarantee stable convergence in the highly nonlinear dynamics of generative processes.

**Natural decay of the error signal**: The authors observe that the discrepancy between conditional and unconditional velocity predictions diminishes progressively throughout denoising. This discrepancy naturally constitutes an "error signal" in control-theoretic terms, providing a theoretical basis for reinterpreting CFG.

**Insights from control theory**: Sliding mode control (SMC) has been extensively validated for robustness and convergence in nonlinear dynamical systems, making it a natural fit for resolving CFG instability at high guidance scales.

**Absence of a unified theoretical perspective**: Prior CFG variants lack a unified analytical framework, making it difficult to systematically compare and design new guidance strategies.

## Method

### Overall Architecture: CFG-Ctrl

The flow matching sampling process is modeled as a continuous-time controlled dynamical system:

$$\frac{d\mathbf{x}_t}{dt} = \mathbf{v}_\theta(\mathbf{x}_t, t) + \mathbf{u}_t$$

where the control signal is decomposed into two core components:

$$\mathbf{u}_t = K_t \, \Pi_t(\mathbf{e}(t))$$

- **Guidance Schedule $K_t$**: controls guidance intensity (scalar/matrix)
- **Direction Operator $\Pi_t$**: adjusts the correction direction (identity/projection, etc.)
- **Semantic Error $\mathbf{e}(t) = \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{c}) - \mathbf{v}_\theta(\mathbf{x}_t, t, \varnothing)$**

Within this framework: standard CFG = proportional controller (P-control); Weight Scheduler = time-varying gain scheduling; APG = projection feedback control; CFG-Zero⋆ = projection feedback control; Rectified-CFG++ = model predictive control.

### Key Designs: SMC-CFG (Sliding Mode Control Guidance)

**1. Sliding surface definition**: An exponentially convergent sliding surface is constructed in the phase space $(\mathbf{e}, \dot{\mathbf{e}})$ of the semantic error:

$$\mathbf{s}(t) = \dot{\mathbf{e}}(t) + \lambda \mathbf{e}(t)$$

When $\mathbf{s}(t) = \mathbf{0}$, the error converges monotonically to zero along the exponential trajectory $\mathbf{e}(t) = \mathbf{e}(T)\exp(-\lambda t)$.

**2. Switching control law**: A nonlinear switching term is introduced to drive the system trajectory onto the sliding surface:

$$\Delta\mathbf{e}(t) = -k \cdot \mathrm{sign}(\mathbf{s}(t))$$

**3. Guidance update**: The final guided velocity is:

$$\hat{\mathbf{v}}_t = \mathbf{v}_\theta(\mathbf{x}_t, t, \varnothing) + w \cdot (\mathbf{e}(t) + \Delta\mathbf{e}(t))$$

### Theoretical Guarantees

Based on Lyapunov stability analysis ($V(\mathbf{s}) = \frac{1}{2}\|\mathbf{s}\|^2$), the paper proves that under the condition $k \cdot b_{\min} > \delta$, the system converges to the sliding surface in finite time:

$$\|\mathbf{s}(t)\| = 0, \quad t \leq \frac{\|\mathbf{s}(0)\|}{\eta}, \quad \eta = k \cdot b_{\min} - \delta > 0$$

### Hyperparameters

- **$\lambda$**: sliding surface shape parameter controlling the convergence rate (optimal at $\lambda=5$ in experiments)
- **$k$**: switching control gain controlling the attraction strength toward the sliding surface (trade-off between semantic alignment and image fidelity)

## Key Experimental Results

### Main Results: Text-to-Image Generation (MS-COCO 5K)

Evaluation on three mainstream models — SD3.5 (8B), Flux-dev (12B), and Qwen-Image (20B):

| Model | Method | FID↓ | CLIP↑ | Aesthetic↑ | ImageReward↑ | HPSv2.1↑ | MPS↑ |
|-------|--------|------|-------|-----------|-------------|----------|------|
| SD3.5 | CFG | 21.42 | 0.3681 | 5.588 | 0.889 | 0.284 | 7.248 |
| SD3.5 | **SMC-CFG** | **20.04** | **0.3694** | 5.579 | **0.949** | **0.288** | **7.572** |
| Flux-dev | CFG | 27.32 | 0.3692 | 5.540 | 0.875 | 0.283 | 7.839 |
| Flux-dev | **SMC-CFG** | **26.40** | **0.3743** | **5.734** | **1.056** | **0.302** | **8.231** |
| Qwen-Image | CFG | 35.43 | 0.3815 | 5.600 | 1.106 | 0.304 | 8.185 |
| Qwen-Image | **SMC-CFG** | **33.37** | **0.3856** | **5.629** | **1.204** | **0.311** | **8.432** |

SMC-CFG comprehensively outperforms standard CFG and other variants (CFG-Zero⋆, Rect-CFG++) across all models, with particularly notable improvements on human preference metrics such as ImageReward and MPS.

### Ablation Study

**$\lambda$ ablation** (fixed $k=0.1$, Flux-dev): $\lambda=5$ achieves the lowest FID (25.95) and highest CLIP (0.3709); performance degrades when $\lambda$ is either too large or too small.

**$k$ ablation** (fixed $\lambda=5$): smaller $k$ favors lower FID (image fidelity), while larger $k$ favors higher CLIP (semantic alignment), forming a clear quality–fidelity trade-off.

### Key Findings

- **Robustness at large guidance scales**: As the CFG scale increases, standard CFG suffers sharp quality degradation, whereas SMC-CFG remains stable, substantially expanding the usable guidance range.
- **Model-agnostic generality**: The method is effective across models ranging from 8B to 20B parameters without model-specific hyperparameter tuning ($\lambda$ and $k$ are fixed across models).
- **Qualitative comparisons**: In complex semantic scenarios involving spatial relationships, text rendering, clothing details, and human actions, SMC-CFG demonstrates superior text consistency and fine-grained fidelity over all baselines.

## Highlights & Insights

- **Unified theoretical framework**: CFG-Ctrl is the first work to systematically unify CFG and its variants (P-control, gain scheduling, projection feedback, MPC) from a control-theoretic perspective, offering a principled theoretical reference for guidance strategy design.
- **Nonlinear control breakthrough**: SMC-CFG is the first work to introduce sliding mode control into diffusion guidance, replacing linear extrapolation with nonlinear feedback and fundamentally addressing instability at high guidance scales.
- **Rigorous theoretical guarantees**: The paper provides complete Lyapunov stability analysis and finite-time convergence proofs, which are rarely seen in the diffusion model guidance literature.
- **Plug-and-play, training-free**: The method only modifies the guidance computation during sampling and requires no changes to the model training procedure.

## Limitations & Future Work

- The $\mathrm{sign}$ function in sliding mode control may introduce chattering; the paper does not discuss the effect of smooth approximations (e.g., $\tanh$).
- Validation is limited to text-to-image generation and has not been extended to other modalities such as text-to-video or 3D generation.
- The Lyapunov analysis relies on assumptions of $\sigma_{\min}(\Gamma_s) \geq b_{\min} > 0$ and boundedness, whose validity for practical neural networks is not sufficiently verified.
- Ablation experiments are conducted solely on Flux-dev; whether the optimal hyperparameters generalize across different models is not analyzed in detail.

## Related Work & Insights

- **CFG variants**: CFG++ (Chung et al., 2024), APG (orthogonal decomposition), Weight Scheduler (time-varying weights), CFG-Zero⋆ (Fan et al., 2025, optimizing guidance scale), Rectified-CFG++ (Yang et al., predictor-corrector)
- **Flow matching foundations**: Flow Matching (Lipman et al.), Rectified Flow (Liu et al.)
- **Control theory applications**: PID control, MPC, adaptive control, Sliding Mode Control (Edwards & Spurgeon, 1998)

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The control-theoretic reinterpretation of CFG is highly original; introducing SMC into diffusion guidance is a first.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive evaluation across three mainstream models and 8 metrics, though non-T2I task validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are complete and rigorous; the unified comparison in Table 1 is particularly well-executed.
- **Value**: ⭐⭐⭐⭐ — Provides a systematic control-theoretic toolkit for CFG improvement and is likely to inspire more advanced control strategies.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)
- [\[NeurIPS 2025\] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](../../NeurIPS2025/image_generation/rectified-cfg_for_flow_based_models.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)

<!-- RELATED:END -->
