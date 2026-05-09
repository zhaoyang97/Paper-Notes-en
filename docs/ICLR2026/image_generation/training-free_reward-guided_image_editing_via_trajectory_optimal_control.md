---
title: >-
  [Paper Note] Training-Free Reward-Guided Image Editing via Trajectory Optimal Control
description: >-
  [ICLR 2026][Image Generation][Optimal Control] This paper reformulates reward-guided image editing as a trajectory optimal control problem, treating the reverse process of diffusion/flow models as a controllable trajectory. By iteratively optimizing the entire trajectory via Pontryagin's Maximum Principle (PMP) with adjoint state methods, it achieves effective reward-guided editing without training and without reward hacking.
tags:
  - ICLR 2026
  - Image Generation
  - Optimal Control
  - Reward-Guided
  - training-free
  - Adjoint State
  - Pontryagin's Maximum Principle
date: 2026-05-08
content_hash: c97d0424ffad101b
---

# Training-Free Reward-Guided Image Editing via Trajectory Optimal Control

**Conference**: ICLR 2026
**arXiv**: [2509.25845](https://arxiv.org/abs/2509.25845)
**Code**: N/A
**Area**: Diffusion Models / Image Editing
**Keywords**: Optimal Control, Reward-Guided, training-free, Adjoint State, Pontryagin's Maximum Principle

## TL;DR
This paper reformulates reward-guided image editing as a trajectory optimal control problem, treating the reverse process of diffusion/flow models as a controllable trajectory. By iteratively optimizing the entire trajectory via Pontryagin's Maximum Principle (PMP) with adjoint state methods, it achieves effective reward-guided editing without training and without reward hacking.

## Background & Motivation
**Background**: Reward-guided sampling has proven successful in T2I generation (DPS, FreeDoM, TFG) by leveraging differentiable reward functions to guide the generation process at inference time. However, these methods are designed for sampling, not specifically for editing.

**Limitations of Prior Work**: Reward-guided editing is more challenging than generation — it must simultaneously maximize rewards and preserve the core identity of the source image. Naïve approaches (inversion + guided sampling) perform poorly: for complex nonlinear reward functions, guidance based on intermediate noisy images or single-step approximations degrades structural fidelity. Direct gradient ascent, while directionally correct, ignores image priors and produces adversarial samples.

**Key Challenge**: Existing guidance methods face a dilemma in editing scenarios — overly strong guidance destroys structure, while overly weak guidance yields insufficient reward improvement. Moreover, they lack theoretical justification for guidance scale selection, requiring extensive hyperparameter tuning.

**Goal**: How can arbitrary differentiable reward functions guide editing without model training, while maintaining structural consistency with the source image?

**Key Insight**: Optimal control theory — elevating the problem from "single-step guidance" to "full-trajectory optimization."

**Core Idea**: Optimize the entire generation trajectory (rather than guiding intermediate states at each step) to simultaneously maximize terminal reward and preserve consistency with the source image.

## Method

### Overall Architecture
Given a source image $\bm{x}_1$, an initial trajectory $\{\bm{x}_t\}_{t=T}^{1}$ is first generated via deterministic inversion. This trajectory is then iteratively optimized: each iteration comprises three steps — (1) backward computation of the adjoint state $p_t$; (2) update of the control variable $u_t$; (3) forward simulation of a new trajectory using the updated control. The terminal state $\bm{x}_1^{u^*}$ is taken as the edited result.

### Key Designs
1. **Unified SDE Framework**: The sampling processes of diffusion models and Flow Matching models are unified as:
$$d\bm{x}_t = b(\bm{x}_t, t) dt + \sigma_t d\mathbf{B}_t$$
where the drift term $b$ takes different expressions depending on the model type (Eq. 4–5), enabling a single theoretical framework to handle both model classes.

2. **Optimal Control Problem Formulation**: A control term $u_t$ is introduced into the drift, and the following problem is solved:
$$\min_{u \in \mathcal{U}} \int_T^1 \frac{1}{2} \|u(\bm{x}_t^u, t)\|^2 dt - r(\bm{x}_1^u)$$
$$\text{s.t.} \quad d\bm{x}_t^u = (b(\bm{x}_t^u, t) + u(\bm{x}_t^u, t)) dt + \sigma_t d\mathbf{B}_t, \quad \bm{x}_T^u = \bm{x}_T$$
The objective maximizes the terminal reward $r(\bm{x}_1^u)$ while minimizing the control norm (regularization to prevent the trajectory from deviating too far from the manifold).

3. **PMP-Based Iterative Solver**: PMP provides necessary conditions for the optimal trajectory (Eq. 8–10), involving three coupled differential equations — the state equation (forward), the adjoint equation (backward), and the optimality condition ($u_t^* = -p_t^*$). Since joint solving is intractable, a coordinate-descent-style iterative method is employed:

    - **Step 1**: Fix the current trajectory $\bm{x}_t$ and solve the adjoint equation backward to obtain $p_t$ (terminal condition $p_1^* = -\nabla_{\bm{x}_1} r(\bm{x}_1^*)$).
    - **Step 2**: Update the control variable via $u_t = u_t - \lambda(u_t + p_t)$.
    - **Step 3**: Forward-simulate a new trajectory $\bm{x}_t$ using the updated $u_t$.

   The Jacobian-vector product $[\nabla_{\bm{x}_t} b(\bm{x}_t^*, t)]^\top p_t^*$ in the adjoint equation is computed efficiently via automatic differentiation.

4. **Inversion Strategy**: DDIM inversion is used for diffusion models, and time-reversed ODE for Flow Matching — both are deterministic ($\sigma_t = 0$), ensuring the initial trajectory faithfully reconstructs the source image.

### Loss & Training
- **Training-free method**: Pure inference-time optimization.
- The core objective is the optimal control cost functional: terminal reward + control norm regularization.
- The reward function $r(\cdot)$ can be scaled by weight $w$ to uniformly control the guidance scale.
- StableDiffusion 1.5 (diffusion) and StableDiffusion 3 (Flow Matching) are used as backbone models.
- Different reward functions are used per task: ImageReward (human preference), Gram matrix discrepancy (style transfer), classifier logits (counterfactual generation), CLIPScore (text-guided editing).

## Key Experimental Results

### Main Results (Human Preference Task, SD 1.5)

| Method | ImageReward↑ | HPSv2↑ | CLIPScore↑ | Aesthetic↑ | LPIPS↓ | CLIP-I_src↑ |
|--------|-------------|--------|-----------|-----------|--------|------------|
| None | 0.154 | 0.239 | 0.289 | 6.052 | 0.000 | 1.000 |
| Gradient Ascent | **1.909** | 0.225 | 0.288 | 5.578 | 0.147 | 0.920 |
| Inv+DPS | 1.599 | 0.232 | 0.265 | 5.828 | 0.288 | 0.851 |
| Inv+TFG | 1.705 | 0.236 | 0.273 | 5.633 | 0.293 | 0.840 |
| **Ours** | 1.891 | **0.253** | **0.290** | **6.109** | 0.172 | **0.924** |

*The proposed method achieves reward scores comparable to GA while leading on all generalization metrics and maintaining superior source image consistency. GA achieves the highest reward but suffers from severe reward hacking (degraded validation metrics).*

### Style Transfer Task

| Method | ‖ΔG‖_F↓ | CLIP-I_sty↑ | DINO_sty↑ | CLIP-I_src↑ |
|--------|---------|------------|----------|------------|
| Gradient Ascent | **4.874** | 0.527 | 0.195 | 0.837 |
| Inv+DPS | 6.844 | 0.540 | 0.169 | 0.686 |
| Inv+FreeDoM | 5.462 | 0.563 | 0.225 | 0.621 |
| **Ours** | 5.019 | **0.578** | **0.247** | 0.717 |

*The proposed method leads on all validation metrics (CLIP-I_sty, DINO_sty) while maintaining substantially better structural preservation than guided sampling baselines.*

### Key Findings
- Gradient ascent (GA) achieves the strongest target reward but consistently exhibits reward hacking — validation metrics degrade, indicating overfitting to the reward function without genuine quality improvement.
- Guided sampling methods (DPS/FreeDoM/TFG) generally damage source image structure in editing scenarios, with large deterioration in LPIPS and CLIP-I_src.
- The proposed method avoids reward hacking through full-trajectory optimization: target reward approaches the optimum while generalization metrics consistently lead.
- Control norm regularization is critical: it bounds trajectory deviation, acting as an implicit structural preservation constraint.
- In counterfactual generation, GA performs reasonably well due to the use of a robust classifier — suggesting that reward function characteristics affect the relative performance of each method.
- The method applies to both diffusion models and Flow Matching models without modification.

## Highlights & Insights
- **Theoretical Rigor**: A complete editing framework is derived from optimal control theory; PMP provides necessary optimality conditions.
- **Unified Framework**: Diffusion and Flow Matching models are handled jointly under a single SDE control formulation.
- **No Guidance Scale Search**: Guidance strength across all steps is governed by a single weight $w$ with theoretical justification.
- **Reward Hacking Prevention**: Control norm regularization naturally prevents excessive editing.
- **Applicable to Abstract Rewards**: Not limited to text conditioning; applicable to concepts difficult to express in language, such as human preference or style.
- **Comparison with Adjoint Matching**: The latter requires model fine-tuning (altering the entire distribution), whereas the proposed method optimizes only a single trajectory (inference-time editing).

## Limitations & Future Work
- The backbone models used (SD 1.5 / SD 3) are relatively outdated; validation on modern architectures such as FLUX has not been conducted.
- The iterative optimization requires multiple forward and backward passes, incurring significant computational overhead ($N$ iterations × trajectory length).
- Computing the Jacobian-vector product in the adjoint equation assumes $b$ is differentiable and its Jacobian is tractable, which may not hold for certain models.
- Deterministic inversion quality directly affects editing quality — additional handling may be needed for CFG-distilled models.
- No comparison is made against more sophisticated conditional editing methods (e.g., InstructPix2Pix, FLUX Kontext).
- Evaluation uses only 300 images, which is a relatively small scale.

## Related Work & Insights
- **DPS / FreeDoM / TFG**: All are training-free guidance methods, but rely on single-step guidance or one-step approximations, making them ineffective for complex nonlinear rewards.
- **Adjoint Matching**: Also employs PMP and adjoint states, but for model fine-tuning (SOC problem); the proposed method applies these tools to single-image inference-time editing.
- **FlowEdit**: An optimization-free flow editing method that directly manipulates text-conditioned flows.
- **RFIN / Rout et al.**: Apply an OC perspective to style personalization and Doob's h-transform, but not to reward-guided editing.
- **Takeaway**: Optimal control theory provides an elegant, theoretically grounded framework for inference-time intervention in generative models, and can be extended to reward-guided control in video editing or 3D generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Reformulating reward-guided editing as an OC problem is highly elegant; the PMP derivation is rigorous.
- Experimental Thoroughness: ⭐⭐⭐ — Four tasks provide broad coverage, but backbone models are outdated, evaluation scale is small, and no user study is included.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, and motivation is developed in a well-structured, progressive manner.
- Value: ⭐⭐⭐⭐ — Provides a novel theoretical framework for reward-guided editing; practical impact depends on validation on modern large-scale models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control](follow-your-shape_shape-aware_image_editing_via_trajectory-guided_region_control.md)
- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](../../AAAI2026/image_generation/melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[ICLR 2026\] EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)

<!-- RELATED:END -->
