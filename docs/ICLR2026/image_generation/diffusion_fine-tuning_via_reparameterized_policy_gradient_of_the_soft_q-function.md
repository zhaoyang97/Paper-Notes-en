---
title: >-
  [Paper Note] Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function
description: >-
  [ICLR 2026][Image Generation][Diffusion model fine-tuning] This paper proposes SQDF (Soft Q-based Diffusion Finetuning), which fine-tunes diffusion models under a KL-regularized RL framework via a training-free differentiable soft Q-function approximation and reparameterized policy gradients. Three complementary components—a discount factor, a consistency model, and an off-policy replay buffer—collectively optimize the target reward while effectively mitigating reward over-optimization, preserving sample naturalness and diversity.
tags:
  - ICLR 2026
  - Image Generation
  - Diffusion model fine-tuning
  - KL-regularized reinforcement learning
  - soft Q-function
  - reward over-optimization
  - text-to-image alignment
date: 2026-05-08
content_hash: 8b30e6093452bc8f
---

# Diffusion Fine-Tuning via Reparameterized Policy Gradient of the Soft Q-Function

**Conference**: ICLR 2026
**arXiv**: [2512.04559](https://arxiv.org/abs/2512.04559)
**Code**: [https://github.com/Shin-woocheol/SQDF](https://github.com/Shin-woocheol/SQDF)
**Area**: Image Generation
**Keywords**: Diffusion model fine-tuning, KL-regularized reinforcement learning, soft Q-function, reward over-optimization, text-to-image alignment

## TL;DR

This paper proposes SQDF (Soft Q-based Diffusion Finetuning), which fine-tunes diffusion models under a KL-regularized RL framework via a training-free differentiable soft Q-function approximation and reparameterized policy gradients. Three complementary components—a discount factor, a consistency model, and an off-policy replay buffer—collectively optimize the target reward while effectively mitigating reward over-optimization, preserving sample naturalness and diversity.

## Background & Motivation

Diffusion models have become the dominant paradigm for high-quality sample generation, yet practical deployment requires alignment with downstream objectives such as aesthetic quality, text-image correspondence, and human preference. Existing fine-tuning methods suffer severely from reward over-optimization, manifesting as:

**Semantic Collapse**: High-reward samples progressively lose semantic alignment with the original prompt, degenerating into unrecognizable abstract textures.

**Diversity Collapse**: Generated outputs converge toward highly homogeneous modes.

Limitations of prior work:
- **RL-based methods (DDPO)**: Do not exploit reward gradients, resulting in low optimization efficiency and rapid diversity collapse.
- **Direct backpropagation methods (DRaFT, ReFL)**: Leverage reward gradients but are prone to over-optimization.
- **KL-regularized methods**: Require training auxiliary value networks—which is notoriously unstable in the diffusion MDP setting—or rely on high-variance Monte Carlo gradient estimators.

**Key Challenge**: How can one exploit powerful reward gradient signals while avoiding over-optimization through KL regularization?

**Core Idea**: The diffusion process is modeled as an MDP; by leveraging the posterior mean approximation from Tweedie's formula, a training-free, differentiable soft Q-function estimate is obtained, enabling direct model updates via reparameterized policy gradients.

## Method

### Overall Architecture

The diffusion reverse process is modeled as a finite-horizon MDP: state $s_t = (x_{T-t}, T-t)$, action $a_t = x_{T-t-1}$, policy $\pi_\theta(a_t|s_t) = p_\theta(x_{T-t-1}|x_{T-t})$. A sparse reward $r(x_0)$ is received at the terminal state $x_0$. The optimization objective is the KL-regularized expected reward.

Pipeline: (1) Sample from diffusion model $p_\theta$ and store in replay buffer → (2) Sample noisy observation $x_t$ from buffer → (3) Denoise one step with $p_\theta$ to obtain $x_{t-1}$ → (4) Predict clean sample $\hat{x}_0$ via consistency model $f_\psi$ → (5) Evaluate with reward model $r_\phi$ → (6) Update $p_\theta$ via reparameterized policy gradient.

### Key Designs

1. **Training-Free Soft Q-Function Approximation**:
   The key insight is that by recursively unrolling the soft Bellman equation and applying the one-step posterior mean approximation (Tweedie's formula), one obtains $Q_{\text{soft}}^*(x_t, x_{t-1}) \approx r(\hat{x}_0(x_{t-1}))$. This entirely avoids training a value network—a procedure known to be highly unstable in the diffusion MDP setting. Crucially, since the approximate Q-function requires only a single forward pass through the parameterized reward model, it is differentiable and admits direct gradient computation.

2. **Reparameterized Policy Gradient**:
   Using the reparameterization trick $x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \epsilon$, the policy gradient is expressed as:
   $$\nabla_\theta \mathcal{L} = \mathbb{E}_{x_t, \epsilon}[-\nabla_{x_{t-1}} r(\hat{x}_0) \cdot \nabla_\theta \mu_\theta + \alpha \nabla_\theta D_{KL}]$$
   This yields a low-variance, efficient gradient signal, significantly outperforming the REINFORCE estimator used in DDPO. The KL divergence term ensures the fine-tuned model does not deviate excessively from the pretrained distribution.

3. **Discount Factor γ for Improved Credit Assignment (4.2.1)**:
   Prior methods implicitly adopt $\gamma=1$, treating all denoising steps equally. However, early denoising steps (high noise levels) have negligible actual influence on the final sample. Introducing $\gamma < 1$ down-weights early steps exponentially, better reflecting the true causal structure. The authors also prove that under the discounted MDP, the Q-function approximation becomes $Q^* \approx \gamma^{t-1} r(\hat{x}_0)$, with upper and lower bounds coinciding under a first-order approximation.

4. **Consistency Model for Improved Q Estimation (4.2.2)**:
   Tweedie's formula yields highly inaccurate posterior mean estimates at high noise levels (clearly illustrated in Figure 2-b). A consistency model $f_\psi$ is used in place of Tweedie's formula for $\hat{x}_0$ prediction. Trained by distilling the integral of the probability flow ODE, the consistency model provides uniformly accurate $\hat{x}_0$ estimates across all timesteps (Figure 2-c), substantially improving Q-function approximation quality.

5. **Off-Policy Replay Buffer (4.2.3)**:
   The SQDF loss function naturally supports off-policy updates (since $x_t$ need not originate from the current policy). A replay buffer is introduced to reuse rare high-reward diverse samples, improving mode coverage and managing the reward–diversity trade-off.

### Loss & Training

Final SQDF loss:
$$\mathcal{L}_{\text{SQDF}} = \mathbb{E}_{x_t \sim \mathcal{D}, x_{t-1} \sim p_\theta}[-\gamma^{t-1} r(f_\psi(x_{t-1})) + \alpha D_{KL}(p_\theta \| p')]$$

- DDPM 50-step sampling
- Base model: Stable Diffusion v1.5 with LoRA fine-tuning
- Small-scale experiments: $\gamma=0.9$, $\alpha=2$, lr=$1\times10^{-3}$, LoRA rank=4, batch=64, 2000 steps
- Large-scale experiments: $\gamma=0.93$, $\alpha=0.05$, lr=$5\times10^{-4}$, LoRA rank=32, batch=258, 500 steps
- Consistency model: LCM-LoRA

## Key Experimental Results

### Main Results

**Text-to-image fine-tuning (Stable Diffusion v1.5, optimizing aesthetic score / HPS):**

From qualitative and quantitative results in Figures 3 and 4:
- ReFL and DRaFT achieve high aesthetic scores but suffer sharp drops in alignment (ImageReward, HPS) and diversity (LPIPS, DreamSim).
- DDPO fails to reach comparable aesthetic scores and exhibits rapid diversity collapse.
- **SQDF consistently achieves the highest alignment and diversity at equivalent reward levels.**

**Comparison with KL-regularized baselines (Figure 4 Pareto curves):**
SQDF occupies the Pareto frontier on nearly all metric pairs. By tuning $\alpha$, SQDF flexibly navigates the trade-off between higher reward and better diversity.

**Online black-box optimization (Table 1):**

| Method | Target (Aesthetic↑) | ImageReward↑ | HPS↑ | LPIPS-Div↑ | DreamSim-Div↑ |
|--------|---------------------|-------------|------|-----------|--------------|
| PPO+KL | 6.63 | -1.35 | 0.24 | 0.47 | 0.44 |
| SEIKO-Bootstrap | 7.80 | -1.69 | 0.23 | 0.36 | 0.24 |
| SEIKO-UCB | 7.49 | -1.08 | 0.24 | 0.40 | 0.32 |
| **SQDF-Bootstrap** | **7.87** | **1.14** | — | — | — |

SQDF dominates all evaluation metrics, most notably lifting ImageReward from negative to positive, demonstrating robustness to inaccurate reward proxies in the black-box optimization setting.

### Ablation Study

| Configuration | Aesthetic Score | DreamSim-Div | LPIPS-Div |
|--------------|----------------|-------------|-----------|
| SQDF (full) | 7.87 | 0.58 | 0.56 |
| w/o consistency model | 7.10 | 0.62 | 0.59 |
| w/o replay buffer | 8.06 | 0.56 | 0.55 |

| Discount Factor | Effect |
|----------------|--------|
| $\gamma=1$ | Higher aesthetic score but severe degradation in alignment and diversity |
| $\gamma=0.9$ | Balanced optimization speed and sample quality |
| $\gamma=0.85$ | Slower optimization but best diversity |

### Key Findings

- The consistency model is critical for accelerating convergence—removing it reduces the target reward from 7.87 to 7.10.
- The replay buffer primarily protects diversity; its removal yields a slightly higher reward (8.06) at the cost of reduced diversity.
- $\gamma$ provides an explicit control knob for the trade-off between optimization speed and sample quality.
- SQDF transfers effectively to SDXL (2.6B), with relative improvements highly consistent with those on SD 1.5.

## Highlights & Insights

- The "training-free Q-function" concept is remarkably elegant—Tweedie's formula transforms the intractable value function training problem into a simple reward evaluation.
- The introduction of discount factor $\gamma$ is straightforward yet well-grounded, supported by both theoretical derivation (first-order coincidence of upper and lower bounds) and thorough empirical validation.
- The consistency model serves as a superior drop-in replacement for Tweedie's formula, outperforming multi-step DDIM (4-step DDIM caused training instability).
- Off-policy update compatibility is a structural advantage of SQDF over DDPO/DRaFT, which are constrained to on-policy samples.
- The experimental design is comprehensive: beyond baseline comparisons, Pareto curve comparisons against KL-augmented baselines confirm that the advantage stems from the framework itself rather than regularization alone.

## Limitations & Future Work

- The one-step Q-function approximation is mathematically coarse—the first-order approximation of the log moment-generating function may be insufficient when $r/\alpha$ is large.
- The method depends on consistency model quality; inaccurate LCM-LoRA estimates will propagate bias into Q-function approximations.
- Validation is currently limited to the Stable Diffusion family; applicability to newer architectures such as flow matching remains untested.
- The replay buffer management strategy (priority sampling) may require task-specific tuning.
- Computational cost analysis is insufficient—per-step overhead of 62s (aesthetic) / 401s (HPS) warrants further optimization.

## Related Work & Insights

- DDPO (Black et al., 2023): A PPO-based method that does not exploit gradients; simple but inefficient.
- DRaFT/ReFL: Direct gradient backpropagation; efficient but severely prone to over-optimization.
- SEIKO (Uehara et al., 2024): KL-regularized direct backpropagation, but relies on truncated backpropagation through the denoising chain.
- The proposed "training-free Q-function + reparameterization" framework may generalize to other generative models requiring RL fine-tuning, such as language model RLHF and protein design.
- The use of consistency models here inspires a broader paradigm of "distilled models as Q-value estimators."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Training-free differentiable Q-function estimation combined with three complementary components is an ingenious design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two task settings, comprehensive ablations, Pareto curve comparisons, and SDXL extension.
- **Writing Quality**: ⭐⭐⭐⭐ — The method section is well-structured, though relegating several derivations to the appendix increases reading difficulty.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a principled solution for diffusion model alignment with open-source code and generalizable methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild](direct_reward_fine-tuning_on_poses_for_single_image_to_3d_human_in_the_wild.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](../../AAAI2026/image_generation/self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[CVPR 2026\] CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think](../../CVPR2026/image_generation/craft_aligning_diffusion_models_with_finetuning_is_easier_than_you_think.md)
- [\[AAAI 2026\] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](../../AAAI2026/image_generation/dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)
- [\[ICCV 2025\] ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning](../../ICCV2025/image_generation/shortft_diffusion_model_alignment_via_shortcut-based_fine-tuning.md)

</div>

<!-- RELATED:END -->
