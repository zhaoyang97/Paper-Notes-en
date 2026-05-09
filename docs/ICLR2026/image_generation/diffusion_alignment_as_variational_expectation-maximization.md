---
title: >-
  [Paper Note] Diffusion Alignment as Variational Expectation-Maximization
description: >-
  [ICLR 2026][Image Generation][diffusion alignment] This paper formalizes diffusion model alignment as a variational EM algorithm: the E-step employs test-time search (soft-Q-guided sampling with importance sampling) to explore multimodal, high-reward trajectories, while the M-step distills the search results into model parameters via forward-KL minimization. The approach simultaneously achieves high reward and high diversity on both image generation and DNA sequence design tasks.
tags:
  - ICLR 2026
  - Image Generation
  - diffusion alignment
  - expectation-maximization
  - test-time search
  - reward optimization
  - mode collapse prevention
date: 2026-05-08
content_hash: 2fe0f584fad22623
---

# Diffusion Alignment as Variational Expectation-Maximization

**Conference**: ICLR 2026
**arXiv**: [2510.00502](https://arxiv.org/abs/2510.00502)
**Code**: [https://github.com/Jaewoopudding/dav](https://github.com/Jaewoopudding/dav)
**Area**: Diffusion Models / Alignment
**Keywords**: diffusion alignment, expectation-maximization, test-time search, reward optimization, mode collapse prevention

## TL;DR
This paper formalizes diffusion model alignment as a variational EM algorithm: the E-step employs test-time search (soft-Q-guided sampling with importance sampling) to explore multimodal, high-reward trajectories, while the M-step distills the search results into model parameters via forward-KL minimization. The approach simultaneously achieves high reward and high diversity on both image generation and DNA sequence design tasks.

## Background & Motivation

**Background**: Diffusion model alignment—making generation conform to external rewards—has primarily followed two paradigms: reinforcement learning (DDPO/DPOK) and direct backpropagation through the reward (DRaFT/AlignProp).

**Limitations of Prior Work**: RL-based methods optimize a reverse-KL objective, inducing mode-seeking behavior that leads to mode collapse and loss of diversity. Direct backpropagation methods rely on gradient signals from the reward model and are prone to reward over-optimization. Both families exhibit high reward but sharp deterioration in image quality and diversity in the late stages of training.

**Key Challenge**: The fundamental tension between reward optimization and diversity preservation. Reverse-KL is inherently mode-seeking and tends to collapse onto a single mode.

**Goal**: To design an alignment framework that effectively optimizes reward while preserving sample diversity and naturalness, applicable to both continuous (image) and discrete (DNA) diffusion models.

**Key Insight**: The alignment problem is formalized as variational EM by introducing an optimality variable $\mathcal{O}$ and a trajectory latent variable $\tau$. The E-step identifies a multimodal posterior, and the M-step performs forward-KL (mode-covering) distillation. Forward-KL naturally encourages coverage of all modes rather than concentration on a single one.

**Core Idea**: The E-step uses test-time search to discover multimodal high-reward samples; the M-step distills them via forward-KL to preserve diversity. Iterating this cycle progressively improves both objectives.

## Method

### Overall Architecture

DAV alternates between two steps during training:
1. **E-step (Exploration)**: Starting from the current model, test-time search (gradient-guided sampling + importance sampling) generates high-reward and diverse trajectories, approximating the variational posterior $\eta_k^*$.
2. **M-step (Distillation)**: The model is trained on trajectories found in the E-step by minimizing the forward-KL $D_{\text{KL}}(\eta_k^* \| p_\theta)$, equivalent to maximizing the log-likelihood of the searched trajectories $-\log p_\theta(\tau)$.

### Key Designs

1. **Variational EM Formulation**:

    - **Function**: Converts reward optimization into marginal likelihood maximization with respect to an optimality variable.
    - **Mechanism**: Defines $p(\mathcal{O}=1|\tau) \propto \exp(\sum r_t/\alpha)$ with trajectory $\tau$ as the latent variable. The ELBO is $\mathcal{J}_{\alpha,\gamma}(\eta, p_\theta)$, with a discount factor $\gamma$ that attenuates credit assignment for time steps far from the terminal state.
    - **Design Motivation**: The EM framework naturally decouples exploration (E-step) from exploitation (M-step), and the forward-KL in the M-step is mode-covering, preventing collapse.

2. **E-step: Test-Time Search**:

    - **Function**: Approximates sampling from the optimal variational posterior $\eta_k^*(x_{t-1}|x_t) \propto p_{\theta_k}(x_{t-1}|x_t) \exp(Q_{\text{soft}}^*/\alpha)$.
    - **Mechanism**: A two-stage procedure—gradient-guided sampling (approximating $Q_{\text{soft}}$ as $\gamma^{t-1} r(\hat{x}_0(x_{t-1}))$ via Tweedie's formula) generates $M$ candidate particles, which are then refined by importance sampling.
    - **Design Motivation**: Naive on-policy reweighting (as in conventional EM-RL) introduces severe bias when the policy deviates from the posterior. Test-time search actively explores high-reward regions outside the current policy's support.

3. **M-step: Forward-KL Distillation**:

    - **Function**: Distills the trajectories found in the E-step into model parameters.
    - **Mechanism**: $\mathcal{L}_{\text{DAV}} = -\mathbb{E}_{\tau \sim \eta_k^*}[\log p_\theta(\tau)]$. An optional KL regularization term $\mathcal{L}_{\text{DAV-KL}} = \mathcal{L}_{\text{DAV}} + \lambda D_{\text{KL}}(p_\theta \| p_{\theta^0})$ constrains deviation from the pretrained model.
    - **Design Motivation**: Minimizing forward-KL is equivalent to maximizing the likelihood of the searched samples, which is mode-covering—the opposite of the mode-seeking reverse-KL used in RL-based methods—and thus naturally preserves diversity.

4. **Modular Design**:

    - The search algorithm in the E-step can be replaced by any test-time search method.
    - The framework is applicable to both continuous and discrete diffusion models.

### Loss & Training

- Built on SD v1.5; rewards include the LAION aesthetic score (differentiable) and compressibility (non-differentiable).
- EM iteration runs for 100 epochs.
- The E-step samples $M$ candidates per step, with importance sampling for selection.
- The discount factor $\gamma$ attenuates credit assignment for early time steps.

## Key Experimental Results

### Main Results (Text-to-Image, SD v1.5, Aesthetic Reward)

| Method | Aesthetic ↑ | LPIPS-A ↑ | ImageReward ↑ | Type |
|--------|-------------|-----------|---------------|------|
| Pretrained | 5.40 | 0.65 | 0.90 | — |
| DDPO | 6.83 | 0.48 | 0.27 | RL |
| DRaFT | 7.22 | 0.46 | 0.19 | Backprop |
| **DAV** | **8.04** | 0.53 | **0.95** | EM |
| DAV-KL | 6.99 | **0.58** | 1.13 | EM+KL |
| DAS (search only) | 7.22 | 0.65 | 1.07 | Inference-time |
| DAV Posterior | **9.18** | 0.53 | 0.91 | EM+search |

### Ablation Study

| Analysis | Key Finding |
|----------|-------------|
| DAV ELBO trend | ELBO increases monotonically (approximately); removing E-step search causes ELBO to decrease. |
| DAV vs. DAV-KL | KL regularization trades reward (8.04→6.99) for diversity (0.53→0.58). |
| DDPO/DRaFT at 100 epochs | Both exhibit severe mode collapse; ImageReward drops to negative values. |

### Key Findings
- DAV's reward (8.04) substantially surpasses DDPO (6.83) and DRaFT (7.22), while maintaining an ImageReward of 0.95 (close to the pretrained baseline of 0.90), indicating the absence of reward over-optimization.
- DDPO and DRaFT suffer severe over-optimization in late training, with ImageReward collapsing to 0.27 and 0.19, respectively.
- DAV Posterior (model + search at inference time) achieves an aesthetic score of 9.18, the highest among all evaluated methods.
- On DNA sequence design, the method comprehensively outperforms baselines across all three dimensions: reward, diversity, and naturalness.

## Highlights & Insights
- The choice of **forward-KL vs. reverse-KL** is the central insight. RL-based methods employ reverse-KL (mode-seeking → collapse), whereas DAV employs forward-KL (mode-covering → diversity preservation). This choice has clear theoretical motivation and is thoroughly validated experimentally.
- **Test-time search amortization** is a broadly applicable paradigm—search first, then distill—converting inference-time computation into model capacity. This idea transfers naturally to any setting requiring expensive inference-time search, such as code generation or molecular design.
- **Cross-modal applicability**: The same framework handles both continuous (image) and discrete (DNA) diffusion models, demonstrating the generality of the methodology.

## Limitations & Future Work
- The test-time search in the E-step increases training cost, as each EM iteration requires multiple ODE/forward passes.
- Approximating $Q^*_{\text{soft}}$ via Tweedie's formula is an approximation that may be inaccurate at high-noise time steps.
- Experiments are limited to SD v1.5; validation on larger models such as SDXL or Flux is absent.
- The optimal selection of the discount factor $\gamma$ has not been systematically studied.
- Forward-KL distillation may not precisely cover all modes of the posterior given finite samples and finite training steps.

## Related Work & Insights
- **vs. DDPO/DPOK**: Both are RL-based alignment approaches, but DAV replaces reverse-KL with forward-KL. The essential distinction is mode-covering vs. mode-seeking; DAV achieves higher reward without collapse.
- **vs. DRaFT/AlignProp**: Direct backpropagation is computationally efficient but relies on fragile gradient signals. DAV does not require a differentiable reward function.
- **vs. DAS (test-time search)**: DAS applies search only at inference time without updating model parameters. DAV distills search results back into the model, incurring no additional cost at inference time.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The variational EM perspective unifies RL-based alignment and test-time search; forward-KL distillation constitutes a key innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Dual-domain validation (image + DNA) with thorough analysis of training dynamics, though limited to SD v1.5.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear, and the motivation–method–experiment narrative is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses the core pain points of diffusion alignment (over-optimization and mode collapse) with broad applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)
- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[ICLR 2026\] Infinity and Beyond: Compositional Alignment in VAR and Diffusion T2I Models](infinity_and_beyond_compositional_alignment_in_var_and_diffusion_t2i_models.md)
- [\[ICLR 2026\] Improved Object-Centric Diffusion Learning with Registers and Contrastive Alignment (CODA)](improved_object-centric_diffusion_learning_with_registers_and_contrastive_alignm.md)

<!-- RELATED:END -->
