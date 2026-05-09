---
title: >-
  [Paper Note] Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning
description: >-
  [NeurIPS 2025][Image Generation][Diffusion models] This paper proposes ResAlign, a framework that leverages Moreau envelope approximation and meta-learning to make safety-driven unlearning in diffusion models resilient against harmful capability recovery induced by downstream fine-tuning, even when fine-tuning is performed exclusively on benign data.
tags:
  - NeurIPS 2025
  - Image Generation
  - Diffusion models
  - safety unlearning
  - fine-tuning resilience
  - meta-learning
  - Moreau envelope
date: 2026-05-08
content_hash: e27207174ceb5e34
---

# Towards Resilient Safety-Driven Unlearning for Diffusion Models Against Downstream Fine-tuning

**Conference**: NeurIPS 2025
**arXiv**: [2507.16302](https://arxiv.org/abs/2507.16302)
**Code**: [https://github.com/](https://github.com/) (Code and pretrained models are publicly available)
**Area**: Image Generation / AI Safety
**Keywords**: Diffusion models, safety unlearning, fine-tuning resilience, meta-learning, Moreau envelope

## TL;DR

This paper proposes ResAlign, a framework that leverages Moreau envelope approximation and meta-learning to make safety-driven unlearning in diffusion models resilient against harmful capability recovery induced by downstream fine-tuning, even when fine-tuning is performed exclusively on benign data.

## Background & Motivation

- **State of the Field**: Text-to-image (T2I) diffusion models (e.g., Stable Diffusion), trained on large-scale web-crawled data, inevitably acquire the ability to generate harmful content (e.g., sexually explicit images). Safety-driven unlearning methods (e.g., ESD, SafeGen, AdvUnlearn) suppress unsafe generation by modifying model parameters, achieving preliminary success.

- **Limitations of Prior Work**: Safety-unlearned models **recover harmful capabilities after downstream fine-tuning**. More critically, the paper's experiments reveal that **even fine-tuning on entirely benign data** causes existing SOTA unlearning methods to regress to unsafe levels approaching those of the original, unmodified model. This implies that even ordinary users with no malicious intent — who merely wish to personalize the model — may inadvertently restore its harmful behaviors.

- **Root Cause**: Existing unlearning methods optimize safety only at the current parameter state (Eq. 2), while the parameter-space neighborhood surrounding the unlearned model may remain "toxic." Parameter drift induced by fine-tuning, even along benign directions, can push the model into these toxic regions.

- **Paper Goals**: Unlearning should not only suppress harmful behaviors in the current state but should also **explicitly minimize the degree of harmful capability recovery after fine-tuning**. The key challenge is computational: since fine-tuning is a multi-step optimization process, how can one efficiently compute the gradient of "post-fine-tuning harmfulness with respect to current parameters"?

## Method

### Overall Architecture

ResAlign augments the standard unlearning objective with a resilience term: $\theta^* = \arg\min_\theta \mathcal{L}_{\text{harmful}}(\theta) + \alpha\mathcal{R}(\theta) + \beta[\mathcal{L}_{\text{harmful}}(\theta_{\text{FT}}^*) - \mathcal{L}_{\text{harmful}}(\theta)]$. The third term explicitly penalizes the increase in harmfulness induced by fine-tuning. Efficient optimization is achieved through Moreau envelope approximation and meta-learning.

### Key Designs

1. **Efficient Hypergradient Approximation via Moreau Envelope**:

    - Directly computing the hypergradient $\nabla_\theta \mathcal{L}_{\text{harmful}}(\theta_{\text{FT}}^*)$ requires storing and backpropagating through the entire fine-tuning trajectory — prohibitive in both computation and memory.
    - Fine-tuning is approximated as minimization of the Moreau envelope: $\theta_{\text{FT}}^* \in \arg\min_{\theta'} \mathcal{L}_{\text{FT}}(\theta') + \frac{1}{2\gamma}\|\theta'-\theta\|^2$
    - Using first-order optimality conditions and the implicit function theorem, the hypergradient is reduced to solving a linear system $Ax=b$.
    - Richardson iteration efficiently solves this system: $x^{(k+1)} = \gamma b - \gamma \nabla^2_{\theta_{\text{FT}}^*} \mathcal{L}_{\text{FT}} \cdot x^{(k)}$, converging in as few as 5 steps.
    - Key advantage: only the final fine-tuned parameters $\theta_{\text{FT}}^*$ and local Hessian-vector products (HVPs) are required — no intermediate trajectory storage is needed.

2. **Cross-Configuration Generalization via Meta-Learning**:

    - Downstream fine-tuning configurations (learning rate, number of steps, loss function, optimizer, full fine-tuning vs. LoRA, etc.) are modeled as meta-variables.
    - Each inner loop: a configuration $\mathcal{C} \sim \pi(\mathcal{C})$ and data $\mathcal{D}_{\text{FT}}$ are randomly sampled, simulated fine-tuning is performed, and the hypergradient is computed.
    - After $J$ repetitions, hypergradients are aggregated to update the base model parameters.
    - This ensures that safety resilience generalizes across diverse possible downstream adaptation scenarios, rather than being limited to a single fine-tuning configuration.

3. **Theoretical Insight (Proposition 1)**:

    - The resilience term is equivalent to implicitly penalizing the Hessian trace of the harmfulness loss, $\text{Tr}(\nabla^2_\theta \mathcal{L}_{\text{harmful}})$.
    - The Hessian trace measures the curvature of the loss landscape — large values correspond to sharp minima (sensitive to parameter perturbations), while small values correspond to flat regions.
    - ResAlign encourages convergence to **flat safety regions**, reducing sensitivity to downstream parameter updates.

### Loss & Training

- $\mathcal{L}_{\text{harmful}}$: Negative denoising loss on harmful prompt–image pairs.
- $\mathcal{R}$: Noise prediction distillation loss relative to the original model on retained prompts.
- Training: Approximately 1 hour on a single A100 GPU; peak memory usage approximately 56 GB.
- Meta-learning configuration distribution: learning rates $\{10^{-4}, 10^{-5}, 10^{-6}\}$, steps $\{5, 10, 20, 30\}$, algorithms $\{\text{full fine-tuning}, \text{LoRA}\}$, optimizers $\{\text{SGD}, \text{Adam}\}$.

## Key Experimental Results

### Main Results

**Safety evaluation under different fine-tuning settings (IP: Inappropriateness Rate ↓, US: Unsafe Score ↓)**

| Model | Pre-FT IP↓ | DreamBench++ FT IP↓ | DiffusionDB FT IP↓ | FID↓ |
|------|----------|-------------------|-------------------|------|
| SD v1.4 (original) | 0.3598 | - | - | 16.90 |
| ESD | 0.0677 | 0.1661 | 0.2209 | 16.88 |
| SafeGen | 0.1199 | 0.3154 | 0.3344 | 17.11 |
| AdvUnlearn | 0.0183 | 0.1038 | 0.2975 | 18.31 |
| LCFDSD-NG | 0.0788 | 0.2238 | 0.2474 | 47.21 |
| **ResAlign** | **0.0014** | **0.0186** | **0.0687** | **18.18** |

### Ablation Study

| Component | IP↓ | FID↓ | Notes |
|------|-----|------|------|
| w/o hypergradient + w/o meta-learning | 0.2266 | 18.24 | Standard unlearning baseline |
| + hypergradient | 0.1826 | 18.07 | Moreau approximation is effective |
| + hypergradient + meta-learning (data) | 0.0322 | 18.35 | Data diversity yields large gains |
| + hypergradient + meta-learning (data + config) | **0.0186** | 18.18 | Full ResAlign |

**Cross-model generalization (IP↓ after LoRA fine-tuning)**

| Model | Pre-FT | DreamBench++ | DiffusionDB |
|------|--------|-------------|-------------|
| SD v2.0 | 0.004 | 0.031 | 0.078 |
| SDXL | 0.033 | 0.044 | 0.059 |
| AnythingXL | 0.015 | 0.062 | 0.087 |
| PonyDiffusion | 0.023 | 0.045 | 0.067 |

### Key Findings

- **All existing methods are vulnerable**: SafeGen's IP recovers from 0.12 to 0.33 after DiffusionDB fine-tuning (approaching the original SD's 0.36); AdvUnlearn recovers from 0.02 to 0.30.
- ResAlign maintains IP = 0.0186 after DreamBench++ fine-tuning — 5.5× lower than AdvUnlearn (0.1038).
- ResAlign consistently maintains low inappropriateness rates with minimal fluctuation across 500 fine-tuning steps.
- ResAlign demonstrates notable robustness to data poisoning — even with 20% harmful data mixed in, IP remains significantly below baselines.
- Component contributions: the hypergradient contributes moderately (0.2266→0.1826); meta-learning over data contributes most substantially (0.1826→0.0322).
- FID increases only marginally from 16.90 to 18.18, indicating that generation quality is largely preserved.

## Highlights & Insights

- **The problem identification itself is a contribution**: revealing that benign fine-tuning can restore harmful capabilities constitutes a critical warning for the safety unlearning community.
- **The Moreau envelope + implicit differentiation optimization framework** is particularly elegant — compressing multi-step fine-tuning into a single-point solve.
- **The flat minima interpretation of safety** provides an intuitive yet mathematically rigorous understanding.
- **The meta-learning configuration distribution design** reflects deep consideration of real-world deployment scenarios.

## Limitations & Future Work

- Achieving perfect resilience against adversarial fine-tuning (using harmful data) is inherently difficult — an adversary can frame harmful capability recovery as a new learning task.
- Experiments are conducted primarily on SD v1.4; although cross-model generalization is validated, more recent models such as Flux are not evaluated.
- The paper focuses exclusively on "sexual" unsafe content; generalization to other harmful categories (e.g., violence, hate speech) is not validated.
- Training requires an A100 GPU with a peak memory footprint of 56 GB, which may not be accessible to most researchers.
- The meta-learning configuration distribution requires manual design and may omit certain extreme configurations.

## Related Work & Insights

- The methodology is conceptually related to MetaCloak (privacy protection via Moreau envelope), though the objectives differ.
- The efficient hypergradient computation framework combining Moreau envelope and Richardson iteration is transferable to other settings requiring anticipation of future fine-tuning effects.
- The connection between flat minima and safety can inspire deeper theoretical investigation.
- This work provides a technical foundation for commercial deployment of "safety unlearning as a service."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Both the problem formulation (benign fine-tuning restoring harmful capabilities) and the solution (Moreau envelope + meta-learning) are highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, fine-tuning methods, configurations, models, data poisoning settings, and component analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, theoretical derivations are rigorous, and experimental presentation is systematic.
- Value: ⭐⭐⭐⭐⭐ Of significant practical value to the AI safety community, addressing a critical vulnerability in unlearning methods.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](../../ICCV2025/image_generation/meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [\[NeurIPS 2025\] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models](prompt-based_safety_guidance_is_ineffective_for_unlearned_text-to-image_diffusio.md)
- [\[ICCV 2025\] ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning](../../ICCV2025/image_generation/shortft_diffusion_model_alignment_via_shortcut-based_fine-tuning.md)

<!-- RELATED:END -->
