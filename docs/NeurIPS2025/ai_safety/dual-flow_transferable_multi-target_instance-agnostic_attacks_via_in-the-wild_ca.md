---
title: >-
  [Paper Note] Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization
description: >-
  [NeurIPS 2025][AI Safety][adversarial attacks] This paper proposes the Dual-Flow framework, which leverages the forward ODE flow of a pretrained diffusion model and the reverse flow of a fine-tuned LoRA velocity function to perform multi-target, instance-agnostic adversarial attacks. Through a cascading distribution shift training strategy, the method significantly improves transfer attack success rates (e.g., +34.58% from Inc-v3 to Res-152) and demonstrates strong robustness against defended models.
tags:
  - NeurIPS 2025
  - AI Safety
  - adversarial attacks
  - black-box transfer attacks
  - diffusion models
  - flow matching
  - multi-target attacks
date: 2026-05-08
content_hash: bf2e0b455cb9a542
---

# Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2502.02096](https://arxiv.org/abs/2502.02096)
**Code**: [github.com/Chyxx/Dual-Flow](https://github.com/Chyxx/Dual-Flow)
**Area**: AI Security / Adversarial Attacks
**Keywords**: adversarial attacks, black-box transfer attacks, diffusion models, flow matching, multi-target attacks

## TL;DR
This paper proposes the Dual-Flow framework, which leverages the forward ODE flow of a pretrained diffusion model and the reverse flow of a fine-tuned LoRA velocity function to perform multi-target, instance-agnostic adversarial attacks. Through a cascading distribution shift training strategy, the method significantly improves transfer attack success rates (e.g., +34.58% from Inc-v3 to Res-152) and demonstrates strong robustness against defended models.

## Background & Motivation

**State of the Field**: Adversarial attacks are broadly categorized as instance-specific or instance-agnostic. Instance-agnostic methods learn perturbations at the data distribution level, yielding better black-box transferability. Generative model-based methods are further divided into single-target (requiring one model per target class) and multi-target (a single conditional model attacks all classes).

**Limitations of Prior Work**: Multi-target generative attacks suffer from low transfer success rates due to limited model capacity; existing diffusion model-based attacks are instance-specific (requiring target model gradients at inference time); the theoretical justification for choosing ODE vs. SDE sampling is lacking.

**Root Cause**: During reverse flow training, the true distribution at intermediate timesteps is inaccessible (the forward ODE trajectory is in-the-wild), making standard diffusion training algorithms inapplicable.

**Paper Goals**: (a) How can diffusion models be leveraged for instance-agnostic multi-target attacks? (b) How can the reverse flow be trained without access to intermediate distributions?

**Starting Point**: The attack is decomposed into two flows — a forward flow (a pretrained diffusion model generates a perturbed distribution) and a reverse flow (fine-tuned LoRA maps it back to the constrained space).

**Core Idea**: The forward ODE of a pretrained diffusion model produces intermediate representations, which are then mapped back into $\ell_\infty$-constrained adversarial examples via a LoRA-fine-tuned velocity function. Cascading optimization progressively improves attack effectiveness.

## Method

### Overall Architecture
An input image $x$ is mapped to a perturbed distribution $X_\tau$ via the forward flow, then mapped to the $\ell_\infty$-constrained space via the reverse flow. No target model gradients are required at inference time.

### Key Designs

1. **Forward Flow**:

    - **Function**: Maps clean images to an intermediate perturbed distribution.
    - **Mechanism**: Uses the pretrained diffusion model's velocity function $v_\phi$, integrating via ODE from $t=0$ to $t=\tau$.
    - **Design Motivation**: The pretrained diffusion model inherently generates structured perturbed distributions without additional training.

2. **Reverse Flow**:

    - **Function**: Maps the perturbed distribution to valid adversarial examples.
    - **Mechanism**: Fine-tunes LoRA to obtain a new velocity function $v_\theta$, integrating via ODE from $t=\tau$ to $t=0$.
    - **Optimization Objective**: Minimize cross-entropy $j = -\mathrm{CE}(f(x), c)$, where $f$ is the surrogate model and $c$ is the target class.

3. **Cascading Distribution Shift Training**:

    - **Function**: Resolves the inaccessibility of intermediate timestep distributions during training.
    - **Mechanism** (Algorithm 1): Backtracking from $t=N$ to $t=1$, each step first estimates $\hat{x}_0$, clips it to the constraint range, and updates $\theta$ via cross-entropy.
    - **Theoretical Guarantee** (Theorem 2): Cascading improvement property — updating $\theta$ at timestep $t$ does not worsen the cross-entropy at timestep $t - \delta$ (for sufficiently small $\delta$).
    - **Design Motivation**: Ensures consistency between training and sampling procedures.

4. **Morse Flow Construction** (Proposition 1):

    - **Core Theory**: Under mild assumptions on $X_\epsilon$ and $j$, there exists a unique smooth flow $\Phi$ whose velocity function $v$ equals $\alpha(x) \cdot \nabla_x j(x)$ almost everywhere.
    - **Significance**: Guarantees that gradient-directed flow improves the attack objective, and that the flow map is a diffeomorphism.

5. **Dynamic Gradient Clipping and ODE vs. SDE Selection**:

    - Estimated $\hat{x}_0$ is clipped with stop-gradient during training.
    - Cascading ODE outperforms cascading SDE (stochastic terms disrupt the cascade) and stochastic SDE (distribution mismatch), validating the necessity of deterministic trajectories.

### Loss & Training
- Cross-entropy loss $\mathrm{CE}(f(\hat{x}_0), c)$
- $\ell_\infty \leq 16/255$ perturbation constraint
- LoRA fine-tuning to minimize additional parameter count

## Key Experimental Results

### Main Results: Multi-Target Attack Success Rate (%) — Normally Trained Models

| Source Model | Method | Inc-v3* | Inc-v4 | Res-152 | DN-121 | VGG-16 | Black-box Avg. |
|---|---|---|---|---|---|---|---|
| Inc-v3 | C-GSP | 93.40 | 66.90 | 41.60 | 46.40 | 45.00 | 51.08 |
| Inc-v3 | CGNC | 96.03 | 59.43 | 42.48 | 62.98 | 52.54 | 52.80 |
| Inc-v3 | Dual-Flow | 90.08 | 77.19 | 77.06 | 82.64 | 67.09 | 73.96 |

### Attack Success Rate (%) on Defended Models — Source Model: Inc-v3

| Method | Inc-v3_adv | IR-v2_ens | Res50_SIN | Res50_Aug | Avg. |
|---|---|---|---|---|---|
| C-GSP | 20.41 | 18.04 | 6.96 | 21.95 | 24.28 |
| CGNC | 24.36 | 22.54 | 8.85 | 22.85 | 28.60 |
| Dual-Flow | 51.54 | 55.62 | 45.86 | 67.56 | 62.28 |

### Key Findings
- Black-box transfer success rates are substantially improved: Inc-v3 → Res-152 increases from 42.48% (CGNC) to 77.06%, an absolute gain of 34.58%.
- The advantage over defended models is even greater: average success rate of 62.28% vs. 28.60% for CGNC (+33.68%).
- Compared to single-target attacks, the multi-target variant underperforms by only ~3%, while eliminating the need to train separate models for each target class.
- Cascading ODE substantially outperforms cascading SDE and stochastic SDE, confirming the necessity of deterministic trajectories.

## Highlights & Insights
- This is the first work to apply flow-based ODE velocity training to adversarial attacks (as opposed to conventional score function training), opening a new direction for diffusion models in the security domain.
- The cascading distribution shift training is elegantly designed — by first integrating forward then optimizing backward step by step, it ensures training-inference consistency with theoretical guarantees.
- LoRA fine-tuning enables adversarial adaptation with minimal additional parameters, making deployment practical.

## Limitations & Future Work
- White-box access to the surrogate model is required during training; transferability to target models operates in the black-box setting.
- Experiments are conducted solely on ImageNet classification; extension to downstream tasks such as detection and segmentation remains unexplored.
- The perturbation constraint is fixed at $\ell_\infty \leq 16/255$; alternative constraints or smaller perturbation budgets are not explored.
- The selection of the forward flow timestep $\tau$ may require careful tuning.

## Related Work & Insights
- **vs. CGNC (2024)**: Both are multi-target conditional generative attacks, but CGNC employs a UNet-GAN whereas Dual-Flow uses diffusion ODE + LoRA; Dual-Flow achieves on average 20+% higher black-box transfer rates.
- **vs. C-GSP**: Also a generative method, but achieves lower transfer rates than both CGNC and Dual-Flow.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First application of flow-based velocity training to multi-target adversarial attacks; the cascading training approach is methodologically innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers normal and defended models, multi- and single-target settings, and ODE vs. SDE comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Theory and experiments are well integrated, with clear intuitive explanations.
- **Value**: ⭐⭐⭐⭐ Significantly advances the state of the art in multi-target transfer attacks, with important implications for model robustness evaluation.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach](../../AAAI2026/ai_safety/rethinking_target_label_conditioning_in_adversarial_attacks_a_2d_tensor-guided_g.md)
- [\[NeurIPS 2025\] Private Zeroth-Order Optimization with Public Data](private_zeroth-order_optimization_with_public_data.md)
- [\[NeurIPS 2025\] Fairness-Regularized Online Optimization with Switching Costs](fairness-regularized_online_optimization_with_switching_costs.md)
- [\[NeurIPS 2025\] Multi-Class Support Vector Machine with Differential Privacy](multi-class_support_vector_machine_with_differential_privacy.md)
- [\[AAAI 2026\] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation](../../AAAI2026/ai_safety/transferable_backdoor_attacks_for_code_models_via_sharpness-aware_adversarial_pe.md)

<!-- RELATED:END -->
