---
title: >-
  [Paper Note] TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Concept Erasure] This paper proposes TINA (Text-free INversion Attack), which bypasses all text-based concept erasure defenses by optimizing DDIM inversion under null-text conditioning to re…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Concept Erasure"
  - "Machine Unlearning"
  - "DDIM Inversion"
  - "Text-to-Image Diffusion"
  - "Adversarial Attack"
date: 2026-05-08
content_hash: c47630fa6843ce9d
---

# TINA: Text-Free Inversion Attack for Unlearned Text-to-Image Diffusion Models

## Basic Information

**Conference**: CVPR 2026
**arXiv**: [2603.17828](https://arxiv.org/abs/2603.17828)
**Code**: [GitHub](https://github.com/qianlong0502/TINA)
**Area**: Image Generation / AI Safety / Concept Erasure Attack
**Keywords**: Concept Erasure, Machine Unlearning, DDIM Inversion, Text-to-Image Diffusion, Adversarial Attack

## TL;DR

This paper proposes TINA (Text-free INversion Attack), which bypasses all text-based concept erasure defenses by optimizing DDIM inversion under null-text conditioning to recover a precise initial noise vector. The work demonstrates that existing erasure methods merely sever the text-to-image mapping without truly deleting the visual knowledge encoded in model parameters.

## Background & Motivation

A fundamental blind spot exists in the concept erasure literature for text-to-image diffusion models (e.g., Stable Diffusion): both erasure methods and adversarial attacks have evolved exclusively around the **text conditioning pathway**. Erasure methods (ESD, UCE, AdvUnlearn, etc.) achieve "forgetting" by severing the mapping from text prompts to target concepts, while attack methods (P4D, UDA, CCE, etc.) attempt to recover erased concepts by finding alternative texts or embeddings.

This "text-centric co-evolution" rests on a critical implicit assumption: **severing the text-to-image link equals deleting visual knowledge**. The authors argue this assumption is fundamentally flawed—even when the text pathway is blocked, the **visual knowledge** corresponding to erased concepts may still reside in the model's parameter space. To validate this hypothesis, an attack that completely bypasses text conditioning is required.

**Core hypothesis**: Even after text-to-image mappings are removed, the **deterministic generation trajectories** of erased concepts remain embedded in the model and can be rediscovered in a fully text-free manner.

## Method

### Overall Architecture

TINA operates in two stages:

1. **Text-Free Inversion**: Given an unlearned model $\epsilon_\theta$ and a reference image $x$ depicting the erased concept, an optimized initial noise $z_T^*$ is recovered under null-text conditioning $c_\text{null}$ such that it deterministically reconstructs the target image.

2. **Deterministic Concept Regeneration**: The optimized $z_T^*$ is fed into the same unlearned model $\epsilon_\theta$, and standard DDIM sampling under $c_\text{null}$ is executed to deterministically regenerate the erased concept.

Key insight: The entire process **involves no text conditioning whatsoever**, thereby completely circumventing all text-based erasure defenses.

### Preliminaries: DDIM Sampling and Inversion

**DDIM sampling** is deterministic—given model $\theta$, condition $c$, and initial noise $z_T$, the generated $z_0$ is uniquely determined. The sampling update is:

$$z_{t-1} = \sqrt{\alpha_{t-1}} \hat{z}_0(z_t) + \sqrt{1-\alpha_{t-1}} \cdot \epsilon_\theta(z_t, t, c)$$

where $\hat{z}_0(z_t) = \frac{z_t - \sqrt{1-\alpha_t} \epsilon_\theta(z_t, t, c)}{\sqrt{\alpha_t}}$ is the predicted clean latent.

**Standard DDIM inversion** attempts to recover $z_T$ from $z_0$. The exact inversion relation is:

$$z_t = C_1(t) z_{t-1} + C_2(t) \cdot \epsilon_\theta(z_t, t, c)$$

where $C_1(t) = \frac{\sqrt{\alpha_t}}{\sqrt{\alpha_{t-1}}}$ and $C_2(t) = \sqrt{1-\alpha_t} - \sqrt{\frac{\alpha_t(1-\alpha_{t-1})}{\alpha_{t-1}}}$.

Since $z_t$ appears on both sides, a circular dependency arises. The standard approximation substitutes $\epsilon_\theta(z_{t-1}, t-1, c)$ for $\epsilon_\theta(z_t, t, c)$, introducing **cumulative approximation errors**.

### Why Naive Inversion Fails

The authors analyze the failure modes of two straightforward baselines:

- **Text-guided inversion** (using the erased concept's prompt): The unlearned model actively resists this text condition, causing complete failure—which itself confirms that text-based erasure defenses are effective along the text pathway.

- **Null-text inversion** (using empty text): Since the standard DDIM inversion approximation relies heavily on meaningful text guidance to steer noise predictions, small per-step errors accumulate rapidly without text guidance, causing the estimated $\hat{z}_T$ to diverge from the true $z_T^*$ and failing to faithfully reconstruct the target concept.

### Core Innovation: Optimization-Based Text-Free Inversion

TINA reformulates inversion as a **fixed-point optimization problem**. Rather than using the approximation formula, it derives the exact inversion relation directly from the DDIM sampling equation:

$$z_t = f_\theta^*(z_t, z_{t-1}, t, c) = C_1(t) z_{t-1} + C_2(t) \cdot \epsilon_\theta(z_t, t, c)$$

This establishes a **self-consistency constraint**: every $z_t$ on the true generation trajectory must be a fixed point of the map $f_\theta^*$.

At each timestep $t$, given the previously computed $z_{t-1}$, TINA optimizes $z_t$ by minimizing:

$$\mathcal{L}_t(z_t) = \| f_\theta^*(z_t, z_{t-1}, t, c_\text{null}) - z_t \|_2^2$$

Optimization procedure:
1. Compute an initial estimate $\tilde{z}_t$ via standard DDIM inversion under $c_\text{null}$
2. Starting from $\tilde{z}_t$, perform $K$ gradient descent steps to refine $z_t$
3. Repeat sequentially for $t = 1, \dots, T$ to obtain the precise $z_T^*$

### Algorithm Pseudocode

| Step | Operation |
|------|-----------|
| Input | Unlearned model $\epsilon_\theta$, target image latent $z_0$, timesteps $T$, null text $c_\text{null}$, optimization rounds $K$, learning rate $\eta$ |
| for $t = 1$ to $T$ | Compute initial estimate $\tilde{z}_t$ from $z_{t-1}$ via standard inversion under $c_\text{null}$; set $z_t \leftarrow \tilde{z}_t$ |
| for $k = 1$ to $K$ | Compute $\mathcal{L}_t$; gradient descent $z_t \leftarrow z_t - \eta \nabla_{z_t} \mathcal{L}_t$ |
| Output | Optimized initial noise $z_T^* \leftarrow z_T$ |

### Implementation Details

- Base model: Stable Diffusion v1.4
- Scheduler: Linear Multistep Scheduler, $T = 50$ steps, CFG = 7.5
- Inner-loop optimization: $K = 25$ iterations, AdamW optimizer, $\eta = 0.001$
- Hardware: Single NVIDIA A100 GPU

## Key Experimental Results

### Main Results: Attack Success Rate on Nudity Concept Erasure

| Attack | ESD | FMN | UCE | MACE | RECE | AdvUnlearn | SalUn | STEREO |
|--------|-----|-----|-----|------|------|------------|-------|--------|
| MMA | 13.1 | 67.0 | 32.6 | 6.0 | 22.8 | 1.7 | 1.7 | 5.5 |
| P4D | 69.0 | 97.9 | 76.1 | 75.4 | 66.2 | 18.3 | 15.5 | 24.7 |
| UDA | 76.1 | 97.9 | 78.9 | 81.7 | 63.4 | 23.2 | 13.4 | 25.4 |
| RAB | 50.5 | 97.9 | 29.5 | 6.3 | 10.5 | 2.1 | 0.0 | 8.4 |
| CCE | 74.7 | 55.0 | 49.3 | 50.0 | 66.9 | 76.8 | 2.8 | 16.9 |
| **TINA** | **82.4** | **97.9** | **82.4** | **93.0** | **80.3** | **78.9** | **71.1** | **81.0** |

TINA achieves the highest ASR across all 8 defenses. Notably, against robust defenses such as AdvUnlearn (78.9%), SalUn (71.1%), and STEREO (81.0%), text-based attacks nearly fail (UDA achieves only 23.2% / 13.4% / 25.4%), while TINA maintains high attack rates throughout.

### Attack Success Rate on Style Erasure (Van Gogh)

| Attack | ESD | FMN | AC | MACE | SPM | RECE | AdvUnlearn | STEREO |
|--------|-----|-----|----|------|-----|------|------------|--------|
| P4D | 30.0 | 54.0 | 68.0 | 42.0 | 78.0 | 62.0 | 0.0 | 0.0 |
| UDA | 32.0 | 56.0 | 77.0 | 56.0 | 88.0 | 64.0 | 2.0 | 0.0 |
| CCE | 8.0 | 18.0 | 14.0 | 26.0 | 36.0 | 40.0 | 44.0 | 4.0 |
| **TINA** | **70.0** | **72.0** | **74.0** | **72.0** | **80.0** | **74.0** | **70.0** | **44.0** |

### Attack Success Rate on Object Erasure (Tench Category)

| Attack | ESD | EraseDiff | SalUn | Scissorhands | STEREO |
|--------|-----|-----------|-------|--------------|--------|
| P4D | 32.0 | 8.0 | 18.0 | 6.0 | 0.0 |
| UDA | 46.0 | 2.0 | 12.0 | 6.0 | 2.0 |
| CCE | 40.0 | 34.0 | 58.0 | 0.0 | 2.0 |
| **TINA** | **70.0** | **68.0** | **72.0** | **78.0** | **72.0** |

### Ablation Study

| Method | ASR (%) | Notes |
|--------|---------|-------|
| Standard Inv. (text-guided) | 30 | Erasure method actively opposes text condition |
| TINA-Less (insufficient optimization steps) | 46 | Error correction is inadequate |
| **TINA** (full optimization $K=25$) | **70** | Sufficient optimization achieves self-consistency |

The 24% ASR gain from TINA-Less to TINA demonstrates that sufficient optimization iterations are critical for accurately tracking the generation trajectory.

### Comparison with DDIM Reconstruction Methods (EasyInv)

| Method | ESD | EraseDiff | SalUn | Scissorhands | STEREO |
|--------|-----|-----------|-------|--------------|--------|
| EasyInv | 24.0 | 26.0 | 30.0 | 34.0 | 24.0 |
| **TINA** | **70.0** | **68.0** | **72.0** | **78.0** | **72.0** |

General-purpose DDIM reconstruction methods are substantially inferior to TINA's dedicated optimization scheme for concept recovery.

### Key Findings

1. **Text erasure ≠ visual knowledge deletion**: TINA successfully bypasses all 12 erasure defenses across all three task categories (nudity / style / object), demonstrating that visual knowledge of erased concepts persists in model parameters.
2. **Robust defenses provide no resistance to TINA**: Adversarially hardened defenses such as AdvUnlearn and STEREO effectively block text-based attacks but pose little obstacle to TINA.
3. **Latent embedding analysis** (t-SNE): The optimized noise $z_T^*$ is indistinguishable from random noise in noise space, yet its activations at the UNet `mid_block` cluster clearly by concept, confirming that concept-specific visual knowledge within the model is precisely activated.
4. **Architectural generality**: TINA remains effective on the DiT architecture (PixArt-XL-2), indicating that this vulnerability is not limited to UNet-based models.

## Highlights & Insights

- **Paradigm shift**: The first work to challenge the effectiveness of concept erasure from a visual perspective, exposing a fundamental flaw in the text-centric paradigm.
- **Methodological elegance**: The approximation error inherent in DDIM inversion is reformulated as a fixed-point optimization problem, requiring neither auxiliary models nor textual information.
- **Comprehensive evaluation**: Covers 12 erasure methods × 5 baseline attacks × 3 concept task categories, providing rigorous empirical support.
- **Security warning**: Provides a critical alert to the AI safety community, motivating a shift toward erasure paradigms that operate on internal visual representations.

## Limitations & Future Work

- Requires a **reference image** of the target concept as the inversion starting point; the attack is not fully zero-shot.
- ASR against STEREO on the style erasure task is only 44%, suggesting adversarial training can partially perturb internal visual representations.
- The attack incurs substantial computational overhead ($K=25$ optimization iterations per timestep, totaling $T \times K = 1250$ forward passes).
- Comprehensive evaluation is conducted only on SD v1.4; larger models such as SDXL are not covered.
- The paper primarily diagnoses the problem without proposing a corresponding defensive solution.

## Rating

⭐⭐⭐⭐ — An important paradigm-challenging contribution to the concept erasure field. The elegant text-free inversion attack reveals a fundamental deficiency in current erasure methods, and the experimental design is rigorous and comprehensive. The requirement for reference images and the absence of a defensive proposal are notable limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[ICCV 2025\] PLA: Prompt Learning Attack against Text-to-Image Generative Models](../../ICCV2025/image_generation/pla_prompt_learning_attack_against_text-to-image_generative_models.md)
- [\[CVPR 2026\] RAISE: Requirement-Adaptive Evolutionary Refinement for Training-Free Text-to-Image Alignment](raise_requirement-adaptive_evolutionary_refinement_for_training-free_text-to-ima.md)

</div>

<!-- RELATED:END -->
