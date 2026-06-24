---
title: >-
  [Paper Note] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][anti-personalized diffusion model] This paper proposes the Anti-Personalized Diffusion Model (APDM), which for the first time shifts privacy protection from the data level (image perturbation) to the model level (parameter update). Through a Direct Protective Optimization (DPO) loss and a Learning to Protect (L2P) dual-path optimization strategy, APDM robustly prevents diffusion models from personalizing to specific subjects while preserving t…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "anti-personalized diffusion model"
  - "privacy protection"
  - "DPO loss"
  - "model-level defense"
  - "dual-path optimization"
date: 2026-05-08
content_hash: 2a0d0a249482597d
---

# Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.01307](https://arxiv.org/abs/2511.01307)  
**Code**: [GitHub](https://github.com/KU-VGI/APDM)  
**Area**: Diffusion Models / Privacy Protection
**Keywords**: anti-personalized diffusion model, privacy protection, DPO loss, model-level defense, dual-path optimization

## TL;DR

This paper proposes the Anti-Personalized Diffusion Model (APDM), which for the first time shifts privacy protection from the data level (image perturbation) to the model level (parameter update). Through a Direct Protective Optimization (DPO) loss and a Learning to Protect (L2P) dual-path optimization strategy, APDM robustly prevents diffusion models from personalizing to specific subjects while preserving the model's generation and personalization capabilities for other subjects.

## Background & Motivation

Personalization techniques for diffusion models (e.g., DreamBooth, Custom Diffusion) allow users to fine-tune models with a small number of images to generate images of specific persons or objects, posing serious privacy risks as malicious users may generate unauthorized content.

Existing data-poisoning protection methods suffer from four critical limitations:

**Impractical assumptions**: They require perturbations to be applied to all personal images—including those already shared and newly created—which is infeasible in practice.

**Easy to circumvent**: Even with perturbed images, attackers can undermine protection by mixing in a small number of clean images or applying simple transformations (flipping, blurring).

**High user burden**: Data poisoning requires technical expertise and is difficult for ordinary users to perform.

**Unsuitable for service providers**: Privacy regulations such as GDPR place privacy protection obligations on service providers, yet user-level data poisoning is inherently ill-suited for provider-side deployment.

These issues point to a fundamental paradigm shift: from protecting data to protecting models. However, naively applying existing data-level protection loss functions to model parameters is not viable.

## Method

### Overall Architecture

APDM directly updates the parameters of a pre-trained diffusion model $\theta \to \hat{\theta}$ so that when an attacker subsequently personalizes the model, it fails to generate the target subject. The framework consists of two core components: the Direct Protective Optimization (DPO) loss and the Learning to Protect (L2P) optimization strategy.

### Key Designs

1. **Impossibility Analysis of the Naïve Approach**

   The intuitive protection loss is $\mathcal{L}_{adv} = -\mathcal{L}_{simple}^{per} + \mathcal{L}_{ppl}$ (maximizing personalization loss while preserving prior generation), but the authors prove that this **necessarily fails to converge**:

   **Proposition 1**: A necessary condition for $\mathcal{L}_{adv}$ to converge to a local minimum is that $\nabla_\theta \mathcal{L}_{simple}^{per}$ and $\nabla_\theta \mathcal{L}_{ppl}$ are aligned in direction.

   However, a first-order Taylor expansion analysis yields a contradictory requirement:
    $|\nabla_\theta \mathcal{L}_{simple}^{per}| < |\nabla_\theta \mathcal{L}_{ppl}| \text{ and } |\nabla_\theta \mathcal{L}_{ppl}| < |\nabla_\theta \mathcal{L}_{simple}^{per}|$

   **Theorem 1**: The two gradient norms cannot simultaneously satisfy the condition of each being smaller than the other; therefore, $\mathcal{L}_{adv}$ cannot effectively converge. This impossibility result directly motivates the design of a new loss function.

2. **Direct Protective Optimization (DPO) Loss**

   Inspired by preference optimization, each protected image $x_0^-$ (negative sample) is paired with a positive sample $x_0^+$ generated from a general category. A preference probability is constructed based on the Bradley-Terry model:
    $p(x_0^+ > x_0^-) = \sigma(r(x_0^+) - r(x_0^-))$

   The DPO loss is defined as:
    $\mathcal{L}_{DPO} = -\mathbb{E} \log \sigma(-\beta(r^+ - r^-))$

   where $r^+ = \|\epsilon_\theta(x_t^+, t, c) - \epsilon\|_2^2 - \|\epsilon_\phi(x_t^+, t, c) - \epsilon\|_2^2$, with $r^-$ defined analogously. Here $\phi$ denotes the frozen pre-trained model and $\beta$ controls the degree of deviation.

   The final protection objective is: $\mathcal{L}_{protect} = \mathcal{L}_{DPO} + \mathcal{L}_{ppl}$.

   Motivation: DPO explicitly guides the model to learn "what to encourage" and "what to suppress," avoiding the gradient conflict inherent in the naïve approach.

3. **Learning to Protect (L2P) Dual-Path Optimization**

   Since personalization is an iterative process, static protection is insufficiently robust. L2P alternates between two paths to simulate future personalization trajectories and adaptively strengthen protection:

    - **Personalization path**: Starting from the current protected state $\theta_j$, simulate $N_{per}$ personalization steps: $\theta'_{i+1} = \theta'_i - \gamma_{per}\nabla_{\theta'_i}\mathcal{L}_{per}$
    - **Protection path**: Compute the protection gradient $\nabla_i = \nabla_{\theta'_i}\mathcal{L}_{protect}$ at each intermediate state $\theta'_i$, then accumulate and apply the update: $\theta_{j+1} = \theta_j - \gamma_{protect}\sum_{i=1}^{N_{per}}\nabla_i$

   This procedure is repeated $N_{protect}$ times to obtain the final protected model $\hat{\theta}$. Motivation: L2P enables protection to "anticipate" the trajectory of personalization, achieving trajectory-aware adaptive defense.

### Loss & Training

- Optimizer: AdamW, learning rate $\gamma_{per} = \gamma_{protect} = 5\text{e-6}$
- DPO hyperparameter: $\beta = 1$
- L2P parameters: $N_{per} = 20$, $N_{protect} = 800$
- Inference: PNDM scheduler, 20 steps
- Based on Stable Diffusion 1.5; approximately 9 hours on a single A6000 GPU

## Key Experimental Results

### Main Results

Protection performance (DreamBooth personalization; DINO↓ indicates lower similarity to the original subject is better; BRISQUE↑ indicates worse generation quality is better):

| Method | # Clean Images | DINO↓ (avg) | BRISQUE↑ (avg) |
|--------|---------------|-------------|----------------|
| DreamBooth (no protection) | N | 0.6525 | 16.80 |
| AdvDM | 0 | 0.4999 | 24.06 |
| AdvDM | N-1 | 0.5596 | 23.83 |
| SimAC | 0 | 0.4411 | 27.69 |
| SimAC | N-1 | 0.6181 | 20.67 |
| **APDM** | **N** | **0.1167** | **50.50** |

APDM achieves a DINO score of only 0.1167 (vs. the best baseline of 0.4411), even when all input images are clean.

### Ablation Study

| Configuration | DINO↓ (person) | DINO↓ (dog) | BRISQUE↑ (person) | BRISQUE↑ (dog) |
|---------------|---------------|-------------|-------------------|----------------|
| Without image pairing | 0.2770 | 0.3487 | 27.32 | 29.87 |
| With image pairing | **0.1375** | **0.0959** | **40.25** | **60.74** |
| Without L2P | 0.4454 | 0.3689 | 24.70 | 30.62 |
| With L2P | **0.1375** | **0.0959** | **40.25** | **60.74** |
| $N_{per}=5$ | 0.3371 | 0.1923 | 37.89 | 39.48 |
| $N_{per}=20$ | **0.1375** | **0.0959** | **40.25** | **60.74** |

### Key Findings

- APDM's protection substantially outperforms all data-poisoning methods, reducing DINO to 0.1167 (best baseline: 0.4411), an improvement of approximately 74%.
- Data-poisoning methods degrade significantly when even a single clean image is mixed in (e.g., SimAC rises from 0.4411 to 0.5181), whereas APDM is entirely unaffected by data mixing.
- L2P is critical to performance: without it, DINO degrades from 0.1375 to 0.4454.
- The protected model retains strong personalization capability for other subjects (cat, sneaker, glasses), with an average DINO of 0.6334 vs. 0.5991 for the original DreamBooth.
- Generation quality is preserved: FID 28.85 vs. 25.98 for the original SD; CLIP 0.2853 vs. 0.2878.

## Highlights & Insights

- The paradigm shift from "protecting data" to "protecting models" represents a conceptually significant breakthrough that fundamentally addresses the practical limitations of data poisoning.
- The theoretical proof of the naïve method's non-convergence provides a rigorous mathematical foundation for the new loss function design.
- The dual-path optimization of L2P elegantly internalizes the adversarial game into the training process, drawing on ideas akin to meta-learning.
- The balance between protection and capability preservation is well achieved: specific-subject personalization is blocked while the model's general generation ability and personalization for other subjects remain largely intact.

## Limitations & Future Work

- The protection procedure requires approximately 9 GPU hours, incurring considerable cost.
- A separate protection process must be executed for each subject to be protected.
- Validation is currently limited to SD 1.5 and 2.1; newer models (SDXL, FLUX) remain to be evaluated.
- For subjects whose information already exists in pre-training data, the effectiveness of model-level protection may be limited.

## Related Work & Insights

- Diffusion-DPO provides direct inspiration for the DPO loss design.
- The dual-path optimization of L2P shares conceptual similarities with meta-learning approaches such as MAML.
- The model-level defense paradigm is extensible to privacy protection scenarios in other generative models.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to propose a model-level anti-personalization framework; both the impossibility theorem and the L2P optimization scheme represent significant innovations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive comparisons across multiple subjects and scenarios with complete ablations, though validation on a broader range of models is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative from motivation through theoretical analysis, method, and experiments is exceptionally coherent.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a core practical pain point in privacy protection with direct implications for service provider compliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Anti-Tamper Protection for Unauthorized Individual Image Generation](../../ICCV2025/image_generation/anti-tamper_protection_for_unauthorized_individual_image_generation.md)
- [\[NeurIPS 2025\] Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)
- [\[CVPR 2025\] Enhancing Facial Privacy Protection via Weakening Diffusion Purification](../../CVPR2025/image_generation/enhancing_facial_privacy_protection_via_weakening_diffusion_purification.md)
- [\[CVPR 2025\] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](../../CVPR2025/image_generation/nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)
- [\[NeurIPS 2025\] StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models](stableguard_towards_unified_copyright_protection_and_tamper_localization_in_late.md)

</div>

<!-- RELATED:END -->
