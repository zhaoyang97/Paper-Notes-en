---
title: >-
  [Paper Note] Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning
description: >-
  [ICLR 2026][Image Generation][machine unlearning attack] Recall proposes the first multi-modal guided attack framework that optimizes adversarial image prompts in the latent space using a single reference image. Combined with the original text prompt, it exploits the image-conditioning channel of diffusion models and achieves an average ASR of 65%–97% across 10 SOTA unlearning methods, substantially outperforming text-only attack methods and exposing the vulnerability of current unlearning mechanisms to image-modality attacks.
tags:
  - ICLR 2026
  - Image Generation
  - machine unlearning attack
  - multi-modal adversarial
  - image prompt
  - diffusion model security
  - robustness auditing
date: 2026-05-08
content_hash: 288cd491bc034e08
---

# Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning

**Conference**: ICLR 2026
**arXiv**: [2507.07139](https://arxiv.org/abs/2507.07139)
**Code**: [GitHub](https://github.com/ryliu68/RECALL)
**Area**: Diffusion Models / Unlearning / Security
**Keywords**: machine unlearning attack, multi-modal adversarial, image prompt, diffusion model security, robustness auditing

## TL;DR

Recall proposes the first multi-modal guided attack framework that optimizes adversarial image prompts in the latent space using a single reference image. Combined with the original text prompt, it exploits the image-conditioning channel of diffusion models and achieves an average ASR of 65%–97% across 10 SOTA unlearning methods, substantially outperforming text-only attack methods and exposing the vulnerability of current unlearning mechanisms to image-modality attacks.

## Background & Motivation

**State of the Field**: Machine unlearning for diffusion models has become a key technique for mitigating the generation of harmful or copyright-infringing content, with more than ten methods proposed, including ESD, UCE, and AdvUnlearn.

**Limitations of Prior Work**: Existing attack methods (P4D, UnlearnDiffAtk, CCE) perturb only the text prompt and suffer from four major drawbacks: ① text perturbations disrupt semantic alignment; ② they rely on external classifiers or the original model, increasing overhead; ③ their effectiveness drops sharply against adversarially hardened unlearning methods (e.g., AdvUnlearn); ④ they entirely overlook the native multi-modal conditioning capability of diffusion models.

**Root Cause**: Unlearning methods are primarily designed to defend against the text modality, yet Stable Diffusion natively supports multi-modal conditioning via image+text inputs (e.g., img2img), a channel that has been largely unexplored as an attack vector.

**Paper Goals**: To bypass unlearning defenses by exploiting the image-modality attack channel while keeping the text semantics unchanged.

**Starting Point**: Rather than modifying the text prompt, the method optimizes a single adversarial image prompt so that the unlearned model regenerates erased content. The optimization is performed directly in the unlearned model's internal latent space, requiring no external models.

**Core Idea**: Adversarial image prompts are generated via reference-image-guided latent-space optimization, "awakening" residual concept memories in the unlearned model through the multi-modal conditioning channel.

## Method

### Overall Architecture

A three-stage pipeline: (1) **Latent Encoding** — the reference image and a noise-initialized image are encoded into the latent space; (2) **Iterative Latent Optimization** — the adversarial latent variable is optimized by aligning its denoising predictions with those of the reference latent; (3) **Multi-Modal Attack** — the adversarial image is decoded and fed into the unlearned model together with the text prompt.

### Key Designs

1. **Noise Initialization Strategy**:

   - Function: Constructs the initial image prompt $P_{img}^{init} = \lambda \cdot P_{ref} + (1-\lambda) \cdot \delta$, where $\delta \sim \mathcal{N}(0, I)$ and $\lambda = 0.25$.
   - Mechanism: A large noise fraction (75%) enlarges the sampling space and increases generation diversity, while a small reference signal (25%) provides a conceptual seed.
   - Design Motivation: Prevents the generated image from being a mere transformation of the reference, ensuring diverse outputs that follow the text prompt.

2. **Iterative Latent Optimization**:

   - Function: Performs gradient-based optimization of $z_{adv}$ over 50 DDIM timesteps, with 20 gradient iterations per step.
   - Mechanism: Minimizes the MSE between the U-Net noise predictions of the adversarial and reference latent variables under the same text conditioning: $\mathcal{L}_{adv} = \|\hat{\epsilon}_{ref,t} - \hat{\epsilon}_{adv,t}\|_2^2$.
   - Momentum gradient update: $v_i = \beta \cdot v_{i-1} + \frac{\nabla_{z_{adv}} \mathcal{L}_{adv}}{\|\nabla_{z_{adv}} \mathcal{L}_{adv}\|_1 + \omega}$, $z_{adv} \leftarrow z_{adv} + \eta \cdot \text{sign}(v_i)$.
   - Periodic reference injection: Every 5 steps, a small fraction of $z_{ref}$ ($\gamma = 0.05$) is blended into $z_{adv}$ to maintain semantic consistency.

3. **Early Stopping**:

   - Optimization halts as soon as the target content is detected to reappear, reducing unnecessary computation.
   - If all 50 steps complete without success, the attack is deemed to have failed.

### Loss & Training

$$\mathcal{L}_{adv} = \|\hat{\epsilon}_{ref,t} - \hat{\epsilon}_{adv,t}\|_2^2$$

- The text prompt is left unmodified → CLIP Score is naturally maximized.
- No external classifiers are used → low computational overhead (~64s vs. ~238s for P4D).
- Only a single reference image is required (obtainable from the web).

## Key Experimental Results

### Main Results (Average ASR across 10 unlearning methods)

| Task | P4D-N | CCE | UnlearnDiffAtk | WACE-C | **Recall** |
|------|-------|-----|----------------|--------|------------|
| Nudity-I2P | 57.61 | 50.42 | 63.87 | 57.04 | **80.77** |
| Nudity-MMA | 69.51 | 55.70 | 76.52 | 66.25 | **88.20** |
| Van Gogh | 92.00 | 77.20 | 97.20 | 48.00 | **97.40** |
| Object-Church | 49.80 | 55.80 | 62.40 | 41.40 | **73.40** |
| Object-Parachute | 38.20 | 60.40 | 59.80 | 38.80 | **97.00** |

### Semantic Alignment (CLIP Score, I2P Task)

| Method | ESD | MACE | RECE | UCE | Receler | DoCo |
|--------|-----|------|------|-----|---------|------|
| P4D | 24.09 | 23.20 | 24.99 | 24.90 | 25.64 | 23.70 |
| UnlearnDiffAtk | 29.61 | 23.11 | 29.25 | 29.17 | 29.00 | 31.18 |
| **Recall** | **32.13** | **24.79** | **30.66** | **31.31** | **31.12** | **31.95** |

### Key Findings

- Against the strongest defense AdvUnlearn, Recall achieves ASR of 60.56% (I2P), 82.81% (MMA), and 92% (Van Gogh), far exceeding competing methods that achieve only single-digit to ~25%.
- Attack efficiency: ~64s on average, approximately 3.5× faster than P4D-N (~238s) and UnlearnDiffAtk (~232s).
- Reference image independence: ASR and diversity metrics remain stable across different reference images.
- Cross-model generalization: The method is effective on SD 2.0 and SD 2.1.
- Diversity: LPIPS/IS scores of generated images are comparable to text-only methods and far superior to image-only methods.

## Highlights & Insights

- **First systematic exploitation of the image modality to attack unlearning**: Exposes a "modality blind spot" in current unlearning methods — they defend against text but not against image inputs.
- **Single reference image + latent optimization = lightweight and efficient**: No original model, no external classifiers, and no large reference datasets are required.
- **Attack as auditing**: Recall simultaneously serves as a robustness auditing tool for model owners to systematically evaluate unlearning quality prior to deployment.
- **Semantics-preserving adversarial design**: Through the triple mechanism of unmodified text, noise initialization, and reference injection, generated content both recovers the target concept and maintains high semantic alignment with the prompt.

## Limitations & Future Work

- The method operates in a white-box setting (model weights required); applicability to black-box scenarios remains to be validated.
- ASR against MACE is relatively lower on certain tasks (e.g., 50% on Church), possibly due to MACE's LoRA-based mechanism.
- Evaluation is limited to the Stable Diffusion architecture family; newer architectures such as Flux/DiT have not been tested.
- A reference image containing the target concept is still required (though the constraint is loose); fully zero-reference scenarios are not supported.

## Related Work & Insights

- **vs. P4D**: A text optimization attack with poor semantic alignment (CLIP Score lower by 6+ points) and ASR of only 2–8% against AdvUnlearn.
- **vs. UnlearnDiffAtk**: Also white-box but optimizes only the text; achieves Avg. ASR of 63.87% on I2P compared to Recall's 80.77%.
- **vs. CCE**: Recovers concepts via textual inversion of a placeholder token, but achieves the lowest CLIP Score (~19), indicating severe semantic deviation.
- **vs. WACE**: A noise-probing-based method with limited effectiveness against strong defenses.
- Inspiration: The multi-modal attack paradigm can be extended to security auditing of video generation models and large multi-modal models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First multi-modal unlearning attack framework, opening an entirely new attack surface.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 unlearning methods × 4 task categories × 6 datasets × multiple baselines; extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and the method pipeline is easy to follow.
- Value: ⭐⭐⭐⭐⭐ — Provides important warnings for the unlearning security community and offers a practical auditing tool.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Follow-Your-Shape: Shape-Aware Image Editing via Trajectory-Guided Region Control](follow-your-shape_shape-aware_image_editing_via_trajectory-guided_region_control.md)
- [\[ICLR 2026\] Unified Multi-Modal Interactive & Reactive 3D Motion Generation via Rectified Flow](unified_multi-modal_interactive_reactive_3d_motion_generation_via_rectified_flow.md)
- [\[ICLR 2026\] There and Back Again: On the Relation between Noise and Image Inversions in Diffusion Models](there_and_back_again_on_the_relation_between_noise_and_image_inversions_in_diffu.md)
- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)
- [\[ICCV 2025\] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning](../../ICCV2025/image_generation/holistic_unlearning_benchmark_a_multi-faceted_evaluation_for_text-to-image_diffu.md)

<!-- RELATED:END -->
