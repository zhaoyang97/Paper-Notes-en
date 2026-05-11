---
title: >-
  [Paper Note] Steer Away From Mode Collisions: Improving Composition In Diffusion Models
description: >-
  [ICLR 2026][Image Generation][Compositional Generation] To address concept missing and collision in multi-concept prompts for diffusion models…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Compositional Generation"
  - "Mode Collision"
  - "Tweedie Mean Composition"
  - "Gradient-Free Correction"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: dc431a4c7a58b1cf
---

# Steer Away From Mode Collisions: Improving Composition In Diffusion Models

**Conference**: ICLR 2026
**arXiv**: [2509.25940](https://arxiv.org/abs/2509.25940)
**Code**: [https://github.com/debottam-dutta7/co3](https://github.com/debottam-dutta7/co3)
**Area**: Diffusion Models / Compositional Generation
**Keywords**: Compositional Generation, Mode Collision, Tweedie Mean Composition, Gradient-Free Correction, Plug-and-Play

## TL;DR

To address concept missing and collision in multi-concept prompts for diffusion models, this paper proposes the "mode collision" hypothesis — that the modes of the joint distribution overlap with those of individual concept distributions — and introduces CO3 (Concept Contrasting Corrector). CO3 corrects sampling via a contrasting distribution $\tilde{p}(x|C) \propto p(x|C) / \prod_i p(x|c_i)$ in Tweedie mean space to steer away from degenerate modes, achieving plug-and-play, gradient-free, and model-agnostic compositional generation improvement.

## Background & Motivation

**Background**: Diffusion models have achieved remarkable progress in text-to-image generation, yet even simple multi-concept prompts (e.g., "a cat and a dog") frequently suffer from concept missing, blurring, or unnatural merging.

**Limitations of Prior Work**:
- Optimization-based correction methods (Attend-Excite, SynGen, ToMe) require gradient computation through the model and are therefore model-specific.
- Composable diffusion methods are model-agnostic but perform poorly, as linear score composition does not correspond to the correct forward distribution for $t > 0$.
- Each family has its own trade-offs, and no unified framework exists.

**Key Challenge**: The joint distribution $p(x|C)$ under multi-concept prompts contains "problematic modes" that overlap heavily with the individual concept distributions $p(x|c_i)$, causing generation to collapse toward a dominant concept.

**Goal**: (i) Theoretically analyze and unify existing approaches (correction vs. composable diffusion); (ii) Design a training-free, gradient-free, and model-agnostic sampling correction strategy to improve multi-concept composition.

**Key Insight**: The paper introduces the "mode collision" hypothesis — when certain modes of $p(x|C)$ overlap with $p(x|c_i)$, sampling tends to collapse to that single concept. The solution is to design a correcting distribution that suppresses these overlapping regions.

**Core Idea**: Steer sampling away from single-concept-dominated degenerate modes via $\tilde{p}(x|C) \propto p(x|C) / \prod_i p(x|c_i)$, and show that Tweedie mean composition constitutes a unifying framework.

## Method

### Overall Architecture

CO3 is a sampling-stage corrector embedded within the DDIM sampling process. Corrections are applied during the first 20% of denoising steps, divided into two phases: (1) **CO3-resampler**: noise resampling for the first 3 steps (weights sum to 0); (2) **CO3-corrector**: latent correction for subsequent steps (weights sum to 1). No model parameters are modified and no gradient computation is required.

### Key Designs

1. **Tweedie Mean Composition Framework**:

    - **Function**: Unifies correction-based and composable diffusion methods under a common theoretical foundation.
    - **Mechanism**: Moves distribution composition from score space to Tweedie mean space: $\tilde{x}_{\text{tweedie}} = w_0 \hat{x}_{\text{tweedie}}[\epsilon_t^{\lambda,C}] + \sum_{k=1}^K w_k \hat{x}_{\text{tweedie}}[\epsilon_t^{\lambda,c_k}]$, where $\hat{x}_{\text{tweedie}}[\epsilon_t^{\lambda,c}] = x_t - \sigma_t \epsilon_t^{\lambda,c}$.
    - **Design Motivation**: Proposition 1 proves that when $\sum_k w_k = 1$, the composed Tweedie mean remains a valid Tweedie mean (CO3-corrector); when $\sum_k w_k = 0$, it reduces to weighted noise (CO3-resampler, theoretically valid only at $t = T$).

2. **CO3-resampler (weights sum to 0)**:

    - **Function**: Replaces the initial noise at early high-noise steps.
    - **Mechanism**: Replaces the current $x_t$ with a weighted combination of concept-conditioned noise estimates, effectively resampling the starting noise from a concept-suppressed distribution.
    - **Design Motivation**: Experiments show that resampling is most effective at high $t$, helping to suppress concept dominance at initialization.

3. **CO3-corrector (weights sum to 1)**:

    - **Function**: Corrects the Tweedie mean at intermediate denoising steps.
    - **Mechanism**: $w_0 > 0$ (joint prompt weight); $w_1, \ldots, w_K < 0$ (negative weights for individual concepts, acting as suppression). Crucially, the composition retains the CFG form: $\tilde{\epsilon}_t^{\tilde{\lambda},C} = \epsilon_t^\phi + \lambda(\sum_k w_k \epsilon_t^{c_k} - \epsilon_t^\phi)$.
    - **Design Motivation**: Unlike composable diffusion, which uses arbitrary $\lambda_i$, CO3-corrector preserves the unconditional-to-conditional ratio of CFG, preventing out-of-distribution samples.

4. **Closeness-Aware Concept Weight Modulation**:

    - **Function**: Adaptively adjusts the suppression weight for each concept.
    - **Mechanism**: Computes the distance $d_k$ between the current noise prediction $\epsilon^C$ and each concept noise $\epsilon^{c_k}$, converts distances to affinities via an exponential kernel $a_k = \exp(-\beta d_k)$, normalizes them, and uses the result as negative weights $w_k = -a_k / \sum_j a_j$.
    - **Design Motivation**: Concepts closer to the current sample should be suppressed more strongly, enabling dynamic balancing across concepts.

### Loss & Training

This is a training-free method. It is built on SDXL with 50-step DDIM sampling and guidance scale $\lambda = 5.0$. Corrections are applied in the first 20% of steps: the resampler is used for the first 3 steps and the corrector for subsequent steps. $\beta = 0.8$.

## Key Experimental Results

### Main Results (Two-concept prompts, Attend-Excite benchmark)

| Method | Training-Free | Gradient-Free | Model-Agnostic | ImageReward (A-A) | ImageReward (A-O) | ImageReward (O-O) |
|--------|--------------|--------------|----------------|-------------------|-------------------|-------------------|
| SDXL | ✓ | ✓ | - | 0.782 | 1.547 | 0.679 |
| Attend-Excite | ✓ | ✗ | ✗ | 0.824 | 1.238 | 0.874 |
| InitNO | ✓ | ✗ | ✓ | 1.008 | 1.393 | 1.138 |
| Tweediemix | ✓ | ✓ | ✓ | 1.002 | 1.313 | 0.796 |
| **CO3 (Ours)** | ✓ | ✓ | ✓ | **1.234** | **1.674** | **1.016** |

### Ablation Study (Incremental component addition)

| Configuration | ImageReward Avg | BLIP-VQA Avg |
|--------------|----------------|-------------|
| SDXL base | 0.843 | — |
| + Resampling | 0.944 | — |
| + Corrector | 0.946 | — |
| + Weight modulation | **1.012** | — |

### Key Findings

- CO3 is the only method that simultaneously satisfies all three properties — training-free, gradient-free, and model-agnostic — while matching or surpassing gradient-based methods across all metrics.
- The weight-sum-to-1 constraint is critical: it preserves the CFG form, whereas composable diffusion with arbitrary weights readily produces out-of-distribution samples.
- The resampler and corrector play complementary roles: the resampler is effective in the high-noise regime, while the corrector operates in the intermediate denoising phase.
- Weight modulation contributes substantially (Avg: 0.946 → 1.012), demonstrating that adaptive suppression is more effective than fixed weights.
- CO3 also outperforms methods specifically designed for complex prompts (T2ICompBench) and rare concepts (RareBench), including specialized approaches such as R2F.

## Highlights & Insights

- **Strong Theoretical Unification**: Proposition 1 unifies existing correction and composable diffusion methods under the Tweedie mean composition framework, revealing the critical role of weight constraints ($\sum w_k = 1$ preserves CFG form; $\sum w_k = 0$ enables resampling). This unified perspective is itself a valuable theoretical contribution.
- **Mode Collision Hypothesis**: Suppressing degenerate modes overlapping with individual concept distributions via $p(x|C) / \prod_i p(x|c_i)$ is intuitive and well-supported by experimental evidence, which indirectly validates the hypothesis.
- **Fully Plug-and-Play**: No model modification, no gradient computation, and no additional training are required, making CO3 directly applicable to any diffusion model.

## Limitations & Future Work

- Each denoising step requires $K+1$ separate noise predictions, so inference cost scales linearly with the number of concepts.
- Evaluation is conducted solely on SDXL; although model-agnosticism is claimed, the method has not been tested on DiT-based architectures (e.g., Flux/SD3).
- CO3-resampler is not theoretically rigorous for $t < T$; while empirically effective up to approximately $t \approx 0.9T$, formal guarantees are lacking.
- The $\beta$ parameter in weight modulation and the proportion of correction steps require manual tuning.
- Concept decomposition relies on prompt parsing; the quality of decomposition for complex natural-language prompts is not discussed.

## Related Work & Insights

- **vs. Attend-Excite**: Attend-Excite optimizes attention maps for attribute binding, requiring gradients and being architecture-dependent; CO3 addresses the problem at the distribution mode level, offering a theoretically higher-level perspective.
- **vs. Composable Diffusion**: Both are model-agnostic score composition approaches, but composable diffusion uses unconstrained linear combinations; CO3 demonstrates that the weight constraint is the key factor.
- **vs. Tweediemix**: Both operate in Tweedie space, but Tweediemix requires a LoRA fine-tuning stage, whereas CO3 is a purely sampling-time correction.
- The mode collision hypothesis may inspire deeper investigation into the training dynamics of diffusion models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The mode collision hypothesis is novel, and the unifying theory based on Tweedie mean composition is insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple benchmarks (simple/complex/rare prompts), complete ablations, and human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical derivations are clear, though the heavy notation raises the reading barrier.
- **Value**: ⭐⭐⭐⭐ Plug-and-play compositional generation improvement carries high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition](compose_your_policies_improving_diffusion-based_or_flow-based_robot_policies_via.md)
- [\[ICLR 2026\] Self-Improving Loops for Visual Robotic Planning](self-improving_loops_for_visual_robotic_planning.md)
- [\[ICLR 2026\] Improving Discrete Diffusion Unmasking Policies Beyond Explicit Reference Policies (UPO)](improving_discrete_diffusion_unmasking_policies_beyond_explicit_reference_polici.md)
- [\[NeurIPS 2025\] Composition and Alignment of Diffusion Models using Constrained Learning](../../NeurIPS2025/image_generation/composition_and_alignment_of_diffusion_models_using_constrai.md)
- [\[ICCV 2025\] Less is More: Improving Motion Diffusion Models with Sparse Keyframes](../../ICCV2025/image_generation/less_is_more_improving_motion_diffusion_models_with_sparse_keyframes.md)

</div>

<!-- RELATED:END -->
