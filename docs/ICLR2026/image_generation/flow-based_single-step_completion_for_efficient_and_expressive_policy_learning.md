---
title: >-
  [Paper Note] SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning
description: >-
  [ICLR 2026][Image Generation][offline RL] This paper proposes the Single-Step Completion Policy (SSCP), which compresses multi-step generative policies into single-step inference by predicting a "completion vector" (the…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "offline RL"
  - "flow matching"
  - "single-step generation"
  - "completion vector"
  - "policy learning"
  - "D4RL"
date: 2026-05-08
content_hash: 822a50f2ea124317
---

# SSCP: Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning

**Conference**: ICLR 2026
**arXiv**: [2506.21427](https://arxiv.org/abs/2506.21427)
**Code**: [GitHub](https://github.com/PrajwalKoirala/SSCP-Single-Step-Completion-Policy)
**Area**: Image Generation
**Keywords**: offline RL, flow matching, single-step generation, completion vector, policy learning, D4RL

## TL;DR
This paper proposes the Single-Step Completion Policy (SSCP), which compresses multi-step generative policies into single-step inference by predicting a "completion vector" (the normalized direction from any intermediate state to the target action) within a flow matching framework. On D4RL, SSCP matches multi-step diffusion/flow policies while achieving 64× faster training and 4.7× faster inference, and extends to GCRL to flatten hierarchical policies.

## Background & Motivation
**Background**: Diffusion/flow matching generative policies excel in offline RL due to their ability to capture multimodal action distributions (e.g., DQL, CAC). However, they require tens of iterative sampling steps, resulting in high inference latency.

**Limitations of Prior Work**:
   - **Inference efficiency**: Diffusion policies require 5–50 denoising steps per action, making them unsuitable for real-time control (DQL ~1.27ms vs. deterministic policies ~0.1ms).
   - **Training instability**: Backpropagating policy gradients through multi-step sampling chains (BPTT) causes gradient instability and long training times (DQL ~8 hours vs. TD3+BC ~30 minutes).
   - **Bootstrap issues in shortcut methods**: The shortcut model of Frans et al. 2024 uses its own predictions as training targets (self-consistency loss), which is unstable in dynamic-target settings such as RL.

**Key Challenge**: Is there an inherent trade-off between the expressiveness of generative policies and their inference/training efficiency?

**Core Idea**: At an intermediate timestep $\tau$ of flow matching, train the model to predict a completion vector pointing directly to the target $x_1$ (rather than the velocity field), supervised by ground-truth data (not bootstrap), enabling single-step generation.

## Method

### Overall Architecture
The linear interpolation path of flow matching $x_\tau = (1-\tau)z + \tau x_1$ → at $\tau$, train the model to predict two quantities: (1) instantaneous velocity $h_\theta(x_\tau, \tau, d{=}0)$ (standard flow loss), and (2) completion vector $h_\theta(x_\tau, \tau, d{=}1{-}\tau)$ (direct jump to $x_1$) → at inference, start from noise $z$ and complete in one step: $\pi_\theta(s) = z + h_\theta(z, s, 0, 1) \cdot 1$

### Key Designs

1. **Completion Vector**

    - Function: Predicts the normalized direction from any intermediate point $x_\tau$ in flow matching to the final target $x_1$.
    - Core formula: $\hat{x}_1 = x_\tau + h_\theta(x_\tau, \tau, 1{-}\tau) \cdot (1{-}\tau)$
    - Training loss: $\mathcal{L}_{completion} = \mathbb{E}[\|x_\tau + h_\theta(x_\tau, \tau, 1{-}\tau)(1{-}\tau) - x_1\|^2]$
    - **Key distinction from shortcut methods**: The completion loss uses the ground-truth $x_1$ from the dataset as the supervision target, rather than bootstrap self-predictions. This eliminates the instability of the self-consistency loss.
    - Design Motivation: Action spaces are low-dimensional (typically <20 dimensions), making direct regression of the completion vector far easier than in image generation.

2. **Joint Three-Objective Training (SSCQL)**

    - Function: Combines flow loss + completion loss + Q-learning policy gradient.
    - Total loss: $\mathcal{L}_\pi = \alpha_1 \mathcal{L}_{flow} + \alpha_2 \mathcal{L}_{completion} + \mathcal{L}_{\pi_Q}$
    - The flow loss constrains the velocity field (preserving expressiveness and distribution matching).
    - The completion loss constrains single-step generation quality (behavioral cloning regularization).
    - The Q-learning policy gradient optimizes action value.
    - Critic loss: Standard twin Q-learning with target networks.

3. **Single-Step Inference**

    - At inference, set $\tau=0, d=1$, i.e., go from pure noise to action in one step: $\pi_\theta(s) = z + h_\theta(z, s, 0, 1)$
    - A single forward pass, comparable in speed to deterministic policies.
    - Outputs are deterministic for a fixed $z$, but different $z$ samples produce a multimodal distribution.

4. **Goal-Conditioned Extension (GC-SSCP)**

    - Function: Compresses hierarchical GCRL (e.g., the high-level + low-level policies of HIQL) into a single flat policy.
    - Mechanism: Trains a flat policy using the completion model to match the combined output of the hierarchical policy, enabling single-step decisions at inference.
    - Analogy: SSCP compresses multi-step flow generation into one step → GC-SSCP compresses multi-level decision-making into a single level.

### Loss & Training
- Actor: $\alpha_1 \mathcal{L}_{flow} + \alpha_2 \mathcal{L}_{completion} + \mathcal{L}_{\pi_Q}$
- Critic: Twin Q-learning with soft target network updates.
- Optimizer: Adam, batch size 256.
- Training time: ~16 minutes (vs. DQL ~8 hours).

## Key Experimental Results

### Main Results on D4RL Offline RL

| Method | Type | D4RL Avg (9 tasks) | Training Time | Inference Latency | Denoising Steps |
|--------|------|-------------------|---------------|-------------------|-----------------|
| DQL | Diffusion policy | 87.9 | ~8h | 1.27ms | 5 |
| CAC | Flow policy | 85.1 | ~5h | 0.85ms | 2 |
| TD3+BC | Deterministic | 85.2 | ~30min | 0.08ms | 1 |
| **SSCQL** | **Single-step completion** | **87.9** | **~16min** | **0.27ms** | **1** |

SSCQL matches the strongest diffusion baseline DQL while achieving **64× faster training and 4.7× faster inference**.

### Offline-to-Online Fine-tuning

| Method | Stability | Notes |
|--------|-----------|-------|
| DQL | Frequent degradation (>10%) | Multi-step sampling chain causes instability during fine-tuning |
| CAC | Frequent degradation | Same issue |
| Cal-QL | Stable | SOTA designed specifically for O2O |
| **SSCQL** | **Stable improvement** | Single-step avoids BPTT instability |

### Online RL

| Method | HalfCheetah | Hopper | Walker2d |
|--------|-------------|--------|----------|
| DQL | Poor | Poor | Poor |
| CAC | Poor | Poor | Poor |
| **SSCQL** | **Best** | **Best** | **Best** |

### Goal-Conditioned RL (OGBench)

GC-SSCP (flat policy) outperforms HIQL (hierarchical policy) on average, demonstrating that the completion model successfully compresses hierarchical structure into flat decision-making.

### Key Findings
- The low dimensionality of action spaces (<20 dimensions) makes direct regression of completion vectors feasible — this is the key reason SSCP is effective in RL but may not transfer to image generation.
- Both the flow loss and completion loss are individually necessary: the flow loss ensures expressiveness, while the completion loss ensures single-step quality.
- Multi-step diffusion/flow policies are unstable in O2O fine-tuning and online RL — BPTT is the root cause.
- GC-SSCP demonstrates the broader applicability of the completion model for policy compression (not just generation step compression).

## Highlights & Insights
- **Ground-truth supervision replacing bootstrap** is the most central innovation: self-consistency losses based on bootstrap are unreliable in RL's dynamic-target setting, whereas completion vectors can be directly supervised by data. Simple but critical.
- **64× training speedup + 4.7× inference speedup** while maintaining equivalent performance — making generative policies viable for real-time control.
- **From generation compression to decision compression**: The extension from SSCP to GC-SSCP demonstrates the generality of the completion model — it compresses not only sampling steps but also decision hierarchies.

## Limitations & Future Work
- The balancing coefficients $\alpha_1, \alpha_2$ require tuning and may need different settings across tasks.
- Completion predictions at early $\tau$ values may be inaccurate (high noise, low information); theoretical analysis is absent.
- Validation is limited to MuJoCo continuous control tasks; performance in high-dimensional action spaces (e.g., robotic manipulation, autonomous driving) has not been tested.
- Comparison with distillation-based methods (e.g., consistency models) is missing.

## Related Work & Insights
- **vs. DQL (Wang et al. 2022)**: DQL uses a diffusion policy + DDPG+BC, requiring 5 denoising steps; SSCQL uses a completion policy with 1 step, achieving equivalent performance but 64× faster.
- **vs. Shortcut Models (Frans et al. 2024)**: Shortcut models use bootstrap self-consistency objectives, which are unstable; SSCP uses ground-truth completion vectors, which are stable.
- **vs. CAC**: CAC uses flow matching + 2-step denoising + consistency distillation; SSCP is simpler and more direct, requiring no distillation.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using completion vectors to replace bootstrap is simple but effective; the insight of ground-truth supervision is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers D4RL + O2O + Online + BC + GCRL comprehensively.
- Writing Quality: ⭐⭐⭐⭐ Clear and progressive exposition.
- Value: ⭐⭐⭐⭐⭐ Makes generative policies viable for real-time control; the 64× training speedup has substantial practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI](../../AAAI2026/image_generation/efficientflow_efficient_equivariant_flow_policy_learning_for_embodied_ai.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](../../CVPR2026/image_generation/renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[AAAI 2026\] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation](../../AAAI2026/image_generation/mp1_meanflow_tames_policy_learning_in_1-step_for_robotic_manipulation.md)
- [\[ICLR 2026\] CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models](cmt_mid-training_for_efficient_learning_of_consistency_mean_flow_and_flow_map_mo.md)
- [\[ICLR 2026\] SoFlow: Solution Flow Models for One-Step Generative Modeling](soflow_solution_flow_models_for_one-step_generative_modeling.md)

</div>

<!-- RELATED:END -->
