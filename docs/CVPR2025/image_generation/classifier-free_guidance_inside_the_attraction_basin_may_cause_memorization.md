---
title: >-
  [Paper Note] Classifier-Free Guidance inside the Attraction Basin May Cause Memorization
description: >-
  [CVPR 2025][Image Generation][Diffusion Model Memorization] From a dynamical systems perspective, the concept of "Attraction Basin" is proposed to explain the memorization phenomenon in diffusion models. Applying CFG inside the attraction basin causes trajectories to converge to memorized training images. Detecting transition points to delay CFG activation (combined with Opposite Guidance, OG) mitigates memorization with zero extra computational overhead.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Diffusion Model Memorization"
  - "Attraction Basin"
  - "Classifier-Free Guidance"
  - "Privacy Leakage"
  - "Mitigation Strategies"
date: 2026-05-08
content_hash: 0d54c57d123b7724
---

# Classifier-Free Guidance inside the Attraction Basin May Cause Memorization

**Conference**: CVPR 2025  
**arXiv**: [2411.16738](https://arxiv.org/abs/2411.16738)  
**Code**: [https://github.com/SonyResearch/mitigating_memorization](https://github.com/SonyResearch/mitigating_memorization)  
**Area**: Diffusion Models / AI Safety  
**Keywords**: Diffusion Model Memorization, Attraction Basin, Classifier-Free Guidance, Privacy Leakage, Mitigation Strategies

## TL;DR
From a dynamical systems perspective, the concept of "Attraction Basin" is proposed to explain the memorization phenomenon in diffusion models. Applying CFG inside the attraction basin causes trajectories to converge to memorized training images. Detecting transition points to delay CFG activation (combined with Opposite Guidance, OG) mitigates memorization with zero extra computational overhead.

## Background & Motivation

**Background**: Diffusion models (such as Stable Diffusion) can replicate images from training data verbatim (verbatim memorization), raising concerns about copyright infringement and privacy leakage. Known factors include training data duplication, overly specific prompts, and fine-tuning on small datasets, but the issues persist even after deduplication.

**Limitations of Prior Work**: Existing mitigation methods either modify the training phase (requiring expensive retraining), perturb the prompt/embedding during inference (damaging text alignment), or reduce the attention weights of trigger tokens (only effective for specific types of memorization). A key drawback is that these methods are designed for specific memorization scenarios and fail to generalize across scenarios. For example, the method by Wen et al. is effective in data duplication scenarios but fails in fine-tuning scenarios, while the method by Ren et al. requires the presence of trigger tokens.

**Key Challenge**: Memorization occurs during the high-noise phase of the denoising process, where the conditional guidance of CFG is exceptionally strong, "attracting" trajectories from different initializations into the same memorized image. However, completely disabling CFG leads to a degradation in image quality and text alignment.

**Goal**: To mitigate diffusion model memorization at inference time with zero extra computational overhead, without relying on prompt modification, and with the ability to generalize across multiple memorization scenarios.

**Key Insight**: It is observed that enabling CFG after a certain timestep in the late denoising phase does not produce memorized outputs. This implies the existence of a "transition point" $\tau^*$—before which the denoising trajectory is inside the attraction basin (where CFG causes memorization), and after which the trajectory escapes the basin (where CFG operates normally). The transition point can be detected via a sudden drop in the magnitude of the difference between conditional and unconditional noise predictions.

**Core Idea**: Avoid applying CFG inside the attraction basin (delaying it until after the transition point), or employ Opposite Guidance (OG) to accelerate escape from the attraction basin, thereby mitigating memorization with zero extra computational overhead.

## Method

### Overall Architecture
During standard diffusion inference, the $L_2$ magnitude of conditional guidance, $\|\epsilon_\theta(x_t, e_p) - \epsilon_\theta(x_t, e_\emptyset)\|^2$, is monitored. When the first local minimum is detected, it is marked as the transition point, after which normal CFG is applied. Before the transition point, one can choose to apply no guidance (zero CFG) or opposite guidance (negative CFG). The entire process reuses the existing conditional and unconditional prediction calculations of CFG, incurring zero extra overhead.

### Key Designs

1. **Attraction Basin Theory**

    - **Function**: Provides a dynamical systems explanation of the memorization phenomenon.
    - **Mechanism**: The denoising process is viewed as a dynamical system, where memorized training images $x^a$ act as attractors. The attraction basin is defined as the set of all points in the state space that will converge to the vicinity of $x^a$ under CFG: $X^b(x^a, \epsilon) = \{(x,t) | \mathbb{P}(\varphi(x,t,e) \in B_D(x^a, \epsilon)) > 1-\delta\}$. The basin is widest at $t=T$ (almost covering the entire space) and gradually narrows as denoising progresses. Key insight: without CFG, the trajectory naturally escapes the basin; applying CFG is equivalent to applying a force directing the trajectory toward the attractor, keeping it trapped inside the basin.
    - **Design Motivation**: Existing explanations focus only on surface factors such as trigger tokens or data duplication; the attraction basin provides a unified mechanism to explain these phenomena.

2. **Transition Point Detection and Delayed CFG (STP/DTP)**

    - **Function**: Finds the critical timestep transitioning from memorization to non-memorization, precisely determining the optimal starting time for CFG.
    - **Mechanism**: **Static Transition Point (STP)**: In some models (such as the fine-tuned SDv2.1), all samples share the same transition point (e.g., $t=500$), which can be directly hardcoded. **Dynamic Transition Point (DTP)**: In other models (such as pretrained SDv1.4), each prompt/initialization pair has a different transition point. By tracking $d_t = \|\epsilon_\theta(x_t, e_p) - \epsilon_\theta(x_t, e_\emptyset)\|_2^2$ in real time, the first local minimum is detected ($d_{t+2} > d_{t+1} < d_t$), after which the system switches to normal CFG. Crucially, conditional and unconditional predictions are already computed in CFG, requiring no extra feedforward passes.
    - **Design Motivation**: Formulated from empirical observations in Figure 2—when the magnitude is high and stable, the trajectory is inside the basin; when it drops sharply, it escapes the basin.

3. **Opposite Guidance (OG)**

    - **Function**: Accelerates the denoising trajectory to escape the attraction basin, making the transition point occur earlier.
    - **Mechanism**: Before the transition point, a negative CFG is applied: $\hat{\epsilon} = \epsilon_\theta(x_t, e_\emptyset) - s(\epsilon_\theta(x_t, e_p) - \epsilon_\theta(x_t, e_\emptyset))$, pushing the trajectory in the direction opposite to the conditional guidance. Experiments show that OG advances the transition point from approximately $t=779$ to about $t=839$, leaving more steps for normal CFG and improving image quality and text alignment.
    - **Design Motivation**: When the transition point occurs late ($t \leq 500$), the zero CFG phase is too long, leaving insufficient steps for normal CFG and degrading image quality. OG resolves this issue by actively pushing the trajectory out of the basin.

### Loss & Training
This is an entirely inference-time method, requiring no training or modification of model weights. All computations (conditional/unconditional predictions) are already included in standard CFG inference. SDv1.4 inference time is 1.26s vs. 2.86s for Wen et al. (A100 GPU).

## Key Experimental Results

### Main Results

| Scenario | Method | Similarity(95%)↓ | CLIP↑ | FID↓ |
|------|------|-------------|-------|------|
| S1: SDv2.1 Fine-tuned LAION-10k | No Mitigation | 0.6504 | 0.3027 | 16.84 |
| S1 | Wen et al. | 0.3853 | 0.2895 | 16.72 |
| S1 | **OG+STP** | **0.3811** | **0.3020** | **15.67** |
| S3: SDv1.4 Data Duplication | No Mitigation | 0.7977 | 0.3105 | 106.49 |
| S3 | Wen et al. (lt=1) | 0.6038 | 0.3050 | 136.34 |
| S3 | **DTP** | **0.5885** | 0.3020 | 138.92 |

### Ablation Study

| Configuration | Similarity(95%)↓ | CLIP↑ | Description |
|------|-------------|-------|------|
| Standard CFG (Full) | 0.6504 | 0.3027 | Memorization |
| Zero CFG [1000,500] + CFG [500,0] (STP) | 0.2857 | 0.2976 | Memorization eliminated, CLIP slightly decreased |
| OG [1000,STP] + CFG [STP,0] | 0.3811 | 0.3020 | OG improves CLIP (more CFG steps) |

### Key Findings
- The attraction basin exists in all four memorization scenarios, validating the universality of the theory.
- Other methods are only effective in specific scenarios: Ren et al. is almost ineffective in Scenario 1 (similarity only drops from 0.6504 to 0.6028), while Wen et al. reduces similarity in Scenario 1 but results in poor image quality.
- The proposed method is the only one that is consistently effective across all scenarios.
- OG improves FID from 19.85 (STP only) to 15.67, indicating that an earlier transition point leaves more generation steps for CFG.
- Zero extra computational overhead (1.26s vs. 2.86s for Wen et al.), as it reuses the existing computations in CFG.

## Highlights & Insights
- **Attraction basin perspective** provides an elegant dynamical systems explanation for diffusion model memorization, unifying multiple causes of memorization (data duplication, fine-tuning overfitting, trigger tokens)—all of which fundamentally stem from the excessively strong guidance of CFG within the basin.
- **Zero-overhead mitigation** is highly practical: it does not modify the model, prompt, or increase computation, but simply alters the temporal scheduling of CFG. It can be directly deployed to any diffusion model utilizing CFG.
- **Opposite Guidance (OG)** is a novel concept: negative CFG is not a negative prompt, but rather an active push away from the memorization trajectory inside the attraction basin, achieving a better balance between image quality and mitigation effectiveness.

## Limitations & Future Work
- It requires prior detection of whether memorization exists (the detection itself requires additional computation or prior knowledge).
- Non-memorized samples do not have transition points; applying the method to them is harmless but does not yield benefits.
- When the transition point is very late ($t \leq 500$), even with OG, the number of CFG steps may still be insufficient, affecting image quality.
- Only validated on SD v1.4/v2.1; generalization to newer architectures such as SDXL and Flux remains unknown.

## Related Work & Insights
- **vs. Wen et al.**: Minimizes conditional noise differences by optimizing prompt embeddings, but increases inference time and does not generalize across scenarios. The proposed method has zero extra overhead and remains effective in scenarios where Wen et al. fails.
- **vs. Ren et al.**: Lowers attention weights of trigger tokens, which is only effective when trigger tokens are present. Without a trigger token, it only reduces similarity from 0.6504 to 0.6028.
- **vs. Chen et al.**: Designs different CFG weight schedulers, but requires prior knowledge of the memorization type. The proposed method automatically detects the transition point.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The theoretical framework of the attraction basin and the concept of opposite guidance are highly original, providing a new paradigm for understanding and mitigating memorization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively validated across four scenarios and compared with multiple baselines, though testing on newer architectures is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical definitions, outstanding visualization (Figure 2 is the highlight of the paper), and fluent writing.
- Value: ⭐⭐⭐⭐⭐ Direct value for the safety and privacy protection of diffusion models; the zero-overhead characteristic makes it highly deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TCFG: Tangential Damping Classifier-Free Guidance](tcfg_tangential_damping_classifier-free_guidance.md)
- [\[CVPR 2025\] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[NeurIPS 2025\] Towards a Golden Classifier-Free Guidance Path via Foresight Fixed Point Iterations](../../NeurIPS2025/image_generation/towards_a_golden_classifier-free_guidance_path_via_foresight_fixed_point_iterati.md)
- [\[CVPR 2026\] CFG-Ctrl: Control-Based Classifier-Free Diffusion Guidance](../../CVPR2026/image_generation/cfg-ctrl_control-based_classifier-free_diffusion_guidance.md)

</div>

<!-- RELATED:END -->
