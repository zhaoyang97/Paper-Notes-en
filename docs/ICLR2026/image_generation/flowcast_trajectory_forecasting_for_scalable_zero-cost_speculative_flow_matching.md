---
title: >-
  [Paper Note] FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching
description: >-
  [ICLR 2026][Image Generation][Flow Matching] FlowCast is a framework that introduces speculative decoding into Flow Matching models. It exploits the local smoothness of the velocity field to extrapolate future states usi…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Flow Matching"
  - "Speculative Decoding"
  - "Zero-Cost Acceleration"
  - "Inference Optimization"
  - "Trajectory Forecasting"
date: 2026-05-08
content_hash: eb5f5593387f24a2
---

# FlowCast: Trajectory Forecasting for Scalable Zero-Cost Speculative Flow Matching

**Conference**: ICLR 2026
**arXiv**: [2602.01329](https://arxiv.org/abs/2602.01329)  
**Code**: None  
**Area**: Diffusion Models / Inference Acceleration
**Keywords**: Flow Matching, Speculative Decoding, Zero-Cost Acceleration, Inference Optimization, Trajectory Forecasting

## TL;DR
FlowCast is a framework that introduces speculative decoding into Flow Matching models. It exploits the local smoothness of the velocity field to extrapolate future states using the current velocity prediction as a zero-cost draft, then selectively skips redundant steps via MSE-based verification, achieving >2.5× speedup without quality degradation.

## Background & Motivation

**Background**: Flow Matching (FM) has become a mainstream approach for high-quality generative modeling (e.g., FLUX.1, Wan video generation), mapping noise to data by solving an ODE. However, ODE integration is inherently sequential—each step depends on the previous output—making inference slow.

**Limitations of Prior Work**: Existing acceleration methods (distillation, trajectory truncation, consistency training) either degrade quality (blurred textures, semantic drift), require expensive retraining, or lack generalizability. The problem is amplified in video generation, where the temporal dimension multiplies the inference burden.

**Key Challenge**: FM's high fidelity depends on a sufficient number of sampling steps, yet more steps mean slower inference. There is a need to intelligently skip "unnecessary" steps without retraining.

**Goal**: To adaptively accelerate FM inference without introducing auxiliary models or performing any training.

**Key Insight**: FM models are trained to maintain a nearly constant velocity, and empirically the velocity field changes slowly between adjacent steps. This implies that the velocity prediction at the current step can serve as a "free draft" for future steps.

**Core Idea**: Use the FM model's own velocity prediction as a zero-cost draft for speculative extrapolation; accept the draft if MSE verification passes, otherwise roll back.

## Method

### Overall Architecture
FlowCast is a plug-and-play inference-time acceleration framework for FM, consisting of three components: **Drafting** (linearly extrapolating all future states using the current velocity) → **Verification** (computing true velocities at all draft points in parallel and checking MSE) → **Correction** (restarting computation from the first rejected point). No training or auxiliary networks are required.

### Key Designs

1. **Zero-Cost Drafting**:

    - **Function**: At the current timestep $t_i$, linearly extrapolate all remaining timestep states using velocity $v(x_{t_i}, t_i)$.
    - **Mechanism**: $\tilde{x}_{t_k} = x_{t_m} + (t_k - t_m) \cdot v_{t_m}$, for all $k = m+1, \ldots, K$.
    - **Design Motivation**: The FM training objective encourages constant velocity (linear interpolation paths), so the velocity changes minimally between adjacent steps. Reusing the current velocity for extrapolation is "free," requiring no additional forward computation.

2. **Parallel Verification**:

    - **Function**: Compute the true velocity $v(\tilde{x}_{t_k}, t_k)$ in parallel for each draft state $\tilde{x}_{t_k}$ and check its MSE against the draft velocity.
    - **Mechanism**: Accept the draft if $\text{MSE}(v_{t_m}, v_{t_k}) < \epsilon$; upon finding the first point $j$ where MSE exceeds the threshold, discard that point and all subsequent drafts.
    - **Design Motivation**: Verification in velocity space rather than data space more sensitively captures local dynamic inconsistencies with low computational overhead. Parallel verification allows multi-point evaluation in a single forward pass.

3. **Correction**:

    - **Function**: Roll back from the rejected point and begin a new extrapolation round using the true velocity computed at that point.
    - **Mechanism**: Accept $\{x_{m+1}, \ldots, x_{j-1}\}$ and re-extrapolate using $v_{t_{j-1}}$.
    - **Design Motivation**: Ensures trajectory fidelity—smooth dynamic regions are skipped aggressively for acceleration, while regions of rapid dynamics are integrated finely for accuracy.

### Theoretical Analysis
- **Lemma 4.1**: Under the conditions that the velocity field is Lipschitz continuous (constant $M$) and its second-order derivatives are bounded ($N$), the global error bound of speculative integration is $\|x(t_k) - x_k\| \leq \frac{e^{Mt_k}-1}{2M}(hN + 2p\sqrt{\epsilon})$.
- **Theorem 4.2**: Given tolerance $q_d$, to ensure the speculative error does not exceed $q_d$, the threshold must satisfy $\epsilon \leq (\frac{q_d}{2A})^2$, where $A = \frac{e^M-1}{M}$.

## Key Experimental Results

### Main Results
Text-to-image generation (GenEval benchmark + FLUX model):

| Method | Overall↑ | CLIPIQA↑ | Speedup |
|--------|----------|----------|---------|
| FLUX 50 steps | 0.65 | 0.83 | 1.0× |
| FLUX 25 steps | 0.64 | 0.80 | 2.0× |
| FLUX 10 steps | 0.57 | 0.59 | 5.0× |
| FlowCast + FLUX | 0.65 | 0.83 | **>2.5×** |

### Multi-Task Validation

| Task | Model | Speedup | Quality Loss |
|------|-------|---------|-------------|
| Text-to-image | FLUX | >2.5× | None |
| Text-to-image | BAGEL | >2.5× | None |
| Image editing | GEdit | >2.5× | None |
| Multi-turn editing | EditBench | >2.5× | None |
| Video generation | VBench | >2.5× | None |

### Key Findings
- FlowCast achieves >2.5× speedup with quality fully consistent with 50-step generation, whereas naively reducing steps to 25 or 10 results in noticeable quality degradation.
- The advantage is especially pronounced in multi-turn editing tasks—step-reduction methods accumulate errors across rounds, while FlowCast maintains accuracy without error accumulation.
- A key insight: over 60% of steps in FM trajectories exhibit velocity changes within the threshold and can be safely skipped.
- Quality is relatively insensitive to the threshold $\epsilon$ within a reasonable range, indicating robustness.

## Highlights & Insights
- **Zero-Cost Draft Insight**: The method exploits the constant-velocity property inherent in the FM training objective, requiring no auxiliary model training. The draft is entirely "free"—more lightweight than speculative decoding in LLMs, which requires a smaller draft model.
- **Transferring Speculative Decoding from LLMs to Vision**: Although FM is not an autoregressive model, its sequential ODE integration shares structural similarity with autoregression. The paper cleverly leverages velocity smoothness to address the two key challenges: how to construct a draft and how to verify it.
- **Plug-and-Play and Composable**: The approach is orthogonal to existing acceleration techniques and can be applied on top of distillation or trajectory truncation for further gains.

## Limitations & Future Work
- The speedup ratio depends on the smoothness of the velocity field; for certain models or scenarios with rich high-frequency details (e.g., audio generation), the gain may be limited.
- Only the Euler solver is evaluated; the effectiveness under higher-order solvers (e.g., midpoint, RK4) remains unexplored.
- Parallel verification requires computing velocities at all draft points simultaneously, increasing memory overhead as the number of steps grows.
- The threshold $\epsilon$ must be set manually; while robust, the optimal value may vary across tasks.

## Related Work & Insights
- **vs. TeaCache**: TeaCache uses timestep embedding similarity for step skipping, whereas FlowCast directly uses velocity MSE for verification—more principled and theoretically grounded.
- **vs. Distillation Methods (InstaFlow, etc.)**: Distillation requires training and incurs quality loss; FlowCast is training-free with zero quality loss.
- **vs. Consistency Models**: Consistency models require modifications to the training pipeline; FlowCast is a purely post-hoc inference-time acceleration strategy.
- **Transferable Idea**: The speculative generation framework generalizes to any ODE-based generative model.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce speculative decoding into FM inference; the zero-cost draft design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers image generation, editing, and video generation with theoretical analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation and methodology are clearly explained; theory and experiments are well integrated.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play, training-free acceleration solution of high value to the FM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FlowCast: Advancing Precipitation Nowcasting with Conditional Flow Matching](flowcast_advancing_precipitation_nowcasting_with_conditional_flow_matching.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](multi-agent_coordination_via_flow_matching.md)
- [\[ICLR 2026\] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting](doflow_flow-based_generative_models_for_interventional_and_counterfactual_foreca.md)
- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation](senseflow_scaling_distribution_matching_for_flow-based_text-to-image_distillatio.md)

</div>

<!-- RELATED:END -->
