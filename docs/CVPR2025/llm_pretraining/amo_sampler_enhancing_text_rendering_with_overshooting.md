---
title: >-
  [Paper Note] AMO Sampler: Enhancing Text Rendering with Overshooting
description: >-
  [CVPR 2025][LLM Pretraining][text rendering] This paper proposes the Attention-Modulated Overshooting (AMO) sampler, a training-free inference-time enhancement method. By introducing an overshooting-noise compensation Langevin dynamics correction during the sampling process of rectified flow models, and adaptively controlling the overshooting intensity using text-image cross-attention scores, it significantly improves text rendering accuracy while maintaining the overall qual…
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "text rendering"
  - "rectified flow"
  - "overshooting sampler"
  - "attention modulation"
  - "training-free"
date: 2026-05-08
content_hash: 0543cfe9cbed9cc9
---

# AMO Sampler: Enhancing Text Rendering with Overshooting

**Conference**: CVPR 2025  
**arXiv**: [2411.19415](https://arxiv.org/abs/2411.19415)  
**Authors**: Xixi Hu, Keyang Xu, Bo Liu, Qiang Liu, Hongliang Fei  
**Affiliations**: Google, University of Texas at Austin  
**Code**: [https://github.com/hxixixh/amo-release](https://github.com/hxixixh/amo-release)  
**Area**: Diffusion Models/Image Generation  
**Keywords**: text rendering, rectified flow, overshooting sampler, attention modulation, training-free  

## TL;DR
This paper proposes the Attention-Modulated Overshooting (AMO) sampler, a training-free inference-time enhancement method. By introducing an overshooting-noise compensation Langevin dynamics correction during the sampling process of rectified flow models, and adaptively controlling the overshooting intensity using text-image cross-attention scores, it significantly improves text rendering accuracy while maintaining the overall quality of generated images.

## Background & Motivation
**Background**: Diffusion models based on rectified flow (e.g., Stable Diffusion 3, Flux) have made breakthrough progress in image generation quality, but still exhibit significant deficiencies in accurately rendering text within images (text rendering). Even the state-of-the-art SD3 model has a text rendering correction rate of only about 32.5%.

**Limitations of Prior Work**: (1) Text rendering requires pixel-level precision—the shape, size, and spacing of each letter must be correct, which places extremely high demands on the sampling accuracy of generative models; (2) existing samplers (such as Euler, DPM-Solver) accumulate discretization errors under a limited number of steps, leading to lost or distorted details in text regions; (3) training specialized text rendering models (e.g., TextDiffuser) requires massive text-image pairs and additional training costs.

**Key Challenge**: Larger sampling steps can reduce discretization errors but increase inference costs, whereas sampling with fewer steps easily leads to errors in regions requiring high precision, such as text. How can sampling accuracy be improved within a limited number of steps without increasing training costs?

**Ours**: Designing a training-free sampler enhancement method specifically aimed at improving the sampling accuracy of rectified flow models in text rendering regions.

**Key Insight**: Drawing inspiration from the "noise-denoise" correction mechanism in Langevin dynamics—deliberately "overshooting" to a noisier position on the ODE sampling trajectory, and then precisely compensating the noise to return to the target position. This "detour" achieves an improvement in local sampling accuracy.

**Core Idea**: Constructing an equivalent Langevin correction step through overshooting and noise compensation, and adaptively applying the correction only to text regions using attention maps to avoid disturbing non-text regions.

## Method

### Overall Architecture
The AMO sampler inserts an additional "overshooting-compensation" operation into each sampling step on top of standard Euler sampling in rectified flow. The detailed pipeline is: (1) standard Euler step progresses from $x_t$ to $x_s$; (2) continues overshooting along the ODE direction to $x_o$ ($o = s + c\epsilon$, where $c$ is the overshooting coefficient); (3) adds precisely calculated noise to bring $x_o$ back to the correct noise level at timestep $s$. The attention modulation module controls the overshooting intensity of each spatial position based on cross-attention maps.

### Key Designs

1. **ODE Overshooting**:

    - **Function**: Continuing to progress to a further timestep along the velocity field direction after completing the standard Euler step.
    - **Mechanism**: Let the current timestep be $t$ and the next step be $s$ ($s < t$). The standard Euler step yields $x_s = x_t + (s-t) v_\theta(x_t, t)$. The overshooting step then continues to progress to $x_o = x_s + c\epsilon \cdot v_\theta(x_s, s)$, where $o = s + c\epsilon$ is the overshooting target time, and $c > 0$ is the overshooting coefficient.
    - **Design Motivation**: Overshooting temporarily deviates the sample from the ODE trajectory into a "noisier" region, creating space for subsequent noise-compensation correction. Analogous to momentum in optimization, a moderate "overshoot" can help escape local extrema.

2. **Noise Compensation**:

    - **Function**: Restoring the overshot sample precisely back to the noise level of the target timestep $s$.
    - **Mechanism**: In rectified flow, the noise level of the sample at timestep $o$ is $\sigma_o$, while the noise level at the target timestep $s$ is $\sigma_s$. By adding precisely calculated Gaussian noise $\eta \sim \mathcal{N}(0, I)$, $x_o$ is mapped back to $\tilde{x}_s = \alpha_{s|o} x_o + \sigma_{s|o} \eta$, where $\alpha_{s|o}$ and $\sigma_{s|o}$ satisfy the correct noise level matching conditions.
    - **Design Motivation**: Overshooting itself corrupts the sample distribution (deviating it from the correct noise level), and noise compensation precisely corrects it back. The combination of overshooting and compensation is equivalent to a single-step Langevin dynamics correction—first injecting additional noise and then denoising, which has been shown to improve sampling accuracy.
    - **Key Constraint**: $c\epsilon$ must be much smaller than $s$ (the remaining noise level); otherwise, extreme overshooting will cause the sample to degenerate into pure noise.

3. **Attention Modulation**:

    - **Function**: Adaptively controlling the overshooting intensity of each spatial position using text-image cross-attention scores.
    - **Mechanism**: During the denoising process, the cross-attention weights $A \in \mathbb{R}^{H \times W}$ between text tokens and image patches are extracted. For regions highly correlated with text content (high attention scores), stronger overshooting correction is applied; for non-text regions such as backgrounds, overshooting is weakened or disabled. The overshooting coefficient changes from a scalar $c$ to a spatially varying 2D field $c(h, w) = c \cdot \text{softmax}(A(h,w) / \tau)$.
    - **Design Motivation**: The overshooting-compensation operation is a double-edged sword—it improves local precision but also introduces extra randomness. While applying strong correction to text regions is necessary (due to the need for high precision), applying the same correction to non-text regions unnecessarily disturbs the already-sufficient background generation. Attention modulation achieves "on-demand correction".
    - $\tau$ is a temperature parameter that controls the sharpness of attention scores. When $\tau \to 0$, only the highest-attention patches are corrected; when $\tau \to \infty$, it degenerates into uniform correction.

4. **Equivalence to Langevin Dynamics**:

    - **Theoretical Proof**: When the overshooting step size $c\epsilon \to 0$, the limiting form of the overshooting and noise compensation operation is strictly equivalent to the Euler step plus a Langevin correction step.
    - This demonstrates that the AMO sampler can be understood as superimposing an SDE correction term on top of the ODE sampler, which has been theoretically proven to reduce sampling errors.
    - In practice, $c$ does not need to be an extremely small value; a moderate $c$ (e.g., $c=2.0$) is sufficient to achieve good results.

## Key Experimental Results

### Main Results: Text Rendering Correction Rate

| Model | Sampler | Steps | Correction Rate (CR) | Relative Gain | FID | CLIP Score |
|------|--------|------|-----------|---------|-----|------------|
| SD3 | Euler | 20 | 32.5% | — | 24.3 | 0.312 |
| SD3 | **AMO** | 20 | **43.0%** | **+32.3%** | 23.8 | 0.315 |
| Flux | Euler | 20 | 74.0% | — | 18.7 | 0.341 |
| Flux | **AMO** | 20 | **82.5%** | **+11.5%** | 18.4 | 0.344 |
| SD3 | Euler | 50 | 38.2% | — | 22.1 | 0.318 |
| SD3 | **AMO** | 50 | **48.7%** | **+27.5%** | 21.6 | 0.321 |

### Steps vs. Correction Rate

| Steps | SD3 Euler | SD3 AMO | Gain | Flux Euler | Flux AMO | Gain |
|------|-----------|---------|------|------------|----------|------|
| 10 | 24.8% | 35.2% | +41.9% | 62.3% | 73.8% | +18.5% |
| 20 | 32.5% | 43.0% | +32.3% | 74.0% | 82.5% | +11.5% |
| 30 | 35.6% | 45.8% | +28.7% | 77.2% | 84.1% | +8.9% |
| 50 | 38.2% | 48.7% | +27.5% | 79.8% | 85.3% | +6.9% |

### Ablation Study

| Configuration | SD3 CR (20 Steps) | Flux CR (20 Steps) | Description |
|------|-------------|---------------|------|
| Euler Baseline | 32.5% | 74.0% | Standard sampling |
| Overshooting Only (No Noise Compensation) | 0.0% | 0.0% | Distribution destroyed, complete failure |
| Overshooting + Noise Compensation (Uniform) | 41.2% | 81.5% | Effective but disturbs background |
| **Overshooting + Noise Compensation + Attention Modulation**| **43.0%** | **82.5%** | Optimal configuration |
| $c=0.5$ | 37.8% | 77.9% | Insufficient overshooting |
| $c=1.0$ | 40.1% | 80.3% | Moderate effect |
| $c=2.0$ | **43.0%** | **82.5%** | Optimal |
| $c=4.0$ | 39.6% | 78.1% | Degenerated due to excessive overshooting |

### Key Findings
- **Overshooting without compensation completely destroys generation**: Overshooting deviates the sample from the correct noise level, reducing the correction rate to 0%, proving that noise compensation is an indispensable component.
- **Larger gains at fewer steps**: The relative gain is 41.9% at 10 steps, compared to only 27.5% at 50 steps. This indicates that AMO's Langevin correction mainly compensates for discretization errors, which are larger at fewer steps, leading to a more pronounced correction effect.
- **No compromise and slight improvement in FID and CLIP scores**: AMO not only enhances text rendering quality but also slightly improves overall image quality (reducing FID by 0.3-0.5), indicating that the accuracy gains from Langevin correction are global.
- **Optimal overshooting coefficient $c=2.0$**: Too small $c$ (<1.0) results in insufficient correction, whereas too large $c$ (>3.0) introduces excessive randomness. $c=2.0$ achieves the best balance between correction intensity and stability.
- **Additional gains from attention modulation**: Applying attention modulation on top of uniform correction gains an extra 1.8% (SD3) and 1.0% (Flux) CR, proving that spatially adaptive correction avoids unnecessary perturbations to already high-quality regions.
- **Smaller room for improvement in Flux**: The Flux baseline CR is already 74.0% (much higher than SD3's 32.5%), leaving limited room for further improvement, yet it still achieves a significant relative gain of +11.5%.

## Highlights & Insights
- **Completely training-free**: AMO is a pure inference-time method that does not require modifying model weights, adding extra modules, or collecting training data. It can be plug-and-played into any rectified flow model, showing immense engineering value in practical applications.
- **Theoretical elegance**: The theoretical connection that overshooting and noise compensation are equivalent to Langevin correction establishes a seemingly ad-hoc sampling trick upon solid SDE/stochastic analysis theoretical foundations, enhancing the interpretability and reliability of the method.
- **"Free" supervision signals from attention maps**: Leveraging the model's own cross-attention as guidance for spatial correction intensity eliminates the need for extra text detection models or segmentation annotations. Since the attention map itself encodes the information of "where the text is being rendered", this utilization strategy is clever and zero-cost.
- **Relationship with step trade-offs**: AMO allows 20-step sampling to outperform 50-step Euler in text rendering quality (43.0% vs. 38.2%), which means inference cost can be reduced by 60% while maintaining text quality.

## Limitations & Future Work
- The overshooting coefficient $c$ and temperature $\tau$ need to be tuned for different models; currently, there is a lack of an adaptive selection mechanism.
- AMO increases the computational overhead by approximately 1.5 times per sampling step (overshooting step + noise compensation). Although this can be compensated by reducing the total number of steps, it remains a bottleneck under strict latency requirements.
- Attention modulation assumes that cross-attention can accurately pinpoint text regions, but attention can be scattered or inaccurate in complex prompts.
- Its effectiveness has only been validated on text rendering tasks; whether it can improve other generation tasks requiring high precision (e.g., facial details, hand/finger anatomy) remains to be investigated.
- Extension to non-rectified flow architectures (e.g., standard DDPM, EDM) requires re-deriving the equivalence conditions.
- Direct comparison with specialized text rendering models (such as TextDiffuser and GlyphDraw) has not been conducted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DreamText: High Fidelity Scene Text Synthesis](dreamtext_high_fidelity_scene_text_synthesis.md)
- [\[NeurIPS 2025\] Enhancing Training Data Attribution with Representational Optimization](../../NeurIPS2025/llm_pretraining/enhancing_training_data_attribution_with_representational_optimization.md)
- [\[CVPR 2025\] ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)
- [\[CVPR 2025\] MR-PLIP: Multi-Resolution Pathology-Language Pre-training Model with Text-Guided Visual Representation](mr_plip_multi_resolution_pathology.md)
- [\[ICML 2025\] How to Synthesize Text Data without Model Collapse?](../../ICML2025/llm_pretraining/how_to_synthesize_text_data_without_model_collapse.md)

</div>

<!-- RELATED:END -->
