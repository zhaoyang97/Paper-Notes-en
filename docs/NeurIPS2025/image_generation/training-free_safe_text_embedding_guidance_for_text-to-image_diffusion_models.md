---
title: >-
  [Paper Note] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models
description: >-
  [NeurIPS 2025][Image Generation][Safe generation] This paper proposes Safe Text embedding Guidance (STG), a training-free approach for safe text-to-image generation that dynamically adjusts text embeddings during diffusion sampling based on a safety function evaluated on the expected denoised image. STG effectively removes unsafe content while maximally preserving the original semantic intent.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Safe generation"
  - "text embedding guidance"
  - "diffusion models"
  - "training-free methods"
  - "content safety"
date: 2026-05-08
content_hash: 9c9bd475248a475e
---

# Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.24012](https://arxiv.org/abs/2510.24012)  
**Code**: [GitHub](https://github.com/aailab-kaist/STG)  
**Area**: Diffusion Models / Image Generation
**Keywords**: Safe generation, text embedding guidance, diffusion models, training-free methods, content safety

## TL;DR
This paper proposes Safe Text embedding Guidance (STG), a training-free approach for safe text-to-image generation that dynamically adjusts text embeddings during diffusion sampling based on a safety function evaluated on the expected denoised image. STG effectively removes unsafe content while maximally preserving the original semantic intent.

## Background & Motivation

Text-to-image diffusion models have achieved remarkable progress by training on large-scale web-crawled datasets; however, such datasets inevitably contain inappropriate or biased content, causing models to generate unsafe outputs—including nudity, violence, or copyright-infringing material—when prompted with adversarial text. The definition of "safety" varies across cultural contexts and individual perception, necessitating flexible and adaptable safe generation strategies.

Existing safe generation methods fall into two broad categories, each with distinct limitations:

**Training-based methods** (e.g., ESD, DUO) fine-tune model weights to "unlearn" unsafe concepts, but require additional safety-annotated data and substantial computational resources, and risk degrading the model's original generation capability.

**Training-free methods** (e.g., SLD, UCE, SAFREE) manipulate inputs or intermediate representations at inference time, but typically do not directly leverage intermediate samples produced by the diffusion model to guide the safety mechanism, and lack a clear theoretical foundation for understanding how their modifications affect the original model distribution.

The root cause of the tension is: how to exploit intermediate sample information produced during diffusion to ensure safety, without relying on additional training, while providing theoretical guarantees that generation quality is preserved. STG's starting point is that unsafe images typically originate from text prompts containing unsafe concepts; therefore, applying safety guidance directly in the text embedding space is more robust than operating in the data space.

## Method

### Overall Architecture
At each diffusion sampling step, STG first estimates the expected denoised image $\bar{\mathbf{x}}_0$ from the current noisy image $\mathbf{x}_t$ via the Tweedie formula, then evaluates a safety score using the safety function $g$ on this expected image, and finally applies the gradient of the safety function with respect to the text embedding $\mathbf{c}$ to update the embedding, steering the generation toward safer outputs.

### Key Designs

1. **Safe Guidance (SG) Theoretical Framework**:

    - Safety conditioning is modeled as the Bayesian conditional probability $q_t(o=1|\mathbf{x}_t, \mathbf{c})$.
    - The score function under the safety condition decomposes into the original text-conditioned score plus a safety guidance term: $\nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t|\mathbf{c}, o=1) = \nabla_{\mathbf{x}_t} \log q_t(\mathbf{x}_t|\mathbf{c}) + \nabla_{\mathbf{x}_t} \log q_t(o=1|\mathbf{x}_t, \mathbf{c})$
    - Design motivation: directly training a time-dependent safety classifier is prohibitively expensive, motivating a training-free alternative.

2. **From Safe Data Guidance (SDG) to Safe Text Guidance (STG)**:

    - SDG applies safety guidance in the data space $\mathbf{x}_t$ (analogous to Universal Guidance), but relies on the assumption that the safety function $g$ is exactly proportional to the true safety probability; mismatches in functional form introduce bias.
    - STG instead performs gradient ascent in the text embedding space: $\mathbf{c} \leftarrow \mathbf{c} + \rho \nabla_{\mathbf{c}} g_t(\mathbf{x}_t, \mathbf{c})$
    - Core formula: $\mathbf{s}_{\text{STG}}(\mathbf{x}_t, \mathbf{c}, t) = \mathbf{s}_\theta\left(\mathbf{x}_t, \mathbf{c} + \rho \nabla_{\mathbf{c}} g\left(\frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t + (1-\bar{\alpha}_t)\mathbf{s}_\theta(\mathbf{x}_t, \mathbf{c}, t))\right), t\right)$
    - Design motivation: operating in the text embedding space simultaneously accounts for the underlying model likelihood and the safety direction, avoiding the mode collapse issues associated with data-space guidance.

3. **Theoretical Guarantee (Theorem 1)**:

    - The equivalent effect of STG in the data space decomposes into the original text-conditioned score, a safety guidance term, and a higher-order error $O(\rho^2)$.
    - The implicit safety probability is $q_t^{\text{STG}}(o=1|\mathbf{x}_t, \mathbf{c}) \propto \exp(\rho \nabla_\mathbf{c} g_t \cdot \nabla_\mathbf{c} \log q_t(\mathbf{x}_t|\mathbf{c}))$, meaning the safety probability is determined by the alignment between the safety function gradient and the model likelihood gradient, naturally balancing safety and generation quality.

4. **Practical Design**:

    - Update threshold $\tau$: determines whether guidance is applied at the current step based on the estimated safety value.
    - Update step ratio $\gamma$: controls guidance frequency, trading off efficiency and safety.
    - Supports FP16 inference, substantially reducing runtime and GPU memory usage.

### Instantiation of Safety Function $g$
- **Nudity detection**: $g$ is the negative sum of NudeNet detector confidence scores for nudity-related labels.
- **Violence detection**: $g$ is the negative CLIP score between the generated image and violence-related text.
- **Artistic style removal**: $g$ is the difference between the CLIP score of the image with the text "art" and the CLIP score with the target artist's name.

## Key Experimental Results

### Main Results: Generation Quality on COCO

| Method | FID ↓ | CLIP ↑ | Notes |
|--------|-------|--------|-------|
| Base (SD v1.4) | 23.22 | 31.96 | Baseline |
| ESD (training-based) | 22.85 | 30.02 | Notable CLIP drop |
| DUO (training-based) | 23.27 | 31.90 | Close to baseline |
| SAFREE (training-free) | 28.39 | 30.27 | Severe FID degradation |
| SDG (training-free) | 26.90 | 29.97 | Similar degradation |
| **STG (Ours)** | **22.00** | **31.14** | FID even surpasses baseline |

### Ablation Study: Hyperparameter Sensitivity

| Configuration | DSR ↑ | PP ↑ | Notes |
|---------------|-------|------|-------|
| $\rho=0.5, \tau=0.15$ | 0.60 | 0.94 | Weak guidance |
| $\rho=2.0, \tau=0.15$ | 0.79 | 0.90 | Balanced configuration |
| $\rho=2.0, \tau=0.40$ | 0.88 | 0.84 | Strong guidance |
| $\rho=2.0, \tau=0.80$ | 0.92 | 0.84 | Strongest guidance |

### Cross-Architecture Generalization (Ring-A-Bell Violence)

| Model | Base DSR | STG DSR ($\tau=0.16$) | FID-1K Change |
|-------|----------|------------------------|---------------|
| FLUX | 0.11 | 0.70 | 56.58→57.77 |
| SDXL | 0.04 | 0.77 | 48.97→49.44 |
| SD3 | 0.12 | 0.68 | 53.70→54.91 |
| LCM | 0.02 | 0.80 | 60.87→62.32 |

### Key Findings
- STG achieves lower FID than the base model on safe COCO prompts, suggesting that the likelihood-preserving term in the safety guidance contributes positively to generation quality.
- STG can be seamlessly combined with training-based methods (e.g., DUO) to further improve defense success rates in violence scenarios.
- 2D toy experiments clearly demonstrate STG's robustness to safety function shape mismatch compared to SDG.

## Highlights & Insights
- **Elegant integration of theory and practice**: Theorem 1 reveals that the safety probability implicitly defined by STG naturally accounts for model likelihood preservation, which is the fundamental reason for its superiority over SDG.
- **Strong framework generality**: The same framework applies to multiple safety scenarios—nudity, violence, artistic style removal, and bias mitigation—requiring only a replacement of the safety function $g$.

## Limitations & Future Work
- Computational overhead arises from gradient computation for updating text embeddings; while FP16 and the threshold mechanism help, inference remains slower than the baseline.
- The selection of the safety function $g$ depends on external classifiers (e.g., NudeNet, CLIP), whose quality directly affects the guidance effectiveness.
- Defense against white-box attacks (e.g., Concept Inversion) still requires combination with training-based methods.

## Related Work & Insights
- **vs. SLD**: SLD applies CFG-style guidance conditioned on unsafe text without directly leveraging intermediate samples; STG feeds safety signals back through the expected denoised image.
- **vs. SAFREE**: SAFREE filters in the unsafe token subspace, decoupled from the diffusion state; STG directly incorporates diffusion intermediate states into the safety guidance.
- **vs. DUO**: DUO is a training-based method with limited effectiveness on diverse categories such as violence; STG offers greater flexibility through test-time CLIP guidance.

## Rating
- Novelty: ⭐⭐⭐⭐ The safety guidance perspective in text embedding space is novel, with rigorous theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers nudity, violence, and style removal; validated across multiple architectures and both black-box and white-box attacks.
- Writing Quality: ⭐⭐⭐⭐⭐ The progressive logic from SG → SDG → STG is clearly articulated.
- Value: ⭐⭐⭐⭐ High practical value; plug-and-play compatible with various diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Adaptive Text Embedding for Text-to-Image Diffusion Models](diffusion_adaptive_text_embedding_for_texttoimage_diffusion.md)
- [\[NeurIPS 2025\] Safe and Stable Control via Lyapunov-Guided Diffusion Models](safe_and_stable_control_via_lyapunov-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models](prompt-based_safety_guidance_is_ineffective_for_unlearned_text-to-image_diffusio.md)
- [\[NeurIPS 2025\] Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models](mitigating_sexual_content_generation_via_embedding_distortion_in_text-conditione.md)
- [\[ICCV 2025\] Addressing Text Embedding Leakage in Diffusion-Based Image Editing](../../ICCV2025/image_generation/addressing_text_embedding_leakage_in_diffusion-based_image_editing.md)

</div>

<!-- RELATED:END -->
