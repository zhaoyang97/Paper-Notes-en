---
title: >-
  [Paper Note] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Implicit Bias Injection] This paper proposes the Implicit Bias Injection Attack framework (IBI-Attacks). By pre-computing a universal bias direction vector in the text embedding space and dynamically adjusting it according to different user inputs using an adaptive feature selection module, this approach implants implicit biases (e.g., emotions, cultural inclinations) into pre-trained text-to-image diffusion models in a plug-and-play manner. This…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Implicit Bias Injection"
  - "Text-to-Image Diffusion Models"
  - "Adversarial Attacks"
  - "Embedding Space Manipulation"
  - "Adaptive Feature Selection"
date: 2026-05-08
content_hash: f0de11e27531b8e0
---

# Implicit Bias Injection Attacks against Text-to-Image Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2504.01819](https://arxiv.org/abs/2504.01819)  
**Code**: [https://github.com/Hannah1102/IBI-attacks](https://github.com/Hannah1102/IBI-attacks)  
**Area**: Image Generation  
**Keywords**: Implicit Bias Injection, Text-to-Image Diffusion Models, Adversarial Attacks, Embedding Space Manipulation, Adaptive Feature Selection

## TL;DR
This paper proposes the Implicit Bias Injection Attack framework (IBI-Attacks). By pre-computing a universal bias direction vector in the text embedding space and dynamically adjusting it according to different user inputs using an adaptive feature selection module, this approach implants implicit biases (e.g., emotions, cultural inclinations) into pre-trained text-to-image diffusion models in a plug-and-play manner. This framework preserves the original semantics of the generated content and achieves over an 80% attack success rate, while only 35.8% of the attacks are perceived by human evaluators.

## Background & Motivation

1. **Background**: Text-to-image diffusion models (such as Stable Diffusion) are widely used, and AI-generated images have permeated daily life. Prior research has shown that these models exhibit inherent biases (such as gender and skin tone biases), but they primarily focus on explicit biases, which possess clearly recognizable visual patterns.

2. **Limitations of Prior Work**: Existing bias exploitation methods (such as Backdooring Bias) can only inject explicit biases (such as specific skin tones). These methods require computationally expensive model fine-tuning and are easily detected. Furthermore, because these biases are expressed through fixed visual features, they lack diversity and stealth.

3. **Key Challenge**: Explicit biases have fixed features, are easy to detect, and present singular expressions. Conversely, real-world implicit biases—such as emotions, cultural stereotypes, and religious inclinations—pose greater danger since they lack fixed visual patterns. They can manifest through diverse semantic forms like facial expressions, body gestures, backgrounds, and group behaviors, making them harder to detect yet more likely to persistently influence user perception.

4. **Goal**: Design an attack framework capable of injecting implicit biases, which requires: (1) diverse and stealthy bias expressions; (2) no user input modifications and no model retraining; (3) generalizability across different prompts.

5. **Key Insight**: The authors observe that in the text embedding space, the average difference vector between neutral prompts and biased prompts already encodes multiple semantic expressions and possesses the capability to generalize to various inputs.

6. **Core Idea**: Use LLMs to generate neutral-biased prompt pairs, compute the average bias direction vector in the embedding space, and train a lightweight adaptive feature selection module to dynamically adjust this direction based on the user input. This achieves plug-and-play implicit bias injection without modifying the model.

## Method

### Overall Architecture
Input: User text prompt. Output: Generated image with specified bias (e.g., negative emotion). Pipeline: (1) Pre-computation phase: Use LLM to generate N neutral prompts and their biased rewrites, then calculate the average difference vector $v^{\text{diff}}$ after encoding; (2) Training phase: Train an adaptive feature selection module to learn how to dynamically scale $v^{\text{diff}}$ based on the input prompt; (3) Inference phase: Embed the trained module behind the text encoder to modify the user prompt's embedding before feeding it into the diffusion model.

### Key Designs

1. **Directional Vector Generation**:

    - **Function**: Find a direction in the embedding space that represents the specified bias.
    - **Mechanism**: Use ChatGPT-4 to generate 200 neutral everyday scene prompts $X_{\text{neu}}$ and rewrite them into $X_{\text{bias}}$ according to a designated bias (e.g., "negative emotion"). The rewriting rules are restricted to only adding appropriate adjectives to minimize bias-unrelated structural changes. Map both sets of prompts into the embedding space $v_i^{\text{neu}}, v_i^{\text{bias}} \in \mathbb{R}^{D \times L}$ (D=1024, L=77) using a pre-trained encoder $\varphi$, and compute the average difference vector $v^{\text{diff}} = \frac{1}{N}\sum_{i=1}^{N}(v_i^{\text{bias}} - v_i^{\text{neu}})$.
    - **Design Motivation**: Restricting rewriting to only adding adjectives avoids interference from syntactic changes in the embedding. A single average direction vector already encodes multiple semantic expressions (expressions, gestures, background, etc.), which is a valuable finding.

2. **Adaptive Feature Selection**:

    - **Function**: Dynamically adjust the weights of each dimension of the fixed bias direction vector based on the specific content of the user input.
    - **Mechanism**: Inspired by SENet, a lightweight module is designed. Global average pooling (Avg) is alternately performed on the token dimension L and embedding dimension D of the text embedding. After compressing one dimension, a two-layer MLP$_\theta$ is used to learn the attention weight of the other dimension. Formula: $\tilde{v}^{\text{diff}} = \text{MLP}_\theta(\text{Avg}(v^{\text{user}})) \odot v^{\text{diff}}$, and the final bias embedding is $\tilde{v}^{\text{bias}} = v^{\text{user}} + \tilde{v}^{\text{diff}}$. The training loss is $L = \frac{1}{N}\sum\|v_i^{\text{diff}} - \text{MLP}_\theta(\text{Avg}(v_i^{\text{neu}})) \odot v^{\text{diff}}\|^2$.
    - **Design Motivation**: Directly adding a fixed bias vector to all inputs leads to over-modification (semantic destruction) for some prompts and under-modification (insufficient bias) for others. The adaptive module selectively activates relevant feature dimensions in the bias direction based on the context.

3. **Plug-and-Play Inference Deployment**:

    - **Function**: Seamlessly embed the attack module into any pre-trained T2I model.
    - **Mechanism**: The trained adaptive module is inserted directly between the text encoder and the diffusion backbone, requiring no access to model parameters/architecture, no user input modification, and no runtime LLM queries. The attack can be selectively deployed to specific user groups (e.g., based on specific IP addresses), offering strong stealthiness.
    - **Design Motivation**: Compared with methods requiring prompt modification or model fine-tuning, the plug-and-play approach is harder to detect and incurs extremely low deployment costs.

### Loss & Training
- Training data: 200 neutral-biased prompt pairs generated by LLM
- Trained for only 50 epochs using the Adam optimizer with lr=0.001
- Extremely lightweight—only trains a small MLP module

## Key Experimental Results

### Main Results
Evaluated on 200K person-related captions from the COCO dataset using Stable Diffusion 2.1:

| Bias Type | Method | Success Rate | CLIP_txt-img↑ | CLIP_img-img↑ | SSIM↑ | FID↓ |
|----------|------|--------|--------------|--------------|-------|------|
| Negative | Original | 14.4% | 0.3633 | 1.000 | 1.000 | 39.30 |
| Negative | Explicit (modified prompt) | 89.1% | 0.3458 | 0.735 | 0.522 | 40.09 |
| Negative | IBI w/o adapt | 82.2% | 0.3596 | 0.814 | 0.588 | 39.22 |
| Negative | **IBI (ours)** | **80.2%** | **0.3637** | **0.879** | **0.699** | **39.14** |
| Positive | IBI (ours) | 83.7% | 0.3602 | 0.825 | 0.604 | 39.50 |

### Ablation Study

| Configuration | Attack Success Rate | CLIP_img-img | SSIM | Description |
|------|-----------|-------------|------|------|
| Direct prompt modification | 89.1% | 0.735 | 0.522 | Highest success rate but extremely poor stealthiness |
| Fixed direction vector (w/o adapt) | 82.2% | 0.814 | 0.588 | Feasible but causes significant semantic degradation |
| Adaptive module (full) | 80.2% | 0.879 | 0.699 | Success rate drops slightly, but stealthiness is significantly improved |

### Key Findings
- While maintaining an 80%+ attack success rate, the adaptive module improves SSIM from 0.588 to 0.699 and CLIP image similarity from 0.814 to 0.879.
- **Outstanding zero-shot transferability**: The bias module trained on human scenes directly transfers to animal scenes (95.1% success rate) and natural scenes (89.2% success rate).
- Human evaluation: Only 35.8% of the 24 participants could identify the attacked images as anomalous (nearly identical to 35.7% for clean images), demonstrating extremely high stealthiness.
- Diverse forms of bias expression—facial expressions, body gestures, environmental atmospheres, and image styles can all convey the bias.

## Highlights & Insights
- **The discovery of "bias direction in the embedding space"**: A single average difference vector can encode multiple semantic expressions and possesses generalization capability, revealing structural characteristics of the T2I model's embedding space. This discovery itself holds academic value and can in turn be utilized for bias detection and mitigation.
- **Extremely lightweight attack module**: Effective attacks require training on only 200 prompt pairs for 50 epochs, indicating a very low barrier to bias injection, which highlights the urgent need for security auditing of T2I models.
- **Exquisite trade-off in stealthy design**: The balance between bias success rate and semantic preservation is achieved through a SENet-style attention mechanism.

## Limitations & Future Work
- Bias evaluation relies heavily on MLLM (LLaVA) judgments, which may result in misjudgments for extremely subtle biases.
- Validated only on SD v2.1; the attack effectiveness on newer models (e.g., SDXL, DALL-E 3) remains unknown.
- The rewriting strategy is restricted to adding adjectives; more complex biases (e.g., narrative biases) might require different direction modeling.
- Positive direction: Conversely utilizing the bias direction vector for bias detection and debiasing.
- Defense measures: Such attacks could be defended against by detecting abnormal offsets in the embedding space.

## Related Work & Insights
- **vs Backdooring Bias**: Requires expensive model fine-tuning, can only handle explicit biases (fixed visual features), and is easy to detect. Conversely, the proposed method is training-free (no model fine-tuning required), plug-and-play, supports implicit biases, and offers diverse expressions.
- **vs Bias mitigation methods (e.g., Fair Diffusion)**: These methods also modify embeddings, but they compute fixed, single-semantic directions. This work reveals the existence of multi-semantic directions and incorporates dynamic adjustment.
- From an attacker's perspective, this study reveals the vulnerability of T2I models in the embedding space, offering important insights for AI security. The attack methodology can provide a threat model for defense design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes the concept of implicit bias injection for the first time; the discovery of the bias direction in the embedding space is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative evaluation, zero-shot transfer, and human evaluation are comprehensive, but lacks validation on non-SD models.
- Writing Quality: ⭐⭐⭐⭐ Clear threat model, and intuitive method pipeline diagram.
- Value: ⭐⭐⭐⭐⭐ Holds significant warning significance for T2I model security research; the low barrier to attack is alarming.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Bias for Action: Video Implicit Neural Representations with Bias Modulation](bias_for_action_video_implicit_neural_representations_with_bias_modulation.md)
- [\[CVPR 2025\] SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-Image Diffusion Models](sleepermark_towards_robust_watermark_against_fine-tuning_text-to-image_diffusion.md)
- [\[CVPR 2025\] Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability](dissecting_and_mitigating_diffusion_bias_via_mechanistic_interpretability.md)
- [\[CVPR 2025\] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models](nearly_zero-cost_protection_against_mimicry_by_personalized_diffusion_models.md)
- [\[CVPR 2025\] Not Just Text: Uncovering Vision Modality Typographic Threats in Image Generation Models](not_just_text_uncovering_vision_modality_typographic_threats_in_image_generation.md)

</div>

<!-- RELATED:END -->
