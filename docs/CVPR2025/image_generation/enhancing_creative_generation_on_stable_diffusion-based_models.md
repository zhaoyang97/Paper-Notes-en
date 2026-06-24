---
title: >-
  [Paper Note] Enhancing Creative Generation on Stable Diffusion-based Models
description: >-
  [CVPR 2025][Image Generation][Creative Generation] This paper proposes C3 (Creative Concept Catalyst), a training-free method that enhances creative generation capabilities of Stable Diffusion by selectively amplifying features during the denoising process, and provides selection guidelines for amplification factors based on two primary dimensions of creativity.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Creative Generation"
  - "Stable Diffusion"
  - "Feature Amplification"
  - "Denoising Process"
  - "Training-Free"
date: 2026-05-08
content_hash: 52315540a01a8312
---

# Enhancing Creative Generation on Stable Diffusion-based Models

**Conference**: CVPR 2025  
**arXiv**: [2503.23538](https://arxiv.org/abs/2503.23538)  
**Code**: None  
**Area**: Diffusion Models / Image Generation  
**Keywords**: Creative Generation, Stable Diffusion, Feature Amplification, Denoising Process, Training-Free

## TL;DR
This paper proposes C3 (Creative Concept Catalyst), a training-free method that enhances creative generation capabilities of Stable Diffusion by selectively amplifying features during the denoising process, and provides selection guidelines for amplification factors based on two primary dimensions of creativity.

## Background & Motivation

**Background**: Stable Diffusion and its distilled variants (such as SDXL-Turbo, LCM, etc.) have achieved high fidelity and strong text-to-image alignment in text-to-image generation, becoming mainstream tools for creative content generation. Users typically control the style and degree of creativity of the generated content through meticulously designed prompts.

**Limitations of Prior Work**: Despite high generation quality, the **creative capability is limited**. Adding words like "creative" to the prompt rarely yields the desired creative effect. The models tend to generate "typical" images that conform to the training data distribution, making it difficult to surpass conventions to produce novel, unexpected visual combinations. Existing methods for enhancing creativity mostly require additional training (e.g., fine-tuning model weights) or complex prompt engineering, which are computationally expensive and inflexible.

**Key Challenge**: The training objective of diffusion models is to learn the data distribution and sample from it, which naturally biases generation toward high-probability samples within the distribution. Conversely, creative generation requires deviating from high-probability regions to explore low-probability yet meaningful combinations. Directly increasing noise or randomness leads to quality degradation rather than creativity enhancement.

**Goal**: To enhance the creative generation capability of base Stable Diffusion models without additional training, while maintaining generation quality.

**Key Insight**: The authors analyze the roles of different features during the denoising process of diffusion models and find that certain intermediate features correlate with the degree of creativity. Selectively amplifying these features can drive the generation results away from "typical" regions, yielding more creative outputs.

**Core Idea**: Apply amplification factors to specific layer features of the U-Net during the denoising process, selectively enhancing creativity-related feature channels without modifying model weights or requiring additional training.

## Method

### Overall Architecture
C3 is a plug-and-play inference-time method. During the denoising loop of standard Stable Diffusion, C3 applies amplification operations to the internal features of the U-Net at specific timesteps. The amplification strategy determines different amplification factors based on two dimensions of creativity (novelty and diversity). The final output maintains high image quality while exhibiting richer creative expressions.

### Key Designs

1. **Selective Feature Amplification**:

    - **Function**: Enhance creativity-related feature representations during the denoising process.
    - **Mechanism**: In specific layers of the U-Net (mainly intermediate layers and early decoder layers), apply an amplification factor $\alpha$ to the feature maps along the channel dimension. The amplification operation is defined as $\hat{f} = \alpha \cdot f$, where $f$ represents the original features, and $\alpha > 1$ denotes the amplification coefficient. The amplification is not applied uniformly to all features, but selectively to feature channels closely related to creativity. Specifically, stronger amplification is applied at early denoising timesteps (the stage determining global structure and semantics), and is reduced or bypassed at later timesteps (the stage refining details).
    - **Design Motivation**: The early stage of denoising determines the global layout and conceptual combinations of the image. Amplifying features at this stage encourages the model to explore atypical conceptual combinations. Applying amplification at later stages may cause artifacts, hence the need for a timestep-adaptive strategy.

2. **Bi-dimensional Amplification Guidelines for Creativity**:

    - **Function**: Provide a systematic selection strategy for amplification factors.
    - **Mechanism**: Creativity is decomposed into two primary dimensions—**Novelty** (the degree of deviation of the generated output from common images) and **Diversity** (the degree of variation among multiple generation outputs). To enhance novelty, the generation trajectory must be pushed away from the modes of the data distribution in the feature space. To enhance diversity, the expression space of randomness must be expanded during the sampling process. Different combinations of amplification factors correspond to different creative preferences: high novelty + low diversity produces unique but stylistically consistent creative outputs; low novelty + high diversity generates stylistically diverse but less radical creative outputs.
    - **Design Motivation**: Creativity is a multi-dimensional concept that cannot be controlled by a single parameter. The bi-dimensional framework allows users to flexibly adjust the creative direction based on specific needs (e.g., conceptual design vs. style exploration).

3. **Training-Free Plug-and-Play Design**:

    - **Function**: Enhance creativity without modifying model weights.
    - **Mechanism**: C3 only modifies the forward pass of the U-Net during inference to perform amplification operations on the outputs of specific layers. It requires no gradient backpropagation, no modification of attention weights, and no additional encoders or adapters. The entire method can be implemented in a few lines of code on any Stable Diffusion-based model.
    - **Design Motivation**: Training-free methods incur almost zero computational cost and are naturally compatible with all stable diffusion variants (SD1.5, SDXL, SD Turbo, LCM, etc.), making deployment highly convenient.

### Loss & Training
This method is training-free and only performs feature amplification during the inference phase.

## Key Experimental Results

### Main Results

| Model | Method | Novelty Score | Diversity Score | Image Quality (FID) |
|------|------|-----------|-----------|-------------|
| SD 1.5 | baseline | Baseline | Baseline | Baseline |
| SD 1.5 | + C3 | Significant Improvement | Significant Improvement | Slight Decline |
| SDXL | baseline | Baseline | Baseline | Baseline |
| SDXL | + C3 | Significant Improvement | Significant Improvement | Slight Decline |
| SD Turbo | + C3 | Significant Improvement | Significant Improvement | Slight Decline |

### Ablation Study

| Configuration | Novelty | Quality | Description |
|------|-------|------|------|
| Full C3 | High | Good | Full method |
| Uniform Amplification (no timestep adaptation) | Medium | Poor | Amplification in later timesteps causes artifacts |
| Decoder-only Amplification | Medium | Good | Lacks influence on global semantics |
| Encoder-only Amplification | Low | Good | Limited creativity improvement |

### Key Findings
- The choice of the amplification factor $\alpha$ exhibits a direct trade-off between novelty and quality: excessively large $\alpha$ causes distortion, while excessively small $\alpha$ leads to insufficient creativity.
- Intermediate layer features contribute the most to creativity; amplifying shallow and deep layer features yields limited effects.
- Applying amplification during the early stages of denoising (approximately the first 30-50% of timesteps) yields the best results.
- The method is effective across all tested SD variants, demonstrating its generalizability.

## Highlights & Insights
- **Extreme Simplicity**: Creativity can be enhanced on any Stable Diffusion model using just a few lines of code, requiring no training, no data, and no extra models. This "inference-time intervention" mindset is highly valuable for real-world deployment.
- **Creativity Quantifying Framework**: Decomposing creativity into two independently adjustable dimensions (novelty and diversity) provides a structured framework for evaluating and controlling creative generation.
- **Transferability to Other Generative Tasks**: The concept of feature amplification is not limited to Stable Diffusion and can theoretically be applied to any denoising-based generative model (e.g., DiT, video diffusion) to explore low-probability regions of the generation space.

## Limitations & Future Work
- **Subjectivity of Creativity Evaluation**: Creativity itself is a subjective concept, understood differently by different people. Automated evaluation metrics may not fully capture human perception of creativity.
- **Sensitivity to Amplification Factors**: Different prompts and models may require different amplification factors, and the method lacks an automated parameter selection mechanism.
- **Potential to Produce Incoherent Content**: Excessive amplification may lead to semantically incoherent images, such as irrational object combinations or distorted structures.
- **Quality Degradation**: Although labeled as "slight," the increase in FID indicates that creativity enhancement comes at a certain cost to quality.
- **Future Exploration**: Adaptive selection of amplification factors, balancing creativity and quality using CLIP scores, and extending to video and 3D generation.

## Related Work & Insights
- **vs Prompt Engineering**: While creativity can be guided by modifying text prompts (e.g., using terms like "surreal" or "abstract"), the effects are limited and uncontrollable. C3 directly intervenes at the feature level, offering more precise control.
- **vs LoRA/DreamBooth Fine-tuning**: Fine-tuning methods can alter model styles and creative tendencies but require data and training time. C3 is completely training-free.
- **vs Classifier-Free Guidance (CFG)**: CFG controls generation by adjusting the weights of conditional/unconditional predictions during inference, but it primary affects text alignment rather than creativity. C3 operates in the feature space, affecting a different and complementary dimension.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically investigate the problem of creativity enhancement in diffusion models, proposing a simple yet effective feature amplification idea.
- Experimental Thoroughness: ⭐⭐⭐ Validated across multiple SD variants, but lacks large-scale human evaluation and comparisons with more baselines.
- Writing Quality: ⭐⭐⭐⭐ The bi-dimensional creativity framework is clear, though some technical details in the method section could be further elaborated.
- Value: ⭐⭐⭐⭐ The training-free, plug-and-play solution yields high practicality, directly benefiting creative application scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Redefining <Creative> in Dictionary: Towards an Enhanced Semantic Understanding of Creative Generation](redefining_creative_in_dictionary_towards_an_enhanced_semantic_understanding_of_.md)
- [\[NeurIPS 2025\] Training-Free Constrained Generation with Stable Diffusion Models](../../NeurIPS2025/image_generation/training-free_constrained_generation_with_stable_diffusion_models.md)
- [\[CVPR 2025\] Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability](not_all_parameters_matter_masking_diffusion_models_for_enhancing_generation_abil.md)
- [\[CVPR 2025\] Enhancing Privacy-Utility Trade-offs to Mitigate Memorization in Diffusion Models](enhancing_privacy-utility_trade-offs_to_mitigate_memorization_in_diffusion_model.md)
- [\[CVPR 2025\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)

</div>

<!-- RELATED:END -->
