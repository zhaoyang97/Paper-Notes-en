---
title: >-
  [Paper Note] Lego: Learning to Disentangle and Invert Personalized Concepts Beyond Object Appearance in Text-to-Image Diffusion Models
description: >-
  [ECCV 2024][Image Generation][Concept Disentanglement] The Lego method is proposed to achieve the disentanglement and inversion of personalized concepts beyond appearance (such as adjectives and verbs) through subject separation and context loss for personalized content generation in diffusion models.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Concept Disentanglement"
  - "Textual Inversion"
  - "Personalized Generation"
  - "Diffusion Models"
  - "Non-Appearance Concepts"
date: 2026-05-08
content_hash: 5aa30c631c1e1999
---

# Lego: Learning to Disentangle and Invert Personalized Concepts Beyond Object Appearance in Text-to-Image Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2311.13833](https://arxiv.org/abs/2311.13833)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Concept Disentanglement, Textual Inversion, Personalized Generation, Diffusion Models, Non-Appearance Concepts

## TL;DR

The Lego method is proposed to achieve the disentanglement and inversion of personalized concepts beyond appearance (such as adjectives and verbs) through subject separation and context loss for personalized content generation in diffusion models.

## Background & Motivation

Personalized content generation is a crucial research direction in the field of text-to-image generation. Methods such as Textual Inversion and DreamBooth have demonstrated the ability to learn and reproduce specific concepts from a small number of exemplar images. However, existing methods primarily focus on the inversion of **object appearance**, such as specific object styles or human faces.

In reality, there exists a plethora of personalized concepts beyond object appearance, including adjectives (e.g., "worn-out", "shiny") and verbs (e.g., "jumping", "spinning"). The inversion of these concepts faces two unique challenges:

(1) **Concept-Subject Entanglement**: The visual representations of adjectives and verbs are inherently tied to their subjects. For instance, the "jumping" action and the appearance of the "cat" in "a jumping cat" are entangled. Existing inversion methods inevitably leak the cat's appearance into the concept embedding when learning the concept "jumping".

(2) **Description Complexity Beyond a Single Token**: Unlike object concepts that can be represented by a single special token, adjective and verb concepts typically require multiple embedding tokens to fully capture their semantic meaning.

The core contribution of this work is the proposed Lego method, which addresses these two challenges through two key techniques: Subject Separation and Context Loss.

## Method

### Overall Architecture

The Lego pipeline consists of three phases: (1) Subject Separation: disentangling the concept and the subject from the input image; (2) Multi-embedding Inversion: learning multiple token embeddings to represent the target concept; (3) Context-guided Generation: utilizing the learned concept embeddings to generate content in new scenes and with new subjects.

### Key Designs

1. **Subject Separation**:
    - **Function**: Disentangle the target concept from its associated subject appearance.
    - **Mechanism**: A pre-trained segmentation model or a manual mask is utilized to divide the image into concept-related regions and subject regions. During the inversion optimization process, the subject regions are constrained using known text descriptions (e.g., "a cat") to ensure that subject appearance information does not leak into the concept embedding. Only the concept embedding is allowed to learn visual information from the non-subject parts.
    - **Design Motivation**: Direct optimization on the complete image leads to entanglement between the concept and the subject, as the optimization objective drives the embedding to encode all visual information.

2. **Context Loss**:
    - **Function**: Guide the learning of multi-embedding concepts.
    - **Mechanism**: A context-aware loss function is designed. During the denoising process, it not only reconstructs the original image but also requires the concept embedding to produce semantically consistent results when applied to a new context (e.g., different subjects). Specifically, this is achieved by combining the concept embedding with random subject descriptions during training and calculating the semantic consistency between the generated results and the target concept.
    - **Design Motivation**: Inversion optimization without context constraints easily overfits to the training images, leading to a loss of generalization capability. The context loss encourages the concept embedding to learn transferable semantic information.

3. **Multi-embedding Representation**:
    - **Function**: Fully represent complex concepts using multiple token embeddings.
    - **Mechanism**: Instead of a single token, the target concept is represented as a sequence of multiple learnable tokens. Optimizing multiple tokens jointly allows the encoding of richer semantic information. The number of embeddings is selected based on performance on the validation set.
    - **Design Motivation**: The semantic complexity of adjective and verb concepts typically exceeds that of object concepts, rendering a single token insufficient to fully capture their meaning.

### Loss & Training

Training utilizes the standard diffusion denoising loss, but incorporates the context loss as a regularization term:
- Reconstruction Loss: The standard denoising loss on the original image.
- Context Loss: Combines the concept embedding with random subjects to constrain the semantic consistency of the concept across different contexts.
- Subject Separation Constraint: Uses fixed text descriptions for subject areas to prevent the leakage of subject information.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Textual Inversion | Gain |
|--------|------|------|----------|------|
| User Study | Concept Fidelity Preference | >70% | <30% | over +40% |
| VQA Evaluation | Concept Alignment | Better | Baseline | Significant Improvement |
| Subject Fidelity | CLIP-I | Higher | Subject leakage observed | No leakage |
| Diversity | Generation Diversity | Maintained | Overfitting | Better |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| W/o Subject Separation | Severe subject leakage | Concept embedding contains appearance information |
| W/o Context Loss | Poor generalization | Overfitting to training images |
| Single embedding | Insufficient concept representation | Complex concepts cannot be fully encoded |
| Full Lego | Optimal | Three components complement each other |

### Key Findings

- Subject separation is the key to resolving appearance leakage, which is simple yet highly effective.
- Context loss significantly enhances the generalization ability of concept inversion in new contexts.
- Multi-embedding representation is crucial for complex concepts such as adjectives and verbs.
- In the user study, Lego achieved a preference rate of over 70%, demonstrating the practical effectiveness of the method.

## Highlights & Insights

- Novel problem definition: Focusing on concept inversion beyond appearance opens up a new research direction.
- Clear methodological design: Tailored solutions are proposed to target the two main challenges (entanglement and complexity).
- The concept of subject separation is highly versatile and can be extended to other scenarios requiring disentanglement.
- User study results strongly support the effectiveness of the proposed method.

## Limitations & Future Work

- Subject separation relies on accurate segmentation or masks, which can be challenging in scenarios with ambiguous concept-subject boundaries.
- The random subject sampling strategy in context loss may not be sufficiently efficient.
- The method is only validated on Stable Diffusion, and its applicability to other diffusion models (such as DALL-E or Imagen) remains unknown.
- For highly abstract concepts (e.g., "elegant", "melancholy"), the effectiveness of the method might be limited.
- Adaptive strategies for selecting the number of embeddings can be explored in future work.

## Related Work & Insights

- **Textual Inversion**: Pioneered the direction of personalized concept inversion, but is limited to appearance concepts.
- **DreamBooth**: Achieves personalization by fine-tuning the entire model, yet still suffers from entanglement issues.
- **Custom Diffusion**: Jointly optimizes text embeddings and a subset of model parameters.
- Insight: Concept disentanglement is the core challenge in personalized generation. Lego's subject separation concept can be generalized to a broader range of concept manipulation scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The problem definition of non-appearance concept inversion is novel, and the method design is ingenious.
- Experimental Thoroughness: ⭐⭐⭐ The user study and VQA evaluation are relatively thorough, but more quantitative benchmarks are lacking.
- Writing Quality: ⭐⭐⭐⭐ The problem motivation is clearly articulated, and the method description is intuitive.
- Value: ⭐⭐⭐⭐ It opens up a new direction for personalized generation and holds practical application value for creative design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Powerful and Flexible: Personalized Text-to-Image Generation via Reinforcement Learning](powerful_and_flexible_personalized_texttoimage_generation_vi.md)
- [\[ECCV 2024\] Enhancing Diffusion Models with Text-Encoder Reinforcement Learning](enhancing_diffusion_models_with_text-encoder_reinforcement_learning.md)
- [\[ECCV 2024\] LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning](lego_learning_egocentric_action_frame_generation_via_vi.md)
- [\[ECCV 2024\] OMG: Occlusion-friendly Personalized Multi-concept Generation in Diffusion Models](omg_occlusion-friendly_personalized_multi-concept_generation_in_diffusion_models.md)
- [\[ECCV 2024\] Pixel-Aware Stable Diffusion for Realistic Image Super-Resolution and Personalized Stylization](pixel-aware_stable_diffusion_for_realistic_image_super-resolution_and_personaliz.md)

</div>

<!-- RELATED:END -->
