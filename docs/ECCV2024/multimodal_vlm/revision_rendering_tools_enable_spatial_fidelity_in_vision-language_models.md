---
title: >-
  [Paper Note] REVISION: Rendering Tools Enable Spatial Fidelity in Vision-Language Models
description: >-
  [ECCV 2024][Multimodal VLM][Spatial relationship reasoning] The REVISION framework is proposed to leverage Blender 3D rendering to generate spatially accurate synthetic images. These images guide text-to-image (T2I) models in a training-free manner to generate spatially consistent images. It also implements the RevQA benchmark to evaluate the spatial reasoning capabilities of MLLMs.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Spatial relationship reasoning"
  - "text-to-image generation"
  - "3D rendering"
  - "multimodal large language model"
  - "benchmark"
date: 2026-05-08
content_hash: cff69372002c5202
---

# REVISION: Rendering Tools Enable Spatial Fidelity in Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2408.02231](https://arxiv.org/abs/2408.02231)  
**Code**: [https://github.com/AgneetchatterjeeASU/REVISION](https://github.com/AgneetchatterjeeASU/REVISION)  
**Area**: Multimodal VLM  
**Keywords**: Spatial relationship reasoning, text-to-image generation, 3D rendering, multimodal large language model, benchmark

## TL;DR
The REVISION framework is proposed to leverage Blender 3D rendering to generate spatially accurate synthetic images. These images guide text-to-image (T2I) models in a training-free manner to generate spatially consistent images. It also implements the RevQA benchmark to evaluate the spatial reasoning capabilities of MLLMs.

## Background & Motivation

**Background**: Text-to-image (T2I) models (such as Stable Diffusion, DALL-E) and multimodal large language models (MLLMs) have made tremendous progress in image generation and visual understanding, but they suffer from severe deficiencies in understanding and generating spatial relationships.

**Limitations of Prior Work**:
- Generated images from T2I models often fail to correctly reflect spatial relationships (such as "left of", "on top of", "in front of") described in input prompts.
- Existing improvement methods either require massive amounts of training data (e.g., SPRIGHT requires relabeling 6 million images) or rely on bounding box annotations (e.g., Layout Guidance), which are highly expensive.
- MLLMs exhibit fragile performance under complex spatial reasoning (involving negation, conjunction, and disjunction).

**Key Challenge**: Graphical rendering tools (such as Blender) can place objects precisely but lack photorealism, while T2I models produce high-quality outputs but possess poor spatial accuracy. How can we leverage the strengths of both?

**Key Insight**: Utilize Blender to render spatially accurate reference images, and inject spatial information into the generation process of existing T2I models through a training-free image guidance mechanism. Core Idea: **Guiding the spatial fidelity of generative models using the deterministic spatial precision of rendering tools**.

## Method

### Overall Architecture
REVISION is a Blender-based image rendering pipeline containing four core components: Asset Library, Coordinate Generator, Scene Synthesizer, and Position Diversifier. Given a text prompt, it parses the objects and spatial relationships, renders a spatially accurate reference image in Blender, and then uses this image to guide the generation of the T2I model.

### Key Designs

1. **Asset Library**:

    - Contains 101 categories of 3D object models (80 of which are from MS-COCO), totaling 410 3D models.
    - Each category is associated with 3-5 royalty-free 3D models to provide texture and shape diversity.
    - All models are uniformly scaled to fit within a 1m cube to ensure visibility.
    - Includes 3 background panoramas (indoor, outdoor, white).
    - Design Motivation: A sufficiently rich asset library is required to cover common visual concepts.

2. **Coordinate Generator**:

    - Deterministically generates 3D coordinates for objects and cameras based on the spatial relationships parsed from the prompt.
    - Supports 11 spatial relationships across 4 categories: horizontal (left/right), vertical (above/below), proximity (next to), and depth (in front of/behind).
    - X-axis = depth, Y-axis = horizontal, Z-axis = vertical; object coordinates are constrained within the $[-1\text{m}, 1\text{m}]$ range.
    - The camera is fixed at x=5m, facing the origin; z=2.5m for depth relationships, and z=1.5m for others.
    - Design Motivation: Deterministic coordinate generation guarantees the absolute correctness of spatial relationships.

3. **Scene Synthesizer + Position Diversifier**:

    - Assembles the 3D scene (camera, light source, background, ground plane, two objects), automatically adding a ground plane to prevent objects from floating, and supports shadows to enhance realism.
    - The Position Diversifier increases diversity by randomly rotating backgrounds, adding camera position jitter, and randomly rotating objects.
    - Design Motivation: Maximize the diversity of generated images while guaranteeing spatial accuracy.

4. **Training-Free Image Generation**:

    - Converts a standard T2I pipeline into an image-to-image pipeline: $\phi(I|x^{(g)}, T)$
    - Option A: Uses SDEdit, adding noise to the reference image and then denoising to generate the final image.
    - Option B: Uses ControlNet (Canny edge condition), extracting low-level features from the reference image for guidance.
    - Design Motivation: SDEdit provides spatial guidance, while ControlNet mitigates asset attribute bias.

5. **RevQA Benchmark**:

    - Includes 16 yes-no question types that contain combinations of negation, conjunction, and disjunction.
    - Introduces Random (replaced with random objects) and Adversarial (replaced with semantically similar objects) variants.
    - Evaluates the robustness of MLLMs in spatial reasoning.

### Loss & Training
The proposed method is entirely training-free, involving no extra loss functions or training processes. The trade-off between spatial accuracy and photorealism is controlled by adjusting the number of denoising steps.

## Key Experimental Results

### Main Results

| Method | OA (%) | VISOR_cond (%) | VISOR_1 (%) | VISOR_4 (%) |
|------|--------|----------------|-------------|-------------|
| SD 1.4 (baseline) | 29.86 | 18.81 | 62.98 | 1.63 |
| SD 1.4 + REVISION | 53.96 | 52.71 | 97.69 | 27.15 |
| SD 1.5 (baseline) | 28.43 | 17.51 | 61.59 | 1.35 |
| SD 1.5 + REVISION | **54.33** | **53.08** | **97.72** | 27.55 |
| Control-GPT | 48.33 | 44.17 | 65.97 | 20.48 |
| ControlNet + REVISION | **56.88** | **55.48** | 97.54 | **31.59** |

On SD 1.5: OA increases by 91.1%, and conditional score increases by 58.6%.

| Method | VISORcond SD $\sigma$ | Description |
|------|-------------------|------|
| Control-GPT | 2.95 | High fluctuation across different spatial relations |
| ControlNet + REVISION | **0.21** | Consistent performance across all spatial relations |
| DALLE-v2 | 3.38 | Significantly better on "below" relation |

### Ablation Study

| Background Type | IS ↑ | OA (%) | VISOR_cond (%) | Description |
|---------|------|--------|----------------|------|
| White | 16.27 | 54.33 | 53.08 | Highest spatial accuracy |
| Indoor | 19.11 | 48.77 | 45.28 | More diverse but slightly lower accuracy |
| Outdoor | 19.66 | 43.99 | 41.51 | Richest diversity, highest IS |

### Key Findings
- REVISION performs consistently across all spatial relationship types ($\sigma$ is only 0.21%), whereas Control-GPT exhibits a deviation of 6.8%.
- A white background provides the highest spatial accuracy, but outdoor backgrounds bring higher diversity and Inception Score.
- RevQA shows that MLLMs perform below chance (< 50%) on opposing spatial relations and double negation questions.
- Extension experiments on depth relations show that REVISION also brings significant improvements (OA: 41.52% → 58.32%).
- Human evaluation: 79.62% accuracy for multi-object, multi-relation prompts, and 63.62% accuracy for OOD objects.

## Highlights & Insights
- **Zero-cost spatial guidance**: Entirely training-free, plug-and-play, and applicable to any T2I model.
- **Deterministic guarantee**: The rendering pipeline guarantees 100% spatial accuracy, eliminating probabilistic biases.
- **Outstanding consistency**: The performance deviation of REVISION across different spatial relationship types is extremely small ($\sigma < 0.3\%$), which is unseen in any of the prior methods.
- **RevQA reveals MLLM fragility**: Even LLaVA 1.5 achieves only 55.9% accuracy on adversarial spatial questions.

## Limitations & Future Work
- The Asset Library only supports 101 object categories, and OOD objects require semantically approximate replacements, resulting in lower accuracy.
- It only supports spatial relationships between two objects, and extension to multi-object scenes is limited.
- The photorealism of rendered images still has a gap with photos, which may introduce visual bias.
- More spatial relationship types (such as "around", "between") and occlusion relationships can be introduced.

## Related Work & Insights
- **vs SPRIGHT**: SPRIGHT requires 6 million images with detailed caption relabeling for training, while REVISION is completely training-free.
- **vs Layout Guidance**: Layout Guidance relies on bounding box annotations, whereas REVISION automatically parses layout from the prompt.
- **vs Control-GPT**: Control-GPT has high training costs and shows large performance fluctuations across different spatial relations.
- **Insight**: Combining rendering tools with generative models is an under-explored direction, which can be extended to video generation, 3D scene generation, etc.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of rendering-guided T2I is novel, but the SDEdit guidance technology itself is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with multiple benchmarks, human evaluation, ablation studies, and RevQA.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and rich figures/tables, but some details are left in the supplementary materials.
- Value: ⭐⭐⭐⭐ High practicality, training-free plug-and-play, and RevQA is also a valuable benchmark contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models](../../ICCV2025/multimodal_vlm/tab_transformer_attention_bottlenecks_enable_user_intervention_and_debugging_in_.md)
- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)
- [\[ECCV 2024\] FlexAttention for Efficient High-Resolution Vision-Language Models](flexattention_for_efficient_high-resolution_vision-language_models.md)
- [\[ECCV 2024\] BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_vision-language_models.md)
- [\[ECCV 2024\] Attention Prompting on Image for Large Vision-Language Models](attention_prompting_on_image_for_large_visionlanguage_models.md)

</div>

<!-- RELATED:END -->
