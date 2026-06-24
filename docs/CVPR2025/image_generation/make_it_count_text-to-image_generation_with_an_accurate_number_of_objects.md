---
title: >-
  [Paper Note] Make It Count: Text-to-Image Generation with an Accurate Number of Objects
description: >-
  [CVPR 2025][Image Generation][Text-to-Image Generation] This paper proposes CountGen, an approach that isolates and counts object instances by identifying features carrying object identity information in the denoising process of diffusion models, and trains a layout prediction model to mitigate under-generation, enabling accurate object-counting text-to-image generation without relying on external layout generators.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Text-to-Image Generation"
  - "Counting Accuracy"
  - "Object Quantity Control"
  - "Diffusion Models"
  - "Attention Mechanism"
date: 2026-05-08
content_hash: fc97000ede317c59
---

# Make It Count: Text-to-Image Generation with an Accurate Number of Objects

**Conference**: CVPR 2025  
**arXiv**: [2406.10210](https://arxiv.org/abs/2406.10210)  
**Code**: Yes (Project Page)  
**Area**: Image Generation / Diffusion Models  
**Keywords**: Text-to-Image Generation, Counting Accuracy, Object Quantity Control, Diffusion Models, Attention Mechanism

## TL;DR
This paper proposes CountGen, an approach that isolates and counts object instances by identifying features carrying object identity information in the denoising process of diffusion models, and trains a layout prediction model to mitigate under-generation, enabling accurate object-counting text-to-image generation without relying on external layout generators.

## Background & Motivation

**Background**: Text-to-image diffusion models (such as Stable Diffusion, DALL-E, etc.) have achieved remarkable generation quality, but still face numerous challenges regarding controllability. One seemingly simple yet difficult issue is accurately generating a specified number of objects based on a text prompt. For instance, "three cats" might result in two or four, and "five apples" might lead to three or seven.

**Limitations of Prior Work**: Controlling object quantity is difficult due to several fundamental reasons: First, generative models must maintain distinct identity representations for each object instance, even when multiple objects have identical appearances or overlap. Second, the model needs to implicitly perform a global calculation during generation to ensure the correct count. Third, it remains unclear whether structures capable of carrying such information exist in the internal representation of diffusion models. Existing methods either rely on external layout generators (such as LayoutGPT) to plan object locations, or repeatedly generate and filter images to obtain the correct count, which is highly inefficient.

**Key Challenge**: The denoising process of diffusion models progressively recovers signal from noise. In the early denoising stages, the image structure is not yet formed, and the object count might still be undetermined, whereas in the later stages, the layout is fixed, making it difficult to insert new objects. How to detect and correct the object count at the right moment during denoising is the core challenge.

**Goal**: (1) Identify internal features within diffusion models that carry object instance identity information; (2) detect in real-time during denoising whether the object count is correct; (3) design correction mechanisms to resolve under- or over-generation.

**Key Insight**: The authors discover that features at specific layers of the diffusion model's cross-attention maps can distinguish different object instances. Even for multiple objects of the same category, they manifest as distinct connected regions in the attention maps. Based on this finding, real-time counting can be performed during the denoising process.

**Core Idea**: Utilizing the diffusion model's own cross-attention features to isolate and count object instances. When under-generation is detected, a lightweight layout model is used to predict the locations and shapes of missing objects, which are then injected through attention guidance to steer the denoising process.

## Method

### Overall Architecture
The workflow of CountGen is as follows: During the denoising process of the diffusion model, attention maps corresponding to target object tokens are extracted from the cross-attention maps every few steps. These maps are separated into distinct instances and counted using thresholding and connected component analysis. If under-generation is detected, a pre-trained layout prediction model is queried to predict the locations and shapes (bounding boxes + rough masks) of the missing objects based on the spatial distribution of existing objects. Attention manipulation is then applied to guide the denoising process to generate new objects at the specified positions. If over-generation is detected, it is suppressed by decreasing the attention responses in the redundant regions.

### Key Designs

1. **Attention-based Instance Separation & Counting**:

    - **Function**: Real-time detection of the number of target object instances generated during denoising.
    - **Mechanism**: For target object tokens in the text prompt (e.g., "cat"), cross-attention maps of specific UNet layers are extracted. The attention values are normalized and thresholded to obtain binary masks, and connected component analysis is applied to identify individual object instances. Each connected region corresponds to an object instance, and the number of regions serves as the current object count. This operation is repeated across multiple denoising steps to obtain stable counts.
    - **Design Motivation**: Cross-attention maps naturally reflect the mapping between image spatial coordinates and text tokens. Different instances of the same category are naturally isolated into distinct spatial attention "hotspots." This is an inherent structure of the model that can be utilized without extra training.

2. **Layout Prediction Model**:

    - **Function**: Predicts the locations and shapes of missing objects when under-generation is detected.
    - **Mechanism**: A lightweight Transformer-based model is trained to take the layout arrangement of existing objects (positions, sizes, masks) and the target count as inputs, and predict the bounding boxes and rough shape masks of the missing objects. The key innovation is that this model leverages the diffusion model's own prior—the training data comes from layout distributions naturally occurring within a massive set of generated images, rather than external data sources. Consequently, the predicted layouts are prompt-dependent and seed-dependent, naturally harmonizing with the context of the current generation.
    - **Design Motivation**: Unlike methods such as LayoutGPT that rely on external large language models for layout planning (which might mismatch the actual style of the generated image), this method directly learns layout priors from the diffusion model's generative distribution, ensuring spatial rationality of the inserted objects.

3. **Attention-Guided Denoising**:

    - **Function**: Guides the model to generate new objects at specified locations during denoising.
    - **Mechanism**: After obtaining the predicted locations of missing objects, the cross-attention maps are modified to enhance the attention response of the target token at the designated positions while moderately suppressing it elsewhere. Specifically, a soft constraint is applied to the attention maps at each denoising step, "pushing" the attention of target tokens toward the predicted areas. For over-generation, the reverse operation is performed to decrease attention values at the locations of extra objects.
    - **Design Motivation**: Attention guidance operates as a gentle control mechanism that does not disrupt global image coherence. Compared to direct inpainting on latent variables, attention manipulation better preserves global consistency.

### Loss & Training
The training of the layout prediction model uses standard regression losses ($L_{1}$ loss for bounding box coordinates, and BCE loss for mask prediction). The training data is obtained by running the diffusion model extensively to generate images, with layout information automatically annotated by a detector. The overall CountGen framework requires no additional training of the diffusion model itself during inference.

## Key Experimental Results

### Main Results

| Method | CountBench Accuracy↑ | Mean Counting Error↓ | FID↓ | CLIP-Score↑ |
|------|-----------------|-------------|------|------------|
| Stable Diffusion | 28.3% | 1.42 | 18.2 | 0.312 |
| Attend-and-Excite | 35.1% | 1.18 | 19.5 | 0.305 |
| LayoutGPT + SD | 41.7% | 0.95 | 22.3 | 0.289 |
| CountGen (Ours) | **54.8%** | **0.71** | **19.1** | **0.308** |

### Ablation Study

| Configuration | Count Accuracy↑ | Explanation |
|------|-----------|------|
| Full CountGen | 54.8% | Full method |
| w/o Layout Prediction (Random Position) | 42.3% | Performance drops significantly with random placement |
| w/o Over-generation Correction | 50.1% | Only handles under-generation, ignoring over-generation |
| w/o Attention Guidance (Direct Inpainting) | 46.5% | Direct inpainting is less natural than attention guidance |
| Using External Layout (LayoutGPT) | 48.2% | External layouts underperform compared to self-priors |
| Different Detection Steps (Step 10) | 51.2% | Layout has not stabilized in early steps |
| Different Detection Steps (Step 25) | 54.8% | Best performance when detected at mid-steps |

### Key Findings
- CountGen works best for counting tasks involving 2-5 objects. Its accuracy drops significantly with 7 or more objects because severe overlap in dense layouts makes it difficult for attention maps to isolate instances.
- Using the diffusion model's own layout prior outperforms the external LayoutGPT (54.8% vs 48.2%), demonstrating the superiority of prompt-dependent layouts.
- The contribution of over-generation correction (suppressing extra objects) is approximately 4.7%, showing that both under-generation and over-generation are crucial issues that need to be addressed.
- CountGen barely sacrifices any image quality in terms of FID (retaining parity with the original SD), indicating that attention manipulation is a gentle and effective control mechanism.
- Cross-model evaluations reveal that CountGen's attention isolation strategy is also effective on SDXL, exhibiting strong generalization capability.

## Highlights & Insights
- **Discovering instance-isolated representations within diffusion models** is the most significant insight of this work. Since same-class objects naturally form independent connected regions in cross-attention maps, diffusion models already "know" where each object is, but this knowledge remains underutilized.
- The design of **self-guided layout prediction** is highly elegant: instead of introducing external knowledge sources, it learns "where to place missing objects" directly from the model's own generative distribution. This ensures realism and consistency in the generated outputs.
- The counting problem seems simple but touches the core of generative model controllability: how to perform discrete structural control in a highly entangled latent space.

## Limitations & Future Work
- Accuracy is still limited for a high number of objects (>5), especially in densely occluded scenarios.
- Currently, the method is optimized for counting objects within a single category; hybrid multi-category counting (e.g., "3 cats and 2 dogs") has not been fully verified.
- Instance separation in attention maps depends heavily on threshold selection, and the optimal threshold may vary across different prompts and seeds.
- The layout prediction model requires an additional training data generation and annotation stage, which, although automated, increases pipeline complexity.
- Future work could explore joint optimization of count control with other attribute controls (such as color, size, and pose).

## Related Work & Insights
- **vs Attend-and-Excite**: A&E ensures that all textual concepts exist in the image by enhancing attention on neglected tokens, but fails to focus on multiple instances of the same concept. CountGen focuses on instance-level count control, extending A&E along the quantity dimension.
- **vs LayoutGPT**: LayoutGPT uses LLMs to generate layouts first and then generates images conditionally; however, LLM-generated layouts may not match the natural distribution of diffusion models. CountGen extracts information entirely from within the diffusion model, ensuring greater self-consistency.
- **vs Training-free Compositional Generation**: Such methods control generated content by fusing denoising results of multiple prompts, but struggle to control quantities precisely. CountGen's instance detection and correction mechanisms are more straightforward and direct.

## Rating
- Novelty: ⭐⭐⭐⭐ Discovering and utilizing the instance isolation property in attention maps is a novel insight; the self-guided layout design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two benchmarks with comprehensive ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear problem formulation, well-motivated approach, and fluent writing.
- Value: ⭐⭐⭐⭐ Object quantity control is a real pain point in text-to-image generation, making this approach highly valuable for direct applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Emuru: Zero-Shot Styled Text Image Generation, but Make It Autoregressive](zero-shot_styled_text_image_generation_but_make_it_autoregressive.md)
- [\[CVPR 2025\] Compass Control: Multi Object Orientation Control for Text-to-Image Generation](compass_control_multi_object_orientation_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)
- [\[CVPR 2025\] A Comprehensive Study of Decoder-Only LLMs for Text-to-Image Generation](a_comprehensive_study_of_decoder-only_llms_for_text-to-image_generation.md)
- [\[CVPR 2025\] Learning to Sample Effective and Diverse Prompts for Text-to-Image Generation](learning_to_sample_effective_and_diverse_prompts_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
