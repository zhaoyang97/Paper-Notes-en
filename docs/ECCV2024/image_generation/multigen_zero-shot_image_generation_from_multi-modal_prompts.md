---
title: >-
  [Paper Note] MultiGen: Zero-Shot Image Generation from Multi-modal Prompts
description: >-
  [ECCV 2024][Image Generation][Multi-modal prompts] This paper proposes MultiGen, which constructs an "augmented token" for each object by fusing text, spatial coordinates, and image features. By training coordinate and feature models to handle missing modalities during inference, it achieves the first zero-shot image generation from multi-object multi-modal prompts, supporting flexible inputs of text-only or arbitrary modality combinations.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Multi-modal prompts"
  - "Zero-shot image generation"
  - "Augmented token"
  - "Diffusion models"
  - "Multi-object controllable generation"
date: 2026-05-08
content_hash: b593e244f02f34a6
---

# MultiGen: Zero-Shot Image Generation from Multi-modal Prompts

**Conference**: ECCV 2024  
**Authors**: Zhi-Fan Wu, Lianghua Huang, Wei Wang, Yanheng Wei, Yu Liu  
**Code**: None  
**Area**: Image Generation / Diffusion Models  
**Keywords**: Multi-modal prompts, Zero-shot image generation, Augmented token, Diffusion models, Multi-object controllable generation  

## TL;DR

This paper proposes MultiGen, which constructs an "augmented token" for each object by fusing text, spatial coordinates, and image features. By training coordinate and feature models to handle missing modalities during inference, it achieves the first zero-shot image generation from multi-object multi-modal prompts, supporting flexible inputs of text-only or arbitrary modality combinations.

## Background & Motivation

**Background**: Text-to-Image Generation has made significant progress in recent years, with models like Stable Diffusion and DALL-E capable of generating high-quality images from textual descriptions. However, purely textual descriptions have inherent limitations in precisely controlling object appearance, locations, and details.

**Limitations of Prior Work**: Relying solely on text makes it difficult to precisely describe specific object appearances (e.g., "a cat identical to my cat") or precise positions (e.g., "place a cup in the upper-left corner"). Existing methods either require fine-tuning (e.g., DreamBooth, Textual Inversion) to learn specific concepts, or only support a single object as an additional condition (e.g., IP-Adapter), failing to directly support zero-shot generation with multi-modal, multi-object prompts. Zero-shot generation with multiple objects and multiple modalities (any combination of text, coordinates, and reference images) remains an unresolved challenge.

**Key Challenge**: Users wish to precisely control the generation of multiple objects using various modalities (textual descriptions, spatial locations, and reference images). However, (1) training requires multi-modally aligned data, and (2) during inference, users might only provide a subset of modalities, requiring the model to handle arbitrary missing modalities.

**Goal**: (1) How to simultaneously support three modalities (text, coordinates, and reference images) in a unified framework? (2) How to support the concurrent controllable generation of multiple objects? (3) How to operate under a zero-shot setting without requiring fine-tuning for specific objects/concepts? (4) How to handle cases where some modalities are missing during inference?

**Key Insight**: The authors encapsulate the multi-modal information of each object into an "augmented token," which can be fed into the diffusion model alongside standard text prompts, thereby reusing the generative capabilities of text-to-image models. For modalities missing during inference, specialized models are trained to generate the missing modalities from the known ones.

**Core Idea**: Fusing the text, coordinates, and image features of each object into an "augmented token" to jointly train the diffusion model with the text prompt, while employing auxiliary models to predict missing modalities to achieve zero-shot, multi-modal controllable generation.

## Method

### Overall Architecture

MultiGen is built upon a pre-trained text-to-image diffusion model (e.g., Stable Diffusion). Given an image-text pair, the system first extracts three types of object-level information: textual descriptions (object-level text), spatial coordinates (bounding boxes), and reference image features (object image features). These three types of information are then fused into an "augmented token" for each object. These augmented tokens, along with the global text prompt, are used as conditional inputs to the diffusion model for denoising training. To address the problem of missing modalities during inference, pre-trained coordinate models (predicting layouts from text) and feature models (predicting visual features from text) are applied to complete the missing information.

### Key Designs

1. **Augmented Token**:

    - **Function**: Encodes the multi-modal information (text, coordinates, and images) of each object into a unified token representation.
    - **Mechanism**: For each object, the CLIP text features of its textual description, the positional encoding of its bounding box, and the CLIP visual features of its reference image are extracted separately. Subsequently, a lightweight fusion network integrates these three features into an "augmented token". The dimension of this token is compatible with the text embeddings, allowing it to be directly concatenated onto the sequence of global prompt tokens. During training, object information is automatically extracted from image-text datasets to construct training samples.
    - **Design Motivation**: Compressing multi-modal information into a representation isomorphic to text tokens enables the cross-attention mechanism of the diffusion model to handle multi-modal conditions naturally without modifying the model architecture.

2. **Coordinate Model**:

    - **Function**: Automatically predicts the spatial layout of objects (bounding box coordinates) from pure text descriptions during inference.
    - **Mechanism**: A lightweight model is trained to output predicted bounding boxes for each object, given the global text description and the text labels of individual objects. This model is trained on a large amount of image-text data to learn the mapping between textual descriptions and spatial layouts. When users provide only text without specifying positions, the coordinate model automatically fills in the spatial coordinates.
    - **Design Motivation**: Users often prefer not to manually specify the precise coordinates of each object. The coordinate model enables MultiGen to function even with text-only inputs, while allowing users to override the automatically predicted locations for manual layout control.

3. **Feature Model**:

    - **Function**: Generates the visual features of objects from text descriptions during inference to substitute for missing reference images.
    - **Mechanism**: A mapping model from CLIP text features to CLIP visual features is trained. When the user does not provide a reference image for a specific object, the feature model generates approximate visual features based on the text description, serving as the image feature component of the augmented token. This "text-to-visual feature" translation is more lightweight than direct "text-to-image" generation, requiring only alignment in the feature space.
    - **Design Motivation**: The zero-shot setting requires the model to work without any reference images. The feature model bridges the gap between the text and visual modalities, ensuring the quality of the augmented tokens across various input combinations.

### Loss & Training

The diffusion model utilizes the standard denoising loss $L_{denoise} = \mathbb{E}_{t,\epsilon}\|\epsilon - \epsilon_\theta(x_t, c)\|^2$, where the condition $c$ includes the global text prompt and all object augmented tokens. The coordinate model employs an L1 coordinate regression loss, while the feature model uses a cosine similarity loss to align text features with visual features. Training is conducted in stages: first, the coordinate model and feature model are trained independently, and then the diffusion model is jointly fine-tuned with the fusion network of the augmented tokens.

## Key Experimental Results

### Main Results

| Task/Setting | Metric | Ours | Prev. SOTA | Description |
|-----------|------|----------|---------|------|
| Multi-object layout control | FID ↓ | Significantly better | GLIGEN | Supports multi-modality instead of only text and coordinates |
| Single-object appearance reference | CLIP-I ↑ | Competitive | IP-Adapter | Zero-shot without fine-tuning |
| Pure text generation | FID ↓ | Comparable to SD | Stable Diffusion | Automatically completed by coordinate and feature models |
| Multi-modal combinations | User Study | High preference rate | No existing counterparts | First method to support this setup |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Full MultiGen | Optimal | Complete inputs of all three modalities |
| w/o Coordinate info | Disordered layout | Object positions are uncontrollable, with severe overlapping |
| w/o Reference image features | Decreased appearance consistency | Text-only descriptions cannot capture detailed appearances |
| w/o Feature model | Failed in text-only mode | Cannot handle cases lacking reference images |
| Single object vs Multi-object | Slightly degraded for multi-object | Inter-object interference leads to some loss in quality |

### Key Findings
- The design of the augmented token allows the model to naturally support an arbitrary number of objects, exhibiting good scalability.
- The quality of the coordinate and feature models significantly impacts the final generation performance—the accuracy of these auxiliary models is the bottleneck of the system.
- Attribute leaking among multiple objects remains a challenge, particularly when the object categories are similar.
- Although the zero-shot performance is inferior to fine-tuning methods (e.g., DreamBooth), it far outperforms other zero-shot baselines.

## Highlights & Insights
- **The unified representation of augmented tokens** is an elegant design: it translates multi-modal information into a representation isomorphic to text tokens, avoiding modifications to the diffusion model's architecture. This design perspective can be generalized to video generation, 3D generation, and other tasks requiring multi-modal conditions.
- **The strategy for handling missing modalities**—employing auxiliary models to predict missing modalities—is a generalizable paradigm. Compared to random modal dropout during training, this approach is more stable during inference.
- This work first defines and addresses the problem of "multi-object multi-modal zero-shot generation," laying foundations and providing a baseline for future research.

## Limitations & Future Work
- The attribute leaking problem during multi-object generation is not fully resolved, and features between objects are prone to cross-contamination.
- The prediction precision of the coordinate model is limited, showing insufficient spatial reasoning capabilities for complex layouts (e.g., >5 objects).
- Currently, only bounding box-level spatial control is supported, while finer-grained segmentation masks or keypoint controls are not accommodated.
- There is still a gap between the visual features generated by the feature model from text and real image features, resulting in insufficient appearance fidelity in zero-shot mode.
- Training requires additional coordinate and feature models, which increases system complexity.

## Related Work & Insights
- **vs GLIGEN**: GLIGEN supports controllable generation via text and coordinates but does not accept reference image inputs. MultiGen unifies three modalities via augmented tokens.
- **vs IP-Adapter**: IP-Adapter supports reference images as conditions but can only handle global references or single objects. MultiGen extends this to multi-object scenarios.
- **vs DreamBooth/Textual Inversion**: These methods learn specific concepts via fine-tuning, delivering high-quality but inflexible results. The zero-shot approach of MultiGen is more suitable for on-the-fly applications.

## Rating
- Novelty: ⭐⭐⭐⭐ Formulates and solves the problem of multi-object, multi-modal zero-shot generation for the first time.
- Experimental Thoroughness: ⭐⭐⭐ Abundant qualitative results, but quantitative evaluation is constrained by the lack of direct competitors for identical settings.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the methodological pipeline is well described.
- Value: ⭐⭐⭐⭐ Explores a new problem setting, and the augmented token design is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] OmniSSR: Zero-shot Omnidirectional Image Super-Resolution using Stable Diffusion Model](omnissr_zero-shot_omnidirectional_image_super-resolution_using_stable_diffusion_.md)
- [\[ECCV 2024\] DreamDrone: Text-to-Image Diffusion Models are Zero-shot Perpetual View Generators](dreamdrone_text-to-image_diffusion_models_are_zero-shot_perpetual_view_generator.md)
- [\[ECCV 2024\] FreeCompose: Generic Zero-Shot Image Composition with Diffusion Prior](freecompose_generic_zero-shot_image_composition_with_diffusion_prior.md)
- [\[ECCV 2024\] Zero-Shot Detection of AI-Generated Images](zero-shot_detection_of_ai-generated_images.md)
- [\[ECCV 2024\] MotionChain: Conversational Motion Controllers via Multimodal Prompts](motionchain_conversational_motion_controllers_via_multimodal_prompts.md)

</div>

<!-- RELATED:END -->
