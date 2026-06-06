---
title: >-
  [Paper Note] Alterbute: Editing Intrinsic Attributes of Objects in Images
description: >-
  [ICML 2026][Multimodal VLM][Image Editing] Alterbute uses VLMs to automatically mine Visual Named Entity (VNE) identity clusters and jointly conditions identity references, attribute text, backgrounds…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Image Editing"
  - "Intrinsic Attribute Editing"
  - "Diffusion Models"
  - "Visual Named Entities"
  - "Identity Preservation"
date: 2026-05-08
content_hash: 1eef347bf9f0f195
---

# Alterbute: Editing Intrinsic Attributes of Objects in Images

**Conference**: ICML 2026  
**arXiv**: [2601.10714](https://arxiv.org/abs/2601.10714)  
**Code**: No public code (Project page: https://talreiss.github.io/alterbute/)  
**Area**: Image Generation / Image Editing  
**Keywords**: Image Editing, Intrinsic Attribute Editing, Diffusion Models, Visual Named Entities, Identity Preservation

## TL;DR
Alterbute uses VLMs to automatically mine Visual Named Entity (VNE) identity clusters and jointly conditions identity references, attribute text, backgrounds, and masks within a diffusion model. This approach unifies the editing of object color, texture, material, and shape while striving to maintain object identity and scene context.

## Background & Motivation
**Background**: Image editing models are already capable of large-scale text-guided modifications, local inpainting, style transfer, and subject-driven generation. While many methods can maintain coarse categories or instance appearances, it is significantly more difficult to modify intrinsic attributes—such as changing a car's color to red, a desk's material to wood, or altering an object's shape—while simultaneously preserving its identity.

**Limitations of Prior Work**: General image editors often edit the wrong object, alter the identity, or ignore target attributes. Conversely, subject personalization methods define identity so strictly that they allow almost no variation in color, material, texture, or shape. Attribute-specific methods typically only address a single attribute like material or texture, failing to cover the full spectrum of intrinsic attributes.

**Key Challenge**: There is an inherent tension between identity preservation and attribute editing. If the identity definition is too coarse (e.g., just "car"), the editing space is large but the object might be replaced by a different car. If the identity definition is too fine (e.g., a specific instance), the model treats color and texture as part of the identity, making meaningful intrinsic editing impossible.

**Goal**: The authors aim to train a single model that supports four types of intrinsic attribute editing—color, texture, material, and shape—while maintaining user-perceived object identity, background, lighting, and composition after the edit.

**Key Insight**: Rather than attempting to collect non-existent paired data of "the same object in the same scene with only intrinsic attribute changes," the paper relaxes the training task. It allows both intrinsic and extrinsic attributes to vary during training and fixes extrinsic factors during inference by reusing the original image background and mask.

**Core Idea**: Use Visual Named Entities (VNE) as an identity definition intermediate between coarse categories and specific instances. VLMs are then used to automatically construct supervised data of the "same VNE with different attributes and scenes," enabling the diffusion model to learn identity-preserving intrinsic attribute changes.

## Method
The methodology of Alterbute can be understood as "redefining identity first, then making the supervision problem collectable." If identity is defined by category, supervision is too loose; if defined by instance, it is too tight. VNE allows the model to observe natural variations of the same nameable object across different colors, materials, textures, shapes, and scenes, thereby learning which changes do not compromise identity.

### Overall Architecture
Training data is sourced from OpenImages. The authors first use Gemini to assign VNE labels to detected objects (e.g., "Porsche 911 Carrera" or "IKEA LACK table"), filtering out generalized or unnameable objects. Objects with the same VNE form identity clusters. Gemini then extracts structured intrinsic attribute descriptions for each object, including color, texture, material, and shape.

The diffusion model is fine-tuned based on SDXL. During training, inputs are organized into a $1\times 2$ image grid: the left half contains the noisy latent of the target image, and the right half contains an identity reference image from the same VNE cluster. The model also receives target attribute text, a background image, and an object mask. The target region in the background image is masked with gray, and the mask specifies the object's position. The loss is only applied to the left half, forcing the model to learn to generate an object with target attributes while maintaining identity within the specified scene.

During inference, given a source image and a single attribute prompt, the system uses a segmentation model to extract the object mask, crops the foreground as an identity reference, and uses the original background and mask as extrinsic conditions. Fine masks are used for color, texture, and material editing. For shape editing, as the target geometry is unknown, a coarse bounding-box mask is used to provide the model with more deformation space.

### Key Designs
1. **Visual Named Entity (VNE)**:
    - **Function**: Provides a supervision unit that preserves identity while allowing intrinsic attribute variations.
    - **Mechanism**: Uses Gemini to assign fine-grained nameable labels based on visual appearance; images under the same VNE are treated as different instances or states of the same perceptual identity.
    - **Design Motivation**: DINOv2 similarity tends to cluster objects with different identities but similar appearances, while instance retrieval is too strict; VNE serves as an intermediate layer that aligns better with human naming conventions.

2. **Relaxed Training, Constrained Inference**:
    - **Function**: Bypasses the lack of strictly paired data for intrinsic attribute editing.
    - **Mechanism**: Training allows target and reference images to differ in intrinsic attributes, poses, and backgrounds, requiring only that they belong to the same VNE. During inference, the original background and mask are fixed to focus changes on intrinsic attributes.
    - **Design Motivation**: Samples of "same object, same scene, only attribute change" are rare in natural data, but samples of "same VNE, different scenes and attributes" can be automatically mined at scale.

3. **Grid-based Identity Conditioning**:
    - **Function**: Enables the diffusion UNet to pass fine-grained identity information between the target region and identity reference via self-attention.
    - **Mechanism**: Places the noisy target and reference object in two $512\times 512$ panels to form a $512\times 1024$ grid. Background and mask conditions are only applied to the left half, while the right half provides a background-removed identity reference.
    - **Design Motivation**: Ablations show that channel-wise concatenation leads to the model performing almost no editing, indicating that identity information must be explicitly propagated across panels via spatial attention.

### Loss & Training
The model utilizes a standard diffusion $L_2$ denoising loss, calculated only on the left target region. It is trained for 100,000 steps with a learning rate of $10^{-5}$ and a batch size of 128 at a resolution of $512\times 1024$. Based on the 7B parameter SDXL architecture, training took approximately 24 hours on 128 v4 TPUs. To improve robustness, 10% of samples randomly drop the identity reference, and another 10% drop the text prompt. Inference uses a text CFG of 7.5 and an image CFG of 2.0.

## Key Experimental Results

### Main Results
The authors constructed an evaluation set featuring 30 objects and 100 attribute editing samples covering color, texture, material, and shape. The user study involved 166 participants, with each sample receiving 5 independent judgments. Pairwise evaluations were also conducted using VLMs (Gemini, GPT-4o, and Claude).

| Evaluator | vs MimicBrush | vs MaterialFusion | vs FlowEdit | vs InstructPix2Pix | vs OmniGen | vs UltraEdit | vs Diptych |
|-----------|---------------|-------------------|-------------|--------------------|------------|--------------|------------|
| User      | 85.0%         | 79.7%             | 89.3%       | 85.0%              | 81.2%      | 80.0%        | 76.2%      |
| Gemini    | 94.3%         | 87.0%             | 89.6%       | 88.8%              | 80.2%      | 86.0%        | 76.8%      |
| GPT-4o    | 89.8%         | 77.6%             | 88.6%       | 87.0%              | 77.4%      | 78.6%        | 74.8%      |
| Claude    | 92.6%         | 81.3%             | 92.6%       | 85.4%              | 78.8%      | 85.6%        | 77.8%      |

### Ablation Study
Analysis focused on identity definition, conditioning methods, and training budget. Standard DINO/CLIP metrics were reported, though the authors emphasized that these metrics are not entirely reliable for intrinsic editing, as "no-edit" results can produce high identity scores.

| Analysis Item | Key Metrics | Note |
|---------------|-------------|------|
| Standard Metrics (Ours) | DINO 0.815 / CLIP-I 0.914 / CLIP-T 0.321 | Highest CLIP-T, indicating best target attribute matching. |
| Standard Metrics (UltraEdit) | DINO 0.841 / CLIP-I 0.922 / CLIP-T 0.303 | High identity metrics, but weaker attribute matching than Alterbute. |
| VNE Data Scale | 69,744 clusters / 1,079,442 images | Automatically constructed by OpenImages and Gemini. |
| Channel-wise conditioning | Qualitative results close to no-op | Identity reference not effectively passed; model tends to output the original. |
| 50K training steps | VLM Win Rate 78.0/75.7/76.3 | Half budget remains significantly stronger than baselines. |
| 100K training steps | VLM Win Rate 86.1/82.0/84.9 | Full training improves results by ~7 percentage points. |
| 100K vs 50K | VLM Preference 58.2/57.1/60.6 | Longer training is beneficial but not the sole decisive factor. |

### Key Findings
- Both users and VLMs significantly prefer Alterbute, with p-values from binomial tests consistently below 0.05, indicating the advantage is not due to evaluator bias.
- When split by attribute, shape editing showed the highest win rate, suggesting Alterbute excels at difficult geometric changes where baselines struggle.
- VNE is not merely label engineering but the core of the supervision; it provides clusters of "same identity but variable attributes," preventing the model from fixing all intrinsic attributes as part of the identity.

## Highlights & Insights
- The most compelling aspect of the paper is the redefinition of "identity." Instead of treating it as an abstract concept, VNE provides a data structure that is automatically labelable, scalable, and aligned with human naming conventions.
- The relaxation of training targets is ingenious. Allowing more variation during training makes data acquisition feasible, while fixing extrinsic variations during inference with backgrounds and masks is more practical than searching for rare paired data.
- The grid-based input demonstrates that the conditioning method of a diffusion model determines whether it truly utilizes the reference image. For identity-preserving editing, spatial self-attention is more critical than simple channel concatenation.

## Limitations & Future Work
- VNE labeling depends on Gemini and may inherit VLM biases regarding brands, object categories, and long-tail cultural entities.
- The evaluation set is limited to 30 objects and 100 samples; while it covers four attribute types, it is too small to fully represent real-world open-domain scenarios.
- Coarse bounding-box masks support shape editing but may introduce background artifacts; changing the shape of rigid objects can also result in unrealistic geometry.
- The focus is currently on single-object editing; multi-object interactions, occlusions, reflections, and physical consistency require more robust scene modeling.

## Related Work & Insights
- **vs InstructPix2Pix / UltraEdit / OmniGen**: These general editors cover a wide range of tasks but lack stable joint constraints for intrinsic attributes and identity preservation. Alterbute specifically learns these variations via VNE supervision.
- **vs DreamBooth / subject-driven generation**: Personalization methods emphasize instance preservation but often bind color and texture to the instance identity. Alterbute allows these to vary within the same VNE, making it better suited for attribute editing.
- **vs MaterialFusion / MimicBrush**: These methods target single attributes like material or texture; Alterbute provides a unified model for color, texture, material, and shape.
- **Insight**: The bottleneck in many generative tasks is not just model architecture, but the semantic granularity of supervision. Finding an appropriate intermediate label layer can transform "impossible-to-collect" data into automatically constructible data.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The VNE identity definition and relaxed training targets are highly creative; the model core remains based on established diffusion editing paradigms.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes user studies, VLM evaluations, standard metrics, and training budget analysis, though the benchmark scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐☆ The motivation is clearly articulated, and the explanation of the identity definition spectrum is particularly helpful for understanding the methodology.
- Value: ⭐⭐⭐⭐☆ Offers significant insights into controllable image editing and supervised data construction, especially for product-level object attribute editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Debate with Images: Detecting Deceptive Behaviors in Multimodal Large Language Models](debate_with_images_detecting_deceptive_behaviors_in_multimodal_large_language_mo.md)
- [\[CVPR 2026\] Zina: Multimodal Fine-grained Hallucination Detection and Editing](../../CVPR2026/multimodal_vlm/zina_multimodal_fine-grained_hallucination_detection_and_editing.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](../../ICCV2025/multimodal_vlm/advancing_textual_prompt_learning_with_anchored_attributes.md)
- [\[NeurIPS 2025\] Learning Skill-Attributes for Transferable Assessment in Video](../../NeurIPS2025/multimodal_vlm/learning_skill-attributes_for_transferable_assessment_in_video.md)
- [\[CVPR 2026\] DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](../../CVPR2026/multimodal_vlm/dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)

</div>

<!-- RELATED:END -->
