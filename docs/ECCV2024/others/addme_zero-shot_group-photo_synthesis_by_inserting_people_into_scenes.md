---
title: >-
  [Paper Note] AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes
description: >-
  [ECCV 2024][Portrait Insertion] This paper proposes AddMe, a zero-shot portrait generator based on diffusion models. Through an identity decoupling adapter and an enhanced portrait attention module, it can naturally insert a given portrait into specified positions of an existing scene image, while maintaining identity consistency and the plausibility of group interactions.
tags:
  - "ECCV 2024"
  - "Portrait Insertion"
  - "Group-Photo Synthesis"
  - "Diffusion Models"
  - "Identity Preservation"
  - "Zero-Shot Generation"
date: 2026-05-08
content_hash: 8151a48197d15965
---

# AddMe: Zero-Shot Group-Photo Synthesis by Inserting People Into Scenes

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Others  
**Keywords**: Portrait Insertion, Group-Photo Synthesis, Diffusion Models, Identity Preservation, Zero-Shot Generation

## TL;DR

This paper proposes AddMe, a zero-shot portrait generator based on diffusion models. Through an identity decoupling adapter and an enhanced portrait attention module, it can naturally insert a given portrait into specified positions of an existing scene image, while maintaining identity consistency and the plausibility of group interactions.

## Background & Motivation

**Background**: Large text-to-image diffusion models (such as Stable Diffusion) have achieved significant progress in high-quality image generation. Personalization methods (such as DreamBooth, IP-Adapter) can generate new images of a specific person based on reference images; image editing methods (such as Inpainting) can modify local regions. However, inserting the facial identity of a specific portrait into an existing photo—especially for group-photo synthesis—remains a challenging task.

**Limitations of Prior Work**: Current methods face three main difficulties: (1) Existing personalization customization methods (e.g., DreamBooth) excel at generating new images containing the target person, but cannot precisely control the person's position and pose in an existing scene, making it difficult to accomplish the task of "inserting a new person at a designated position in an existing group photo"; (2) Existing local image editing methods (e.g., Inpainting) can generate content in designated areas, but lack sufficient capability to handle facial details, often resulting in generated faces that lack realism or fail to maintain the target identity; (3) Most crucially, the inserted person needs to establish reasonable spatial and social interaction relationships with the existing people in the scene (such as gaze direction, body orientation, and relative distance), which requires a deep understanding of the group photo's context.

**Key Challenge**: There is an inherent contradiction between identity preservation and scene integration. Keeping the identity requires fidelity to the reference portrait, but natural integration into the scene requires adjusting the person's pose, lighting, and expression according to the scene context, making it difficult to balance both.

**Goal**: (1) How to maintain facial identity while ensuring the generated portrait is consistent with the scene in terms of pose, lighting, and style? (2) How to ensure reasonable social interactions between the inserted person and the existing people in the scene? (3) How to achieve zero-shot inference without requiring additional training for each new identity?

**Key Insight**: The authors design a two-stage solution: first, an identity adapter is used to learn a facial representation decoupled from the existing people in the scene, and then an enhanced portrait attention module is employed to capture the scene context during the generation process, ensuring logical interactions between the generated portrait and the existing people.

**Core Idea**: To naturally insert new portraits into group photos under zero-shot conditions through decoupled identity representations and context-aware portrait attention.

## Method

### Overall Architecture

AddMe is built upon a pretrained text-to-image diffusion model. The input consists of three parts: (1) a scene image (which may already contain other people), (2) reference portrait photos of the person to be inserted, and (3) spatial conditions (such as a mask or keypoints) specifying the insertion location. The overall workflow divides into the collaboration of two key modules: the Identity Adapter extracts facial representations highly related to the identity but independent of the existing people in the scene; the Enhanced Portrait Attention module allows the generated region to perceive information about the existing people in the scene during the diffusion denoising process, thereby producing reasonable interaction effects. The final output is a synthesized group photo containing the new person.

### Key Designs

1. **Identity Adapter**:

    - **Function**: Learns a facial identity representation from the reference portrait, which is decoupled from the characteristics of existing people in the scene, ensuring there is no identity confusion.
    - **Mechanism**: Uses a pretrained face recognition model (such as ArcFace) to extract the identity embedding vector of the reference portrait, which is then mapped to a conditioning vector compatible with the diffusion model through a set of learnable mapping layers. The key design is the decoupling mechanism: during training, the model simultaneously sees both the existing people in the scene and the facial features of the person to be inserted. Contrastive learning constraints are applied to ensure that the conditioning vector output by the identity adapter only encodes the identity information of the person to be inserted, without confusing it with the features of the existing people in the scene. Specifically, for each existing person in the scene, the similarity between their facial features and the identity conditioning vector is computed, and this similarity is minimized via a contrastive loss.
    - **Design Motivation**: Without decoupling, the model may "leak" the facial features of the existing people in the scene to the newly generated region during generation, causing the generated portrait to resemble one of the existing people in the scene rather than the target identity.

2. **Enhanced Portrait Attention**:

    - **Function**: Captures scene context information during the diffusion denoising process, enabling the generated portrait to have reasonable social interactions with existing people in the scene.
    - **Mechanism**: In the cross-attention layer of the U-Net, scene-level context conditions are introduced in addition to the standard text conditions. Specifically, the scene image (excluding the region to be generated) is passed through an encoder to obtain scene feature maps. Then, in the attention computation of each denoising step, the query of the generated region not only attends to the text embedding but also attends to the regions in the scene features related to the existing people. This allows the generation process to "see" the poses and positions of surrounding people, thereby adjusting the orientation, expression, and body language of the generated person.
    - **Design Motivation**: People in group photos usually exhibit social interactions—they face the camera, stand close to each other, or even have physical contact. If the newly inserted person "ignores" the surrounding people, the synthesis result will look highly unnatural. By enabling the generation process to perceive the context, a more realistic group-photo effect can be generated.

3. **Spatial Conditioning Control**:

    - **Function**: Precisely controls the position, size, and basic pose of the generated portrait.
    - **Mechanism**: The method is compatible with text conditions and various spatial condition inputs, including binary masks (specifying the generation region), keypoints (specifying the body skeleton), and semantic segmentation maps. These spatial conditions are injected into the diffusion model through ControlNet or similar architectures, allowing users to precisely specify the position and basic pose of the new person.
    - **Design Motivation**: Practical usage scenarios for group-photo synthesis require users to precisely control the position of the new person (e.g., standing between two specific people). Such fine-grained control cannot be achieved by text descriptions alone.

### Loss & Training

Training employs a combination of the standard diffusion denoising loss, identity consistency loss, and decoupled contrastive loss. The identity consistency loss minimizes the distance between the generated portrait and the reference portrait in the facial feature space; the decoupled contrastive loss ensures that the identity representation is not contaminated by existing people in the scene. The training data comes from a large-scale group-photo dataset, where training pairs are constructed by randomly masking one person in a group photo.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (AddMe) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| In-house Group Photo Test Set | Identity Similarity (FaceNet) | Significantly outperforms | Paint-by-Example, IP-Adapter | Significant lead |
| In-house Group Photo Test Set | FID | Outperforms | Baselines | Better image quality |
| In-house Group Photo Test Set | Human Preference Evaluation | Outperforms | Baselines | Superior in both naturalness and identity preservation |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| w/o Identity Adapter | Identity similarity drops significantly | Fails to maintain target person's identity |
| w/o Decoupling Mechanism | Identity confusion occurs | Generated portrait may blend features of existing people in the scene |
| w/o Enhanced Portrait Attention | Plausibility of interaction decreases | Generated person lacks a sense of interaction with surrounding people |
| Full Model | Optimal on all metrics | Modules work synergistically |

### Key Findings

- Identity decoupling is a crucial design in group-photo synthesis; without it, the model is prone to identity confusion.
- The Enhanced Portrait Attention effectively improves the naturalness of the interaction between the generated portrait and the scene, which shows the most significant improvement in human evaluations.
- The method maintains good performance across group photos with varying numbers of people (2-8 people), demonstrating robustness to different scene complexities.
- Zero-shot inference is highly efficient, allowing immediate deployment without extra training for each new identity.

## Highlights & Insights

- The design of identity decoupling is critical and ingenious; in multi-person scenarios, simple identity injection leads to identity confusion, and the decoupling mechanism fundamentally solves this issue.
- Achieving scene-context awareness through the attention mechanism is an elegant solution, implicitly generating plausible interactions without explicit social relationship modeling.
- The method has high practical application value; "helping someone who was absent to join a group photo" is a very common real-world demand.
- Compatibility with various spatial conditions (masks, keypoints, etc.) provides users with flexible control.

## Limitations & Future Work

- Code is not publicly available, limiting reproducibility.
- The synthesis quality may degrade when people in the scene are highly dense or heavily occluded.
- For reference portraits with profile views or extreme angles, the quality of identity preservation may be affected.
- Video scenarios are not considered; inserting the same person across continuous frames in a video while maintaining temporal consistency is a more challenging direction.
- Potential ethical issues: this technology could be abused for deepfakes, necessitating considerations for security and safety safeguards.

## Related Work & Insights

- Personalized image generation (DreamBooth, Textual Inversion, IP-Adapter) is a popular topic in recent years; this paper extends it from "generation" to "insertion", a more practical scenario.
- Image inpainting technologies (such as LaMa, Stable Diffusion Inpaint) provide the foundation for regional generation, but preserving facial identity requires additional design.
- The conflict between facial identity preservation and scene integration has also been deeply studied in the face swapping domain; the decoupling concept of this method could be borrowed for face swapping tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Group-photo portrait insertion is a novel and valuable task definition, and the decoupled + context-aware solution is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐ The comparison with various baselines is sufficient, but lacks a large-scale quantitative evaluation benchmark.
- Writing Quality: ⭐⭐⭐⭐ The task definition is clear, the methodology is well-motivated, and the writing is fluent.
- Value: ⭐⭐⭐⭐ Highly valuable for practical applications, presenting a comprehensive technical solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Zero-Shot Head Swapping in Real-World Scenarios](../../CVPR2025/others/zero-shot_head_swapping_in_real-world_scenarios.md)
- [\[ECCV 2024\] Non-parametric Sensor Noise Modeling and Synthesis](non-parametric_sensor_noise_modeling_and_synthesis.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](../../ACL2025/others/zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[CVPR 2026\] Hyperbolic Defect Feature Synthesis for Few-Shot Defect Classification](../../CVPR2026/others/hyperbolic_defect_feature_synthesis_for_few-shot_defect_classification.md)
- [\[CVPR 2026\] NAF: Zero-Shot Feature Upsampling via Neighborhood Attention Filtering](../../CVPR2026/others/naf_zero-shot_feature_upsampling_via_neighborhood_attention_filtering.md)

</div>

<!-- RELATED:END -->
