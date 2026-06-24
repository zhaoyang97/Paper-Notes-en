---
title: >-
  [Paper Note] Open-World Amodal Appearance Completion
description: >-
  [CVPR 2025][3D Vision][Open-world object completion] This paper proposes a training-free open-world amodal appearance completion framework that accepts flexible natural language queries (including both direct names and abstract descriptions). By unifying segmentation, occlusion analysis, and iterative inpainting, the framework reconstructs the complete appearance of occluded objects and outputs RGBA formats to support downstream applications such as 3D reconstruction and imag…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Open-world object completion"
  - "occlusion reasoning"
  - "training-free framework"
  - "language-guided"
  - "RGBA output"
date: 2026-05-08
content_hash: 1562ea7111ebb48b
---

# Open-World Amodal Appearance Completion

**Conference**: CVPR 2025  
**arXiv**: [2411.13019](https://arxiv.org/abs/2411.13019)  
**Code**: None  
**Area**: 3D Vision / Amodal Completion  
**Keywords**: Open-world object completion, occlusion reasoning, training-free framework, language-guided, RGBA output

## TL;DR

This paper proposes a training-free open-world amodal appearance completion framework that accepts flexible natural language queries (including both direct names and abstract descriptions). By unifying segmentation, occlusion analysis, and iterative inpainting, the framework reconstructs the complete appearance of occluded objects and outputs RGBA formats to support downstream applications such as 3D reconstruction and image editing.

## Background & Motivation

Amodal completion aims to infer and reconstruct the occluded parts of objects, which is crucial for AR, 3D reconstruction, and content creation. However, existing methods have severe limitations: (1) **Closed-world categories**—methods like PD-MC rely on predefined object categories and fail when encountering unseen classes; (2) **Need for training data**—Pix2gestalt relies on a large amount of supervised training data; (3) **Lack of language interaction support**—users cannot specify target objects using natural language.

In open-world scenarios, object categories are diverse and unpredictable, and occlusion relationships are complex (including ambiguous background occlusions). A general framework that is **training-free, supports natural language specification of arbitrary objects, and handles complex occlusions** is highly demanded.

This paper introduces the concept of "reasoning amodal completion": the system infers and reconstructs the complete appearance of the queried object based on the image and a language query, supporting both concrete terms (e.g., "polar bear") and abstract queries (e.g., "What is the mammal in this image").

## Method

### Overall Architecture

The pipeline consists of four steps: (1) **Text query parsing and segmentation**: A VLM (LISA) is used to generate the visible region mask $M_{\text{visible}}$ based on the text query, while an automatic labeling system segments all objects and the background; (2) **Occlusion analysis**: InstaOrderNet is employed to determine which segments occlude the target, generating an occlusion mask $M_{\text{occ}}$; (3) **Prompt generation**: The optimal inpainting prompt is selected via CLIP matching; (4) **Iterative inpainting**: A pre-trained inpainting model is used to progressively reconstruct the occluded regions, outputting RGBA.

### Key Designs

**1. Open-World Segmentation and Background Processing**

- **Function**: Identify all possible occluders in the scene, including hard-to-identify background regions
- **Mechanism**: First, use an open-set labeling model + open-set detector + SAM to segment all nameable objects, obtaining a set $S$. Then, segment the unsegmented region $B = I - \bigcup S_i$ into independent background segments $\{B_1, \ldots, B_k\}$ via morphological operations (erosion and dilation), where $B_j = \text{Morph}(I - \bigcup_{i=1}^{m} S_i)$
- **Design Motivation**: Traditional segmentation ignores background regions (e.g., bushes, ground) that cannot be described with category labels, but these regions may occlude the target object. Background segmentation ensures all potential occluders are considered

**2. Occlusion Analysis and Boundary Awareness**

- **Function**: Determine which segments occlude the target object and generate an occlusion mask to guide the inpainting
- **Mechanism**: InstaOrderNet is used to perform pairwise occlusion order judgment between each segment (including background segments) and the target object, merging all occluding segments into $M_{\text{occ}} = \bigcup_{occ_i=1} S_i \cup \bigcup_{occ_j=1} B_j$. For cases touching the image boundaries, the occlusion mask is expanded via dilation operations: $M_{\text{occ}} \leftarrow M_{\text{occ}} \cup (d(M_{\text{visible}}) \cap \bigcup_{e \in E} \text{edge}_e)$
- **Design Motivation**: Occlusion relationships cannot be simply determined by spatial location; they require a dedicated occlusion order reasoning model. Boundary awareness handles cases where the target object extends beyond the image boundaries

**3. CLIP-Guided Prompt Selection and Iterative Inpainting**

- **Function**: Automatically generate the optimal inpainting prompt and iteratively reconstruct the occluded regions
- **Mechanism**: The target visible region and all candidate labels $T \cup Q$ are matched via CLIP to select the optimal prompt $P = \arg\max_{t_i} \text{CLIP}(I_{\text{target}}, t_i)$. Inpainting is performed iteratively: $I_{\text{inpaint}}^{(t+1)} = \phi(I_{\text{inpaint}}^{(t)}, M_{\text{occ}}^{(t)}, P)$, updating the occlusion mask and amodal mask at each step, and terminating when $\Delta M_{\text{occ}}^{(t)} < \epsilon$ or the maximum number of iterations is reached. Finally, the original visible region and the reconstructed region are merged via alpha blending
- **Design Motivation**: User queries can be abstract descriptions, and CLIP matching ensures that the prompt aligns with the visual attributes of the target object. Iterative inpainting progressively expands the reconstructed region to handle complex occlusions

### Loss & Training

- **Training-free framework**: No additional training is required, leveraging a combination of pre-trained models (LISA, SAM, InstaOrderNet, CLIP, and inpainting models)
- Iteration termination condition: The change in the occlusion mask is below the threshold $\epsilon$ or the maximum number of iterations $T$ is reached
- Alpha blending ensures that pixels of the original visible region are not modified

## Key Experimental Results

### Main Results

Comparison on an evaluation dataset with 2565 instances and 553 categories:

| Method | CLIP↑ | LPIPS↓ | Feature Sim.↑ | SSIM↑ |
|------|-------|--------|---------------|-------|
| PD w/o MC | 24.553 | 0.614 | 0.404 | 0.395 |
| PD-MC | 27.984 | 0.628 | 0.364 | 0.413 |
| Pix2gestalt | 27.417 | 0.442 | 0.548 | 0.714 |
| **Ours** | **28.181** | **0.320** | **0.646** | **0.731** |

### Ablation Study

The impact of prompt variants and background segmentation:

| Configuration | CLIP↑ | LPIPS↓ | Feature Sim.↑ | SSIM↑ |
|------|-------|--------|---------------|-------|
| Q only | 28.563 | 0.327 | 0.633 | 0.724 |
| T only | 28.043 | 0.324 | 0.636 | 0.725 |
| T∪Q w/o bg seg. | 28.071 | 0.333 | 0.620 | 0.713 |
| **T∪Q w/ bg seg.** | **28.181** | **0.320** | **0.646** | **0.731** |

### Key Findings

1. **Background segmentation is crucial for complex occlusions**: After adding background segments, SSIM improved from 0.713 to 0.731, and LPIPS decreased from 0.333 to 0.320.
2. **Significantly outperforms closed-category methods in the open world**: PD-MC completely fails when encountering unseen categories, while Pix2gestalt sometimes makes only minimal changes to the input.
3. **The combined prompt $T \cup Q$ is most effective in practical scenarios**: Although Q-only achieves the highest CLIP score (due to direct label matching), it is not applicable to abstract queries.
4. A human preference study shows that the proposed method achieves the highest preference rate in terms of reconstruction quality.

## Highlights & Insights

1. **The concept of "reasoning amodal completion" is forward-looking**: It allows for object completion with abstract queries (e.g., "What is the mammal in the image?"), leading to more natural human-computer interaction.
2. **The modular design of the training-free pipeline is excellent**: Each module can be upgraded independently (e.g., with better segmentation/inpainting models) while the framework remains unchanged.
3. **RGBA output format is downstream-friendly**: It can be directly utilized in image editing, 3D reconstruction, and AR scenarios.

## Limitations & Future Work

1. It relies on pre-trained image generation models, which may introduce artifacts (such as animal pose inconsistency).
2. For severe occlusions or ambiguous queries, the segmentation accuracy may be insufficient.
3. Evaluation metrics are limited in scenarios where ground-truth amodal data is lacking.
4. Exploring the end-to-end joint optimization of occlusion reasoning and the generation process is a promising direction.

## Related Work & Insights

- **PD-MC**: Utilizes pre-trained models but is limited to predefined categories; ours removes the category constraint.
- **Pix2gestalt**: A supervised learning method that requires a large amount of training data and exhibits limited generalization.
- **LISA**: A language-guided segmentation model; ours leverages it for initial visible mask generation.
- **InstaOrderNet**: An occlusion order reasoning model, which enables ours to determine which regions occlude the target.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to extend amodal completion to the open-world + language-guided setting.
- **Experimental Thoroughness**: ⭐⭐⭐ — Includes quantitative comparisons and human evaluations, but the evaluation dataset is relatively small and lacks ground truth.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition and detailed pipeline description.
- **Value**: ⭐⭐⭐⭐ — Possesses practical value for AR/3D reconstruction, and the modular, training-free design is easy to deploy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos](depthcrafter_generating_consistent_long_depth_sequences_for_open-world_videos.md)
- [\[CVPR 2025\] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)
- [\[NeurIPS 2025\] Towards 3D Objectness Learning in an Open World](../../NeurIPS2025/3d_vision/towards_3d_objectness_learning_in_an_open_world.md)
- [\[ICCV 2025\] Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting](../../ICCV2025/3d_vision/contact-aware_amodal_completion_for_human-object_interaction_via_multi-regional_.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](../../ICCV2025/3d_vision/amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
