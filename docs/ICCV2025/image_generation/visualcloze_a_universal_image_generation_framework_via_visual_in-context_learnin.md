---
title: >-
  [Paper Note] VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning
description: >-
  [ICCV 2025][Image Generation][universal image generation] This paper presents VisualCloze, which unifies diverse image generation tasks under a "visual cloze" paradigm—defining tasks via visual in-context examples rather than text instructions, performing unified generation through an image infilling model, and constructing the Graph200K graph-structured dataset to enhance cross-task knowledge transfer. The framework supports in-distribution tasks, unseen-task generalization, multi-task composition, and reverse generation.
tags:
  - ICCV 2025
  - Image Generation
  - universal image generation
  - visual in-context learning
  - image infilling
  - Graph200K
  - multi-task unification
date: 2026-05-08
content_hash: 2d0c01ca74caa0ea
---

# VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning

**Conference**: ICCV 2025
**arXiv**: [2504.07960](https://arxiv.org/abs/2504.07960)
**Code**: [https://visualcloze.github.io/](https://visualcloze.github.io/)
**Area**: Image Generation / Unified Framework / In-Context Learning
**Keywords**: universal image generation, visual in-context learning, image infilling, Graph200K, multi-task unification

## TL;DR

This paper presents VisualCloze, which unifies diverse image generation tasks under a "visual cloze" paradigm—defining tasks via visual in-context examples rather than text instructions, performing unified generation through an image infilling model, and constructing the Graph200K graph-structured dataset to enhance cross-task knowledge transfer. The framework supports in-distribution tasks, unseen-task generalization, multi-task composition, and reverse generation.

## Background & Motivation

Diffusion models have driven rapid advances in image generation, enabling a wide range of applications including conditional generation, style transfer, virtual try-on, and personalized generation. However, the dominant paradigm still trains task-specific models for each application (e.g., ControlNet for conditional generation, IP-Adapter for style preservation, InstructPix2Pix for editing), which is inefficient and difficult to scale.

Although unified generation models have made progress, they face three core challenges:

**Ambiguity in task instructions**: Existing methods (OmniGen, ACE++) rely on text instructions or task-specific tokens to distinguish tasks, but the complexity of visual tasks and the vision-language gap frequently cause **task confusion**, with poor generalization to unseen tasks.

**Sparsity of visual task distributions**: Unlike NLP tasks, which overlap heavily, visual tasks (segmentation, depth estimation, super-resolution, etc.) draw on nearly disjoint datasets, isolating task-specific knowledge and impeding the learning of transferable shared representations.

**Absence of a unified architecture design**: A framework must simultaneously support flexible task formats and be compatible with state-of-the-art pretrained models.

**Core insights**: (1) Visual in-context learning (Visual ICL) is better suited than text instructions for defining visual tasks—directly demonstrating input–output examples lets the model "see" the task definition; (2) the objective of unified image generation (filling in a target image region) is naturally aligned with that of image infilling, enabling direct exploitation of the strong generative prior of pretrained infilling models.

## Method

### Overall Architecture

VisualCloze reformulates all image generation tasks as a **visual cloze** problem:

- Given $C$ in-context examples (each containing a task instance of $L$ images) and a query ($L-1$ conditioning images plus one blank target slot),
- All images are concatenated into a grid of size $(L \times W,\, (C+1) \times H)$,
- A binary mask $M$ marks the target position, and an infilling model completes the target region.

Formally: $\hat{X} = f(X \mid T, M)$, where $X$ is the concatenated image, $T$ is the language instruction, and $M$ is the mask condition.

### Key Designs

#### 1. Visual In-Context Learning

Rather than describing the task type in text, the model is provided with 1–2 (input, output) demonstration pairs as task exemplars. The model infers the task definition by observing the transformation relationship in the examples, yielding four key advantages:

- **Reduced task ambiguity**: Visual examples convey task intent more precisely than text; increasing the number of examples further reduces confusion.
- **Generalization to unseen tasks**: Novel transformation types not seen during training can be executed solely via examples, without any retraining.
- **Multi-task composition**: Multiple sub-tasks (e.g., depth-guided generation + relighting) can be merged into a single unseen task.
- **Reverse generation**: The model can infer conditions from targets (e.g., decomposing a stylized image into content and style references).

During training, up to $C=2$ in-context examples are randomly provided; at inference time this can be extended to more.

#### 2. Graph200K Dataset

Starting from Subject200K, 49 annotations are constructed for each image, spanning five meta-tasks:

- **Conditional generation**: 12 condition types (Canny edge, HED edge, depth, normal, keypose, SAM2 mask, foreground segmentation, open-world detection boxes, etc.)
- **Image restoration**: 32 online degradation augmentations
- **Image editing**: background-preserving editing (object replacement) and background-changing editing
- **IP preservation**: subject-driven generation
- **Style transfer**: semantics-preserving (InstantStyle) and semantics-varying (FLUX.1-Redux) settings

The dataset is constructed as a **strongly connected graph**—every pair of nodes is connected by a bidirectional path, with intermediate nodes serving as conditions and terminal nodes as targets. Through combinatorial sampling, up to **134 highly overlapping tasks** can be derived, substantially increasing task density and cross-task transfer.

Supplementary data include: VITON-HD (virtual try-on), PhotoDoodle (artistic editing), OmniEdit (object addition/removal), as well as painting process and multi-view generation data.

#### 3. Infilling-Based Unified Architecture

A core finding is that the unified generation formulation of VisualCloze is **naturally aligned** with the objective of image infilling models—both complete masked regions given context. Consequently:

- The model is fine-tuned directly on FLUX.1-Fill-dev without architectural modifications or task-specific modules.
- LoRA (rank=256) fine-tuning is used to reduce training cost while preserving base model capabilities.
- The LoRA can be merged with other community LoRAs.

For positional encoding, the 3D-RoPE of FLUX.1-Fill-dev is exploited by concatenating different examples along the temporal dimension, accommodating examples with varying aspect ratios.

Language instructions comprise three components: (1) layout instructions describing the grid arrangement; (2) task instructions specifying the task type; and (3) content instructions describing the target image content.

### Loss & Training

- Standard flow matching loss, computed only within the masked region of the concatenated grid image.
- Lognorm noise scheduling and dynamic time shifting following FLUX.1-Fill-dev.
- During training, one of the first $L-1$ grid slots is randomly masked with probability 0.5 to promote reverse generation capability.
- AdamW optimizer, learning rate $1\text{e-}4$, accumulated batch size 64, trained for 20k steps on $8\times$ A100 GPUs.
- Images are uniformly resized to $384\times384$ or $512\times512$ area before concatenation.

## Key Experimental Results

### Main Results

Quantitative comparison on conditional generation and image restoration:

| Condition | Method | Context | Controllability (F1/RMSE) | FID↓ | SSIM↑ | MAN-IQA↑ | MUSIQ↑ | CLIP-Score↑ |
|-----------|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Canny | ControlNet (task-specific) | - | 0.13 | 46.06 | 0.34 | 0.31 | 45.45 | 34.10 |
| Canny | OminiControl | - | 0.47 | 29.58 | 0.61 | 0.44 | 61.40 | 34.40 |
| Canny | OmniGen | - | 0.43 | 51.58 | 0.47 | 0.47 | 62.66 | 33.66 |
| Canny | **Ours_fill** | 0 | 0.35 | 30.60 | 0.55 | **0.49** | **64.39** | **34.98** |
| Canny | **Ours_fill** | 2 | 0.36 | 31.15 | 0.56 | **0.49** | 64.08 | 34.85 |
| Depth | ControlNet (task-specific) | - | 23.70 | 36.83 | 0.41 | 0.44 | 60.17 | 34.49 |
| Depth | OneDiffusion | - | 10.35 | 39.03 | 0.49 | 0.49 | 60.49 | 34.71 |
| Depth | **Ours_fill** | 0 | **10.31** | **33.88** | **0.54** | 0.48 | **64.85** | **35.10** |
| Depth | **Ours_fill** | 2 | **9.68** | 34.88 | 0.54 | 0.48 | 64.29 | 34.89 |
| Deblur | OminiControl | - | 19.70 | 26.17 | 0.85 | 0.45 | 60.70 | 34.53 |
| Deblur | **Ours_fill** | 2 | 25.57 | **36.28** | 0.76 | **0.48** | **61.77** | **34.82** |

Ours_fill leads comprehensively in visual quality (MAN-IQA, MUSIQ) and text alignment (CLIP-Score), and achieves the best controllability in depth-to-image generation (RMSE 9.68).

Quantitative comparison on subject-driven generation:

| Method | Context | DINOv2↑ | CLIP-I↑ | CLIP-T↑ |
|--------|:---:|:---:|:---:|:---:|
| OminiControl (task-specific) | - | 73.17 | 87.70 | 33.53 |
| OneDiffusion | - | 73.88 | 86.91 | 34.85 |
| OmniGen | - | 67.73 | 83.43 | 34.53 |
| Ours_dev | 0 | 78.05 | 87.68 | 35.06 |
| **Ours_fill** | 0 | **80.41** | **89.63** | **35.16** |
| **Ours_fill** | 2 | 80.32 | 89.36 | 35.01 |

Compared to the task-specific model OminiControl, DINOv2, CLIP-I, and CLIP-T improve by 7.15%, 1.66%, and 1.48%, respectively.

Quantitative comparison on style transfer:

| Method | Text↑ | Image↑ |
|--------|:---:|:---:|
| InstantStyle (task-specific) | 0.27 | **0.60** |
| OmniGen | 0.27 | 0.52 |
| **Ours_fill** | **0.29** | 0.55 |

Text alignment surpasses the task-specific model InstantStyle by 2%; style consistency is marginally lower.

### Ablation Study

Infilling model vs. dev model (core ablation):

| Task | Metric | Ours_dev | Ours_fill | Gain |
|------|--------|:---:|:---:|:---:|
| Depth→Image | RMSE↓ | 25.06 | **10.31** | −58.8% |
| Depth→Image | FID↓ | 42.14 | **33.88** | −19.6% |
| Deblur | RMSE↓ | 25.03 | 26.53 | — |
| Deblur | MUSIQ↑ | 46.68 | **59.62** | +27.7% |
| Subject-driven | DINOv2↑ | 78.05 | **80.41** | +3.0% |
| Subject-driven | CLIP-I↑ | 87.68 | **89.63** | +2.2% |

Ours_fill (based on FLUX.1-Fill-dev) significantly outperforms Ours_dev (based on FLUX.1-dev) on most tasks, validating the critical role of infilling objective alignment. Qualitative comparisons reveal that Ours_dev frequently produces diagonal stripe artifacts in depth-to-image generation.

Effect of the number of in-context examples (depth-to-image task):

| Context count | RMSE↓ | FID↓ | SSIM↑ | MAN-IQA↑ |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 10.31 | 33.88 | 0.54 | 0.48 |
| 1 | 9.91 | 34.44 | 0.54 | 0.49 |
| 2 | **9.68** | 34.88 | 0.54 | 0.48 |

Adding in-context examples further improves controllability (RMSE decreases from 10.31 to 9.68), though not all metrics improve monotonically.

### Key Findings

1. **In-context learning mitigates task confusion**: On tasks such as pose estimation and edge detection, the model occasionally produces noisy outputs without ICL examples; providing 1–2 examples substantially improves both performance and stability.
2. **Generalization to unseen tasks**: Although only object addition/removal editing tasks are seen during training, the model generalizes via ICL to unseen editing types such as environment modification and attribute transformation; similarly, training on single-subject generation generalizes to multi-subject-driven generation.
3. **Multi-task composition**: ICL enables the merging of multiple sub-tasks into a single inference step (e.g., Depth→Image + Relighting) without additional training.
4. **Reverse generation**: The model can decompose a stylized image back into the original content and style reference, or infer a real image, depth, and normal from an edge map.
5. **Sensitivity to example quality**: ICL examples must accurately convey task intent; if examples deviate from the task core (e.g., a side-face example that is too close in appearance to a frontal face), success rates drop significantly.

## Highlights & Insights

1. The **visual cloze = image infilling unified paradigm** is remarkably elegant: conceptually simple (all tasks are "fill in the blank"), requires no architectural changes (directly leverages a pretrained infilling model), and incurs minimal engineering overhead (LoRA fine-tuning for 20k steps).
2. **Visual ICL is better suited than text instructions for defining visual tasks**: This represents an important paradigm shift—allowing users to *show* rather than *describe* the desired transformation.
3. The **graph-structured design of Graph200K** cleverly addresses the visual task sparsity problem: 49 annotations and 134 overlapping tasks per image enable the model to learn compact shared representations.
4. **Four emergent capabilities**—in-distribution performance, unseen-task generalization, multi-task composition, and reverse generation—arise naturally from a single framework without separate design effort.

## Limitations & Future Work

1. **Resolution constraints**: Concatenating multiple images into a grid requires each image to be resized to $384\times384$ or $512\times512$, potentially losing fine-grained details.
2. **Controllability gap with task-specific models**: The F1 score on the Canny→Image task is lower than OminiControl (0.36 vs. 0.47); precise edge control remains inferior to dedicated models.
3. **Sensitivity to ICL example selection**: Generation quality varies considerably across different examples; automatic selection of optimal examples warrants further investigation.
4. **Graph200K data quality**: The dataset relies on an automated pipeline (Qwen2-VL for caption generation, FLUX.1-Fill for editing), which may introduce noisy annotations.
5. **Training data bias**: Only object addition/removal editing tasks are included in training; all other editing types depend entirely on generalization.
6. **Inference efficiency**: Processing large grid images incurs greater computational overhead than single-image inference.

## Related Work & Insights

- **OmniGen**: Unifies multiple tasks via a vision-language model and text instructions, but generalization to unseen tasks is limited; VisualCloze substantially improves generalization through visual ICL.
- **ControlNet / OminiControl**: Task-specific conditional injection modules, each requiring separate training per condition; VisualCloze constitutes a truly unified framework.
- **OneDiffusion**: Unifies diverse generation tasks, but achieves lower controllability and quality than VisualCloze.
- **UniReal**: Unifies image generation as discontinuous video generation; complementary to VisualCloze's infilling-based unification.
- **Insights**: The Visual ICL paradigm can be extended to video generation—demonstrating a few frames of video transformation as examples to define video editing tasks; the graph-structured dataset methodology can be transferred to multimodal learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The "visual cloze = infilling" unification formula is exceptionally elegant; using Visual ICL for task definition represents an important paradigm innovation.
- **Technical Quality**: ⭐⭐⭐⭐ — Graph200K is cleverly designed, and the analysis of infilling objective alignment is rigorous, though controllability still has room for improvement.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers conditional generation, restoration, style transfer, and subject-driven generation, with the critical dev-vs-fill ablation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Based on FLUX.1-Fill-dev LoRA fine-tuning with minimal engineering cost; LoRA weights are composable with community resources.
- **Value**: ⭐⭐⭐⭐⭐ — Defines a new paradigm for universal image generation; both Graph200K and the Visual ICL methodology carry long-term value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Pretrained Reversible Generation as Unsupervised Visual Representation Learning](pretrained_reversible_generation_as_unsupervised_visual_representation_learning.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)
- [\[ICCV 2025\] Less-to-More Generalization: Unlocking More Controllability by In-Context Generation](less-to-more_generalization_unlocking_more_controllability_by_in-context_generat.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[ICCV 2025\] OminiControl: Minimal and Universal Control for Diffusion Transformer](ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)

<!-- RELATED:END -->
