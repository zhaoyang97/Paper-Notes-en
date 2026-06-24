---
title: >-
  [Paper Note] V-CLR: View-Consistent Learning for Open-World Instance Segmentation
description: >-
  [CVPR 2025][Segmentation][Open-world instance segmentation] v-CLR proposes a view-consistent learning framework. By transforming natural images into appearance-invariant views such as depth maps and stylized images, enforcing cross-view query feature consistency within a DETR architecture, and utilizing unsupervised object proposals to guide the matching direction, the framework effectively overcomes the texture bias of detection networks. It achieves state-of-the-art perform…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Open-world instance segmentation"
  - "appearance-invariant representations"
  - "view consistency"
  - "texture bias"
  - "cross-category generalization"
date: 2026-05-08
content_hash: 9ee67c3880483f6c
---

# V-CLR: View-Consistent Learning for Open-World Instance Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2504.01383](https://arxiv.org/abs/2504.01383)  
**Code**: [https://visual-ai.github.io/vclr](https://visual-ai.github.io/vclr)  
**Area**: Segmentation  
**Keywords**: Open-world instance segmentation, appearance-invariant representations, view consistency, texture bias, cross-category generalization

## TL;DR

v-CLR proposes a view-consistent learning framework. By transforming natural images into appearance-invariant views such as depth maps and stylized images, enforcing cross-view query feature consistency within a DETR architecture, and utilizing unsupervised object proposals to guide the matching direction, the framework effectively overcomes the texture bias of detection networks. It achieves state-of-the-art performance on multiple open-world segmentation benchmarks.

## Background & Motivation

**Background**: Open-world instance segmentation requires a model, trained on known classes, to discover and segment objects of unknown classes during inference. Existing methods mainly include: OLN, which replaces the classification branch with localization-aware scores; LDET, which synthesizes training images; and SWORD, which applies the DETR architecture to the open-world setting.

**Limitations of Prior Work**: Numerous studies demonstrate that visual neural networks naturally bias towards learning appearance information (particularly texture) to recognize objects. This implicit bias causes models to fail in the open world when encountering novel objects with unseen textures. For instance, a detector trained on red metallic objects shows a significantly decreased detection rate for objects of other colors and materials.

**Key Challenge**: Textures and appearances in training data are coupled with object identities. Models exploit textures as "shortcut features" rather than learning true structural features of objects. This severely limits the generalization capability from known to unknown classes.

**Goal**: To enable the model to learn appearance-invariant yet object-relevant representations, allowing it to generalize to unknown objects with arbitrary textures and appearances.

**Key Insight**: The authors validate their hypothesis through a toy experiment on the CLEVR dataset: when depth maps are incorporated as auxiliary inputs, the detection rate for objects with novel colors or materials improves substantially. This demonstrates that introducing appearance-invariant information can indeed significantly enhance generalization.

**Core Idea**: Transform images into multiple appearance-invariant views (depth maps, stylized images, edge maps), enforce appearance-invariant representation learning using cross-view query feature consistency loss within a DETR framework, and utilize unsupervised object proposals to ensure that the consistency optimization is directed toward actual objects.

## Method

### Overall Architecture

v-CLR consists of two branches: a natural image branch (EMA teacher) that always receives the original images, and a transformed image branch (student) that randomly receives either the original images or transformed views (e.g., depth maps or stylized images). Each branch produces query predictions using a DETR variant. Object proposals pre-trained with CutLER are used to associate the queries from both branches with real objects, which then enforces feature similarity between matched query pairs. Meanwhile, the student branch is trained using ground-truth labels for the detection loss.

### Key Designs

1. **Appearance-Invariant Transformation**:

    - **Function**: Transforms natural images into multiple views that destroy appearance but preserve structure.
    - **Mechanism**: Utilizes three types of views: natural images, colorized depth maps (estimated via MiDaS and colorized), and auxiliary views (artistic style transfer or edge maps). During training, one view is chosen at random with equal probability for each sample. Additionally, random patch cropping and paste-back are applied to further disrupt appearance consistency.
    - **Design Motivation**: Depth maps preserve the 3D geometric structure of objects while completely rewriting texture and color information, making them the most ideal appearance-invariant transformation. The random combination of multiple transformations enhances training diversity and prevents the model from relying on features from any single domain.

2. **Object Feature Matching**:

    - **Function**: Enforces consistency between the output query features of both branches when they correspond to the same object.
    - **Mechanism**: Uses a Cascade-Mask-RCNN pre-trained with CutLER to generate object proposals $\mathcal{P}_o$. Hungarian matching is applied to pair the predictions of both branches $\mathcal{P}_1, \mathcal{P}_2$ with the proposals, forming query triplets. Then, the cosine similarity loss of matched query pairs is computed as $L_{sim} = \frac{1}{\tilde{N}} \sum (1 - \cos(q_1, q_2))$, while the student branch's detection output $L_{obj}$ is supervised using the masks/boxes of the proposals.
    - **Design Motivation**: Simply forcing consistency across all query pairs may lead to "shortcut solutions," where the model extracts similar features without actually focusing on objects. Object proposals serve as "object anchors" to ensure that the consistency optimization points toward actual objects, making the learned invariant representations object-relevant.

3. **EMA Teacher-Student Architecture**:

    - **Function**: Prevents feature collapse of the two branches.
    - **Mechanism**: Transformer parameters of the natural image branch are updated via an EMA (Exponential Moving Average) of the transformed image branch's parameters, without receiving direct gradients. This follows established collapse-prevention strategies in self-supervised learning (e.g., BYOL, DINO).
    - **Design Motivation**: If both branches share a single set of parameters or are updated simultaneously via gradients, the features will quickly degenerate into a trivial solution. EMA provides a stable teacher signal, guaranteeing the stability of the learning process.

### Loss & Training

- Total loss: $L = \lambda_{match} L_{match} + \lambda_{gt} L_{gt}$
- Matching loss: $L_{match} = \lambda_{obj} L_{obj} + \lambda_{sim} L_{sim}$, where $L_{obj}$ contains dice loss, mask loss, score loss, box loss, and GIoU loss.
- GT loss: $L_{gt}$ uses ground-truth labels to compute detection and segmentation losses of the same form as $L_{obj}$.
- Trained for 8 epochs, with learning rate decayed at the 7th epoch; uses DINO-DETR + ResNet-50 backbone, with 1000/1500 queries for decoder.

## Key Experimental Results

### Main Results

| Setting | Metric | v-CLR (DINO) | SWORD | Gain |
|------|------|------|----------|------|
| VOC→Non-VOC | AR100_b | 40.9 | 35.3 | +5.6 |
| VOC→Non-VOC | AR100_m | 34.1 | 30.2 | +3.9 |
| VOC→UVO | AR100_b | 47.2 | 43.1 | +4.1 |
| VOC→UVO | AR100_m | 35.9 | 34.9 | +1.0 |
| COCO→LVIS | AR100_b | 28.4 | 23.5 | +4.9 |
| COCO→LVIS | AR100_m | 23.6 | 20.4 | +3.2 |
| COCO→Objects365 | AR100_b | 48.9 | - | - |

### Ablation Study

| Configuration | AR100_b (VOC→Non-VOC) | Description |
|------|---------|------|
| v-CLR (DINO) | 40.9 | Full model |
| w/o L_sim | ~36 | Removing cross-view consistency significantly degrades performance |
| w/o object proposals | ~35 | Without proposal guidance, consistency degenerates |
| w/o depth view | ~38 | Using only style-transfer transformations, less effective than depth maps |
| DINO-DETR baseline | 31.1 | Without any open-world enhancements |

### Key Findings

- Depth maps are the most effective appearance-invariant transformation because they completely destroy textures while preserving all geometric structures.
- Object proposals are critical to preventing consistency learning from collapsing; without them, cross-view consistency may instead lead the model to learn invariant features unrelated to objects.
- v-CLR built on DINO-DETR performs better than that on Deformable-DETR, and denoising queries help accelerate training convergence.
- In all settings, the improvement in AR@10 is greater than that in AR@100, indicating that v-CLR is particularly adept at high-confidence top-k object detection.

## Highlights & Insights

- **Guiding the direction of consistency learning with object proposals** is highly ingenious, resolving the "shortcut solution" problem in self-supervised consistency learning. This idea can be transferred to any scenario requiring the learning of invariant representations.
- **The concept of view transformation is not limited to domain adaptation**: Traditional multi-domain/style-transfer methods seek cross-domain consistency of the same semantic class, whereas this work seeks cross-appearance consistency of the same object instance, which are orthogonal problems. This paper explicitly points out that domain shift and semantic shift are distinct challenges.
- The **CLEVR toy experiment** validates the core hypothesis concisely and powerfully, serving as a highlight in the paper's narrative structure.

## Limitations & Future Work

- Relies on a pre-trained CutLER object proposal network; if the proposals themselves are biased, it limits the performant upper bound.
- The quality of depth maps is limited by MiDaS, and depth estimation may be inaccurate in extreme scenarios (such as strong reflections or highly textured surfaces).
- Only verified using the ResNet-50 backbone; stronger backbones (e.g., Swin-T, ViT) could yield further improvements.
- Future directions: Replacing CutLER with SAM/DINOv2 as the proposal source may yield better object coverage; combining proposals with 3D geometry (e.g., point cloud clustering) to handle outdoor scenarios.

## Related Work & Insights

- **vs SWORD**: SWORD first applied DETR to open-world segmentation, proposing stop-gradients, an IoU branch, and one-to-many assignment. v-CLR demonstrates that cross-view consistency learning addresses generalization more fundamentally than these techniques by eliminating texture bias at the representation level.
- **vs OLN**: OLN replaces the classification branch with localization scores, which is a model-level modification. v-CLR solves the problem from the data/learning strategy level, making the two approaches orthogonal and combinable.
- **vs Texture Bias Studies (Geirhos et al., SIN)**: Previous studies on texture-shape bias primarily focused on classification tasks, while this work systematically applies it to open-world scenarios in detection/segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ Combining appearance-invariant learning with object-aware constraints to solve open-world segmentation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 settings across 5 datasets with complete ablation studies and a CLEVR toy experiment validating the hypothesis.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clear, and the CLEVR experiment serves as an excellent narrative starting point.
- Value: ⭐⭐⭐⭐ Provides an effective new paradigm for open-world instance segmentation, with highly transferable concepts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SOS: Segment Object System for Open-World Instance Segmentation With Object Priors](../../ECCV2024/segmentation/sos_segment_object_system_for_open-world_instance_segmentation_with_object_prior.md)
- [\[CVPR 2025\] ROCKET-1: Mastering Open-World Interaction with Visual-Temporal Context Prompting](rocket-1_mastering_open-world_interaction_with_visual-temporal_context_prompting.md)
- [\[CVPR 2026\] Learning Cross-View Object Correspondence via Cycle-Consistent Mask Prediction](../../CVPR2026/segmentation/learning_cross-view_object_correspondence_via_cycle-consistent_mask_prediction.md)
- [\[CVPR 2025\] Foveated Instance Segmentation](foveated_instance_segmentation.md)
- [\[CVPR 2025\] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)

</div>

<!-- RELATED:END -->
