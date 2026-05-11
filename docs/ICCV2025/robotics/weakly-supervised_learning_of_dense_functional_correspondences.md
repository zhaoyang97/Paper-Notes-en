---
title: >-
  [Paper Note] Weakly-Supervised Learning of Dense Functional Correspondences
description: >-
  [ICCV 2025][Robotics][Dense Functional Correspondence] This paper defines the task of *Dense Functional Correspondence*—establishing pixel-level dense correspondences between objects of different categories based on shar…
tags:
  - "ICCV 2025"
  - "Robotics"
  - "Dense Functional Correspondence"
  - "Weakly-Supervised Learning"
  - "Vision-Language Models"
  - "Contrastive Learning"
  - "Robotic Manipulation"
date: 2026-05-08
content_hash: 4f45338f04f9e0d5
---

# Weakly-Supervised Learning of Dense Functional Correspondences

**Conference**: ICCV 2025
**arXiv**: [2509.03893](https://arxiv.org/abs/2509.03893)
**Code**: [Project Page](https://dense-functional-correspondence.github.io/)
**Area**: Robotics
**Keywords**: Dense Functional Correspondence, Weakly-Supervised Learning, Vision-Language Models, Contrastive Learning, Robotic Manipulation

## TL;DR

This paper defines the task of *Dense Functional Correspondence*—establishing pixel-level dense correspondences between objects of different categories based on shared functionality (e.g., "pouring")—and proposes a weakly-supervised learning framework that distills functional and structural knowledge into a new model via VLM-based pseudo-labeling of functional parts combined with multi-view contrastive learning.

## Background & Motivation

Establishing pixel-level correspondences across images is fundamental to shape reconstruction, image editing, and robotic manipulation. Existing approaches face increasing challenges across three levels of difficulty:

**Same object, different viewpoints**: Multi-view correspondence; relatively straightforward.

**Same category, different instances**: E.g., correspondence between two cats; existing methods (NOCS, keypoint detection) already provide reasonable solutions.

**Different categories**: E.g., functional correspondence between a kettle and a bottle; **the most challenging yet practically important scenario**.

**Key Insight**: "Form follows function"—parts of objects that perform similar functions (e.g., a kettle's spout and a bottle's mouth) tend to share shape and appearance similarities, even when the overall objects look very different. This provides a natural bridge for establishing dense correspondences across object categories.

Limitations of prior work:

- **Self-supervised representations** (DINOv2, Stable Diffusion): Effective for intra-category correspondence but suffer significant accuracy degradation across categories.
- **Vision-language models** (CogVLM, ManipVQA): Capable of zero-shot detection of functional part bounding boxes, but unable to perform fine-grained pixel-level correspondence reasoning.
- **Keypoint methods** (Lai et al.): Define only 5 keypoints, insufficient to capture subtle similarities between highly dissimilar objects.
- **Affordance learning**: Identifies interaction regions within a single image; cannot establish dense correspondences across images.

## Method

### Overall Architecture

The method consists of three stages:

1. **Evaluation dataset construction**: 2D dense functional correspondence annotations are derived via 3D object alignment.
2. **Training dataset construction**: Large-scale functional part labels are obtained using VLM (CogVLM) pseudo-labeling with GPT-4-generated prompts.
3. **Model training**: A function-conditioned MLP is trained on top of frozen DINOv2 features, combining a functional part contrastive loss and a multi-view spatial contrastive loss.

### Key Designs

1. **Formal Definition of Dense Functional Correspondence**:

    - Provides a rigorous mathematical definition that transforms "functional similarity" into a computable 3D distance.
    - **Mechanism**: Given a function $\mathcal{F}$ and an image pair $(I_1, I_2)$, a functional correspondence mapping $f(I_1, I_2; \mathcal{F}): M(I_1;\mathcal{F}) \to M(I_2;\mathcal{F})$ is defined to minimize $\sum_{p \in M(I_1;\mathcal{F})} \|\pi^{-1}(p) - \pi^{-1}(f(p))\|_2$, where $\pi^{-1}$ denotes back-projection from pixel to 3D surface point.
    - **Design Motivation**: Defining 2D correspondences through 3D alignment avoids the infeasibility of manual dense annotation and naturally provides an evaluation benchmark.

2. **VLM Pseudo-Labeling Pipeline**:

    - **Function**: Automatically annotates functional parts of Objaverse 3D assets using large-scale pre-trained VLMs.
    - **Mechanism**: GPT-4 generates category–function–part text prompts → CogVLM predicts bounding boxes on multi-view renderings → 2D labels are back-projected and aggregated onto 3D point clouds → post-processing yields 2D pixel-level pseudo-labels. The pipeline covers 24 functional categories, 160 object categories, and 8,285 3D assets.
    - **Design Motivation**: Manual annotation of dense correspondences is infeasible; the pipeline leverages the zero-shot capability of VLMs for pseudo-labeling, with multi-view aggregation and 3D consistency to improve label quality.

3. **Function-Conditioned MLP + Dual Contrastive Learning**:

    - **Function**: Trains a function-text-conditioned feature extraction network to jointly learn functional semantics and spatial structure.
    - **Mechanism**: The model $g_\theta(p|I,\mathcal{F})$ consists of a 3-layer MLP on top of weighted multi-layer DINOv2 features and CLIP text embeddings. Training incorporates:
        - **Functional part contrastive loss** $\mathcal{L}_{func}$: InfoNCE loss where functional part pixels form positive pairs and non-functional part pixels form negative pairs, with non-functional pixels also mutually repelled:
      $$\mathcal{L}_{func} = -\log\frac{e^{\text{sim}(p_1^+, p_2^+)/\tau}}{e^{\text{sim}(p_1^+, p_2^+)/\tau} + e^{\text{sim}(p_1^+, p_2^-)/\tau} + e^{\text{sim}(p_1^-, p_2^-)/\tau}}$$
        - **Multi-view spatial contrastive loss** $\mathcal{L}_{spatial}$: Corresponding pixels of the same object across different viewpoints serve as positive pairs, preventing feature collapse:
      $$\mathcal{L}_{spatial} = -\log\frac{e^{\text{sim}(q, q_+^\prime)/\tau}}{e^{\text{sim}(q, q_+^\prime)/\tau} + e^{\text{sim}(q, q_-^\prime)/\tau}}$$
        - An optional **mask prediction loss** $\mathcal{L}_{mask}$.
    - **Design Motivation**: Using the functional contrastive loss alone leads to mode collapse (all spout features become identical); the spatial contrastive loss preserves structural information (top and bottom of a spout should have distinct features). The two losses are complementary and indispensable.

### Loss & Training

- Final loss: $\mathcal{L} = \mathcal{L}_{func} + \lambda_{spatial}\mathcal{L}_{spatial} + \lambda_{mask}\mathcal{L}_{mask}$
- $\lambda_{spatial} = 10$, $\lambda_{mask} = 1$
- DINOv2-B backbone is frozen; only the MLP (3 layers, 1024-dim hidden) is trained.
- Adam optimizer, learning rate $1 \times 10^{-4}$, batch size of 50 image pairs, 128 sampled points per image.
- Random color background augmentation is applied.

## Key Experimental Results

### Main Results

**Synthetic evaluation dataset (1,800+ image pairs, 24 functions, 85% cross-category):**

| Method | Norm.Dist↓ | PCK@23p↑ | Best F1@23p↑ | AP@23p↑ |
|--------|-----------|---------|-------------|--------|
| Chance | 0.310 | 0.165 | 0.416 | 0.256 |
| DINO | 0.212 | 0.381 | 0.578 | 0.381 |
| SD-DINO | 0.227 | 0.376 | 0.563 | 0.341 |
| CogVLM + DINO | 0.180 | 0.416 | 0.678 | 0.556 |
| **Ours (full)** | **0.170** | **0.486** | **0.768** | **0.685** |

**Real-world evaluation dataset (HANDAL, 500+ image pairs, 13 functions):**

| Method | Norm.Dist↓ | PCK@23p↑ | Best F1@23p↑ | AP@23p↑ |
|--------|-----------|---------|-------------|--------|
| DINO | 0.206 | 0.408 | 0.589 | 0.382 |
| CogVLM + DINO | 0.172 | 0.440 | 0.695 | 0.561 |
| **Ours (full w/ mask)** | **0.153** | **0.501** | **0.808** | **0.730** |

### Ablation Study

| Configuration | Norm.Dist↓ | PCK@23p↑ | AP@23p↑ | Note |
|--------------|-----------|---------|--------|------|
| Functional only | 0.228 | 0.287 | 0.441 | Mode collapse; structural information lost |
| Spatial only | 0.204 | 0.470 | 0.412 | Lacks functional semantics |
| Full (w/o mask) | 0.170 | 0.486 | 0.685 | Functional + spatial are complementary |
| Full (w/ mask) | 0.172 | 0.480 | 0.684 | Mask loss improves real-world performance |

### Key Findings

1. **Training on purely synthetic data generalizes to real images**: Models trained on synthetic Objaverse data perform well on the HANDAL real-world dataset.
2. **Functional and spatial contrastive losses are mutually necessary**: Using functional loss alone causes mode collapse (PCK only 0.287); spatial loss alone lacks functional understanding (AP only 0.412).
3. **Inference speed advantage**: The model runs approximately 50× faster than CogVLM and approximately 1,000× faster than ManipVQA.
4. Prompting ManipVQA with function names (ManipVQA-F) performs substantially worse than prompting with part names (ManipVQA-P), indicating that zero-shot functional reasoning remains difficult.

## Highlights & Insights

- **Task definition contribution**: The paper is the first to formally define the *Dense Functional Correspondence* task, filling the gap between sparse functional keypoint correspondence and intra-category dense correspondence.
- **Clever data strategy**: GPT-4 and CogVLM are combined for large-scale pseudo-labeling, with 3D consistency verification to minimize human intervention.
- **Complementary distillation**: The semantic understanding capability of VLMs and the spatial correspondence capability of DINOv2 are jointly distilled into a lightweight MLP.
- **Evaluation methodology**: Ground-truth 2D dense correspondences are automatically derived via 3D object alignment, providing a scalable evaluation pipeline.
- **Practical value**: The approach has direct applicability to robotic imitation learning, e.g., transferring manipulation demonstrations from one object to another of a different category.

## Limitations & Future Work

1. The method assumes input images have already been segmented; additional segmentation modules would be required in practical deployment.
2. The granularity of functional categories (24 functions) may be insufficient to cover all real-world needs.
3. Objaverse asset quality varies; functional part annotations for some assets may contain noise.
4. Evaluation is limited to tool-type and container-type objects; applicability to broader object categories (e.g., furniture, electronic devices) remains to be validated.
5. The practical effectiveness of functional correspondences in downstream robotic manipulation tasks has not been explored.

## Related Work & Insights

- Compared to intra-category correspondence methods such as NOCS, functional correspondence operates at a higher level of abstraction that transcends category boundaries.
- The paradigm of VLM pseudo-labeling combined with contrastive learning distillation is generalizable to other tasks requiring dense annotations with high annotation costs.
- The idea of defining 2D correspondences via 3D alignment is extensible to other correspondence tasks involving 3D structure.
- Distinction from affordance grounding: affordance focuses on "how to interact with an object," whereas functional correspondence focuses on "aligning functionally equivalent parts across different objects."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Novel task definition + innovative data construction pipeline + elegant training methodology
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on both synthetic and real-world datasets with clear ablations, but lacks downstream task validation
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, method pipeline is coherent, and figures are well-crafted
- Value: ⭐⭐⭐⭐⭐ Opens a new research direction with significant application potential in robotic manipulation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Selective Contrastive Learning for Weakly Supervised Affordance Grounding](selective_contrastive_learning_for_weakly_supervised_affordance_grounding.md)
- [\[ICCV 2025\] Self-supervised Learning of Hybrid Part-aware 3D Representations of 2D Gaussians and Superquadrics](self-supervised_learning_of_hybrid_part-aware_3d_representations_of_2d_gaussians.md)
- [\[ICCV 2025\] iManip: Skill-Incremental Learning for Robotic Manipulation](imanip_skill-incremental_learning_for_robotic_manipulation.md)
- [\[ICCV 2025\] Rep-MTL: Unleashing the Power of Representation-Level Task Saliency for Multi-Task Learning](rep-mtl_unleashing_the_power_of_representation-level_task_saliency_for_multi-tas.md)
- [\[ICCV 2025\] TesserAct: Learning 4D Embodied World Models](learning_4d_embodied_world_models.md)

</div>

<!-- RELATED:END -->
