---
title: >-
  [Paper Note] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling
description: >-
  [ICCV 2025][3D Vision][zero-shot 3D scene modeling] This paper presents Diorama, the first zero-shot open-world system for 3D indoor scene modeling. By modularly composing foundation models (GPT-4o, SAM, DINOv2, Metric3D, etc.), Diorama converts a single RGB image into a complete, compositional 3D indoor scene containing architectural structures and CAD objects, requiring neither end-to-end training nor manual annotation.
tags:
  - ICCV 2025
  - 3D Vision
  - zero-shot 3D scene modeling
  - CAD retrieval
  - scene graph
  - pose estimation
  - layout optimization
date: 2026-05-08
content_hash: c8acb305785c52d1
---

# Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling

**Conference**: ICCV 2025
**arXiv**: [2411.19492](https://arxiv.org/abs/2411.19492)
**Code**: [https://3dlg-hcvc.github.io/diorama/](https://3dlg-hcvc.github.io/diorama/)
**Area**: 3D Vision
**Keywords**: zero-shot 3D scene modeling, CAD retrieval, scene graph, pose estimation, layout optimization

## TL;DR

This paper presents Diorama, the first zero-shot open-world system for 3D indoor scene modeling. By modularly composing foundation models (GPT-4o, SAM, DINOv2, Metric3D, etc.), Diorama converts a single RGB image into a complete, compositional 3D indoor scene containing architectural structures and CAD objects, requiring neither end-to-end training nor manual annotation.

## Background & Motivation

3D scene modeling from a single image is a fundamental task in computer vision. Existing approaches follow two main paradigms: (1) reconstruction methods (e.g., NeRF, 3DGS), which produce imperfect surface meshes unsuitable for simulation or interaction; and (2) CAD retrieval-and-alignment methods (e.g., Mask2CAD, ROCA, DiffCAD), which yield clean and compact outputs but rely on large amounts of annotated training data and are restricted to fixed category sets, precluding generalization to open-world scenarios.

The authors' core insight is: given that foundation models have already demonstrated 3D perception capabilities, can one assemble a complete single-view 3D scene modeling system purely by composing multiple pretrained models, without any training?

## Core Problem

How can a **complete** editable 3D scene (including both architectural structures and objects) be reconstructed from a single RGB image **without training any model**? The key challenges are: (1) open-vocabulary object recognition and localization; (2) understanding complex spatial support relationships; (3) retrieving and aligning inexact CAD models from a shape database; and (4) ensuring physical plausibility, i.e., preventing object interpenetration and floating.

## Method

### Overall Architecture

The system consists of two major components operating as a sequential pipeline:

**Component 1: Open-World Perception** (understanding what is in the image, where it is, and how objects relate spatially)
1. **Scene Parsing**: GPT-4o identifies all object categories → OWLv2 detects bounding boxes → SAM segments instance masks → Metric3DV2 estimates metric depth and surface normals → back-projection yields per-instance point clouds.
2. **Scene Graph Generation**: Objects in the image are annotated with Set-of-Mark indices and fed to GPT-4o to infer support relationships (e.g., "book on table," "painting on wall"), producing a scene graph $G=\langle V, E\rangle$.
3. **PlainRecon Architectural Reconstruction**: Furniture is removed via segmentation and inpainting to obtain an empty room → depth and normals are estimated → normal clustering (K-means + DBSCAN) partitions the point cloud into planar regions → RANSAC fits plane equations → rotating calipers compute minimum-area bounding rectangles. The output is a set of 3D planar meshes representing walls, floor, and other architectural elements.

**Component 2: CAD Scene Modeling** (assembling the 3D scene from CAD models)
1. **Multimodal Shape Retrieval**: DuoDuoCLIP encodes text, images, and 3D shapes into a joint embedding space. A text query first narrows candidates to the correct semantic category; an image query then re-ranks among candidates for the best geometric match.
2. **Zero-Shot 9DoF Pose Estimation**: 180 grayscale multi-view renders are generated per CAD model → DINOv2 features establish 2D correspondences via cyclical distance matching → the most similar viewpoint provides a coarse pose → depth maps lift 2D correspondences to 3D → Umeyama + RANSAC solves the rigid transformation → GigaPose's network supplements more robust scale prediction.
3. **Semantics-Aware Multi-Stage Scene Optimization**: A four-stage differentiable optimization progressively refines object poses.

### Key Designs

**PlainRecon** is an elegant engineering contribution. Existing planar reconstruction methods (e.g., RaC) fail in approximately 30% of cases due to constrained discrete optimization, whereas PlainRecon performs bottom-up plane fitting via normal clustering, achieving a 100% success rate and running more than 20× faster (23 s vs. 543 s).

**Hierarchical Retrieval Strategy** addresses a practical challenge: in a database of 60,000 CAD models, purely visual retrieval may confuse books with cardboard boxes (geometrically similar but semantically distinct). Locking the category with text first and then re-ranking by image is highly effective.

**Zero-Shot Pose Estimation via DINOv2 Correspondences** is among the most central technical contributions. Without any pose-annotated data, semantic ViT features establish correspondences between image patches and CAD renders; the cyclical distance mechanism enforces mutual consistency of correspondences.

### Loss & Training

The system **requires no training**, but the scene optimization stage employs explicit differentiable objectives:
- **Stage 1 (Orientation)**: $e_1 = 3 \cdot e_{\text{align}} + e_{\text{sem}}$, aligning contact-surface normals with support-surface normals while preserving object orientation.
- **Stage 2 (Placement)**: $e_2 = 5 \cdot e_{\text{place}} + e_{\text{rel}}$, ensuring objects rest on their support surfaces while maintaining relative inter-object positions.
- **Stage 3 (Space)**: $e_3 = e_{\text{vol}}$, penalizing objects that extend beyond their support volume (e.g., books protruding from a bookshelf).
- **Stage 4 (Refinement)**: $e_4 = 5 \cdot e_{\text{place}} + e_{\text{col}}$, re-correcting placement and penalizing interpenetration via SAT collision detection.

Each stage runs SGD for 200 steps with lr = 0.01, momentum = 0.9, and a decay of 0.1 every 50 steps.

## Key Experimental Results

### System-Level Comparison (SSDB Dataset, vs. ACDC)

| Metric | ACDC | Diorama | Notes |
|--------|------|---------|-------|
| Acc ↑ | 0.04 | **0.08** | Scene-aware alignment accuracy |
| CD ↓ | 14.0 | **9.5** | Shape retrieval similarity |
| User Preference | 14.9% | **85.1%** | Human evaluation of scene quality |
| API Cost | $1.44 | **$0.12** | 12× cheaper |
| Time | 23.2 min | **3.7 min** | 6× faster |

### Architectural Reconstruction (PlainRecon vs. RaC / ACDC)

| Method | Success Rate | IoU ↑ | PE ↓ | CDb ↓ | Time |
|--------|-------------|-------|------|-------|------|
| RaC | 236/344 | 40.3 | 39.8 | 0.645 | 543 s |
| ACDC | 344/344 | 46.8 | 17.0 | 0.563 | 32 s |
| **PlainRecon** | **344/344** | **47.8** | **13.3** | **0.503** | **23 s** |

### Pose Estimation (GT Objects + GT Depth)

| Method | rAcc | Acc | Collision ↓ | Relation ↑ |
|--------|------|-----|-------------|------------|
| GigaPose | 0.36 | 0.27 | 7.91 | 0.61 |
| **Ours** | **0.47** | **0.37** | **5.84** | **0.63** |

### Ablation Study

Contribution of each optimization stage (GT objects + GT depth):
- No optimization → Collision 6.42, Support Relation 0.00
- +Orientation → Collision 5.17
- +Placement → Support relation jumps from 0.01 to 0.93
- +Space → Collision decreases but support relation temporarily drops to 0.54 (conflict between spatial and placement constraints)
- +Refinement → **Collision 3.78, Support 0.95** (final balance)

Compared to all-in-one optimization: the all-in-one variant yields Collision 12.80 and Support 0.16, far worse than the stage-wise results of 3.78 and 0.93, demonstrating the critical importance of the staged strategy.

Ablation on retrieval candidate count: increasing candidates from 1 → 4 → 8 improves Acc from 0.11 → 0.20 → 0.23, confirming that more candidates provide better pose initialization.

## Highlights & Insights

1. **"Training-free" does not mean "weak"**: By carefully decomposing the problem into sub-tasks and selecting the most appropriate foundation model for each, it is possible to achieve better generalization than end-to-end trained methods. This represents a pragmatic and powerful system design philosophy.

2. **Stage-wise optimization >> all-in-one optimization**: When the objective contains multiple conflicting terms (orientation vs. placement vs. space vs. collision), progressive decoupling is far more effective than joint optimization — all-in-one yields Collision 12.80 vs. stage-wise 3.78. This lesson generalizes broadly to 3D optimization problems.

3. **"Semantics first, then vision" retrieval**: Text narrows the candidate pool (ensuring semantic correctness), and image re-ranks within it (ensuring geometric match). This two-level strategy is worth borrowing for other multimodal retrieval tasks.

4. **Minimalist design of PlainRecon**: No discrete optimizer or complex CNN is needed — a simple four-step pipeline of inpainting → depth → normal clustering → plane fitting outperforms prior methods while being more robust.

## Limitations & Future Work

1. **Geometric and texture inaccuracy**: CAD retrieval inherently cannot yield exact geometric or material matches, which is a bottleneck for applications requiring precise reconstruction. Future work could incorporate shape deformation and texture transfer post-retrieval.
2. **Unstable scale estimation**: RANSAC-based solving is sensitive to noisy correspondences, particularly under heavy occlusion, causing sAcc to fall short of GigaPose.
3. **Error propagation from depth estimation**: The entire system depends heavily on monocular depth quality; inaccurate depth estimates compound across modules.
4. **Scene graph errors from GPT-4o**: Large multimodal models still have limitations in reasoning about complex spatial relationships; incorrect support relations drive optimization in the wrong direction.
5. **Limited to indoor scenes**: The planar reconstruction assumption restricts applicability to outdoor or non-planar architectural environments.

## Related Work & Insights

| Method | Training | Arch. Recon. | Open Vocabulary | Input | Key Difference |
|--------|----------|-------------|-----------------|-------|----------------|
| Mask2CAD / ROCA | Fully supervised | ✗ | ✗ | RGB | Fixed categories, requires annotation |
| DiffCAD | Weakly supervised | ✗ | ✗ | RGB | Trained on synthetic data, per-category |
| ACDC | Zero-shot | ✓ | ✓ | RGB | Relies on physics engine, high LLM cost |
| SceneComplete | Zero-shot | ✗ | ✓ | RGB | Desktop manipulation scenes only |
| **Diorama** | **Zero-shot** | **✓** | **✓** | **RGB** | **Complete system, lowest cost** |

Diorama's most significant advantage over DiffCAD is generalizability — DiffCAD fails entirely on categories such as bathtub, bin, and display on ScanNet due to insufficient training coverage, whereas Diorama handles arbitrary categories. Compared to ACDC, Diorama reduces GPT-4o calls from "3 per object" to "2 per scene," cutting cost by 12×.

## Relevance to My Research

2. **Scene graphs as intermediate representations**: The scene graph generated by GPT-4o makes implicit spatial understanding explicit, serving as a bridge between 2D visual perception and 3D layout optimization. The paradigm of LMM → structured representation → geometric optimization is worth exploring in other 3D tasks.
3. **Complementarity of synthetic data and foundation models**: Diorama demonstrates that foundation models can achieve strong performance even without synthetic data training. A further question is whether fine-tuning bottleneck modules (e.g., scale estimation, occlusion handling) with a small amount of synthetic data could yield significant improvements.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first zero-shot open-world complete 3D scene modeling system; innovation is more at the system level than in individual technical modules.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation on both synthetic and real data, detailed ablations, user study, multi-dimensional metrics, and rich supplementary material.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear structure, polished figures, rigorous problem decomposition logic; reads as a coherent engineering narrative.
- **Value**: ⭐⭐⭐⭐ — The modular zero-shot system design philosophy, stage-wise optimization strategy, and multimodal retrieval approach all offer transferable insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICCV 2025\] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)
- [\[ICCV 2025\] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](buffer-x_towards_zero-shot_point_cloud_registration_in_diverse_scenes.md)

</div>

<!-- RELATED:END -->
