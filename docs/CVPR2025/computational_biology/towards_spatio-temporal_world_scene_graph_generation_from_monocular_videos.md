---
title: >-
  [Paper Note] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos
description: >-
  [CVPR 2025][Computational Biology][Scene Graph Generation] This paper proposes the World Scene Graph Generation (WSGG) task and the ActionGenome4D dataset, upgrading video scene graphs from frame-centric 2D representations to world-centric 4D representations. It requires models to perform 3D localization and relation prediction in the world coordinate system for all objects, including invisible ones that are occluded or out of view. Three complementary methods (PWG/MWAE/4DST)…
tags:
  - "CVPR 2025"
  - "Computational Biology"
  - "Scene Graph Generation"
  - "4D Scene Understanding"
  - "Object Permanence"
  - "World State Modeling"
  - "Monocular Videos"
date: 2026-05-08
content_hash: fb872159bab66eb2
---

# Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos

**Conference**: CVPR 2025  
**arXiv**: [2603.13185](https://arxiv.org/abs/2603.13185)  
**Code**: [https://github.com/rohithpeddi/WorldSGG](https://github.com/rohithpeddi/WorldSGG)  
**Area**: Computational Biology  
**Keywords**: Scene Graph Generation, 4D Scene Understanding, Object Permanence, World State Modeling, Monocular Videos

## TL;DR
This paper proposes the World Scene Graph Generation (WSGG) task and the ActionGenome4D dataset, upgrading video scene graphs from frame-centric 2D representations to world-centric 4D representations. It requires models to perform 3D localization and relation prediction in the world coordinate system for all objects, including invisible ones that are occluded or out of view. Three complementary methods (PWG/MWAE/4DST) are proposed to explore different inductive biases for invisible object reasoning.

## Background & Motivation
**Background**: Video Scene Graph Generation (VidSGG) has existing datasets like Action Genome and Transformer-based methods, and 3D/4D scene graphs have also been explored.

**Limitations of Prior Work**: Existing methods are essentially "frame-centric"—modeling only currently visible objects. Once an object is occluded or leaves the field of view, it disappears from the graph. This contradicts the cognitive approach of real-world agents (object permanence).

**Key Challenge**: Embodied agents need to maintain continuous awareness of all objects in the environment (including invisible ones), but existing datasets and tasks lack (1) 3D world coordinate system localization, (2) cross-frame object consistency tracking, and (3) relation annotations for invisible objects.

**Goal**: How to construct temporally persistent, world-anchored scene graphs from monocular videos, covering all interacting objects (visible + invisible)?

**Key Insight**: Introduce the cognitive science concept of 'object permanence' into scene graph generation—objects continue to exist in the world state even when they are invisible, requiring their relations to be reasoned about.

**Core Idea**: Upgrade the scene graph from frame-centric 2D to world-centric 4D, where the core challenge lies in the representation and relation reasoning of invisible objects.

## Method

### Overall Architecture
Input monocular video $\rightarrow$ $\pi^3$ feedforward 3D reconstruction + BA pose refinement $\rightarrow$ 3D OBB in world coordinate system (GDINO detection + SAM2 segmentation + PCA OBB + Kalman smoothing) $\rightarrow$ three WSGG methods handling invisible objects $\rightarrow$ output per-frame world scene graph $\mathcal{G}_{\mathcal{W}}^t$ (containing three categories of relations: attention/spatial/contacting).

### ActionGenome4D Dataset
- Performs 3D reconstruction using $\pi^3$ on the Action Genome videos
- Generates 3D Oriented Bounding Boxes (OBB) via GDINO + SAM2 + ground alignment + PCA OBB + Kalman filtering
- Generates pseudo-labels for relations of invisible objects via VLM + manual correction
- Predicate set: attention (3), spatial (6), contacting (17)
- Covers all object pairs: visible-visible, visible-invisible, and invisible-invisible

### Three Complementary Methods

1. **PWG (Persistent World Graph)**:
    - **Function**: Achieves object permanence via a Last-Known-State (LKS) buffer
    - **Mechanism**: Non-differentiable zero-order hold—freezes DINO features of each object from its last visible frame, with an expiration metric $\Delta_n^{(t)} = |t - \tau^*|$. Fuses geometry, features, camera, and expiration before passing them through Spatial GNN + relation predictor
    - **Design Motivation**: The simplest implementation of object permanence—since the object existed, its features are preserved

2. **MWAE (Masked World Auto-Encoder)**:
    - **Function**: Treats occlusion/disappearance as natural masking and reconstructs invisible object representations using an MAE framework
    - **Mechanism**: Masks the visual stream of invisible objects, and the Associative Retriever reconstructs missing features using asymmetric cross-attention (all tokens as queries, only visible tokens as keys/values). Supervised by simulated occlusion and cross-view reconstruction during training
    - **Design Motivation**: Instead of simply freezing old features, it infers the possible current state of invisible objects based on the current scene context

3. **4DST (4D Scene Transformer)**:
    - **Function**: Replaces static buffers with a differentiable temporal Transformer, jointly processing visible/invisible tokens across all timesteps
    - **Mechanism**: Fuses multimodal tokens (visual + 3D structure + motion + camera pose) in Fusion Node, followed by unmasked bidirectional temporal self-attention + Spatial GNN to output globally aware spatio-temporal representations
    - **Design Motivation**: Allows invisible objects to extract information from all historical frames rather than only the last visible frame

### Shared Components
- **Global Structural Encoder**: OBB 8 corner points $\rightarrow$ MLP $\rightarrow$ structural token
- **Spatial Positional Encoding**: Euclidean distance + orientation + volume ratio of object pairs
- **Camera/Motion Encoder**: 6D rotation representation + relative inter-frame poses + object 3D velocity/acceleration
- **Relation Predictor**: CLIP text embedding + union ROI features $\rightarrow$ three-headed prediction (attention/spatial/contacting)

## Key Experimental Results

### Main Results (ActionGenome4D, PredCls)

| Method | Backbone | R@10 (WC) | R@20 (WC) | R@50 (NC) | R@50 (NC) |
|------|----------|-----------|-----------|-----------|-----------|
| PWG | DINOv2-L | 65.07 | 67.99 | 94.39 | 99.59 |
| MWAE | DINOv2-L | — | — | — | — |
| 4DST | DINOv2-L | — | — | — | — |

### Ablation Study
- 3D geometric features (OBB) contribute the most to spatial relation prediction
- Camera pose encoding is crucial for invisible object reasoning
- Motion features significantly aid contacting relation prediction
- VLM pseudo-label quality is close to ground truth after manual correction

### Key Findings
- Relation prediction for invisible objects is the core challenge—the Recall of all methods on invisible-object pairs is significantly lower than on visible-object pairs
- Although simple, PWG shows competitive performance in many configurations, indicating that "freezing the last visible features" is a reasonable baseline
- VLM (Graph RAG-based) has potential in non-localization relation prediction but still underperforms compared to specialized methods
- 3D localization is the main bottleneck under the SGDet mode

## Highlights & Insights
- **Paradigm shift from frame-centric to world-centric**: Systematically introduces the cognitive science concept of "object permanence" into scene graph generation, defining a clear WSGG task
- **Exploration of three complementary inductive biases**: PWG (feature freezing), MWAE (masked reconstruction), and 4DST (all-time attention) represent a spectrum of invisible object reasoning from simple to complex
- **ActionGenome4D Dataset**: Upgrades Action Genome to 4D scenes, with a foundation model-driven annotation pipeline ($\pi^3$ + GDINO + SAM2 + VLM) that can be reused for other datasets
- **Transferable insights**: World-centric scene graphs can directly serve planning and reasoning in embodied AI

## Limitations & Future Work
- ActionGenome4D is based on indoor scenes from Action Genome, with limited scene diversity (mainly household activity scenes)
- 3D reconstruction depends on the quality of $\pi^3$—it may degrade in poorly textured or fast-moving scenes, affecting OBB accuracy
- Pseudo-labels for relations of invisible objects rely on VLMs and may still be biased even after manual correction, especially for objects invisible for a long duration
- Only human-object interaction relations are considered; object-object relations are not included (e.g., "cup on table")
- Integration with actual embodied tasks (e.g., navigation/manipulation) to validate downstream value has not been performed
- High computational cost: the full pipeline of $\pi^3$ reconstruction + GDINO detection + SAM2 segmentation + Kalman smoothing incurs a significant overhead

## Related Work & Insights
- **vs Action Genome (VidSGG)**: AG only annotates relations of visible objects. ActionGenome4D extends to invisible objects, upgrading from 2D to 3D
- **vs 3D SGG (ScanNet)**: 3D SGG processes static 3D scans with no temporal dimension. WSGG processes temporal relational changes in dynamic videos
- **vs 4D Panoptic SG**: 4D PSG is still restricted to the visible range of the camera. WSGG explicitly models invisible objects

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Defines a brand new WSGG task; introducing object permanence into scene graphs is an original contribution
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three methods + VLM baseline + ablation + two evaluation modes
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous problem definition, complete mathematical formulation, and highly informative figures and tables
- **Value**: ⭐⭐⭐⭐⭐ New dataset + new task + new methods, with significant importance for embodied AI

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation](diffvsgg_diffusion-driven_online_video_scene_graph_generation.md)
- [\[ICLR 2026\] Scalable Spatio-Temporal SE(3) Diffusion for Long-Horizon Protein Dynamics](../../ICLR2026/computational_biology/scalable_spatio-temporal_se3_diffusion_for_long-horizon_protein_dynamics.md)
- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](../../ICML2025/computational_biology/neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [\[ICLR 2026\] MicroVerse: A Preliminary Exploration Toward a Micro-World Simulation](../../ICLR2026/computational_biology/microverse_a_preliminary_exploration_toward_a_micro-world_simulation.md)
- [\[ICLR 2026\] VCWorld: A Biological World Model for Virtual Cell Simulation](../../ICLR2026/computational_biology/vcworld_a_biological_world_model_for_virtual_cell_simulation.md)

</div>

<!-- RELATED:END -->
