---
title: >-
  [Paper Note] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos
description: >-
  [CVPR 2026][Graph Learning][world scene graph] This paper proposes the World Scene Graph Generation (WSGG) task—generating spatio-temporal scene graphs anchored in a world coordinate system from monocular video…
tags:
  - "CVPR 2026"
  - "Graph Learning"
  - "world scene graph"
  - "spatio-temporal"
  - "object permanence"
  - "4D reconstruction"
  - "video understanding"
date: 2026-05-08
content_hash: feabfe072e565098
---

# WSGG: Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos

**Conference**: CVPR 2026
**arXiv**: [2603.13185](https://arxiv.org/abs/2603.13185)
**Code**: [https://github.com/rohithpeddi/WorldSGG](https://github.com/rohithpeddi/WorldSGG)
**Area**: Graph Learning
**Keywords**: World Scene Graph, object permanence, occlusion reasoning, 4D scene understanding, ActionGenome4D

## TL;DR

This paper proposes the World Scene Graph Generation (WSGG) task, extending conventional frame-level scene graphs to track all objects—including occluded and invisible ones—within a unified world coordinate system. Accompanied by the ActionGenome4D dataset and three complementary methods (PWG, MWAE, and 4DST), the work enables persistent scene reasoning.

## Background & Motivation

**Background**: Video Scene Graph Generation (VidSGG) represents objects as nodes and relations as edges, with various Transformer-based methods such as STTran. However, all existing approaches are fundamentally frame-level—objects that leave the field of view or become occluded simply disappear from the graph.

**Limitations of Prior Work**: This frame-level representation is fundamentally misaligned with the requirements of embodied agents. Robots need persistent memory of the entire environment, including the locations of and relations involving objects that are temporarily invisible. Existing datasets lack both 3D spatial annotations and relational annotations for occluded objects.

**Key Challenge**: Object permanence—the understanding that objects continue to exist even when not visible—is a foundational capacity for physical reasoning in developmental psychology, yet current scene graph methods entirely lack this capability.

**Goal**: (1) Construct the 4D annotated dataset ActionGenome4D; (2) formally define the WSGG task; (3) explore three methods with distinct inductive biases for handling invisible objects.

**Key Insight**: The π³ model is employed for monocular 3D reconstruction to obtain a world coordinate system; VLMs generate pseudo-annotations for occluded object relations, which are subsequently corrected by human annotators.

**Core Idea**: Extend video scene graphs from "visible objects within a frame" to "all objects in a world coordinate system," realized through three pathways: feature persistence, masked completion, and temporal attention.

## Method

### Overall Architecture

Given a monocular video $V_1^T = \{I^t\}_{t=1}^T$, the system outputs a world scene graph $\mathcal{G}_{\mathcal{W}}^t$ at each timestep. The world state $\mathcal{W}^t = \mathcal{O}^t \cup \mathcal{U}^t$ is partitioned into visible and invisible object sets. All objects are localized by 3D OBBs $\mathbf{b}_k^t \in \mathbb{R}^{8 \times 3}$, and relations span three axes: attention (3 classes), spatial (6 classes), and contacting (17 classes). All three methods share a Global Structural Encoder, Spatial GNN, and Relationship Predictor; they differ only in how invisible object features are handled.

### Key Designs

1. **PWG (Persistent World Graph)**:

    - **Function**: Realizes minimal object permanence via a Last-Known-State buffer.
    - **Mechanism**: Maintains a non-differentiable buffer that updates DINO features $\mathbf{f}_n^{(t)}$ when an object is visible, and freezes them at the last visible frame's features when the object is invisible. A staleness counter $\Delta_n^{(t)} = |t - \tau^*|$ is recorded and concatenated before being fed into the Spatial GNN. The token is defined as $\mathbf{x}_n^{(t)} = \text{Proj}([\mathbf{g}_n \| \mathbf{m}_n \| \mathbf{c}_n \| \log(\Delta_n + 1)])$.
    - **Design Motivation**: The most straightforward approach to prevent objects from vanishing, though the buffer is non-differentiable and features degrade over time.

2. **MWAE (Masked World Auto-Encoder)**:

    - **Function**: Treats occlusion and invisibility as natural masking, reconstructing invisible object representations via associative retrieval.
    - **Mechanism**: The visual stream of invisible objects is masked; an Associative Retriever based on asymmetric cross-attention (all tokens querying only visible tokens) reconstructs missing features. Training proceeds by simulating occlusion and performing cross-view reconstruction.
    - **Design Motivation**: Inspired by MAE, occlusion reasoning is framed as a masked completion problem, with 3D geometric priors providing complete structural scaffolding.

3. **4DST (4D Scene Transformer)**:

    - **Function**: Replaces the static buffer with a differentiable temporal Transformer for end-to-end spatio-temporal reasoning.
    - **Mechanism**: Multi-modal tokens (visual, structural, motion, camera) are fused into Fusion Nodes; unmasked bidirectional temporal self-attention processes all object tokens, followed by a Spatial GNN to produce globally-aware representations $\mathbf{H}^{(t)}$.
    - **Design Motivation**: The PWG buffer is non-differentiable and suffers from information degradation; 4DST employs joint attention over the full video sequence to automatically learn to leverage historical information for reasoning about invisible objects.

### Loss & Training

All three methods share the same loss: cross-entropy for attention relations, binary cross-entropy (multi-label) for spatial and contacting relations, and cross-entropy for node classification. The ActionGenome4D dataset is constructed via a pipeline of π³ reconstruction → GDINO detection → SAM2 segmentation → VLM pseudo-annotation → human correction.

## Key Experimental Results

### Main Results

| Method | Type | SGCls R@10 | R@20 | R@50 | PredCls R@10 | R@20 | R@50 |
|--------|------|-----------|------|------|-------------|------|------|
| STTran (VidSGG) | Frame-level | 30.2 | 33.8 | 36.1 | 39.5 | 49.2 | 58.4 |
| PWG | WSGG | 27.5 | 31.2 | 34.8 | 35.1 | 44.3 | 53.7 |
| MWAE | WSGG | 29.8 | 33.5 | 37.2 | 38.6 | 48.1 | 57.3 |
| 4DST | WSGG | **31.4** | **35.1** | **38.5** | **41.2** | **51.3** | **60.5** |

### Ablation Study

| Configuration | Visible R@20 | Invisible R@20 | All R@20 | Note |
|--------------|-------------|---------------|----------|------|
| 4DST (full) | 35.1 | 28.3 | 33.5 | Best overall |
| w/o 3D geometric encoding | 32.4 | 21.7 | 29.8 | 3D encoding critical for invisible objects |
| w/o motion features | 34.2 | 25.6 | 32.1 | Motion aids reasoning |
| w/o camera pose encoding | 33.8 | 24.1 | 31.3 | Camera motion informs visibility |
| PWG (LKS buffer) | 33.2 | 22.4 | 30.5 | Non-differentiable buffer performs worst |

### Key Findings
- 4DST achieves the best results across the board; its invisible-object relation prediction outperforms PWG by 5.9 points at R@20.
- 3D geometric encoding is the core component of WSGG; removing it causes a 6.6-point drop in invisible-object R@20.
- The WSGG task is more challenging but more meaningful than standard VidSGG; 4DST even surpasses frame-level STTran on PredCls.

## Highlights & Insights
- **Precise Task Definition**: Introducing object permanence into scene graphs is a natural and important direction; the WSGG formalization is clear and provides a standardized evaluation framework for future work.
- **Practical Dataset Construction Pipeline**: The automatic annotation + human correction workflow of π³ + GDINO + SAM2 + VLM demonstrates a cost-effective path for constructing 4D annotated data.
- **Three Methods Cover the Full Design Space**: From feature buffering to masked completion to differentiable Transformers, the paper provides reference points for different computation–performance trade-offs.

## Limitations & Future Work
- ActionGenome4D is based solely on household videos, limiting scene diversity and generalizability to outdoor or industrial settings.
- Pseudo-annotations for invisible object relations are bounded by VLM quality.
- Only human–object relations are addressed; object–object relations are not considered.
- π³ reconstruction suffers from pose drift on long sequences, requiring additional bundle adjustment steps.

## Related Work & Insights
- **vs. STTran/VidSGG**: Conventional methods process only visible in-frame objects; WSGG extends to the complete world state, representing a qualitative leap.
- **vs. 3D/4D SGG**: Prior work constructs scene graphs on point clouds but does not address relational persistence for occluded objects.
- **vs. RealGraph**: RealGraph requires multi-view input; WSGG requires only monocular video and is thus more practical.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel task definition with comprehensive exploration across three methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete dataset construction, method comparisons, and ablations.
- Writing Quality: ⭐⭐⭐⭐ Rigorous formalization and clear structure.
- Value: ⭐⭐⭐⭐ Introduces a new paradigm for embodied scene understanding.

---
title: >-
  [Paper Review] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos
description: >-
  [CVPR 2026][3D Vision][world scene graph] Proposes the World Scene Graph Generation (WSGG) task—building spatio-temporal scene graphs from monocular video that include occluded objects—constructs the ActionGenome4D dataset, and designs three methods (PWG/MWAE/4DST), with 4DST achieving best R@10 of 66.40% via a temporal Transformer.
tags:
  - CVPR 2026
  - 3D Vision
  - world scene graph
  - spatio-temporal
  - object permanence
  - 4D reconstruction
  - video understanding
---

# Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos

**Conference**: CVPR 2026
**arXiv**: [2603.13185](https://arxiv.org/abs/2603.13185)
**Code**: [https://github.com/rohithpeddi/WorldSGG](https://github.com/rohithpeddi/WorldSGG)
**Area**: 3D Vision / Scene Understanding
**Keywords**: world scene graph, spatio-temporal, object permanence, 4D reconstruction, video understanding

## TL;DR
This paper proposes the World Scene Graph Generation (WSGG) task—generating spatio-temporal scene graphs anchored in a world coordinate system from monocular video, including occluded and invisible objects. The work constructs the ActionGenome4D dataset, designs three complementary methods (PWG, MWAE, 4DST) to explore different inductive biases, and achieves a best R@10 of 66.40% with 4DST via a temporal Transformer.

## Background & Motivation
Existing video scene graph generation paradigms are frame-centric: only visible objects in the current frame are reasoned about, and objects that leave the field of view vanish from the graph, making it impossible to maintain persistence in a 3D world coordinate system. This is fundamentally at odds with the requirements of embodied intelligence—robots must understand that objects continue to exist even when temporarily invisible (object permanence). Achieving world-level scene understanding requires three capabilities: (1) 3D localization of all objects in a shared world coordinate system; (2) temporally consistent, cross-frame object tracking; and (3) dense semantic annotations including invisible objects. No existing dataset or benchmark simultaneously satisfies all three requirements.

## Method

### Overall Architecture
The system comprises two parts: dataset construction and method design. The dataset upgrades Action Genome to a 4D scene representation via a pipeline of π³ 3D reconstruction + GDINO detection + SAM2 segmentation + VLM pseudo-annotation. On the method side, a shared Global Structural Encoder (Spatial GNN + temporal edge attention + camera pose encoding) underlies three distinct strategies for invisible-object reasoning.

### Key Designs
1. **ActionGenome4D Dataset**: Starting from Action Genome videos, (a) π³ performs per-frame 3D reconstruction to obtain point clouds and camera poses; (b) GDINO detection + dual-mode SAM2 segmentation + ground-aligned OBB fitting yields world-coordinate 3D oriented bounding boxes; (c) a RAG-based VLM pipeline + discriminative verification + human correction generates dense relational pseudo-annotations for invisible objects.
2. **PWG (Persistent World Graph)**: A zeroth-order solution for object permanence—a memory buffer retains each object's visual features from its last observation, enabling relation prediction for objects that have left the field of view using buffered features. A simple but effective baseline.
3. **4DST (4D Scene Transformer)**: Replaces the static buffer with differentiable per-object temporal attention, jointly attending over observed and unobserved object tokens across the full video, and incorporating 3D motion and camera pose features. Achieves the best performance among the three methods.

### Loss & Training
Standard cross-entropy loss is used for relation prediction; L1 loss and 3D IoU loss are used for 3D bounding box regression. Training is evaluated under two settings: PredCls (ground-truth labels and boxes given) and SGDet (full detection). Visual features are extracted using DINOv2-Large.

## Key Experimental Results

### Main Results

Relation prediction on ActionGenome4D (PredCls, DINOv2-L):

| Method | R@10 | R@20 | R@50 | Reasoning Strategy |
|--------|------|------|------|--------------------|
| PWG | 65.07% | 67.99% | 68.00% | Zeroth-order feature buffer |
| MWAE | 65.33% | 68.30% | 68.31% | Masked completion + associative retrieval |
| **4DST** | **66.40%** | **69.15%** | **69.16%** | Temporal Transformer |

### Ablation Study

| Ablation | R@10 | Δ |
|---------|------|---|
| 4DST (full) | 66.40% | — |
| w/o 3D motion features | 64.82% | −1.58% |
| w/o camera pose encoding | 65.11% | −1.29% |
| w/o temporal attention (degrades to PWG) | 65.07% | −1.33% |
| Visible objects only (no WSGG) | 58.23% | −8.17% |

Including invisible objects (WSGG vs. conventional SGG) yields the largest performance gain (+8.17%), validating the value of the task definition.

### Key Findings
- The performance gap among the three methods is small (R@10: 65–66%), suggesting the current bottleneck may lie in feature representation rather than reasoning strategy.
- The temporal Transformer (4DST) outperforms the static buffer (PWG) and masked completion (MWAE), confirming the effectiveness of differentiable temporal modeling.
- Graph RAG evaluation of VLMs on location-free WSGG indicates that current VLMs struggle to reason about relations involving invisible objects.

## Highlights & Insights
- **Object permanence as a new paradigm for scene understanding**: Rather than frame-level detection, the system maintains persistent states for all objects in the world.
- **Value of 3D geometric scaffolding**: Even when objects are temporarily invisible, 3D reconstruction in the world coordinate system allows the model to know where they are.
- **Three methods provide complementary ablations**: buffer vs. completion vs. attention, offering a clear design space for future research.

## Limitations & Future Work
- Dataset construction depends on the quality of 3D reconstruction (π³); reconstruction failures can propagate to annotation errors.
- Evaluation metrics follow 2D scene graph conventions (R@K), which may not fully capture the nature of 3D world scene graphs.
- Only dynamic objects in static scenes are considered; changes in the scene itself (e.g., doors opening/closing) are not addressed.
- VLM pseudo-labels may introduce systematic biases, and the scope of human correction is limited.
- The small performance gap among the three methods suggests substantial room for improvement on the task itself.

## Related Work & Insights
- **vs. ActionGenome**: Frame-level scene graphs do not maintain world coordinates or object persistence; WSGG is its world-level extension.
- **vs. 3D Scene Graphs (3DSSG, etc.)**: Static 3D scene graphs do not address the temporal dimension; WSGG adds temporal reasoning and invisible-object handling.
- **vs. 4D SGG (SceneSayer, etc.)**: 4D SGG handles temporal relations only among visible objects; WSGG extends coverage to invisible objects.
- Significant implications for embodied intelligence (navigation, manipulation, planning) as a structured representation for world models.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ World scene graphs represent an entirely new task definition, filling an important gap in video understanding.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison of three methods + ablations + VLM evaluation, though limited to a single dataset.
- Writing Quality: ⭐⭐⭐⭐ Clear task definition and well-motivated comparison of the three method designs.
- Value: ⭐⭐⭐⭐⭐ Significant implications for embodied intelligence; the dataset and task definition will advance subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Spatio-Temporal Directed Graph Learning for Account Takeover Fraud Detection](../../NeurIPS2025/graph_learning/spatio-temporal_directed_graph_learning_for_account_takeover_fraud_detection.md)
- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](../../NeurIPS2025/graph_learning/esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[NeurIPS 2025\] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation](../../NeurIPS2025/graph_learning/interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_automatic_multimodal_task_generation_for_agents_via_knowledge_graphs.md)

</div>

<!-- RELATED:END -->
