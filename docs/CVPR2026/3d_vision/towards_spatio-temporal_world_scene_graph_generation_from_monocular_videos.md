---
title: >-
  [Paper Note] Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos
description: >-
  [CVPR 2026][3D Vision][Scene Graph Generation] This paper introduces the World Scene Graph Generation (WSGG) task, which constructs spatio-temporally persistent, world-coordinate-anchored scene graphs from monocular videos, covering all objects including occluded and out-of-frame ones. The paper also presents the ActionGenome4D dataset and three complementary methods (PWG/MWAE/4DST).
tags:
  - CVPR 2026
  - 3D Vision
  - Scene Graph Generation
  - Object Permanence
  - 3D Scene Understanding
  - Spatio-Temporal Reasoning
  - Vision-Language Models
date: 2026-05-08
content_hash: 4cd6a9ca1896c48e
---

# Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos

**Conference**: CVPR 2026
**arXiv**: [2603.13185](https://arxiv.org/abs/2603.13185)
**Code**: [Available](https://github.com/rohithpeddi/WorldSGG)
**Area**: 3D Vision
**Keywords**: Scene Graph Generation, Object Permanence, 3D Scene Understanding, Spatio-Temporal Reasoning, Vision-Language Models

## TL;DR

This paper introduces the World Scene Graph Generation (WSGG) task, which constructs spatio-temporally persistent, world-coordinate-anchored scene graphs from monocular videos, covering all objects including occluded and out-of-frame ones. The paper also presents the ActionGenome4D dataset and three complementary methods (PWG/MWAE/4DST).

## Background & Motivation

### 1. State of the Field
Scene graph generation (SGG) has expanded from static images to videos (VidSGG), 3D point clouds (3D SGG), 4D scenes, and beyond. Nevertheless, mainstream methods remain frame-centric: each frame independently infers currently visible objects and produces a 2D planar scene graph.

### 2. Limitations of Prior Work
- **Viewpoint dependency**: All object locations are expressed in 2D image coordinates, lacking a unified spatial reference frame.
- **Observation-gating**: Objects disappear from the graph once they leave the frame or become occluded, with no persistent memory.
- **Temporal fragmentation**: Even methods with temporal modeling (e.g., STTran, Tempura) only process frames within a sliding window and do not maintain a globally consistent world model.

### 3. Root Cause
Intelligent agents operating in real-world scenes must maintain a world model with *object permanence*—objects continue to exist in the environment even when unobservable. The frame-centric design of existing SGG methods cannot meet the demands of downstream tasks such as robotic manipulation, embodied navigation, and long-horizon activity understanding, all of which require reasoning over persistent world states.

### 4. Paper Goals
To construct a scene graph representation that is **temporally persistent, anchored in world coordinates, and covers all objects (including unobservable ones)**, with relationship prediction across three types of object pairs: observed–observed, observed–unobserved, and unobserved–unobserved.

### 5. Starting Point
The paper incorporates the cognitive science principle of object permanence into scene graph generation. The world state $\mathcal{W}^t$ is partitioned into an observable set $\mathcal{O}^t$ and an unobservable set $\mathcal{U}^t$, requiring models to map the complete world state at every timestamp.

### 6. Core Idea
- **ActionGenome4D dataset**: Upgrades Action Genome to a 4D representation, providing world-coordinate OBBs and dense relationship annotations for unobservable objects.
- **WSGG task**: Requires outputting a world scene graph covering all objects in $\mathcal{W}^t$ at every timestamp.
- **Three methods**: Explore different inductive biases for reasoning about unobservable objects.

## Method

### Overall Architecture

All three methods share a unified input pipeline and component suite: pre-extracted DINOv2/v3 visual features, 3D OBB corner coordinates reconstructed by π³, and camera extrinsic matrices. Shared components include:

- **Global Structural Encoder**: Encodes the 8 OBB corners as 27-dimensional input and produces structural tokens via an MLP.
- **Spatial Positional Encoding**: Computes 5D features between object pairs, including Euclidean distance, direction vector, and volume ratio.
- **Spatial GNN**: Intra-frame Transformer Encoder with spatial positional encoding to model object interactions.
- **Relationship Predictor**: Fuses person/object tokens, union RoI features, and CLIP text embeddings to predict attention (3 classes), spatial (6 classes), and contacting (17 classes) relations.
- **Camera Pose / Motion Encoder**: Encodes camera motion and per-object 3D velocity and acceleration.

### Key Designs

#### PWG (Persistent World Graph)
- **Function**: Retains the most recent visual features of unobservable objects from the last frame in which they were visible.
- **Mechanism**: A Last-Known-State (LKS) memory buffer implementing zeroth-order feature persistence. Current features are used when an object is visible; the most recent visible-frame features are retrieved when it is not; zero vectors are used for objects never observed.
- **Design Motivation**: Directly implements the object permanence principle. A staleness counter $\Delta_n^{(t)} = |t - \tau^*|$ is additionally recorded for feature fusion, enabling the model to be aware of feature "freshness."
- **Novelty**: The memory is non-differentiable and cannot learn temporal context end-to-end, yet it achieves strong performance through 3D geometric priors alone.

#### MWAE (Masked World Auto-Encoder)
- **Function**: Reformulates unobservable object reasoning as a masked completion problem.
- **Mechanism**: Occlusion and camera motion naturally provide "masks"; the model must infer representations of unobservable objects from visible ones. During training, a portion of visible objects is additionally masked at random to strengthen learning.
- **Design Motivation**: Transfers the MAE paradigm from the patch domain to the object/relation domain. Asymmetric cross-attention is employed (queries include all tokens; keys/values are restricted to visible tokens) to prevent unobservable tokens from attending to each other.
- **Loss**: $\mathcal{L}_{\text{MWAE}} = \mathcal{L}_{\text{SG}} + \lambda_{\text{recon}} \cdot \lambda_{\text{dom}} \cdot \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{sim}}$, comprising a scene graph loss, a feature reconstruction MSE loss, and a relation re-prediction loss for masked visible objects.

#### 4DST (4D Scene Transformer)
- **Function**: Replaces PWG's static buffer with a differentiable temporal Transformer.
- **Mechanism**: For each object, a token sequence is constructed along the temporal dimension (fusing visual, structural, camera, motion, and ego-motion features), and bidirectional Transformer self-attention is applied across the full video.
- **Design Motivation**: The LKS buffer in PWG is non-differentiable and cannot learn temporal context end-to-end. 4DST extends the factorized spatio-temporal attention paradigm from 2D visible objects to the complete 4D setting, incorporating sinusoidal positional encoding and learnable visibility embeddings.

### Loss & Training

All three methods share a unified multi-axis BCE loss structure. Object pairs are divided into visible pairs (clean ground truth) and unobserved pairs (VLM pseudo-labels, weighted by $\lambda_{\text{vlm}}$); attention, spatial, and contacting losses as well as node classification loss are computed for each group separately. MWAE additionally incorporates feature reconstruction and similarity losses.

## Key Experimental Results

### Main Results

**Table 2: Recall (R@K) — PredCls & SGDet on ActionGenome4D**

| Method | Backbone | PredCls R@10 | PredCls R@20 | SGDet R@10 | SGDet R@50 |
|--------|----------|-------------|-------------|-----------|-----------|
| PWG | DINOv2-L | 65.07 | 67.99 | 41.69 | 69.63 |
| MWAE | DINOv2-L | 65.33 | 68.30 | 41.69 | 69.50 |
| 4DST | DINOv2-L | 64.31 | 67.26 | **42.64** | 70.32 |
| PWG | DINOv3-L | 65.58 | 68.57 | 39.96 | 70.93 |
| MWAE | DINOv3-L | 65.57 | 68.58 | 39.67 | 70.90 |
| 4DST | DINOv3-L | **66.11** | **69.11** | 40.84 | **71.95** |

**Table 4: VLM Relationship Prediction — Micro-Averaged F1**

| Pipeline | Model | Mode | Attn F1 | Contact F1 | Spatial F1 | Micro F1 |
|----------|-------|------|---------|-----------|-----------|----------|
| Graph RAG | Qwen 2.5-VL | PredCls | 61.4 | 56.9 | 42.5 | **53.3** |
| Graph RAG | InternVL 2.5 | PredCls | 53.8 | 42.7 | 27.2 | 40.8 |
| Subtitle-Only | Qwen 2.5-VL | PredCls | 61.8 | 53.0 | 39.8 | 51.2 |

### Ablation Study

**Inter-method ablation findings**:
- **4DST** most consistently leads under the SGDet setting (R@10=42.64 with DINOv2-L; R@50=71.95 with DINOv3-L), with its differentiable temporal Transformer improving end-to-end gradient propagation.
- **MWAE** achieves the best performance in the multi-label (No Constraint) setting, with PredCls R@10=81.50 and mR@10=55.09 (DINOv3-L), where reconstruction and simulated-occlusion losses act as complementary regularizers.
- **PWG** trails the best method by only 1–2 points in most PredCls settings, confirming that 3D geometric priors alone constitute a strong structural inductive bias.

**VLM ablation findings**:
- Graph RAG consistently outperforms Subtitle-Only, though the margin narrows for the stronger VLM (Qwen: +2.1 vs. InternVL: +3.8).
- Recall under SGDet drops to roughly half that under PredCls, identifying world-level object detection as the primary bottleneck.

### Key Findings

1. Persistent 3D geometric priors alone (zeroth-order feature persistence in PWG) are sufficient to achieve highly competitive world scene graph generation.
2. Unobservable object reasoning can be further improved through differentiable temporal modeling (4DST), particularly in the end-to-end SGDet detection setting.
3. While VLMs can provide useful pseudo-annotations, substantial room remains for improvement in fine-grained spatial and contacting relation reasoning (micro F1 53.3 vs. macro F1 26.6, indicating severe long-tail imbalance).
4. Predicate difficulty increases in the order: Attention > Contacting > Spatial.

## Highlights & Insights

1. **Precise and necessary task formulation**: WSGG captures the critical shift from frame-centric to world-centric representation, with a clear definition of $\mathcal{W}^t = \mathcal{O}^t \cup \mathcal{U}^t$ and a world scene graph covering all interaction pairs.
2. **Complete dataset construction pipeline**: The pipeline from π³ 3D reconstruction → GDINO+SAM2 geometric annotation → VLM pseudo-labeling with manual correction → ActionGenome4D is systematic and reproducible.
3. **Clear design philosophy across three methods**: PWG (memory buffer), MWAE (masked completion), and 4DST (temporal Transformer) correspond respectively to zeroth-order persistence, auto-encoding, and full attention as inductive biases—complementary and progressively more expressive.
4. **Comprehensive experimental design**: Full matrix evaluation across PredCls/SGDet × With/No Constraint × R@K/mR@K, supplemented by VLM baselines and two inference pipelines.
5. **Cognitive science inspiration**: Introducing object permanence into technical design is well-motivated; PWG's staleness awareness and MWAE's natural masking from occlusion are both elegantly grounded.

## Limitations & Future Work

1. **Multi-stage pipeline lacks end-to-end training**: The cascade of 3D reconstruction (π³) → geometric annotation (GDINO+SAM2) → feature extraction (DINO) → relationship prediction propagates errors across stages.
2. **VLM pseudo-label quality**: Relationship annotations for unobservable objects rely on VLM generation with manual correction; label noise is mitigated by the $\lambda_{\text{vlm}}$ weight but not fundamentally resolved.
3. **Severe long-tail distribution**: Macro F1 is substantially lower than micro F1, indicating significant predicate class imbalance.
4. **Limited to person–object interactions**: The current framework only predicts person–object relation pairs and does not extend to arbitrary object pairs.
5. **Offline processing**: 4DST requires bidirectional attention over the complete video, precluding online streaming inference.
6. **Dataset scale constraints**: As an upgrade of Action Genome, scene diversity and generalization capability remain to be validated.

## Related Work & Insights

- **Relation to VidSGG (STTran/Tempura)**: WSGG is a strict superset, extending frame-level graphs to world-level graphs by adding two core dimensions: 3D localization and unobservable object reasoning.
- **Relation to 3D/4D SGG**: Existing 3D SGG methods process static scans, and 4D SGG typically requires RGB-D or multi-view inputs; WSGG operates from monocular video and covers unobservable objects.
- **MAE → object-level MAE**: MWAE generalizes masked autoencoders from the patch level to the object/relation level, replacing artificial masks with natural occlusion—a meaningful paradigm transfer.
- **VLMs as annotators**: The Graph RAG pipeline (event graph → retrieval → frame-level prediction → discriminative verification) is a practical paradigm for generating structured annotations with VLMs.
- **Implications for embodied intelligence**: World scene graphs serve as a critical intermediate representation bridging visual perception and embodied action; the temporal modeling approach in 4DST offers reference value for deployable systems.

## Rating

⭐⭐⭐⭐ The task formulation is forward-looking, the dataset construction is rigorous, the method design is systematic and progressive, and the experiments comprehensively cover multiple evaluation protocols and VLM baselines. Challenges remain, however, in achieving end-to-end training across the multi-stage pipeline and addressing long-tail predicate imbalance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning](../../AAAI2026/3d_vision/open-world_3d_scene_graph_generation_for_retrieval-augmented_reasoning.md)
- [\[CVPR 2026\] STS-Mixer: Spatio-Temporal-Spectral Mixer for 4D Point Cloud Video Understanding](sts_mixer_4d_point_cloud.md)
- [\[CVPR 2026\] STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction](stac_plug-and-play_spatio-temporal_aware_cache_compression_for_streaming_3d_reco.md)
- [\[CVPR 2026\] RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space](raynova_scale-temporal_autoregressive_world_modeling_in_ray_space.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)

</div>

<!-- RELATED:END -->
