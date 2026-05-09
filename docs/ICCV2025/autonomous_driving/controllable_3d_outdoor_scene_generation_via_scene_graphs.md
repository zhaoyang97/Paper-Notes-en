---
title: >-
  [Paper Note] Controllable 3D Outdoor Scene Generation via Scene Graphs
description: >-
  [ICCV 2025][Autonomous Driving][3D scene generation] This work proposes the first method to use scene graphs as control signals for large-scale 3D outdoor scene generation. A GNN encodes sparse scene graphs into BEV embedding maps, which are then fed into a cascaded 2D→3D discrete diffusion model to generate semantic 3D scenes. An accompanying interactive system allows users to directly edit scene graphs to control the generation.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 3D scene generation
  - scene graph
  - discrete diffusion model
  - BEV embedding
  - graph neural network
date: 2026-05-08
content_hash: 9b75a674e4086b05
---

# Controllable 3D Outdoor Scene Generation via Scene Graphs

**Conference**: ICCV 2025
**arXiv**: [2503.07152](https://arxiv.org/abs/2503.07152)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: 3D scene generation, scene graph, discrete diffusion model, BEV embedding, graph neural network

## TL;DR

This work proposes the first method to use scene graphs as control signals for large-scale 3D outdoor scene generation. A GNN encodes sparse scene graphs into BEV embedding maps, which are then fed into a cascaded 2D→3D discrete diffusion model to generate semantic 3D scenes. An accompanying interactive system allows users to directly edit scene graphs to control the generation.

## Background & Motivation

Large-scale 3D outdoor scene generation is in high demand for autonomous driving, gaming, and metaverse applications, yet existing methods are either uncontrollable (purely stochastic) or rely on suboptimal control signals. Text-conditioned methods (e.g., Text2LiDAR) cannot precisely specify object counts or spatial relationships; BEV layout/semantic map conditions require pixel-level annotations from users, incurring prohibitive interaction costs. Indoor scene generation approaches (e.g., bounding-box-based composition) do not transfer well to outdoor settings, where scenes are unbounded, large-scale, LiDAR-based without texture, and contain complex continuous background structures such as roads and buildings.

Scene graphs are inherently **structured, sparse, and editable**: nodes encode object categories and approximate positions, edges encode spatial relationships, and users can intuitively add, delete, or modify nodes to control the scene. These properties make scene graphs an ideal control signal for controllable 3D outdoor scene generation.

## Core Problem

How can a **sparse and abstract** scene graph be transformed into a conditioning signal that effectively guides a diffusion model to generate **large-scale 3D outdoor semantic scenes**? The core challenge lies in bridging the representational gap between the discrete graph structure of a scene graph and the dense conditioning format (e.g., 2D feature maps) expected by diffusion models.

## Method

### Overall Architecture

The pipeline follows a sparse-to-dense cascaded process consisting of three stages:

1. **Scene Graph → BEV Embedding Map (BEM)**: A GNN encodes scene graph node features, and an Allocation Module places node embeddings onto a 2D BEV grid to form a sparse BEV Embedding Map.
2. **BEM → 2D Semantic Map**: A 2D discrete diffusion model conditioned on the BEM generates a complete 2D semantic bird's-eye-view map, filling in background elements such as roads and buildings.
3. **2D Semantic Map → 3D Scene**: A 3D discrete diffusion model conditioned on the upsampled 2D semantic map generates the final 3D semantic voxel scene.

An additional **interactive system** enables users to manually drag and edit scene graphs, or to input natural language prompts for an LLM to automatically construct the scene graph.

### Key Designs

**1. Scene Graph Definition**: Nodes are of two types — instance nodes (vehicles, pedestrians, poles, etc., with category labels and BEV positions) and scene road nodes (encoding road types: straight, T-junction, intersection, curve, etc.). Edges are also of two types: physical proximity edges (instance pairs within distance threshold $\delta$) and road connectivity edges (connections between instance and road nodes).

**2. Context-Aware Node Embedding (CANE)**: Two-layer GAT message passing is applied, after which each node embedding is concatenated with a global graph pooling feature and passed through an MLP, yielding a globally contextualized node embedding $\mathbf{h}_i^{\text{CANE}}$. This ensures each node is aware of its context within the entire scene.

**3. Allocation Module**: The core function is to place graph nodes onto the BEV grid. During inference, an MLP localization head with Gumbel Softmax (temperature $\tau = 2.0$) samples a position $\hat{p}_i$, generates a binary mask $\mathcal{M}(\hat{p}_i)$, and element-wise multiplies node embeddings onto the corresponding locations, which are then aggregated to produce the BEM $\mathbf{L} \in \mathbb{R}^{H_b \times W_b \times C}$. During training, ground-truth positions are used directly; the localization head is trained separately via post-training.

**4. Auxiliary Tasks**: To enhance the GNN's scene understanding capability, two auxiliary tasks are introduced — edge reconstruction (reconstructing the adjacency matrix via a GAE-style objective) and node classification (predicting node categories) — ensuring that CANE encodes both structural relationships and semantic information.

**5. Two-Stage Discrete Diffusion**: Both the 2D and 3D diffusion models operate in a discrete (categorical) state space with a 3D-UNet backbone. The 2D diffusion model is conditioned on the BEM to generate a complete 2D semantic map; the 3D diffusion model is conditioned on the upsampled 2D semantic map to generate the 3D voxel scene.

### Loss & Training

- **Auxiliary loss** $\mathcal{L}_a$: BCE loss for edge reconstruction + CE loss for node classification.
- **2D diffusion loss** $\mathcal{L}_\theta$: KL divergence between forward and reverse diffusion processes + auxiliary reconstruction term (weight $\lambda$).
- **3D diffusion loss** $\mathcal{L}_\phi$: Same formulation as above, conditioned on the 2D semantic map.
- **Joint training**: The GNN and 2D diffusion model are trained jointly (loss $\mathcal{L}_a + \mathcal{L}_\theta$) to ensure feature–diffusion co-learning; the localization head (LOC) is post-trained after the GNN and diffusion model are frozen.
- **Data augmentation**: 10% unconditional data (analogous to classifier-free guidance) and 30% feature masking to simulate cases where users do not provide position information.

Ablation studies confirm that the optimal training strategy is **strategy (d)**: jointly train diffusion + GNN from scratch → freeze GNN → post-train LOC.

## Key Experimental Results

The dataset is CarlaSG (a scene graph–3D scene paired dataset constructed from CarlaSC), with a test set of 1k scene graphs.

| Method | Condition | mIoU↑ | MA↑ | F3D↓ | MAE↓ | Jaccard↑ | M-Pole↓ | M-Pede↓ | M-Vech↓ |
|--------|-----------|-------|-----|------|------|----------|---------|---------|---------|
| Uncon-Gen | None | - | - | - | - | - | - | - | - |
| SG2Im | Scene Graph | 65.43 | 81.72 | 0.486 | 0.97 | 0.81 | 2.25 | 2.79 | 2.64 |
| LLM | Text-Embedding | 68.19 | 85.62 | 0.386 | 1.44 | 0.70 | 3.41 | 3.57 | 3.51 |
| **Ours** | **Scene Graph** | **68.69** | **85.01** | **0.393** | **0.63** | **0.93** | **1.39** | **1.81** | **1.35** |

Key finding: scene quality (mIoU/MA/F3D) is on par with the LLM baseline, while **control precision substantially outperforms all baselines** — MAE of 0.63 (less than half of LLM's 1.44) and Jaccard Index of 0.93 vs. LLM's 0.70.

### Ablation Study

1. **Both auxiliary tasks are necessary**: Using both edge reconstruction and node classification yields MAE=0.63 and Jaccard=0.93; removing either reduces Jaccard to 0.83–0.84.
2. **Unconditional ratio**: 0.1 is the optimal balance — higher ratios improve mIoU but at the cost of control precision.
3. **Training strategy**: Jointly training DM+GNN followed by post-training LOC substantially outperforms end-to-end training from scratch (MAE 0.63 vs. 1.01) and frozen pretraining schemes.
4. **User study**: DMOS scores from 20 participants are significantly higher than both baselines ($p < 10^{-3}$).

## Highlights & Insights

1. **The sparse-to-dense cascaded design is elegant**: The pipeline scene graph → BEM → 2D map → 3D scene progressively densifies information, with each stage employing an appropriate generative model (GNN → 2D diffusion → 3D diffusion).
2. **Gumbel Softmax trick in the Allocation Module**: This converts the discrete position assignment problem into a differentiable sampling operation; GT positions are used during training while sampled positions are used during inference — a clean design.
3. **The practical impact of auxiliary tasks is surprisingly large**: Edge reconstruction and node classification appear simple, yet they yield a substantial gain in control precision (Jaccard 0.81 → 0.93), demonstrating that making the encoder truly understand graph structure is more important than simply scaling model capacity.
4. **Training strategy choice has a large impact**: The choice between end-to-end and staged training is non-trivial; the combination of joint DM+GNN training with staged LOC post-training is the result of careful empirical exploration.

## Limitations & Future Work

1. **Dataset limited to CARLA simulation**: CarlaSG is constructed from a simulator; paired scene graph–3D scene data for real-world scenes is unavailable, and generalization to real-world environments remains unvalidated.
2. **Semantic voxels only**: The output is a semantic-label voxel grid without texture or appearance information, leaving a significant gap from photorealistic renderable scenes.
3. **Fixed scene graph schema**: Node categories and edge types are predefined, making it difficult to extend to new object categories.
4. **Two-stage diffusion efficiency**: The cascaded 2D-then-3D diffusion may lead to error accumulation, and inference speed is constrained accordingly.
5. **Incorporating 3DGS or NeRF to produce renderable scenes, rather than purely semantic voxels, is a promising direction.**

## Related Work & Insights

- **vs. Text2LiDAR**: Text-based control is coarse-grained and cannot precisely specify object counts or positions; this paper's scene graph provides structured and precise control.
- **vs. SG2Im**: Originally a 2D image generation method, its adaptation for BEM generation yields lower scene quality and control precision than the proposed approach (mIoU 65.43 vs. 68.69; Jaccard 0.81 vs. 0.93).
- **vs. indoor scene graph methods (CommonScenes, EchoScene, etc.)**: These methods compose objects via bounding boxes, which is ill-suited for unbounded outdoor scenes with continuous background structures.
- **vs. SemCity/Pyramid Diffusion**: These are unconditional or weakly conditioned 3D scene generation methods; the proposed work extends them with scene graph conditioning.

The scene graph as intermediate representation can inspire **autonomous driving data augmentation**: scene graphs could describe corner cases and drive the generation of corresponding 3D training scenes. The sparse-to-dense conversion paradigm via GNN + Allocation Module is broadly applicable to other tasks requiring graph-structured conditioning (e.g., indoor scene rearrangement, video generation control). This work is also related to the "Streaming World Scene Graph" concept: this paper generates 3D scenes from scene graphs, while the reverse direction — understanding scene graphs from 3D scenes or videos — is equally worth exploring.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First use of scene graphs for 3D outdoor scene generation; the sparse-to-dense cascaded framework is clearly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablations (auxiliary tasks / training strategy / unconditional ratio / user study), though limited to a single simulated dataset.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, figures are intuitive, and the pipeline is easy to follow.
- **Value**: ⭐⭐⭐ The scene graph → 3D generation paradigm is inspiring, though somewhat distant from the current research focus.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[ICCV 2025\] Decoupled Diffusion Sparks Adaptive Scene Generation](decoupled_diffusion_sparks_adaptive_scene_generation.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)

<!-- RELATED:END -->
