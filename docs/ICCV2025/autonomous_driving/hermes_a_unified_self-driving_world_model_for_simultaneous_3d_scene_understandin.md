---
title: >-
  [Paper Note] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation
description: >-
  [ICCV 2025][Autonomous Driving][Driving world model] This paper proposes Hermes, the first unified driving world model that simultaneously performs 3D scene understanding (VQA/captioning) and future scene generation (poi…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Driving world model"
  - "3D scene understanding"
  - "point cloud generation"
  - "BEV"
  - "large language model"
date: 2026-05-08
content_hash: e7d41742a639a394
---

# Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation

**Conference**: ICCV 2025
**arXiv**: [2501.14729](https://arxiv.org/abs/2501.14729)
**Code**: [https://github.com/LMD0311/HERMES](https://github.com/LMD0311/HERMES)
**Area**: Autonomous Driving / World Models
**Keywords**: Driving world model, 3D scene understanding, point cloud generation, BEV, large language model

## TL;DR

This paper proposes Hermes, the first unified driving world model that simultaneously performs 3D scene understanding (VQA/captioning) and future scene generation (point cloud prediction). By leveraging BEV representations and world queries to inject LLM world knowledge into future scene generation, Hermes reduces 3s point cloud generation error by 32.4% and improves scene understanding CIDEr by 8.0%.

## Background & Motivation

Existing driving world models (DWMs) fall into two camps:
- **Scene generation** (e.g., ViDAR, OccWorld): skilled at predicting future scene evolution (2D video / 3D point clouds / occupancy grids), but incapable of understanding or describing the current scene.
- **Scene understanding** (e.g., OmniDrive, DriveGPT4): leverage VLMs for scene captioning and VQA, but lack future prediction capability.

Both capabilities are critical for autonomous driving decision-making, yet no prior framework unifies them. Unification faces two key challenges:

**Large spatial extent of multi-view inputs**: Autonomous driving requires processing six surround-view camera images; naively converting them to tokens exceeds the LLM context length and fails to capture inter-view interactions.

**Bridging understanding and generation**: Simply sharing features while modeling the two tasks separately cannot exploit the latent interaction between scene understanding and generation.

## Method

### Overall Architecture

The Hermes pipeline: multi-view images $I_t$ → BEV Tokenizer → flattened BEV $\mathcal{F}_t$ → LLM (understanding + world queries) → encoded BEV $\mathcal{B}_t$ + enriched world queries → Current-to-Future Link → future BEV → shared Render → point cloud sequence.

### Key Designs

1. **BEV-based World Tokenizer**: Six-view images are processed by a CLIP image encoder and BEVFormer v2 to produce BEV features $\mathcal{F}^{bev}_t \in \mathbb{R}^{w \times h \times c}$ ($w=h=200, c=256$), then spatially compressed 4× to a $50 \times 50$ token sequence fed into the LLM. The BEV representation offers two advantages: (1) it compresses multi-view inputs into a unified latent space, resolving the token length issue; (2) it preserves geometric spatial relationships, capturing inter-object interactions across views.

2. **World Queries Mechanism**: $\Delta t$ groups of world queries (each with $n=4$ queries) are initialized via max pooling over BEV features, augmented with ego-motion embeddings and frame position embeddings, and appended to the LLM input sequence. The LLM's causal attention mechanism allows world queries to attend to world knowledge produced during scene understanding (scene descriptions, VQA reasoning), thereby transferring semantic and conceptual information to the generation task. After LLM processing, the world queries interact with the encoded BEV $\mathcal{B}_t$ through a Current-to-Future Link (3-layer cross-attention blocks) to generate future BEV features.

3. **BEV-to-Point Render**: BEV features are upsampled and reshaped into a 3D volumetric representation $\mathcal{F}^{vol}_t \in \mathbb{R}^{w \times h \times z \times c'}$ ($z=32$), followed by differentiable volume rendering based on an implicit SDF field. For each LiDAR ray, sample points are drawn, local features are extracted via trilinear interpolation, a shallow MLP predicts SDF values, and the rendered depth is obtained by weighted integration. Current-frame point cloud prediction serves as an auxiliary task to regularize the BEV representation.

### Loss & Training

- **Scene understanding loss**: Next Token Prediction (NTP), $\mathcal{L}_N = -\sum_i \log P(\mathcal{T}_i | \mathcal{F}_t, \mathcal{T}_{1:i-1}; \Theta)$
- **Point cloud generation loss**: L1 depth supervision, $\mathcal{L}_D = \sum_{i=0}^{\Delta t} \lambda_i \frac{1}{N_i} \sum_k |d(\mathbf{r}_k) - \tilde{d}(\mathbf{r}_k)|$
- Frame weight $\lambda_i = 1 + 0.5i$ (larger weights for distant frames)
- Total loss $\mathcal{L} = \mathcal{L}_N + 10 \mathcal{L}_D$
- Three-stage training: (1) Tokenizer + Render pre-training; (2) BEV-text alignment + LoRA fine-tuning; (3) joint understanding and generation training.
- The LLM backbone is InternVL2-2B (1.8B parameters); the BEV backbone uses OpenCLIP ConNeXt-L.

## Key Experimental Results

### Main Results

| Method | Type | Gen. 0s↓ | 1s↓ | 2s↓ | 3s↓ | METEOR↑ | ROUGE↑ | CIDEr↑ |
|--------|------|-----------|-----|-----|-----|---------|--------|--------|
| 4D-Occ | Gen. only | - | 1.13 | 1.53 | 2.11 | - | - | - |
| ViDAR | Gen. only | - | 1.12 | 1.38 | 1.73 | - | - | - |
| GPT-4o | Und. only | - | - | - | - | - | 0.223 | 0.244 |
| OmniDrive | Und. only | - | - | - | - | 0.380 | 0.326 | 0.686 |
| Separated | Unified | 0.60 | 0.84 | 1.08 | 1.37 | 0.384 | 0.327 | 0.745 |
| **Hermes** | **Unified** | **0.59** | **0.78** | **0.95** | **1.17** | **0.384** | **0.327** | **0.741** |

Hermes achieves a 3s point cloud generation Chamfer Distance of 1.17, a 32.4% reduction over ViDAR (1.73→1.17). Scene understanding CIDEr reaches 0.741, an 8.0% improvement over OmniDrive. Notably, ViDAR uses 3s of historical frames while Hermes relies only on the current frame.

### Ablation Study

| Setting | Gen. 3s↓ | METEOR↑ | ROUGE↑ | CIDEr↑ |
|---------|---------|---------|--------|--------|
| Understanding only | - | 0.379 | 0.323 | 0.728 |
| Generation only | 1.687 | - | - | - |
| Separated unified | 1.875 | 0.377 | 0.321 | 0.722 |
| Unified (Hermes) | 1.718 | 0.377 | 0.321 | 0.720 |
| w/o world queries | ~1.30 (est.) | - | ~0.745 | - |
| w/ world queries | ~1.17 | - | ~0.741 | - |
| BEV 25×25 | 1.698 | 0.367 | 0.311 | 0.671 |
| BEV 50×50 | 1.718 | 0.377 | 0.321 | 0.720 |
| LLM 0.8B | 1.809 | 0.372 | 0.318 | 0.703 |
| LLM 1.8B | 1.718 | 0.377 | 0.321 | 0.720 |
| LLM 3.8B | 1.701 | 0.381 | 0.325 | 0.730 |

### Key Findings

1. **Unified outperforms separated**: Hermes significantly surpasses the separated unified baseline in generation (3s: 1.718 vs. 1.875), validating effective cross-task knowledge transfer.
2. **World queries are critical**: Introducing world queries reduces 3s generation Chamfer Distance by approximately 10%.
3. **Max pooling is optimal**: Among world query initialization strategies, max pooling outperforms attention pooling and average pooling.
4. **BEV resolution matters**: The 50×50 BEV improves CIDEr by 7.3% and 0s point cloud generation by 10% over the 25×25 BEV.
5. **LLM scaling is effective**: Understanding and generation performance consistently improve as LLM size increases from 0.8B to 3.8B.

## Highlights & Insights

- Hermes is the first unified driving world model to simultaneously perform 3D scene understanding and generation, opening a new research direction.
- The world queries design is elegant: the LLM's causal attention naturally transfers knowledge from understanding to generation without additional modules.
- BEV representations serve as a bridge, resolving both the multi-view token explosion problem and the need to preserve geometric structure.
- Current-frame point cloud prediction as an auxiliary task regularizes BEV encoding for free, without additional inference overhead.

## Limitations & Future Work

- Integration of perception tasks (detection, segmentation, etc.) into the unified framework has not been explored.
- Future image generation is not supported (only point clouds); incorporating image generation is an important extension direction.
- Only a 1.8B LLM is employed; the observed scaling law suggests larger models offer further gains.
- Evaluation is limited to the nuScenes dataset; generalization to other large-scale driving datasets remains untested.
- Performance degrades when the number of world queries increases (n=8/16 underperforms n=4), indicating an optimization challenge that warrants further investigation.

## Related Work & Insights

- **ViDAR**: Self-supervised future point cloud prediction from images; the primary baseline for the generation component.
- **OmniDrive**: Q-Former-based VQA for driving scenes; the primary baseline for the understanding component.
- **InternVL2**: Provides the LLM backbone; its causal attention is fundamental to the effectiveness of world queries.
- **BEVFormer v2**: Provides the BEV representation, forming the foundation for unifying multi-view information.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First to unify driving scene understanding and 3D generation; world queries design is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablations (world queries / BEV resolution / LLM scale / task interaction), but limited to nuScenes.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, figures are intuitive, and the method is presented coherently.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a unified driving world model paradigm with significant implications for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Controllable 3D Outdoor Scene Generation via Scene Graphs](controllable_3d_outdoor_scene_generation_via_scene_graphs.md)
- [\[ICCV 2025\] Epona: Autoregressive Diffusion World Model for Autonomous Driving](epona_autoregressive_diffusion_world_model_for_autonomous_driving.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)
- [\[ICCV 2025\] MCAM: Multimodal Causal Analysis Model for Ego-Vehicle-Level Driving Video Understanding](mcam_multimodal_causal_analysis_model_for_ego-vehicle-level_driving_video_unders.md)
- [\[NeurIPS 2025\] SPIRAL: Semantic-Aware Progressive LiDAR Scene Generation and Understanding](../../NeurIPS2025/autonomous_driving/spiral_semantic-aware_progressive_lidar_scene_generation_and_understanding.md)

</div>

<!-- RELATED:END -->
