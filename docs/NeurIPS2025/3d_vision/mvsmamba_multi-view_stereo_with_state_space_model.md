---
title: >-
  [Paper Note] MVSMamba: Multi-View Stereo with State Space Model
description: >-
  [NeurIPS 2025][3D Vision][Multi-View Stereo] This paper proposes MVSMamba, the first Mamba-based Multi-View Stereo (MVS) network, which achieves efficient intra-view and inter-view global omnidirectional feature aggregation via a reference-centered dynamic scanning strategy, attaining state-of-the-art performance on DTU and Tanks-and-Temples with superior efficiency.
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Multi-View Stereo"
  - "Mamba"
  - "State Space Model"
  - "Feature Matching"
  - "Depth Estimation"
date: 2026-05-08
content_hash: ff40d0d68bc94dc0
---

# MVSMamba: Multi-View Stereo with State Space Model

**Conference**: NeurIPS 2025
**arXiv**: [2511.01315](https://arxiv.org/abs/2511.01315)  
**Code**: [https://github.com/JianfeiJ/MVSMamba](https://github.com/JianfeiJ/MVSMamba)  
**Area**: LLM Evaluation
**Keywords**: Multi-View Stereo, Mamba, State Space Model, Feature Matching, Depth Estimation

## TL;DR

This paper proposes MVSMamba, the first Mamba-based Multi-View Stereo (MVS) network, which achieves efficient intra-view and inter-view global omnidirectional feature aggregation via a reference-centered dynamic scanning strategy, attaining state-of-the-art performance on DTU and Tanks-and-Temples with superior efficiency.

## Background & Motivation

Multi-View Stereo (MVS) aims to reconstruct dense 3D geometry from calibrated multi-view images, relying critically on high-quality cross-view feature matching. Robust feature representation is the foundation of reliable matching.

Evolution and limitations of existing methods:
- **CNN-based methods** (e.g., CasMVSNet): computationally efficient but constrained by local receptive fields, underperforming in challenging regions such as textureless and reflective surfaces.
- **Transformer-based methods** (e.g., TransMVSNet, MVSFormer++): introduce global dependency modeling to improve performance, but the quadratic complexity of attention mechanisms incurs significant efficiency costs. Even with optimizations such as linear attention and epipolar window attention, multiple alternating rounds of self-attention and cross-attention remain costly. Some approaches also rely on large pretrained ViTs with substantial parameter counts (e.g., MVSFormer++ uses 39.48M parameters).

**Core problem**: How to minimize computational cost while maintaining high performance?

Mamba, as an efficient variant of state space models, offers linear-complexity long-range dependency modeling and is well-suited to address this challenge. However, Mamba's 1D sequential scanning cannot be directly adapted to the one-to-many feature matching paradigm in MVS and requires specialized scanning strategies.

## Method

### Overall Architecture

MVSMamba adopts a coarse-to-fine paradigm. The overall pipeline proceeds as follows: (1) an FPN encoder extracts multi-scale features; (2) a Dynamic Mamba Module (DM-module) is inserted at scale 0 of the FPN to perform global reference–source feature interaction; (3) a Simplified DM-module (SDM-module) is inserted at scale 1 of the FPN decoder for single-view feature enhancement; (4) enhanced pyramid features are used for depth estimation from coarse to fine via warping, cost volume construction, and regularization.

### Key Designs

1. **Dynamic Mamba Module (DM-module)**: The core innovation. For each reference–source feature pair, the source feature is concatenated to the top, bottom, left, and right of the reference feature, forming four concatenated feature maps. Four skip scans in different directions (with stride 2) are applied to these maps, generating four directional sequences. Crucially, the **reference-centered** design ensures that each scan originates from the reference view toward the source view, enabling source features to learn global representations informed by the reference. The four sequences are processed by four independent Mamba blocks and then merged back into 2D feature maps. Stride 2 reduces sequence length to one-quarter of the total pixel count, improving efficiency.

2. **Reference-Centered Dynamic Scanning**: The scan starting coordinate $(h_k, w_k)$ is dynamically updated according to the source image index $k$. Different source views induce different scan directions for each reference pixel; thus, when $K \geq 5$ (at least 4 source views), the reference features acquire an omnidirectional global receptive field. This resolves the anisotropy caused by a single fixed scan direction without requiring multiple alternating self-attention and cross-attention rounds as in Transformer-based methods.

3. **Simplified DM-module (SDM-module) and Multi-Scale Aggregation**: The SDM-module processes single-view features only (without reference–source concatenation), directly scanning the input to generate four directional sequences. The DM-module is deployed only at scale 0 to capture inter-view interaction, while the SDM-module operates at scale 1 for intra-view enhancement. Experiments show that adding DM/SDM modules at additional scales yields no further gain—since scale-0 interactions propagate to all scales through the decoder—thereby preserving efficiency.

4. **Feature Merging**: The four enhanced sequences are inverse-scanned back into four concatenated feature maps. Enhanced features are extracted separately from the reference and source regions, and the four directional components of the reference features are summed to produce the final omnidirectionally enhanced features.

### Loss & Training

Cross-entropy loss supervises the probability volume at each scale. DTU training: 5 views × 512 × 640, batch size 4, 15 epochs, $lr = 0.001$ with staged decay. BlendedMVS fine-tuning for Tanks-and-Temples evaluation: 11 views × 576 × 768, batch size 2, 15 epochs. High-resolution training: 5 views × 1024 × 1280, 10 epochs. Depth hypothesis counts: 32–16–8–4 (coarse to fine); depth intervals: 2–1–1–0.5. Adam optimizer is used. Final point clouds are obtained via a dynamic fusion strategy.

## Key Experimental Results

### Main Results

**DTU Dataset (Point Cloud Reconstruction Quality + Efficiency)**:

| Method | Type | Overall ↓ | Acc. ↓ | Comp. ↓ | GPU(G) ↓ | Time(s) ↓ | Params(M) ↓ |
|--------|------|-----------|--------|---------|----------|-----------|-------------|
| MVSMamba* (Ours) | Mamba | **0.280** | **0.308** | **0.252** | 2.82 | 0.11 | 1.31 |
| MVSFormer++ | Trans. | 0.281 | 0.309 | 0.252 | 4.71 | 0.23 | 39.48 |
| ET-MVSNet | Trans. | 0.291 | 0.329 | 0.253 | 2.91 | 0.16 | 1.09 |
| CasMVSNet | CNN | 0.355 | 0.324 | 0.385 | 4.48 | 0.18 | 0.93 |

MVSMamba achieves the best performance with the lowest GPU memory, fastest inference speed, and only 1.31M parameters—far fewer than MVSFormer++'s 39.48M. It ranks first overall (Avg. Rank: 2.50).

**Tanks-and-Temples (Generalization Evaluation)**:

| Method | Intermediate Mean ↑ | Advanced Mean ↑ |
|--------|---------------------|-----------------|
| MVSMamba (Ours) | **67.67** | **43.32** |
| MVSFormer++ | 67.18 | 41.60 |
| GoMVS | 66.44 | 43.07 |
| GeoMVSNet | 65.89 | 41.52 |

MVSMamba achieves the best F-score on both Intermediate and Advanced splits.

### Ablation Study

| Configuration | Overall ↓ | MAE ↓ | GPU(G) | Time(s) | Notes |
|---------------|-----------|-------|--------|---------|-------|
| Full MVSMamba | **0.287** | **5.21** | 2.82 | 0.11 | Complete model |
| w/o DM | 0.295 | 5.58 | 2.82 | 0.104 | Removing DM-module causes notable degradation |
| w/o SDM | 0.289 | 5.45 | 2.82 | 0.097 | Smaller impact |
| w/o MLP | 0.293 | 5.23 | 2.82 | 0.108 | Without MLP enhancement |
| w/ VMamba scan | 0.291 | 5.30 | 2.82 | 0.13 | Four-directional scan replacement |
| w/ EVMamba scan | 0.298 | 5.81 | 2.82 | 0.11 | Skip scan without dynamic strategy |
| w/ JamMa scan | 0.301 | 6.01 | 2.82 | 0.11 | Joint scan unsuitable for MVS |

### Key Findings

- The DM-module contributes the most (removing it increases Overall from 0.287 to 0.295), as it captures inter-view long-range dependencies at the FPN's lowest scale and propagates them to all scales.
- The dynamic scanning strategy substantially outperforms VMamba's four-directional scan, EVMamba's skip scan, and JamMa's joint scan, validating the reference-centered design with dynamic starting coordinates.
- Using four independent (non-shared) Mamba blocks outperforms weight sharing (at the cost of only +0.1M parameters), indicating that different scan directions must learn distinct information.
- Reference-centered scanning outperforms source-centered scanning, as source features need to learn consistent global representations from the reference.
- Deploying the DM-module only at scale 0 is sufficient; adding it at more scales provides no gain or slightly degrades performance.

## Highlights & Insights

- This work is the first to introduce Mamba into MVS, demonstrating that linear-complexity SSMs can match Transformer performance with significantly superior efficiency.
- The reference-centered dynamic scanning strategy is the core innovation: it elegantly adapts Mamba's 1D scan to the one-to-many matching paradigm in MVS, where different source views contribute different scan directions to give reference features an omnidirectional receptive field.
- Compared to MVSFormer++: comparable performance (0.280 vs. 0.281) with 30× fewer parameters (1.31M vs. 39.48M), 2× faster inference (0.11s vs. 0.23s), and 40% less GPU memory (2.82G vs. 4.71G).
- State-of-the-art results on the Tanks-and-Temples Advanced split further demonstrate strong generalization.

## Limitations & Future Work

- Mamba is a causal model that exploits only preceding context; although four-directional scanning partially mitigates this, the model is not fully bidirectional.
- The DM-module is currently deployed only at the lowest resolution; Mamba's efficiency advantages at higher resolutions remain unexplored.
- Validation is limited to the coarse-to-fine paradigm; integration with iterative update frameworks (e.g., RAFT-Stereo-derived methods) is worth exploring.
- The dynamic scanning strategy requires at least 4 source views to achieve omnidirectional coverage.

## Related Work & Insights

- VMamba introduced four-directional scanning for visual tasks; MVSMamba further advances this with reference-centered design and dynamic starting coordinates.
- TransMVSNet first applied Transformers to MVS but requires alternating self-attention and cross-attention, resulting in low efficiency.
- JamMa proposed Joint Mamba for feature matching, but its joint scan is ill-suited for the one-to-many MVS scenario.
- EVMamba's skip scan improves efficiency but limits the receptive field; MVSMamba resolves this via dynamic starting coordinates.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](../../CVPR2025/3d_vision/mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)
- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](../../CVPR2025/3d_vision/mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[ICCV 2025\] MeshMamba: State Space Models for Articulated 3D Mesh Generation and Reconstruction](../../ICCV2025/3d_vision/meshmamba_state_space_models_for_articulated_3d_mesh_generation_and_reconstructi.md)
- [\[ICCV 2025\] Thermal Polarimetric Multi-view Stereo](../../ICCV2025/3d_vision/thermal_polarimetric_multi-view_stereo.md)
- [\[ICCV 2025\] Global-Aware Monocular Semantic Scene Completion with State Space Models](../../ICCV2025/3d_vision/global-aware_monocular_semantic_scene_completion_with_state_space_models.md)

</div>

<!-- RELATED:END -->
