---
title: >-
  [Paper Note] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning
description: >-
  [NeurIPS 2025][3D Vision][mesh generation] This paper proposes Mesh-RFT, a framework that achieves face-level fine-grained mesh quality optimization through a topology-aware scoring system and Masked Direct Preference Op…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "mesh generation"
  - "reinforcement fine-tuning"
  - "DPO"
  - "topology-aware"
  - "fine-grained optimization"
date: 2026-05-08
content_hash: 2f43aca9e15423a9
---

# Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2505.16761](https://arxiv.org/abs/2505.16761)
**Code**: [Project Page](https://hitcslj.github.io/mesh-rft/)
**Area**: 3D Vision / Mesh Generation
**Keywords**: mesh generation, reinforcement fine-tuning, DPO, topology-aware, fine-grained optimization

## TL;DR

This paper proposes Mesh-RFT, a framework that achieves face-level fine-grained mesh quality optimization through a topology-aware scoring system and Masked Direct Preference Optimization (M-DPO), significantly improving the geometric integrity and topological regularity of generated meshes.

## Background & Motivation

High-quality 3D mesh generation faces two major challenges:

**Limitations of Prior Work**: Autoregressive mesh generation methods (MeshGPT, MeshXL, etc.) are prone to structural ambiguities and "hallucinations" (inconsistent edges, non-manifold vertices, deformations, holes) when generating long-sequence, high-resolution meshes.

**Shortcomings of Global Reinforcement Learning**: DeepMesh applies DPO for preference alignment but relies on manually annotated preference pairs (only 5,000 samples), and its global reward signal fails to capture local topological variations. A key observation is that high-quality and low-quality structures frequently coexist within the same mesh.

Core problem: How to achieve face-level fine-grained optimization rather than applying a uniform global reward to the entire mesh?

## Method

### Overall Architecture

A three-stage pipeline:
1. **Pre-training**: Supervised learning using a Hourglass Autoregressive Transformer with a Shape Encoder.
2. **Preference Dataset Construction**: The pre-trained model generates candidate meshes, and a topology-aware scoring system establishes preference pairs.
3. **Post-training**: Fine-grained reinforcement fine-tuning using Masked DPO.

### Key Designs

1. **Topology-Aware Scoring System**: Two objective topological metrics are proposed to replace manual annotation:

    - **Boundary Edge Ratio (BER)**: $BER(\mathcal{M}) = E_{\partial\mathcal{M}} / E_{\mathcal{M}}$, measuring mesh integrity. A closed manifold mesh should have BER = 0; a high BER indicates surface discontinuities, holes, and similar issues.
    - **Topology Score (TS)**: $TS(\mathcal{M}) = \sum_{i=1}^{4} w_i s_i(\mathcal{Q}(\mathcal{M}))$, evaluated by converting the triangle mesh to a quadrilateral mesh and assessing: quad ratio (0.4), angle quality (0.2), aspect ratio (0.3), and adjacency consistency (0.1).
    - **Hausdorff Distance (HD)** is additionally used to measure geometric consistency.

2. **Preference Dataset Construction**: Eight candidate meshes are generated per input point cloud, yielding $C(8,2)=28$ exhaustive pairs. A preference relation is established only when one mesh strictly outperforms another across all three metrics—BER, TS, and HD—thereby avoiding ambiguous preferences.

3. **Masked Direct Preference Optimization (M-DPO)**: The core innovation. Quality is assessed at the individual triangle face level to construct a token-level binary mask $\phi(\mathcal{M}) \in \{0,1\}^{|\mathcal{M}|}$. For chosen samples, the mask amplifies contributions from high-quality regions; for rejected samples, the inverted mask focuses the penalty on low-quality regions:

    - $\mathcal{L}^+$: tokens corresponding to high-quality regions in chosen samples are selected by the mask.
    - $\mathcal{L}^-$: tokens corresponding to low-quality regions in rejected samples are selected by the inverted mask.

    This achieves the effect of preserving good regions while focusing repair efforts on defective regions.

### Loss & Training

- Pre-training: truncated training (fixed-length segments) + sliding-window inference (sliding begins after 40% window coverage, retaining the most recent 30%).
- M-DPO loss: $\mathcal{L}_{M-DPO} = -\mathbb{E}[\log \sigma(\beta \mathcal{L}^+ - \beta \mathcal{L}^-)]$
- Model architecture: Hourglass Transformer (with 2 shortening and 2 upsampling operations); the Hunyuan3D 2.0 point cloud encoder is injected via cross-attention.
- Pre-training: 256 × H20 GPUs, 10 days; M-DPO: 64 GPUs, 8 hours, learning rate 5e-7.
- Training data: 2M meshes for pre-training, 800K filtered meshes for fine-tuning, 10K meshes for preference data construction.

## Key Experimental Results

### Main Results

| Method | CD↓ | HD↓ | TS↑ | BER↓ | US↑ |
|--------|-----|-----|-----|------|-----|
| MeshAnythingV2 | 0.2265 | 0.4760 | 72.0 | 0.0913 | 8% |
| BPT | 0.1615 | 0.3347 | 73.7 | 0.0113 | 18% |
| DeepMesh* (0.5B) | 0.1760 | 0.3570 | 75.8 | 0.0044 | 20% |
| **Mesh-RFT** | **0.1286** | **0.2411** | **79.4** | **0.0015** | **40%** |

(Dense Meshes results; user preference score US improves from 20% to 40%.)

### Ablation Study

| Configuration | CD↓ | HD↓ | TS↑ | BER↓ | US↑ |
|---------------|-----|-----|-----|------|-----|
| Pretrain | 0.1588 | 0.3196 | 76.5 | 0.0033 | 30% |
| N-DPO (HD only) | 0.1455 | 0.2919 | 75.7 | 0.0028 | 32% |
| S-DPO (scoring system) | 0.1348 | 0.2625 | 77.9 | 0.0023 | 35% |
| **M-DPO (masked)** | **0.1286** | **0.2411** | **79.4** | **0.0015** | **40%** |

### Key Findings

- Compared to the pre-trained model, M-DPO reduces HD by 24.6% and improves TS by 3.8%.
- Compared to global DPO (S-DPO), M-DPO reduces HD by 17.4% and improves TS by 4.9%.
- Using HD alone as the preference criterion (N-DPO) actually degrades TS, demonstrating the necessity of multi-metric composite scoring.
- M-DPO achieves 40% user preference (vs. 30% for the pre-trained model), validating perceptual quality gains.
- Strong performance on out-of-distribution Hunyuan2.5-generated meshes demonstrates generalization capability.

## Highlights & Insights

- **First face-level RL optimization method**: Breaks the limitation of global reward signals by enabling precise repair of local defects.
- The **objective topology scoring system** replaces manual annotation and offers strong scalability (vs. DeepMesh's 5,000 annotated samples).
- The BER and TS designs are elegant: evaluating triangle mesh quality via quadrilateral conversion aligns well with the industrial preference for quad meshes.
- The engineering design of truncated training combined with sliding-window inference addresses practical challenges in long-sequence mesh generation.

## Limitations & Future Work

- Only point-cloud-conditioned generation is evaluated; text- and image-conditioned variants remain unexplored.
- The comparison with DeepMesh is limited to the 0.5B version, which may not be fully fair.
- The weights $w_i$ in the scoring system are manually specified; adaptive learning could be considered.
- Using quadrilateral quality as a proxy metric for triangle mesh quality may introduce bias.
- The number of faces and resolution of generated meshes are constrained by sequence length.

## Related Work & Insights

- Compared to DeepMesh, which uses global DPO with manual annotations, Mesh-RFT employs local M-DPO with automated scoring.
- Transferring DPO/RLHF paradigms from NLP to 3D mesh generation is a growing trend, but adaptation to 3D structural properties is essential.
- The Masked DPO concept is generalizable to other sequence generation scenarios with significant local quality variation (e.g., code generation, music generation).
- The hierarchical design of the Hourglass Transformer provides a useful reference for long-sequence generation tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First face-level RL optimization combined with an objective topology scoring system; the M-DPO design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Complete ablations with user studies and OOD testing, though the number of baselines is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Rich figures and tables; method description is clear.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable to production-level mesh generation; the objective scoring system is reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[NeurIPS 2025\] Fin3R: Fine-tuning Feed-forward 3D Reconstruction Models via Monocular Knowledge Distillation](fin3r_fine-tuning_feed-forward_3d_reconstruction_models_via_monocular_knowledge_.md)
- [\[ICCV 2025\] DeepMesh: Auto-Regressive Artist-Mesh Creation with Reinforcement Learning](../../ICCV2025/3d_vision/deepmesh_auto-regressive_artist-mesh_creation_with_reinforcement_learning.md)
- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](../../ICCV2025/3d_vision/meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[NeurIPS 2025\] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)

</div>

<!-- RELATED:END -->
