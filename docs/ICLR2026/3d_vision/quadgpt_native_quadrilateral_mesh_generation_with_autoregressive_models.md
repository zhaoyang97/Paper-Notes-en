---
title: >-
  [Paper Note] QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models
description: >-
  [ICLR 2026][3D Vision][quad mesh generation] This paper presents QuadGPT—the first end-to-end autoregressive framework for native quadrilateral mesh generation. It achieves comprehensive superiority over existing triangle-to-quad conversion pipelines and cross-field-guided methods in Chamfer Distance, Hausdorff Distance, quad ratio, and user preference, via unified mixed-topology tokenization (padding triangular faces into 4-vertex blocks), a Hourglass Transformer architecture, and topology-reward-based truncated DPO (tDPO) fine-tuning.
tags:
  - ICLR 2026
  - 3D Vision
  - quad mesh generation
  - autoregressive model
  - mixed topology
  - tDPO
  - Hourglass Transformer
date: 2026-05-08
content_hash: 6408da82f76ed40b
---

# QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models

**Conference**: ICLR 2026
**arXiv**: [2509.21420](https://arxiv.org/abs/2509.21420)
**Code**: Unavailable (API release planned)
**Area**: 3D Generation / Mesh Generation
**Keywords**: quad mesh generation, autoregressive model, mixed topology, tDPO, Hourglass Transformer

## TL;DR

This paper presents QuadGPT—the first end-to-end autoregressive framework for native quadrilateral mesh generation. It achieves comprehensive superiority over existing triangle-to-quad conversion pipelines and cross-field-guided methods in Chamfer Distance, Hausdorff Distance, quad ratio, and user preference, via unified mixed-topology tokenization (padding triangular faces into 4-vertex blocks), a Hourglass Transformer architecture, and topology-reward-based truncated DPO (tDPO) fine-tuning.

## Background & Motivation

**State of the Field**: Quadrilateral (quad) meshes are the industry standard in games and visual effects, ensuring modeling efficiency, subdivision surface smoothness, deformation stability, and convenient UV unwrapping. Existing 3D generation methods either extract dense, unstructured triangle meshes via implicit representations and isosurface extraction (e.g., Marching Cubes), or perform quadrangulation on existing meshes via cross-field guidance—the latter requiring clean inputs and lacking robustness.

**Limitations of Prior Work**: Autoregressive mesh generation methods (MeshAnything, BPT, DeepMesh, Mesh-RFT) have demonstrated the ability to generate triangle meshes with artist-like topology, but are restricted to triangles. Converting triangular outputs to quads still relies on heuristic merging algorithms (e.g., triangle-pair merging), and such post-processing often disrupts natural edge flow and introduces topological artifacts. Even high-quality triangle meshes are difficult to translate into production-ready quad layouts.

**Root Cause**: Generative methods can learn artist-like topology but are confined to triangles, while post-processing methods can produce quads but cannot guarantee global topological quality—the problem lies in the artificial decoupling of quad mesh generation from post-processing.

**Starting Point**: If an autoregressive model could directly predict sequences of quad faces, it could learn the global structure of quad topology (edge loops, edge density distribution) end-to-end, without requiring a triangular intermediate step. The key challenges are: (1) how to represent mixed topology (real artist meshes are typically quad-dominant with a small number of triangles); and (2) how to optimize global topological quality (cross-entropy loss only optimizes local token prediction).

**Core Idea**: Achieve the first native quad mesh autoregressive generation via unified face-block representation combined with topology-aware RL fine-tuning.

## Method

### Overall Architecture

QuadGPT takes a point cloud ($N_p = 40960$ points with normals) as input and outputs a quad-dominant face sequence end-to-end. The pipeline consists of three stages: (1) serializing mixed-topology meshes into uniform-length token blocks; (2) autoregressive pre-training with a Hourglass Transformer to learn basic geometry and connectivity distributions; and (3) RL fine-tuning via truncated DPO (tDPO) to optimize global topological quality. At inference, the model supports a context window of 36,864 tokens, employs a sampling strategy of top-k=10, top-p=0.95, T=0.5, and achieves approximately 230 tokens/s on a single A100.

### Key Designs

1. **Unified Mixed-Topology Serialization**:

    - Function: Unifies triangular and quadrilateral faces into fixed-length token blocks, enabling the Transformer to naturally handle mixed topology.
    - Mechanism: Each face—regardless of type—is serialized into a fixed-length block of 12 tokens. Quad faces are flattened directly from the 3D coordinates of 4 vertices ($4 \times 3 = 12$ tokens). Triangular faces are prepended with 3 padding tokens $\tau_{\text{pad}}$ (integer value 1024), followed by the coordinates of 3 vertices ($3 + 3 \times 3 = 12$ tokens). Vertex coordinates are normalized to $[-0.95, 0.95]^3$ and quantized at 1024 levels (10-bit precision). All vertices are sorted in $(z, x, y)$ lexicographic order to ensure a deterministic sequence representation.
    - Design Motivation: Uniform-length blocks create a highly parallelizable tokenization pipeline and simplify the model architecture. The model implicitly learns face type from the presence of padding tokens, without requiring explicit type tokens. This is considerably more elegant than designing separate encoding paths for triangles and quads.

2. **Hourglass Transformer + Curriculum Learning Pre-training**:

    - Function: Efficiently processes long sequences (token counts for high-resolution meshes can reach tens of thousands) while stably learning complex quad topology.
    - Mechanism: A multi-level hourglass architecture processes the input sequence. Initial token embeddings $\mathbf{E}^{(0)} \in \mathbb{R}^{L \times D_0}$ first pass through a Transformer Block, then are compressed by a causality-preserving shortening layer by a factor of 3 to $\mathbb{R}^{(L/3) \times D_1}$, and further compressed by a factor of 4 to $\mathbb{R}^{(L/12) \times D_2}$, where a bottleneck layer efficiently captures global context, before upsampling back to the original length. The model has 1.1B parameters across 24 Transformer layers. The training strategy initializes from weights pre-trained on pure triangle meshes, then gradually anneals the training data distribution via a quad-dominance parameter $r \in [0, 1]$—transitioning from pure triangles ($r=0$) to quad-dominant meshes ($r \to 1$). Point clouds are encoded into global shape embeddings via a pre-trained Michelangelo encoder and injected into the decoder via cross-attention.
    - Design Motivation: Directly training quad generation is unstable (predicting a quad face is equivalent to simultaneously predicting two correlated triangles); curriculum learning allows the model to first master basic geometric grammar before learning the more complex rules of quad topology. The hierarchical compression of the Hourglass architecture substantially improves computational efficiency for long sequences.

3. **Truncated DPO (tDPO) Topology Fine-tuning**:

    - Function: Optimizes global topological properties (edge loop continuity, reduction of fractures), compensating for the limitation of cross-entropy loss, which can only optimize local token prediction.
    - Mechanism: A topology scoring criterion is designed, primarily rewarding the formation of long continuous edge loops ($L_{\text{avg}}$) and penalizing generated fractures ($R_{\text{frac}}$). Candidate meshes are generated from the current policy $\pi_\theta$, ranked by topological reward to construct preference pairs $(y_w, y_l)$ satisfying $L_{\text{avg}}(y_w) > L_{\text{avg}}(y_l)$ and $R_{\text{frac}}(y_w) < R_{\text{frac}}(y_l)$. The key innovation is **truncation**: since full mesh sequences are excessively long, DPO optimization is applied over a window of length $\tau = 36864$ starting at a random position $m$. The tDPO loss is: $\mathcal{L}_{\text{tDPO}}(\theta) = -\mathbb{E}_{\mathcal{D}}\mathbb{E}_m[\log\sigma(\beta[\log\frac{\pi_\theta(y_{w,m:m+\tau}|x)}{\pi_{\text{ref}}(y_{w,m:m+\tau}|x)} - \log\frac{\pi_\theta(y_{l,m:m+\tau}|x)}{\pi_{\text{ref}}(y_{l,m:m+\tau}|x)}])]$. RL fine-tuning requires only 4 hours on 64 A100 GPUs.
    - Design Motivation: Topological quality (e.g., edge loop coherence, presence of fractures) is an emergent global property that cross-entropy loss cannot directly optimize. The truncation strategy enables DPO to scale to high-face-count meshes—making locally optimal decisions to achieve globally superior topology. Preference data is constructed from 500 high-quality dense meshes (from Hunyuan3D 2.5), yielding approximately 2,000 preference pairs.

### Data Strategy

A pre-training dataset of 1.3 million high-quality quad models is constructed through large-scale data curation, sourced from ShapeNetV2, 3D-FUTURE, Objaverse, Objaverse-XL, and professionally licensed assets, processed via automatic triangle-to-quad conversion and multi-stage quality filtering.

## Key Experimental Results

### Main Results

| Method | Dense CD ↓ | Dense HD ↓ | Dense QR ↑ | Dense US ↑ | Artist CD ↓ | Artist HD ↓ | Artist QR ↑ | Artist US ↑ |
|------|-----------|-----------|-----------|-----------|------------|------------|------------|------------|
| QuadriFlow | 0.045 | 0.099 | 100% | 1.6 | 0.281 | 0.531 | 100% | 0.3 |
| MeshAnythingV2 | 0.153 | 0.394 | 53% | 1.4 | 0.096 | 0.251 | 60% | 2.1 |
| BPT | 0.115 | 0.283 | 43% | 2.7 | 0.051 | 0.125 | 49% | 3.1 |
| DeepMesh | 0.246 | 0.435 | 64% | 3.3 | 0.236 | 0.417 | 66% | 2.8 |
| FastMesh | 0.105 | 0.257 | 3% | 1.1 | 0.052 | 0.141 | 17% | 1.9 |
| **QuadGPT** | **0.057** | **0.147** | **80%** | **4.9** | **0.043** | **0.095** | **78%** | **4.8** |

QuadGPT leads comprehensively across all metrics: CD on dense meshes is more than 46% lower than the best baseline, user study scores (US) are decisively superior (4.9 vs. 3.3), and QR reaches 80% (only QuadriFlow achieves a higher 100%, but at substantially worse geometric quality and with frequent failures).

### Ablation Study

| Training Strategy | CD ↓ | HD ↓ | QR ↑ | US ↑ |
|---------|------|------|------|------|
| From Scratch | 0.081 | 0.203 | 75% | 0.6 |
| Finetune (Curriculum Learning) | 0.065 | 0.167 | 72% | 1.3 |
| DPO (Full Sequence) | 0.073 | 0.188 | 74% | 1.1 |
| tDPO (Truncated) | 0.061 | 0.156 | 78% | 3.3 |
| **tDPO-Pro (Full Reward)** | **0.057** | **0.147** | **80%** | **3.7** |

| Configuration | CD ↓ | HD ↓ | QR ↑ | US ↑ |
|---------|------|------|------|------|
| TriGPT (Triangle→Quad Conversion) | 0.062 | 0.160 | 70% | 0.2 |
| TriGPT+RL (Same + RL Fine-tuning) | 0.051 | 0.138 | 72% | 0.5 |
| **QuadGPT (Native Quad)** | **0.057** | **0.147** | **80%** | **1.3** |

### Key Findings

- **Curriculum learning is critical for training stability**: Training quad generation from scratch (From Scratch) suffers from convergence difficulties, with CD as high as 0.081; curriculum learning initialization (Finetune) reduces it to 0.065. Predicting a quad face is equivalent to predicting two correlated triangles, necessitating a foundation built on simpler tasks.
- **Standard DPO fails to generalize to complex meshes**: Full-sequence DPO fine-tuning on low-face-count meshes fails to generalize to complex, high-face-count meshes (CD actually increases from 0.065 to 0.073). tDPO resolves this through truncated training.
- **Decisive gap between native generation and conversion**: In a fully controlled comparison (TriGPT and QuadGPT sharing the same architecture, data, and RL strategy), TriGPT+RL achieves marginally better CD/HD, but the gaps in QR (72% vs. 80%) and user preference (0.5 vs. 1.3) are substantial—post-processing conversion cannot recover natural edge flow.

## Highlights & Insights

- **Minimalist elegance of the padding strategy**: Using 3 padding tokens to unify triangles into the 12-token block of quads—no type tokens, no branching encoding paths—the model implicitly learns face type from the padding pattern. This design maximizes sequence regularity and parallelism.
- **Rationale for curriculum learning**: A quad face is topologically equivalent to two correlated triangles, so learning triangles before quads constitutes a natural curriculum aligned with cognitive gradients—this simple-to-complex training strategy is systematically validated in mesh generation for the first time.
- **Scalability design of tDPO**: The truncation strategy for extending DPO to long-sequence generation warrants reference in other long-sequence RL tasks (e.g., long-form text generation, music generation).

## Limitations & Future Work

- **QR does not reach 100%**: A quad ratio of 80% implies that 20% of faces remain triangular, leaving a gap from pure quad meshes.
- **Strong data dependency**: The curation cost of 1.3 million high-quality quad meshes is substantial, and the paper acknowledges that data quality is critical to performance—making reproduction difficult.
- **Point cloud input only**: No complete pipeline from text/image to quad mesh is demonstrated; the approach depends on an external model (e.g., Hunyuan3D) to first generate the point cloud.
- **Limited inference speed**: At 230 tokens/s for a 36,864-token context, each mesh requires approximately 2.7 minutes, which may necessitate further acceleration for production environments.

## Related Work & Insights

- **vs. MeshAnything/BPT/DeepMesh**: These methods generate high-quality triangle meshes but require post-processing conversion to quads, during which the edge flow structure is disrupted. QuadGPT's end-to-end approach fundamentally avoids this problem.
- **vs. QuadriFlow**: Cross-field-guided methods can produce perfect quad meshes (QR=100%) on ideal inputs, but are highly non-robust to complex topology or sharp features—frequently producing failures. QuadGPT substantially outperforms in robustness and user preference.
- **vs. Mesh-RFT (RL for meshes)**: Mesh-RFT applies RL to triangle mesh generation; QuadGPT's tDPO extends analogous ideas to topological quality optimization for quads, with the truncation strategy as the key contribution.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First autoregressive framework for native quad mesh generation; both unified serialization and tDPO represent innovative contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive ablation (curriculum learning / DPO variants / native vs. conversion), rich baselines, and user studies included.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed method description, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ Fills the gap in native quad mesh generation with direct applicability to the games and visual effects industries.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](../../CVPR2026/3d_vision/pixarmesh_autoregressive_mesh-native_single-view_scene_reconstruction.md)
- [\[AAAI 2026\] Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation](../../AAAI2026/3d_vision/learning_conjugate_direction_fields_for_planar_quadrilateral_mesh_generation.md)
- [\[NeurIPS 2025\] ARMesh: Autoregressive Mesh Generation via Next-Level-of-Detail Prediction](../../NeurIPS2025/3d_vision/armesh_autoregressive_mesh_generation_via_next-level-of-detail_prediction.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICLR 2026\] PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data](partsam_a_scalable_promptable_part_segmentation_model_trained_on_native_3d_data.md)

<!-- RELATED:END -->
