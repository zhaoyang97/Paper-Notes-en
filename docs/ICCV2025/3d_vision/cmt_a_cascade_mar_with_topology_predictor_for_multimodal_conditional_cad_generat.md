---
title: >-
  [Paper Note] CMT: A Cascade MAR with Topology Predictor for Multimodal Conditional CAD Generation
description: >-
  [ICCV 2025][3D Vision][CAD generation] This paper proposes CMT, the first multimodal CAD generation framework based on B-Rep representation. By employing a cascade MAR (edges first, then faces) and a topology predictor, CMT achieves accurate topology and geometry generation. The authors also construct mmABC, a multimodal CAD dataset of over 1.3 million models.
tags:
  - ICCV 2025
  - 3D Vision
  - CAD generation
  - B-Rep
  - cascade autoregression
  - topology prediction
  - multimodal conditional generation
date: 2026-05-08
content_hash: f6952cc2a0058c42
---

# CMT: A Cascade MAR with Topology Predictor for Multimodal Conditional CAD Generation

**Conference**: ICCV 2025
**arXiv**: [2504.20830](https://arxiv.org/abs/2504.20830)
**Code**: N/A (dataset mmABC is publicly available)
**Area**: 3D Vision / CAD Generation
**Keywords**: CAD generation, B-Rep, cascade autoregression, topology prediction, multimodal conditional generation

## TL;DR

This paper proposes CMT, the first multimodal CAD generation framework based on B-Rep representation. By employing a cascade MAR (edges first, then faces) and a topology predictor, CMT achieves accurate topology and geometry generation. The authors also construct mmABC, a multimodal CAD dataset of over 1.3 million models.

## Background & Motivation

Traditional CAD design requires manually executing a complex pipeline of 2D sketching → 3D operations → B-Rep conversion, which is time-consuming and demands professional expertise. An ideal CAD generation tool should simultaneously satisfy: (1) topological and geometric accuracy; (2) support for multimodal conditional inputs (text, image, point cloud); and (3) accessibility for non-expert users.

Existing methods suffer from fundamental limitations:
- **MLLM-based methods** (DeepCAD, CAD-MLLM, SolidGen): serialize CAD models into discrete tokens for sequential generation, but are constrained by discrete representations — supporting only line/arc primitives and extrusion operations, and unable to model chamfers, fillets, or freeform surfaces.
- **Diffusion-based methods** (BrepGen): can generate continuous B-Rep models, but struggle with multimodal conditional inputs.

The recently proposed MAR (Masked AutoRegressive) architecture unifies the token dependency modeling of autoregression with the continuous distribution learning of diffusion, offering an opportunity for "accurate + multimodal" CAD generation. However, vanilla MAR lacks an inherent mechanism to capture B-Rep topological relationships (the vertex–edge–face hierarchy).

## Method

### Overall Architecture

CMT consists of four components: (1) continuous B-Rep tokenization; (2) a unified multimodal condition encoder; (3) a cascade autoregressive generation network (Edge MAR → Surface MAR); and (4) a topology predictor. The overall pipeline is: B-Rep → continuous tokenization → condition encoding → cascade edge-then-face generation → topology relation prediction → complete B-Rep.

### Key Designs

1. **Continuous Tokenization**:

    - **Face tokens**: Surface geometry (uniformly sampled points) is encoded via a Surface VAE, with bounding box coordinates concatenated as topological features.
    - **Edge tokens**: Edge geometry is encoded via an Edge VAE, with bounding box coordinates and adjacent vertex coordinates concatenated. Vertex information is integrated into edge tokens, eliminating the need for separate vertex generation.
    - **Ordering**: Tokens are sorted in ascending order by 3D bounding box coordinates $(x_1, y_1, z_1, x_2, y_2, z_2)$, enabling deterministic serialization.

2. **Cascade Autoregressive Network (CAN)**:
   Following the topological prior of B-Rep that "edges bound faces," a two-stage generation pipeline (edges first, then faces) is designed:
    - **Edge MAR**: Given condition embedding $Z$ and visible edge tokens $E_v$, an edge transformer $\mathcal{G}_e$ generates features $c_e$ for masked edge tokens, which are then denoised into edge tokens via an edge diffusion MLP $\mathcal{D}_e$.
    - **Surface MAR**: A variable-length edge sequence is compressed into a fixed-length edge condition embedding $Q$ via an edge projector (self-attention + learnable edge tokens). Combined with $Z$ and visible face tokens $S_v$, a surface transformer $\mathcal{G}_s$ and surface diffusion $\mathcal{D}_s$ generate face tokens.

   The cascade design allows face generation to leverage the already-generated edge information, reducing modeling difficulty.

3. **Topology Predictor**:
   A simple cross-attention layer directly predicts the adjacency matrix $A \in \mathbb{R}^{N_e \times N_s}$ from generated edge tokens $\hat{E}$ and face tokens $\hat{S}$, with a threshold $\tau = 0.5$ to determine edge–face adjacency. This is **4200× faster** than Point2CAD's post-processing algorithm (256 CAD models: 0.038s vs. 161.15s).

4. **Unified Multimodal Condition Encoder**: Text and images are processed using the CLIP tokenizer; point clouds are processed via 3D convolution. All modalities are encoded by a frozen CLIP-ViT encoder, and a learnable projector generates fixed-length condition embeddings $Z$.

### Loss & Training

The total loss is $L = L_{edge} + L_{surf} + L_{topo}$, where:
- $L_{edge}$ and $L_{surf}$: diffusion denoising losses constraining the MSE between predicted and ground-truth noise.
- $L_{topo}$: MSE loss for adjacency matrix prediction.

Training: unconditional generation for 2100 epochs + conditional generation for 1000 epochs, on 8× A100 GPUs. Maximum sequence lengths: 64 edges / 32 faces on DeepCAD; 128 edges / 64 faces on ABC. At inference, the default setting generates 1 token per step.

## Key Experimental Results

### Main Results

**Unconditional Generation (DeepCAD & ABC)**:

| Dataset | Method | COV↑ | MMD↓ | JSD↓ | Valid↑ |
|--------|------|------|------|------|--------|
| DeepCAD | DeepCAD | 65.46 | 1.29 | 1.67 | 46.1 |
| DeepCAD | SolidGen | 71.03 | 1.08 | 1.31 | 60.3 |
| DeepCAD | BrepGen | 71.26 | 1.04 | 0.09 | 62.9 |
| DeepCAD | **CMT** | **75.71** | **0.92** | 1.02 | **70.1** |
| ABC | BrepGen | 57.92 | 1.35 | 3.69 | 48.2 |
| ABC | **CMT** | **68.60** | 1.35 | **2.79** | **58.5** |

**Conditional Generation (mmABC)**:

| Condition | Method | Chamfer↓ | F-score↑ | Normal C↑ |
|------|------|----------|----------|-----------|
| Point Cloud | DeepCAD | 5.11 | 83.56 | 69.58 |
| Point Cloud | NVDNet (reconstruction) | 0.77 | 98.17 | 94.36 |
| Point Cloud | **CMT** | **0.64** | **99.07** | **95.48** |
| Image | InstantMesh | 6.18 | 84.71 | 52.72 |
| Image | **CMT** | **2.17** | **92.93** | **70.14** |
| Text | Michelangelo | 25% (GPT-4o win) | 20% (Human win) | - |
| Text | **CMT** | **75%** (GPT-4o) | **80%** (Human) | - |

### Ablation Study

| Cascade | Sampling Steps | COV↑ | Valid↑ |
|------|---------|------|--------|
| ✓ | 64/32 | **75.71** | **70.1** |
| ✗ | 64+32 | 65.80 | 47.17 |
| ✓ | 32/16 | 74.97 | 67.93 |
| ✓ | 16/8 | 70.26 | 47.80 |
| ✓ | 8/4 | N.A. | 1.20 |
| ✓ | 1/1 | N.A. | 0.10 |

### Key Findings

- The cascade network yields **+9.91% Coverage and +22.96% Valid**, demonstrating that the edges-first-then-faces paradigm is critical for B-Rep generation.
- CMT **surpasses the SOTA reconstruction method** NVDNet on point-cloud-conditioned generation (+0.90 F-score, +1.12 Normal C).
- Generation quality is strongly correlated with sampling steps: at 1/1 step (all tokens generated simultaneously), Valid is only 0.10%, whereas token-by-token generation achieves 70.1%.
- The topology predictor is 4200× faster than post-processing algorithms.
- mmABC is currently the largest multimodal B-Rep dataset (1.3M+), supporting freeform surface modeling.

## Highlights & Insights

1. **First multimodal B-Rep generation framework**: CMT fills the gap of "accurate B-Rep + multimodal conditioning."
2. **Cascade design encodes domain knowledge**: the topological prior that edges bound faces is naturally embedded into the architecture.
3. **Successful application of MAR**: CMT extends MAR from image generation to CAD generation, demonstrating its broad applicability.
4. **Large-scale dataset contribution**: the construction of mmABC (multi-body decomposition + deduplication + VLM annotation) lays the groundwork for future CAD generation research.

## Limitations & Future Work

- The maximum sequence length is limited (128 edges / 64 faces), covering only 95% of the data; more complex models cannot be generated.
- Evaluation of text-conditioned generation relies on human and VLM scoring, lacking automated quantitative metrics.
- Text descriptions in the dataset are automatically generated by a VLM, which may result in inconsistent quality.
- More complex tasks such as multi-body generation and model completion are not supported (explicitly noted as future work in the paper).
- Inference speed: since sampling steps equal sequence length in token-by-token generation, speed may become a bottleneck.

## Related Work & Insights

- **BrepGen**: current SOTA in B-Rep generation; CMT significantly outperforms it on all metrics.
- **MAR (Kaiming He)**: the architectural foundation of CMT; its successful application demonstrates MAR's utility beyond image generation.
- **Point2CAD**: a method for reconstructing CAD models from point clouds; its post-processing topology algorithm is directly superseded by CMT's topology predictor.
- **OneLLM**: a source of inspiration for the unified multimodal encoder.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First multimodal B-Rep generation framework; the cascade design and topology predictor are highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers unconditional + three conditional generation settings, with thorough ablations and comprehensive comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear and the pipeline is described in a well-organized manner.
- **Value**: ⭐⭐⭐⭐⭐ Dual contributions of dataset and method; highly significant for the CAD and industrial design communities.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AnyI2V: Animating Any Conditional Image with Motion Control](anyi2v_animating_any_conditional_image_with_motion_control.md)
- [\[ICCV 2025\] SHeaP: Self-Supervised Head Geometry Predictor Learned via 2D Gaussians](sheap_self-supervised_head_geometry_predictor_learned_via_2d_gaussians.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[ICLR 2026\] Topology-Preserved Auto-regressive Mesh Generation in the Manner of Weaving Silk](../../ICLR2026/3d_vision/topology-preserved_auto-regressive_mesh_generation_in_the_manner_of_weaving_silk.md)
- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](egom2p_egocentric_multimodal_multitask_pretraining.md)

<!-- RELATED:END -->
