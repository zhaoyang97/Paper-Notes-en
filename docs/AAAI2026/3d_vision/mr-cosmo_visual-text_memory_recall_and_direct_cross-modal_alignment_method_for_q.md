---
title: >-
  [Paper Note] MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation
description: >-
  [AAAI 2026][3D Vision][3D segmentation] This paper proposes MR-CoSMo, a coarse-to-fine query-driven 3D segmentation model that establishes explicit alignment between 3D point clouds and text/2D images via a Direct Cross-Modal Alignment module (DCMA), and incorporates a Visual-Text Memory Module to store high-confidence feature pairs for enhanced cross-scene segmentation consistency. The method achieves state-of-the-art performance across three tasks: 3D instruction segmentation, referring segmentation, and semantic segmentation.
tags:
  - AAAI 2026
  - 3D Vision
  - 3D segmentation
  - cross-modal alignment
  - visual-text memory
  - point cloud segmentation
  - query-driven segmentation
date: 2026-05-08
content_hash: 1110cfd9f87836d5
---

# MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation

**Conference**: AAAI 2026
**arXiv**: [2506.20991](https://arxiv.org/abs/2506.20991)
**Code**: None
**Area**: 3D Vision
**Keywords**: 3D segmentation, cross-modal alignment, visual-text memory, point cloud segmentation, query-driven segmentation

## TL;DR

This paper proposes MR-CoSMo, a coarse-to-fine query-driven 3D segmentation model that establishes explicit alignment between 3D point clouds and text/2D images via a Direct Cross-Modal Alignment module (DCMA), and incorporates a Visual-Text Memory Module to store high-confidence feature pairs for enhanced cross-scene segmentation consistency. The method achieves state-of-the-art performance across three tasks: 3D instruction segmentation, referring segmentation, and semantic segmentation.

## Background & Motivation

Text-guided 3D segmentation aims to leverage natural language inputs to segment 3D objects/scenes, which is a critical capability for autonomous driving and embodied intelligence. Existing methods suffer from several fundamental limitations:

**Indirect alignment strategies**: Methods such as PointCLIP and Seal use 2D images as intermediaries between 3D point clouds and text. This indirect strategy is highly dependent on the accuracy of camera intrinsic and extrinsic parameters, making it susceptible to parameter estimation errors and pixel-point alignment artifacts.

**Insufficient local feature–text context linkage**: Fine-grained segmentation requires recognizing subtle structural variations within objects, demanding a deep understanding of 3D geometry and the ability to capture correspondences between local details and textual context. Existing methods fail to establish stable and accurate coordinate correspondences between 3D point clouds and 2D images/text.

**Class sample imbalance**: Inherent intra-class variation in texture and contextual features across instances of the same category leads to misclassification of co-category objects and poor accuracy on underrepresented classes.

## Method

### Overall Architecture

The model follows a coarse-to-fine architecture:
1. **Coarse stage**: Multi-modal feature extraction → DCMA cross-modal alignment → Multi-layer Transformer updates → Detection head generating 3D bounding boxes
2. **Fine stage**: Point features within bounding boxes + text features → Memory Module enhancement → Binary classifier iteratively generating segmentation masks

**Input**: Point cloud + corresponding 2D images + text query

**Feature extraction**:
- Point cloud: MLP extracts per-point features $f_{point}$ + 4-layer 3D window-shifted Transformer extracts voxel features $f_{voxel}$
- Image: Pre-trained ResNet-50 extracts visual features $f_{image}$
- Text: Frozen LLaMA2-7B extracts text features $f_{txt}$

### Key Designs

#### 1. **Direct Cross-Modal Alignment Module (DCMA)**

DCMA consists of two sub-modules:

**Alignment Constrains Block**: Prior to aligning 3D features, this block applies contrastive learning to constrain the relationship between 2D image features and text features. Independent encoders map image/text features into an embedding space, and a symmetric cross-entropy loss is used to bring matching pairs closer while pushing non-matching pairs apart. This establishes a correct cross-modal semantic foundation for subsequent 3D alignment.

**Bidirectional Direct Alignment Block**:

The core innovation employs **bidirectional Mamba attention** to directly align 3D features with text/image features, bypassing indirect 2D projection.

Modality pairing strategy:
- **Text ↔ per-point features**: Avoids pixel-point misalignment caused by 2D projection
- **Image ↔ voxel features**: Leverages regular voxel structure to reduce geometric distortion

For text-point alignment, a three-element sequence $X = [\phi_{txt}, \phi_{points}, \phi_{txt}^{copy}]$ is constructed and processed via a bidirectional state space model:

Forward pass $(\phi_{txt} \to \phi_{points} \to \phi_{txt}^{copy})$:

$$h_t^f = \tilde{A}_f h_{t-1}^f + \tilde{B}_f X_t$$

$$\psi_t^f = \tilde{C}_f h_t^f + \tilde{D}_f X_t$$

The backward pass processes the sequence in reverse order, and the final aligned feature is:

$$\psi_{point}^* = \text{LayerNorm}(\psi_3^f + \psi_1^b)$$

By placing text features (original and copy) at both ends of the sequence, the forward and backward passes respectively capture text-to-point semantic transfer and cross-modal interaction-refined text representations.

#### 2. **Memory Module**

Addresses segmentation inconsistency caused by class sample imbalance and intra-class variation.

**Feature pair storage**: Text features $f_{txt}^i$ and 3D point features $f_{box}^i$ within detection boxes are stored in dedicated text/visual memory banks, and concatenated to form a feature-pair memory bank:

$$\mathcal{M}_p = \{[f_{txt}^i; f_{box}^i] | i=1,...,N\}$$

**Confidence-weighted storage**: Initial weights are computed based on BCE loss as $w_i^{(\text{init})} = \frac{1}{\mathcal{L}_{BCE_i} + \tau}$ (lower loss → higher confidence → larger weight), and normalized within the same category:

$$w_i = \frac{1}{\mathcal{L}_{BCE_i} + \tau} \cdot \frac{1}{\sum_{j \in C} \frac{1}{\mathcal{L}_{BCE_j} + \tau}}$$

**Three-step attention retrieval**: When processing a new scene:
1. Text self-attention: current text query retrieves from the text memory bank $\mathcal{M}_t$
2. Feature-pair self-attention: concatenated current text and point features retrieve from $\mathcal{M}_p$
3. Cross-attention: text attention output queries the feature-pair attention output

The retrieved result is passed to the binary classifier to generate segmentation masks, whose BCE loss then updates the current feature-pair weights, forming a dynamic weight optimization loop.

### Loss & Training

Total loss: $\mathcal{L}_{all} = \mathcal{L}_{task} + \mathcal{L}_{DCMA}$

$$\mathcal{L}_{task} = \alpha\mathcal{L}_{det} + \beta\mathcal{L}_{seg} = \alpha(\mathcal{L}_{smoothL1} + \mathcal{L}_{WCE}) + \beta\mathcal{L}_{BCE}$$

$$\mathcal{L}_{DCMA} = \mathcal{L}_{SCE} = \gamma(-\sum y_i\log p_i) + \delta(-\sum p_i\log y_i)$$

Training details:
- 4× Nvidia V100 (32G), AdamW optimizer, cosine scheduling
- Initial learning rates: 0.005 (indoor) / 0.002 (outdoor); 500 / 100 training epochs
- LLaMA2-7B is frozen; only vectorized text is used as input
- Memory Module features: float32 → float16; memory footprint < 50MB
- $\tau = 0.05$ (optimal); random seeds 42/888/2026; ≥ 3 runs per experiment; std < 0.2%

## Key Experimental Results

### Main Results

**3D Instruction Segmentation** (Instruct3D/ScanNet++):

| Method | Acc | mIoU |
|--------|-----|------|
| **MR-CoSMo** | **33.8** | **28.5** |
| MR-CoSMo (w/o Memory) | 31.9 | 27.4 |
| SegPoint | 31.6 | 27.5 |
| M3DRef | 18.1 | 12.8 |
| EDA | 16.6 | 12.1 |

**3D Referring Segmentation** (ScanRefer/ScanNet):

| Method | mIoU |
|--------|------|
| **MR-CoSMo** | **45.6** |
| RefMask3D | 44.8 |
| SegPoint | 41.7 |
| 3D-STMN | 39.5 |

**3D Semantic Segmentation**:

| Method | S3DIS Area5 mIoU | SemanticKITTI val mIoU |
|--------|-----------------|----------------------|
| **MR-CoSMo** | **75.6** | **73.4** |
| PTv3+PPT | 74.7 | 72.3 |
| PTv2 | 72.6 | 70.3 |

### Ablation Study

Module ablation on Instruct3D:

| Configuration | mIoU | ΔmIoU | Note |
|---------------|------|-------|------|
| Baseline | 26.4 | +0.0 | Baseline |
| + DCMA | 27.4 | +1.0 | Direct alignment effective |
| + Memory Module | 27.5 | +1.1 | Independent contribution of Memory Module |
| Only Voxel Encoder | 27.7 | +1.3 | Dual encoder outperforms single |
| Reversed matching | 27.9 | +1.5 | Validates modality pairing strategy |
| w/o Alignment Constrains | 28.0 | +1.6 | Contrastive constraint effective |
| w/o Loss on BBox | 28.4 | +2.0 | Box constraint has marginal impact |
| Full model | **28.5** | +2.1 | Full model is optimal |

Backbone replacement ablation:

| Configuration | Speed (fps) | GPU (GB) | mIoU |
|---------------|-------------|----------|------|
| Mamba → Transformer | 2.21 | 30.4 | 28.4 |
| ResNet-50 → ViT | 1.89 | 33.7 | 28.7 |
| LLaMA2-7B → 13B | 2.35 | 33.0 | 28.6 |
| LLaMA2-7B → 2B | 2.74 | 27.6 | 28.2 |
| **Default configuration** | **2.66** | **28.9** | **28.5** |

### Key Findings

- **DCMA + Memory Module jointly improve mIoU by 2.1%**, with each module individually contributing ~1%
- DCMA via Mamba attention (vs. Transformer) improves inference speed (2.66 vs. 2.21 fps) and reduces GPU memory (28.9 vs. 30.4 GB) with comparable performance
- The Memory Module improves Acc from 31.9 to 33.8 (+1.9%), primarily by distinguishing individual instances among multiple similar objects
- MR-CoSMo surpasses PTv3+PPT on semantic segmentation while training on a single dataset (vs. multiple datasets), demonstrating the benefit of category-aware priors
- The temperature parameter $\tau = 0.05$ is optimal; sensitivity analysis confirms the importance of tuning this hyperparameter
- The default backbone achieves the best performance-efficiency trade-off: LLaMA2-7B vs. 13B yields only a 0.1% mIoU difference while being 22% faster

## Highlights & Insights

1. **Direct alignment over indirect alignment**: The proposed approach bypasses error accumulation from 2D projection by establishing direct correspondences between 3D features and text/images.
2. **Sequence construction for bidirectional Mamba**: Placing text feature copies at both ends of the sequence enables the forward and backward passes to respectively realize "text-guided point feature enhancement" and "point-guided text refinement."
3. **Dynamic weight updating in the Memory Module**: Loss-based confidence weighting combined with intra-class normalization elegantly addresses sample imbalance.
4. **Generality**: A unified framework handles three distinct 3D segmentation tasks — instruction, referring, and semantic segmentation.

## Limitations & Future Work

- Inference speed (2.66 fps) is lower than 3D-STMN (3.53 fps)
- Storage and retrieval overhead of the Memory Module grows with the number of training samples
- LLaMA2 is used in a frozen state; fine-tuning may yield further improvements
- Outdoor scene training (SemanticKITTI) uses only 100 epochs, which may be insufficient
- Open-vocabulary scenarios have not been explored

## Related Work & Insights

- SegPoint is the most directly comparable baseline (also leverages LLMs for text understanding)
- RefMask3D is a strong baseline for referring segmentation (44.8 → 45.6 mIoU)
- PTv3 serves as the foundational backbone for semantic segmentation
- The Memory Module paradigm is generalizable to other 3D tasks requiring cross-scene consistency (e.g., 3D object detection, 3D instance segmentation)
- The efficiency of Mamba architectures in 3D processing warrants further exploration

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of direct cross-modal alignment and the Memory Module is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three tasks, four datasets, comprehensive ablation and backbone analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with detailed formulations
- Value: ⭐⭐⭐⭐ — A unified framework for multiple 3D segmentation tasks with strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STMI: Segmentation-Guided Token Modulation with Cross-Modal Hypergraph Interaction for Multi-Modal Object Re-Identification](stmi_segmentation-guided_token_modulation_with_cross-modal_hypergraph_interactio.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](../../ICCV2025/3d_vision/easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)
- [\[AAAI 2026\] NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling](nurbgen_high-fidelity_text-to-cad_generation_through_llm-driven_nurbs_modeling.md)
- [\[NeurIPS 2025\] Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation](../../NeurIPS2025/3d_vision/walking_the_schrödinger_bridge_a_direct_trajectory_for_text-to-3d_generation.md)
- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](point-sra_self-representation_alignment_for_3d_representation_learning.md)

</div>

<!-- RELATED:END -->
