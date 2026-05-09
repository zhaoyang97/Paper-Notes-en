---
title: >-
  [Paper Note] PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior
description: >-
  [ICCV 2025][Graph Learning][Sketch-to-3D Generation] This paper proposes the PASTA framework, which integrates VLM-derived text priors to compensate for semantic deficiencies in sketches, and introduces ISG-Net (a dual graph convolutional network comprising IndivGCN and PartGCN) to model inter-part structural relationships, achieving state-of-the-art sketch-to-3D shape generation and part-level editing.
tags:
  - ICCV 2025
  - Graph Learning
  - Sketch-to-3D Generation
  - Part-Level Editing
  - Vision-Language Model
  - Graph Convolutional Network
  - Gaussian Mixture Model
date: 2026-05-08
content_hash: c6dbcead6672a94f
---

# PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior

**Conference**: ICCV 2025
**arXiv**: [2503.12834](https://arxiv.org/abs/2503.12834)
**Code**: N/A
**Area**: Graph Learning / 3D Generation
**Keywords**: Sketch-to-3D, Text Prior, Graph Convolutional Network, Part-Aware, VLM

## TL;DR
This paper proposes the PASTA framework, which leverages VLM-derived text priors to compensate for semantic deficiencies in sketches, and designs ISG-Net (IndivGCN + PartGCN) to model inter-part structural relationships, achieving state-of-the-art performance in sketch-to-3D shape generation with support for part-level editing.

## Background & Motivation

**Background**: Conditional 3D generation primarily relies on either sketches or text as input, each with notable limitations — text lacks precise geometric control, while sketches lack semantic information and suffer from inherent ambiguity.

**Limitations of Prior Work**: Pure sketch-based methods (LAS-D, SENS) struggle to recover complete structures from simplified sketches (e.g., missing armrests, incorrect leg counts); pure text-based methods cannot precisely control geometry.

**Key Challenge**: The fundamental challenge is how to simultaneously exploit the geometric controllability of sketches and the semantic expressiveness of text for accurate 3D shape generation.

**Key Insight**: A VLM is employed to automatically extract part-level descriptions from sketches (e.g., "chair back shape, seat, 4 legs, no armrests"), serving as text priors to supplement the missing semantic cues in sketches.

**Core Idea**: Text priors + a visual-text Transformer decoder fusing both conditions + ISG-Net dual-GCN modeling inter-part structural relationships.

## Method

### Overall Architecture
Input sketch → Visual backbone extracts visual embedding $\mathcal{V}$ + VLM extracts text embedding $\mathcal{T}$ → Text-Visual Transformer Decoder fuses both conditions into learnable queries $\mathbf{Q}$ → ISG-Net refines part structure → SPAGHETTI shape decoder generates 3D mesh.

### Key Designs

1. **Text-Visual Transformer Decoder**:

    - Function: Fuses visual and text conditions into $N$ learnable queries, each corresponding to a GMM component.
    - Mechanism: Queries undergo self-attention → cross-attention with visual embeddings $\mathbf{Q}_\mathcal{V}$ → self-attention → cross-attention with text embeddings $\mathbf{Q}_{\mathcal{TV}}$, iterated 12 times.
    - Design Motivation: The abstraction and simplification inherent in sketches leads to insufficient visual information; text priors supply part composition details such as leg count and presence of armrests.

2. **IndivGCN (Fine-Grained Feature Processing)**:

    - Function: Models spatial relationships among individual GMM components.
    - Mechanism: An MLP predicts the adjacency matrix $\tilde{\mathbf{A}}_I$ (supervised by pseudo ground truth derived from inter-GMM centroid distances), followed by graph convolution $\mathbf{Q}_{indiv} = \sigma(\tilde{\mathbf{A}}_I \mathbf{Q}_{\mathcal{TV}} \mathbf{W}_I)$.
    - Design Motivation: Distance relationships between GMM components reflect their spatial connectivity in 3D space.

3. **PartGCN (Part-Level Structural Aggregation)**:

    - Function: Clusters GMM components into parts and models inter-part structural relationships.
    - Mechanism: Hierarchical clustering groups $N$ GMM components into $K$ parts; average pooling yields part-level queries $\mathbf{Q}_P$; a part adjacency matrix is predicted and graph convolution is applied; results are unpooled back to the original resolution.
    - Design Motivation: Part-level relationships (e.g., "legs connect to seat") provide stronger structural integrity guarantees than individual GMM-level relationships.

Final output: $\mathbf{Q}_{final} = norm(\alpha \mathbf{Q}_{indiv} + (1-\alpha)\mathbf{Q}_{part} + \mathbf{Q}_{\mathcal{TV}})$

### Loss & Training
$\mathcal{L} = \lambda_{align}\mathcal{L}_{align} + \lambda_{indiv}\mathcal{L}_{indiv} + \lambda_{part}\mathcal{L}_{part}$, where $\mathcal{L}_{align}$ uses L1 loss to align predicted latent vectors with GT vectors inverted from SPAGHETTI, and $\mathcal{L}_{indiv}$, $\mathcal{L}_{part}$ use MSE loss to supervise adjacency matrix predictions.

## Key Experimental Results

### Main Results

| Method | AmateurSketch CD↓ | EMD↓ | FID↓ | ProSketch CD↓ | EMD↓ | FID↓ |
|--------|-------------------|------|------|----------------|------|------|
| Sketch2Mesh | 0.257 | 0.211 | 392.2 | 0.228 | 0.171 | 297.8 |
| LAS-D | 0.159 | 0.128 | 197.5 | 0.195 | 0.147 | 193.5 |
| SENS | 0.121 | 0.096 | 171.3 | 0.116 | 0.076 | 160.5 |
| DY3D | 0.109 | 0.091 | - | 0.093 | 0.087 | - |
| **PASTA** | **0.090** | **0.071** | **143.9** | **0.055** | **0.049** | **112.2** |

### Ablation Study

| Configuration | CD↓ | EMD↓ | FID↓ |
|---------------|-----|------|------|
| w/o Text Prior | 0.121 | 0.096 | 171.3 |
| w/o ISG-Net | 0.105 | 0.083 | 156.2 |
| Full PASTA | **0.090** | **0.071** | **143.9** |

### Key Findings
- PASTA comprehensively outperforms existing methods across all metrics; on ProSketch, CD decreases from 0.093 to 0.055 (a 41% improvement).
- Text priors effectively compensate for missing part information in sketches (e.g., absent armrests, incorrect leg counts).
- The dual-GCN design of ISG-Net significantly improves structural consistency.
- The method generalizes to other categories such as airplanes and lamps.

## Highlights & Insights
- **Automatic part description extraction from sketches via VLM as text prior**: This eliminates the need for manual text annotation; the VLM can identify structural details such as "4 legs" and "circular seat surface."
- **Hierarchical graph modeling with IndivGCN + PartGCN**: Analogous to pixel-to-region hierarchical reasoning, preserving fine-grained details while ensuring global structural coherence.
- **Part-level editing support**: The GMM-based part decomposition naturally enables adding, removing, and transforming individual parts.

## Limitations & Future Work
- Depends on the pretrained SPAGHETTI decoder, which constrains the supported object categories (chairs, airplanes, lamps).
- The VLM's capacity to interpret complex sketches is limited.
- Only single-view sketch input is supported.
- Future work may explore extension to broader object categories and open-vocabulary 3D generation.

## Related Work & Insights
- **vs. SENS**: SENS is a purely visual sketch-to-3D approach; incorporating text priors in this work yields significant improvements.
- **vs. DY3D**: DY3D supports user-interactive editing without text augmentation; this work employs VLM to automatically supplement semantic information.
- **vs. Text-to-3D methods**: Text-to-3D methods lack precise geometric control; this work uses sketches to provide geometric constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of VLM text priors and dual-GCN structural modeling is clear and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-category, multi-dataset quantitative evaluation with ablations and editing demonstrations.
- Writing Quality: ⭐⭐⭐⭐ Figures and tables are clear; method description is well-organized.
- Value: ⭐⭐⭐⭐ A practical approach for sketch-to-3D generation with part-level editing support.

---

# PASTA: Part-Aware Sketch-to-3D Shape Generation with Text-Aligned Prior

**Conference**: ICCV 2025
**arXiv**: [2503.12834](https://arxiv.org/abs/2503.12834)
**Code**: N/A
**Area**: 3D Vision / Graph Learning
**Keywords**: Sketch-to-3D Generation, Part-Level Editing, Vision-Language Model, Graph Convolutional Network, Gaussian Mixture Model

## TL;DR
This paper proposes the PASTA framework, which integrates VLM-derived text priors to compensate for semantic deficiencies in sketches, and introduces ISG-Net (a dual graph convolutional network comprising IndivGCN and PartGCN) to model inter-part structural relationships, achieving state-of-the-art sketch-to-3D shape generation and part-level editing.

## Background & Motivation

1. **Background**: Conditional 3D shape generation primarily employs two input modalities — sketches and text. Sketches provide geometrically precise control but lack semantic information, while text supplies semantics but lacks precise geometric control.
2. **Limitations of Prior Work**: Single-sketch inputs are overly simplified and ambiguous, leading to missing parts (e.g., absent armrests on chairs) and structural inaccuracies. Existing methods such as SENS and DY3D rely solely on visual features and cannot compensate for semantic cues absent from sketches.
3. **Key Challenge**: The core challenge is accurately inferring complete 3D part structures and semantic attributes from highly simplified 2D sketches.
4. **Key Insight**: A VLM is used to generate textual descriptions from sketches (e.g., "chair with 4 legs and armrests"), and graph-based structural reasoning is performed over a part-level GMM representation.
5. **Core Idea**: Text priors compensate for visual deficiencies; graph convolutional networks model part relationships — together enabling more accurate and complete 3D shape generation.

## Method

### Overall Architecture
Input sketch → Visual backbone extracts visual embedding $\mathcal{V}$ → VLM extracts text embedding $\mathcal{T}$ (describing part composition) → Text-Visual Transformer Decoder fuses both modalities into $N$ learnable queries → ISG-Net (IndivGCN + PartGCN) refines structural relationships → MLP maps to SPAGHETTI latent vectors → Shape decoder generates 3D mesh.

### Key Designs

1. **Text-Visual Transformer Decoder**:
    - Function: Fuses visual and text conditions into learnable queries.
    - Mechanism: $N$ learnable queries undergo self-attention → visual cross-attention with visual embeddings → text cross-attention with text embeddings, iterated 12 times. $\mathbf{Q}_{\mathcal{TV}} = Attn(W_Q^T \cdot \mathbf{Q}_\mathcal{V}, W_K^T \cdot \mathcal{T}, W_V^T \cdot \mathcal{T})$
    - Design Motivation: Text priors supply semantic information not readily observable in sketches, such as part count and presence of specific components (e.g., armrests), compensating for the limitations of visual backbones.

2. **IndivGCN (Fine-Grained Feature Processing)**:
    - Function: Models spatial relationships among individual GMM components.
    - Mechanism: An MLP predicts the adjacency matrix $\tilde{\mathbf{A}}_I$ from queries (supervised with pseudo ground truth based on GMM centroid distances), followed by graph convolution $\mathbf{Q}_{indiv} = \sigma(\tilde{\mathbf{A}}_I \mathbf{Q}_{\mathcal{TV}} \mathbf{W}_I)$.
    - Design Motivation: Enables each GMM component to aggregate information from its spatial neighbors, refining local geometric details.

3. **PartGCN (Part-Level Structural Aggregation)**:
    - Function: Clusters GMM components into parts and models inter-part structural relationships.
    - Mechanism: Hierarchical clustering groups $N$ GMM components into $K$ part clusters → average pooling produces part-level queries → part adjacency matrix is predicted → part-level graph convolution is applied → results are unpooled back to individual-level resolution.
    - Design Motivation: Coarse-grained part-level structural modeling ensures global consistency.

Final fusion: $\mathbf{Q}_{final} = norm(\alpha \mathbf{Q}_{indiv} + (1-\alpha)\mathbf{Q}_{part} + \mathbf{Q}_{\mathcal{TV}})$

### Loss & Training
$\mathcal{L} = \lambda_{align}\mathcal{L}_{align} + \lambda_{indiv}\mathcal{L}_{indiv} + \lambda_{part}\mathcal{L}_{part}$, where $\mathcal{L}_{align}$ is the L1 distance between predicted and GT latent vectors, and $\mathcal{L}_{indiv}$, $\mathcal{L}_{part}$ are MSE losses supervising adjacency matrix predictions.

## Key Experimental Results

### Main Results

| Method | AmateurSketch-3D | | ProSketch-3D | |
|--------|------|------|------|------|
| | CD↓ | EMD↓ | CD↓ | EMD↓ |
| Sketch2Mesh | 0.257 | 0.211 | 0.228 | 0.171 |
| LAS-D | 0.159 | 0.128 | 0.195 | 0.147 |
| SENS | 0.121 | 0.096 | 0.116 | 0.076 |
| DY3D | 0.109 | 0.091 | 0.093 | 0.087 |
| **PASTA** | **0.090** | **0.071** | **0.055** | **0.049** |

| Method | Airplane CD↓ | Lamp CD↓ |
|--------|-------------|----------|
| SENS | 0.240 | 0.253 |
| **PASTA** | **0.188** | **0.195** |

### Ablation Study

| Configuration | CD↓ | EMD↓ |
|---------------|-----|------|
| Visual backbone only | 0.115 | 0.092 |
| + Text Prior | 0.098 | 0.078 |
| + IndivGCN | 0.095 | 0.075 |
| + PartGCN (Full PASTA) | **0.090** | **0.071** |

### Key Findings
- On ProSketch-3D, CD decreases by 41% and EMD by 44% relative to DY3D, representing substantial gains.
- Text priors contribute the largest single improvement (CD reduced from 0.115 to 0.098), confirming that VLM semantic information is critical for compensating sketch ambiguity.
- The dual-GCN design yields consistent further improvements; PartGCN's part-level modeling contributes more than IndivGCN.
- The method generalizes to real image inputs, demonstrating robustness of the system.

## Highlights & Insights
- **VLM as a semantic enhancer for sketches is highly practical**: The VLM can identify structural details such as leg count and presence of armrests that are difficult to infer even by human observers from simplified line drawings.
- **The dual-granularity GCN design is elegant**: IndivGCN handles fine-grained details while PartGCN handles global structure; the two modules complement each other and cover geometric relationships at different scales.
- **Part-level editing is naturally supported**: The GMM-based representation inherently enables adding, deleting, and transforming individual parts.

## Limitations & Future Work
- Training and evaluation are limited to chair, airplane, and lamp categories from ShapeNet.
- The framework depends on the pretrained SPAGHETTI shape decoder, constraining its representational capacity.
- The quality of VLM-generated descriptions may be inconsistent for complex sketches.
- Future work may explore extension to more complex shapes (e.g., multi-part mechanical objects, human bodies).

## Related Work & Insights
- **vs. SENS**: SENS relies solely on visual features; the incorporation of text priors and graph-based structural reasoning in this work yields significantly superior results across all metrics.
- **vs. DY3D**: DY3D also employs part-level representations but lacks text augmentation and graph convolutional modeling of inter-part relationships.
- **vs. Text-to-3D methods**: Text conditioning alone lacks geometric control; this work combines sketch and text to leverage the advantages of both modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of text–sketch fusion and dual-GCN is novel, though not paradigm-shifting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset quantitative and qualitative evaluation with ablations, though category coverage is limited.
- Writing Quality: ⭐⭐⭐⭐ Figures and tables are clear; architectural description is detailed and well-structured.
- Value: ⭐⭐⭐⭐ Practically valuable for interactive 3D content creation with part-level editing support.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/graph_learning/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](../../NeurIPS2025/graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[NeurIPS 2025\] Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](../../NeurIPS2025/graph_learning/sketch-augmented_features_improve_learning_long-range_dependencies_in_graph_neur.md)
- [\[NeurIPS 2025\] Nonlinear Laplacians: Tunable Principal Component Analysis under Directional Prior Information](../../NeurIPS2025/graph_learning/nonlinear_laplacians_tunable_principal_component_analysis_under_directional_prio.md)
- [\[AAAI 2026\] Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data](../../AAAI2026/graph_learning/format_as_a_prior_quantifying_and_analyzing_bias_in_llms_for_heterogeneous_data.md)

</div>

<!-- RELATED:END -->
