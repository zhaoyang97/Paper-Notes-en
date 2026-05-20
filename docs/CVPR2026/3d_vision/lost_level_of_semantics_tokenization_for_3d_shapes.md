---
title: >-
  [Paper Note] LoST: Level of Semantics Tokenization for 3D Shapes
description: >-
  [CVPR 2026][3D Vision][3D generation] This paper proposes Level-of-Semantics Tokenization (LoST), which orders 3D shape tokens by semantic saliency so that short prefixes can already decode into complete and semantically…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D generation"
  - "shape tokenization"
  - "autoregressive models"
  - "semantic hierarchy"
  - "diffusion decoding"
date: 2026-05-08
content_hash: 8aeac813667e846d
---

# LoST: Level of Semantics Tokenization for 3D Shapes

**Conference**: CVPR 2026
**arXiv**: [2603.17995](https://arxiv.org/abs/2603.17995)  
**Code**: [Project Page](https://lost3d.github.io)  
**Area**: 3D Vision
**Keywords**: 3D generation, shape tokenization, autoregressive models, semantic hierarchy, diffusion decoding

## TL;DR

This paper proposes Level-of-Semantics Tokenization (LoST), which orders 3D shape tokens by semantic saliency so that short prefixes can already decode into complete and semantically coherent shapes. Combined with the RIDA semantic alignment loss and GPT-style autoregressive generation, LoST achieves significant improvements over existing 3D AR methods that require tens of thousands of tokens, using only 128 tokens.

## Background & Motivation

Tokenization is the foundation of autoregressive (AR) generative models and determines both generation quality and efficiency. The current state of 3D shape tokenization includes:

- **Flat element streams** (e.g., MeshGPT, Llama-Mesh): directly serialize vertices/faces into token streams, producing extremely long sequences (thousands to tens of thousands of tokens) with quadratic attention costs, where early prefixes cannot be decoded into meaningful shapes.
- **Geometric LoD hierarchies** (e.g., OctGPT, VertexRegen): coarse-to-fine spatial hierarchies based on octrees or progressive meshes, originating from rendering and compression, but with two systematic issues:
    - **Coarse-level token bloat**: even after geometric simplification, early stages still require a large number of spatial tokens to outline basic structure.
    - **Unusable early decoding**: aggressive geometric simplification causes coarse-level shapes to be geometrically and semantically dissimilar to the final result.

**Core Observation**: Geometric LoD is designed for rendering/compression, not for AR modeling. The ideal token sequence for AR models should satisfy: short prefix = complete + semantically coherent, long prefix = fine-grained detail. This corresponds to ordering by **semantic saliency** rather than spatial detail.

## Method

### Overall Architecture

A three-stage training system:
1. **RIDA Pre-training**: trains a 3D semantic extractor to align the triplane latent space with the DINO semantic space.
2. **LoST Autoencoder**: a ViT encoder compresses the triplane into a semantically ordered token sequence + a DiT diffusion decoder reconstructs the full triplane from any prefix.
3. **LoST-GPT**: a GPT-style Transformer performs autoregressive generation in the continuous token space.

### Key Designs

1. **LoST Encoder — Semantic Hierarchy Token Sequence**: A ViT encoder processes patchified triplanes (768 tokens) and introduces up to 512 learnable **register tokens** $\mathcal{T}_R$, which extract information from the original triplane tokens via attention. Key mechanisms:

    - **Causal masking**: causal attention among register tokens forces hierarchical organization of information from front to back.
    - **Nested dropout**: during training, only prefixes of length $[1, 2, 4, 8, ..., k]$ are randomly retained while the rest are discarded, forcing the model to front-load the most important information.
    - The type of hierarchy (semantic vs. geometric) is determined by the training loss: geometric loss → frequency hierarchy (low to high frequency); semantic loss → semantic hierarchy.
    - After encoding, only register tokens (projected to 32 dimensions) are retained; original triplane tokens are discarded to form an information bottleneck.

2. **RIDA — Relational Inter-Distance Alignment Loss**: The 2D REPA loss directly aligns with DINO features, but 3D shapes lack a direct semantic feature extractor. Instead of directly regressing DINO features (which suffers from large modality gap), RIDA aligns the **relational structure**:

    - A "student" Transformer $f_\theta$ maps triplanes into a semantic space.
    - **Global relational contrastive loss** $\mathcal{L}_{global}$: mines positive and negative pairs using DINO feature similarities and applies Multi-positive InfoNCE in the student space.
    - **Inter-instance ranking distillation** $\mathcal{L}_{rank}$: preserves continuous pairwise distance structure in DINO space (inspired by RKD).
    - **Spatial structure distillation** $\mathcal{L}_{spatial}$: aligns token-level affinity matrices.
    - Total loss: $\mathcal{L}_{RIDA} = 1.0 \cdot \mathcal{L}_{global} + 1.0 \cdot \mathcal{L}_{rank} + 0.5 \cdot \mathcal{L}_{spatial}$

3. **LoST-GPT — Continuous-Space Autoregressive Generation**: Tokens remain in continuous space (no quantization). A LlamaGen architecture (depth 24, 16 heads, dim 1024) performs autoregressive next-token prediction. A small MLP diffusion head at each position models the conditional distribution (following MAR), avoiding the information loss of VQ. Conditional generation uses OpenCLIP embeddings. High-quality generation is achieved with only 128 tokens.

### Loss & Training

- LoST autoencoder: $\mathcal{L} = \mathcal{L}_{denoise} + 1.0 \cdot \mathcal{L}_{semantic}$, where $\mathcal{L}_{semantic} = 1 - \langle f_\theta(\hat{\mathbf{X}}_0), f_\theta(\mathbf{X}_0) \rangle$
- DiT decoder: depth 24, dim 1024, 16 heads, 2×2 patchification
- Training data: a custom 300K shape dataset (prompts generated by Gemini 2.5 Pro → images generated by Flux.1 → 3D shapes generated by Direct3D)
- LoST trained for 250 epochs; RIDA trained for 100 epochs; both on 8×A100

## Key Experimental Results

### Main Results

| Method | #Tokens | CD(×10⁻²)↓ | FID↓ | DINO↑ |
|--------|---------|------------|------|-------|
| OctGPT (best layer) | ~61962 | 0.533 | 100.78 | 0.619 |
| VertexRegen (best layer) | ~7530 | 0.034 | 86.10 | 0.791 |
| **LoST (64 tokens)** | 64 | 0.382 | 21.13 | 0.880 |
| **LoST (512 tokens)** | 512 | 0.234 | 13.59 | 0.921 |
| LoST (1 token) | 1 | 2.271 | 31.65 | 0.731 |

AR Generation:

| Method | #Tokens | FID↓ | DINO↑ |
|--------|---------|------|-------|
| OctGPT | ~50000 | 66.93 | — |
| ShapeLLM-Omni | 1024 | 48.70 | 0.680 |
| **LoST-GPT** | **128** | **34.25** | **0.758** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Geometry loss only (no RIDA) | High FID, low DINO | Lack of semantic guidance; poor prefix decoding semantics |
| With RIDA semantic loss | Significant FID and DINO improvement | Semantic alignment is key to the LoS hierarchy |
| 1-token decoding | DINO 0.731, FID 31.65 | Already outperforms OctGPT's best layer (~62K tokens) |
| 4-token decoding | DINO 0.765, FID 29.26 | Further improvement in semantic fidelity |

### Key Findings

- LoST decodes a complete, recognizable shape (e.g., basic semantics of a chair or car) from just 1 token, while LoD methods still produce unusable geometric primitives at thousands of tokens.
- AR generation with 128 tokens comprehensively outperforms OctGPT requiring 50,000 tokens, representing approximately 400× improvement in token efficiency.
- Semantic hierarchy (LoS) substantially outperforms geometric hierarchy (LoD) for AR modeling, as semantic prefixes can generate complete structures.
- RIDA successfully bridges the modality gap between 2D DINO features and 3D triplane latent space by aligning relational structure rather than absolute feature values.

## Highlights & Insights

- **Deep core insight**: Geometric LoD is designed for rendering; semantic LoS is what suits AR generation — this observation reframes the thinking around 3D tokenization.
- **Elegant RIDA design**: Rather than directly regressing cross-modal features (which would fail), it aligns relational topological structure — a general approach for cross-modal knowledge transfer.
- **Extreme token efficiency**: 1 token encodes a complete shape; 128 tokens outperform methods requiring 50,000 tokens.
- **Semantically progressive prefix decoding**: from "generic category" to "specific instance detail," e.g., from "generic mountain" to "mountain with a face."
- Continuous tokens with diffusion loss avoid the information loss of VQ, serving as an effective alternative paradigm for AR generation.

## Limitations & Future Work

- The method is built on a VAE triplane latent space; future work should extend to other 3D representations (e.g., Gaussian Splats).
- The diffusion decoder increases inference cost compared to pure AR decoding.
- Artifacts may still appear with very few tokens (a problem shared with 2D semantic tokenizers).
- The AR generator currently uses a fixed target length and does not support adaptive EOS stopping.
- Training data is synthetically generated (Direct3D pipeline); real scanned data may require additional adaptation.
- Conditional control (e.g., part-level editing) has not been explored.

## Related Work & Insights

- **FlexTok/Semanticist** (2D): directly inspired LoST, extending the idea of semantic hierarchy tokenization from 2D to 3D.
- **RKD**: relational knowledge distillation inspired RIDA's relational alignment strategy.
- **MAR**: continuous-space autoregressive modeling inspired the diffusion loss design in LoST-GPT.
- **Direct3D**: the VAE provides the triplane latent space foundation.
- The LoST framework can be generalized to video tokenization (temporal semantic hierarchy) or scene-level 3D tokenization.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — paradigm shift from LoD to LoS, RIDA cross-modal relational alignment, and continuous AR generation — three independent contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — separate evaluation of tokenization reconstruction and AR generation, comparison with multiple SOTAs, and downstream semantic retrieval tasks.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — clear paper structure, compelling LoD vs. LoS comparison, high-quality figures.
- **Value**: ⭐⭐⭐⭐⭐ — dramatically improves 3D AR generation efficiency (400× token compression) and defines a new direction for 3D tokenization.
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](../../ICCV2025/3d_vision/can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)
- [\[CVPR 2026\] ExtrinSplat: Decoupling Geometry and Semantics for Open-Vocabulary Understanding in 3D Gaussian Splatting](extrinsplat_decoupling_geometry_and_semantics_for_open-vocabulary_understanding_.md)
- [\[CVPR 2026\] Lifting Unlabeled Internet-level Data for 3D Scene Understanding](lifting_unlabeled_internet-level_data_for_3d_scene_understanding.md)
- [\[ICLR 2026\] LiTo: Surface Light Field Tokenization](../../ICLR2026/3d_vision/lito_surface_light_field_tokenization.md)
- [\[CVPR 2026\] TR2M: Transferring Monocular Relative Depth to Metric Depth with Language Descriptions and Dual-Level Scale-Oriented Contrast](tr2m_transferring_monocular_relative_depth_to_metric_depth_with_language_descrip.md)

</div>

<!-- RELATED:END -->
