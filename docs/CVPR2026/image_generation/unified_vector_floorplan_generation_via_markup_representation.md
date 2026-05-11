---
title: >-
  [Paper Note] Unified Vector Floorplan Generation via Markup Representation
description: >-
  [CVPR 2026][Image Generation][floorplan generation] This paper proposes the Floorplan Markup Language (FML), which encodes floorplan elements such as rooms and doors into structured token sequences. A LLaMA-style Transfo…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "floorplan generation"
  - "markup language"
  - "autoregressive sequence model"
  - "constrained decoding"
  - "vector representation"
date: 2026-05-08
content_hash: 9811941312f28da4
---

# Unified Vector Floorplan Generation via Markup Representation

**Conference**: CVPR 2026
**arXiv**: [2604.04859](https://arxiv.org/abs/2604.04859)
**Code**: [https://mapooon.github.io/FMLPage](https://mapooon.github.io/FMLPage)
**Area**: Image Generation
**Keywords**: floorplan generation, markup language, autoregressive sequence model, constrained decoding, vector representation

## TL;DR

This paper proposes the Floorplan Markup Language (FML), which encodes floorplan elements such as rooms and doors into structured token sequences. A LLaMA-style Transformer model (FMLM) trained on this representation unifies unconditional, boundary-conditioned, graph-conditioned, and completion tasks within a single framework, achieving over 80% lower FID than HouseDiffusion.

## Background & Motivation

1. **Background**: Automated floorplan generation is a core requirement in architectural design and real estate. Existing methods are task-specific — Graph2Plan handles boundary conditions, while HouseGAN++/HouseDiffusion address adjacency graph conditions — each requiring a dedicated model.
2. **Limitations of Prior Work**: (1) Different generation tasks rely on different architectures, precluding unification; (2) diffusion-based methods (e.g., GSDiff) produce raster images, and post-processing conversion to vector format introduces errors; (3) GAN-based methods suffer from mode collapse and limited generation diversity.
3. **Key Challenge**: Floorplans are inherently structured vector data (room polygons + door positions + connectivity), yet existing methods either operate in pixel space (losing structural information) or require task-specific graph neural networks.
4. **Goal**: Design a unified representation that reformulates all floorplan generation tasks as a single sequence prediction problem.
5. **Key Insight**: Inspired by markup languages (HTML/XML) in NLP — token sequences defined by syntactic rules naturally represent structured information and are directly amenable to autoregressive Transformer modeling.
6. **Core Idea**: Define FML grammar to encode floorplans as token sequences of "tag + coordinate + index + type", and apply constrained decoding to guarantee syntactic validity of generated outputs.

## Method

### Overall Architecture

Optional input conditions (boundary point sequence / adjacency graph / partial floorplan) → encoded as FML condition segment → FMLM autoregressively generates FML sequence → constrained decoding enforces syntactic validity → FML parsed into vector floorplan (room polygons and door positions).

### Key Designs

1. **Floorplan Markup Language (FML)**

   - **Function**: Encodes all floorplan elements into a linear token sequence.
   - **Mechanism**: Defines four token types — tags (e.g., `<room>`, `<door>`), coordinates (1D encoding $z = x + y \times W$, $W=256$), room indices, and room types. The grammar follows `<sequence> → <condition> → <floorplan> → rooms → doors → front_door → </sequence>`. Rooms are ordered by descending index.
   - **Design Motivation**: 1D coordinate encoding avoids the high-dimensional sparsity of 2D positional representations. Descending ordering is validated by ablation, reducing FID from 94.57 to 25.50. Tag tokens provide structural supervision signals.

2. **FMLM Model Architecture**

   - **Function**: Autoregressively generates FML token sequences.
   - **Mechanism**: A LLaMA-3-style Transformer with 24 layers, 512-dimensional hidden states, and 32 attention heads. Coordinate tokens use sinusoidal positional encoding with a learnable projection; tag/index/type tokens use learnable embeddings. A unified output head $W \in \mathbb{R}^{(C_{tag}+C_{coord}+C_{index}+C_{type}) \times C}$ is shared across all token types.
   - **Design Motivation**: The unified output head allows the model to learn when to generate each token type automatically, eliminating the need for manually switching decoding modes.

3. **Constrained Decoding**

   - **Function**: Guarantees syntactic validity of generated FML sequences at inference time.
   - **Mechanism**: Hard constraints include: doors must have exactly 2 vertices; room polygons must not overlap with existing rooms; doors must lie on room boundaries. These rules are enforced by masking invalid token probabilities during decoding.
   - **Design Motivation**: Autoregressive models may generate syntactically invalid sequences (e.g., a door with 3 vertices). Constrained decoding guarantees 100% valid outputs at zero additional computational cost.

### Loss & Training

Standard cross-entropy loss is computed over non-structural tag tokens in the FML sequence. Room permutation augmentation (random shuffling of room order) is applied during training to encourage the model to learn permutation equivariance — ablation shows this reduces FID from 24.36 to 14.17.

## Key Experimental Results

### Main Results

| Task | Method | FID↓ | GED↓ | IoU↑ |
|------|--------|------|------|------|
| Unconditional | GSDiff | 15.02 | - | - |
| Unconditional | **FMLM** | **7.22** | - | - |
| Boundary-conditioned | Graph2Plan | 34.20 | - | 95.87% |
| Boundary-conditioned | **FMLM** | **6.51** | - | **97.86%** |
| Graph-conditioned (ALL) | HouseGAN++ | 48.44 | 2.57 | - |
| Graph-conditioned (ALL) | HouseDiffusion | 29.31 | 1.55 | - |
| Graph-conditioned (ALL) | **FMLM** | **3.41** | **1.21** | - |
| Boundary+Graph (ALL) | Graph2Plan | 22.87 | 3.43 | 92.96% |
| Boundary+Graph (ALL) | **FMLM** | **14.17** | **1.24** | **97.59%** |

### Ablation Study

| Configuration | FID↓ | GED↓ | IoU↑ | Note |
|---------------|------|------|------|------|
| Full + permutation aug. | 14.17 | 1.24 | 97.59% | Full model |
| w/o permutation aug. | 24.36 | 2.35 | 95.82% | FID +72% |
| Ascending index order | 94.57 | - | - | Very poor FID |
| Descending index order | 25.50 | - | - | Descending far superior |

### Key Findings

- Room permutation augmentation is critical — removing it increases FID from 14.17 to 24.36 (+72%), indicating that learning permutation equivariance is essential for generalization.
- FMLM substantially outperforms GAN-based and diffusion-based methods across all conditioning settings.
- Constrained decoding guarantees 100% syntactically valid outputs, whereas post-processing pipelines in methods such as HouseDiffusion cannot provide this guarantee.
- Performance degrades slightly for 8-room layouts (FID increases from 3.41 to 4.64) due to limited training samples.

## Highlights & Insights

- **Elegance of the markup representation**: Reformulating structured generation as sequence prediction via grammar rules is a clean and transferable paradigm, applicable to other structured generation tasks such as circuit layout or molecular structure generation.
- **Zero-overhead hard constraints**: Masking invalid tokens at inference time enforces hard syntactic constraints without additional computation, which is more reliable than post-hoc correction.
- **Multi-task unification**: A single model handles unconditional, boundary-conditioned, graph-conditioned, and completion tasks simultaneously, eliminating the redundancy of maintaining task-specific architectures.

## Limitations & Future Work

- Only single-story floorplans are supported; multi-story buildings would require extending the FML grammar.
- Performance degrades for layouts with more than 8 rooms due to insufficient training data.
- Coordinate quantization to a 256×256 grid may sacrifice precision; higher resolutions would increase vocabulary size.
- Integrating with LLMs (natural language specification → floorplan generation) is a promising future direction.

## Related Work & Insights

- **vs. HouseDiffusion**: Diffusion methods model continuous spaces and require vectorization post-processing, whereas FMLM directly generates vector results in discrete token space with greater precision.
- **vs. Graph2Plan**: Requires a GNN encoder for adjacency graph conditioning, resulting in architectural complexity. FMLM serializes adjacency relationships directly into the FML condition segment, requiring no additional encoder.
- **vs. GSDiff**: The raster-based diffusion approach achieves FID 15.02, compared to FMLM's 7.22; the gap primarily stems from the structural prior embedded in the vector representation.

## Rating

- Novelty: ⭐⭐⭐⭐ The markup language representation is a novel perspective, though autoregressive generation itself is not new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across four conditioning settings, with ablations and multi-room-count analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear and fluent, with a rigorous definition of FML grammar.
- Value: ⭐⭐⭐⭐ Directly applicable to architectural design, with a transferable markup-based generation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ExpPortrait: Expressive Portrait Generation via Personalized Representation](expportrait_expressive_portrait_generation_via_personalized_representation.md)
- [\[ICLR 2026\] Purrception: Variational Flow Matching for Vector-Quantized Image Generation](../../ICLR2026/image_generation/purrception_variational_flow_matching_for_vector-quantized_image_generation.md)
- [\[CVPR 2026\] BiGain: Unified Token Compression for Joint Generation and Classification](bigain_unified_token_compression_for_joint_generation_and_classification.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] CoLoGen: Progressive Learning of Concept-Localization Duality for Unified Image Generation](cologen_progressive_learning_of_concept-localization_duality_for_unified_image_g.md)

</div>

<!-- RELATED:END -->
