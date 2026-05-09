---
title: >-
  [Paper Note] AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer
description: >-
  [ICLR 2026][LLM/NLP][3D generation] This paper proposes AssetFormer, an autoregressive Transformer based on the Llama architecture that models modular 3D assets (composed of primitive sequences) as discrete token sequences. Through DFS/BFS graph traversal reordering and joint vocabulary decoding, it enables the generation of modular 3D assets directly usable in game engines from text descriptions.
tags:
  - ICLR 2026
  - LLM/NLP
  - 3D generation
  - autoregressive transformer
  - modular assets
  - UGC
  - Llama
  - text-to-3D
date: 2026-05-08
content_hash: 6ea6531454600baa
---

# AssetFormer: Modular 3D Assets Generation with Autoregressive Transformer

**Conference**: ICLR 2026
**arXiv**: [2602.12100](https://arxiv.org/abs/2602.12100)
**Code**: [https://github.com/Advocate99/AssetFormer](https://github.com/Advocate99/AssetFormer)
**Area**: LLM/NLP
**Keywords**: 3D generation, autoregressive transformer, modular assets, UGC, Llama, text-to-3D

## TL;DR

This paper proposes AssetFormer, an autoregressive Transformer based on the Llama architecture that models modular 3D assets (composed of primitive sequences) as discrete token sequences. Through DFS/BFS graph traversal reordering and joint vocabulary decoding, it enables the generation of modular 3D assets directly usable in game engines from text descriptions.

## Background & Motivation

3D asset generation is a core requirement in UGC (user-generated content) and game development. While mainstream 3D generation methods (e.g., NeRF, 3D Gaussian Splatting, or direct mesh generation) have achieved significant progress in visual quality, several critical issues remain:

1. **Incompatibility with existing workflows**: Generated 3D content is typically a monolithic mesh or implicit representation that is difficult to import into game engines for editing, decomposition, or recombination, requiring extensive post-processing.
2. **Lack of modular structure**: Real-world game development relies on standardized modular components (akin to building blocks) assembled into scenes and objects—a structure that existing methods struggle to produce.
3. **Difficulty modeling discrete attributes**: Each primitive in a modular asset carries mixed attributes such as class, rotation, and position, which are discrete and structurally constrained—properties that continuous generation methods handle poorly.

The motivating insight of AssetFormer is: **since modular 3D assets are fundamentally sequences of primitives with discrete attributes, why not model them directly with the mature autoregressive Transformer paradigm from the NLP domain?**

## Core Problem

How to reformulate modular 3D asset generation as a sequence-to-sequence modeling problem, enabling an autoregressive Transformer to effectively generate valid 3D assets composed of discrete primitives from text descriptions.

## Method

### Modular 3D Asset Representation

A modular 3D asset consists of a set of **primitives**, each with three attributes:
- **Class**: selected from a predefined primitive library (e.g., building blocks of various shapes), with $K$ types in total
- **Rotation**: discretized into a finite set of rotation angles, with the valid count varying per primitive based on its symmetry
- **Position**: coordinates $(x, y, z)$ on a 3D grid, discretized as integers

A complete 3D asset is represented as:
$$
A = \{(c_i, r_i, p_i)\}_{i=1}^{N}
$$
where $N$ is the number of primitives, and $c_i, r_i, p_i$ denote the class, rotation, and position of the $i$-th primitive, respectively.

### Lossless Discrete Tokenization

Unlike many methods that require training a VQ-VAE codebook, AssetFormer adopts **lossless tokenization**:

- Each primitive's attributes are directly mapped to discrete tokens without an additional encoder-decoder
- Class, rotation, and position each occupy a fixed number of tokens
- This guarantees lossless tokenization—decoded tokens can unambiguously reconstruct the exact 3D asset

This design avoids the information loss and training instability associated with codebook learning.

### Token Reordering Strategy

When flattening a 3D asset into a 1D token sequence, the ordering of primitives is critical for autoregressive modeling. The authors observe that primitives share spatial adjacency relations, which can be captured by an **adjacency graph**:

- **Nodes**: individual primitives
- **Edges**: spatially adjacent primitive pairs (sharing a face or edge)

Two traversal strategies are compared on this graph:
- **BFS (Breadth-First Search)**: expands level by level, producing sequences where adjacent tokens tend to occupy the same spatial layer
- **DFS (Depth-First Search)**: follows a single path deeply, producing sequences where adjacent tokens tend to belong to the same branch

Experiments show **DFS slightly outperforms BFS**, as DFS yields sequences with more localized spatial relationships between adjacent tokens, facilitating the autoregressive model's capture of short-range dependencies. DFS sequences also exhibit smoother variation in attribute values (especially coordinates), which benefits normalization during training.

### Token Set Modeling

A key challenge is that each primitive's three attributes (class, rotation, position) have distinct valid value ranges. Naively placing all possible tokens into a single vocabulary leads to a large number of illegal combinations.

AssetFormer proposes **Token Set Modeling**:
- A **joint vocabulary** is constructed containing all possible values across all attribute types
- During decoding, **filtered decoding** is applied: tokens not belonging to the current attribute type are dynamically masked, ensuring only legal values are generated at each step
- This constitutes **structured constrained decoding**, enforcing output validity without modifying the model architecture

### Model Architecture

AssetFormer is built on the **Llama architecture**:
- A standard decoder-only Transformer
- Text prompts are encoded by a pretrained text encoder and provided as prefix conditioning
- The 3D asset token sequence is generated autoregressively as the target
- RoPE positional encoding is adopted
- The model operates at a moderate scale, demonstrating effectiveness without requiring extreme parameter counts

### Classifier-Free Guidance (CFG)

Drawing on the success of Classifier-Free Guidance in diffusion models, AssetFormer adapts CFG to the autoregressive framework:

- During training, text conditions are randomly dropped (replaced with empty prompts) with a fixed probability
- At inference, both conditional and unconditional logits are computed, and text guidance is amplified via linear extrapolation:
$$
\text{logits}_{\text{guided}} = \text{logits}_{\text{uncond}} + \lambda \cdot (\text{logits}_{\text{cond}} - \text{logits}_{\text{uncond}})
$$
where $\lambda > 1$ is the guidance scale. This effectively improves alignment between generated assets and text descriptions.

### Dataset Construction

Dataset construction is a notable contribution of this work:
- **Real user data**: Modular 3D assets created by real users are collected from an online UGC platform (similar to Roblox)
- **PCG synthetic data**: Procedural Content Generation is used to augment the dataset
- **Text annotation**: 3D assets are rendered into 2D images from multiple viewpoints, and **GPT-4o** is used to generate corresponding text descriptions
- This semi-automatic data construction pipeline effectively addresses the scarcity of text-annotated modular 3D assets

## Key Experimental Results

### Generation Quality

- AssetFormer significantly outperforms simple baselines in visual quality and text-3D alignment
- Compared to end-to-end 3D mesh generation methods, AssetFormer's key advantage is that generated assets are **natively editable and directly usable in game engines**
- Quantitative metrics include FID and CLIP Score

### DFS vs. BFS

- DFS ordering outperforms BFS across multiple metrics
- The authors attribute this to smoother coordinate transitions in DFS sequences, which help the model learn local spatial patterns
- This finding has reference value for sequence modeling of spatially structured data such as modular assets

### CFG Effectiveness

- Incorporating CFG substantially improves text-to-3D alignment
- The guidance scale $\lambda$ has an optimal range; excessively large values reduce diversity

### Comparison with Other 3D Generation Methods

- Comparisons are conducted against diffusion-based and other autoregressive 3D generators
- AssetFormer demonstrates a clear structural advantage on the modular asset generation task
- Competing methods cannot directly produce editable modular structures

## Highlights & Insights

1. **Elegant problem formulation**: Reformulating modular 3D asset generation as discrete token sequence modeling fully leverages the maturity of the large language model technology stack.
2. **Lossless tokenization**: Avoids the information compression loss of VQ-VAE-based methods, with a one-to-one correspondence between tokens and raw attributes.
3. **Graph traversal reordering**: Constructing an adjacency graph from spatial topology and using DFS/BFS traversal to identify sequences more amenable to autoregressive modeling is a concise and effective design.
4. **Token Set Modeling + Filtered Decoding**: The combination of a joint vocabulary and dynamic masked decoding elegantly enforces validity constraints across multiple attribute types.
5. **Practical orientation**: Generated assets can be directly imported into game engines, offering genuine application value for UGC platforms and game development.
6. **Reproducible data pipeline**: The GPT-4o annotation combined with PCG augmentation approach has broad transferability.

## Limitations & Future Work

1. **Fixed primitive library**: The method relies on a predefined set of primitives and cannot generate entirely new types of basic components, limiting generative diversity.
2. **Texture and material**: The paper focuses primarily on geometric structure generation, leaving texture, material, and lighting attributes unaddressed.
3. **Scalability**: Since token sequence length grows proportionally with the number of primitives, very large-scale 3D scenes may produce prohibitively long sequences.
4. **Limited evaluation metrics**: Standards for evaluating modular 3D assets remain immature; the quantitative metrics reported may not fully reflect real-world usability.
5. **Single data source**: Training data predominantly originates from a specific UGC platform, potentially introducing stylistic bias.
6. **Small DFS vs. BFS gap**: Although DFS is slightly superior, the margin is narrow, suggesting that sequence ordering may not be the primary performance bottleneck.

## Related Work & Insights

- **Another instance of "everything is a sequence"**: Following code generation and molecular generation, modular 3D assets have been successfully serialized and modeled with autoregressive models, demonstrating the broad applicability of the autoregressive Transformer paradigm to structured discrete data.
- **Integrating domain knowledge**: Graph traversal reordering and filtered decoding incorporate structural priors of 3D assets (spatial adjacency, attribute type constraints) elegantly into a general-purpose Transformer framework, rather than requiring specialized architectures.
- **Connection to CAD generation**: The methodology shares conceptual similarities with CAD model generation approaches (e.g., DeepCAD), both employing sequence models to capture sequences of discrete geometric operations; the key difference is that primitives in this work are more standardized.
- **Generality of Classifier-Free Guidance**: The successful transfer of CFG from image diffusion models to 3D autoregressive generation reinforces its paradigm-level universality in conditional generation tasks.
- **Real industry demand**: The collaboration with LIGHTSPEED Studios grounds this work in engineering applicability and reflects genuine industry demand for AI-assisted content creation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ICLR 2026\] AP-OOD: Attention Pooling for Out-of-Distribution Detection](ap-ood_attention_pooling_for_out-of-distribution_detection.md)
- [\[ICLR 2026\] LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery](llema_evolutionary_search_with_llms_for_multi-objective_material_design.md)
- [\[ICLR 2026\] Enhancing Persona Following at Decoding Time via Dynamic Importance Estimation for Role-Playing Agents](enhancing_persona_following_at_decoding_time_via_dynamic_importance_estimation_f.md)

</div>

<!-- RELATED:END -->
