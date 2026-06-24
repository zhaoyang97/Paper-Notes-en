---
title: >-
  [Paper Note] FreeMesh: Boosting Mesh Generation with Coordinates Merging
description: >-
  [ICML 2025][3D Vision][Mesh Generation] This work proposes the Per-Token-Mesh-Entropy (PTME) metric to evaluate mesh tokenizer quality without training, and introduces Rearrange & Merge Coordinates (RMC), a coordinate merging technique borrowed from NLP, achieving a compression rate of up to 21.2% across three tokenizers (MeshXL, MeshAnythingV2, and EdgeRunner), while significantly increasing the number of generatable faces and preserving geometric details.
tags:
  - "ICML 2025"
  - "3D Vision"
  - "Mesh Generation"
  - "PTME"
  - "Coordinates Merging"
  - "BPE"
  - "Autoregressive Transformer"
date: 2026-05-08
content_hash: f9eadf74e3fb6cc7
---

# FreeMesh: Boosting Mesh Generation with Coordinates Merging

**Conference**: ICML 2025  
**arXiv**: [2505.13573](https://arxiv.org/abs/2505.13573)  
**Code**: None  
**Area**: 3D Vision / Mesh Generation  
**Keywords**: Mesh Generation, PTME, Coordinates Merging, BPE, Autoregressive Transformer

## TL;DR

This work proposes the Per-Token-Mesh-Entropy (PTME) metric to evaluate mesh tokenizer quality without training, and introduces Rearrange & Merge Coordinates (RMC), a coordinate merging technique borrowed from NLP, achieving a compression rate of up to 21.2% across three tokenizers (MeshXL, MeshAnythingV2, and EdgeRunner), while significantly increasing the number of generatable faces and preserving geometric details.

## Background & Motivation

**Background**: Autoregressive mesh generation has become the dominant paradigm for directly generating high-quality triangular meshes. MeshGPT pioneered generating meshes using VQ-VAE + Transformer; subsequently, MeshXL shifted to direct coordinate-level generation, while MeshAnythingV2 and EdgeRunner introduced compressed tokenization in the geometric dimension, reducing the number of coordinates per face from 9 to approximately 4-5.

**Limitations of Prior Work**: (1) The absence of training-free tokenizer evaluation metrics—one must undergo expensive full training to compare different tokenizers, which is often hindered by uncontrollable randomness; (2) coordinate sequences exhibit abundant high-frequency repeating patterns (e.g., coordinate repetition because of adjacent faces sharing vertices), which existing methods fail to exploit despite this statistical redundancy.

**Key Challenge**: The sequence length of autoregressive models limits the number of generatable faces. While sequence length is determined by the tokenizer, there lacks a theoretical tool to guide tokenizer design—forcing researchers to rely on blind trial-and-error through training.

**Goal**: (1) To establish a training-free theoretical framework for tokenizer evaluation; (2) to further compress sequence length on top of existing tokenizers.

**Key Insight**: From an information theory perspective, sequences with lower information content are easier for autoregressive models to learn. The concept of BPE in NLP, which shortens sequences by merging high-frequency subwords, can be directly transferred to coordinate sequences.

**Core Idea**: PTME evaluates tokenizer quality via the product of information entropy and compression rate; BPE-style coordinate merging serves as a plug-and-play module to further compress sequences.

## Method

### Overall Architecture

Given a 3D triangular mesh, the pipeline is as follows: choose a base tokenizer (RAW/AMT/EDR) → serialize the mesh into a 1D coordinate sequence → perform regularized rearrangement (Rearrange) → merge high-frequency coordinate patterns using BPE (Merge) → obtain the compressed token sequence → train the autoregressive Transformer generative model. During inference, tokens are decoded and mapped back to coordinates to reconstruct the mesh. The entire coordinate merging process is a plug-and-play post-processing step that can be stacked on top of any coordinate-level tokenizer.

### Key Designs

1. **Per-Token-Mesh-Entropy (PTME) Metric**

    - **Function**: A theoretical, training-free metric to evaluate the quality of mesh tokenizers.
    - **Mechanism**: For a tokenizer that encodes mesh $\mathcal{M}$ into token sequence $S$, PTME is defined as $\text{PTME} = H(S) \times CR$, where $H(S)=-\sum_i p_i \log p_i$ is the information entropy of the token sequence (statistically computing the frequency of all tokens) and $CR = |S|/|S_{raw}|$ is the compression rate relative to the raw RAW representation. A lower PTME implies that each token carries less average information and the sequence is shorter, making it easier for the autoregressive model to learn. This extends from Per-Coordinate-Mesh-Entropy (PCME) by generalizing the basic unit from coordinates to merged tokens.
    - **Design Motivation**: Different tokenizers yield sequences of varying lengths, making the direct comparison of entropy unfair. Multiplying by the compression rate standardizes the scale for comparison. The core intuition is that sequences with lower overall information are easier to fit for sequence models.

2. **Coordinate Rearrangement (Rearrange)**

    - **Function**: Regularizes the arrangement of coordinates to make high-frequency patterns more concentrated, preparing them for merging.
    - **Mechanism**: The serialized coordinates are reordered through a regularized rearrangement so that repeating patterns appear consecutively in the sequence. It is observed that different tokenizers output coordinates in different orders (e.g., RAW orders coordinates as $x_1y_1z_1x_2y_2z_2x_3y_3z_3$). Rearrange adjusts the sequence order (e.g., positioning shared coordinates of adjacent faces adjacently), enabling BPE to discover and merge high-frequency pairs more effectively.
    - **Design Motivation**: Applying BPE merging directly to a sequence without rearrangement (the MC method) **fails to lower PTME**, because high-frequency patterns are scattered. Rearrangement is a **necessary prerequisite** for effective merging, which is a key empirical finding of this work.

3. **BPE Coordinate Merging (Merge)**

    - **Function**: Merges high-frequency coordinate patterns into new tokens to shorten the sequence.
    - **Mechanism**: Using BPE training implemented by SentencePiece, the most frequent adjacent token pairs from the training set coordinate sequences are statistically computed and iteratively merged into new tokens until the target vocabulary size is met. For example, if the coordinate pair $(x_i, y_i)$ occurs frequently, it is merged into a single token $[x_i, y_i]$. Increasing vocabulary size allows for compressing more coordinate pairs, continuously reducing PTME.
    - **Design Motivation**: BPE in NLP balances the word level and character level via subword merging. Coordinate sequences of 3D meshes exhibit statistical redundancy similar to natural language, making BPE directly transferable.

### Loss & Training

The Transformer is trained using a standard autoregressive cross-entropy loss. RMC is a pure data preprocessing step that does not alter the training process. Training and testing are performed under a unified 7-bit discretization setting (where coordinates are quantized to 128 levels).

## Key Experimental Results

### Compression Rate Comparison (7-bit Discretization)

| Tokenizer | Original Compression Rate | +MC | +RMC | PTME Change |
|-----------|-----------|-----|------|---------|
| RAW (MeshXL) | 100% | ~100% (No improvement) | Significantly Reduced | Visibly Reduced |
| AMT (MeshAnythingV2) | ~50% | ~50% | Further Reduced | Reduced |
| EDR (EdgeRunner) | ~45% | ~45% | **21.2%** | **Lowest** |

### Generation Quality (Objaverse / Objaverse-XL, Point Cloud Conditioned)

| Configuration | Max Faces | Geometric Details | Topological Quality |
|------|----------|---------|---------|
| EDR Baseline | ~800 | Baseline | Baseline |
| EDR + MC | ≈800 | ≈Baseline | ≈Baseline |
| **EDR + RMC** | **~1600** | **Significantly Improved** | **Better** |

### Ablation Study

| Configuration | PTME Change | Description |
|------|---------|------|
| MC only (Without Rearrange) | Unchanged or slightly increased | Rearrangement is a necessary condition for effective merging |
| Rearrange only (Without Merge) | Slightly reduced | Rearrangement alone is insufficient; must combined with merging |
| RMC (Complete) | Significantly reduced | Both steps must work synergistically to be effective |
| Vocabulary 256→1024→4096 | Continually reduced | More merges → shorter sequence → more faces |

### Key Findings

- PTME is highly positively correlated with actual post-training generation quality (low PTME → high quality), validating the effectiveness of the metric.
- MC (merging without rearrangement) fails on all tokenizers—PTME increases instead of decreasing.
- RMC can double the maximum number of generatable faces for EdgeRunner (~800 → ~1600), achieving this without increasing model parameters or computational budget.

## Highlights & Insights

- **PTME serves as the first training-free theoretical evaluation metric for mesh tokenizers.** Information entropy $\times$ compression rate elegantly unifies the two dimensions of sequence length and learning difficulty. This provides theoretical guidance for tokenizer design, avoiding the blind trial-and-error of "training to know the performance."
- **Cross-domain method transfer from NLP to 3D**: The successful application of BPE subword tokenization on coordinate sequences demonstrates the universality of NLP techniques for 3D generation under a serialized representation. "Rearrange" as a necessary preprocessing step to make BPE work is an important insight.

## Limitations & Future Work

- BPE merging is a purely statistical operation and does not consider geometric semantics; merged coordinate patterns may span across different geometric structures.
- Increasing vocabulary size leads to a linear growth in embedding parameters, presenting a trade-off between compression rate and model scale.
- PTME assumes independent and identically distributed (i.i.d.) tokens to compute entropy, ignoring conditional dependencies in the sequence.
- The method has only been validated under 7-bit discretization; the impact under higher precision remains unexplored.
- Inference requires backward mapping of merged tokens, increasing decoding complexity.

## Related Work & Insights

- **vs MeshGPT (Siddiqui et al., 2023)**: MeshGPT uses VQ-VAE for compression to a latent space, whereas FreeMesh operates at the coordinate level; both are orthogonal.
- **vs EdgeRunner (Tang et al., 2024a)**: EdgeRunner performs compression in the geometric dimension (reducing coordinates per face), while RMC performs compression in the statistical dimension (merging high-frequency patterns). They are complementary, and their combination yields a compression rate of 21.2%.
- **vs NLP BPE (Sennrich et al., 2016)**: Directly transferring the concept of BPE from natural language to 3D meshes, where the isomorphic nature of the problem structure makes the transfer natural.

## Rating

- Novelty: ⭐⭐⭐⭐ The PTME metric and the migration of BPE to 3D meshes are both pioneering, and the necessity of Rearrange is a non-obvious insight.
- Experimental Thoroughness: ⭐⭐⭐ Consistency is validated across three tokenizers, but downstream application evaluation and more quantitative metrics are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear concepts, intuitive diagrams, and concise mathematical derivation of PTME.
- Value: ⭐⭐⭐⭐ A plug-and-play compression module accompanied by a training-free evaluation metric offers direct practical value to the mesh generation community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Scaling Mesh Generation via Compressive Tokenization](../../CVPR2025/3d_vision/scaling_mesh_generation_via_compressive_tokenization.md)
- [\[CVPR 2025\] TreeMeshGPT: Artistic Mesh Generation with Autoregressive Tree Sequencing](../../CVPR2025/3d_vision/treemeshgpt_artistic_mesh_generation_with_autoregressive_tree_sequencing.md)
- [\[ICCV 2025\] MeshAnything V2: Artist-Created Mesh Generation with Adjacent Mesh Tokenization](../../ICCV2025/3d_vision/meshanything_v2_artist-created_mesh_generation_with_adjacent_mesh_tokenization.md)
- [\[NeurIPS 2025\] Mesh-RFT: Enhancing Mesh Generation via Fine-Grained Reinforcement Fine-Tuning](../../NeurIPS2025/3d_vision/mesh-rft_enhancing_mesh_generation_via_fine-grained_reinforcement_fine-tuning.md)
- [\[ICCV 2025\] VertexRegen: Mesh Generation with Continuous Level of Detail](../../ICCV2025/3d_vision/vertexregen_mesh_generation_with_continuous_level_of_detail.md)

</div>

<!-- RELATED:END -->
