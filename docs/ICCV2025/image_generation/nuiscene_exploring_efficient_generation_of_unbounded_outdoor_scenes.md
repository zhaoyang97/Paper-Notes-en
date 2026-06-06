---
title: >-
  [Paper Note] NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes
description: >-
  [ICCV 2025][Image Generation][unbounded scene generation] NuiScene proposes an efficient vector set encoding scheme for scene chunks, paired with an explicitly trained outpainting diffusion model…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "unbounded scene generation"
  - "vector set representation"
  - "outpainting"
  - "diffusion model"
  - "outdoor scenes"
  - "3D generation"
date: 2026-05-08
content_hash: f3c9320e2561c056
---

# NuiScene: Exploring Efficient Generation of Unbounded Outdoor Scenes

**Conference**: ICCV 2025
**arXiv**: [2503.16375](https://arxiv.org/abs/2503.16375)  
**Area**: 3D Generation · Scene Generation
**Keywords**: unbounded scene generation, vector set representation, outpainting, diffusion model, outdoor scenes, 3D generation

## TL;DR

NuiScene proposes an efficient vector set encoding scheme for scene chunks, paired with an explicitly trained outpainting diffusion model, to enable fast unbounded outdoor scene generation. The work also curates NuiScene43, a high-quality outdoor scene dataset.

## Background & Motivation

Large-scale outdoor scene generation is critical for open-world games, film CGI, and VR simulation. Compared to indoor scene generation, outdoor scenes present unique challenges:

**Extreme height variation**: Scene chunk heights vary dramatically, from low-rise buildings to skyscrapers. Prior methods rely on spatially structured latents such as triplanes, which require fixed sizes and either lose detail by compressing tall structures or incur prohibitive memory costs.

**Slow inference**: Existing approaches depend on RePaint-style resampling inpainting for outpainting, requiring additional diffusion steps.

**Lack of high-quality outdoor data**: Semantic driving datasets have poor mesh quality, while Objaverse scenes lack a unified scale.

## Method

### 1. Data Curation

43 high-quality scenes are selected from Objaverse through the following pipeline:
- Scene filtering using DuoduoCLIP embeddings
- Relative scale annotation to establish a unified normalization
- Ground geometry cleaning and uniform ground thickness
- Scene decomposition into chunks of size $(50, h_{vox}, 50)$

### 2. Vector Set VAE

**Encoder**: For each scene chunk, $N_p$ point cloud samples $\mathbf{p} \in \mathbb{R}^{N_p \times 3}$ are drawn uniformly and aggregated into a compact representation via cross-attention:

$$\mathbf{z}^{\mathbf{p}} = \mathcal{E}(\mathbf{p}) \in \mathbb{R}^{V \times c}$$

where $V=16$ denotes the number of vectors and $c=64$ the channel dimension. Compared to the triplane representation with $V=3 \times 4^2=48$, the vector set requires only 16 tokens, yielding a higher compression ratio.

**Preventing posterior collapse**: Two point cloud samples $\mathbf{p}, \mathbf{q}$ are drawn from the same chunk, and their embeddings are constrained to be consistent: $\mathcal{L}_{emb} = (\mathbf{z}^{\mathbf{p}} - \mathbf{z}^{\mathbf{q}})^2$

**Height prediction**: A learned height embedding $\mathbf{e}_h$ queries the latent to predict the chunk height, which is used at inference time to prune unnecessary voxel queries.

**Decoder**: The vector set predicts occupancy via cross-attention: $\hat{o}_r = \text{FC}(\text{CA}(\mathbf{f}_{out}, \text{PE}(\mathbf{r})))$

**Total loss**: $\mathcal{L} = \lambda_{kl}\mathcal{L}_{kl} + \lambda_{emb}\mathcal{L}_{emb} + \lambda_{ce}\mathcal{L}_{ce} + \lambda_{height}\mathcal{L}_{height}$

### 3. Explicit Outpainting Diffusion Model

Rather than relying on RePaint-style resampling, a diffusion model is explicitly trained to generate four chunks on a $2 \times 2$ grid, conditioned on a mask and the embeddings of already-generated chunks.

Four conditioning configurations cover all cases encountered in raster-scan generation:
- $\{0,0,0,0\}$: unconditional generation
- $\{1,0,1,0\}$: left column given
- $\{1,1,0,0\}$: top row given
- $\{1,1,1,0\}$: only the bottom-right chunk to be generated

Training objective: $\mathbb{E}[\|\boldsymbol{\epsilon} - \epsilon_\theta((\mathbf{X}_t \oplus \mathbf{C}), t)\|_2^2]$

## Key Experimental Results

### VAE Reconstruction Quality

| Method | Output Res./S | IoU↑ | CD↓ | F-Score↑ |
|--------|--------------|------|-----|----------|
| triplane | 3×32²/6 | 0.734 | 0.168 | 0.508 |
| triplane | 3×64²/6 | 0.940 | 0.064 | 0.831 |
| **vecset** | **-** | **0.989** | **0.055** | **0.864** |

The vector set outperforms the triplane on all metrics, achieving an IoU of 0.989.

### Diffusion Generation Quality and Efficiency

| Method | FPD↓ | KPD↓ | # Tokens | Training Time | Memory |
|--------|------|------|----------|--------------|--------|
| triplane | 1.406 | 2.589 | 192 | 27.6h | 24.4GB |
| **vecset** | **0.571** | **0.951** | **64** | **11.1h** | **10.4GB** |

The vector set diffusion model trains 2.5× faster, consumes only 42% of the GPU memory, and reduces FPD by 59%.

### Outpainting Speed Comparison

| Method | Time for 21×21 chunks (s) |
|--------|--------------------------|
| RePaint (r=5) | 1022.20 |
| **Explicit outpainting** | **215.92** |

The explicit outpainting approach is approximately 4.7× faster and maintains coherence without resampling.

## Highlights & Insights

1. **Vector set vs. triplane**: The vector set achieves better compression with fewer tokens and naturally accommodates scene chunks of varying heights.
2. **Explicit outpainting**: Training on four conditioning configurations eliminates the resampling overhead of RePaint.
3. **Cross-scene interpolation**: After joint training on multiple scenes, the model can generate scenes that blend castles and skyscrapers, demonstrating generalization capability.

## Limitations & Future Work

- The dataset is small (43 scenes), limiting generalization.
- The absence of global context precludes large-scale planning (e.g., road network layout).
- Inter-chunk connections occasionally exhibit discontinuities or noisy artifacts.
- No support for conditional or label-guided generation.

## Related Work & Insights

- Unbounded indoor generation: BlockFusion (triplane), LT3SD (dense feature grids)
- Outdoor scenes: SemCity (semantic driving), CityDreamer, SceneDreamer
- 3D scene representations: 3DShape2VecSet, NeRF, Gaussian Splatting

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 4 |
| **Overall** | **4.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[NeurIPS 2025\] Exploring Variational Graph Autoencoders for Distribution Grid Data Generation](../../NeurIPS2025/image_generation/exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

</div>

<!-- RELATED:END -->
