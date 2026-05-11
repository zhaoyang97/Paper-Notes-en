---
title: >-
  [Paper Note] Conditional Panoramic Image Generation via Masked Autoregressive Modeling
description: >-
  [NeurIPS 2025][Image Generation][panoramic image generation] This paper proposes PAR (Panoramic AutoRegressive model), the first framework to unify text-to-panorama (T2P) and panorama outpainting (PO) under masked autore…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "panoramic image generation"
  - "masked autoregressive modeling (MAR)"
  - "equirectangular projection"
  - "circular padding"
  - "consistency alignment"
date: 2026-05-08
content_hash: 9798b3f3b9a020f7
---

# Conditional Panoramic Image Generation via Masked Autoregressive Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2505.16862](https://arxiv.org/abs/2505.16862)
**Code**: [https://wang-chaoyang.github.io/project/par](https://wang-chaoyang.github.io/project/par) (project page)
**Area**: Panoramic Image Generation / Autoregressive Modeling
**Keywords**: panoramic image generation, masked autoregressive modeling (MAR), equirectangular projection, circular padding, consistency alignment

## TL;DR
This paper proposes PAR (Panoramic AutoRegressive model), the first framework to unify text-to-panorama (T2P) and panorama outpainting (PO) under masked autoregressive modeling. PAR addresses the boundary discontinuity inherent in ERP panoramas through a circular translation consistency loss and dual-space circular padding, achieving an FID of 37.37 on Matterport3D while demonstrating strong scalability and zero-shot generalization.

## Background & Motivation
Panoramic (360°) image generation is in high demand for VR/AR, autonomous driving, and visual navigation. Existing methods suffer from two major limitations:

1. **Theoretical flaws in diffusion models**: Most existing methods are built on diffusion models, yet mapping a sphere to a 2D plane via ERP introduces non-uniform spatial distortion—pixels near the poles exhibit higher variance than equatorial pixels. This violates the i.i.d. Gaussian noise assumption central to diffusion models (Appendix A provides a rigorous mathematical proof: ERP pixel noise variance is inversely proportional to the sine of latitude).

2. **Task fragmentation**: T2P and PO are typically treated as independent tasks—the former fine-tunes Stable Diffusion while the latter uses SD-inpainting variants—resulting in separate architectures and datasets. Even Omni2, which attempts unification, requires elaborate multi-task data engineering.

Additionally, existing methods suffer from **redundant modeling**: bottom-up methods accumulate errors through iterative inpainting, while top-down methods incur unnecessary computational overhead via global-local dual branches.

## Core Problem
How to design a **theoretically sound** and **task-unified** framework for panoramic image generation? Specifically: (1) avoid the i.i.d. conflict between diffusion models and ERP; (2) handle both T2P and PO with a single architecture and objective, without task-specific data engineering.

## Method

### Overall Architecture
PAR is built on masked autoregressive modeling (MAR). The overall pipeline is as follows:
- **Input**: panoramic images are compressed into a latent representation via a VAE encoder, then patchified into a sequence of visual tokens.
- **Masked encoder**: a subset of tokens is randomly masked; unmasked tokens are fused with text embeddings (encoded by Phi-2) in the encoder.
- **Decoder**: encoder outputs are fed into the decoder and interact with masked tokens to produce a conditioning signal $z$.
- **Denoising MLP**: a lightweight MLP $\epsilon_\theta$ conditioned on $z$ denoises noise-corrupted latents to generate continuous tokens (rather than discrete tokens, reducing quantization error).
- **VAE decoder**: reconstructs the panoramic image in pixel space.

**Key unification insight**: T2P corresponds to $\mathcal{S}_k = \emptyset$ (all tokens must be generated), while PO corresponds to $\mathcal{S}_k \neq \emptyset$ (tokens from known regions serve as conditions). Both tasks are naturally unified under the MAR framework. Traditional raster-scan AR cannot handle PO (since known regions are not necessarily at the beginning of the sequence), whereas MAR supports generation in arbitrary order, elegantly resolving this issue.

### Key Designs

1. **Circular Translation Consistency Loss**: ERP panoramas are equivariant under horizontal circular translation—shifting an image by $v$ pixels horizontally preserves semantic content (only the starting longitude changes). Exploiting this property, the model forward-passes both the original input $(x, \epsilon, M)$ and the translated input $(\mathcal{T}_v(x), \mathcal{T}_v(\epsilon), \mathcal{T}_v(M))$, enforcing equivariance between their outputs: $\mathcal{L}_{consistency} = M' \circ ||\mathcal{T}_v(y) - y'||^2$. This compels the model to internalize the cyclic nature of ERP. Note: this constraint is valid only for panoramas; translating perspective images introduces discontinuous boundaries that break semantic equivalence.

2. **Dual-space Circular Padding**: During VAE encoding/decoding, edge pixels have incomplete receptive fields (only one-sided context), causing left-right boundary discontinuities. The solution applies circular padding in **two spaces**:

    - **Pre-padding** (pixel space): before VAE encoding, strips of width $rW/2$ are cropped from the left and right edges and appended to the opposite sides, providing sufficient boundary context for the encoder → ensures **semantic-level** continuity.
    - **Post-padding** (latent space): the same operation is applied to latents before VAE decoding → ensures **pixel-level** smooth transitions.

   Padding regions are discarded after transformation: $C_r(x) = \text{concat}(x[...,-rW/2:], x, x[...,:rW/2])$

3. **NOVA-based Initialization**: The model is initialized from NOVA (a vector-quantization-free autoregressive video generation model) at resolution 512×1024, requiring only 20K fine-tuning iterations.

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_{va} + \lambda \mathcal{L}_{consistency}$, where $\lambda = 0.1$

- $\mathcal{L}_{va}$: standard denoising loss, computed only over masked regions.
- $\mathcal{L}_{consistency}$: circular translation consistency loss.

Training details: batch size 32, AdamW, lr=5e-5, linear scheduling; inference uses CFG=5, 64 AR steps, and 25 denoising steps.

## Key Experimental Results

### Text-to-Panorama (Matterport3D)

| Method | Type | Params | FAED↓ | FID↓ | CS↑ | DS↓ |
|--------|------|--------|-------|------|------|------|
| PanFusion | DM | - | 5.12 | 45.21 | 30.29 | 2.67 |
| DiffPano | DM | - | 10.03 | 53.29 | 30.31 | 6.16 |
| UniPano | DM | - | 5.87 | 44.74 | **30.45** | 0.77 |
| Text2Light | AR | 0.8B | 68.90 | 70.42 | 27.90 | 7.55 |
| PanoLlama | AR | 0.8B | 33.15 | 103.51 | 32.54 | 13.99 |
| **PAR (ours)** | AR | 0.3B | 3.39 | 41.15 | 30.21 | 0.58 |
| **PAR (ours)** | AR | 0.6B | 3.34 | 39.31 | 30.34 | 0.57 |
| **PAR (ours)** | AR | 1.4B | **3.75** | **37.37** | 30.41 | **0.58** |

### Panorama Outpainting (Matterport3D)

| Method | FID↓ | FID-h↓ |
|--------|------|--------|
| AOG-Net | 83.02 | 37.88 |
| 2S-ODIS | 52.59 | 35.18 |
| **PAR w/o prompt** | 41.63 | 25.97 |
| **PAR w/ prompt** | **32.68** | **12.20** |

### Inference Speed Comparison (PAR-0.3B vs. PanFusion)

| Method | Inference Speed (sec/img) | FID |
|--------|--------------------------|-----|
| PanFusion | 28.91 | 45.21 |
| PAR-0.3B | 10.03 | 41.15 |

### Ablation Study
- **Consistency loss**: removing it raises FID from 37.37 to 39.55 (+2.18), demonstrating a clear quality improvement from the consistency loss.
- **Circular padding**: pre-padding ensures semantic-level continuity (without it, even a large post-padding ratio cannot repair semantic breaks); post-padding ensures pixel-level smoothness. DS largely converges at $r_{pre}=0.25$, $r_{post}=0.125$.
- **Scalability**: FID decreases monotonically from 0.3B → 0.6B → 1.4B (41.15 → 39.31 → 37.37), with visual quality also improving with model size and training compute.
- **CFG coefficient**: CFG=5 is optimal; FID=40.04 at CFG=3 and FID=39.76 at CFG=10.
- **Denoising steps**: 25 steps is optimal (FID=40.19); both 10 and 50 steps perform slightly worse.
- **Circular padding incurs negligible overhead**: padding primarily affects the VAE; transformer and MLP inference time is nearly unchanged (padding ratio 0–0.5, inference time 2.99–3.05 sec/img).
- **Structured3D dataset**: PAR-0.3B achieves FID=47.02, far outperforming PanoLlama's 125.35.
- **OOD generalization**: on zero-shot outpainting on SUN360, PAR achieves FID=127.01 vs. Diffusion360's 140.91; zero-shot T2P DS=0.63 outperforms StitchDiffusion's 1.12.

## Highlights & Insights
- **Theory-driven design**: rather than stacking techniques, the method starts from the fundamental conflict between ERP and the i.i.d. assumption to motivate the AR modeling choice, with rigorous mathematical proof in the appendix.
- **Elegant task unification**: T2P and PO are unified without any data engineering—switching tasks requires only controlling the known token set $\mathcal{S}_k$, and image editing is supported zero-shot.
- **Dual-space circular padding**: concise and effective; the two-space design (pixel and latent) complementarily resolves semantic and pixel-level discontinuities.
- **Circular translation consistency**: cleverly exploits the geometric prior of ERP without adding inference overhead (used only during training).
- **Inference speed advantage**: the 0.3B model is approximately 3× faster than PanFusion while achieving a lower FID.
- **Continuous token design**: continuous tokens with MLP denoising avoid the quantization error of discrete token approaches.

## Limitations & Future Work
- **Insufficient fine-grained detail**: the authors acknowledge failure cases on small objects such as chairs and tables (Fig. 15).
- **Data scarcity**: panoramic data is far less abundant than perspective data, limiting further quality gains; the authors suggest that training on larger-scale real-world panoramic data may alleviate this.
- **Limited resolution**: the current design is fixed at 512×1024; high-resolution panorama generation remains unexplored.
- **Polar region blurriness**: experiments on Structured3D show lower generation quality at polar regions (ceiling/floor), though the authors attribute this to dataset characteristics.
- **Vertical consistency**: circular padding and the consistency loss focus primarily on the horizontal direction; distortion adaptation along the vertical direction (poles to equator) has not been specifically addressed.
- **Realism gap**: texture and detail quality still fall short of real panoramic images.

## Related Work & Insights

| Dimension | PanFusion (CVPR 2024) | Omni2 (2025) | PAR (Ours) |
|-----------|----------------------|--------------|-----------|
| Base model | Stable Diffusion | Diffusion model | NOVA (MAR) |
| i.i.d. issue | Present | Present | Avoided |
| Task unification | T2P only | T2P+PO but requires data engineering | T2P+PO+editing, no data engineering |
| Architecture | Dual-branch (panorama + perspective) | Unified but complex | Single encoder-decoder |
| Inference speed | 28.91s | — | 10.03s |
| FID | 45.21 | — | 37.37 (1.4B) |

Compared to AR methods such as PanoLlama and Text2Light, PAR employs MAR rather than raster-scan order, supports generation at arbitrary positions, and achieves substantially better quality (FID 37.37 vs. 103.51/70.42).

The MAR framework's flexibility in unifying multiple tasks by controlling $\mathcal{S}_k$ is transferable to other scenarios requiring unified conditional/unconditional generation. The circular translation consistency idea generalizes to other data with equivariance priors (e.g., spherical data, periodic signals). The dual-space padding strategy is applicable to other settings where a VAE processes data with periodic or cyclic boundary conditions.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Theoretically motivated selection of MAR to resolve the i.i.d. conflict demonstrates conceptual depth, though MAR itself is not original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers T2P and PO tasks, ablations, OOD generalization, editing, and speed analysis comprehensively.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, theoretical derivations are rigorous (with appendix proofs), and method descriptions are well-structured.
- **Value**: ⭐⭐⭐⭐ Introduces a new paradigm for panoramic image generation with practical value through its unified design, though the target domain is relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [\[NeurIPS 2025\] Watermarking Autoregressive Image Generation](watermarking_autoregressive_image_generation.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](../../ICCV2025/image_generation/dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[NeurIPS 2025\] Understand Before You Generate: Self-Guided Training for Autoregressive Image Generation](understand_before_you_generate_self-guided_training_for_autoregressive_image_gen.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](../../ICCV2025/image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)

</div>

<!-- RELATED:END -->
