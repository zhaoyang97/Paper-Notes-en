---
title: >-
  [Paper Note] Auto-Regressively Generating Multi-View Consistent Images
description: >-
  [ICCV 2025][3D Vision][Multi-view generation] This paper proposes MV-AR, the first autoregressive model for multi-view image generation. It progressively generates each subsequent view conditioned on all previously generated views, incorporating a unified multimodal condition injection module and a Shuffle View data augmentation strategy. MV-AR achieves consistency comparable to diffusion-based methods under text, image, and shape conditioning.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Multi-view generation"
  - "autoregressive models"
  - "3D content creation"
  - "multimodal conditioning"
  - "view consistency"
date: 2026-05-08
content_hash: d444280ba42561fb
---

# Auto-Regressively Generating Multi-View Consistent Images

**Conference**: ICCV 2025
**arXiv**: [2506.18527](https://arxiv.org/abs/2506.18527)  
**Code**: Available (released as mentioned in the paper)  
**Area**: 3D Vision
**Keywords**: Multi-view generation, autoregressive models, 3D content creation, multimodal conditioning, view consistency

## TL;DR
This paper proposes MV-AR, the first autoregressive model for multi-view image generation. It progressively generates each subsequent view conditioned on all previously generated views, incorporating a unified multimodal condition injection module and a Shuffle View data augmentation strategy. MV-AR achieves consistency comparable to diffusion-based methods under text, image, and shape conditioning.

## Background & Motivation
Generating multi-view consistent images from text, reference images, or geometric shapes is fundamental to 3D content creation. Current mainstream approaches are diffusion-based (e.g., MVDream, Zero123++, SyncDreamer), achieving cross-view information exchange via cross-attention or multi-view concatenation.

**Limitations of Prior Work**:

**Poor consistency for distant views**: Diffusion-based methods use a single reference view (e.g., frontal image) to generate all views. When generating back-facing views, the overlap with the reference is nearly zero, causing the guidance signal to degrade and resulting in texture inconsistencies.

**Inflexible condition switching**: Diffusion-based methods typically require substantial architectural modifications when switching condition types (text → image → shape).

**Key Challenge**: Diffusion methods model the joint distribution $p(v_1, v_2, ..., v_n)$, simultaneously generating all views in a manner inconsistent with the progressive way humans observe 3D objects.

**Core Idea**: Autoregressive models are naturally suited for progressive generation — when generating the $n$-th view, all information from the preceding $n-1$ views is available. Adjacent views always share significant overlap, and even back-facing views can receive effective reference from side views.

## Method

### Overall Architecture
MV-AR is built upon a pretrained text-to-image AR model (LLamaGen). Multi-view token sequences encoded by a 2D VQVAE are concatenated and generated autoregressively token by token. The conditioning module supports text (prefill tokens + SSA), camera pose (Shift Position Embedding), reference image (Image Warp Controller), and geometry (pretrained shape encoder tokens). All modalities are unified through a progressive training strategy.

### Key Designs

1. **Split Self-Attention (SSA)**: Text tokens and image tokens are concatenated and processed through self-attention within the Transformer, but the residual contribution of text positions in the self-attention output is zeroed out: $SSA(X_{in}) = X_{in} + Concat(0 \cdot O_{text}, O_{image})$.

    - **Design Motivation**: In standard self-attention, subsequent image tokens back-propagate interference into the text token representations, degrading text guidance. SSA allows text tokens to influence image generation without being inversely affected by image tokens, improving text-image alignment (higher CLIP Score).

2. **Shift Position Embedding (SPE)**: Plücker ray encodings $r_{i,j} = (o \times d, d)$ are used as shift position embeddings and added directly to token embeddings.

    - **Design Motivation**: Tokens from different views need awareness of their corresponding 3D positions and view directions. The physically grounded angular information provided by SPE enables the model to distinguish the same spatial location across different views, improving spatial understanding.

3. **Image Warp Controller (IWC)**: Texture features in overlapping regions are predicted using the current view's camera pose and reference image features: $X_{IWC} = FFN(CA(SA(X_{ref}), r))$, injected into the model as a residual.

    - **Design Motivation**: High-level image features (e.g., CLIP) discard low-level detail. IWC retains color and texture details using encoder-level features and injects them per token for precise control. Experiments show IWC substantially outperforms in-context and cross-attention injection methods.

4. **Shuffle View Data Augmentation**: During training, the order of $N$ views is randomly shuffled, expanding each 3D object into $\frac{N(N-1)}{2}$ distinct training sequences.

    - **Design Motivation**: AR models require large amounts of data to prevent overfitting, yet high-quality multi-view data is scarce. ShufV leverages the ordering flexibility of multi-view sequences to substantially increase data volume, while training the model to learn view transitions in both directions.

### Loss & Training
- Standard AR loss: $\mathcal{L}_{ar} = -\frac{1}{T}\sum_{t=1}^T \log p(q_t | q_{<t})$
- **Progressive learning**: A text-to-multi-view (t2mv) model is trained first, followed by training an any-condition-to-multi-view (X2mv) model initialized from t2mv.
- Text conditions are randomly dropped and replaced with instruction templates (e.g., "Generate multi-view images of the following <<<img>>>").
- Drop probability increases linearly from 0 to 0.5 over the first 10K iterations, then remains at 0.5.
- Optimizer: AdamW; 16× A800 GPUs; batch size 1024; learning rate $4 \times 10^{-4}$; trained for 30K iterations.

## Key Experimental Results

### Main Results

**Text → Multi-view (GSO 30 objects)**

| Method | FID↓ | IS↑ | CLIP-Score↑ |
|--------|------|-----|-------------|
| MVDream (diffusion) | 141.05 | 7.49 | 28.71 |
| LLamaGen (baseline AR) | 146.11 | 5.78 | 28.36 |
| **MV-AR** | **144.29** | **8.00** | **29.49** |

**Image → Multi-view (GSO 30 objects)**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| Zero123 | 18.93 | 0.779 | 0.166 |
| SyncDreamer | 19.89 | 0.801 | 0.129 |
| Wonder3D | 22.82 | 0.892 | 0.062 |
| **MV-AR** | **22.99** | 0.907 | 0.084 |

MV-AR achieves the best PSNR (22.99) and the highest CLIP-Score (29.49), demonstrating that the AR approach is competitive with, and in some metrics superior to, diffusion-based methods in terms of multi-view consistency and text alignment.

### Ablation Study

| Component | FID↓ / IS↑ (t2mv) | PSNR↑ / SSIM↑ / LPIPS↓ (i2mv) |
|-----------|-------------------|-------------------------------|
| w/o SPE | 147.29 / 7.26 | 21.30 / 0.843 / 0.118 |
| w/o ShufV | 173.51 / 4.77 | 18.27 / 0.778 / 0.194 |
| **Full MV-AR** | **144.29 / 8.00** | **22.99 / 0.907 / 0.084** |

| Image Conditioning Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------------------------|-------|-------|--------|
| In-context | 11.92 | 0.538 | 0.477 |
| Cross Attention | 15.13 | 0.709 | 0.310 |
| **IWC** | **22.99** | **0.907** | **0.084** |

Shuffle View is the most impactful component (its removal degrades FID by +29 and PSNR by 4.72). IWC substantially outperforms alternative image condition injection strategies.

### Key Findings
- AR generation is inherently superior to diffusion models for front-back view consistency, as it can leverage intermediate transitional views.
- SSA significantly improves CLIP Score (text-image alignment) over standard self-attention.
- IWC leverages low-level features to guide texture generation, substantially outperforming high-level semantic features such as CLIP or DINO.
- MV-AR is the first unified multi-view generation model capable of simultaneously handling text, image, and shape conditioning.

## Highlights & Insights
- **First successful introduction of the AR paradigm to multi-view generation**, offering a new direction beyond diffusion-based methods.
- **Unified multimodal conditioning (X2mv)**: Text, image, and shape conditions can be used simultaneously, providing strong flexibility.
- Shuffle View data augmentation is simple yet effective and applicable to all sequence-based multi-view generation methods.
- The SSA design addresses the general problem of text representations being corrupted by image tokens under in-context conditioning, and is transferable to other multimodal AR models.

## Limitations & Future Work
- The **unidirectionality** and **discrete encoding** of AR models are inherent constraints; generation quality is bounded by VQVAE reconstruction quality.
- **Error accumulation**: Poor quality in earlier views degrades subsequent views; this issue is acknowledged but not fully resolved.
- The current 2D VQVAE encoding may be superseded by causal 3D VAEs in future work, potentially improving cross-view consistency.
- Generation resolution is limited (256×256), requiring higher-resolution tokenizers.
- Inference speed is slower than diffusion models due to token-by-token generation across all views.

## Related Work & Insights
- **MVDream / Zero123++ / SyncDreamer**: Diffusion-based multi-view generation methods; this paper demonstrates that the AR route is equally viable.
- **LLamaGen / VQGAN / VAR**: Foundational work in AR visual generation; this paper extends the paradigm to multi-view generation.
- **PixelCNN → VQVAE → VAR**: The evolution of AR image generation.
- **Insight**: Progressive AR generation is naturally suited to tasks with inherent sequential structure (e.g., video, multi-view, panorama); it may eventually unify generation and understanding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First successful application of AR to multi-view generation, opening a new research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three conditioning tasks with sufficient ablations; however, evaluation is limited to GSO 30 objects, which is a relatively small scale.
- Writing Quality: ⭐⭐⭐⭐ Problem-solution correspondence is clear, though notation is dense.
- Value: ⭐⭐⭐⭐ Establishes the viability of the AR route and provides meaningful inspiration for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MV-Adapter: Multi-view Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)
- [\[ICCV 2025\] PanSt3R: Multi-view Consistent Panoptic Segmentation](panst3r_multi-view_consistent_panoptic_segmentation.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[CVPR 2025\] 3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement](../../CVPR2025/3d_vision/3denhancer_consistent_multi-view_diffusion_for_3d_enhancement.md)
- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)

</div>

<!-- RELATED:END -->
