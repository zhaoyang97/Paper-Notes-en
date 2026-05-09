---
title: >-
  [Paper Note] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models
description: >-
  [ICCV 2025][Image Generation][content-style decomposition] This work is the first to explore content-style decomposition (CSD) in visual autoregressive (VAR) models. Through three key innovations—scale-aware alternating optimization, SVD-based style embedding rectification, and augmented key-value memory—CSD-VAR achieves content preservation and style transfer quality that surpasses existing diffusion-model-based methods.
tags:
  - ICCV 2025
  - Image Generation
  - content-style decomposition
  - visual autoregressive models
  - textual inversion
  - personalized generation
  - multi-scale representation
date: 2026-05-08
content_hash: 97be47a0acf493a7
---

# CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models

**Conference**: ICCV 2025
**arXiv**: [2507.13984](https://arxiv.org/abs/2507.13984)
**Code**: None
**Area**: Image Generation
**Keywords**: content-style decomposition, visual autoregressive models, textual inversion, personalized generation, multi-scale representation

## TL;DR

This work is the first to explore content-style decomposition (CSD) in visual autoregressive (VAR) models. Through three key innovations—scale-aware alternating optimization, SVD-based style embedding rectification, and augmented key-value memory—CSD-VAR achieves content preservation and style transfer quality that surpasses existing diffusion-model-based methods.

## Background & Motivation

Content-style decomposition (CSD) aims to disentangle content (subject structural details) and style (artistic technique) from a single image, enabling two downstream applications: content re-contextualization and style transfer. Existing methods such as B-LoRA and UnZipLoRA are designed specifically for diffusion models. Visual autoregressive models (VAR), as an emerging alternative to generative models, adopt a "next-scale prediction" paradigm that naturally provides multi-scale generation characteristics; however, their potential for CSD tasks remains unexplored.

Directly applying textual inversion to VAR for CSD yields poor results, as the strong coupling between content and style prevents effective decomposition through simple text-prompt guidance. This motivates the authors to leverage the scale-specific properties of VAR to improve decomposition quality.

## Method

### Overall Architecture

CSD-VAR is built upon text-to-image VAR models (Switti and Infinity as backbones) and employs textual inversion to optimize a content embedding $y_c$ and a style embedding $y_s$, using the prompt format "A photo of a \<$y_c$\> object in \<$y_s$\> style." Model weights are kept frozen; only the text embeddings and K-V memory are optimized. Training uses teacher forcing to enable multi-scale parallel training.

### Key Designs

1. **Scale-aware Alternating Optimization**: Empirical analysis reveals that smaller scales ($k=1,2,3$) and the final scale ($k=10$) in VAR primarily encode style information, while intermediate scales ($k=4,\ldots,9$) primarily encode content information. Accordingly, scales are partitioned into a style group $S_{\text{style}}=\{1,2,3,10\}$ and a content group $S_{\text{content}}=\{4,\ldots,9\}$, with separate style and content losses defined as:

    $$\mathcal{L}_{\text{style}} = \sum_{k \in S_{\text{style}}} \mathcal{L}_k + \alpha \sum_{k' \in S_{\text{content}}} \mathcal{L}_{k'}$$

    $$\mathcal{L}_{\text{content}} = \sum_{k \in S_{\text{content}}} \mathcal{L}_k$$

   where $\alpha=0.1$ controls the influence of large-scale tokens on style. Content and style embeddings are optimized alternately in successive iterations to prevent gradient mixing.

2. **SVD-based Style Embedding Rectification**: Style embeddings may leak content information. To address this, an LLM is used to generate 200 sub-concept variants of the target concept (e.g., "dog" → "Golden Retriever", etc.). SVD is applied to their CLIP text embeddings, and the top $r=10$ singular vectors are taken to construct a projection matrix $P_{\text{proj}} = V_r^\top V_r$. The projection of the style embedding onto the content subspace is then subtracted:

    $$e'_s = e_s - e_s^\top P_{\text{proj}}$$

   This enforces orthogonality between the style embedding and content variants, effectively eliminating content leakage.

3. **Augmented Key-Value Memory**: Text embeddings have limited capacity to represent complex content or style. To address this, $O$ learnable K-V memory pairs are inserted before the self-attention layers of the autoregressive Transformer, injected at scale 1 (style) and scale 4 (content) respectively:

    $$\text{Attn}(Q,K,V;\tilde{K},\tilde{V}) = \text{Attn}(Q, [\tilde{K};K], [\tilde{V};V])$$

   K-V memory pairs are initialized with Xavier uniform initialization. Applying them only in the first Transformer block yields the best efficiency-performance trade-off.

### Loss & Training

- The training loss is a multi-scale cross-entropy loss, with content and style embeddings optimized alternately.
- Adam optimizer, learning rate $10^{-3}$, 200 training steps, batch size = 1.
- At inference, style or content K-V memory is selectively injected based on the prompt type.

## Key Experimental Results

### Main Results

The authors propose the CSD-100 benchmark (100 images, 50 inference prompts, 50K evaluation images in total).

| Method | CSD-C↑ | CLIP-I↑ | CSD-S↑ | DINO↑ | CLIP-T↑ |
|------|--------|---------|--------|-------|---------|
| DreamBooth-C | 0.594 | 0.721 | - | - | 0.271 |
| DreamBooth-S | - | - | 0.537 | 0.519 | 0.289 |
| B-LoRA | 0.523 | 0.592 | 0.476 | 0.346 | 0.278 |
| Inspiration Tree | 0.497 | 0.575 | 0.404 | 0.353 | 0.257 |
| **CSD-VAR (Switti)** | **0.603** | **0.754** | **0.564** | **0.521** | **0.332** |
| **CSD-VAR (Infinity)** | **0.660** | **0.795** | **0.552** | **0.536** | **0.319** |

CSD-VAR comprehensively outperforms existing methods on both content alignment and style alignment while achieving the highest text alignment scores.

### Ablation Study

| SA | SVD | KV | CSD-C↑ | CLIP-I↑ | CSD-S↑ | DINO↑ | CLIP-T↑ |
|----|-----|------|--------|---------|--------|-------|---------|
| ✓ | ✓ | ✓ | 0.603 | 0.751 | 0.564 | 0.517 | 0.330 |
| ✓ | ✓ | - | 0.581 | 0.702 | 0.559 | 0.509 | 0.315 |
| ✓ | - | ✓ | 0.601 | 0.725 | 0.503 | 0.422 | 0.289 |
| - | ✓ | ✓ | 0.501 | 0.612 | 0.547 | 0.508 | 0.270 |
| - | - | - | 0.482 | 0.527 | 0.431 | 0.320 | 0.302 |

- Removing the scale-aware strategy (SA) has the largest negative impact, with significant drops in content alignment and text alignment.
- Removing SVD rectification leads to severe degradation in style alignment (DINO drops from 0.517 to 0.422).
- Removing K-V memory weakens representation capacity for both content and style.

### Key Findings

- Selecting the top-10 singular vectors in SVD rectification ($r=10$) yields the best performance.
- K-V memory needs to be applied only in the first Transformer block; adding more blocks yields only marginal improvements while degrading text alignment.
- Using 4 tokens for textual inversion is optimal; too many tokens (e.g., 16) introduce artifacts.
- A user study (100 participants, 7,500 ratings) demonstrates that CSD-VAR achieves significantly higher preference rates on both content alignment and style alignment.

## Highlights & Insights

- **In-depth exploitation of VAR scale properties**: This work is the first to reveal the correspondence between different scales and content/style in VAR, and translates this insight into a concrete scale-partitioned optimization strategy.
- **Novel SVD orthogonalization approach**: Forcing style embeddings to be orthogonal to content by constructing a content subspace and projecting it out is both elegant and effective.
- **Contribution of the CSD-100 benchmark**: Fills the gap of a standardized evaluation benchmark for CSD tasks.

## Limitations & Future Work

- Decomposition quality remains limited for subjects with fine-grained details (e.g., objects with complex textures).
- CSD-100 is currently used only for evaluation; its potential as a training dataset warrants future exploration.
- The reliance on an LLM to generate sub-concept variants leaves room for further automation.
- No direct comparison with UnZipLoRA is provided, as its code has not been released.

## Related Work & Insights

- B-LoRA achieves implicit decomposition by fine-tuning sensitive layers; the proposed method offers a more flexible alternative through textual inversion combined with VAR-specific properties.
- TokenVerse explores the ability of offset text embeddings in DiT models to encode multiple concepts.
- The scale analysis methodology proposed in this work is generalizable to other multi-scale generative frameworks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to combine VAR with CSD; the scale analysis perspective is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive ablations and a large-scale user study, though comparison with UnZipLoRA is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Method motivation is clear and experimental organization is well-structured.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for personalized applications of VAR models; the CSD-100 dataset offers community value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[ICCV 2025\] AIComposer: Any Style and Content Image Composition via Feature Integration](aicomposer_any_style_and_content_image_composition_via_feature_integration.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ICCV 2025\] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance](stylekeeper_prevent_content_leakage_using_negative_visual_query_guidance.md)
- [\[ICCV 2025\] Randomized Autoregressive Visual Generation](randomized_autoregressive_visual_generation.md)

<!-- RELATED:END -->
