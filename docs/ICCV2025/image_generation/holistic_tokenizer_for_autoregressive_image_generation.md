---
title: >-
  [Paper Note] Holistic Tokenizer for Autoregressive Image Generation
description: >-
  [ICCV 2025][Image Generation][Image Tokenizer] This paper proposes Hita, a holistic-to-local image tokenizer that captures global attributes such as texture, material, and shape via learnable global queries, and integrates dual codebooks with a causal-attention fusion module. Without modifying the AR model architecture, Hita reduces ImageNet 256×256 generation FID to 2.59, accelerates training convergence by 2.1×, and supports zero-shot style transfer and image completion.
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Image Tokenizer"
  - "Autoregressive Generation"
  - "Holistic Tokens"
  - "VQGAN"
  - "LlamaGen"
date: 2026-05-08
content_hash: 815a11cca61dba3b
---

# Holistic Tokenizer for Autoregressive Image Generation

**Conference**: ICCV 2025
**arXiv**: [2507.02358](https://arxiv.org/abs/2507.02358)  
**Code**: [https://github.com/CVMI-Lab/Hita](https://github.com/CVMI-Lab/Hita)  
**Area**: Image Generation / Autoregressive Models / Image Tokenization
**Keywords**: Image Tokenizer, Autoregressive Generation, Holistic Tokens, VQGAN, LlamaGen

## TL;DR
This paper proposes Hita, a holistic-to-local image tokenizer that captures global attributes such as texture, material, and shape via learnable global queries, and integrates dual codebooks with a causal-attention fusion module. Without modifying the AR model architecture, Hita reduces ImageNet 256×256 generation FID to 2.59, accelerates training convergence by 2.1×, and supports zero-shot style transfer and image completion.

## Background & Motivation

### State of the Field
Autoregressive (AR) image generation models follow the GPT paradigm: images are first encoded into discrete token sequences via VQVAE/VQGAN, and then predicted token-by-token using causal Transformers such as Llama. Existing tokenizers are primarily based on patch-level representations and lack global information.

### Core Challenges

**Absence of Global Information**: Patch-level tokens carry only local information, making it difficult for AR models to maintain global consistency during sequential generation.

**Limitations of Causal Attention**: AR models predict tokens autoregressively with causal attention and cannot access future tokens, which hinders long-range dependency modeling.

**Poor Zero-Shot Completion Quality**: When models such as LlamaGen are tasked with completing the lower half of an image, the generated content is semantically inconsistent with the upper half (e.g., a fish merges into a fish–bird hybrid).

### Key Motivation
If the tokenizer can supply global information (e.g., overall texture, color, and material) at the beginning of the token sequence, these global tokens can serve as a prefix to guide the generation of subsequent patch tokens, without requiring any modification to the AR model itself.

## Method

### Overall Architecture
Hita comprises three core modules: (1) **Global Feature Extraction** — learnable queries aggregate global attributes from patch embeddings and foundation model features; (2) **Dual Codebook Independent Quantization** — global tokens and patch tokens are quantized with separate codebooks; (3) **Token Fusion and Decoding** — a lightweight causal fusion module prioritizes global tokens before passing representations to the decoder for image reconstruction.

### Global Feature Extraction
Patch embeddings are extracted via the VQGAN encoder $\mathcal{E}(\cdot)$. Additionally, $M$ learnable queries $Q \in \mathbb{R}^{M \times C}$ aggregate information from the global patch embeddings through an attention mechanism. Pre-trained DINOv2 is introduced to inject semantic features:

$$Q, Z = \mathcal{E}_{\text{trans}}(Q \oplus \mathcal{E}(I) \oplus \mathcal{H}(I))$$
$$\overline{Q}, \overline{Z} = \mathcal{E}_{\text{causal}}(Q \oplus Z)$$

where $\mathcal{H}(\cdot)$ denotes DINOv2 and $\oplus$ denotes concatenation. A key design choice is the **causal Transformer** $\mathcal{E}_{\text{causal}}$, which places global queries at the front of the sequence and patch embeddings (in raster-scan order) at the back, naturally aligning the latent space with the causal generation pattern of AR models.

### Dual Codebook Independent Quantization
The global queries $\overline{Q}$ and patch embeddings $\overline{Z}$ are quantized independently using separate codebooks:
- Global quantizer $\mathcal{Q}_H(\cdot)$: a dedicated codebook capturing global attributes.
- Patch quantizer $\mathcal{Q}_P(\cdot)$: a standard patch-level codebook.
- Both codebooks adopt $\ell_2$ normalization, low-dimensional vectors, and a large codebook size (16,384).

### Token Fusion and Decoding (Key Innovation)
Naively concatenating global and patch tokens before passing them through a Transformer decoder leads to **global codebook collapse** — patch tokens directly influence the reconstruction of their corresponding patches via skip connections, bypassing the global tokens entirely.

The proposed solution is that, after causal Transformer fusion, the **last $k$ global tokens replace the first $k$ patch tokens** before being fed into the decoder:

$$\tilde{Q}, \tilde{Z} = \hat{\mathcal{E}}_{\text{causal}}(\hat{Q} \oplus \hat{Z}_p)$$
$$\hat{I} = \mathcal{D}(\mathcal{R}(\tilde{Q}_{[-k:]} \oplus \tilde{Z}_{[:-k]}))$$

This renders the information supplied by patch tokens to the decoder incomplete, compelling them to interact with global tokens for compensation, thereby preventing the degenerate solution and codebook collapse.

### Loss & Training
The total loss is: $\mathcal{L} = \alpha \cdot \mathcal{L}_{vq} + \lambda \cdot \mathcal{L}_{AE}$, where:
- $\mathcal{L}_{vq} = \mathcal{L}_{vq}(\overline{Q}) + \mathcal{L}_{vq}(\overline{Z}_p)$ (sum of quantization losses for both codebooks)
- $\mathcal{L}_{AE} = \mathcal{L}_2 + \mathcal{L}_P(\text{LPIPS}) + \lambda_G \cdot \mathcal{L}_G(\text{PatchGAN})$

### AR Generation
Once Hita is trained, a standard Llama AR model can be integrated without any modification: global tokens are first generated as a prefix prompt, followed by sequential patch token generation, which is then passed through the fusion module and decoder to produce the final image.

## Key Experimental Results

### Main Results: ImageNet 256×256 Class-Conditional Generation

| Model | Params | FID↓ | IS↑ | Precision↑ | Recall↑ |
|-------|--------|------|-----|------------|---------|
| LDM-4 | 400M | 3.60 | 247.7 | 0.87 | 0.48 |
| DiT-XL/2 | 675M | 2.27 | 278.2 | 0.83 | 0.57 |
| LlamaGen-B | 111M | 8.31 | 154.7 | 0.84 | 0.38 |
| **Hita-B** | 111M | **5.85** | **212.3** | 0.84 | 0.41 |
| LlamaGen-L | 343M | 4.24 | 206.7 | 0.83 | 0.49 |
| **Hita-L** | 343M | **3.75** | **262.1** | 0.85 | 0.48 |
| LlamaGen-XXL | 1.4B | 2.89 | 236.2 | 0.81 | 0.56 |
| **Hita-XXL** | 1.4B | **2.70** | **274.8** | 0.84 | 0.55 |
| LlamaGen-3B | 3B | 2.61 | 251.9 | 0.80 | 0.56 |
| **Hita-2B** | 2B | **2.59** | **281.9** | 0.84 | 0.56 |

Hita consistently and substantially outperforms LlamaGen at all model scales. The 2B model surpasses LlamaGen-3B with fewer parameters, and outperforms the LDM-4 diffusion model by 0.7 FID and 19.6 IS.

### Ablation Study

| Configuration | rFID↓ | gFID↓ | gIS↑ | Linear Probe↑ |
|---------------|-------|-------|------|---------------|
| Baseline (no global) | 1.31 | 9.37 | 162.6 | 14.2 |
| + Learnable Queries | 1.15 | 6.32 | 187.9 | 28.2 |
| + DINOv2 Injection | **1.03** | **5.85** | **212.3** | **36.6** |

Learnable queries alone significantly improve both reconstruction and generation quality (gFID decreases by 3.05), and DINOv2 injection further enhances semantic representation. The linear probe accuracy rising from 14.2 to 36.6 confirms that global tokens capture rich semantic information.

### $k$ Value Ablation (Token Fusion Design Validation)
- At $k=0$, the global codebook utilization collapses to a very low value, as patch tokens completely bypass global tokens.
- At $k>0$, the global codebook functions properly; $k=4$ achieves the best reconstruction and generation quality.
- This validates the necessity of enforcing patch token dependence on global tokens.

### Training Acceleration
Training time to reach FID=4.22 is reduced by **2.1×** — the global token prefix guides faster convergence of the AR model.

## Highlights & Insights
- **Global-local disentangled tokenization paradigm**: Distinguished from TiTok's compressed 1D tokens and VAR's multi-scale tokens, Hita explicitly separates semantic global information from spatial local information.
- **Emergent zero-shot capabilities**: The trained tokenizer directly supports style transfer (by replacing global tokens) and image completion without additional training.
- **Elegant anti-collapse mechanism in token fusion**: By truncating the direct reconstruction path of patch tokens, the model is forced to exploit global information, simultaneously resolving codebook collapse and improving generation quality.
- **Seamless compatibility with AR models**: No modification to the Llama architecture or introduction of bidirectional attention is required.

## Limitations & Future Work
- Validation is limited to ImageNet 256×256 class-conditional generation; experiments on text-guided high-resolution generation are absent.
- The semantic interpretability of global tokens is demonstrated primarily through style transfer in an indirect manner, lacking systematic representation analysis.
- Approximately 128 additional tokens are introduced (441→569), increasing inference sequence length by roughly 29%.
- Fair comparisons with recent methods incorporating bidirectional attention, such as VAR and MAR, are not provided.

## Related Work & Insights
- **Image Tokenizers**: VQVAE/VQGAN, ViT-VQGAN, RQ-VAE, MAGVIT-v2, TiTok, VQGAN-LC
- **AR Image Generation**: DALL-E, Parti, LlamaGen, VAR, MAR, Show-o
- **Semantic Injection**: DINOv2 feature extraction and fusion

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The global-local disentanglement paradigm is novel, and the anti-collapse token fusion design is technically elegant.
- **Technical Depth**: ⭐⭐⭐⭐ — The analysis of codebook collapse and the corresponding solution are rigorously motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations and comparisons against mainstream methods.
- **Value**: ⭐⭐⭐⭐ — Open-source code, seamless Llama compatibility, and practical zero-shot capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Spectral Image Tokenizer](spectral_image_tokenizer.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)
- [\[ICCV 2025\] Music-Aligned Holistic 3D Dance Generation via Hierarchical Motion Modeling](music-aligned_holistic_3d_dance_generation_via_hierarchical_motion_modeling.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[CVPR 2025\] TokenFlow: Unified Image Tokenizer for Multimodal Understanding and Generation](../../CVPR2025/image_generation/tokenflow_unified_image_tokenizer_for_multimodal_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
