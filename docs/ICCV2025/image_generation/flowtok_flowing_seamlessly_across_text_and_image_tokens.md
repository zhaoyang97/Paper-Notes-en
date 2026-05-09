---
title: >-
  [Paper Note] FlowTok: Flowing Seamlessly Across Text and Image Tokens
description: >-
  [Image Generation] FlowTok proposes encoding both text and images as compact 1D token representations ($77 \times 16$) and directly evolving between text and image tokens via flow matching, eliminating the need for complex conditioning mechanisms or noise schedules, thereby enabling efficient cross-modal generation.
tags:
  - Image Generation
date: 2026-05-08
content_hash: c6ba97107bd50b2a
---

# FlowTok: Flowing Seamlessly Across Text and Image Tokens

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2503.10772](https://arxiv.org/abs/2503.10772)
- **Code**: [GitHub](https://github.com/TACJu/FlowTok)
- **Area**: Image Generation / Cross-modal Generation
- **Keywords**: Flow Matching, 1D Token, Text-to-Image Generation, Compact Representation, Cross-modal

## TL;DR

FlowTok proposes encoding both text and images as compact 1D token representations ($77 \times 16$) and directly evolving between text and image tokens via flow matching, eliminating the need for complex conditioning mechanisms or noise schedules, thereby enabling efficient cross-modal generation.

## Background & Motivation

Conventional text-to-image generation methods treat text as a conditioning signal, progressively guiding a denoising process from Gaussian noise toward the target image. This requires complex conditioning mechanisms (e.g., cross-attention, concatenation) and noise scheduling strategies.

FlowTok explores a simpler paradigm: **directly evolving between text and image modalities via flow matching**. This requires projecting both modalities into a shared latent space, where the representational gap between text (1D sequences, high-dimensional semantics) and images (2D spatial structure, redundant information) constitutes the core challenge.

Prior work CrossFlow mapped text into a 2D latent space to match image embeddings, but the additional computational overhead of the text variational autoencoder made it slower than SD1.5/2.1, undermining its efficiency motivation.

## Method

### Overall Architecture

The core idea of FlowTok is to encode both text and images as compact 1D tokens of shape $77 \times 16$:
- **Image side**: A modified TA-TiTok encodes images into $\mathbf{Z}_I \in \mathbb{R}^{K \times D}$ ($K=77$, $D=16$).
- **Text side**: A CLIP text encoder extracts initial embeddings, which are then mapped to a low-dimensional variational latent space $\mathbf{Z}_T \in \mathbb{R}^{N \times D}$ via a text projector.
- **Generative model**: Vanilla flow matching with DiT blocks, where text tokens serve directly as the source distribution.

Compared to the conventional 2D flow matching latent space of $32 \times 32 \times 4$, FlowTok achieves a **3.3× compression ratio**.

### Key Designs

**1. Image Tokenizer Improvements**
- Based on the TA-TiTok architecture, the number of latent tokens $K$ is set to 77 to match the CLIP text length.
- RoPE replaces learnable 1D positional encodings to improve positional modeling.
- SwiGLU FFN replaces the standard MLP to enhance latent space quality.
- Encoder uses ViT-B; decoder uses ViT-L; patch size = 16.

**2. Text Projector**
- 6 Transformer blocks with skip connections.
- Projects CLIP text embeddings ($77 \times 768$) into a low-dimensional space ($77 \times 16$).
- KL divergence regularization is applied to the projected text tokens to introduce generative diversity.

**3. Text Alignment Loss**

To prevent semantic information loss due to channel dimensionality reduction, a CLIP-style contrastive loss is introduced:

$$\mathcal{L}_{\text{align}} = \frac{1}{2}(\text{CE}(\text{logits}_{TZ}, \text{labels}) + \text{CE}(\text{logits}_{ZT}, \text{labels}))$$

where logits are computed as scaled cosine similarities using a learnable temperature parameter $\tau$.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{fm}} + \gamma_1 \cdot \mathcal{L}_{\text{kld}} + \gamma_2 \cdot \mathcal{L}_{\text{align}}$$

- $\mathcal{L}_{\text{fm}}$: flow matching velocity prediction loss
- $\mathcal{L}_{\text{kld}}$: KL divergence regularization ($\gamma_1 = 10^{-4}$)
- $\mathcal{L}_{\text{align}}$: text alignment contrastive loss ($\gamma_2 = 1$)

### Model Scales

| Model | Depth | Width | MLP | Heads | Parameters |
|-------|-------|-------|-----|-------|------------|
| FlowTok-B | 12 | 768 | 3072 | 12 | 153M |
| FlowTok-XL | 28 | 1152 | 4608 | 16 | 698M |
| FlowTok-H | 36 | 1280 | 5120 | 20 | 1.1B |

## Key Experimental Results

### Main Results: Zero-shot Text-to-Image Generation

| Method | Params | Open Data | Training Cost (8-A100 days) | Inference Speed (samples/s) | COCO FID-30K↓ | MJHQ-30K FID↓ |
|--------|--------|-----------|-----|------|------|------|
| PixArt-α | 630M | ✗ | 94.1 | 7.9 | 7.32 | 9.85 |
| SD-2.1 | 860M | ✓ | 1041.6 | - | 13.45 | 26.96 |
| Show-o | 1.3B | ✓ | - | 1.0 | 9.24 | 14.99 |
| CrossFlow | 950M | ✗ | 78.8 | 1.1 | 9.63 | - |
| **FlowTok-XL** | **698M** | **✓** | **20.4** | **22.7** | **10.06** | **7.68** |
| **FlowTok-H** | **1.1B** | **✓** | **26.1** | **18.2** | **9.67** | **7.15** |

### Ablation Study: Text Alignment Loss

| Alignment Target | COCO FID-30K↓ |
|------------------|---------------|
| Average Pooling | 36.02 |
| **MLP** | **29.14** |

| Loss Type | COCO FID-30K↓ |
|-----------|---------------|
| Cosine | 31.80 |
| **Contrastive** | **29.14** |

| $\gamma_2$ | COCO FID-30K↓ |
|------------|---------------|
| **1.0** | **29.14** |
| 2.0 | 30.59 |

### Key Findings

1. **Extreme efficiency**: FlowTok-H requires only 26.1 eight-A100 GPU days for training — 1/40 of SD-2.1 and 1/3.6 of PixArt-α.
2. **Fast inference**: FlowTok-XL generates 22.7 images per second, 20× faster than CrossFlow and 22× faster than Show-o.
3. **Memory efficiency**: The largest model supports a batch size of 8K on 8 A100s without gradient checkpointing or gradient accumulation.
4. **Bidirectional generation**: The same framework naturally supports image-to-text generation; FlowTok-XL achieves a CIDEr score of 117.0 on COCO Karpathy.

## Highlights & Insights

1. **Paradigm innovation**: Text is recast from a "conditioning signal" to a "source distribution," with flow matching evolving directly between modalities, eliminating complex conditioning mechanisms.
2. **Unified 1D token representation**: By encoding images as 1D tokens, the method elegantly unifies the representations of text and images.
3. **Only 20 sampling steps**: The compact 1D latent space requires far fewer sampling steps than conventional 2D approaches.
4. **Fully open-source data**: All training uses publicly available datasets, ensuring reproducibility.

## Limitations & Future Work

1. Image resolution is limited to 256; higher resolutions have not been validated.
2. Reliance on the CLIP text encoder (77-token limit) constrains the handling of long textual descriptions.
3. The 1D representation may be less effective than 2D methods for spatially fine-grained control.
4. Image-to-text generation performance is competitive but not state-of-the-art.

## Related Work & Insights

- **CrossFlow**: Also explores cross-modal flow matching, but uses a 2D latent space, incurring substantial computational overhead.
- **TA-TiTok**: Provides the foundational architecture for 1D image tokenization.
- **DiT**: FlowTok's generative model is based on DiT blocks, with cross-attention and other conditioning mechanisms removed.

## Rating

⭐⭐⭐⭐ — Strong paradigm innovation with significant efficiency gains; however, limitations in resolution and text length reduce practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EmotiCrafter: Text-to-Emotional-Image Generation based on Valence-Arousal Model](emoticrafter_text-to-emotional-image_generation_based_on_valence-arousal_model.md)
- [\[ICCV 2025\] TRCE: Towards Reliable Malicious Concept Erasure in Text-to-Image Diffusion Models](trce_towards_reliable_malicious_concept_erasure_in_text-to-image_diffusion_model.md)
- [\[ICCV 2025\] What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?](what_makes_for_text_to_360-degree_panorama_generation_with_stable_diffusion.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] Trans-Adapter: A Plug-and-Play Framework for Transparent Image Inpainting](trans-adapter_a_plug-and-play_framework_for_transparent_image_inpainting.md)

</div>

<!-- RELATED:END -->
