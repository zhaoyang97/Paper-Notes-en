---
title: >-
  [Paper Note] MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding
description: >-
  [ICCV2025][Multimodal VLM][Unified vision-language model] This paper proposes Semantic Discrete Encoding (SDE), which injects pretrained CLIP semantic features into the quantization process of a visual tokenizer, enabling discrete visual tokens to be naturally aligned with language tokens. With only 24M image-text pairs, the resulting unified model achieves state-of-the-art performance on both visual understanding and generation.
tags:
  - ICCV2025
  - Multimodal VLM
  - Unified vision-language model
  - visual discretization
  - semantic encoding
  - visual understanding and generation
  - autoregressive
date: 2026-05-08
content_hash: 99275485bcb73e1e
---

# MUSE-VL: Modeling Unified VLM through Semantic Discrete Encoding

**Conference**: ICCV2025
**arXiv**: [2411.17762](https://arxiv.org/abs/2411.17762)
**Authors**: Rongchang Xie, Chen Du, Ping Song, Chang Liu (ByteDance)
**Code**: Not open-sourced
**Area**: Multimodal VLM
**Keywords**: Unified vision-language model, visual discretization, semantic encoding, visual understanding and generation, autoregressive

## TL;DR

This paper proposes Semantic Discrete Encoding (SDE), which injects pretrained CLIP semantic features into the quantization process of a visual tokenizer, enabling discrete visual tokens to be naturally aligned with language tokens. With only 24M image-text pairs, the resulting unified model achieves state-of-the-art performance on both visual understanding and generation.

## Background & Motivation

Unified multimodal large models that simultaneously support visual understanding and visual generation represent a prominent research direction. The central challenge is how to convert visual inputs into discrete tokens analogous to text tokens, enabling LLMs to handle both vision and language under a unified next-token prediction paradigm.

Existing methods suffer from the following limitations:

**Traditional VQ tokenizers (e.g., VQGAN) focus solely on low-level pixel reconstruction.** Since training objectives consist only of image reconstruction loss, the extracted visual tokens contain no high-level semantic information and are difficult to align with language tokens. This causes unified models using discrete tokens (e.g., Chameleon, Show-o) to significantly underperform dedicated understanding models.

**High training cost**: Due to the semantic deficiency of visual tokens, Emu3 requires training an 8B model from scratch, Chameleon demands massive data, and VILA-U requires 720M image-text pairs for alignment.

**Existing solutions each have drawbacks**: Emu3 circumvents unified modeling difficulties by fine-tuning two separate models; Janus increases model complexity by using separate encoders; VILA-U attempts to combine contrastive and reconstruction losses but faces convergence difficulties due to loss conflicts.

Core motivation: **Can semantic information be injected at the visual discretization stage so that visual tokens are inherently aligned with language?** This would substantially reduce the data requirements and alignment difficulty of subsequent VLM training.

## Method

### 3.1 Semantic Discrete Encoding (SDE)

SDE is the core contribution of this paper. Built upon the VQGAN architecture, it incorporates semantic information alongside pixel information during quantization.

**Architecture**: On top of the standard image encoder + codebook + image decoder pipeline, two additional components are introduced:

- **Semantic Encoder**: A frozen pretrained SigLIP model that extracts semantic features $T$ from images. Having been trained on large-scale image-text pairs, SigLIP features are naturally aligned with language.
- **Semantic Decoder**: A Vision Transformer that reconstructs semantic features from quantized discrete representations, ensuring that the discrete codes retain semantic information.

**Encoding pipeline**:

1. Image $x \in \mathbb{R}^{H \times W \times 3}$ is passed through the image encoder to obtain features $z = Enc(x)$
2. The frozen SigLIP extracts semantic features $T$
3. **Key innovation**: Semantic features are fused with image features before quantization: $z_q = \text{Quant}(T + z)$
4. The quantized features $z_q$ are fed into the image decoder to reconstruct the original image $\hat{x}$, and into the semantic decoder to reconstruct semantic features $z_s$

**Loss function**:

$$L_{\text{total}} = L_{\text{sem}} + L_{\text{img}} + L_{\text{vq}}$$

- $L_{\text{sem}} = 1 - \cos(Dec_s(z_q), T)$: semantic reconstruction loss that maximizes the cosine similarity between decoded semantic features and SigLIP features
- $L_{\text{img}} = \ell_2(x, \hat{x}) + L_P(x, \hat{x}) + \lambda_G L_G(\hat{x})$: pixel reconstruction + perceptual loss + adversarial loss
- $L_{\text{vq}}$: standard VQ commitment loss

**Key distinction from VILA-U**: VILA-U employs a text encoder for contrastive learning, which introduces loss conflicts. SDE instead uses the CLIP **image encoder** to extract semantic features (which already encode text-aligned information), and avoids direct conflict between contrastive and reconstruction losses through feature addition fusion and semantic reconstruction.

### 3.2 Unified Vision-Language Modeling (MUSE-VL)

A unified VLM is constructed on top of the SDE tokenizer with a remarkably simple design:

- Images are converted by SDE into discrete token sequences of length $h \times w$, which are concatenated with text tokens
- Only the LLM embedding layer needs to be extended (adding 32,768 visual token IDs)
- `<soi>` and `<eoi>` markers delimit the visual token sequence
- **No modification to the LLM architecture is required**; the training objective is standard next-token prediction

**Two-stage training**:

1. **Pretraining**: Using image-text pair data, loss is computed over all tokens (visual + text) to learn visual token embeddings and cross-modal alignment
2. **Instruction fine-tuning**:
   - Understanding tasks: visual tokens appear in the prompt; loss is computed only over the response text
   - Generation tasks: text descriptions appear in the prompt; loss is computed over the generated visual tokens

**Base LLMs**: Compatible with Yi-1.5 (9B/34B) and Qwen-2.5 (7B/32B), both directly adaptable.

## Key Experimental Results

### Tokenizer Comparison (Same LLM + Same Data)

| Tokenizer | MMBench | SEED | MMStar | AVG |
|-----------|---------|------|--------|-----|
| VQGAN | 32.0 | 42.7 | 29.1 | 34.6 |
| SEED | 63.1 | 57.8 | 39.1 | 53.3 |
| LaVIT | 63.3 | 59.5 | 40.3 | 54.4 |
| **SDE (ours)** | **70.6** | **68.1** | **43.8** | **60.8** |

SDE outperforms VQGAN by +26.2% (AVG) and SEED/LaVIT by +6.4–7.5%.

### Understanding Performance of Unified Models

| Model | LLM | Token Type | MMBench | SEED | MMMU | SQA-I | AVG |
|-------|-----|-----------|---------|------|------|-------|-----|
| Chameleon | 7B scratch | Discrete | 31.1 | 30.6 | 25.4 | 46.8 | 33.3 |
| Emu3 | 8B scratch | Discrete | 58.5 | 68.2 | 31.6 | 89.2 | 58.8 |
| LLaVA-NeXT | Yi-34B | Continuous | 79.3 | 75.9 | 51.1 | 81.8 | 66.4 |
| **MUSE-VL** | Qwen2.5-7B | Discrete | 72.1 | 69.1 | 39.7 | 93.5 | 63.6 |
| **MUSE-VL** | Qwen2.5-32B | Discrete | **81.8** | 71.0 | 50.1 | **95.0** | **70.1** |

- The 7B model outperforms Emu3 by +13.6% on MMBench and +4.8% on average
- The 32B model surpasses dedicated understanding models including LLaVA-NeXT 34B

### Visual Generation Performance

| Model | MJHQ-30K FID↓ | GenEval |
|-------|---------------|---------|
| SD-XL | 9.55 | 0.55 |
| Show-o | 9.24 | 0.53 |
| Emu3 | - | 0.54 |
| **MUSE-VL** | **7.73** | **0.56** |

### Data Efficiency

| Model | Image-Text Pairs |
|-------|-----------------|
| Show-o | 35M |
| VILA-U | 720M |
| **MUSE-VL** | **24M** |

MUSE-VL surpasses VILA-U (which uses 720M pairs) with only 24M pairs.

### Ablation Study

| Image Branch | Semantic Branch | rFID | MMBench | SEED | MMStar | AVG |
|:---:|:---:|------|---------|------|--------|-----|
| ✓ | - | 2.63 | 42.8 | 48.5 | 38.1 | 43.1 |
| - | ✓ | - | 72.5 | 67.5 | 48.1 | 62.7 |
| ✓ | ✓ | 2.26 | 72.1 | 69.1 | 49.6 | 63.6 |

The semantic branch provides a decisive improvement to understanding performance (+20.5% AVG), while adding the image branch preserves generation capability and further marginally improves understanding.

## Highlights & Insights

1. **Elegant avoidance of loss conflict**: Rather than directly applying contrastive loss (the pain point of VILA-U), SDE uses feature addition fusion with pretrained CLIP image encoder features plus semantic reconstruction. CLIP image features are inherently aligned with text, obviating the need for cross-modal contrastive training.
2. **Exceptional data efficiency**: The 24M training pairs represent only 1/30 of VILA-U's data requirement. The root cause is that SDE makes visual tokens naturally semantically aligned, substantially reducing the alignment burden at the VLM stage.
3. **Plug-and-play**: No LLM architecture modifications are needed; any pretrained LLM can be adapted simply by extending the embedding layer, demonstrating strong scalability.
4. **Semantic encoding visualization**: The paper shows that identical codebook IDs correspond to identical semantic concepts (e.g., cat ears, strawberries), directly demonstrating that discrete codes capture high-level semantics.
5. **Favorable scaling behavior**: Performance consistently improves from 7B to 32B and from 256 to 384 resolution, consistent with scaling laws.

## Limitations & Future Work

1. **Limited generation resolution**: The current maximum is 384×384, yielding 27×27 = 729 discrete tokens, which falls considerably short of the 1024 resolution achievable by diffusion models.
2. **Reconstruction quality trade-off**: SDE achieves an rFID of 2.26, roughly on par with the pure reconstruction method LLamaGEN (2.19), but it remains unclear whether semantic constraints would impair reconstruction fidelity at higher resolutions or in more complex scenes.
3. **Dependence on frozen SigLIP**: The quality ceiling of the semantic encoder is bounded by SigLIP's capabilities; if SigLIP has limited understanding of certain visual concepts, SDE's semantic injection will be correspondingly constrained.
4. **No video understanding/generation**: The current framework supports only single images and has not been extended to the video modality.
5. **Codebook utilization**: The actual utilization rate of the 32,768-entry codebook is not reported; an oversized codebook may result in certain codes being insufficiently trained.

## Related Work & Insights

- **Emu3**: Also a discrete unified model but trained from scratch on an 8B model. MUSE-VL demonstrates that leveraging pretrained LLMs with a semantic tokenizer is a more efficient path.
- **Janus**: Uses separate encoders (CLIP for understanding, VQGAN for generation). MUSE-VL unifies both with a single SDE tokenizer.
- **VILA-U**: The most direct point of comparison, which similarly attempts to inject semantics into the tokenizer but suffers from convergence difficulties due to contrastive loss conflicts. SDE's feature addition + semantic reconstruction approach is more elegant.
- **TokenFlow**: A concurrent work that uses dual codebooks to decouple semantic and pixel features. MUSE-VL achieves similar results with a single codebook.
- **BEITv2**: Serves as the design inspiration for SDE's semantic decoder; the idea of reconstructing pretrained features via a ViT decoder is successfully transferred here.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MaTVLM: Hybrid Mamba-Transformer for Efficient Vision-Language Modeling](matvlm_hybrid_mamba-transformer_for_efficient_vision-language_modeling.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)
- [\[ICCV 2025\] CVPT: Cross Visual Prompt Tuning](cvpt_cross_visual_prompt_tuning.md)
- [\[ICCV 2025\] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation](multimodal_llms_as_customized_reward_models_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
