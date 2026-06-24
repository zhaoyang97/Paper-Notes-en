---
title: >-
  [Paper Note] Emuru: Zero-Shot Styled Text Image Generation, but Make It Autoregressive
description: >-
  [CVPR 2025][Image Generation][Handwritten Text Generation] Proposes Emuru, the first autoregressive model for handwritten text image generation (HTG), combining a specialized VAE and a T5 Transformer encoder-decoder. Trained solely on synthetic data with 100k+ fonts, it generalizes zero-shot to unseen handwritten styles and supports arbitrary-length text generation.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Handwritten Text Generation"
  - "Autoregressive Model"
  - "Style Transfer"
  - "VAE"
  - "Zero-Shot Generalization"
date: 2026-05-08
content_hash: 3764d6d0f27c9b63
---

# Emuru: Zero-Shot Styled Text Image Generation, but Make It Autoregressive

**Conference**: CVPR 2025  
**arXiv**: [2503.17074](https://arxiv.org/abs/2503.17074)  
**Code**: [HuggingFace](https://huggingface.co/blowing-up-groundhogs/emuru)  
**Area**: Image Generation  
**Keywords**: Handwritten Text Generation, Autoregressive Model, Style Transfer, VAE, Zero-Shot Generalization

## TL;DR

Proposes Emuru, the first autoregressive model for handwritten text image generation (HTG), combining a specialized VAE and a T5 Transformer encoder-decoder. Trained solely on synthetic data with 100k+ fonts, it generalizes zero-shot to unseen handwritten styles and supports arbitrary-length text generation.

## Background & Motivation

Stylized handwritten text generation (HTG) aims to generate text images with specified content and styles, finding wide applications in document analysis data augmentation, graphic design, and assistive technologies. Existing approaches suffer from the following key limitations:

- **Poor Generalization**: Existing GANs and diffusion models struggle to generalize to handwritten styles that differ significantly from the training set.
- **Limited Output Length**: Technical architectural constraints lead to a fixed maximum output length (e.g., normalized character widths, fixed canvas size).
- **Background Artifacts**: Inability to properly disentangle the writing style from the reference image background, leading to unwanted background artifacts in the generated images.
- **Low Training Efficiency**: Adversarial training of GANs or multi-step denoising of diffusion models incurs high computational costs.
- **Word-Concatenation Issues**: Most models only generate word-level images, causing scale inconsistencies and baseline alignment issues when concatenated into long text.

## Method

### Overall Architecture

Emuru consists of two independently trained components: (1) a $\beta$-VAE that encodes text images into a sequence of continuous, variable-length vectors in a dense latent space and removes the background during decoding; (2) a T5 Transformer encoder-decoder that autoregressively generates a sequence of visual embeddings compatible with the VAE's latent space, which is then reconstructed into the final image by the VAE decoder. Both components are trained solely on a large-scale synthetic dataset (2.2 million images, >100k fonts).

### Key Designs

#### Key Design 1: Specialized VAE for Text Images — Background Removal and Style Encapsulation

**Function**: Encodes a background-containing text line image into a sequence of latent variables storing only writing style details, automatically stripping the background during decoding.

**Mechanism**: The convolutional VAE encoder downsamples the input image $I^{3 \times W \times H}$ into an embedding tensor of size $c \times h \times w$ ($c=1$, $h=H/8$, $w=W/8$), modeled as a sequence of $w$ vectors of dimension $h$ (each vector encoding a vertical slice of the text line). The objective of the VAE decoder is to **reconstruct a background-free grayscale text image** $I_T$ (rather than the original $I$ with background).

Training losses include: $\mathcal{L}_{MAE}$ ($L_1$ reconstruction), $\mathcal{L}_{KL}$ (KL divergence, with weight $\beta = 10^{-6}$), $\mathcal{L}_{WID}$ (auxiliary style classification loss), and $\mathcal{L}_{HTR}$ (auxiliary text recognition loss).

**Design Motivation**: Utilizing a single-channel latent space with $c=1$ (whereas SD1.5 uses 4 channels and SD3 uses 16 channels) drastically compresses information, making the downstream Transformer lightweight and feasible. Training targets background-free text to ensure the latent variables code writing style only.

#### Key Design 2: Continuous Token Autoregressive Transformer — Arbitrary-Length Generation and Automatic Stopping

**Function**: Autoregressively generates a variable-length sequence of visual embeddings based on style references and desired text.

**Mechanism**: Employs a T5-Large encoder-decoder architecture. The encoder receives the target text (single-character tokenized) and the decoder receives the styled image VAE embeddings along with noise (preventing exposure bias) to autoregressively predict the next visual embedding using causal masked self-attention. Training uses MSE loss and teacher-forcing. The model learns to output "padding" embeddings after the text ends, stopping generation automatically when $P=10$ consecutive padding embeddings are detected.

**Design Motivation**: Unlike discrete token autoregression, continuous tokens avoid information compression and training optimization hurdles introduced by vector quantization. The automatic stopping mechanism removes the output length constraint. Two-stage curriculum learning is adopted: first training on 4-7 word short texts, then fine-tuning on 1-32 word long texts.

#### Key Design 3: Large-Scale Diverse Synthetic Data Training — Foundation for Zero-Shot Generalization

**Function**: Empowers the model with zero-shot generalization capabilities to unseen styles (including real handwriting and printed fonts) via rich training data.

**Mechanism**: Collects English texts from the NLTK corpus, renders them with >100k online fonts (calligraphy + printed), and overlays diverse background images to generate 2.2 million synthetic training samples. Ensuring rare and common character frequencies are roughly uniform.

**Design Motivation**: Existing HTG models are trained on individual datasets, which limits style and text diversity. Large-scale synthetic data provides abundant style variations, allowing the model to learn highly generic style representations. Training only on synthetic data is sufficient for zero-shot handling of real handwriting.

### Loss & Training

- VAE: $\mathcal{L} = \mathcal{L}_{MAE} + 0.005 \cdot \mathcal{L}_{WID} + 0.3 \cdot \mathcal{L}_{HTR} + 10^{-6} \cdot \mathcal{L}_{KL}$
- Transformer: $\mathcal{L}_{MSE}$ (Mean Squared Error between predicted and ground-truth VAE embeddings)

## Key Experimental Results

### VAE Reconstruction Quality Comparison

| VAE Type | FID↓ | BFID↓ | KID↓ | HWD↓ |
|---------|------|-------|------|------|
| SD1.5 VAE | 29.39 | 7.36 | 32.14 | 0.77 |
| SD3 VAE | 21.90 | 3.61 | 23.01 | 0.74 |
| **Emuru VAE** | **19.22** | **1.62** | **16.35** | 0.85 |

### IAM Words Handwritten Word Generation

| Method | FID↓ | BFID↓ | KID↓ | ΔCER↓ | HWD↓ |
|------|------|-------|------|-------|------|
| HWT | 27.83 | 15.09 | 19.64 | 0.15 | 2.01 |
| One-DM | 27.54 | 10.73 | 21.39 | 0.10 | 2.28 |
| DiffPen | **15.54** | **6.06** | **11.55** | 0.06 | **1.78** |
| Emuru | 63.61 | 37.73 | 62.34 | 0.19 | 3.03 |

### Key Findings

- Emuru VAE outperforms general-purpose VAEs like SD1.5/SD3 in reconstruction quality, using only about 16% of the parameters.
- On IAM Words word-level generation, Emuru gets a higher FID because it is trained solely on synthetic data and not fine-tuned on IAM.
- However, in **line-level generation and cross-dataset generalization** (CVL, RIMES, Karaoke), Emuru exhibits superior generalization performance.
- It can generate arbitrary-length text lines with consistent baseline alignment, which is unachievable by existing methods.
- Generated images have no background artifacts, making them directly usable for downstream applications like OCR.

## Highlights & Insights

1. **First Autoregressive HTG Model**: The continuous token + automatic stopping mechanism fundamentally solves the limitations on output length.
2. **Zero-Shot Generalization via Synthetic-Only Training**: The diversity of 100k+ fonts gives the model strong style generalization capabilities.
3. **Background Removal Native to VAE**: With the training targeted at background-free text, a clean output is naturally produced during decoding.
4. **Single-Channel Latent Space Design**: Extreme compression makes the downstream Transformer lightweight and highly efficient, needing only a single 4090 GPU for training.

## Limitations & Future Work

- Performs worse than specialized fine-tuned methods on specific datasets (like IAM), showing a trade-off for zero-shot generalization.
- Currently supports only Latin characters; applicability to complex writing systems like Chinese or Arabic has not been verified.
- Autoregressive generation is slower than single-forward diffusion or GAN methods.
- Extreme compression with $c=1$ may result in loss of color and detail information, only generating grayscale text.

## Related Work & Insights

- **VATr/VATr++**: Transformer-GAN-based HTG methods.
- **DiffPen / One-DM**: Diffusion-model-based HTG methods.
- **GIVT**: A general framework for continuous token autoregressive generation, which is the core inspiration for Emuru.
- **T5**: The base architecture of Emuru's Transformer.

## Rating

⭐⭐⭐⭐ — Paradigm innovation (first autoregressive HTG). Zero-shot generalization from purely synthetic training is impressive, and arbitrary-length generation combined with background removal offers unique, practical value. Although it might be outperformed by specialized methods on specific datasets, its generalization and scalability are superior. The design choices (single-channel VAE, automatic stopping) are highly thoughtful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Make It Count: Text-to-Image Generation with an Accurate Number of Objects](make_it_count_text-to-image_generation_with_an_accurate_number_of_objects.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[CVPR 2025\] Z-Magic: Zero-shot Multiple Attributes Guided Image Creator](z-magic_zero-shot_multiple_attributes_guided_image_creator.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)
- [\[NeurIPS 2025\] Flex-Judge: Text-Only Reasoning Unleashes Zero-Shot Multimodal Evaluators](../../NeurIPS2025/image_generation/flex-judge_text-only_reasoning_unleashes_zero-shot_multimodal_evaluators.md)

</div>

<!-- RELATED:END -->
