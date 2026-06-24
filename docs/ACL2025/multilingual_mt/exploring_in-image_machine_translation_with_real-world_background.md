---
title: >-
  [Paper Note] Exploring In-Image Machine Translation with Real-World Background
description: >-
  [ACL 2025][Multilingual & Machine Translation][In-Image Machine Translation] This paper proposes the DebackX model, which processes images by separating them into background and text-image components. It addresses the In-Image Machine Translation (IIMT) task under real-world complex backgrounds for the first time, outperforming existing methods in both translation quality and visual presentation.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "In-Image Machine Translation"
  - "Background Separation"
  - "Vector Quantization"
  - "Visual Text Generation"
  - "Multimodal Translation"
date: 2026-05-08
content_hash: 44ad9acfb015dddc
---

# Exploring In-Image Machine Translation with Real-World Background

**Conference**: ACL 2025  
**arXiv**: [2505.15282](https://arxiv.org/abs/2505.15282)  
**Code**: [GitHub](https://github.com/BITHLP/DebackX)  
**Area**: Multilingual Translation  
**Keywords**: In-Image Machine Translation, Background Separation, Vector Quantization, Visual Text Generation, Multimodal Translation  

## TL;DR

This paper proposes the DebackX model, which processes images by separating them into background and text-image components. It addresses the In-Image Machine Translation (IIMT) task under real-world complex backgrounds for the first time, outperforming existing methods in both translation quality and visual presentation.

## Background & Motivation

**Background**: In-Image Machine Translation (IIMT) aims to translate text within an image from one language to another, where both the input and output are in image modality. Key application scenarios include video subtitle translation and visual translation tools.

**Limitations of Prior Work**: Prior IIMT research was conducted only under highly simplified scenarios: Tian et al. (2023) utilized single-line black text on white backgrounds, while Lan et al. (2024) employed monochrome backgrounds. These settings drastically differ from real-world complex background texts (such as video subtitles overlaid on natural images) and cannot be directly applied to real-world scenarios.

**Key Challenge**: Advancing IIMT to real-world scenarios faces two major challenges: first, complex backgrounds interfere with translation quality—traditional OCR-NMT-Render cascaded frameworks suffer from error propagation; second, erasing text areas disrupts background integrity and fails to maintain font style consistency, resulting in poor visual quality.

**Goal**: To achieve high-quality translation combined with superior visual performance in IIMT under real-world background conditions.

**Key Insight**: To "deconstruct" the image into background and text-image, directly perform image-to-image translation on the text-image (avoiding OCR cascade errors), and then fuse the translated text-image back with the background.

**Core Idea**: Background-text separation + direct text-image translation + fusion reconstruction, avoiding OCR error propagation and background damage.

## Method

### Overall Architecture

DebackX consists of three components:
1. **Text-Image Background Separation**: Decomposes the source image into a background image and a source text-image.
2. **Image Translation**: Transforms the source text-image into a target text-image.
3. **Text-Image Background Fusion**: Fuses the background with the target text-image to generate the final output.

### Key Designs

#### 1. Text-Background Separation Module

Two sets of ViT Encoder-Decoders are used to extract the background and text respectively:

$$\text{Background} = G_{\text{back}}(E_{\text{deback}}(x))$$
$$\text{Source Text-Image} = G_{\text{text}}(E_{\text{detext}}(x))$$

All ViT configurations: patch_size=16, d_model=512, layers=8, heads=8, d_ff=2048.

#### 2. Image Translation Module

Trained in two stages:

**Stage 1 — Vector Quantization (VQ)**: A ViT Encoder is used to encode the text-image, which is then quantized into a sequence of discrete tokens via a codebook (size 8192, dimension 32):

$$z_i = q(E(x_i)) = \arg\min_{e_k \in q} \|E(x_i) - e_k\|_2$$

**Stage 2 — Translation**: Reformulates image translation into a code-sequence to code-sequence transformation. Key designs include:
- **Code Encoder**: Encodes the source code sequence.
- **Pivot Decoder**: Utilizes a TIT (Text-Image Translation) auxiliary task to inject semantic information, outputting $H_{\text{pivot}}^D$ for both auxiliary text translation and subsequent code decoding.
- **Linear Adapter**: Adapts the output of the Pivot Decoder to the input of the Code Decoder.
- **Code Decoder**: Autoregressively generates the target code sequence.

During inference, the auxiliary text is decoded first, and the target code sequence is decoded after obtaining the full Pivot representation.

#### 3. Text-Background Fusion Module

$$\text{Target Image} = G_{\text{fuse}}(E_{\text{back}}(i_b) + E_{\text{text}}(i_t))$$

Two ViT Encoders encode the background and target text-image, respectively, and their features are summed and passed through a ViT Decoder to generate the final image.

### Loss & Training

- **Separation Module**: $\mathcal{L}_{\text{sep}} = \mathcal{L}_{\text{img}}(i_b, \hat{i_b}) + \mathcal{L}_{\text{img}}(i_t, \hat{i_t})$, where $\mathcal{L}_{\text{img}} = \|y - \hat{y}\|^2 + 0.1 \cdot \mathcal{L}_{\text{Perceptual}}$
- **VQ Stage**: $\mathcal{L}_{\text{VQ}} = \|y - \hat{y}\|^2 + 0.1 \cdot \mathcal{L}_{\text{Perceptual}} + \|\text{sg}[z_q] - E(x)\|_2^2$ (including commitment loss and EMA update)
- **Translation Stage**: $\mathcal{L}_{\text{trans}} = \mathcal{L}_{\text{code}} + \mathcal{L}_{\text{TIT}}$ (both using cross-entropy with label-smoothing=0.1)
- **Fusion Module**: Also utilizes $\mathcal{L}_{\text{img}}$
- **Training Steps**: Separation 25K, VQ 50K, Pre-training 100K + Fine-tuning 50K, Fusion 15K

## Key Experimental Results

### Datasets

The IIMT30k dataset is built based on Multi30k, containing three types of fonts (TNR, Arial, Calibri), with approximately 25K training, 864 validation, and 2740 testing pairs each. The image size is $48 \times 512$.

### Main Results

| System | BLEU (De→En Test) | COMET (De→En Test) | FID (De→En Test) ↓ | FID (En→De Test) ↓ |
|---|---|---|---|---|
| VQGAN | 0.6 | 24.4 | 21.3 | 20.7 |
| TIT-Render | 12.1 | 49.6 | 133.2 | 119.0 |
| McTIT-Render | 11.7 | 47.3 | 137.5 | 117.5 |
| Translatotron-V | 1.6 | 24.8 | 10.1 | 17.5 |
| **DebackX** | **12.8** | **50.0** | **9.0** | **8.7** |

DebackX is slightly superior to the TIT-Render series in translation quality (BLEU 12.8 vs. 12.1), and substantially leads in visual quality (FID 9.0 vs. 133.2).

### Ablation Study

| Setting | Pivot | Deback | BLEU (De→En Test) |
|---|---|---|---|
| #1 Full Model | ✓ | ✓ | 5.9 |
| #2 w/o Pivot | ✗ | ✓ | 1.4 |
| #3 w/o Deback | ✓ | ✗ | 1.1 |
| #4 w/o Both | ✗ | ✗ | 0.6 |

### Key Findings

- **Significant Pre-training Effect**: Training only with IIMT30k yields a BLEU of 5.9; incorporating IWSLT pre-training increases it to 12.8, and adding WMT14 pre-training further increases it to 17.5.
- **Multi-font Adaptation**: Training with a mixture of three fonts achieves font consistency of 96.5%–97.3%.
- **Impact of OCR Errors**: Even on ground-truth images, OCR recognition achieves only 67.5–81.0 BLEU, indicating that the upper bound of real-world evaluation is limited.
- **GPT-4o Comparison**: GPT-4o is capable of generating meaningful translations but fails to correctly handle layout and fonts.

## Highlights & Insights

- **Elegant Decoupled Design**: Decomposes the complex end-to-end IIMT task into three independently trainable sub-modules, each with a clear objective.
- **Unique Pre-training Advantage**: Text-images can easily be constructed in batches using parallel corpora, allowing the pre-training data to scale massively (from 100K to 1M), which is a capability other IIMT methods lack.
- **Dual-purpose Pivot Decoder**: Cleverly designed to provide semantic supervision for the auxiliary TIT task and semantic guidance for the Code Decoder simultaneously.

## Limitations & Future Work

- All sub-modules employ the most basic ViT, leaving more advanced vision architectures (such as DiT, SwinTransformer) unexplored.
- The multi-stage training is computationally costly, making end-to-end optimization difficult.
- While simulating real video subtitles, the dataset is still synthetic, presenting a gap with genuine natural scene text (characterized by arbitrary angles and locations).
- Only the German-to-English translation direction was tested; generalization to other language pairs has not been verified.

## Related Work & Insights

- Translatotron-V (Lan et al., 2024) is effective on simple backgrounds but fails to handle complex scenes → Insight: Complex scenes require explicit background-text decoupling.
- OCR-VQGAN (Rodríguez et al., 2023) uses pre-trained OCR features to calculate text-aware loss → Insight: Decoupled text-images can mitigate OCR error rates.
- Text generation efforts like AnyText/GlyphDraw → Future work can integrate diffusion models to enhance generation quality.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — For systematically addressing IIMT with complex backgrounds for the first time, using a novel framework design of background separation and fusion.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Including comparisons against multiple models, ablation studies, pre-training research, multi-font experiments, and GPT-4o comparison.
- **Writing Quality**: ⭐⭐⭐⭐ — The problem definition is clear, well-illustrated, and the experimental design is highly sensible.
- **Value**: ⭐⭐⭐⭐ — Fills the research gap in real-world scenario IIMT, possessing clear practical application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World](lemonade_a_large_multilingual_expert-annotated_abstractive_event_dataset_for_the.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)

</div>

<!-- RELATED:END -->
