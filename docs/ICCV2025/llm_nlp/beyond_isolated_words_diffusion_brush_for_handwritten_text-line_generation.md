---
title: >-
  [Paper Note] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation
description: >-
  [ICCV 2025][LLM/NLP][Handwritten text generation] This paper proposes DiffBrush, the first diffusion-based method for handwritten text-line generation. Through content-decoupled style learning (column/row masking) and a multi-scale content discriminator (line/word level), DiffBrush substantially outperforms existing methods in both style imitation and content accuracy.
tags:
  - ICCV 2025
  - LLM/NLP
  - Handwritten text generation
  - diffusion model
  - text-line generation
  - style imitation
  - content accuracy
date: 2026-05-08
content_hash: f012d65b5e3be722
---

# Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation

**Conference**: ICCV 2025
**arXiv**: [2508.03256](https://arxiv.org/abs/2508.03256)
**Code**: [https://github.com/dailenson/DiffBrush](https://github.com/dailenson/DiffBrush)
**Area**: Text Generation
**Keywords**: Handwritten text generation, diffusion model, text-line generation, style imitation, content accuracy

## TL;DR

This paper proposes DiffBrush, the first diffusion-based method for handwritten text-line generation. Through content-decoupled style learning (column/row masking) and a multi-scale content discriminator (line/word level), DiffBrush substantially outperforms existing methods in both style imitation and content accuracy.

## Background & Motivation

Existing handwritten text generation methods predominantly focus on **isolated word** generation (e.g., VATr, One-DM, DiffusionPen). However, concatenating words into text lines introduces two critical problems:

**Vertical misalignment**: Human writers maintain consistent vertical baselines across words, a global alignment property that cannot be captured by isolated word generation.

**Loss of horizontal spacing**: Different writers exhibit unique inter-word spacing characteristics, yet word-level generation can only apply fixed spacing during concatenation.

Methods that directly generate text lines are scarce. Representative approaches TS-GAN and CSA-GAN suffer from two fundamental limitations:

- **Ineffective style extraction**: Content recognition loss and style learning loss are jointly optimized on the same model output, causing mutual interference. Minimizing the content recognition loss encourages the model to produce simplified, easily recognizable styles (regular fonts, standard strokes), impeding faithful imitation of diverse handwriting styles.
- **Difficulty ensuring character-level accuracy**: On the IAM dataset, for instance, a text line averages 42 characters (roughly six times that of a single word); line-level content loss promotes global correctness but cannot guarantee the structural accuracy of individual characters.

## Method

### Overall Architecture

DiffBrush consists of three main components: (1) a content-decoupled style module $\xi_{style}$ that extracts content-independent style features; (2) a conditional diffusion generator $\mathcal{G}$ that denoises and generates images conditioned on style and content; and (3) a multi-scale content discriminator $\mathcal{D}$ that provides content supervision at both the line and word levels.

### Key Designs

1. **Content-Decoupled Style Learning**: The core innovation. Directly extracting features from style references conflates content information, causing style–content entanglement. Naïve random masking simultaneously destroys both style and content information. The authors propose two directional masking strategies:

    - **Column-wise masking**: Style features $S_{ver}$ are reshaped into spatial features $\hat{S}_{ver} \in \mathbb{R}^{h \times w \times c}$ and then randomly masked column-wise. Column masking preserves vertical-direction information (character style, vertical alignment) while disrupting horizontal content information. A vertical augmentation head is trained with a Proxy-NCA loss.

    - **Row-wise masking**: Analogously, features are randomly masked row-wise, preserving horizontal-direction information (inter-word spacing, character ligatures) while disrupting vertical content. A horizontal augmentation head is trained accordingly.

   Each augmentation head is paired with a Proxy-NCA loss that pulls masked features from the same writer closer together and pushes those from different writers apart:

    $\mathcal{L}_{style} = \mathcal{L}_{ver} + \mathcal{L}_{hor}$

2. **Multi-Scale Content Discriminator**: Addresses the inability of line-level content supervision to guarantee character-level accuracy.

    - **Line content discriminator $\mathcal{D}_{line}$**: The generated image $x_0$ and the content guidance image $I_{line}$ are concatenated and divided into $n=32$ segments, processed by a 3D CNN to capture global character-order context and verify whether the overall character sequence is correct.

    - **Word content discriminator $\mathcal{D}_{word}$**: A CNN-LSTM attention module localizes each word within the text line, extracts attention-masked word images $x_{word}^t = a_t \cdot x_0$, and judges content correctness word by word. The attention module is pre-trained on the training set and then frozen.

    $\mathcal{L}_{content} = \mathcal{L}_{line} + \mathcal{L}_{word}$

3. **Conditional Diffusion Generator**: Built upon the VAE of Stable Diffusion 1.5, diffusion is performed in the latent space. Style features $S_{ver}$ and $S_{hor}$ are fused with content features $Q$ via a blender module (6-layer Transformer decoder) into a conditioning vector $c$, which guides denoising through cross-attention.

### Loss & Training

$$\mathcal{L}_\mathcal{G} = \mathcal{L}_{diff} + \mathcal{L}_{style} + \lambda \mathcal{L}_{content}$$

- For the first 750 epochs, only $\mathcal{L}_{diff} + \mathcal{L}_{style}$ is used.
- For the final 50 epochs, the content discriminators are introduced; 5 denoising steps are performed per iteration to obtain a coarse image before feeding it to the discriminators.
- $\lambda = 0.05$; conditioning dropout probability is 0.1 (classifier-free guidance); inference uses DDIM with 50 steps.

## Key Experimental Results

### Main Results

Evaluated on the IAM and CVL English handwriting datasets:

| Method | Shot | HWD ↓ | D_CER ↓ | D_WER ↓ | FID ↓ | IS ↑ | GS ↓ |
|--------|------|-------|---------|---------|-------|------|------|
| TS-GAN | one | 2.11 | 44.20 | 87.13 | 16.76 | 1.76 | 2.87e-2 |
| CSA-GAN | few | 2.25 | 42.27 | 84.14 | 13.52 | 1.74 | 1.62e-2 |
| VATr | few | 1.87 | 28.80 | 71.77 | 12.51 | 1.69 | 1.45e-2 |
| DiffusionPen | few | 1.72 | 54.75 | 84.70 | 10.24 | 1.83 | 6.42e-3 |
| One-DM | one | 1.80 | 20.91 | 54.27 | 10.60 | 1.82 | 8.42e-3 |
| **DiffBrush** | **one** | **1.41** | **8.59** | **28.60** | **8.69** | **1.85** | **2.35e-3** |

*IAM dataset. DiffBrush achieves state-of-the-art performance across all metrics in the one-shot setting, with an 18% improvement in HWD and a 59% reduction in D_CER.*

### Ablation Study

| Configuration | HWD ↓ | D_CER ↓ | D_WER ↓ |
|---------------|-------|---------|---------|
| Single style encoder (no masking) | 1.82 | - | - |
| + Random masking | 1.75 | - | - |
| + Content-decoupled style module $\xi_{style}$ | 1.47 | 54.64 | 84.33 |
| + $\xi_{style}$ + $\mathcal{D}_{line}$ | 1.45 | 15.72 | 44.29 |
| + $\xi_{style}$ + $\mathcal{D}_{word}$ | 1.43 | 11.34 | 34.14 |
| + $\xi_{style}$ + $\mathcal{D}_{line}$ + $\mathcal{D}_{word}$ (full) | **1.41** | **8.59** | **28.60** |

*The style module yields a 19.23% improvement in HWD; combining both discriminators reduces D_CER by 84% without degrading HWD.*

### Key Findings

- **CTC recognizer vs. discriminator**: The CTC variant reduces D_CER but severely compromises style imitation (HWD rises above 1.67), producing text with simplified styles; the discriminator improves content accuracy without harming style.
- The vertical augmentation head improves inter-word vertical alignment; the horizontal augmentation head improves inter-word spacing; the two are complementary.
- Directly generated text lines significantly outperform concatenation-based approaches on all metrics (HWD 1.41 vs. 2.17; FID 8.69 vs. 23.92).
- The method generalizes effectively to the Chinese CASIA-HWDB dataset, handling thousands of character categories.

## Highlights & Insights

- The column/row masking strategy is an elegant and principled design: masking in different directions naturally corresponds to distinct style dimensions of a text line (vertical alignment vs. horizontal spacing), and proves more effective than naïve random masking.
- The key insight behind the multi-scale discriminator is that line-level supervision ensures correct global character order while word-level supervision ensures local structural correctness; neither alone is sufficient.
- DiffBrush is the first work to successfully apply diffusion models to handwritten text-line generation, achieving substantial gains over GAN-based methods in both quality and controllability.

## Limitations & Future Work

- Low-frequency characters (punctuation, Greek letters, etc.) occasionally exhibit structural errors, which could be mitigated through data oversampling.
- Text lines wider than 1024 pixels require resizing, potentially losing fine detail.
- Training requires approximately 4 days on 8 RTX 4090 GPUs, and inference still requires 50 DDIM steps.

## Related Work & Insights

- The content-decoupling paradigm is transferable to other generative tasks requiring style–content separation, such as font generation and image style transfer.
- The design philosophy of the multi-scale discriminator offers reference value for other long-sequence generation tasks, such as document image generation.
- The user study methodology (preference study + credibility study) is worth adopting in future work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/llm_nlp/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ICLR 2026\] DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas](../../ICLR2026/llm_nlp/dreamon_diffusion_language_models_for_code_infilling_beyond_fixed-size_canvas.md)
- [\[AAAI 2026\] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](../../AAAI2026/llm_nlp/c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)
- [\[ACL 2026\] FastDiSS: Few-step Match Many-step Diffusion Language Model on Sequence-to-Sequence Generation](../../ACL2026/llm_nlp/fastdiss_few-step_match_many-step_diffusion_language_model_on_sequence-to-sequen.md)
- [\[NeurIPS 2025\] GeoCAD: Local Geometry-Controllable CAD Generation with Large Language Models](../../NeurIPS2025/llm_nlp/geocad_local_geometry-controllable_cad_generation_with_large_language_models.md)

<!-- RELATED:END -->
