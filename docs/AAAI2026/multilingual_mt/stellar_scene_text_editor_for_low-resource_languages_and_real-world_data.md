---
title: >-
  [Paper Note] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data
description: >-
  [AAAI 2026][Scene Text Editing] This paper proposes STELLAR, a framework for scene text editing (STE) in low-resource languages such as Korean, Arabic, and Japanese. STELLAR introduces a language-adaptive glyph encoder and a two-stage training strategy (synthetic pretraining followed by real-data fine-tuning). A reference-free TAS metric is proposed to evaluate font, color, and background style preservation without requiring ground-truth images. Korean recognition accuracy improves from a baseline maximum of 22.1% to 80.4%.
tags:
  - AAAI 2026
  - Scene Text Editing
  - Low-Resource Languages
  - Diffusion Models
  - Domain Adaptation
  - Text Appearance Similarity
date: 2026-05-08
content_hash: 3eed7f5ef28863f2
---

# STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data

**Conference**: AAAI 2026
**arXiv**: [2511.09977](https://arxiv.org/abs/2511.09977)
**Code**: [github.com/yongchoooon/stellar](https://github.com/yongchoooon/stellar)
**Area**: Multilingual Translation
**Keywords**: Scene Text Editing, Low-Resource Languages, Diffusion Models, Domain Adaptation, Text Appearance Similarity

## TL;DR

This paper proposes STELLAR, a framework for scene text editing (STE) in low-resource languages such as Korean, Arabic, and Japanese. STELLAR introduces a language-adaptive glyph encoder and a two-stage training strategy (synthetic pretraining followed by real-data fine-tuning). A reference-free TAS metric is proposed to evaluate font, color, and background style preservation without requiring ground-truth images. Korean recognition accuracy improves from a baseline maximum of 22.1% to 80.4%.

## Background & Motivation

**State of the Field**: Scene text editing (STE) aims to modify text content in images while preserving visual style attributes such as font, color, and background. Growing demand for multilingual STE has emerged from global content industries including advertising, product packaging, game and film localization, and AR sign translation. Technical approaches have evolved from GAN-based methods to diffusion models, with two dominant paradigms: mask-and-inpaint (erasing the text region and inpainting) and direct substitution (disentangling style and content before replacement).

**Limitations of Prior Work**:
- **(1) Insufficient low-resource language support**: The AnyWord-3M dataset contains millions of samples each for Chinese and English, whereas Korean, Arabic, and Japanese each have only approximately 2K samples. Complex writing systems—Arabic's right-to-left directionality and context-dependent glyph variations, Korean's jamo-based combinatorial glyph structure—cannot be correctly handled by models pretrained on English.
- **(2) Synthetic-to-real domain gap**: Most STE models are trained exclusively on synthetically rendered data (e.g., SynthText, SynthTIGER), which fails to capture real-world lighting, texture, and noise. At inference time, noticeable degradation such as color distortion and texture artifacts appears.
- **(3) Inapplicable evaluation metrics**: Metrics such as SSIM, PSNR, MSE, and FID penalize differences even when only text content changes while style is fully preserved, and are inapplicable in the absence of ground-truth images—severely limiting evaluation in real-world scenarios.

**Root Cause**: Low-resource languages exhibit complex glyph structures under data-scarce conditions; models trained on synthetic data fail to generalize to real scenes; and existing metrics cannot distinguish between text content changes and style changes.

**Starting Point**: A language-specific pretrained OCR recognizer (PPOCRv4) is used to supervise a glyph encoder in learning language-relevant structural features. The first real-world multilingual text image pair dataset, STIPLAR, is collected for domain-adaptive fine-tuning. A style-decomposition-based TAS evaluation metric is designed accordingly.

## Method

### Overall Architecture

STELLAR is built upon TextCtrl (direct substitution paradigm), adopting a two-encoder–one-generator architecture:

- **Text Style Encoder $S$** (ViT-B backbone): Extracts style features $C_\text{style}$ from source text images; projects them linearly into texture features $c_\text{tex}$ (driving color/font subtasks) and spatial features $c_\text{spa}$ (driving background removal/text segmentation subtasks).
- **Language-Adaptive Glyph Encoder $T$** (lightweight Transformer): Extracts language-specific glyph features $C_\text{glyph}$ from target text, supervised by visual features from language-specific OCR recognizers.
- **Diffusion Generator $G$** (Stable Diffusion v1.5): $C_\text{glyph}$ is injected via cross-attention to guide text rendering; $C_\text{style}$ is injected via skip-connections and middle blocks to guide style preservation.

### Key Designs

**1. Language-Adaptive Glyph Encoder**

The glyph encoder in the baseline TextCtrl uses ABINet—pretrained exclusively on English—to extract glyph features, yielding poor performance on non-Latin scripts. STELLAR replaces this with a modular design: each target language is paired with its corresponding PPOCRv4 pretrained recognizer (korean_PP-OCRv4, arabic_PP-OCRv4, japan_PP-OCRv4). The glyph encoder aligns character-level features with language-specific OCR visual features under CLIP Loss supervision. This design naturally accommodates each language's structural characteristics—Arabic's right-to-left writing and context-dependent glyphs, Korean's jamo combinatorial structure, and Japanese's mixed kanji/kana system. Extending to a new language requires only plugging in the corresponding OCR recognizer, with no architectural modification.

**2. Multi-Task Learning for the Text Style Encoder**

The style encoder $S$ is trained via four subtasks to learn interpretable style representations:
- **Color transfer**: Colorizes grayscale text (ResNet34 + AdaIN), learning color information from texture features.
- **Font transfer**: Converts template fonts to source style (ResNet34 + PPM), capturing font structural features.
- **Text removal**: Reconstructs the background image using spatial features.
- **Text segmentation**: Generates binary text region masks.

This multi-task learning enables the encoder to acquire structured, transferable style representations, and also provides the computational basis for the TAS metric.

**3. Multi-Stage Training Strategy**

- **Stage 1 (Synthetic Pretraining)**: 200K synthetic text image pairs per language, trained for 100 epochs (66 hours, 2×H100). A key detail: PPOCRv4 is used to filter and retain only high-quality samples with correct OCR recognition; data for each language is organized and trained independently to capture language-specific typographic styles.
- **Stage 2 (Real-Data Fine-Tuning)**: Fine-tuned on the STIPLAR dataset for only 10 epochs (0.3 hours). Using less than 5% of Stage 1 data volume and approximately 10% of the training epochs, the model rapidly adapts to the real domain without any post-hoc inference techniques.

**4. STIPLAR Dataset**

The first real-world scene text image-pair (I2I) dataset targeting low-resource languages. Data is sourced from two streams:
- **Open-source data**: Text regions are cropped from the MLT-2019 training set; 1,000 images per language are selected, with label errors manually corrected and pairs constructed (Korean: 1,818 pairs; Arabic: 2,328 pairs; Japanese: 453 pairs).
- **Web-crawled data**: GPT-4o generates multilingual and English search queries; CC-licensed images are retrieved, then processed through OCR detection, cropping, quality filtering, safety checking, and privacy redaction (Korean: 7,946 pairs; Arabic: 3,988 pairs; Japanese: 1,570 pairs).
- **Total**: Korean 9,764 pairs; Arabic 6,316 pairs; Japanese 2,023 pairs; split 8:2 into training and evaluation sets.

**5. Text Appearance Similarity (TAS) Metric**

The style encoder $S$ is used to decompose the styles of two images into three independent dimensions:

$$\text{TAS} = \frac{s_\text{clr} + s_\text{fnt} + s_\text{bg}}{3}$$

- $s_\text{clr}$: Color similarity, normalized using CIEDE2000 color difference.
- $s_\text{fnt}$: Font similarity, measured via FSIM feature similarity.
- $s_\text{bg}$: Background similarity, measured via MS-SSIM.

Key advantage: unaffected by text content changes; applicable without ground-truth references.

### Loss & Training

- **Style Encoder**: $\mathcal{L} = \mathcal{L}_\text{clr}(\text{MSE}) + \mathcal{L}_\text{fnt}(\text{Dice}) + \mathcal{L}_\text{bg}(\text{MAE}) + \mathcal{L}_\text{seg}(\text{Dice})$
- **Glyph Encoder**: CLIP Loss aligns character-level features with language-specific OCR visual features.
- **Diffusion Generator**: Standard diffusion denoising loss; inference uses 50 denoising steps with CFG scale 2.0.
- **Optimizer**: AdamW, LR $1\times10^{-5}$, weight decay 0.01.

## Key Experimental Results

### Main Results: Comparison with Multilingual Baselines on the STIPLAR Evaluation Set

| Language | Metric | STELLAR | TextFlux | AnyText2 | AnyText |
|----------|--------|---------|----------|----------|---------|
| Korean | SSIM | **0.5061** | 0.3409 | 0.2626 | 0.2822 |
| Korean | TAS | **0.8596** | 0.8464 | 0.7173 | 0.7227 |
| Korean | Rec.Acc | **0.8042** | 0.2213 | 0.0899 | 0.0010 |
| Korean | NED | **0.9115** | 0.4836 | 0.2796 | 0.0116 |
| Arabic | TAS | **0.8601** | 0.8049 | 0.6671 | 0.6908 |
| Arabic | Rec.Acc | **0.6840** | 0.0714 | 0.0082 | 0.0000 |
| Arabic | NED | **0.8985** | 0.4449 | 0.0576 | 0.0054 |
| Japanese | TAS | 0.7714 | **0.7857** | 0.5833 | 0.6393 |
| Japanese | Rec.Acc | **0.4338** | 0.4156 | 0.1013 | 0.0000 |

### Ablation Study: Multi-Stage Training Strategy Analysis

| Configuration | Korean TAS | Korean Rec.Acc | Arabic TAS | Arabic Rec.Acc | Japanese TAS | Japanese Rec.Acc |
|---------------|-----------|----------------|-----------|----------------|-------------|------------------|
| S1 (Synthetic pretraining only) | 0.8362 | 0.6676 | 0.8308 | 0.6412 | 0.7657 | 0.2987 |
| S1 + Post-hoc | 0.8375 | 0.6710 | 0.8333 | 0.6290 | 0.7660 | 0.2961 |
| S1 + S2 downsampled data | 0.8543 | 0.7710 | 0.8562 | 0.6799 | — | — |
| **STELLAR (Full)** | **0.8596** | **0.8042** | **0.8601** | **0.6840** | **0.7714** | **0.4338** |

### TAS Metric Validation: Controlled Synthetic Variation Experiments (Korean)

| Variation Type | SSIM | PSNR | FID | TAS |
|----------------|------|------|-----|-----|
| Text content only (style unchanged) | 0.5768 | 17.29 | 27.28 | **0.8933** |
| Font changed (text unchanged) | 0.6379 | 18.09 | 24.92 | 0.8374 |
| Color changed (text unchanged) | 0.7783 | 21.04 | 21.25 | 0.8379 |
| Background changed (text unchanged) | 0.5653 | 10.25 | 41.77 | 0.7076 |
| All style changed (text unchanged) | 0.3859 | 8.61 | 42.61 | 0.5555 |

### Key Findings

- **Dominant recognition accuracy improvement**: Korean Rec.Acc improves from the highest baseline of 0.2213 (TextFlux) to 0.8042, an absolute gain of +58.3%, demonstrating the fundamental effectiveness of language-adaptive glyph encoding.
- **Stage 2 fine-tuning is highly efficient**: Only 0.3 hours of real-data fine-tuning improves Korean Rec.Acc from 0.6676 to 0.8042 (+13.7%) and background preservation $s_\text{bg}$ from 0.7748 to 0.8325.
- **Post-hoc techniques are largely ineffective**: S1 vs. S1+Post-hoc yields only +0.0013 in TAS, with Rec.Acc even declining on Arabic (0.6412 → 0.6290), confirming that real-data fine-tuning far outperforms inference-stage patching.
- **TAS correctly reflects style preservation**: When only text content changes, the conventional SSIM (0.58) incorrectly assigns a low score, whereas TAS (0.89) correctly identifies that style is unchanged; when color changes, SSIM (0.78) overestimates similarity while TAS (0.84) is more sensitive.
- **Japanese is an exception**: TextFlux slightly outperforms STELLAR on Japanese TAS, as Japanese kanji are visually similar to Chinese characters and the baseline benefits from large-scale Chinese training data transfer; however, STELLAR still leads on Rec.Acc.

## Highlights & Insights

- **Elegant and reusable TAS metric design**: Style evaluation is decomposed into three independent dimensions—color, font, and background—without requiring ground-truth references. This approach generalizes to related tasks such as image style transfer and font generation evaluation.
- **Highly efficient domain adaptation paradigm**: Stage 2 achieves significant gains using less than 5% of the data volume and approximately 0.5% of the training time of Stage 1, demonstrating the strong practical utility of large-scale synthetic pretraining combined with minimal real-data fine-tuning in data-scarce scenarios.
- **Modular language extension design**: Adding a new language requires only plugging in the corresponding PPOCRv4 recognizer without any architectural modification. This plug-and-play design philosophy is worth adopting in multilingual and multimodal systems.
- **Data quality over data quantity**: Stage 1 employs OCR filtering to ensure training data quality; ablation experiments confirm that high-quality data is more effective than simply increasing data volume.

## Limitations & Future Work

- Only three languages (Korean, Arabic, Japanese) are supported; many more low-resource languages such as Hindi, Thai, and Bengali are not covered.
- The STIPLAR dataset totals only approximately 18K pairs, limiting generalization to highly complex scenes and rare fonts.
- Performance degrades on long-text editing due to sparse long-text samples in training data.
- The framework is based on SD v1.5, yielding lower generation resolution and quality than newer architectures (e.g., SDXL, Flux).
- The TAS metric depends on the quality of the style encoder and may be unreliable for languages not specifically trained on.
- Editing curved or perspective-distorted text is not addressed.
- Future directions include unsupervised domain adaptation and zero-shot generalization to unseen languages.

## Related Work & Insights

- **vs. TextCtrl**: The direct baseline for STELLAR, trained on English synthetic data only. This paper extends it via a language-adaptive encoder and real-data fine-tuning, demonstrating the multilingual scalability of the direct substitution paradigm.
- **vs. AnyText/AnyText2**: Representatives of the mask-and-inpaint paradigm, which nearly completely fail on low-resource languages (Korean Rec.Acc: 0.001–0.09), indicating that inpainting-based approaches cannot accurately render target text in the absence of corresponding language training data.
- **vs. TextFlux**: A recent DiT/Flux-based method with reasonable background and style consistency, but far inferior text accuracy compared to STELLAR, suggesting that even powerful generative models require language-specific glyph guidance.
- **Insights**: Real-domain adaptation does not require massive data; the key lies in multi-stage training strategies and data quality control. Evaluation metrics should align with task objectives (style preservation vs. pixel-level similarity), and TAS's decomposition approach is worth generalizing.

## Rating

4/5 Stars

- **Novelty** 4/5: The language-adaptive glyph encoder, two-stage domain adaptation strategy, and TAS metric complement each other to form a complete low-resource language STE solution.
- **Experimental Thoroughness** 4/5: Three languages, four baselines, full ablation, and controlled TAS validation experiments are comprehensive; the dataset and code are both open-sourced.
- **Writing Quality** 4/5: The problem-solution correspondence is clear, and figures and tables are well designed.
- **Value** 4/5: Addresses the industry-critical problem of low-resource language text editing; the dataset and metric are directly reusable.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] ViDia2Std: A Parallel Corpus and Methods for Low-Resource Vietnamese Dialect-to-Standard Translation](vidia2std_a_parallel_corpus_and_methods_for_low-resource_vietnamese_dialect-to-s.md)
- [\[CVPR 2026\] SEA-Vision: A Multilingual Benchmark for Document and Scene Text Understanding in Southeast Asia](../../CVPR2026/multilingual_mt/sea-vision_a_multilingual_benchmark_for_comprehensive_document_and_scene_text_un.md)
- [\[NeurIPS 2025\] Reflective Translation: Improving Low-Resource Machine Translation via Structured Self-Reflection](../../NeurIPS2025/multilingual_mt/reflective_translation_improving_low-resource_machine_translation_via_structured.md)
- [\[AAAI 2026\] Bridging the Multilingual Safety Divide: Efficient, Culturally-Aware Alignment for Global South Languages](bridging_the_multilingual_safety_divide_efficient_culturally-aware_alignment_for.md)
- [\[ICLR 2026\] Prior-based Noisy Text Data Filtering: Fast and Strong Alternative For Perplexity](../../ICLR2026/multilingual_mt/prior-based_noisy_text_data_filtering_fast_and_strong_alternative_for_perplexity.md)

<!-- RELATED:END -->
