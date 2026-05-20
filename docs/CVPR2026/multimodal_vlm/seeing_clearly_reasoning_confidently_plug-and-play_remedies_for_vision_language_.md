---
title: >-
  [Paper Note] Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness
description: >-
  [CVPR 2026][Multimodal VLM][rare object recognition] This paper proposes an efficient plug-and-play module that learns multimodal class embeddings to enhance VLM recognition and reasoning on rare objects. On the visual s…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "rare object recognition"
  - "visual token enhancement"
  - "multimodal class embeddings"
  - "plug-and-play"
  - "VLM robustness"
date: 2026-05-08
content_hash: a7f4916b551b8aa2
---

# Seeing Clearly, Reasoning Confidently: Plug-and-Play Remedies for Vision Language Model Blindness

**Conference**: CVPR 2026
**arXiv**: [2602.19615](https://arxiv.org/abs/2602.19615)  
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: rare object recognition, visual token enhancement, multimodal class embeddings, plug-and-play, VLM robustness

## TL;DR
This paper proposes an efficient plug-and-play module that learns multimodal class embeddings to enhance VLM recognition and reasoning on rare objects. On the visual side, a cross-attention adapter refines visual tokens; on the textual side, object detection prompts are injected. Without fine-tuning the VLM, the method achieves a significant gain from 72.8 to 75.4 on CODA-LM.

## Background & Motivation
**Background**: VLMs perform well on general visual understanding, but exhibit notable degradation on reasoning tasks involving rare or uncommon objects.

**Limitations of Prior Work**:
- Attention weights in intermediate decoding layers of VLMs are significantly lower for rare object regions compared to common objects.
- Approaches that introduce stronger visual encoders or perform full model fine-tuning incur high computational costs and are not optimized at the object level.
- Retrieval-augmented learning (RAL) requires large-scale external data and VLM fine-tuning, and may cause catastrophic forgetting.

**Key Challenge**: Rare objects appear with extremely low frequency in pretraining data, leading to insufficient visual-language alignment. Existing improvement methods are not designed at the object level and require expensive full model fine-tuning.

**Goal**: Efficiently improve VLM perception and reasoning for rare objects without fine-tuning the VLM.

**Key Insight**: Attention visualizations reveal that VLMs pay insufficient attention to rare objects in intermediate decoding layers. This motivates remedies from two directions — enhancing visual tokens (making rare objects more salient) and enriching text prompts (guiding attention to target regions).

**Core Idea**: Learn multimodal class embeddings that fuse features from visual foundation models and synonym-augmented text descriptions. These embeddings serve both as anchors for visual token refinement and as object detectors for generating text prompts.

## Method

### Overall Architecture
Three stages: (a) learning multimodal class embeddings (visual + text alignment) → (b) visual token enhancement (cross-attention adapter) → (c) text prompt injection (class embeddings as detector → generating object prompts).

### Key Designs

1. **Multimodal Class Embedding Learning**:

    - **Adaptive Semantic Augmentation**: An LLM generates synonyms and descriptive text for each rare category. Categories with fewer samples receive more text variants (re-sampling) to mitigate class imbalance.
    - **Dual-Branch Feature Extraction**: A VFM (DINOv3) extracts object visual features $z_v$; CLIP extracts text features $z_t$; both are projected into the LLM embedding space.
    - **Cross-Modal Alignment**: $\mathcal{L}_{align}$ applies contrastive learning to pull together visual and textual features of the same class.
    - **Class Embedding Optimization**: $\mathcal{L}_{class}$ classification loss with EMA updates, making class embeddings serve as unified anchors for both modalities.
    - Initialization: Class embeddings are initialized from the mean visual features of same-class samples, yielding greater stability than random initialization.

2. **Visual Token Enhancement (Cross-Attention Adapter)**:

    - Input: frozen VLM visual tokens $V$ and class embeddings $W$.
    - Cross-attention: $V$ as query, $W$ as key-value → refined output $\hat{V} = V + \mathcal{C}_{att}(V, W)$.
    - Refined tokens are injected only at the first decoding layer of the VLM.
    - Loss = reconstruction loss $\mathcal{L}_{rec}$ (keeping $\hat{V}$ close to the distribution of $V$) + autoregressive loss $\mathcal{L}_{autoreg}$.
    - **Design Motivation**: Class embeddings carry discriminative knowledge of rare objects, which is injected into visual tokens via cross-attention.

3. **Text Prompt Injection at Inference**:

    - Class embeddings $W$ are used as a detector: cosine similarity is computed between VFM visual tokens and each class embedding.
    - Top-$k$ categories are selected as candidate objects.
    - Candidate object names are injected into the text prompt, e.g., "In this image, there might be objects such as: [bollard, debris, ...]".
    - **Design Motivation**: Explicit text prompts guide the LLM's attention to relevant objects.

### Loss & Training
- Stage 1: $\mathcal{L}_{align} + \mathcal{L}_{class}$ (training class embeddings and projection layers, 20 epochs).
- Stage 2: $\mathcal{L}_{adapter} = \mathcal{L}_{rec} + \mathcal{L}_{autoreg}$ (training the adapter, 10 epochs).
- The VLM remains frozen throughout. All training can be completed on a single RTX 4090.

## Key Experimental Results

### Main Results (CODA-LM GPT Score)

| Model | Barrier↑ | Cone↑ | Vehicle↑ | All↑ |
|------|:-------:|:-----:|:--------:|:----:|
| LLaVA-1.5-7B | 39.3 | 54.5 | 48.9 | 46.5 |
| **LLaVA-1.5-7B + Ours** | **68.3** | **84.9** | **73.0** | **72.8** |
| Qwen2.5-VL-7B | 70.9 | 84.9 | 66.5 | 67.9 |
| **Qwen2.5-VL-7B + Ours** | **79.8** | **91.7** | **71.0** | **75.4** |
| InternVL3-8B | 59.7 | 73.3 | 66.9 | 65.4 |
| **InternVL3-8B + Ours** | **76.4** | **85.8** | **73.8** | **74.2** |

### Ablation Study

| Configuration | All↑ | Note |
|------|:----:|------|
| LLaVA-1.5-7B baseline | 46.5 | No enhancement |
| + Text prompt only | 56.2 | Prompt is helpful but insufficient |
| + Visual enhancement only | 65.8 | Visual enhancement contributes more |
| + Visual enhancement + Text prompt | **72.8** | Best with both components |

### Key Findings
- LLaVA-1.5-7B improves by 26.3 points (46.5→72.8), a remarkably large gain.
- Generalizes across models: effective for LLaVA, Qwen2.5-VL, and InternVL3.
- Visual enhancement contributes more than text prompt injection, but the two are complementary.
- Requires only a single RTX 4090 and minimal training data (CODA-LM, tens of thousands of QA pairs).
- The largest gain is on the Barrier category (39.3→68.3), a prototypical rare object class.

## Highlights & Insights
- **Dual utility of multimodal class embeddings**: The same set of class embeddings serves both as visual refinement anchors (keys and values in cross-attention) and as object detectors (similarity matching), achieving two goals simultaneously.
- **Efficient frozen-VLM paradigm**: Only a lightweight cross-attention adapter and class embeddings are trained, achieving substantial improvements without modifying any VLM parameters — highly valuable for deploying existing large models.
- **Attention visualization analysis**: Directly demonstrates insufficient attention to rare objects in intermediate VLM layers, providing clear motivation for the proposed method.

## Limitations & Future Work
- Requires a predefined set of rare categories; cannot handle entirely unseen categories at test time.
- The number of class embeddings is bounded by the number of rare categories $C$; very large-scale category sets would require architectural adjustments.
- Top-$k$ detection may introduce false positives, generating incorrect text prompts that mislead reasoning.
- Performance gains on GeoBench-VLM (satellite imagery) are weaker than on CODA-LM, indicating remaining challenges under extremely scarce data.

## Related Work & Insights
- **vs. VLM internal feature supervision methods (LLaVA-Grounding)**: These methods align all visual tokens with VFM features without targeting rare objects; the proposed method uses class embeddings for object-level refinement, achieving greater precision and efficiency.
- **vs. Retrieval-Augmented Learning (RAL)**: RAL retrieves from large-scale external data and fine-tunes the VLM, incurring high computational cost and risk of forgetting; the proposed method requires neither large-scale data nor VLM fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐ The dual-purpose design of multimodal class embeddings is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model validation with attention visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and figures are intuitive.
- Value: ⭐⭐⭐⭐ A practical solution for rare object understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](prune2drive_a_plug-and-play_framework_for_accelerating_vision-language_models_in.md)
- [\[AAAI 2026\] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit](../../AAAI2026/multimodal_vlm/llmc_benchmarking_vision-language_model_compression_with_a_plug-and-play_toolkit.md)
- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[AAAI 2026\] Global Compression Commander: Plug-and-Play Inference Acceleration for High-Resolution Large Vision-Language Models](../../AAAI2026/multimodal_vlm/global_compression_commander_plug-and-play_inference_acceler.md)

</div>

<!-- RELATED:END -->
