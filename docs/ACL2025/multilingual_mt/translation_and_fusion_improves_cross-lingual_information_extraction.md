---
title: >-
  [Paper Note] Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction
description: >-
  [ACL 2025][Multilingual & Machine Translation][Cross-lingual Transfer] TransFusion is proposed, which first translates low-resource language texts into English at inference time, performs information extraction annotation on English, and then uses a fusion model to combine English annotations with the source text to generate final predictions. It significantly outperforms baselines on zero-shot cross-lingual IE tasks across 50 languages (increasing average F1 on MasakhaNER2 f…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Cross-lingual Transfer"
  - "Information Extraction"
  - "Machine Translation"
  - "Low-resource Languages"
  - "Annotation Fusion"
date: 2026-05-08
content_hash: 2f38b27136f1f52f
---

# Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction

**Conference**: ACL 2025  
**arXiv**: [2305.13582](https://arxiv.org/abs/2305.13582)  
**Code**: [https://github.com/edchengg/gollie-transfusion](https://github.com/edchengg/gollie-transfusion)  
**Area**: Multilingual Translation  
**Keywords**: Cross-lingual Transfer, Information Extraction, Machine Translation, Low-resource Languages, Annotation Fusion

## TL;DR

TransFusion is proposed, which first translates low-resource language texts into English at inference time, performs information extraction annotation on English, and then uses a fusion model to combine English annotations with the source text to generate final predictions. It significantly outperforms baselines on zero-shot cross-lingual IE tasks across 50 languages (increasing average F1 on MasakhaNER2 from 47.9 to 62.4).

## Background & Motivation

**Background**: Large language models combined with instruction tuning (such as GoLLIE) have demonstrated strong zero-shot generalization capabilities in information extraction (IE) tasks, enabling them to perform IE on unseen datasets based on annotation guidelines. However, these models are typically pre-trained centered around English, exhibiting severely insufficient processing capabilities for low-resource languages.

**Limitations of Prior Work**: Even top-tier models like GPT-4 experience a sharp decline in NER performance from 80 F1 in English to 55 F1 in low-resource African languages. Although traditional cross-lingual transfer methods (such as translate-train and translate-test) show some effectiveness, they cannot be easily applied to span-level annotation tasks like IE because of alignment difficulties between the translated text and the original annotations.

**Key Challenge**: Low-resource languages lack annotated data for fine-tuning, as well as sufficient unannotated texts for pre-training. Although machine translation models (such as NLLB-200) support translation across 200 languages, their translation quality varies, meaning simple translate-test or translate-train approaches cannot fully leverage the translation information.

**Goal**: How to enable English-centric IE models to utilize external MT systems to significantly improve performance on low-resource languages without requiring target language annotation data.

**Key Insight**: The authors observe that while directly performing IE on the translated English text yields decent English predictions, these predictions might suffer from errors due to translation noise. If the model can learn to "fuse" the English predictions with the original low-resource source text, it can benefit from both worlds.

**Core Idea**: To train the model to translate first, then annotate the English version, and finally fuse the English annotations with the original text during inference, achieving significant improvements in cross-lingual IE via a three-step autoregressive reasoning chain.

## Method

### Overall Architecture

TransFusion is a three-step reasoning framework: (1) Translate: translate the low-resource text into English using NLLB-200; (2) Annotate: annotate the translated text using an English IE model; (3) Fuse: the fusion model combines English annotations with the original low-resource text to generate final predictions. This process is implemented in a decoder-only LLM as a single-pass autoregressive decoding process.

### Key Designs

1. **TransFusion Inference Chain (Autoregressive Annotate-and-Fuse)**:

    - **Function**: Unifies translation, annotation, and fusion into a single autoregressive generation step.
    - **Mechanism**: Based on GoLLIE's Python code representation format, TransFusion instructions are embedded in the prompt. The model takes "annotation guidelines + target language text + English translation + TransFusion instructions" as input, first autoregressively generating the IE annotations on the English translation $\tilde{y}_{src}^{trans}$, and then generating final predictions on the target language $y_{tgt}$ based on these annotations.
    - **Design Motivation**: Unifying into a single decoding pass avoids error accumulation in cascades and allows the fusion model to directly observe the context of English annotations to correct target language predictions.

2. **Cross-lingual Training Data Construction (EasyProject Annotation Projection)**:

    - **Function**: Automatically generates bilingual parallel IE data required for TransFusion training.
    - **Mechanism**: Using the mark-then-translate method of EasyProject, English IE training data is translated into 36 target languages while projecting span-level annotations. A mixed dataset $\mathcal{D}_{mix} = \{x_{src}, y_{src}, x_{tgt}^{trans}, y_{tgt}^{trans}\}$ is constructed, which includes 19,109 English samples and only 891 translated samples, totaling approximately 20,000 instances.
    - **Design Motivation**: Requiring only an extremely small amount of translation data (only 8 samples per language per task) combined with a large amount of English data allows highly efficient cross-lingual transfer, while retaining performance in English and generalization to unseen label schemas.

3. **Two-step TransFusion for Encoder-only Models**:

    - **Function**: Extends the TransFusion framework to encoder-only architectures (such as AfroXLM-R).
    - **Mechanism**: Since encoder-only models cannot generate text, a two-step pipeline is adopted: first, an English fine-tuned model annotates the translated text (using XML tags to label spans); then, the annotated English translation is concatenated with the original text separated by "||", and classification loss is computed on the target language tokens only.
    - **Design Motivation**: Verifies that the TransFusion framework does not depend on specific model architectures and can achieve SOTA even with encoder-only models.

### Loss & Training

For decoder-only models (GoLLIE-TF), continual fine-tuning is conducted based on GoLLIE-7B using QLoRA with LoRA rank=128, alpha=16, learning rate of 1e-4, batch size of 16, and a cosine scheduler. Training takes approximately 6 hours on 2 NVIDIA A40 GPUs. Inference uses greedy decoding.

The fusion training loss is the conditional language modeling loss: $\mathcal{L}_{fusion}(\theta, \mathcal{D}_{mix}) = \sum \mathcal{L}(P(y | x_{tgt}^{trans}, x_{src}, y_{src}; \theta_{fusion}), y_{tgt}^{trans})$, where the next-token prediction loss is computed only on tokens following the TransFusion instructions.

## Key Experimental Results

### Main Results

| Dataset | Metric | GoLLIE-TF | GoLLIE-7B | GPT-4 | Gain (vs GoLLIE) |
|--------|------|-----------|-----------|-------|------|
| MasakhaNER2 (20 languages) | F1 | 62.4 | 47.9 | 54.2 | +14.5 |
| UNER (13 languages) | F1 | 77.8 | 73.6 | 69.0 | +4.2 |
| ACE05 NER (en/ar/zh) | F1 | 61.5 | 58.7 | 41.6 | +2.8 |
| MultiCoNER2 (12 languages, unseen) | F1 | 34.5 | 22.2 | 46.1 | +12.2 |
| Massive (15 low-resource languages, unseen) | F1 | 19.0 | 5.8 | 33.3 | +13.1 |
| Average over all datasets | F1 | 45.7 | 40.2 | 36.6 | +5.5 |

### Ablation Study

| Configuration | MasakhaNER2 F1 | Description |
|------|---------|------|
| GoLLIE-TF (Full) | 62.4 | Includes annotate+fuse |
| w/o annotate | 55.7 (-6.7) | Generated directly from unannotated English translations, validating the critical role of English annotations |
| AfroXLM-R (TransFusion) | 72.1 | Also effective for encoder-only models |
| AfroXLM-R (Trans-train) | 65.8 | TransFusion outperforms simple translate-train |
| AfroXLM-R (Baseline) | 58.8 | No translation augmentation |

### Key Findings

- TransFusion achieves significantly larger gains on low-resource languages than on high-resource ones: a 14.5 F1 gain on MasakhaNER2 (low-resource) versus only 4.2 F1 on UNER (mixed).
- Moderate robustness to translation quality: using different sizes of NLLB models (600M/1.3B/3.3B) shows minimal performance variance, though stronger translation models still yield minor improvements.
- Error analysis indicates that out of 31 errors, 22 originate from the English prediction stage and 12 from the fusion stage, indicating that continuing to improve the English IE model remains the primary direction for enhancement.
- TransFusion can also be applied via prompting to GPT-4, raising F1 on MasakhaNER2 from 53.4 to 62.

## Highlights & Insights

- **The three-step unified decoding design of translation + annotation + fusion** is elegant, preventing error propagation across multi-stage pipelines while utilizing the contextual modeling capability of autoregressive models. This paradigm of "performing on a simplified version first and then fusing/correcting" can be generalized to other cross-modal or cross-domain tasks.
- Significant improvements are achieved with extremely minimal translation data (only 8 samples per language per task), indicating that TransFusion learns a general "translation -> annotation -> fusion" reasoning pattern rather than memorizing mappings of specific languages.
- The framework is highly architecture-agnostic, demonstrating effectiveness on decoder-only (GoLLIE), encoder-only (AfroXLM-R), and proprietary models (GPT-4).

## Limitations & Future Work

- Dependency on the availability of external MT systems, which may not work for extremely low-resource languages unsupported by MT.
- Introduces additional latency and computational cost due to the translation step during inference.
- Error analysis indicates that 71% of errors stem from the English prediction stage; future work should consider how to enable the fusion model to better correct upstream errors.
- Currently verified only on IE tasks; it can be extended to other, more complex NLP tasks such as relation extraction and event detection.

## Related Work & Insights

- **vs Translate-Train**: TransFusion is not only trained on translated data but also learns the inference-time annotate-fuse chain, outperforming Trans-Train by nearly 10 F1 points on MasakhaNER2.
- **vs GPT-4 Zero-shot**: GoLLIE-TF (7B) outperforms GPT-4 on seen label schemas (61.8 vs 33.7) and is also superior on low-resource language NER, demonstrating that smaller models can surpass the zero-shot capabilities of large language models through specialized training.
- **vs Codec**: TransFusion on AfroXLM-R (72.1 F1) outperforms the previous SOTA Codec (70.1 F1), which uses constrained decoding for label projection in translation models.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of TransFusion's three-step reasoning chain is novel, though translation-assisted cross-lingual transfer is a known concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, validated across 50 languages, 12 datasets, and multiple architectures.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear and the experimental setup is detailed, though extensive equations in the methodology section slightly affect readability.
- Value: ⭐⭐⭐⭐ Highly valuable for practical low-resource language IE applications, with strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Machine Translation Models are Zero-Shot Detectors of Translation Direction](machine_translation_models_are_zero-shot_detectors_of_translation_direction.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [\[ACL 2025\] KnowCoder-X: Boosting Multilingual Information Extraction via Code](knowcoder-x_boosting_multilingual_information_extraction_via_code.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)

</div>

<!-- RELATED:END -->
