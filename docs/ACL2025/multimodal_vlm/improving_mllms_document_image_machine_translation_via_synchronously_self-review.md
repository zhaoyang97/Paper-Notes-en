---
title: >-
  [Paper Note] Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency
description: >-
  [ACL 2025 (Findings)][Multimodal VLM][Document Image Translation] This paper proposes the Synchronously Self-Reviewing (SSR) paradigm. By requiring the MLLM to first generate OCR text before outputting the translation during the document image translation process, SSR leverages the "bilingual cognitive advantage" to alleviate catastrophic forgetting caused by fine-tuning, while simultaneously enhancing both OCR and Document Image Machine Translation (DIMT) performance.
tags:
  - "ACL 2025 (Findings)"
  - "Multimodal VLM"
  - "Document Image Translation"
  - "Multimodal Large Language Models (MLLMs)"
  - "OCR"
  - "Catastrophic Forgetting"
  - "Self-Reviewing Mechanism"
date: 2026-05-08
content_hash: 71eefffff1d2b843
---

# Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2507.08309](https://arxiv.org/abs/2507.08309)  
**Code**: None  
**Area**: Multimodal VLM / Machine Translation  
**Keywords**: Document Image Translation, Multimodal Large Language Models (MLLMs), OCR, Catastrophic Forgetting, Self-Reviewing Mechanism

## TL;DR

This paper proposes the Synchronously Self-Reviewing (SSR) paradigm. By requiring the MLLM to first generate OCR text before outputting the translation during the document image translation process, SSR leverages the "bilingual cognitive advantage" to alleviate catastrophic forgetting caused by fine-tuning, while simultaneously enhancing both OCR and Document Image Machine Translation (DIMT) performance.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have demonstrated superb performance on document image tasks, especially possessing powerful optical character recognition (OCR) capabilities. Document Image Machine Translation (DIMT) is a more complex task that requires the model to simultaneously handle both cross-modal (image $\rightarrow$ text) and cross-lingual (source language $\rightarrow$ target language) conversions.

**Limitations of Prior Work**: When fine-tuning MLLMs on DIMT datasets via supervised fine-tuning (SFT), catastrophic forgetting occurs. Although translation capability improves, the original OCR ability degrades significantly. This occurs because training the model solely on translation pairs alters how the model understands document images, causing it to "forget" how to accurately recognize source language text.

**Key Challenge**: DIMT essentially involves two steps—first accurately recognizing the source language text in the image (OCR), and then translating it into the target language. However, standard SFT only trains the model with translation pairs without explicitly maintaining OCR capabilities, resulting in a trade-off between these two abilities.

**Goal**: Design a fine-tuning paradigm that enhances DIMT translation quality while maintaining or even boosting the model's OCR capabilities.

**Key Insight**: The authors draw inspiration from the "Bilingual Cognitive Advantage" theory in cognitive science. Research shows that bilinguals display better performance in cognitive tasks because the interaction between two languages in the brain enhances cognitive flexibility. Similarly, allowing the model to perform OCR recognition before translating can leverage the synergetic effects between the two tasks.

**Core Idea**: During MLLM fine-tuning for DIMT, the model is compelled to generate OCR-recognized text before producing the final translation (i.e., "self-reviewing" its OCR proficiency). Through this synchronous self-reviewing mechanism, the model preserves its monolingual capabilities while learning cross-lingual translation.

## Method

### Overall Architecture

The overall workflow of the SSR method is highly straightforward: given a document image and a translation prompt as input, the model is required to generate two parts sequentially—first the OCR text of the source language, followed by the translation in the target language. These two parts are generated sequentially in a single forward pass, separated by a special token. The supervision signal during training contains both OCR labels and translation labels.

### Key Designs

1. **Synchronously Self-Reviewing (SSR)**:

    - **Function**: Force-activate the model's OCR capability during the translation generation process.
    - **Mechanism**: The output format of the DIMT task is defined as `"[OCR] {source text} [Translation] {target text}"`. During training, the OCR part is supervised using the document's ground-truth source text, while the translation part is supervised using the reference translation. During inference, the model first autonomously recognizes the source text and then translates it; the OCR output serves as a contextual translation reference.
    - **Design Motivation**: By forcing the model to perform OCR prior to translation, it must consistently maintain its OCR capability throughout fine-tuning, thereby avoiding catastrophic forgetting. Meanwhile, the OCR output provides an explicit source text reference for the translation, improving translation accuracy.

2. **Dual-task Joint Training Framework**:

    - **Function**: Co-optimize both OCR and translation objectives.
    - **Mechanism**: The total loss consists of two parts: the OCR recognition loss and the translation loss, which are distinguished within the same sequence via positional tokens. Both tasks share the model parameters and the visual encoder. The gradient signal from the OCR task helps maintain the accuracy of visual feature extraction, whereas the gradient signal from the translation task promotes learning the cross-lingual mapping.
    - **Design Motivation**: Compared to separately incorporating OCR data for multi-task training in SFT, SSR's advantage lies in executing both tasks on the exact same input, ensuring semantic consistency between OCR and translation.

3. **Progressive Training Strategy**:

    - **Function**: Introduce training data of varying difficulties in phases.
    - **Mechanism**: The first stage utilizes simple documents with high OCR accuracy, allowing the model to learn the format and baseline capabilities. The second stage introduces more complex documents (handwritten text, complex layouts, etc.) to gradually ramp up the difficulty. This curriculum learning strategy avoids training instability that might occur from starting on difficult samples.
    - **Design Motivation**: The quality of DIMT data is highly variable. Progressive training allows the model to learn more stably.

### Loss & Training

The total loss is a standard autoregressive cross-entropy loss calculated across both OCR tokens and translation tokens. An attention mask ensures that the OCR portion only attends to the image input, whereas the translation portion can attend to both the image and the preceding OCR output.

## Key Experimental Results

### Main Results

Comparison of BLEU scores on multiple document image translation benchmarks:

| Method | Zh→En BLEU | En→De BLEU | OCR CER↓ | OCR F1 |
|------|-----------|-----------|---------|--------|
| Baseline MLLM (SFT) | 28.3 | 21.7 | 8.2% | 86.5% |
| Baseline MLLM (No Fine-tuning) | 15.6 | 12.4 | 3.1% | 94.2% |
| Pipeline (OCR+NMT) | 26.8 | 20.9 | 3.1% | 94.2% |
| SSR (Ours) | 31.5 | 24.2 | 3.8% | 93.1% |

The SSR method outperforms all baselines in translation quality (BLEU +3.2/+2.5), while showing only a slight degradation in OCR performance (CER increase of 0.7%), which is significantly better than standard SFT (where CER degraded from 3.1% to 8.2%).

### Ablation Study

| Configuration | BLEU | OCR CER↓ | Description |
|------|------|---------|------|
| SSR Full | 31.5 | 3.8% | Full model |
| SFT Translation Only | 28.3 | 8.2% | Severe catastrophic forgetting |
| Translation + OCR Multi-task | 29.7 | 5.1% | Decoupled multi-task mitigates to some extent |
| SSR without OCR Supervision | 29.1 | 6.7% | Obvious degradation in unsupervised OCR part |
| SSR + Progressive Training | 32.1 | 3.6% | Progressive training further improves |

### Key Findings

- **Catastrophic forgetting is the core bottleneck of DIMT**: Standard SFT degrades OCR CER from 3.1% to 8.2%, indicating that fine-tuning purely for translation severely damages visual understanding capabilities.
- The full SSR model improves by 1.8 BLEU compared to decoupled multi-task training, proving that synchronous generation (OCR $\rightarrow$ translation) is more effective than independent multi-tasking.
- The OCR output serves as an "intermediate step" for translation, similar to applying the Chain-of-Thought (CoT) concept to multimodal translation.
- The progressive training strategy brings steady additional gains, yielding significant improvements particularly on complex documents (handwritten text, multi-column layouts).

## Highlights & Insights

- **A simple yet effective solution inspired by cognitive science**: The mapping from bilingual cognitive advantage to SSR is highly natural and compelling. The methodology is exceptionally straightforward to implement, requiring only modifications to the output format and the addition of OCR labels, without any extra modules or complex training strategies.
- **An elegant solution to catastrophic forgetting**: Instead of utilizing indirect means such as data replay or regularization, this approach directly forces the model to retain its original capabilities through task design, which is a novel perspective.
- **High generalizability**: The core concept of SSR can be extended to any multi-capability MLLM fine-tuning scenario—explicitly maintaining the generation of legacy capabilities while learning new ones.

## Limitations & Future Work

- The paper mainly validates the approach on Chinese-English and English-German translation pairs; its generalizability to more language pairs (especially morphologically rich, low-resource languages) remains unexplored.
- SSR increases generation length (requiring OCR generation prior to translation), which incurs higher inference latency and negatively impacts efficiency on long documents.
- The cascading effect of OCR errors on translation quality is not explored—if the OCR recognition is incorrect, the erroneous information may mislead the subsequent translation.
- The method could be extended to similar situations in multimodal question answering, such as describing the image before answering the question.

## Related Work & Insights

- **vs. Pipeline Methods (OCR $\rightarrow$ NMT)**: Pipeline methods completely decouple the two steps, which causes error cascades and prevents end-to-end optimization; SSR integrates both steps under an end-to-end framework, allowing for joint optimization.
- **vs. Standard SFT**: Standard SFT only focuses on the final translation output and ignores the retention of intermediate capabilities; SSR maintains intermediate capabilities through explicit OCR supervision.
- **vs. Chain-of-Thought**: SSR can be viewed as an adaptation of CoT in multimodal translation—first "thinking" (OCR) and then "answering" (translation), leveraging intermediate steps to assist the final output.

## Rating

- Novelty: ⭐⭐⭐⭐ The self-reviewing mechanism inspired by cognitive science is both novel and concise.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive oblations and comparisons validate the efficacy of the method from multiple dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clarify of motivation and an intuitive, easy-to-understand description of the methodology.
- Value: ⭐⭐⭐½ Although the problem focuses on the relatively niche area of document image translation, the core concept of the method has broader applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation](single-to-mix_modality_alignment_with_multimodal_large_language_model_for_docume.md)
- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[CVPR 2025\] DocoPilot: Improving Multimodal Models for Document-Level Understanding](../../CVPR2025/multimodal_vlm/docopilot_improving_multimodal_models_for_document-level_understanding.md)
- [\[CVPR 2025\] Multimodal OCR: Parse Anything from Documents](../../CVPR2025/multimodal_vlm/multimodal_ocr_parse_anything_from_documents.md)

</div>

<!-- RELATED:END -->
