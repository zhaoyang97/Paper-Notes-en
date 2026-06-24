---
title: >-
  [Paper Note] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation
description: >-
  [ACL 2025 (Main)][Multimodal VLM][Document Image Machine Translation] This paper proposes M4Doc, a document image machine translation framework based on "single-to-mix modality alignment." During the training phase, it leverages the joint vision-language representation of Multimodal Large Language Models (MLLMs) to enhance a lightweight image encoder. During inference, the MLLM is discarded to maintain efficiency. This approach achieves significant translation quality improve…
tags:
  - "ACL 2025 (Main)"
  - "Multimodal VLM"
  - "Document Image Machine Translation"
  - "Modality Alignment"
  - "Multimodal Large Language Model"
  - "Knowledge Distillation"
  - "Cross-Domain Generalization"
date: 2026-05-08
content_hash: a71c2c9ca0b93852
---

# Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation

**Conference**: ACL 2025 (Main)  
**arXiv**: [2507.07572](https://arxiv.org/abs/2507.07572)  
**Code**: None  
**Area**: Multimodal VLM / Document Translation  
**Keywords**: Document Image Machine Translation, Modality Alignment, Multimodal Large Language Model, Knowledge Distillation, Cross-Domain Generalization

## TL;DR

This paper proposes M4Doc, a document image machine translation framework based on "single-to-mix modality alignment." During the training phase, it leverages the joint vision-language representation of Multimodal Large Language Models (MLLMs) to enhance a lightweight image encoder. During inference, the MLLM is discarded to maintain efficiency. This approach achieves significant translation quality improvements in cross-domain generalization and complex document scenarios.

## Background & Motivation

**Background**: Document Image Machine Translation (DIMT) aims to directly translate text within document images without explicit intermediate OCR steps. Such end-to-end approaches avoid the cascading propagation of OCR errors but face two major challenges: limited training data and complex interaction between vision and text information. Existing DIMT models are usually based on CNN or ViT encoders to extract image features and then generate translations through a Transformer decoder.

**Limitations of Prior Work**: The image encoders of existing DIMT models only learn visual features and lack a deep semantic understanding of text. When encountering document styles outside the training domain (different fonts, layouts, language pairs), generalization performance drops significantly. Although multimodal large language models (such as InternVL, Qwen-VL, etc.) exhibit outstanding performance in document understanding, directly employing them for DIMT tasks is computationally too expensive for large-scale deployment.

**Key Challenge**: MLLMs possess powerful joint vision-text understanding capabilities but suffer from prohibitive computational costs, whereas lightweight DIMT models are highly efficient but lack deep multimodal knowledge. How to let lightweight models "borrow" the capabilities of MLLMs is the key to improving DIMT performance.

**Goal**: To design a framework that leverages the multimodal representations of an MLLM during the training phase to enhance the encoding capability of a lightweight DIMT model, while requiring no dependency on the MLLM during inference to maintain computational efficiency.

**Key Insight**: The authors observe that the intermediate representations of MLLMs implicitly contain rich vision-text correlation knowledge (since MLLMs have been pre-trained on large-scale document data). This knowledge can be "injected" into a lightweight encoder through alignment learning. This is similar to knowledge distillation but aligns intermediate representation spaces rather than distilling output distributions.

**Core Idea**: Propose single-to-mix modality alignment—aligning the representation space of a lightweight encoder (which only processes images) with the joint representation space of an MLLM processing mixed "image + text" inputs. This forces the lightweight encoder to generate features infused with text semantics even when only looking at an image.

## Method

### Overall Architecture

M4Doc consists of three core components: (1) a pre-trained MLLM teacher model that receives document images and corresponding text as a mixed input to generate multimodal representations; (2) a lightweight image encoder (student) that only receives document images as input; and (3) an alignment module that aligns the student encoder's output with the teacher MLLM's multimodal representations during training. During inference, only the student encoder and the Transformer decoder are retained, completely discarding the MLLM.

### Key Designs

1. **Multimodal Representation Extraction from MLLM Teacher**:

    - **Function**: Provides the "gold standard" representation containing both visual and textual semantics.
    - **Mechanism**: It feeds the document image and the corresponding source language text simultaneously into a pre-trained MLLM (e.g., based on InternVL or similar architectures) and extracts the hidden states of its intermediate layers as teacher representations. These representations encode image layout information, textual content, and fine-grained correspondences between them. Since the MLLM has been pre-trained on large-scale document data, its representations contain rich cross-modal correlation knowledge.
    - **Design Motivation**: Directly learning vision-text alignment with an image encoder is difficult (requiring massive parallel data), while MLLMs have already learned this alignment and can act as a "knowledge source" to transfer it to the lightweight model.

2. **Single-to-Mix Modality Alignment Module**:

    - **Function**: Aligns the representation space of the image-only encoder with the mixed-modality representation space of the MLLM.
    - **Mechanism**: A projection layer (project head) is designed to map the output of the image encoder into the same dimensional space as the MLLM representation. During training, the two spaces are aligned using MSE or cosine similarity loss. The key innovation lies in aligning an asymmetric mapping of "single modality (image) -> mixed modality (image + text)", rather than traditional unimodal alignment. This forces the image encoder to learn to "complete" the missing textual semantic information from the image.
    - **Design Motivation**: Traditional knowledge distillation aligns output distributions, but the output of DIMT is translated text, and directly distilling outputs has limited effects in sequence-to-sequence tasks. Aligning intermediate representations is more flexible and can exploit the structured knowledge in MLLM representations.

3. **MLLM Bypass Design during Inference**:

    - **Function**: Maintains inference efficiency.
    - **Mechanism**: After training is completed, the MLLM teacher is completely removed from the inference pipeline, leaving only the aligned lightweight image encoder and the Transformer translation decoder. Due to the alignment learning in the training phase, the image encoder has already "internalized" the multimodal knowledge of the MLLM and does not need to access the MLLM during inference.
    - **Design Motivation**: MLLMs typically have billions of parameters, and their inference cost is tens to hundreds of times higher than that of lightweight DIMT models. The bypass design keeps the size and speed of the finally deployed model comparable to baseline models that do not use MLLMs.

### Loss & Training

Training uses a weighted combination of two loss functions: (1) translation loss—standard cross-entropy loss that supervises the Transformer decoder to generate correct translations; (2) alignment loss—distance loss (such as MSE or cosine similarity loss) between the encoder output and the MLLM teacher representations, where a weighting coefficient balances the two objectives. The training is divided into a warmup phase (freezing the encoder, training only the alignment projection layer) and a joint fine-tuning phase.

## Key Experimental Results

### Main Results

| Dataset/Direction | Metric | Ours | Baseline (No Alignment) | Prev. SOTA | Gain |
|-----------|------|-------|-------------|----------|------|
| In-domain (Zh→En) | BLEU | **Optimal** | Baseline | Competitive | +2-3 BLEU |
| Cross-domain (Zh→En) | BLEU | **Significant Gain** | Significant Drop | Moderate | +5-8 BLEU |
| In-domain (En→De) | BLEU | **Optimal** | Baseline | Competitive | +1-2 BLEU |
| Complex Layout Docs | BLEU | **Significant Gain** | Drastic Drop | Moderate | Significant Gain |

### Ablation Study

| Configuration | Cross-domain BLEU | Description |
|------|----------|------|
| Full M4Doc | **Optimal** | Full alignment framework |
| w/o alignment loss | Significant Drop | Degenerates to a normal DIMT model |
| Image-image alignment only | Drop | Unimodal alignment is less effective than single-to-mix |
| Text-text alignment only | Drop | Lack of visual information transfer |
| Using smaller MLLM | Slight Drop | Stronger teacher models yield better results |
| Frozen encoder | Significant Drop | The encoder needs joint fine-tuning to fully absorb knowledge |

### Key Findings

- **Cross-domain generalization is the biggest highlight**: The improvement of M4Doc on cross-domain tests (where training and testing document types differ) is much larger than that on in-domain tests, with BLEU score gains reaching 5-8 points. This indicates that the multimodal knowledge from MLLM effectively improves the robustness of the encoder.
- **Single-to-mix outperforms unimodal alignment**: Aligning the single-image representation to the mixed image+text representation of the MLLM works better than aligning it to the pure image representation of the MLLM, validating the core hypothesis of "forcing the encoder to learn to complete textual semantics."
- **Zero extra cost at inference**: Since the MLLM is completely discarded during inference, the inference speed of M4Doc is identical to that of the baseline model.
- **Complex document scenarios benefit the most**: In documents containing charts, formulas, special fonts, and other complex elements, the advantages of M4Doc are even more pronounced.

## Highlights & Insights

- **A paradigm of distilling during training and discarding during inference**: This pattern of "borrowing large models during training and keeping it lightweight during inference" is highly practical and can be directly migrated to other multimodal tasks requiring deployment efficiency (such as document QA, visual translation, etc.).
- **Asymmetric modality alignment**: The innovative idea of aligning "fewer modalities" to "more modalities" essentially forces the model to learn to reason richer semantics from limited information. This concept could also be useful in tasks like visual assistance for the blind or low-resolution image understanding.
- **MLLMs as general knowledge sources**: Instead of directly using MLLMs to perform tasks, leveraging their representations to enhance specialized models represents an important direction for the application of large models.

## Limitations & Future Work

- The alignment effect depends on the quality of the MLLM teacher; if the MLLM provides poor representations on certain document types, the student cannot benefit either.
- The training phase requires running the MLLM to extract teacher representations, which increases training costs (though they can be pre-computed).
- Current experiments are mainly conducted on high-resource language pairs such as Chinese-English and English-German; the performance on low-resource languages remains unknown.
- Only the encoder-decoder architecture of DIMT models has been validated; whether decoder-only architectures can also benefit has not been explored.
- Future work can extend this to more complex scenarios like handwritten documents, scanned documents, and multilingual mixed documents.

## Related Work & Insights

- **vs Traditional DIMT Methods**: Traditional methods only use image encoders to extract visual features, lacking textual semantic understanding. M4Doc compensates for this deficiency through MLLM alignment, showing distinct advantages especially in cross-domain scenarios.
- **vs Direct Usage of MLLMs**: Although directly using MLLMs for DIMT yields good results, the inference cost is too high. M4Doc achieves a compromise of "comparable performance, equal cost" through knowledge transfer during the training phase.
- **vs Traditional Knowledge Distillation**: Traditional distillation aligns output distributions (soft labels), whereas M4Doc aligns intermediate representation spaces, which is more effective in seq2seq tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Single-to-mix modality alignment is an inspiring new concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic experimental design with multiple language pairs, cross-domain tests, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear framework description and logical derivation of motivations.
- Value: ⭐⭐⭐⭐ The proposed paradigm is widely transferable and has practical value for the Document AI field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency](improving_mllms_document_image_machine_translation_via_synchronously_self-review.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](../../CVPR2025/multimodal_vlm/post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[ACL 2025\] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus](moscar_a_large-scale_multilingual_and_multimodal_document-level_corpus.md)
- [\[ACL 2025\] AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models](alignmmbench_evaluating_chinese_multimodal_alignment_in_large_vision-language_mo.md)
- [\[CVPR 2026\] Towards Dynamic Modality Alignment in Multimodal Continual Learning](../../CVPR2026/multimodal_vlm/towards_dynamic_modality_alignment_in_multimodal_continual_learning.md)

</div>

<!-- RELATED:END -->
