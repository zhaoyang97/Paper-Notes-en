---
title: >-
  [Paper Note] MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures
description: >-
  [CVPR 2026][Multimodal VLM][Chemical Structure Recognition] MarkushGrapher-2 proposes an end-to-end multimodal chemical structure recognition model that jointly encodes image, text…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Chemical Structure Recognition"
  - "Markush Structures"
  - "Multimodal Encoding"
  - "Patent Document Analysis"
  - "OCR"
date: 2026-05-08
content_hash: 91555473cbcf327e
---

# MarkushGrapher-2: End-to-end Multimodal Recognition of Chemical Structures

**Conference**: CVPR 2026
**arXiv**: [2603.28550](https://arxiv.org/abs/2603.28550)  
**Code**: [https://github.com/DS4SD/MarkushGrapher](https://github.com/DS4SD/MarkushGrapher)  
**Area**: Multimodal VLM / Document Understanding
**Keywords**: Chemical Structure Recognition, Markush Structures, Multimodal Encoding, Patent Document Analysis, OCR

## TL;DR

MarkushGrapher-2 proposes an end-to-end multimodal chemical structure recognition model that jointly encodes image, text, and layout information via a dedicated chemical OCR module. Combined with a two-stage training strategy (first adapting to OCSR features, then integrating multimodal encoding), the model substantially outperforms existing methods on Markush structure recognition (M2S accuracy 56% vs. 38%), while remaining competitive on standard molecular structure recognition.

## Background & Motivation

1. **Background**: Automated extraction of chemical structures from documents is fundamental to large-scale chemical literature analysis. Existing methods handle molecular structures in images (OCSR) or chemical named entities in text separately, but remain inadequate for Markush structures — multimodal descriptions that combine both image and text.

2. **Limitations of Prior Work**: Markush structures are critical in patent analysis (for prior art search, freedom-to-operate evaluation, etc.), yet are currently indexed only by two manually annotated proprietary databases: MARPAT and DWPIM. The predecessor MarkushGrapher-1 requires pre-annotated OCR output as input (precluding end-to-end processing), and its visual recognition accuracy leaves room for improvement. General-purpose VLMs (GPT-5, DeepSeek-OCR) perform poorly on Markush recognition (GPT-5 achieves only 3% on M2S).

3. **Key Challenge**: The visual style of Markush structures varies enormously across patent offices and publication years; textual descriptions lack standardization and include conditional/recursive specifications; and large-scale real-world training data are scarce.

4. **Goal**: Build a unified end-to-end model capable of recognizing both standard molecules and multimodal Markush structures.

5. **Key Insight**: Exploit complementary dual encoders — an OCSR visual encoder and a VTL multimodal encoder — together with a dedicated chemical OCR module and a two-stage training strategy.

6. **Core Idea**: A dual-encoder pipeline fuses visual structural features with multimodal text-layout features to enable end-to-end recognition of chemical Markush structures.

## Method

### Overall Architecture

Given a chemical structure image as input, the model outputs a CXSMILES representation (a graph-based description of the Markush scaffold) and a substituent table (molecular fragments that may substitute variable groups). The overall architecture is encoder–decoder:

1. **Pipeline 1**: Image → Visual Encoder (MolScribe's Swin-B ViT, frozen) → MLP Projector → Visual Embedding $e_1$
2. **Pipeline 2**: Image → ChemicalOCR → Text + Bounding Boxes → VTL Encoder (T5-base) → Multimodal Embedding $e_2$
3. **Fusion**: $e_1$ and $e_2$ are concatenated → Text Decoder → Autoregressive generation of CXSMILES + Substituent Table

### Key Designs

1. **ChemicalOCR Module**:

    - **Function**: Extracts character-level text and bounding boxes from chemical structure images to enable end-to-end processing.
    - **Mechanism**: Fine-tuned from Smoldocling (a lightweight 256M-parameter VLM). Pre-trained on 235k synthetic chemical structures (with automatic OCR annotations), then fine-tuned on 7k manually annotated IP5 patent document chemical structures. The extracted text and bounding boxes provide text and layout modality inputs to the VTL encoder.
    - **Design Motivation**: General-purpose OCR models (PaddleOCR, EasyOCR) perform extremely poorly on chemical images (F1 of 7.7/10.2 vs. ChemicalOCR's 87.2), commonly misidentifying chemical bonds as minus signs or equal signs and failing to handle chemical abbreviations. Chemical OCR is critical for accurately recognizing Markush features such as brackets and indices.

2. **Dual-Encoder Fusion (OCSR + VTL)**:

    - **Function**: Complementarily captures visual structural features and multimodal text-layout features.
    - **Mechanism**: The OCSR visual encoder (Swin-B ViT from MolScribe) excels at molecular scaffold recognition but cannot handle Markush features; the VTL encoder (T5-base, following the UDOP fusion paradigm) aligns and fuses spatially co-located visual and text tokens, excelling at Markush features but weaker on molecular structure. After projection, both embeddings are concatenated and fed into the text decoder. Ablation studies confirm: Pipeline 1 alone achieves 89.1% on USPTO SMILES but only 8% on M2S; Pipeline 2 alone achieves 39% on M2S but only 46% on USPTO; the fused model achieves competitive performance on both.
    - **Design Motivation**: Markush structures simultaneously contain visual information (molecular scaffold) and textual information (variable group definitions), requiring complementary encoding for complete recognition.

3. **Two-Stage Training Strategy**:

    - **Function**: Effectively integrates the two encoders without disrupting pre-trained OCSR features.
    - **Mechanism**: *Phase 1 (Adaptation)*: The visual encoder is frozen; the projector and text decoder are trained for standard SMILES prediction (243k real samples, 3 epochs), allowing the decoder to adapt to the OCSR feature space. *Phase 2 (Fusion)*: The visual encoder and projector are frozen; the OCR and VTL encoders are introduced; the VTL encoder and text decoder are trained end-to-end for CXSMILES + substituent table prediction (235k synthetic + 145k real, 2 epochs).
    - **Design Motivation**: Direct single-stage training (fusion only) yields 44% M2S accuracy; the two-stage strategy improves this to 50% (+6%). Freezing the OCSR encoder preserves the original visual features, allowing the VTL encoder to focus on learning the complementary information required for Markush features.

### Loss & Training

The model employs standard autoregressive cross-entropy loss throughout. Phase 1 trains on SMILES prediction; Phase 2 trains on CXSMILES + substituent table prediction. The total model has 831M parameters, of which 744M are trainable. Training is conducted on NVIDIA A100 GPUs.

## Key Experimental Results

### Main Results

| Method | M2S (CXSMILES A) | USPTO-M A | WildMol-M A | IP5-M A |
|--------|-------------------|-----------|-------------|---------|
| MolParser-Base (Image) | 39 | 30 | 38.1 | 47.7 |
| MolScribe (Image) | 21 | 7 | 28.1 | 22.3 |
| GPT-5 (Multimodal) | 3 | — | — | — |
| DeepSeek-OCR | 0 | 0 | 1.9 | 0.0 |
| MarkushGrapher-1 | 38 | 32 | — | — |
| **MarkushGrapher-2** | **56** | **55** | **48.0** | **53.7** |

### Ablation Study

| Configuration | M2S A | M2S A_InChIKey | USPTO-M A | IP5-M A |
|---------------|-------|----------------|-----------|---------|
| Without OCR Input | 4 | 39 | 3 | 15.4 |
| With OCR Input | 56 | 80 | 55 | 53.7 |
| Single-Stage Training (Fusion only) | 44 | 53 | — | — |
| Two-Stage Training (Adapt + Fusion) | 50 | 68 | — | — |

### Key Findings

- The OCR module is the single most critical component: removing it causes M2S accuracy to plummet from 56% to 4%, as textual information such as brackets and indices is essential for predicting Markush features.
- ChemicalOCR substantially outperforms general-purpose OCR: F1 = 86.5 on IP5-M vs. 1.9 for PaddleOCR and 18.4 for EasyOCR.
- General-purpose VLMs completely fail on Markush recognition: GPT-5 achieves only 3%, DeepSeek-OCR achieves 0%.
- Two-stage training improves M2S scaffold accuracy by 15 percentage points over single-stage training (53% → 68%).
- The model remains competitive on standard molecular recognition (OCSR): 96.6% on UOB (best overall) and 68.4% on WildMol.

## Highlights & Insights

- **Complementary Dual-Encoder Design**: The model separately leverages the molecular scaffold recognition capability of the visual encoder and the multimodal fusion capability of the VTL encoder — a generalizable multimodal architecture design pattern. This approach can be transferred to other tasks requiring simultaneous processing of structured visual and textual information (e.g., table understanding, circuit diagram analysis).
- **USPTO-MOL-M Data Generation Pipeline**: A pipeline for automatically extracting real Markush training data from USPTO MOL files addresses the scarcity of annotated data. The strategy of leveraging existing structured data to automatically generate training samples is broadly applicable.
- **Domain-Specialized OCR**: General-purpose OCR is entirely unusable on chemical images, yet a high-accuracy domain OCR model can be trained with only 7k manually annotated samples plus 235k synthetic data — demonstrating that domain adaptation remains highly important in OCR.

## Limitations & Future Work

- **Overall accuracy remains modest**: 56% on M2S and 53.7% on IP5-M fall short of practical deployment requirements, particularly for substituent table prediction (M2S table accuracy is only 22%).
- **Training data remains predominantly synthetic**: The combination of 235k synthetic and 145k real samples may introduce distribution gaps relative to real-world patent documents.
- **Cascading OCR errors**: Errors in the OCR module directly propagate to downstream Markush recognition, and cascading errors may be amplified for complex structures.
- **2D structures only**: 3D molecular conformation information is not addressed.
- **Inference efficiency not discussed**: Whether the 831M-parameter model can meet the throughput requirements of large-scale patent scanning remains unexamined.

## Related Work & Insights

- **vs. MarkushGrapher-1**: The predecessor requires pre-annotated OCR input; this work achieves end-to-end processing and improves accuracy from 38% to 56%.
- **vs. MolParser**: MolParser processes only the image modality and supports only a limited set of Markush features; this work jointly processes image and text and provides more comprehensive coverage.
- **vs. GPT-5/DeepSeek-OCR**: The complete failure of general-purpose VLMs on this task demonstrates that chemical structure recognition still requires domain-specialized methods.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of dual-encoder fusion, two-stage training, and specialized OCR is novel, though the individual components are not themselves entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple benchmarks, diverse baselines, detailed ablation studies, and the release of a new benchmark (IP5-M).
- Writing Quality: ⭐⭐⭐⭐ The paper is clearly structured with sufficient background on chemical context, though some sections are verbose.
- Value: ⭐⭐⭐⭐ Fills a critical gap in end-to-end Markush recognition with significant practical value for cheminformatics and patent analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition](../../ACL2026/multimodal_vlm/e2e-gmner_end-to-end_generative_grounded_multimodal_named_entity_recognition.md)
- [\[AAAI 2026\] SpeakerLM: End-to-End Versatile Speaker Diarization and Recognition with Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/speakerlm_end-to-end_versatile_speaker_diarization_and_recognition_with_multimod.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](../../ICLR2026/multimodal_vlm/webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[CVPR 2026\] Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models](taxonomy-aware_representation_alignment_for_hierarchical_visual_recognition_with.md)
- [\[CVPR 2026\] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)

</div>

<!-- RELATED:END -->
