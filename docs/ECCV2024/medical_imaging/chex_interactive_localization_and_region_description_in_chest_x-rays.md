---
title: >-
  [Paper Note] CheX: Interactive Localization and Region Description in Chest X-rays
description: >-
  [ECCV 2024][Medical Imaging][Chest X-ray] This paper proposes ChEX, an interactive chest X-ray interpretation model that supports both text prompts and bounding box queries. Through a DETR-style prompt detector and multi-task joint training, ChEX achieves competitive performance with SOTA on 9 chest X-ray tasks while providing unique grounding interpretability and user interaction capabilities.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Chest X-ray"
  - "Report Generation"
  - "Visual Grounding"
  - "Interactive Diagnosis"
  - "Multi-task Learning"
date: 2026-05-08
content_hash: bd79b81eedc88b8b
---

# CheX: Interactive Localization and Region Description in Chest X-rays

**Conference**: ECCV 2024  
**arXiv**: [2404.15770](https://arxiv.org/abs/2404.15770)  
**Code**: [https://github.com/philip-mueller/chex](https://github.com/philip-mueller/chex)  
**Area**: Medical Imaging  
**Keywords**: Chest X-ray, Report Generation, Visual Grounding, Interactive Diagnosis, Multi-task Learning

## TL;DR

This paper proposes ChEX, an interactive chest X-ray interpretation model that supports both text prompts and bounding box queries. Through a DETR-style prompt detector and multi-task joint training, ChEX achieves competitive performance with SOTA on 9 chest X-ray tasks while providing unique grounding interpretability and user interaction capabilities.

## Background & Motivation

Although automatic report generation models for chest X-rays have progressed rapidly, they face two core obstacles in clinical applications: **lack of interpretability** (opaque model decision-making process, leaving clinicians unable to verify the basis of predictions) and **lack of interactivity** (static outputs that cannot be adjusted based on user focus).

Existing works have their respective limitations: RGRG [Tanida et al.] provides some interpretability via bounding boxes of anatomical regions but does not support text queries and focuses solely on anatomical structures. RaDialog, Med-PaLM M, etc., support textual interaction but cannot predict bounding boxes for visual localization. Although OmniFM-DR can predict bounding boxes for text prompts, it does not describe the content within the box.

**Key Challenge**: No existing model simultaneously possesses "bidirectional text/bounding-box interaction" and "visual localization interpretability".

**Key Insight**: Designing a unified multi-task architecture that integrates text prompts and bounding boxes into a query mechanism. This framework supports various tasks such as localization, classification, region description, sentence grounding, and full report generation, obtaining zero-shot generalization capability through joint multi-dataset training.

## Method

### Overall Architecture

The pipeline of ChEX consists of four stages: (1) **Image Encoder** extracts patch features of chest X-rays; (2) **Prompt Encoder** (frozen CLIP text encoder) encodes text queries into prompt tokens; (3) **Prompt Detector** (DETR-style decoder) predicts bounding boxes and ROI features based on prompt tokens and patch tokens; (4) **Sentence Generator** (GPT2-medium) independently generates text descriptions for each region based on ROI tokens.

When a user provides a bounding box query, ROI features are directly computed via Gaussian ROI Pooling, bypassing the text processing portion of the detector.

### Key Designs

1. **Prompt Detector (DETR-style Decoder)**: Predicts $M=3$ bounding boxes for each prompt token. Specifically, each prompt token is added to $M$ learnable tokens to form $Q \times M$ decoder query tokens. After processing through a 6-layer DETR decoder, an MLP is used to predict box coordinates and confidence scores. Then, box features are computed on patch features via **Gaussian ROI Pooling**, and finally, the $M$ box features are weighted averaged based on confidence to obtain the ROI token for each prompt. Design Motivation: Supporting multiple localized regions for a single query (e.g., bilateral pleural effusion) while ensuring gradient flow through random skip connections.

2. **Multi-type Prompt Tokens**: Three types of prompts are used during training: (a) **pathology tokens**—predefined pathology names (e.g., "pleural effusion"); (b) **anatomy tokens**—anatomical region names (e.g., "right lung"); (c) **sentence tokens**—individual sentences from reports. Different samples selectively use corresponding types of tokens based on available annotations. This design enables the model to learn a unified localization-description ability under different types of supervision signals.

3. **Bounding Box Query Mode**: In addition to text queries, a fraction of the batch is randomly selected to use target bounding boxes to directly compute ROI features via Gaussian ROI Pooling, skipping detection and encoding. This allows the model to support both text and bounding box queries during inference, which can also be combined for more accurate predictions.

4. **Sentence Generator**: GPT2-medium (pretrained on PubMed) is utilized to independently generate descriptions for each ROI token using P-tuning v2 as a condition. An additional 3-layer post-decoder (cross-attention to patch features) is introduced to inject global context.

### Loss & Training

- **Bounding Box Loss**: Modified DETR Hungarian matching using L1 + gIoU loss, omitting cross-entropy in favor of Focal Loss for training box confidence.
- **Pathology Classification Loss**: InfoNCE contrastive loss, pairing ROI tokens with pathology prompts (positive examples like "pleural effusion" and negative examples like "no pleural effusion" or other non-existent pathologies).
- **Text Generation Loss**: Autoregressive language modeling + contrastive learning between ROI tokens and corresponding sentences + global CLIP loss.

Training Data: MIMIC-CXR (227K images, including 29 anatomical region boxes + 53 class labels from Chest ImaGenome) + VinDr-CXR (15K images, 22 pathology box classes). Over-sampling was applied to VinDr-CXR to balance the data volume.

## Key Experimental Results

### Main Results

In a comprehensive evaluation across 9 tasks, ChEX competes with (within 1-std) or outperforms the best baseline in 8 out of 9 tasks.

| Task/Dataset | Metric | ChEX | Best Baseline | Description |
|-------------|------|------|-------------|------|
| SG/MS-CXR | mAP | **44.47** | 44.05 (SupVG) | On par with specifically trained TransVG |
| OD/NIH8 | mAP | **11.14** | 6.69 (SupOD) | Nearly 2x higher than the best supervised detector |
| OD/MS-CXR | mAP | **16.60** | 15.83 (SupOD) | Outperforms supervised and weakly-supervised detectors |
| RC/MS-CXR | AUROC | **82.33** | 76.13 (SupOD) | Gain of 8% |
| RC/CIG | wAUROC | **70.46** | 66.96 (Contrastive) | Gain of 5% |
| RE/CIG | METEOR | **10.18** | 7.88 (RGRG) | Gain of 29% |
| RE/CIG | Mac-F1-14 | **29.13** | 20.88 (RGRG) | Gain of 40% |
| RG/MIMIC-CXR | Ex-F1-14 | **58.76** | 47.6 (Med-PaLM M) | Gain of 23%, new SOTA |
| RG/MIMIC-CXR | Mic-F1-14 | 52.32 | 55.7 (MAIRA-1) | MAIRA-1 is a model 7x larger |

### Ablation Study

| Configuration | Key Findings | Description |
|------|---------|------|
| Pathology tokens only | Good OD performance but severe decline in RC/RE | Anatomy tokens are crucial for region-based tasks |
| Anatomy tokens only | Optimal RC/RE but drop in OD/SG | Pathology tokens are indispensable for localization tasks |
| Without sentence tokens | Slight improvement in localization, drop in text generation | Sentence tokens primarily serve generation quality |
| Without bounding box supervision | Decline across all tasks | Bounding box supervision is the most critical factor |
| Without contrastive learning | Decrease in OD and some RC/RE | Contrastive learning enhances understanding of pathology regions |
| Report-level generation (non-region level) | Mic-F1-14 decreases by about 3-5% | Region-level sentence-by-sentence generation is key to ChEX's strong performance |

### Key Findings

- **Interactive capability**: Adding coarse-grained region prompts (e.g., "left lung") significantly improves localization quality; fine-grained prompts (e.g., "left upper lung") bring further minor improvements.
- **Directional guidance**: When a query points to the contralateral lung without pathology, the model correctly shifts focus to that region (instead of always pointing to the pathology), indicating that the model understands user intent.
- **Text + box combined query**: Providing both text and bounding boxes simultaneously yields the best prediction accuracy.
- **Customizable Prompt Set**: Using different prompt sets balances precision and recall, with Mic-F1-14 ranging from 50.08 to 52.37.

## Highlights & Insights

- **Unique grounding**: The only medical imaging model that simultaneously supports "text prompt -> box + description" and "box query -> description".
- **Small model, big capability**: ChEX has only 1/10 the parameter size of Med-PaLM M and 1/7 of MAIRA-1, yet excels or competes in most tasks.
- **Multi-dataset joint training**: A single architecture seamlessly integrates three heterogeneous supervisions: bounding box annotations, classification labels, and report texts.
- Reports outputting with bounding boxes can directly assist radiologists in rapid verification.

## Limitations & Future Work

- Text queries are limited to regional prompts or pathology names, not supporting complex reasoning questions (e.g., "compare left and right lungs").
- Answers are based on report sentences, which may introduce hallucinations regarding comparison with prior studies (as only a single image is used).
- Future work could combine instruction tuning or LLMs to enhance conversational capabilities.
- Systematic user experience evaluation by radiologists is lacking.

## Related Work & Insights

- Compared with RGRG, ChEX expands token types (pathology + sentence) and contrastive learning, leading to comprehensive performance improvements.
- The successful migration from DETR to medical applications demonstrates that region-centric object detection paradigms possess natural advantages for medical image understanding.
- The spillover effects of multi-task training (e.g., report generation benefiting from localization tasks) are worth validating in more medical scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ First unified model to achieve bidirectional interaction + visual grounding in medical imaging.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 tasks, multiple datasets, and thorough ablation and interaction analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich diagrams, and intuitive interactive cases.
- Value: ⭐⭐⭐⭐ Provides a practical and scalable infrastructure for interactive diagnosis in medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Temporal Inversion for Learning Interval Change in Chest X-Rays](../../CVPR2026/medical_imaging/temporal_inversion_for_learning_interval_change_in_chest_x-rays.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](../../NeurIPS2025/medical_imaging/cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[CVPR 2026\] Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset](../../CVPR2026/medical_imaging/instruction-guided_lesion_segmentation_for_chest_x-rays_with_automatically_gener.md)
- [\[ICLR 2026\] Learning Self-Critiquing Mechanisms for Region-Guided Chest X-Ray Report Generation](../../ICLR2026/medical_imaging/learning_self-critiquing_mechanisms_for_region-guided_chest_x-ray_report_generat.md)
- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)

</div>

<!-- RELATED:END -->
