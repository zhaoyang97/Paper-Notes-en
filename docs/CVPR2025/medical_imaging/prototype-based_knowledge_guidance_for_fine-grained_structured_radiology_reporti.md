---
title: >-
  [Paper Note] Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting
description: >-
  [CVPR2025][Medical Imaging][structured radiology reporting] ProtoSR proposes to mine template-aligned prototype knowledge bases from large-scale free-text radiology reports and inject them into structured report prediction through a prototype-conditioned late-fusion residual module, achieving SOTA on the Rad-ReStruct benchmark, particularly gaining a 72.1% relative improvement on fine-grained attribute questions (L3).
tags:
  - "CVPR2025"
  - "Medical Imaging"
  - "structured radiology reporting"
  - "prototype learning"
  - "knowledge base"
  - "free-text mining"
  - "late fusion"
  - "VQA"
date: 2026-05-08
content_hash: 94f3b5d828b6375c
---

# Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting

**Conference**: CVPR2025  
**arXiv**: [2603.11938](https://arxiv.org/abs/2603.11938)  
**Code**: To be released  
**Area**: Medical Imaging  
**Keywords**: structured radiology reporting, prototype learning, knowledge base, free-text mining, late fusion, VQA

## TL;DR

ProtoSR proposes to mine template-aligned prototype knowledge bases from large-scale free-text radiology reports and inject them into structured report prediction through a prototype-conditioned late-fusion residual module, achieving SOTA on the Rad-ReStruct benchmark, particularly gaining a 72.1% relative improvement on fine-grained attribute questions (L3).

## Background & Motivation

- **Value of Structured Reporting (SR)**: Compared to free-text, structured reporting uses predefined fields and standardized answer options to improve consistency and completeness, while supporting quality monitoring and downstream analysis.
- **Challenges of Automated SR**: Fine-grained templates contain a large number of rare attributes (e.g., location, appearance, severity), whereas structured datasets (such as Rad-ReStruct with only 3,597 samples) are limited in scale, leading to sparse supervision signals for long-tail attributes.
- **Abundance of Free-Text Reports**: Public datasets like MIMIC-CXR contain over 220k paired chest X-rays and free-text reports, which implicitly contain rich image-associated fine-grained information.
- **Gap in Existing Methods**: General medical VLMs (such as MedGemma and CheXagent) still lag behind specialized models on SR tasks; knowledge integration methods mostly target unstructured outputs and fail to address fine-grained discrete decision-making.
- **Key Motivation**: How to transform abundant but unformatted free-text reports into knowledge aligned with SR templates and effectively inject it into structured prediction pipelines.

## Method

### 1. Knowledge Base Construction

Mapping large-scale free-text report corpora (MIMIC-CXR) to the structured template label space:

- **Terminology Expansion**: Uses a zero-shot LLM to generate synonyms, abbreviations, and alternative expressions for each template label, constructing a mapping dictionary from variants to canonical labels.
- **Template-Constrained Extraction**: For each report, queries the LLM hierarchically to determine the presence of findings; if present, the attribute values are extracted. Constrained decoding is used to ensure only valid answers are output.
- **Post-processing and Prototype Construction**: Employs rule-based filtering for denoising and hierarchical consistency checks (removing parent labels if no sub-label is supported). For each label, a maximum of K=5 images are evenly sampled, and their features are extracted using an image encoder and aggregated into a prototype vector via element-wise max pooling.
- **Extraction Model Selection**: Comparing Llama 3.1 8B / Mistral 8B / Qwen2.5 7B, Qwen2.5-7B-Instruct + terminology expansion yields the best performance.

### 2. Knowledge-Enhanced Late Fusion Architecture (ProtoSR)

- **Base Model**: Follows the hierarchical VQA architecture of Rad-ReStruct: EfficientNet-B5 (image) + RadBERT (text) → Transformer fusion → classification head outputting base logits.
- **Prototype-Conditioned Knowledge Branch**:
  1. Project the fused representation S and prototype embeddings P into a shared space.
  2. Calculate the cosine similarity weight α (considering only prototypes corresponding to legal answers of the current question).
  3. Perform weighted aggregation to obtain the prototype feature vector v and answer support vector u.
  4. Concatenate [v; u] and feed it into an MLP to generate the support bias $b_{\text{sup}}$.
- **Late Fusion**: $z_{\text{final}} = z_{\text{base}} + s \odot b_{\text{sup}}$, where s is a learnable per-answer dimension scaling vector.
- **Design Advantages**: Preserves the decision path of the base model, performing targeted corrections only when prototype evidence is meaningful; when no prototype is available, the bias is zero and does not cause interference.
- **Prototype Alignment**: Uses an EMA copy of the image encoder to refresh prototype embeddings every 10k steps, maintaining alignment during training.

### 3. Training Details
- End-to-end training using the same multi-label objective as Rad-ReStruct.
- Single RTX 3090 (24GB), 34 epochs, learning rate of 1e-5, batch size of 8 + gradient accumulation of 4.

## Key Experimental Results

### Rad-ReStruct Benchmark Performance

| Method | Overall F1 | L1-F1 | L2-F1 | L3-F1 |
|------|-----------|-------|-------|-------|
| MedGemma | 26.8 | 38.2 | 63.4 | 2.8 |
| CheXagent | 32.4 | 62.1 | 69.8 | 6.2 |
| hi-VQA (Rad-ReStruct) | 32.0 | 64.6 | 71.6 | 4.1 |
| Context-VQA | 32.9 | 67.2 | 71.8 | 3.2 |
| **ProtoSR** | **34.4** | **66.2** | **72.8** | **7.4** |

- On L3 (fine-grained attributes), ProtoSR reaches 7.4, achieving a **72.1%** relative improvement over the baseline (4.3).
- Overall F1 increases by an absolute value of 1.5 compared to Context-VQA.

### Ablation Study

| Variant | Overall F1 | L3-F1 |
|------|-----------|-------|
| No knowledge | 32.5 | 4.3 |
| Early Fusion | 32.5 | 4.3 |
| Random Prototype | 32.7 | 4.4 |
| **ProtoSR** | **34.4** | **7.4** |

- Both early fusion and random prototypes are ineffective, proving that the gains stem from the prototype content rather than the extra parameter capacity.
- The performance of random prototypes is equivalent to the no-knowledge baseline, showing that the model learns to ignore uninformative signals.

### Knowledge Base Coverage

| Level | Total Categories | Covered Categories | Coverage Rate |
|------|--------|---------|--------|
| L1 | 56 | 56 | 100% |
| L2 | 326 | 314 | 96% |
| L3 | 1167 | 966 | 82% |

## Highlights & Insights

1. **Bridge from Free-Text to Structured Knowledge**: Systematically converts unstructured reports into a template-aligned prototype knowledge base, addressing the scarcity of SR data.
2. **Elegant Late Fusion Design**: The residual + learnable scaling design is conservative yet effective, preserving the patterns learned by the base model while performing targeted corrections.
3. **Breakthrough in L3 Fine-Grained Attributes**: The 72.1% relative improvement is concentrated on the most challenging long-tail fine-grained attributes, which are precisely the areas where structured reporting most needs improvement.
4. **Thorough Ablation Validation**: Ablations on early fusion, random prototypes, and no-knowledge clearly demonstrate the effectiveness of the proposed method.
5. **Lightweight Extension**: The knowledge branch adds only an MLP and a scaling vector, introducing almost no inference overhead.
6. **Importance of Terminology Expansion**: Simple LLM-based synonym expansion brings substantial improvements across all extraction models.

## Limitations & Future Work

1. **Absolute Performance remains low**: Although L3-F1 is significantly improved, it is still only 7.4, leaving a huge gap for practical application; the Overall F1 of 34.4 is also low.
2. **Verified on a Single Benchmark Only**: Tested only on Rad-ReStruct (ChestXR), with no verification of generalization to other modalities or anatomical regions.
3. **Quality of Knowledge Base Extraction Limited by LLMs**: The extraction accuracy of 7B models is limited (L3-F1 around 80.6%), and noise can propagate into the prototypes.
4. **Simple Prototype Construction**: Aggregating K=5 samples via max pooling might be insufficient to represent the visual diversity of labels, and scalability to larger scales might be limited.
5. **Suboptimal Report Accuracy**: The report-level accuracy of ProtoSR (36.6%) is lower than that of Context-VQA (39.7%), indicating that global consistency needs improvement.
6. **Missing Comparison with the Latest VLMs**: Direct application of MedGemma/CheXagent to SR is not entirely fair, and comparisons with more suitable multi-turn VLMs are missing.

## Rating
- Novelty: ⭐⭐⭐⭐ (The strategy of mining prototype knowledge from free-text and injecting it into structured prediction is highly inspiring)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Thorough ablations, but limited to a single benchmark)
- Writing Quality: ⭐⭐⭐⭐ (The method is clearly described, and the diagrams are well-designed)
- Value: ⭐⭐⭐⭐ (Provides practical guidance for automated structured reporting)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Unleashing Video Language Models for Fine-grained HRCT Report Generation](unleashing_video_language_models_for_fine-grained_hrct_report_generation.md)
- [\[CVPR 2025\] Thin-Shell-SfT: Fine-Grained Monocular Non-Rigid 3D Surface Tracking with Neural Deformation Fields](thin-shell-sft_fine-grained_monocular_non-rigid_3d_surface_tracking_with_neural_.md)
- [\[AAAI 2026\] FaNe: Towards Fine-Grained Cross-Modal Contrast with False-Negative Reduction and Text-Conditioned Sparse Attention](../../AAAI2026/medical_imaging/fane_towards_fine-grained_cross-modal_contrast_with_false-negative_reduction_and.md)
- [\[ECCV 2024\] A Rotation-Invariant Texture ViT for Fine-Grained Recognition of Esophageal Cancer Endoscopic Ultrasound Images](../../ECCV2024/medical_imaging/a_rotation-invariant_texture_vit_for_fine-grained_recognition_of_esophageal_canc.md)
- [\[ICLR 2026\] CerebraGloss: Instruction-Tuning a Large Vision-Language Model for Fine-Grained Clinical EEG Interpretation](../../ICLR2026/medical_imaging/cerebragloss_instruction-tuning_a_large_vision-language_model_for_fine-grained_c.md)

</div>

<!-- RELATED:END -->
