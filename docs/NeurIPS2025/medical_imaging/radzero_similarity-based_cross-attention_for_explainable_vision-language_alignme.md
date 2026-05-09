---
title: >-
  [Paper Note] RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray
description: >-
  [NeurIPS 2025][Medical Imaging][Vision-language alignment] This paper proposes RadZero, a framework centered on VL-CABS (Vision-Language Cross-Attention Based on Similarity), enabling explainable and fine-grained vision-language alignment on chest X-rays with unified support for zero-shot classification, localization, and segmentation.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Vision-language alignment
  - chest X-ray
  - zero-shot
  - explainability
  - cross-attention
date: 2026-05-08
content_hash: 9152623c0c49c49f
---

# RadZero: Similarity-Based Cross-Attention for Explainable Vision-Language Alignment in Chest X-ray

**Conference**: NeurIPS 2025

**arXiv**: [2504.07416](https://arxiv.org/abs/2504.07416)

**Code**: [GitHub](https://github.com/deepnoid-ai/RadZero)

**Area**: Medical Imaging

**Keywords**: Vision-language alignment, chest X-ray, zero-shot, explainability, cross-attention

## TL;DR

This paper proposes RadZero, a framework centered on VL-CABS (Vision-Language Cross-Attention Based on Similarity), enabling explainable and fine-grained vision-language alignment on chest X-rays with unified support for zero-shot classification, localization, and segmentation.

## Background & Motivation

### Limitations of Prior Work

**Background**: Multimodal vision-language (VL) alignment has seen significant progress in radiology, yet existing methods exhibit critical shortcomings:

**Underutilization of reports**: Radiology reports have complex structures, and current methods struggle to exploit their fine-grained semantic information effectively.

**Poor explainability**: Traditional attention probability visualizations offer limited interpretability, which is insufficient for clinical adoption.

**Weak multi-task capability**: Separate models are typically required for classification, localization, and segmentation.

**Insufficient zero-shot generalization**: Generalization to unseen disease categories remains limited.

## Method

### Overall Architecture

RadZero comprises three novel components: (1) the VL-CABS cross-attention mechanism; (2) LLM-driven semantic sentence extraction; and (3) multi-positive contrastive training.

### Key Designs

**1. VL-CABS (Vision-Language Cross-Attention Based on Similarity)**

- **Core Idea**: Computes **similarity** between text embeddings and local image patch features, rather than attention probabilities.
- Similarity computation: $S(t, p) = \frac{\text{sim}(f_t, f_p)}{\tau}$, where $f_t$ is the text embedding and $f_p$ is the image patch feature.
- **Classification**: Zero-shot inference via similarity probabilities.
- **Localization/Segmentation**: Pixel-level VL similarity maps directly provide spatial localization.

**2. LLM-Assisted Semantic Extraction**

- A large language model decomposes complex radiology reports into concise semantic sentences.
- Each sentence describes a single independent medical finding (e.g., "infiltration is present in the right lower lobe").
- Reduces redundant information and improves matching precision.

**3. Multi-Positive Contrastive Learning**

- A single image may correspond to multiple valid text descriptions (multiple findings).
- Conventional contrastive learning considers only a single positive pair.
- This work employs a multi-positive InfoNCE loss: $\mathcal{L} = -\sum_{k \in P(i)} \log \frac{\exp(s_{ik}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$

**4. Frozen Pretrained Visual Encoder + Trainable Transformer Layers**

- The pretrained visual encoder (e.g., BiomedCLIP) is frozen.
- Additional trainable Transformer layers are appended to handle high-resolution images.
- An efficient parameter strategy that avoids full fine-tuning.

### Loss & Training

- Multi-positive contrastive loss + KL divergence regularization.
- Two-stage training: VL alignment is trained first, followed by fine-tuning of the segmentation head.
- Data: Publicly available chest X-ray datasets including MIMIC-CXR.

## Key Experimental Results

### Main Results

Zero-shot classification (CheXpert 5×200, AUC):

| Method | Atelectasis | Cardiomegal. | Consolidat. | Edema | Pl. Effusion | Avg. |
|------|-------------|-------------|-------------|-------|-------------|------|
| BioViL | 72.5 | 85.3 | 78.1 | 82.6 | 88.2 | 81.3 |
| MedCLIP | 74.8 | 87.1 | 79.5 | 84.3 | 89.7 | 83.1 |
| CheXzero | 76.2 | 88.5 | 81.3 | 85.8 | 90.5 | 84.5 |
| RadZero | **79.5** | **90.8** | **84.2** | **88.1** | **92.3** | **87.0** |

Zero-shot localization (MS-CXR, mIoU / Pointing Game):

| Method | mIoU | Pointing Game |
|------|------|-------------|
| GradCAM | 18.5 | 52.3 |
| BioViL-T | 25.3 | 61.8 |
| MedKLIP | 28.7 | 65.2 |
| RadZero | **35.2** | **72.8** |

### Ablation Study

Contribution of each component (CheXpert AUC):

| Model | Zero-shot Cls. | Zero-shot Loc. | Zero-shot Seg. |
|------|----------|----------|----------|
| Base CLIP | 81.3 | 18.5 | 22.1 |
| + VL-CABS | 84.5 | 30.8 | 35.6 |
| + LLM Semantic Extraction | 85.8 | 32.5 | 37.2 |
| + Multi-Positive Contrastive | **87.0** | **35.2** | **40.8** |

### Key Findings

1. VL-CABS similarity maps localize lesion regions more precisely than conventional attention maps.
2. LLM-based semantic extraction produces more focused textual features, yielding approximately 1.3 AUC points improvement in classification.
3. RadZero substantially outperforms prior methods on zero-shot segmentation (40.8 vs. 22.1 mIoU), demonstrating the advantage of fine-grained alignment.
4. The model exhibits open-vocabulary semantic segmentation capability, generalizing to disease descriptions unseen during training.

## Highlights & Insights

- **Explainability**: VL similarity maps provide clinically interpretable visual explanations, facilitating physician trust.
- **Unified Multi-Task**: A single model supports classification, localization, and segmentation without task-specific heads.
- **High-Resolution Processing**: Additional Transformer layers effectively leverage high-resolution chest X-ray information.

## Limitations & Future Work

1. Validation is currently limited to chest X-rays; other imaging modalities (CT, MRI) remain unexplored.
2. The quality of LLM-based semantic extraction depends on the LLM's medical knowledge.
3. Zero-shot segmentation outperforms baselines but still has room for improvement in absolute terms.
4. The multi-positive strategy may lack sufficient positive samples for extremely rare diseases.

## Related Work & Insights

- **BioViL/BioViL-T** (Bannur et al.): Biomedical vision-language pretraining.
- **CheXzero**: Zero-shot chest X-ray diagnosis.
- **CLIP**: Foundational work on contrastive language-image pretraining.

## Rating

- ⭐ Novelty: 8/10 — The similarity-based cross-attention design is elegant and well-motivated.
- ⭐ Value: 9/10 — Open-source code with direct applicability to clinical scenarios.
- ⭐ Writing Quality: 8/10 — Comprehensive experiments with intuitive qualitative analysis.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[NeurIPS 2025\] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation](toward_a_vision-language_foundation_model_for_medical_data_multimodal_dataset_an.md)
- [\[NeurIPS 2025\] Sequential Attention-based Sampling for Histopathological Analysis](sequential_attention-based_sampling_for_histopathological_analysis.md)

<!-- RELATED:END -->
