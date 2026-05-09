---
title: >-
  [Paper Note] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network
description: >-
  [CVPR 2026][Remote Sensing][Vision-Language Model] AVION proposes a knowledge distillation framework that generates semantically rich text prototypes via LLMs and employs visual-textual dual-side prompt tuning with tri-aspect alignment distillation, addressing semantic poverty and visual rigidity in remote sensing VLM adaptation and comprehensively surpassing SOTA on few-shot classification, base-to-novel generalization, and cross-modal retrieval.
tags:
  - CVPR 2026
  - Remote Sensing
  - Vision-Language Model
  - Knowledge Distillation
  - Parameter-Efficient Fine-Tuning
  - Remote Sensing Scene Classification
  - Prompt Learning
date: 2026-05-08
content_hash: 1ccc20eb39f6b391
---

# AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network

**Conference**: CVPR 2026
**arXiv**: [2603.12659](https://arxiv.org/abs/2603.12659)
**Code**: [https://github.com/yuhu990424/AVION](https://github.com/yuhu990424/AVION)
**Area**: Remote Sensing
**Keywords**: Vision-Language Model, Knowledge Distillation, Parameter-Efficient Fine-Tuning, Remote Sensing Scene Classification, Prompt Learning

## TL;DR

AVION proposes a knowledge distillation framework that generates semantically rich text prototypes via LLMs and employs visual-textual dual-side prompt tuning with tri-aspect alignment distillation, addressing semantic poverty and visual rigidity in remote sensing VLM adaptation and comprehensively surpassing SOTA on few-shot classification, base-to-novel generalization, and cross-modal retrieval.

## Background & Motivation

Remote sensing (RS) vision-language models (e.g., RemoteCLIP, GeoRSCLIP) exhibit strong zero-shot capabilities after pretraining, but still require efficient adaptation for new scenarios. Full-parameter fine-tuning is computationally expensive and prone to overfitting. Parameter-efficient fine-tuning (PEFT) serves as a lightweight alternative, but existing methods face two core bottlenecks in remote sensing:

**Semantic Poverty**: RS datasets provide only category names (e.g., "airport") without describing the vast visual variations of the same category across different regions, seasons, and sensors. Methods like CoOp that learn from "a photo of [CLASS]" templates leave the text encoder unable to fully express diverse appearance patterns.

**Visual Rigidity**: Most PEFT methods update only the text encoder while freezing the visual encoder, preventing the model from capturing RS-specific scale variations and cross-source heterogeneity.

**Key Insight**: Using large language models to generate rich category descriptions as teacher signals while injecting learnable prompts on both the visual and textual sides, achieving efficient adaptation through tri-aspect alignment distillation.

**Core Idea**: LLM-enhanced text prototypes serve as the teacher to guide tri-aspect distillation alignment for a student model with visual-textual dual-side prompt learning.

## Method

### Overall Architecture

AVION adopts a teacher-student distillation architecture: a frozen large teacher model (GeoRSCLIP ViT-H/14) constructs semantically rich text prototypes offline; the student model (GeoRSCLIP ViT-B/32) has learnable prompts injected into both visual and text encoders and is trained via tri-aspect alignment losses. Only the student model is used during inference.

### Key Designs

1. **LLM Domain Prompting + Selective Prototype Aggregation (Textual Prototype Enhancement)**

    - **Function**: Generates semantically rich text prototypes for each category, replacing single category names.
    - **Mechanism**: (1) An LLM (Gemini 2.5 Flash) generates up to 50 RS-related descriptions per class; (2) RS-Flag rules filter out non-RS descriptions; (3) the teacher's visual prototype is used as a query to compute similarity for each description; (4) a robust z-score based on median/MAD removes outlier descriptions; (5) softmax-weighted aggregation produces the final prototype, with weights incorporating RS-Flag prior boosting.
    - **Design Motivation**: LLM-generated descriptions may contain hallucinations or non-RS content, which must be validated through visual prototype verification and RS-Flag filtering. This aggregation process resembles parameter-free cross-attention, ensuring prototypes are both semantically rich and visually aligned.

2. **Dual-Side Deep Prompt Tuning**

    - **Function**: Injects learnable prompts into both the visual and text encoders of the student model simultaneously.
    - **Mechanism**: Similar to VPT and CoOp, deep prompt tokens are injected across multiple layers of the ViT, enabling the student encoder to gain adaptation flexibility while keeping pretrained weights frozen.
    - **Design Motivation**: Adjusting only the text side cannot handle RS images' scale variations and overhead perspective features; adjusting only the visual side lacks semantic guidance. Dual-side prompts allow both encoders to accumulate RS knowledge under teacher guidance.

3. **Tri-Aspect Alignment Distillation**

    - **Function**: Achieves comprehensive knowledge transfer through three complementary losses.
    - **Mechanism**:
        - **Visual alignment**: Brings student and teacher visual embeddings closer (cosine distance)
        - **Textual alignment**: Brings student text embeddings closer to teacher semantic prototypes (cosine distance)
        - **Similarity alignment**: Aligns cross-modal probability distributions using temperature-scaled KL divergence
    - **Design Motivation**: Visual alignment addresses visual rigidity, textual alignment addresses semantic poverty, and logit alignment transfers implicit knowledge about inter-class relationships.

### Loss Function / Training Strategy

The total objective is the weighted sum of task loss and three alignment losses. Visual and textual alignment weights are set to 0.5, logit alignment weight to 1.0, with 30% linear warm-up for the logit loss. Distillation temperature τ=2. AdamW optimizer, lr 5e-4, batch size 4. All experiments are conducted on a single NVIDIA L4 GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | AVION | Previous SOTA (APPLeNet) | Improvement |
|--------|------|-------|---------------------|------|
| 6-dataset avg (1-shot) | Accuracy | **74.27%** | 74.27% | Tied |
| 6-dataset avg (8-shot) | Accuracy | **91.85%** | 89.20% | +2.65pp |
| 6-dataset avg (16-shot) | Accuracy | **93.69%** | 91.61% | +2.08pp |
| 6-dataset avg (B2N) | HM | **87.05%** | 83.84% | +3.21pp |
| 6-dataset avg (B2N) | Novel | **79.94%** | 75.75% | +4.19pp |
| RSITMD | mR | **52.92%** | - | +1.11pp vs GeoRSCLIP-FT |
| RSICD | mR | **39.80%** | - | +0.93pp vs GeoRSCLIP-FT |

### Ablation Study

| Configuration | HM (%) | 1-shot (%) | Description |
|------|--------|------------|------|
| B0: CoOp-style shallow text prompts | 78.88 | 69.98 | Baseline |
| B1: + deep prompts | 66.71 | 66.95 | Severe novel-class degradation |
| B2: + visual alignment | 72.74 | 70.21 | Regularization restores generalization |
| B5: + LLM prototypes + selective aggregation | 83.05 | 72.52 | Largest HM improvement |
| B7: + logit alignment + warm-up | **87.05** | **74.27** | Overall best |

### Key Findings

- AVION is the only method that simultaneously exceeds the GeoRSCLIP baseline in the base-to-novel setting (Novel 79.94% vs 79.75%)
- As shot count increases, the gap between AVION and the runner-up widens from 0pp to +2.65pp
- LLM domain prompting + selective aggregation is the largest contributing component (HM +10.31pp)
- t-SNE visualization shows AVION maintains good multimodal alignment on novel classes

## Highlights & Insights

- Precisely diagnoses two core bottlenecks in RS PEFT and systematically addresses each one
- Selective prototype aggregation uses parameter-free cross-attention to align LLM knowledge with visual semantics, leveraging rich semantics while filtering hallucinations
- Step-by-step ablation of tri-aspect distillation is clear, with quantified evidence for each component's contribution
- Outperforms full-parameter fine-tuning on cross-modal retrieval with fewer trainable parameters

## Limitations & Future Work

- Offline pre-computation of teacher prototypes still incurs overhead, potentially inefficient with extremely large numbers of categories
- Description quality depends on LLM prompt design; effects of different LLMs are unexplored
- Validated only on optical RS; applicability to other modalities such as SAR is unknown
- Experiments use only RS-specific VLMs; generalization to general-purpose CLIP is unverified

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of dual-side prompts + LLM prototype enhancement + tri-aspect distillation is novel, though individual components are not entirely new
- **Experimental rigor**: ⭐⭐⭐⭐⭐ — 6 classification + 2 retrieval datasets, three evaluation protocols, thorough ablations
- **Writing quality**: ⭐⭐⭐⭐⭐ — Clear problem diagnosis, well-motivated method, progressively layered ablations
- **Impact**: ⭐⭐⭐⭐ — Practical value for RS VLM adaptation; the LLM-assisted prototype construction approach is inspirational

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[CVPR 2026\] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [\[CVPR 2026\] ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery](acpv-net_all-class_polygonal_vectorization_for_seamless_vector_map_generation_fr.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[CVPR 2026\] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_co.md)

<!-- RELATED:END -->
