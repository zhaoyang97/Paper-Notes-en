---
title: >-
  [Paper Note] AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network
description: >-
  [CVPR 2026][Remote Sensing] AVION proposes a knowledge distillation framework that uses LLM-generated semantically rich remote sensing text prototypes as teacher supervision while injecting learnable prompts into both the visual and text encoders of the student, achieving tri-aspect alignment distillation that significantly outperforms existing PEFT methods on few-shot classification and cross-modal retrieval.
tags:
  - CVPR 2026
  - Remote Sensing
  - Knowledge Distillation
  - Prompt Tuning
  - Vision-Language Model
  - Cross-Modal Retrieval
date: 2026-05-08
content_hash: 4e4190f3d859e203
---

# AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network

**Conference**: CVPR 2026
**arXiv**: [2603.12659](https://arxiv.org/abs/2603.12659)
**Code**: [https://github.com/yuhu990424/AVION](https://github.com/yuhu990424/AVION)
**Area**: Remote Sensing / Vision-Language Model
**Keywords**: Remote Sensing, Knowledge Distillation, Prompt Tuning, Vision-Language Model, Cross-Modal Retrieval

## TL;DR

AVION proposes a knowledge distillation framework that uses LLM-generated semantically rich remote sensing text prototypes as teacher supervision while injecting learnable prompts into both the visual and text encoders of the student, achieving tri-aspect alignment distillation that significantly outperforms existing PEFT methods on few-shot classification and cross-modal retrieval.

## Background & Motivation

**Background**: RS-specific VLMs such as RemoteCLIP and GeoRSCLIP perform well on downstream tasks, but full fine-tuning is expensive. PEFT methods (CoOp, MaPLe, etc.) adapt to new tasks by learning a small number of parameters.

**Limitations of Prior Work**: (1) **Semantic poverty** — RS datasets typically only have category name labels (e.g., "airport") and cannot describe the vast visual variations of the same category across different regions, seasons, and sensors; (2) **Visual rigidity** — most PEFT methods only update text-side prompts while freezing the visual encoder, unable to capture RS-specific overhead perspective and scale variation features.

**Key Challenge**: The gap between simple category names and the rich visual patterns of RS images, combined with the inability of frozen visual encoders to adapt to the RS domain.

**Goal**: Simultaneously address semantic poverty and visual rigidity to make PEFT methods work effectively in RS scenarios.

**Key Insight**: Leverage LLMs to generate rich category descriptions as textual supervision, and achieve robust adaptation through visual-textual-logit tri-aspect distillation constraints.

**Core Idea**: Use LLM-enriched text prototypes to solve semantic poverty, employ dual-side prompts + tri-aspect distillation to solve visual rigidity, and incur no additional overhead at inference through the teacher-student framework.

## Method

### Overall Architecture

Offline Teacher stage: a large model (GeoRSCLIP ViT-H/14) encodes LLM-generated category descriptions, validates them with visual prototypes, and aggregates them into text prototypes $\mathbf{t}_k^{T*}$. Training Student stage: a smaller model (GeoRSCLIP ViT-B/32) with visual and text prompts injected, trained through tri-aspect distillation alignment. Inference stage: only the student model performs forward passes.

### Key Designs

1. **Selective Prototype Aggregation**

    - **Function**: Filters and aggregates text prototypes for each class from LLM-generated candidate descriptions.
    - **Mechanism**: First uses Gemini 2.5 Flash to generate up to 50 RS-perspective descriptions per class, marking RS relevance with RS-Flag. Computes teacher visual prototypes $\hat{\mathbf{v}}_k^T = \frac{1}{|\mathcal{B}_k|}\sum_i \mathbf{v}_{k,i}^T$, evaluates the similarity of each description to the visual prototype $s_{k,j} = (\hat{\mathbf{v}}_k^T)^\top \mathbf{t}_{k,j}^T$, removes outliers using median/MAD z-score, and finally produces weighted aggregation with $w_{k,j} \propto \exp(\beta s_{k,j} + \gamma \cdot \text{RS-Flag}_{k,j})$ as the final prototype.
    - **Design Motivation**: LLMs may produce non-visual or non-RS-related hallucinated descriptions; visual prototype verification and statistical filtering ensure prototype quality and RS relevance.

2. **Dual-Side Prompt Tuning**

    - **Function**: Simultaneously injects learnable prompts into the student's visual and text encoders.
    - **Mechanism**: The text side learns context tokens similar to CoOp; the visual side injects prompt tokens at each ViT layer similar to VPT. Both sides keep the backbone frozen, updating only prompt parameters.
    - **Design Motivation**: Visual-side prompts give the encoder flexibility to adapt to RS overhead perspective features, addressing visual rigidity; text-side prompts absorb the teacher's rich semantic knowledge.

3. **Tri-Aspect Alignment**

    - **Function**: Simultaneously aligns visual embeddings, text embeddings, and similarity logits.
    - **Mechanism**: $\mathcal{L}_{\text{img}} = 1 - (\mathbf{v}_i^S)^\top \mathbf{v}_i^T$ aligns visual features; $\mathcal{L}_{\text{text}} = 1 - (\mathbf{t}_k^S)^\top \mathbf{t}_k^{T*}$ aligns text prototypes; $\mathcal{L}_{\text{logit}} = \tau^2 \text{KL}(\sigma(\mathbf{s}^T/\tau) \| \sigma(\mathbf{s}^S/\tau))$ aligns cross-modal similarity distributions. Total loss: $\mathcal{L} = \mathcal{L}_{\text{task}} + 0.5\mathcal{L}_{\text{img}} + 0.5\mathcal{L}_{\text{text}} + \mathcal{L}_{\text{logit}}$, with 30% linear warmup for the logit term.
    - **Design Motivation**: Embedding alignment alone is insufficient; aligning inter-class relational structure (via logit distillation) enables the student to learn not only individual class representations but also relative inter-class relationships.

### Loss Function / Training Strategy

Total loss adds task classification cross-entropy. AdamW optimizer, lr=5e-4, 100 epochs for few-shot, 50 epochs for base-to-novel. Distillation temperature $\tau=2$, with linear warmup for logit weight.

## Key Experimental Results

### Main Results (6-Dataset Average Few-Shot Classification Accuracy)

| Method | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|------|--------|--------|--------|--------|---------|
| GeoRSCLIP (zero-shot) | 72.95 | — | — | — | — |
| CoOp | 69.98 | 78.95 | 84.52 | 87.57 | 90.24 |
| CoCoOp | 70.27 | 80.56 | 85.74 | 88.93 | 91.41 |
| MMRL | 70.57 | 79.47 | — | — | — |
| **AVION** | **73.12** | **82.34** | **87.21** | **90.48** | **92.85** |

### Ablation Study (AID Dataset, 16-shot)

| Configuration | Base Acc | Novel Acc | HM | Description |
|------|----------|-----------|------|------|
| AVION full | 95.2 | 88.7 | 91.8 | Only method exceeding baseline on both base and novel |
| w/o textual alignment | 94.1 | 85.3 | 89.5 | Text prototype supervision is critical for novel-class generalization |
| w/o visual prompts | 94.8 | 86.1 | 90.3 | Visual-side prompts improve domain adaptation |
| w/o logit alignment | 94.5 | 87.2 | 90.7 | Logit distillation provides inter-class relationships |
| w/o RS-Flag | 94.9 | 87.5 | 91.1 | RS flag filtering improves prototype quality |

### Key Findings

- AVION is the only method where both base and novel accuracy exceed the GeoRSCLIP baseline in the base-to-novel setting, demonstrating that distillation does not harm generalization
- Both RS-Flag and visual verification in text prototype aggregation are indispensable; removing either degrades Novel Acc
- Cross-modal retrieval mR also improves, indicating that tri-aspect distillation enhances overall modal alignment quality

## Highlights & Insights

- **Precise diagnosis of semantic poverty**: RS datasets having only category name labels is the fundamental cause of PEFT failure; generating rich descriptions via LLMs is an elegant solution
- **Ingenious selective prototype aggregation**: Functions like parameter-free cross-attention, with visual prototypes as queries and text descriptions as keys/values, automatically filtering poor descriptions and balancing aggregation weights
- **Tri-aspect distillation preserves generalization**: Logit alignment preserves inter-class relational structure, which is key to AVION not degrading on novel classes

## Limitations & Future Work

- Depends on LLM description quality; may produce low-quality descriptions for non-English or unconventional RS categories
- Teacher model is fixed as GeoRSCLIP ViT-H/14; switching to other backbones requires rebuilding prototypes
- Not validated on more complex RS downstream tasks such as detection and segmentation

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Accurate problem diagnosis with a systematic and complete solution
- **Experimental rigor**: ⭐⭐⭐⭐ — Six datasets + three task settings + thorough ablations
- **Writing quality**: ⭐⭐⭐⭐ — Clear motivation derivation, well-designed figures and tables
- **Impact**: ⭐⭐⭐⭐ — Practical solution for RS VLM adaptation

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [\[CVPR 2026\] Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [\[CVPR 2026\] ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery](acpv-net_all-class_polygonal_vectorization_for_seamless_vector_map_generation_fr.md)
- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[CVPR 2026\] Conflated Inverse Modeling for Urban Vegetation Patterns](conflated_inverse_urban_vegetation.md)

<!-- RELATED:END -->
