---
title: >-
  [Paper Note] Chimera: Improving Generalist Model with Domain-Specific Experts
description: >-
  [ICCV 2025][Multimodal VLM][Multimodal reasoning] This paper proposes Chimera, a scalable and low-cost multimodal pipeline that integrates domain-specific expert knowledge (tables, charts, math, documents) into a generalist multimodal large model via a lightweight routing module for dynamic expert selection, a progressive training strategy, and a Generalist-Specialist Collaboration Masking (GSCM) mechanism. Chimera achieves 64.9% on MathVista (SOTA) and matches or surpasses specialist models on multiple visual structure extraction tasks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Multimodal reasoning
  - expert model integration
  - domain adaptation
  - routing mechanism
  - visual content extraction
date: 2026-05-08
content_hash: ea82601ee8ce52c3
---

# Chimera: Improving Generalist Model with Domain-Specific Experts

**Conference**: ICCV 2025
**arXiv**: [2412.05983](https://arxiv.org/abs/2412.05983)
**Code**: Open-source (weights, data, evaluation)
**Area**: Multimodal Large Models / VLM
**Keywords**: Multimodal reasoning, expert model integration, domain adaptation, routing mechanism, visual content extraction

## TL;DR

This paper proposes Chimera, a scalable and low-cost multimodal pipeline that integrates domain-specific expert knowledge (tables, charts, math, documents) into a generalist multimodal large model via a lightweight routing module for dynamic expert selection, a progressive training strategy, and a Generalist-Specialist Collaboration Masking (GSCM) mechanism. Chimera achieves 64.9% on MathVista (SOTA) and matches or surpasses specialist models on multiple visual structure extraction tasks.

## Background & Motivation

Large multimodal models (LMMs) perform well on general tasks but remain inadequate for **specialized domain tasks**:

- **Limitations of generalist LMMs**: Training data is dominated by natural images, whereas specialized tasks involve charts, tables, geometric figures, and function plots — content that is denser in text and more abstract.
- **"One for One" paradigm problems**: Domain-specific expert models achieve strong performance but poor generalization; distribution gaps across sub-domains are large.
- **Data privacy issues**: Domain-specific data is often proprietary and unavailable for post-training LMMs.

Directly integrating expert models faces two core challenges:

**Representation gap**: Large distribution shifts exist across encoders from different domains.

**Optimization imbalance**: The generalist visual encoder is already well-aligned with the language model, causing the model to over-rely on the generalist encoder and ignore expert features.

## Method

### Overall Architecture

Chimera consists of the following components:
- Generalist visual encoder $E^g$ + generalist projector $P^g$ + language model $f$ (initialized from a pretrained LMM, e.g., InternVL2)
- Router $R$ (a linear layer that predicts based on the CLS token of the generalist encoder)
- Expert model set $S^e = \{E^{table}, E^{chart}, E^{math}\}$
- Expert projector set $S^p = \{P^{table}, P^{chart}, P^{math}\}$

Inference flow: the router determines whether and which expert to invoke based on the visual input → expert features are concatenated with generalist features → fed into the language model.

### Key Designs

1. **Lightweight Routing Module (Router)**:

    - Takes the CLS token $\mathcal{Z}_v^{cls}$ of the generalist encoder as input
    - A linear layer outputs $N_e + 1$ predictions ($N_e$ experts + 1 no-expert option)
    - $i = \arg\max_i(\mathcal{H}_r)_i$ selects the expert with the highest confidence
    - Achieves 95.4% routing accuracy on MathVista; all errors are generalist–expert confusions, with no inter-expert confusion
    - Routing loss uses categorical cross-entropy: $\mathcal{L}_c = -\sum_{i=0}^{N_e+1}\log P(c_i|\mathcal{X}_v, \theta)$

2. **Generalist-Specialist Collaboration Masking (GSCM)**:

    - **Core problem**: The generalist encoder is already well-aligned with the language model, so directly concatenating expert features causes the model to "take shortcuts" — relying solely on generalist features and ignoring expert features.
    - **Solution**: During training, a proportion of generalist visual tokens is randomly sampled from a uniform distribution and their attention masks are set to False.
    - Effect: Forces the model to utilize domain information from expert models as a complement to generalist information.
    - The uniform distribution prevents bias introduced by concentrating masking on image centers or specific regions.
    - Attention analysis validates: after applying GSCM, the model's output attention to expert visual tokens increases significantly.

3. **Progressive Training Strategy**:

    - **Stage 1 — Domain-generalist knowledge alignment**: Freezes the generalist encoder, expert models, and language model; trains only the router, generalist projector, and expert projectors.
        - Data: natural image captioning, table format conversion, chart structure extraction, math diagram description, paragraph-level OCR.
    - **Stage 2 — Visual instruction fine-tuning**: Unfreezes the router, projectors, and language model (expert encoders remain frozen throughout); applies GSCM.
        - Data: multi-domain instruction-following datasets.

### Loss & Training

Total loss: $\mathcal{L} = \mathcal{L}_c + \mathcal{L}_m$

- $\mathcal{L}_m$: token-level cross-entropy loss for autoregressive modeling
- $\mathcal{L}_c$: router classification loss

Preference optimization (optional Stage 3): Preference pairs are constructed from 60K public data for DPO, pushing Chimera to 68.3%.

## Key Experimental Results

### Main Results

**MathVista testmini accuracy (%)**

| Model | Scale | Overall | GPS | MWP | TQA | GEO |
|-------|-------|---------|-----|-----|-----|-----|
| GPT-4o | - | 63.8 | - | - | - | - |
| InternVL2-8B | 8B | 61.6 | 64.4 | 61.3 | 64.6 | 61.9 |
| Qwen2-VL | 7B | 58.2 | - | - | - | - |
| Math-LLaVA* | 13B | 46.6 | 57.7 | 56.5 | 51.3 | 56.5 |
| **Chimera-8B** | **8B** | **64.9** | **71.6** | **72.6** | **65.2** | **69.5** |
| **Chimera†-8B** | **8B** | **68.3** | **76.9** | **80.1** | 60.8 | **74.5** |

**MathVerse accuracy (%)**

| Model | Scale | Overall | Text Dominant | Vision Only |
|-------|-------|---------|---------------|-------------|
| GPT-4V | - | 39.4 | 54.7 | 31.6 |
| MAVIS-7B* | 7B | 27.5 | 41.4 | 14.6 |
| InternVL2-8B | 8B | 31.3 | 38.8 | 17.0 |
| **Chimera-8B** | **8B** | **32.4** | **39.6** | **19.3** |

**Visual Structure Extraction (ChartQA-SE AP@strict / Table-SE TEDS)**

| Method | ChartQA-SE AP@strict | Table-SE TEDS | Table-SE Edit Dist↓ |
|--------|---------------------|---------------|---------------------|
| GOT* | 74.7 | **0.745** | 0.257 |
| InternVL-2 | 73.7 | 0.676 | 0.229 |
| **Chimera** | **74.1** | 0.740 | **0.165** |

### Ablation Study

**Domain-level analysis (MathVista testmini)**

| Model | Overall | General | Chart | Table | Math |
|-------|---------|---------|-------|-------|------|
| InternVL2-2B | 48.3 | 45.3 | 58.9 | 50.0 | 44.2 |
| Chimera-2B | 53.1 | 46.0 | 60.3 | 62.9 | 56.1 |
| InternVL2-4B | 57.0 | 50.1 | 66.2 | 65.7 | 58.3 |
| Chimera-4B | 61.3 | 54.0 | 64.8 | 72.9 | 66.9 |
| InternVL2-8B | 61.6 | 52.7 | 71.2 | 67.1 | 66.5 |
| Chimera-8B | 64.9 | 57.5 | 71.2 | 62.9 | 71.9 |

**Router error statistics (MathVista testmini)**

| GT \ Pred | General | Table | Chart | Math |
|-----------|---------|-------|-------|------|
| General | – | 0 | 16 | 6 |
| Table | 1 | – | 0 | 0 |
| Chart | 1 | 0 | – | 0 |
| Math | 22 | 0 | 0 | – |

Routing accuracy: 95.4%. All misclassifications occur between general↔expert categories; no inter-expert confusion is observed.

### Key Findings

- Chimera-8B achieves 64.9% on MathVista, surpassing GPT-4o (63.8%) and setting a new SOTA among open-source LMMs under 70B.
- **Expert knowledge even improves performance on general-domain tasks**: Chimera outperforms the InternVL2 baseline in the "General" category (57.5% vs. 52.7%), indicating that domain knowledge provides diversified perspectives.
- DPO post-training with only 60K data improves performance from 64.9% to 68.3% (+3.4%), demonstrating the scalability of the framework.
- GSCM attention analysis validates: with masking, expert tokens receive significantly more attention; without masking, the model relies almost entirely on the generalist encoder.
- The math expert exhibits the most consistent gains — it improves performance across all model scales; the table expert's high specialization may introduce noise at the 8B scale.
- Chimera substantially outperforms InternVL2 on document structure extraction (Doc-SE) in both English and Chinese tasks.

## Highlights & Insights

- **"One for All + Domain Experts" paradigm**: Achieves specialized capabilities without sacrificing generality, offering more practical value than either pure expert models or pure generalist models.
- **Deep insight behind GSCM**: The mechanism identifies and addresses the counter-intuitive problem of "optimization imbalance caused by pretraining alignment advantage."
- **Minimal routing design**: A single linear layer achieves 95.4% routing accuracy, indicating that visual content from different domains is already well-separable in feature space.
- **Efficient DPO integration**: Demonstrates that Chimera, as a base framework, can be seamlessly combined with preference optimization.

## Limitations & Future Work

- Chimera-8B underperforms InternVL2-8B on the table domain (62.9% vs. 67.1%), as the table expert (StructEqTable) is overly specialized and the task gap introduces noise.
- Routing labels are assigned at the dataset level rather than the image level; the "general" category may contain images from mixed domains.
- Only three domain experts (table, chart, math) are currently integrated; the effect of extending to more domains remains to be validated.
- Expert models remain frozen throughout training, limiting further adaptation of domain knowledge.
- Inference requires running all encoders (generalist + selected expert), incurring higher computational cost than purely generalist models.

## Related Work & Insights

- **InternVL2**: The base model for Chimera; its pretrain–finetune paradigm informs the progressive alignment strategy.
- **GOT**: An expert model for document structure extraction, trained on millions of proprietary data samples.
- **ChartVLM**: Its routing structure for chart QA inspires Chimera's routing design.
- **MAVIS / Math-LLaVA**: Math reasoning expert models that demonstrate the importance of domain specialization but lack cross-domain generalization.
- **MoE**: Chimera's router-plus-expert design shares similarities with Mixture of Experts but is more lightweight and targets the integration of pretrained specialist models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The GSCM mechanism addresses an important yet easily overlooked optimization imbalance problem; the "integrate existing experts" paradigm is practical and scalable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers both reasoning and extraction scenarios, three model scales (2B/4B/8B), multiple benchmarks (MathVista/MathVerse/ChartQA-SE/Table-SE/Doc-SE), and fine-grained domain-level analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture diagrams are clear; the progressive method description is well-structured; the training strategy is presented in a logical hierarchy.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a low-cost, highly efficient framework for LMMs to rapidly acquire domain capabilities; reproducible using public data and models, with high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[ICCV 2025\] DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing](dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)
- [\[CVPR 2026\] DC-Merge: Improving Model Merging with Directional Consistency](../../CVPR2026/multimodal_vlm/dc-merge_improving_model_merging_with_directional_consistency.md)
- [\[ICCV 2025\] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](a_quality-guided_mixture_of_score-fusion_experts_framework_for_human_recognition.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](compcap_improving_multimodal_large_language_models_with_composite_captions.md)

<!-- RELATED:END -->
