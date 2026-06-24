---
title: >-
  [Paper Note] STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection
description: >-
  [CVPR 2025][Multimodal VLM][X-ray security inspection] Constructs the first multimodal X-ray baggage security dataset **STCray** (46,642 image-description pairs, 21 threat classes including IEDs and 3D printed guns), designs the **STING protocol** to systematically generate domain-aware high-quality descriptions, and trains the domain-specific VLM **STING-BEE**, establishing new baselines in scene understanding, threat localization, visual grounding, and VQA…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "X-ray security inspection"
  - "domain-specific VLM"
  - "multimodal dataset"
  - "threat detection"
  - "visual grounding"
date: 2026-05-08
content_hash: 8a805676a26b11b3
---

# STING-BEE: Towards Vision-Language Model for Real-World X-ray Baggage Security Inspection

**Conference**: CVPR 2025  
**arXiv**: [2504.02823](https://arxiv.org/abs/2504.02823)  
**Code**: [https://divs1159.github.io/STING-BEE/](https://divs1159.github.io/STING-BEE/)  
**Area**: Multimodal VLM  
**Keywords**: X-ray security inspection, domain-specific VLM, multimodal dataset, threat detection, visual grounding

## TL;DR

Constructs the first multimodal X-ray baggage security dataset **STCray** (46,642 image-description pairs, 21 threat classes including IEDs and 3D printed guns), designs the **STING protocol** to systematically generate domain-aware high-quality descriptions, and trains the domain-specific VLM **STING-BEE**, establishing new baselines in scene understanding, threat localization, visual grounding, and VQA, while demonstrating SOTA cross-domain generalization capabilities.

## Background & Motivation

With the surge in global passenger traffic, baggage security inspection faces massive pressure. X-ray baggage screening is a core component of security checks, but traditional methods face multiple challenges:

**Background**: Existing computer-aided screening (CAS) systems primarily rely on vision methods like edge detection, contour mapping, and attention mechanisms, which are limited to **closed-set predefined categories** and fail to generalize to novel threats.

**Limitations of Prior Work**:
1. **Dataset Deficiencies**—Existing X-ray datasets (such as SIXray, OPIXray, PIDRay, etc.) are entirely unimodal, lacking text descriptions; simple category labels cannot represent the occlusion relationships and concealment strategies in complex scenes.
2. **Poor Generalization**—Models trained on handguns might fail to detect rifles (morphological differences) or 3D printed guns (color differences); cross-device domain shifts further degrade detection performance.
3. **VLM Cannot Be Used Directly**—General-purpose VLMs like GPT-4 and Gemini suffer from severe hallucinations on X-ray images, misinterpreting baggage scans as medical images and entirely failing to recognize threat items.

**Key Challenge**: VLMs have the potential for complex scene understanding and zero-shot generalization, but they lack multimodal data in the X-ray domain; meanwhile, existing datasets only contain simple labels, which cannot support VLM training.

**Key Insight**: Starting from data—designing a systematic Strategic Threat ConcealING Protocol (STING Protocol) to generate high-quality domain-aware descriptions, constructing the first X-ray multimodal dataset, and then training a domain-specific VLM on top of it.

## Method

### Overall Architecture

STING-BEE comprises three core contributions: (1) a STING protocol-driven dataset construction pipeline that systematically varies threat item positions, angles, and occlusion levels to generate detailed descriptions; (2) the STCray dataset with 46,642 images across 21 threat categories, including IEDs and 3D printed guns; and (3) the STING-BEE model, a domain-specific VLM based on the LLaVA architecture trained via a two-stage process (multitask instruction tuning and visual grounding instruction tuning), supporting scene understanding, referring expression comprehension, visual grounding, and VQA.

### Key Designs

1. **STING Protocol (Strategic Threat ConcealING Protocol)**
    - **Function**: Systematically generates X-ray baggage scan images accompanied by high-quality domain-aware descriptions.
    - **Mechanism**:
     - Select baggage types (suitcases, backpacks, duffel bags, etc.) and threat categories.
     - Systematically vary the **position** (center/corner/sides) and **pose** (flat/tilted/upright) of threat items.
     - Incrementally increase **occlusion complexity**: from no occlusion $\rightarrow$ minor clutter $\rightarrow$ moderate $\rightarrow$ heavy clutter $\rightarrow$ deliberate concealment.
     - Each occlusion tier is paired with obstructing items of different densities and materials to simulate real-world smuggling strategies.
     - Descriptions record details: threat position, orientation, material characteristics, overlapping items, and degree of occlusion.
    - **Design Motivation**: General-purpose VLMs (GPT-4, Gemini) exhibit severe hallucinations and incorrect descriptions on X-ray images, meaning they cannot be directly utilized to generate training data. The STING protocol ensures description accuracy from the source via structured scanning processes, rather than relying on model generation.

2. **Task-Identification Tokens**
    - **Function**: Enables a single model to distinguish visual-linguistic tasks of different granularities without training separate models for each task.
    - **Mechanism**: Introduces two special tokens:
     - `[refer]`—triggers the referring threat localization task, with the model outputting normalized bounding box coordinates.
     - `[grounding]`—triggers the visual grounding task, with the model outputting interleaved descriptive text and spatial coordinates.
     - Scene understanding and VQA do not require special tokens, requiring only image-level comprehension.
    - **Design Motivation**: X-ray security scanning requires multi-tier understanding from coarse to fine—first identifying the presence of a threat (classification), then locating it (localization), and finally describing details (description). Task tokens allow the model to flexibly switch between different granularities.

3. **Two-Stage Multi-Task Instruction Tuning Strategy**
    - **Function**: Enables the model to progressively acquire X-ray domain knowledge and visual grounding capabilities.
    - **Mechanism**:
     - **Stage 1: Multi-task Threat Instruction Tuning**—Employs 120,190 instruction training samples (comprising scene understanding, referring expressions, and VQA) to build a basic understanding of X-ray data.
     - **Stage 2: Threat Visual Grounding Instruction Tuning**—Incorporates 29,444 visual grounding instructions to train the model to generate descriptive responses interleaved with bounding box coordinates.
     - LoRA is applied to fine-tune the LLM, while the MLP projector is trained and the visual encoder is frozen.
    - **Design Motivation**: Establishing domain understanding prior to learning fine-grained grounding avoids multi-task conflicts. LoRA fine-tuning preserves general language capabilities while adapting to domain-specific features.

### Loss & Training

Based on the standard autoregressive language modeling loss of the LLaVA framework. The image encoder adopts the CLIP-pretrained ViT-L/14, and the language model uses Vicuna. LoRA is utilized for efficient fine-tuning of the LLM parameters.

## Key Experimental Results

### VQA Performance (In-domain, 7 question categories)

| Model | Instance Localization | Complex Reasoning | Instance Recognition | Counting | Misleading | Attribute | Interaction | Overall |
|------|---------|---------|---------|------|-------|------|------|------|
| Florence-2 | 30.11 | 37.50 | 39.84 | 29.95 | 21.16 | 35.80 | 29.12 | 32.27 |
| LLaVA 1.5 | 29.73 | 56.67 | 74.04 | 34.24 | 13.76 | 40.85 | 24.03 | 41.94 |
| **STING-BEE** | **49.22** | **79.21** | **80.04** | **45.24** | **27.76** | **52.85** | **35.03** | **52.81** |

### STCray Dataset Statistics

| Metric | Value |
|------|------|
| Total Images | 46,642 |
| Threat Classes | 21 |
| Threat Instances | 57,218 |
| Training Set | 30,044 |
| Test Set | 16,598 |
| Single-threat Images | 36,438 |
| Multi-threat Images | 9,255 |

### Comparison with Existing Datasets

STCray is the unique dataset that simultaneously satisfies the following conditions:
- ✓ Multimodal (image-description pairs)
- ✓ Strategic concealment (systematic occlusion)
- ✓ Novel threats (IEDs, 3D printed guns)
- ✓ Zero-shot capability

### Key Findings

1. STING-BEE outperforms LLaVA 1.5 by 10.87 percentage points on overall VQA (52.81 vs 41.94), showing major advantages especially in complex reasoning (+22.54%) and instance localization (+19.49%).
2. General VLMs (GPT-4, Gemini) completely fail to identify threat items in X-ray images, even misidentifying baggage scans as medical images.
3. STING-BEE exhibits SOTA generalization in cross-domain settings, which indicates that the data generated via the STING protocol successfully helps the model acquire domain-specific knowledge of X-rays.

## Highlights & Insights

1. **Data-driven rather than model-driven**—Overcoming the core bottleneck (lack of multimodal X-ray data) through a meticulously designed data collection protocol (STING Protocol), rather than complex model architectural innovations.
2. **Real-world threat item data**—Includes real security threats like IEDs and 3D printed guns, which have never appeared in previous datasets, greatly enhancing practical application value.
3. **End-to-end multi-task unification**—Unifies four tasks (classification, localization, visual grounding, and VQA) within a single model using task tokens, avoiding the complexity of maintaining multiple dedicated models.
4. **Massive dataset construction effort**—The team spent approximately 3,109 hours (equivalent to ~130 full-time days) compiling the STCray dataset, reflecting the severity of the data bottleneck in this domain.

## Limitations & Future Work

1. Although the dataset size is substantial, the category distribution is imbalanced (e.g., Explosive has 6,491 instances, whereas Shaving Razor only has 1,284).
2. Models based on the LLaVA architecture may suffer from pretraining bias of the visual encoder in specialized visual domains like X-ray (as CLIP is pretrained on natural images).
3. The overall VQA accuracy is only 52.81%, leaving a significant gap before actual security deployment can be realized.
4. Data acquisition relies on real X-ray scanners, making it difficult to scale up to more threat categories and diverse configurations.

## Related Work & Insights

- **X-ray Security Inspection Datasets**: Escalating from GDXray (3 categories) to PIDRay (13 categories), this work extends the category space to 21 classes and introduces multimodal annotations for the first time, laying a data foundation for transitioning this field from closed-set to open-set paradigms.
- **Domain-Specific VLMs**: Similar to Med-PaLM in the medical field, this work demonstrates that VLMs require domain-specific data and training to perform in professional visual domains like X-ray security inspection.
- **Insights**: When general-purpose LLMs fail completely in specialized domain tasks, the remedy is "data first”—designing a systematic data acquisition protocol rather than attempting to alter model architectures.

## Rating

⭐⭐⭐⭐ — Massive workload (~3,109 hours of data construction), outstanding dataset contribution (the first multimodal X-ray dataset), and high practical application value; however, the model innovation is relatively limited (primarily LLaVA + LoRA), and the absolute performance in VQA remains modest.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VisionArena: 230K Real World User-VLM Conversations with Preference Labels](visionarena_230k_real_world_user-vlm_conversations_with_preference_labels.md)
- [\[ICCV 2025\] AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?](../../ICCV2025/multimodal_vlm/advdreamer_unveils_are_visionlanguage_models_truly_ready_for.md)
- [\[ACL 2025\] REAL-MM-RAG: A Real-World Multi-Modal Retrieval Benchmark](../../ACL2025/multimodal_vlm/real-mm-rag_a_real-world_multi-modal_retrieval_benchmark.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)

</div>

<!-- RELATED:END -->
