---
title: >-
  [Paper Note] Vision-Language Models Struggle to Align Entities across Modalities
description: >-
  [ACL 2025][Multimodal VLM][Cross-modal Entity Linking] This paper proposes the MATE benchmark (5,500 QA instances) to systematically evaluate the entity linking performance of VLMs via cross-modal attribute retrieval tasks in synthetic 3D scenes. The study reveals that even the strongest closed-source models still lag behind humans by approximately 15 percentage points, and their performance drops sharply as the number of objects in the scene increases—primarily due to cross-…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Cross-modal Entity Linking"
  - "Vision-Language Models"
  - "Benchmark"
  - "Entity Attribute Alignment"
  - "Visual Search"
date: 2026-05-08
content_hash: 42b1567581969cf8
---

# Vision-Language Models Struggle to Align Entities across Modalities

**Conference**: ACL 2025  
**arXiv**: [2503.03854](https://arxiv.org/abs/2503.03854)  
**Code**: [GitHub](https://github.com/hitz-zentroa/MATE)  
**Area**: Multimodal VLM / Cross-modal Alignment  
**Keywords**: Cross-modal Entity Linking, Vision-Language Models, Benchmark, Entity Attribute Alignment, Visual Search  

## TL;DR

This paper proposes the MATE benchmark (5,500 QA instances) to systematically evaluate the entity linking performance of VLMs via cross-modal attribute retrieval tasks in synthetic 3D scenes. The study reveals that even the strongest closed-source models still lag behind humans by approximately 15 percentage points, and their performance drops sharply as the number of objects in the scene increases—primarily due to cross-modal feature binding difficulties rather than single-modal perception.

## Background & Motivation

**Problem Definition**: Cross-modal entity linking refers to the fundamental ability to align the same entity and its attributes across different modalities (e.g., image and text). This capability is a prerequisite for multimodal AI systems to execute downstream tasks.

**Application Scenarios**: This ability is crucial in multiple real-world scenarios—in **autonomous driving**, aligning vehicles in images with sensor text data (speed, trajectory) is needed to construct a unified representation; in **multimodal code generation**, UI images must be aligned with code entities; in **fake news detection**, consistency between image and text information needs verification; in **scene understanding**, visual and structured data must be unified.

**Limitations of Prior Work**: Although tasks like Referring Expression Comprehension (REC), Multimodal Entity Linking (MEL), and SIMMC are related to cross-modal alignment, none of them directly test the model's ability to align entity attributes from raw multimodal inputs. REC only requires regional localization rather than attribute alignment; MEL focuses on linking mentions to knowledge bases and typically involves single-entity scenarios; SIMMC bypasses the linking challenge by providing gold standard object IDs.

**Core Motivation**: There is a lack of **systematic, controlled evaluation** of this fundamental ability. The core question proposed in this paper is: **Can current VLMs reliably align the representations of the same entity between visual and textual modalities?**

## Method

### Overall Architecture

MATE (Multimodal Attribute-based Entity linking) is a benchmark containing **5,500 QA instances**. Each instance includes a synthetic 3D scene image and its corresponding JSON text representation, featuring 3-10 geometric objects with different colors, shapes, materials, and sizes. The core of the task is: given a **pointer attribute** that uniquely identifies an object in one modality, the model is required to retrieve the **target attribute** of that object in the other modality.

| Task Direction | Pointer Attribute (Localization Modality) | Target Attribute (Retrieval Modality) | Example |
|---------|-------------------|-------------------|------|
| Image→Text | Visual attributes in image (e.g., color "red") | Attributes in text (e.g., name "Object_0") | What is the name of the red object? |
| Text→Image | Attributes in text (e.g., name "Object_0") | Visual attributes in image (e.g., color) | What color is Object_0? |

### Key Designs

**1. Controlled Experimental Methodology with Synthetic Scenes**: Based on an extension of the CLEVR dataset, synthetic scenes containing 3-10 3D geometric objects are generated. The core idea is to use synthetic images instead of real-world ones to control all variables, excluding confounding factors like object recognition or visual ambiguity to ensure the purity of the evaluation—measuring only cross-modal entity linking ability, rather than object detection. The dataset maintains a uniform distribution across the number of objects, task directions, and attribute pairs ($43 \pm 1.5$ samples per configuration), eliminating distribution bias.

**2. Attribute Information Isolation and Cross-modal Forcing Mechanism**: Attributes are categorized into three types: visual-only attributes (color, shape), textual-only attributes (name, rotation, size, 3D coordinates), and shared attributes (material, not used as pointer or target). The key design is that the **serialized scene text does not contain the pointer/target attributes present only in the image**, forcing the model to retrieve across modalities, making it impossible to complete the task relying on a single modality alone. For example, when color in the image serves as the pointer, the textual JSON does not contain the color field.

**3. Three-Step Human Problem-Solving Process Modeling**: Through human evaluation, three cognitive steps of cross-modal entity linking are identified: (a) **Visual Search**: locate the target object in the image using the pointer attribute (e.g., finding the red object); (b) **Linking Attribute Identification**: find shared attributes (linking attributes) that distinguish this object from other objects in the scene, such as being the only cylinder; (c) **Textual Search**: use the linking attributes to locate the corresponding object in the other modality and retrieve the target attribute. This decomposition provides an analytical framework for subsequent ablation studies.

### Experimental Setup

| Setting | Details |
|-------|------|
| Open-source Models | LLaVA 1.5 (13B), LLaVA 1.6 (34B), Molmo-7B, Llama-3.2-11B, Qwen2-VL-7B, Qwen2.5-VL-7B |
| Closed-source Models | GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Flash |
| Prompting Strategy | zero-/one-/two-shot, reporting two-shot results (most stable) |
| Text Format | JSON (no significant difference from YAML/XML/verbalized) |
| Human Evaluation | 384 subsets, 5 participants, ensuring representative feature distribution |
| Compute Resources | $4 \times \text{NVIDIA A100 80GB}$, approx. 300 GPU hours |
| Evaluation Metric | Exact match accuracy |

## Key Experimental Results

### Main Results

| Model | Image→Text | Text→Image | Average |
|------|-----------|-----------|------|
| **Human** | **97.9** | **97.9** | **97.9** |
| Random Baseline | 25.4 | 18.5 | 22.0 |
| LLaVA 1.5 | 29.3 | 35.7 | 32.5 |
| LLaVA 1.6 | 48.7 | 61.6 | 55.2 |
| Molmo | 18.1 | 20.9 | 19.5 |
| Llama 3.2 | 37.4 | 11.4 | 24.4 |
| Qwen2-VL | 72.1 | 77.2 | 74.7 |
| Qwen2.5-VL | 75.7 | 84.5 | 80.1 |
| Gemini 1.5 | 63.2 | 71.2 | 67.2 |
| GPT-4o | 76.4 | 79.1 | 77.8 |
| **Claude 3.5** | **80.9** | **85.7** | **83.3** |

The best VLM (Claude 3.5) still lags behind humans by **14.6** percentage points. All VLMs perform better in the Text→Image direction than in Image→Text (except for Llama 3.2), suggesting that locating from text first and then checking the image is easier; in contrast, human performance is consistent in both directions. As the number of objects increases from 3 to 10, VLM performance drops significantly—Claude 3.5 lags behind humans by nearly 30 percentage points in 10-object scenes, whereas human performance remains stable.

### Ablation Study

| Model | Image→Image | Text→Text | Average |
|------|------------|----------|------|
| Human | 100.0 | 99.0 | 99.5 |
| Qwen2.5-VL | 99.7 | 99.4 | 99.5 |
| GPT-4o | 98.4 | 100.0 | 99.2 |
| Claude 3.5 | 97.3 | 100.0 | 98.7 |

**Key Comparison**: Qwen2.5-VL achieves 99.5% in single-modal tasks (on par with humans), but only 80.1% in cross-modal ones, showing a performance plunge of **19.4** percentage points. This proves that the root cause of the difficulty lies in **cross-modal linking** rather than single-modal attribute extraction, and single-modal performance is unaffected by the number of objects.

### CoT Prompting and Self-Reflection

| Model | All (+CoT) | $\Delta$ vs Standard | 10-Object Scenes |
|------|-----------|----------|-----------|
| Claude 3.5 | 86.2 | +2.9 | 70.5 |
| GPT-4o | 82.8 | +5.0 | 64.6 |
| Qwen2.5-VL | 78.9 | -1.2 | 62.8 |
| Llama 3.2 | 53.7 | +29.2 | 36.3 |

CoT significantly improves weaker models (Llama 3.2 +29.2) but provides limited help to stronger models. Even with CoT, the performance of all models still drops significantly as the number of objects increases. Comparing the self-reflection model VL-Rethinker-7B to its base Qwen2.5-VL also shows no significant improvement (79.6 vs 80.1), indicating that self-reflection techniques are ineffective for cross-modal entity linking.

### Linking Attribute Analysis

Analysis of Qwen2.5-VL in 7-object scenes reveals that accuracy is highest when only **1** linking attribute is required, and performance degrades when a combination of **2-3** attributes is needed. Performance is poorest when the only linking attribute is 3D coordinates, indicating that the model struggles to leverage spatial positions for cross-modal matching. The low OSE (Out-of-Scene Error) rate confirms that errors primarily originate from **entity linking errors** rather than hallucination.

## Key Findings

1. **Huge Discrepancy Between VLMs and Humans**: Humans approach 100%, while the best VLM still lags by about 15 percentage points.
2. **Difficulty Lies in Cross-Modal Linking Rather Than Single-Modal Search**: Single-modal is nearly perfect, whereas cross-modal performance plunges by nearly 20 percentage points.
3. **Number of Objects is the Key Difficulty Factor**: VLM performance drops linearly as the number of objects increases (increased feature interference), whereas human performance remains stable.
4. **Neither CoT nor Self-Reflection is the Fundamental Solution**: CoT brings only limited improvements, and self-reflection is entirely ineffective.
5. **Correspondence with the Binding Problem in Cognitive Science**: The difficulty of cross-modal alignment in VLMs can be analogized to the feature binding problem in cognitive science.

## Highlights & Insights

**Highlights**:  
(1) Locates the issue to cross-modal binding rather than single-modal perception through elegant controlled experimental design;  
(2) Synthesized data excludes confounding factors to keep evaluation pure;  
(3) The three-step decomposition analysis framework (visual search $\rightarrow$ linking attribute identification $\rightarrow$ textual search) is clear and reproducible;  
(4) The benchmark can be easily extended to scenes with more objects or multiple pointer/target attributes.

**Limitations**:  
(1) The gap between synthetic geometric scenes and the real world is relatively large;  
(2) Only simple geometric attributes are used, without testing semantically richer attributes;  
(3) The paper identifies the problem but does not explore solutions;  
(4) It only tests single-hop linking, without evaluating multi-hop cross-modal reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Defines a new task and designs an elegant controlled experiment, but the task itself is relatively basic
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Main results + single-modal ablation + CoT + self-reflection + linking attribute analysis + attribute type analysis, extremely comprehensive
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear logical chain (capability verification $\rightarrow$ problem localization $\rightarrow$ root cause analysis), elegant charts and figures
- **Value**: ⭐⭐⭐⭐ — Uncovers foundational capability defects of VLMs, providing an important benchmark and research direction for the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Performance Gap in Entity Knowledge Extraction Across Modalities in Vision Language Models](performance_gap_in_entity_knowledge_extraction_across_modalities_in_vision_langu.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](../../CVPR2025/multimodal_vlm/self-supervised_spatial_correspondence_across_modalities.md)
- [\[ACL 2025\] Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities](con_instruction_universal_jailbreaking_of_multimodal_large_language_models_via_n.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](negvqa_can_vision_language_models_understand_negation.md)

</div>

<!-- RELATED:END -->
