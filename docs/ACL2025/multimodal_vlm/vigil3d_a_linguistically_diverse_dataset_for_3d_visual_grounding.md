---
title: >-
  [Paper Note] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding
description: >-
  [ACL 2025][Multimodal VLM] Proposes ViGiL3D—a linguistically diverse diagnostic dataset and an automatic analysis framework—to evaluate 3D visual grounding (3DVG) methods across various linguistic phenomena such as negation, coarse-grained references, and coreference resolution, revealing a significant performance drop (up to 20+ points) of existing methods on out-of-distribution prompts.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
date: 2026-05-08
content_hash: 383e2e7feb9c6beb
---

# ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding

**Conference**: ACL 2025  
**arXiv**: [2501.01366](https://arxiv.org/abs/2501.01366)  
**Code**: None  
**Area**: Multimodal VLM  

## TL;DR

Proposes ViGiL3D—a linguistically diverse diagnostic dataset and an automatic analysis framework—to evaluate 3D visual grounding (3DVG) methods across various linguistic phenomena such as negation, coarse-grained references, and coreference resolution, revealing a significant performance drop (up to 20+ points) of existing methods on out-of-distribution prompts.

## Background & Motivation

1. **Linguistic monotony in 3DVG datasets**: Descriptions in existing mainstream datasets (ScanRefer, Nr3D, Sr3D) are dominated by direct, simple sentence structures. They do not comprehensively cover sentence structures and linguistic phenomena, failing to reflect the diverse expressions of users in real-world applications.
2. **LLM scaling fails to address diversity**: Although recent works use LLMs (e.g., GPT) to automatically generate large volumes of 3DVG prompts, scaling up has left the language patterns highly homogenized—characterized by excessive attributes, overly specific target class names, and a lack of negative sentences.
3. **Severe lack of negative prompts**: Most datasets contain almost no negative sentences (e.g., "find the food storage cabinet that has no green objects on it"), even though negation is a crucial natural language phenomenon where models must exclude objects based on "what they are not."
4. **Monotonous granularity in target reference**: In the vast majority of datasets, target objects appear with precise class names (e.g., "chair") and rarely use generalized references (e.g., "object", "equipment"). This allows models to bypass other semantic signals by simply matching the class name.
5. **Lack of systematic linguistic evaluation frameworks**: Prior to this work, there was no systematic framework to quantitatively analyze the distribution and coverage of various linguistic phenomena (such as attribute types, relation types, reference styles, and anchor types) in 3DVG datasets.
6. **Downstream application requirements**: In applications like robotics and AR/VR conversational assistants, user descriptions are complex and volatile, involving viewpoint dependency, spatial arrangement, and ordinal relations, necessitating a diagnostic benchmark to measure true model capabilities.

## Method

### Linguistic Analysis Framework

The authors design an automatic linguistic analysis pipeline containing **35 metrics** to comprehensively measure 3DVG prompts across three dimensions:

- **Linguistic Diversity (DIV)**: Attribute type coverage (color, size, shape, quantity, material, function, texture, style, text tag, state, total of 10 types), relation type coverage (near/far/direction/vertical/contain/arrangement, total of 6 types), lexical bigram unique rate.
- **Linguistic Resolution (RES)**: Separation and parsing of target attributes from anchor attributes, separation of target relations from anchor relations, coreference resolution detection, and whether the target is the first noun phrase.
- **Attribute and Relation Understanding (UAR)**: Overall statistics of attributes/relations, ratio of generalized/coarse-grained/fine-grained references, proportion of single-object/multi-object/non-object/viewpoint anchors, and negation sentence ratio.

The pipeline uses **GPT-4o** to extract enhanced scene graphs (objects, attributes, relations) and **SpaCy** for dependency parsing to measure bigram diversity. Validated on 225 manual annotations, 28 binary metrics achieve an average precision of 0.86 and recall of 0.91.

### ViGiL3D Dataset Construction

- **Scene Sources**: ScanNet (26 scenes) + ScanNet++, where the former controls distribution for comparability with existing methods, and the latter provides higher-quality point clouds.
- **Annotation Method**: Manual annotation, where annotators author localization descriptions for sampled objects based on RGB video streams and 3D point clouds.
- **Annotation Protocols**: Cover multiple linguistic patterns (negation, generalized reference, coarse-grained reference, viewpoint anchor, ordinal relation, etc.), using natural phrasing with moderate constraints (avoiding ambiguity without over-specifying).
- **Zero-target Descriptions**: Designed to be similar to real-object descriptions in the scene but modified to be inapplicable, which is more challenging than simply describing non-existent categories.
- **Dataset Scale**: 350 prompts, 35 scenes, vocabulary size of 942, and support for 0/1/multi-target localization.
- **Diversity Edge**: Achieves the best coverage across all 35 metrics, with a lexical bigram ratio of 0.45, far exceeding other datasets (which reach a max of 0.28).

### Evaluated 3DVG Methods

Encompasses 7 open-vocabulary methods across three categories:
1. **CLIP-aligned 3D representations**: OpenScene (direct point cloud projection), LERF (NeRF)
2. **LLM zero-shot reasoning**: ZSVG3D (program synthesis), LLM-Grounder (natural language reasoning)
3. **3DVG-data trained models**: 3D-VisTA, 3D-GRAND (trained on LLM-expanded data), PQ3D (trained on aggregated human-annotated datasets)

## Key Experimental Results

### Table 1: Performance Comparison between ViGiL3D and ScanRefer (ScanNet Scenes, %)

| Method | ViGiL3D Acc/GT | ViGiL3D F1/GT | ScanRefer Acc@25 | ScanRefer Acc@50 |
|:---|:---:|:---:|:---:|:---:|
| OpenScene | 2.1 | 2.1 | 13.2 | 6.5 |
| LERF | 2.5 | 2.5 | 4.8 | 0.9 |
| ZSVG3D | 18.9 | 12.2 | 36.4 | 32.7 |
| LLM-Grounder | 2.5 | 2.5 | 17.1 | 5.3 |
| 3D-VisTA | 14.2 | 14.1 | 50.6 | 45.8 |
| 3D-GRAND | 17.9 | 17.9 | 38.0 | 27.4 |
| **PQ3D** | **26.2** | **26.8** | **57.0** | **51.2** |

The best-performing model PQ3D experiences a **24.4 percentage point** drop in F1 on ViGiL3D compared to ScanRefer, verifying that existing methods exhibit severe deficiencies when handling linguistically diverse prompts.

### Table 2: Subgroup Analysis—Acc/GT of Different Linguistic Phenomena (%)

| Method | Overall | Negation | Generalized Reference | Coarse-grained | Text Tag | Ordinal Relation | Comparative Relation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenScene | 2.1 | 8.1 | 2.5 | 1.9 | 4.0 | 0.0 | 0.0 |
| ZSVG3D | 18.9 | 10.8 | 15.8 | 13.2 | 12.0 | 19.2 | 25.0 |
| 3D-GRAND | 17.9 | 21.6 | 7.5 | 13.2 | 4.0 | 14.8 | 18.4 |
| PQ3D | 26.2 | 13.5 | 20.0 | 24.5 | 8.0 | 7.4 | 24.5 |

No single model consistently outperforms others across all subgroups; text tags are extremely challenging for all methods (due to insufficient point cloud resolution); negative prompts universally lead to degraded performance.

## Highlights & Insights

- **First systematic 3DVG linguistic diversity analysis framework**: 35 metrics in three dimensions (DIV/RES/UAR) provide a comprehensive measurement, with a highly precise and scalable automated pipeline.
- **Novel concept of diagnostic dataset design**: Prioritizes coverage over mere scale, with 350 meticulously annotated prompts sufficient to expose major defects across all models.
- **Subgroup analysis offers actionable improvements**: Clear vulnerability exposures in negation, generalized reference, text tags, etc.
- **Demonstrated "data scale $\neq$ capability enhancement"**: Training on aggregated human-annotated datasets (PQ3D) outperforms models trained on LLM-expanded data.

## Limitations & Future Work

- **Small dataset scale** (only 350 prompts, 35 scenes): The reliability of statistical conclusions is limited, and the sample sizes in subgroups may be insufficient to support robust conclusions.
- **Limited to English and indoor scenes**: Outdoor or multilingual scenes are not involved, and cultural differences are not considered.
- **Difficulty in VLM scaling**: The authors attempted to use VLMs for automatic generation of diverse descriptions, but with poor results—VLMs lack 3D spatial understanding, generate unnatural diverse descriptions, and struggle to reliably distinguish between object instances of the same category.
- **Only open-vocabulary methods evaluated**: Comparisons with closed-set models are omitted, limiting the scope of evaluation.

## Related Work & Insights

| Dimension | ViGiL3D | ScanRefer (Chen et al., 2020) | SceneVerse (Jia et al., 2024) |
|:---|:---|:---|:---|
| Prompt Source | Human annotators, emphasizing diversity | Crowdsourced annotation | LLM + template automatic generation |
| Linguistic Diversity | bigram 0.45, covering all phenomena | bigram 0.20, lacking negation/generalized reference | bigram 0.16, very few attributes/relations |
| Target Reference | Generalized + coarse-grained + fine-grained all present | Almost completely fine-grained | Almost completely fine-grained |
| Negation | ✓ (significant proportion) | ✗ | ✗ |
| Design Goal | Diagnostic evaluation, exposing weaknesses | General purpose training + evaluation | Large-scale pre-training |

## Rating

- ⭐⭐⭐⭐ Novelty: First to systematically quantify the linguistic diversity gap in 3DVG and build a diagnostic benchmark
- ⭐⭐⭐⭐ Value: The analysis framework can be directly reused to evaluate any 3DVG dataset, with subgroup analysis guiding model improvements
- ⭐⭐⭐ Experimental Thoroughness: 7 methods span across three categories, though the dataset scale limits statistical significance
- ⭐⭐⭐⭐ Writing Quality: Well-structured, rich with tables and visualizations, and elaborates thoroughly on the linguistic analysis framework

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)
- [\[ACL 2025\] Aria-UI: Visual Grounding for GUI Instructions](aria-ui_visual_grounding_for_gui_instructions.md)
- [\[ICCV 2025\] ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition](../../ICCV2025/multimodal_vlm/viewsrd_3d_visual_grounding_via_structured_multi-view_decomposition.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)
- [\[ACL 2025\] ConECT Dataset: Overcoming Data Scarcity in Context-Aware E-Commerce MT](conect_dataset_overcoming_data_scarcity_in_context-aware_e-commerce_mt.md)

</div>

<!-- RELATED:END -->
