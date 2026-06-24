---
title: >-
  [Paper Note] Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][3D scene change understanding] This work introduces the Situat3DChange dataset (174K data instances) that unifies dynamic scene change perception and situated awareness understanding under a perception–action paradigm, and proposes SCReasoner—an efficient 3D MLLM for point cloud comparative reasoning.
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "3D scene change understanding"
  - "situation awareness"
  - "multimodal large language models"
  - "point cloud comparison"
  - "dataset"
date: 2026-05-08
content_hash: 3d045d457dd06548
---

# Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2510.11509](https://arxiv.org/abs/2510.11509)  
**Code**: [https://github.com/RuipingL/Situat3DChange](https://github.com/RuipingL/Situat3DChange)  
**Area**: Multimodal VLM
**Keywords**: 3D scene change understanding, situation awareness, multimodal large language models, point cloud comparison, dataset

## TL;DR

This work introduces the Situat3DChange dataset (174K data instances) that unifies dynamic scene change perception and situated awareness understanding under a perception–action paradigm, and proposes SCReasoner—an efficient 3D MLLM for point cloud comparative reasoning.

## Background & Motivation

Physical environments are inherently dynamic, yet existing 3D datasets either focus on dynamic scenes (object change detection) or dynamic situations (viewpoint-situated reasoning), lacking a unified framework that integrates both. Specifically, 3D situated reasoning datasets (SQA3D, MSQA) assume static scenes, while 3D change understanding datasets (Dy2Change, ChangeSim) rely on synthetic data and lack situational context.

The authors raise two critical questions: (Q1) Do LLM-generated data genuinely reflect human shared mental maps and situational awareness? Interviews with 30 participants of diverse backgrounds (including two visually impaired individuals) reveal that humans perceive space in **cylindrical coordinates** (specifying a reference direction when saying "left/right," and treating "front" as closer to the observer), whereas robotics and engineering conventionally use Cartesian coordinates—a fundamental cognitive discrepancy. (Q2) Can scene graphs effectively represent spatial changes? 3DSSG lacks sensitivity to object rotations and subtle displacements of approximately 10 cm.

## Method

### Overall Architecture

Situat3DChange follows a perception–action model: **perception** comprises 121K QA pairs + 36K change descriptions, and **action** comprises 17K rearrangement instructions. It is built upon 903 real-world scan pairs from 3RScan, grounded in 11K human annotations, integrating egocentric and allocentric perspectives as well as categorical/coordinate spatial relations, and expanded into full situated data via LLMs.

### Key Designs

1. **Human Cognition-Aligned Data Construction**: Seven collaborators experienced in assisting visually impaired individuals annotate each changed object in 3RScan across four fields: change reason (Reason), warning information (Warning), change description (Description), and rearrangement instruction (Rearrangement). Annotations incorporate horizontal cues from the allocentric perspective. Vertical relations and object attributes are then extracted, egocentric positional changes are computed, and GPT-4 is used to generate complete situated change descriptions and rearrangement instructions. Change descriptions are arranged clockwise and expressed in meters; rearrangement instructions are expressed in steps (more accessible for visually impaired users).

2. **Distinctive Feature Query**: To uniquely identify changed objects within a scene, objects that are unique in the scene are first promoted to landmarks, and three candidate distinguishing features are then extracted for the remaining objects: distinctive color, horizontal extremity (nearest/farthest relative to the landmark), and vertical spatial relation. Human review supplements features where necessary to ensure each query uniquely identifies the target object.

3. **SCReasoner Architecture**: Addressing the challenge of effectively comparing two highly similar point clouds, existing methods concatenate all modality tokens before the decoder input—causing redundancy for similar point clouds. SCReasoner leverages **Mamba's selectivity** to filter informative tokens from the previous scene's point cloud, then fuses them with current scene tokens via the **star operation (element-wise multiplication)**, without introducing additional tokens to the language decoder at minimal parameter overhead. The architecture builds on the LEO framework, adding only a selective comparison projector.

### Loss & Training

The training setup follows LEO, with joint training for 5 epochs across three tasks in Situat3DChange. Evaluation employs a multi-dimensional metric system: long-text tasks use CIDEr/BLEU-4/METEOR/ROUGE/sentence embedding similarity/GPT score; QA tasks use GPT score (normalized to 1–5); distance-type questions adopt an improved REL metric that resolves the division-by-zero issue of conventional REL when the ground-truth distance is zero.

## Key Experimental Results

### Main Results

| Task | Metric | SCReasoner (mamba*) | LEO | InternVL2-7B (FT) |
|------|--------|-------------------|-----|------------------|
| Change Description | GPT Score | 13.9% | 12.7% | 8.2% |
| Rearrangement Instruction | GPT Score | 30.7% | 30.1% | 16.3% |
| QA (average) | GPT Score | Best | 2nd Best | — |

2D MLLMs vs. 3D MLLMs: 3D models perform better on allocentric understanding, while fine-tuned 2D models perform slightly better on egocentric tasks (distance/direction).

### Ablation Study

| Configuration | Change Desc. GPT | Rearrangement GPT | Note |
|---------------|-----------------|-------------------|------|
| LEO (baseline) | 12.7 | 30.1 | Naive concatenation of two point clouds |
| SCReasoner (linear+) | 12.6 | 30.3 | Linear projection + additive fusion |
| SCReasoner (linear*) | 13.4 | 30.3 | Linear projection + multiplicative fusion |
| SCReasoner (mamba+) | 13.3 | 30.5 | Mamba projection + additive fusion |
| SCReasoner (mamba*) | **13.9** | **30.7** | Mamba projection + multiplicative fusion, best |

### Key Findings

- Mamba's selectivity outperforms simple linear projection, effectively filtering change-relevant informative tokens from similar point clouds.
- The star operation (multiplicative fusion) outperforms additive fusion; its ability to map inputs to higher-dimensional representations helps highlight differences.
- Panoramic images as 2D MLLM input show some feasibility, but the lack of comprehensive allocentric context leads to poor performance on long-text tasks.
- Zero-shot and one-shot MLLMs perform poorly on change understanding tasks, indicating that specialized fine-tuning is required for scene change understanding.
- Data scaling experiments demonstrate a positive scaling effect, and cross-domain transfer experiments confirm the task-agnostic training value of the dataset.

## Highlights & Insights

- **Human-centered data construction**: Pioneering integration of egocentric/allocentric perspectives and categorical/coordinate spatial relations, with cognitive alignment ensured through annotators experienced in assisting visually impaired individuals.
- **Incisive critique of existing LLM-generated data**: Reveals the cognitive gap between Cartesian and cylindrical coordinate systems and the insensitivity of scene graphs to subtle changes.
- **Minimalist and efficient SCReasoner**: Achieves effective point cloud comparison with only a small number of additional parameters, without introducing extra tokens to the decoder.
- The improved REL distance evaluation metric resolves the division-by-zero problem at zero distance, offering general utility for the scene understanding field.
- Three tasks (QA, change description, rearrangement instruction) are unified under the perception–action model with a comprehensive evaluation framework.
- Multi-layer quality control mechanisms (human annotation + automatic cross-validation + GPT expansion) ensure data quality.

## Limitations & Future Work

- Reliance on the 3RScan dataset limits scene scale (903 scan pairs), with a predominant focus on indoor scenes.
- Human annotations are provided by only 7 collaborators, potentially introducing annotator bias and cultural background limitations.
- SCReasoner's improvement over LEO is relatively modest (~1–2%), indicating substantial room for further advances in point cloud comparison methods.
- Dynamic change understanding based on video sequences or continuous frames is not explored.
- Test set labels for 3RScan are not publicly available, necessitating evaluation on the validation set only, which may affect evaluation fairness.
- The panorama rendering method relies on cubemap projection, which may introduce geometric distortions.
- The improved REL metric for distance evaluation, while resolving the zero-distance issue, may be overly tolerant for very large distances.
- Broader comparisons with recent 3D foundation models (e.g., PointLLM, 3D-LLM) are absent.

## Related Work & Insights

- Complements situated QA datasets such as SQA3D and MSQA by filling the gap of "dynamic scenes + situated awareness."
- The Mamba + star operation combination in SCReasoner is generalizable to other scenarios requiring comparison of two similar inputs (e.g., video understanding, before/after comparison).
- Has direct application value for assistive technologies (navigation for the visually impaired, indoor rearrangement robots).
- The human cognition-aligned data construction approach (cylindrical coordinate perception, step-based rearrangement instructions) provides important reference for embodied AI dataset design.
- The improved distance REL evaluation metric has general reference value for all tasks involving distance regression.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First 3D dataset integrating dynamic scenes + situated awareness, with a unique cognitive alignment perspective
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive multi-task/multi-baseline evaluation, including scaling and transfer experiments
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is deeply articulated; data construction pipeline is described in detail
- **Value**: ⭐⭐⭐⭐ Dataset fills an important gap with significant implications for embodied AI and assistive technology

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](../../CVPR2025/multimodal_vlm/sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[NeurIPS 2025\] SmartWilds: Multimodal Wildlife Monitoring Dataset](smartwilds_multimodal_wildlife_monitoring_dataset.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](../../ACL2025/multimodal_vlm/vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)

</div>

<!-- RELATED:END -->
