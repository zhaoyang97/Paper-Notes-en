---
title: >-
  [Paper Note] VinaBench: Benchmark for Faithful and Consistent Visual Narratives
description: >-
  [CVPR 2025][Visual Narrative] VinaBench is constructed to annotate commonsense links and discourse constraints for visual narrative samples, propose faithfulness and consistency evaluation metrics, and verify that utilizing these constraints substantially improves the quality of visual narrative generation.
tags:
  - "CVPR 2025"
  - "Visual Narrative"
  - "Consistency Evaluation"
  - "Commonsense Constraints"
  - "Discourse Structure"
  - "Benchmark Dataset"
date: 2026-05-08
content_hash: 8e9cc2283380a70b
---

# VinaBench: Benchmark for Faithful and Consistent Visual Narratives

**Conference**: CVPR 2025  
**arXiv**: [2503.20871](https://arxiv.org/abs/2503.20871)  
**Code**: [https://silin159.github.io/Vina-Bench](https://silin159.github.io/Vina-Bench)  
**Area**: LLM Evaluation  
**Keywords**: Visual Narrative, Consistency Evaluation, Commonsense Constraints, Discourse Structure, Benchmark Dataset

## TL;DR

VinaBench is constructed to annotate commonsense links and discourse constraints for visual narrative samples, propose faithfulness and consistency evaluation metrics, and verify that utilizing these constraints substantially improves the quality of visual narrative generation.

## Background & Motivation

**Background**: Visual Narrative Generation (VNG) aims to convert textual narratives into image sequences, widely applied in scenarios such as film storyboarding and educational illustrations. Existing approaches, such as ARLDM, StoryGen, and MM-Interleaved, primarily rely on pre-trained vision Transformers and diffusion models to learn the direct mapping from text to visual narratives.

**Limitations of Prior Work**: Existing methods suffer from two core issues: (1) **Narrative alignment**—textual narratives are typically abstract and visually under-described, requiring models to infer commonsense knowledge to generate visual content (e.g., "bad news" should manifest as a character's sad facial expression), but existing methods do not model this "manifestation gap" between text and vision. (2) **Visual consistency**—narrative elements such as character appearance, scene location, and time in an image sequence should remain consistent across frames, but existing methods do not explicitly learn these discourse constraints, leading to inconsistent outputs in terms of character appearance, background, etc.

**Key Challenge**: An inherent manifestation gap exists between textual and visual narratives, and visual narratives possess implicit discourse structural constraints, yet existing methods fail to explicitly model these constraints.

**Goal**: (1) Build a visual narrative benchmark dataset annotated with commonsense and discourse constraints; (2) design new metrics to evaluate faithfulness and consistency; (3) verify that learning these constraints can improve generation quality.

**Key Insight**: Utilize LLMs and VLMs (such as Llama3.1 and LLaVA-OV) to automatically annotate implicit commonsense links and discourse features in visual narrative samples, using these constraints as additional learning signals and evaluation dimensions.

**Core Idea**: Establish a dual-layer annotation framework of "commonsense links + discourse constraints" for visual narrative samples, leveraging these structured constraints as auxiliary training signals and evaluation standards to systematically improve the faithfulness and consistency of visual narratives.

## Method

### Overall Architecture

VinaBench comprises approximately 25K pairs of visual-textual narrative samples (sourced from VWP, Storyboard20K, and StorySalon), with two types of constraints annotated for each pair: (1) Commonsense constraints, which link visual entities in image descriptions to related entities in the textual narrative; (2) discourse constraints, including global features (character profiles, style) and scene features (characters present, time, location). New evaluation metrics for faithfulness and consistency are also proposed based on these constraints.

### Key Designs

1. **Commonsense Constraints Construction**:

    - **Function**: Bridges the manifestation gap between textual and visual narratives.
    - **Mechanism**: Completed in three steps. First, Mantis-Idefics2 is used to generate dense captions for each narrative image, with the textual narrative provided as context to prevent hallucination. Next, Llama3.1 is used to extract visual entities (noun/verb phrases) from the captions. Finally, Llama3.1 links each visual entity to its associated entity in the textual narrative, marking it as "no link" if no correlation exists. For instance, a "green shirt" in an image might have no corresponding textual entity, but "washing dishes" can link to "preparing dinner" in the narrative.
    - **Design Motivation**: Existing vision-language alignment research only focuses on general token/region-level matching, ignoring more implicit commonsense alignments in narrative contexts (e.g., "bad news" $\rightarrow$ sad expression). By explicitly annotating this alignment relationship, models can learn better manifestation strategies.

2. **Discourse Constraints Construction**:

    - **Function**: Explicitly represents the structural features of visual narratives to promote consistency.
    - **Mechanism**: Annotates two types of features. **Global features** include character profiles (name, age, gender, social role, persistent appearance features) and style (realistic, cartoon, comic, etc.). **Scene features** include characters present in each frame (detected in three steps using LLaVA-OV: counting characters first, then matching profiles), time period (morning/afternoon/evening/night/unknown), and location. Global features are expected to remain static throughout the narrative, while scene features track the dynamic changes of narrative elements.
    - **Design Motivation**: Just as natural language has syntactic structure, visual narratives also possess discourse structure (persistence and change of characters, time, and space). Explicitly annotating these structures provides clear optimization objectives and evaluation standards for consistency.

3. **New Evaluation Metric System**:

    - **Function**: Overcomes the limitations of traditional reference-based metrics by providing fine-grained evaluations of faithfulness and consistency.
    - **Mechanism**: Three types of metrics are designed: (1) **Alignment Ranking**: Uses CLIP-T or VQAScore to rank generated images from a top-100 pool of the test set, reporting MRR to avoid single-reference bias. (2) **Fine-Grained Alignment**: 5 VQA-based metrics evaluate the alignment of non-character entities, character counts, character attributes, time, and location. (3) **Consistency**: 3 VQA-based metrics check cross-frame consistency in style, character, and location. All VQA metrics utilize VLMs outputting in a Yes/No judgment format.
    - **Design Motivation**: Reference-based metrics like FID are biased toward specific reference images (e.g., the heroine's hair color), whereas the visual expression of a narrative is open-ended; absolute scores from CLIP-T are not comparable across samples. Ranking metrics combined with constraint-based VQA metrics provide a fairer and more fine-grained measurement of generation quality.

### Loss & Training

VinaBench itself is a benchmark rather than an independent model. In the experiments, the authors test three training settings on three generation models (ARLDM, StoryGen, and MM-Interleaved): (1) No Constraint—original training; (2) LLM Constraints—training with constraints generated by an LLM as additional inputs; (3) Gold Constraints—training with constraints annotated by VinaBench. Constraint information is concatenated into the model inputs as text.

## Key Experimental Results

### Main Results

| Model | Setting | FID↓ | CLIP-T MRR↑ | Consistency-Style | Consistency-Char | Consistency-Loc |
|------|------|------|-------------|------------|------------|------------|
| ARLDM | No Constraint | 42.6 | 0.110 | 0.466 | 0.379 | 0.376 |
| ARLDM | LLM Constraints | 37.6 | 0.151 | 0.859 | 0.551 | 0.689 |
| ARLDM | Gold Constraints | 35.3 | 0.155 | 0.854 | 0.569 | 0.697 |
| MM-Inter. | No Constraint | 48.3 | 0.066 | 0.947 | 0.582 | 0.449 |
| MM-Inter. | LLM Constraints | 42.2 | 0.111 | 0.986 | 0.678 | 0.764 |
| MM-Inter. | Gold Constraints | 39.3 | 0.118 | 0.976 | 0.688 | 0.856 |

### Ablation Study

Ablation based on MM-Interleaved + LLM Constraints:

| Setting | FID↓ | CLIP-T MRR↑ | Consistency-Loc |
|------|------|-------------|------------|
| Full (CL + DF) | 42.2 | 0.111 | 0.764 |
| w/o Commonsense Links (CL) | 42.9 | 0.109 | 0.758 |
| w/o Discourse Features (DF) | 43.3 | 0.107 | 0.684 |
| w/o Global Discourse Features | 42.6 | 0.110 | 0.760 |
| w/o Scene Discourse Features | 42.6 | 0.109 | 0.685 |
| Random Constraints | 53.7 | 0.048 | 0.447 |

### Key Findings

- Across all three models, integrating constraint learning consistently improves all metrics, with FID dropping by up to 26 points (StoryGen).
- Discourse constraints provide the most substantial improvement for consistency metrics (particularly location consistency, which increases from 0.449 to 0.764).
- Randomly shuffling constraints leads to a severe performance drop, validating the content value of constraint information rather than just the effect of increased input volume.
- Scene discourse features (per-image) contribute more to consistency than global discourse features.
- Expert evaluation shows that the automatic annotation accuracy of VinaBench is up to 85-95%, validating the reliability of the LLM/VLM construction pipeline.

## Highlights & Insights

- Pioneeringly introduces visual narrative structure theory (character/spatiotemporal tracking in discourse analysis) into generative tasks' constraint modeling.
- Comprehensive benchmark design: datasets (25K samples + constraint annotations) + metrics (alignment + consistency) + methodological validation.
- The pipeline design of leveraging LLMs/VLMs for large-scale automatic annotation can be generalized to other tasks requiring structural annotations.
- The approach of ranking-based evaluation instead of absolute scoring is worth adopting in other open-ended generation tasks.

## Limitations & Future Work

- Annotation quality depends on the capabilities of LLMs/VLMs, which may lead to omissions or errors in complex narratives.
- Currently, constraints are concatenated as text inputs; more structured constraint injection methods (such as graph structures) have not been explored.
- Evaluation metrics rely on VLM judgments, which may introduce the VLM's own biases.
- Significant room for improvement remains for *generative models* (with obvious gaps compared to human references); constraint learning is only one direction.
- Future work could consider applying similar discourse constraint frameworks to video generation scenarios.

## Related Work & Insights

- Unlike physical commonsense like ConceptNet, this work focuses on more implicit commonsense alignment in narrative contexts ("bad news" $\rightarrow$ sad expression).
- The concept of discourse constraints can be used for reference to maintain character/scene consistency in long video generation.
- The ranking-based MRR evaluation method can be generalized to the evaluation of other open-ended image generation tasks.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 4.5 |
| Writing Quality | 4 |
| Overall Rating | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [\[CVPR 2025\] Scene-Agnostic Pose Regression for Visual Localization](scene-agnostic_pose_regression_for_visual_localization.md)
- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](../../NeurIPS2025/others/face_faithful_automatic_concept_extraction.md)
- [\[CVPR 2025\] CARE Transformer: Mobile-Friendly Linear Visual Transformer via Decoupled Dual Interaction](care_transformer_linear_attention.md)
- [\[ACL 2025\] Consistent Client Simulation for Motivational Interviewing-based Counseling](../../ACL2025/others/consistent_client_simulation_for_motivational_interviewing-based_counseling.md)

</div>

<!-- RELATED:END -->
