---
title: >-
  [Paper Note] More than the Sum: Panorama-Language Models for Adverse Omni-Scenes
description: >-
  [CVPR 2026][Multimodal VLM][Panoramic image understanding] This paper proposes the Panorama-Language Modeling (PLM) paradigm and the PanoVQA large-scale panoramic VQA dataset (653K QA pairs). A plug-and-play panoramic sp…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Panoramic image understanding"
  - "360° vision"
  - "VQA"
  - "sparse attention"
  - "autonomous driving"
date: 2026-05-08
content_hash: 1d4485485fa97b93
---

# More than the Sum: Panorama-Language Models for Adverse Omni-Scenes

**Conference**: CVPR 2026
**arXiv**: [2603.09573](https://arxiv.org/abs/2603.09573)  
**Code**: [https://github.com/InSAI-Lab/PanoVQA](https://github.com/InSAI-Lab/PanoVQA)  
**Area**: Multimodal VLM
**Keywords**: Panoramic image understanding, 360° vision, VQA, sparse attention, autonomous driving

## TL;DR
This paper proposes the Panorama-Language Modeling (PLM) paradigm and the PanoVQA large-scale panoramic VQA dataset (653K QA pairs). A plug-and-play panoramic sparse attention (PSA) module is designed to enable existing VLMs to process equirectangular projection (ERP) panoramic images without retraining, achieving superior global reasoning over multi-view stitching approaches in adverse scenarios such as occlusion and accidents.

## Background & Motivation

**Background**: VLMs (e.g., LLaVA, BLIP-2) have achieved strong performance on pinhole images, yet panoramic (360°) inputs are increasingly prevalent in real-world applications such as autonomous driving, robotics, and AR/VR. Existing methods adopt a "stitching" strategy: sampling multiple narrow-field-of-view crops, processing them independently, and concatenating the results.

**Limitations of Prior Work**: Multi-view stitching disrupts the spatial continuity of 360° scenes, ignoring global spatial relationships (e.g., left-right boundary connectivity) and failing to model wrap-around properties. For instance, multi-camera systems may miss a hazardous vehicle at the front-left because it straddles the boundary between two views.

**Key Challenge**: (1) The absence of large-scale panoramic VQA benchmarks — existing datasets are either multi-view pinhole VQA or panoramic data without QA pairs. (2) Architectural incompatibility — ERP images exhibit severe geometric distortion and significantly higher resolution than pinhole images, making the $O(n^2)$ complexity of dense attention intractable.

**Goal**: To validate the hypothesis that "panoramic language understanding exceeds the sum of multi-view stitching" and to establish foundational infrastructure for 360° VLMs.

**Key Insight**: The empirical observation that a single panoramic image (41.42%) outperforms six camera images (40.22%).

**Core Idea**: Construct the PanoVQA dataset and design the panoramic sparse attention (PSA) module to enable existing VLMs to directly process panoramic inputs.

## Method

### Overall Architecture
The PLM framework consists of two components: (1) PanoVQA dataset construction — 653K QA pairs covering three scenario types: normal driving, occlusion, and accidents; (2) Panoramic Sparse Attention (PSA) — a plug-and-play module that replaces dense attention in VLMs to handle ERP image distortion and long-range dependencies.

### Key Designs

1. **PanoVQA Dataset**:

    - **Function**: The first large-scale panoramic VQA benchmark.
    - **Composition**: PanoVQA-N (normal scenes, sourced from NuScenes, four task categories N1–N4) + PanoVQA-O (occlusion scenes, sourced from BlendPASS, three task categories O1–O3) + PanoVQA-D (accident scenes, sourced from DeepAccident, five task categories D1–D5), totaling 12 VQA task categories.
    - **Panorama generation**: A purely geometric pipeline stitches multi-camera images into panoramas for NuScenes and DeepAccident.
    - **QA generation**: QA pairs are generated using GPT-5-mini (reasoning effort = minimal/low) from structured annotations, followed by automatic and manual quality control.
    - **Scale**: 44.6K frames, 653K QA pairs.

2. **Panoramic Sparse Attention (PSA)**:

    - **Function**: Enables VLMs to efficiently process the high resolution and distortion of ERP panoramic images.
    - **Mechanism**: Queries dynamically learn which tokens to attend to, rather than performing full token interactions as in dense attention, reducing complexity while adapting to the topological structure of ERP projection.
    - **Design Motivation**: Distortion in ERP increases with latitude, and fixed-window attention cannot reflect true proximity relationships on the sphere.
    - **Compatibility**: Plug-and-play; no retraining of pretrained models is required.

3. **Object Annotation Quadruple Format**:

    - (category, direction, distance, visibility/speed)
    - Both machine-readable and intuitive, supporting diverse tasks such as spatial reasoning, occlusion inference, and collision risk assessment.

### Loss & Training
- The PSA module can be fine-tuned on PanoVQA or directly inserted into existing inference pipelines.
- Supports per-scenario evaluation across the PanoVQA-N, PanoVQA-O, and PanoVQA-D subsets.

## Key Experimental Results

### Main Results

| Method | Input | PanoVQA Accuracy | Note |
|--------|-------|-----------------|------|
| Existing VLM (6-cam) | 6 pinhole images | 40.22% | Multi-view stitching |
| PLM (1-Pano) | 1 panoramic image | **41.42%** | Panoramic model |
| All other models | — | Below PLM | Trailing across all categories |

### Scene-Level Performance

| Scene | Description | PLM Advantage |
|-------|-------------|--------------|
| Normal (N) | Scene description, object recognition, spatial relations | Clear advantage in spatial reasoning |
| Occlusion (O) | Occlusion relationship inference | Global context aids inference of occluded objects |
| Accident (D) | Collision risk, evasion decisions | 360° field of view eliminates blind spots |

### Ablation Study
- Replacing PSA with dense attention substantially increases inference cost and reduces accuracy.
- The 1-Pano setting significantly outperforms the 6-cam setting on directional judgment tasks, where the latter frequently misidentifies directions.
- All three scenario types in the dataset challenge existing models, with accident scenes being the most difficult.

## Highlights & Insights
- Introduces the first panoramic VLM paradigm (PLM), with experimental evidence that panoramic understanding exceeds the sum of multi-view stitching.
- PanoVQA is the first large-scale benchmark combining panoramic images with VQA, including rare occlusion and accident scenarios.
- The PSA module is plug-and-play and requires no retraining of existing VLMs, lowering the adoption barrier.
- The dataset construction pipeline is reusable (NuScenes/BlendPASS/DeepAccident → panoramic VQA).
- The 12 VQA task categories span scene description, spatial reasoning, occlusion inference, and collision assessment.
- The object triplet/quadruple representation format is both machine-readable and intuitive, facilitating reuse in future research.

## Limitations & Future Work
- Panorama stitching quality is limited by the alignment accuracy of original multi-camera setups and handling of occluded regions.
- The specific attention pattern design and hyperparameters of PSA require further ablation.
- PanoVQA currently focuses on driving scenes; coverage of indoor, pedestrian, AR/VR, and other settings is insufficient.
- Future work may explore integrating PLM with BEV representations to combine the global advantage of panoramic images with the precise localization of BEV.
- Severe distortion in ERP projection at the poles (sky/ground) may impair understanding in those regions.
- PanoVQA-O (occlusion scenes) and PanoVQA-D (accident scenes) have relatively small sample sizes (<1.3K and <144K, respectively), potentially insufficient for training large models.

### QA Generation Quality
- QA pairs are generated by GPT-5-mini and subjected to dual quality control via automated machine filtering and human evaluation.
- Average question length is 15.6 words and average answer length is 34.8 words, substantially exceeding single-word-answer benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage](../../ACL2026/multimodal_vlm/more_than_meets_the_eye_measuring_the_semiotic_gap_in_vision-language_models_via.md)
- [\[CVPR 2026\] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)
- [\[CVPR 2026\] VLM-Loc: Localization in Point Cloud Maps via Vision-Language Models](vlm-loc_localization_in_point_cloud_maps_via_vision-language_models.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)

</div>

<!-- RELATED:END -->
