---
title: >-
  [Paper Note] The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation
description: >-
  [CVPR 2025][LLM (Other)][Semantic change detection] This paper proposes HySCDG (Hybrid Semantic Change Detection Data Generation), a hybrid data generation pipeline that combines real very-high-resolution (VHR) remote sensing imagery with image inpainting techniques to generate large-scale training data for semantic change detection, achieving strong temporal and spatial generalization capabilities under a simple architectural design.
tags:
  - "CVPR 2025"
  - "LLM (Other)"
  - "Semantic change detection"
  - "VHR imagery"
  - "Hybrid data generation"
  - "Inpainting"
  - "Land cover"
date: 2026-05-08
content_hash: cdb6c5edd8152511
---

# The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation

**Conference**: CVPR 2025  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Semantic change detection, VHR imagery, Hybrid data generation, Inpainting, Land cover

## TL;DR

This paper proposes HySCDG (Hybrid Semantic Change Detection Data Generation), a hybrid data generation pipeline that combines real very-high-resolution (VHR) remote sensing imagery with image inpainting techniques to generate large-scale training data for semantic change detection, achieving strong temporal and spatial generalization capabilities under a simple architectural design.

## Background & Motivation

### Background

**Background**: Bi-temporal change detection is a core capability in Earth observation, used for urban expansion monitoring, disaster assessment, agricultural change tracking, etc. Semantic change detection based on VHR (Very High Resolution) imagery must not only identify "where has changed" (binary change detection) but also "what it has changed into" (semantic categories), such as "farmland $\rightarrow$ building".

**Limitations of Prior Work**: (1) Severe lack of annotated data—semantic change detection requires pixel-level semantic annotations for images at two different time points, which is highly expensive. Existing datasets (e.g., SECOND, HRSCD) are small and have limited geographic coverage. (2) Disconnection between binary and semantic detection—most works focus on simple binary change detection (changed/unchanged), whereas practical applications require semantic-level changes. (3) Unrealistic synthetic data—purely synthetic data (via style transfer, GAN generation, etc.) exhibits a significant domain gap compared to real remote sensing imagery, leading to poor generalization of models trained on synthetic data. (4) Complex architectures—many methods employ complex multi-branch architectures that require large amounts of annotated data to train.

**Key Challenge**: Semantic change detection requires large-scale and diverse paired annotated data (two time points, pixel-level semantics), but manual annotation is prohibitively expensive; meanwhile, purely synthetic data lacks realism.

**Goal**: How to generate large-scale, high-quality training data for semantic change detection at a low cost and train a detection model with strong generalization capability?

**Key Insight**: Utilizing a "hybrid" strategy—selecting unchanged regions from real remote sensing imagery as the base, modifying specific regions into other land cover types via image inpainting, and simultaneously automatically generating change labels. Real background map + localized synthetic changes = hybrid data.

**Core Idea**: Locally synthesize land cover changes on real VHR remote sensing images using inpainting, automatically generate change labels, and construct large-scale hybrid training data.

## Method

### Overall Architecture

The HySCDG pipeline: (1) Collect a large amount of real VHR remote sensing imagery and its corresponding land cover semantic maps. (2) Select the change types of interest (e.g., "vegetation $\rightarrow$ building") and locate the corresponding regions in the semantic maps. (3) Employ an inpainting model (e.g., SD-Inpaint) to generate the texture of the target class in the selected region (e.g., repainting vegetation areas into building textures) while keeping the surrounding environment unchanged. (4) The original image serves as T1 and the modified image as T2; the difference in the semantic maps automatically constitutes the change annotation. (5) Train a simple semantic change detection network on the generated hybrid data.

### Key Designs

1. **Hybrid Data Generation Strategy**:
    - **Function**: Generate large-scale, diverse paired data for semantic change detection at low cost.
    - **Mechanism**: Utilize existing remote sensing image semantic segmentation datasets (requiring only single-temporal annotations) to "manually manufacture" land cover changes through inpainting. For each image, randomly select some semantic regions and repaint them as another land cover type using the inpainting function of diffusion models. Diverse changes are generated by controlling the inpainting prompts (e.g., "urban buildings", "farmland") and the shapes of the regions. Key constraint: the boundaries of the modified regions should transition naturally, and unmodified regions must remain consistent.
    - **Design Motivation**: Traditional synthetic data is entirely generated by models, introducing a domain gap across the entire image. In contrast, hybrid data only modifies localized regions, preserving most of the real image information and drastically reducing the domain gap.

2. **Semantic-Consistency-Constrained Inpainting**:
    - **Function**: Ensure that the inpainting results are semantically and visually consistent with remote sensing scenes.
    - **Mechanism**: Apply conditional control to the inpainting model: (1) Text prompts specify the target land cover types. (2) Surrounding pixels serve as context conditions. (3) Post-processing verification—use a pre-trained semantic segmentation model to verify whether the inpainted region is indeed classified as the target class, filtering out inconsistent samples. Run multiple samplings to select the results with the highest semantic consistency.
    - **Design Motivation**: General-purpose inpainting models have limited understanding of remote sensing scenes, which may lead to textures that do not conform to remote sensing features (e.g., generating street-view style buildings instead of bird's-eye view rooftops).

3. **Simple Change Detection Architecture**:
    - **Function**: Verify the effectiveness of the hybrid data.
    - **Mechanism**: Adopt a simple architecture containing a Siamese encoder, feature differences, and a segmentation decoder to avoid complex designs. The encoder is based on a pre-trained backbone (such as ResNet or ViT). After extracting bi-temporal features, it calculates the differences or concatenates them, outputting semantic change maps through a lightweight decoder. The purpose of a simple architecture is to prove the value of hybrid data rather than relying on complex models. Meanwhile, pre-training is performed on multiple real remote sensing scenes to enhance spatial generalization.
    - **Design Motivation**: Complex architectures might mask the actual impact of data quality.

## Key Experimental Results

### Key Findings

- On the SECOND and HRSCD test sets, the simple model trained with hybrid data achieved or outperformed complex models trained with real annotations.
- Compared to purely synthetic data, hybrid data improved the mIoU of semantic change detection on real test sets by approximately 10-15%.
- Temporal generalization: Training in one city yields good performance on images of the same city in different years.
- Spatial generalization: Training in one region still delivers reasonable detection performance on VHR images from other continents.
- The quality of inpainting significantly impacts the final detection performance; semantic consistency filtering can boost mIoU by about 5%.

## Highlights & Insights

- **Paradigm Value of Data Generation**: It possesses broad reference significance for the remote sensing field, which suffers from a severe lack of annotated data.
- **Hybrid Strategy as the Key**: The concept of "real background map + localized synthesis" effectively narrows the synthetic-to-real domain gap.
- **Simplicity Proves Effectiveness**: It does not rely on complex architectures, instead driving performance through data.

## Limitations & Future Work

- The generation quality of inpainting models for views unique to remote sensing (bird's-eye view) still has room for improvement.
- Currently, mainly urbanization-related change types have been verified; more change types (e.g., flooding, deforestation) remain to be tested.
- The boundary transitions of changed regions in hybrid data may not be as natural as real changes.
- Future work can incorporate real temporal satellite imagery for semi-supervised learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](../../ICML2026/llm_nlp/token-efficient_change_detection_in_llm_apis.md)
- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](../../ACL2025/llm_nlp/explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[CVPR 2025\] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)
- [\[ACL 2025\] Theorem Prover as a Judge for Synthetic Data Generation](../../ACL2025/llm_nlp/theorem_prover_as_a_judge_for_synthetic_data_generation.md)
- [\[ACL 2025\] Literature Meets Data: A Synergistic Approach to Hypothesis Generation](../../ACL2025/llm_nlp/literature_meets_data_hypothesis.md)

</div>

<!-- RELATED:END -->
