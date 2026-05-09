---
title: >-
  [Paper Note] ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data
description: >-
  [CVPR 2026][Autonomous Driving][Anomaly Segmentation] This paper proposes ClimaDrive, a data generation framework, and ClimaOoD, a benchmark dataset. By combining semantically guided multi-weather scene generation with perspective-aware anomaly object placement, the framework constructs a 10K+ training set covering 6 weather conditions × 93 anomaly categories. Training on this dataset yields an average AP improvement of 3.25% across four state-of-the-art methods.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Anomaly Segmentation
  - OoD Detection
  - Synthetic Data
  - Weather Augmentation
  - Diffusion Models
  - ControlNet
date: 2026-05-08
content_hash: f1d09e73c5cdfe23
---

# ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data

**Conference**: CVPR 2026
**arXiv**: [2512.02686](https://arxiv.org/abs/2512.02686)
**Code**: Unavailable
**Area**: Autonomous Driving
**Keywords**: Anomaly Segmentation, OoD Detection, Synthetic Data, Weather Augmentation, Diffusion Models, ControlNet

## TL;DR

This paper proposes ClimaDrive, a data generation framework, and ClimaOoD, a benchmark dataset. By combining semantically guided multi-weather scene generation with perspective-aware anomaly object placement, the framework constructs a 10K+ training set covering 6 weather conditions × 93 anomaly categories. Training on this dataset yields an average AP improvement of 3.25% across four state-of-the-art methods.

## Background & Motivation

Anomaly (OoD) segmentation in autonomous driving aims to detect unknown out-of-distribution objects (e.g., fallen cargo, animals, barriers), a safety-critical capability. The core bottleneck is **data scarcity**:

- **Existing datasets are small-scale and lack diversity**:
    - LostAndFound: only 1 terrain type (urban), 9 anomaly categories
    - Fishyscapes: 1 terrain type, 7 anomaly categories
    - SMIYC (SegmentMeIfYouCan): 4 terrain types, 26 anomaly categories
- **Nearly zero weather coverage**: Existing datasets are predominantly captured under clear weather, yet OoD detection under adverse conditions represents the true safety blind spot.
- **High cost of real-world collection**: Anomalous events are rare by nature, and exhaustively covering combinations of weather × scene × anomaly type is infeasible in the real world.

Synthetic data is the key pathway to overcoming the data bottleneck. However, naive copy-paste synthesis lacks physical realism and cannot produce convincing weather effects. ClimaDrive leverages the generative capacity of diffusion models to systematically address both diversity and realism.

## Method

### Overall Architecture

ClimaDrive consists of two core modules forming a complete data generation pipeline:

1. **Multi-Scene Weather Generator**: Generates multi-weather driving scene images from clean semantic maps.
2. **AnomPlacer**: Places anomaly objects in a physically plausible manner within the generated scenes.

The final output is the ClimaOoD dataset, comprising a training split and a manually curated test split.

### Key Designs

**Module 1: Multi-Scene Weather Generator**

Semantic-guided image-to-image generation based on ControlNet:

- **Input**: Semantic segmentation map + scene description text prompt
- **Control condition**: ControlNet uses the semantic map as a spatial structural constraint; the text prompt specifies weather and scene type.
- **Output**: Driving scene images rendered under the specified weather conditions.

Six weather conditions are supported: Clear, Rain, Fog, Snow, Overcast, and Night. Scene text prompts incorporate both weather descriptions and scene types (urban/suburban/highway, etc.) to guide the generation of diverse backgrounds.

**Module 2: AnomPlacer**

This is the core innovation of the method, addressing the question of how to place anomaly objects in a physically plausible manner within a scene:

**Step 1 – Feasible Region Sampling**: 64 candidate positions are uniformly sampled within the Drivable Region of the semantic map to generate pseudo bounding boxes.

**Step 2 – Perspective Prior Adjustment**: Bounding box sizes are adjusted according to their vertical position $y_i$ in the image:

$$h_i = \frac{H}{y_i}$$

Objects farther away (higher in the image) are rendered smaller, consistent with the physics of perspective projection. This prevents unnaturally large objects being placed in the distance or unnaturally small objects in the foreground.

**Step 3 – Detection and Matching**: A detection backbone predicts confidence scores for candidate bounding boxes, and Hungarian Matching selects the optimal placement locations, ensuring objects do not overlap and the layout appears natural.

**Step 4 – Diffusion Model Inpainting**: At selected locations, a diffusion model (conditioned on the anomaly object category as a prompt) performs inpainting to generate anomaly objects consistent with the scene's lighting and style.

**Step 5 – Mask Generation**: Grounding-SAM is applied to the inpainted regions to generate precise anomaly object segmentation masks, serving as ground-truth labels for training.

### Loss & Training

ClimaDrive is a data generation framework and does not introduce additional loss functions. Downstream segmentation models are trained using their respective original losses:

- **RbA (Residual-based Anomaly)**: Residual-based anomaly scoring
- **RPL (Robust Pixel-Level)**: Pixel-level robust training
- **Mask2Former**: Mask-based segmentation with an anomaly branch
- **DenseHybrid**: Density estimation and classification hybrid

Training strategy: The ClimaOoD training set is mixed with existing normal driving data for training, with anomaly region labels derived from Grounding-SAM-generated masks.

## Key Experimental Results

### Main Results

Performance gains of four state-of-the-art methods after training with ClimaOoD:

| Method | AP (Baseline → +ClimaOoD) | AUROC (Baseline → +ClimaOoD) | Gain |
|--------|--------------------------|------------------------------|------|
| RbA | Baseline → +ClimaOoD | Baseline → +ClimaOoD | AP↑, AUROC↑ |
| RPL | Baseline → +ClimaOoD | Baseline → +ClimaOoD | AP↑, AUROC↑ |
| Mask2Former | Baseline → +ClimaOoD | Baseline → +ClimaOoD | AP↑, AUROC↑ |
| DenseHybrid | Baseline → +ClimaOoD | Baseline → +ClimaOoD | AP↑, AUROC↑ |
| **Average Gain** | — | — | **AP +3.25%, AUROC +0.66%** |

### Ablation Study

| Ablation Condition | AP Change | Notes |
|-------------------|-----------|-------|
| Clear weather only (no weather augmentation) | Decrease | Weather diversity is critical for generalization |
| No perspective prior (fixed bbox size) | Decrease | Implausible object scales cause the model to learn incorrect patterns |
| No Hungarian Matching (random placement) | Slight decrease | Object overlaps reduce data quality |
| Copy-paste only (no inpainting) | Notable decrease | Boundary artifacts serve as shortcuts for the model |
| Fewer anomaly categories (93 → 20) | Decrease | Anomaly diversity is a key factor |

### Key Findings

1. **ClimaOoD dataset scale**: 10K+ training images covering 6 weather conditions × 6 scene types × 93 anomaly categories, substantially surpassing existing datasets.
2. **Test set quality**: 1,200 high-quality test images selected through manual curation.
3. **Adverse weather remains challenging**: FPR95 increases from 7.8% under clear weather to 11.0% under adverse conditions, indicating substantial room for improvement in OoD detection under weather variations.
4. **Generalizability**: ClimaOoD yields consistent improvements across four architecturally distinct methods, demonstrating that gains from improved data diversity are model-agnostic.

## Highlights & Insights

- **Systematic resolution of the data bottleneck**: Rather than proposing a new model, this work constructs a high-quality dataset — a contribution with more lasting value under the current data-centric paradigm.
- **Simplicity and effectiveness of the perspective prior**: The straightforward formula $h_i = H / y_i$ substantially improves the physical plausibility of object placement, exemplifying the engineering wisdom of "simple but effective."
- **93 anomaly categories**: Coverage spans from common objects (cones, tires) to rare ones (sofas, shopping carts), significantly improving the generalization capacity of OoD detection.
- **Method-agnostic gains**: Improvements across four methodologically distinct approaches validate the insight that data quality outweighs model design in this setting.

## Limitations & Future Work

1. **Inpainting quality depends on the diffusion model**: Generation quality for certain anomaly objects (e.g., highly reflective objects) may be unstable, introducing noisy labels.
2. **Grounding-SAM mask accuracy**: Automatically generated masks are less precise than manual annotations; boundary regions may contain errors.
3. **FPR95 under adverse weather remains high (11.0%)**: Data augmentation mitigates but does not resolve the weather robustness problem; model-level improvements are also needed.
4. **Absence of 3D information**: Pure 2D image synthesis cannot model 3D consistency such as occlusion and cast shadows; generated anomaly objects may lack proper depth cues.
5. **Test set bias**: Manual curation of 1,200 images may introduce selection bias, making it difficult to represent the true long-tail distribution.
6. **Dependency on semantic map sources**: The reliance on existing semantic segmentation ground truth for ControlNet limits the degree of automation in the data generation pipeline.

## Related Work & Insights

- **LostAndFound / Fishyscapes / SMIYC**: Existing OoD segmentation benchmarks → ClimaOoD surpasses all of these in both scale and diversity.
- **ControlNet**: Conditional diffusion generation → Using semantic maps to control scene structure is an elegant design choice.
- **Grounding-SAM**: Open-vocabulary segmentation → Cleverly repurposed to automatically generate anomaly object mask labels.
- **Inspiration**: This "generative engine + automatic annotation" data factory paradigm is transferable to other safety-critical tasks suffering from data scarcity (e.g., medical anomaly detection).

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 3.5 | Individual technical modules are not entirely new, but their systematic combination and benchmark construction are valuable. |
| Practicality | 4.5 | The dataset is directly usable; all four SOTA methods benefit. |
| Experimental Thoroughness | 4.0 | Four methods and ablation studies are comprehensive, but validation on real adverse-weather data is lacking. |
| Writing Quality | 3.5 | Structure is clear, but descriptions of generation details could be more thorough. |
| **Overall** | **3.9** | A strong, practically oriented contribution; the dataset itself offers more lasting value than the method. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Unraveling the Effects of Synthetic Data on End-to-End Autonomous Driving](../../ICCV2025/autonomous_driving/unraveling_the_effects_of_synthetic_data_on_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](terraseg_self-supervised_ground_segmentation_for_any_lidar.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Scaling-Aware Data Selection for End-to-End Autonomous Driving Systems](scaling-aware_data_selection_for_end-to-end_autonomous_driving_systems.md)
- [\[CVPR 2026\] ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes](rescene4d_temporally_consistent_semantic_instance_segmentation_of_evolving_indoo.md)

</div>

<!-- RELATED:END -->
