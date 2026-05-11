---
title: >-
  [Paper Note] SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving
description: >-
  [CVPR 2026][Autonomous Driving][Rare image retrieval] SearchAD introduces the first large-scale rare image retrieval dataset for autonomous driving, comprising 420K+ frames, 510K+ annotated bounding boxes, and 90 rare categories. It supports both text-to-image and image-to-image retrieval, and through comprehensive evaluation reveals the deficiencies of current multimodal retrieval models in retrieving rare objects.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Rare image retrieval
  - long-tail distribution
  - semantic retrieval
  - dataset benchmark
date: 2026-05-08
content_hash: c207fa7d4c94a9f8
---

# SearchAD: Large-Scale Rare Image Retrieval Dataset for Autonomous Driving

**Conference**: CVPR 2026
**arXiv**: [2604.08008](https://arxiv.org/abs/2604.08008)
**Code**: [https://github.com/iis-esslingen/searchad_devkit](https://github.com/iis-esslingen/searchad_devkit)
**Area**: Autonomous Driving / Dataset
**Keywords**: Rare image retrieval, autonomous driving, long-tail distribution, semantic retrieval, dataset benchmark

## TL;DR

SearchAD introduces the first large-scale rare image retrieval dataset for autonomous driving, comprising 420K+ frames, 510K+ annotated bounding boxes, and 90 rare categories. It supports both text-to-image and image-to-image retrieval, and through comprehensive evaluation reveals the deficiencies of current multimodal retrieval models in retrieving rare objects.

## Background & Motivation

**Background**: The safety of autonomous driving (AD) systems critically depends on the ability to handle rare and safety-critical scenarios. As dataset scales continue to grow (now reaching millions of frames), the key challenge has shifted from "collecting more data" to "efficiently retrieving the most relevant samples."

**Limitations of Prior Work**: (1) Existing AD datasets focus primarily on common categories (vehicles, pedestrians, cyclists, etc.) with minimal coverage of rare objects such as pedestrians with crutches, animals, and abnormal road markings; (2) existing image retrieval benchmarks are designed for instance-level retrieval (different viewpoints of the same object) rather than the semantic-level retrieval required in AD (finding images containing a specific rare category); (3) no unified large-scale benchmark exists to evaluate and advance retrieval techniques for rare driving scenarios.

**Key Challenge**: Rare safety-critical scenarios occur at extremely low frequencies (a "needle-in-a-haystack" problem), yet they are crucial to the safety of AD systems. Existing methods have not been specifically evaluated for retrieval capability under such extreme long-tail distributions.

**Goal**: To construct the first large-scale dataset and benchmark focused on rare object/scene retrieval in autonomous driving scenarios, filling a critical gap in the field.

**Key Insight**: Integrating data from 11 existing AD datasets and creating a unified retrieval benchmark through manual annotation of bounding boxes for 90 rare categories.

**Core Idea**: To address the "needle-in-a-haystack" problem in autonomous driving by constructing a large-scale rare image retrieval dataset that supports semantic-level text-to-image and image-to-image retrieval.

## Method

### Overall Architecture

The SearchAD construction pipeline consists of: (1) extracting 423,798 frames from 11 existing AD datasets; (2) defining 90 rare categories and performing manual bounding box annotation (513,265 boxes in total); (3) designing appropriate train/validation/test splits; (4) providing textual descriptions and visual examples as retrieval support sets; and (5) establishing a public benchmark server for standardized evaluation.

### Key Designs

1. **Rare Category Taxonomy**:

    - Function: Define rare and safety-critical object categories in autonomous driving scenarios.
    - Mechanism: The 90 rare categories are organized into 9 superclasses—Animal, Human (special pedestrian states such as crutch or wheelchair users), Marking (abnormal road markings), Object (road obstacles), Rideable (rideable devices), Scene (special scenes), Sign (non-standard traffic signs), Trailer, and Vehicle (special vehicles). Some categories appear fewer than 50 times across the entire dataset, reflecting extreme rarity.
    - Design Motivation: The definition of rare categories must balance safety criticality and frequency of occurrence. Organizing categories into a hierarchical structure facilitates retrieval evaluation at different levels of granularity.

2. **Multi-source Data Integration**:

    - Function: Construct a dataset with both scale and diversity.
    - Mechanism: Data is integrated from 11 datasets: Lost and Found, WildDash2, ACDC, IDD, KITTI, Cityscapes, Mapillary Vistas, ECP, nuScenes, BDD100K, and Mapillary Sign. Each source contributes distinct sensor configurations, geographic distributions, and weather conditions, increasing retrieval difficulty. All bounding boxes for rare categories are manually annotated at high quality.
    - Design Motivation: A single dataset offers limited diversity; integrating multi-source data better simulates the varied conditions under which rare objects appear in the real world.

3. **Semantic Retrieval Evaluation Framework**:

    - Function: Support standardized evaluation of both text-to-image and image-to-image retrieval modes.
    - Mechanism: For each rare category, a language support set (textual descriptions) and a visual support set (example images) are provided. Retrieval models are required to locate images containing the target category within the large-scale dataset. Evaluation metrics include mean Average Precision (mAP) and Recall@K. A public test server ensures fair evaluation.
    - Design Motivation: Semantic-level retrieval (based on category semantics rather than specific instances) better reflects the practical needs of data mining in autonomous driving—developers typically need to "find all images containing animals" rather than "find this specific dog."

### Loss & Training

SearchAD is a dataset contribution and does not involve training a novel model. However, baseline experiments are provided:

- **Zero-shot retrieval**: Pretrained multimodal models (CLIP, SigLIP, RegionCLIP, etc.) are used directly for retrieval without any fine-tuning.
- **Fine-tuned baseline**: Retrieval models are fine-tuned on the SearchAD training set using standard contrastive learning loss (InfoNCE).
- **Evaluation metrics**: Standard retrieval metrics including mAP and Recall@1/5/10.

## Key Experimental Results

### Main Results

| Method Type | Model | mAP (T2I) | Recall@10 (T2I) | Notes |
|-------------|-------|-----------|-----------------|-------|
| Global features | CLIP | Baseline level | Low | Global semantic matching |
| Global features | SigLIP | Slightly above CLIP | Slightly better | Stronger pretraining |
| Spatial alignment | RegionCLIP | Best zero-shot | Best zero-shot | Superior spatial vision-language alignment |
| Fine-tuned | CLIP-ft | Significant gain | Significant gain | Fine-tuning yields large improvement |
| Image-to-image | CLIP (I2I) | Below T2I | Below T2I | Image retrieval weaker than text |

### Ablation Study

| Analysis Dimension | Finding | Notes |
|--------------------|---------|-------|
| Text vs. image retrieval | Text outperforms image | Text carries stronger semantic priors |
| Effect of rarity | Rarer categories are harder to retrieve | Categories with <50 occurrences are extremely difficult |
| Effect of object size | Smaller objects are harder | Small rare objects achieve the lowest retrieval precision |
| Effect of fine-tuning | Significant improvement but still insufficient | Absolute retrieval capability remains unsatisfactory |

### Key Findings

- Text-to-image retrieval significantly outperforms image-to-image retrieval, indicating that the semantic grounding capability of language is essential for rare object retrieval.
- Models that directly align spatial visual features with language (e.g., RegionCLIP) achieve the best zero-shot retrieval performance.
- Even after fine-tuning, retrieval precision for extremely rare categories (<50 occurrences) remains low, demonstrating that the problem is far from solved.
- The long-tail distribution within the dataset is the core challenge—the frequency of the 90 categories spans more than three orders of magnitude.

## Highlights & Insights

- **Unique problem formulation**: The work focuses on the "needle-in-a-haystack" retrieval of rare safety-critical scenes, an important yet largely overlooked problem. In autonomous driving, a system's ability to respond to one-time rare events can be a matter of life and death.
- **Balance between dataset scale and quality**: 420K+ frames from 11 datasets ensure diversity, while 510K+ manually annotated bounding boxes guarantee annotation quality. The design of 90 rare categories reflects a deep understanding of autonomous driving safety requirements.
- **Revealing limitations of current methods**: Even the best models after fine-tuning exhibit insufficient rare object retrieval capability, providing the community with a clear direction for improvement.

## Limitations & Future Work

- Annotations are limited to 2D bounding boxes; the absence of 3D annotations and semantic segmentation masks constrains finer-grained retrieval evaluation.
- Although the 90 rare categories cover major safety-critical scenarios, the long-tail distribution in the real world may be even more extreme.
- Only static single-frame retrieval is currently supported; temporal retrieval (finding video clips containing rare events) is not addressed.
- The geographic distribution of source datasets may introduce regional bias in the rare category distribution.
- Future work could consider integrating active learning or few-shot detection techniques to improve rare object retrieval capability.

## Related Work & Insights

- **vs. nuScenes/Waymo retrieval**: Retrieval in traditional AD datasets relies primarily on metadata tags, whereas SearchAD emphasizes multimodal semantic-based retrieval.
- **vs. Oxford5K/Paris6K retrieval benchmarks**: Traditional retrieval benchmarks target instance-level retrieval of landmarks, while SearchAD targets semantic-level retrieval of rare categories.
- **vs. OpenImages/LVIS long-tail detection**: These datasets focus on long-tail detection, while SearchAD focuses on long-tail retrieval; the emphasis differs but the problem structure is analogous.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale semantic retrieval benchmark for rare AD scenarios; the problem formulation is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison of multiple baselines with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ The dataset construction process is described clearly.
- Value: ⭐⭐⭐⭐ Dataset + benchmark + public server represent a clear contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Ghost-FWL: A Large-Scale Full-Waveform LiDAR Dataset for Ghost Detection and Removal](ghost-fwl_a_large-scale_full-waveform_lidar_dataset_for_ghost_detection_and_remo.md)
- [\[CVPR 2026\] Learning to Drive is a Free Gift: Large-Scale Label-Free Autonomy Pretraining from Unposed In-The-Wild Videos](learning_to_drive_is_a_free_gift_large-scale_label-free_autonomy_pretraining_fro.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](traffic_scene_generation_from_natural_language_description_for_autonomous_vehicl.md)
- [\[ICLR 2026\] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](../../ICLR2026/autonomous_driving/egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)

</div>

<!-- RELATED:END -->
