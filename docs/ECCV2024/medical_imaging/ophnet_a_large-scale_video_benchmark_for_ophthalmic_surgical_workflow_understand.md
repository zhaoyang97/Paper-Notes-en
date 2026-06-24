---
title: >-
  [Paper Note] OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding
description: >-
  [ECCV 2024][Medical Imaging][Surgical Workflow Understanding] Constructs OphNet, currently the largest video benchmark dataset for ophthalmic surgery (2,278 videos, 285 hours, 66 surgery types, 102 surgical phases, and 150 fine-grained steps). It supports four main tasks: surgery type recognition, phase recognition, temporal localization, and phase prediction, with a scale approximately 20 times larger than the largest prior benchmark for surgical workflow analysis.
tags:
  - "ECCV 2024"
  - "Medical Imaging"
  - "Surgical Workflow Understanding"
  - "Ophthalmic Surgery"
  - "Video Benchmark"
  - "Action Recognition"
  - "Temporal Localization"
date: 2026-05-08
content_hash: 7ef29948b83be9a4
---

# OphNet: A Large-Scale Video Benchmark for Ophthalmic Surgical Workflow Understanding

**Conference**: ECCV 2024  
**arXiv**: [2406.07471](https://arxiv.org/abs/2406.07471)  
**Code**: [https://minghu0830.github.io/OphNet-benchmark/](https://minghu0830.github.io/OphNet-benchmark/)  
**Area**: Medical Images  
**Keywords**: Surgical Workflow Understanding, Ophthalmic Surgery, Video Benchmark, Action Recognition, Temporal Localization

## TL;DR
Constructs OphNet, currently the largest video benchmark dataset for ophthalmic surgery (2,278 videos, 285 hours, 66 surgery types, 102 surgical phases, and 150 fine-grained steps). It supports four main tasks: surgery type recognition, phase recognition, temporal localization, and phase prediction, with a scale approximately 20 times larger than the largest prior benchmark for surgical workflow analysis.

## Background & Motivation
Video understanding in surgical scenes is crucial for advancing robotic surgery, telesurgery, and AI-assisted surgery, particularly in ophthalmology. However, existing datasets suffer from five key limitations: (1) **Small scale**: Most datasets contain no more than 100 videos (e.g., CATARACTS has only 50, and CatRelDet has only 21); (2) **Limited surgery and phase categories**: Almost all ophthalmic datasets only include cataract surgery, with few phase categories (e.g., CatRelDet has only 4 phases); (3) **Coarse annotation granularity**: Coarse-grained definitions lead to the same action being classified into different categories in different phases, resulting in annotation bias; (4) **Lack of hierarchical annotations**: Ignoring the hierarchical construction and continuity of the surgery-phase-step relations; (5) **Single domain**: Meticulously collected videos have a uniform style, which is unfavorable for testing domain generalization capabilities.

Key Challenge: The huge gap between the demand of deep learning techniques for large-scale data and the scarcity of surgical video datasets. Research shows that I3D models require training on more than 100 videos to reach 80% accuracy, and performance continues to improve only after exceeding 700 videos.

Key Insight: **Leverages YouTube as a data source to bypass privacy concerns, constructing a large-scale, hierarchically annotated benchmark covering 66 surgery types across three major categories: cataract, glaucoma, and corneal surgeries**. The Core Idea is to perform fine-grained hierarchical annotation (surgery → phase → step) through a professional team of ophthalmologists, while simultaneously providing classification and temporal localization tasks.

## Method

### Overall Architecture
The construction of OphNet is divided into three stages: (1) Data collection and preprocessing — searching and filtering ophthalmic surgery videos from YouTube; (2) Data annotation — classification annotation and hierarchical localization annotation; (3) Evaluation benchmark — establishing baselines on four tasks. The overall pipeline revolves around dataset construction rather than proposing a new method.

### Key Designs
1. **Data Collection and Quality Control**: 

    - Leverages text search algorithms to retrieve surgical keywords on YouTube, expanding with synonyms and abbreviations (e.g., PHACO = phacoemulsification, IOL = intraocular lens implantation, ECCE = extracapsular cataract extraction).
    - Involves initial screening by five professionals followed by a review by six attending ophthalmologists.
    - Filtering criteria: Excludes low-resolution, black-and-white, animated demonstrations, and non-human eye (pig/rabbit/artificial eye) videos.
    - Design Motivation: YouTube videos bypass medical data privacy issues while ensuring diversity in video styles.

2. **Hierarchical Classification and Localization Annotation**: 

    - **Classification Annotation**: Unlike single-label classification in natural videos, ophthalmic surgery often treats multiple eye diseases simultaneously in one procedure (e.g., cataract + glaucoma), thereby supporting multi-label classification.
    - Three-layer annotation structure: Primary surgery category (only one) → Secondary surgery type (can be multiple) → Phase → Step.
    - **Localization Annotation**: 523 videos were selected for fine-grained temporal annotation, averaging 22 step annotations per video.
    - Adopts a complete linkage algorithm to aggregate the temporal boundaries of multiple annotators into stable and consistent boundaries.
    - Design Motivation: Since the eye is a complex organ and multiple conditions often coexist, multi-label and hierarchical annotations are required to accurately reflect real-world clinical scenarios.

3. **Design of Four Evaluation Tasks**: 

    - **Surgery Type Recognition**: Weakly supervised recognition of the primary surgery type in untrimmed videos (66 classes).
    - **Phase/Step Recognition**: Classifying and recognizing phases (102 classes) and steps (150 classes) in trimmed clips.
    - **Phase Localization**: Locating the start and end times of each phase in untrimmed videos.
    - **Phase Prediction**: Predicting the current phase given partial video observations.
    - Design Motivation: To cover all aspects of demand in surgical workflow understanding, going from coarse to fine, and from recognition to prediction.

### Loss & Training
As a benchmark dataset paper, it primarily evaluates existing models:
- Classification tasks: Evaluates I3D, SlowFast, X3D, and MViT V2 (both randomly initialized and K400 pre-trained); introduces CLIP-based models such as X-CLIP and ViFi-CLIP.
- Localization tasks: Evaluates ActionFormer and TriDet, using CSN, SwinViViT, and SlowFast as backbones.
- Prediction task: Evaluates Top-1 accuracy under different observation ratios.
- Data split: 70% training / 10% validation / 20% testing.

## Key Experimental Results

### Main Results
Surgery type recognition (All classes Top-1/Top-5 Accuracy %):

| Method | Backbone | Top-1 | Top-5 |
|------|----------|-------|-------|
| I3D | - | 29.8 | 53.2 |
| SlowFast | - | 27.2 | 54.4 |
| X3D | - | 28.5 | 62.7 |
| MViT V2 | - | 29.1 | 60.1 |
| X-CLIP₃₂ | ViT-B/16 | **58.9** | 81.0 |
| ViFi-CLIP₁₆ | ViT-B/16 | 58.9 | **79.8** |

Phase localization (mAP %):

| Method | Backbone | IoU=0.1 | IoU=0.3 | IoU=0.5 | IoU=0.7 | Avg. |
|------|----------|---------|---------|---------|---------|------|
| ActionFormer | SwinViViT | 59.3 | 54.7 | 43.3 | 26.3 | 46.4 |
| ActionFormer | SlowFast | 60.0 | 55.9 | 45.1 | 26.0 | 47.5 |
| TriDet | SwinViViT | **61.0** | **57.1** | **47.1** | **33.1** | **50.4** |
| TriDet | SlowFast | 61.3 | 56.0 | 45.6 | 30.4 | 48.6 |

### Ablation Study
Impact of different input frame numbers and pre-training in phase/step classification (ViFi-CLIP model):

| Model Configuration | Phase Top-1 (All) | Step Top-1 (All) |
|---------|----------------|----------------|
| ViFi-CLIP₁₆ | 66.1 | 65.0 |
| ViFi-CLIP₃₂ | **68.4** | 64.8 |
| X-CLIP₁₆ | 63.4 | 62.5 |
| X-CLIP₃₂ | 62.7 | 62.0 |

Phase prediction (Top-1 accuracy % under different observation ratios):

| Method | ratio=0.1 | ratio=0.3 | ratio=0.5 | ratio=0.7 | Avg. |
|------|-----------|-----------|-----------|-----------|------|
| MViT V2 | 25.6 | 43.7 | 49.3 | 52.3 | 47.5 |
| MViT V2* (K400) | **27.8** | **43.8** | **50.5** | 51.7 | **48.2** |
| SlowFast* | 27.5 | 43.2 | 49.9 | **52.3** | 47.8 |

### Key Findings
- **CLIP-based models significantly outperform traditional video models**: X-CLIP scores 58.9% vs. I3D's 29.8%, almost doubling the performance, which indicates that vision-language pre-training is highly beneficial for surgical video understanding.
- **ViFi-CLIP performs the best in phase and step classification**, especially achieving 75.9% Top-1 accuracy in cataract surgeries.
- **Kinetics 400 pre-training is generally helpful**, but less effective than CLIP pre-training.
- **TriDet+SwinViViT performs best in localization tasks**, with a more pronounced advantage under high IoU thresholds.
- **More input frames generally have a positive impact**, but the gains are inconsistent.
- **The overall accuracy remains low** (surgical recognition Top-1 is only ~59%), indicating that the dataset is indeed challenging.
- **Attention visualization shows** that the models correctly focus on surgical instruments and the ocular regions.

## Highlights & Insights
- **Significant breakthroughs in scale and diversity**: 2,278 videos, 285 hours, 66 surgeries, and 150 steps — far exceeding all prior surgical workflow datasets.
- **Rational hierarchical annotation design**: The three-layer annotation (surgery → phase → step) aligns with real-world clinical needs, where a single video may contain multiple surgery types.
- **Utilizing YouTube to bypass privacy concerns**: This solves medical data privacy issues while introducing style diversity to test domain generalization.
- **Involvement of 15 professional ophthalmologists in annotation**: Ensures the professionalism and accuracy of the annotations.
- **Comprehensive coverage of four tasks**: Ranging from recognition to localization and prediction, providing a complete evaluation framework for surgical workflow understanding.
- **The complete linkage algorithm** used to aggregate boundaries from multiple annotators is a practical solution for annotation consistency.

## Limitations & Future Work
- Video quality from YouTube sources is inconsistent, and some videos may contain labeling noise.
- The Top-1 accuracy for surgery recognition is only 59%, indicating the need for stronger models or better representation learning.
- Localization annotations cover only 523 videos (~23%), with most videos only having classification labels.
- Pixel-level segmentation annotations for instruments and anatomical structures are not provided, limiting the potential for multi-task learning.
- **Severe label class imbalance**: Some rare surgery types have extremely few samples.
- Future work could explore weakly supervised or semi-supervised methods to utilize the large amount of unlabeled videos.

## Related Work & Insights
- Complements endoscopic surgery datasets like Cholec80/CholecT50, extending surgical video understanding to the field of ophthalmic microscopy.
- The advantages of CLIP-based models in surgical video classification suggest that the medical video field can benefit more from vision-language pre-training.
- The hierarchical annotation design can be generalized to other video understanding tasks involving complex, multi-step procedures.
- Directly promotes the development of surgical education training and AI-assisted surgery systems.
- Large-scale annotation requires substantial expert team involvement; the dataset's value lies in its irreplaceable professionalism.

## Rating
- Novelty: ⭐⭐⭐ (mainly historical dataset contribution, limited methodological novelty)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (comprehensive evaluation across four tasks, comparing multiple baselines)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, detailed statistics)
- Value: ⭐⭐⭐⭐⭐ (fills an important data gap in ophthalmic surgical video understanding with a leading scale)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GEMeX: A Large-Scale, Groundable, and Explainable Medical VQA Benchmark for Chest X-ray Diagnosis](../../ICCV2025/medical_imaging/gemex_a_large-scale_groundable_and_explainable_medical_vqa_benchmark_for_chest_x.md)
- [\[CVPR 2026\] X-PCR: A Benchmark for Cross-modality Progressive Clinical Reasoning in Ophthalmic Diagnosis](../../CVPR2026/medical_imaging/x-pcr_a_benchmark_for_cross-modality_progressive_clinical_reasoning_in_ophthalmi.md)
- [\[NeurIPS 2025\] THUNDER: Tile-level Histopathology image UNDERstanding benchmark](../../NeurIPS2025/medical_imaging/thunder_tile-level_histopathology_image_understanding_benchmark.md)
- [\[CVPR 2026\] Efficient Unrolled Networks for Large-Scale 3D Inverse Problems](../../CVPR2026/medical_imaging/efficient_unrolled_networks_for_large-scale_3d_inverse_problems.md)
- [\[CVPR 2026\] TRCoRSurg: Temporal-Relational Co-Reasoning for Surgical Video Triplet Recognition](../../CVPR2026/medical_imaging/trcorsurg_temporal-relational_co-reasoning_for_surgical_video_triplet_recognitio.md)

</div>

<!-- RELATED:END -->
