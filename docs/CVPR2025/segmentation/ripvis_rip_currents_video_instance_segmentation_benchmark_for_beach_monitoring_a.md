---
title: >-
  [Paper Note] RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring
description: >-
  [CVPR 2025][Segmentation][Rip current segmentation] RipVIS introduces the first large-scale rip current video instance segmentation benchmark dataset (184 videos / 210k frames) and proposes a post-processing method named Temporal Confidence Aggregation (TCA). TCA enhances the stability and recall of rip current segmentation through cross-frame confidence accumulation, providing a systematic computer vision solution for beach safety monitoring.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Rip current segmentation"
  - "Video instance segmentation"
  - "Beach safety"
  - "Temporal confidence aggregation"
  - "Amorphous object detection"
date: 2026-05-08
content_hash: 62736f5b25454e52
---

# RipVIS: Rip Currents Video Instance Segmentation Benchmark for Beach Monitoring

**Conference**: CVPR 2025  
**arXiv**: [2504.01128](https://arxiv.org/abs/2504.01128)  
**Code**: [https://ripvis.ai](https://ripvis.ai)  
**Area**: Segmentation  
**Keywords**: Rip current segmentation, Video instance segmentation, Beach safety, Temporal confidence aggregation, Amorphous object detection

## TL;DR

RipVIS introduces the first large-scale rip current video instance segmentation benchmark dataset (184 videos / 210k frames) and proposes a post-processing method named Temporal Confidence Aggregation (TCA). TCA enhances the stability and recall of rip current segmentation through cross-frame confidence accumulation, providing a systematic computer vision solution for beach safety monitoring.

## Background & Motivation

**Background**: Rip currents are strong, narrow seaward-flowing streams of water occurring near beaches, causing numerous drowning incidents globally each year. Traditional detection methods include GPS drifter tracking and UAV dye tracking, but these are costly and have limited coverage. Recently, automated detection based on computer vision has emerged, with existing work utilizing YOLO and Faster R-CNN for bounding box detection.

**Limitations of Prior Work**: (1) Severe lack of datasets: the largest existing dataset has only around 30k frames with low geographical diversity (a single location) and mostly provides only bounding box annotations. (2) Rip currents are "amorphous objects": they continuously change shape, feature blurry boundaries, and are highly integrated with the background (seawater), making them harder to detect than smoke or fire. (3) Accurate annotation requires both oceanographic expertise and CV annotation skills, resulting in extremely high labor costs.

**Key Challenge**: The amorphous nature of rip currents and their high similarity to the background make bounding box detection insufficient, as boxes either contain significant background noise or miss parts of the rip current. However, the high cost of segmentation annotation limits the scale of datasets.

**Goal**: To establish a high-quality rip current video instance segmentation benchmark with an order of magnitude increase in scale, providing annotated data across multiple countries, viewpoints, and environmental conditions globally, and exploring effective segmentation methods.

**Key Insight**: Over a three-year period, the authors mobilized more than 30 annotation volunteers and 2 oceanography experts to collect beach videos across 10 countries using UAVs, mobile phones, and stationary cameras, constructing the RipVIS benchmark, which is an order of magnitude larger than previous datasets.

**Core Idea**: Utilizing a large-scale diverse dataset combined with Temporal Confidence Aggregation (TCA) post-processing to bridge the instability of single-frame segmentation, leveraging temporal consistency to reduce false positives and false negatives.

## Method

### Overall Architecture

The RipVIS project consists of two parts: (1) Dataset construction: 184 videos / 212,328 frames, where 150 videos are annotated with rip currents and 34 videos serve as negative samples without rip currents; the dataset is partitioned into 112/36/36 videos for training/validation/testing. (2) Baseline evaluation: establishing baselines on Mask R-CNN, Cascade Mask R-CNN, SparseInst, and YOLO11, and proposing TCA post-processing to improve temporal consistency in video segmentation.

### Key Designs

1. **Large-scale Diverse Dataset Construction**:

    - **Function**: Provides the first video-level instance segmentation annotated and globally diverse rip current dataset.
    - **Mechanism**: Data was collected from three categories of sources: 76 videos shot in situ by the author team using UAVs/mobile phones across 10 countries, 87 from the internet, and 21 from existing data by de Silva et al. The videos cover four shooting perspectives (horizontal beach, elevated beach, UAV oblique, UAV nadir). Keyframe annotations were manually completed using polygon masks primarily sampled at 5 FPS, with non-annotated frames generated via interpolation and validated. The labeling process involved 30 volunteers and 2 oceanography experts, achieving a Cohen's $\kappa$ of 0.82.
    - **Design Motivation**: Data diversity is key to model generalization: beach morphology, wave patterns, and water color vary greatly across different countries. Previous datasets with narrow geographical scopes led to poor performance on new beaches.

2. **Temporal Confidence Aggregation (TCA)**:

    - **Function**: Improves the stability and accuracy of video segmentation by leveraging cross-frame temporal information.
    - **Mechanism**: Consists of four steps: (a) Downsampling: reducing the resolution of predicted masks to lower computational cost; (b) Instance tracking: matching masks in adjacent frames via IoU and maintaining instance ID consistency with the Hungarian algorithm; (c) Temporal smoothing: maintaining a heatmap for each tracked instance, where confidence increments for detected pixels and decrements for undetected ones; (d) Hysteresis thresholding: adopting a dual-threshold strategy inspired by Canny edge detection, where high-threshold pixels serve as seed points and connected low-threshold pixels are included.
    - **Design Motivation**: Single-frame segmentation is noisy: the shape of a rip current changes dynamically, leading to potential missed detections in the current frame that might reappear later. TCA aggregates these transient detections through "temporal memory", effectively reducing false positives (requiring accumulated evidence to confirm) and false negatives (retaining detections via historical evidence).

3. **Safety-Oriented F2 Evaluation Metric**:

    - **Function**: An evaluation framework prioritizing the reduction of missed detections.
    - **Mechanism**: Uses the $F_\beta$ score with $\beta=2$, i.e., $F_2 = \frac{5 \cdot precision \cdot recall}{4 \cdot precision + recall}$, placing higher emphasis on recall than F1.
    - **Design Motivation**: In beach safety monitoring, a missed detection (false negative) means someone could potentially drown without a system warning, which is far more critical than a false alarm (false positive, which only inconveniences beachgoers). Prioritizing lower false negatives through $F_2$ aligns with the requirements of safety-critical scenarios.

### Loss & Training

- Baseline models are trained using their original settings (Detectron2 + YOLO ultralytics).
- Training is conducted on an RTX 4090, with inference FPS measured on an RTX 3060.
- TCA post-processing hyperparameters: the slow-increase/slow-decrease mode is suitable for stationary camera videos, while the fast-increase/fast-decrease mode fits moving cameras.

## Key Experimental Results

### Main Results

| Model | Precision | Recall | AP50 | F1 | F2 | FPS |
|------|-----------|--------|------|----|----|-----|
| SparseInst PVTv2 | 0.683 | 0.770 | 0.721 | 0.724 | 0.751 | 28.0 |
| SparseInst PVTv2 +TCA | 0.712 | 0.798 | 0.751 | 0.753 | **0.780** | 17.6 |
| YOLO11s | 0.757 | 0.612 | 0.705 | 0.677 | 0.636 | 116.3 |
| YOLO11s +TCA | 0.752 | 0.647 | 0.723 | 0.696 | 0.666 | 33.8 |
| Cascade Mask R-CNN | 0.606 | 0.660 | 0.628 | 0.632 | 0.648 | 9.5 |
| Cascade Mask R-CNN +TCA | 0.613 | 0.686 | 0.639 | 0.647 | 0.670 | 7.9 |

### Ablation Study

| Configuration | F2 Change | Description |
|------|---------|------|
| SparseInst PVTv2 Original | 0.751 | Baseline without TCA |
| + TCA | 0.780 (+0.029) | TCA consistently improves all metrics |
| YOLO11n Original | 0.583 | Most lightweight |
| + TCA | 0.613 (+0.030) | Smaller models benefit more |
| Mask R-CNN Original | 0.593 | Two-stage detector |
| + TCA | 0.625 (+0.032) | Two-stage detector also benefits significantly |

### Key Findings

- SparseInst PVTv2 performs the best in $F_2$ (0.780 with TCA), achieving the best balance between precision and recall.
- The YOLO11 series exhibits high precision but low recall—tending to make conservative predictions, which results in lower $F_2$ scores than SparseInst in safety-critical scenarios.
- TCA has a positive effect on all models, improving $F_2$ by approximately 0.03 on average, though it reduces inference speed by around 40-50%.
- One-stage detectors perform better and are more stable than two-stage detectors.
- The RipVIS dataset is substantially more challenging than conventional video segmentation datasets—all SOTA models perform relatively poorly on this dataset.
- TCA may produce a few false alarms in moving camera scenarios (as the historical heatmap may mismatch the current frame during rapid camera movement).

## Highlights & Insights

- **Dataset scale and diversity represent the primary contribution**: Three years of annotation involving 30+ individuals, covering 10 countries and 4 viewing perspectives with Cohen's $\kappa=0.82$. Constructing such a professional, large-scale dataset constitutes a highly valuable research utility in its own right.
- **TCA post-processing is highly generalizable**: Temporal Confidence Aggregation does not depend on any specific model architecture and can be integrated into any video segmentation pipeline. Adopting the dual-threshold strategy from Canny edge detection for the temporal domain is a clever adaptation.
- **The choice of the $F_2$ evaluation metric** reflects how CV systems must re-evaluate performance metrics when moving from academic settings to real-world deployment—the weights of precision and recall should differ in safety-critical scenarios.

## Limitations & Future Work

- The dataset lacks flash and transient rip currents due to their highly unpredictable nature, meaning gaps in category coverage remain.
- The increment/decrement rates of TCA need to be manually adjusted according to the camera type (stationary vs. mobile), lacking an adaptive mechanism.
- The performance of all methods remains low (best $F_2 = 0.780$), indicating that rip current segmentation is still a highly challenging open problem.
- Future directions: incorporating optical flow information as auxiliary features (since rip current direction differs from surrounding seawater); utilizing SAM or video foundation models to improve segmentation accuracy; and designing adaptive TCA to automatically adjust temporal smoothing parameters.

## Related Work & Insights

- **vs Dumitriu et al. (2023)**: The only prior rip current segmentation dataset comprised only 37k frames and a single location (Black Sea). RipVIS thoroughly surpasses it in scale (210k frames), geographic diversity (10 countries), and annotation quality (expert audit with $\kappa = 0.82$).
- **vs Optical Flow Methods (RipViz, Philip et al.)**: Optical flow methods can detect directional anomalies in rip currents but require stationary cameras. RipVIS includes videos from moving cameras, necessitating more robust methods. The temporal aggregation concept of TCA is complementary to optical flow methods.
- **vs Generic Video Object Segmentation (DAVIS, YouTube-VIS)**: The amorphous nature of rip currents makes them harder to segment than objects in standard video segmentation datasets—they lack a fixed shape and are highly integrated with the background, representing a unique type of challenge.

## Rating

- **Novelty**: ⭐⭐⭐ TCA post-processing has some novelty, but the methodological innovation is relatively limited; the main contribution lies in the dataset.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple baseline models, with/without TCA comparisons, and complete hyperparameter analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Detailed dataset description and clear motivation of the research problem.
- **Value**: ⭐⭐⭐⭐ Fills a significant gap in rip current segmentation datasets, offering practical societal value for beach safety.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MammAlps: A Multi-view Video Behavior Monitoring Dataset of Wild Mammals in the Swiss Alps](mammalps_a_multi-view_video_behavior_monitoring_dataset_of_wild_mammals_in_the_s.md)
- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)
- [\[CVPR 2025\] MaSS13K: A Matting-level Semantic Segmentation Benchmark](mass13k_a_matting-level_semantic_segmentation_benchmark.md)
- [\[ICCV 2025\] Hierarchical Visual Prompt Learning for Continual Video Instance Segmentation](../../ICCV2025/segmentation/hierarchical_visual_prompt_learning_for_continual_video_instance_segmentation.md)
- [\[CVPR 2025\] Foveated Instance Segmentation](foveated_instance_segmentation.md)

</div>

<!-- RELATED:END -->
