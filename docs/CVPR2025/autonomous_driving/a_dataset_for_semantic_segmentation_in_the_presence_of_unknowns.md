---
title: >-
  [Paper Note] A Dataset for Semantic Segmentation in the Presence of Unknowns
description: >-
  [CVPR 2025][Autonomous Driving][Anomaly Segmentation Dataset] This paper proposes the ISSU anomaly segmentation dataset, which represents the first benchmark to simultaneously support the joint evaluation of known classes (closed-set) and unknown anomalies (open-set). It is twice the size of existing anomaly segmentation datasets, covers multiple domains, sensors, and lighting conditions, and its benchmarks reveal significant deficiencies in state-of-the-art (SOTA) methods re…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Anomaly Segmentation Dataset"
  - "Open-set Recognition"
  - "Unknown Object Detection"
  - "Domain Generalization"
  - "Autonomous Driving Safety"
date: 2026-05-08
content_hash: 13c74aa0d499efe5
---

# A Dataset for Semantic Segmentation in the Presence of Unknowns

**Conference**: CVPR 2025  
**arXiv**: [2503.22309](https://arxiv.org/abs/2503.22309)  
**Code**: No public code yet  
**Area**: Autonomous Driving / Semantic Segmentation / Anomaly Detection  
**Keywords**: Anomaly Segmentation Dataset, Open-set Recognition, Unknown Object Detection, Domain Generalization, Autonomous Driving Safety

## TL;DR

This paper proposes the ISSU anomaly segmentation dataset, which represents the first benchmark to simultaneously support the joint evaluation of known classes (closed-set) and unknown anomalies (open-set). It is twice the size of existing anomaly segmentation datasets, covers multiple domains, sensors, and lighting conditions, and its benchmarks reveal significant deficiencies in state-of-the-art (SOTA) methods regarding domain generalization and the segmentation of large/small objects.

## Background & Motivation

**Background**: Semantic segmentation is crucial for autonomous driving perception, with mainstream methods achieving excellent performance on closed-set benchmarks such as Cityscapes and ADE20K. However, real-world deployment scenarios inevitably encounter unknown objects outside the training distribution (e.g., spilled cargo, animals, irregular traffic barriers), making the correct identification of these anomalies vital for safety.

**Limitations of Prior Work**: Existing anomaly segmentation evaluation datasets suffer from a clear fragmentation issue—datasets like Fishyscapes, RoadAnomaly, and SegmentMeIfYouCan (SMIYC) focus solely on anomaly detection performance, while Cityscapes evaluates only closed-set segmentation. Consequently, researchers cannot simultaneously evaluate both the correct segmentation of known classes and the rejection capability of unknown objects on the same dataset, hindering a comprehensive assessment of "in-the-wild" deployment feasibility. Furthermore, existing anomaly segmentation datasets are small in scale (Fishyscapes Lost\&Found has ~2,000 images, RoadAnomaly ~60 images), lack training sets, and suffer from limited scene diversity, covering only restricted driving environments.

**Key Challenge**: A natural trade-off exists between closed-set segmentation and open-set anomaly detection—increasing anomaly detection sensitivity often decreases the segmentation accuracy of known classes (producing more false positives). Existing evaluation systems fail to measure this trade-off within a unified framework, resulting in a lack of comprehensive feedback for method design.

**Goal**: To construct a large-scale, multi-scenario dataset with annotations for both known classes and unknown anomalies, supporting the joint evaluation of closed-set segmentation and open-set anomaly detection, and exposing the performance bottlenecks of existing methods through systematic benchmarking.

**Key Insight**: The authors observe that existing datasets either lack anomaly annotations (strictly closed-set) or annotate anomalies while ignoring the segmentation quality of known classes. Real-world deployment requires models to perform well in both aspects: segmenting known classes accurately and identifying unknown objects. Therefore, a dual-purpose dataset is needed.

**Core Idea**: The ISSU (In-domain Semantic Segmentation with Unknowns) dataset is proposed. By providing fine-grained annotations for both known classes and anomaly objects in diverse real-world driving scenarios, it establishes the first large-scale anomaly segmentation benchmark supporting joint closed-set and open-set evaluation.

## Method

### Overall Architecture

ISSU is an anomaly segmentation dataset designed for autonomous driving scenarios. The input consists of RGB images of real-world driving scenes (captured from various camera sensors), and the annotation output contains two layers of information: (1) pixel-level semantic labels of known categories (e.g., road, sidewalk, building, vehicle, etc.), and (2) binary labels of unknown/anomaly regions. The dataset is partitioned into training, validation, and test sets, with the test set further divided into a static image part and a temporal video part containing continuous video sequences.

### Key Designs

1. **Multi-Domain and Multi-Sensor Data Collection**:

    - Function: To ensure the dataset covers diverse driving environments and sensor configurations.
    - Mechanism: Data collection spans multiple geographic regions and driving environments using different camera sensor models. These scenes cover various road conditions such as urban roads and suburban highways, with collection times spanning daytime, nighttime, and diverse weather conditions. Differences in resolution and color characteristics among different sensors naturally create cross-sensor shifts, while different regional scenes constitute domain shifts. This design enables researchers to perform specialized ablation analyses on domain and sensor generalization.
    - Design Motivation: Existing datasets (e.g., Fishyscapes) are mostly collected in a single city or region with a single sensor type, making it impossible to evaluate the domain shift issues encountered in real-world deployments. The diverse design of ISSU directly addresses this shortcoming.

2. **Dual-Layer Annotation System (Closed-set + Anomaly)**:

    - Function: To simultaneously support the evaluation of closed-set semantic segmentation and open-set anomaly detection.
    - Mechanism: Each image is annotated with both pixel-level semantic labels of known classes and anomaly regions. The known classes adopt an annotation system compatible with Cityscapes (including classes like road, sidewalk, building, traffic sign, vegetation, sky, pedestrian, vehicle, etc.), while anomaly regions are annotated as the "unknown" class. This allows the simultaneous calculation of closed-set segmentation mIoU and anomaly detection AUROC/FPR95/AP metrics during evaluation, establishing a unified evaluation framework. Under this protocol, models must correctly segment known regions while marking anomaly regions; performance degradation in either aspect is reflected in the joint metrics.
    - Design Motivation: This is ISSU's most core distinguishing design. The "fragmented evaluation" of existing datasets fails to reflect real deployment needs—methods performing exceptionally well in anomaly detection alone might severely degrade closed-set segmentation, and vice versa.

3. **Static + Temporal Test Sets**:

    - Function: To support both single-frame and multi-frame temporal anomaly detection evaluations.
    - Mechanism: The test set is divided into two parts: the static portion contains independent single-frame images, and the temporal portion contains continuous video sequences. The static part is used for standard single-frame segmentation and anomaly detection evaluations, while the temporal part allows evaluating methods that exploit temporal context (e.g., using optical flow or temporal consistency to boost anomaly detection). Anomaly objects appear across multiple frames in the video sequences, testing the stability and robustness of methods in the temporal dimension.
    - Design Motivation: Real driving scenarios are continuous video streams rather than static images. An increasing number of methods utilize temporal information to improve segmentation and detection, and the temporal test set of ISSU provides the evaluation foundation for these methods.

### Evaluation Protocols

ISSU defines a unified evaluation protocol: closed-set segmentation uses the mIoU metric to measure the quality of known category segmentation, while anomaly detection uses three metrics—AUROC, FPR@95%TPR, and AP (Average Precision)—to measure the capability of identifying unknown anomalies. For comprehensive evaluation, fine-grained assessments grouped by scene conditions (lighting, sensor type, domain) and analysis grouped by anomaly object size are also provided.

## Key Experimental Results

### Main Results: Comparison of Anomaly Detection Performance

The authors evaluate multiple mainstream anomaly detection methods on the ISSU test set, including softmax-based baselines, energy/logit-based methods, and synthetic-data-assisted methods.

| Method | Backbone | AUROC ↑ | FPR@95 ↓ | AP ↑ | mIoU (Closed-set) ↑ |
|------|---------|---------|----------|------|--------------|
| MSP (Baseline) | DeepLabv3+ | 76.2 | 52.3 | 28.4 | 72.8 |
| MaxLogit | DeepLabv3+ | 79.5 | 45.1 | 33.6 | 72.8 |
| SynBoost | DeepLabv3+ | 82.1 | 38.7 | 39.2 | 71.5 |
| PEBAL | DeepLabv3+ | 85.3 | 31.4 | 44.8 | 70.9 |
| DenseHybrid | DeepLabv3+ | 87.6 | 27.8 | 48.3 | 71.2 |
| RbA | Mask2Former | 89.1 | 24.5 | 52.7 | 74.6 |
| Mask2Anomaly | Mask2Former | 90.4 | 22.1 | 55.3 | 75.1 |

Note: Compared to their performance on Fishyscapes or SMIYC, these methods generally experience significant performance degradation on ISSU, demonstrating that the ISSU dataset is more challenging. Methods based on the Mask2Former backbone (RbA, Mask2Anomaly) outperform traditional DeepLab-based methods in both closed-set segmentation quality and anomaly detection.

### Ablation Study: Performance Analysis Grouped by Scene Conditions

| Evaluation Condition | AUROC (Mask2Anomaly) | FPR@95 | Description |
|---------|---------------------|--------|------|
| Full Test Set | 90.4 | 22.1 | Complete evaluation |
| In-domain | 93.2 | 16.8 | Test scenes from the same training domain |
| Cross-domain | 84.7 | 33.5 | Domain shift scenarios; AUROC drops by ~8.5 points |
| Cross-sensor | 86.3 | 29.4 | Different sensors; performance shows noticeable degradation |
| Daylight | 91.8 | 19.3 | Best performance under sufficient lighting |
| Night / Low-light | 85.1 | 31.7 | AUROC drops by ~6.7 points under insufficient lighting |
| Small Anomaly Objects | 81.3 | 39.6 | Detection performance for small objects drops significantly |
| Large Anomaly Objects | 93.7 | 13.2 | Large objects are relatively easy to detect |

### Key Findings

- **Domain generalization is the biggest bottleneck**: In cross-domain scenarios, AUROC drops by approximately 8-10 points. This indicates that anomaly detection performance degrades significantly when deployed in new geographic regions/environments, and the domain generalization capability of existing methods falls far short of practical requirements.
- **Inadequate detection of small objects**: The detection performance of small-sized anomaly objects drops by more than 10 points compared to large objects. This severely impacts autonomous driving safety, as small-sized road obstacles are often the most dangerous.
- **Clear closed-set vs. open-set trade-off**: Methods like SynBoost and PEBAL sacrifice about 1-2 points in closed-set mIoU to improve anomaly detection, while Mask2Former-based methods exhibit a superior trade-off.
- **Great potential in utilizing temporal information**: On the temporal test set, methods leveraging multi-frame information show obvious improvements compared to single-frame methods, suggesting that temporal consistency is an important direction for improving anomaly detection.
- **Mask-level methods outperform pixel-level methods across the board**: RbA and Mask2Anomaly outperform DeepLab-series methods across all metrics, validating the advantages of mask-level recognition in anomaly segmentation.

## Highlights & Insights

- **Pioneering a Joint Evaluation Framework**: ISSU is the first large-scale dataset to simultaneously provide both closed-set segmentation annotations and anomaly annotations. This allows researchers to evaluate both capacities within a unified framework, which aligns much better with actual deployment requirements than existing segmented evaluations. This approach can be extended to other safety-critical tasks, such as known lesion segmentation + unknown lesion detection in medical image analysis.
- **Multi-Dimensional Ablation Capability**: Through its multi-domain, multi-sensor, and multi-light design, ISSU serves not only as a benchmark but also as a diagnostic tool—it can precisely locate the weak points of a method (poor domain generalization? missing small objects? degradation at night?). This design concept is highly instructive for other benchmarks.
- **Foresight of the Temporal Test Set**: Introducing video evaluation to the field of anomaly segmentation is a forward-looking design. Since actual driving is a continuous perception process, temporally consistent detection of anomaly objects is much closer to real stakes than single-frame detection.
- **Balance of Scale and Diversity**: The dataset size is twice that of existing anomaly segmentation datasets, complete with a full training/validation/test split. This enables controlled in-domain experiments rather than relying solely on zero-shot evaluation.

## Limitations & Future Work

- **High Annotation Costs**: The definition of anomaly objects is inherently subjective. Consequently, the annotation costs and quality control challenges of the dual-layer annotation (known categories + anomalies) are substantial, which may lead to annotation consistency issues.
- **Distribution of Anomaly Categories**: Although it covers a wide variety of anomaly objects, the long-tailed distribution of anomaly types may leave certain categories underrepresented.
- **Lack of 3D Information**: The dataset only provides RGB images, missing LiDAR or depth information. In actual autonomous driving perception systems, multi-modal fusion is the mainstream; integrating 3D information could further improve anomaly detection.
- **Limitations of Evaluation Metrics**: Current AUROC and FPR@95 might not fully reflect safety-critical needs (e.g., FPR@99.9 is more meaningful for autonomous driving). Future work could introduce stricter safety-oriented metrics.
- **Potential Improvement Directions**: Employing domain adaptation/generalization techniques (such as style transfer, domain randomization) to enhance cross-domain performance; combining temporal information (optical flow, tracking) to improve the detection of small objects; exploring the zero-shot capabilities of foundation models (like SAM) in anomaly segmentation.

## Related Work & Insights

- **vs Fishyscapes**: Fishyscapes provides only a small number of test images (Lost\&Found ~2000, Static ~30), lacks a training set, and only annotates anomaly regions. ISSU completely surpasses it in scale, completeness (train/val/test splits), and annotation types (dual-layer annotation). However, Fishyscapes benefits from using the high-quality Cityscapes annotations as the foundation for known classes.
- **vs SegmentMeIfYouCan (SMIYC)**: SMIYC integrates multiple anomaly segmentation datasets (including RoadAnomaly, RoadObstacle, etc.) to offer a unified evaluation platform. However, SMIYC also lacks training sets and does not track closed-set segmentation quality. The joint evaluation framework of ISSU represents a key advancement over SMIYC.
- **vs RoadAnomaly**: RoadAnomaly contains only about 60 test images, making it extremely small with limited scene diversity. ISSU offers order-of-magnitude increases in both scale and scene coverage.
- **DenseHybrid (Grcic et al., ECCV 2022)**: The third author of this paper, Grcic, is also the author of DenseHybrid, which integrates discriminative and generative anomaly detection into a unified framework. On the ISSU benchmark, DenseHybrid exhibits solid upper-middle performance, showing that hybrid strategies are effective but still leave room for improvement.

## Rating

- Novelty: ⭐⭐⭐⭐ First large-scale anomaly segmentation dataset to simultaneously support joint closed-set and open-set evaluation, filling an important evaluation gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ The benchmarks cover several mainstream methods and provide multi-dimensional ablation analyses, though details regarding online evaluation platforms and long-term maintenance are limited.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the dataset design logic is complete, although the method's technical innovation is relatively limited for a dataset-focused paper.
- Value: ⭐⭐⭐⭐⭐ Represents a major contribution to the field of anomaly segmentation, addressing the evaluation gap while highlighting key performance bottlenecks (domain generalization, small objects) that point the way for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Exploring Scene Affinity for Semi-Supervised LiDAR Semantic Segmentation](exploring_scene_affinity_for_semi-supervised_lidar_semantic_segmentation.md)
- [\[NeurIPS 2025\] Leveraging Depth and Language for Open-Vocabulary Domain-Generalized Semantic Segmentation](../../NeurIPS2025/autonomous_driving/leveraging_depth_and_language_for_open-vocabulary_domain-generalized_semantic_se.md)
- [\[ECCV 2024\] Reliability in Semantic Segmentation: Can We Use Synthetic Data?](../../ECCV2024/autonomous_driving/reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)
- [\[CVPR 2025\] Zero-Shot 4D Lidar Panoptic Segmentation](zero-shot_4d_lidar_panoptic_segmentation.md)
- [\[CVPR 2025\] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation](lisu_a_dataset_and_method_for_lidar_surface_normal_estimation.md)

</div>

<!-- RELATED:END -->
