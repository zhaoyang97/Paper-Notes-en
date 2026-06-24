---
title: >-
  [Paper Note] I Can't Believe It's Not Scene Flow!
description: >-
  [ECCV 2024][LLM Pretraining][Scene Flow Estimation] Reveals that the catastrophic failure of existing scene flow methods on small objects like pedestrians is masked by current evaluation metrics, and proposes a category-aware and velocity-normalized Bucket Normalized EPE evaluation protocol, alongside a simple yet SOTA baseline, TrackFlow (generating scene flow from a detector + tracker), achieving a 1.5x improvement in pedestrian motion description.
tags:
  - "ECCV 2024"
  - "LLM Pretraining"
  - "Scene Flow Estimation"
  - "Evaluation Protocol"
  - "Category-Aware"
  - "3D Object Tracking"
  - "LiDAR"
date: 2026-05-08
content_hash: d5158172c4c7334f
---

# I Can't Believe It's Not Scene Flow!

**Conference**: ECCV 2024  
**arXiv**: [2403.04739](https://arxiv.org/abs/2403.04739)  
**Code**: [https://github.com/kylevedder/BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval)  
**Area**: LLM Pre-training  
**Keywords**: Scene Flow Estimation, Evaluation Protocol, Category-Aware, 3D Object Tracking, LiDAR

## TL;DR
Reveals that the catastrophic failure of existing scene flow methods on small objects like pedestrians is masked by current evaluation metrics, and proposes a category-aware and velocity-normalized Bucket Normalized EPE evaluation protocol, alongside a simple yet SOTA baseline, TrackFlow (generating scene flow from a detector + tracker), achieving a 1.5x improvement in pedestrian motion description.

## Background & Motivation

**Background**: Scene flow estimation is a core task in autonomous driving, aiming to describe the 3D motion field between two consecutive point cloud frames. Current SOTA methods (such as ZeroFlow XL 5x) achieve an average error of approximately 4.9cm on the standard Threeway EPE metric, seemingly reaching centimeter-level accuracy. Mainstream methods are divided into supervised methods (FastFlow3D, DeFlow) and unsupervised methods (NSFP, ZeroFlow).

**Limitations of Prior Work**: However, these seemingly excellent numbers mask a serious issue—all existing methods almost completely fail at scene flow estimation on small objects such as pedestrians and cyclists. Through visualization, the authors find that even in the "simplest" cases where the pedestrian LiDAR return density is exceptionally high, all prior methods fail to describe pedestrian motion. However, the standard Threeway EPE metric completely fails to expose this issue, as pedestrian points constitute less than 1% of the total dynamic points, being overwhelmed by a massive number of car points.

**Key Challenge**: Existing evaluation protocols suffer from dual defects: (1) not category-aware—the few points of small targets are drowned out in the average by the massive points of large targets; (2) not velocity-normalized—an error of 0.5m/s is negligible (<2.5%) for a 20m/s car, but for a 0.5m/s pedestrian, it means 100% of the motion is unexplained. This conceals the catastrophic failure of methods on safety-critical categories.

**Goal**: (1) Design evaluation metrics that can reveal scene flow failures on small objects; (2) Demonstrate that leveraging category rebalancing techniques can significantly improve the quality of scene flow on small objects.

**Key Insight**: Inspired by how mAP in object detection evaluates each category with equal weight, scene flow evaluation should also be category-aware. Meanwhile, observing that the ground truth for scene flow itself originates from 3D bounding box tracking, directly generating scene flow using a high-quality detector + tracker is a natural baseline.

**Core Idea**: Reveal the failure of existing methods using category-aware + velocity-normalized evaluation metrics, and surpass all prior methods with the "embarrassingly simple" pipeline of "detection + tracking $\rightarrow$ scene flow".

## Method

### Overall Architecture
Our contributions consist of two parts: (1) a new evaluation metric, Bucket Normalized EPE; (2) a new baseline method, TrackFlow. The pipeline of TrackFlow is extremely simple: input two frames of LiDAR point clouds $\rightarrow$ SOTA 3D object detector (LE3DE2E) $\rightarrow$ Kalman filter tracker (AB3DMOT) $\rightarrow$ generate point-wise scene flow based on the rigid transformation of tracked bounding boxes.

### Key Designs

1. **Bucket Normalized EPE Evaluation Protocol**:

    - **Function**: Provides category-aware and velocity-normalized scene flow evaluation, allowing fair comparison of performance across different categories.
    - **Mechanism**: Assigns all points to a category-velocity matrix based on their ground truth category and velocity. For each bucket, the Average EPE and average velocity are calculated. Two metrics are reported: Static EPE (error in static buckets) and Dynamic Normalized EPE (the mean normalized EPE of each non-empty velocity bucket, i.e., $\text{Average EPE} / \text{average speed}$$). Dynamic Normalized EPE measures the "proportion of unexplained motion"—0 represents perfection, and 1.0 indicates that only zero flow after ego-motion compensation was predicted. Finally, the mean Dynamic Normalized EPE (similar to mAP) is obtained by averaging across all categories.
    - **Design Motivation**: Resolves the two blind spots of Threeway EPE. Category bucketing ensures rare categories like pedestrians are not drowned out by cars; velocity normalization allows direct comparison of errors across targets with different speeds. This leaves the true performance of methods on safety-critical categories with nowhere to hide.

2. **TrackFlow Scene Flow Baseline**:

    - **Function**: Generates scene flow predictions via rigid transformations from 3D object tracking.
    - **Mechanism**: Runs the SOTA 3D detector LE3DE2E (using a low confidence threshold of 0.2 to maximize recall), and then associates detection boxes using the Kalman filter tracker AB3DMOT to generate trajectories. For each detected object, its internal point cloud's motion is described using the rigid transformation between tracked boxes. Undetected points use zero flow after ego-motion compensation. The effectiveness of this method stems from directly simulating the ground truth generation process—since ground truth flow itself is derived from the rigid transformation of bounding box tracking.
    - **Design Motivation**: The key insight is that modern 3D detectors leverage mature category rebalancing techniques (such as copy-paste augmentation, focal loss, etc.), resulting in excellent detection capabilities on rare categories like pedestrians. In contrast, existing scene flow methods fail to utilize these category rebalancing techniques, leading to extremely poor performance on small objects.

3. **Detector Selection and Confidence Tuning**:

    - **Function**: Selects the optimal detector configuration for TrackFlow.
    - **Mechanism**: Differing from conventional object detection which uses high confidence thresholds (0.7-0.9), TrackFlow utilizes a low threshold (0.2) to maximize recall. This is because the cost of a missed detection is catastrophic—each false negative means all points of that object can only receive zero flow, where 100% of the motion is unexplained. Conversely, false positives can be filtered out by the association logic of the tracker. Experiments show that LE3DE2E's recall at low thresholds is far superior to BEVFusion, which directly leads to TrackFlow significantly outperforming TrackFlowBEVF.
    - **Design Motivation**: Reveals that a "good detector suitable for TrackFlow" is not the one with the highest mAP, but rather the one with the highest recall at low thresholds and accurate orientation estimation—two detectors with similar mAP can yield drastically different scene flow qualities.

### Loss & Training
TrackFlow itself does not require training a scene flow model—it directly employs pre-trained detectors and trackers. The detectors are trained using standard category rebalancing techniques. Evaluation is performed using the test split of Argoverse 2.

## Key Experimental Results

### Main Results

| Metric | TrackFlow | DeFlow | ZeroFlow XL 5x | NSFP |
|------|-----------|--------|----------------|------|
| Threeway EPE | **SOTA** (↓1.5mm) | Second Best | Third | Fourth |
| mean Dyn. Norm. EPE | **0.287** | ~0.39 | ~0.45 | ~0.50 |
| Pedestrian Dyn. Norm. EPE | **~0.40** | ~0.60 | ~0.80 | ~0.70 |
| Pedestrian Motion Description Rate | **>50%** | ~30% | ~20% | ~30% |

### Ablation Study

| Configuration | mean Dyn. Norm. EPE | Explanation |
|------|---------------------|------|
| TrackFlow (LE3DE2E) | **0.287** | Full model, high recall |
| TrackFlowBEVF (BEVFusion) | +10-22% degradation | mAP is only 2% lower but a large gap in recall |
| BEVFusion threshold 0.1 | 0.4816 | Low threshold still cannot compensate for poor recall |
| BEVFusion threshold 0.4 | 0.8176 | Severe degradation at high threshold |

### Key Findings
- TrackFlow explains over 50% of the motion on pedestrians, which is 20% more than DeFlow (a 1.5x improvement), representing an order of magnitude difference.
- On Threeway EPE, TrackFlow leads by only 1.5mm, but Bucket Normalized EPE reveals a huge performance gap—this perfectly demonstrates the failure of the old metric.
- DeFlow actually outperforms TrackFlow in the car category, but lags far behind on pedestrians.
- A detector's recall is more important than its mAP: BEVFusion's mAP is only 2% lower than LE3DE2E, but TrackFlowBEVF's performance degrades by 10-22%.
- In the subsequent Argoverse 2 2024 Scene Flow Challenge, Flow4D halved TrackFlow's dynamic error through architectural improvements, but did not use any category-aware loss.

## Highlights & Insights
- **Beating all prior work with the simplest method**: TrackFlow is essentially "detection + tracking" without any design specific to scene flow, yet it achieves SOTA. This is not because TrackFlow is incredibly strong, but because existing methods are incredibly poor on small targets—a wake-up call. The brilliance lies in leveraging mature category-rebalancing techniques from the detection field.
- **Deep reflection on evaluation metrics**: The core value of the paper lies not in the method itself, but in exposing the evaluation blind spots of the entire field. This research paradigm of "fixing the metrics before proposing methods" is highly worth learning—sometimes, identifying the problem is more important than solving it.
- **The recall vs precision trade-off is completely different in flow estimation**: While the two need to be balanced in detection tasks, the cost of missed detection in scene flow is catastrophic. This insight can be transferred to other tasks requiring dense point-wise predictions.

## Limitations & Future Work
- TrackFlow can only predict rigid flow (based on bounding boxes) and cannot handle non-rigid motions (such as the joint movements of pedestrian gait).
- It relies on the fixed classification taxonomy of closed-set detectors, failing to handle unknown object categories in the open world (though the authors note it can be replaced with a class-agnostic detector).
- The bucketing strategy depends on semantic annotations. Although the authors also demonstrated an alternative approach using volume-based bucketing, it requires further validation.
- It leaves unexplored the possibility of directly injecting category rebalancing techniques into end-to-end scene flow methods—which might be a more elegant direction.

## Related Work & Insights
- **vs FastFlow3D**: A supervised method based on PointPillars, serving as the foundational architecture for many subsequent works, but fails on pedestrians due to the lack of category rebalancing.
- **vs ZeroFlow**: An unsupervised method via distillation, achieving near-SOTA on Threeway EPE but explaining less than 20% of pedestrian motion.
- **vs NSFP**: Unsupervised method based on online optimization that optimizes an MLP per frame to minimize Chamfer distance, is computationally expensive and performs poorly on small targets.
- **vs Flow4D**: A follow-up work in the challenge that halved the error using a 4D voxel architecture, proving that architectural innovation can also benefit small targets.

## Rating
- Novelty: ⭐⭐⭐⭐ The contribution of the evaluation metric is greater than the method itself; work that exposes blind spots in the field is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The analysis of metrics is extremely thorough; detector ablation, category analysis, and visualizations are all well-executed.
- Writing Quality: ⭐⭐⭐⭐⭐ The title is engaging, the argument logic is clear, and the FAQ section candidly addresses reservations.
- Value: ⭐⭐⭐⭐⭐ Reshapes the evaluation standards of the entire scene flow field, and has been adopted by the Argoverse 2 challenge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] If open source is to win, it must go public](../../ICML2026/llm_pretraining/if_open_source_is_to_win_it_must_go_public.md)
- [\[CVPR 2025\] DreamText: High Fidelity Scene Text Synthesis](../../CVPR2025/llm_pretraining/dreamtext_high_fidelity_scene_text_synthesis.md)
- [\[CVPR 2025\] Robust Message Embedding via Attention Flow-Based Steganography](../../CVPR2025/llm_pretraining/robust_message_embedding_via_attention_flow-based_steganography.md)
- [\[CVPR 2025\] The Scene Language: Representing Scenes with Programs, Words, and Embeddings](../../CVPR2025/llm_pretraining/the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)
- [\[NeurIPS 2025\] Flatness is Necessary, Neural Collapse is Not: Rethinking Generalization via Grokking](../../NeurIPS2025/llm_pretraining/flatness_is_necessary_neural_collapse_is_not_rethinking_generalization_via_grokk.md)

</div>

<!-- RELATED:END -->
