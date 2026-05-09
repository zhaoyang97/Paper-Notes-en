---
title: >-
  [Paper Note] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos
description: >-
  [CVPR 2026][Object Detection][Novel Object Detection] This paper proposes the "Show, Don't Tell" paradigm: a SODC pipeline (HOIST-Former for hand-object detection → SAMURAI for tracking → DBSCAN for spatiotemporal clustering) automatically creates annotated datasets from human demonstration videos to train a lightweight F-RCNN customized detector (MOD). Without any language prompts, MOD achieves instance-level detection of novel objects, surpassing VLM baselines such as GroundingDINO, RexOmni, and YoloWorld in mAP and precision on the Meccano and in-house datasets, and is integrated end-to-end into a real robotic sorting system.
tags:
  - CVPR 2026
  - Object Detection
  - Novel Object Detection
  - Human Demonstration
  - Automatic Dataset Creation
  - Customized Detector
  - Robotic Sorting
date: 2026-05-08
content_hash: e00110566b3dd7dd
---

# Show, Don't Tell: Detecting Novel Objects by Watching Human Videos

**Conference**: CVPR 2026
**arXiv**: [2603.12751](https://arxiv.org/abs/2603.12751)
**Code**: None (from Robotics and AI Institute)
**Area**: Object Detection / Robotic Manipulation
**Keywords**: Novel Object Detection, Human Demonstration, Automatic Dataset Creation, Customized Detector, Robotic Sorting

## TL;DR

This paper proposes the "Show, Don't Tell" paradigm: a SODC pipeline (HOIST-Former for hand-object detection → SAMURAI for tracking → DBSCAN for spatiotemporal clustering) automatically creates annotated datasets from human demonstration videos to train a lightweight F-RCNN customized detector (MOD). Without any language prompts, MOD achieves instance-level detection of novel objects, surpassing VLM baselines such as GroundingDINO, RexOmni, and YoloWorld in mAP and precision on the Meccano and in-house datasets, and is integrated end-to-end into a real robotic sorting system.

## Background & Motivation

**Background**: When robots learn tasks from human demonstrations, they must rapidly identify novel objects appearing in those demonstrations. Large-scale object detection models fall into two categories: closed-set detectors (YOLO, Faster R-CNN), which only detect objects within trained categories and are powerless against OOD objects; and open-set detectors/VLMs (GroundingDINO, RexOmni, YoloWorld), which can theoretically detect novel objects but rely heavily on the precision of language prompts.

**Limitations of Prior Work**: (1) Closed-set detectors cannot detect custom-manufactured parts, assembled objects, or other out-of-distribution objects; (2) VLMs require tedious manual prompt engineering to uniquely describe each object instance—some objects are difficult to distinguish linguistically (e.g., "green yellow red worm toy" in Figure 1 still cannot be recognized by VLMs); (3) the community has shifted toward automated prompt tuning, which introduces additional complexity.

**Key Challenge**: The inherent ambiguity and incompleteness of language make it a natural bottleneck for describing specific object instances—two visually similar but functionally distinct parts may be indistinguishable in language yet perfectly discriminable visually.

**Goal**: Bypass the linguistic intermediary entirely and self-supervisedly create training data from visual signals in human demonstration videos to train customized detectors.

**Key Insight**: Rather than "telling" the detector what to find, directly "show" it. Hand-object interactions in human demonstration videos serve as natural supervisory signals—whatever the human grasps is the task-relevant object.

**Core Idea**: Automatically create annotated datasets from human demonstration videos using hand-object interaction detection, tracking, and spatiotemporal clustering, then train a customized F-RCNN detector, completely bypassing language prompts.

## Method

### Overall Architecture

A three-stage pipeline: (1) SODC (Salient Objects Dataset Creation) automatically generates annotated datasets from human demonstration videos; (2) MOD (Manipulated Objects Detector) fine-tunes a lightweight F-RCNN on the SODC dataset; (3) end-to-end robotic system integration (plan skeleton generation + MOD detection + scene graph + execution). The full understanding pipeline requires approximately 4–7 minutes on a 15-second video.

### Key Designs

1. **SODC Pipeline: Detecting Grasped Objects**

   - **Function**: Identify objects being manipulated by humans in the video.
   - **Mechanism**: HOIST-Former serves as the hand-object interaction detector, outputting segmentation masks of grasped objects per frame. Each frame is processed independently without inter-frame association (due to excessive label persistence noise in HOIST-Former). This step only produces masks in frames where a hand is actively grasping an object; no output is generated when objects are occluded or not in hand.
   - **Design Motivation**: Hand-object interaction defines "task-relevant objects"—whatever the human grasps is what needs to be detected—eliminating any linguistic definition.

2. **SODC Pipeline: Tracking + Spatiotemporal Clustering**

   - **Function**: Extend sparse grasp masks into complete object tracks across the full video, then cluster them into per-object annotated datasets.
   - **Mechanism**: **Tracking**—each mask output by HOIST-Former serves as a seed; SAMURAI tracks bidirectionally across the entire video, producing a large number of tracks (far exceeding the actual object count, as each object is detected in multiple frames). **Spatial clustering**—DBSCAN (with an IoU distance function) clusters overlapping bounding boxes within each frame. **Temporal clustering**—for each track, a "cluster track" is generated (the sequence of spatial clusters the track belongs to across frames); Jaccard similarity between tracks is computed to aggregate tracks belonging to the same object, and clusters with insufficient track counts are discarded as noise.
   - **Design Motivation**: Pure spatial clustering incorrectly merges distinct objects when they overlap (e.g., five overlapping boxes from two objects at $t=30$ in Figure 3); the temporal dimension exploits motion trajectory differences to resolve correct associations. The combination of spatial and temporal clustering is robust to noise and brief occlusions.

3. **MOD: Customized Object Detector**

   - **Function**: Train a lightweight detector on datasets automatically created by SODC.
   - **Mechanism**: Fine-tune a pretrained F-RCNN (ResNet50 backbone) with standard RCNN losses (classification + objectness). Training takes approximately 3–4 minutes on 4×T4 GPUs. Extensive data augmentation is applied (flipping, distortion, brightness, contrast, color, cropping, scaling, blurring, and affine transforms).
   - **Design Motivation**: Rather than pursuing a general-purpose detector, a small specialized detector is trained per task. Although generality is sacrificed, task-specific precision far exceeds that of general VLMs, and training is extremely fast.

4. **End-to-End Robotic Integration**

   - **Function**: Integrate SODC+MOD into a real robotic sorting task.
   - **Mechanism**: (1) Record a human sorting demonstration video; (2) GPT-4o generates a plan skeleton (pick/place sequence) from the video; (3) SODC creates the dataset and MOD trains the detector; (4) MOD detects manipulated objects, a VLM detects placement targets (e.g., baskets), a scene graph aggregates point clouds, and pick-and-place actions are executed following the plan skeleton.
   - **Design Motivation**: MOD handles manipulated objects (novel, requiring instance-level discrimination) while a VLM handles placement targets (semantic-level suffices, e.g., "basket")—each detector serves its appropriate role.

### Loss & Training

- Standard F-RCNN loss (classification + bounding box regression + objectness)
- Training on 4×T4 GPUs for 3–4 minutes
- Extensive geometric and color augmentation to compensate for limited training data

## Key Experimental Results

### Main Results — Object Detection Performance Comparison

| Dataset | Method | Prompt | mAP₀.₅₋₀.₉₅ | mAR₁ | F1₀.₅₋₀.₉₅ | Precision | Recall |
|---------|--------|--------|-------------|------|-----------|-----------|--------|
| Meccano | RexOmni | Human | 0.05 | 0.09 | 0.30 | 0.59 | 0.23 |
| Meccano | GroundingDINO | Human | 0.19 | 0.26 | 0.24 | 0.46 | 0.18 |
| Meccano | YoloWorld | Human | 0.03 | 0.03 | 0.00 | 0.01 | 0.00 |
| Meccano | **MOD (Ours)** | **None** | 0.06 | 0.10 | 0.18 | **0.71** | 0.12 |
| In-House #1 | RexOmni | GPT | 0.06 | 0.09 | **0.98** | 1.00 | 0.97 |
| In-House #1 | GroundingDINO | Human | 0.04 | 0.08 | 0.87 | 1.00 | 0.82 |
| In-House #1 | **MOD (Ours)** | **None** | **0.10** | **0.17** | 0.92 | 1.00 | 0.87 |
| In-House #2 | RexOmni | GPT | 0.09 | 0.12 | **0.99** | 1.00 | 0.99 |
| In-House #2 | GroundingDINO | GPT | 0.08 | 0.10 | 0.98 | 1.00 | 0.96 |
| In-House #2 | **MOD (Ours)** | **None** | **0.15** | **0.19** | 0.95 | 1.00 | 0.91 |

### Key Metric Comparison

| Dimension | MOD (Ours) | Best VLM Baseline | Notes |
|-----------|-----------|-------------------|-------|
| mAP₀.₅₋₀.₉₅ (In-House #2) | **0.15** | 0.09 (RexOmni) | MOD leads by 67% on strict mAP |
| Precision (Meccano) | **0.71** | 0.59 (RexOmni) | MOD prediction accuracy far exceeds VLMs |
| Requires manual prompts | **No** | Yes | MOD is fully automated |
| Training time | 3–4 min | 0 (inference only) | MOD requires modest training time |

### Key Findings

- MOD outperforms all VLM baselines on strict mAP (IoU 0.5–0.95) across all in-house datasets; however, on Meccano, GroundingDINO (human-prompt) achieves higher mAP (0.19 vs. 0.06) because Meccano objects are more common categories.
- MOD achieves substantially higher precision (0.71–1.00), indicating a very low false detection rate—for robotic manipulation, precision is more critical than recall (the cost of a wrong grasp is high).
- VLM baselines sometimes achieve higher F1 (e.g., RexOmni reaches 0.98 on In-House #1) because VLMs tend to detect all objects (high recall), whereas MOD is more conservative (precision-first).
- GPT prompts benefit VLMs inconsistently: for RexOmni on In-House #1, GPT prompts actually degrade performance relative to human prompts in mAP (GPT: 0.06 vs. Human: 0.04), underscoring the unreliability of prompt engineering.
- The end-to-end pipeline (video → dataset → detector → robot execution) completes in 4–7 minutes, validating practical deployment feasibility.

## Highlights & Insights

- **Paradigm Shift**: Transitioning from "language-description-driven" (Tell) to "visual-demonstration-driven" (Show) represents a fundamental reorientation in novel object detection. Language has inherent limitations in describing specific object instances, whereas visual presentation conveys appearance characteristics precisely.
- **Elegant SODC Spatiotemporal Clustering Design**: Spatial DBSCAN first clusters overlapping boxes per frame, then Jaccard similarity aggregates cluster tracks along the temporal dimension. Pure spatial clustering fails when objects overlap; the temporal dimension leverages motion trajectory differences to resolve correct associations.
- **Pragmatic Trade-off Between Customization and Generalization**: Rather than pursuing a universal detector, a small specialized detector is trained per task. A 3–4 minute training cost is acceptable in exchange for precision far superior to general-purpose methods.
- **End-to-End Robotic Deployability**: The complete system closes the loop from demonstration video to robot execution; the hybrid strategy of MOD for manipulated objects and VLM for semantic targets leverages the strengths of each approach.

## Limitations & Future Work

- MOD achieves only 0.06 mAP on the Meccano dataset (below GroundingDINO's 0.19); SODC data quality for small components and complex assembly scenarios requires improvement.
- The pipeline depends on HOIST-Former as the seed detector—if hand-object interaction detection quality is poor, the entire pipeline fails in a cascading manner.
- Only objects directly manipulated by human hands are detected; task-relevant but unmanipulated objects in the scene (e.g., obstacles) cannot be detected.
- Retraining the detector (3–4 min) is required for each new task, and cumulative costs increase when many novel objects are involved.
- The experimental datasets are limited in scale (Meccano: 19 videos; In-House: 54 and 61 videos), and large-scale evaluation is absent.
- The method is not evaluated on standard object detection benchmarks (e.g., LVIS rare categories), and comparisons with few-shot detection methods are lacking.

## Related Work & Insights

- **vs. GroundingDINO / OWL-ViT / RexOmni (open-set detectors)**: These methods rely on language prompts and lack instance-level discriminability for novel objects. MOD bypasses the language bottleneck through visual demonstration, achieving significantly higher precision.
- **vs. Few-Shot Object Detection (FSOD, etc.)**: Support-set-based methods require a small number of annotated samples. MOD eliminates annotation requirements entirely through automated SODC.
- **vs. HOIST-Former (hand-object interaction detection)**: HOIST-Former only detects objects currently in hand and operates frame-independently. MOD uses SODC to extend it into a full-scene detector capable of detecting objects not currently in hand.
- **vs. Behavior Cloning / Imitation Learning (RT-2, etc.)**: End-to-end approaches fold object detection into the model but require training data collected on the robot. MOD requires no robot data—only human demonstration videos.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐: The Show Don't Tell paradigm presents a clear conceptual shift in novel object detection; the SODC spatiotemporal clustering design is creative.
- **Experimental Thoroughness** ⭐⭐⭐: Dataset scale is limited, standard benchmark evaluation is absent, and ablation studies are insufficiently systematic.
- **Writing Quality** ⭐⭐⭐⭐: The "Show vs. Tell" analogy is intuitive and memorable; the SODC pipeline is explained clearly.
- **Value** ⭐⭐⭐⭐: The approach has direct practical value for rapid robotic deployment and lowers the barrier to use for non-expert users.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Detecting Unknown Objects via Energy-based Separation for Open World Object Detection](detecting_unknown_objects_via_energy-based_separation_for_open_world_object_dete.md)
- [\[CVPR 2026\] NoOVD: Novel Category Discovery and Embedding for Open-Vocabulary Object Detection](noovd_novel_category_discovery_and_embedding_for_open-vocabulary_object_detectio.md)
- [\[CVPR 2026\] PHAC: Promptable Human Amodal Completion](phac_promptable_human_amodal_completion.md)
- [\[CVPR 2026\] AR²-4FV: Anchored Referring and Re-identification for Long-Term Grounding in Fixed-View Videos](ar2-4fv_anchored_referring_and_re-identification_for_long-term_grounding_in_fixe.md)
- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)

<!-- RELATED:END -->
