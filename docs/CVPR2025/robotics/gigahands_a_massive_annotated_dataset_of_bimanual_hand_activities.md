---
title: >-
  [Paper Note] GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities
description: >-
  [CVPR 2025][Robotics][Bimanual Activity Dataset] GigaHands is the largest bimanual activity dataset to date. By designing an "Instruct-to-Annotate" procedural acquisition strategy and a 51-camera markerless capture system, it collects 34 hours of bimanual activities from 56 subjects interacting with 417 objects. It contains 183 million RGB image frames and 84K detailed text annotations, demonstrating the value of data scale in text-driven hand motion generation and motion cap…
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Bimanual Activity Dataset"
  - "Hand Motion Capture"
  - "Text Annotation"
  - "Motion Generation"
  - "Markerless Capture"
date: 2026-05-08
content_hash: c78e921502af12b7
---

# GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities

**Conference**: CVPR 2025  
**arXiv**: [2412.04244](https://arxiv.org/abs/2412.04244)  
**Code**: [https://ivl.cs.brown.edu/research/gigahands.html](https://ivl.cs.brown.edu/research/gigahands.html)  
**Area**: Robotics  
**Keywords**: Bimanual Activity Dataset, Hand Motion Capture, Text Annotation, Motion Generation, Markerless Capture

## TL;DR

GigaHands is the largest bimanual activity dataset to date. By designing an "Instruct-to-Annotate" procedural acquisition strategy and a 51-camera markerless capture system, it collects 34 hours of bimanual activities from 56 subjects interacting with 417 objects. It contains 183 million RGB image frames and 84K detailed text annotations, demonstrating the value of data scale in text-driven hand motion generation and motion captioning tasks.

## Background & Motivation

**Background**: Understanding bimanual activities is a core problem in AI and robotics. Current data acquisition mainly relies on two paradigms: (1) in-the-wild collection (monocular/handheld cameras), which provides realistic data but poor 3D reconstruction accuracy; (2) studio collection (markers/multi-camera setups), which offers high precision but markers hinder natural movements, and diversity is limited by manually designed scenarios.

**Limitations of Prior Work**: Existing datasets are insufficient in scale, activity coverage, and annotation granularity to support the training of large-scale hand activity models. For instance, ARCTIC only contains 121 minutes with 11 objects; TACO has 202 minutes with 196 objects; OakInk2 contains 557 minutes but with sparse text annotations. More importantly, marker-based capture systems not only suppress natural gestures (such as self-contact) but also increase the cost of post-processing to restore real appearance.

**Key Challenge**: The trilemma of scale vs. quality vs. diversity—in-the-wild data is large-scale but has poor 3D quality, while studio data has high quality but limited diversity and high collection costs.

**Goal**: How to efficiently capture a large-scale, highly diverse, and high-precision bimanual activity dataset with detailed text annotations.

**Key Insight**: Design a procedural "Instruct-to-Annotate" pipeline. LLMs are leveraged to automatically generate activity instruction scripts to guide subjects. This ensures diversity (covered by scripts) while significantly reducing the subsequent annotation workload (since each recording session directly corresponds to an instruction). The acquisition system employs 51 synchronized RGB cameras to replace markers, achieving high-precision markerless capture.

**Core Idea**: Utilize procedural instruction guidance combined with markerless multi-view capture and an automated 3D estimation pipeline to simultaneously address issues of data scale, diversity, and annotation quality.

## Method

### Overall Architecture

The entire pipeline is divided into four stages: (1) procedural instruction generation—extracting a verb pool from existing datasets and using LLMs to organize them into scenario-based activity scripts; (2) capture—subjects operate on objects in a 51-camera cubic system following instructions; (3) annotation and augmentation—annotators segment and correct the recorded sequences, then use LLMs to augment text descriptions; (4) 3D estimation—a fully automated pipeline estimates 3D hand and object shapes and poses.

### Key Designs

1. **Procedural Instruction Guidance (Instruct-to-Annotate)**:

    - **Function**: Automatically generate recording instruction scripts covering diverse activities while reducing annotation workload.
    - **Mechanism**: (a) Parse atomic actions from datasets such as Ego4D, Ego-Exo4D, OakInk2, and TACO to establish a verb pool of 533 items; (b) After manually associating verbs with objects, use an LLM to group them into scenarios (5 major scenarios like cooking, office, handcrafts, and 25 sub-scenarios); (c) The LLM organizes verbs and objects within each sub-scenario into temporally coherent activity lists, automatically generating detailed instruction scripts; (d) Instructions are converted to audio and played sequentially during recording to guide the subjects. Ultimately, 1,370 instructions covering 533 verbs and 191 types of activities are generated.
    - **Design Motivation**: Traditional collection where subjects perform freely often leads to repetitive actions and high post-annotation costs. In contrast, preset instructions make each recording segment naturally map to a specific text description.

2. **51-Camera Markerless Capture System and Automated 3D Estimation**:

    - **Function**: Achieve high-precision 3D hand and object motion estimation without using physical markers.
    - **Mechanism**: Hand estimation pipeline: YOLOv8 for hand detection $\rightarrow$ HaMeR to estimate MANO mesh $\rightarrow$ ViTPose to determine left/right hands $\rightarrow$ multi-view triangulation for precise 3D keypoints $\rightarrow$ One-Euro filter for temporal smoothing $\rightarrow$ EasyMoCap to fit MANO parameters. Object estimation pipeline: DINOv2 + Grounding DINO for object detection $\rightarrow$ OpenCLIP to filter false positives $\rightarrow$ SAM2 for mask segmentation $\rightarrow$ Instant-NGP to build a neural radiance field to initialize translation $\rightarrow$ FoundPose + DINOv2 to initialize rotation $\rightarrow$ PyTorch3D differentiable rendering to optimize 6D poses under multi-view mask supervision.
    - **Design Motivation**: Markerless capture ensures natural movements are undistracted, while 51 camera viewpoints provide sufficient multi-view redundancy to compensate for any accuracy loss.

3. **Text Annotation and Augmentation**:

    - **Function**: Provide diverse and fine-grained text descriptions for each action segment.
    - **Mechanism**: Annotators segment 13K recording sequences into 14K clips, correcting discrepancies between instructions and actual actions (e.g., when subjects misunderstood instructions or improvised). An LLM is then used to paraphrase each description 5 times, augmenting the 14K clips into 84K action-text pairs covering 1,467 unique verbs.
    - **Design Motivation**: Diverse textual expressions facilitate text-action alignment learning. Furthermore, the selection of 1,467 verbs exceeds any existing dataset (including the in-the-wild Ego4D dataset).

### Loss & Training

The dataset itself does not involve training. In downstream applications, hand motion generation is trained using the T2M-GPT framework with the standard VQ-VAE + GPT training strategy, while motion captioning employs the TM2T framework.

## Key Experimental Results

### Dataset Scale Comparison

| Dataset | Duration (min) | # Actions | # Poses | # Views | # Frames | Subjects | Objects |
|--------|----------|--------|--------|--------|------|------|------|
| ARCTIC | 121 | 339 | 218k | 9 | 2.1M | 10 | 11 |
| TACO | 202 | 2.3k | 363k | 13 | 4.7M | 14 | 196 |
| OakInk2 | 557 | 2.8k | 993k | 4 | 4.01M | 9 | 75 |
| **GigaHands** | **2,034** | **13.9k** | **3.7M** | **51** | **183M** | **56** | **417** |

### Text-Driven Hand Motion Generation (T2M-GPT)

| Training Dataset | R@1 (%) | R@3 (%) | FID ↓ | Diversity | MM. |
|-----------|--------|--------|------|-----------|-----|
| TACO | 18.9 | 52.9 | 11.0 | 11.1 | 6.83 |
| OakInk2 | 17.9 | 47.9 | 19.6 | 6.88 | 3.45 |
| **GigaHands** | **31.2** | **53.1** | **4.70** | **10.5** | **9.11** |

### Key Findings

- Growth in dataset scale consistently brings performance improvements: training with $10\% \rightarrow 100\%$ of GigaHands data monotonically improves FID, MM Dist, and Top-1/3 accuracy, indicating that hand activity modeling is not yet saturated.
- Models trained on GigaHands not only perform best on its own test set but also generate plausible hand motions for texts from other datasets (OakInk2, TACO), demonstrating the generalization capability enabled by data diversity.
- In the motion captioning task, GigaHands achieves an R@1 of 57.0%, significantly outperforming OakInk2's 40.4%, proving that larger-scale and more diverse text annotations effectively enhance text-motion alignment learning.

## Highlights & Insights

- **Procedural Data Collection Paradigm**: Generating instruction scripts via LLMs not only ensures activity diversity but also shifts the annotation challenge from "post-hoc extraction from video" to "upfront constraint via instructions," drastically reducing annotation costs. This paradigm can be extended to any scenario requiring large-scale behavioral data collection.
- **Diversity Verification via t-SNE**: The paper does not merely claim data diversity; it quantitatively verifies it using t-SNE visualizations of hand poses/motions alongside UpSet plots for verb coverage analysis, presenting a robust methodology.
- **Fully Automated 3D Estimation Pipeline**: A complete 3D hand+object estimation workflow is constructed by integrating recent foundation models (HaMeR, SAM2, DINOv2, Grounding DINO). While the individual modules are not novel, this integration scheme holds high practical value.

## Limitations & Future Work

- The studio environment remains artificial. Although instructions mimic in-the-wild activities, they lack real scenario context (e.g., a real kitchen instead of kitchenware laid out on a tabletop).
- There is still a gap in 3D hand precision between markerless capture and marker-based methods, particularly in detailed areas like fingertips.
- Objects are limited to tabletop-manipulable rigid/semi-rigid items, lacking deformable objects (such as cloth) or liquids.
- Text augmentation relies on LLM paraphrasing, which may introduce semantic drifts or hallucinations.
- The high replication cost of the 51-camera system limits data expansion by other research teams.

## Related Work & Insights

- **vs. ARCTIC**: ARCTIC captures bimanual hand-object interactions using markers, yielding high precision but small scale (121 minutes, 11 objects). Additionally, markers impede movements like self-contact. GigaHands is 17 times larger in scale, with activity diversity far exceeding ARCTIC.
- **vs. Ego4D / Ego-Exo4D**: These in-the-wild datasets are large-scale but suffer from poor 3D precision and sparse annotations. GigaHands features more unique verbs (1,467) than even the in-the-wild Ego4D dataset, demonstrating the effectiveness of procedural instructions.
- **vs. OakInk2**: OakInk2 also provides text annotations, but its scale and verb diversity are inferior to GigaHands.

## Rating

- Novelty: ⭐⭐⭐⭐ The procedural instruction-guided data acquisition paradigm is the core contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Demonstrates two downstream tasks (motion generation and captioning) along with data scale ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured, rich in visualization, and provides comprehensive dataset comparisons.
- Value: ⭐⭐⭐⭐⭐ Large-scale datasets hold long-term value for the community, and the acquisition methodology is highly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SortScrews: A Dataset and Baseline for Real-time Screw Classification](sortscrews_a_dataset_and_baseline_for_real-time_screw_classification.md)
- [\[CVPR 2025\] ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping](manivideo_generating_hand-object_manipulation_video_with_dexterous_and_generaliz.md)
- [\[CVPR 2025\] ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning](maniptrans_efficient_dexterous_bimanual_manipulation_transfer_via_residual_learn.md)
- [\[ICCV 2025\] GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices](../../ICCV2025/robotics/guiodyssey_a_comprehensive_dataset_for_cross-app_gui_navigation_on_mobile_device.md)
- [\[ICCV 2025\] AnyBimanual: Transferring Unimanual Policy for General Bimanual Manipulation](../../ICCV2025/robotics/anybimanual_transferring_unimanual_policy_for_general_bimanual_manipulation.md)

</div>

<!-- RELATED:END -->
