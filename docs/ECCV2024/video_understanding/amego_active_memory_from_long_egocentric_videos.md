---
title: >-
  [Paper Note] AMEGO: Active Memory from Long EGOcentric Videos
description: >-
  [ECCV 2024][Video Understanding][Egocentric Video] Proposes AMEGO, a method for online construction of structured "active memory" from long egocentric videos. By combining HOI tracklets, location segments, and semantic-free visual queries, it outperforms Video QA baselines by 12.7% on the newly proposed AMB benchmark.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Egocentric Video"
  - "Long Video Understanding"
  - "Episodic Memory"
  - "Hand-Object Interaction"
  - "Structured Representation"
date: 2026-05-08
content_hash: 6b0939b5ce09d587
---

# AMEGO: Active Memory from Long EGOcentric Videos

**Conference**: ECCV 2024  
**arXiv**: [2409.10917](https://arxiv.org/abs/2409.10917)  
**Code**: [https://gabrielegoletto.github.io/AMEGO/](https://gabrielegoletto.github.io/AMEGO/)  
**Area**: Video Understanding / Egocentric Vision  
**Keywords**: Egocentric Video, Long Video Understanding, Episodic Memory, Hand-Object Interaction, Structured Representation

## TL;DR
Proposes AMEGO, a method for online construction of structured "active memory" from long egocentric videos. By combining HOI tracklets, location segments, and semantic-free visual queries, it outperforms Video QA baselines by 12.7% on the newly proposed AMB benchmark.

## Background & Motivation
**Background**: Egocentric video understanding of long videos (ranging from tens of minutes to hours) is a popular research direction. Existing methods either uniformly sample frame features (losing activity structure) or use LLMs for captioning and QA (which are semantically bound and coarse-grained), failing to adequately understand the details of object interactions in truly long videos.

**Limitations of Prior Work**:
   - **Neglecting human activity structure**: Uniform sampling fails to capture where the person is, when they interact with objects, or which hand is used.
   - **Reliance on semantic labels**: Training encoders requires semantically annotated QA pairs, which is limited by a fixed vocabulary.
   - **Lack of explainability**: Implicit feature representations cannot directly reveal the content of human activities.
   - **Inability to distinguish fine-grained objects**: Semantic labels (e.g., "cup") cannot differentiate between different instances of cups.

**Key Challenge**: Long video understanding requires fine-grained activity perception (who used what where), but existing methods are either too coarse (uniform sampling) or too heavily reliant on semantics (fixed vocabulary).

**Goal**: (a) To build a structured representation for long videos that does not rely on semantic labels; (b) To support semantic-free visual queries.

**Key Insight**: Mimicking human episodic memory—processing videos online, recording only object interactions and location transitions to construct a lightweight "active memory".

**Core Idea**: Utilizing an HOI detector, a single-object tracker, and DINOv2 features to construct hand-object interaction tracklets and location segments online, forming a queryable, semantic-free structured memory.

## Method

### Overall Architecture
Takes a long egocentric video as input and outputs the AMEGO representation $\mathcal{E} = \{\mathcal{O}, \mathcal{L}\}$, consisting of a set of HOI tracklets (object interaction trajectories) and Location segments. The processing is done online in three steps: initializing new interactions $\rightarrow$ tracking ongoing interactions $\rightarrow$ matching object/location instances after termination. Queries are semantic-free, performed using visual feature matching.

### Key Designs

1. **HOI Tracklet Construction (Online three steps)**:

    - *Initialization*: Frame-by-frame detection using a class-agnostic hand-object detector. Spatial overlap across $s_o$ consecutive frames is required to confirm a new interaction (to filter out noise).
    - *Update/Tracking*: After initialization, a single-object tracker (SOT) is employed to continuously track the object, maintaining the trajectory even when the hand leaves the field of view. The termination condition is defined as the hand being visible but without associated detections for $e_o$ consecutive frames.
    - *Instance Matching*: Upon termination, DINOv2 is used to extract object appearance features, which are matched with existing instances via cosine similarity. If the similarity exceeds a predefined threshold, the existing instance is assigned; otherwise, a new instance is created.
    - Design Motivation: The HOI detector is adept at identifying new interactions but lacks tracking stability, whereas the SOT provides stable tracking but does not know when to start or end—the two are complementary.

2. **Location Segment Construction**:

    - Function: To identify the time intervals when the camera wearer is in different "activity hotspots".
    - Mechanism: Utilizing low optical flow values combined with the presence of hand detections to determine when "the person has stopped to perform an action"; a location segment starts if the condition is met for $s_l$ consecutive frames, and ends if it is unmet for $e_l$ consecutive frames. A separate visual feature extractor $\sigma$ is used for location instance matching.
    - Design Motivation: Location is crucial for understanding activity context—the significance of using the same object in the kitchen versus the living room can be entirely different.

3. **Semantic-free Query**:

    - Function: Given a cropped image of an object or location, retrieve all relevant interaction information.
    - Mechanism: Extract query image features using DINOv2 $\rightarrow$ match with instances in AMEGO $\rightarrow$ return all associated tracklets/segments.
    - Design Motivation: Eliminates the need for a predefined vocabulary and allows distinguishing different instances of the same category (e.g., two different cups)—which semantic approaches fail to achieve.

### Loss & Training
- **Completely training-free**: All components (HOI detector, SOT tracker, DINOv2, optical flow) are off-the-shelf pre-trained models.
- No training or fine-tuning is required.

## Key Experimental Results

### Main Results (AMB Benchmark)

| Method | Type | Overall Accuracy |
|------|------|-----------|
| SF-QA (obj) | Semantic-free QA | 21.2% |
| S-QA (BLIP-2) | Semantic QA | 23.6% |
| LLoVi (LaViLa+BLIP-2) | LLM pipeline | 22.6% |
| **AMEGO - S** | Structured Representation | **33.8%** |
| **AMEGO - L** | Structured Representation | **36.3%** |
| Random Guess | - | 20.0% (1-out-of-5) |

$\rightarrow$ AMEGO-L outperforms the best baseline by +12.7% (36.3% vs 23.6%).

### Sub-results
- **Sequencing**: AMEGO takes a substantial lead (~35% vs ~22%).
- **Concurrency**: AMEGO shows the most significant advantage (~40% vs ~27%).
- **Temporal Grounding**: AMEGO demonstrates a clear advantage in location-related queries (~45% vs ~22%).
- **Q5 (Concurrent objects with the same hand)**: AMEGO performs the poorest (24.7%) because the HOI detector struggles to handle the same hand holding multiple objects simultaneously.

### Key Findings
- **All VQA baselines performance is near-random**: Even powerful models such as BLIP-2 and LaViLa are only slightly better than random guessing on such fine-grained long video understanding.
- **LLM pipeline (LLoVi) performs worse**: Textual summarization loses too much fine-grained information.
- **The longer the video, the faster the performance of Semantic-Free QA drops**: However, AMEGO maintains its advantage on long videos.
- **AMEGO-L (with location info) outperforms AMEGO-S (object only) by 2.5%**: Location information is valuable.

## Highlights & Insights
- **"Remembering" rather than "understanding" long videos**: Unlike traditional approaches that attempt to "understand" video content, AMEGO merely "records" structured interaction events—aligning closely with how human episodic memory functions.
- **Semantic-free design**: Avoids binding to a fixed vocabulary, employing visual feature matching instead of semantic matching. This enables the system to distinguish different instances of the same category and achieves zero-shot generalization to novel objects.
- **Training-free pipeline**: Uses entirely off-the-shelf models without requiring any training—resulting in exceptionally high reproducibility and deployment convenience.
- **Complementary combination of HOI detection and SOT tracking**: Leverages the "when to start" signals from the HOI detector alongside the stable tracking capabilities of the SOT, elegantly addressing the issue of objects frequently entering and leaving the field of view in egocentric videos.

## Limitations & Future Work
- **Weakness in Q5**: The HOI detector struggles to recognize cases where the same hand holds multiple objects simultaneously; better multi-object interaction detection is needed.
- **Coarse location segmentation**: Relying solely on optical flow and hand detection may lead to misjudgments (e.g., looking down at a phone displays low optical flow and hand visibility, but does not represent a physical location-based activity).
- **Exclusive focus on hand-object interactions**: Neglects other activity cues such as body posture and gaze direction.
- **Limited scale of the AMB benchmark**: Although containing over 20K queries, it is based solely on EPIC-KITCHENS (kitchen environments); generalization to other environments remains to be validated.
- **Sensitivity to DINOv2 matching thresholds**: Object and location instance matching relies heavily on manually defined similarity thresholds.

## Related Work & Insights
- **vs ReST Benchmark**: ReST also employs visual queries but focuses only on objects rather than locations. AMB is more comprehensive, considering objects, locations, and interactions simultaneously.
- **vs Semantic-QA (BLIP-2, etc.)**: Semantic approaches are limited in fine-grained object differentiation (where "cup" refers to all cups); AMEGO resolves this via instance-level visual matching.
- **vs LLoVi**: Multi-stage LLM pipelines lose fine-grained visual details during video-to-text translation, performing worse than direct visual matching.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of structured active memory is novel, the semantic-free design is unique, and the AMB benchmark is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ New benchmark + multiple baseline comparisons + video length analysis + qualitative analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Methodology diagrams are clear, the pipeline is described in detail, and the motivation is convincingly articulated.
- Value: ⭐⭐⭐⭐ Possesses direct application value for long video understanding and wearable displays/devices (AR/VR, robotics).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Spherical World-Locking for Audio-Visual Localization in Egocentric Videos](spherical_world-locking_for_audio-visual_localization_in_egocentric_videos.md)
- [\[ECCV 2024\] Goldfish: Vision-Language Understanding of Arbitrarily Long Videos](goldfish_vision-language_understanding_of_arbitrarily_long_videos.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](../../CVPR2025/video_understanding/rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [\[ECCV 2024\] RGNet: A Unified Clip Retrieval and Grounding Network for Long Videos](rgnet_a_unified_clip_retrieval_and_grounding_network_for_long_videos.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)

</div>

<!-- RELATED:END -->
