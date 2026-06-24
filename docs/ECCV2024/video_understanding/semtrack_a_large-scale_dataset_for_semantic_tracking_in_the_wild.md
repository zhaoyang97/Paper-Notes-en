---
title: >-
  [Paper Note] SemTrack: A Large-Scale Dataset for Semantic Tracking in the Wild
description: >-
  [ECCV 2024][Video Understanding][Semantic Tracking] This paper proposes the SemTrack dataset and SemTracker method, expanding traditional object tracking from "locating where the target is" to "understanding what the target is doing"—tracking targets while capturing their semantic trajectories (who/what they interact with, when, where, and how they interact), and introducing a meta-learning strategy to address the challenges of long-tailed interaction categories.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Semantic Tracking"
  - "Dataset"
  - "Object Tracking"
  - "Interaction Recognition"
  - "Semantic Trajectory"
date: 2026-05-08
content_hash: d240b0744e9d991e
---

# SemTrack: A Large-Scale Dataset for Semantic Tracking in the Wild

**Conference**: ECCV 2024  
**Code**: [https://github.com/sutdcv/SemTrack](https://github.com/sutdcv/SemTrack)  
**Area**: Video Understanding  
**Keywords**: Semantic Tracking, Dataset, Object Tracking, Interaction Recognition, Semantic Trajectory

## TL;DR
This paper proposes the SemTrack dataset and SemTracker method, expanding traditional object tracking from "locating where the target is" to "understanding what the target is doing"—tracking targets while capturing their semantic trajectories (who/what they interact with, when, where, and how they interact), and introducing a meta-learning strategy to address the challenges of long-tailed interaction categories.

## Background & Motivation

**Background**: Object tracking is one of the most classic tasks in computer vision, and existing methods are already capable of accurately locating targets across various scenarios. However, "knowing where the target is" is far from enough for many real-world applications. For instance, in customer analytics, merchants care not only about customers' movement trajectories but also about which products they interact with, how long they stay, and what behavioral patterns they exhibit. In public safety, understanding the behavioral semantics of a tracked target rather than just its position is crucial.

**Limitations of Prior Work**: (1) Existing tracking datasets (such as GOT-10K, LaSOT, TAO, etc.) only provide target position trajectory annotations (bounding boxes) and lack semantic information; (2) Although the video understanding field has tasks like action recognition and video relation detection, they usually handle short clips instead of long-term tracking and do not focus on the continuous semantic state of specific targets; (3) Research combining tracking and semantic understanding remains blank—there is no suitable dataset to train and evaluate this integrated capability.

**Key Challenge**: Traditional tracking only provides spatial localization (where) and lacks semantic understanding (what, who, how, when). However, real-world application demands go far beyond location tracking—they require simultaneously obtaining the "semantic trajectory" of targets, i.e., the sequence of interaction behaviors evolving over time.

**Goal**: (1) Construct a large-scale dataset that annotates both spatial and semantic trajectories of targets; (2) Define the new task of "semantic tracking"—tracking a target and capturing its semantic trajectory based on user input; (3) Propose an effective baseline method to validate the feasibility and challenges of the task.

**Key Insight**: The authors define semantic tracking as a joint task of tracking + interaction recognition. The semantic trajectory is modeled as a sequence of timestamped interaction events, where each event contains "who/what the target is interacting with" and "how the interaction occurs." By integrating multiple existing video datasets and performing large-scale annotation, they construct a dataset covering diverse scenarios and interaction categories.

**Core Idea**: Define the new task of semantic tracking, construct the large-scale annotated SemTrack dataset containing 6.7 million frames and 52 interaction categories, and propose the meta-learning-based SemTracker method to address the challenge of long-tailed distribution in interaction categories.

## Method

### Overall Architecture
The workflow of the SemTrack system is: (1) receive user input (text description or clicks) to initialize the target to be tracked; (2) continuously track the target location (bounding box sequence) throughout the video; (3) simultaneously recognize the target's interaction behaviors during each time period (which object it interacts with and what the interaction category is) while tracking; (4) output the complete semantic trajectory—a sequence of timestamped (subject, interaction object, interaction category, time interval) quadruplets.

### Key Designs

1. **SemTrack Dataset Construction (Dataset Construction)**:

    - **Function**: Provide large-scale, diverse semantic tracking annotations to support training and evaluation.
    - **Mechanism**: Integrate video resources from 8 existing video datasets (YFCC100M, TAO, VIDOR, VIDVRD, HACS, AVA, GOT-10K, ILSVRC2016), and generate semantic trajectory annotations through a multi-stage labeling process. The annotations include tracking bounding box sequences of targets, interaction relations between targets and other objects in the scene, and categories and time intervals of interaction behaviors. The dataset scales up to 6.7 million frames across 6,961 videos, covering 52 interaction categories, 115 object categories, and 10 supercategories across 12 different indoor and outdoor scenes.
    - **Design Motivation**: Existing datasets either only have location annotations (tracking datasets) or only have clip-level semantic annotations (action recognition datasets), lacking datasets that exhibit both long-term tracking and frame-by-frame semantic annotations. SemTrack fills this gap.

2. **SemTracker Baseline Method (Baseline Method)**:

    - **Function**: An end-to-end model capable of simultaneously executing target tracking and semantic trajectory prediction.
    - **Mechanism**: SemTracker adopts a dual-branch architecture—the tracking branch is responsible for predicting the bounding box position of the target in each frame, while the semantic branch is responsible for identifying the target's interaction behaviors. The tracking branch is a Transformer-based object tracker that takes the initial template and current frame as input to output target positions. Based on the target region features extracted by the tracking branch, combined with scene context features, the semantic branch predicts the interaction category and the interacted object in the current frame. The two branches share the underlying visual feature extractor and run synchronously during inference.
    - **Design Motivation**: Decoupling tracking and semantic understanding into two branches allows leveraging mature tracking techniques as localization foundations while enabling the semantic branch to focus on interaction recognition. The shared feature extractor reduces computational redundancy.

3. **Meta-learning Approach (Meta-learning Approach)**:

    - **Function**: Handle the long-tailed distribution challenge of interaction categories in the semantic tracking dataset.
    - **Mechanism**: Interaction categories in the SemTrack dataset exhibit a severe long-tailed distribution—common interactions like "walking" and "standing" have abundant samples, whereas rare interactions like "diving" and "rock climbing" have very few samples. Direct training severely biases the model towards head categories. SemTracker introduces a MAML-style meta-learning strategy, treating the semantic tracking task of each video as an independent episode, balancing the learning of different categories through rapid adaptation on few-shot samples within the episode. During the meta-training phase, the model learns a good initialization, allowing it to quickly adapt with only a few samples when facing new videos (which may contain rare interactions).
    - **Design Motivation**: Traditional oversampling or undersampling strategies have limited efficacy in video tasks because interaction events in videos are temporally dependent and cannot be sampled independently on a simple basis. Meta-learning naturally handles the class imbalance problem through task-level optimization.

### Loss & Training
The tracking sub-branch uses standard bounding box regression losses (L1 + GIoU). The semantic branch uses cross-entropy classification loss. Meta-learning employs episode-based training, where each episode samples a support set and a query set from a single video, rapidly adapting parameters using the support set in the inner loop, and updating meta-parameters using the query set in the outer loop.

## Key Experimental Results

### Dataset Statistics

| Metric | Value |
|------|------|
| Total Videos | 6,961 |
| Total Frames | 6.7M (6.7 million) |
| Interaction Categories | 52 |
| Object Categories | 115 |
| Supercategories | 10 |
| Scene Types | 12 (Indoor + Outdoor) |
| Data Sources | 8 Public Datasets |

### Main Results

| Method | Tracking Acc↑ | Interaction F1↑ | Semantic Traj. Acc↑ |
|------|--------------|----------------|---------------------|
| Tracker + Independent Classification | Higher | Lower | Lower |
| End-to-End (w/o Meta-learning) | Higher | Moderate | Moderate |
| **SemTracker (w/ Meta-learning)** | **Higher** | **Highest** | **Highest** |

### Ablation Study

| Configuration | Interaction F1↑ | Description |
|------|----------------|------|
| Full SemTracker | Optimal | Dual-branch + Meta-learning |
| w/o Meta-learning | Significant Decrease | Substantial drop in recognition rates for long-tailed categories |
| w/o Scene Context | Decrease | Missing interaction object information |
| Tracking Only (No Semantics) | N/A | Position tracking remains unaffected |

### Key Findings
- The main bottleneck of semantic tracking lies in interaction recognition rather than target tracking—modern trackers are already sufficiently accurate, but recognizing fine-grained interaction behaviors from tracking regions remains highly challenging.
- Meta-learning yields significant improvements for long-tailed categories—the recognition rate for tail categories (rare interactions) increases the most, while head categories remain almost unchanged.
- Scene context is vital for interaction recognition—many interaction behaviors require integrating information about the interacted objects to be correctly identified (e.g., "picking up" requires knowing what is being picked up).
- Existing models still perform poorly on complex interactions (involved in composite interactions across multiple objects), indicating that the challenges of the dataset are far from being resolved.

## Highlights & Insights
- **The task definition itself is a major contribution**: Expanding target tracking from "localization" to "understanding" opens up a new research direction. The concept of semantic trajectory—a sequence of interaction behaviors evolving over time—provides far richer information than simple location trajectories, holding immense application value (retail analysis, security surveillance, human-computer interaction, etc.).
- **Dataset scale and diversity are impressive**: 6.7 million frames, 52 interactions, 115 objects, and 12 scenes, integrated from 8 datasets, showing extensive coverage. The methodology of constructing new task datasets by integrating existing resources is also highly worth learning from.
- The idea of using meta-learning to handle long-tailed distribution can be directly migrated to other video understanding tasks with class imbalances—such as long-tailed action recognition, rare event detection, etc.

## Limitations & Future Work
- Some videos in the dataset come from different datasets, which may lead to discrepancies in annotation standards and quality; cross-dataset annotation consistency needs validation.
- The 52 interaction categories are still limited, whereas real-world interactions are far richer—subsequent works could consider open-vocabulary interaction recognition.
- The semantic branch of SemTracker is relatively simple (essentially a classification network); LLMs/VLMs can be introduced for more flexible semantic descriptions.
- The design of evaluation metrics still warrants further exploration—how to measure the quality of semantic trajectory prediction requires more detailed definitions.
- The training overhead of meta-learning is relatively high; more lightweight long-tailed handling strategies (such as decoupled training, class-conditional generation, etc.) could be explored.

## Related Work & Insights
- **vs GOT-10K/LaSOT**: These are classic object tracking datasets that only provide location annotations. SemTrack adds the semantic trajectory dimension on top of spatial annotations, serving as a conceptual extension of these datasets.
- **vs VIDOR/VidVRD**: These video relation detection datasets annotate relations between objects but do not involve continuous tracking. SemTrack integrates relation detection and long-term tracking into a unified task.
- **vs AVA**: The AVA dataset annotates atomic human actions, but each action is labeled independently without forming continuous semantic trajectories. SemTrack focuses on the temporal evolution sequence of a target's outer behaviors throughout the entire video.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Defined the completely new task of "semantic tracking" and built the first large-scale semantic tracking dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐ Provided a complete baseline method and ablation studies, establishing a benchmark.
- Writing Quality: ⭐⭐⭐⭐ Clear task definition; detailed dataset construction process.
- Value: ⭐⭐⭐⭐⭐ A dataset work that pioneers a new direction, carrying long-term impact on the community with broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VideoNet: A Large-Scale Dataset for Domain-Specific Action Recognition](../../CVPR2026/video_understanding/videonet_a_large-scale_dataset_for_domain-specific_action_recognition.md)
- [\[ECCV 2024\] SLAck: Semantic, Location, and Appearance Aware Open-Vocabulary Tracking](slack_semantic_location_and_appearance_aware_open-vocabulary_tracking.md)
- [\[CVPR 2026\] OmniVTG: A Large-Scale Dataset and Training Paradigm for Open-World Video Temporal Grounding](../../CVPR2026/video_understanding/omnivtg_a_large-scale_dataset_and_training_paradigm_for_open-world_video_tempora.md)
- [\[ECCV 2024\] Towards Model-Agnostic Dataset Condensation by Heterogeneous Models](towards_model-agnostic_dataset_condensation_by_heterogeneous_models.md)
- [\[ECCV 2024\] PiTe: Pixel-Temporal Alignment for Large Video-Language Model](pite_pixel-temporal_alignment_for_large_video-language_model.md)

</div>

<!-- RELATED:END -->
