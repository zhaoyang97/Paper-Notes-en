---
title: >-
  [Paper Note] SocialGesture: Delving into Multi-Person Gesture Understanding
description: >-
  [CVPR 2025][Human Understanding][Multi-person gesture recognition] SocialGesture is the first large-scale dataset focusing on deictic gestures (pointing/showing/giving/reaching) in multi-person social scenarios, covering 9,889 video clips and 42,533 gesture instances. It establishes three benchmark tasks: temporal localization, classification, and VQA, systematically revealing the severe deficiencies of current models in multi-person gesture understanding.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Multi-person gesture recognition"
  - "social gesture dataset"
  - "deictic gestures"
  - "video understanding"
  - "Visual Question Answering (VQA)"
date: 2026-05-08
content_hash: 7dff5be4eb3f2edf
---

# SocialGesture: Delving into Multi-Person Gesture Understanding

**Conference**: CVPR 2025  
**arXiv**: [2504.02244](https://arxiv.org/abs/2504.02244)  
**Code**: [huggingface.co/datasets/IrohXu/SocialGesture](https://huggingface.co/datasets/IrohXu/SocialGesture)  
**Area**: Human Understanding  
**Keywords**: Multi-person gesture recognition, social gesture dataset, deictic gestures, video understanding, Visual Question Answering (VQA)

## TL;DR

SocialGesture is the first large-scale dataset focusing on deictic gestures (pointing/showing/giving/reaching) in multi-person social scenarios, covering 9,889 video clips and 42,533 gesture instances. It establishes three benchmark tasks: temporal localization, classification, and VQA, systematically revealing the severe deficiencies of current models in multi-person gesture understanding.

## Background & Motivation

**Background**: Gesture recognition is an important branch of human action understanding. Existing datasets (such as Jester, EgoGesture, HaGRID, LD-ConGR, etc.) mainly focus on device control gestures in single-person scenarios or sign language recognition, or are collected in controlled environments, lacking natural gestures in real social interactions.

**Limitations of Prior Work**: (1) Existing datasets are almost exclusively single-person scenarios, failing to capture social communication through gestures between people; (2) Gesture categories lean heavily towards HCI interactions (e.g., waving, thumbs-up), ignoring deictic gestures, which are the core of social communication; (3) The lack of annotation for the relationship between the gesture initiator and the target prevents the study of social gesture semantics; (4) Gestures are not aligned with natural language, limiting the development of VLMs in gesture understanding.

**Key Challenge**: In real-world social communication, gestures and language co-originate in a unified cognitive system. However, current research isolates gestures within a single-person, controlled, HCI-oriented framework, which is severely disconnected from real social scenarios.

**Goal**: (1) Build the first multi-person social gesture dataset; (2) Provide multi-level annotations (gesture types, spatiotemporal localization, interpersonal relationships, VQA); (3) Establish comprehensive baseline experiments to expose the limitations of existing models.

**Key Insight**: Taking inspiration from cognitive science theories on gesture research—where deictic gestures (pointing, showing, giving, reaching) are the most fundamental gesture types for humans to establish joint attention and facilitate social interactions—this work focuses on these four crucial categories of social gestures.

**Core Idea**: Build a large-scale, multi-person, natural-scene dataset of deictic gestures, accompanied by all-around annotations and multi-task benchmarks to advance research in multimodal social understanding.

## Method

### Overall Architecture

The construction pipeline of SocialGesture: (1) Collect videos containing multi-person interactions from YouTube and Ego4D, covering diverse scenarios such as social games (44.51%), variety entertainment (22.31%), Ego4D (21.91%), etc.; (2) Preprocess videos to 720p/30FPS and downsample them to 360p/5FPS for annotation; (3) Annotate temporal segments, keyframes, spatial bounding boxes of initiators and targets, and natural language descriptions for the four categories of deictic gestures; (4) Establish three major benchmark tasks based on these annotations.

### Key Designs

1. **Four Categories of Deictic Gesture Definition and Annotation Taxonomy**:

    - **Function**: To provide a clear, actionable gesture classification standard to support high-quality annotation.
    - **Mechanism**: Based on McNeill's gesture theory, deictic gestures are sub-categorized into four types: **Pointing** (using fingers to direct others' attention to a specific target), **Showing** (displaying objects to others), **Giving** (actions with the intention of transferring objects), and **Reaching** (the intention of reaching out to grab objects). The key distinction for each gesture class lies in the "intention" rather than the "movement morphology". For example, the core of pointing is directing attention rather than the specific way the fingers extend. Annotations contain temporal segments, keyframes, initiator bounding boxes, target bounding boxes (person or object), and social relationship descriptions.
    - **Design Motivation**: Previous gesture datasets defined categories by "movement morphology" (e.g., five open fingers, OK gesture), but the core of social gestures is "intention". Thus, a classification system based on intention is required.

2. **Multi-level Benchmark Task Design**:

    - **Function**: To evaluate model capability in multi-person gesture understanding from different levels of difficulty and perspectives.
    - **Mechanism**: Three major tasks are designed: (a) **Temporal Localization** (Task 1)—localizing the temporal segments of all gestures in long videos and classifying them, evaluated using mAP@IoU; (b) **Gesture Recognition** (Task 2-1 binary classification + Task 2-2 four-way classification)—determining the presence and category of gestures in short video clips; (c) **VQA** (Task 3-1/3-2/3-3)—global perception (scene description, person counting), gesture understanding (detection and classification), gesture localization (spatially localizing the initiator and target), used to evaluate VLMs.
    - **Design Motivation**: Action classification tasks alone are insufficient for a comprehensive evaluation of social gesture understanding. Temporal localization tests detection capability, classification tests recognition, and VQA tests reasoning and multimodal alignment.

3. **Data Diversity and Quality Control**:

    - **Function**: To ensure the dataset covers diverse scenes, populations, and gesture types.
    - **Mechanism**: High-definition videos are selected, containing 2-10 people, 2-30 minutes long, and spanning diverse scenes (race, gender, age). Data sources include various YouTube channel genres (social games, variety shows, education, product reviews, gather dining, cooking) and Ego4D. To address class imbalance (where pointing far outnumbers the other three), resampling is done on the training set.
    - **Design Motivation**: Gestures in the real world are naturally imbalanced and occur in diverse scenarios, necessitating treatments in both data collection and training strategies.

### Loss & Training

Each benchmark task adopts standard training strategies: Temporal localization uses ActionFormer; video recognition fine-tunes various pretrained video models; VQA evaluates various VLMs through zero-shot or fine-tuning. A unified batch size of 16, a learning rate of 5e-4, and standard data augmentation are used.

## Key Experimental Results

### Main Results

Temporal localization (ActionFormer + different feature extractors):

| Feature Extractor | mAP@0.3 | mAP@0.5 | mAP@0.7 | Avg mAP |
|-----------|---------|---------|---------|---------|
| I3D | 24.85 | 9.31 | 0.96 | 10.73 |
| R(2+1)D | 14.38 | 7.23 | 1.77 | 7.29 |
| VideoMAEV2 | 27.23 | 13.33 | 2.76 | 14.73 |

Gesture vs. Non-gesture binary classification:

| Model | Pre-training | Params | Accuracy |
|------|--------|--------|----------|
| SlowFast-R50 | K400 | 35M | 80.82% |
| MViTv2-B | K400 | 51M | 83.29% |
| UniFormerV2-B/16 | CLIP | 115M | **84.43%** |

### Ablation Study

Four-way gesture recognition (Full frame vs. Cropped initiator region):

| Model | Full Frame Top1 | Cropped Region Top1 | Description |
|------|----------|-------------|------|
| TSN-R50 | 54.83% | 55.06% | CNN baseline |
| VideoSwin-L | **56.18%** | 54.94% | Best full frame |
| UniFormerV2-B/16 | 53.37% | **64.72%** | Significant improvement after cropping |

Impact of sliding window stride in temporal localization:

| Stride | Avg mAP |
|--------|---------|
| 16 | 5.94 |
| 8 | 10.73 |
| 4 | 19.19 |

### Key Findings

- All models perform extremely poorly on the four-way classification task (highest is only 56.18%/64.72%), showing that social gesture recognition is highly challenging for existing models.
- Cropping the initiator region improves UniFormerV2 by over 11 percentage points, indicating severe background interference in multi-person scenarios.
- The avg mAP for temporal localization is only 14.73 (VideoMAEV2), far below the performance of these features on THUMOS/ActivityNet—gestures in multi-person scenes are fine-grained and subtle.
- Decreasing the sliding window stride significantly improves localization accuracy (from 5.94 to 19.19), but overall results remain unsatisfactory.
- Feature extractors are pre-trained on data lacking multi-person interactions, leading to poor feature alignment with multi-person gesture tasks.

## Highlights & Insights

- **First multi-person gesture dataset**: Fills a crucial gap in social gesture research. All previous datasets focus on single-person scenarios, while SocialGesture introduces the interpersonal relationship dimension for the first time, which is vital for building AI systems that truly understand social contexts.
- **Intention-based rather than morphology-based classification**: The definitions of the four deictic gestures are based on communicative intentions (directing attention vs. showing vs. transferring vs. acquiring) rather than the physical morphology of fingers/palms. This aligns better with the nature of human social communication and is transferrable to other human action understanding tasks.
- **Elegant multi-level task design**: The progressive benchmark design ranging from detection → recognition → reasoning evaluates both traditional video models and VLMs, fully exposing shortcomings across various dimensions.

## Limitations & Future Work

- The high proportion of pointing gestures leads to severe class imbalance, which may affect the model's ability to learn the other three gesture types.
- The data source is predominantly YouTube, introducing selection bias (mainly entertainment/game scenes) and lacking more everyday social scenarios like workplaces or classrooms.
- It only focuses on deictic gestures, omitting beat gestures, iconic gestures, and metaphoric gestures.
- Some VQA annotations were generated via GPT-4o, which may introduce distribution bias.
- Future work could introduce audio/speech modalities to research joint gesture-speech modeling.

## Related Work & Insights

- **vs HaGRID**: HaGRID focuses on 18 HCI gesture categories with high-resolution single-person hands, while SocialGesture focuses on 4 deictic gestures in multi-person social scenarios. The two are complementary but have completely different scenario assumptions.
- **vs LD-ConGR**: LD-ConGR focuses on long-distance single-person gestures (robustness), while SocialGesture focuses on hand-level multi-person gestures (social semantics), targeting different objectives.
- **vs Ego4D**: Ego4D provides first-person hand-object interactions. SocialGesture reuses some Ego4D data but re-annotates social gesture relationships, representing a shift from an egocentric to a social-centric perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ The first multi-person social gesture dataset, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers localization, recognition, and VQA tasks with plenty of baselines and solid ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation with solid theoretical grounding for the gesture taxonomy.
- Value: ⭐⭐⭐⭐ Highly facilitates the multimodal social understanding field, though technical method novelty is limited (the main contribution is the dataset).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Ego4o: Egocentric Human Motion Capture and Understanding from Multi-Modal Input](ego4o_egocentric_human_motion_capture_and_understanding_from_multi-modal_input.md)
- [\[ICML 2025\] LLaVA-ReID: Selective Multi-Image Questioner for Interactive Person Re-Identification](../../ICML2025/human_understanding/llava-reid_selective_multi-image_questioner_for_interactive_person_re-identifica.md)
- [\[CVPR 2026\] MAMMA: Markerless Accurate Multi-person Motion Acquisition](../../CVPR2026/human_understanding/mamma_markerless_accurate_multi-person_motion_acquisition.md)
- [\[CVPR 2026\] DyaDiT: A Multi-Modal Diffusion Transformer for Socially Favorable Dyadic Gesture Generation](../../CVPR2026/human_understanding/dyadit_a_multi-modal_diffusion_transformer_for_socially_favorable_dyadic_gesture.md)
- [\[AAAI 2026\] Spatiotemporal-Untrammelled Mixture of Experts for Multi-Person Motion Prediction](../../AAAI2026/human_understanding/spatiotemporal-untrammelled_mixture_of_experts_for_multi-person_motion_predictio.md)

</div>

<!-- RELATED:END -->
