---
title: >-
  [Paper Note] IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants
description: >-
  [NeurIPS 2025][3D Vision][Egocentric Vision] This paper presents IndEgo — the first large-scale multimodal egocentric vision dataset targeting real industrial environments. It comprises 3,460 egocentric video clips (~197 hours) and 1,092 exocentric recordings (~97 hours), spanning five major task categories including assembly/disassembly, logistics, maintenance, woodworking, and miscellaneous tasks, as well as collaborative work scenarios. Three benchmarks are established: mistake detection, reasoning-based QA, and collaborative task understanding.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Egocentric Vision
  - Industrial Scenarios
  - Multimodal Dataset
  - Collaborative Work
  - Mistake Detection
  - Video Question Answering
date: 2026-05-08
content_hash: 6053405c1db818eb
---

# IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants

**Conference**: NeurIPS 2025  
**arXiv**: [2511.19684](https://arxiv.org/abs/2511.19684)  
**Authors**: Vivek Chavan (Fraunhofer IPK / TU Berlin), Yasmina Imgrund, Tung Dao, Sanwantri Bai, Bosong Wang, Ze Lu, Oliver Heimann, Jörg Krüger  
**Code**: [Project Page](https://indego-dataset.github.io/) / [HuggingFace](https://huggingface.co/datasets/FraunhoferIPK/IndEgo)  
**Area**: 3D Vision  
**Keywords**: Egocentric Vision, Industrial Scenarios, Multimodal Dataset, Collaborative Work, Mistake Detection, Video Question Answering  

## TL;DR

This paper presents IndEgo — the first large-scale multimodal egocentric vision dataset targeting real industrial environments. It comprises 3,460 egocentric video clips (~197 hours) and 1,092 exocentric recordings (~97 hours), spanning five major task categories including assembly/disassembly, logistics, maintenance, woodworking, and miscellaneous tasks, as well as collaborative work scenarios. Three benchmarks are established: mistake detection, reasoning-based QA, and collaborative task understanding.

## Background & Motivation

### State of the Field
Egocentric Vision and AI assistants represent a prominent research direction, aiming to develop intelligent systems capable of understanding user actions and intentions while providing contextual guidance. Industrial environments — where workers perform complex manual operations, use diverse tools, and navigate cluttered spaces — pose unique challenges for egocentric vision systems.

### Limitations of Prior Work
- Existing datasets (e.g., EPIC-KITCHENS, Ego4D) are **heavily biased toward daily-life and kitchen scenarios**, with industrial settings severely underrepresented.
- Existing quasi-industrial datasets (Meccano: only 7 hours; Assembly101: only 42 hours) are limited to tabletop tasks and do not cover mobility, material handling, or collaboration demands present in real industrial work.
- **Dyadic collaboration** data is absent — future AI assistants and embodied agents will need to collaborate with humans on shared tasks.
- No dataset provides comprehensive **multimodal sensing** (gaze, hand pose, motion trajectories, 3D point clouds) in an integrated manner.
- Existing datasets consist predominantly of short clips, with **long-horizon tasks** (over 20 minutes) severely lacking.

### Core Problem
To address the gap in egocentric vision datasets for industrial settings by constructing a large-scale benchmark that incorporates real industrial tasks, dyadic collaboration, rich multimodal sensing, and long-duration recordings, while evaluating the performance of state-of-the-art multimodal models in such contexts.

## Method

### Dataset Design and Collection

**Task Categories** (5 types):
1. **Assembly/Disassembly**: Assembly and disassembly of mechanical equipment and PC chassis, with guided and unguided variants.
2. **Logistics & Organization**: Tool transport, item sorting, and warehousing operations.
3. **Maintenance**: Equipment inspection and fault repair, with faults pre-seeded by experimenters.
4. **Woodworking**: Filing, drilling, jointing, ranging from basic to complex operations.
5. **Miscellaneous**: Wearing PPE, first aid, packaging, etc.

**Collaboration Modes** (3 types):
- **Coworking**: Two participants collaborate as equal partners.
- **Supervision**: An expert guides a novice.
- **Teacher-Student**: A teacher demonstrates while a student follows using separate setups.

**Hardware and Multimodal Sensing**: Data are collected using Meta Project Aria devices with sensors including an 8MP RGB camera (2880×2880@10FPS), SLAM cameras, eye tracking, and IMU. Exocentric footage is captured with Sony A6400 cameras (1080p). Derived outputs include gaze estimation, hand pose, semi-dense 3D point clouds, and user trajectories.

**Dataset Scale**:
- 3,460 egocentric clips totaling 197.1 hours and 7.1M RGB frames.
- 1,092 exocentric clips totaling 96.8 hours and 10.5M frames.
- 20 participants (15 male, 5 female) from 10 countries, with experience ranging from novice to expert.
- ~34k fine-grained action annotations with POS tagging of verbs, nouns, and adjectives.
- Annotation agreement: Krippendorff's $\alpha = 0.97$ for key steps; $\alpha = 0.54$ for fine-grained actions.

### Benchmark Task Design

**Mistake Detection**: 1,166 clips covering 25 tasks, containing both correct executions and intentional/unintentional errors. Error types include skipped steps, ordering errors, redundant steps, and safety violations. A severity grading scheme is introduced: Severe (2.3%), Process Failure (18.7%), Impact Future (7%), and Harm (5%).

**Reasoning-based QA**: 3,105 QA pairs spanning four categories: temporal understanding (14%), situational reasoning (28%), visual recognition (32%), and analogical/abductive reasoning (26%).

**Collaborative Task Understanding**: Predicts the individual actions of the wearer and co-worker during collaboration, along with their role relationship.

## Key Experimental Results

### Experiment 1: Mistake Detection Benchmark

| Method | Setting | Precision | Recall | F1 | F1_Severe | F1_ProcessFail | F1_Harm |
|--------|---------|-----------|--------|-----|-----------|----------------|---------|
| QVL2.5 | Zero-shot | 15.9 | 50.1 | 24.1 | 38.8 | 36.5 | 34.1 |
| GFT (Gemini 2.0) | Zero-shot | 35.6 | 48.2 | **40.9** | **51.2** | **42.2** | **48.0** |
| QVL2.5 | MLP fine-tuned | 31.4 | 51.6 | 39.1 | 42.6 | 39.8 | 44.0 |
| VL3 | Transformer fine-tuned | 34.5 | 33.3 | 33.9 | 39.2 | 35.5 | 38.5 |
| QVL2.5 | Early detection (50% frames) | 24.1 | 51.0 | 32.7 | 34.2 | 32.0 | 40.1 |

- In the zero-shot setting, Gemini 2.0 Flash Thinking achieves 40.9% F1, substantially outperforming other VLMs (~24%).
- After fine-tuning, open-source models reach ~39% F1 but still fall short of Gemini's zero-shot performance.
- Early detection using only 50% of frames reduces F1 to ~31–33%, highlighting the importance of temporal information.
- Combining egocentric and exocentric views improves F1 (GFT: 0.43→0.44).

### Experiment 2: Reasoning-based QA Benchmark

| Model | Temporal | Situational | Visual | Analogical | Overall |
|-------|----------|-------------|--------|------------|---------|
| VideoLLaMA3 8B | 52.2 | 60.3 | 59.4 | 57.5 | 58.2 |
| InternVL2.5 | 51.7 | 61.1 | 58.2 | 56.0 | 57.6 |
| Qwen2.5-VL 7B | 53.2 | 60.8 | 59.3 | 56.5 | 58.1 |
| Gemini 2.0 Flash | 55.4 | 62.1 | 67.2 | 68.3 | **64.1** |
| **Human** | **92.6** | **89.6** | **90.4** | **88.6** | **90.0** |

- The best-performing model achieves only 64.1% accuracy, approximately 26 percentage points below the human baseline of 90%.
- Temporal understanding (~52–55%) is the weakest dimension across all models.
- The text-only model Mistral-Large2 with labels reaches 61.4%, suggesting that current VLMs offer limited additional gains from visual inputs.

### Modality Ablation Study

| Model | RGB | +Audio | +Gaze | +Audio+Gaze |
|-------|-----|--------|-------|-------------|
| GFT | 0.38 | 0.41 | 0.39 | **0.42** |
| VL3 | 0.27 | 0.26 | 0.28 | **0.30** |
| IVL2.5 | 0.30 | 0.28 | 0.29 | 0.29 |

The benefits of multimodal fusion are highly context-dependent across tasks, while the full modality combination yields the best overall performance.

## Highlights & Insights

- **First large-scale egocentric dataset for real industrial environments**: Covers 5 major industrial task categories with 197 hours of egocentric and 97 hours of exocentric footage, substantially exceeding prior industrial datasets (Meccano: 7h; Assembly101: 42h).
- **Collaborative work data**: The first systematic collection of multi-perspective egocentric data for dyadic collaboration, encompassing coworking, supervision, and teacher-student modes, providing a foundation for research on AI collaborative assistants.
- **Rich multimodal annotations**: Integrates gaze, hand pose, 3D point clouds, speech, and action annotations with high inter-annotator agreement ($\alpha = 0.97$).
- **Three challenging benchmarks**: Mistake detection (with severity grading), reasoning-based QA, and collaborative task understanding all expose significant shortcomings in state-of-the-art models.
- **Hardware scalability**: The Project Aria-based lightweight wearable acquisition pipeline can be extended to a broader range of industrial settings.

## Limitations & Future Work

- **Single collection site**: All data are collected at Fraunhofer IPK in Berlin, limiting generalizability across factories and industries.
- **Limited participant pool**: 20 participants with a 15:5 gender ratio reflects industry demographics but may introduce bias.
- **English only**: All annotations and narrations are in English, restricting applicability to multilingual scenarios.
- **Maximum clip duration constrained**: Device storage limits individual recordings to ~68 minutes, precluding coverage of longer continuous industrial workflows.
- **Limited baseline models**: Only four VLMs are evaluated; larger-scale models and video-specialized models are not assessed.
- **3D point clouds and hand pose** are provided as auxiliary outputs but are not deeply integrated into benchmark modeling.

## Related Work & Insights

- **EPIC-KITCHENS**: 100 hours in kitchen scenarios, single modality (no gaze/motion/collaboration); IndEgo surpasses it in both scene diversity and modality richness.
- **Ego4D**: The largest-scale dataset (3,670 hours) but centered on everyday life, with no industrial scenarios or collaboration annotations.
- **Ego-Exo4D**: 221 hours with dual perspectives but focused on daily activities and sports; IndEgo specializes in industrial tasks and provides richer mistake detection and QA benchmarks.
- **Assembly101**: 42 hours of quasi-industrial data limited to tabletop assembly, with no collaboration, gaze, or audio.
- **HoloAssist**: 166 hours of assistive scenarios with collaboration and key-step annotations, but lacking industry-specific tasks and error severity grading.
- **Meccano**: Only 7 hours; inferior to IndEgo in both scale and task diversity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first large-scale egocentric dataset for real industrial environments, filling an important gap; methodological innovation is inherently limited for a dataset paper.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three benchmarks with modality and viewpoint ablations provide comprehensive evaluation; the number of baseline models is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with detailed statistics and rich figures.
- **Value**: ⭐⭐⭐⭐⭐ — Significant contribution to industrial AI assistant and embodied intelligence research; data are publicly available.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views](look_and_tell_a_dataset_for_multimodal_grounding_across_egocentric_and_exocentri.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](../../CVPR2026/3d_vision/ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [\[NeurIPS 2025\] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span](gaze_beyond_the_frame_forecasting_egocentric_3d_visual_span.md)
- [\[NeurIPS 2025\] FlareX: A Physics-Informed Dataset for Lens Flare Removal via 2D Synthesis and 3D Rendering](flarex_a_physics-informed_dataset_for_lens_flare_removal_via_2d_synthesis_and_3d.md)
- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](../../ICCV2025/3d_vision/egom2p_egocentric_multimodal_multitask_pretraining.md)

<!-- RELATED:END -->
