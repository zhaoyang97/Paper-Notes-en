---
title: >-
  [Paper Note] EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark
description: >-
  [ICLR 2026][3D Vision] This paper introduces EgoNight, the first systematic nighttime egocentric vision benchmark, comprising day-night aligned videos and 3,658 manually verified QA pairs. It reveals that MLLMs suffer up to 32.8% performance degradation under low-light conditions.
tags:
  - ICLR 2026
  - 3D Vision
date: 2026-05-08
content_hash: 0f60842c0a6f6ae2
---

# EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark

**Conference**: ICLR 2026
**arXiv**: [2510.06218](https://arxiv.org/abs/2510.06218)
**Code**: [https://github.com/dehezhang2/EgoNight](https://github.com/dehezhang2/EgoNight)
**Area**: 3D Vision / Egocentric Vision

## TL;DR

This paper introduces EgoNight, the first systematic nighttime egocentric vision benchmark, comprising day-night aligned videos and 3,658 manually verified QA pairs. It reveals that MLLMs suffer up to 32.8% performance degradation under low-light conditions.

## Background & Motivation

### State of the Field

Egocentric vision understanding has advanced substantially in recent years, driven by large-scale datasets such as EPIC-KITCHENS, Ego4D, and Ego-Exo4D, which have facilitated progress in action recognition, object detection, and video question answering. MLLMs (e.g., GPT-4V, Gemini, Qwen-VL) have demonstrated strong video understanding capabilities, and egocentric-specific MLLMs (e.g., EgoGPT, Exo2Ego) have also emerged.

### Limitations of Prior Work

Nearly all existing egocentric vision datasets and benchmarks are confined to daytime or well-lit scenarios, overlooking nighttime low-light conditions that are unavoidable in real-world applications. Consequently, the robustness of current models under nighttime conditions remains entirely unknown, severely limiting the practical deployment of intelligent assistants and navigation systems in such settings.

### Root Cause

Egocentric vision systems in practice (e.g., intelligent navigation assistants) must operate at night, facing challenges such as low illumination, uneven lighting, and severely restricted visibility. However, the absence of appropriate nighttime benchmarks prevents researchers from both evaluating model performance at night and making targeted improvements. Furthermore, nighttime annotation is itself extremely difficult due to low visibility, making it hard to ensure annotation quality.

### Paper Goals

The paper proposes EgoNight, the first systematic nighttime egocentric vision benchmark. The core innovation lies in introducing day-night aligned videos: precisely aligned day-night video pairs are synthesized using Blender (EgoNight-Synthetic), real-world day-night aligned videos are collected via a video-guided recording strategy (EgoNight-Sofia), and existing nighttime data are integrated (EgoNight-Oxford). On this basis, the EgoNight-VQA benchmark and two auxiliary tasks are constructed.

---

## Method

### Overall Architecture

EgoNight consists of three components: (1) video source collection — synthetic (50 pairs), real-world recordings (20 pairs), and existing data (20 clips); (2) the EgoNight-VQA benchmark — 12 QA types and 3,658 QA pairs generated through a three-stage daytime-enhanced automatic annotation pipeline; and (3) two auxiliary benchmarks — day-night correspondence retrieval and nighttime depth estimation.

### Key Design 1: Day-Night Aligned Video Collection Strategy

**Synthetic Alignment (EgoNight-Synthetic)**: Diverse indoor 3D scenes are generated using Infinigen; human annotators clean the scenes and simulate walking trajectories; daytime and nighttime versions are rendered under identical trajectories via Blender, ensuring pixel-level precise alignment. The subset comprises 50 video pairs covering 100+ environment assets and 50+ object categories.

**Real-World Alignment (EgoNight-Sofia)**: A video-guided recording strategy is designed — daytime videos are first recorded, and during nighttime recording the daytime video is played on a phone as a visual guide, helping the wearer match pace, viewpoint, and actions. Post-hoc trimming further optimizes spatiotemporal consistency. The subset comprises 20 video pairs covering diverse scenes including apartments, offices, supermarkets, and streets.

### Key Design 2: Daytime-Enhanced Automatic QA Generation Pipeline

A three-stage pipeline leverages daytime videos to assist nighttime annotation:

1. **Nighttime Description Generation**: GPT-4.1 is prompted to generate detailed descriptions for nighttime clips targeting specific QA types.
2. **Nighttime Question Generation**: Descriptions and nighttime clips are fed into an MLLM to generate diverse question candidates.
3. **Daytime-Enhanced Pseudo-Answer Synthesis**: For paired types, daytime clips are used to generate more accurate answers; for non-paired types, answers are inferred directly from nighttime clips.

All generated QA pairs are refined through three rounds of operations (deletion/modification/addition) by human annotators, with each QA pair undergoing at least one manual verification, accumulating 300+ hours of human effort in total.

### Key Design 3: Diverse QA Type Taxonomy

Twelve QA types are defined, divided into:

- **Paired Types** (8): object recognition, text recognition, spatial reasoning, scene sequence, navigation, static counting, action recognition, and uncommon-sense reasoning.
- **Non-Paired Types** (4): illumination recognition, illumination change, dynamic detection, and dynamic counting.

Among these, navigation, scene sequence, illumination recognition/change, and uncommon-sense reasoning are newly proposed task types in this work.

---

## Key Experimental Results

### Main Results

Ten state-of-the-art MLLMs are evaluated on EgoNight-VQA:

| Model | Synthetic (Night) | Sofia (Night) | Oxford (Night) | Avg. Accuracy |
|---|---|---|---|---|
| GPT-4.1 | 30.73% | 26.33% | 35.72% | **30.93%** |
| Gemini 2.5 Pro | 27.18% | 25.00% | 33.21% | 28.46% |
| InternVL3-8B | 19.28% | 17.10% | 23.80% | **20.06%** |
| Qwen2.5-VL-72B | 18.56% | 16.73% | 22.41% | 19.23% |
| Qwen2.5-VL-7B | 14.58% | 13.28% | 16.71% | 14.86% |
| EgoGPT | 12.88% | 14.03% | 15.95% | **14.29%** |

Day-to-night performance gap: an average drop of **32.8%** on EgoNight-Synthetic and **25.0%** on EgoNight-Sofia.

### Ablation Study / In-Depth Analysis

| Fine-tuning Strategy | Synthetic Acc. | Real Acc. | Gain |
|---|---|---|---|
| Zero-shot (baseline) | 14.83% | — | — |
| Full model fine-tuning | **24.67%** | **21.88%** | +9.84% |
| Visual encoder only | 19.23% | 18.56% | +4.40% |
| LLM only | 21.15% | 19.02% | +6.32% |
| Synthetic train → Real test | — | 20.57% | +5.74% |

**Key Findings**:

1. Synthetic and real data are highly correlated (Pearson $r = 0.9359$, $p = 6.847 \times 10^{-5}$), indicating that fine-tuning on synthetic data effectively improves real-scene performance.
2. Perception-oriented tasks perform better in daytime but suffer larger degradation at night; reasoning-oriented tasks are overall harder but are relatively less affected by illumination.
3. Newly proposed QA types (illumination recognition, navigation, uncommon-sense reasoning) pose substantial challenges to existing MLLMs.
4. In auxiliary tasks, GPT-4.1 achieves 80%+ accuracy on spatial retrieval but performs poorly on temporal localization; fisheye-specific depth estimation models outperform general-purpose ones.

---

## Highlights & Insights

- The work fills a critical gap in nighttime egocentric vision understanding; the day-night aligned design is elegant and enables quantitative analysis of the performance gap.
- The QA type taxonomy is comprehensive, introducing multiple previously unexplored task dimensions such as navigation and illumination recognition.
- The daytime-enhanced annotation pipeline cleverly exploits daytime information to assist nighttime annotation, balancing both efficiency and quality.
- The high correlation between synthetic and real data ($r = 0.9359$) validates the research value of synthetic data.

## Limitations & Future Work

- The dataset scale is relatively small (90 videos, 3,658 QA pairs) compared to large-scale benchmarks.
- The proportion of synthetic data is high (approximately 55%), which may not fully reflect the complexity of real-world environments.
- Only VQA and two auxiliary tasks are evaluated; a broader range of nighttime egocentric tasks remains unexplored.
- The daytime-enhanced annotation strategy relies on GPT-4.1, and generation quality is bounded by that model's capabilities.

## Related Work & Insights

- **vs. Ego4D/EPIC-KITCHENS**: These large-scale egocentric datasets focus exclusively on daytime scenes; EgoNight is the first benchmark dedicated to nighttime conditions.
- **vs. NightBench**: NightBench addresses general nighttime image understanding, whereas EgoNight focuses on the egocentric perspective with day-night alignment.
- **vs. Low-Light Enhancement Methods**: Traditional methods focus on pixel-level enhancement, while this paper targets the gap in semantic-level understanding.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic nighttime egocentric vision benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 10 MLLMs with fine-tuning analysis and auxiliary tasks
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-presented data
- Value: ⭐⭐⭐⭐ Fills an important research gap with high practical significance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Egocentric Action-aware Inertial Localization in Point Clouds with Vision-Language Guidance](../../ICCV2025/3d_vision/egocentric_action-aware_inertial_localization_in_point_clouds_with_vision-langua.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](../../CVPR2026/3d_vision/ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [\[CVPR 2026\] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models](../../CVPR2026/3d_vision/ava_bench_atomic_visual_ability_vfm.md)
- [\[ICLR 2026\] EgoWorld: Translating Exocentric View to Egocentric View using Rich Exocentric Observations](egoworld_translating_exocentric_view_to_egocentric_view_using_rich_exocentric_ob.md)

</div>

<!-- RELATED:END -->
