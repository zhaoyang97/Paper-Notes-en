---
title: >-
  [Paper Note] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments
description: >-
  [CVPR 2026][Video Understanding][Multimodal dataset] This paper presents OpenMarcie, the largest-scale multimodal action recognition dataset for industrial environments, integrating 8 sensing modalities, 200+ channels…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Multimodal dataset"
  - "human action recognition"
  - "industrial manufacturing"
  - "wearable sensors"
  - "cross-modal alignment"
date: 2026-05-08
content_hash: 0e16236d79be40b1
---

# OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments

**Conference**: CVPR 2026
**arXiv**: [2603.02390](https://arxiv.org/abs/2603.02390)  
**Code**: Available (dataset and code provided on the OpenMarcie official website)  
**Area**: Video Understanding
**Keywords**: Multimodal dataset, human action recognition, industrial manufacturing, wearable sensors, cross-modal alignment

## TL;DR

This paper presents OpenMarcie, the largest-scale multimodal action recognition dataset for industrial environments, integrating 8 sensing modalities, 200+ channels, and 37+ hours of recordings from wearable sensors and visual data. Three benchmarks—HAR classification, open-vocabulary description, and cross-modal alignment—demonstrate the superiority of inertial+vision fusion.

## Background & Motivation

### 1. State of the Field
Smart factories rely on human activity recognition (HAR) to quantify worker performance, improve efficiency, and ensure safety. Video data has long served as the primary source for HAR, but single-modality visual approaches face risks of privacy leakage and technical confidentiality in industrial settings. Several industrial HAR datasets have emerged in recent years (InHARD, LARa, OpenPack, Assembly101, IKEA-ASM, etc.), yet all exhibit notable shortcomings.

### 2. Limitations of Prior Work
Existing industrial HAR datasets suffer from three major limitations:
- **Lack of truly synchronized multimodal data**: Most datasets cover only visual or IMU modalities and lack coordinated collection of wearable sensors, vision, and audio.
- **Overly constrained tasks**: Heavy reliance on protocol-driven, tightly controlled tasks fails to reflect open-ended, procedural workflows found in real industrial environments.
- **Insufficient demographic diversity and task complexity**: Most datasets capture only short, isolated actions without representing the long-duration, multi-step continuous activities characteristic of manufacturing.

### 3. Root Cause
Human actions are inherently multimodal—integrating visual, auditory, tactile, cognitive, and affective cues—yet existing datasets are either unimodal or lack natural variability and realistic industrial noise. Enabling AI systems to genuinely understand human activity in industrial scenes requires a comprehensive dataset spanning diverse sensors, multi-view video, and natural language narrations.

### 4. Paper Goals
The paper aims to construct a unified, large-scale industrial multimodal benchmark that simultaneously supports three tasks—activity classification, open-vocabulary description generation, and cross-modal alignment—filling the gap left by current datasets in terms of modality richness, task diversity, and annotation granularity.

### 5. Starting Point
Two complementary experimental scenarios are designed: bicycle assembly and disassembly (open-ended, ad-hoc improvisation) and 3D printer assembly (procedural, instruction-following). These respectively capture free goal-directed behavior and procedural knowledge acquisition, while sequential collaborative assembly introduces realistic manufacturing handoff dynamics.

### 6. Core Idea
OpenMarcie is the first dataset to simultaneously cover wearable sensors, egocentric/exocentric multi-view video, and overlapping multi-action annotations across fully industrial scenes. With 8 sensing modalities, 282 raw channels, 36 participants, and over 37 hours of data, it provides the most comprehensive multimodal benchmark for industrial HAR.

## Method

### Overall Architecture

OpenMarcie is organized around three major modules: **data collection → annotation → validation benchmarks**.

1. **Data Collection**: Two experimental scenarios (ad-hoc bicycle + procedural 3D printer), each equipped with three ZED X AI stereo cameras for exocentric coverage. Participants wear devices including IMUs, barometers, thermometers, spectrometers, thermal cameras, RGB-LiDAR units, and stereo microphones.
2. **Annotation Pipeline**: A hybrid approach combining manual annotation and LLM-assisted structured label generation.
3. **Validation Benchmarks**: HAR classification, open-vocabulary description, and cross-modal alignment.

### Key Designs

#### Design 1: Dual-Scenario Complementary Collection

- **Function**: Two contrasting scenarios—bicycle assembly (ad-hoc) and 3D printer assembly (procedural).
- **Mechanism**: The bicycle task is familiar to participants, encouraging free decision-making and goal-directed improvisation; the 3D printer task is unfamiliar, requiring interpretation of detailed instructions and procedural knowledge acquisition. Together they cover open-ended maintenance and structured assembly-line procedures.
- **Design Motivation**: Real industrial environments involve both skilled workers improvising and novices following procedures; a single scenario cannot fully represent this range. The 3D printer scenario additionally incorporates **sequential collaborative assembly** (each subsequent participant continues from where the previous one stopped), requiring assessment of prior progress and planning of next steps to simulate real production handoffs.

#### Design 2: Coverage Across 8 Sensing Modalities

- **Function**: Synchronized collection of IMU (wrist, forehead), magnetometer, barometer, temperature sensors, spectrometer, thermal imaging, RGB-LiDAR, stereo audio, and exocentric RGBD cameras, totaling 282 raw channels.
- **Mechanism**: Different modalities carry complementary information—IMU captures motion dynamics, vision captures spatial context, audio captures tool-use sounds, and LiDAR provides depth information.
- **Design Motivation**: No single modality can fully characterize industrial actions (e.g., vision alone cannot distinguish tightening from loosening; IMU alone cannot identify the object being manipulated). Multimodal fusion is essential for improving HAR accuracy, and the multi-sensor design also enables sensor substitution when vision is unavailable.

#### Design 3: Hybrid Annotation Pipeline (Manual + LLM)

- **Function**: Scenario (a) is manually annotated on the best exocentric view using a verb-object-tool scheme with multi-label support (e.g., "walking while carrying"). Scenario (b) is narrated in real time by an external observer, transcribed via Whisper large-v3, and then processed through a two-stage LLM pipeline (DeepSeek-R1 extracts action categories → GPT-4o generates structured hard labels).
- **Mechanism**: Manual annotation ensures precise ground truth; LLM-assisted annotation scales to large scenarios at lower cost. Bidirectional consistency checks (structured → description → structured) validate label quality.
- **Design Motivation**: Manually annotating 37 hours of data is prohibitively costly; LLMs serve as structured translators from natural language narrations to training labels. Validation results show Scenario (a) Macro F1 = 0.715 and Scenario (b) METEOR = 0.531, confirming the reliability of LLM-generated labels.

### Validation Benchmark Methods

- **HAR Classification**: ViT (video) + DeepConvLSTM (IMU) + EnCodec + temporal classifier (audio), each trained independently per modality, then fused via a late-fusion transformer; 12 action classes, with train/test splits by subject.
- **Open-Vocabulary Description**: Modality-specific encoders regress sentence embeddings of narration text (OV-HAR scheme), decoded via Vec2Text embedding retrieval without requiring a large language model.
- **Cross-Modal Alignment**: Inspired by ImageBind, contrastive learning (multimodal InfoNCE loss) aligns video, IMU, audio, and language into a shared embedding space.

## Key Experimental Results

### Main Results

**Table 1: HAR Classification Macro F1 (↑)**

| Modality | Scenario (a) No Null | Scenario (a) Null | Scenario (b) No Null | Scenario (b) Null |
|----------|---------------------|-------------------|---------------------|-------------------|
| Inertial (I) | 0.834 | 0.811 | 0.750 | 0.674 |
| Acoustic (A) | 0.489 | 0.469 | 0.425 | 0.432 |
| Vision (V) | 0.757 | 0.729 | 0.705 | 0.655 |
| I + A | 0.803 | 0.782 | 0.744 | 0.666 |
| A + V | 0.739 | 0.714 | 0.695 | 0.646 |
| **I + V** | **0.882** | **0.851** | **0.773** | **0.685** |
| I + A + V | 0.859 | 0.831 | 0.763 | 0.676 |

**Table 2: Cross-Modal Alignment Recall and Top-1 Accuracy**

| Modality Combination | Scenario (a) R@1 | R@5 | Top-1 | Scenario (b) R@1 | R@5 | Top-1 |
|----------------------|-----------------|-----|-------|-----------------|-----|-------|
| I + T | 0.324 | 0.655 | 0.481 | 0.312 | 0.642 | 0.468 |
| A + T | 0.241 | 0.583 | 0.342 | 0.227 | 0.567 | 0.329 |
| V + T | 0.437 | 0.768 | 0.556 | 0.421 | 0.751 | 0.541 |
| I + A + T | 0.347 | 0.679 | 0.495 | 0.334 | 0.663 | 0.479 |
| A + V + T | 0.412 | 0.740 | 0.533 | 0.395 | 0.723 | 0.517 |
| **I + V + T** | **0.485** | **0.803** | **0.587** | **0.467** | **0.787** | **0.570** |
| I + A + V + T | 0.470 | 0.795 | 0.579 | 0.453 | 0.779 | 0.563 |

### Ablation Study

Cosine similarity results for open-vocabulary description further confirm modality complementarity:
- **I + V is consistently best**: Scenario (a) 0.561, Scenario (b) 0.655, consistently outperforming three-modality fusion (I+A+V = 0.547 / 0.647).
- **Acoustic alone is the weakest**: Only 0.361 in Scenario (a), far below Inertial (0.518) and Vision (0.479).
- **Marginal benefit from adding Acoustic**: I+A (0.512) falls slightly below I alone (0.518), suggesting that audio may introduce noise under the current setup.
- Removing the Null class consistently improves all metrics, indicating that null activity segments are the primary source of classification difficulty.

### Key Findings

1. **Inertial + Vision is the golden combination**: Consistently achieves the best performance across all three tasks—HAR, description, and alignment—demonstrating that motion dynamics and visual spatial information are highly complementary.
2. **Three-modality fusion underperforms two-modality fusion**: I+A+V falls below I+V on most metrics, suggesting that the noisier acoustic modality may dilute effective signals in late fusion.
3. **Ad-hoc scenario consistently outperforms procedural scenario**: HAR F1 for bicycle assembly (0.882) is substantially higher than for 3D printer assembly (0.773), as the latter involves less familiar small-component operations and greater cognitive demands.
4. **Audio modality shows limited but non-negligible contribution**: Its weak standalone performance is primarily attributable to data collection in a lab rather than a real factory, lacking authentic industrial noise (e.g., machine vibration); it still provides marginal gains in fusion.

## Highlights & Insights

- **Broadest scale and coverage**: 8 modalities, 282 channels, 37+ hours, 36 participants—the largest known industrial multimodal HAR dataset.
- **High ecological validity**: Sequential collaborative assembly design (each participant continues where the previous one stopped) authentically reflects production handoff scenarios.
- **Innovative annotation methodology**: Manual annotation combined with a two-stage LLM pipeline and bidirectional consistency checking balances annotation cost and quality.
- **Multi-label action support**: The verb-object-tool scheme uniquely permits overlapping annotations (e.g., walking while carrying), more closely reflecting real industrial activity.
- **Three complementary benchmarks**: The combination of HAR + description + alignment comprehensively evaluates the dataset's value across multiple dimensions.

## Limitations & Future Work

1. **Limited participant diversity**: Participants are predominantly right-handed engineers (72% engineers, 86% right-handed), limiting demographic generalizability.
2. **Weak audio modality performance**: The laboratory environment lacks authentic industrial noise, and the effectiveness of audio signals in real factory conditions remains to be validated.
3. **Incomplete annotation coverage**: Current annotations exploit only a portion of the dataset's potential; multi-view recordings could support richer annotations for objects, interactions, and pose.
4. **Sensor configurations are not fully consistent across scenarios**: Wearable device placement differs slightly between the two scenarios; although key modalities (wrist IMU, chest LiDAR, stereo microphone) are consistent, cross-scenario comparison is still complicated.
5. **Baseline methods are relatively straightforward**: HAR relies on ViT + DeepConvLSTM late fusion, without exploring more advanced early fusion or attention-based fusion strategies.

## Related Work & Insights

- **Complementary to Ego-Exo4D**: Ego-Exo4D is larger in scale (1,200 h) but industrial data accounts for only approximately 6%; OpenMarcie provides 100% industrial coverage and includes wearable sensors.
- **Extension of OpenPack**: OpenPack focuses on logistics scenarios with 50+ hours of IMU/IoT data but lacks visual and egocentric modalities; OpenMarcie supplements these with vision and egocentric perspectives.
- **Practical validation of ImageBind**: Transfers the multimodal alignment concept of ImageBind from internet-scale data to structured industrial sensor data.
- **Implications for future research**: Early fusion strategies could be explored to better leverage audio signals; object detection could be enhanced using 3D printer STL part models; the sequential collaborative design could be extended into a human-robot collaboration dataset.

## Rating

⭐⭐⭐⭐ A high-quality contribution of an industrial multimodal dataset with the most comprehensive modality coverage and scenario design in its domain, along with a systematic set of three validation benchmarks. The primary weaknesses are the unvalidated practical utility of the audio modality and the relatively basic baseline methods; as a dataset paper, the overall contribution is outstanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[NeurIPS 2025\] SmartWilds: Multimodal Wildlife Monitoring Dataset](../../NeurIPS2025/video_understanding/smartwilds_multimodal_wildlife_monitoring_dataset.md)
- [\[AAAI 2026\] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation](../../AAAI2026/video_understanding/emovid_a_multimodal_emotion_video_dataset_for_emotion-centric_video_understandin.md)
- [\[CVPR 2026\] Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection](decompose_and_transfer_cot-prompting_enhanced_alignment_for_open-vocabulary_temp.md)
- [\[CVPR 2026\] EgoXtreme: A Dataset for Robust Object Pose Estimation in Egocentric Views under Extreme Conditions](egoxtreme_a_dataset_for_robust_object_pose_estimation_in_egocentric_views_under_.md)

</div>

<!-- RELATED:END -->
