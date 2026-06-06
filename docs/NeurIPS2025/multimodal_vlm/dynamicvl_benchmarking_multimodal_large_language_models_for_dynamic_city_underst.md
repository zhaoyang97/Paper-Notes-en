---
title: >-
  [Paper Note] DynamicVL: Benchmarking MLLMs for Dynamic City Understanding
description: >-
  [NeurIPS 2025][Multimodal VLM][Remote sensing imagery] This paper proposes DVL-Suite, a framework comprising the DVL-Bench evaluation benchmark and the DVL-Instruct instruction-tuning dataset…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Remote sensing imagery"
  - "urban dynamic understanding"
  - "multi-temporal analysis"
  - "vision-language benchmark"
  - "change detection"
date: 2026-05-08
content_hash: c0e5432f6374989d
---

# DynamicVL: Benchmarking MLLMs for Dynamic City Understanding

**Conference**: NeurIPS 2025  
**arXiv**: [2505.21076](https://arxiv.org/abs/2505.21076)  
**Code**: [GitHub](https://github.com/weihao1115/dynamicvl)  
**Area**: Multimodal VLM  
**Keywords**: Remote sensing imagery, urban dynamic understanding, multi-temporal analysis, vision-language benchmark, change detection

## TL;DR

This paper proposes DVL-Suite, a framework comprising the DVL-Bench evaluation benchmark and the DVL-Instruct instruction-tuning dataset, covering 42 U.S. cities and 14,871 high-resolution multi-temporal remote sensing images. It systematically evaluates 18 MLLMs on long-term urban dynamic understanding and introduces DVLChat as a baseline model.

## Background & Motivation

Remote sensing enables urban monitoring via satellite imagery, yet most existing studies are limited to bi-temporal comparisons and lack vision-language datasets spanning longer temporal ranges. Although MLLMs perform well on general visual understanding tasks, two key bottlenecks remain in multi-temporal remote sensing analysis: (1) the absence of temporally aligned visual-language datasets covering long time series, and (2) existing multi-temporal remote sensing MLLMs being tested only on high-level semantic understanding without pixel-level precise quantitative analysis capabilities.

Existing datasets (e.g., CDVQA, TEOChatlas, EarthDial) are either limited to bi-temporal settings, task-restricted, or low-resolution (224–512 pixels). DVL-Suite addresses these gaps by providing 1024×1024 high-resolution imagery, averaging 6.73–6.94 temporal frames per scene (2005–2023), and covering six task categories spanning pixel-level to scene-level analysis.

## Method

### Overall Architecture

DVL-Suite consists of two components:

1. **DVL-Bench**: An evaluation benchmark containing 3,469 multi-temporal image sequences with 1,391 referring segmentation instructions, 5,854 QA pairs, and 1,437 comprehensive descriptions.
2. **DVL-Instruct**: An instruction-tuning dataset with 63,771 text pairs and 11,402 multi-temporal images for training DVLChat.

Data are sourced from NAIP (National Agriculture Imagery Program) with a GSD of 1.0 m, covering 42 major U.S. cities.

### Key Designs

#### Six-Task Taxonomy

The paper defines a hierarchical task framework covering urban dynamic understanding from fine-grained to global levels:

- **BCA (Basic Change Analysis)**: Identifies and compares multi-temporal land-use changes, covering 20 change events across 5 land cover types (vegetation, non-vegetation, water, buildings, playgrounds).
- **CSE (Change Speed Estimation)**: Tracks and quantifies temporal trends of urban elements (e.g., building expansion rate, vegetation loss).
- **EA (Environmental Assessment)**: Evaluates urban livability and economic indicators through visual analysis.
- **RCD (Referring Change Detection)**: Performs dense reasoning and precise spatial localization of changed regions, requiring pixel-level segmentation.
- **RCC (Region Change Captioning)**: Generates detailed change descriptions for user-specified geographic regions.
- **DTC (Dense Temporal Captioning)**: Generates comprehensive reports documenting long-term temporal changes.

#### Data Annotation Pipeline

A semi-automatic annotation workflow is adopted:

1. Urban experts perform base annotation (semantic change region segmentation, key frame identification).
2. GPT-4.1 integrates expert annotations to generate diverse instructions.
3. Three-round quality control: self-checking, cross-checking, and supervised review.
4. BCA/CSE: Correct answers computed from segmentation masks; distractors generated at ±20% and ±40%.
5. RCD: Domain experts design event-specific prompts with manual mask annotation.
6. DTC/RCC: Annotators identify key frames → write stage-wise descriptions → GPT-4.1 refinement.

#### DVLChat Model Design

Built upon the LISA architecture with two key modifications:

1. **Dual LoRA Routing Mechanism**: Prefix tokens route requests — `[QA]` activates the VQA LoRA and `[SE]` activates the change detection LoRA, preventing inter-task interference.
2. **Multi-temporal Image Interleaving**: Image features from multiple temporal frames are interleaved prior to decoding to enable cross-temporal analysis.
3. **Segmentation Capability**: `<SEG>` token embeddings are decoded through SAM's frozen visual backbone and unfrozen decoder to generate precise segmentation masks.

The underlying MLLM is Qwen2.5-VL, though the architecture is MLLM-agnostic.

### Loss & Training

- Two independent LoRA modules are trained separately for VQA and segmentation tasks.
- The QA branch uses instruction–ground-truth pairs from DVL-Instruct.
- The segmentation branch uses mask annotations from the RCD task.
- Training is conducted on 8 H100 GPUs.

## Key Experimental Results

### Main Results

**Table 1: QA Task Results (Accuracy %)**

| Model | AVG | BCA-Single | BCA-Multi | CSE-Single | CSE-Multi | EA |
|-------|-----|-----------|----------|-----------|----------|----|
| o4-mini | 34.1 | 62.8 | 36.1 | 33.8 | 12.4 | 25.3 |
| GPT-4.1 | 32.5 | 66.1 | 39.7 | 31.3 | 5.4 | 20.2 |
| Qwen2.5-VL 32B | 31.4 | 62.0 | 33.3 | 36.9 | 3.2 | 21.6 |
| DVLChat 7B | **33.3** | 64.9 | 21.3 | 31.3 | **18.6** | **30.6** |
| TEOChat | 17.2 | 35.1 | 8.7 | 17.0 | 10.8 | 14.6 |

**Table 2: Captioning Task Results (0–5 scale)**

| Model | RCC-AVG | DTC-AVG |
|-------|---------|---------|
| o4-mini | 4.58 | 4.14 |
| GPT-4.1 | 4.46 | 3.98 |
| DVLChat 7B | 3.98 | 3.40 |
| InternVL3 78B | 3.92 | 3.33 |
| TEOChat | 1.66 | 1.45 |

### Ablation Study

- **Referring Change Detection**: The specialized model ChangeMamba achieves 32.41% IoU; DVLChat achieves 29.06% (a gap of only 3.35%), outperforming LISA (13.85%) and PSALM (26.93%).
- **Non-monotonic Model Scaling**: The Qwen2.5-VL series peaks at 31.4% with 32B parameters and drops to 29.7% at 72B; InternVL3 similarly declines after peaking at 14B — indicating that simply increasing parameter count is insufficient to improve precise change detection.

### Key Findings

1. The strongest commercial model, o4-mini, achieves only 34.1% overall QA accuracy, exposing severe deficiencies in long-sequence understanding and quantitative analysis among current MLLMs.
2. CSE multi-choice accuracy peaks at only 13.6%, and the Change Rate Precision (CRP) consistently remains below 1.21, indicating models cannot capture fine-grained temporal changes.
3. DVLChat at 7B surpasses general-purpose 72B–78B models on multiple tasks through domain-specific training data, demonstrating that domain data matters more than model scale.
4. A significant gap exists between open-source and commercial models on captioning tasks (approximately 1-point difference in average DTC scores).

## Highlights & Insights

- The first long-sequence remote sensing VL benchmark spanning pixel-level to scene-level tasks, filling the gap in multi-temporal analysis.
- The dual LoRA routing design elegantly integrates QA and segmentation capabilities within a single model without mutual interference.
- The non-monotonic scaling phenomenon reveals a profound insight: improving general capability and domain-specific precise analysis capability requires fundamentally different strategies.
- The semi-automatic annotation pipeline (domain experts + GPT-4.1) achieves a good balance between quality and efficiency.

## Limitations & Future Work

- NAIP imagery contains near-infrared band information that current MLLMs cannot effectively exploit.
- DVLChat does not yet leverage pixel-level segmentation data to enhance cross-task numerical quantification capabilities.
- DVLChat still lags behind commercial models in overall performance, requiring dedicated algorithms and larger-scale parameters.
- Coverage is limited to U.S. cities, lacking global geographic diversity.

## Related Work & Insights

- Compared to existing multi-temporal remote sensing VL datasets such as TEOChat and EarthDial, DVL-Suite spans significantly longer temporal sequences (average 6.94 frames vs. 2.07 frames) at higher resolution (1024 vs. 224–512 pixels).
- The dual LoRA routing mechanism is generalizable to other multi-task scenarios requiring the integration of understanding and segmentation.
- The non-monotonic scaling phenomenon has implications for scaling law research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic long-sequence remote sensing VL benchmark with a comprehensive task taxonomy.
- **Technical Depth**: ⭐⭐⭐ — The DVLChat architecture is straightforward but practical; the core contributions lie in the data and benchmark design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluates 18 models with thorough multi-dimensional analysis.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to urban planning, disaster assessment, and related domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] in the eye of mllm benchmarking egocentric video intent understanding with gaze-](in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[NeurIPS 2025\] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video](rtv_bench_benchmarking_mllm_continuous_perception_through_realtime_video.md)
- [\[ACL 2026\] CNSL-bench: Benchmarking the Sign Language Understanding Capabilities of MLLMs on Chinese National Sign Language](../../ACL2026/multimodal_vlm/cnsl-bench_benchmarking_the_sign_language_understanding_capabilities_of_mllms_on.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](../../ICCV2025/multimodal_vlm/spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[NeurIPS 2025\] rtv-bench benchmarking mllm continuous perception understanding and reasoning th](rtv-bench_benchmarking_mllm_continuous_perception_understanding_and_reasoning_th.md)

</div>

<!-- RELATED:END -->
