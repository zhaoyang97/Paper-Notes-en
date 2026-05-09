---
title: >-
  [Paper Note] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing
description: >-
  [CVPR 2026][Remote Sensing][Geoscience] This work proposes GeoMMBench (1053 expert-level geoscience multiple-choice questions) and GeoMMAgent (a retrieval-perception-reasoning multi-agent framework), systematically evaluating 36 MLLMs in the remote sensing domain and revealing systematic deficiencies in domain knowledge, perceptual grounding, and reasoning capabilities.
tags:
  - CVPR 2026
  - Remote Sensing
  - Geoscience
  - Benchmark
  - Multi-Agent
  - MLLM Evaluation
date: 2026-05-08
content_hash: 0e5c8b42a299e312
---

# GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing

**Conference**: CVPR 2026  
**arXiv**: [2604.08896](https://arxiv.org/abs/2604.08896)  
**Code**: [https://geo-mm-agi.github.io](https://geo-mm-agi.github.io)  
**Area**: Remote Sensing / Multimodal Benchmark  
**Keywords**: Geoscience, Remote Sensing, Benchmark, Multi-Agent, MLLM Evaluation

## TL;DR

This work proposes GeoMMBench (1053 expert-level geoscience multiple-choice questions) and GeoMMAgent (a retrieval-perception-reasoning multi-agent framework), systematically evaluating 36 MLLMs in the remote sensing domain and revealing systematic deficiencies in domain knowledge, perceptual grounding, and reasoning capabilities.

## Background & Motivation

MLLMs have advanced rapidly in general domains, but their evaluation in geoscience and remote sensing remains limited. Existing RS benchmarks are narrow in scope, primarily focusing on perception tasks (classification, detection, segmentation) and typically limited to optical imagery. Geoscience requires multi-sensor data fusion, spatiotemporal reasoning, and cross-disciplinary knowledge integration, which existing benchmarks cannot adequately assess.

GeoMMBench is the first expert-level, knowledge-driven geoscience multimodal benchmark spanning multiple disciplines, sensors, and task hierarchies.

## Method

### Overall Architecture

GeoMMBench contains 1053 image-based multiple-choice questions covering three key dimensions. GeoMMAgent integrates LLMs with domain-specific tools and models within an adaptive multi-agent framework.

### Key Designs

1. **Three-dimensional coverage**: (a) Multi-discipline—remote sensing, photogrammetry, GIS, GNSS, and foundational subjects including mathematics, physics, and geography; (b) Multi-sensor—optical, SAR, multispectral/hyperspectral, LiDAR, DEM, thermal imaging, etc.; (c) Multi-task hierarchy—from theoretical concepts and data preprocessing to perception tasks and advanced geospatial applications.

2. **GeoMMAgent three-stage pipeline**: Retrieval stage (acquiring domain knowledge from external knowledge bases) → Perception stage (leveraging domain-specific RS models to analyze complex data) → Reasoning stage (spatial reasoning and comprehensive analysis). Task decomposition and tool selection enable capabilities beyond those of general-purpose MLLMs.

3. **Expert-level question design**: Scope and content were established by PhD researchers and doctoral students in geoscience through extensive discussion, drawing from authoritative references including educational resources and academic literature. The val set contains 37 questions (for evaluating human expert performance), while the test set contains 1016 questions.

### Loss & Training

GeoMMBench is an evaluation benchmark with no training process. GeoMMAgent relies on the zero-shot capabilities of existing LLMs, augmented through tool invocation.

## Key Experimental Results

### Main Results

| Model Category | Representative Models | GeoMMBench Accuracy | Note |
|---------|---------|----------------|------|
| Open-source MLLM | InternVL2.5, Qwen2.5-VL | Moderate | Insufficient domain knowledge |
| Commercial MLLM | GPT-4o, Gemini | Higher but still gaps | Reasoning advantage |
| GeoMMAgent | LLM + tools | Significant improvement | Tool augmentation effective |

The val set of 37 questions is used for evaluating human expert performance and model selection, while the test set of 1016 questions serves as the final evaluation. Questions were designed by PhD researchers and doctoral students in geoscience through extensive discussion, drawing from educational resources, academic literature, and other authoritative references. GeoMMAgent's three-stage division of labor: the retrieval stage acquires domain knowledge such as spectroscopy and surveying from external knowledge bases; the perception stage leverages RS-specific models to analyze complex data such as SAR and hyperspectral imagery; the reasoning stage performs spatial reasoning and comprehensive analysis.

### Key Findings

- All 36 tested MLLMs exhibit systematic deficiencies in the geoscience domain
- Domain knowledge (e.g., spectroscopy, surveying) is the largest shortcoming
- GeoMMAgent significantly outperforms standalone LLMs through tool augmentation
- Multi-sensor understanding (particularly SAR, hyperspectral) lags far behind optical imagery
- GeoMMBench covers 4 core disciplines (remote sensing, photogrammetry, GIS, GNSS) plus foundational subjects including mathematics, physics, and geography
- Task hierarchy ranges from theoretical concepts and data preprocessing to perception tasks and advanced geospatial applications, providing comprehensive evaluation

## Highlights & Insights

- First expert-level geoscience multimodal benchmark spanning multiple disciplines, sensors, and task hierarchies
- Large-scale evaluation of 36 models establishes a current performance baseline
- GeoMMAgent demonstrates the effectiveness of tool augmentation in bridging the domain gap
- Questions carefully designed by domain experts ensure quality and authoritativeness

## Limitations & Future Work

- The scale of 1053 questions is still limited, and coverage of some sub-domains may be insufficient
- Multiple-choice format may not fully capture open-ended reasoning capabilities
- GeoMMAgent's toolset requires continuous expansion to cover more tasks
- Coverage of temporal analysis tasks such as multi-temporal analysis and change detection remains insufficient
- The significant performance gap between open-source and closed-source MLLMs suggests that domain-specific fine-tuning is still needed in the RS field

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First expert-level geoscience multimodal benchmark
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation of 36 models
- Writing Quality: ⭐⭐⭐⭐ — Well-designed multi-agent framework
- Value: ⭐⭐⭐⭐⭐ — Provides standardized evaluation for RS-AI development

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)
- [\[NeurIPS 2025\] GeoLink: Empowering Remote Sensing Foundation Model with OpenStreetMap Data](../../NeurIPS2025/remote_sensing/geolink_empowering_remote_sensing_foundation_model_with_openstreetmap_data.md)
- [\[ICCV 2025\] SkySense V2: A Unified Foundation Model for Multi-Modal Remote Sensing](../../ICCV2025/remote_sensing/skysense_v2_a_unified_foundation_model_for_multi-modal_remote_sensing.md)
- [\[NeurIPS 2025\] RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [\[ICCV 2025\] RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model](../../ICCV2025/remote_sensing/rs-vheat_heat_conduction_guided_efficient_remote_sensing_foundation_model.md)

</div>

<!-- RELATED:END -->
