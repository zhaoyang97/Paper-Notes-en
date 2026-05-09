---
title: >-
  [Paper Note] SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence
description: >-
  [CVPR 2026][Multimodal VLM][Spatial Intelligence] This paper introduces SpatialScore, currently the most comprehensive multimodal spatial intelligence benchmark (5K samples / 30 tasks), and proposes two complementary approaches to enhance spatial understanding in MLLMs: a data-driven fine-tuning scheme via SpatialCorpus (331K QA pairs) and a training-free SpatialAgent system equipped with 12 specialized tools.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Spatial Intelligence
  - Multimodal Evaluation
  - Spatial Reasoning
  - Agent Systems
  - Spatial Corpus
date: 2026-05-08
content_hash: 52e5beba7d04c378
---

# SpatialScore: Towards Comprehensive Evaluation for Spatial Intelligence

**Conference**: CVPR 2026
**arXiv**: [2505.17012](https://arxiv.org/abs/2505.17012)
**Code**: [https://github.com/haoningwu3639/SpatialScore/](https://github.com/haoningwu3639/SpatialScore/)
**Area**: Multimodal VLM
**Keywords**: Spatial Intelligence, Multimodal Evaluation, Spatial Reasoning, Agent Systems, Spatial Corpus

## TL;DR
This paper introduces SpatialScore, currently the most comprehensive multimodal spatial intelligence benchmark (5K samples / 30 tasks), and proposes two complementary approaches to enhance spatial understanding in MLLMs: a data-driven fine-tuning scheme via SpatialCorpus (331K QA pairs) and a training-free SpatialAgent system equipped with 12 specialized tools.

## Background & Motivation
1. **Background**: Multimodal large language models (MLLMs) have demonstrated strong performance on semantic QA and mathematical reasoning, yet evaluation of spatial intelligence remains fragmented and limited in scope.
2. **Limitations of Prior Work**: Existing spatial benchmarks suffer from two major deficiencies: (i) tasks are overly simplistic, focusing primarily on coarse-grained spatial relations (e.g., object existence/location) while neglecting rigorous visual-geometric perception (e.g., camera pose estimation, dynamic scene understanding); (ii) evaluation coverage is narrow, relying on simple true/false questions, unimodal inputs, or single-skill assessments that fail to comprehensively measure spatial intelligence.
3. **Key Challenge**: Classical computer vision has established mature geometric optimization tools and mathematical foundations, yet these advances remain confined to the purely visual paradigm and lack tight integration with language as well as a unified evaluation protocol.
4. **Goal**: (i) Construct the most comprehensive spatial intelligence benchmark to date; (ii) Conduct extensive evaluation of 49 representative MLLMs; (iii) Improve spatial reasoning capability through both data-driven and agent-based approaches.
5. **Key Insight**: The fusion of semantic understanding and spatial perception is identified as the next frontier, motivating a systematic investigation of the extent to which existing MLLMs possess spatial intelligence.
6. **Core Idea**: A comprehensive benchmark spanning 30 tasks is proposed, accompanied by large-scale training corpora and a multi-tool agent system, advancing spatial intelligence from both evaluation and enhancement perspectives.

## Method

### Overall Architecture
The paper comprises three contributions: (1) **SpatialScore** benchmark — 5,025 manually verified samples covering diverse data types (real/simulated/AIGC), input modalities (image/video), and QA formats (true/false/multiple-choice/open-ended); (2) **SpatialCorpus** — a 331K multimodal QA training resource; (3) **SpatialAgent** — a multi-agent system equipped with 12 specialized spatial perception tools.

### Key Designs

1. **SpatialScore Benchmark Construction**:
    - **Function**: Provides comprehensive and diverse evaluation of spatial intelligence.
    - **Mechanism**: 500 scenes are randomly sampled from existing 3D datasets (ScanNet++, Omni3D, etc.), and precise 3D annotations are used to generate open-ended QA pairs, which are subsequently paraphrased via LLMs to increase linguistic diversity. Spatially relevant samples from 23 existing datasets are further integrated. After GPT-based filtering and manual screening by five volunteers, 5,025 high-quality samples are retained, covering 30 specific tasks across 10 broad categories.
    - **Design Motivation**: Existing benchmarks are either overly simplistic or too narrow in scope to comprehensively measure spatial intelligence. By combining newly constructed data with existing datasets, this benchmark achieves a balance between task diversity and evaluation coverage.

2. **SpatialCorpus Training Resource**:
    - **Function**: Provides large-scale spatial reasoning fine-tuning data for MLLMs.
    - **Mechanism**: A 2D simulator and existing 3D annotations (ScanNet++, WildRGB-D, Omni3D, PointOdyssey, etc.) are leveraged to construct 331K multimodal spatial QA samples. The corpus supports supervised fine-tuning of models such as Qwen3-VL, yielding significant performance improvements on spatial reasoning tasks.
    - **Design Motivation**: Evaluation alone cannot improve model capability; training data of sufficient scale and quality is required to close the gap between models and humans in spatial understanding.

3. **SpatialAgent Multi-Agent System**:
    - **Function**: Enhances spatial reasoning in existing MLLMs in a training-free manner.
    - **Mechanism**: Twelve specialized spatial perception tools (depth estimator, camera pose estimator, motion estimator, etc.) are orchestrated under two reasoning paradigms — **Plan-Execute** (hierarchical subtask decomposition with sequential tool invocation) and **ReAct** (interleaved reasoning-action with iterative tool interaction). Dynamic tool orchestration improves spatial understanding without any additional training.
    - **Design Motivation**: Data-driven approaches incur additional training costs, whereas the agent scheme offers a plug-and-play lightweight alternative; the two approaches are complementary.

### Loss & Training
SpatialCorpus fine-tuning follows the standard supervised fine-tuning paradigm. SpatialAgent is a training-free framework and therefore involves no additional loss function design.

## Key Experimental Results

### Main Results

| Model | Overall | Mental Anim. | Counting | Depth Est. | Obj-Dist | Camera |
|-------|---------|-------------|----------|------------|----------|--------|
| Human | 86.60 | 96.87 | 89.72 | 82.33 | 78.96 | 86.89 |
| GPT-5 (Text-only) | 30.62 | 18.79 | 20.34 | 29.36 | 24.20 | 32.01 |
| Qwen3-VL-2B | 41.41 | 35.35 | 52.74 | 34.64 | 35.42 | 30.59 |
| InternVL3-1B | 33.03 | 26.85 | 47.69 | 24.74 | 24.02 | 25.71 |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Qwen3-VL + SpatialCorpus | Significant overall improvement | Data-driven fine-tuning is effective |
| SpatialAgent (Plan-Execute) | Substantially outperforms base model | Training-free paradigm is viable |
| SpatialAgent (ReAct) | Complementary to Plan-Execute | Better suited for tasks requiring iterative reasoning |

### Key Findings
- Even the strongest existing models fall far short of human-level performance on SpatialScore (86.60 vs. ~50+), indicating that spatial intelligence remains a formidable challenge.
- Text-only GPT-5 performs near chance level (30.62 vs. 28.29), confirming the necessity of visual information for spatial reasoning.
- Both SpatialCorpus fine-tuning and SpatialAgent yield significant performance gains and are mutually complementary.
- Camera pose and motion estimation tasks are the most challenging, with the largest gap between model and human performance.

## Highlights & Insights
- **Unprecedented evaluation scale**: A systematic evaluation spanning 30 tasks, 5,025 samples, and 49 models establishes a solid foundation for spatial intelligence research.
- **Dual-path enhancement strategy**: Data-driven and training-free agent approaches are complementary and can be flexibly selected based on deployment requirements.
- **Transferable 3D data repurposing pipeline**: The pipeline for converting 3D annotations into QA format is generalizable to other domains.

## Limitations & Future Work
- The benchmark remains primarily focused on static evaluation, lacking assessment of interactive spatial reasoning.
- The agent system depends on the accuracy of external tools, and tool failures may have cascading effects on downstream results.
- Future work could extend evaluation to embodied AI and autonomous navigation in real-world settings.

## Related Work & Insights
- **vs. VSI-Bench/STI-Bench**: These benchmarks cover only a limited number of tasks and formats; SpatialScore substantially surpasses them in both scale and diversity.
- **vs. OmniSpatial**: Although OmniSpatial covers more tasks (50), its sample size is considerably smaller (1,533); SpatialScore is superior in terms of quality and task balance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Systematic integration with newly constructed evaluation benchmark represents a major contribution, though core technical innovation is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluation of 49 models with human baselines and validation of multiple enhancement approaches.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure with comprehensive and detailed data presentation.
- **Value**: ⭐⭐⭐⭐⭐ An important infrastructure contribution to the spatial intelligence research community.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[ICLR 2026\] On the Generalization Capacities of MLLMs for Spatial Intelligence](../../ICLR2026/multimodal_vlm/on_the_generalization_capacities_of_mllms_for_spatial_intelligence.md)
- [\[ICLR 2026\] OmniSpatial: Towards Comprehensive Spatial Reasoning Benchmark for Vision Language Models](../../ICLR2026/multimodal_vlm/omnispatial_towards_comprehensive_spatial_reasoning_benchmark_for_vision_languag.md)
- [\[CVPR 2026\] Medic-AD: Towards Medical Vision-Language Model's Clinical Intelligence](medic-ad_towards_medical_vision-language_models_clinical_intelligence.md)
- [\[CVPR 2026\] Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy](nano-emox_unifying_multimodal_emotional_intelligence_from_perception_to_empathy.md)

<!-- RELATED:END -->
