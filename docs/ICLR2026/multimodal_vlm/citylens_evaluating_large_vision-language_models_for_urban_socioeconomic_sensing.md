---
title: >-
  [Paper Note] CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing
description: >-
  [ICLR2026][Multimodal VLM][urban computing] CityLens is introduced as the largest urban socioeconomic sensing benchmark to date (17 cities, 6 domains, 11 prediction tasks)…
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "urban computing"
  - "socioeconomic sensing"
  - "benchmark"
  - "vision-language model"
  - "street view"
date: 2026-05-08
content_hash: 42be3b401c662933
---

# CityLens: Evaluating Large Vision-Language Models for Urban Socioeconomic Sensing

**Conference**: ICLR2026
**arXiv**: [2506.00530](https://arxiv.org/abs/2506.00530)
**Code**: [https://github.com/tsinghua-fib-lab/CityLens](https://github.com/tsinghua-fib-lab/CityLens)
**Area**: Multimodal VLM
**Keywords**: urban computing, socioeconomic sensing, benchmark, vision-language model, street view

## TL;DR
CityLens is introduced as the largest urban socioeconomic sensing benchmark to date (17 cities, 6 domains, 11 prediction tasks), evaluating 17 LVLMs across three paradigms—direct metric prediction, normalized metric estimation, and feature-based regression—for inferring socioeconomic indicators from satellite and street-view imagery. Results show that general-purpose LVLMs still fall short of domain-specialized contrastive learning methods on most tasks.

## Background & Motivation

**Background**: Inferring socioeconomic indicators (GDP, crime rates, education levels, etc.) from urban imagery is a core task in urban computing. Conventional approaches apply contrastive learning (e.g., UrbanCLIP, UrbanVLP) to extract visual features from street-view or satellite images for regression, but suffer from poor cross-country generalization, inability to handle unstructured multimodal data, and lack of cultural-semantic understanding.

**Limitations of Prior Work**: (a) LVLMs possess multimodal understanding and broad world knowledge, making them theoretically suitable for such tasks, yet systematic evaluation is lacking—existing work covers limited spatial areas, uses narrow metrics, and involves small model scales. (b) No unified benchmark exists to measure LVLMs' urban sensing capabilities across diverse tasks, regions, and modalities.

**Key Challenge**: LVLMs exhibit strong visual understanding and reasoning capabilities, yet whether they can effectively extract socioeconomic signals from urban imagery remains an open question requiring large-scale systematic evaluation.

**Goal**: Construct the most comprehensive urban socioeconomic benchmark and systematically evaluate the capability boundaries of LVLMs.

**Key Insight**: A unified benchmark spanning multiple cities, domains, and modalities, combined with three complementary evaluation paradigms.

**Core Idea**: Conduct large-scale experiments across 17 cities × 11 indicators × 3 evaluation paradigms × 17 models to comprehensively measure the capabilities and limitations of LVLMs in urban socioeconomic sensing.

## Method

### Overall Architecture
On the data side: 17 global cities (spanning 6 continents including the US, UK, China, Africa, and South America), with 1 satellite image and 10 street-view images per region, paired with ground-truth labels for 11 socioeconomic indicators. On the evaluation side: 3 complementary paradigms test different capability dimensions of LVLMs.

### Key Designs

1. **Dataset Construction**:

    - Function: Construct a multimodal urban socioeconomic dataset
    - Indicator selection: Narrowed from an initial 28 indicators to 11, based on visual inferability and Pearson correlation-based redundancy removal. Covers economics (GDP, housing prices, income Gini), education (bachelor's degree attainment), crime (violent/non-violent), transportation (transit/driving share), health (mental health, healthcare accessibility, life expectancy), and environment (carbon emissions, building height)
    - Spatial mapping: US census tract level, UK MSOA level, and satellite-image-covered regions globally. Up to 500–1,000 samples per task
    - Design Motivation: Only indicators that humans can reasonably infer from imagery are selected (excluding visually unrelated ones such as "daily commute distance"), ensuring the evaluation targets visual perception rather than guessing

2. **Direct Metric Prediction**:

    - Function: Given regional imagery, prompt the LVLM to predict specific indicator values directly
    - Mechanism: Prompts instruct the model to act as an urban socioeconomic expert and estimate quantities such as "What is the public transit share in this area?"
    - Design Motivation: Tests whether LVLMs can translate visual cues into precise numerical values—the most challenging paradigm

3. **Normalized Metric Estimation**:

    - Function: Normalize indicators to a 0.0–9.9 scale and ask the model to estimate the relative level
    - Mechanism: Following GeoLLM, absolute value prediction is simplified to relative rank estimation, reducing task difficulty
    - Design Motivation: Tests whether LVLMs possess coarse-grained spatial knowledge—even without knowing exact GDP, can they judge "this area has a high economic level"?

4. **Feature-Based Regression**:

    - Function: Ask the LVLM to score street-view images along 13 predefined visual attributes (greenery, vehicles, building facades, etc.), then apply LASSO regression to predict indicators
    - Mechanism: Rather than requiring direct numerical prediction, this paradigm tests whether the visual features extracted by the LVLM carry socioeconomic information
    - Design Motivation: This serves as an "upper-bound" paradigm for LVLMs as feature extractors—if the features are inadequate, direct prediction will be even worse

## Key Experimental Results

### Main Results (Feature-Based Regression, R² scores)

| Model | GDP | Population | Housing Price | Crime | Transit | Building Height | Mental Health | Bachelor's | Mean |
|-------|-----|------------|---------------|-------|---------|----------------|---------------|------------|------|
| UrbanVLP | 0.717 | 0.132 | 0.559 | 0.149 | 0.551 | 0.807 | 0.403 | 0.422 | **0.417** |
| GPT-4o | 0.500 | 0.330 | 0.140 | 0.083 | 0.470 | 0.620 | 0.138 | 0.300 | 0.310 |
| Gemma3-27B | 0.463 | 0.324 | 0.141 | 0.077 | 0.567 | 0.590 | 0.211 | 0.297 | 0.338 |
| Qwen2.5VL-72B | ~0.52 | ~0.35 | ~0.10 | ~0.08 | ~0.53 | ~0.65 | ~0.22 | ~0.30 | ~0.35 |

### Ablation Study (Effect of Number of Street-View Images)

| # Street-View Images | GDP R² | Housing Price R² | Bachelor's R² | Note |
|----------------------|--------|-----------------|---------------|------|
| 1 | Low | Low | Low | Insufficient information |
| 5 | Medium | Medium | Medium | Performance rises rapidly |
| 10 | Highest | Highest | Highest | Near saturation |

### Key Findings
- **General-purpose LVLMs underperform domain-specialized methods on most tasks**: UrbanVLP (contrastive learning baseline) substantially outperforms all LVLMs on GDP, housing prices, transportation, and building height, indicating that generic visual features are less effective than domain-specialized representations for urban sensing
- **Mental health and bachelor's degree attainment are the hardest tasks**: These indicators have weak correspondence with visual cues (R² near 0), demonstrating that current LVLMs cannot infer deep social characteristics from imagery
- **Scaling model size yields limited gains**: R² improvements from 3B to 72B parameters are marginal (~0.05–0.10), suggesting the bottleneck lies not in model scale but in fundamental methodology for urban visual understanding
- **Normalized estimation outperforms direct prediction**: Coarse-grained relative judgment is substantially easier than precise numerical prediction—LVLMs possess a degree of spatial intuition but lack precise quantification capability
- **Building height is the easiest task**: R² consistently exceeds 0.5, as it is the most directly observable visual indicator

## Highlights & Insights
- **Most comprehensive urban socioeconomic benchmark**: 17 cities × 11 indicators × 3 evaluation paradigms × 17 models—far exceeding prior work such as GeoLLM in scale, providing the community with a unified evaluation infrastructure
- **Complementary three-paradigm design**: Direct prediction measures precision; normalized estimation measures coarse-grained perception; feature-based regression measures representation quality—together providing a thorough diagnosis of LVLMs' capability boundaries
- **Visual inferability as an indicator selection principle**: Not all socioeconomic indicators should be predicted from imagery—only those that "humans can also reasonably infer from images" are retained, avoiding ill-posed evaluation settings
- **Systematic deficiencies of LVLMs in urban sensing are revealed**: This finding provides important directional guidance—domain-specialized visual pretraining for urban contexts is needed rather than simply scaling general-purpose models

## Limitations & Future Work
- **Benchmark rather than methods paper**: The core contribution is the evaluation framework, not a novel method. The paper lacks exploration of how to improve LVLMs' urban sensing capabilities
- **Label temporality**: Socioeconomic data and street-view imagery may have been collected at different times (e.g., 2019 crime data vs. 2024 street views), and temporal misalignment may affect results
- **Cultural bias**: LVLMs' training data skews toward cities in developed countries, potentially introducing systematic underperformance in African and South American cities—an issue insufficiently analyzed in the paper
- **Future directions**: (a) Visual instruction fine-tuning for urban domains; (b) multi-source fusion of street view, satellite imagery, and POI data; (c) temporal street-view analysis for tracking urban change

## Related Work & Insights
- **vs. GeoLLM**: GeoLLM relies solely on text prompts without imagery and operates only at coarse global granularity. CityLens is multimodal (satellite + street view) and fine-grained (census tract level)
- **vs. UrbanVLP/UrbanCLIP**: These are domain-specialized contrastive learning methods—more effective but with poor generalization. CityLens reveals the performance gap between general LVLMs and domain methods, providing an evaluation benchmark for closing that gap
- **vs. PlacePulse/StreetScore**: Early work focused only on "urban perception scoring" (safety, aesthetics); CityLens extends the scope to quantifiable socioeconomic indicators

## Rating
- Novelty: ⭐⭐⭐⭐ The most comprehensive urban socioeconomic benchmark with a novel three-paradigm evaluation design, though methodological innovation is limited
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 17 models × 11 tasks × 3 paradigms, with comprehensive ablations over modality, number of images, and model scale
- Writing Quality: ⭐⭐⭐⭐ Dataset construction pipeline is clearly described and analysis is thorough, though the paper is lengthy
- Value: ⭐⭐⭐⭐ Provides much-needed evaluation infrastructure for applying LVLMs to urban computing

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[NeurIPS 2025\] CHOICE: Benchmarking the Remote Sensing Capabilities of Large Vision-Language Models](../../NeurIPS2025/multimodal_vlm/choice_benchmarking_the_remote_sensing_capabilities_of_large_vision-language_mod.md)
- [\[ICLR 2026\] Detecting Misbehaviors of Large Vision-Language Models by Evidential Uncertainty Quantification](detecting_misbehaviors_of_large_vision-language_models_by_evidential_uncertainty.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)
- [\[ICLR 2026\] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models](frieda_benchmarking_multi-step_cartographic_reasoning_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
