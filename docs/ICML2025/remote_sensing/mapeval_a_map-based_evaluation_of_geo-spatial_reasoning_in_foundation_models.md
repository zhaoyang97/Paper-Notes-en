---
title: >-
  [Paper Note] MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models
description: >-
  [Remote Sensing] This paper proposes the MapEval benchmark, which systematically evaluates the geo-spatial reasoning capabilities of 30 foundation models in map scenarios using 700 multiple-choice questions across textual, API, and visual tasks. The results show that the strongest model achieves an accuracy of no more than 67%, and all models lag behind human performance by over 20%.
tags:
  - "Remote Sensing"
date: 2026-05-08
content_hash: 5cf1f45aa2450399
---

# MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models

- **Conference**: ICML 2025 (Spotlight)
- **arXiv**: [2501.00316](https://arxiv.org/abs/2501.00316)
- **Code**: [GitHub - MapEval](https://github.com/MapEval)
- **Area**: Remote Sensing / Geo-Spatial Reasoning
- **Keywords**: Geo-Spatial Reasoning, Benchmark, Foundation Models, Map-Based QA, LLM Evaluation

## TL;DR

This paper proposes the MapEval benchmark, which systematically evaluates the geo-spatial reasoning capabilities of 30 foundation models in map scenarios using 700 multiple-choice questions across textual, API, and visual tasks. The results show that the strongest model achieves an accuracy of no more than 67%, and all models lag behind human performance by over 20%.

## Background & Motivation

Foundation models (such as GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro) have made significant progress in natural language reasoning and tool use, but their **geo-spatial reasoning** capabilities in map scenarios have not been fully explored. Existing geo-spatial QA benchmarks suffer from the following limitations:

**Single Task Types**: Most benchmarks focus on simple location queries (such as POI retrieval) and lack coverage of complex spatial relationships or navigation planning.

**Lack of Multimodal Evaluation**: They fail to simultaneously evaluate reasoning capabilities across textual, visual, and API interaction modalities.

**Insufficient Geographical Coverage**: Existing datasets tend to concentrate on a few cities or countries, lacking global diversity.

**Lack of Tool Interaction Evaluation**: Real-world map usage involves API calls (e.g., Google Maps API), which existing benchmarks do not evaluate.

The motivation of MapEval is to construct a **comprehensive, multimodal, and globally representative** map reasoning benchmark to systematically reveal the shortcomings of current foundation models in spatial reasoning.

## Method

### Overall Architecture

MapEval consists of three subtasks designed to evaluate the geo-spatial reasoning capabilities of models under different information input modalities:

| Subtask | Input Format | Evaluation Focus | Question Count |
|---|---|---|---|
| MapEval-Textual | Structured text (place names, coordinates, operating hours, etc.) | Long-context reasoning, spatial relation understanding | 300 |
| MapEval-API | Models calling map APIs via tool functions | Agent tool use, API interaction reasoning | 300 |
| MapEval-Visual | Map screenshots (Google Maps visual snapshots) | Visual map understanding, map information extraction | 100 |

The overall dataset covers **180 cities across 54 countries**, totaling 700 multiple-choice questions.

### Key Designs

**1. Question Taxonomy**

MapEval divides map reasoning tasks into 5 major categories:

- **Place Info**: Attribute information about specific locations (ratings, operating hours, addresses, etc.)
- **Nearby**: POI search and recommendation based on spatial proximity
- **Route/Navigation**: Route planning, distance calculation, navigation direction judgment
- **Trip**: Multi-stop trip planning, time budgeting, itinerary optimization
- **Unanswerable**: Questions that cannot be answered due to insufficient information (evaluating the model's ability to refuse to answer)

**2. Data Construction Process**

- Expert annotators manually create questions based on Google Maps to ensure authenticity and diversity.
- Use the MapQaTor tool to cache API call results, constructing a static evaluation database to ensure reproducibility.
- Employ LLM filters to filter out simple questions that can be answered solely based on prior training knowledge (the no-context baseline achieves only 6.67% accuracy).

**3. Agent Evaluation Framework for MapEval-API**

In API tasks, models act as agents that can call the following simplified tool functions:
- `PlaceDetailsTool(placeId)` — Retrieve place details
- `NearbySearchTool(location, keyword, radius)` — Search for nearby places
- `TravelTimeTool(origin, destination, travelMode)` — Query travel time
- `DirectionsTool(origin, destination)` — Retrieve navigation routes

These tools encapsulate actual Google Maps API calls, reducing evaluation variance caused by API parameter variations.

### Evaluation Metrics

Multiple-choice question accuracy is used as the primary metric, computed separately across subtasks and categories, and compared against human performance.

## Key Experimental Results

### Main Results: Overall Performance of 30 Models

| Model | Textual Overall | Place Info | Nearby | Route | Trip | Unans. |
|---|---|---|---|---|---|---|
| Claude-3.5-Sonnet | **66.33** | 73.44 | 73.49 | 75.76 | 49.25 | 40.00 |
| GPT-4o | ~64 | — | — | — | — | — |
| Gemini-1.5-Pro | 66.33 | 65.63 | 74.70 | 69.70 | 47.76 | 85.00 |
| Llama-3.2-90B | 58.33 | 68.75 | 66.27 | 66.67 | 38.81 | 30.00 |
| Gemma-2.0-27B | 49.00 | 39.07 | 71.08 | 59.09 | 31.34 | 15.00 |
| Human Performance | **>86** | — | — | — | — | 65.00 |

**Key Finding in MapEval-API**: The Claude-3.5-Sonnet Agent outperforms GPT-4o and Gemini-1.5-Pro by approximately **16%** and **21%** in API tasks, respectively, while open-source models show an even wider gap.

### Ablation Study & In-depth Analysis

| Analysis Dimension | Key Findings |
|---|---|
| No-context Baseline | Claude-3.5-Sonnet achieves only 6.67% without context, demonstrating the necessity of external context |
| Open-ended vs MCQ | Open-ended response accuracy is significantly lower than MCQ (Textual: 55.33% vs 66.33%) |
| Fine-tuning Effect | Fine-tuning small open-source models (Phi-3.5-mini, Llama-3.2-3B, etc.) on MapEval-Textual yields less than a 5% improvement |
| Large VLMs | Qwen2.5-VL-72B reaches 60.35% on Visual tasks, narrowing the gap with closed-source models (vs 61.65%) |
| Calculator Assistance | Adding a calculator tool improves performance on questions involving distance/time calculations |

### Key Findings

1. **No model surpasses 67% accuracy**, and even the strongest closed-source model still has a **20%+** gap compared to human performance.
2. **Open-source models significantly lag behind closed-source models**, especially in API interaction and visual reasoning tasks.
3. Models perform the worst in **distance estimation, direction judgment, and route planning**.
4. Performance on the **Unanswerable category** is highly polarized: Claude-3.5-Sonnet (90%) far outperforms humans (65%) because models rely strictly on the context, whereas humans tend to guess.
5. Fine-tuning smaller models fails to significantly improve performance, indicating that the root cause lies in the **fundamental deficiencies of models in geo-spatial reasoning capabilities**.

## Highlights & Insights

1. **Unique Three-in-One Evaluation Framework**: This benchmark is the first to integrate textual, API, and visual modalities for map reasoning within a single framework, providing a comprehensive evaluation perspective.
2. **Elegant Agent Evaluation Design**: By encapsulating tool functions and caching API responses, it ensures both the authenticity and reproducibility of the evaluation.
3. **Reveals fundamental capability deficiencies**: Fine-tuning experiments prove that the performance gap is not due to a lack of training data, but rather a fundamental limitation in the models' spatial reasoning capabilities.
4. **MapQA Ecosystem**: MapEval is part of the larger MapQA ecosystem (MapQaTor → MapEval → MapAgent), forming a complete research pipeline from data construction to evaluation and agents.

## Limitations & Future Work

1. **Limited Data Scale**: 700 questions provide insufficient sample size in some fine-grained categories (e.g., only 64 questions for Place Info), making 1-2% differences between models lack statistical significance.
2. **MCQ Format Constraints**: Multiple-choice questions cannot fully reflect the open-ended spatial reasoning requirements in real-world scenarios.
3. **Single Map Source**: The benchmark is built solely on Google Maps, failing to cover other map services such as OpenStreetMap or Baidu Maps.
4. **Timeliness Issues**: Map data change over time (e.g., store closures, road modifications), and static caching may lead to evaluation discrepancies over time.
5. **Lack of Remote Sensing Imagery**: The benchmark uses only digital map screenshots and does not include remote sensing modalities like satellite imagery.

## Related Work & Insights

- **MapQaTor (ACL 2025)**: The data annotation tool for MapEval, used to cache API calls and construct a static evaluation database.
- **TravelPlanner (ICML 2024)**: Another benchmark focusing on travel planning, but limited strictly to textual reasoning.
- **GeoQuestions1089**: An early geo-spatial QA dataset, which lacks multimodal and tool interaction evaluations.
- **MapAgent (EACL 2026)**: A follow-up work that builds a stronger map reasoning agent based on MapEval.

**Insights**: Geo-spatial reasoning is a critical weakness in foundation models. Future research could combine remote sensing imagery, GIS tools, and structured geographic knowledge graphs to enhance the spatial understanding capabilities of models.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First map reasoning benchmark to integrate textual, API, and visual modalities.
- **Value**: ⭐⭐⭐⭐ — Directly serves to evaluate high-frequency application scenarios such as navigation and trip planning.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Evaluates 30 models and includes multi-dimensional ablation analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with comprehensive experiments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Causal Foundation Models: Disentangling Physics from Instrument Properties](causal_foundation_models_disentangling_physics_from_instrument_properties.md)
- [\[AAAI 2026\] Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments](../../AAAI2026/remote_sensing/consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra.md)
- [\[ICLR 2026\] Towards Faithful Reasoning in Remote Sensing: A Perceptually-Grounded Geospatial Chain-of-Thought for Vision-Language Models](../../ICLR2026/remote_sensing/towards_faithful_reasoning_in_remote_sensing_a_perceptually-grounded_geospatial_.md)
- [\[CVPR 2026\] WRIVINDER: Towards Spatial Intelligence for Geo-locating Ground Images onto Satellite Imagery](../../CVPR2026/remote_sensing/wrivinder_towards_spatial_intelligence_for_geo-locating_ground_images_onto_satel.md)
- [\[CVPR 2026\] IMAIA: Interactive Maps AI Assistant for Travel Planning and Geo-Spatial Intelligence](../../CVPR2026/remote_sensing/imaia_interactive_maps_ai_assistant_for_travel_planning_and_geo-spatial_intellig.md)

</div>

<!-- RELATED:END -->
