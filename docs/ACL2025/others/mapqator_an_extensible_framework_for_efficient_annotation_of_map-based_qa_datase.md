---
title: >-
  [Paper Note] MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets
description: >-
  [ACL 2025][Geospatial Reasoning] This paper proposes MapQaTor, an extensible open-source Web framework that integrates multiple map APIs (Google Maps, OpenStreetMap, etc.) to accelerate geospatial QA dataset annotation by at least 30 times, while ensuring data reproducibility through API response caching.
tags:
  - "ACL 2025"
  - "Geospatial Reasoning"
  - "QA Dataset Annotation"
  - "Map APIs"
  - "LLM Evaluation"
  - "Tool Augmentation"
date: 2026-05-08
content_hash: b215a05d7a38faab
---

# MapQaTor: An Extensible Framework for Efficient Annotation of Map-Based QA Datasets

**Conference**: ACL 2025  
**arXiv**: [2412.21015](https://arxiv.org/abs/2412.21015)  
**Code**: [Available](https://github.com/mapqator/)  
**Area**: Others  
**Keywords**: Geospatial Reasoning, QA Dataset Annotation, Map APIs, LLM Evaluation, Tool Augmentation

## TL;DR

This paper proposes MapQaTor, an extensible open-source Web framework that integrates multiple map APIs (Google Maps, OpenStreetMap, etc.) to accelerate geospatial QA dataset annotation by at least 30 times, while ensuring data reproducibility through API response caching.

## Background & Motivation

Map and navigation services (Google Maps, Apple Maps, etc.) provide rich geospatial data but struggle to process natural language queries. LLMs have demonstrated strong capabilities in question-answering tasks, but creating reliable training and evaluation datasets for geospatial reasoning faces the following challenges:

**Inefficient Manual Collection**: Manually copying place information, route data, etc., from map services is extremely time-consuming.

**Irreproducibility**: Real-world map data constantly changes (restaurant ratings, opening hours, etc.), leading to unstable dataset quality.

**Lack of Provenance**: Manual methods struggle to trace data sources, affecting dataset trustworthiness.

**Lack of Tools**: There is currently no platform dedicated to annotating "language-map reasoning" tasks.

MapQaTor aims to provide a one-stop solution that integrates data acquisition, visualization, and annotation into a single platform.

## Method

### Overall Architecture

MapQaTor is a Web application comprising the following core modules:

1. **Context Designer**: Obtains structured data from map APIs using data collection tools.
2. **QA Design and Annotation Interface**: Creates QA pairs based on the acquired context.
3. **Context Optimizer**: Converts structured contexts into a compact, readable format.
4. **Caching Mechanism**: Caches API responses in a PostgreSQL database.
5. **Visualization Tool**: Displays embedded maps using the Google Maps JavaScript API.

### Key Designs

#### 1. Modular Data Collection Tools

**Function**: Provides five standardized tools to unify the functionalities of different map APIs.

**Five Tools**:
- **Text Search**: Free-text search for places (e.g., "Eiffel Tower" or "Starbucks near Central Park").
- **Place Details**: Retrieves detailed place information (opening hours, accessibility facilities, etc.).
- **Nearby Search**: Searches for nearby POIs, supporting filtering by rating and price range.
- **Compute Routes**: Generates navigation routes, supporting multiple waypoints and different travel modes.
- **Search Along Route**: Searches for POIs along a route.

**Design Motivation**: Each tool follows a unified input-output-context pattern, abstracting API differences through configurable adapters. A new API can be integrated simply by extending the base class and implementing `convertRequest`/`convertResponse` methods. Currently, 20 APIs from 6 providers have been integrated (Google Maps, OpenStreetMap, Mapbox, TomTom, HERE, Azure Maps).

#### 2. API Response Caching Mechanism

**Function**: Caches all API responses in PostgreSQL to ensure data consistency and reproducibility.

**Mechanism**:
- Identical queries return identical cached results, even if real-world data has changed.
- Reduces duplicate API calls to save costs (allowing large-scale annotation within the free tier of Google Maps).
- Caches retain original JSON responses, standardized fields, and metadata (timestamps, API providers, query parameters).

**Design Motivation**: The time-varying nature of geospatial data is the core reason for the irreproducibility of QA datasets. Caching fundamentally resolves this issue.

#### 3. Context Optimization

**Function**: Converts structured contexts into formatted contexts, substantially reducing token usage.

**Effect**: Reduces structured context from an average of 17,534 characters to a formatted context of 2,536 characters, a **85.54%** reduction. This significantly lowers evaluation costs without losing key information.

#### 4. Secure API Handling

**Function**: The backend securely proxies API requests; the frontend uses credential placeholders, and the backend injects the actual API Key.

**Design Motivation**: API keys are never exposed in the client-side code, ensuring security.

#### 5. QA Design and Annotation Interface

**Function**: Supports four answer formats (Yes/No, single-choice, multiple-choice, open-ended), supports question categorization, provides place name autocompletion, and integrates AI-assisted question generation powered by Gemini-2.0-Flash.

### Loss & Training

This is a tool/framework paper and does not involve model training.

## Key Experimental Results

### Main Results

**Efficiency comparison: MapQaTor vs. Manual Method**:

| Task | MapQaTor | Manual Method | Gain |
|------|----------|---------------|--------|
| Place Details | 10.17 s | 487 s | **47.9×** |
| Nearby Search | 12.50 s | 456 s | **36.5×** |
| Compute Routes | 14.00 s | 516.5 s | **36.9×** |
| Search Along Route | 15.66 s | 476 s | **30.4×** |

MapQaTor is at least **30 times** faster than the manual method across all tasks.

### MapEval Benchmark Experiments

**Evaluating LLM geospatial reasoning capabilities based on 300 MCQs annotated by MapQaTor**:

| Model | Accuracy |
|------|--------|
| Claude-3.5-Sonnet | 66.33% |
| Gemini-1.5-Pro | 66.33% |
| GPT-4o | 63.33% |
| **Human (using MapQaTor visualization)** | **86.67%** |

The highest accuracy of LLMs is 66.33%, which is over 20 percentage points lower than that of humans (86.67%). The gap primarily stems from the fact that humans can leverage map visualization to understand spatial relations, whereas LLMs can only process text context.

### Key Findings

1. **Significant Efficiency Gain**: At least a 30x speedup, approaching nearly 50x on certain tasks.
2. **Obvious Shortcomings in LLM Geospatial Reasoning**: Even the best model achieves only 66% accuracy, with complex spatial tasks (e.g., route reasoning) being particularly challenging.
3. **Visualization is Crucially Important for Human Understanding**: Human accuracy is over 20% higher with map visualization support than without.
4. **Context Optimization Saves over 85% of Tokens**: Dramatically reducing LLM evaluation costs.
5. **Caching Mechanism Enables Annotation Within Free API Tiers**: Lowering the financial barrier to dataset construction.
6. **Modular Design Guarantees Extensibility**: Integrating a new API requires implementing only two methods.

## Highlights & Insights

- **Filling an Important Gap**: The first framework specifically tailored for annotating map reasoning QA datasets, with precise problem targeting.
- **Excellent Engineering Design**: Details such as caching, secure API handling, and context optimization reflect high engineering maturity.
- **Dataset Reproducibility is the Core Innovation**: The time-varying nature of geospatial data has been a long-standing pain point in this field, which the caching mechanism solves elegantly.
- **Open Ecosystem**: 6 API providers + modular extensibility, lowering the barrier to adoption for the community.
- **Revealing the Root Cause of the LLM-Human Gap**: The difference between visualization and text-only capabilities, offering insights for multimodal geospatial reasoning research.

## Limitations & Future Work

1. Dependency on paid map APIs (currently free during the demo period, but requiring users to configure their own API keys in the long term).
2. Platform functionality is affected by the stability of external APIs; changes or deprecations in APIs will impact data collection.
3. The quality of QA pairs depends on the annotators' question design ability, which may introduce subjective bias.
4. Visualization currently only utilizes the Google Maps JavaScript API, lacking support for other map rendering engines.
5. Evaluation metrics focus primarily on efficiency, lacking systematic assessment of annotation quality (e.g., Kappa inter-annotator agreement).
6. Non-map platforms that are rich in geographical information, such as TripAdvisor, are not yet supported.

## Related Work & Insights

- **Tool-use Datasets** (ToolBench, APIBench) contain location tasks but lack traceability and reproducibility.
- **Geospatial NLP** (Cai & Hovy 2010; Zheng et al. 2014) explored text processing of geographical information but did not focus on QA dataset construction.
- **MapEval** (Dihan et al. 2025) is the first map reasoning evaluation benchmark built on MapQaTor, demonstrating the practical utility of the framework.
- **WebArena / VisualWebArena** include map-use scenarios but do not provide systematic dataset construction tools.

## Rating

- **Novelty**: ⭐⭐⭐ — The framework design concept (API integration + caching + annotation) is novel at the system level, though technical depth is moderate.
- **Experimental Thoroughness**: ⭐⭐⭐ — The efficiency comparison is clear but the scenarios are limited (4 tasks, 2 subjects), and the MapEval benchmark contains only 300 questions.
- **Writing Quality**: ⭐⭐⭐⭐ — Standardized formatting for a demo paper, with clear architecture diagrams, interface screenshots, and complete functional descriptions.
- **Value**: ⭐⭐⭐⭐ — Open-source framework + 20 integrated APIs + delivery of the MapEval benchmark, exhibiting practical tool value for the geospatial NLP community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GoR: A Unified and Extensible Generative Framework for Ordinal Regression](../../ICLR2026/others/gor_a_unified_and_extensible_generative_framework_for_ordinal_regression.md)
- [\[ACL 2025\] Map&Make: Schema Guided Text to Table Generation](mapmake_schema_guided_text_to_table_generation.md)
- [\[ACL 2025\] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [\[ACL 2025\] DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](dape_v2_process_attention_score_as_feature_map_for_length_extrapolation.md)
- [\[ACL 2025\] Inter-Passage Verification for Multi-evidence Multi-answer QA](inter-passage_verification_for_multi-evidence_multi-answer_qa.md)

</div>

<!-- RELATED:END -->
