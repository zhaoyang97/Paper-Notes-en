---
title: >-
  [Paper Note] TripCraft: A Benchmark for Spatio-Temporally Fine Grained Travel Planning
description: >-
  [ACL 2025][LLM Evaluation][Travel planning benchmark] Proposes TripCraft, a travel planning benchmark dataset integrating real-world spatiotemporal constraints (public transit, activity availability, user personas, etc.), accompanied by five continuous evaluation metrics to systematically assess the quality of LLM-generated itineraries. It improves the temporal meal score from 61% to 80% under the parameter-informed setting.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Travel planning benchmark"
  - "Spatiotemporal constraints"
  - "Personalized itinerary generation"
  - "Continuous evaluation metrics"
date: 2026-05-08
content_hash: 180df66e75693231
---

# TripCraft: A Benchmark for Spatio-Temporally Fine Grained Travel Planning

**Conference**: ACL 2025  
**arXiv**: [2502.20508](https://arxiv.org/abs/2502.20508)  
**Code**: None (declared to be public after acceptance)  
**Area**: LLM Evaluation  
**Keywords**: Travel planning benchmark, Spatiotemporal constraints, LLM evaluation, Personalized itinerary generation, Continuous evaluation metrics

## TL;DR

Proposes TripCraft, a travel planning benchmark dataset integrating real-world spatiotemporal constraints (public transit, activity availability, user personas, etc.), accompanied by five continuous evaluation metrics to systematically assess the quality of LLM-generated itineraries. It improves the temporal meal score from 61% to 80% under the parameter-informed setting.

## Background & Motivation

**Background**: Recently, researchers have begun exploring the potential of utilizing Large Language Models (LLMs) as personalized travel planning agents. LLMs can generate multi-day itineraries based on user requirements, covering various aspects such as attraction visits, dining arrangements, and transportation planning. Existing benchmark datasets, such as TravelPlanner and TravelPlanner+, have laid the foundation for this field.

**Limitations of Prior Work**: Existing datasets suffer from three core issues: (1) Semi-synthetic data: relying on manually constructed or simplified data that fails to reflect the complexity of real-world travel scenarios; (2) Spatial inconsistency: inaccurate spatial information such as distances and flight/transit times between attractions, leading to physically infeasible generated itineraries; (3) Absence of key travel constraints: missing real-world constraints such as public transit timetables, attraction opening hours, activity availability, and personalized user preferences. Moreover, most existing evaluation methods adopt binary verification (satisfied/unsatisfied), which cannot measure itinerary quality in a fine-grained manner.

**Key Challenge**: Travel planning is inherently a complex decision-making problem involving multi-dimensional constraints. Temporal constraints (opening hours, dining time windows), spatial constraints (distances between attractions, transportation modes), and personal preference constraints (user personas, interest categories) must be satisfied simultaneously. Existing benchmarks lack both the modeling of these real-world constraints and the corresponding evaluation methods.

**Goal**: Construct a spatiotemporally consistent travel planning benchmark that integrates real-world constraints; design continuous evaluation metrics to replace binary verification; and evaluate the travel planning capabilities of mainstream LLMs.

**Key Insight**: Starting from real-world data, integrate actual distance data from Google Maps, public transportation timetables, attraction classification information, and user personas to construct a constraint-aware dataset.

**Core Idea**: Replace simple binary verification with five continuous evaluation metrics to quantify itinerary quality across temporal, spatial, ordering, and personalization dimensions, and introduce a parameter-informed setting to enable LLMs to generate superior itineraries using constraint parameters.

## Method

### Overall Architecture

The construction of TripCraft consists of three phases: (1) Data collection and construction: obtaining city attractions, restaurants, and transportation information from real-world sources to build a spatiotemporally consistent dataset; (2) Evaluation metric design: proposing five continuous evaluation metrics to quantify itinerary quality from different dimensions; (3) LLM evaluation experiments: assessing the travel planning capabilities of multiple LLMs under both direct generation and parameter-informed settings. The input is a user requirement description (destination, number of days, preferences, etc.), and the output is a complete daily itinerary.

### Key Designs

1. **Spatiotemporally Consistent Dataset Construction**:

    - **Function**: Provides a travel planning test environment under real-world constraints.
    - **Mechanism**: The dataset integrates multi-source real-world data: attraction information includes opening hours, categories (museums, parks, shopping, etc.), and geographic coordinates; restaurant information includes dining time windows (breakfast, lunch, dinner); transportation information comes from real public transit timetables, including actual travel times between stops; activity information includes time-sensitive events (such as exhibitions, shows) with available time windows. Spatial consistency is ensured by actual distances and travel times retrieved via the Google Maps API. User personas include personalized attributes such as age, interest preferences, and dining habits, used to test personalized planning capabilities. The data covers multiple cities and travel lengths ranging from 1 to 7 days.
    - **Design Motivation**: Existing datasets have inconsistent spatial information (e.g., the distance between attractions does not match reality), leading to physically infeasible itineraries generated by LLMs. Integrating real-world data addresses this fundamental problem.

2. **Five-Dimensional Continuous Evaluation Metric System**:

    - **Function**: Quantifies itinerary quality in a multi-dimensional, fine-grained manner.
    - **Mechanism**: Five continuous scores (0-100%) are designed to replace simple pass/fail judgments: (a) **Temporal Meal Score (TMS)** evaluates the temporal rationality of dining arrangements—whether breakfast, lunch, and dinner are scheduled within reasonable time windows; (b) **Temporal Attraction Score (TAS)** assesses whether attraction visiting times fall within opening hours; (c) **Spatial Score (SS)** evaluates whether spatial distances between adjacent attractions are reasonable, calculated based on actual travel times; (d) **Ordering Score (OS)** evaluates whether the visiting order of attractions is logical (e.g., visiting near before far, avoiding doubling back); (e) **Persona Score (PS)** evaluates whether the itinerary matches preferences in the user persona (e.g., whether users interested in history and culture are assigned enough museums). Each metric is a continuous value, finely reflecting the itinerary quality in that dimension.
    - **Design Motivation**: Binary validation cannot distinguish between "almost satisfying constraints" and "completely failing to satisfy constraints." Continuous metrics provide richer signals to guide model improvement.

3. **Parameter-Informed Setting**:

    - **Function**: Improves itinerary quality by providing constraint parameters to the LLMs.
    - **Mechanism**: Key constraint parameters—such as restaurant operation hours, approximate distances between attractions, and available time windows of activities—are explicitly provided in the prompt. LLMs no longer need to "guess" these constraints and can directly utilize this information for planning. Two modes are set up for comparative experiments: (a) Direct Generation: LLMs generate portfolios based solely on user requirements without additional constraints; (b) Parameter-Informed: constraint parameters are embedded in the prompt to guide generation.
    - **Design Motivation**: Although LLMs possess certain common-sense reasoning abilities, they lack reliable knowledge of precise spatiotemporal constraints (e.g., what time a specific restaurant opens). Explicitly providing parameters mitigates this deficiency.

### Loss & Training

This paper focuses on a benchmark dataset and evaluation framework; thus, it does not involve model training. Experiments conduct inference using APIs of existing LLMs.

## Key Experimental Results

### Main Results

| Model | TMS (7 days) | TAS | SS | OS | PS |
|------|----------|-----|-----|-----|-----|
| GPT-4 (Direct) | 61% | Relatively High | Medium | Relatively High | Medium |
| GPT-4 (Parameter-Informed) | **80%** | **High** | **Relatively High** | **High** | **Relatively High** |
| GPT-3.5 (Direct) | Low | Medium | Medium-Low | Medium | Medium-Low |
| GPT-3.5 (Parameter-Informed) | Medium | Relatively High | Medium | Relatively High | Medium |
| LLaMA Series | Relatively Low | Medium-Low | Low | Medium-Low | Low |

### Ablation Study

| Configuration | TMS | Description |
|------|-----|------|
| Direct Generation (7 days) | 61% | No constraint parameters |
| Parameter-Informed (7 days) | 80% | Temporal constraints provided |
| Direct Generation (1 day) | Relatively High | Fewer constraints for short itineraries |
| Parameter-Informed (1 day) | Higher | Relatively smaller improvement margin |
| Without Persona Info | PS drops significantly | Persona matching degrades |
| Without Spatial Constraints | SS drops | Spatial rationality weakens |

### Key Findings

- The parameter-informed setting significantly improves the performance of all models, especially in dining schedule arrangements (TMS from 61% to 80%), indicating weak temporal perception during LLM execution without explicit constraints.
- As the travel days increase (from 1 to 7 days), the performance of all models drops significantly, showing that long-term planning presents a greater challenge for LLMs.
- The Spatial Score (SS) is the worst-performing dimension across all LLMs, indicating that LLMs have limited capabilities in estimating actual geographic distances and transit times.
- There is a significant gap between open-source models (e.g., LLaMA series) and closed-source models (such as GPT-4), especially in scenarios that require adhering to complex constraints.
- The personalization dimension (PS) is another weakness for all models, as LLMs tend to generate "generic" itineraries rather than customizing them for specific user personas.

## Highlights & Insights

- The **five-dimensional continuous evaluation metrics** represent the most significant highlight of this work—breaking down the vague notion of "itinerary quality" into quantifiable dimensions such as time, space, order, and personalization, providing a fine-grained tool for travel planning evaluation. This evaluation framework can be transferred to other multi-constraint generation tasks (e.g., scheduling, logistics optimization).
- The **parameter-informed setting** is simple yet highly effective, revealing a fundamental limitation of LLMs in constraint awareness—they excel at following explicitly provided constraints rather than inferring implicit constraints on their own.
- Integrating dynamic constraints like public transit timetables and activity availability into the dataset makes the evaluation closer to real-world application scenarios.

## Limitations & Future Work

- The dataset scale and coverage may be limited, containing only a subset of cities, without fully considering different cultural backgrounds and travel habits.
- Although multi-dimensional, the evaluation metrics lack subjective quality measures (e.g., the fun and richness of the itinerary).
- Budget constraints and price sensitivity are not considered, which are highly critical in real-world travel planning.
- Future work can introduce multi-turn dialog scenarios, allowing LLMs to dynamically adjust itineraries based on user feedback.
- Tool augmentation methods combining external APIs (e.g., maps, ticket booking systems) can also be explored to improve the spatiotemporal reasoning capabilities of LLMs.

## Related Work & Insights

- **vs TravelPlanner**: TravelPlanner is the earliest LLM travel planning benchmark, but it relies on semi-synthetic data and suffers from spatial inconsistency. TripCraft displays significant improvements in both data authenticity and evaluation dimensions.
- **vs TravelPlanner+**: TravelPlanner+ adds some constraints, but still misses key constraints like public transit and activity availability, and its evaluation remains primarily based on binary verification.
- **vs NaturalPlan**: Google's NaturalPlan focuses on planning under natural language constraints but does not address spatiotemporal consistency issues. The evaluation perspectives of the two are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of the five-dimensional continuous evaluation metrics is novel, but the benchmark construction method is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ 18 tables comprehensively cover different models and settings, with detailed analysis.
- Writing Quality: ⭐⭐⭐⭐ 27 pages of detailed description with a clear structure.
- Value: ⭐⭐⭐⭐ Provides a more reasonable benchmark for LLM travel planning evaluation, driving progress in this field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TripTailor: A Real-World Benchmark for Personalized Travel Planning](triptailor_a_real-world_benchmark_for_personalized_travel_planning.md)
- [\[ACL 2026\] Beyond Itinerary Planning: A Real-World Benchmark for Multi-Turn and Tool-Using Travel Tasks](../../ACL2026/llm_evaluation/beyond_itinerary_planning-a_real-world_benchmark_for_multi-turn_and_tool-using_t.md)
- [\[ICCV 2025\] OmniDiff: A Comprehensive Benchmark for Fine-grained Image Difference Captioning](../../ICCV2025/llm_evaluation/omnidiff_a_comprehensive_benchmark_for_fine-grained_image_difference_captioning.md)
- [\[ACL 2025\] EcomScriptBench: A Multi-task Benchmark for E-commerce Script Planning via Step-wise Intention-Driven Product Association](ecomscriptbench.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)

</div>

<!-- RELATED:END -->
