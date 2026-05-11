---
title: >-
  [Paper Note] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction
description: >-
  [AAAI 2026][Multimodal VLM][Geospatial Understanding] This paper proposes VIR-Bench — a benchmark based on 200 Japanese travel vlog videos that evaluates MLLMs' geospatial and temporal understanding via an itinerary reco…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Geospatial Understanding"
  - "Temporal Reasoning"
  - "Travel Video"
  - "Itinerary Reconstruction"
  - "MLLM Evaluation"
date: 2026-05-08
content_hash: b045eb6d3248acbd
---

# VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2509.19002](https://arxiv.org/abs/2509.19002)
**Code**: [https://github.com/nlp-waseda/VIR-Bench](https://github.com/nlp-waseda/VIR-Bench)
**Area**: Multimodal VLM
**Keywords**: Geospatial Understanding, Temporal Reasoning, Travel Video, Itinerary Reconstruction, MLLM Evaluation

## TL;DR
This paper proposes VIR-Bench — a benchmark based on 200 Japanese travel vlog videos that evaluates MLLMs' geospatial and temporal understanding via an itinerary reconstruction task (visiting order graph construction). Findings reveal that SOTA models (including GPT-4.1 and Gemini-2.5) still struggle significantly with POI recognition and temporal transition reasoning.

## Background & Motivation
**Background**: The video understanding capabilities of MLLMs are advancing rapidly, yet existing benchmarks (Ego4D, HourVideo, VSI-Bench, etc.) primarily focus on indoor scenes or short-range outdoor activities, lacking evaluation of geospatial-temporal understanding for long-distance travel (cross-city/cross-region).

**Limitations of Prior Work**: Long-distance geospatial-temporal reasoning is critical for embodied AI applications such as planning and navigation, yet no relevant benchmark exists.

**Key Challenge**: The spatial scale of existing benchmarks is too small (indoor/short-range) to evaluate models' understanding at the macro-geographic level (inter-city navigation) and over long temporal spans (multi-day itineraries).

**Goal**: Construct a video benchmark for evaluating MLLMs' long-distance geospatial-temporal understanding.

**Key Insight**: Use itinerary reconstruction from travel vlog videos as the test task — models must identify all visited locations along with their hierarchical relationships and temporal order from video content.

**Core Idea**: Reconstruct a visiting order graph from travel videos (nodes = locations, edges = inclusion/transition relations) to evaluate MLLMs' geospatial and temporal intelligence.

## Method

### Overall Architecture
VIR-Bench defines a hierarchical visiting order graph:
- **Node types**: Root → Prefecture → City (municipality) → POI (attractions/stations/restaurants, etc.)
- **Edge types**: Inclusion (containment relations, e.g., Tokyo → Shinjuku) and Transition (temporal transitions, e.g., Senso-ji → Tokyo Tower)
- **Two subtasks**: Node Prediction (identifying all visited locations) and Edge Prediction (inferring inclusion and transition relations)

### Key Designs

1. **Dataset Construction**:

    - 200 YouTube travel vlogs (100 English + 100 Japanese), covering 43 of Japan's 47 prefectures
    - 10 native Japanese annotators, each collecting and annotating 20 videos
    - Each POI annotated with timestamps and Google Maps URLs; detailed information retrieved via the Google Places API
    - A total of 3,689 POIs; visiting order graphs manually constructed with quality checks

2. **Node Prediction Task**:

    - Given a video, the model outputs a JSON list of all visited prefectures/cities/POIs
    - Evaluates geospatial understanding ability (analogous to the GeoGuessr game)
    - Assessed using macro-averaged Precision/Recall/F1

3. **Edge Prediction Task**:

    - Given a video along with all visited locations (gold labels, shuffled), the model predicts inclusion and transition edges
    - Inclusion evaluates geographic knowledge (A belongs to B); Transition evaluates temporal understanding (A precedes B)

### Loss & Training
This is a purely evaluative benchmark with no training involved. Mainstream MLLMs are evaluated in a zero-shot setting.

## Key Experimental Results

### Main Results

| Model | Node-Prefecture F1 | Node-City F1 | Node-POI F1 | Edge-Inclusion F1 | Edge-Transition F1 |
|------|----|----|----|----|----|
| VideoLLaMA3 | Low | Low | Very Low | Low | Very Low |
| InternVL3 | Medium | Medium | Low | Medium | Low |
| Qwen2.5-VL | Medium | Medium | Low | Medium | Low |
| GPT-4.1 | Relatively High | Medium-High | Medium | Medium-High | Medium |
| Gemini-2.5-Pro | Highest | Highest | Medium | Highest | Medium |

### Ablation Study

| Configuration | Effect | Notes |
|------|------|------|
| More video frames | Steady improvement | Additional visual context is beneficial |
| More reasoning effort | Improvement | o4-mini > GPT-4.1 |
| Audio input | Improvement | Gemini can leverage audio for additional cues |
| Open-source vs. closed-source | Closed-source leads by a large margin | Open-source models lack sufficient geographic knowledge |

### Key Findings
- **POI node prediction is the primary bottleneck**: All models perform substantially worse at the POI granularity than at the prefecture/city level — inferring specific attraction names from video frames is extremely challenging.
- **Transition edge prediction (temporal order reasoning) is difficult**: Understanding the full temporal timeline of a video remains beyond current model capabilities.
- **Open-source models lack geographic knowledge**: Models such as InternVL3 struggle even with prefecture recognition, indicating insufficient geographic knowledge in pretraining.
- **A travel-planning agent case study validates the benchmark's practical value**: Agents that integrate both video and itinerary information generate more feasible and appealing travel plans.

## Highlights & Insights
- **Elegant task design**: The visiting order graph jointly evaluates geographic knowledge (nodes + inclusion edges) and temporal reasoning (transition edges), covering two dimensions within a single task.
- **Filling the macro-scale gap**: Bridges the benchmark gap between the micro scale (indoor/short-range) and the macro scale (cross-city/cross-region).
- **Closed loop from evaluation to application**: Beyond benchmarking, a travel-planning agent is developed to validate the practical utility of the benchmark.

## Limitations & Future Work
- **Limited geographic coverage**: Only Japan is included; travel videos from other countries/regions may exhibit different characteristics.
- **POI annotation quality depends on video content**: Some vlogs have limited camera angles, making certain POIs difficult to annotate.
- **End-to-end generation is too challenging**: Node prediction and edge prediction are ultimately evaluated separately, forgoing the challenge of fully end-to-end assessment.

## Related Work & Insights
- **vs. VSI-Bench**: VSI-Bench evaluates 3D scene understanding (indoor); VIR-Bench evaluates cross-city geographic understanding (macro scale) — the two operate at entirely different spatial scales.
- **vs. CityGuessr**: CityGuessr performs geographic localization based on driving videos and static positioning, whereas VIR-Bench involves temporal sequences and itinerary reconstruction.
- **Implications for embodied AI**: Future embodied AI systems will need to perform planning at the city/regional scale; VIR-Bench precisely targets this capability.

## Rating
- Novelty: ⭐⭐⭐⭐ First video benchmark evaluating macro-scale geospatial-temporal understanding
- Experimental Thoroughness: ⭐⭐⭐⭐ Eight models evaluated with ablation analysis and agent case study
- Writing Quality: ⭐⭐⭐⭐ Task definitions are clear and dataset construction is well-specified
- Value: ⭐⭐⭐⭐ Fills a critical gap in macro-scale video understanding evaluation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](../../CVPR2026/multimodal_vlm/enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[ICLR 2026\] GTR-Bench: Evaluating Geo-Temporal Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/gtr-bench_evaluating_geo-temporal_reasoning_in_vision-language_mod.md)
- [\[ICCV 2025\] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?](../../ICCV2025/multimodal_vlm/sti-bench_are_mllms_ready_for_precise_spatial-temporal_world_understanding.md)
- [\[AAAI 2026\] CrossVid: A Comprehensive Benchmark for Evaluating Cross-Video Reasoning in Multimodal Large Language Models](crossvid_a_comprehensive_benchmark_for_evaluating_cross-vide.md)
- [\[AAAI 2026\] Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)

</div>

<!-- RELATED:END -->
