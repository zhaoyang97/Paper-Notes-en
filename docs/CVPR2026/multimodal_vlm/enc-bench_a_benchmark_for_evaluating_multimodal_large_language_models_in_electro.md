---
title: >-
  [Paper Note] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding
description: >-
  [CVPR 2026][Multimodal VLM][chart understanding] This paper introduces ENC-Bench, the first professional-grade benchmark for Electronic Navigational Chart (ENC) understanding, comprising 20…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "chart understanding"
  - "multimodal benchmark"
  - "spatial reasoning"
  - "safety-critical AI"
  - "symbol grounding"
date: 2026-05-08
content_hash: 1e05176703329485
---

# ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.22763](https://arxiv.org/abs/2603.22763)
**Code**: None
**Area**: Multimodal / VLM Benchmark
**Keywords**: chart understanding, multimodal benchmark, spatial reasoning, safety-critical AI, symbol grounding

## TL;DR

This paper introduces ENC-Bench, the first professional-grade benchmark for Electronic Navigational Chart (ENC) understanding, comprising 20,490 samples organized under a three-level hierarchical evaluation framework (Perception → Spatial Reasoning → Maritime Decision-Making). Systematic evaluation of 10 MLLMs reveals that the best-performing model achieves only 47.88% accuracy, exposing a critical capability gap of general-purpose models in safety-critical specialized domains.

## Background & Motivation

**Background**: Maritime shipping accounts for over 90% of global trade, and Electronic Navigational Charts (ENCs) have been mandated by the International Maritime Organization for commercial vessels. The maritime AI market is projected to reach $4.13 billion. MLLMs have demonstrated strong performance on general visual understanding tasks, yet their actual capabilities in specialized professional domains remain largely unknown.

**Limitations of Prior Work**: Existing benchmarks address statistical charts (ChartQA), documents (DocVQA), and geographic reasoning (MapQA), but none specifically targets ENCs — safety-critical professional charts that employ standardized vector symbology (IHO S-57), scale-dependent rendering, and multi-constraint spatial geometry.

**Key Challenge**: ENCs encode navigational regulations, water depths, and routing constraints in ways fundamentally different from natural images or statistical charts — requiring standardized symbol interpretation, spherical geodetic coordinate computation, and multi-constraint safety decision-making — none of which general-purpose MLLMs have been trained on.

**Goal**: To systematically evaluate whether current MLLMs can reliably interpret ENCs, and to quantify the practical capability boundaries of these models in the specialized maritime domain.

**Key Insight**: The evaluation framework is designed to mirror the cognitive workflow of a licensed mariner, spanning from symbol recognition to safety-critical decision-making in a hierarchical manner.

**Core Idea**: Construct a three-level hierarchical ENC benchmark (Perception → Spatial Reasoning → Decision-Making) to rigorously evaluate MLLM performance in maritime safety-critical scenarios for the first time.

## Method

### Overall Architecture

ENC-Bench employs a four-stage data generation pipeline: starting from 840 official NOAA S-57 navigational charts, the pipeline proceeds through multi-condition rendering via OpenCPN (3 lighting modes × 6 scale levels) → GDAL parsing and image registration (control-point matching to establish bidirectional pixel–geographic coordinate transformation) → feature annotation (graph coloring for overlap prevention, density control, and expert review) → task-template QA generation, ultimately producing 20,490 expert-validated samples.

### Key Designs

1. **Three-Level Hierarchical Evaluation Framework**:
   - **Function**: Decomposes ENC understanding into 10 task categories across three levels — Perception (L-1, 4 task types), Spatial Reasoning (L-2, 3 task types), and Maritime Decision-Making (L-3, 3 task types).
   - **Mechanism**: L-1 tests symbol recognition and attribute extraction for point/line/area features; L-2 requires numerical geometric computation including coordinate localization, bearing calculation (compass 0–360°), and distance measurement (nautical miles); L-3 requires heading direction judgment, safe passage assessment considering vessel draft, and emergency anchorage selection under multiple constraints.
   - **Design Motivation**: To simulate the progressive cognitive hierarchy of mariners, from basic symbol recognition to complex multi-constraint safety decision-making.

2. **Multi-Condition Rendering and Rigorous Quality Control**:
   - **Function**: Each chart is rendered under 3 lighting modes (day/dusk/night) × 6 zoom levels (1:50K–1:300K), yielding 18 rendering variants per chart.
   - **Mechanism**: Spatial reasoning answers are computed using verified nautical formulas including the Haversine distance formula and azimuth calculation. A two-stage validation process is applied — automated consistency checks (cross-validation of coordinates, depth values, and feature classifications) followed by expert review by maritime navigation specialists.
   - **Design Motivation**: To cover all display conditions encountered in real-world navigation operations, ensuring robustness of evaluation rather than performance under incidental conditions.

### Loss & Training

ENC-Bench is a purely evaluative benchmark with no training components. All models are assessed under a unified zero-shot protocol.

## Key Experimental Results

### Main Results

| Model | Symbol Rec. | Point Feat. | Line Feat. | Area Feat. | Heading | Safe Passage | Anchorage | Mean |
|-------|-------------|-------------|------------|------------|---------|--------------|-----------|------|
| Gemini-2.5-Pro | 69.53 | 45.38 | 30.05 | 39.95 | 63.12 | 57.55 | 29.55 | **47.88** |
| Qwen3-VL-235B | 57.03 | 51.79 | 29.70 | 29.93 | 74.47 | 58.97 | 26.48 | 46.91 |
| GPT-4o | 50.78 | 36.39 | 21.58 | 23.97 | 45.39 | 45.44 | 20.57 | 34.87 |
| GLM-4.5V | 38.80 | 43.44 | 20.92 | 21.61 | 53.19 | 65.67 | 26.24 | 38.55 |
| InternVL-3-38B | 55.99 | 27.36 | 19.87 | 30.29 | 51.06 | 54.13 | 20.57 | 37.04 |
| Random Baseline | 25.00 | 25.00 | 25.00 | 25.00 | 33.33 | 50.00 | 25.00 | 29.76 |

### Ablation Study

| Analysis Dimension | Key Metric | Notes |
|--------------------|------------|-------|
| Spatial Reasoning – Coordinate Localization | Gemini best Acc@200px = 21.43% | Even under a lenient threshold, most predictions remain substantially off (mean error: 480px) |
| Spatial Reasoning – Bearing Calculation | Gemini Acc@20° = 46.86% | Requires understanding of chart orientation and coordinate system transformation |
| Spatial Reasoning – Distance Measurement | Gemini Acc@0.2 = 25.67%, mean error 42.31% | Errors exceeding 20% indicate extremely weak scale interpretation ability |
| Lighting Mode Comparison | Night mode causes significant performance drop | High-contrast symbols on dark backgrounds interfere with model visual processing |
| Zoom Level Comparison | Small-scale charts (1:200K–300K) yield the worst performance | Map generalization alters feature density and symbology |

### Key Findings

- **Symbol Grounding Bottleneck**: All models systematically fail to interpret formal annotations such as coordinate grids and scale bars, indicating a fundamental lack of understanding of standardized symbol systems.
- **Multi-Constraint Reasoning Deficiency**: The best model achieves only 29.55% on anchorage selection (which simultaneously requires satisfying depth, distance, and restricted area constraints), with models tending toward greedy local optimization.
- **Scale and Lighting Fragility**: Performance degrades significantly under night-mode and small-scale conditions, revealing insufficient robustness.
- **Validation of Hierarchical Task Design**: Difficulty increases progressively from Perception → Spatial Reasoning → Decision-Making tasks, confirming the validity of the hierarchical evaluation design.

## Highlights & Insights

- ENC-Bench fills a critical gap in maritime AI evaluation; given the rapid advancement of MLLM capabilities, exposing this "blind spot" in specialized vertical domains carries significant cautionary value.
- The three-level cognitive hierarchy design is generalizable to benchmark construction in other safety-critical vertical domains such as medical imaging and air traffic control.
- The data generation pipeline (S-57 → rendering → registration → annotation → QA) is methodologically rigorous, and the 20,490-sample scale is sufficient.
- A counterintuitive finding is revealed: Qwen3-VL-235B surpasses Gemini-2.5-Pro on heading identification (74.47% vs. 63.12%), yet falls slightly short in overall mean accuracy.

## Limitations & Future Work

- Coverage is limited to NOAA (U.S.) navigational charts; ENCs from other countries and regions are absent.
- Only zero-shot evaluation is conducted; few-shot settings and the performance gains achievable through maritime domain fine-tuning remain unexplored.
- Absolute performance on spatial reasoning tasks is extremely low (coordinate localization < 22%), suggesting that dedicated spatial reasoning enhancement modules may be necessary.
- Sequential decision-making scenarios (e.g., multi-step route planning) are not included; the benchmark could be extended into a sequential decision-making setting.

## Related Work & Insights

- **vs. ChartQA/DocVQA**: These benchmarks focus on unstructured statistical charts and documents, whereas ENC-Bench addresses standardized vector symbology, regulatory encoding, and safety constraints.
- **vs. GeoQA/MathVista**: These perform geometric reasoning in idealized Euclidean coordinate systems, whereas ENC-Bench requires spherical geodetic coordinates and DMS notation.
- **vs. MapQA/MapEval**: These involve coarse spatial reasoning over consumer-grade maps, whereas ENC-Bench demands legally mandated positional accuracy and safety compliance.
- **Insights**: The core value of a specialized domain benchmark lies in defining the domain-specific "cognitive capability hierarchy" — this is more meaningful than simply accumulating sample volume.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First ENC understanding benchmark, opening an entirely new research direction with a clearly defined problem formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 10 MLLMs, 10 task categories, 18 rendering conditions, and detailed error analysis; exceptionally comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, thorough introduction of domain background, and rigorous statistical analysis.
- **Value**: ⭐⭐⭐⭐ — Carries important cautionary significance for safety-critical AI, urging the field to confront capability gaps in specialized domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)
- [\[AAAI 2026\] VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction](../../AAAI2026/multimodal_vlm/vir-bench_evaluating_geospatial_and_temporal_understanding_of_mllms_via_travel_v.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[CVPR 2026\] SpatiaLQA: A Benchmark for Evaluating Spatial Logical Reasoning in Vision-Language Models](spatialqa_a_benchmark_for_evaluating_spatial_logical_reasoning_in_vision-languag.md)
- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)

</div>

<!-- RELATED:END -->
