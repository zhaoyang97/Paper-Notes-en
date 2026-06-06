---
title: >-
  [Paper Note] FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models
description: >-
  [ICLR2026][Multimodal VLM][cartographic reasoning] This paper introduces FRIEDA, a benchmark that systematically evaluates large vision-language models (LVLMs) on multi-step…
tags:
  - "ICLR2026"
  - "Multimodal VLM"
  - "cartographic reasoning"
  - "map VQA"
  - "spatial relations"
  - "multi-image reasoning"
  - "benchmark"
date: 2026-05-08
content_hash: bb772d96189588f4
---

# FRIEDA: Benchmarking Multi-Step Cartographic Reasoning in Vision-Language Models

**Conference**: ICLR2026
**arXiv**: [2512.08016](https://arxiv.org/abs/2512.08016)  
**Code**: [knowledge-computing/FRIEDA](https://github.com/knowledge-computing/FRIEDA)  
**Area**: Multimodal VLM
**Keywords**: cartographic reasoning, map VQA, spatial relations, multi-image reasoning, benchmark

## TL;DR

This paper introduces FRIEDA, a benchmark that systematically evaluates large vision-language models (LVLMs) on multi-step, cross-map cartographic reasoning. The strongest model, Gemini-2.5-Pro, achieves only 38.20% accuracy, far below the human baseline of 84.87%.

## Background & Motivation

- Cartographic reasoning is a core human cognitive capability, involving comprehensive interpretation of legends, scale bars, compasses, map text, and geometric features, and is indispensable in real-world applications such as urban planning and disaster response.
- Existing LVLM research typically treats maps as a special case of charts, neglecting the unique symbolic grammar and spatial relational reasoning that maps require.
- Current map VQA benchmarks exhibit significant shortcomings: (1) most cover only a subset of spatial relations (e.g., navigation or entity recognition only); (2) map styles are limited (predominantly choropleth or web basemaps); (3) cross-map reasoning is rarely addressed; (4) in-document map retrieval scenarios are absent.
- Consequently, existing benchmarks cannot comprehensively assess whether LVLMs possess human-level map reading ability.

## Core Problem

How to design a cartographic reasoning benchmark that covers all three categories of spatial relations (topological, metric, and directional), requires multi-step reasoning and cross-map integration, and closely reflects real-world document usage scenarios?

## Method

### Task Definition

FRIEDA organizes questions around four core dimensions:

1. **Spatial Relation Reasoning**: Based on the three major spatial relation categories in the GIS literature:
    - Topological relations: border (shared boundary), equal (geometric coincidence), intersect (overlap), within (containment)
    - Metric relations: distance (computing real-world distances using scale bars)
    - Directional relations: orientation (determining cardinal directions using compass roses)
2. **Map Element Interpretation**: Requires understanding the semantics of map text, legend, map scale, and compass.
3. **Cross-Map Reasoning**: Requires aligning shared symbols, labels, and scale bars across multiple maps and integrating multi-source evidence.
4. **Contextual Setting**: The model must first retrieve the relevant map from multiple maps within the same document before answering.

### Benchmark Construction Pipeline

1. **Map Collection**: Maps are collected from publicly available government reports, environmental assessment documents, geological surveys, and other sources across six thematic domains, covering 32 countries with highly diverse styles.
2. **Question Generation**: GPT-4/GPT-o3 is used to generate candidate questions, ensuring that no question can be answered via search engines or without viewing the map.
3. **Expert Review**: Two GIS experts (with 7 and 2 years of experience, respectively) manually verify answers and resolve ambiguous questions.
4. **Annotation Validation**: Eleven doctoral researchers (8 with cartographic expertise) conduct a four-week annotation process; only questions for which ≥2/3 of annotators agree on the gold-standard answer are retained, yielding a final set of 500 questions.

### Dataset Statistics

| Item | Count |
|------|-------|
| Total questions | 500 |
| Source documents | 210 |
| Total maps | 17,030 |
| Single-map questions | 202 (40.4%) |
| Multi-map questions | 298 (59.6%) |
| Questions requiring legend | 417 (83.4%) |
| Avg. maps per contextual question | 9.5 |

### Evaluation Protocol

Answers fall into three categories, each evaluated differently:

- **Text answers**: Mistral Small 3.1 is used as an LLM-as-Judge for semantic matching rather than exact string comparison.
- **Distance answers**: Unit-aware parsing + MAPE; answers within 20% error are considered correct.
- **Directional answers**: Adjacent cardinal direction tolerance is permitted (e.g., if the gold standard is North, NW and NE are also accepted).

## Key Experimental Results

### Overall Performance

| Model | Accuracy |
|-------|----------|
| Human average | 84.87% |
| Gemini-2.5-Pro | 38.20% |
| GPT-5-Think | 37.20% |
| Claude-Sonnet-4 | 31.60% |
| Qwen2.5-VL-72B (best open-source) | 25.60% |
| Ovis2.5-9B-Think | 25.80% |

### Analysis by Spatial Relation

- **Orientation** is the category where models perform best: Gemini-2.5-Pro reaches 71.59%.
- **Distance** is the most challenging: the best model achieves only 27.47% (GPT-5-Think), and human performance is also relatively lower at 78.28%.
- For **equal** relations, GPT-5-Think (44.44%) significantly outperforms Gemini-2.5-Pro (33.33%), reflecting its advantage in multi-map reasoning.
- Claude-Sonnet-4 performs best on **distance** questions, demonstrating stronger scale bar interpretation.

### Key Findings

- The accuracy gap between direct and contextual settings is minimal (88.03% question-level consistency), indicating that the primary bottleneck lies in cartographic reasoning itself rather than map retrieval.
- Model size shows no clear positive correlation with performance; training data and reasoning mechanisms are more critical.
- Enabling the Think mode improves Ovis2.5-9B by approximately 5%, primarily in directional judgment and multi-map alignment.

### Error Analysis (Gemini-2.5-Pro)

| Error Type | Proportion |
|------------|------------|
| Legend misinterpretation (color/symbol mapping errors) | 25.61% |
| Cross-map interpretation failure | 23.78% |
| Spatial relation semantic confusion | 16.46% |
| Scale bar errors | 9.76% |
| Incorrect map text selection | 8.93% |
| Counting errors | 6.71% |

## Highlights & Insights

- **Comprehensive spatial relation coverage**: The first map VQA benchmark to systematically cover all three major categories—topological, metric, and directional—comprising six relation types in total.
- **Cross-map reasoning**: 59.6% of questions require joint reasoning across multiple maps, filling a critical evaluation gap in multi-map cartographic reasoning.
- **Real-world map diversity**: Sourced from 210 real documents across 32 countries and six domains (geology, urban planning, environmental assessment, etc.), avoiding the simplification bias inherent in synthetic maps.
- **Rigorous quality control**: Expert curation, annotation by 11 doctoral researchers, and a ≥2/3 consensus filter ensure high question quality.
- **Dual-mode evaluation**: Direct and contextual settings decouple reasoning capability from retrieval capability.

## Limitations & Future Work

- The dataset covers only Latin-script documents, excluding maps in Chinese, Arabic, and other languages.
- The scale of 500 questions is relatively limited, and sample sizes across spatial relation subcategories are uneven.
- Evaluation of fine-tuned models is absent, making it difficult to assess whether domain adaptation can substantially improve performance.
- The reliability of LLM-as-Judge evaluation depends on the specific evaluation model and may introduce bias.
- The effects of chain-of-thought prompting or tool augmentation (e.g., GIS API calls) on performance remain unexplored.

## Related Work & Insights

| Dimension | MapQA/MapWise | MapEval | FRIEDA |
|-----------|--------------|---------|--------|
| Map types | Primarily choropleth | Web basemaps | Diverse real-document maps |
| Spatial relations | None | Partial | All three categories, six types |
| Multi-map reasoning | No | No | Yes (59.6%) |
| Document context | No | No | Yes (contextual setting) |
| Answer format | Multiple choice | MC/short answer | Open-ended |

Unlike spatial reasoning work on natural images such as SpatialVLM and SpatialRGPT, FRIEDA focuses on map-specific symbolic systems (legends, scale bars, compass roses), evaluating symbol-to-semantics mapping ability rather than spatial perception in natural scenes.

The benchmark reveals systematic deficiencies in current LVLMs' understanding of symbolic visual representations; legend misinterpretation accounts for the largest share of errors, suggesting insufficient modeling of discrete symbol-to-semantics mappings. Cross-map reasoning failures resemble alignment issues in multi-image VQA and may require explicit spatial alignment modules or attention mechanisms. Distance estimation—which demands scale bar comprehension followed by numerical computation—represents a distinctive failure mode where tool-augmented LLMs may offer a viable solution. Directional reasoning performs relatively well, indicating that models have acquired basic compass recognition, yet still fail when the compass is rotated.

## Rating

- Novelty: ⭐⭐⭐⭐ — First benchmark to comprehensively cover multiple spatial relation categories using real-world maps
- Experimental Thoroughness: ⭐⭐⭐⭐ — 11 models + human baseline + fine-grained error analysis
- Writing Quality: ⭐⭐⭐⭐ — Task definitions are clear, and GIS theory is tightly integrated with LVLM evaluation
- Value: ⭐⭐⭐⭐ — Fills an important evaluation gap with practical significance for advancing spatial intelligence in LVLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](../../ACL2026/multimodal_vlm/omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ICLR 2026\] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)
- [\[AAAI 2026\] FinMMDocR: Benchmarking Financial Multimodal Reasoning with Scenario Awareness, Document Understanding, and Multi-Step Computation](../../AAAI2026/multimodal_vlm/finmmdocr_benchmarking_financial_multimodal_reasoning_with_scenario_awareness_do.md)
- [\[ACL 2026\] OMHBench: Benchmarking Balanced and Grounded Omni-Modal Multi-Hop Reasoning](../../ACL2026/multimodal_vlm/omhbench_benchmarking_balanced_and_grounded_omni-modal_multi-hop_reasoning.md)
- [\[ICLR 2026\] VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs](visiomath_benchmarking_figure-based_mathematical_reasoning_in_lmms.md)

</div>

<!-- RELATED:END -->
