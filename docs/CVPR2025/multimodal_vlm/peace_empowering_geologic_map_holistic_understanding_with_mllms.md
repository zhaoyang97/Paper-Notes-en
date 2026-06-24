---
title: >-
  [Paper Note] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs
description: >-
  [CVPR 2025][Multimodal VLM][Geologic Map Understanding] This paper constructs the first geologic map understanding benchmark, GeoMap-Bench (covering 5 capabilities, 25 tasks, and 3,864 questions), and proposes GeoMap-Agent (hierarchical information extraction + domain knowledge injection + enhanced QA), which significantly outperforms GPT-4o (scoring 0.811 overall vs. 0.369) in geologic map understanding.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Geologic Map Understanding"
  - "Multimodal Large Language Models (MLLMs)"
  - "AI Agent"
  - "Domain Knowledge Injection"
  - "Benchmark"
date: 2026-05-08
content_hash: 0dda975e5575d079
---

# PEACE: Empowering Geologic Map Holistic Understanding with MLLMs

**Conference**: CVPR 2025  
**arXiv**: [2501.06184](https://arxiv.org/abs/2501.06184)  
**Code**: None  
**Area**: Human understanding / Multimodal VLM  
**Keywords**: Geologic Map Understanding, Multimodal Large Language Models (MLLMs), AI Agent, Domain Knowledge Injection, Benchmark

## TL;DR
This paper constructs the first geologic map understanding benchmark, GeoMap-Bench (covering 5 capabilities, 25 tasks, and 3,864 questions), and proposes GeoMap-Agent (hierarchical information extraction + domain knowledge injection + enhanced QA), which significantly outperforms GPT-4o (scoring 0.811 overall vs. 0.369) in geologic map understanding.

## Background & Motivation

**Background**: Geologic maps are foundational diagrams in geology, crucial for hazard detection, resource exploration, and civil engineering. Although Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities in general image understanding, they perform poorly in geologic map understanding.

**Limitations of Prior Work**: Geologic map understanding faces three major challenges: (1) Ultra-high resolution: Geologic maps can reach resolutions of up to $10,000^2$ pixels, which far exceeds the input processing limits of typical MLLMs; (2) Multi-component association: Multiple components such as the title, legend, main map, cross-sections, and stratigraphic columns are highly interrelated; (3) Domain expertise: They contain symbolized geological features and diverse visual representations, requiring interdisciplinary knowledge across geology, geography, seismology, etc. Currently, there are no benchmarks or methods specifically designed for geologic map understanding.

**Key Challenge**: Even experienced geologists find it difficult to quickly retrieve and correlate knowledge from external data sources (e.g., geological, geographical, and seismological data), let alone AI models. Meanwhile, although MLLMs possess general image understanding capabilities, they lack the specialized processing abilities required for cartographic generalization.

**Goal**: (1) To construct a comprehensive evaluation benchmark for geologic map understanding; (2) to design a specialized AI agent for geologic map question answering and analysis.

**Key Insight**: Inspired by the interdisciplinary collaboration of human scientists, this work designs an AI expert panel as consultants, leveraging a diverse toolset (detection, OCR, segmentation, etc.) to comprehensively analyze complex questions.

**Core Idea**: Geologic maps are digitized into structured data via hierarchical information extraction, then domain knowledge bases are injected to enhance reasoning, and finally, enhanced prompts drive MLLMs to answer questions.

## Method

### Overall Architecture
GeoMap-Agent consists of three modules working in sequence: (1) Hierarchical Information Extraction (HIE) detects and crops various components of the geologic map (titles, legends, main maps, etc.) and extracts them into structured metadata using OCR and segmentation tools; (2) Domain Knowledge Injection (DKI) links the extracted metadata with external geological knowledge bases to inject professional information such as lithology and geologic ages; (3) Prompt-Enhanced Question Answering (PEQA) organizes the structured information and domain knowledge into enhanced prompts to guide the MLLM (GPT-4o) to generate accurate answers.

### Key Designs

1. **Hierarchical Information Extraction (Hierarchical Information Extraction, HIE)**:

    - Function: Digitize ultra-high-resolution geologic maps into structured, queryable metadata.
    - Mechanism: First detect the locations and bounding boxes of various components on the geologic map (title, scale bar, legend, main map, index map, cross-sections, stratigraphic columns). Then perform fine-grained extraction on each component: the color, text, lithology, and geologic age of each unit in the legend; fault lines and rock formation distributions in the main map; and stratigraphic structures in cross-sections. Specialized detection, OCR, and color analysis tools are used to construct a hierarchical information structure.
    - Design Motivation: The ultra-high resolution of geologic maps makes direct input to MLLMs infeasible (as details are lost through compression), requiring decomposition into component-level information first. The hierarchical structure ensures the integrity and queryability of the information.

2. **Domain Knowledge Injection (Domain Knowledge Injection, DKI)**:

    - Function: Link the extracted geological metadata with external geological knowledge bases to provide the professional background knowledge that MLLMs lack.
    - Mechanism: Construct a geological domain knowledge graph containing the geological time scale, rock classification systems, and structural geology knowledge. Map the metadata extracted by HIE (e.g., lithology codes, stratigraphic symbols) to corresponding entries in the knowledge graph to retrieve detailed geological meanings. For example, mapping "Qal" in the legend to "Quaternary alluvium" and associating it with its formation environment and potential mineral resources.
    - Design Motivation: Geologic maps employ highly stylized coding systems (e.g., color-to-stratigraphy correspondences, abbreviation-to-lithology mappings). This professional knowledge cannot be inferred from the image alone and must be retrieved from external knowledge sources.

3. **Prompt-enhanced Question Answering (Prompt-enhanced Question Answering, PEQA)**:

    - Function: Organize structured information and domain knowledge into effective prompts to guide the MLLM to generate accurate and detailed answers.
    - Mechanism: Selectively organize relevant structured information and knowledge as context to construct enhanced prompts based on the question type (extraction, localization, citation, reasoning, analysis). Similar to the design of an "AI expert panel", outputs from different modules are aggregated as opinions from different experts and provided to the MLLM.
    - Design Motivation: Directly feeding the question into an MLLM yields poor performance (GPT-4o scores only 0.369). There is a critical need to transform the complex geologic map understanding problem into a "reasoning given context" format where MLLMs excel.

### Loss & Training
GeoMap-Agent is an inference-time agent system and does not involve training. GeoMap-Bench contains 124 geologic maps and 3,864 questions sourced from USGS (English) and CGS (Chinese), covering 25 tasks across five capabilities: extraction, localization, citation, reasoning, and analysis.

## Key Experimental Results

### Main Results
Performance of various models on GeoMap-Bench:

| Model | Extraction | Localization | Citation | Reasoning | Analysis | Overall |
|------|------|------|------|------|------|------|
| GPT-4o (Direct) | Low | Low | Low | Low | Low | **0.369** |
| Gemini-1.5-Pro | Low | Low | Low | Low | Low | Lower |
| Qwen-VL | Even Lower | Even Lower | Even Lower | Even Lower | Even Lower | Even Lower |
| **GeoMap-Agent** | **High** | **High** | **High** | **High** | **High** | **0.811** |

### Ablation Study

| Configuration | Overall Score | Description |
|------|---------|------|
| GPT-4o baseline | 0.369 | Direct understanding without assistance |
| + HIE | Significant improvement | Structured information aids in extraction and localization |
| + HIE + DKI | Further improvement | Domain knowledge aids in reasoning and analysis |
| + HIE + DKI + PEQA | **0.811** | Prompt engineering maximizes MLLM capabilities |

### Key Findings
- MLLMs perform far worse than humans in geologic map understanding—GPT-4o achieves only 0.369, exposing major deficiencies of these models in professional domains.
- The HIE module contributes the most; digitizing geologic maps into structured data is the key breakthrough.
- Domain knowledge injection yields the most significant improvement for reasoning and analytical questions, both of which require professional background knowledge.
- Understanding Chinese geologic maps (from CGS sources) is more difficult than English ones (from USGS sources).

## Highlights & Insights
- **First Geologic Map Understanding Benchmark and Agent**: GeoMap-Bench fills the evaluation gap for the AI understanding of geologic maps, with 25 tasks covering the complete capability spectrum from basic extraction to high-level analysis.
- **Tool-Augmented Agent Paradigm**: Instead of training an end-to-end model, this approach enhances existing MLLMs through a combination of tools such as detection, OCR, and knowledge graphs. This paradigm can be directly applied to other domain-specific diagram understanding tasks.
- **Structured Intermediate Representation**: Transforming unstructured, high-resolution images into structured metadata before reasoning effectively bypasses the resolution limits of MLLMs.

## Limitations & Future Work
- GeoMap-Bench currently contains only 124 geologic maps, making its scale limited.
- The component detection of HIE relies on standard geological map formats, and its adaptability to non-standard formats remains unverified.
- Tests were conducted only on USGS and CGS sources; geological maps from other countries or regions may follow different cartographic standards.
- The reasoning latency of the agent is relatively high (due to multi-step tool invocations), and its real-time performance needs improvement.

## Related Work & Insights
- **vs GeoBench/K2**: These works focus on text-only geological QA, without involving geologic map (image) understanding. GeoMap-Bench is the first multimodal geologic benchmark.
- **vs GeoGPT**: GeoGPT uses GIS tools to handle geospatial tasks but does not address geologic maps. GeoMap-Agent specifically targets geologic map information extraction and QA.
- **vs LHRS-Bench**: LHRS-Bench evaluates remote sensing image understanding, but remote sensing images and geologic maps differ fundamentally in content and complexity.

## Rating
- Novelty: ⭐⭐⭐⭐ Defining the geologic map understanding problem systematically for the first time; both the benchmark and the agent are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model comparisons and ablations are thorough, although the data scale could be expanded.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, though the method description could be more refined.
- Value: ⭐⭐⭐⭐ Provides a significant boost to AI applications in geology, and the agent paradigm is highly generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Don't Just Chase "Highlighted Tokens" in MLLMs: Revisiting Visual Holistic Context Retention](../../NeurIPS2025/multimodal_vlm/dont_just_chase_highlighted_tokens_in_mllms_revisiting_visual_holistic_context_r.md)
- [\[CVPR 2025\] SegAgent: Exploring Pixel Understanding Capabilities in MLLMs by Imitating Human Annotator Trajectories](segagent_exploring_pixel_understanding_capabilities_in_mllms_by_imitating_human_.md)
- [\[NeurIPS 2025\] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images](../../NeurIPS2025/multimodal_vlm/gem_empowering_mllm_for_grounded_ecg_understanding_with_time_series_and_images.md)
- [\[ICML 2026\] ACTIVE-o3: Empowering MLLMs with Active Perception via Pure Reinforcement Learning](../../ICML2026/multimodal_vlm/active-o3_empowering_mllms_with_active_perception_via_pure_reinforcement_learnin.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](../../ICCV2025/multimodal_vlm/spatial_preference_rewarding_for_mllms_spatial_understanding.md)

</div>

<!-- RELATED:END -->
