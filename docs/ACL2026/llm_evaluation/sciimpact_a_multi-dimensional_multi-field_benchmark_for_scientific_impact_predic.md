---
title: >-
  [Paper Note] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction
description: >-
  [ACL 2026][LLM Evaluation][Scientific Impact Prediction] This paper constructs SciImpact—the first large-scale scientific impact prediction benchmark across 19 disciplines and 7 impact dimensions (citations, awards…
tags:
  - "ACL 2026"
  - "LLM Evaluation"
  - "Scientific Impact Prediction"
  - "Multi-Dimensional Benchmark"
  - "Citation Prediction"
  - "Academic Awards"
  - "Multi-Task Instruction Tuning"
date: 2026-05-08
content_hash: 684d13cce52ef3df
---

# SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction

**Conference**: ACL 2026  
**arXiv**: [2604.17141](https://arxiv.org/abs/2604.17141)  
**Code**: [Project Homepage](https://flypig23.github.io/sciimpact-homepage/)  
**Area**: LLM Evaluation  
**Keywords**: Scientific Impact Prediction, Multi-Dimensional Benchmark, Citation Prediction, Academic Awards, Multi-Task Instruction Tuning

## TL;DR

This paper constructs SciImpact—the first large-scale scientific impact prediction benchmark across 19 disciplines and 7 impact dimensions (citations, awards, patents, media, code, datasets, models). It contains 215,928 comparative paper pairs, enabling a 4B model to outperform large models like o4-mini through multi-task fine-tuning.

## Background & Motivation

**Background**: Scientific literature is growing exponentially, requiring automated methods for evaluating and predicting research impact. Existing work primarily focuses on citation count prediction.

**Limitations of Prior Work**: (1) Citation counts are only a proxy for impact and fail to capture other dimensions like award recognition, public attention, and technical translation; (2) Existing datasets often only cover computer science and biomedicine, lacking interdisciplinary coverage; (3) No unified benchmark exists for systematic multi-dimensional and multi-field comparisons.

**Key Challenge**: Scientific impact is multi-dimensional, but evaluation benchmarks remain single-dimensional.

**Goal**: Build a unified prediction benchmark covering 7 impact dimensions and 19 academic disciplines.

**Key Insight**: Model impact prediction as a comparative pair classification (given two papers/artifacts, determine which has more impact) by integrating heterogeneous data sources (OpenAlex, Papers with Code, HuggingFace, SciSciNet).

**Core Idea**: Jointly train on all dimensions through multi-task instruction fine-tuning to enable small models to surpass large models in multi-dimensional impact prediction.

## Method

### Overall Architecture

SciImpact construction involves three stages: (1) Candidate retrieval—retrieving papers and artifacts from various sources; (2) Impact labeling and pair generation—constructing meaningful comparison pairs based on dimension-specific rules; (3) Filtering and quality control—ensuring text integrity and disciplinary balance.

### Key Designs

1. **Seven-Dimensional Impact System**:

    - Function: Comprehensively covers various aspects of academic impact.
    - Mechanism: Citations (academic citation count), Awards (Best Paper Awards/Nobel Prizes/MDPI awards), Patents (number of patent citations), Media (news and social media mentions), Code (GitHub stars), Datasets (HuggingFace downloads), Models (HuggingFace downloads).
    - Design Motivation: Different dimensions reflect different types of impact—academic impact (citations), honorary recognition (awards), technological translation (patents), public attention (media), and practical adoption (code/data/models).

2. **Comparative Pair Construction Rules**:

    - Function: Ensures comparative pairs reflect meaningful impact differences.
    - Mechanism: Count-based dimensions require both papers to exceed a minimum threshold (e.g., citations $\ge 10$) and a ratio $\ge 2$; the award dimension uses binary comparison (award-winning vs. non-award-winning papers from the same venue). Constraints like same year/same venue/same author ensure comparability.
    - Design Motivation: To avoid trivial comparisons (e.g., 0 vs. 100 citations) and incomparable comparisons (e.g., citation counts of papers from different years).

3. **Multi-Task Instruction Fine-Tuning**:

    - Function: Trains a unified impact prediction model.
    - Mechanism: Aggregates training data from all dimensions, uses instruction formats to represent different prediction tasks uniformly, and fine-tunes on Qwen3-4B and LLaMA-3.2-3B.
    - Design Motivation: Possible transfer learning effects exist between dimensions; joint training is more efficient than dimension-specific training.

### Loss & Training

Standard Supervised Fine-Tuning (SFT) with cross-entropy loss. Evaluation uses binary classification accuracy.

## Key Experimental Results

### Main Results

| Model | Citations | Awards | Patents | Media | Code | Datasets | Models | Average |
|------|------|------|------|------|------|--------|------|------|
| o4-mini | Mid | Mid | Mid | Mid | Mid | Mid | Mid | ~65% |
| Qwen3-4B (Original) | Low | Low | Low | Low | Low | Low | Low | ~55% |
| **SFT-Qwen3-4B** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **Highest** |

### Ablation Study

| Analysis Dimension | Result |
|----------|------|
| Single-task vs. Multi-task | Multi-task consistently outperforms single-task |
| Model Scale | 4B SFT > 30B Zero-shot |
| Inter-dimensional Difficulty | Predicting awards and model downloads is most difficult |

### Key Findings

- Off-the-shelf LLMs vary significantly in scientific impact prediction performance and lack consistency across dimensions.
- Multi-task SFT consistently improves all dimensions, with the 4B model surpassing o4-mini.
- Award prediction is the most difficult dimension—because award decisions involve non-content factors such as politics and networking.

## Highlights & Insights

- Expanding scientific impact from a single citation count to seven dimensions is a significant conceptual contribution.
- The effectiveness of multi-task fine-tuning suggests transferable patterns exist across different impact dimensions.
- Coverage across 19 disciplines provides a foundation for interdisciplinary comparative research.

## Limitations & Future Work

- Construction of comparative pairs depends on available metadata, leading to uneven data coverage.
- Predictions are based solely on text content and do not utilize graph structural information such as citation networks.
- Impact changes over time; the current benchmark is a static snapshot.

## Related Work & Insights

- **vs. SciSciNet**: SciSciNet is a data lake, while SciImpact is an evaluation benchmark; the two are complementary.
- **vs. Citation Prediction Work**: This work extends the prediction range from citation counts to seven dimensions.

## Rating

- Novelty: ⭐⭐⭐⭐ Significant conceptual contribution with a multi-dimensional impact prediction benchmark.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage with 11 models, 7 dimensions, and 19 fields.
- Writing Quality: ⭐⭐⭐⭐ Data construction process is clear and transparent.
- Value: ⭐⭐⭐⭐ Provides a standardized evaluation tool for scientometrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] K-MetBench: A Multi-Dimensional Benchmark for Fine-Grained Evaluation of Expert Reasoning, Locality, and Multimodality in Meteorology](k-metbench_a_multi-dimensional_benchmark_for_fine-grained_evaluation_of_expert_r.md)
- [\[ACL 2026\] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)
- [\[ACL 2026\] DiningBench: A Hierarchical Multi-view Benchmark for Perception and Reasoning in the Dietary Domain](diningbench_a_hierarchical_multi-view_benchmark_for_perception_and_reasoning_in_.md)

</div>

<!-- RELATED:END -->
