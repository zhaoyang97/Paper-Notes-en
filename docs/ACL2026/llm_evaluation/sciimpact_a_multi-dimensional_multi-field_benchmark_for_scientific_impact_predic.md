---
title: >-
  [Paper Note] SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction
description: >-
  [ACL 2026][LLM Evaluation][scientific impact prediction] This paper introduces SciImpact — the first large-scale scientific impact prediction benchmark spanning 19 disciplines and 7 impact dimensions (citations, awards, patents, media, code, datasets, and models), comprising 215,928 contrastive paper pairs. Multi-task fine-tuning enables a 4B model to outperform large models such as o4-mini.
tags:
  - ACL 2026
  - LLM Evaluation
  - scientific impact prediction
  - multi-dimensional benchmark
  - citation prediction
  - academic awards
  - multi-task instruction fine-tuning
date: 2026-05-08
content_hash: df217a4670c9c3fc
---

# SciImpact: A Multi-Dimensional, Multi-Field Benchmark for Scientific Impact Prediction

**Conference**: ACL 2026
**arXiv**: [2604.17141](https://arxiv.org/abs/2604.17141)
**Code**: [Project Page](https://flypig23.github.io/sciimpact-homepage/)
**Area**: LLM Evaluation
**Keywords**: scientific impact prediction, multi-dimensional benchmark, citation prediction, academic awards, multi-task instruction fine-tuning

## TL;DR

This paper introduces SciImpact — the first large-scale scientific impact prediction benchmark spanning 19 disciplines and 7 impact dimensions (citations, awards, patents, media, code, datasets, and models), comprising 215,928 contrastive paper pairs. Multi-task fine-tuning enables a 4B model to outperform large models such as o4-mini.

## Background & Motivation

**Background**: Scientific literature is growing exponentially, necessitating automated methods for assessing and predicting research impact. Existing work has focused primarily on citation count prediction.

**Limitations of Prior Work**: (1) Citation counts are only one proxy for impact and fail to capture other dimensions such as award recognition, public attention, and technology transfer; (2) existing datasets typically cover only computer science and biomedicine, lacking cross-disciplinary breadth; (3) no unified benchmark exists to support systematic multi-dimensional, multi-field comparison.

**Key Challenge**: Scientific impact is inherently multi-dimensional, yet evaluation benchmarks remain single-dimensional.

**Goal**: Construct a unified prediction benchmark covering 7 impact dimensions across 19 academic disciplines.

**Key Insight**: Impact prediction is framed as contrastive pair classification (given two papers or artifacts, determine which has greater impact), integrating heterogeneous data sources (OpenAlex, Papers with Code, HuggingFace, SciSciNet).

**Core Idea**: Joint training across all dimensions via multi-task instruction fine-tuning enables small models to outperform large models on multi-dimensional impact prediction.

## Method

### Overall Architecture

SciImpact is constructed in three stages: (1) candidate retrieval — collecting papers and artifacts from various data sources; (2) impact annotation and contrastive pair generation — constructing meaningful pairs according to dimension-specific rules; (3) filtering and quality control — ensuring textual completeness and domain balance.

### Key Designs

1. **Seven-Dimensional Impact Framework**:

    - Function: Comprehensively covers all aspects of scientific impact.
    - Mechanism: Citations (academic citation counts), Awards (best paper awards / Nobel prizes / MDPI awards), Patents (patent citation counts), Media (news and social media mentions), Code (GitHub stars), Datasets (HuggingFace downloads), Models (HuggingFace downloads).
    - Design Motivation: Different dimensions reflect different types of impact — academic influence (citations), honorary recognition (awards), technology transfer (patents), public attention (media), and practical adoption (code/data/models).

2. **Contrastive Pair Construction Rules**:

    - Function: Ensure that contrastive pairs reflect meaningful differences in impact.
    - Mechanism: Count-based dimensions require both papers to exceed a minimum threshold (e.g., citations ≥ 10) with a ratio ≥ 2; the awards dimension uses binary contrast (award-winning vs. non-winning papers from the same venue). Constraints such as same year, same venue, and same author ensure comparability.
    - Design Motivation: Avoids trivial comparisons (e.g., 0 vs. 100 citations) and incomparable comparisons (e.g., citation counts across different publication years).

3. **Multi-Task Instruction Fine-Tuning**:

    - Function: Train a unified impact prediction model.
    - Mechanism: Training data from all dimensions are aggregated and represented in a unified instruction format; fine-tuning is performed on Qwen3-4B and LLaMA-3.2-3B.
    - Design Motivation: Transfer learning effects across dimensions may exist; joint training is more efficient than training separate models per dimension.

### Loss & Training

Standard supervised fine-tuning (SFT) with cross-entropy loss. Evaluation uses binary classification accuracy.

## Key Experimental Results

### Main Results

| Model | Citation | Award | Patent | Media | Code | Dataset | Model | Avg. |
|-------|----------|-------|--------|-------|------|---------|-------|------|
| o4-mini | Moderate | Moderate | Moderate | Moderate | Moderate | Moderate | Moderate | ~65% |
| Qwen3-4B (base) | Low | Low | Low | Low | Low | Low | Low | ~55% |
| **SFT-Qwen3-4B** | **High** | **High** | **High** | **High** | **High** | **High** | **High** | **Best** |

### Ablation Study

| Analysis Dimension | Result |
|--------------------|--------|
| Single-task vs. multi-task | Multi-task consistently outperforms single-task |
| Model scale | 4B SFT > 30B zero-shot |
| Difficulty across dimensions | Award and model download prediction are most difficult |

### Key Findings

- Off-the-shelf LLMs exhibit high variance and cross-dimensional inconsistency in scientific impact prediction.
- Multi-task SFT consistently improves performance across all dimensions, enabling a 4B model to surpass o4-mini.
- Award prediction is the most challenging dimension, as award decisions involve non-content factors such as politics and social connections.

## Highlights & Insights

- Expanding scientific impact from a single citation count to seven dimensions represents a significant conceptual contribution.
- The effectiveness of multi-task fine-tuning suggests the existence of transferable patterns across different impact dimensions.
- Coverage across 19 disciplines provides a foundation for cross-disciplinary comparative research.

## Limitations & Future Work

- Contrastive pair construction relies on available metadata, leading to uneven data coverage.
- Predictions are based solely on textual content, without leveraging graph-structured information such as citation networks.
- Impact evolves over time; the current benchmark represents a static snapshot.

## Related Work & Insights

- **vs. SciSciNet**: SciSciNet is a data lake; SciImpact is an evaluation benchmark — the two are complementary.
- **vs. citation prediction work**: This paper extends the prediction scope from citation counts to seven dimensions.

## Rating

- Novelty: ⭐⭐⭐⭐ Multi-dimensional impact prediction benchmark with significant conceptual contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 11 models, 7 dimensions, and 19 fields comprehensively.
- Writing Quality: ⭐⭐⭐⭐ Data construction process is clearly and transparently described.
- Value: ⭐⭐⭐⭐ Provides a standardized evaluation tool for scientometrics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MultiFileTest: A Multi-File-Level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms](multifiletest_a_multi-file-level_llm_unit_test_generation_benchmark_and_impact_o.md)
- [\[ACL 2026\] Modeling Multi-Dimensional Cognitive States in Large Language Models under Cognitive Crowding](modeling_multi-dimensional_cognitive_states_in_large_language_models_under_cogni.md)
- [\[ACL 2026\] SessionIntentBench: A Multi-Task Inter-Session Intention-Shift Modeling Benchmark](sessionintentbench_a_multi-task_inter-session_intention-shift_modeling_benchmark.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](towards_self-improving_error_diagnosis_in_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
