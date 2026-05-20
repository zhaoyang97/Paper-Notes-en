---
title: >-
  [Paper Note] Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models
description: >-
  [AAAI 2026][Interpretability][LLM reliability] This paper proposes the Composite Reliability Score (CRS), which unifies calibration, robustness…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "LLM reliability"
  - "calibration"
  - "robustness"
  - "uncertainty quantification"
  - "composite metric"
date: 2026-05-08
content_hash: 204d40c472a856dd
---

# Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2512.24058](https://arxiv.org/abs/2512.24058)  
**Code**: [https://github.com/rohitsalla/CRS.git](https://github.com/rohitsalla/CRS.git)  
**Area**: Interpretability
**Keywords**: LLM reliability, calibration, robustness, uncertainty quantification, composite metric

## TL;DR
This paper proposes the Composite Reliability Score (CRS), which unifies calibration, robustness, and uncertainty quantification into a single interpretable metric. A systematic evaluation of 10 open-source LLMs across 5 QA datasets reveals that Mistral-8x22B achieves the highest overall reliability (CRS=0.81), and that model size does not directly determine reliability.

## Background & Motivation

**Background**: Open-source LLMs are increasingly deployed in high-stakes domains such as healthcare, law, and finance, yet their reliability remains uncertain—models frequently exhibit overconfidence, fragility to input perturbations, and a lack of clear uncertainty estimates.

**Limitations of Prior Work**: Existing evaluations are fragmented—focusing either on accuracy alone or on only one of calibration or robustness—and thus fail to comprehensively characterize deployment reliability.

**Key Challenge**: A single metric (e.g., ECE or accuracy) can mask weaknesses along other dimensions; a model with high accuracy but poor calibration may be more dangerous in high-stakes settings than one with slightly lower accuracy but better calibration.

**Goal**: Design a unified reliability measurement framework that simultaneously evaluates calibration, robustness, and uncertainty quantification.

**Key Insight**: Each of the three dimensions is normalized to $[0,1]$ and then combined via a weighted average—a simple yet effective approach for reflecting composite reliability.

**Core Idea**: $\text{CRS} = \alpha C + \beta R + \gamma U$, where $C$ = calibration score, $R$ = robustness score, $U$ = uncertainty quantification score.

## Method

### Overall Architecture
CRS consists of three pillars, each normalized to $[0,1]$ and combined with equal weights ($\alpha=\beta=\gamma=1/3$):

### Key Designs

1. **Calibration (C)**

    - **Function**: Measures how well model confidence aligns with actual accuracy.
    - **Mechanism**: Based on ECE, converted to a positive score: $C = \max(0, 1 - \text{ECE}_{model}/\text{ECE}_{max})$.
    - **Evaluation**: Baseline calibration plus two post-hoc calibration methods (temperature scaling, isotonic regression).

2. **Robustness (R)**

    - **Function**: Measures the model's ability to maintain performance under input perturbations.
    - **Mechanism**: Computes the relative performance drop under three perturbation types (5% typos, back-translation paraphrase, TextFooler adversarial attack): $R = 1 - \text{Avg Acc Drop}/\text{Avg Acc}_{clean}$.
    - **Design Motivation**: Using relative rather than absolute accuracy drop enables fair comparison across models with different baseline capabilities.

3. **Uncertainty Quantification (U)**

    - **Function**: Evaluates the model's ability to produce high uncertainty when making incorrect predictions.
    - **Mechanism**: Uncertainty is estimated via MC Dropout (10 forward passes) and a 3-model ensemble; AUROC is used to assess discrimination between correct and incorrect predictions, normalized as $U = (\text{AUROC} - 0.5)/0.5$. The better of the two methods is taken as the final $U$ score.

### CRS Interpretability Tiers
- CRS ≥ 0.8: High reliability — suitable for deployment with minimal supervision.
- 0.6–0.8: Moderate reliability — requires human oversight.
- < 0.6: Low reliability — unsuitable for safety-critical scenarios.

## Key Experimental Results

### Main Results (CRS Rankings)

| Model | Params | C | R | U | CRS | Tier |
|-------|--------|------|------|------|------|------|
| Mistral-8x22B | 22B | 0.91 | 0.78 | 0.73 | **0.81** | High |
| Qwen3-235B | 22B | 0.84 | 0.74 | 0.70 | 0.76 | Moderate |
| DeepSeek R1 | 27B | 0.87 | 0.76 | 0.63 | 0.75 | Moderate |
| Gemma 2 | 27B | 0.71 | 0.68 | 0.71 | 0.70 | Moderate |
| LLaMA-3-7B | 7B | 0.16 | 0.54 | 0.44 | 0.57 | Low |
| Falcon-7B | 7B | 0.00 | 0.51 | 0.41 | 0.52 | Low |

### Calibration Metrics (Baseline)

| Model | Avg ECE↓ | Avg Brier↓ | Avg NLL↓ |
|-------|---------|-----------|---------|
| Mistral-8x22B | **0.031** | **0.128** | **0.332** |
| Falcon-7B | 0.062 | 0.179 | 0.566 |

### Key Findings
- Only Mistral-8x22B reaches the "high reliability" tier (CRS ≥ 0.8); most models fall in the moderate range.
- **Model size does not determine reliability**: the 27B Gemma 2 achieves a CRS of only 0.70, while the 22B Mistral-8x22B reaches 0.81.
- DeepSeek R1 demonstrates strong calibration ($C=0.87$) but weak uncertainty quantification ($U=0.63$), illustrating that a single dimension cannot represent overall reliability.
- Sensitivity analysis shows that top and bottom model rankings remain stable across different weight configurations.
- LLaMA-3-7B exhibits severely poor calibration ($C=0.16$), a weakness that would be obscured by accuracy-only evaluation.

## Highlights & Insights
- The **three-dimensional unified metric** fills a gap in LLM reliability evaluation—providing substantially richer information than accuracy or ECE alone.
- The **tiered deployment recommendations** (High/Moderate/Low) are directly oriented toward practical applications, offering actionable guidance for practitioners.
- The paper reveals that calibration, robustness, and uncertainty quantification are not simply positively correlated—some models are well-calibrated yet fragile to perturbations, a trade-off that only a composite metric can expose.

## Limitations & Future Work
- **Normalization has inherent limitations**: the calibration score $C$ is anchored to the worst model's ECE in the evaluation set, making scores incomparable across different evaluation setups.
- Evaluation is limited to QA tasks and does not cover generation, summarization, or dialogue scenarios.
- MC Dropout and ensemble-based uncertainty estimation have limited applicability to architectures without dropout layers, as found in some LLMs.
- The equal weighting scheme ($1/3, 1/3, 1/3$) is simple, but the relative importance of the three dimensions should vary across application domains.
- Hallucination detection is not incorporated—despite the title's claim of going "Beyond Hallucinations," the framework does not directly measure hallucination rates.

## Related Work & Insights
- **vs. GLUE/SuperGLUE**: These benchmarks evaluate accuracy but not reliability; this work is complementary.
- **vs. single-dimension calibration studies (RestoreCalib, CogCalib, etc.)**: Prior work focuses on calibration alone, whereas this paper unifies three dimensions.

## Rating
- **Novelty**: ⭐⭐⭐ — The framework is intuitive but not complex; the three-dimension weighted average is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 10 models, 5 datasets, three perturbation types, and two uncertainty estimation methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Formulations are clear and the framework is interpretable.
- **Value**: ⭐⭐⭐⭐ — Addresses a gap in unified reliability evaluation with strong practical utility.

## Additional Notes
- The methodology and experimental design of this work offer useful reference points for related research.
- Future work could validate the generalizability and scalability of the approach across broader scenarios and larger model scales.
- Potential research value exists in combining this framework with recent related work (e.g., intersections with RL/MCTS or multimodal methods).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](../../NeurIPS2025/interpretability/auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[AAAI 2026\] HSKBenchmark: Modeling and Benchmarking Chinese Second Language Acquisition in Large Language Models through Curriculum Tuning](hskbenchmark_modeling_and_benchmarking_chinese_second_language_acquisition_in_la.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](../../ICLR2026/interpretability/beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)

</div>

<!-- RELATED:END -->
