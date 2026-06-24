---
title: >-
  [Paper Note] VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare
description: >-
  [ACL 2025][Medical LLM][Pluralistic alignment] This paper constructs VITAL, the first pluralistic alignment benchmark dataset for the healthcare domain, containing 13.1K value scenarios and 5.4K multiple-choice questions. Extensive evaluation of 8 LLMs demonstrates that existing pluralistic alignment techniques (especially ModPlural) perform poorly in medical scenarios, and simple prompting yields better results.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Pluralistic alignment"
  - "healthcare"
  - "LLM benchmark"
  - "value diversity"
  - "dataset"
date: 2026-05-08
content_hash: 56c008f846fc68ce
---

# VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare

**Conference**: ACL 2025  
**arXiv**: [2502.13775](https://arxiv.org/abs/2502.13775)  
**Code**: [https://github.com/anudeex/VITAL.git](https://github.com/anudeex/VITAL.git)  
**Area**: Medical NLP  
**Keywords**: Pluralistic alignment, healthcare, LLM benchmark, value diversity, dataset

## TL;DR
This paper constructs VITAL, the first pluralistic alignment benchmark dataset for the healthcare domain, containing 13.1K value scenarios and 5.4K multiple-choice questions. Extensive evaluation of 8 LLMs demonstrates that existing pluralistic alignment techniques (especially ModPlural) perform poorly in medical scenarios, and simple prompting yields better results.

## Background & Motivation

**Background**: LLM alignment techniques (e.g., RLHF, DPO) are increasingly mature, but typically model the "average" preference, neglecting value diversity across different cultures, demographics, and communities. Sorensen et al. (2024) proposed a pluralistic alignment framework, defining three modes: Overton (covering all diverse perspectives), Steerable (steering based on user-specified attributes), and Distributional (matching real-world distributions). Feng et al. (2024) proposed the ModPlural multi-LLM collaboration scheme.

**Limitations of Prior Work**: (1) Existing alignment datasets (such as OpinionQA, GlobalOpinionQA) are not focused on the healthcare domain; (2) Plurality is particularly critical in medical scenarios, where culture, religion, and personal values influence health decisions; (3) The effectiveness of existing pluralistic alignment techniques in the healthcare domain remains unverified.

**Key Challenge**: General-purpose pluralistic alignment techniques may not transfer to domain-specific settings. Misalignment in medical scenarios can lead to harmful health advice or belief homogenization.

**Goal**: (1) Construct the first pluralistic alignment benchmark for healthcare; (2) Systematically evaluate existing methods on this benchmark; (3) Explore future improvement directions.

**Key Insight**: Approaching the problem from the highly sensitive and controversial healthcare domain, the authors use surveys, opinion polls, and moral dilemmas to construct a pluralistic dataset.

**Core Idea**: The healthcare domain requires specialized pluralistic alignment benchmarks and methods, as general-purpose solutions show limited effectiveness here.

## Method

### Overall Architecture
Construct the VITAL dataset $\rightarrow$ Evaluate 8 LLMs using four alignment techniques (Vanilla, Prompting, MoE, ModPlural) $\rightarrow$ Analyze performance across three pluralistic modes (Overton, Steerable, Distributional).

### Key Designs

1. **Dataset Construction (VITAL)**:

    - **Function**: To construct a healthcare pluralistic alignment benchmark containing 13.1K value scenarios + 5.4K multiple-choice questions
    - **Mechanism**: Collecting multiple-choice questions from various surveys and moral datasets (such as OpinionQA, GlobalOpinionQA, and MoralChoice), and using few-shot classification via FLAN-T5 to filter out samples that are health-related, represent pluralistic viewpoints, and require actions
    - **Data Distribution**: The Overton mode contains 1,649 text samples, the Steerable mode contains 15,340 samples (text+QA), and the Distributional mode contains 1,857 QA samples
    - **Quality Validation**: Human annotation verified 10% of the samples, confirming 80% as health-related (Fleiss' Kappa: 0.49)

2. **Evaluation Techniques**:

    - **Vanilla**: Direct LLM output with no instructions
    - **Prompting**: Appending pluralistic instructions within the prompt
    - **MoE**: The primary LLM functions as a router to select the most relevant community LLM (perspective/culture LLM), feeding its response back to the primary LLM for final generation
    - **ModPlural**: The primary LLM collaborates with multiple community LLMs. In the Overton mode, it concatenates community messages to perform multi-document summarization; in the Steerable mode, it selects the most relevant community LLM; in the Distributional mode, it aggregates the community probability distributions

3. **Evaluation Metrics**:

    - **Overton**: Sentence-level entailment calculated using an NLI model for value coverage, augmented by human evaluation and GPT-as-Judge
    - **Steerable**: Accuracy (whether the final response adheres to the designated steering attribute)
    - **Distributional**: Jensen-Shannon Divergence (lower is better, indicating greater similarity to the ground-truth distribution)

### LLM Agent Experiments
The experiment investigated replacing fine-tuned community LLMs with lightweight LLM agents (role-playing agents based on Mistral-7B). A healthcare-specific agent pool was constructed, with GPT-4o selecting the 6 most relevant agents. The NLI coverage of the 6 agents was 44.16% (vs. 47.84% for original community LLMs), which increased to 49.37% when using 10 agents.

## Key Experimental Results

### Main Results: Overton Mode Coverage (%)

| Method | LLaMA2-7B | Gemma-7B | Qwen2.5-7B | LLaMA3-8B | ChatGPT | Average |
|------|-----------|----------|------------|-----------|---------|------|
| Aligned Vanilla | 20.76 | 38.60 | 32.41 | 18.93 | 26.70 | 26.10 |
| + Prompting | 22.88 | **40.61** | **34.42** | 27.41 | **32.22** | **30.46** |
| + MoE | 19.58 | 26.00 | 28.14 | 24.70 | 18.84 | 22.79 |
| + ModPlural | 15.38 | 22.18 | 22.30 | 24.51 | 18.06 | 20.09 |

### Ablation Study: Comparison of Community LLM Sources

| Configuration | LLaMA2-7B | LLaMA3-8B | Gemma-7B |
|------|-----------|-----------|----------|
| Perspective Community LLM | 15.15 | 23.82 | 22.37 |
| Culture Community LLM | 17.61 | 25.11 | 22.45 |
| Healthcare LLM as Primary Model | 12.00 (ModPlural) | - | - |
| Replacing Community LLMs with 6 LLM Agents | 44.16 (NLI) | - | - |
| 10 LLM Agents | 49.37 (NLI) | - | - |

### Key Findings
- **Prompting > ModPlural**: Across all 8 models and 3 alignment modes, simple prompting consistently outperformed the more complex ModPlural multi-LLM collaboration scheme, with the maximum performance gap reaching 55.5%.
- **Invariance to Model Scale**: Scaling up models did not yield consistent performance improvements.
- **NLI Evaluation Bias**: Overton coverage is positively correlated with the number of generated response sentences. ModPlural's summarization tends to compress multiple arguments into a single sentence, resulting in artificially lower NLI scores.
- **Inadequacy of Direct Domain-Specific Model Replacement**: Using a medical-specialized LLM (mental-llama2-7b) as the primary model yielded no substantial improvements, suggesting that simple domain "patching" is insufficient.
- **Slightly Comparable in Distributional Mode**: ModPlural performed reasonably well in the distributional mode, narrowing the gap with other methods.

## Highlights & Insights
- **Counter-Intuitive Finding**: Highly complex multi-LLM collaboration schemes (ModPlural, MoE) underperform simple prompting in healthcare scenarios, indicating that general pluralistic schemes may fail in domain-specific tasks. This highlights the vital importance of domain-specific design.
- **Feasibility of Replacing Community LLMs with Agents**: 10 lightweight agents outperformed fine-tuned community LLMs in coverage without requiring expensive fine-tuning, offering a highly scalable and dynamically expandable alternative.
- **Dataset Design**: The methodology of combining textual scenarios with multiple-choice questions to cover three pluralistic dimensions is highly transferable to other sensitive domains such as law and education.

## Limitations & Future Work
- Data construction heavily relies on FLAN-T5 filtering, and human verification only covered 10% of the sample with moderate agreement (Kappa 0.49), implying potential noise in data quality.
- The Overton evaluation leverages NLI models at the sentence level to judge entailment, which suffers from biases related to sentence count and semantic compression.
- State-of-the-art LLMs (e.g., GPT-4, LLaMA3-70B, and other larger models) were not evaluated.
- No novel alignment method was proposed; research was restricted to benchmark evaluation.

## Related Work & Insights
- **vs. ModPlural (Feng et al., 2024)**: ModPlural is the state-of-the-art pluralistic alignment method. This paper exposes its shortcomings in healthcare, despite its strong performance in general domains.
- **vs. OpinionQA (Santurkar et al., 2023)**: OpinionQA is a widely used benchmark dataset for alignment but does not focus on healthcare. VITAL addresses this critical gap.
- **vs. MoralChoice (Liu et al., 2024)**: MoralChoice offers moral scenarios but is not medical-specific; VITAL filters and extends a healthcare-related subset from it.

## Rating
- Novelty: ⭐⭐⭐⭐ The first healthcare pluralistic alignment benchmark, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across 8 models, 4 methods, and 3 modes for a comprehensive study, but lacks a proposed novel methodology.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed analysis.
- Value: ⭐⭐⭐⭐ Both the dataset and counter-intuitive findings provide valuable reference for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](../../NeurIPS2025/medical_nlp/healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)
- [\[ACL 2025\] Online Iterative Self-Alignment for Radiology Report Generation](oisa_radiology_report_gen.md)
- [\[ACL 2026\] IndicMedDialog: A Parallel Multi-Turn Medical Dialogue Dataset for Accessible Healthcare in Indic Languages](../../ACL2026/medical_nlp/indicmeddialog_a_parallel_multi-turn_medical_dialogue_dataset_for_accessible_hea.md)
- [\[ACL 2025\] AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset](afrimed_qa_pan_african.md)
- [\[ACL 2025\] CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation](clinidial_a_naturally_occurring_multimodal_dialogue_dataset_for_team_reflection_.md)

</div>

<!-- RELATED:END -->
