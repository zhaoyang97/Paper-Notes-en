---
title: >-
  [Paper Note] On the Out-of-Distribution Generalization of Multimodal Large Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Multimodal Large Language Models] This paper systematically evaluates the out-of-distribution (OOD) generalization capabilities of 14 MLLMs across 20 datasets, finding that MLLMs perform near-randomly on domain-specific data such as medical and molecular imaging. Through a three-hypothesis analysis, "semantic-visual mapping deficits" are identified as the primary cause. Additionally, the study demonstrates that In-Context Learning (ICL) significant…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "Out-of-Distribution Generalization"
  - "In-Context Learning"
  - "Domain-Specific Tasks"
  - "Mapping Deficits"
date: 2026-05-08
content_hash: 1922604d6ad96e14
---

# On the Out-of-Distribution Generalization of Multimodal Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2402.06599](https://arxiv.org/abs/2402.06599)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models, Out-of-Distribution Generalization, In-Context Learning, Domain-Specific Tasks, Mapping Deficits

## TL;DR

This paper systematically evaluates the out-of-distribution (OOD) generalization capabilities of 14 MLLMs across 20 datasets, finding that MLLMs perform near-randomly on domain-specific data such as medical and molecular imaging. Through a three-hypothesis analysis, "semantic-visual mapping deficits" are identified as the primary cause. Additionally, the study demonstrates that In-Context Learning (ICL) significantly mitigates this issue but remains sensitive to label shifts and spurious correlation shifts.

## Background & Motivation

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in general visual understanding tasks. Models such as GPT-4V, Gemini, and LLaVA have achieved excellent performance on benchmarks like common object recognition and multimodal reasoning. However, does this superior performance on public benchmarks truly reflect the generalization capabilities of these models?

Existing research lacks a systematic evaluation of MLLMs in out-of-distribution (OOD) scenarios. Some sporadic case studies suggest that GPT-4V is prone to erroneous outputs when encountering distribution shifts, but in-depth analysis is lacking. **Where are the boundaries of MLLMs' generalization capabilities? What is the root cause of generalization failure? How can it be improved?** These key questions have not been fully answered.

The motivation of this paper is to: (1) comprehensively delineate the boundaries of MLLM generalization capabilities, (2) systematically analyze the causes of generalization failure, and (3) explore feasible mitigation schemes. By comparing different scenarios, including synthetic images, natural distribution shifts, medical imaging, and molecular images, the vulnerability of MLLMs outside the training data distribution is revealed.

## Method

### Overall Architecture

This paper is an evaluation and analysis study consisting of three progressive steps:
1. **Zero-shot generalization evaluation**: Evaluating 14 MLLMs on three categories of data: synthetic, natural shifts, and domain-specific.
2. **Failure analysis**: Proposing and testing three hypotheses to explain the reasons for generalization failure.
3. **ICL generalization exploration**: Investigating the effectiveness of In-Context Learning in mitigating mapping deficits and its robustness under distribution shifts.

### Key Designs

1. **Three-Hypothesis Failure Analysis Framework**:
    - Function: Systematically diagnose the root cause of MLLM generalization failures.
    - Mechanism: Propose three potential causes—(a) Semantic misunderstanding: MLLMs cannot comprehend domain-specific scientific concepts; (b) Insufficient visual feature extraction: High dimensionality or complex features of the data exceed the capability of model encoding; (c) Mapping deficits: Insufficient training data leads to imperfect mapping between semantic and visual features.
    - Design Motivation: By conducting experiments that control each factor individually, the bottleneck can be precisely pinpointed.

    **Exclusion of Hypothesis (a)**: Enhanced prompts containing domain instructions, expert guidance, and detailed category explanations were designed. The results showed almost no improvement (e.g., GPT-4V performance on HAM10000 dropped from 53% to 29.1%), ruling out semantic understanding as the primary bottleneck.

    **Exclusion of Hypothesis (b)**: Using CLIP as a feature extractor combined with a linear probing classifier yielded performance far superior to zero-shot MLLMs on the same dataset (e.g., COVID-CT: 83% vs. GPT-4V 43.2%). Since CLIP's visual feature extraction capability is weaker than that of most MLLMs, this indicates that visual encoding is not the bottleneck.

    **Confirmation of Hypothesis (c) as the Primary Cause**: After excluding the first two hypotheses, mapping deficits (namely, the lack of domain data leading to insufficient mapping from semantics to vision) were confirmed as the primary barrier.

2. **Scaling Law Analysis of Zero-Shot Generalization**:
    - Function: Evaluate whether scaling up model size can improve OOD generalization.
    - Mechanism: Test zero-shot performance on 5 OOD datasets using CLIP models with varying ViT sizes.
    - Design Motivation: If scaling up consistently improves OOD performance, the problem could be solved via scaling.
    - **Key Findings**: Unlike typical scaling laws observed in ID (In-Distribution) tasks, OOD generalization performance does not consistently improve with increased model size, and even shows a declining trend on TerraInc and SVIRO. This indicates that simply scaling up cannot solve the OOD generalization problem.

3. **In-Context Learning (ICL) Generalization Exploration**:
    - Function: Investigate ICL as a low-cost solution to bridge mapping deficits.
    - Mechanism: Systematically evaluate the efficacy of ICL under three distribution relationships: (a) In-Context Examples (ICE) from the target distribution (ideal case), (b) ICE with a domain shift from the test data, and (c) ICE with label or spurious correlation shifts from the test data.
    - Design Motivation: ICL does not require updating model parameters; instead, it adapts models to new tasks by adding examples to the input.

    **ICL Experimental Setup**: The dataset domains and categories are evenly divided into two groups to ensure class balance. The number of ICE is systematically varied (0, 2, 4, 8), with ICE uniformly sampled across categories.

### Loss & Training

Since this is an evaluation study, no model training is involved. Evaluation strategies include:
- Zero-shot evaluation: Directly evaluated using carefully designed prompt templates.
- Enhanced prompt evaluation: Prompts containing domain contextual information.
- Linear probing: Freezing the CLIP visual encoder and training a linear classification head.
- ICL evaluation: Adding varying numbers and distributions of examples to the input.

## Key Experimental Results

### Main Results

Zero-shot Generalization — Common OOD Datasets (11 indeed? Actually the text says 11, let's keep the table as is):

| Model | PACS | VLCS | DomainNet | iWildCam | Average |
|------|------|------|-----------|----------|------|
| GPT-4V | 96.9% | 87.2% | 74.8% | 52.3% | 69.1% |
| Gemini | 98.7% | 83.2% | 75.3% | 68.2% | 76.9% |
| LLaVA | 98.0% | 97.5% | 48.0% | 5.4% | 64.4% |

Zero-shot Generalization — Domain-Specific Datasets (9):

| Model | Camelyon17 | HAM10000 | NIH-Chest | DrugOOD_A | Average |
|------|-----------|----------|-----------|-----------|------|
| GPT-4V | 46.2% | 53.0% | 5.7% | 42.1% | 38.3% |
| Gemini | 50.2% | 41.0% | 7.8% | 52.2% | 43.3% |
| CLIP(LP) | 50.2% | **84.0%** | **74.0%** | **76.0%** | - |

ICL Enhancement Effect (GPT-4V, from 0 to 8 examples):

| Dataset | 0-shot | 2-shot | 8-shot | Gain |
|--------|--------|--------|--------|------|
| iWildCam | 63.4% | 78.0% | ~100% | +36.6% |
| HAM10000 | 53.9% | 66.4% | 74.2% | +20.3% |
| Camelyon17 | 48.4% | 52.0% | 57.4% | +9.0% |
| CT-XCOV | 44.3% | - | - | Stable |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Standard Prompt vs. Enhanced Prompt | GPT-4V HAM10000: 53%→29.1% | Enhanced prompts can actually be harmful |
| CLIP Zero-shot vs. CLIP Linear Probing | HAM10000: 21.9%→84.0% | Proves that visual features are adequate |
| ViT-B/16 → ViT-L/14 | TerraInc: Downward trend | OOD does not follow standard scaling laws |
| ICL Target Domain vs. ICL Domain Shift | Domain-shifted ICL still yields significant gains | Mapping deficits can be partially mitigated through related domain examples |

### Key Findings

- **Duality of MLLM Generalization**: Excellent performance is observed on common datasets like PACS ($>96\%$) and VLCS ($>80\%$), but performance drops to near random guessing (e.g., $\sim 50\%$ on binary classification tasks) on medical/molecular data.
- **Mapping Deficits as the Primary Cause**: After ruling out semantic misunderstanding and insufficient visual extraction, the lack of domain mapping knowledge in the training data is confirmed as the core reason for generalization failure.
- **Scale Cannot Handle OOD**: OOD generalization does not follow typical scaling laws; larger models do not always yield better performance.
- **ICL is Effective but Limited**: ICL provides significant improvements (iWildCam $+36.6\%$), but remains ineffective on molecular datasets—ICL is insufficient to bridge domains requiring deep, specialized knowledge.
- **ICL is Sensitive to Distribution Shifts**: Label shift and spurious correlation shift can degrade ICL performance and lead to instability.
- **Inconsistent Error Patterns Among MLLMs**: The absence of shared systematic bias suggests that model ensembling might hold potential.

## Highlights & Insights

- **Comprehensive Evaluation Benchmark**: A systematic evaluation across 14 MLLMs and 20 datasets, offering one of the most comprehensive OOD generalization benchmarks in the field.
- **Hypothesis-Driven Analysis Paradigm**: Formulating and verifying hypotheses (proposal-exclusion-confirmation) to diagnose generalization failures provides a rigorous and convincing methodology.
- **Concept of Mapping Deficits**: Attributing generalization failures to missing "semantic-visual mappings" rather than inherent incapacity provides a clear direction for prospective research.
- **ICL as a Low-Cost Adaptation Scheme**: Demonstrates that MLLM performance in new domains can be significantly enhanced through a few in-context examples without micro-tuning.
- **Dethroning Scale Obsession**: The finding that OOD does not obey scaling laws serves as a reminder for paradigm approaches over-reliant on simply scaling up models.

## Limitations & Future Work

- The evaluation is limited by the context window sizes of API-accessible models (limiting the number of ICE).
- Focuses solely on image classification tasks, leaving complex visual reasoning under OOD scenarios unexamined.
- The number of domain-specific datasets (mainly medical and molecular) is relatively limited; whether the conclusions generalize to other specialized fields requires further validation.
- The current ICL strategy is basic (random sampling); more intelligent selection strategies for examples may yield greater improvements.
- Other adaptation paradigms, such as Retrieval-Augmented Generation (RAG), remain unexplored.
- Future directions include: optimal ICE selection strategies, domain-adaptive fine-tuning, multi-MLLM ensembles, and expanding the analysis to VQA and image captioning tasks.

## Related Work & Insights

- **vs. Existing MLLM Benchmarks (MMBench/SEED, etc.)**: Existing benchmarks mainly evaluate general capabilities, whereas this study focuses on OOD scenarios, demonstrating that high scores on general benchmarks do not equate to true generalization.
- **vs. Domain Generalization Methods (DG/OOD)**: Traditional OOD methods primarily target small models in a supervised setup, whereas this work evaluates zero-shot generalization of large models, proving scaling is not the sole solution.
- **vs. ICL for VLM**: Previous work on ICL mostly focuses on In-Distribution (ID) performance. This study is the first to systematically investigate the efficacy and vulnerability of ICL in OOD scenarios.
- **vs. GPT-4V Evaluation Works**: Earlier works mostly present case studies, whereas this study offers quantitative, systematic evaluations and causal analyses.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic evaluation of MLLM OOD generalization with hypothesis testing to locate the root cause.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 14 models and 20 datasets, paired with detailed failure and ICL analyses.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with a logically complete chain of hypothesis-evidence-conclusion.
- Value: ⭐⭐⭐⭐ Serves as a vital reference and offers design guidelines for deploying MLLMs in specialized real-world domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy](playing_the_fool_jailbreaking_llms_and_multimodal_llms_with_out-of-distribution_.md)
- [\[CVPR 2025\] COUNTS: Benchmarking Object Detectors and Multimodal Large Language Models under Distribution Shifts](counts_benchmarking_object_detectors_and_multimodal_large_language_models_under_.md)
- [\[ICML 2025\] LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models](../../ICML2025/multimodal_vlm/laion-c_an_out-of-distribution_benchmark_for_web-scale_vision_models.md)
- [\[ICCV 2025\] FA: Forced Prompt Learning of Vision-Language Models for Out-of-Distribution Detection](../../ICCV2025/multimodal_vlm/fa_forced_prompt_learning_of_vision-language_models_for_out-of-distribution_dete.md)
- [\[CVPR 2025\] Teaching Large Language Models to Regress Accurate Image Quality Scores Using Score Distribution](teaching_large_language_models_to_regress_accurate_image_quality_scores_using_sc.md)

</div>

<!-- RELATED:END -->
