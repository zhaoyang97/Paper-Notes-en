---
title: >-
  [Paper Note] Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks
description: >-
  [ICLR 2026][vlm_reasoning][Two-Axis Evaluation] This paper identifies that existing medical VLM benchmarks focus only on classification accuracy, creating an "evaluation illusion." It proposes a "Breadth-Depth" two-axis evaluation framework and builds Neural-MedBench, a deep reasoning benchmark for neurology (120 multimodal cases, 200 reasoning tasks). Empirical res
tags:
  - ICLR 2026
  - vlm_reasoning
  - Two-Axis Evaluation
date: 2026-05-08
content_hash: 98c19baba01338cf
---
# Beyond Classification Accuracy: Neural-MedBench and the Need for Deeper Reasoning Benchmarks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=KKA59ai0x6](https://openreview.net/forum?id=KKA59ai0x6)  
**Code**: [https://neuromedbench.github.io/](https://neuromedbench.github.io/)  
**Area**: Multimodal Medical Reasoning Evaluation / VLM Benchmark  
**Keywords**: Medical VLM, Clinical Reasoning, Neurology, Evaluation Illusion, Two-Axis Evaluation  

## TL;DR
This paper identifies that existing medical VLM benchmarks focus only on classification accuracy, creating an "evaluation illusion." It proposes a "Breadth-Depth" two-axis evaluation framework and builds Neural-MedBench, a deep reasoning benchmark for neurology (120 multimodal cases, 200 reasoning tasks). Empirical results show that top models like GPT-5, Claude-4, and MedGemma fail collectively in deep reasoning, with failures primarily stemming from reasoning rather than perception.

## Background & Motivation
**Background**: In recent years, Vision-Language Models (VLMs) have achieved near-human or even super-human performance on standard medical benchmarks like MedMNIST v2 and MultiMedQA, creating an optimistic impression that "medical VLMs are nearing clinical readiness."

**Limitations of Prior Work**: Most existing benchmarks are limited to shallow classification tasks such as label prediction and image-text alignment, rarely touching upon multimodal synthesis, uncertainty resolution, and "clinical logical argumentation" required for real diagnosis. The authors term the phenomenon where "high scores on shallow benchmarks mask weaknesses in deep reasoning" as the **evaluation illusion**—models appear capable but systematically fail in high-stakes diagnostic reasoning.

**Key Challenge**: The "breadth" (data scale, population/disease coverage) and "depth" (reasoning fidelity) of a benchmark are dimensions that are **likely uncorrelated**. Recent works like DiagnosisArena and MetaMedQA suggest that evaluating only on breadth overestimates model capabilities, while model metacognition and self-correction on challenging cases remain weak. Increasing data volume alone cannot expose these flaws.

**Goal**: To explicitly operationalize the long-neglected "depth axis" by creating a "stress test" ruler specifically for clinical reasoning fidelity and empirically demonstrating the decoupling of breadth and depth.

**Core Idea**: **[Two-Axis Evaluation]** Instead of expanding benchmark scale, create a "small yet difficult" depth benchmark focusing on reasoning density over data volume. Following the OSCE (Objective Structured Clinical Examination) format, models are forced to integrate multi-sequence MRI, Electronic Health Records (EHR), and clinical narratives to provide reasoned diagnoses.

## Method

### Overall Architecture
The construction of Neural-MedBench follows a "funnel" pipeline: starting from a pool of 2000+ candidate cases, it undergoes multi-source aggregation, multi-stage expert screening, and model validation to yield 120 high-complexity neurology cases with 200 reasoning tasks. A "clinically calibrated hybrid scoring" pipeline (accuracy + semantic similarity + LLM grader) is used to evaluate a fleet of VLMs, anchored by human baselines.

```mermaid
flowchart LR
    A[Case Pool 2000+<br/>ADNI/OASIS/Radiopaedia/Reports] --> B[1. Screening: Multimodal Integrity]
    B --> C[2. Expert Curation<br/>2 Neurologists + 1 Neuroradiologist]
    C --> D[3. Ground-truth Annotation<br/>Diag/Diff/Lesion/Reasoning]
    D --> E[4. Consensus & Challenge Verification<br/>Baseline filtering of trivial cases]
    E --> F[120 Cases / 200 Tasks<br/>3 Task Families × 3 Levels]
    F --> G[Zero-shot Evaluation of 16 VLMs]
    G --> H[Hybrid Scoring: pass@k + BERTScore + LLM Grader]
    H --> I[Error Classification + Human Baseline]
```

### Key Designs

**1. Two-Axis Evaluation Framework: Separating "Depth" from Breadth.** The conceptual foundation decomposes medical AI evaluation into two orthogonal axes: the Breadth axis (large-scale datasets for statistical generalization) and the Depth axis (expert-curated complex cases for reasoning fidelity). The authors hypothesize that success on breadth does not imply competence on depth, using Neural-MedBench to validate this in clinical neurology.

**2. Funnel Curation Focused on Reasoning Density.** Rather than data volume, the focus is on "every case being worth reasoning about." Cases are filtered from 2000+ down to 120 through four stages: maintaining multimodal integrity (imaging + neuropsychological scores + history), expert curation by specialists for diagnostic complexity and educational value, and annotating ground-truth as **structured narratives** (diagnosis, differential, lesion characterization, explanatory reasoning) rather than single labels. Trivial cases are filtered out using baseline models.

**3. Hierarchical Design of 3 Task Families × 3 Difficulty Levels.** Tasks consist of Differential Diagnosis (ranked hypotheses with evidence), Lesion Identification (spatial multimodal reasoning), and Argumentation Generation (professional-level explanatory synthesis). Difficulty levels span from Level 1 (Direct Diagnosis via pattern recognition) to Level 3 (Iterative Diagnosis simulating multi-turn consultation and dynamic hypothesis updates).

**4. Clinically Calibrated Two-stage Hybrid Scoring.** To balance reliability and scalability, the framework uses a two-stage approach. **Stage 1: Grader Calibration**: An LLM grader (GPT-4o) is guided by neurology-specific rubrics and calibrated against independent expert ratings (achieving Pearson $r > 0.9$). **Stage 2: Automated Community Evaluation**: The calibrated grader is open-sourced with the benchmark. Metrics include pass@1/pass@5, BERTScore, and LLM reasoning scores, supplemented by manual error classification (Perceptual, Reasoning, Knowledge Gap, Grounding, or Hallucination).

## Key Experimental Results

### Main Results (Zero-shot pass@1 / pass@5)

| Category | Model | Direct Diag p@1 | Direct Diag p@5 | Complex p@1 | Complex p@5 | Multi-turn p@1 | Multi-turn p@5 |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Base | GPT-5 | **36.7** | 43.3 | **28.3** | 45.0 | **19.5** | 27.5 |
| Base | GPT-4o | 20.0 | 36.7 | 8.3 | 40.0 | 8.5 | 16.5 |
| Base | Gemini 2.5-Pro | 30.0 | **50.0** | 15.0 | 38.3 | 11.5 | 19.5 |
| General | Claude 4.0-Sonnet | 16.7 | 43.3 | 13.3 | 31.6 | 6.5 | 18.0 |
| Medical | MedGemma-27B | 30.0 | 36.7 | 18.3 | 38.3 | 10.5 | 15.5 |
| Human | Medical Students | 3.3 | — | 3.3 | — | 6.0 | — |
| Human | Senior Clinicians | **40.0** | — | **35.5** | — | 15.0 | — |

> Even on the simplest tasks, the strongest medical model (MedGemma) achieves only 30% pass@1, lagging 10% behind senior clinicians. In complex cases, senior clinicians (35.5%) nearly double the performance of MedGemma (18.3%).

### Error Analysis & Evaluation Efficiency

| Error Distribution (100 responses) | Percentage |
|---|:---:|
| **Reasoning Failure** | **51%** |
| Perceptual Failure | 27% |
| Others (Knowledge/Grounding/Hallucination) | 22% |

| Cost Comparison (GPT-4o) | Image Count | Token Cost | Pass Rate |
|---|:---:|:---:|:---:|
| GMAI-MMBench | 12K | \$30.00 | 53.96% |
| **Neural-MedBench** | **1K** | **\$2.50** | **9.67%** |

### Key Findings
- **Breadth ≠ Depth**: Models scoring high on breadth benchmarks (MMLU-Pro) plummet on Neural-MedBench, proving the axes are uncorrelated.
- **Cognitive vs. Perceptual Bottleneck**: 51% of errors are reasoning failures (correct features, wrong synthesis), nearly double the perceptual failures (27%).
- **Anchoring Bias**: Unlike medical students, VLMs show strong anchoring bias, failing to update initial hypotheses when presented with contradictory evidence in multi-turn dialogues.
- **Cost-Efficiency**: Neural-MedBench achieves a much lower pass rate (9.67%) at one-tenth the cost of existing benchmarks, demonstrating the value of "small yet difficult" designs.

## Highlights & Insights
- **Defining a Real Problem**: The "Evaluation Illusion" precisely captures the discrepancy between high leaderboard scores and clinical non-viability.
- **Methodology over Dataset**: The two-axis framework provides a evaluation philosophy—breadth for generalization and depth for fidelity—guiding future benchmark design.
- **Human Anchoring**: The low model scores are validated against senior clinicians (35-40%), confirming the benchmark measures difficulty rather than being overly restrictive.

## Limitations & Future Work
- **Scale & Scope**: Currently limited to 120 cases in neurology; cross-departmental generalization remains to be verified.
- **Scorer Dependency**: The LLM grader depends on GPT-4o; long-term reliability against model updates requires monitoring.
- **Zero-shot Constraint**: The study focuses on intrinsic reasoning without exploring the impact of CoT, tool-use, or RAG.
- **Diagnostic Focus**: The work identifies reasoning failures in diagnosis but does not address therapeutic planning.

## Related Work & Insights
- **Relation to Breadth Benchmarks**: Positions as a **complement** to MedMNIST, MultiMedQA, and OmniMedVQA.
- **Methodological Heritage**: Extends semantic metrics (BERTScore, RadGraph-F1) and LLM-as-a-Judge frameworks with clinical calibration.
- **Broad Impact**: The "small-yet-deep" paradigm is applicable to other domains (law, finance) where high benchmark scores often mask real-world reasoning failures.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Conceptualizing "Evaluation Illusion" and the two-axis framework is impactful; filling the multimodal neurology reasoning gap is significant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation of 16 models, dual human baselines, and five-category error analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear progression from concept to empirical evidence; strong narrative and terminology.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses clinical trustworthiness; open-source grader and benchmark provide immediate utility for the community.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ICLR 2026\] Thyme: Think Beyond Images](thyme_think_beyond_images.md)
- [\[ICLR 2026\] More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models](more_thought_less_accuracy_on_the_dual_nature_of_reasoning_in_vision-language_mo.md)
- [\[CVPR 2026\] ChartR: Evaluating Reasoning Accuracy and Robustness in Chart Question Answering](../../CVPR2026/vlm_reasoning/chartr_evaluating_reasoning_accuracy_and_robustness_in_chart_question_answering.md)
- [\[CVPR 2026\] See Further, Think Deeper: Advancing VLM's Reasoning Ability with Low-level Visual Cues and Reflection](../../CVPR2026/vlm_reasoning/see_further_think_deeper_advancing_vlms_reasoning_ability_with_low-level_visual_.md)
- [\[CVPR 2026\] Deeper Thought, Weaker Aim: Understanding and Mitigating Perceptual Impairment during Reasoning in Multimodal Large Language Models](../../CVPR2026/vlm_reasoning/deeper_thought_weaker_aim_understanding_and_mitigating_perceptual_impairment_dur.md)

</div>

<!-- RELATED:END -->
