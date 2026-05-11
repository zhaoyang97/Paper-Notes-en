---
title: >-
  [Paper Note] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning
description: >-
  [ACL 2026][Medical Imaging][Clinical Diagnostic Reasoning] This paper proposes the Clinical Diagnostic Reasoning Data (CDRD) structure to capture abstract clinical reasoning logic from symptoms to differential diagnosis.…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Clinical Diagnostic Reasoning"
  - "Reinforcement Learning"
  - "Structured Data"
  - "Inquiry Guidance"
  - "CDSS"
date: 2026-05-08
content_hash: c0aec830033facb4
---

# Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning

**Conference**: ACL 2026
**arXiv**: [2601.13690](https://arxiv.org/abs/2601.13690)
**Code**: [GitHub](https://github.com/YGswu/Dr.-Assistant)
**Area**: Medical Imaging
**Keywords**: Clinical Diagnostic Reasoning, Reinforcement Learning, Structured Data, Inquiry Guidance, CDSS

## TL;DR

This paper proposes the Clinical Diagnostic Reasoning Data (CDRD) structure to capture abstract clinical reasoning logic from symptoms to differential diagnosis. Based on CDRD, a two-stage SFT+RL training pipeline is employed to build the Dr. Assistant model (14B), which surpasses HuatuoGPT-o1-72B by 13.59% in ICD-Recall on clinical inquiry benchmarks, reaching a level competitive with GPT-5.

## Background & Motivation

**Background**: Clinical Decision Support Systems (CDSS) provide physicians with reasoning and inquiry guidance. LLMs have been widely applied in medical consultation owing to their broad medical knowledge, demonstrating strong performance on medical benchmarks.

**Limitations of Prior Work**: (1) Traditional CDSS rely on structured knowledge bases and rule-based algorithms, incurring high development and maintenance costs with poor adaptability; (2) Existing medical LLMs (e.g., Baichuan-M2, HuatuoGPT-o1) primarily optimize the patient consultation experience and lack professional clinical diagnostic reasoning and inquiry skills; (3) Diagnostic reasoning logic embedded in clinical guidelines is scattered across different chapters, making it difficult to use directly for training; (4) Even with high-quality data, training models to master clinical inquiry skills remains a significant challenge.

**Key Challenge**: LLMs possess broad medical knowledge but lack systematic clinical diagnostic reasoning logic—they cannot perform structured symptom analysis and differential diagnosis like experienced clinicians under zero-shot prompting.

**Goal**: (1) Design the CDRD data structure to capture abstract diagnostic reasoning logic; (2) Build the Dr. Assistant model equipped with diagnostic reasoning and inquiry skills; (3) Construct an evaluation benchmark for clinical diagnostic reasoning and inquiry.

**Key Insight**: Structured diagnostic reasoning logic (CDRD) is first extracted from clinical guidelines and then used as seeds to synthesize SFT and RL training data. A two-stage training procedure enables the model to internalize clinical reasoning capabilities.

**Core Idea**: Clinical diagnostic reasoning can be abstracted as structured triples of (core symptom, diagnostic evidence, differential diagnosis). These triples serve as seeds for generating training data, and an RL reward function incorporating a "logical deviation penalty" constrains the model's reasoning behavior.

## Method

### Overall Architecture

CDRD construction pipeline (LLM + physician collaboration in three stages: symptom extraction → disease matching → logic completion) → Data synthesis (CDRD → QA pairs for SFT; CDRD → multi-turn inquiry dialogues for RL) → Dr. Assistant two-stage training (SFT for memorizing reasoning logic + RL for reinforcing inquiry skills).

### Key Designs

1. **CDRD Data Structure and Construction Pipeline**:

    - Function: Extract abstract diagnostic reasoning logic from clinical guidelines.
    - Mechanism: CDRD is defined as a triple $\mathcal{C} = (\mathcal{S}, \mathcal{E}, \mathcal{D})$—core symptom $\mathcal{S}$ (e.g., headache), diagnostic evidence $\mathcal{E}$ (associated symptoms, examinations, and laboratory results), and differential diagnosis $\mathcal{D}$ (candidate diseases with their clinical manifestations and required examinations). The three-stage construction proceeds as follows: LLM extracts candidate symptoms → physicians refine and standardize → LLM matches diseases → physicians validate → LLM completes reasoning logic → physicians review.
    - Design Motivation: Diagnostic reasoning logic in clinical guidelines is scattered across chapters. CDRD reorganizes it into symptom-driven differential diagnosis pathways, with physician review at each stage to ensure reliability.

2. **Two-Stage Training Strategy (SFT + RL)**:

    - Function: Enable the model to first memorize reasoning logic and then reinforce inquiry skills through practice.
    - Mechanism: Stage 1 applies SFT using CDRD-generated QA pairs to instill preliminary diagnostic reasoning logic. Stage 2 applies RL using CDRD-generated multi-turn inquiry dialogues (dual-agent simulation: physician agent + patient agent), with a reward function comprising two components: clinical reasoning and inquiry skill scores, and CDRD logical fidelity (penalizing logical deviation from CDRD).
    - Design Motivation: SFT alone is insufficient for the model to flexibly apply reasoning logic in dynamic multi-turn inquiry. The logical deviation penalty in RL constrains the model from straying from correct diagnostic reasoning paths during free exploration.

3. **Structured Reasoning–Inquiry Template**:

    - Function: Formalize the reasoning process of each inquiry turn into six steps.
    - Mechanism: Known information → user intent → information already provided → diagnostic hypothesis → information yet to be collected → response strategy → inquiry/diagnosis output. This template ensures that each reasoning step is evidence-grounded.
    - Design Motivation: Unstructured inquiry tends to overlook critical information or produce unjustified reasoning leaps.

### Loss & Training

SFT stage: standard cross-entropy loss. RL stage: reward function = clinical reasoning and inquiry skill score (evaluated by LLM on coverage, accuracy, and inquiry coherence) + CDRD logical fidelity (penalizing deviation from CDRD-defined reasoning standards). The backbone model has 14B parameters.

## Key Experimental Results

### Main Results

**Diagnostic Reasoning Evaluation (242 real clinical cases across 8 secondary clinical departments)**

| Model | Parameters | ICD-Recall ↑ | Overall Score |
|-------|------------|-------------|---------------|
| HuatuoGPT-o1 | 72B | Baseline | — |
| GPT-5 | — | High | Competitive |
| **Dr. Assistant** | **14B** | **+13.59%** | **Competitive with GPT-5** |

### Ablation Study

| Configuration | ICD-Recall | Inquiry Quality |
|---------------|-----------|-----------------|
| SFT only | Baseline | Moderate |
| SFT + RL (w/o logical penalty) | Improved | Improved but with logical deviations |
| SFT + RL (full reward) | **Highest** | **Highest** |

### Key Findings

- Dr. Assistant (14B) outperforms HuatuoGPT-o1 (72B) with a 13.59% gain in ICD-Recall, demonstrating that specialized diagnostic reasoning training is more important than model scale.
- The CDRD logical fidelity penalty in RL is critical—without it, the model tends to produce seemingly plausible but logically unsound reasoning.
- The structured reasoning template ensures that each inquiry turn is evidence-grounded, improving the systematicity and completeness of the inquiry process.
- Dr. Assistant achieves a level competitive with GPT-5, providing a viable solution for practical CDSS deployment.

## Highlights & Insights

- The CDRD data structure serves as a general clinical knowledge representation scheme that is extensible to a broader range of clinical guidelines.
- The LLM + physician collaborative data construction pipeline balances efficiency and reliability.
- The "logical deviation penalty" in the RL reward function ensures that the model's free exploration does not deviate from the correct clinical reasoning trajectory.

## Limitations & Future Work

- CDRD construction is currently based solely on internal medicine clinical guidelines, limiting departmental coverage.
- The evaluation benchmark is relatively small (242 cases, 147 inquiry turns), constraining statistical power.
- No prospective evaluation has been conducted in real clinical environments.
- Tuning the weights of the RL reward function may require domain expert involvement.

## Related Work & Insights

- **vs. Baichuan-M2 / HuatuoGPT-o1**: These models optimize general medical consultation experience, whereas this work focuses on the specialization of clinical diagnostic reasoning and inquiry skills.
- **vs. Traditional CDSS**: Traditional systems rely on rules that are difficult to scale; Dr. Assistant achieves flexible adaptation through LLMs combined with structured reasoning data.
- **vs. Doctor-R1**: Doctor-R1 emphasizes the reasoning process, while this work places greater emphasis on the structuring of diagnostic reasoning logic and the development of inquiry skills.

## Rating

- Novelty: ⭐⭐⭐⭐ The CDRD data structure and the logical deviation penalty in RL are novel designs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons are comprehensive, but the evaluation scale is limited.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly and systematically presented, with precise clinical problem formulation.
- Value: ⭐⭐⭐⭐⭐ Provides an effective LLM-based solution for practical CDSS deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[NeurIPS 2025\] CXReasonBench: A Benchmark for Evaluating Structured Diagnostic Reasoning in Chest X-rays](../../NeurIPS2025/medical_imaging/cxreasonbench_a_benchmark_for_evaluating_structured_diagnostic_reasoning_in_ches.md)
- [\[ACL 2026\] Inflated Excellence or True Performance? Rethinking Medical Diagnostic Benchmarks with Dynamic Evaluation](inflated_excellence_or_true_performance_rethinking_medical_diagnostic_benchmarks.md)

</div>

<!-- RELATED:END -->
