---
title: >-
  [Paper Note] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning
description: >-
  [ACL 2026][Medical NLP][Clinical Diagnostic Reasoning] This paper proposes the Clinical Diagnostic Reasoning Data (CDRD) structure to capture the abstract clinical reasoning logic from symptoms to differential diagnosis.…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Clinical Diagnostic Reasoning"
  - "Reinforcement Learning"
  - "Structured Data"
  - "Inquiry Guidance"
  - "CDSS"
date: 2026-05-08
content_hash: 548062fd757133b1
---

# Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.13690](https://arxiv.org/abs/2601.13690)  
**Code**: [GitHub](https://github.com/YGswu/Dr.-Assistant)  
**Area**: Medical Imaging  
**Keywords**: Clinical Diagnostic Reasoning, Reinforcement Learning, Structured Data, Inquiry Guidance, CDSS

## TL;DR

This paper proposes the Clinical Diagnostic Reasoning Data (CDRD) structure to capture the abstract clinical reasoning logic from symptoms to differential diagnosis. Based on CDRD, the Dr. Assistant model (14B) is constructed through a two-stage SFT+RL training process. On clinical inquiry benchmarks, its ICD-Recall exceeds HuatuoGPT-o1-72B by 13.59%, achieving a performance level competitive with GPT-5.

## Background & Motivation

**Background**: Clinical Decision Support Systems (CDSS) provide reasoning and inquiry guidance for physicians. LLMs have been widely applied in medical consultation due to their extensive medical knowledge, performing excellently on medical benchmarks.

**Limitations of Prior Work**: (1) Traditional CDSS rely on structured knowledge bases and rule-based algorithms, which entail high development and maintenance costs and suffer from poor adaptability; (2) Existing medical LLMs (e.g., Baichuan-M2, HuatuoGPT-o1) primarily optimize the patient consultation experience, lacking professional clinical diagnostic reasoning and inquiry skills; (3) Diagnostic reasoning logic in clinical guidelines is scattered across different chapters, making it difficult to use directly for training; (4) Training models to master clinical inquiry skills remain a significant challenge even with high-quality data.

**Key Challenge**: LLMs possess extensive medical knowledge but lack systematic clinical diagnostic reasoning logic—failing to perform structured symptom analysis and differential diagnosis like experienced physicians under zero-shot prompting.

**Goal**: (1) Design the CDRD data structure to capture abstract diagnostic reasoning logic; (2) Construct the Dr. Assistant model equipped with diagnostic reasoning and inquiry skills; (3) Build a clinical diagnostic reasoning and inquiry evaluation benchmark.

**Key Insight**: Extracting structured diagnostic reasoning logic (CDRD) from clinical guidelines, then using CDRD as a seed to synthesize SFT and RL training data, enables the model to internalize clinical reasoning capabilities through two-stage training.

**Core Idea**: Clinical diagnostic reasoning can be abstracted into a structured triplet of (Core Symptom, Diagnostic Evidence, Differential Diagnosis). This structure serves as a seed to generate training data, followed by a Reinforcement Learning (RL) reward function containing "logic deviation penalties" to constrain the model's reasoning behavior.

## Method

### Overall Architecture

CDRD construction pipeline (three-stage LLM+physician collaboration: symptom extraction -> disease matching -> logic completion) -> Data synthesis (CDRD to QA pairs for SFT + CDRD to multi-turn inquiry dialogues for RL) -> Two-stage training of Dr. Assistant (SFT for memorizing reasoning logic + RL for reinforcing inquiry skills).

### Key Designs

1.  **CDRD Data Structure and Construction Pipeline**:

    - **Function**: Extracts abstract diagnostic reasoning logic from clinical guidelines.
    - **Mechanism**: Defines CDRD as a triplet $\mathcal{C} = (\mathcal{S}, \mathcal{E}, \mathcal{D})$—where $\mathcal{S}$ represents core symptoms (e.g., headache), $\mathcal{E}$ represents diagnostic evidence (related symptoms, examinations, or lab results), and $\mathcal{D}$ represents differential diagnosis (possible diseases with clinical manifestations and required tests). The construction involves three stages: LLM extracts candidate symptoms -> physicians refine and standardize -> LLM matches diseases -> physicians verify -> LLM completes reasoning logic -> physicians audit.
    - **Design Motivation**: Reasoning logic in clinical guidelines is scattered across chapters; CDRD reorganizes it into a differential diagnosis path starting from symptoms, with physician audits at each stage to ensure reliability.

2.  **Two-stage Training Strategy (SFT + RL)**:

    - **Function**: Enables the model to first memorize reasoning logic and then reinforce inquiry skills through practice.
    - **Mechanism**: Stage 1 uses QA pairs generated from CDRD for SFT to let the model memorize preliminary diagnostic reasoning logic. Stage 2 uses multi-turn inquiry dialogues generated from CDRD for RL (dual-agent simulation: doctor agent and patient agent), utilizing a reward function with two dimensions: clinical reasoning and inquiry skill score + CDRD logic fidelity (penalty for logic deviation from CDRD).
    - **Design Motivation**: SFT alone cannot enable models to flexibly use reasoning logic for dynamic multi-turn inquiries; the logic deviation penalty in RL constrains the model from drifting off the correct diagnostic reasoning path during exploration.

3.  **Structured Reasoning-Inquiry Template**:

    - **Function**: Structures the reasoning process of each inquiry round into six steps.
    - **Mechanism**: Known Information -> User Intent -> Information Provided -> Diagnostic Hypothesis -> Information to be Collected -> Response Strategy -> Inquiry/Diagnostic Output. This template ensures that every round of reasoning is evidence-based.
    - **Design Motivation**: Unstructured inquiries are prone to missing key information or making groundless leap reasoning.

### Loss & Training

SFT Stage: Standard cross-entropy loss. RL Stage: Reward Function = Clinical Reasoning and Inquiry Skill Score (evaluated by LLM for coverage, accuracy, and inquiry logicality) + CDRD Logic Fidelity (penalizing deviations from CDRD standard logic). The base model has 14B parameters.

## Key Experimental Results

### Main Results

**Diagnostic Reasoning Evaluation (242 real-world clinical cases, 8 secondary departments)**

| Model | Params | ICD-Recall ↑ | Composite Score |
| :--- | :--- | :--- | :--- |
| HuatuoGPT-o1 | 72B | Baseline | - |
| GPT-5 | - | High | Competitive Level |
| **Dr. Assistant** | **14B** | **+13.59%** | **Competitive with GPT-5** |

### Ablation Study

| Configuration | ICD-Recall | Inquiry Quality |
| :--- | :--- | :--- |
| SFT Only | Basic Level | Medium |
| SFT + RL (No Logic Penalty) | Improved | Improved but with logic deviations |
| SFT + RL (Full Reward) | **Highest** | **Highest** |

### Key Findings

- Dr. Assistant (14B) outperforms HuatuoGPT-o1 (72B) despite its smaller size, with a 13.59% improvement in ICD-Recall—demonstrating that specialized diagnostic reasoning training is more important than model scale.
- The CDRD logic fidelity penalty in RL is critical; without it, the model tends to produce reasoning that seems plausible but lacks logical rigor.
- The structured reasoning template ensures that each round of inquiry is grounded, enhancing the systematic nature and completeness of the process.
- Dr. Assistant achieves a performance level competitive with GPT-5, providing a feasible solution for the practical deployment of CDSS.

## Highlights & Insights

- The CDRD data structure serves as a universal clinical knowledge representation scheme that can be extended to more clinical guidelines.
- The LLM+physician collaborative data construction pipeline balances efficiency and reliability.
- The "logic deviation penalty" in the RL reward function ensures that the model's free exploration does not deviate from the correct clinical reasoning path.

## Limitations & Future Work

- Currently, CDRD is primarily built on clinical guidelines related to internal medicine, covering limited departments.
- The evaluation benchmark scale is relatively small (242 cases, 147 rounds of inquiry), which limits statistical power.
- Prospective evaluations in real-world clinical environments have not yet been conducted.
- Adjusting weights in the RL reward function might require the involvement of more domain experts.

## Related Work & Insights

- **vs Baichuan-M2/HuatuoGPT-o1**: These models optimize the general medical consultation experience, whereas this work focuses on the specialization of clinical diagnostic reasoning and inquiry skills.
- **vs Traditional CDSS**: Traditional systems depend on rules and are difficult to scale; Dr. Assistant achieves flexible adaptation through LLM and structured reasoning data.
- **vs Doctor-R1**: While Doctor-R1 emphasizes the reasoning process, this study focuses more on the structuralization of diagnostic reasoning logic and inquiry skills.

## Rating

- Novelty: ⭐⭐⭐⭐ The design of the CDRD data structure and logic deviation penalty in RL is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparisons are comprehensive, though the evaluation scale is limited.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clear and systematic, and clinical problems are precisely defined.
- Value: ⭐⭐⭐⭐⭐ Provides an effective LLM-based solution for practical CDSS deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning](from_answers_to_arguments_toward_trustworthy_clinical_diagnostic_reasoning_with_.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[ACL 2026\] CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning](cure-med_curriculum-informed_reinforcement_learning_for_multilingual_medical_rea.md)
- [\[ICML 2026\] MedCase-Structured: A Text-to-FHIR Dataset for Benchmarking Diagnostic Reasoning in Clinically Realistic EHR Settings](../../ICML2026/medical_nlp/medcase-structured_a_text-to-fhir_dataset_for_benchmarking_diagnostic_reasoning_.md)

</div>

<!-- RELATED:END -->
