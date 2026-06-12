---
title: >-
  [Paper Note] PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions
description: >-
  [NeurIPS 2025][LLM Safety][patient simulator] This paper introduces PatientSim — an LLM-based patient simulator grounded in real MIMIC clinical data and a four-dimensional persona framework (personality…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "patient simulator"
  - "LLM role-playing"
  - "doctor-patient dialogue"
  - "persona modeling"
  - "clinical education"
date: 2026-05-08
content_hash: 58ed0081bef68f00
---

# PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions

**Conference**: NeurIPS 2025
**arXiv**: [2505.17818](https://arxiv.org/abs/2505.17818)  
**Authors**: Daeun Kyung, Hyunseung Chung, Seongsu Bae, Jiho Kim (KAIST), Jae Ho Sohn (UCSF), Taerim Kim (Samsung Medical Center), Soo Kyung Kim (Ewha Womans University), Edward Choi (KAIST)
**Code**: [GitHub](https://github.com/dek924/PatientSim)  
**Area**: Medical Imaging
**Keywords**: patient simulator, LLM role-playing, doctor-patient dialogue, persona modeling, clinical education

## TL;DR

This paper introduces PatientSim — an LLM-based patient simulator grounded in real MIMIC clinical data and a four-dimensional persona framework (personality, language proficiency, medical history recall, and cognitive confusion), generating 37 unique persona combinations. The system is evaluated across 8 LLMs for factual accuracy and persona fidelity, and validated by 4 clinical experts with a mean quality score of 3.89/4.

## Background & Motivation

### State of the Field
LLMs have surpassed human experts on medical QA benchmarks such as MedQA; however, these benchmarks are single-turn in nature and present patient data directly. In real clinical settings, physicians must actively gather patient information through multi-turn, context-aware conversations. Evaluating physician LLMs therefore requires realistic patient interaction systems. Traditional Standardized Patients rely on trained human actors, which is costly, poorly reproducible, and difficult to scale.

### Limitations of Prior Work
- Existing LLM patient simulators primarily focus on accurate symptom transmission while neglecting the behavioral diversity of real patients.
- Some studies attempt to enhance realism using Big Five personality traits or occupational keywords, but do not simultaneously *implement and evaluate* multi-dimensional, clinically relevant personas.
- Mental health counseling simulators emphasize emotional modeling but are ill-suited for general diagnostic scenarios such as emergency medicine.
- When faced with information not defined in the patient profile, existing approaches typically refuse to answer or assume normal values, limiting conversational naturalness.

### Root Cause
There is a need for an open-source simulator that jointly ensures clinical factual accuracy and diverse patient personas, providing a reproducible, scalable, and privacy-compliant evaluation platform for physician LLMs and medical education.

## Method

### Scope and Constraints
The study focuses on single-session, first-visit encounters in the emergency department (ED):
- **History-taking only**: Physical examination and laboratory results are excluded, as approximately 80% of diagnoses can be made through history-taking alone.
- **Single session**: Longitudinal treatment effects and disease progression are not modeled, to avoid generating misleading clinical conclusions.
- **Differential diagnosis**: Based solely on verbal information.

### Patient Profile Construction
Three real-world datasets are used: MIMIC-IV, MIMIC-IV-ED, and MIMIC-IV-Note:
- Structured tables provide precise data; clinical notes capture richer contextual information such as lifestyle and current symptoms.
- Each patient profile contains **24 entries** covering demographics, social and medical history, and ED visit details.
- A total of **170 profiles** are constructed across 5 high-prevalence diseases: myocardial infarction, pneumonia, urinary tract infection, bowel obstruction, and cerebral infarction.
- Disease selection criteria: high clinical prevalence, symptom profiles distinguishable through history-taking alone, and sufficient data availability in MIMIC-ED.

### Four-Dimensional Persona Definition
1. **Personality**: 6 types — impulsive, overly anxious, distrustful, overly optimistic, talkative, and neutral (baseline), designed based on literature review and guidance from medical experts.
2. **Language Proficiency**: 3 levels — A (basic), B (intermediate), C (advanced), integrated from the CEFR framework.
3. **Medical History Recall**: 2 levels — high recall and low recall.
4. **Cognitive Confusion**: 2 levels — high confusion and normal. High-confusion patients are restricted to neutral personality, intermediate language proficiency, and high recall to avoid dimensional overlap.

Combining these axes yields $6 \times 3 \times 2 = 36$ standard personas + 1 high-confusion persona = **37 unique personas**.

### Prompt Design
- **PatientSim prompt**: Incorporates profile information, the four-dimensional persona axes, and general behavioral guidelines; iteratively refined through LLM evaluation, qualitative analysis by the authors, and two rounds of expert feedback.
- **Physician LLM prompt**: Designed based on medical textbooks and expert recommendations to ensure coverage of all standard necessary questions.

### Evaluation Framework
Three research questions are investigated:
- **RQ1** (Persona Fidelity): Can LLMs naturally reflect 37 persona combinations? Assessed via automatic and human evaluation using a 5-criterion, 4-point scale.
- **RQ2** (Factual Accuracy): Dual-layer evaluation at sentence and dialogue levels, including NLI entailment/contradiction classification, Information Coverage (ICov), and Information Consistency (ICon).
- **RQ3** (Clinical Plausibility): Evaluates whether responses to profile-undefined information are clinically reasonable, using a 4-point rating scale.

## Key Experimental Results

### Experiment 1: Persona Fidelity Evaluation (RQ1)

Gemini-2.5-Flash is used as the evaluator to assess the persona-reflecting capabilities of 8 LLMs across 37 persona combinations.

| Model | Personality | Language | Recall | Confusion | Authenticity | Average |
|-------|-------------|----------|--------|-----------|--------------|---------|
| Gemini-2.5-Flash | 3.94 | 3.54 | 3.64 | 3.38 | 3.37 | 3.57 |
| GPT-4o mini | 3.58 | 3.55 | 3.78 | 3.88 | 3.26 | 3.61 |
| DeepSeek-R1-Distill-Llama-70B | 3.87 | 3.58 | 3.42 | 2.50 | 3.19 | 3.31 |
| Qwen2.5-72B | 3.30 | 3.68 | 3.63 | 3.50 | 3.22 | 3.46 |
| **Llama-3.3-70B** | **3.92** | 3.40 | **3.78** | **4.00** | 3.28 | **3.68** |
| Llama-3.1-70B | 3.65 | 3.51 | 3.62 | 4.00 | 3.23 | 3.60 |
| Llama-3.1-8B | 3.53 | 3.29 | 3.70 | 4.00 | 3.20 | 3.54 |
| Qwen2.5-7B | 3.23 | 3.49 | 3.31 | 3.50 | 3.16 | 3.34 |

The Llama series excels in affective expression (personality, confusion); general benchmark performance does not directly correspond to simulation fidelity.

### Experiment 2: Factual Accuracy and Clinical Plausibility (RQ2 & RQ3)

Sentence-level evaluation covers support/non-support classification, NLI entailment rate, and clinical plausibility scoring.

| Model | Info. Sent. % | Supported % | Unsupported % | Entailment | Contradiction | Plausibility |
|-------|--------------|-------------|---------------|------------|---------------|--------------|
| Gemini-2.5-Flash | 97.2% | 76.3% | 31.6% | 97.8% | 2.2% | 3.953 |
| GPT-4o mini | 95.7% | 72.1% | 42.8% | 96.8% | 3.2% | 3.929 |
| **Llama-3.3-70B** | 95.8% | **79.6%** | 38.7% | **98.1%** | **1.9%** | **3.963** |
| Llama-3.1-70B | 94.8% | 81.3% | 40.7% | 96.8% | 3.2% | 3.955 |
| Llama-3.1-8B | 94.4% | 77.1% | 48.8% | 94.4% | 5.6% | 3.897 |
| Qwen2.5-72B | 97.5% | 68.3% | 46.8% | 95.4% | 4.6% | 3.928 |
| Qwen2.5-7B | 98.7% | 70.3% | 45.3% | 93.9% | 6.1% | 3.862 |

Larger models (≥70B) consistently outperform smaller models (≤8B) in factual accuracy and plausibility; Llama-3.3-70B achieves the best entailment rate and plausibility among open-source models.

### Human Evaluation
Four clinical experts evaluate PatientSim based on Llama-3.3-70B:
- **Mean quality score of 3.89/4** across 6 evaluation criteria.
- Score of **3.75/4** on the criterion "This chatbot is useful for clinical education."
- Plausibility scores for unsupported sentences: 3.955, 3.923, 3.985, and 3.781 from the four experts respectively.
- Inter-rater agreement (Gwet's AC1): highest 0.968, lowest 0.853, indicating overall strong consistency.

## Highlights & Insights

- **Multi-dimensional persona modeling**: The first general-purpose doctor-patient simulator to systematically define and implement a 4-axis, 37-combination persona framework, substantially advancing beyond simple Big Five keyword descriptions.
- **Dual-layer factual evaluation framework**: Sentence-level NLI combined with dialogue-level coverage and consistency metrics provides a comprehensive methodology for evaluating patient simulators.
- **Strong clinical expert endorsement**: A mean quality score of 3.89/4 from 4 clinical experts, with high inter-rater agreement (AC1 > 0.85), validates the clinical reliability of the framework.
- **Open-ended generation strategy**: Allowing clinically plausible responses to profile-undefined information — rather than refusing to answer — enhances conversational naturalness, achieving a plausibility score of 3.91/4.
- **Open-source and reproducible**: Full code and an open-source implementation based on Llama-3.3-70B lower the barrier to adoption.

## Limitations & Future Work

- **Single data source**: Reliance solely on the MIMIC database may limit the generalizability of findings.
- **Text-only modality**: Non-verbal cues (facial expressions, body language) cannot be simulated, leaving persona representation incomplete.
- **Limited human evaluation scale**: Only 4 clinical experts participated, constraining the generalizability of the human evaluation results.
- **Narrow disease coverage**: Only 5 common ED conditions are included; broader clinical scenarios remain unaddressed.
- **Single-session constraint**: Longitudinal multi-visit simulations are not supported, precluding evaluation of treatment follow-up capabilities.
- **Language limitation**: Currently restricted to English; multilingual scenarios are not addressed.

## Related Work & Insights

- **Agent Hospital (Li et al., 2025)**: Simulates the entire hospital workflow (patients, nurses, physicians), focusing on final task accuracy; this work instead centers on the realism and diversity of patient simulation.
- **MedIQ (Li et al., 2024a)**: Evaluates the interactive inquiry capabilities of physician LLMs but does not assess the persona fidelity of the patient simulator itself.
- **Du et al. (2024)**: Simulates standardized patients via agent co-evolution; this work emphasizes systematic persona definition and multi-dimensional evaluation.
- **Mental health counseling simulators (Qiu et al., Wang et al.)**: Prioritize emotional and subjective response modeling, but are unsuitable for emergency diagnostic scenarios.
- **Fan et al. (2025)**: AI Hospital multi-agent benchmark focuses on benchmark evaluation rather than simulator realism.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The systematic design of a four-dimensional persona framework with 37 combinations is innovative, though the broader concept of LLM role-playing is not new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comparisons across 8 LLMs, validation by 4 clinical experts, and dual-layer sentence/dialogue-level evaluation are comprehensive; however, only 5 diseases and 170 profiles represent a limited scale.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, precise problem formulation, and detailed evaluation methodology.
- **Value**: ⭐⭐⭐⭐ — Open-source and reproducible, with practical application to medical AI education and training; strong clinical expert endorsement enhances credibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] From Single to Societal: Analyzing Persona-Induced Bias in Multi-Agent Interactions](../../AAAI2026/llm_safety/from_single_to_societal_analyzing_persona-induced_bias_in_multi-agent_interactio.md)
- [\[NeurIPS 2025\] TRUST -- Transformer-Driven U-Net for Sparse Target Recovery](trust_--_transformer-driven_u-net_for_sparse_target_recovery.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](../../ACL2026/llm_safety/subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](../../ICLR2026/llm_safety/ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](../../ACL2026/llm_safety/on_safety_risks_in_experience-driven_self-evolving_agents.md)

</div>

<!-- RELATED:END -->
