---
title: >-
  [Paper Note] CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation
description: >-
  [ACL 2025][Medical LLM][Clinical dialogue dataset] This work constructs CliniDial, a dataset collected from natural dialogues during simulated clinical operations, containing multimodal data like audio transcriptions, dual-angle videos, and patient physiological signals. Annotated with team reflection action coding, CliniDial reveals substantial shortcomings of state-of-the-art LLMs in handling class imbalance, natural conversational interactions…
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Clinical dialogue dataset"
  - "multimodal dialogue"
  - "team collaboration analysis"
  - "action coding"
  - "medical NLP"
date: 2026-05-08
content_hash: bb5e2bdec54c715c
---

# CliniDial: A Naturally Occurring Multimodal Dialogue Dataset for Team Reflection in Action During Clinical Operation

**Conference**: ACL 2025  
**arXiv**: [2506.12936](https://arxiv.org/abs/2506.12936)  
**Code**: [https://github.com/MichiganNLP/CliniDial](https://github.com/MichiganNLP/CliniDial)  
**Area**: Medical NLP  
**Keywords**: Clinical dialogue dataset, multimodal dialogue, team collaboration analysis, action coding, medical NLP  

## TL;DR

This work constructs CliniDial, a dataset collected from natural dialogues during simulated clinical operations, containing multimodal data like audio transcriptions, dual-angle videos, and patient physiological signals. Annotated with team reflection action coding, CliniDial reveals substantial shortcomings of state-of-the-art LLMs in handling class imbalance, natural conversational interactions, and domain-specific multimodal data.

## Background & Motivation

**Background**: Team collaboration in clinical operations directly impacts surgical outcomes. Approximately 250,000 preventable deaths annually in the United States are related to communication failures. Ineffective collaboration can lead to a 58% higher mortality rate than expected. Understanding team interaction patterns in the operating room is vital to improving medical safety.

**Limitations of Prior Work**: 
   - Existing NLP dialogue datasets are mostly collected in highly controlled environments (e.g., MultiWOZ typically contains only 30 turns per dialogue), failing to reflect the complex interactions in real clinical operations.
   - Existing multimodal datasets mainly focus on vision and text, lacking key physiological signals (vital signs).
   - Clinical operation data features severe label imbalance, ultra-long dialogues, and strong domain-specificity, making it difficult for existing methods to cope.

**Key Challenge**: Team interaction data in actual clinical settings is extremely difficult to obtain due to ethical and legal constraints, yet existing simulated or conventional dialogue datasets cannot reflect real interaction patterns during surgeries.

**Goal**: To build a natural multimodal dialogue dataset from simulated clinical operations and systematically evaluate the performance boundaries of existing LLMs on such real-world data.

**Key Insight**: Data is collected from simulated malignant hyperthermia (MH) surgical scenarios, focusing on the team interaction behaviors among three roles: anesthesiologists, support staff, and surgeons.

**Core Idea**: Multimodal dialogue in real clinical operations exhibits unique features (class imbalance, ultra-long natural interactions, domain-specific multimodality) that current LLMs are far from effectively handling.

## Method

### Overall Architecture

Rather than proposing a new model architecture, CliniDial focuses on:
1. Dataset construction: Multimodal dialogue data during simulated surgeries.
2. Action coding annotation: Seek / Evaluate / Implement / Plan + None.
3. Three case studies: Investigating the impacts of class imbalance, dialogue context, and multimodality, respectively.

### Key Designs

1. **Data Collection and Scenario Design**: 

    - 22 surgical simulation sessions, each lasting about 19 minutes.
    - Scenario: A 36-year-old female patient experiences malignant hyperthermia (MH) complications during minimally invasive surgery.
    - Roles: Surgeon (confederate), Anesthesiologist (trainee, primary decision-maker), and Support staff.
    - A total of 6.5k conversational turns and 49.9k words.
    - Average dialogue length of 311 turns (far exceeding the 30 turns in conventional dialogue datasets).

2. **Multimodal Data**: 

    - **Audio + Transcriptions**: Dialogue content of each role.
    - **Dual-angle Video**: Two cameras recording the surgery from different perspectives.
    - **9 Types of Physiological Signals**: Simulated physical data from the mannequin (heart rate, blood oxygen level, end-tidal CO2, etc.).
    - All modalities are timestamp-aligned.

3. **Action Coding Annotation**: 

    - Based on the team reflection behavioral framework by Schmutz et al. (2021).
    - Four action labels + None: Seek (seeking information), Evaluate (assessing status), Implement (executing operations), and Plan (formulating plans).
    - 10-fold cross-validation: 17/2/3 sessions split into train/val/test.

4. **Role-specific Analysis**: 

    - Surgeons have the most "Seek" labels (30.4%)—relying on team information.
    - Support staff have the most "Implement" labels (13.7%)—executing auxiliary operations.
    - Lexical features: Surgeons and trainees frequently say "thank you", while support staff often say "alright".

### Loss & Training

Annotated data is used for evaluation; no new training methodology is proposed. Evaluation methods include:
- Fine-tuned BERT_base
- Few-shot prompting with LLMs (Llama 3 8B/70B, GPT-4/4o)

## Key Experimental Results

### Case 1: Class Imbalance

| Method | Macro F1 | Micro F1 |
|------|----------|----------|
| BERT_base (fine-tuned) | 48.6 | **66.6** |
| Llama 3 8B (1-shot) | 37.0 | - |
| Llama 3 70B (5-shot) | 48.2 | - |
| GPT-4 (5-shot) | 47.0 | - |
| GPT-4o (5-shot) | **51.1** | - |

### Case 2: Dialogue Context

| Model | Context=1 | Context=3 | Context=5 |
|------|-----------|-----------|-----------|
| GPT-4o (1-shot) Macro F1 | 47.3 | **49.8** | 48.5 |
| GPT-4o (1-shot) Micro F1 | 55.0 | **58.0** | 56.0 |
| Llama 70B (1-shot) Macro F1 | **46.0** | 40.9 | 36.4 |

### Case 3: Multimodality

| Input Modality | Macro F1 | Micro F1 |
|----------|----------|----------|
| Text only (T) | **48.2** | - |
| T + Vision (Direct) | 46.8 | - |
| T + Physiology (Direct) | 44.9 | - |
| T + Vision (Textualized) | 46.9 | - |
| T + Physiology (Textualized) | 42.9 | - |

### Label Distribution Statistics

| Label | None | Seek | Evaluate | Implement | Plan | Total |
|------|------|------|----------|-----------|------|------|
| Total (k) | 3.7 | 1.3 | 0.8 | 0.6 | 0.3 | 6.9 |
| Percentage | 53.6% | 18.8% | 11.6% | 8.7% | 4.3% | - |

### Key Findings

- **High discrepancy between Macro and Micro F1 in fine-tuned BERT** (48.6 vs 66.6) indicates that the model is heavily biased toward majority classes.
- **LLMs are less affected by class imbalance**: The gap between Macro and Micro F1 in few-shot LLMs is smaller, but overall performance remains limited.
- **Increasing the number of shots for Llama 3 8B degrades performance** (37.0 $\rightarrow$ 32.7) as the smaller model's attention is distracted by long inputs.
- **Context aids GPT-4o but degrades Llama 70B**: The 128K context window of GPT-4o leverages dialogue information more effectively, whereas Llama 3's 8K window begins performance degradation around 1000 input tokens.
- **Multimodal inputs degrade performance**: Directly inputting video frames or screenshots of physiological signals leads to a drop in F1. GPT-4o hallucinates when reading physiological signals (e.g., misinterpreting an Et value of 64 as heart rate).
- **The highest Macro F1 is only 51.1%**, indicating that existing methods are far from meeting clinical requirements.

## Highlights & Insights

- **First dialogue dataset to include physiological signals**: CliniDial introduces 9 physiological signals, including heart rate, blood oxygen, and end-tidal CO2, presenting a novel challenge for multimodal fusion.
- **Naturally emerging dialogues**: Participants were not asked to "generate dialogues" but communicated naturally while performing clinical operations.
- **Role dynamics analysis**: Interaction patterns of different roles in the surgical team are uncovered through word frequency and label distributions.
- **Domain limitations of LLMs**: GPT-4o's hallucinations when reading physiological signals are highly insightful—AI requires domain-specific expertise to accurately interpret clinical data.
- **Practical impact**: The dataset directly links to medical safety—understanding team interactions can help improve surgical training.

## Limitations & Future Work

- Data comes from simulated surgeries rather than real ones, so dialogues may lack the ultimate urgency of real-world procedures.
- The dataset size is relatively limited (22 sessions), which may not suffice for large-scale pre-training.
- Video data will not be made public due to ethical considerations, limiting research on visual modalities.
- The role of acoustic features (such as tone, speech rate, and emotion) has not been explored.
- Advanced prompting strategies like Chain-of-Thought were not tested.
- The annotation framework only contains 4 behavioral categories, which might be insufficient to capture all meaningful team interaction patterns.

## Related Work & Insights

- **MultiWOZ** (Budzianowski et al., 2018): The benchmark dataset for task-oriented dialogue, which has limited scale and naturalness.
- **Ego4D** (Grauman et al., 2022): An egocentric video understanding dataset that inspired the multi-view data collection.
- **TeamSTEPPS**: A healthcare team training framework providing the theoretical basis for action coding.
- CliniDial can serve as a seed dataset for study at the intersection of NLP and medical collaboration.
- Insight: Multimodal fusion of real-world data is far from a simple concatenation; it requires guidance from domain knowledge.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The first multimodal dialogue dataset for clinical operations, introducing a brand-new physiological signal modality)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Three case studies systematically covering the three main features of the dataset, with a comparison across multiple models)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, thorough data analysis, and convincing qualitative cases)
- **Value**: ⭐⭐⭐⭐ (Fills the gap in clinical operation dialogue datasets and provides a benchmark for evaluating LLMs in high-risk domains)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](reflectool_clinical_agent.md)
- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](../../NeurIPS2025/medical_nlp/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[ACL 2025\] VITAL: A New Dataset for Benchmarking Pluralistic Alignment in Healthcare](vital_pluralistic_alignment_healthcare.md)
- [\[ACL 2025\] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)
- [\[ACL 2025\] Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training](adaptive-vp_a_framework_for_llm-based_virtual_patients_that_adapts_to_trainees_d.md)

</div>

<!-- RELATED:END -->
