---
title: >-
  [Paper Note] LLMs Can Simulate Standardized Patients via Agent Coevolution
description: >-
  [ACL 2025][Medical LLM][Standardized Patients] EvoPatient proposes a multi-agent coevolution framework. Through autonomous simulated dialogues between patient and doctor agents, LLMs learn to simulate standardized patients (SP) without human supervision, surpassing existing reasoning methods by more than 10% in requirement alignment.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Standardized Patients"
  - "Multi-Agent Coevolution"
  - "Medical Education"
  - "Role-Playing"
  - "Dialogue Simulation"
date: 2026-05-08
content_hash: 24302812055e2e04
---

# LLMs Can Simulate Standardized Patients via Agent Coevolution

**Conference**: ACL 2025  
**arXiv**: [2412.11716](https://arxiv.org/abs/2412.11716)  
**Code**: [https://github.com/ZJUMAI/EvoPatient](https://github.com/ZJUMAI/EvoPatient)  
**Area**: Medical NLP  
**Keywords**: Standardized Patients, Multi-Agent Coevolution, Medical Education, Role-Playing, Dialogue Simulation  

## TL;DR
EvoPatient proposes a multi-agent coevolution framework. Through autonomous simulated dialogues between patient and doctor agents, LLMs learn to simulate standardized patients (SP) without human supervision, surpassing existing reasoning methods by more than 10% in requirement alignment.

## Background & Motivation
**Background**: Standardized Patients (SPs) are critical in medical education, but training human SPs is costly and can negatively impact their mental health.

**Limitations of Prior Work**:
   - Rule-based digital patients lack authenticity and diversity.
   - Existing LLM-based SPs focus only on data retrieval or manual feedback-driven prompt adjustment, ignoring the critical need to learn standardized expression patterns.
   - LLMs must play a dual role of possessing professional medical knowledge while behaving like a medical layperson, which cannot be solved by simple prompt engineering alone.

**Key Challenge**: How to enable LLMs to autonomously learn to transform clinical record data into human-like standardized patient responses under zero human supervision.

**Goal**: To design a framework that allows patient agents to automatically accumulate experience through simulated diagnostic processes, evolving from novices into qualified SPs.

**Key Insight**: Coevolution between doctor and patient agents—the more professional the doctor's questions become, the more standardized the patient's responses are, forming a positive feedback loop.

**Core Idea**: Empowering patient agents to automatically accumulate standardized expression experience through a coevaluation mechanism in multi-agent simulated diagnostic processes.

## Method

### Overall Architecture
The input consists of real clinical records → Three core modules: (1) Simulated Flow models the diagnostic process into multi-stage dialogues → (2) Simulated Agent Pair performs autonomous multi-round dialogues (patient + multiple doctors) → (3) Coevolution Mechanism validates and stores high-quality dialogues to construct an attention library and a trajectories library → Output the trained virtual SP for human doctor training.

### Key Designs
1. **Simulated Flow**:

    - **Function**: Models the real diagnostic process into a structured, multi-stage dialogue sequence.
    - **Mechanism**: Divides the diagnostic workflow into phases such as chief complaint generation, triage, inquiry, and conclusion $\mathcal{F} = \langle \mathcal{S}^1, \mathcal{S}^2, \dots \rangle$, where each phase contains multi-round dialogues between doctor and patient.
    - **Design Motivation**: Structured workflows make the simulation resemble real-world scenarios while allowing flexible customization of different phases.

2. **Simulated Agent Pair**:

    - **Function**: Enables autonomous multi-round diagnostic dialogues between the patient agent and multiple doctor agents.
    - **Mechanism**: The patient agent is equipped with 5,000 diverse background profiles (family, education, personality, etc.) + RAG retrieval of clinical records; the doctor agent dynamically recruits agents from other departments via a **multi-disciplinary consultation mechanism**, forming a DAG topological structure to prevent information backflow and increase question diversity.
    - **Design Motivation**: Questions from multi-department doctors cover different perspectives, prompting the patient agent to learn more comprehensive standardized expressions.

3. **Coevolution Mechanism**:

    - **Function**: Automatically validates, stores, and utilizes high-quality dialogue experience.
    - **Mechanism**: Two experience libraries work collaboratively:
        - **Attention Library**: Decomposes SP requirements into question-level attention requirements, storing <question, clinical record, response, attention requirement> quadruples. At inference time, it retrieves similar questions to obtain refined requirements and few-shot demonstrations.
        - **Trajectories Library**: Stores high-quality dialogue trajectories $t_i = \{(q_{j-1}, a_{j-1}, q_j, a_j)\}$, helping doctor agents learn more professional inquiry pathways.
    - **Design Motivation**: Enables both sides to learn from past experiences—the patient learns standardized expressions, while the doctor learns professional questioning, forming a positive feedback loop.

### Loss & Training
Training-free. Based on GPT-4 / GPT-3.5-turbo. The coevolution process requires approximately 200 cases and reaches an optimal balance in 10 hours, demonstrating strong generalization capability.

## Key Experimental Results

### Main Results

| Method | Relevance | Faithfulness | Robustness | Ability (Comprehensive) |
|------|-----------|-------------|------------|-------------|
| CoT | 0.716 | 0.557 | 0.671 | 0.648 |
| ToT | 0.747 | 0.714 | 0.771 | 0.744 |
| Self-Align | 0.721 | 0.727 | 0.815 | 0.754 |
| Few-shot | 0.725 | 0.742 | 0.821 | 0.763 |
| **EvoPatient** | **0.759** | **0.879** | **0.941** | **0.860** |

### Ablation Study

| Configuration | Description |
|------|------|
| w/o Attention Library | Faithfulness and Robustness drop significantly |
| w/o Trajectories Library | Doctor inquiry quality degrades, indirectly affecting patient evolution |
| w/o Multi-disciplinary Consultation | Question diversity decreases, slowing down the improvement rate of SP capability |

### Key Findings
- **Robustness improves most significantly** (0.941 vs. second-best 0.821), proving that the coevolution mechanism effectively prevents the patient agent from leaking excessive medical information.
- Multi-disciplinary consultation contributes positively to the evolution of the patient agent—doctors from more disciplines ask more diverse questions.
- The evolution process stabilizes after about 200 cases, with reasonable resource consumption.
- In human doctor preference evaluation, EvoPatient achieves the highest score.

## Highlights & Insights
- The coevolution intuition is compelling: the doctor and patient push each other to improve, eliminating the need for human annotation.
- The Attention Library decomposes SP requirements into question-level attention requirements, which is more practical than global requirements.
- The DAG topology design of the multi-disciplinary consultation is elegant, avoiding information backflow while ensuring question diversity.
- The framework is highly generalizable and can be transferred to other role-playing simulation scenarios such as education and law.

## Limitations & Future Work
- Reliance on high-quality real clinical records, which poses ethical and privacy hurdles for data acquisition.
- Evaluation metrics (Relevance/Faithfulness/Robustness) are primarily based on automatic evaluations, with a limited scale for human trials.
- The performance of open-source models has not been fully explored (only GPT-4/3.5 were evaluated), resulting in high costs.
- Emotional expressions and non-verbal behaviors of patient agents are not yet addressed.

## Related Work & Insights
- **vs. Agent Hospital**: Agent Hospital performs self-evolution in a simulated world but lacks training for standardized expressions; EvoPatient focuses explicitly on the standardization capabilities of SPs.
- **vs. ExpeL**: ExpeL accumulates experience from successful trajectories but works as a single agent, whereas EvoPatient coevolves multiple agents to improve both sides simultaneously.
- **vs. Self-Align**: Self-Align uses principle-driven reasoning for alignment, whereas EvoPatient accumulates experience through actual simulated dialogues.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of multi-agent coevolution to train SPs is novel and practically meaningful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive baseline comparisons + ablation studies + human evaluations + resource analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with standardized formal formulation.
- Value: ⭐⭐⭐⭐ A highly practical framework in the field of medical education, with transferable coevolution paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Beyond Prompt: Fine-grained Simulation of Cognitively Impaired Standardized Patients via Stochastic Steering](../../ACL2026/medical_nlp/beyond_prompt_fine-grained_simulation_of_cognitively_impaired_standardized_patie.md)
- [\[ACL 2025\] Improving Automatic Evaluation of LLMs in Biomedical Relation Extraction via LLMs-as-the-Judge](biore_llm_judge_evaluation.md)
- [\[ACL 2025\] Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training](adaptive-vp_a_framework_for_llm-based_virtual_patients_that_adapts_to_trainees_d.md)
- [\[ICLR 2026\] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](../../ICLR2026/medical_nlp/can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)
- [\[ACL 2025\] Are LLMs Effective Psychological Assessors? Leveraging Adaptive RAG for Interpretable Mental Health Screening through Psychometric Practice](are_llms_effective_psychological_assessors_leveraging_adaptive_rag_for_interpret.md)

</div>

<!-- RELATED:END -->
