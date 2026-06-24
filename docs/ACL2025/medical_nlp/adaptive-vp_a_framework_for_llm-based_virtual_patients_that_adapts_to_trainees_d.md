---
title: >-
  [Paper Note] Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training
description: >-
  [ACL 2025][Medical LLM][Virtual patients] Proposes the Adaptive-VP framework, which utilizes LLMs to build Virtual Patients (VPs) that dynamically adjust their behavior based on the communication quality of nursing trainees. Through a four-module pipeline of multi-Agent evaluation $\rightarrow$ dynamic adaptation $\rightarrow$ dialogue generation $\rightarrow$ safety monitoring, the framework significantly improves the perceived realism of VP interactions (persona fidelity $\…
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Virtual patients"
  - "Adaptive dialogue"
  - "Nurse communication training"
  - "LLM Agent"
  - "Safety monitoring"
date: 2026-05-08
content_hash: 22be4c3409c01d31
---

# Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue to Facilitate Nurse Communication Training

**Conference**: ACL 2025  
**arXiv**: [2506.00386](https://arxiv.org/abs/2506.00386)  
**Code**: [https://github.com/keyeun/adaptive-vp](https://github.com/keyeun/adaptive-vp)  
**Area**: Medical NLP  
**Keywords**: Virtual patients, Adaptive dialogue, Nurse communication training, LLM Agent, Safety monitoring

## TL;DR

Proposes the Adaptive-VP framework, which utilizes LLMs to build Virtual Patients (VPs) that dynamically adjust their behavior based on the communication quality of nursing trainees. Through a four-module pipeline of multi-Agent evaluation $\rightarrow$ dynamic adaptation $\rightarrow$ dialogue generation $\rightarrow$ safety monitoring, the framework significantly improves the perceived realism of VP interactions (persona fidelity $\eta_p^2 = 0.151$, dialogue realism $\eta_p^2 = 0.254$) in a between-subjects experiment with 28 nursing experts.

## Background & Motivation

**Background**: Effective nurse-patient communication is crucial for treatment compliance, yet approximately 15% of clinical encounters are perceived as "difficult" by clinicians. Standardized Patient (SP) simulation is the core of traditional training but is costly, lacks flexibility, and relies heavily on scripted interactions. Virtual Patients (VPs) are rapidly developing as scalable alternatives, and recent LLM-enhanced VPs have enabled more natural context-aware interactions.

**Limitations of Prior Work**: Existing LLM VP systems still lack a natural feedback loop—when trainees use ineffective communication strategies, the VP should escalate difficult behaviors (such as increasing frustration) accordingly, and de-escalate otherwise. Most systems focus on maintaining pre-defined role fidelity and scenario consistency, failing to simulate dynamic patient-clinician interactions. Furthermore, safety guarantees (overly hostile content that may harm learners) are generally lacking.

**Key Challenge**: VPs need to simultaneously possess realism (including difficult behaviors) and safety (without causing psychological harm to learners), and their behavior must dynamically adapt to trainee performance—these three goals exist in tension.

**Goal**: How to construct an LLM virtual patient that can adjust its behavior in real time based on trainees' communication skills while ensuring learner safety?

**Key Insight**: Deconstruct VP interaction into four independent, customizable modules: evaluation, adaptation, generation, and safety, forming a closed-loop feedback system.

**Core Idea**: Drive dynamic adjustment of VP behavior through multi-agent communication evaluation, accompanied by safety monitoring to ensure learner safety, thereby achieving an adaptive virtual patient with a feedback loop.

## Method

### Overall Architecture

Adaptive-VP consists of two major parts: the VP case development pipeline (offline preparation) and the four-module dialogue management system (online interaction). The case pipeline is responsible for building clinically-grounded VP scenarios, while the four-module system handles real-time evaluation of trainees, adjustment of VP behaviors, dialogue generation, and safety review.

### Key Designs

1. **VP Case Development Pipeline**:

    - **Function**: Create clinically-grounded and customizable VP training scenarios.
    - **Mechanism**: A five-step workflow—(1) Specify training goals (e.g., handling difficult patients), (2) Incorporate best practices from nursing literature, (3) Specify training context (region/culture/trainee background), (4) Generate VP profiles (including demographics, medical history, behavioral details) using an LLM (Claude-3.5 Sonnet) and generate communication traits based on the seven-dimensional communication styles of De Vries et al. (2009), (5) Expert validation (reviewed by 10 nursing professionals).
    - **Design Motivation**: Standardized Patient (SP) protocols provide a solid foundation but are overly scripted; there is a need to balance clinical efficacy and flexibility.
    - **Practical Application**: Generated 8 VP cases (4 difficult patient types $\times$ 2 scenarios), with expert ratings for realism $M = 3.81$ (on a 5-point scale) and trait accuracy $M = 4.00$.

2. **Evaluation Module**:

    - **Function**: Evaluate the communication quality of each trainee utterance in real time, producing a communication effectiveness score of 0-5.
    - **Mechanism**: Two-tier evaluation—utterance level (calm and clear tone +1, empathy level $\ge 3$ +1, prohibited behavior such as premature empathy/dismissing beliefs/imperative responses -1) + dialogue level (use of de-escalation strategies: autonomy/setting boundaries/problem solving, +1 each).
    - **Reliability Assurance**: Employs a multi-agent evaluation (three roles: nursing professor + communication trainer + clinical psychologist), scoring only when unanimous, Fleiss' $\kappa > 0.75$.
    - **Design Motivation**: Single-agent evaluation suffers from positional bias and self-preference bias; multi-agent methods refer to Chan et al. (2023) to improve reliability.

3. **Dynamic Adjustment Module**:

    - **Function**: Decide the behavioral direction of the VP's next response based on the evaluation score.
    - **Mechanism**: Adjust three dimensions based on the 0-5 evaluation score—communication style, complaint intensity, and response attitude toward the nurse. High score $\rightarrow$ more cooperative and mild; low score $\rightarrow$ more resistant, emotional, or confrontational.
    - **Design Motivation**: Static VP behavior does not reflect the real world—patient behavior is itself a dynamic response to healthcare communication.
    - **Constraints**: Behavioral adjustments are restricted within predefined ranges to ensure role consistency.

4. **Dialogue Generation Module**:

    - **Function**: Generate contextally appropriate VP dialogue based on the adaptation direction.
    - **Mechanism**: Follows five rules—(1) Follow the predefined patient profile, (2) Natural Korean spoken style, (3) Include non-verbal cues, (4) Appropriately incorporate rude expressions, (5) Limit references to superior authority. The response structure has three parts: inner monologue (hidden), verbal response (cognitive + emotional state), and non-verbal annotation (e.g., "sighs").
    - **Design Motivation**: The three-part structure ensures consistency between internal states and external expressions.

5. **Safety Monitoring Module**:

    - **Function**: Review the VP response before presenting it to the trainee.
    - **Mechanism**: Four checks—safety assurance (no excessive hostility/derogation), training goal alignment, patient profile consistency, behavior direction compliance. If any check fails, it returns to the generation module for revision.
    - **Design Motivation**: Overly confrontational VP dialogue can cause emotional distress and reduce learner confidence.

## Key Experimental Results

### Main Results (Human Evaluation, N=28 experienced nurses)

| Dimension | Static VP | Dynamic VP (Adaptive-VP) | Effect Size | p-value |
|------|-----------|-------------------------|--------|-----|
| Persona Fidelity | Lower | **Significantly higher** | $\eta_p^2 = 0.151$ | 0.043 |
| Dialogue Realism | Lower | **Significantly higher** | $\eta_p^2 = 0.254$ | 0.008 |

### Ablation Study (Evaluation Module Validation, Expert vs Novice, N=30)

| Metric | Expert Group (N=15) | Novice Group (N=15) | Statistical Test |
|------|-------------|-------------|----------|
| Total Score | Significantly higher | Lower | $U = 160960, p = 0.001$ |
| Average Dialogue Turns | 7.45 turns | 5.3 turns | - |
| Tone Management | Strong | Weak | Subcomponent analysis significant |
| Use of De-escalation Strategies | Rich | Limited | Subcomponent analysis significant |

### Key Findings

- Dynamic VP was rated as significantly more realistic, with no significant differences across patient types (indicating consistent adaptability across scenarios).
- In open-ended feedback, nurses in the Dynamic group commented, "The VP feels very realistic; I have heard similar responses from real patients before."
- Nurses in the Static group pointed out, "If my response was effective, the patient should have calmed down, but they didn't"—the lack of a feedback loop is a fatal flaw.
- Role disagreement in multi-agent evaluation is systematic rather than random (GEE analysis); communication trainers and nursing professors systematically rated tone lower than clinical psychologists.

## Highlights & Insights

- The adaptive feedback loop is the core innovation—VPs react to communication styles just like real patients, which is unique compared to similar systems (Table 1 shows it is the only system integrating all four capabilities: expert validation, real-time evaluation, adaptive behavior, and safety assurance).
- Fine balance between safety and realism—instead of simply filtering out negative content, it preserves the realism of difficult scenarios while preventing extremely harmful content. This "controlled negative interaction" design paradigm is transferable to other educational AI scenarios.
- Multi-role Agent evaluation yields systematic, role-specific disagreements (rather than random variation), enhancing the multi-dimensional reliability of the evaluation.

## Limitations & Future Work

- Focuses solely on difficult patient interactions in South Korean nursing scenarios; generalizability across cultures and clinical contexts has not been validated.
- Uses only Claude-3.5 Sonnet; performance comparisons with other LLMs (e.g., GPT-4, LLaMA) were not conducted.
- Pure text dialogue lacks non-verbal modalities (intonation/expressions/gestures), limiting immersion and training authenticity.
- Evaluated VP realism but did not measure the actual long-term improvement of trainees' communication skills.
- The experimental sample size (28 participants) is limited.

## Related Work & Insights

- **vs Wang et al. (2024b) CBT Training**: Expert-validated but lacks real-time evaluation and adaptive behavior.
- **vs Steenstra et al. (2025) Motivational Interviewing**: Features real-time evaluation and adaptivity but lacks expert validation and safety assurance.
- **vs Louie et al. (2024) Psychological Counseling**: Expert-validated and features safety assurance but lacks real-time evaluation and adaptive behavior.
- **vs Traditional SP Simulation**: Costly and lacks flexibility; Adaptive-VP is scalable and available 24/7.

## Rating

- Novelty: ⭐⭐⭐⭐ The four-module closed-loop adaptive VP architecture is novel, although the individual module technologies are not cutting-edge.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three rounds of validation (10 in case review + 30 in evaluation + 28 in main experiment) are rigorous but the sample size is relatively small.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem-driven narrative of "four major challenges $\rightarrow$ four modules" is clear and fluent.
- Value: ⭐⭐⭐⭐ Has direct application value for AI in medical education, and the closed-loop architecture is transferable to other domains.
---
title: >-
  [Paper Insight] Adaptive-VP: A Framework for LLM-Based Virtual Patients that Adapts to Trainees' Dialogue
description: >-
  [ACL 2025][LLM/NLP][Virtual Patients] Proposes Adaptive-VP—an LLM-based virtual patient dialogue generation framework that dynamically adjusts virtual patient behaviors based on the communication quality of nursing trainees (poor communication -> escalates hostility, good communication -> de-escalates). It includes five components: a case development pipeline, an evaluation module, a dynamic adjustment module, a dialogue generation module, and a safety monitoring module. Evaluation by expert nurses indicates its interaction naturalness and realism are significantly superior to existing methods.
tags:
  - ACL 2025
  - LLM/NLP
  - Virtual Patients
  - Nurse Communication Training
  - Adaptive Dialogue
  - LLM
  - Difficult Patients
  - Safety Monitoring
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification](redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)
- [\[NeurIPS 2025\] CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](../../NeurIPS2025/medical_nlp/cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)
- [\[ACL 2025\] LLMs Can Simulate Standardized Patients via Agent Coevolution](evopatient_standardized_patient.md)
- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](../../ICLR2026/medical_nlp/atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[AAAI 2026\] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](../../AAAI2026/medical_nlp/a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)

</div>

<!-- RELATED:END -->
