---
title: >-
  [Paper Note] ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education
description: >-
  [ICML 2026][Medical NLP][Clinical Education] This paper proposes ClinTutor-R1, the first Vision-Language Agent designed for one-to-many alignment in clinical Socratic teaching. By constructing the 48k ClinTeach dialogue…
tags:
  - "ICML 2026"
  - "Medical NLP"
  - "Clinical Education"
  - "One-to-Many Alignment"
  - "Socratic Teaching"
  - "Multi-Agent Simulation"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: f99e9053ed9ecdc3
---

# ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2512.05671](https://arxiv.org/abs/2512.05671)  
**Code**: https://github.com/Zhitao-He/ClinTutor-R1  
**Area**: Medical NLP  
**Keywords**: Clinical Education, One-to-Many Alignment, Socratic Teaching, Multi-Agent Simulation, Vision-Language Models  

## TL;DR

This paper proposes ClinTutor-R1, the first Vision-Language Agent designed for one-to-many alignment in clinical Socratic teaching. By constructing the 48k ClinTeach dialogue dataset via the multi-agent simulator ClinEdu, and employing explicit Theory of Mind (ToM) reasoning and three-axis rubric reinforcement learning, the model maintains stable teaching quality even as the student count scales to 10, surpassing baseline models by 20% and achieving performance parity with GPT-4o.

## Background & Motivation

**Background**: Current LLM alignment techniques (e.g., RLHF) have achieved significant success in one-to-one interaction scenarios. However, many real-world settings require AI to serve multiple users simultaneously, such as a mentor guiding several students during clinical rounds.

**Limitations of Prior Work**: Existing models face two core issues in one-to-many scenarios: (1) **Context dilution**, where the model loses its ability to track individual cognitive states as the number of students increases; (2) **Goal misalignment**, making it difficult to balance personalized guidance with overall group progress. Experiments show that baseline models experience a "performance cliff" when the number of students exceeds 3, with quality dropping by nearly 15%.

**Key Challenge**: Standard alignment methods optimize for a single user's reward signal and lack the ability to model Theory of Mind (ToM). Consequently, they cannot simultaneously maintain each student's cognitive state while coordinating group consensus, which is particularly critical in clinical scenarios requiring both safety and pedagogical depth.

**Goal**: To build a scalable one-to-many alignment framework that allows an AI tutor to provide high-quality, personalized Socratic teaching as the student group scales.

**Key Insight**: Clinical rounds were chosen as the testbed. This scenario naturally features heterogeneous cognitive states (from novices to senior residents) and dual clinical-pedagogical objectives (deep reasoning vs. safety guardrails), making it an ideal environment for one-to-many alignment.

**Core Idea**: Generate large-scale pedagogical dialogue data through a multi-agent simulator and combine explicit ToM reasoning mechanisms with multi-axis rubric reinforcement learning to train a Vision-Language Agent that maintains stable teaching quality in one-to-many settings.

## Method

### Overall Architecture

The system consists of three core components: (1) **ClinEdu**, a multi-agent teaching simulator that models tutor-student-patient interaction dynamics during clinical rounds; (2) **ClinTeach**, a dataset containing 48k Socratic teaching dialogues (31k single-turn + 17k multi-turn); (3) **ClinTutor-R1**, a model based on Qwen2.5VL-7B trained via a two-stage SFT + RL process. Inputs include clinical cases (text and medical images like X-ray/CT), and outputs provide Socratic guidance for multiple students.

### Key Designs

1. **ClinEdu Multi-Agent Simulator**:

    - **Function**: Generates high-fidelity clinical teaching interaction data, covering five agent types: tutor, patient, student, expert review, and safety monitor.
    - **Mechanism**: Decouples the patient's objective medical record (Patient Script) from their subjective personality (Persona). The flexible combination of these elements allows for infinite clinical scenarios. Student agents are sampled from a pool of 300 personas, each with different knowledge levels, cognitive styles, and learning methods. Interactions follow a three-stage closed-loop protocol: independent student analysis → Socratic guidance by the tutor (reviewed by experts and safety monitors) → student exploration.
    - **Design Motivation**: Real-world clinical teaching data is scarce due to privacy regulations. The decoupled design enables scalable data generation, while persona-driven interactions capture the emergent pedagogical conflicts that static templates miss.

2. **Explicit Theory of Mind (ToM) Reasoning Mechanism**:

    - **Function**: Before generating guidance, the model performs structured internal reasoning to model each student's cognitive state and the group consensus.
    - **Mechanism**: The reasoning chain includes four dimensions: `<think history>` tracks dialogue progress; `<think question>` aligns pedagogical goals; `<think student student_id="X">` analyzes each student's understanding individually; and `<think group>` synthesizes group analysis to identify collective blind spots. By writing a dedicated reasoning trajectory for each student, the model maintains independent mental models as student numbers grow.
    - **Design Motivation**: Addresses context dilution by explicitly decoupling multi-agent interaction into independent individual analyses, preventing information leakage across long contexts. The reasoning trajectories also serve as verifiable audit trails.

3. **Three-Axis Rubric Reinforcement Learning**:

    - **Function**: Optimizes the model's dynamic adaptation to diverse student inputs after SFT.
    - **Mechanism**: Reward functions are decomposed along three axes: **Instructional Structure** (IS: reasoning tag integrity, Socratic question quality), **Analysis Quality** (AQ: depth of individual assessment, group synthesis ability), and **Clinical Safety** (CS: factual accuracy, safety priority). A key design is the veto mechanism: if any safety-related criterion $\{CS\text{-}1, CS\text{-}2, IS\text{-}1\}$ receives a score $s_i < 0$, the final reward is vetoed to a large negative value $R_{\text{final}} = P_{\text{veto}}$. The policy is optimized using the GRPO algorithm.
    - **Design Motivation**: A single holistic score cannot distinguish between the need for pedagogical flexibility and the rigidity required for safety. The veto mechanism allows the policy to quickly learn safety boundaries (triggering 8-12% during early exploration and dropping to <2% after stabilization) without suppressing Socratic diversity.

## Key Experimental Results

### Main Results

| Model | MedXpertQA Avg | MVME Avg | MSM (MedXpert) | MSM (MVME) |
|------|---------------|----------|----------------|------------|
| LLaVA-v1.6 | 5.87 | 5.56 | 6.15 | 5.74 |
| Qwen2.5VL (Baseline) | 6.96 | 6.83 | 7.04 | 7.13 |
| TutorRL | 7.42 | 7.13 | 7.49 | 7.01 |
| Med-SocraticLM | 7.41 | 7.28 | 7.33 | 7.18 |
| GPT-4o | 8.36 | 8.47 | 8.26 | 8.39 |
| o3 | 8.42 | 8.45 | 8.18 | 8.23 |
| **ClinTutor-R1** | **8.35** | **8.49** | **8.41** | **8.55** |

ClinTutor-R1 surpasses GPT-4o on MVME (8.49 vs. 8.47) and significantly outperforms it in the Multi-Student Management (MSM) dimension (8.55 vs. 8.39). In human expert evaluations, ClinTutor-R1 scored 8.73, exceeding o3's 8.41. In a 200-person user study, it received a recommendation score of 8.70.

### Ablation Study

| Configuration | MedXpertQA Avg | MVME Avg | Description |
|------|---------------|----------|-------------|
| Full model | 8.35 | 8.49 | Complete model |
| w/o RL | 7.69 | 7.58 | Largest drop (0.66/0.91) without RL |
| w/o Thinking | 7.94 | 7.79 | Drop (0.41/0.70) without ToM chain |
| w/ Vanilla reward | 8.01 | 7.88 | Single reward instead of 3-axis rubric |
| w/o reward veto | 7.87 | 8.03 | MPS drops (8.26→6.92) without veto |
| w/ One-Student | 7.86 | 7.69 | Poor generalization when trained on single student |

### Key Findings

- **RL contributes most**: Removing reinforcement learning leads to the largest performance drop, indicating that SFT alone is insufficient for learning to adapt dynamically to diverse student inputs.
- **Veto mechanism is crucial for safety**: Removing the veto causes the MPS (Medical Safety) dimension to plunge from 8.26 to 6.92, showing that policies learn "reward hacking" without hard constraints.
- **Scalability advantage**: As the student count scales from 1 to 10, ClinTutor-R1 maintains an average score above 8.20, while Med-SocraticLM drops by 15% after 3 students.
- **Error Correction**: In error injection experiments, ClinTutor-R1 achieved a Corrective Success Rate (CSR) of 88.50%, performing particularly well in categories of premature closure (89.10%) and safety/ethical risks (88.60%).

## Highlights & Insights

- **Explicit ToM Decoupling**: Writing independent `<think student>` trajectories for each student is an elegant solution to context dilution in one-to-many scenarios. This "think before you speak" design not only improves performance but also makes the AI tutor's decisions auditable and interpretable.
- **"Safety Floor" via Veto Mechanism**: Treating safety as a hard constraint rather than a soft reward component ensures the clinical safety baseline without suppressing pedagogical variety. The rapid drop in veto triggers from 12% to 2% suggests the strategy successfully internalizes safety boundaries.
- **Decoupled Data Generation**: The Patient Script/Persona decoupling approach is transferable to any scenario requiring role-playing training data (e.g., legal consultation, management training), enabling exponential growth in data diversity.

## Limitations & Future Work

- Perception is limited to text and static medical images (X-ray, CT), lacking dynamic environmental awareness (e.g., patient expressions, physical examination maneuvers) found in real rounds.
- While simulator data is high-fidelity, a gap remains with real classroom environments (e.g., unmodeled student distraction or emotional shifts).
- Training and evaluation are primarily based on MedXpertQA, and generalization across different medical systems (e.g., non-USMLE standards) requires further validation.
- Future work could explore combining ToM reasoning with online learning to allow the model to continuously update its cognitive models of students during real interactions.

## Related Work & Insights

- **SocraticLM** (Liu et al., 2024b): Uses a Dean-Teacher-Student multi-agent pipeline for math dialogues but is limited to single-student scenarios.
- **TutorRL** (Dinucu-Jianu et al., 2025): An RL framework that balances guidance versus answer leakage but does not handle multi-student management.
- **MEDCO** (Wei et al., 2024): Simulates multi-agent clinical teams but uses 1:1 patient-doctor mapping and lacks Script/Persona decoupling.
- The three-axis rubric + veto RL framework introduced here can be generalized to any RLHF task requiring multi-dimensional quality constraints (e.g., correctness-safety-readability in code generation).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](../../ACL2026/medical_nlp/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ACL 2026\] PrinciplismQA: A Philosophy-Grounded Approach to Assessing LLM-Human Clinical Medical Ethics Alignment](../../ACL2026/medical_nlp/principlismqa_a_philosophy-grounded_approach_to_assessing_llm-human_clinical_med.md)
- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_nlp/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[ICLR 2026\] MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](../../ICLR2026/medical_nlp/medagentgym_agentic_training_biomedical.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_nlp/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)

</div>

<!-- RELATED:END -->
