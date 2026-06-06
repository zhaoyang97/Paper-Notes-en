---
title: >-
  [Paper Note] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning
description: >-
  [ACL 2026][Medical Imaging][Clinical Reasoning] This paper adapts the Toulmin argument model to clinical diagnosis and proposes CGCL, a three-stage curriculum training framework (fact collection → hypothesis testing → sy…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Clinical Reasoning"
  - "Toulmin Argument Model"
  - "Curriculum Learning"
  - "Goal-Conditioned Learning"
  - "Trustworthy Diagnosis"
date: 2026-05-08
content_hash: f478ef76e540e8b1
---

# From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning

**Conference**: ACL 2026
**arXiv**: [2604.11137](https://arxiv.org/abs/2604.11137)  
**Code**: [https://github.com/Leonard-zc/CGCL](https://github.com/Leonard-zc/CGCL)  
**Area**: Medical Imaging
**Keywords**: Clinical Reasoning, Toulmin Argument Model, Curriculum Learning, Goal-Conditioned Learning, Trustworthy Diagnosis

## TL;DR
This paper adapts the Toulmin argument model to clinical diagnosis and proposes CGCL, a three-stage curriculum training framework (fact collection → hypothesis testing → synthesis), paired with T-Eval for quantifying reasoning structural completeness. The approach achieves diagnostic reasoning quality comparable to RL-based methods without requiring reinforcement learning.

## Background & Motivation

**Background**: LLMs have achieved impressive, even superhuman, performance on medical benchmarks such as MedQA and USMLE. However, standardized examinations do not reflect real clinical practice, which requires reasoning under uncertainty, integrating incomplete information, and tolerating the high cost of errors.

**Limitations of Prior Work**: (1) Current LLMs exhibit a dangerous "correct answer, flawed reasoning" phenomenon—reaching correct conclusions via pattern matching while the underlying reasoning is defective, lacking robust signal understanding. (2) Existing evaluations focus solely on final answer correctness, without examining the logical coherence or evidential support of the reasoning path. (3) RL-based methods can theoretically optimize reasoning quality but suffer from difficult reward model design, training instability, and high computational cost.

**Key Challenge**: In the medical domain, a correct answer accompanied by flawed reasoning is more dangerous than an incorrect answer—it generates false confidence and fails unpredictably when confronted with real clinical complexity. Current evaluation paradigms systematically overestimate LLM capability by examining only outcomes.

**Goal**: (1) Establish a structured evaluation framework for clinical reasoning; (2) Design a stable and efficient training method to teach LLMs Toulmin-style argumentative reasoning.

**Key Insight**: The Toulmin argument model requires that claims be supported by evidence, qualified under uncertainty, and defended against rebuttals—closely mirroring the reasoning process clinicians employ from symptom observation to diagnosis. This model is instantiated as structured output for clinical diagnosis.

**Core Idea**: The three-stage curriculum simulates the natural progression of medical training—interns extract facts and generate preliminary differentials → senior residents test hypotheses and address rebuttals → attending physicians synthesize and qualify conclusions.

## Method

### Overall Architecture
CGCL consists of two components: (1) **T-Eval**—a quantitative evaluation framework for reasoning quality based on the Toulmin model; (2) **CGCL training pipeline**—a three-stage goal-conditioned offline imitation learning approach, in which a frozen policy model generates candidate reasoning trajectories, T-Eval selects the optimal trajectory, and the result is distilled into the target model via SFT.

### Key Designs

1. **T-Eval Reasoning Evaluation Framework**:

    - **Function**: Measures the structural completeness of diagnostic argumentation beyond answer accuracy.
    - **Mechanism**: Formalizes diagnostic reasoning as a Toulmin argument $A = \{D, R, W, B, Q, Y\}$—where $D$ is case evidence, $R$ is a ranked differential diagnosis, $W$ is the warrant linking evidence to hypotheses (pathophysiological reasoning), $B$ is supporting clinical principles, $Q$ is uncertainty calibration, and $Y$ is the final diagnosis. Each component is scored independently to produce a composite measure of argumentative completeness.
    - **Design Motivation**: Evaluating only whether the final diagnosis is correct ignores the reasoning path. A model that "guesses correctly" via pattern matching and one that "argues correctly" through rigorous reasoning may yield identical answer accuracy, yet differ fundamentally in clinical reliability.

2. **Three-Stage Curriculum Goal-Conditioned Learning**:

    - **Function**: Progressively teaches LLMs the complete clinical reasoning pipeline from evidence to argumentation.
    - **Mechanism**: Stage 1 (Fact Collection, $C^{(1)} = \{D, R\}$)—the model extracts clinical findings and generates preliminary differential diagnoses; Stage 2 (Hypothesis Testing, $C^{(2)} = C^{(1)} \cup \{W, B\}$)—the model argues for the leading hypothesis using pathophysiological evidence and rebuts alternatives; Stage 3 (Synthesis, $C^{(3)} = C^{(2)} \cup \{Q, Y, \Delta\}$)—the model integrates all analyses, generates qualified conclusions, and performs evidence-driven diagnostic revision when necessary. At each stage, the policy model generates candidates, T-Eval selects the optimal trajectory, trajectories are merged into a coherent sequence, and SFT distillation is applied.
    - **Design Motivation**: Mirrors the natural progression of medical training—rather than directly teaching the model to produce complete arguments, the curriculum first establishes foundational skills (fact extraction), then intermediate skills (inferential reasoning), and finally advanced skills (synthetic judgment). Each stage is initialized from the preceding stage's model to ensure cumulative capability development.

3. **Evidence-Driven Diagnostic Revision Mechanism**:

    - **Function**: Enforces evidence-based revision of initial judgments in Stage 3, cultivating metacognitive capacity.
    - **Mechanism**: When the final diagnosis $Y$ diverges from the preliminary ranking in Stage 1, the model must generate a revision rationale $\Delta$ explicitly identifying which evidence prompted the diagnostic change. A revision indicator $\mathbb{I}_{\text{rev}}$ flags whether revision has occurred.
    - **Design Motivation**: Cultivates the ability to recognize and correct errors—a competent clinician not only reaches correct diagnoses but also revises initial judgments based on new evidence when warranted. This is a core element of clinical trustworthiness.

### Loss & Training
Standard SFT negative log-likelihood loss, applied sequentially across three stages. Training data at each stage is constructed by generating candidates via a policy model (e.g., GPT-4), scoring with T-Eval, selecting the optimal trajectory, and merging into coherent sequences. No RL is employed; only imitation learning is used.

## Key Experimental Results

### Main Results

| Method | Diagnostic Accuracy | T-Eval Reasoning Score | Training Stability |
|--------|--------------------|-----------------------|-------------------|
| Direct SFT | Moderate | Low | High |
| RL methods (GRPO, etc.) | High | Moderate–High | Low (unstable) |
| CGCL | **High (comparable to RL)** | **High** | **High** |

### Ablation Study

| Configuration | Accuracy | T-Eval | Notes |
|---------------|----------|--------|-------|
| Full CGCL (3 stages) | Best | Best | Complete curriculum |
| Single-stage direct generation of $C^{(3)}$ | Lower | Lower | Lacks progressive capability building |
| w/o diagnostic revision | Slightly lower | Below full | Contribution of metacognitive capacity |
| w/o T-Eval selection | Degraded | Degraded | Random trajectories insufficient in quality |

### Key Findings
- CGCL achieves diagnostic accuracy comparable to RL-based methods while offering greater training stability and computational efficiency.
- T-Eval reveals a hidden "correct answer, flawed reasoning" problem—some high-accuracy methods exhibit substantially deficient reasoning quality.
- Curriculum training yields the largest gains for smaller models, where progressive guidance is more critical given limited capacity.
- The evidence-driven revision mechanism is essential for clinical trustworthiness.

## Highlights & Insights
- **Paradigm shift from "getting the right answer" to "making the case"**: T-Eval elevates clinical reasoning quality from subjective assessment to a quantifiable automated metric, with broad applicability to any domain requiring interpretable reasoning.
- **Curriculum learning as an alternative to RL**: Carefully designed curriculum training can match RL-level reasoning quality while avoiding reward engineering complexity and training instability, offering a practical option for resource-constrained settings.
- **Clinical instantiation of the Toulmin model**: Precisely aligns philosophical argumentation theory with clinical practice, providing an actionable structured reasoning framework.

## Limitations & Future Work
- Reliance on GPT-4 as the policy model for generating candidate trajectories incurs non-trivial cost.
- T-Eval scoring quality depends on the capability of the evaluating LLM.
- Validation is limited to diagnostic reasoning; extension to treatment decision-making or prognosis assessment remains unexplored.
- The three-stage curriculum division is manually designed; finer-grained or adaptive curricula warrant further investigation.

## Related Work & Insights
- **vs. HuatuoGPT-O1**: Employs CoT distillation and RL training but does not evaluate reasoning structure. CGCL directly optimizes Toulmin completeness of the reasoning process.
- **vs. MedPRM**: Uses process reward models to supervise reasoning paths during training at high cost. CGCL replaces online PRM with T-Eval-based offline quality assessment.
- **vs. general clinical LLMs**: Most clinical LLMs optimize only answer accuracy; CGCL is the first to treat reasoning structure as a first-class optimization objective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The clinical instantiation of the Toulmin model and the T-Eval framework represent highly original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparison against multiple baselines; T-Eval provides a new evaluation dimension.
- Writing Quality: ⭐⭐⭐⭐⭐ The problem motivation is incisive (the danger of "correct answer, flawed reasoning"), and the method design is elegant.
- Value: ⭐⭐⭐⭐⭐ Offers a fundamental methodological contribution to the trustworthiness of medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)
- [\[ACL 2026\] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)
- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)

</div>

<!-- RELATED:END -->
