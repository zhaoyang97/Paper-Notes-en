---
title: >-
  [Paper Note] From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning
description: >-
  [ACL 2026][Medical Imaging][Clinical Reasoning] This paper adapts the Toulmin argumentation model to the clinical diagnostic process and proposes CGCL…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Clinical Reasoning"
  - "Toulmin Argumentation Model"
  - "Curriculum Learning"
  - "Goal-Conditioned Learning"
  - "Trustworthy Diagnosis"
date: 2026-05-08
content_hash: 0e80bee3913c79ee
---

# From Answers to Arguments: Toward Trustworthy Clinical Diagnostic Reasoning with Toulmin-Guided Curriculum Goal-Conditioned Learning

**Conference**: ACL 2026  
**arXiv**: [2604.11137](https://arxiv.org/abs/2604.11137)  
**Code**: [https://github.com/Leonard-zc/CGCL](https://github.com/Leonard-zc/CGCL)  
**Area**: Medical Imaging  
**Keywords**: Clinical Reasoning, Toulmin Argumentation Model, Curriculum Learning, Goal-Conditioned Learning, Trustworthy Diagnosis

## TL;DR
This paper adapts the Toulmin argumentation model to the clinical diagnostic process and proposes CGCL, a three-stage curriculum training framework (Fact Collection → Hypothesis Testing → Comprehensive Conclusion). Accompanied by T-Eval for quantifying reasoning structure integrity, it achieves diagnostic reasoning quality comparable to RL methods without the need for RL.

## Background & Motivation

**Background**: LLMs perform excellently on medical benchmarks (e.g., MedQA, USMLE), even surpassing human experts. However, standardized tests $\neq$ real clinical practice. Clinical decision-making requires reasoning under uncertainty, integrating incomplete information, and bearing the cost of errors.

**Limitations of Prior Work**: (1) Current LLMs exhibit the dangerous "right answer + wrong reasoning" phenomenon—reaching correct conclusions through pattern matching despite flawed reasoning processes and lacking robust understanding of signals. (2) Existing evaluations focus only on final answer accuracy, failing to examine the logic and evidentiary support of the reasoning path. (3) RL methods can theoretically optimize reasoning quality, but reward model design is difficult, training is unstable, and computational demands are high.

**Key Challenge**: In the medical field, a correct answer with incorrect reasoning is more dangerous than an incorrect answer—it provides false confidence and fails unpredictably when facing real clinical complexity. Current evaluation paradigms systematically overestimate the actual capabilities of LLMs by looking only at results.

**Goal**: (1) Establish a structured clinical reasoning evaluation framework; (2) Design a stable and efficient training method to teach LLMs Toulmin-style argumentative reasoning.

**Key Insight**: The Toulmin argumentation model emphasizes that claims must have evidentiary support, uncertainty qualifiers, and rebuttal defenses—highly consistent with the reasoning process of clinicians from symptoms to diagnosis. This model is instantiated as a structured output for clinical diagnosis.

**Core Idea**: A three-stage curriculum simulates the natural progression of medical training—interns extracting facts and preliminary differentials → senior residents performing hypothesis testing and rebuttal → attending physicians synthesizing judgments and qualifying conclusions.

## Method

### Overall Architecture
CGCL consists of evaluation and training components: (1) T-Eval—a quantitative evaluation framework for reasoning quality based on the Toulmin model; (2) CGCL training pipeline—a three-stage goal-conditioned offline imitation learning approach, using a frozen policy model to generate candidate reasoning trajectories, selecting the best via T-Eval scores, and distilling them into the target model via SFT.

### Key Designs

1. **T-Eval Reasoning Evaluation Framework**:

    - **Function**: Beyond answer accuracy, it directly measures the structural integrity of diagnostic arguments.
    - **Mechanism**: Formalizes diagnostic reasoning as a Toulmin argument $A = \{D, R, W, B, Q, Y\}$—where $D$ is case evidence, $R$ is differential diagnosis ranking, $W$ is the argument from evidence to hypothesis (pathophysiological links), $B$ is supporting clinical principles, $Q$ is uncertainty calibration, and $Y$ is the final diagnosis. Each component is scored independently to measure argument completeness comprehensively.
    - **Design Motivation**: Looking only at whether the final diagnosis is correct ignores the reasoning path—a model that "guesses right" via pattern matching and a model that "argues right" via rigorous reasoning may have the same accuracy but differ vastly in clinical reliability.

2. **Three-Stage Curriculum Goal-Conditioned Learning**:

    - **Function**: Progressively teaches LLMs the complete clinical reasoning process from facts to arguments.
    - **Mechanism**: Stage 1 (Fact Collection, $C^{(1)} = \{D, R\}$)—model extracts clinical findings and generates preliminary differential diagnoses; Stage 2 (Hypothesis Testing, $C^{(2)} = C^{(1)} \cup \{W, B\}$)—model argues for primary hypotheses and rebuts alternatives using pathophysiological evidence; Stage 3 (Comprehensive Conclusion, $C^{(3)} = C^{(2)} \cup \{Q, Y, \Delta\}$)—model integrates all analyses, generates qualified conclusions, and performs evidence-driven diagnostic revision if necessary. Each stage uses a policy model to generate candidates → T-Eval scores to select the best → fusion into a coherent trajectory → SFT distillation.
    - **Design Motivation**: Simulates the natural progression in medical training—rather than teaching the model to generate full arguments directly, it first learns basics (extracting facts), then intermediate (reasoning and argumentation), and finally advanced (synthetic judgment). Each stage initializes from the previous stage's model to ensure capability accumulation.

3. **Evidence-Driven Diagnostic Revision Mechanism**:

    - **Function**: Mandates the model to revise initial judgments based on evidence in Stage 3, fostering metacognitive abilities.
    - **Mechanism**: When the final diagnosis $Y$ disagrees with the preliminary ranking from Stage 1, the model must generate a revision rationale $\Delta$, specifying which evidence led to the diagnostic change. A revision indicator $\mathbb{I}_{\text{rev}}$ marks whether a revision occurred.
    - **Design Motivation**: Develops the ability to "correct mistakes"—a good clinician not only makes correct diagnoses but also revises based on new evidence when initial judgments are found to be erroneous. This is a core element of clinical trustworthiness.

### Loss & Training
Standard SFT negative log-likelihood loss is applied with sequential three-stage training. Training data for each stage consists of candidates generated by a policy model (e.g., GPT-4), scored by T-Eval, and fused after optimal selection. No RL is used, only imitation learning.

## Key Experimental Results

### Main Results

| Method | Diagnostic Accuracy | T-Eval Reasoning Score | Training Stability |
|------|-----------|---------------|-----------|
| Direct SFT | Moderate | Low | High |
| RL Methods (GRPO, etc.) | High | Medium-High | Low (Unstable) |
| **CGCL (Ours)** | **High (Comparable to RL)** | **High** | **High** |

### Ablation Study

| Configuration | Accuracy | T-Eval | Description |
|------|--------|--------|------|
| Full CGCL (3 Stages) | Optimal | Optimal | Full Curriculum |
| Single-stage direct generation of $C^{(3)}$ | Lower | Lower | Lacks progressive capability building |
| w/o Diagnostic Revision | Slightly lower | Lower than full | Contribution of metacognitive ability |
| w/o T-Eval Selection | Decrease | Decrease | Random trajectory quality is insufficient |

### Key Findings
- CGCL is comparable to RL methods in diagnostic accuracy but is more stable in training and more computationally efficient.
- T-Eval reveals the hidden issue of "correct answers but flawed reasoning"—some high-accuracy methods are significantly deficient in reasoning quality.
- Curriculum training most significantly benefits smaller models—limited capabilities require more progressive guidance.
- Evidence-driven revision mechanisms are crucial for clinical trustworthiness.

## Highlights & Insights
- **Paradigm shift from "correct answers" to "clear explanations"**: T-Eval elevates clinical reasoning quality from subjective assessment to quantifiable automated metrics, which is valuable for all fields requiring interpretable reasoning.
- **Curriculum learning as an alternative to RL**: A carefully designed curriculum can achieve RL-level reasoning quality while avoiding reward design and training instability. This provides a practical choice for resource-constrained scenarios.
- **Clinical instantiation of the Toulmin model**: Precisely aligns philosophical argumentation theory with clinical practice, providing an actionable structured reasoning framework.

## Limitations & Future Work
- Dependence on GPT-4 as a policy model for generating candidate trajectories remains costly.
- T-Eval's scoring quality depends on the capabilities of the evaluating LLM.
- Validated only on diagnostic reasoning, not extended to treatment decisions or prognosis assessment.
- The three-stage division is manually designed; finer-grained or adaptive curricula could be explored.

## Related Work & Insights
- **vs HuatuoGPT-O1**: Uses CoT distillation + RL training but does not evaluate reasoning structure. CGCL directly optimizes the Toulmin integrity of reasoning.
- **vs MedPRM**: Trains Process Reward Models to supervise reasoning paths, which is costly. CGCL uses T-Eval for offline quality assessment as a substitute for online PRMs.
- **vs General Clinical LLMs**: Most clinical LLMs only optimize answer accuracy; CGCL is the first to treat reasoning structure as a first-class optimization objective.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Clinical instantiation of the Toulmin model and the T-Eval framework are highly original contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Sufficient comparison with various baselines and methods; T-Eval provides a new dimension of evaluation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Deep problem motivation (danger of "correct answer + wrong reasoning"), elegant method design.
- **Value**: ⭐⭐⭐⭐⭐ Makes fundamental methodological contributions to the trustworthiness of medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning](cure-med_curriculum-informed_reinforcement_learning_for_multilingual_medical_rea.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[ACL 2026\] CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning](cure-med_curriculum-informed_reinforcement_learning_for_multilingual_medical_rea.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[NeurIPS 2025\] FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning](../../NeurIPS2025/medical_imaging/fairgrpo_fair_reinforcement_learning_for_equitable_clinical_reasoning.md)

</div>

<!-- RELATED:END -->
