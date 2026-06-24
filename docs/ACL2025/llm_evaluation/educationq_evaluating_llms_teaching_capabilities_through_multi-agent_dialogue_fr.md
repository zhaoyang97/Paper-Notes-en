---
title: >-
  [Paper Note] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework
description: >-
  [ACL 2025][LLM Evaluation][Teaching Capability Evaluation] This work proposes EducationQ, a multi-agent dialogue framework designed to evaluate the teaching capabilities of LLMs by simulating teacher-student informal formative assessment interactions in real classrooms. The study reveals that teaching effectiveness is not linearly correlated with model size or general reasoning ability, with Llama 3.1 70B demonstrating the best teaching performance.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Teaching Capability Evaluation"
  - "Multi-Agent"
  - "Formative Assessment"
  - "Educational AI"
  - "LLM-as-Teacher"
date: 2026-05-08
content_hash: f9b8418d620a2d63
---

# EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework

**Conference**: ACL 2025  
**arXiv**: [2504.14928](https://arxiv.org/abs/2504.14928)  
**Code**: [github](https://github.com/SunriserFuture/EducationQ)  
**Area**: LLM Evaluation  
**Keywords**: Teaching Capability Evaluation, Multi-Agent, Formative Assessment, Educational AI, LLM-as-Teacher

## TL;DR

This work proposes EducationQ, a multi-agent dialogue framework designed to evaluate the teaching capabilities of LLMs by simulating teacher-student informal formative assessment interactions in real classrooms. The study reveals that teaching effectiveness is not linearly correlated with model size or general reasoning ability, with Llama 3.1 70B demonstrating the best teaching performance.

## Background & Motivation

While LLMs are increasingly applied in education, existing evaluation methods possess fundamental limitations:

**Misalignment of Evaluation Goals**: Current benchmarks (MMLU, GPQA, etc.) primarily evaluate knowledge recall and reasoning capabilities rather than interactive teaching skills. The core of education lies in guiding the learning process, facilitating knowledge construction, providing personalized feedback, and scaffolding skills.

**Limitations of Evaluation Methods**:
   - Closed-ended questions only test baseline knowledge levels and fail to capture the dynamic nature of instruction.
   - Open-ended evaluations rely on human judgment, making them difficult to scale.
   - Multi-turn dialogue frameworks lack dedicated mechanisms to elicit and evaluate teaching effectiveness.

**Lack of Teacher Initiative**: Existing frameworks fail to assess the active role of a teacher in posing questions, evaluating understanding, and making real-time adjustments.

The core theoretical foundation of this work is Vygotsky's Zone of Proximal Development (ZPD) and Informal Formative Assessment (IFA), where teachers assess learning progress through continuous dialogue, identify gaps, and adapt teaching strategies.

## Method

### Overall Architecture

EducationQ adopts a tri-role multi-agent architecture comprising a **Teacher Agent** (to be evaluated), a standardized **Student Agent**, and an **Evaluator Agent** (to analyze teaching quality), simulating a cyclical teacher-student interaction in a classroom setting.

### Key Designs

1. **Student Agent**: Llama 3.1 70B Instruct is used as the fixed student model (with an accuracy of 46.97% on the GPQA Diamond benchmark). Ablation studies show that switching the student model (e.g., to Qwen 72B or Mistral Nemo) does not alter the teacher rankings, proving that the method effectively isolates the differences in teacher performance.

2. **Teacher Agent**: The agent is prompted to dynamically assess the student's thought process, using probing questions to gauge comprehension and provide feedback. A key constraint is that the teacher **cannot access the multiple-choice options** and must guide learning solely based on the student's reasoning patterns and correctness feedback, preventing direct answer leakage.

3. **Evaluator Agent**: This agent utilizes a qualitative analysis framework with 17 rubric dimensions, covering teacher dimensions (questioning, assessment, feedback) and student impact dimensions (metacognitive reflection, cognitive dimensions, etc.). Evaluation by human experts demonstrates a 78% agreement rate with the automated qualitative analysis.

4. **Interaction Protocol**:

    - **Pre-test**: Establish the student's initial knowledge baseline.
    - **Interaction**: 5 rounds of dialogue per question, with a maximum of 150 tokens per turn for the teacher and 260 tokens for the student.
    - **Post-test**: Incorporate pre-test reasoning traces and teacher-student dialogue history, maintaining evaluation parameters consistent with the baseline.

5. **Dataset Construction**: A dataset of 1,498 questions was meticulously curated from GPQA (448 questions) and MMLU-Pro (12,032 questions), covering 13 subjects and 10 difficulty levels. MMLU-Pro Stratified ensures a balanced distribution of subjects and difficulty via stratified sampling.

### Loss & Training

This work introduces an evaluation framework rather than a training method. The evaluation metric system includes:
- **Absolute Learning Gain (ALG)**: $ALG = ACC_{post} - ACC_{pre}$, which directly measures teaching effectiveness.
- **Positive-Negative Impact Ratio (PNIR)**: $PNIR = N_{neg} / N_{pos}$, which measures teaching consistency (lower is better).
- **Cross-Subject Stability (CSS)**: The standard deviation of learning gains across subjects (lower is better).
- **Unique Improvement Count (UIC)**: The number of unique questions that only a specific teacher model can improve.

## Key Experimental Results

### Main Results

**Overall Teaching Performance (14 LLMs, 1,498 questions)**:

| Teacher Model | Pre-test | Post-test | ALG↑ | CSS↓ | PNIR↓ | UIC |
|---------|------|------|------|------|-------|-----|
| Llama 3.1 70B Instruct | 47.73 | 58.74 | **11.01** | 0.041 | **0.18** | **37** |
| Gemini 1.5 Pro 002 | 47.73 | 55.21 | 7.48 | **0.030** | 0.40 | 37 |
| OpenAI o1-mini | 47.73 | 53.57 | 5.84 | 0.051 | 0.25 | 7 |
| Qwen 2.5 72B Instruct | 47.73 | 53.14 | 5.41 | 0.054 | 0.33 | 7 |
| Llama 3.1 8B Instruct | 47.73 | 52.60 | 4.87 | 0.051 | 0.40 | 13 |

**GPQA Diamond Subset (Consistency Across Student Models)**:

| Teacher \ Student | Llama 70B | Qwen 72B | Mistral Nemo |
|------------|-----------|----------|-------------|
| Llama 70B Teacher | +12.63% | +8.08% | +4.55% |
| Qwen 72B Teacher | +8.59% | +4.55% | +2.53% |
| Mistral Nemo Teacher | +7.07% | +2.53% | 0.00% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| 250 tokens per turn (vs. 150) | No significant improvement | Relaxing output constraints does not improve teaching effectiveness. |
| 70-100 tokens per turn | Teaching performance degradation | Insufficient expression space constrains teaching strategies. |
| 10 dialogue rounds (vs. 5 rounds) | No significant improvement | Doubling compute costs yields limited gains. |
| Test-retest stability (GPQA-main) | σ²=0.00832 | Extremely low ALG variance, indicating high framework stability. |
| Cross-dataset consistency | r=0.871, p<0.001 | High ranking consistency across GPQA and MMLU-Pro datasets. |

### Key Findings

1. **Teaching capability is not proportional to model size**: Llama 3.1 70B outperforms larger 405B and commercial models, suggesting that teaching capabilities require specialized optimization.
2. **Different models possess unique teaching advantages**: Llama 70B excels in subtle questioning strategies and knowledge-intensive subjects, o1-mini shines in reasoning-intensive subjects, and Gemini 1.5 Pro is proficient in providing targeted feedback.
3. **Surprising improvement in specific subjects**: Llama 70B achieved up to a 24% accuracy improvement in certain subjects.
4. **Teacher rankings remain consistent across different student models**: This validates the robustness of the evaluation framework.

## Highlights & Insights

- **Theory-driven evaluation design**: Integrates Vygotskian learning theory and formative assessment theory into the AI evaluation framework, grounding technical assessment in pedagogical theory.
- **Rigorous data flow control**: Teachers cannot access options, students cannot access pre-test results, and learning occurs solely through dialogue. These constraints ensure the fairness of the evaluation and the authenticity of the teaching behaviors.
- **Discovery of "Teaching Effectiveness ≠ Knowledge Level"**: Challenges the assumption that "larger models are inherently better," charting a clear direction for educational AI development.
- **Mixed-method evaluation**: Combines quantitative metrics (learning gain) with qualitative analysis (17 dimensions of teaching behaviors) to offer a comprehensive perspective.

## Limitations & Future Work

1. **Authenticity of student models**: The assumption that the student behavior simulated by LLMs truly reflects human student learning processes requires further validation.
2. **Limitations of short interactions**: Only 5 rounds of dialogue per question may be insufficient to evaluate long-term instructional strategies.
3. **Uneven subject coverage**: Biology has only 19 questions in GPQA Diamond, which impacts the reliability of cross-subject comparisons.
4. **Overlapping evaluation dimensions**: There is overlap among the 17 qualitative analysis dimensions (as acknowledged by the authors).
5. **Evaluation limited to multiple-choice questions**: The post-test uses a multiple-choice question (MCQ) format, failing to evaluate open-ended learning outcomes.

## Related Work & Insights

- **GPQA (Rein et al., 2023)**: A high-difficulty Q&A benchmark designed by domain experts.
- **MMLU-Pro (Wang et al., 2024)**: An enhanced reasoning evaluation benchmark with 10 options.
- **TeachTune (Jin et al., 2025)**: Generates teaching dialogues for human evaluation, complementing the automated approach proposed in this work.
- **Insight**: Evaluating the teaching capability of LLMs might be the closest way to measure "true understanding," as teaching others is inherently harder than solving problems on one's own.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Establishes the first systematic evaluation of LLMs' teaching capabilities with a novel framework design grounded in pedagogical theory.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 14 models with thorough stability validation and ablation studies, though lacking sufficient validation with human students.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, tightly coupling theory with empirical practice.
- Value: ⭐⭐⭐⭐⭐ Reveals the decoupling of teaching capabilities from baseline knowledge levels, offering critical insights for the development of educational AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)
- [\[ACL 2025\] Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework](learning_to_align_multi-faceted_evaluation_a_unified_and_robust_framework.md)
- [\[ACL 2025\] GuessArena: Guess Who I Am? A Self-Adaptive Framework for Evaluating LLMs in Domain-Specific Knowledge and Reasoning](guessarena_guess_who_i_am_a.md)
- [\[ICLR 2026\] PACEbench: A Framework for Evaluating Practical AI Cyber-Exploitation Capabilities](../../ICLR2026/llm_evaluation/pacebench_a_framework_for_evaluating_practical_ai_cyber-exploitation_capabilitie.md)
- [\[ICCV 2025\] StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition](../../ICCV2025/llm_evaluation/streammind_unlocking_full_frame_rate_streaming_video_dialogue_through_event-gate.md)

</div>

<!-- RELATED:END -->
