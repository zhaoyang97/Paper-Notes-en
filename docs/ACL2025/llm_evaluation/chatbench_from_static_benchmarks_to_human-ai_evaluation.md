---
title: >-
  [Paper Note] ChatBench: From Static Benchmarks to Human-AI Evaluation
description: >-
  [ACL 2025][LLM Evaluation][Benchmark Evaluation] Through user studies, this work converts the static MMLU benchmark into human-AI dialogues, constructing the ChatBench dataset (396 questions, 7,336 dialogues). It reveals that AI-alone accuracy cannot predict user-AI accuracy, and trains a user simulator that improves correlation by 22–26 percentage points, laying the foundation for scalable interactive evaluation.
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Benchmark Evaluation"
  - "Human-AI Interaction"
  - "User Simulation"
  - "MMLU"
  - "Dynamic Evaluation"
date: 2026-05-08
content_hash: 4119726bd8e78139
---

# ChatBench: From Static Benchmarks to Human-AI Evaluation

**Conference**: ACL 2025  
**arXiv**: [2504.07114](https://arxiv.org/abs/2504.07114)  
**Code**: [Available](https://huggingface.co/datasets/microsoft/ChatBench)  
**Area**: LLM Evaluation  
**Keywords**: Benchmark Evaluation, Human-AI Interaction, User Simulation, MMLU, Dynamic Evaluation

## TL;DR

Through user studies, this work converts the static MMLU benchmark into human-AI dialogues, constructing the ChatBench dataset (396 questions, 7,336 dialogues). It reveals that AI-alone accuracy cannot predict user-AI accuracy, and trains a user simulator that improves correlation by 22–26 percentage points, laying the foundation for scalable interactive evaluation.

## Background & Motivation

- Nearly 40% of US adults used generative AI in 2024, making the practical significance of LLM evaluation increasingly critical.
- Standard benchmarks (e.g., MMLU) exhibit a large gap compared to real-world user interactions:
    - **Benchmarks**: Full question text $\rightarrow$ single-letter option, in a fixed format.
    - **Real Interactions**: Varied user phrasing, incomplete information, multi-turn dialogues, and context dependency.
- Existing human-computer interaction evaluations (e.g., WildChat, ChatBot Arena, MT-Bench) are **disconnected** from standard benchmarks:
    - Distribution shift: Real user questions vs. benchmark questions.
    - Lack of ground truth: Requiring LLM-as-a-judge, which prevents direct comparison with MMLU results.
- **Core Problem**: **Can AI-alone benchmark scores predict actual performance when users collaborate with AI?**
- Lee et al. (2023) conducted similar exploration but with only 30 questions, which lacks scale and provides no simulator.

## Method

### Overall Architecture

Designing a pipeline to convert MMLU into user-AI dialogues:
1. Select high-quality questions from MMLU.
2. Collect three types of data: AI-alone (model solves independently), User-alone (user solves independently), and User-AI (user solves after chatting with the model).
3. Analyze the disparity between AI-alone and User-AI scenarios.
4. Train user simulators for scaling up.

### Key Designs

#### 1. User Study Design

**Two-stage pipeline**:
- **Phase 1**: Users answer questions independently (user-alone data).
- **Phase 2**: Users answer after chatting with the AI chatbot (user-AI data).
    - At least one message must be sent (forced interaction).
    - User confidence is recorded for each question.

**Two experimental conditions**:
- **Answer-first**: In Phase 2, users solve the question independently before chatting with the AI (within-subjects design).
- **Direct-to-AI**: In Phase 2, users chat with the AI directly (closer to real-world scenarios).

**Incentive mechanism**: A base fee of $\$5.00$ plus a $\$0.10$ bonus for each correct answer to enhance ecological validity.

#### 2. Question Selection

- 5 MMLU subsets: Elementary/High School/College Mathematics + Conceptual Physics + Moral Scenarios.
- Mathematics is chosen because it remains challenging (GPT-4o achieves 84% overall on MMLU, but only 48% on HS Math).
- Quality control: Manual labeling using MMLU-Redux + cross-verification with o1 models.
- Batched design (19 math batches, 7 physics/morality batches) to minimize variance in the number of responses per question.

#### 3. AI-Alone Evaluation Methods

Three AI-alone variants:
- **Letter-only zero-shot**: Outputs only the answer letter (standard benchmark style).
- **Letter-only few-shot**: Includes 5 MMLU dev questions as in-context examples.
- **Free-text** (novel design in this paper): No format constraints on replies, using GPT-4o to extract answers—closer to the user experience.

#### 4. User Simulator

**Two-step simulator architecture**:
- **Task 1**: Generates the user's first message given an MMLU question.
- **Task 2**: Decides whether to output the final answer or to ask a follow-up question given the dialogue history.

**Fine-tuning data construction**: Every $k$-turn user dialogue yields $k+1$ training samples.

**Fine-tuning approach**: Supervised fine-tuning of GPT-4o on ChatBench data.

### Data Scale

| Data Type | Quantity |
|---------|------|
| Total Questions | 396 |
| Tested Models | GPT-4o, Llama-3.1-8b |
| Confidence Responses | 10,828 |
| User-alone Answers | 7,148 |
| User-AI Dialogues | 7,336 |
| Total Answers | 144,000+ |

## Key Experimental Results

### Main Results: AI-alone vs. User-AI Accuracy

**Mean Absolute Error (MAE) between Letter-only few-shot and user-AI: 21 percentage points**  
**Mean Absolute Error (MAE) between Free-text and user-AI: 10 percentage points** (improved but still significantly different)

Key observations:
- Math: GPT-4o performs well in free-text, but user-AI accuracy is significantly lower than AI-alone (users introduce ambiguity).
- Llama-3.1-8b Math: The AI-alone $\rightarrow$ user-AI gap is smaller (the gap for weak models is already at the baseline).
- **While the two models differ by 25 percentage points in AI-alone accuracy, the gap boils down to only 5–9 percentage points in user-AI accuracy.**

### Question-level Correlation

| Metric | Correlation (Pearson $r$) |
|------|-------------------|
| Free-text vs. User-AI (direct-to-AI) | 0.45 |
| Free-text vs. User-AI (answer-first) | 0.46 |
| Free-text predicting user-AI improvement | 0.26–0.27 |
| Linear prediction of user-AI using User-alone + AI-alone | 0.55–0.63 |

AI-alone also fails to predict user-AI performance well at the question level.

### Only 39.8% of Dialogues "Mirror" the AI Benchmark

Conditions for interaction to mirror the AI benchmark: the user precisely copies the original question + the AI provides the answer in a single turn + the user adopts this answer.  
Most interactions do not meet these criteria—users rephrase questions, omit information, and ask multi-turn follow-ups.

### Net Effect of AI on Users

| Effect | Proportion |
|------|------|
| User error corrected by AI | 54% |
| User correct answer misled by AI | 10% |
| Reasons why AI-alone was 100% correct but user-AI failed | 67% AI did not output the correct answer (user rephrased the question) |

### User Simulator Results

| Method | GPT-4o Corr. $\uparrow$ | GPT-4o MAE $\downarrow$ | Llama Corr. $\uparrow$ | Llama MAE $\downarrow$ |
|------|-------------|------------|------------|-----------|
| Letter-only few-shot | 0.30 | 0.31 | 0.21 | 0.40 |
| Free-text | 0.49 | 0.20 | 0.61 | 0.20 |
| IQA-EVAL | 0.50 | 0.18 | 0.43 | 0.22 |
| Two-Step (un-finetuned) | 0.41 | 0.19 | 0.39 | 0.23 |
| **ChatBench-Sim (finetuned)** | **0.63** | **0.15** | **0.65** | **0.17** |

Fine-tuning improves correlation by 22–26 percentage points and reduces MAE by 21–26%.

### Ablation Study

- **Condition Comparison**: Under the answer-first condition, the accuracy gap between users and AI is smaller (since users have already thought about the problem).
- **Model Capability Comparison**: Although GPT-4o's AI-alone performance is far superior to Llama-3.1-8b, the user-AI performance gap narrows drastically.
- **Impact of User Rephrasing**: In approximately 66% of the cases where "AI should have got it right but the user-AI interaction failed", the user's initial prompt was not an exact paraphrase/copy of the original question.

### Key Findings

1. **AI-alone accuracy cannot predict user-AI accuracy**: Statistically significant differences are observed across multiple subjects.
2. **Letter-only format severely overestimates model capabilities**: The deviation from user-AI accuracy reaches up to 21 percentage points.
3. **Free-text evaluation is more realistic** but still suffers from a 10 percentage point deviation.
4. **The capability gap between the two models shrinks significantly after user interaction**: From 25pp $\rightarrow$ 5–9pp.
5. **Only 40% of user-AI dialogues align with the benchmark evaluation paradigm**.
6. **Fine-tuned user simulators can significantly improve prediction accuracy**: Offering a feasible path for scalable evaluation.

## Highlights & Insights

- First to systematically compare AI-alone and user-AI evaluation on a large scale (396 questions, 7,336 dialogues).
- The finding that "AI-alone benchmarks can mislead model selection" has direct industry implications—weaker models may perform closely to stronger models in interactive settings.
- The fine-tuning method for the user simulator is simple yet effective: decomposing a single dialogue into multiple SFT samples and designing a two-step architecture.
- Rigorous experimental design: pre-registration analysis, incentive schemes, quality control, and two experimental conditions.

## Limitations & Future Work

- Only 5 MMLU subsets were tested; generalization to other benchmarks or task types remains to be validated.
- Users were sourced from the Prolific platform, which may not represent all user cohorts.
- The simulator was only fine-tuned on ChatBench, utilizing limited training data (dialogues of 237 questions).
- The sensitivity of AI-alone results to different prompt templates was not evaluated.
- Only two models (GPT-4o and Llama-3.1-8b) were tested.
- User-AI evaluation is costly; exploring how to further reduce simulator costs is highly worthwhile.

## Related Work & Insights

- **Complementary to WildBench/ArenaHard/MT-Bench**: While these evaluate natural-occurring dialogues, they lack ground truth; ChatBench provides labeled comparisons.
- **Extension of Lee et al. (2023)**: Scaled from 30 questions to 396 questions + incorporated simulators.
- **Li et al. (2024b)** conducted similar conversions in the healthcare domain (benchmark $\rightarrow$ simulated interaction), sharing similar ideas.
- Inspiration: This method can be extended to the interactive evaluation of non-QA tasks like code generation and creative writing.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ Systematically bridges the gap between AI-alone and user-AI evaluation |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ Large-scale user study + pre-registration + simulator |
| Value | ⭐⭐⭐⭐ Direct guidance for LLM evaluation practices |
| Writing Quality | ⭐⭐⭐⭐⭐ Rigorous experimental design, thorough analysis |
| Overall Recommendation | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)
- [\[ACL 2026\] Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation](../../ACL2026/llm_evaluation/beyond_static_benchmarks_synthesizing_harmful_content_via_persona-based_simulati.md)
- [\[ICML 2026\] From Human-Level AI Tales to AI Leveling Human Scales](../../ICML2026/llm_evaluation/from_human-level_ai_tales_to_ai_leveling_human_scales.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] Revisiting 3D LLM Benchmarks: Are We Really Testing 3D Capabilities?](revisiting_3d_llm_benchmarks_are_we_really_testing_3d_capabilities.md)

</div>

<!-- RELATED:END -->
