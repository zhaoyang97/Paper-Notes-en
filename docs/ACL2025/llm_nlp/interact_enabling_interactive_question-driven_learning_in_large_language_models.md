---
title: >-
  [Paper Note] INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Interactive Learning] The INTERACT framework is proposed to simulate teacher-student dialogue, enabling a "student" LLM to learn new concepts from a "teacher" LLM by actively asking questions. Experiments on 1,347 unseen contexts demonstrate that interactive learning can improve comprehension accuracy by up to 25%, matching the static learning baseline in just 5 conversational turns.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Interactive Learning"
  - "Teacher-Student Dialogue"
  - "Concept Acquisition"
  - "Active Questioning"
  - "Knowledge Transfer"
date: 2026-05-08
content_hash: 60a310a9a0445c4a
---

# INTERACT: Enabling Interactive, Question-Driven Learning in Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2412.11388](https://arxiv.org/abs/2412.11388)  
**Code**: [github.com/aumken/interact](https://github.com/aumken/interact)  
**Area**: LLM/NLP  
**Keywords**: Interactive Learning, Teacher-Student Dialogue, Concept Acquisition, Active Questioning, Knowledge Transfer

## TL;DR

The INTERACT framework is proposed to simulate teacher-student dialogue, enabling a "student" LLM to learn new concepts from a "teacher" LLM by actively asking questions. Experiments on 1,347 unseen contexts demonstrate that interactive learning can improve comprehension accuracy by up to 25%, matching the static learning baseline in just 5 conversational turns.

## Background & Motivation

The current training paradigm of LLMs is static—they passively absorb fixed datasets and lack the ability to refine knowledge by asking and following up with questions. However, a key strategy of natural human learning is interactive: asking questions, challenging explanations, and iterating until comprehension is achieved.

The value of introducing interactive, question-driven learning to LLMs includes:
- **Knowledge-intensive domains**: LLMs can request clarification, seek missing details, and test their comprehension through dialogue rather than passively receiving data.
- **Educational scenarios**: An AI "student" can interact with a "teacher" model to address the learner's specific weak points instead of providing the same summary to everyone.
- **Professional domains**: In medicine or scientific research, iterative questioning can refine diagnoses, optimize hypotheses, and uncover overlooked details.

Core research question: **Can LLMs effectively learn new concepts through conversational interaction?** The student model constructs knowledge through active questioning rather than merely receiving summarized information.

## Method

### Overall Architecture

The INTERACT (INTERactive learning for Adaptive Concept Transfer) framework simulates teacher-student dialogue for interactive learning in LLMs:
- **Teacher** (𝒯): Possesses the full contextual information of the concepts and is responsible for answering student questions.
- **Student** (𝒮): Accesses only teacher responses (and an optional static curriculum), learning concepts by actively asking questions.
- **Evaluation**: Measures learning effectiveness through a 9-question quiz per concept (3 difficulty levels $\times$ 3 questions).

### Key Designs

1. **Dataset Construction—Ensuring Concepts are Truly Unseen**:

    - Collects content published after December 2023 (postdating the pre-training data cutoffs of the LLMs under test), covering five domains:
        - Song Lyrics (Genius, 467 samples) — learning figurative language
        - News Articles (CNN, 346 samples) — learning factual knowledge
        - Movie Plots (Wikipedia, 214 samples) — learning narrative elements
        - Academic Papers (arXiv, 170 samples) — learning technical knowledge
        - Images (COCO, 150) — learning visual analysis
    - Utilizes an adversarial filtering strategy to exclude questions that gpt-4o-mini could answer without context, ensuring that the quiz demands genuine comprehension.
    - Manual audit of a random subset: 97% of the questions meet three criteria (effectively testing comprehension, answerable from context, and requiring no external specialist knowledge).

2. **Three Interaction Scenarios**:

    - **Static Student with Curriculum**: Receives only a static curriculum summary (no conversation) before taking the quiz.
    - **Dynamic Student without Curriculum**: Starts from scratch and acquires information entirely through questioning.
    - **Dynamic Student with Curriculum**: Receives the static curriculum first, then deepens understanding through questioning.

3. **Interaction Mechanism**:

    - After each conversational turn, the student model integrates newly acquired information by appending the conversation history to its context.
    - Evaluates learning progress with the quiz after each turn to dynamically track the learning curve.
    - Temperature is set to 1.0 for dialogue generation, and 0 for quiz answering. All experiments are repeated across 3 random seeds.

4. **Five Research Questions**:

    - RQ1: How well can students learn from static curricula?
    - RQ2: How well can students learn through interaction?
    - RQ3: How do the qualities of the teacher and the curriculum affect dynamic learning?
    - RQ4: Can borrowed interaction traces substitute for active participation?
    - RQ5: What patterns exist in student-generated questions?

### Loss & Training

This work does not involve model training or fine-tuning; all experiments utilize the zero-shot and few-shot capabilities of off-the-shelf LLMs. The core evaluation metric is concept quiz accuracy—the proportion of correctly answered quiz questions.

## Key Experimental Results

### Main Results

Performance of the gpt-4o-mini student model over 5 interaction turns:

| Setting | Initial Accuracy | Final Accuracy | Gain |
|------|-----------|-----------|------|
| Dynamic Student without Curriculum | 47.91 | 73.68 | **+25.77** |
| Static Student with Curriculum | 78.83 | - | Baseline |
| Dynamic Student with Curriculum | 78.83 | 81.23 | +2.40 |
| Teacher (Upper Bound) | 90.05 | - | - |

Cross-model summary (averaged over text domains):

| Model | No Curr. Initial | No Curr. Final ($\Delta$) | Curr. Initial | Curr. Final ($\Delta$) | Teacher Perf. | Recovery Rate vs. Teacher |
|------|-----------|-------------|-----------|-------------|---------|--------------|
| gpt-4o-mini | 47.91 | 73.68 (+25.77) | 78.83 | 81.23 (+2.40) | 90.05 | 81.47% |
| LLaMA-8B | 38.12 | 60.13 (+22.01) | 70.88 | 72.43 (+1.55) | 85.70 | 75.49% |
| LLaMA-70B | 60.34 | 76.81 (+16.47) | 80.58 | 82.94 (+2.36) | 91.24 | 85.13% |
| Ministral-8B | 33.82 | 59.66 (+25.84) | 67.68 | 69.36 (+1.68) | 82.38 | 72.33% |
| Mistral-Nemo | 46.81 | 70.61 (+23.80) | 74.06 | 76.82 (+2.76) | 82.05 | 81.90% |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| gpt-4o curr. vs. LLaMA-8B curr. (Static) | 70.88 vs 65.42 (~5% diff) | Strong teacher curriculum has a clear advantage in static settings |
| gpt-4o curr. vs. LLaMA-8B curr. (Post-interaction) | 72.43 vs ~72 (<1% diff) | Interaction compensates for curriculum quality gaps |
| Weak Teacher (8B) vs. Strong Teacher (70B) | <1% difference | Teacher capability has minimal impact on interactive learning |
| Borrowed strong student (70B) interaction traces $\rightarrow$ weak student (8B) | No significant gain | Passive exposure cannot substitute for active questioning |

### Key Findings

- **Interactive learning is highly effective**: Students without curricula achieve absolute gains of 16-26% through 5 rounds of interaction, highlighting the value of active questioning in learning.
- **"Cold-start" students catch up rapidly**: Dynamic students without curricula match the static curriculum baseline within approximately 5 conversational turns.
- **Interaction mitigates gaps in curriculum and teacher quality**: Under interactive settings, the performance gap for weak teachers or weak curricula narrows to <1%, indicating that student-driven inquiry is the crux of learning.
- **Passive exposure is no substitute for active engagement**: Utilizing borrowed high-quality interaction traces does not significantly improve passive student performance. The benefit of interactive learning lies in "active questioning" rather than passive "information acquisition."
- **Students still lag behind teachers**: Despite the benefits of interaction, student models perform significantly below teachers with full context access, highlighting the need for superior interaction strategies.
- **Predictors of learning gains**: Cumulative exposure (number of unique tokens), overlap between student questions and quiz questions, semantic alignment, and response information density are key factors correlated with learning gains.

## Highlights & Insights

- **Paradigm Shift**: Shifting from passive data absorption to active conversational learning provides a brand-new perspective on knowledge acquisition for LLMs.
- **Rigorous Experimental Design**: Collecting post-cutoff training data ensures concepts are genuinely novel, and adversarial filtering eliminates guessable questions.
- **Profound Insights**: Revealing that the efficacy of learning depends on the "questioning strategy" rather than "information quality," as passive exposure to strong interaction traces yields negligible gains.
- **Cross-Domain and Cross-Modal**: Spanning five heterogeneous domains—song lyrics, news, movies, academic papers, and images—demonstrating the robust generalizability of the findings.

## Limitations & Future Work

- Only 5 turns of dialogue were evaluated; whether longer interactions can further narrow the gap with the teacher remains to be explored.
- The student models employ a fixed prompting strategy to ask questions, without exploring adaptive or meta-learning questioning strategies.
- Evaluation is limited to quiz accuracy, without reflecting long-term knowledge retention and transfer capabilities.
- The teacher model directly accesses the full context, which differs from empirical teaching by human educators.
- Parameter updating or model fine-tuning via interactive learning was not explored (currently restricted to in-context learning during inference).
- Interactive learning in the image domain rapidly saturates, suggesting a need for improved multimodal interaction strategies.

## Related Work & Insights

INTERACT bridges active learning, knowledge distillation, and conversational machine learning. Unlike traditional knowledge distillation, the student learns by actively questioning instead of passively receiving information. Educational research in human learning has shown that interactive settings consistently outperform passive instruction (Bloom 1984); this study empirically validates a similar phenomenon in LLMs. This provides theoretical and empirical foundations for downstream applications like AI tutoring systems, collaborative scientific research, and personalized education.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces human learning theory to LLM knowledge acquisition; the perspective is novel, though the methodology is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 models, 1,347 concepts, 5 domains, and 5 research questions make the evaluation systematic and comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear research questions, rigorous experimental design, and concise summaries of findings.
- Value: ⭐⭐⭐⭐ Lays the groundwork for interactive learning in LLMs, though practical deployment remains a future step.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Interactive and Expressive Code-Augmented Planning with Large Language Models](interactive_and_expressive_code-augmented_planning_with_large_language_models.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)
- [\[ACL 2025\] Planning-Driven Programming: A Large Language Model Programming Workflow](planning-driven_programming_a_large_language_model_programming_workflow.md)
- [\[ACL 2025\] WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models](warriorcoder_learning_from_expert_battles_to_augment_code_large_language_models.md)

</div>

<!-- RELATED:END -->
