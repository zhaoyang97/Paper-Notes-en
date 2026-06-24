---
title: >-
  [Paper Note] GPT-4 as a Homework Tutor can Improve Student Engagement and Learning Outcomes
description: >-
  [ACL 2025][GPT-4] An 8-week randomized controlled trial (RCT) was conducted in an English as a Second Language (ESL) course at an Italian technical high school, replacing traditional homework with GPT-4 as an interactive tutoring tool. The evaluation found that the GPT-4 group improved in student engagement (significant increases in interest and helper sufficiency) and learning gains under specific conditions (Grade 3 Cohen's $d = 0.603$). The system is highly practical…
tags:
  - "ACL 2025"
  - "GPT-4"
  - "Homework Tutoring"
  - "Randomized Controlled Trial"
  - "Learning Gain"
  - "Student Engagement"
  - "ESL Instruction"
date: 2026-05-08
content_hash: 7f5d7d6812878594
---

# GPT-4 as a Homework Tutor can Improve Student Engagement and Learning Outcomes

**Conference**: ACL 2025  
**arXiv**: [2409.15981](https://arxiv.org/abs/2409.15981)  
**Code**: Not publicly available  
**Authors**: Alessandro Vanzo, Sankalan Pal Chowdhury, Mrinmaya Sachan  
**Institution**: ETH Zürich  
**Area**: Other  
**Keywords**: GPT-4, Homework Tutoring, Randomized Controlled Trial, Learning Gain, Student Engagement, ESL Instruction

## TL;DR

An 8-week randomized controlled trial (RCT) was conducted in an English as a Second Language (ESL) course at an Italian technical high school, replacing traditional homework with GPT-4 as an interactive tutoring tool. The evaluation found that the GPT-4 group improved in student engagement (significant increases in interest and helper sufficiency) and learning gains under specific conditions (Grade 3 Cohen's $d = 0.603$). The system is highly practical, requiring teachers to provide only the homework targets and descriptions. It maintained a hallucination rate of less than 1%, and all participating students expressed a desire to continue using it.

## Background & Motivation

**Bloom's 2 Sigma Problem**:
   - One-on-one tutoring can improve student performance by 2 standard deviations ($\sigma$), but it is difficult to scale.
   - Homework with grading and feedback improves performance by about $0.8\sigma$, while homework without feedback yields only $0.3\sigma$.
   - Homework in most schools currently operates in a "no-feedback" state.

**Limitations of Prior Work**:
   - Intelligent Tutoring Systems (ITS) require extensive content preparation and scale poorly.
   - Private tutoring is unaffordable for many families.
   - MOOCs lack homework feedback and corrective instruction.

**Opportunities and Concerns of LLMs**:
   - GPT-4 can generate interactive educational content at nearly zero cost.
   - Concerns: Student over-reliance, plagiarism, and hallucinated information.
   - **Severe lack of empirical research**: There are almost no real-classroom RCTs in secondary education and non-computer science subjects.

**Ours**:
   - Non-intrusive: Intervenes only in the homework segment, without altering classroom instruction.
   - Minimal engineering: Requires only prompt design, enabling seamless future upgrades to more powerful LLMs.
   - Context-aware: The prompt incorporates the teacher's educational goals.

## Method

### Overall Architecture

```text
Teacher provides homework components → GPT-4 generates tutoring strategy → GPT-4 + Strategy = Interactive Tutoring Prompt → Student online interaction
```

### Prompting Strategies

Teachers provide three components for each homework assignment (with zero additional workload):
1. **Exercise Purpose**: The educational goal of the homework (described in a few sentences)
2. **Exercise Description**: Task description
3. **Exercise Example**: A typical homework example

Two-step prompting:
1. **Strategy Generation**: Submits the homework components to GPT-4 and asks it to generate a step-by-step tutoring strategy.
2. **Tutoring Prompt**: Combines the homework components, the generated strategy, and a general task description into the final prompt.

Model used: gpt-4-0125-preview, deployed via a dedicated website.

### RCT Design

- **Location**: A technical high school in Italy, 4 classes.
- **Participants**: 76 students (39 in Grade 3 + 37 in Grade 5), ESL English courses.
- **Grouping**: Stratified random assignment based on English GPA.
- **Intervention**:
    - Treatment Group: Replaced traditional homework with GPT-4 interactive tutoring.
    - Control Group: Traditional homework.
    - Both groups shared the same classroom instruction.
- **Duration**: Originally planned for 6 weeks, extended to 8 weeks.
- **Teacher Blinding**: Teachers did not know the grouping of the students.

### Evaluation Tools

1. **Pre-test/Post-test**: 24 multiple-choice questions each, designed by the teachers.
2. **Initial Questionnaire**: Background, Self-Efficacy Scale (SESQ), ARCS framework.
3. **Weekly Questionnaire**: Interest, usefulness, completeness, helper sufficiency.
4. **Final Questionnaire**: Mirrored the initial questionnaire + treatment group feedback.

### Differences in Homework between Grade 3 and Grade 5

| Feature | Grade 3 | Grade 5 |
|------|--------|--------|
| Question Type | Objective (fill-in-the-blank/error correction), with clear answers | Open-ended (literary/historical essays), no single correct answer |
| Median Student Response Word Count | 114 | 314 |
| Median Agent Message Count | 15 | 12 |
| Interaction Mode | Attempt answer → Feedback → Retry | Draft → Iterative editing → Polishing grammar |

## Experimental Results

### Learning Gains

| Group | Cohen's $d$ | $P$-value | Conclusion |
|------|:---:|:---:|------|
| Overall | 0.251 | 0.314 | No significant difference |
| **Grade 3** | **0.603** | **0.087** | Marginally significant, treatment group performed better |
| Grade 5 | -0.004 | 0.991 | No difference |

**Analysis**: Objective questions in Grade 3 are more suitable for GPT-4 tutoring (with explicit correct/incorrect feedback), and the pre-/post-tests were also in objective formats. The open-ended homework in Grade 5 did not align with the objective test format, making it difficult to capture learning gains through the tests.

### Student Engagement (Weekly Questionnaire, Treatment vs. Control)

| Dimension | Cohen's $d$ | $P$-value | Conclusion |
|------|:---:|:---:|------|
| **Interest** | **0.593** | **0.011** | **Significant** |
| **Helper Sufficiency** | **0.586** | **0.015** | **Significant** |
| Usefulness | 0.356 | 0.125 | Not significant |
| Completeness | 0.281 | 0.234 | Not significant |

### Student Evaluation of the Tutoring System

- **32/33** of students believed the tutoring system helped them with their homework.
- **30/35** felt the tutoring system improved their performance on a practical level.
- **26/34** believed it helped them keep up with the English curriculum progress.
- **32/35 expressed a desire to continue using it** (the 3 who said "no" were all in their final year).

Most appreciated aspects of the tutoring system:
- Explanations (63%)
- Feedback and correction (57%)
- Step-by-step guidance in solving exercises (45%)
- The exercises themselves (25%)

### Secondary Analysis

#### Lower-performing Students Benefited More

- In the treatment group, pre-test scores and learning gains were **negatively correlated** ($R = -0.777, P < 0.001$).
- This contradicts previous studies (e.g., Prather et al., 2024, which suggested that higher-performing students benefit more).
- Isolated verification in the Grade 3 treatment group yielded $R = -0.667, P < 0.005$.

#### Relationship between Engagement and Learning Gains

- The volume of student typing was positively correlated with learning gains ($R = 0.316, P = 0.009$).
- The treatment group's typing volume was significantly higher than that of the control group ($d = 1.421, P < 0.001$).
- OLS regression showed that the coefficient for the treatment condition was not significant, indicating that **the effect is primarily mediated by student engagement**.

#### Extremely Low Hallucination Rate

- Only 16 issues were reported across 160 weekly feedback responses.
- Manual verification revealed only **4 errors out of 1,549 turns**.
- The hallucination rate was $< 1\%$, and the system never doubled down on incorrect explanations.

#### Test of Novelty Effect

- No downward trend over time was observed in the weekly ratings for interest, usefulness, completeness, and helper sufficiency.
- This does not support the existence of a novelty effect.

## Highlights & Insights

1. **Minimalist yet Practical Design**: It only requires teachers to provide homework goals and examples, demanding no LLM expertise or extra workload—teachers would have prepared these materials anyway.
2. **Lower-performing Students Benefited More**: This challenges the concern that "AI tutoring exacerbates educational inequality," demonstrating that GPT-4 can actually bridge the gap among students.
3. **Engagement is the Key Mediator**: The learning gains likely stem primarily from the higher engagement stimulated by GPT-4 rather than the intrinsic quality of the tutoring—implying that "making homework more engaging" is valuable in itself.
4. **Controllable Hallucinations**: In highly constrained homework scenarios (English language learning, rather than open-domain knowledge Q&A), GPT-4's hallucination rate is remarkably low.
5. **All participating students requested to continue using it**: This provides a strong signal for product-market fit.

## Limitations & Future Work

1. The sample size is small (76 students across 4 classes), which limits statistical power.
2. The study is limited to a single subject (English as a Second Language) in a single school, reducing external validity.
3. Use of ChatGPT by the control group on their own was not restricted (approximately 2/3 of Italian students might already be using it), which may have diluted the treatment effect.
4. The open-ended homework for Grade 5 was assessed using an objective pre- and post-test format, which might underestimate the tutoring effect.
5. The intervention lasted only 8 weeks, leaving long-term effects unknown.
6. Deep pedagogical indicators such as self-directed learning ability and metacognition were not measured.
7. Changes in subsequent teacher workload were not quantified (reported only qualitatively).

## Related Work & Insights

- **History of ITS**: Bloom's (1984) 2 Sigma Problem; Nwana's (1990) ITS survey; Cai et al. (2019) highlighting challenges in ITS scalability and content preparation.
- **Educational Applications of LLMs**: Kasneci et al. (2023) discussing opportunities and challenges of ChatGPT; substantial work focusing on programming tutoring (CodeHelp, CodeAid, etc.).
- **Mathematics Tutoring**: Chowdhury et al. (2024) AutoTutor + LLM; Butgereit et al. (2023) GPT-4 + WhatsApp Arabic math.
- **Empirical RCTs**: Padiyath et al. (2024), Tanay et al. (2024) primarily in Higher Education STEM domains.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐ Methodological novelty is limited (prompt engineering + RCT), but the choice of topic is highly valuable (the first RCT of LLM-based tutoring in secondary school ESL contexts).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Rigorously designed RCT, multi-dimensional evaluation (learning gains, engagement, long-term experience, hallucination rate, novelty effects), and deep secondary analysis.
- **Value**: ⭐⭐⭐⭐⭐ Minimalist design, low deployment cost, and real-classroom validation, providing direct instructional implications for educational practice.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, standard statistical analysis (Cohen's $d$, $P$-value), and comprehensive ethical considerations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can Uniform Meaning Representation Help GPT-4 Translate from Indigenous Languages?](can_uniform_meaning_representation_help_gpt-4_translate_from_indigenous_language.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)
- [\[ACL 2025\] MEXMA: Token-level Objectives Improve Sentence Representations](mexma_token-level_objectives_improve_sentence_representations.md)
- [\[ACL 2025\] MockConf: A Student Interpretation Dataset: Analysis, Word- and Span-level Alignment and Baselines](mockconf_a_student_interpretation_dataset_analysis_word-_and_span-level_alignmen.md)
- [\[ACL 2025\] Improve Rule Retrieval and Reasoning with Self-Induction and Relevance ReEstimate](improve_rule_retrieval_and_reasoning_with_self-induction_and_relevance_reestimat.md)

</div>

<!-- RELATED:END -->
