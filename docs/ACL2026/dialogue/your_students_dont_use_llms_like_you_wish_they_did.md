---
title: >-
  [Paper Note] Your Students Don't Use LLMs Like You Wish They Did
description: >-
  [ACL2026][Dialogue Systems][Educational AI] This paper proposes six computational behavioral metrics for educational AI dialogues. Through an analysis of 500 real-world student-AI conversations…
tags:
  - "ACL2026"
  - "Dialogue Systems"
  - "Educational AI"
  - "Dialogue Evaluation"
  - "Learning-oriented"
  - "Scaffolding Resistance"
  - "Crisis-driven usage"
date: 2026-05-08
content_hash: 54ee8eba1b9a0090
---

# Your Students Don't Use LLMs Like You Wish They Did

**Conference**: ACL2026  
**arXiv**: [2604.23486](https://arxiv.org/abs/2604.23486)  
**Code**: No public code  
**Area**: Educational Dialogue Systems / Dialogue Evaluation / Learning Analytics  
**Keywords**: Educational AI, Dialogue Evaluation, Learning-oriented, Scaffolding Resistance, Crisis-driven usage

## TL;DR
This paper proposes six computational behavioral metrics for educational AI dialogues. Through an analysis of 500 real-world student-AI conversations, the study reveals that students frequently utilize LLM tools—originally intended to facilitate learning—as answer extractors. Furthermore, the mode of deployment is a stronger determinant of this misalignment than system design or student preferences.

## Background & Motivation
**Background**: Papers in educational NLP and AI tutoring typically evaluate systems using satisfaction surveys, engagement levels, message counts, and self-reported learning gains. While such evaluations indicate whether students like a tool, they rarely demonstrate whether the tool actually achieves educational objectives, such as promoting conceptual understanding, guiding reflection, or reducing the direct copying of answers.

**Limitations of Prior Work**: Educational psychology has long noted that students often mistake "smooth interaction" and "perceptual familiarity with an answer" for genuine mastery—a phenomenon known as the *illusion of fluency*. As LLM dialogue systems become more fluent and prone to providing direct answers, they are likely to achieve higher satisfaction scores while simultaneously circumventing productive struggle. Consequently, satisfaction and pedagogical effectiveness may be inversely correlated.

**Key Challenge**: While teachers want AI tutors to maintain scaffolded learning dialogues, students under pressure and deadlines often prioritize efficiency and direct answers. Traditional dialogue-level metrics might misinterpret high-frequency interaction as high-quality learning, failing to detect behavioral patterns such as "demanding direct answers," "bypassing prompts," or "last-minute cramming before exams."

**Goal**: The authors aim to provide a set of scalable computational metrics to directly measure whether students use AI systems in accordance with pedagogical intent. These metrics are designed to cover dialogue engagement, learning orientation, scaffolding resistance, assignment dependence, crisis-driven usage, and temporal concentration, with their reliability validated via human annotation.

**Key Insight**: The paper translates well-established concepts from learning analytics and educational data mining—such as gaming the system, help-seeking, and deadline procrastination—into computable NLP metrics, rather than focusing solely on evaluating the quality of individual tutor responses.

**Core Idea**: To evaluate "how students actually use LLMs" using multi-dimensional behavioral metrics, shifting educational AI evaluation from satisfaction/engagement toward *pedagogical alignment*—i.e., whether tool-use behaviors align with instructional goals.

## Method
The contribution of the paper is not a new tutor model but an evaluation framework. The input consists of student-AI dialogue logs and timestamps, and the output comprises six behavioral scores (between 0-1) or category distributions used to judge whether student behavior aligns with pedagogical intent. The authors compare turn-by-turn analysis with whole-dialogue analysis, find that the former is better at capturing fine-grained patterns like immediate answer-seeking or bypassing scaffolds.

### Overall Architecture
The framework first decomposes each student-AI conversation into student turns and AI responses, then applies rule-based detection or zero-shot LLM judgment for different metrics. CES, LOI, and SRS rely primarily on turn-by-turn analysis to determine if a student is sustaining a discussion, exploring concepts, or skipping guidance to obtain a solution. ADR utilizes both rules and LLMs for whole-dialogue analysis to identify the direct input of assignment questions. CMI and UCI utilize temporal information to compare behavioral changes between routine periods and periods near exams or deadlines.

To validate the metrics, the authors manually annotated 248 dialogues, with 100 dialogues overlapping with a second annotator to estimate human agreement. For the LLM side, GPT-4.1-mini and GPT-5 were compared across turn-by-turn and whole-dialogue granularities. Finally, these metrics were applied to 500 dialogues (12,650 messages) covering five datasets and two deployment paradigms: optional instructional scaffolding tools and unrestricted AI tools integrated into course workflows.

### Key Designs
1.  **Six behavioral metrics covering various pedagogical risks**:
	- **Function**: Decomposes "whether a student is learning" into multiple observable behavioral dimensions rather than a single satisfaction score.
	- **Mechanism**: CES measures dialogue engagement (turn count, follow-ups, context referencing, and acknowledgments); LOI measures the ratio of exploratory learning to direct answer-seeking; SRS measures student resistance to prompts, guiding questions, or Socratic scaffolding; ADR detects usage driven by assignment questions; CMI detects crisis-driven usage near exams or deadlines; UCI uses a Gini-like temporal concentration index to measure if usage is clustered in high-pressure periods.
	- **Design Motivation**: Failures in educational AI are not monolithic. Some systems result in high engagement but low learning; some are only used during finals; some are treated as assignment answer machines. Multiple metrics are necessary to distinguish these scenarios.

2.  **Turn-by-turn analysis prioritized over whole-dialogue analysis**:
	- **Function**: Captures answer extraction and scaffolding resistance as they occur turn-by-turn.
	- **Mechanism**: Turn-by-turn analysis judges for every student response whether they follow AI guidance, demand a direct answer, or ignore scaffolds. Whole-dialogue analysis compresses the entire conversation into a single judgment. Experiments show that GPT-5 turn-by-turn analysis has a higher correlation with human judgment in LOI/SRS/CES.
	- **Design Motivation**: A dialogue might appear long and active as a whole, but most turns within it might involve pressuring the AI for an answer. Pedagogical risks are often hidden in local transitions and cannot be identified by final dialogue length alone.

3.  **Continuous scores and zero-shot LLM discrimination**:
	- **Function**: Enables the metrics to transfer across courses and tools without large-scale annotated training sets.
	- **Mechanism**: All LLM metrics use zero-shot prompting without fine-tuning or few-shot examples; outputs are continuous scores (0-1) rather than hard binary classifications. For example, in SRS, direct resistance is weighted at 1.0, bypassing scaffolds at 0.5, and partial engagement is given an intermediate value.
	- **Design Motivation**: Student behavior often exists in a gray area. A student might initially attempt to understand but subsequently demand a direct answer; binary classification would erase this dynamic. Continuous scores are better suited for analyzing behavioral intensity and deployment variances.

### Loss & Training
As the paper does not train a new model, there is no supervised loss function. The evaluation side employs zero-shot LLM prompts: LOI, SRS, ADR, and CMI are primarily judged by GPT-4.1-mini or GPT-5. Some components of CES also use LLMs for binary classification, while ADR has a rule-based detection version. Rather than using data-driven methods to tune metric weights, the authors set them based on pedagogical experience (e.g., turn count has the highest weight in CES, while panic indicators and query directness shifts have the highest weights in CMI).

The trade-off of this strategy is clear: sacrificing optimal fitting for cross-course generalizability. Training a classifier for every course would make the framework a high-cost tool, whereas zero-shot prompts with human validation are more suitable for cross-scenario diagnostics.

## Key Experimental Results

### Main Results
Human validation demonstrated that GPT-5’s turn-by-turn analysis approaches human consistency across multiple metrics, whereas whole-dialogue analysis is significantly weaker. This directly supports the authors' emphasis on turn-by-turn evaluation.

| Method | LOI $r/\kappa$ | CES $r/\kappa$ | SRS $r/\kappa$ | ADR $r/\kappa$ | Key Conclusion |
|------|-------:|-------:|-------:|-------:|----------|
| GPT-4.1-mini Turn | 0.62 | 0.42 | 0.64 | - | Usable, but CES is weak |
| GPT-4.1-mini Whole | 0.33 | 0.21 | 0.25 | 0.22 | Whole dialogue loses detail |
| GPT-5 Turn | 0.72 | 0.59 | 0.67 | - | Most consistent with human |
| GPT-5 Whole | 0.47 | 0.46 | 0.49 | 0.31 | Still weaker than turn-by-turn |
| Rule-based | - | - | - | 0.35 | ADR rules are slightly usable but limited |
| Human-Human | 0.58 | 0.67 | 0.64 | 0.65 | Human agreement baseline |

After applying the metrics to 500 real dialogues, the authors found misalignment across different deployment modes: optional tools became crisis management tools during exam periods, while integrated unrestricted platforms—despite having more distributed interactions—were significantly more biased toward direct answer-seeking.

| Dataset | LOI | CES | SRS | ADR Rule | ADR-LLM | CMI | UCI |
|--------|----:|----:|----:|--------:|--------:|----:|----:|
| DrMattTabolism | 33 | 35 | 16 | 3 | 41 | 20 | 64 |
| DrNucleicAlice | 38 | 72 | 33 | 2 | 39 | 13 | 67 |
| MEDS2004 | 13 | 67 | 27 | 2 | 72 | 14 | 74 |
| OLiMent | 34 | 70 | 16 | 5 | 26 | 18 | 67 |
| StudyChat | 15 | 71 | 22 | 12 | 58 | - | 39 |

### Ablation Study
The "ablation" in this paper is reflected in the evaluation granularity, deployment paradigms, and cross-metric comparisons rather than removing model modules. The most critical comparison is the distribution of learning orientation between constrained platforms and StudyChat.

| Platform Type | Answer Seeking | Exploratory | Mixed | Explanation |
|----------|---------------:|------------:|------:|------|
| Constrained optional tools (n=400) | 66.5 | 15.5 | 18.0 | Scaffolding exists, but answer seeking is still prevalent |
| StudyChat unrestricted (n=100) | 92.0 | 2.0 | 6.0 | High availability/low friction leads almost entirely to answer extraction |

| Analysis Item | Key Finding | Description |
|--------|----------|------|
| Turn-by-turn vs Whole | GPT-5 turn-by-turn correlation of 0.72/0.67 on LOI/SRS | Whole-dialogue summaries mask local scaffold-bypassing behaviors |
| External Satisfaction Validation | No significant correlation between satisfaction and any pedagogical metrics on RECIPE4U, $|r| < 0.12$ | Student preference $\neq$ student learning |
| Crisis-driven usage | Optional tools avg UCI 0.681; DrNucleicAlice saw 59.7% of semester use in a single exam week | Optional tools often turn into emergency services before exams |
| ADR Detection | LLM false positive rate of 72% on MEDS2004 assignment dependence (vs 7% human) | Automatically judging "assignment copying" is difficult; model mistakes legitimate practice for cheating |

### Key Findings
- Students indeed "do not use LLMs the way instructors hope." Even when systems are designed with Socratic scaffolding, students demand answers, bypass prompts, or cluster usage during exam weeks.
- High engagement can be a misleading signal. StudyChat's CES was higher than constrained platforms, but its LOI was lower, indicating that more dialogue does not equal deeper learning.
- Deployment modality is more influential than system design. Optional tools cluster around deadlines, while tools integrated into the curriculum see more distributed use but may still primarily serve as assignment answer extractors.
- Academic integrity detection is not a simple LLM classification problem. The false positive/negative rates of ADR suggest that students often wrap assignment needs in natural questions, and models easily misjudge normal practice as copying.

## Highlights & Insights
- The most powerful aspect of the paper is shifting the evaluation of educational AI from "is the AI's response good?" to "does the student's behavior align with pedagogical intent?" This is much closer to the real-world challenges of the classroom than single-turn quality assessment.
- The "satisfaction-effectiveness inversion" is a crucial concept to remember. The smoother and more effortless the LLM experience, the more likely students are to bypass beneficial difficulties.
- The metric design is transferable. Similar phenomena may occur in code assistants, writing assistants, or medical Q&A: users may be satisfied with rapid task completion, even if the system's goal is learning, understanding, or safe decision-making.
- The authors do not simply moralize answer-seeking. The paper acknowledges these behaviors as rational student responses to systemic pressure, making the conclusions more balanced: the problem lies not just in "lazy students," but in deployment and evaluation design.

## Limitations & Future Work
- Data reproducibility is limited. Only approximately 20% of the StudyChat data is public; the remaining 400 dialogues cannot be released due to ethics and privacy, limiting external replication.
- The sample is primarily from STEM courses. The form of answer extraction in humanities, writing, or language learning courses may differ, requiring recalibration of metric definitions.
- The metrics depend on commercial LLMs, particularly GPT-5; the authors reported an analysis cost of approximately $145, and future model updates may affect reproducibility.
- The high error rate of ADR suggests that current methods are not yet reliable for academic integrity monitoring. Direct use of these metrics for student supervision carries a significant risk of false accusations.
- Human validation was performed by the authors, potentially representing a narrow educational culture and annotation perspective. Future work should involve external teachers, students, and cross-cultural experts.
- The paper does not directly measure test scores or long-term learning gains; longitudinal studies are needed to link these metrics to actual learning outcomes.

## Related Work & Insights
- **vs. Satisfaction Survey Evaluation**: Traditional surveys answer "do students like it," while these metrics answer "do students use it according to instructional goals." The two may be completely unrelated or even inversely related.
- **vs. Tutor Response Quality Taxonomy**: Work like Maurya et al. evaluates whether each AI tutor response possesses pedagogical capability; this paper evaluates student behavioral patterns over time as a complementary layer.
- **vs. Learning Analytics / Gaming Detection**: Educational data mining has long studied gaming, help-seeking, and procrastination; this paper contributes by migrating these concepts to the evaluation of LLM dialogue systems.
- **Insights**: Future educational AI should not just optimize for "answering better" but also adjust friction based on real-time behavioral metrics—for example, by adding prompts, requiring explanations, or delaying direct solutions when a student persistently seeks answers, or by embedding optional tools more deeply into the curriculum workflow.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Turning pedagogical alignment at the level of student behavior into computable metrics is a fresh perspective that hits a real pain point in educational LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Features real course data, human validation, and external satisfaction comparisons, though data is not fully public and learning outcome validation is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Sharp arguments, clear concepts, experimental data supports core conclusions, and the discussion avoids over-blaming students.
- Value: ⭐⭐⭐⭐⭐ Highly insightful for educational AI evaluation, classroom deployment, and dialogue system design; worthy of being an evaluation baseline for subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives](../../ICML2026/dialogue/is_your_llm_overcharging_you_tokenization_transparency_and_incentives.md)
- [\[ACL 2026\] Simulated Students in Tutoring Dialogues: Substance or Illusion?](simulated_students_in_tutoring_dialogues_substance_or_illusion.md)
- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)
- [\[ACL 2026\] Preference Learning Unlocks LLMs' Psycho-Counseling Skills](preference_learning_unlocks_llms_psycho-counseling_skills.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)

</div>

<!-- RELATED:END -->
