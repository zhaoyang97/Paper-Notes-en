---
title: >-
  [Paper Note] Substance over Style: Evaluating Proactive Conversational Coaching Agents
description: >-
  [ACL 2025][LLM (Other)][Conversational Coaching] Through health coaching expert interviews and a user study (31 participants, 155 conversations), this study systematically evaluates LLM coaching agents across five different conversational styles (Directive, Interrogative, Facilitative). The findings show that users highly value core functionality (substance) and holds negative attitudes toward stylistic embellishments (style) when substance is lacking…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Conversational Coaching"
  - "Proactive Agent"
  - "User Study"
  - "Human Evaluation"
  - "Mixed-Initiative Dialogue"
date: 2026-05-08
content_hash: 0d5b4bde989f5d4a
---

# Substance over Style: Evaluating Proactive Conversational Coaching Agents

**Conference**: ACL 2025  
**arXiv**: [2503.19328](https://arxiv.org/abs/2503.19328)  
**Code**: None  
**Area**: Other  
**Keywords**: Conversational Coaching, Proactive Agent, User Study, Human Evaluation, Mixed-Initiative Dialogue  

## TL;DR

Through health coaching expert interviews and a user study (31 participants, 155 conversations), this study systematically evaluates LLM coaching agents across five different conversational styles (Directive, Interrogative, Facilitative). The findings show that users highly value core functionality (substance) and holds negative attitudes toward stylistic embellishments (style) when substance is lacking, while also revealing significant inconsistencies between first-person user evaluations and third-party expert/LLM evaluations.

## Background & Motivation

Recent NLP research has achieved remarkable progress in conversational tasks, but has primarily focused on task scenarios characterized by: a single explicit goal, a single correct answer, single-turn or short interactions, objective evaluability by third parties, and a clear interaction structure. However, coaching conversations present uniquely different challenges:

**Open-ended multi-turn interactions**: No predefined stopping conditions.

**Vastly ambiguous initial tasks**: Goals must be clarified through multiple turns of understanding the user.

**Shifting goals**: Priorities need to be dynamically adjusted.

**Potential tangents**: Users may drift off-topic during long conversations.

**Diverse preferences**: Different users prefer different conversational styles.

**Mixed-initiative**: Coaches must balance satisfying user goals with empowering users.

**Unspoken needs**: Users' real needs may not be directly articulated.

**No single correct answer**: Evaluation is inherently subjective.

These characteristics make coaching conversations one of the most challenging scenarios for proactive Agent research, and a systematic framework for design and evaluation is currently lacking.

## Method

### Overall Architecture

The study comprises three phases:

1. **Health Expert Interviews** (N=11) $\rightarrow$ Distill key coaching competencies.
2. **Design and Implementation of Five Coaching Agents** $\rightarrow$ Based on combinations of different conversational paradigms.
3. **User Study** (N=31, 155 conversations) $\rightarrow$ Multi-dimensional evaluation.

### Key Designs

#### Taxonomy of Expert Insights: Style vs Substance

Through interviews with 11 health coaches (4-46 years of experience), six key insights were identified:

**Substance (Core Functionality)**:
- **I1 Goal and Purpose Understanding**: Understand the user's goals and motivations to keep the conversation goal-oriented.
- **I2 Context Clarification**: Gather user constraints, preferences, and past attempts to personalize suggestions.
- **I3 Related Recommendations**: Provide relevant, actionable, and context-sensitive advice.
- **I4 Feedback Seeking**: Solicit user feedback and update recommendations accordingly.

**Style**:
- **I5 Active Listening**: Occasionally paraphrase to ensure correct understanding and goal alignment.
- **I6 User Empowerment**: Build trust and guide users to discover solutions on their own.

#### Three Conversational Paradigms

1. **Interrogative**: One party continuously asks questions, and the other only answers; this maximizes information acquisition but minimizes engagement.
2. **Directive**: The LLM proactively offers continuous solutions and instructions (similar to the default behavior of ChatGPT).
3. **Facilitative**: Guides users to find solutions on their own rather than directly providing answers—this is a style highly recommended by human coaches but not naturally exhibited by LLMs.

#### Five Agent Designs

Based on combinations of two dimensions:

**Coaching Expertise Variations**:
- **Base Module**: A well-intentioned coach without formal training, containing guidelines for proactive questioning but lacking defined specific topics.
- **Expert Module**: An experienced coach with clearly defined, goal-oriented questioning paths (targeting goal $\rightarrow$ constraints $\rightarrow$ preferences $\rightarrow$ barriers $\rightarrow$ recommendations), emphasizing behavior reinforcement, active listening, and user empowerment.

**Conversation Flow Variations**:
- **Probing Module**: Further clarifies when user statements are vague or uncertain.
- **Recommendation Module**: Decides when to make recommendations and seek feedback.
- **Resolution Module**: Decides when the conversation has reached a reasonable conclusion.

The final five Agents are:
1. **Base-Interrogative**: Base + Interrogative flow (accentuates I2, downplays I1/I3/I4)
2. **Expert-Interrogative**: Expert + Interrogative flow
3. **Directive**: Directive style implemented via simple prompting (representing the standard recommendation-first mode of default LLMs)
4. **Base-Facilitative**: Base + Facilitative flow (accentuates I1/I2/I4, moderate I3)
5. **Expert-Facilitative**: Expert + Facilitative flow

#### Explicit Conversation Flow Control

To achieve non-Directive conversational patterns, an **Explicit Conversation Flow** is introduced—running a series of LLM reasoning chains before generating each Agent response:

- **First Reasoning**: Outputs a binary decision (whether to probe / recommend / wrap up) after the user speaks.
- **Second Reasoning**: Generates a specific Agent response based on the decision from the previous step.
- The first reasoning of multiple modules runs in parallel. When multiple modules return a positive decision, continuing the conversation is prioritized (questioning takes precedence over recommending).

All Agents utilize Gemini 1.5 Pro as the base LM.

### Loss & Training

This paper does not involve model training. All agents are implemented via prompt engineering and multi-level LM reasoning chains; the core innovation lies in the design of the dialogue control flow rather than model parameter optimization.

## Key Experimental Results

### Main Results

**User Study Setup**:
- 31 participants, 1.5 hours per person.
- Each person conversed with each of the 5 Agents once.
- 33 open-ended health scenarios (sleep, fitness, daily habits, etc.).
- Balanced Latin square order design.

**Overall Ranking (Win Rate = Combined Share of Top 1 + Top 2 Rankings)**:

| Agent | Win Rate | Top 1 Share |
|---|---|---|
| Expert-Facilitative | **61.29%** | **41.9%** |
| Base-Facilitative | 58.06% | 25.8% |
| Directive | 41.94% | 22.6% |
| Expert-Interrogative | 35.48% | 6.5% |
| Base-Interrogative | 3.22% | 3.2% |

**Substance Dimension Win Rate**:

| Agent | Purpose | Context | Rec. | Personalized | Feedback | Avg. |
|---|---|---|---|---|---|---|
| Expert-Facilitative | 61.29 | 67.74 | 67.74 | 67.74 | 64.52 | **65.81** |
| Base-Facilitative | 51.61 | 54.84 | 51.61 | 45.16 | 54.84 | 51.61 |
| Directive | 48.39 | 41.94 | 41.94 | 45.16 | 41.94 | 43.87 |
| Expert-Interrogative | 6.45 | 6.45 | 3.23 | 9.68 | 9.68 | 7.10 |

**Style Dimension Win Rate**:

| Agent | Length | Concise | Tone | Encourage | Credibility | Empathy | Avg. |
|---|---|---|---|---|---|---|---|
| Base-Facilitative | 54.84 | 54.84 | 51.61 | 61.29 | 61.29 | 61.29 | **57.36** |
| Expert-Facilitative | 35.48 | 29.03 | 48.39 | 41.94 | 48.39 | 61.29 | 44.42 |
| Directive | 51.61 | 45.16 | 38.71 | 51.61 | 45.16 | — | — |

### Key Findings

1. **Substance > Style**: Users value core functionality far more than conversational style. While Expert-Facilitative leads substantially in the substance dimension, Base-Facilitative slightly outperforms it in the style dimension. This indicates that users are far more satisfied with coaching featuring "solid substance but average style" than "good style but lacking substance."

2. **Negative Effects of Over-Questioning**: The Interrogative agent performed the worst. User feedback: "The agent initially asked a lot of open ended questions… in the end responded with a suggestion. This made me feel the conversation was one sided." —P37. Excessive questioning left users feeling that the Agent had drifted from the original goal, thereby reducing engagement.

3. **Forced vs. Natural Ending**: Dialogues with the Interrogative agent were forced to end 58%-71% of the time, compared to only 16%-19% for the Facilitative agent. This suggests that purely interrogative conversations yield a very poor user experience in open-ended scenarios.

4. **LLM Experience Bias**: Among users with extensive LLM experience, 29.2% preferred the Directive agent, whereas 0% of users with less experience preferred it. This indicates a familiarity bias toward directive responses among heavy LLM users.

5. **Evaluation Inconsistency**: Significant differences exist between first-person user evaluations, third-person expert evaluations, and automatic LLM evaluations. LLMs perform poorly as automatic evaluators on subjective, human-centric tasks, but align reasonably well with user ratings on objective tasks.

## Highlights & Insights

1. **Core Discovery of "Substance over Style"**: The title captures the insight: when core functionality is lacking, stylistic embellishments are not only unhelpful but potentially counterproductive. This yields profound insights for LLM Agent design: full performance must be secured before pursuing conversational eloquence.
2. **Introduction of the Facilitative Paradigm**: This work identifies a third conversational paradigm—facilitative interaction—bridging the gap between Directive and Interrogative modalities, and aligning best with human coaching practices.
3. **Explicit Conversation Flow Control**: Implementing complex dialogue control logic through multi-level LLM reasoning chains successfully resolves the control flow dilemma of "when to stop asking questions vs. when to make recommendations," which is otherwise difficult to achieve in a single prompt.
4. **Contribution to Evaluation Methodology**: Systematically comparing first-person (user), third-person (expert), and automatic (LLM) evaluations reveals their inconsistencies, warning researchers not to simply rely on LLM assessments to replace human user studies.

## Limitations & Future Work

1. **Limited Sample Size**: 31 participants may not fully capture demographic nuances.
2. **Domain Specificity (Health)**: Conclusions might not directly generalize to other coaching areas (e.g., career development, academic tutoring).
3. **Single-Turn Interactions**: Each Agent interacted with each user only once, preventing the evaluation of long-term coaching relationship dynamics.
4. **Gender Bias**: 81.7% of the participants were male, which could influence conclusions regarding conversational style preferences.
5. **Gemini 1.5 Pro Specificity**: The results might be tied to the capabilities of the underlying LM, and other LMs could yield different behaviors.

## Related Work & Insights

- **Proactive Dialogue**: Extends work like proactive information seeking by Li et al. (2024) and personalized goal-oriented dialogue by Deng et al. (2024) to complex coaching scenarios.
- **Mixed-Initiative Interaction**: Traditionally dichotomized into user-initiated and system-initiated styles, the Facilitative mode presented here represents a more balanced interaction paradigm.
- **Coaching Research**: Guided coaching theory from Schwarz and Davidson (2008) provides the theoretical foundation.
- **LLM Evaluation**: While extensive studies explore LLM-as-a-judge, this paper exposes its limitations in subjective evaluation tasks.
- **Insights**: The research framework (expert interviews $\rightarrow$ Agent design $\rightarrow$ user study $\rightarrow$ multidimensional evaluation) serves as a methodological blueprint for evaluating other human-centric AI applications. For the commercialization of LLM Agents, user studies remain indispensable and cannot be replaced solely by automated evaluations.

## Rating

| Dimension | Rating (1-5) |
|------|-----------|
| Novelty | 4 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 4.5 |
| Value | 4.5 |
| Writing Quality | 4.5 |
| Overall Rating | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)
- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)
- [\[ACL 2025\] X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](xturing_enhanced_turing_test.md)

</div>

<!-- RELATED:END -->
