---
title: >-
  [Paper Note] LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph
description: >-
  [AAAI 2026][Video Understanding][Personalized Learning] This paper proposes LOOM, an agentic pipeline system that observes users' daily LLM conversations, infers learning needs, maintains a Dynamic Learner Memory Graph, and automatically generates personalized mini-courses. LOOM unifies **continuity** (long-term progress tracking) and **initiative** (immediate responsiveness to emerging interests) in a single framework.
tags:
  - AAAI 2026
  - Video Understanding
  - Personalized Learning
  - LLM Conversations
  - Learner Memory Graph
  - Agentic Pipeline
  - Adaptive Curriculum
date: 2026-05-08
content_hash: c21fcd9ef028da6d
---

# LOOM: Personalized Learning Informed by Daily LLM Conversations Toward Long-Term Mastery via a Dynamic Learner Memory Graph

**Conference**: AAAI 2026
**arXiv**: [2511.21037](https://arxiv.org/abs/2511.21037)
**Code**: [github](https://anonymous.4open.science/r/LoomDemo)
**Area**: Video Understanding / Personalized Learning
**Keywords**: Personalized Learning, LLM Conversations, Learner Memory Graph, Agentic Pipeline, Adaptive Curriculum

## TL;DR

This paper proposes LOOM, an agentic pipeline system that observes users' daily LLM conversations, infers learning needs, maintains a Dynamic Learner Memory Graph, and automatically generates personalized mini-courses. LOOM unifies **continuity** (long-term progress tracking) and **initiative** (immediate responsiveness to emerging interests) in a single framework.

## Background & Motivation

### State of the Field

General-purpose conversational assistants such as ChatGPT have become important tools for everyday learning. However, they are **fundamentally passive** — they only respond to questions explicitly posed by users, without proactively guiding learning, tracking mastery, or helping identify knowledge gaps. Although frequent user–LLM interactions accumulate rich contextual information, this information is rarely leveraged to construct structured learning paths.

### Limitations of Prior Work

Existing LLM-based learning tools can be grouped into three categories, each with notable limitations:

**Goal-oriented tutoring systems** (ChatTutor, TutorLLM): Maintain learning paths and progress tracking, providing **continuity**, but rely on users' explicit learning goals (e.g., "learn Python") and **cannot sense emerging needs arising from users' everyday activities**.

**Opportunistic learning systems** (WaitChatter, AiGet, VocabEncounter): Embed micro-learning in idle moments with **initiative and flexibility**, but **do not track long-term mastery**; learned content remains fragmented and fails to accumulate.

**Memory-augmented assistants** (GUM, knowledge-enhanced LLMs): Maintain personal context for personalized recommendations, but **lack pedagogical functionality** — they neither track mastery nor organize curricula.

### Root Cause

**The continuity–initiative dilemma**:
- **Continuity**: Maintaining a learning plan across sessions and tracking long-term progress.
- **Initiative**: Sensing users' current activities and responding immediately to new interests.

Existing systems can only emphasize one side. Learners are left to bear the cognitive burden of integrating fragmented learning themselves — manually stitching together scattered immediate interactions into a coherent learning trajectory.

### Starting Point

The paper proposes **proactively inferring learning needs from users' daily LLM conversations**, maintaining long-term progress via a Dynamic Learner Memory Graph while remaining immediately responsive to new interests. The core idea is to transform "passive answering" into "proactive teaching companion."

## Method

### Overall Architecture

LOOM comprises a four-stage agentic pipeline, with each stage implemented by a lightweight LLM agent:

1. **Conversation Observation & Summarization**
2. **Topic Decision & Outline Generation**
3. **Course Content Generation**
4. **Progress Tracking & Graph Updates**

The central data structure is the **Dynamic Learner Memory Graph**.

### Key Designs

#### 1. Conversation Observation & Summarization (Stage 1)

**Function**: Extracts learning signals from users' daily LLM conversations without requiring explicit educational intent.

**Mechanism**:
- For each conversation, a summarization agent generates a **learner-centered one-sentence summary**.
    - Example: a user asks about K-means clustering → summarized as "how to cluster customers."
- Each summary is annotated with a **topic label** (e.g., "supervised learning") and a **difficulty estimate** (beginner / intermediate / advanced), inferred from the depth of understanding demonstrated in the conversation.
- An **activity-based filter** is maintained: only conversations referenced recently (e.g., within 10 days) are marked as "active," ensuring recommendations reflect current priorities.

**Design Motivation**: Leverages users' natural conversational behavior as implicit learning signals, avoiding the need for users to explicitly state learning goals. Temporal filtering ensures course recommendations do not draw on stale interests.

#### 2. Topic Decision & Course Outline Generation (Stage 2)

**Function**: Identifies coherent and timely learning topics based on recent conversation summaries and the Learner Memory Graph.

**Mechanism**:
- A topic-decision agent selects topics in two modes:
    - **Strengthen mode**: Selects topics that appear frequently in recent conversations to consolidate existing knowledge.
    - **Explore mode**: Introduces new concepts adjacent to existing knowledge, proactively exposing "unknown unknowns."
- A mini-course outline is generated for each topic:
    - 3–4 modules.
    - Each module is explicitly linked to recent source conversations.
    - Time estimates are included to support fragmented learning.

**Design Motivation**: A mixed-signal strategy that both consolidates existing knowledge and actively expands knowledge boundaries. Each course outline is **traceable to the user's own conversations**, ensuring personal relevance.

#### 3. Dynamic Learner Memory Graph (Core Data Structure)

**Function**: Maintains a panoramic view of the learner's knowledge across sessions, providing structured progress tracking.

**Mechanism**:
- Two-level hierarchical structure:
    - **Goal Umbrellas**: Broad thematic categories (e.g., "Machine Learning," "Decision Analysis").
    - **Courses**: Specific courses linked to goals, maintaining module-level progress.
- Dynamic reorganization mechanism:
    - After completing new courses, a reorganization agent proposes structured updates.
    - Courses may be added to existing goals, goals may be renamed (e.g., "Supervised Learning" → "Machine Learning"), or new goals may be created.
    - Each reorganization is directly anchored to recent learning activity.

**Design Motivation**: The key distinction from general memory assistants — rather than simply recording what users have done, the graph explicitly supports pedagogically meaningful learning trajectories. The simple two-level structure maintains interpretability while enabling easy user comprehension and progress visualization.

#### 4. Course Content Generation (Stage 3)

**Function**: Transforms recent conversations into personalized mini-courses.

**Mechanism**:
- Actual chat excerpts are extracted from tagged source conversations.
- Combined with the course outline, the LLM generates 3–4 modules of content.
- Each module includes concise lesson text and a short multiple-choice quiz.
- The generation agent enforces lightweight pedagogical constraints: content progresses from core concepts → application → synthesis.

**Design Motivation**: Grounding courses in **users' own conversations** maintains immediate personal relevance, while quizzes diagnose understanding and expose remaining knowledge gaps.

### Loss & Training

This paper involves no model training; instead, it relies on a **prompt engineering**-based multi-agent pipeline. Each stage uses carefully designed prompts to guide LLMs in completing specific tasks. The system's "learning" is embodied in the continuous updates to the Learner Memory Graph throughout interaction.

## Key Experimental Results

### User Study Design

- 10 participants, using LOOM for 2 days.
- Each participant created at least 10–15 new conversations.
- Post-study 10-item 7-point Likert scale survey + open-ended feedback.

### Main Results

| Evaluation Dimension | Trend | Notes |
|---|---|---|
| Usefulness (Q1) | Positive | Most participants selected Agree/Strongly Agree |
| Coherence / Relevance to Conversations (Q7) | Positive | Courses highly relevant to recent conversations |
| Motivation for Further Learning (Q9) | Positive | Users willing to continue exploring |
| Willingness to Reuse (Q10) | Positive | High overall acceptance |
| Novelty / Discovery of Knowledge Gaps (Q4, Q5) | Positive | LOOM effectively exposes useful "unknowns" |
| Trust in Correctness (Q8) | Mixed | Uncertainty regarding factual accuracy |
| Course Length / Depth Match (Q6) | Mixed | Alignment with time budget varies |
| Redundancy (Q2, reverse-scored) | Mixed | Some courses exhibit redundancy |

### Ablation Study

| Category | Representative Feedback |
|---|---|
| **Modular design appreciated** | P2: "The modular flashcards and quizzes made me feel like I actually understood the material." |
| **Strong sense of personalization** | P7: "The courses felt very personal and sometimes gave me content I hadn't thought of, yet was relevant." |
| **Progress visibility** | P5: "I liked the grouping of completed courses — it showed what I had learned." |
| **Need for better grouping/continuity** | P1: "Suggested courses should be grouped, allowing continued learning within a group." |
| **Inconsistent content quality** | P8: "The course quality was not always good; sometimes it lacked sufficient detail." |
| **Redundancy issues** | P9: "Some courses were too similar and felt redundant." |
| **Quiz control desired** | P7: "I don't always want to take quizzes — sometimes it feels unnecessary." |

### Key Findings

1. **Transforming daily conversations into courses is feasible**: Users found generated courses highly relevant to their recent conversations.
2. **Discovery of "unknown unknowns"**: The system proactively provides learning opportunities users would not have sought themselves.
3. **Balancing continuity and initiative remains difficult**: At times the system resembles a planned tutor advancing long-term goals; at other times it resembles an interruptive assistant offering content not immediately actionable.
4. **Fragility of LLM prompt pipelines**: Content quality, factual reliability, and course granularity vary considerably across participants.

## Highlights & Insights

1. **Precise problem definition**: The continuity vs. initiative framework clearly encapsulates the limitations of existing LLM learning tools.
2. **Innovative design philosophy**: Elevating "passive Q&A" to "proactive teaching companion" by leveraging users' existing conversational behavior as implicit learning signals.
3. **Learner Memory Graph**: A simple yet effective two-level knowledge graph that supports both progress tracking and dynamic reorganization.
4. **Mixed initiative mode**: The dual-mode topic selection (strengthen + explore) simultaneously consolidates and expands knowledge.
5. **Practical considerations**: Time filtering, difficulty estimation, and module time estimates are all designed for real-world usage scenarios.

## Limitations & Future Work

1. **Very small user study**: Only 10 participants over 2 days; no statistically significant conclusions can be drawn.
2. **No objective learning outcome measurement**: Lacks pre/post tests, spaced repetition, or delayed recall metrics from learning science.
3. **Complete reliance on prompt pipeline**: Content quality is highly susceptible to LLM capability fluctuations, with no auditable intermediate representations.
4. **Memory graph updates may overgeneralize**: Marking a concept as "mastered" after a single interaction is insufficiently robust.
5. **No baseline comparison**: No comparative study against fixed curriculum systems, purely manual learning, or other alternatives.
6. **Insufficient privacy consideration**: The system requires access to all of a user's LLM conversations, yet the paper does not discuss privacy protection mechanisms in depth.

### Future Directions Proposed by Authors

- **Pipeline robustness**: Introduce structured intermediate representations (learning objectives, prerequisite concepts, misconceptions to address), separating planning from text generation.
- **Mixed-initiative interaction**: Allow users to directly edit and expand the Learner Memory Graph rather than passively responding to system proposals.
- **Diversified progress evidence**: Combine query frequency, self-reports, and quiz performance as multi-signal inputs for updating mastery estimates.

## Related Work & Insights

- **Relationship to GUM (General User Model)**: GUM learns user models from everyday computer usage without teaching; LOOM adds the pedagogical dimension.
- **Distinction from ChatTutor**: ChatTutor relies on declared goals and fixed curricula; LOOM infers goals from natural conversations.
- **Application of conversation summarization**: Compressing conversations into learner-centered one-sentence summaries is applicable to other personalization systems.
- **The "interaction contract" problem**: Proactive guidance must remain aligned with content users can actually act on at the moment; otherwise it becomes noise — a common challenge for all proactive systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The problem framing (continuity vs. initiative) and the Memory Graph design are novel.
- **Experimental Thoroughness**: ⭐⭐⭐ — Only a 10-person, 2-day formative study; lacks objective learning outcome metrics.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; related work is precisely categorized.
- **Value**: ⭐⭐⭐⭐ — The direction is important, but the current work is a prototype validation with a gap to practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments](../../NeurIPS2025/video_understanding/memtrack_evaluating_long-term_memory_and_state_tracking_in_multi-platform_dynami.md)
- [\[CVPR 2026\] Temporally Consistent Long-Term Memory for 3D Single Object Tracking](../../CVPR2026/video_understanding/chronotrack_temporally_consistent_long_term_memory_for_3d_single_object_tracking.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](../../CVPR2026/video_understanding/question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)
- [\[AAAI 2026\] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems](rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm-based_conversati.md)
- [\[AAAI 2026\] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective](rethinking_progression_of_memory_state_in_robotic_manipulation_an_object-centric.md)

</div>

<!-- RELATED:END -->
