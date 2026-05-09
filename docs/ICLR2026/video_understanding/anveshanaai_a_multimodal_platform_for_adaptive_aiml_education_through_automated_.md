---
title: >-
  [Paper Note] AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment
description: >-
  [ICLR 2026][Video Understanding][AI education] This paper presents AnveshanaAI, an adaptive AI/ML education platform grounded in Bloom's cognitive taxonomy. The system employs fine-tuned GPT-2 for automated question generation, semantic similarity-based deduplication, explainable AI (XAI) techniques for transparency, and gamification mechanisms (points/badges/leaderboards) to deliver a personalized learning assessment system spanning seven domains from data science to multimodal AI. Experiments demonstrate a significant reduction in perplexity after fine-tuning and a notable improvement in learner engagement.
tags:
  - ICLR 2026
  - Video Understanding
  - AI education
  - question generation
  - Bloom's taxonomy
  - gamification
  - explainable AI
date: 2026-05-08
content_hash: eaa23b54ee34422d
---

# AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment

**Conference**: ICLR 2026
**arXiv**: [2509.23811](https://arxiv.org/abs/2509.23811)
**Code**: None
**Area**: Video Understanding / AI Education
**Keywords**: AI education, question generation, Bloom's taxonomy, gamification, explainable AI

## TL;DR
This paper presents AnveshanaAI, an adaptive AI/ML education platform grounded in Bloom's cognitive taxonomy. The system employs fine-tuned GPT-2 for automated question generation, semantic similarity-based deduplication, explainable AI (XAI) techniques for transparency, and gamification mechanisms (points/badges/leaderboards) to deliver a personalized learning assessment system spanning seven domains from data science to multimodal AI. Experiments demonstrate a significant reduction in perplexity after fine-tuning and a notable improvement in learner engagement.

## Background & Motivation

**Background**: Demand for AI/ML education has grown explosively, yet existing online learning platforms (e.g., Coursera, Kaggle) largely rely on static question banks that cannot dynamically adjust difficulty or content to individual learners. Question coverage is limited, and transparency in the generation process is lacking.

**Limitations of Prior Work**:
   - Static question banks cannot adapt to varying learner cognitive levels — beginners and experts encounter identical questions.
   - Existing automated question generation systems lack pedagogical theory guidance, resulting in imbalanced difficulty distributions.
   - Semantic redundancy among questions is prevalent, with no effective deduplication mechanism.
   - Learning platforms overemphasize rote practice, lacking interactivity and motivation mechanisms, leading to low user retention.

**Key Challenge**: High-quality adaptive education requires large quantities of stratified questions, yet manual authoring is costly and slow; automatically generated questions are difficult to guarantee in terms of quality and pedagogical alignment.

**Goal**: To construct an end-to-end adaptive AI education platform that automatically generates multi-level questions conforming to Bloom's taxonomy, while enhancing learner engagement and trust through gamification and explainability.

**Key Insight**: Bloom's cognitive taxonomy (Remember → Understand → Apply → Analyze → Evaluate → Create) serves as the structural backbone for question stratification; LLM fine-tuning enables automatic generation; semantic similarity prevents duplication; XAI provides transparency.

**Core Idea**: Pedagogical theory (Bloom's taxonomy) + LLM fine-tuning for question generation + semantic deduplication + gamification = adaptive AI education platform.

## Method

### Overall Architecture
AnveshanaAI is a full-stack educational platform comprising the following core modules:
- **Input**: AI/ML domain knowledge corpora covering seven areas: data science, machine learning, deep learning, Transformers, generative AI, large language models, and multimodal AI.
- **Processing Pipeline**: Bloom's taxonomy annotation → GPT-2 fine-tuning for question generation → semantic similarity checking → XAI annotation → adaptive delivery.
- **Output**: Personalized learning paths + stratified assessment questions + explainable feedback.

### Key Designs

1. **Bloom's Taxonomy-Based Dataset Construction**:

    - Function: Creates question sets covering all six Bloom cognitive levels for each AI/ML knowledge point.
    - Mechanism: Each knowledge point is associated with questions at six levels — Remember (factual recall), Understand (conceptual explanation), Apply (scenario-based usage), Analyze (comparison and differentiation), Evaluate (judgment of strengths/weaknesses), and Create (designing new solutions). Questions at different levels for the same knowledge point form a progressive relationship.
    - Design Motivation: Bloom's taxonomy is a classical framework in education, ensuring complete coverage of cognitive dimensions from lower-order to higher-order thinking, with each level corresponding to a distinct depth of learning.

2. **GPT-2 Fine-Tuning for Automated Question Generation**:

    - Function: Fine-tunes GPT-2 on the Bloom-annotated dataset to automatically generate questions of appropriate difficulty given a knowledge point and cognitive level.
    - Mechanism: The tuple {domain, knowledge point, Bloom level} is used as a prompt; GPT-2 is fine-tuned to generate multiple-choice questions (question stem + options + correct answer + explanation) at the target cognitive level.
    - Design Motivation: GPT-2 offers a moderate model size and low fine-tuning cost, making it suitable for deployment on educational platforms. Fine-tuning (rather than prompt engineering) ensures consistency in generation quality and output format.

3. **Semantic Similarity Checking and Deduplication**:

    - Function: Performs semantic-level deduplication on automatically generated questions to prevent the question bank from accumulating near-duplicate items that differ only in phrasing.
    - Mechanism: The embedding similarity (e.g., cosine similarity) between a newly generated question and existing questions is computed; items exceeding a threshold are deemed duplicates and discarded.
    - Design Motivation: Traditional string matching cannot detect semantic duplication (e.g., "What is gradient descent?" and "Please explain the gradient descent algorithm"), necessitating semantic-level checking.

4. **XAI Explainability Module**:

    - Function: Provides interpretable annotations for each generated question and assessment result.
    - Mechanism: Explainable AI techniques (e.g., attention visualization, feature importance) are used to explain why a particular question was generated and why a given answer is correct.
    - Design Motivation: Educational contexts demand high transparency — learners need to understand why option A is incorrect and option B is correct, while educators need to verify the validity of generated questions.

5. **Gamification Incentive System**:

    - Function: Enhances user engagement through gamification elements including points, badges, learning streaks, and leaderboards.
    - Mechanism: Users navigate five modules — Playground (free practice), Challenges (timed challenges), Simulator (simulated environments), Dashboard (learning analytics visualization), and Community (discussion forums) — earning points and achievements upon task completion.
    - Design Motivation: Educational research demonstrates that gamification mechanisms (points, badges, leaderboards) significantly improve learning motivation and retention rates.

### Loss & Training
- GPT-2 fine-tuning employs the standard language modeling objective (causal LM loss) on the Bloom-annotated dataset.
- Perplexity is monitored as a generation quality indicator throughout training.
- Post-generation semantic similarity thresholding filters redundant questions to construct the final question bank.

## Key Experimental Results

### Main Results

| Evaluation Dimension | Metric | Result | Notes |
|---|---|---|---|
| Dataset Coverage | 7 domains × 6 Bloom levels | Broad coverage | Questions available for every domain–level combination |
| Fine-tuning Stability | Perplexity | Significant reduction | Generation quality improves stably after fine-tuning |
| Learner Engagement | Interaction frequency / completion rate | Notable improvement | Gamification effectively promotes sustained learning |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Remove gamification | Engagement drops | Validates gamification's role in user retention |
| Remove Bloom stratification | Imbalanced question distribution | LLM alone cannot guarantee balanced cognitive-level coverage |
| Remove semantic deduplication | Redundancy rate increases | Automatic generation inevitably produces semantically similar questions |

### Key Findings
- Perplexity decreases steadily after GPT-2 fine-tuning, validating the feasibility of using medium-scale language models for domain-specific question generation.
- The Bloom taxonomy framework ensures balanced question distribution from lower-order (Remember) to higher-order (Create) levels.
- Semantic similarity checking effectively filters approximately 15–20% of near-duplicate questions.
- Gamification mechanisms (streaks + badges + leaderboards) yield substantially higher sustained engagement rates compared to traditional platforms.
- XAI explanations enhance user trust in the platform's assessment outcomes.

## Highlights & Insights
- **Organic integration of pedagogical theory and LLMs**: Rather than simply prompting an LLM to generate questions, the system uses Bloom's taxonomy as a structural scaffold to guide generation across distinct cognitive levels, ensuring educational validity.
- **End-to-end system thinking**: A complete closed loop is formed from data annotation and model fine-tuning through quality control to user interaction, rather than addressing only a single component.
- **Engineering value of gamification design**: Seemingly simple mechanisms such as points, badges, and learning streaks have a substantive impact on user retention in educational platforms.
- **Lightweight approach**: Choosing GPT-2 over larger models keeps deployment costs manageable, which is important for broad accessibility in educational contexts.

## Limitations & Future Work
- **Limited technical depth**: At 11 pages and under review status, the method description leans toward system design, lacking deep technical novelty.
- **Predominantly qualitative evaluation**: Large-scale quantitative user studies are absent (e.g., A/B tests comparing effect sizes against other platforms).
- **Outdated generative model**: GPT-2 lags behind GPT-3.5/4 or LLaMA-series models in generation quality; upgrading the backbone model could substantially improve performance.
- **Language limitations**: The current system focuses on English-language AI/ML content, with no multilingual or cross-disciplinary extension.
- **Simplistic adaptive strategy**: Difficulty adaptation relies primarily on linear progression through Bloom levels, without incorporating more sophisticated knowledge tracing models (e.g., BKT, DKT).
- **Lack of comparison with ChatGPT-style tutoring**: No comparison is made against approaches that directly use GPT-4 for tutoring.

## Related Work & Insights
- **Intelligent Tutoring Systems (ITS)**: Platforms such as Khan Academy and Duolingo have long employed adaptive learning, but largely depend on manually authored question banks.
- **Automatic Question Generation**: The NLP literature on question generation is rich but predominantly focused on reading comprehension, with little work specifically targeting STEM education.
- **Bloom's Taxonomy in AI Education**: Combining this classical pedagogical framework with modern AI represents a practically valuable direction.
- **Insight**: For vertical-domain AI education tools, pedagogical theory constraints may matter more than simply scaling up LLM capacity.

## Rating
- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value: ⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[ICLR 2026\] Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences](paper_copilot_tracking_the_evolution_of_peer_review_in_ai_conferences.md)
- [\[AAAI 2026\] EmoVid: A Multimodal Emotion Video Dataset for Emotion-Centric Video Understanding and Generation](../../AAAI2026/video_understanding/emovid_a_multimodal_emotion_video_dataset_for_emotion-centric_video_understandin.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](../../CVPR2026/video_understanding/movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[ICLR 2026\] AdAEM: An Adaptively and Automated Extensible Measurement of LLMs' Value Difference](adaem_an_adaptively_and_automated_extensible_measurement_of_llms_value_differenc.md)

</div>

<!-- RELATED:END -->
