---
title: >-
  [Paper Note] Help Me Write a Story: Evaluating LLMs' Ability to Generate Writing Feedback
description: >-
  [ACL 2025][LLM Evaluation][Writing Feedback] This paper defines the novel task of "LLM writing feedback generation," constructs a dataset (StoryFeedback) containing 1,300 stories with controlled writing defects (totaling 83K story-feedback pairs), and systematically evaluates the performance of 8 LLMs across four dimensions (specificity, correctness, error detection, and praise appropriateness) using automatic metrics and human evaluation. The study finds that while models pr…
tags:
  - "ACL 2025"
  - "LLM Evaluation"
  - "Writing Feedback"
  - "Creative Writing"
  - "Story Generation"
  - "Human Evaluation"
date: 2026-05-08
content_hash: 2d1998d55e18175a
---

# Help Me Write a Story: Evaluating LLMs' Ability to Generate Writing Feedback

**Conference**: ACL 2025  
**arXiv**: [2507.16007](https://arxiv.org/abs/2507.16007)  
**Code**: [https://github.com/google-deepmind/igen](https://github.com/google-deepmind/igen)  
**Area**: LLM Evaluation  
**Keywords**: Writing Feedback, Creative Writing, LLM Evaluation, Story Generation, Human Evaluation

## TL;DR

This paper defines the novel task of "LLM writing feedback generation," constructs a dataset (StoryFeedback) containing 1,300 stories with controlled writing defects (totaling 83K story-feedback pairs), and systematically evaluates the performance of 8 LLMs across four dimensions (specificity, correctness, error detection, and praise appropriateness) using automatic metrics and human evaluation. The study finds that while models provide specific and generally correct feedback, they often miss major writing flaws and struggle to determine when to praise.

## Background & Motivation

**Background**: The role of LLMs in creative writing assistance has received increasing attention, including text continuation and rewriting. However, the task of "generating writing feedback"—evaluating and guiding human authors instead of directly generating content—remains largely unstudied. Existing NLP feedback research primarily focuses on academic peer review or scholarly writing, leaving creative writing feedback without specialized datasets and evaluation frameworks.

**Limitations of Prior Work**: (1) Absence of dedicated creative writing feedback datasets and evaluation benchmarks; (2) Generating feedback is distinct from generating content—it requires identifying writing issues and articulating them constructively, which is a unique combination of abilities; (3) The performance boundaries of current "out-of-the-box" LLMs in providing feedback remain unclear, particularly regarding which specific dimensions are strong or weak.

**Key Challenge**: High-quality writing feedback must concurrently satisfy multiple interrelated requirements: specificity (not generic), correctness (suggestions should actually improve the writing), target-oriented focus (identifying major issues rather than trivial ones), and appropriate encouragement (avoiding unnecessary criticism on good texts, and avoiding empty praise on problematic ones). How existing models perform across these dimensions in combination is unknown.

**Goal**: (1) How to construct a writing feedback dataset suitable for large-scale automatic evaluation? (2) How do LLMs perform across the four key dimensions of feedback? (3) How do different model architectures, prompting methods, and error types affect feedback quality?

**Key Insight**: The authors design an ingenious "controlled defect injection" method. Starting from high-quality seed stories, they introduce known writing issues through three corruption techniques (backtranslation, sentence swapping, and sentence deletion). This allows precise evaluation of whether the models detect these known issues.

**Core Idea**: Constructing a controllable evaluation set by injecting known defects into high-quality stories to systematically test LLMs on four dimensions of writing feedback: specificity, correctness, error detection, and praise appropriateness.

## Method

### Overall Architecture

The workflow consists of three steps: (1) Collect 326 seed short stories from public datasets and generate approximately 1,300 stories with known defects using three automatic corruption methods; (2) Generate feedback using 8 LLMs, 4 types of prompts, and 2 few-shot configurations, obtaining 83,456 story-feedback pairs; (3) Analyze feedback quality using automatic metrics ("perfect-as-is" precision, trigram repetitiveness) and human evaluation (multi-level scoring across 7 dimensions).

### Key Designs

1. **Controlled Story Corruption**:

    - **Function**: Systematically inject known types of writing issues into high-quality stories.
    - **Mechanism**: Three corruption methods—**backtranslate** (looping English $\rightarrow$ German $\rightarrow$ English 10 times to introduce grammar/phrasing/coreference errors); **swap** (swapping two adjacent sentences to disrupt coherence/event order); **delete** (randomly deleting a sentence to cause information gaps/context breakage). Combined with original stories, this yields 4 categories (326 stories each).
    - **Design Motivation**: Using known corruption types allows precise measurement of whether the model detects the specific injected issues (relevance metric) without relying purely on subjective judgment.

2. **Multi-Dimension Human Evaluation Framework**:

    - **Function**: Define 7 evaluation dimensions and design a branching questionnaire, with each feedback annotated by 3 annotators.
    - **Mechanism**: The evaluation dimensions include sanity-check (is it valid feedback), feedback-type (positive/criticism/mixed), perfect-agree (whether saying "perfect" is accurate), correctness (whether suggestions improve the writing), error-detection (whether the main issue is identified), specificity (whether feedback is specific to the story or generic), and relevance (whether suggestions address the known corruption). The annotation workflow is conditional-branching—first categorizing feedback type and then proceeding along specific evaluation paths.
    - **Design Motivation**: Different dimensions probe different aspects of feedback capability. Hierarchical evaluation helps locate the models' advantages and shortfalls.

3. **多样化提示策略 (Prompt Variations)**:

    - **Function**: Design 4 prompting styles to test model sensitivity to instructions.
    - **Mechanism**: **BL Full** (list format + guide with writing issue categories), **BL Only** (list only without categories), **1-Sent** (one-sentence feedback), and **SpotProb** (challenge style: identify and describe the single main issue). Each has zero-shot and two-shot versions.
    - **Design Motivation**: Simulate different levels of instructions that non-expert users might provide in real scenarios.

### Loss & Training

This study focuses purely on evaluation and does not involve model training. All models are evaluated using their instruction-tuned versions out-of-the-box.

## Key Experimental Results

### Main Results

| Model | Correctness | Error Detection | Specificity | Relevance |
|------|--------|---------|--------|--------|
| GPT 4 | **0.834** | **0.757** | 0.942 | **0.593** |
| Gemini Pro | 0.792 | 0.703 | **0.953** | 0.497 |
| Gemini Flash | 0.763 | 0.687 | 0.917 | 0.481 |
| GPT 3.5 | 0.766 | 0.663 | 0.862 | 0.438 |
| Gemma 27B | 0.755 | 0.614 | 0.902 | 0.455 |
| Gemma 9B | 0.734 | 0.609 | 0.904 | 0.462 |
| Bloomz 176B | 0.316 | 0.254 | 0.460 | 0.243 |
| Bloomz 7B | 0.292 | 0.219 | 0.288 | 0.226 |

### Ablation Study

| Corruption Type | Correctness | Error Detection | Relevance |
|---------|--------|---------|--------|
| backtranslate | 0.770 | 0.712 | 0.577 |
| swap | 0.691 | 0.621 | 0.378 |
| delete | 0.666 | 0.545 | 0.367 |
| original | 0.630 | 0.485 | - |

### Key Findings

- **Specific but struggles to focus on core issues**: All mainstream models score highly on specificity ($>0.9$), indicating feedback is grounded in the specific stories. However, error-detection scores are generally low (even the best, GPT-4, achieves only 0.757), indicating models focus on minor details while overlooking major writing defects.
- **Backtranslate errors are easiest to detect**: Grammatical/phrasing issues are easier to spot than coherence/relevance issues. Structural problems introduced by delete and swap are difficult for models to identify (relevance scores of 0.37-0.38 vs. 0.58 for backtranslation).
- **Models struggle to evaluate when to praise**: When models output that a story is "perfect as is," the human agreement (perfect-agree) is low (only 0.573 for GPT-4), showing a tendency to falsely label flawed stories as perfect.
- **BL Full prompt is most effective**: Providing a categorization guide of writing issues significantly boosts error-detection (0.719 vs. 0.437 for SpotProb) and relevance (0.513 vs. 0.414).
- **Proprietary models lead by a wide margin**: GPT-4 and Gemini Pro lead across most dimensions, while the Bloomz series trails significantly.
- **Limited model scaling effects**: Performance differences between smaller and larger variants within the same family are minimal (Gemma 9B vs. 27B).

## Highlights & Insights

- **Ingenious controlled corruption paradigm**: Injecting known flaws (instead of relying on natural defects) enables scalable, quantifiable evaluation. This paradigm can be adapted to other evaluation tasks, such as code review or academic writing feedback.
- **Valuable finding on the 'missing-the-forest-for-the-trees' failure mode**: Models supply correct and specific feedback but often address trivial issues while ignoring fundamental flaws. This is a crucial warning for adopting LLMs as writing tutors.
- **Fine-grained, reusable 7-dimension evaluation framework**: The conditional branching design avoids irrelevant questions, improving annotation efficiency and quality.

## Limitations & Future Work

- Evaluation is limited to short stories ($\le 5$ sentences). Feedback capability on long-form narratives, which represents a more practical need, is untested.
- Only three basic text-manipulation corruption methods are used, ignoring deeper writing issues like plot logic or character arcs.
- Lack of validation on actual feedback efficacy—does human writing actually improve after receiving this feedback?
- Only single-turn feedback is evaluated, whereas real-world writing mentorship is a multi-turn iterative process.
- The rate of "perfect-as-is" responses varies wildly across models (Gemma $<5\%$ vs. GPT-3.5 $43\%$), limiting the comparability of the perfect-agree metric between models.
- The dataset is primarily in English, leaving cross-lingual feedback capabilities unexplored.

## Related Work & Insights

- **vs. Academic Peer Review Generation (Chamoun et al., 2024)**: Peer review targets argumentative logic and experimental design, while story feedback targets narrative coherence and prose quality. Their domains differ, but the evaluation frameworks can inspire each other.
- **vs. Text Revision Detection (Dou et al., 2022)**: Revision detection focuses on classifying error types. This work requires the model to not only identify issues but also articulate them constructively, acting as a "coach" rather than just a "referee."
- **vs. Direct Text Rewriting (Shu et al., 2024)**: Rewriting outputs corrected text directly, while feedback preserves the author's editorial agency. Combining these two complementary paradigms could yield better results.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic creative writing feedback evaluation framework; the controlled corruption evaluation paradigm is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 models $\times$ 4 prompts $\times$ 4 corruption types $\times$ 2 n-shot settings, 83K data pairs, dual evaluation (automatic and human).
- Writing Quality: ⭐⭐⭐⭐ Well-structured and systematic, rich in charts, with clearly defined dimensions.
- Value: ⭐⭐⭐⭐ Provides the first systematic benchmark for LLM-assisted writing, with practical insights from identified failure modes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] EvoWiki: Evaluating LLMs on Evolving Knowledge](evowiki_evaluating_llms_on_evolving_knowledge.md)
- [\[ICCV 2025\] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](../../ICCV2025/llm_evaluation/lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)
- [\[ACL 2025\] From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions](from_tools_to_teammates_evaluating_llms_in_multi-session_coding_interactions.md)
- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[ACL 2026\] HoWToBench: Holistic Evaluation for LLM's Capability in Human-level Writing using Tree of Writing](../../ACL2026/llm_evaluation/howtobench_holistic_evaluation_for_llms_capability_in_human-level_writing_using_.md)

</div>

<!-- RELATED:END -->
