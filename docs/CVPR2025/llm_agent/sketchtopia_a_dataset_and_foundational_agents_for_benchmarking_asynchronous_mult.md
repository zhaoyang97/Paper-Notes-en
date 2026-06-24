---
title: >-
  [Paper Note] Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback
description: >-
  [CVPR 2025][LLM Agent][Sketch] Proposes the Sketchtopia large-scale dataset (20K+ game sessions, 263K sketches, 916 players) and a three-component Agent framework (ActionDecider + DRAWBOT + GUESSBOT) to study asynchronous, goal-driven multimodal collaborative communication in Pictionary scenarios, introducing three new evaluation metrics: AAO, FRS, and MATS.
tags:
  - "CVPR 2025"
  - "LLM Agent"
  - "Sketch"
  - "Multimodal Communication"
  - "Asynchronous Interaction"
  - "Pictionary"
  - "Agent Evaluation"
date: 2026-05-08
content_hash: a666ae9cddc4f944
---

# Sketchtopia: A Dataset and Foundational Agents for Benchmarking Asynchronous Multimodal Communication with Iconic Feedback

**Conference**: CVPR 2025  
**Code**: [https://sketchtopia25.github.io/](https://sketchtopia25.github.io/)  
**Area**: LLM Agent  
**Keywords**: Sketch, Multimodal Communication, Asynchronous Interaction, Pictionary, Agent Evaluation

## TL;DR
Proposes the Sketchtopia large-scale dataset (20K+ game sessions, 263K sketches, 916 players) and a three-component Agent framework (ActionDecider + DRAWBOT + GUESSBOT) to study asynchronous, goal-driven multimodal collaborative communication in Pictionary scenarios, introducing three new evaluation metrics: AAO, FRS, and MATS.

## Background & Motivation
**Background**: Multimodal AI research primarily focuses on synchronous interaction scenarios (e.g., visual question answering, dialogue systems), where agents generate responses immediately after receiving inputs, and the interaction pattern is mostly turn-based.

**Limitations of Prior Work**: (a) Existing multimodal research overlooks asynchronous communication—real-world human collaboration is often non-synchronous, where both parties can act simultaneously and provide feedback at any time; (b) There is a lack of large-scale datasets to study AI's collaborative communication capability under constraints (e.g., communicating via sketches only, without linguistic descriptions of target words); (c) Existing sketch datasets are mostly static single frames (such as QuickDraw) and lack the dynamic interaction process of iterative drawing and feedback.

**Key Challenge**: To achieve effective asynchronous multimodal collaboration, agents must simultaneously possess three capabilities—generative sketching, open-ended guessing based on visual cues, and responsiveness to iconic feedback (👍👎❓)—which far exceeds the scope of conventional VQA or dialogue systems.

**Goal**: (1) Construct a large-scale asynchronous multimodal interaction dataset; (2) Design multimodal agents capable of collaborating in asynchronous scenarios; (3) Define appropriate evaluation metrics to measure the quality of asynchronous communication.

**Key Insight**: Using the Pictionary game as a medium. This scenario is inherently asynchronous (drawers and guessers can act simultaneously), multimodal (sketches + textual guesses + iconic feedback), and goal-driven (one must make the other guess the target word).

**Core Idea**: Use the Pictionary game as a benchmark for asynchronous multimodal communication research and design ActionDecider to achieve true asynchronous agent interaction.

## Method

### Overall Architecture
The Sketchtopia system consists of three parts: (1) **Dataset**: 20K+ game sessions from 916 real players, containing complete sequential data of asynchronous interactions; (2) **Agent Framework**: DRAWBOT (drawer) + GUESSBOT (guesser) + ActionDecider (asynchronous controller), three components working collaboratively to complete the Pictionary task; (3) **Evaluation System**: 800 human-agent interaction sessions + three dedicated evaluation metrics.

### Key Designs

1. **ActionDecider (Asynchronous Controller)**:

    - **Function**: Serves as a lightweight controller that continuously monitors the game state (current canvas, guessing history, feedback signals) to decide when the agent should act and what actions to execute.
    - **Mechanism**: Breaks the traditional turn-based interaction, allowing the drawer and guesser to operate concurrently in parallel. ActionDecider determines if the agent needs to act based on changes in the game state (e.g., canvas update amount, occurrence of new guesses, feedback signal triggers).
    - **Design Motivation**: Human interaction in Pictionary is naturally fluid (the drawer draws while the guesser guesses, and the drawer adjusts based on guesses). Enforcing a turn-based system would disrupt this natural flow.

2. **DRAWBOT (Drawer Agent)**:

    - **Function**: Generates sketches based on target words and iteratively refines the drawing based on the guesser's guesses and feedback.
    - **Mechanism**: Uses state-of-the-art generative models for sketch generation, dynamically adjusting the drawing strategy by combining the current canvas state and guesser feedback (👍 indicating correct direction, 👎 indicating a need to change strategy, ❓ indicating lack of understanding).
    - **Design Motivation**: Single-shot static generation cannot adapt to collaborative scenarios; the agent needs to adaptively supplement or modify the sketch based on communication feedback.

3. **GUESSBOT (Guesser Agent)**:

    - **Function**: Observes the evolving sketch and generates guesses by integrating historical interaction context.
    - **Mechanism**: Employs a retrieval-augmented framework—using visual models to understand the current sketch content and combining historical interaction data (previous guesses, received feedback) for retrieval and filtering to generate new textual guesses.
    - **Design Motivation**: Open-ended guessing requires a synthesis of visual understanding, common-sense reasoning, and the ability to eliminate refuted guesses.

4. **Sketchtopia Dataset**:

    - Scale: 20K+ game sessions, 263K sketches, 10K erase operations, 56K open-ended guesses, 19.4K iconic feedback actions.
    - Participants: 916 real players to ensure diversity of interaction patterns.
    - Characteristics: Preserves complete temporal information (precise timestamps for every action) to support asynchronous behavior analysis.

### Evaluation Metric Design
- **AAO (Asynchronous Action Overlap)**: Measures the temporal overlap of actions between agents; an AAO closer to human values indicates more natural interaction.
- **FRS (Feedback Responsiveness Score)**: Quantifies the responsiveness of agents to feedback (👍👎), i.e., whether behavior adjusts towards the target direction after receiving feedback.
- **MATS (Multimodal Action Timing Similarity)**: Compares the similarity of action timing patterns between agents and humans to evaluate the naturalness of pacing.

## Key Experimental Results

### Dataset Statistics

| Metric | Value | Description |
|------|------|------|
| Game sessions | 20K+ | Complete asynchronous interaction records |
| Participants | 916 people | Diverse player base |
| Number of sketches | 263K | Includes iterative drawing process |
| Number of guesses | 56K | Open-ended textual guesses |
| Iconic feedback | 19.4K | 👍👎❓ three types |
| Human-Agent sessions | 800 | Used for benchmarking |

### Key Findings
- AI agents still significantly lag behind humans in asynchronous communication tasks, especially with a notable gap in MATS (naturalness of temporal pacing).
- Iconic feedback has a significant positive impact on collaboration success rate—sessions with feedback have higher success rates than sessions without feedback.
- The generative sketching capability of DRAWBOT is the core bottleneck of the agent system; its sketch recognizability and iterative refinement capabilities fall far behind human drawers.
- GUESSBOT performs reasonably well on simple target words but still struggles with semantically abstract concepts (e.g., emotional words like ANGRY).
- The introduction of asynchronous control (ActionDecider) makes agent interaction closer to the natural interaction patterns of humans.

## Highlights & Insights
- The **"game-as-medium" research paradigm** is highly ingenious: Pictionary naturally possesses asynchrony, multimodality, and goal-directedness, allowing the collection of high-quality collaborative communication data without artificially constructing complex scenarios. This "game as benchmark" concept can be extended to other collaborative AI research.
- The **asynchronous architectural design of ActionDecider** is a key innovation: Most multi-agent systems employ turn-based interactions, whereas ActionDecider allows agents to autonomously decide when to act based on environment changes, which is closer to real human collaboration.
- The **three evaluation metrics (AAO/FRS/MATS)** fill the gap in evaluating asynchronous interactions, assessing not only task completion but also the naturalness and responsiveness of the interaction process.

## Limitations & Future Work
- Although Pictionary is ingenious, it is relatively simple (single target word, limited action space); its transferability to complex real-world asynchronous collaboration needs verification.
- The sketch generation capability of DRAWBOT is limited; incorporating stronger sketch generation models (such as improved versions of SketchRNN or diffusion models for sketch generation) could be considered.
- Only three types of iconic feedback (👍👎❓) are currently supported, which could be expanded to richer non-verbal feedback channels (e.g., pointing annotations, local highlighting).
- Multi-turn strategy learning remains unexplored—i.e., whether agents can learn communication strategies from historical sessions.
- The paper is not on arXiv and must be accessed through CVF, which is unfavorable for broad dissemination.

## Related Work & Insights
- **vs QuickDraw Dataset**: QuickDraw only contains single-frame sketch recognition without iterative interaction and feedback mechanisms, whereas Sketchtopia captures the complete dynamic collaborative process.
- **vs Traditional Multi-Agent Communication Research**: Works like Emergent Communication focus on agents autonomously developing communication protocols, whereas Sketchtopia investigates whether agents can understand and use existing human communication modalities (sketches + icons).
- **Inspiration for the LLM Agent field**: Asynchronous interaction and non-verbal feedback mechanisms provide valuable references for building more natural AI assistants.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Unique perspective on asynchronous multimodal communication, and the ActionDecider concept is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Large-scale dataset + 800 human-agent sessions + dedicated metrics, though it lacks comparisons with a wider range of baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Clear and comprehensive, with a well-designed project page.
- **Value**: ⭐⭐⭐ Valuable for multimodal AI collaboration evaluation, but the application scope is relatively narrow within the mainstream computer vision conference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[CVPR 2025\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](sceneassistant_a_visual_feedback_agent_for_open-vocabulary_3d_scene_generation.md)
- [\[ICLR 2026\] MMSearch-Plus: Benchmarking Provenance-Aware Search for Multimodal Browsing Agents](../../ICLR2026/llm_agent/mmsearch-plus_benchmarking_provenance-aware_search_for_multimodal_browsing_agent.md)
- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](../../NeurIPS2025/llm_agent/are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[NeurIPS 2025\] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](../../NeurIPS2025/llm_agent/t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)

</div>

<!-- RELATED:END -->
