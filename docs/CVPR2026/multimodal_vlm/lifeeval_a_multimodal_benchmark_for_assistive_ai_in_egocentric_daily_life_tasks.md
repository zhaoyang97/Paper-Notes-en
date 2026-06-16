---
title: >-
  [Paper Note] LifeEval: A Multimodal Benchmark for Assistive AI in Egocentric Daily Life Tasks
description: >-
  [CVPR 2026][Multimodal VLM][MLLM] LifeEval constructs the first multimodal assistant evaluation benchmark oriented toward "egocentric, real-time, and task-oriented" scenarios. Using 591 Ego4D video slices and 4,075 QA pairs with reasoning chains, it examines whether 26 mainstream MLLMs can assist humans in daily tasks in real-time like a personal assis
tags:
  - CVPR 2026
  - Multimodal VLM
  - MLLM
date: 2026-05-08
content_hash: 1006cdfd719fe295
---
# LifeEval: A Multimodal Benchmark for Assistive AI in Egocentric Daily Life Tasks

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_LifeEval_A_Multimodal_Benchmark_for_Assistive_AI_in_Egocentric_Daily_CVPR_2026_paper.html)  
**Code**: TBD  
**Area**: Multimodal VLM / Egocentric Video / Evaluation Benchmark  
**Keywords**: Egocentric Video, Real-time Assistant, Multimodal Evaluation, Human-AI Collaboration, MLLM

## TL;DR
LifeEval constructs the first multimodal assistant evaluation benchmark oriented toward "egocentric, real-time, and task-oriented" scenarios. Using 591 Ego4D video slices and 4,075 QA pairs with reasoning chains, it examines whether 26 mainstream MLLMs can assist humans in daily tasks in real-time like a personal assistant across six capability dimensions (Perception/Reasoning/Retrieval/Planning/Safety/Multi-turn Collaboration). The results reveal significant collective shortcomings in dynamic reasoning and goal planning.

## Background & Motivation
**Background**: Multimodal Large Language Models (MLLMs) are being integrated into wearable devices like AI glasses and egocentric cameras. The ideal form is a real-time assistant that "watches what you are doing and provides help at any time." To measure this capability, the community has developed numerous video understanding benchmarks covering long-term video memory, comprehensive reasoning, and egocentric understanding.

**Limitations of Prior Work**: Existing benchmarks have two common flaws. First, they are almost all evaluated in **offline** settings—feeding the complete video to the model for post-hoc retrospective analysis rather than real-time perception while viewing. Second, they evaluate **isolated** understanding capabilities (recognizing objects, answering questions) with little connection to the user's specific goals and intentions.

**Key Challenge**: Current research focuses on "what the model can see and understand," but rarely asks "what the model can actually do to help people." The latter is the essence of a truly useful intelligent assistant—it must continuously align with the user's evolving intentions and provide actionable guidance. In third-person videos, the model is a detached observer, whereas egocentric interaction requires the model to be an embedded, active participant. This imposes much higher requirements on dynamic perception, causal reasoning, and adaptive response.

**Goal**: To shift the evaluation paradigm from "passive perception" to "active collaboration," specifically addressing three characteristics: task-oriented holistic evaluation, egocentric real-time streaming perception, and natural conversational human-AI interaction.

**Key Insight**: The authors argue that a good assistant's capabilities can be decomposed hierarchically, from basic perception to high-level interactive reasoning. High-quality QA data, anchored with reasoning chains and timestamps, is key to examining these capabilities.

**Core Idea**: Build a **streaming, task-oriented, multi-turn conversational** QA benchmark using Ego4D egocentric daily-life videos. Design a six-dimensional capability taxonomy and a multi-stage generation-filtering-augmentation-rewriting annotation pipeline to quantify "helpfulness."

## Method

### Overall Architecture
LifeEval is an evaluation benchmark rather than a single model. Its "method" consists of a **capability taxonomy + data construction pipeline + evaluation protocol**. The inputs are raw Ego4D egocentric videos, and the outputs are a set of 4,075 high-quality QA pairs (including multiple-choice and open-ended questions) along with evaluation results for 26 MLLMs.

The authors sampled 100 videos from Ego4D covering daily scenarios such as cooking, cleaning, handicrafts, shopping, laundry, and equipment maintenance (totaling 44.19 hours), cut into 30-second segments. Gemini 2.5 Pro was used to **directly read the videos** and generate QA pairs. These passed through four stages: "automatic + manual" quality control, difficulty augmentation, and open-ended rewriting. Evaluation uses a **streaming protocol**—models only see the frames within the question's time window. Multiple-choice questions are scored by accuracy, while open-ended questions use GPT-5 as a judge for scoring on a $0$ to $1$ scale.

The backbone of the benchmark is the six-dimensional capability taxonomy (SEP→DTR→CKR→GP→SFA→MIC), which progresses from basic perception to high-level collaboration.

### Key Designs

**1. Task-oriented Six-dimensional Capability Taxonomy: Decomposing "Helping" into Evaluable Axes**

Existing benchmarks evaluate "understanding," whereas LifeEval evaluates "helpfulness after understanding." The authors reconstruct assistant capabilities from passive understanding to active, context-aware collaboration across six depth-graded dimensions: Static Environment Perception (SEP, identifying immediately visible elements), Dynamic Task Reasoning (DTR, inferring task progress and user intent from video streams), Contextual Knowledge Retrieval (CKR, combining commonsense or manuals with current visuals), Goal-oriented Planning (GP, providing actionable next-step instructions), Safety & Feasibility Assessment (SFA, identifying risks in user actions), and Multi-turn Interactive Collaboration (MIC, maintaining context consistency and resolving ambiguity). This taxonomy decomposes the vague goal of being a "good assistant" into six diagnosable axes.

**2. Multi-stage Annotation Pipeline with Direct Video Input: Generation → Quality Check → Augmentation → Rewriting**

To obtain high-quality data cost-effectively, the authors designed a four-step pipeline centered on Gemini 2.5 Pro's video reasoning. First, **MCQs are generated directly from video**: unlike previous methods that feed text captions to LLMs (losing information), 30-second clips are fed directly to video MLLMs to produce questions, answers, reasoning chains, and timestamps. Second, **two-stage quality checks**: Gemini 2.5 Pro reviews its own output for dimension mismatch or ambiguity, followed by manual verification. Third, **controllable difficulty augmentation**: Gemini rewrites questions to be harder, requiring multi-step reasoning or subtle visual cues. Fourth, **open-ended rewriting**: MCQs are converted into concise open-ended questions. This pipeline leverages MLLM efficiency while suppressing bias through manual verification.

**3. Streaming Evaluation Protocol + Dual-format Grading: Restoring Real-time Constraints**

To approach real-world scenarios, the evaluation uses a streaming protocol: for each question, the model only receives visual content within the corresponding time window. Models without native video support sample 8 frames from that interval. Scoring differentiates two formats: MCQs are matched directly for accuracy, while open-ended questions use the LLM-as-a-Judge paradigm, where GPT-5 provides scores from $0$ to $1$ in $0.25$ increments. This grading captures nuances in partial correctness that binary scoring misses.

## Key Experimental Results

### Comparison with Existing Benchmarks
LifeEval's differentiated positioning (Excerpt from Tab.1): Unlike previous offline and isolated benchmarks, it satisfies multiple attributes: egocentric, interactive, real-time, and task-assistive.

| Benchmark | #QAs | #Clips | Egocentric | Real-time | Task Assist | Multi-turn | Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| MVBench | 4000 | 3641 | ✗ | ✗ | ✗ | ✗ | Closed |
| VideoMME | 2700 | 900 | ✗ | ✗ | ✗ | ✗ | Closed |
| EgoSchema | 5063 | 5063 | ✓ | ✗ | ✗ | ✗ | Closed |
| OmniMMI | 2290 | 1121 | ❍ | ✓ | ✗ | ❍ | Open |
| **LifeEval (Ours)** | **4075** | **591** | **✓** | **✓** | **✓** | **❍** | **Both** |

The 4,075 QA pairs are balanced across six dimensions and two question formats (MCQ 50.77%, Open 49.23%).

### Main Results (26 MLLM Evaluation)
All scores are linearly normalized to 0–100. Closed-source models lead overall, but even the strongest performers show clear drops in dynamic reasoning and planning.

| Model | MCQ Overall | Open-ended Overall | Remarks |
| :--- | :--- | :--- | :--- |
| Gemini-2.5-Pro | **85.61** | 69.32 | Highest MCQ |
| GPT-5 | 82.48 | **72.39** | Highest Open |
| GPT-5-mini | 79.85 | 67.44 | — |
| GPT-4o | 74.23 | 55.19 | — |
| Qwen3-VL Series | Best Open | Best Open | Strongest Open-source |
| Video-specific Models | Significantly Trail | Significantly Trail | Largest gap with closed-source |

### Key Findings
- **Three distinct tiers: Closed-source vs. Open-source vs. Video-specific**: In MCQ, Gemini-2.5-Pro is 13.42 points ahead of the best open-source model and 27.19 points ahead of the best video-specific model. Models dedicated to video understanding perform the worst in this "real-time helper" scenario.
- **Dimensional Weaknesses**: Models are generally strong in Contextual Knowledge Retrieval (CKR), but collective performance drops in Dynamic Task Reasoning (DTR) and Goal-oriented Planning (GP)—the exact procedural reasoning and adaptive planning capabilities required for intelligent task execution.
- **Open-ended vs. MCQ Difficulty**: Scores for open-ended questions are typically 10–15 points lower than MCQs, suggesting that generating precise, semantically grounded free-text responses remains a bottleneck.

## Highlights & Insights
- **Contribution of the Evaluation Paradigm**: Moving from "what a model understands" to "how a model helps" is a powerful shift. It forces subsequent research to optimize for real-time performance, planning, and executability rather than just comprehension metrics.
- **Direct Video-based QA Generation**: Bypassing intermediate text captions avoids information loss and text bias, ensuring QA pairs are directly aligned with raw visual streams. This "auto-review + manual check + difficulty augmentation" loop is a transferable template for high-quality multimodal data.
- **Streaming Protocol + Time-window Constraints**: These are clever means of implementing "real-time" requirements. Models cannot look at future frames or rewind perfectly, preventing "hindsight" shortcuts.

## Limitations & Future Work
- The data source is solely Ego4D, which is biased toward daily housework/manual tasks; coverage of professional scenarios like industry, medicine, or outdoors is limited.
- The Multi-turn Interaction (MIC) dimension is only "partially supported," as dialogues were largely simulated rather than generated during real-time human interaction.
- Open-ended questions rely on GPT-5 as a judge, making scores susceptible to the judge model's preferences and capability limits.
- Evaluation is primarily zero-shot; the potential for improvement through targeted fine-tuning or tool enhancement has not been fully explored.

## Related Work & Insights
- **vs. General Video Benchmarks (MVBench / Video-MME)**: These perform offline, third-person, isolated evaluations; LifeEval emphasizes egocentric, real-time streaming, and task-oriented collaboration.
- **vs. Egocentric Benchmarks (EgoSchema / EgoPlan-Bench)**: These often focus on post-hoc analysis or single capabilities (temporal understanding, planning) loosely coupled with user goals. LifeEval covers the full chain with a six-dimensional taxonomy.
- **vs. Streaming Dialogue (OmniMMI)**: While both focus on interaction, LifeEval adds the critical constraint of perceiving the world from a natural human perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First benchmark for egocentric real-time task assistants; paradigm shift is highly significant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive testing of 26 MLLMs across six dimensions and two formats.
- Writing Quality: ⭐⭐⭐⭐ Taxonomy and pipeline are clear, though some dimension definitions are slightly abstract.
- Value: ⭐⭐⭐⭐⭐ Provides a quantifiable direction for human-centric multimodal intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PAI-Bench: A Comprehensive Benchmark for Physical AI](pai-bench_a_comprehensive_benchmark_for_physical_ai.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)
- [\[ECCV 2024\] Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild](../../ECCV2024/multimodal_vlm/nymeria_a_massive_collection_of_multimodal_egocentric_daily_motion_in_the_wild.md)
- [\[CVPR 2026\] PhyCritic: Multimodal Critic Models for Physical AI](phycritic_multimodal_critic_models_for_physical_ai.md)
- [\[CVPR 2026\] Twin-T & TwintVQA: A Reliable Structure-Detail Separating VLM and a Comprehensive Benchmark for Chart and Table Tasks](twin-t_twintvqa_a_reliable_structure-detail_separating_vlm_and_a_comprehensive_b.md)

</div>

<!-- RELATED:END -->
