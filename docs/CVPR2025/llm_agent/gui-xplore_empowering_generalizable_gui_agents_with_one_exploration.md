---
title: >-
  [Paper Note] GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration
description: >-
  [CVPR 2025][LLM Agent][GUI Agent] Proposed the GUI-Xplore dataset (312 applications, 32K+ QA pairs, 5-level tasks) and the Xplore-Agent framework (Action-aware GUI Modeling + GUI Transition Graph Reasoning). By simulating the human strategy of "exploring before reasoning", it improves StepSR by approximately 10% on unfamiliar applications compared to state-of-the-art GUI agents.
tags:
  - "CVPR 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "cross-application generalization"
  - "exploration video"
  - "graph-guided reasoning"
  - "multi-task evaluation"
date: 2026-05-08
content_hash: 078c40798fc62b7a
---

# GUI-Xplore: Empowering Generalizable GUI Agents with One Exploration

**Conference**: CVPR 2025  
**arXiv**: [2503.17709](https://arxiv.org/abs/2503.17709)  
**Code**: To be confirmed  
**Area**: LLM Agent  
**Keywords**: GUI Agent, cross-application generalization, exploration video, graph-guided reasoning, multi-task evaluation

## TL;DR
Proposed the GUI-Xplore dataset (312 applications, 32K+ QA pairs, 5-level tasks) and the Xplore-Agent framework (Action-aware GUI Modeling + GUI Transition Graph Reasoning). By simulating the human strategy of "exploring before reasoning", it improves StepSR by approximately 10% on unfamiliar applications compared to state-of-the-art GUI agents.

## Background & Motivation
**Background**: GUI Agents aim to automate human-device interaction. Existing methods learn general GUI operation knowledge through large-scale pre-training data (such as AITW, Mind2Web) or train within interactive environments like WebArena.

**Limitations of Prior Work**: Insufficient generalization across two dimensions—(a) **Cross-application**: Applications designed by different developers exhibit structural discrepancies (interaction logic, page hierarchies). Existing datasets ignore these differences, causing severe performance drops on unfamiliar applications. (b) **Cross-task**: Most datasets focus only on a single task type (e.g., navigation/automation) and fail to evaluate the agent's comprehensive understanding of applications.

**Key Challenge**: The lack of a mechanism for agents to acquire application-specific knowledge during the inference stage. Existing pre-training paradigms only learn general knowledge during training, leaving the agent without context when facing a new application during inference.

**Goal**: (1) Enable GUI agents to adapt quickly to unfamiliar applications (cross-application generalization); (2) Evaluate the agent's multi-level GUI capabilities (cross-task generalization).

**Key Insight**: Simulating how humans interact with unfamiliar software: first exploring the application to understand its layout and functionality, and then executing specific tasks based on the acquired knowledge.

**Core Idea**: Provide the GUI agent with an exploration video of the application as prior knowledge, and model the video as a GUI Transition Graph to assist in reasoning.

## Method

### Overall Architecture
Xplore-Agent is a two-stage framework: (1) **Action-aware GUI Modeling** stage: Keyframes and action sequences are extracted from exploration videos, transforming unstructured video into structured textual exploration sequences; (2) **Graph-Guided Environment Reasoning** stage: The exploration sequence is clustered into a GUI Transition Graph, and the node and edge information of the graph serves as prompts for the LLM to complete five downstream QA tasks.

### Key Designs

1. **Action-aware Frame Extraction**:

    - **Function**: Intelligently extract keyframes marking the start and end of actions from exploration videos (rather than uniform sampling).
    - **Mechanism**: Leverage absolute luminance differences (Y-Diff) in the YUV color space to detect visual changes between adjacent frames. GUI actions are typically accompanied by distinct screen transitions, where Y-Diff peaks precisely capture action boundaries. This extracts roughly 115 keyframes from an average 20-minute video (compared to 629 frames at 1 FPS).
    - **Design Motivation**: Uniform sampling yields substantial redundancy of static frames, while the operational patterns of GUI videos (click -> transition -> static) are naturally suited for change-detection-based keyframe extraction.

2. **Exploration Sequence Generation (VH + Action Generation)**:

    - **Function**: Convert keyframes into textual representations—using an LVLM (QwenVL-7B fine-tuned on Pix2Struct objectives) to generate a simplified View Hierarchy for each keyframe, and infer action types and target elements between adjacent keyframes via a pre-trained Action Generation module.
    - **Mechanism**: Image information to text compression, preserving semantics while drastically reducing token consumption.
    - **Design Motivation**: Directly feeding around 200 keyframes into a VLM would exceed context window limits, whereas converting them to text allows highly efficient processing by the LLM.

3. **GUI Clustering + Transition Graph**:

    - **Function**: Aggregate the linear exploration sequence into a graph structure. An LVLM determines frame-by-frame whether a new keyframe belongs to an existing page node (via functional category matching); if it matches, it is classified into that node, otherwise, a new node is created. The resulting nodes represent page cluster centers (equipped with functional descriptions), and edges symbolize action transitions.
    - **Mechanism**: Utilize the semantic understanding capabilities of GPT for online clustering without requiring View Hierarchy (VH) ground truth.
    - **Design Motivation**: Linear action streams fail to capture the non-linear page transitions of applications. A graph structure is inherently suited to model many-to-many page transitions.

4. **GUI-Xplore Dataset**:

    - 312 applications (207 automatic exploration + 105 manual exploration), spanning 6 major domains and 33 subcategories.
    - 115 hours of exploration videos, averaging 23.73 minutes per application.
    - 32,569 QA pairs covering 5 hierarchal levels of tasks: Application Overview (global functionality summarization), Page Analysis (single-page functionality analysis), Application Usage (action sequence reasoning), Action Recall (temporal localization of actions), and Action Sequence Verification (action topology verification).
    - All tasks are formulated as 5-option multiple-choice questions.

### Loss & Training
- Fine-tune QwenVL-7B during the Action-aware GUI Modeling stage.
- Use GPT for page clustering and downstream QA during the Graph-Guided Reasoning stage (no training required).

## Key Experimental Results

### Main Results: Cross-App Automation

| Method | Ele. Acc. | Op. Acc. | StepSR |
|------|-----------|----------|--------|
| GPT | 5.06% | 66.12% | 4.02% |
| AUTO-UI | 7.40% | 24.87% | 2.17% |
| SeeClick | 6.64% | – | 6.64% |
| CogAgent | 17.18% | 73.54% | 15.80% |
| **Xplore-Agent** | **30.73%** | **84.63%** | **30.39%** |

### Cross-Task Performance Comparison

| Method | Overview | Page | Usage | Recall | SeqVerify | Avg. |
|------|----------|------|-------|--------|-----------|------|
| GPT (8frame) | 96.88% | 82.12% | 66.48% | 22.6% | 28.85% | 59.39% |
| VideoTree | 89.75% | 91.05% | 65.73% | 21.7% | 21.61% | 57.97% |
| **Xplore-Agent** | **99.25%** | **92.86%** | **68.21%** | **24.36%** | **36.54%** | **64.24%** |

### Ablation Study

| Configuration | Avg. Acc. | Token Count | Description |
|------|-----------|----------|------|
| w/o Clustering | - | 5,144,107 | Token explosion, unable to run |
| Rule-based Clustering | 61.87% | 63,771 | Based on VH and screenshot similarity |
| GPT Clustering (Full) | **64.24%** | 45,199 | Fewer tokens with higher accuracy |

### Key Findings
- On cross-application automation, Xplore-Agent improves StepSR by nearly 15 percentage points compared to CogAgent (15.80% → 30.39%), primarily due to the application priors provided by the exploration videos.
- All models perform significantly better in Environment Understanding (Overview, Page) than in Operational Behavior (Recall, SeqVerify), indicating that existing VLMs lack the capability to model relationships across pages and interactive behaviors.
- Increasing the number of input frames degrades end-to-end model performance, indicating that more input is not always better; information compression and structuring are critical.
- Action-aware keyframe extraction reduces the frame count from 629 to 115, slashing computational cost by 82% without performance loss.

## Highlights & Insights
- **"Explore-to-Adapt" Paradigm Innovation**: Breaks the traditional "pre-train -> infer" paradigm of GUI agents by introducing application-specific priors during inference. This mimics human behavior when familiarizing themselves with new software before execution. This paradigm might be more efficient than indefinitely scaling pre-training data.
- **Graph Structure for Environment Modeling**: The GUI Transition Graph elegantly transforms linear video streams into a structured knowledge representation. Moreover, GPT-based clustering reduces token usage while boosting accuracy compared to rule-based alternatives, showcasing the unique advantages of LLMs' semantic understanding in structuring tasks.
- **Five-Level Task Design**: Spanning from global understanding to local operations, and from static screens to temporal behaviors, establishing a comprehensive evaluation framework for GUI agent capabilities.

## Limitations & Future Work
- Exploration videos need to be pre-recorded. In real-world scenarios, agents should possess autonomous "explore-on-the-fly" capabilities, which represents a more fundamental challenge.
- Current outputs are restricted to text-based answers rather than concrete GUI operations (e.g., coordinate clicks), leaving a gap before fully realizing end-to-end GUI automation.
- Data collection poses privacy challenges, as application screenshots may contain sensitive information.
- Graph clustering depends heavily on GPT, with the performance of open-source alternatives remaining unverified.
- Automatic exploration (using DroidBot) has limited coverage, and manual exploration is costly. RL-based autonomous exploration strategies could be explored.

## Related Work & Insights
- **vs CogAgent**: CogAgent acquires general GUI capabilities through large-scale pre-training but performs poorly on unfamiliar apps (StepSR of only 15.80%). Xplore-Agent boosts this to 30.39% by leveraging exploration priors.
- **vs VideoTree**: Both are two-stage methods; however, VideoTree employs uniform sampling and general video understanding, whereas Xplore-Agent uses GUI-specific keyframe extraction coupled with graph-based reasoning, demonstrating a pronounced edge in SeqVerify (36.54% vs. 21.61%).
- **vs Mind2Web/AITW**: These datasets provide operation trajectories but overlook differences in application architecture. GUI-Xplore supplements application-level knowledge via exploration videos.

## Rating
- Novelty: ⭐⭐⭐⭐ The exploration-reasoning paradigm and GUI Transition Graph concept are novel, but the individual components within the two-stage pipeline (e.g., VH generation, GPT clustering) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across both cross-application and cross-task dimensions with complete ablations, though certain implementation details (e.g., prompt templates for GPT clustering) lack transparency.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined problems and intuitive system diagrams, though the organization of Sections 3 and 4 appears slightly redundant.
- Value: ⭐⭐⭐⭐ Proposes a new paradigm and benchmarking suite for GUI agent generalization, offering clear guidance for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] SpiritSight Agent: Advanced GUI Agent with One Look](spiritsight_agent_advanced_gui_agent_with_one_look.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](../../ACL2025/llm_agent/gui_explorer_autonomous.md)
- [\[ICCV 2025\] Less is More: Empowering GUI Agent with Context-Aware Simplification](../../ICCV2025/llm_agent/less_is_more_empowering_gui_agent_with_context-aware_simplification.md)
- [\[ACL 2025\] GUI Agents: A Survey](../../ACL2025/llm_agent/gui_agents_a_survey.md)
- [\[ICCV 2025\] UIPro: Unleashing Superior Interaction Capability for GUI Agents](../../ICCV2025/llm_agent/uipro_unleashing_superior_interaction_capability_for_gui_agents.md)

</div>

<!-- RELATED:END -->
