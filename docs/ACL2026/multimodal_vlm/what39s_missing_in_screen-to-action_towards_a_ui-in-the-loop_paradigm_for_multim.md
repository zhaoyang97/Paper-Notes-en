---
title: >-
  [Paper Note] What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning
description: >-
  [ACL 2026][Multimodal VLM][GUI Reasoning] This paper proposes the UILoop (UI-in-the-Loop) paradigm, restructuring GUI reasoning from the traditional "Screen $\rightarrow$ Action" into a cyclic "Screen $\rightarrow$ UI El…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "GUI Reasoning"
  - "UI Understanding"
  - "Reinforcement Learning Fine-tuning"
  - "Multimodal Agent"
  - "UI Element Grounding"
date: 2026-05-08
content_hash: 3ebfc6956d4391d3
---

# What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning

**Conference**: ACL 2026  
**arXiv**: [2604.06995](https://arxiv.org/abs/2604.06995)  
**Code**: None  
**Area**: Multimodal VLM / LLM Agent  
**Keywords**: GUI Reasoning, UI Understanding, Reinforcement Learning Fine-tuning, Multimodal Agent, UI Element Grounding

## TL;DR

This paper proposes the UILoop (UI-in-the-Loop) paradigm, restructuring GUI reasoning from the traditional "Screen $\rightarrow$ Action" into a cyclic "Screen $\rightarrow$ UI Element $\rightarrow$ Action" process. Through UI-element-driven reinforcement fine-tuning, the model is taught to explicitly locate, understand, and utilize key UI elements, achieving SOTA performance in GUI reasoning tasks.

## Background & Motivation

**Background**: GUI automation simulates user interaction with device screens using AI. Current methods leverage advanced MLLMs like GPT-4o and Qwen-VL to interpret user instructions and perform reasoning, but generally follow the "Screen-to-Action" paradigm—generating actions (e.g., click coordinates, input text, scrolling) directly from screen input, which is a black-box decision process.

**Limitations of Prior Work**: Existing GUI Agents have severe flaws in UI element understanding. Experiments show that advanced models score below 0.1 on average across three key dimensions (UI element localization, semantic function description, and practical usage). When correct UI descriptions are provided, reasoning performance improves significantly in all scenarios; when incorrect descriptions are provided, failure rates increase markedly. This indicates that UI element understanding is crucial for GUI reasoning yet overlooked by the current paradigm.

**Key Challenge**: The Screen-to-Action paradigm implicitly embeds UI understanding within action prediction, lacking explicit focus on UI elements. Models often fail to accurately locate key elements or understand their semantics and functions (e.g., misidentifying a scroll bar as a clickable button), leading to interaction errors and task failures.

**Goal**: To enable the model to explicitly learn the localization, semantic functions, and practical usage of UI elements, establishing an interpretable bridge between screen understanding and action execution.

**Key Insight**: UI elements are the key intermediate representation from screen to action. By having the model first identify and understand key UI elements before making decisions based on them, both reasoning accuracy and interpretability can be improved simultaneously.

**Core Idea**: Restructure GUI reasoning into a cyclic "Screen–UI Element–Action" process, using reinforcement learning to master three capabilities: Locate, Lingualize, and Leverage.

## Method

### Overall Architecture

UILoop consists of two main stages: (1) Data construction stage—designing a synthetic pipeline to build UI Comprehension-Bench (26K samples), enhancing existing GUI datasets to include localization, semantic descriptions, and usage information of key UI elements; (2) Training stage—proposing UI-element-driven Reinforcement Fine-Tuning (RFT) to train the model to master UI elements via three specialized reward functions.

### Key Designs

1.  **UI Comprehension-Bench Data Construction**:

    - **Function**: Provides the first GUI reasoning benchmark containing GT (Ground Truth) for key UI elements, supporting an interpretable "Screen $\rightarrow$ UI Element $\rightarrow$ Action" reasoning chain.
    - **Mechanism**: Collect existing datasets such as Android Control, OmniAct, and GUI-Act. Use a Set-of-Marks model (OmniParser V2) to mark positions of all recognizable UI elements, then use GPT-4o to filter out key UI elements useful for completing user instructions, supplementing them with semantic function descriptions and practical usage. The final data format is extended from the original $(I, S, a)$ to $(I, S, U^*, a)$.
    - **Design Motivation**: Existing GUI datasets only provide screen and action annotations, lacking finegrained information at the UI element level. In the 26K benchmark, there are 1,576,068 UI elements, with only 57,332 ($<4\%$) being key UI elements, making localization highly challenging.

2.  **Three-Dimensional Reward-Driven Reinforcement Fine-Tuning**:

    - **Function**: Reinforces the model's capabilities in locating, understanding semantics, and utilizing UI elements respectively.
    - **Mechanism**: Design three rewards based on the GRPO algorithm—Location Reward calculated by normalized Euclidean distance between predicted coordinates and GT; Lingualization Reward calculated by text similarity between predicted semantic descriptions and GT; Leverage Reward evaluating UI element utilization accuracy based on action type (coordinate matching for clicks, text matching for input/scroll). Total reward $r = r^{format} + \alpha_1 \cdot r^{loc} \cdot r^{lin} + \alpha_2 \cdot 1_U(r^{loc} \cdot r^{lin}) \cdot r^{lev}$, where the indicator function ensures the model learns localization and understanding before learning utilization.
    - **Design Motivation**: Traditional action prediction loss cannot explicitly optimize UI understanding. Decomposing into three independent rewards allows precise guidance for learning each capability.

3.  **UI Comprehension Evaluation Tasks**:

    - **Function**: Provides interpretable intermediate evaluation for GUI reasoning, beyond just evaluating final action accuracy.
    - **Mechanism**: Design three evaluation metrics—Locate (localization accuracy), Lingualize (semantic function understanding accuracy), and Leverage (utilization accuracy), with an overall score $\text{Overall} = \text{Locate} \times \text{Lingualize} \times \text{Leverage}$.
    - **Design Motivation**: Existing evaluations only measure final action accuracy, which is black-box and cannot diagnose which step the model fails at.

### Loss & Training

Using Qwen2.5-VL-3B and 7B as base models, RFT is performed using GRPO on the UI Comprehension-Bench training set, with 5 rollouts, training for 3-6 epochs until reward convergence. $\alpha_1 = 4$, $\alpha_2 = 5$, UI indicator threshold $\eta = 0.5$, trained on 8 A100 80G GPUs.

## Key Experimental Results

### Main Results

| Method | ScreenSpot-Pro GR | AndroidControl-High SR |
|------|------------------|----------------------|
| GPT-4o (zero-shot) | 0.8% | 21.2% |
| Qwen2.5-VL-7B (zero-shot) | 17.4% | 47.1% |
| SeeClick | 1.1% | 59.1% |
| OS-Atlas-7B | 18.9% | 29.8% |
| GUI-Owl-7B | 21.3% | 37.5% |
| Qwen2.5-VL-7B* (SFT) | 18.5% | - |
| **UILoop-3B** | - | **63.3%** |
| **UILoop-7B** | **23.6%** | **67.8%** |

| UI Comprehension Metrics | Locate | Lingualize | Leverage | Overall |
|---------------------|--------|-----------|---------|---------|
| GPT-4o | Low | Low | Low | <0.1 |
| Qwen2.5-VL-7B | Low | Low | Low | <0.1 |
| **UILoop-7B** | **Significant Gain** | **Significant Gain** | **Significant Gain** | **SOTA** |

### Ablation Study

| Configuration | AndroidControl-High SR | Description |
|------|----------------------|------|
| UILoop (Full) | 67.8% | Full model |
| w/o Location Reward | Decrease | Inaccurate localization leads to errors in subsequent steps |
| w/o Lingualization Reward | Decrease | Lack of semantic understanding leads to incorrect operations |
| w/o Leverage Reward | Decrease | Degraded element utilization capability |
| Provided correct UI description | Large Gain | Validates importance of UI understanding |
| Provided incorrect UI description | Large Decrease | Validates criticality of UI understanding |

### Key Findings

- **UI element understanding is a key bottleneck in GUI reasoning**: All models (including GPT-4o) score extremely low ($<0.1$) across the three dimensions of UI understanding, but reasoning performance improves significantly once correct UI information is provided, proving the fundamental flaw of the "Screen-to-Action" paradigm.
- **UILoop outperforms larger zero-shot models and specialized GUI agents at both 3B and 7B scales.**
- **Reinforcement learning is more suitable for learning UI understanding than supervised fine-tuning**: GRPO's group relative advantage estimation handles complex sequential decision-making in GUI reasoning more effectively.
- **Key UI elements account for less than 4% of all elements on screen**: This indicates that identifying key elements among a large number of irrelevant UI elements is a highly challenging task in itself.

## Highlights & Insights

- **Paradigm shift is persuasive**: "Screen $\rightarrow$ UI $\rightarrow$ Action" aligns better with human cognitive processes when using interfaces, where humans first identify key buttons/input boxes and understand their functions before interacting.
- **Hierarchical design of the three-dimensional reward is clever**: $1_U(r^{loc} \cdot r^{lin}) \cdot r^{lev}$ ensures the model must master localization and understanding before starting to optimize utilization, simulating the human learning sequence of "see first $\rightarrow$ then understand $\rightarrow$ finally operate."
- **UI Comprehension-Bench has long-term value as infrastructure**: 26K samples with complete GT UI elements and interpretable reasoning chains can support systematic evaluation and improvement of future GUI agents.
- The approach is transferable to other domains: Any task involving complex interface interaction (e.g., IDE automation, medical system operation) can benefit from explicit intermediate element understanding.

## Limitations & Future Work

- Reliance on GPT-4o and OmniParser V2 for constructing GT UI elements; data quality is limited by the capabilities of these tools.
- Only validated on 3B and 7B models; whether larger models still require explicit UI understanding remains to be verified.
- The current framework handles static screenshots without considering dynamic UI (animation, video stream) scenarios.
- Evaluation primarily focuses on Android and desktop apps; the complexity of Web interactions (e.g., dynamic loading, iframe nesting) might present new challenges.

## Related Work & Insights

- **vs Screen-to-Action methods (SeeClick, OS-Atlas)**: These focus on improving localization but ignore semantic functions and practical usage; UILoop's three-dimensional UI understanding is more comprehensive.
- **vs RL methods like GUI-R1, UI-R1**: These apply RL to action prediction in the "Screen-to-Action" paradigm; UILoop applies RL to the intermediate UI understanding stage, addressing a more fundamental issue.
- **vs UI-Vision, ScreenSpot-Pro**: These are benchmarks focusing only on localization; UILoop's UI Comprehension-Bench covers localization + semantics + utilization.

## Rating

- Novelty: ⭐⭐⭐⭐ Paradigm restructuring from "Screen $\rightarrow$ Action" to "Screen $\rightarrow$ UI $\rightarrow$ Action" is creative, though the basic idea is intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks and complete ablation, but lacks validation across more scales and domains.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, rich charts, and standardized formal definitions.
- Value: ⭐⭐⭐⭐ The 26K benchmark and three-dimensional evaluation system provide practical contributions to the GUI Agent community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[ACL 2026\] Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations](beyond_screenshots_evaluating_vlms_understanding_of_ui_animations.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](../../ICML2026/multimodal_vlm/learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[CVPR 2026\] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](../../CVPR2026/multimodal_vlm/widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)
- [\[ACL 2026\] Measuring What Matters Beyond Text: Evaluating Multimodal Summaries by Quality, Alignment, and Diversity (MM-Eval)](measuring_what_matters_beyond_text_evaluating_multimodal_summaries_by_quality_al.md)

</div>

<!-- RELATED:END -->
