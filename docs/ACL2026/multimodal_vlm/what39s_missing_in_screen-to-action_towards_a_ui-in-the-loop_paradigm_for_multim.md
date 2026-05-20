---
title: >-
  [Paper Note] What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning
description: >-
  [ACL 2026][Multimodal VLM][GUI reasoning] This paper proposes UILoop (UI-in-the-Loop), a paradigm that restructures GUI reasoning from the conventional "screen→action" pipeline into a cyclic "screen→UI elements→action" p…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "GUI reasoning"
  - "UI understanding"
  - "reinforcement fine-tuning"
  - "multimodal agent"
  - "UI element grounding"
date: 2026-05-08
content_hash: 51674df19a725d26
---

# What's Missing in Screen-to-Action? Towards a UI-in-the-Loop Paradigm for Multimodal GUI Reasoning

**Conference**: ACL 2026
**arXiv**: [2604.06995](https://arxiv.org/abs/2604.06995)  
**Code**: None  
**Area**: Multimodal VLM / LLM Agent
**Keywords**: GUI reasoning, UI understanding, reinforcement fine-tuning, multimodal agent, UI element grounding

## TL;DR

This paper proposes UILoop (UI-in-the-Loop), a paradigm that restructures GUI reasoning from the conventional "screen→action" pipeline into a cyclic "screen→UI elements→action" process. Through UI element-driven reinforcement fine-tuning, the model is trained to explicitly locate, understand, and leverage key UI elements, achieving state-of-the-art performance on GUI reasoning benchmarks.

## Background & Motivation

**Background**: GUI automation employs AI to simulate user interactions with device screens. Current approaches leverage advanced MLLMs such as GPT-4o and Qwen-VL to interpret user instructions and perform reasoning, yet they uniformly follow a "Screen-to-Action" paradigm—directly generating actions (e.g., click coordinates, text input, scrolling) from screen observations—which constitutes a black-box decision process.

**Limitations of Prior Work**: Existing GUI agents exhibit critical deficiencies in UI element understanding. Experiments demonstrate that state-of-the-art models score below 0.1 on average across three key dimensions—UI element grounding, semantic function description, and practical usage. When accurate UI descriptions are provided, reasoning performance improves substantially across all scenarios; conversely, erroneous descriptions lead to significant performance degradation. This indicates that UI element understanding is essential for GUI reasoning, yet is systematically neglected by current paradigms.

**Key Challenge**: The Screen-to-Action paradigm embeds UI understanding implicitly within action prediction, lacking explicit attention to UI elements. Models frequently fail to accurately localize critical elements or comprehend their semantics and functionality (e.g., misidentifying a scrollbar as a clickable button), resulting in interaction errors and task failures.

**Goal**: To enable models to explicitly learn the grounding, semantic functionality, and practical usage of UI elements, thereby establishing an interpretable bridge between screen comprehension and action execution.

**Key Insight**: UI elements serve as the critical intermediate representation between screens and actions. By having the model first identify and understand key UI elements before making decisions based on them, both reasoning accuracy and interpretability can be simultaneously improved.

**Core Idea**: GUI reasoning is reformulated as a cyclic "screen–UI elements–action" process. Through reinforcement learning, the model acquires three capabilities with respect to UI elements: Locate, Lingualize, and Leverage.

## Method

### Overall Architecture

UILoop comprises two primary stages: (1) a data construction stage, in which a synthetic pipeline is designed to build UI Comprehension-Bench (26K samples), augmenting existing GUI datasets with grounding information, semantic descriptions, and usage details for key UI elements; and (2) a training stage, in which UI element-driven reinforcement fine-tuning (RFT) is proposed, training the model via three specialized reward functions to master UI element understanding.

### Key Designs

1. **UI Comprehension-Bench Data Construction**:

    - Function: Provides the first GUI reasoning benchmark containing ground-truth key UI elements, supporting interpretable "screen→UI elements→action" reasoning chains.
    - Mechanism: Existing datasets—including Android Control, OmniAct, and GUI-Act—are collected; a Set-of-Marks model (OmniParser V2) annotates the positions of all recognizable UI elements; GPT-4o then filters out key UI elements relevant to completing user instructions, supplementing them with semantic function descriptions and practical usage information. The final data format is extended from the original $(I, S, a)$ to $(I, S, U^*, a)$.
    - Design Motivation: Existing GUI datasets provide only screen and action annotations, lacking fine-grained UI element-level information. Among the 1,576,068 UI elements in the 26K benchmark, only 57,332 (<4%) are key UI elements, making localization highly challenging.

2. **Three-Dimensional Reward-Driven Reinforcement Fine-Tuning**:

    - Function: Independently reinforces the model's capabilities in grounding, semantic understanding, and leveraging UI elements.
    - Mechanism: Three rewards are designed based on the GRPO algorithm. The Location Reward is computed via normalized Euclidean distance between predicted coordinates and ground truth. The Lingualization Reward measures text similarity between the predicted semantic description and ground truth. The Leverage Reward evaluates the accuracy of UI element utilization according to action type (coordinate matching for clicks; text matching for input/scroll). The total reward is $r = r^{format} + \alpha_1 \cdot r^{loc} \cdot r^{lin} + \alpha_2 \cdot 1_U(r^{loc} \cdot r^{lin}) \cdot r^{lev}$, where the indicator function ensures the model first masters localization and understanding before optimizing for leveraging.
    - Design Motivation: Conventional action prediction losses cannot explicitly optimize UI understanding. Decomposing the objective into three independent rewards enables precise guidance for each capability.

3. **UI Comprehension Evaluation Tasks**:

    - Function: Provides interpretable intermediate evaluation of GUI reasoning, going beyond assessment of final action accuracy alone.
    - Mechanism: Three evaluation metrics are designed—Locate (UI element grounding accuracy), Lingualize (semantic function understanding accuracy), and Leverage (utilization accuracy)—with the final score defined as Overall = Locate × Lingualize × Leverage.
    - Design Motivation: Existing evaluations measure only final action accuracy, which is black-box in nature and cannot diagnose at which stage a model fails.

### Loss & Training

Qwen2.5-VL-3B and 7B are used as base models. RFT is conducted via GRPO on the UI Comprehension-Bench training set with 5 rollouts, training for 3–6 epochs until reward convergence. $\alpha_1 = 4$, $\alpha_2 = 5$, and the UI indicator threshold is $\eta = 0.5$. Training is performed on 8× A100 80G GPUs.

## Key Experimental Results

### Main Results

| Method | ScreenSpot-Pro GR | AndroidControl-High SR |
|--------|------------------|----------------------|
| GPT-4o (zero-shot) | 0.8% | 21.2% |
| Qwen2.5-VL-7B (zero-shot) | 17.4% | 47.1% |
| SeeClick | 1.1% | 59.1% |
| OS-Atlas-7B | 18.9% | 29.8% |
| GUI-Owl-7B | 21.3% | 37.5% |
| Qwen2.5-VL-7B* (SFT) | 18.5% | — |
| **UILoop-3B** | — | **63.3%** |
| **UILoop-7B** | **23.6%** | **67.8%** |

| UI Comprehension Metric | Locate | Lingualize | Leverage | Overall |
|------------------------|--------|-----------|---------|---------|
| GPT-4o | Low | Low | Low | <0.1 |
| Qwen2.5-VL-7B | Low | Low | Low | <0.1 |
| **UILoop-7B** | **Significant gain** | **Significant gain** | **Significant gain** | **SOTA** |

### Ablation Study

| Configuration | AndroidControl-High SR | Note |
|--------------|----------------------|------|
| UILoop (Full) | 67.8% | Full model |
| w/o Location Reward | Drops | Inaccurate grounding corrupts subsequent steps |
| w/o Lingualization Reward | Drops | Missing semantic understanding leads to misoperation |
| w/o Leverage Reward | Drops | Degraded element utilization capability |
| Oracle UI descriptions provided | Large gain | Validates importance of UI understanding |
| Incorrect UI descriptions provided | Large drop | Validates criticality of UI understanding |

### Key Findings

- **UI element understanding is the key bottleneck in GUI reasoning**: All models (including GPT-4o) achieve extremely low scores (<0.1) across all three UI understanding dimensions, yet reasoning performance improves substantially when accurate UI information is provided, demonstrating a fundamental deficiency of the Screen-to-Action paradigm.
- **UILoop outperforms larger zero-shot models and specialized GUI agents at both the 3B and 7B scales.**
- **Reinforcement learning is better suited than supervised fine-tuning for learning UI understanding**: GRPO's group-relative advantage estimation handles complex sequential decision-making in GUI reasoning more effectively.
- **Key UI elements account for less than 4% of all on-screen elements**, indicating that identifying critical elements among a large number of irrelevant ones is itself an extremely challenging task.

## Highlights & Insights

- **The paradigm shift is well-motivated**: "Screen→UI→Action" more closely mirrors the human cognitive process of interface interaction—humans also first identify key buttons or input fields, understand their function, and then act.
- **The hierarchical design of the three-dimensional reward is elegant**: $1_U(r^{loc} \cdot r^{lin}) \cdot r^{lev}$ ensures the model must first master grounding and understanding before optimizing leveraging, simulating the human learning sequence of "perceive→comprehend→act."
- **UI Comprehension-Bench has long-term value as infrastructure**: With 26K samples, complete ground-truth UI elements, and interpretable reasoning chains, it supports systematic evaluation and improvement of future GUI agents.
- The approach is transferable to other domains: any task involving complex interface interaction (e.g., IDE automation, medical system operation) can benefit from explicit intermediate element understanding.

## Limitations & Future Work

- Ground-truth UI elements are constructed using GPT-4o and OmniParser V2; data quality is therefore bounded by the capabilities of these tools.
- Validation is limited to 3B and 7B scale models; whether larger models still require explicit UI understanding remains to be investigated.
- The current framework processes static screenshots and does not account for dynamic UI scenarios (animations, video streams).
- Evaluation is conducted primarily on Android and desktop applications; the complexity of web interactions (e.g., dynamic loading, nested iframes) may introduce new challenges.

## Related Work & Insights

- **vs. Screen-to-Action methods (SeeClick, OS-Atlas)**: These methods focus on improving grounding while neglecting semantic functionality and practical usage; UILoop's three-dimensional UI understanding is more comprehensive.
- **vs. RL-based methods (GUI-R1, UI-R1)**: These apply RL to action prediction within the Screen-to-Action paradigm, whereas UILoop applies RL to the intermediate UI understanding stage, addressing a more fundamental problem.
- **vs. UI-Vision, ScreenSpot-Pro**: These are evaluation benchmarks focused solely on grounding; UILoop's UI Comprehension-Bench covers all three dimensions of grounding, semantics, and leveraging.

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm restructuring from "Screen→Action" to "Screen→UI→Action" is insightful, though the core intuition is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation with complete ablations, though validation across more scales and domains is lacking.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, figures are informative, and formal definitions are rigorous.
- Value: ⭐⭐⭐⭐ The 26K benchmark and three-dimensional evaluation framework offer practical contributions to the GUI agent community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Don't Act Blindly: Robust GUI Automation via Action-Effect Verification and Self-Correction](don39t_act_blindly_robust_gui_automation_via_action-effect_verification_and_self.md)
- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[CVPR 2026\] Widget2Code: From Visual Widgets to UI Code via Multimodal LLMs](../../CVPR2026/multimodal_vlm/widget2code_from_visual_widgets_to_ui_code_via_multimodal_llms.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
