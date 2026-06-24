---
title: >-
  [Paper Note] Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction
description: >-
  [ICML 2025][LLM Agent][GUI Agent] Proposes Aguvis, the first fully pure vision-based cross-platform autonomous GUI Agent framework. By unifying visual observation space, standardizing action spaces, and utilizing an inner monologue mechanism, Aguvis achieves SOTA results on offline and online benchmarks without relying on closed-source models.
tags:
  - "ICML 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "Pure Vision"
  - "Cross-Platform"
  - "Inner Monologue"
  - "Two-Stage Training"
date: 2026-05-08
content_hash: 3d729275f929d4e4
---

# Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction

**Conference**: ICML 2025

**arXiv**: [2412.04454](https://arxiv.org/abs/2412.04454)

**Authors**: Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, Caiming Xiong (HKU & Salesforce Research)

**Area**: LLM Agent

**Keywords**: GUI Agent, Pure Vision, Cross-Platform, Inner Monologue, Two-Stage Training

**Code**: [GitHub](https://github.com/xlang-ai/aguvis) | [Project Page](https://aguvis-project.github.io/)

---

## TL;DR

Proposes Aguvis, the first fully pure vision-based cross-platform autonomous GUI Agent framework. By unifying visual observation space, standardizing action spaces, and utilizing an inner monologue mechanism, Aguvis achieves SOTA results on offline and online benchmarks without relying on closed-source models.

## Background & Motivation

Existing GUI automation methods face three core bottlenecks:

**Dependency on textual representations**: Most GUI agents require textual structures such as HTML DOM trees or accessibility trees. However, these representations are not universal across platforms, contain redundant information, and struggle to capture visual layouts.

**Fragmented action spaces**: Web, desktop, and mobile platforms each define different sets of interactive operations, making cross-platform migration difficult for models.

**Insufficient reasoning capability**: Existing methods lack explicit planning and reasoning chains, frequently failing in complex multi-step GUI tasks.

The authors observe that humans operate GUIs primarily relying on visual perception rather than underlying code structures. Therefore, taking a "pure vision" approach by unifying inputs across all platforms into screenshot images is a more natural scheme with superior generalization ability.

## Method

### Overall Architecture

Aguvis comprises three key designs: **unified pure vision input**, **standardized cross-platform action space**, and **inner monologue reasoning mechanism**.

- **Input**: Directly receives screenshots as observations without any textual intermediate representations.
- **Output**: A unified sequence of actions (click, type, scroll, press, etc.), adapted to different platforms via a plugin system.
- **Reasoning**: Generates a natural language "inner monologue" before action prediction, explicitly performing task decomposition and current state analysis.

### Dataset Construction — Aguvis Data Collection

A large-scale multimodal GUI Agent training dataset is constructed, including:

- **GUI Grounding Annotations**: Coordinate annotations of UI elements in screenshots, training the model to understand "what is where on the screen."
- **Reasoning Annotations**: Natural language descriptions of planning and reasoning for each operational step.
- Covering multiple platforms, including Web, desktop (Windows/Linux/macOS), and mobile (Android).

### Two-Stage Training Pipeline

**Stage 1 — GUI Grounding**:

- Goal: Enable the model to precisely locate UI elements in screenshots.
- Training Task: Predict bounding box coordinates of elements given natural language descriptions.
- The loss function is a standard coordinate regression loss:

$$\mathcal{L}_{\text{ground}} = \frac{1}{N} \sum_{i=1}^{N} \| \hat{b}_i - b_i \|_2^2$$

where $\hat{b}_i$ is the predicted coordinate, and $b_i$ is the ground truth coordinate.

**Stage 2 — Planning & Reasoning**:

- Goal: Learn multi-step task planning and reasoning on top of established grounding capabilities.
- Introduces the inner monologue mechanism, where the model first generates current state analysis and the next step's plan before outputting specific actions.
- Training data consists of complete task trajectories containing sequences of (screenshot, inner monologue, action) triples.

### Key Designs — Inner Monologue

The inner monologue consists of three elements:

1. **Observation Summary**: Describes the current screen state (e.g., "Currently in the file manager, document has been opened").
2. **Task Progress**: Evaluates the remaining steps to achieve the goal.
3. **Next-step Plan**: Clearly defines the next action and its rationale.

This explicit reasoning chain significantly improves the success rate of complex tasks (e.g., cross-application operations).

## Key Experimental Results

### Offline Experiments — ScreenSpot Visual Grounding

| Model | Mobile Text | Mobile Icon | Desktop Text | Desktop Icon | Web Text | Web Icon | Average |
|------|:-----------:|:-----------:|:------------:|:------------:|:--------:|:--------:|:----:|
| GPT-4o | 78.0 | 24.9 | 72.9 | 43.6 | 70.3 | 22.3 | 52.0 |
| CogAgent | 67.0 | 24.9 | 74.2 | 20.0 | 70.4 | 28.6 | 47.5 |
| SeeClick | 78.0 | 52.0 | 72.2 | 30.0 | 55.7 | 32.5 | 53.4 |
| OS-Atlas-7B | 93.0 | 72.2 | 91.8 | 62.7 | 85.7 | 49.0 | 75.7 |
| **Aguvis-7B** | **93.4** | **76.9** | **91.0** | **66.8** | **85.7** | **55.5** | **78.2** |

### Online Experiments — OSWorld Real Desktop Environment

| Model | Overall Success Rate (%) |
|------|:-----------:|
| GPT-4o | 5.03 |
| GPT-4o + SoM | 4.59 |
| SecClick | 9.21 |
| OS-Atlas-Base-4B | 11.65 |
| OS-Atlas-Base-7B | 14.63 |
| **Aguvis-7B** | **14.79** |
| Human | 72.36 |

### Online Experiments — AndroidWorld & Mind2Web-Live

In the AndroidWorld real Android environment, Aguvis-72B achieves a **21.8%** task success rate, and **62.5%** on MobileMiniWob++, outperforming baselines of the same scale.

In Mind2Web-Live real web interaction, Aguvis-7B with pure vision input is competitive in task success rate with methods using HTML text inputs while maintaining lower inference costs.

### Ablation Study

| Configuration | ScreenSpot Avg | OSWorld SR |
|------|:--------------:|:----------:|
| Full Aguvis-7B | 78.2 | 14.79 |
| W/o Stage 1 (no grounding pre-training) | 69.5 | 10.2 |
| W/o Inner Monologue | 74.8 | 11.6 |
| Trained only on Web data | 72.1 | 8.9 |
| Trained only on Mobile data | 70.3 | 7.5 |

Key Findings:
- In the two-stage training, Stage 1 contributes the most to the grounding capability (+8.7 pp).
- Inner Monologue significantly improves complex online tasks (+3.2 pp OSWorld SR).
- Cross-platform mixed training data brings distinct transfer gains.

## Highlights & Insights

1. **Feasibility verification of the pure vision route**: For the first time, it is proven that a pure vision GUI agent, which does not rely on HTML/DOM, can achieve SOTA in real environments, carrying major significance for cross-platform deployment.
2. **Ingenious design of decoupled training**: Separately training "where to look" (grounding) and "what to do" (planning) avoids mutual interference between the two capabilities.
3. **Contribution to the open-source ecosystem**: The complete dataset, model, and training code are open-sourced, filling the community vacancy in open-source visual GUI agents.

## Limitations & Future Work

1. Compared to the human success rate of 72.36% on OSWorld, the 14.79% success rate still exhibits a huge gap.
2. The pure vision approach may be less accurate than DOM parsing in text-dense UIs (e.g., code editors, spreadsheets).
3. Currently, the inner monologue is in natural language format, lacking generalized error recovery mechanisms.
4. It places high demands on the resolution and inference speed of vision models, resulting in high deployment costs for the 72B version.

## Related Work & Insights

- **CogAgent / OS-Atlas**: Also explore visual grounding methods but do not achieve fully autonomous pure vision agents.
- **SeeClick / WebAgent**: Rely on textual representations like HTML, with limited cross-platform capabilities.
- **AppAgent / MobileAgent**: Focus on the mobile side, lacking a unified cross-platform framework.

**Insight**: The pure vision + inner monologue paradigm can be generalized to scenarios requiring visual understanding and multi-step planning, such as robot manipulation and game AI.

## Rating

| Dimension | Score (1-5) | Description |
|------|:----------:|------|
| Novelty | 4 | The combination of pure vision route + two-stage training + inner monologue is relatively novel |
| Experimental Thoroughness | 5 | Comprehensive coverage of offline + online, multi-platform, and multiple benchmarks |
| Value | 4 | Fully open-sourced, directly applicable to GUI automation products |
| Writing Quality | 4 | Clear description of methods, with systematic organization of experiments |
| **Total Score** | **4.25** | High-quality systematic work |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] UIPro: Unleashing Superior Interaction Capability for GUI Agents](../../ICCV2025/llm_agent/uipro_unleashing_superior_interaction_capability_for_gui_agents.md)
- [\[ACL 2025\] OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents](../../ACL2025/llm_agent/os-kairos_adaptive_interaction_for_mllm-powered_gui_agents.md)
- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](../../ACL2025/llm_agent/guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](../../ACL2025/llm_agent/gui_explorer_autonomous.md)
- [\[CVPR 2026\] MMBench-GUI: A Unified Hierarchical Evaluation Framework for Multi-Platform GUI Agents](../../CVPR2026/llm_agent/mmbench-gui_a_unified_hierarchical_evaluation_framework_for_multi-platform_gui_a.md)

</div>

<!-- RELATED:END -->
