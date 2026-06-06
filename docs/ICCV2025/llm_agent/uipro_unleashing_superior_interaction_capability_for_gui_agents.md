---
title: >-
  [Paper Note] UIPro: Unleashing Superior Interaction Capability for GUI Agents
description: >-
  [ICCV 2025][LLM Agent][GUI agent] UIPro is proposed to achieve state-of-the-art GUI interaction performance across mobile, web, and desktop platforms by constructing 20.6M GUI understanding samples for pre-training and i…
tags:
  - "ICCV 2025"
  - "LLM Agent"
  - "GUI agent"
  - "unified action space"
  - "GUI grounding"
  - "vision-language model"
  - "multi-platform interaction"
date: 2026-05-08
content_hash: d698785a81d73d4d
---

# UIPro: Unleashing Superior Interaction Capability for GUI Agents

**Conference**: ICCV 2025
**arXiv**: [2509.17328](https://arxiv.org/abs/2509.17328)  
**Code**: [GitHub](https://github.com/ZJULiHongxin/UIPro)  
**Area**: LLM Agent
**Keywords**: GUI agent, unified action space, GUI grounding, vision-language model, multi-platform interaction

## TL;DR

UIPro is proposed to achieve state-of-the-art GUI interaction performance across mobile, web, and desktop platforms by constructing 20.6M GUI understanding samples for pre-training and introducing a unified action space to integrate heterogeneous GUI agent task data.

## Background & Motivation

Building autonomous GUI agents capable of operating graphical interfaces as humans do is a long-standing vision in AI. The core capabilities for GUI interaction include: (1) visual understanding and grounding of GUI elements; and (2) planning and executing action sequences that fulfill user goals.

Existing approaches face two critical bottlenecks:

**Insufficient data scale**: Existing GUI interaction datasets typically lack sufficient scale and scenario diversity. The advantages of large-scale training cannot emerge at small scales (emergent capabilities), yet large-scale datasets such as CogAgent (247M) and ScreenAI (421M) are not publicly available.

**Training pipeline deficiencies**: Different GUI trajectory datasets adopt **heterogeneous action spaces** (e.g., AITW defines swipe as DUAL_POINT(start, end), while AndroidControl uses scroll(direction)), and naively mixing them during training leads to action conflicts and performance degradation.

**Mechanism**: (1) Constructing the largest open-source GUI understanding dataset (20.6M samples) to establish a strong grounding foundation for agents; (2) designing a unified action space to integrate heterogeneous data sources and unlock the potential of multi-source data.

## Method

### Overall Architecture

UIPro adopts a two-stage training paradigm:
- **Stage 1: GUI Understanding Pre-training** — Training on 20.6M multi-platform, multi-task GUI understanding samples to acquire strong grounding capabilities.
- **Stage 2: GUI Agent Task Fine-tuning** — Fine-tuning on agent trajectory data unified under the proposed action space to develop action prediction capabilities.

Two backbone models are used: UIPro-SLiME (3B, trained from scratch) and UIPro-Qwen2VL (7B, fine-tuned from Qwen2-VL).

### Key Designs

1. **Large-Scale GUI Understanding Data Construction**: GUI data is collected and cleaned from multiple sources (Common Crawl web pages, Android emulators, RICO, MobileViews, etc.), generating $\langle$screenshot, referring expression, coordinate$\rangle$ triplets across 13 task types:

    - **Element description (elemgnd/elemref)**: Describes visual appearance, element type, and location.
    - **User intent (intentgnd)**: Describes how users interact with an element, e.g., "tap the password input field."
    - **Contextual functionality (funcgnd/funcref)**: Describes interaction affordances, e.g., "this element enables the user to share content."
    - **Text grounding (textgnd/OCR)**, **icon classification (icongnd/iconref)**, widget listing, GUI question answering, and GUI summarization.
    - The final 20.6M samples are associated with 2.5M unique screenshots; 67% are newly annotated and 33% are cleaned from open-source data.

2. **Unified Action Space Design**: To resolve conflicts arising from heterogeneous action definitions, an **action superset** is designed:

    - Swipe is unified as `swipe(start, direction, distance)`, compatible with AITW's DUAL_POINT and AndroidControl's scroll(direction).
    - Separate unified action spaces are defined for mobile, web, and desktop platforms (the mobile space includes 12 actions such as tap, long_press, drag, input_text, swipe, and navigate).
    - All outputs are formatted as JSON, e.g., `{"action_type": "click", "target": (x, y)}`.
    - Action definitions are excluded from prompts (experiments show that omitting them improves training efficiency).

3. **Systematic Denoising Pipeline**: Because raw GUI data suffers from severe noise (95.9% of home pages contain accessibility errors, and one data source has a noise rate of 29%), a seven-step denoising procedure is designed:

    - Detection of blank elements (color standard deviation $< 5$).
    - OCR-based detection of invisible elements.
    - Removal of invalid, oversized, or undersized bounding boxes.
    - Removal of duplicate boxes and mismatched elements.

### Loss & Training

- Pre-training stage: Coordinates are normalized to $[0, 1000]$; UIPro-SLiME trains with a frozen ViT and full parameter fusion for 1 epoch; UIPro-Qwen2VL is fine-tuned via LoRA on a 4.4M subset.
- Agent fine-tuning stage: Trained for 6 epochs until performance saturates; prompts include task descriptions and action histories; ground-truth actions are formatted as JSON objects.
- Mobile: 6 data sources are mixed (380K samples); Web: 3 data sources are mixed (145K samples).

## Key Experimental Results

### Main Results (Tables)

**AITW Mobile Benchmark (Step SR%)**:

| Method | Scale | General | Install | GoogleApps | Single | WebShop | Overall |
|--------|-------|---------|---------|------------|--------|---------|---------|
| GPT-4V-OmniParser | - | 48.3 | 57.8 | 51.6 | 77.4 | 52.9 | 57.7 |
| SeeClick | 10B | 54.0 | 66.4 | 54.9 | 63.5 | 57.6 | 59.3 |
| OS-ATLAS | 7B | 57.9 | 63.4 | 55.5 | 79.1 | 59.7 | 63.1 |
| **UIPro-Qwen2VL** | **7B** | **64.4** | **74.6** | **67.9** | **79.4** | **67.6** | **70.4** |
| **UIPro-SLiME** | **3B** | **67.0** | **71.4** | **65.4** | 73.2 | 62.9 | **68.0** |

**Mind2Web Web Benchmark (Step SR%)**:

| Method | Scale | Cross-Task | Cross-Website | Cross-Domain |
|--------|-------|------------|---------------|--------------|
| OmniParser (GPT-4V) | - | 39.4 | 36.5 | 42.0 |
| OS-ATLAS | 7B | 36.7 | 35.7 | 37.2 |
| **UIPro-Qwen2VL** | **7B** | **48.4** | **43.6** | **45.5** |

### Ablation Study (Tables)

**Effect of GUI Understanding Pre-training Data Scale**:

| Pre-training Data Size | Avg. Grounding Accuracy | AITW Step SR | AndroidControl Step SR |
|------------------------|------------------------|-------------|----------------------|
| 0 | ~30% | ~52% | ~40% |
| 5.9M | ~55% | ~63% | ~55% |
| 20.6M | ~60% | ~68% | ~61% |

**Effect of Unified Action Space**: Mixing data sources without unifying the action space leads to significant performance degradation across all benchmarks, primarily due to a sharp drop in action type accuracy and inconsistent swipe direction prediction.

### Key Findings

- UIPro-SLiME (3B) outperforms CogAgent (18B) and GPT-4V-OmniParser.
- Grounding accuracy is positively correlated with downstream agent task performance, confirming that grounding is the foundation of agent capability.
- The unified action space improves not only the accuracy of shared actions but also that of platform-specific actions (e.g., Wait), indicating cross-task knowledge transfer and a regularization effect from data diversity.
- Both GUI understanding data and agent task data exhibit clear scaling laws.
- Performance gains from denoising are consistently significant across all 6 grounding benchmarks.

## Highlights & Insights

- The largest open-source GUI understanding dataset (20.6M samples) covering 13 task types.
- The unified action space design is elegant and effective — a superset accommodates different definitions, and different platforms share similar interaction principles.
- The systematic denoising pipeline reveals the severity of GUI data quality issues (29% noise rate in one data source).
- The introduction of functional grounding tasks (funcgnd) is a notable contribution — enabling the model to understand what an element *can do* rather than merely *what it is*.

## Limitations & Future Work

- Desktop training data is far sparser than mobile and web data, limiting UIPro's performance on Windows/macOS.
- Evaluation is currently limited to offline settings; on-device real-time interaction evaluation remains to be explored.
- The action space unification is still manually designed; future work could explore automatic learning of cross-platform action alignment.
- Benchmarks such as AITW do not account for alternative valid solutions, potentially leading to underestimated evaluation scores.

## Related Work & Insights

- Compared to SeeClick (5.3M data), UIPro uses 4× more data and incorporates functional annotations.
- Compared to OS-ATLAS (13.6M data), UIPro uses 50% more data and achieves superior performance on most benchmarks.
- The unified action space concept is generalizable to other multi-source mixed training scenarios (e.g., action space unification in robotic manipulation).

## Rating

- Novelty: ⭐⭐⭐⭐ (systematic contributions through unified action space + large-scale data engineering)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5 agent benchmarks + 6 grounding benchmarks + comprehensive ablations + transfer experiments)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, complete details)
- Value: ⭐⭐⭐⭐⭐ (significant contributions to both data resources and methodology for the GUI agent community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Video2GUI: Synthesizing Large-Scale Interaction Trajectories for Generalized GUI Agent Pretraining](../../ICML2026/llm_agent/video2gui_synthesizing_large-scale_interaction_trajectories_for_generalized_gui_.md)
- [\[ACL 2026\] LPO: Towards Accurate GUI Agent Interaction via Location Preference Optimization](../../ACL2026/llm_agent/lpo_towards_accurate_gui_agent_interaction_via_location_preference_optimization.md)
- [\[ICCV 2025\] Less is More: Empowering GUI Agent with Context-Aware Simplification](less_is_more_empowering_gui_agent_with_context-aware_simplification.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](../../CVPR2026/llm_agent/gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)

</div>

<!-- RELATED:END -->
