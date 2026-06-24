---
title: >-
  [Paper Note] GUICourse: From General Vision Language Model to Versatile GUI Agent
description: >-
  [ACL 2025][LLM Agent][GUI Agent] This paper introduces GUICourse, a suite of datasets (GUIEnv/GUIAct/GUIChat) designed to train versatile GUI agents from general Vision-Language Models (VLMs). Through a two-stage training pipeline, it first enhances OCR and grounding capabilities, and then injects GUI-specific knowledge, enabling a small model with only 3.1B parameters to achieve effective performance on web and smartphone GUI navigation tasks.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "GUI Agent"
  - "Vision Language Model"
  - "OCR and grounding"
  - "Data-driven"
  - "Web navigation"
date: 2026-05-08
content_hash: f7b5f45c9debac0c
---

# GUICourse: From General Vision Language Model to Versatile GUI Agent

**Conference**: ACL 2025  
**arXiv**: [2406.11317](https://arxiv.org/abs/2406.11317)  
**Code**: [https://github.com/RUCBM/GUICourse](https://github.com/RUCBM/GUICourse)  
**Area**: LLM Agent  
**Keywords**: GUI Agent, Vision Language Model, OCR and grounding, Data-driven, Web navigation

## TL;DR
This paper introduces GUICourse, a suite of datasets (GUIEnv/GUIAct/GUIChat) designed to train versatile GUI agents from general Vision-Language Models (VLMs). Through a two-stage training pipeline, it first enhances OCR and grounding capabilities, and then injects GUI-specific knowledge, enabling a small model with only 3.1B parameters to achieve effective performance on web and smartphone GUI navigation tasks.

## Background & Motivation
Graphical User Interfaces (GUIs) are the core medium for human-computer interaction, and GUI agents aim to automate complex user tasks across various GUI systems. Although existing VLMs (such as LLaVA, Qwen-VL) show excellent performance in image captioning and visual question answering, they face two core challenges in GUI scenarios: (1) Insufficient OCR and grounding capabilities—they fail to accurately recognize text in different fonts and positions within web screenshots and provide precise locations; (2) Lack of GUI-specific knowledge—they do not understand the functions and control methods of GUI elements, making them unable to execute navigation instructions such as "click login" or "search for products."

Existing GUI datasets are either overly simplified in environments (e.g., MiniWoB++), too restricted in domain (e.g., WebShop focusing only on shopping), or extremely limited in data scale (e.g., RUSS with only 80 instructions). Under realistic web scenarios, large-scale datasets containing both single-step and multi-step instructions are highly scarce.

The core idea of this work is: by building a systematic data-training pipeline, first bridge the perception gaps (OCR + grounding) of general VLMs, and then inject GUI interaction knowledge through navigation data, transforming general VLMs into practical GUI agents in a purely data-driven manner.

## Method

### Overall Architecture
GUICourse adopts a two-stage training pipeline:
- **Stage 1 (Pre-training)**: Employs the GUIEnv-global dataset (10 million samples) to enhance the foundational OCR and grounding capabilities of the VLM.
- **Stage 2 (SFT Fine-tuning)**: Uses GUIEnv-local (70k) + GUIAct (all) + GUIChat (all) to inject GUI element knowledge, navigation capability, and conversational interaction ability.

The GUI agent receives only screenshots as input (purely visual) and outputs position-based actions, defining a unified action space with 11 types of operations (e.g., click, tap, swipe).

### Key Designs

1. **GUIEnv: Large-scale OCR and Grounding Dataset**

    - GUIEnv-global: Collected 4 million URLs from the C4 corpus and rendered them using Playwright to obtain 10 million webpage screenshot-annotation pairs. Each sample contains all describable contents of the page (texts, bounding boxes, layout sequences) to serve pre-training.
    - GUIEnv-local: Filtered 50,000 screenshots from the global split, cropped them into sub-images of size $\le 1920\times 1080$, and randomly sampled 10 elements with texts and positions from each sub-image to construct 700k SFT data with bi-directional tasks: "text2bbox" and "bbox2text".
    - Design Motivation: Screenshots in GUI scenarios are typically large, dense in text, and contain diverse fonts, where the OCR performance of general VLMs falls severely short.

2. **GUIAct: GUI Navigation Dataset**

    - web-single (67k items): Covers 50 scenarios and 13k real websites. GPT-4V was used to automatically annotate single-step instruction-action pairs, followed by human verification (improving accuracy from 55% to 92%).
    - web-multi (5,696 items): Spans 8 top-level scenarios, 32 sub-scenarios, and 121 well-known websites. Multi-step navigation instructions were crowd-sourced, averaging 7.9 steps per task.
    - smartphone (9,157 items): Converted from the General subset of the AITW dataset to fit the unified action space.
    - This represented the largest navigation dataset containing both single-step and multi-step instructions in realistic web scenarios at the time.

3. **GUIChat: Text-rich Multimodal QA Dataset**

    - Comprises 44k single-turn QA and 6k multi-turn conversations, covering four categories: visual information querying, human-centric questions, world knowledge, and complex reasoning.
    - Dialogues include webpage screenshots with bounding boxes, generated by GPT-4 based on textual representations of webpages.
    - The purpose is to enhance the natural language interaction capabilities of GUI agents.

### Loss & Training
- MiniCPM-GUI is trained based on MiniCPM-V (3.1B) by first integrating GUIEnv-global into the pre-training data, followed by fine-tuning with all SFT data.
- It supports a high-resolution version ($1344\times 1334$ vs. default $448\times 448$) using a flexible patch-slice strategy.
- Qwen-GUI and Fuyu-GUI were also trained based on Qwen-VL and Fuyu-8B, respectively, but utilizing only the SFT data.

## Key Experimental Results

### Main Results (Self-constructed Test Set)

| Model | Web-Single StepSR | Web-Multi StepSR | Smartphone StepSR | Mean |
|------|-------------------|-------------------|---------------------|------|
| GPT-4o-mini | 57.0 | 17.0 | 22.0 | 32.0 |
| MiniCPM-GUI (High-res + GUIChat)| 70.6 | 47.5 | 53.3 | 57.1 |
| Fuyu-GUI | 63.5 | 47.1 | 40.4 | 50.4 |
| Qwen-GUI | 66.7 | 46.8 | 58.1 | **57.2** |

### Cross-Task Generalization (Mind2Web)

| Model | Cross-Task StepSR | Cross-Website StepSR | Cross-Domain StepSR |
|------|-------------------|----------------------|---------------------|
| Qwen-VL (Baseline) | 20.3 | 14.0 | 12.3 |
| Qwen-GUI | 24.4 | 15.6 | 17.5 |
| MiniCPM-V (Baseline) | 8.5 | 6.0 | 5.2 |
| MiniCPM-GUI | 20.8 | 17.3 | 14.6 |

### Ablation Study (Impact of GUIEnv Data Volume)

| GUIEnv Data Volume | Text2Bbox IoU@0.5 | Web-Single StepSR |
|-------------|-------------------|-------------------|
| 0 | 2.15 | 52.84 |
| 2.5M | 25.32 | 64.82 |
| 10M | 47.96 | **70.57** |

### Key Findings
- **Small models can also perform well**: The 3.1B MiniCPM-GUI (57.1) performs comparably to the 9.6B Qwen-GUI (57.2).
- **High resolution is crucial**: High-resolution performance improves the mean score from 49.0 to 56.1.
- **OCR and grounding are prerequisites for navigation**: Without GUIEnv, the grounding IoU@0.5 is only 2.15; after incorporating 10M samples, it rises to 47.96, and the navigation StepSR also increases from 52.84 to 70.57.
- **GUIChat data benefits web tasks**: Despite its primary goal being interaction capability, it provides an additional 1% improvement in StepSR.
- Error analysis reveals that among 50 error samples, 13 are "reasonable but mismatching ground truth" false errors, 22 are action type errors, and 15 are position errors.

## Highlights & Insights
- **Systematic data-training pipeline methodology**: Distinctly decomposes the challenge into "foundational capabilities + domain knowledge" levels and addresses them in stages using corresponding data, offering a clear methodology.
- **Rigorous engineering details in data construction**: The pipeline of GPT-4V auto-annotation combined with human verification is highly practical, lifting accuracy from 55% to 92%, indicating that automated annotation necessitates quality control.
- **Unified action space design**: The 11 action types cover both web and smartphone scenarios, facilitating cross-platform migration.
- **Ablation study reveals positive correlation between grounding ability and navigation ability**: This finding offers practical guidance for future research on GUI agents.

## Limitations & Future Work
- Only pre-training and SFT are utilized without incorporating RLHF or reinforcement learning to further boost the agent's capabilities.
- It only covers web and smartphone scenarios, lacking training data for desktop operating systems or professional applications.
- Static evaluation has inherent limitations: there may reside multiple reasonable action paths for the same instruction, yet evaluation only matches pre-defined ground truths.
- Evaluation of multi-step navigation is limited to the step level, lacking end-to-end task success rate metrics.
- **Research idea**: Consider introducing online interactive evaluation (executing agent actions in simulated environments) and enabling continuous learning from execution feedback via RLHF.

## Related Work & Insights
- SeeClick (2024): Also focuses on GUI grounding capability, with which this work compares on Mind2Web.
- CogAgent: A representative work employing vision-language models for GUI agents.
- Mind2Web and AITW: Two mainstream GUI navigation benchmarks.
- This work demonstrates the efficacy of the training paradigm "strengthening foundational visual capabilities first, then injecting domain-specific knowledge." This concept can be generalized to other agent scenarios requesting specific visual capabilities.

## Rating
- Novelty: ⭐⭐⭐ Methodologically, it is a standard data-driven + two-stage SFT paradigm; its novelty lies primarily in data construction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers comprehensive evaluations, including self-constructed testing, cross-task generalization, ablation studies, case studies, and error analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, abundant statistical tables for datasets, and well-defined pipeline descriptions.
- Value: ⭐⭐⭐⭐ Provides open-source, large-scale GUI training datasets, delivering practical contributions to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GUI-explorer: Autonomous Exploration and Mining of Transition-aware Knowledge for GUI Agent](gui_explorer_autonomous.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[ICML 2025\] Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction](../../ICML2025/llm_agent/aguvis_unified_pure_vision_agents_for_autonomous_gui_interaction.md)
- [\[NeurIPS 2025\] BTL-UI: Blink-Think-Link Reasoning Model for GUI Agent](../../NeurIPS2025/llm_agent/btlui_blinkthinklink_reasoning_model_for_gui_agent.md)
- [\[ACL 2025\] OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](os_genesis_gui_agent_trajectory.md)

</div>

<!-- RELATED:END -->
