---
title: >-
  [Paper Note] Aria-UI: Visual Grounding for GUI Instructions
description: >-
  [ACL 2025][Multimodal VLM][GUI Grounding] This paper proposes Aria-UI, a vision-only multimodal model specifically designed for GUI visual grounding. By utilizing a scalable instruction synthesis data pipeline and a interleaved text-image action history mechanism, Aria-UI achieves state-of-the-art (SOTA) performance on both offline and online agent benchmarks, including 1st place on AndroidWorld (44.8%) and 3rd place on OSWorld (15.2%).
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "GUI Grounding"
  - "visual grounding"
  - "Multimodal Large Language Models (MLLMs)"
  - "GUI Agent"
  - "Action History"
date: 2026-05-08
content_hash: 0e3a81f8915048e7
---

# Aria-UI: Visual Grounding for GUI Instructions

**Conference**: ACL 2025  
**arXiv**: [2412.16256](https://arxiv.org/abs/2412.16256)  
**Code**: [https://github.com/AriaUI/Aria-UI](https://github.com/AriaUI/Aria-UI)  
**Area**: Multimodal / GUI Agent  
**Keywords**: GUI Grounding, visual grounding, Multimodal Large Language Models (MLLMs), GUI Agent, Action History

## TL;DR

This paper proposes Aria-UI, a vision-only multimodal model specifically designed for GUI visual grounding. By utilizing a scalable instruction synthesis data pipeline and a interleaved text-image action history mechanism, Aria-UI achieves state-of-the-art (SOTA) performance on both offline and online agent benchmarks, including 1st place on AndroidWorld (44.8%) and 3rd place on OSWorld (15.2%).

## Background & Motivation

Digital automation agents need to locate and interact with target elements (such as buttons, input fields, etc.) on GUI interfaces based on language instructions, a process known as **GUI Grounding**. This is the core capability of a GUI agent; only by accurately mapping instructions to specific on-screen elements can the agent perform operations like clicking and typing.

Existing methods face three major limitations:

**Dependency on structured auxiliary inputs (HTML/AXTree)**: Most methods rely on DOM trees or accessibility trees (AXTree) as inputs. However, acquiring such information is inconsistent across different platforms (web, mobile, desktop) and unavailable in certain scenarios (e.g., native applications, game interfaces). More importantly, these structured inputs can be extremely verbose and expensive to process.

**Insufficient diversity in instruction formats**: Different planning agents generate grounding instructions in various formats. Some are short element descriptions ("search button"), while others are long instructions containing reasoning processes ("To complete the task, I need to click on..."). Existing models struggle to adapt to this heterogeneity.

**Lack of context awareness**: During multi-step task execution, the grounding of the current step requires understanding previous operation history. For instance, the target element of the instruction "click next" depends heavily on what was done previously, yet most existing methods ignore historical context.

Core Idea: **Replacing the reliance on structured inputs with a vision-only approach, and enhancing GUI grounding capabilities by synthesizing diverse instruction data and modeling action history.**

## Method

### Overall Architecture

Aria-UI is based on the Aria multimodal model (MoE architecture, with 3.9B activated parameters/token). Its input consists of a GUI screenshot + a text instruction (optionally including history), and its output is the relative coordinate $(x, y)$ of the target element, with coordinates normalized to the range of $[0, 1000]$.

The pipeline consists of two stages:
- **Stage 1**: Supervised Fine-Tuning (SFT) on large-scale synthetically diverse grounding instruction data.
- **Stage 2**: Context-aware fine-tuning on episode data containing action histories.

### Key Designs

1. **Scalable Data Pipeline**:

    - **Function**: Automatically synthesizes diverse and high-quality grounding instruction samples from existing GUI datasets.
    - **Mechanism**: Collects GUI screenshots and element annotation data from multiple sources, and then leverages an LLM (e.g., GPT-4) to generate instructions in different formats—including short descriptive ones ("the search icon"), reasoning-based ones ("To find videos, I should click on the search button"), and direct coordinate-based instructions. Multiple expressions are generated for each element, significantly increasing the diversity of the training data.
    - **Design Motivation**: Instruction formats output by planning agents in real-world scenarios vary drastically. If the training data contains only a single format, the model's generalization capability will be limited. By synthesizing multiple formats, Aria-UI can adapt to the outputs of different upstream agents.

2. **Action History Modeling**:

    - **Function**: Encodes historical operations into the context in a text-and-image interleaved manner to assist grounding in the current step.
    - **Mechanism**: Supports two historical formats: (a) **text-only history**—only describing past operations in text (e.g., "Step 1: clicked on Settings"); (b) **interleaved text-image history**—with each operational step accompanied by its corresponding GUI screenshot. The model is trained on both formats during the SFT phase.
    - **Design Motivation**: During dynamic task execution (e.g., AndroidWorld), the current interface state is the outcome of prior operations. It is impossible to correctly ground target elements without understanding the history—such as "clicking confirm" having completely different targets in different dialog boxes. Image histories provide much richer contextual clues than text-only histories.

3. **MoE Architecture and Super-Resolution Support**:

    - **Function**: Efficiently handles GUI screenshots of varying sizes and aspect ratios.
    - **Mechanism**: Based on the Mixture-of-Experts (MoE) architecture of the Aria model, where each token activates only 3.9B parameters (while the total parameters are larger). It supports splitting the GUI screenshot into multiple patches for super-resolution encoding (`split_image=True`, `max_image_size=980`), which is capable of processing image patches up to ~980px.
    - **Design Motivation**: Text and small icons in GUI screenshots require high resolution for accurate recognition. The MoE architecture provides sufficient model capacity while maintaining lightweight inference.

### Loss & Training

- Stage 1: SFT on synthetic static grounding data to learn the basic capability of instruction-element alignment.
- Stage 2: Further fine-tuning on episode data containing action histories (~992K instruction-output pairs) to learn context-aware capabilities.
- Output Format: The model directly outputs coordinate text such as `[523, 187]`, representing the relative location of the target element in the coordinate system.

## Key Experimental Results

### Main Results (Offline Grounding Benchmarks - ScreenSpot Series)

| Benchmark | Metric | Aria-UI | Prev. Vision SOTA | AXTree-based Methods | Description |
|------|------|---------|-------------|-----------|------|
| ScreenSpot | Acc | SOTA | - | - | Multi-platform GUI localization accuracy |
| ScreenSpot-V2 | Acc | SOTA | - | -| Extended version |
| ScreenSpot-Pro | Acc | SOTA | - | - | Professional-level difficulty |

Aria-UI outperforms both vision-only methods and AXTree-dependent methods across all sub-benchmarks in the ScreenSpot series.

### Online Agent Benchmarks

| Benchmark | Metric | Aria-UI | Rank | Description|
|------|------|---------|------|------|
| AndroidWorld | Task SR (%) | 44.8 | 🏆 1st | Mobile, multi-step real-world tasks |
| OSWorld | Task SR (%) | 15.2 | 🥉 3rd | Desktop, complex real-world tasks |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Without instruction diversity | Significant drop | Single-format instructions lead to poor generalization |
| Without action history | Obvious drop | Multi-step tasks rely heavily on context |
| Text-only history vs. Interleaved text-image history | Interleaved is better | Images provide richer contextual signals |
| Base model vs. Context-aware model | Context-aware is better | Gain from the two-stage training |

### Key Findings

- **Vision-only methods can outperform structured input methods**: Aria-UI outperforms baselines that rely on HTML/AXTree without utilizing them.
- **Instruction diversity is critical**: Synthesizing training instructions in multiple formats is crucial for cross-agent generalization.
- **Action history significantly improves multi-step task performance**: Especially on online benchmarks requiring multi-turn interactions, such as AndroidWorld and OSWorld.
- **Efficiency advantages of MoE architecture**: Achieving SOTA with only 3.9B activated parameters, which is much smaller than full-parameter dense models.

## Highlights & Insights

- **Success of the vision-only approach**: Proving that vision-only methods not relying on DOM/AXTree are not only feasible but actually superior, which is of great significance for the practical deployment of cross-platform GUI agents.
- **Data Engineering > Model Engineering**: The core innovation of Aria-UI lies in the training data rather than the model architecture—this highlights the critical importance of data quality and diversity in the GUI agent domain.
- **Complete open-source ecosystem**: Model weights, training data, and inference code are fully open-sourced, including online demos on Hugging Face Spaces.
- **High scalability**: The data synthesis pipeline and context modeling methods can be readily transferred to other platforms and tasks.

## Limitations & Future Work

- May lack robustness for highly dynamic interfaces (such as video streams and animations).
- Coordinate prediction accuracy depends heavily on screenshot resolution; extremely small elements may face localization difficulties.
- The context-aware mode requires storing and processing historical screenshots, which increases inference overhead.
- Currently validated primarily on English GUIs; performance on multilingual interfaces remains unexplored.
- Deploying MoE models is more complex than dense models, requiring support from specific inference frameworks.

## Related Work & Insights

- **vs. CogAgent**: CogAgent similarly employs a vision-only approach but features a larger model (18B). Aria-UI archives better results with fewer activated parameters via the MoE architecture.
- **vs. SeeClick/UGround**: These methods typically process only a single instruction format. Aria-UI's instruction synthesis pipeline significantly enhances generalization capabilities.
- **vs. Set-of-Mark**: SoM overlays annotations on screenshots to assist localization, which is a semi-structured method, whereas Aria-UI is entirely end-to-end.

## Rating

- Novelty: ⭐⭐⭐⭐ The vision-only approach and the instruction synthesis pipeline represent clear technical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of both offline and online benchmarks, complete with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and detailed methodological descriptions.
- Value: ⭐⭐⭐⭐⭐ Completely open-sourced, offering high reference value to the GUI agent community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding](r-vlm_region-aware_vision_language_model_for_precise_gui_grounding.md)
- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[CVPR 2026\] DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](../../CVPR2026/multimodal_vlm/drs-gui_dynamic_region_search_for_training-free_gui_grounding.md)
- [\[CVPR 2026\] MVP: Multiple View Prediction Improves GUI Grounding](../../CVPR2026/multimodal_vlm/mvp_multiple_view_prediction_improves_gui_grounding.md)
- [\[ICLR 2026\] GuirlVG: Incentivize GUI Visual Grounding via Empirical Exploration on Reinforcement Learning](../../ICLR2026/multimodal_vlm/guirlvg_incentivize_gui_visual_grounding_via_empirical_exploration_on_reinforcem.md)

</div>

<!-- RELATED:END -->
