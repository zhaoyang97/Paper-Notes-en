---
title: >-
  [Paper Note] Agent3D-Zero: An Agent for Zero-shot 3D Understanding
description: >-
  [ECCV 2024][LLM Agent][Zero-shot 3D Understanding] Agent3D-Zero proposes a VLM-based zero-shot 3D scene understanding agent framework. By utilizing Set-of-Line visual prompting on the bird's-eye view (BEV) to guide the VLM to actively select observation viewpoints and synthesizing multi-view images for 3D reasoning, it outperforms fine-tuned 3D-LLM methods on tasks like ScanQA.
tags:
  - "ECCV 2024"
  - "LLM Agent"
  - "Zero-shot 3D Understanding"
  - "VLM Agent"
  - "Visual Prompting"
  - "Multi-view Reasoning"
  - "Set-of-Line Prompting"
date: 2026-05-08
content_hash: a07d14265b9a7720
---

# Agent3D-Zero: An Agent for Zero-shot 3D Understanding

**Conference**: ECCV 2024  
**arXiv**: [2403.11835](https://arxiv.org/abs/2403.11835)  
**Code**: None  
**Area**: LLM Agent / 3D Scene Understanding  
**Keywords**: Zero-shot 3D Understanding, VLM Agent, Visual Prompting, Multi-view Reasoning, Set-of-Line Prompting

## TL;DR

Agent3D-Zero proposes a VLM-based zero-shot 3D scene understanding agent framework. By utilizing Set-of-Line visual prompting on the bird's-eye view (BEV) to guide the VLM to actively select observation viewpoints and synthesizing multi-view images for 3D reasoning, it outperforms fine-tuned 3D-LLM methods on tasks like ScanQA.

## Background & Motivation

**Background**: 3D scene understanding is a core task in computer vision, crucial for robotics, autonomous driving, and augmented reality. Recent developments in LLMs/VLMs have demonstrated powerful capabilities in 2D tasks, naturally leading to the demand for extending these capabilities to the 3D domain.

**Limitations of Prior Work**: Existing mainstream approaches (e.g., 3D-LLM, 3DMIT) require acquiring 3D data (point clouds/meshes) first, and then using a perceiver to fine-tune and align 3D features to the LLM space. However, 3D datasets are significantly smaller in scale than 2D datasets—acquisition relies on specialized equipment like depth cameras/LiDAR, annotation costs are extremely high, and public datasets are limited to CAD models, indoor scenes, and autonomous driving scenarios.

**Key Challenge**: The ceiling of the fine-tuning paradigm is constrained by the scale and diversity of 3D training data, whereas pre-trained VLMs have absorbed a massive amount of 2D knowledge that has not been fully utilized for 3D understanding.

**Goal**: How to leverage the existing 2D capabilities of VLMs to achieve 3D scene understanding without requiring any 3D data training.

**Key Insight**: Inspired by human cognition—humans do not require explicit 3D reconstruction to understand the 3D world; rather, they establish spatial understanding by observing from multiple angles. Therefore, 3D scene understanding can be formulated as a problem of "understanding and synthesizing multiple images."

**Core Idea**: Empower the VLM as an agent to actively select observation viewpoints, render the corresponding images, and then perform synthetic reasoning. This approach addresses 3D tasks using 2D vision and language capabilities in a completely zero-shot manner.

## Method

### Overall Architecture

Pipeline: 3D Mesh $M$ → Render bird's eye view (BEV) $I_b$ → VLM + SoLP visual prompting to select $N$ camera poses $\mathcal{T}$ → Render $N$ viewpoint images $\mathcal{I}$ → VLM synthesizes multi-view images to answer downstream tasks.

Core workflow in three steps:
1. **BEV Viewpoint Planning**: Render a BEV image from the 3D mesh, enhance it with SoLP, and input it into the VLM to select camera poses.
2. **Multi-view Image Rendering**: Render images based on the camera poses selected by the VLM.
3. **Task Reasoning**: Input the multi-view images + task prompt into the VLM to complete downstream reasoning.

The overall process is iterative—the VLM selects $N'$ viewpoints at each step, choosing the next batch based on its prior observation experiences.

### Key Designs

1. **Set-of-Line Prompting (SoLP)**:

    - **Function**: Overlays grid lines and coordinate scales on the BEV map to convert the raw image $I_b$ into a prompted image $I_b^p$ with a spatial reference system.
    - **Mechanism**: Inspired by the latitude and longitude system of maps, a Cartesian coordinate system is established on the BEV image. The VLM can specify camera positions using grid intersections (e.g., $(0,0)$) and orientation using a set of directions $\{left, right, front, back\}$, discretizing continuous space localization into a textually describable form:
    $$\mathcal{T} = \textbf{VLM}(I_b^p, P_b^p)$$
    - **Design Motivation**: VLMs are weak at accurate distance measurement and spatial localization; directly feeding a raw BEV image to let the VLM output camera poses is completely infeasible (the 0x0 configuration fails directly in the ablation studies). By introducing a discretized visual reference frame, SoLP transforms the precise positioning problem into a discrete decision-making problem of "selecting grid points," significantly reducing the difficulty of spatial reasoning for the VLM.

2. **Active Viewpoint Selection Strategy**:

    - **Function**: Enables the VLM to autonomously decide which angles to observe the scene next, based on existing observations.
    - **Mechanism**: Instead of asking the VLM to output all $N$ viewpoints at once, the approach iteratively outputs $N'$ viewpoints at each step, allowing the VLM to utilize previous observation history. Formally:
    $$A_t = \textbf{VLM}(\mathcal{I}, P_t^p)$$
    - **Design Motivation**: Planning all viewpoints at once may lead to redundancy or omissions. Iterative selection adaptively adjusts based on the acquired information, resembling human environment exploration strategies. Ablation experiments demonstrate that the "selected" strategy outperforms the "random" strategy by 31.8 points in CIDEr.

3. **Tool-augmented Perception**:

    - **Function**: Extends Agent3D-Zero from pure reasoning tasks to perception tasks such as 3D semantic segmentation.
    - **Mechanism**: The agent can invoke external visual tools (such as the SAM segmentation model) to first perform 2D semantic segmentation on each selected image. The VLM then assigns semantic labels to the segmented regions, which are back-projected to the 3D point cloud using depth information:
    $$R_i = \textbf{VLM}(I_i, P_f, f)$$
    - **Design Motivation**: Although VLMs naturally lack pixel-level perception capability, acting as an agent allows them to orchestrate tools to accomplish perception tasks, demonstrating the generality and extensibility of the agent framework.

### Loss & Training

Agent3D-Zero is a training-free, completely zero-shot method that **does not require any training or fine-tuning**. The core VLM uses GPT-4V, and the entire system is driven by prompt engineering. All "learning" is implicitly derived from the pre-trained knowledge of the VLM.

## Key Experimental Results

### Main Results

**ScanQA Validation Set**:

| Method | Type | METEOR | ROUGE-L | CIDEr | EM |
|------|------|--------|---------|-------|-----|
| ScanQA | Two-stage | 13.1 | 33.3 | 64.9 | 21.0 |
| 3D-LLM (BLIP2-flant5) | Fine-tune | 14.5 | 35.7 | 69.4 | 20.5 |
| Agent3D-Zero (random) | Zero-shot | 12.2 | 26.9 | 40.0 | 4.9 |
| **Agent3D-Zero (selected)** | **Zero-shot** | **16.0** | **37.0** | **71.8** | 17.5 |

**ScanQA Test Set**:

| Method | METEOR | ROUGE-L | CIDEr | EM |
|------|--------|---------|-------|-----|
| 3D-LLM (BLIP2-flant5) | 14.9 | 35.3 | 69.6 | 19.1 |
| **Agent3D-Zero** | **16.9** | **39.3** | **77.5** | 21.3 |

**3D-assisted Dialog（Held-In Dataset）**:

| Method | METEOR | ROUGE-L |
|------|--------|---------|
| 3D-LLM (BLIP2-flant5) | 18.9 | 39.3 |
| Agent3D-Zero (selected) | **19.3** | 39.3 |

**Task Decomposition**:

| Method | BLEU-4 | METEOR | ROUGE-L |
|------|--------|--------|---------|
| 3D-LLM (BLIP2-flant5) | 7.4 | 15.9 | 37.8 |
| Agent3D-Zero (selected) | **15.5** | **22.9** | **45.1** |

### Ablation Study

**Impact of the number of viewpoints (ScanQA Validation Set 20% subset)**:

| Viewpoints | BLEU-1 | METEOR | CIDEr | EM |
|--------|--------|--------|-------|-----|
| 6 | 17.1 | 12.8 | 50.8 | 13.1 |
| 12 | 23.3 | 15.0 | 67.9 | 16.9 |
| 24 | 34.1 | 16.5 | 82.0 | 21.1 |

**Impact of SoLP grid line density**:

| Density | BLEU-1 | METEOR | CIDEr | Description |
|------|--------|--------|-------|------|
| 0×0 | - | - | - | VLM is completely unable to output valid results |
| 4×4 | 23.2 | 14.2 | 66.0 | Basic usability |
| 8×8 | 34.1 | 16.5 | 82.0 | Optimal configuration |
| 16×16 | - | - | - | Too dense, leading to VLM identification failure |

### Key Findings

- **Zero-shot beats Fine-tuning**: On semantic evaluation metrics such as METEOR, ROUGE-L, and CIDEr, Agent3D-Zero significantly outperforms the 3D-LLM that requires 3D data fine-tuning (CIDEr: 77.5 vs 69.6).
- **Active Viewpoint Selection vs Random**: Strategically selecting viewpoints brings a massive performance boost (CIDEr: 71.8 vs 40.0), indicating that the VLM is capable of planning observation strategies effectively.
- **More Viewpoints are Better**: Model performance scales monotonically with the number of viewpoints, but is ultimately capped by the VLM's context window.
- **SoLP is Necessary**: Without grid lines (0x0), the VLM is completely incapable of performing viewpoint planning. 8x8 is the optimal density.
- **Weaker EM/BLEU Metrics than Fine-tuned Methods**: These metrics focus on exact matching, which is naturally unfavorable for zero-shot methods; however, semantic metrics (METEOR, CIDEr) better reflect actual comprehension capability.

## Highlights & Insights

- **Paradigm Innovation**: First to demonstrate that 3D scene understanding can be achieved entirely without 3D data. Simply pairing a VLM with multi-view images can outperform fine-tuned methods, showing a paradigm-shifting contribution.
- **SoLP Visual Prompting**: Discretizing continuous spatial positioning into grid selection is a simple yet extremely elegant and effective mechanism, which can be generalized to any task requiring spatial reasoning from VLMs.
- **Generality of the Agent Framework**: By employing different task prompts and tool invocations, the identical framework covers various tasks including QA, dialogue, segmentation, and navigation.
- **Iterative Viewpoint Selection**: Mimics human exploration behavior, allowing the Agent to adaptively select the next steps based on prior observations, representing a natural mindset for embodied AI.
- **Human Cognition-Inspired Transferable Trick**: Decomposing 3D understanding into a synthesis of multiple 2D observations—this "dimension reduction" thinking can be transferred to other modalities.

## Limitations & Future Work

- **Reliance on 3D Mesh**: Requires a pre-reconstructed 3D mesh to render arbitrary viewpoints, making it inapplicable to scenarios with only images/videos.
- **High Cost**: Using GPT-4V for multi-turn reasoning results in extremely high computation and API costs.
- **Poor 3D Semantic Segmentation Performance**: Achieves an mIoU of only 8.7, which is far below traditional 3D segmentation methods, indicating a substantial gap in perception capabilities.
- **Limited Number of Viewpoints**: Constrained by the VLM's context length limit, currently capped at 24 images.
- **Lack of Depth Reasoning Ability**: Relies on the rendering pipeline to provide depth information; the VLM cannot accurately estimate exact distances on its own.
- **Constrained SoLP Density**: If it is too dense (16x16), the VLM fails to recognize it; if it is too sparse, the spatial resolution is insufficient. The optimal density must be adjusted for different VLMs.

## Related Work & Insights

- **vs 3D-LLM**: 3D-LLM takes the "fine-tuning perceiver paradigm", which is data-inefficient and constrained by 3D data distributions; Agent3D-Zero employs a "zero-shot agent paradigm", which is flexible but bottlenecked by VLM capabilities.
- **vs SpatialVLM**: SpatialVLM enhances spatial understanding of a single image; Agent3D-Zero utilizes multi-view images to cover complete scenes.
- **vs Set-of-Mark (SoM)**: SoM overlays segment indices on images to enhance the grounding capabilities of VLMs; SoLP adds coordinate grids on BEV maps to strengthen spatial localization. Both represent excellent visual prompting engineering.
- **Inspiration**: Any VLM application requiring precise spatial reasoning can learn from SoLP's idea of "adding a coordinate frame of reference."

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The agent framework for zero-shot 3D understanding is a brand new paradigm, and SoLP is a highly original design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of multiple tasks such as QA, dialogue, segmentation, and navigation with ablation studies, though 3D segmentation experiments are not deeply investigated.
- Writing Quality: ⭐⭐⭐⭐ The logic is clear and the method descriptions are intuitive, though some mathematical notations are slightly redundant.
- Value: ⭐⭐⭐⭐⭐ Demonstrates the feasibility of zero-shot 3D understanding, offering valuable inspiration for future Agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MIND: A Multi-agent Framework for Zero-shot Harmful Meme Detection](../../ACL2025/llm_agent/mind_a_multi-agent_framework_for_zero-shot_harmful_meme_detection.md)
- [\[CVPR 2026\] History to Future: Evolving Agent with Experience and Thought for Zero-shot Vision-and-Language Navigation](../../CVPR2026/llm_agent/history_to_future_evolving_agent_with_experience_and_thought_for_zero-shot_visio.md)
- [\[ECCV 2024\] VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](videoagent_a_memory-augmented_multimodal_agent_for_video_understanding.md)
- [\[ACL 2025\] Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play](../../ACL2025/llm_agent/play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](../../NeurIPS2025/llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

</div>

<!-- RELATED:END -->
