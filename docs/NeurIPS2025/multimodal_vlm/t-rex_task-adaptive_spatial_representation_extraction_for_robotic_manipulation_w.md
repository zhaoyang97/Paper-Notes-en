---
title: >-
  [Paper Note] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs
description: >-
  [NeurIPS 2025][Multimodal VLM][VLM] This paper proposes the T-Rex framework, which dynamically selects the optimal spatial representation extraction scheme (point / vector / 6D pose) according to task complexity, and introduces Chain of Grounding (CoG) to guide VLMs through step-by-step reasoning, enabling training-free open-vocabulary robotic manipulation.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - VLM
  - robotic manipulation
  - spatial representation
  - task-adaptive
  - Chain of Grounding
date: 2026-05-08
content_hash: 5d955c8d9b8cf687
---

# T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs

**Conference**: NeurIPS 2025
**arXiv**: [2506.19498](https://arxiv.org/abs/2506.19498)
**Code**: [https://github.com/](https://github.com/) (not provided)
**Area**: Multimodal VLM / Robotic Manipulation
**Keywords**: VLM, robotic manipulation, spatial representation, task-adaptive, Chain of Grounding

## TL;DR

This paper proposes the T-Rex framework, which dynamically selects the optimal spatial representation extraction scheme (point / vector / 6D pose) according to task complexity, and introduces Chain of Grounding (CoG) to guide VLMs through step-by-step reasoning, enabling training-free open-vocabulary robotic manipulation.

## Background & Motivation

Vision-language models (VLMs), endowed with rich world knowledge acquired from large-scale data, are increasingly being applied to robotic manipulation tasks. Spatial representations—such as points encoding object positions and vectors encoding object orientations—serve as the bridge between VLM reasoning capabilities and real-world scenarios.

**Core Limitations of Prior Work**: Existing VLM-guided robotic methods (e.g., ReKep, VoxPoser) adopt **fixed spatial representation extraction schemes**—regardless of task complexity, the same approach is used to extract spatial information. This leads to two problems:

**Insufficient representational capacity**: Simple point representations cannot handle tasks that require object orientation information (e.g., "make the stuffed toy face the camera").

**Excessive extraction time**: Expensive 6D pose estimation is applied even to simple tasks that only require centroid points.

**Key Challenge**: Task complexity determines the type and granularity of spatial representation required, yet stronger representational capacity typically entails higher system runtime cost. How can a balance between representational capacity and efficiency be achieved?

**Key Insight**: The paper enables VLMs to autonomously determine what level of spatial representation each object requires at each task stage, and dynamically invokes the corresponding extraction tool. The CoG method is designed to explicitly guide the VLM's stage-wise reasoning process, ensuring reasoning stability.

## Method

### Overall Architecture

Given a natural language instruction and scene observation, the VLM uses CoG to decompose the instruction into multi-stage subtasks, selects the optimal spatial representation extraction scheme for each relevant object at each stage, and generates constraint functions. A low-level action sequence generator produces robot actions based on these constraints and the tracked spatial representations.

### Key Designs

1. **Task-Adaptive Heterogeneous Spatial Representation Extraction**: An extensible spatial representation extraction toolkit is constructed, comprising multiple large vision models (e.g., Grounding DINO for keypoints, FoundationPose for 6D pose). Each tool is defined as $(I_i, o_i, f_i, s_i, h_i)$, encompassing input, output type, format, implementation summary, and historical average execution time. The VLM selects the optimal tool for each object based on the task and scene: $t_{s,o}^* = \arg\max_{t \in \mathcal{R}} [P_{\text{succ}}(t|I,X,s,o) - \lambda h_t]$, trading off success probability against extraction cost.

2. **Task-Adaptive Multi-Granularity Spatial Representation Extraction**: When the VLM determines that a finer-grained spatial representation is required for a given task stage (e.g., the leg orientation of a robot dog), a "local zoom-in" strategy is triggered: SAM first segments the target object region, which is then expanded with padding and cropped into a local sub-image, upon which adaptive extraction is applied. This attention-inspired zoom-in strategy is activated only when necessary, introducing no overhead for simple tasks.

3. **Chain of Grounding (CoG)**: The reasoning process of the VLM is explicitly guided through four sequentially dependent stages:

    - **Manipulation Prompt Reasoning**: The task is decomposed into multiple stages, generating representation-agnostic manipulation prompts.
    - **Constraint Reasoning**: The required spatial constraints (in natural language) are inferred for each prompt.
    - **Tool Selection**: The Toolkit Registry is queried to select the optimal extraction tool for each object.
    - **Constraint Code Generation**: Natural language constraints are translated into executable Python functions (returning scalar costs).

### Loss & Training

T-Rex is a zero-shot method that **requires no training**. It relies entirely on the reasoning capabilities of VLMs (GPT-4.1) and pretrained visual foundation models. Constraint functions are generated as Python code and solved via numerical optimizers to produce robot action sequences.

## Key Experimental Results

### Main Results

**15 real-world open-vocabulary manipulation tasks (10 independent trials per task)**

| Task | VoxPoser | ReKep | T-Rex | T-Rex Time (s) |
|------|----------|-------|-------|---------------|
| Open Drawer | 4/10 | 2/10 | 6/10 | 14.3 |
| Pour Water | 0/10 | 3/10 | 7/10 | 24.1 |
| Close Lid of Laptop | 4/10 | 2/10 | 7/10 | 21.6 |
| Setup: Tools Insert | 0/10 | 3/10 | 7/10 | 56.3 |
| Setup: Mixed | 0/10 | 0/10 | 2/10 | 217.5 |
| **Total** | **30%** | **36.4%** | **60.7%** | **45.5** |

### Ablation Study

| Configuration | Success Rate (%) | Time (s) |
|---------------|-----------------|---------|
| T-Rex (full) | 60.7%±2.1% | 45.5±1.3 |
| w/o CoG | 52.1%±2.4% | 41.4±2.1 |
| w/o Toolkit (points only) | 30.7%±3.7% | 33.6±3.6 |
| w/o Toolkit (VLM points + vectors) | 55.0%±2.9% | 47.9±3.2 |
| w/o CoG, w/o Toolkit (points only) | 27.9%±1.5% | 30.0±0.9 |

### Key Findings

- T-Rex achieves an overall success rate of 60.7%, approximately 2× over VoxPoser (30%) and 1.7× over ReKep (36.4%).
- Removing the Toolkit (points only) reduces success rate to 30.7%, demonstrating the necessity of heterogeneous spatial representations.
- Removing CoG reduces success rate to 52.1%; CoG provides a consistent absolute gain of 8.6% with negligible additional latency.
- The primary source of system error lies in the spatial representation tracking module, rather than in VLM reasoning or extraction.
- The advantage is particularly pronounced in tasks requiring 6D pose (e.g., stuffed toy placement).

## Highlights & Insights

- The **task-driven representation selection** paradigm is highly pragmatic—different tasks genuinely require different granularities of spatial perception.
- **Strong extensibility**: New tools can be integrated by registering only a few parameters in a configuration file.
- **Zero-training deployment**: The system relies entirely on combinations of pretrained models, making it well-suited for rapid prototyping.
- **Design philosophy of CoG**: Decomposing complex one-shot VLM reasoning into four chained stages significantly reduces the risk of hallucination.

## Limitations & Future Work

- Spatial representation tracking remains the primary bottleneck, as existing tools lack continuous tracking capability.
- Success rates on complex multi-stage tasks (e.g., Mixed setup) remain low (2/10).
- The system is highly dependent on VLM reasoning quality (GPT-4.1); weaker VLMs lead to erroneous constraint generation.
- Experiments are conducted solely in tabletop manipulation settings; more complex scenarios such as mobile manipulation are not evaluated.
- Processing time is substantial (>200 s for complex tasks), making real-time deployment infeasible.

## Related Work & Insights

- Complementary to works such as ReKep: ReKep uses fixed keypoints, whereas T-Rex's Toolkit can incorporate ReKep as one of its tools.
- The chain-of-reasoning philosophy underlying CoG is analogous to Chain-of-Thought, but is specifically designed for the grounding process in robotic manipulation.
- The open registration framework of the Toolkit may inspire other applications that require the composition of multiple visual tools.

## Rating

- Novelty: ⭐⭐⭐⭐ The task-adaptive representation extraction idea is original; CoG is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ 15 real-world tasks + ablations + error analysis constitute a comprehensive evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete formal definitions, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides a practical engineering solution for VLM-driven robotic systems.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)
- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[ICLR 2026\] Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic Scenes](../../ICLR2026/multimodal_vlm/seeing_across_views_benchmarking_spatial_reasoning_of_vision-language_models_in_.md)
- [\[NeurIPS 2025\] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation](forcevla_enhancing_vla_models_with_a_force-aware_moe_for_contact-rich_manipulati.md)

<!-- RELATED:END -->
