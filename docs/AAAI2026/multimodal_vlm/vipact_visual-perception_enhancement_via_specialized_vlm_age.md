---
title: >-
  [Paper Note] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use
description: >-
  [AAAI 2026][Multimodal VLM][VLM Agent] VipAct proposes a multi-agent collaboration framework that significantly improves VLM performance on fine-grained visual perception tasks through three-tier collaboration: an Orches…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "VLM Agent"
  - "Multi-Agent Collaboration"
  - "Tool Use"
  - "Fine-Grained Visual Perception"
  - "System-2 Reasoning"
date: 2026-05-08
content_hash: 4eb764e8ffd16317
---

# VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use

**Conference**: AAAI 2026
**arXiv**: [2410.16400](https://arxiv.org/abs/2410.16400)
**Code**: None
**Area**: Multimodal VLM / Agent
**Keywords**: VLM Agent, Multi-Agent Collaboration, Tool Use, Fine-Grained Visual Perception, System-2 Reasoning

## TL;DR
VipAct proposes a multi-agent collaboration framework that significantly improves VLM performance on fine-grained visual perception tasks through three-tier collaboration: an Orchestrator Agent (task analysis, planning, and coordination), specialized agents (captioning, comparison, and visual prompt interpretation), and vision expert models (depth estimation, object detection, segmentation, etc.). The framework improves accuracy on Blink from 63.74% (zero-shot GPT-4o) to 73.79%.

## Background & Motivation
**Background**: VLMs such as GPT-4o excel at general vision-language understanding tasks, but struggle with fine-grained visual perception tasks that require pixel-level analysis—such as line crossing detection, object boundary judgment, and depth comparison—despite these tasks being trivial for humans.

**Limitations of Prior Work**: Existing solutions fall into two categories: (1) Visual programming methods (ViperGPT, VisProg) use LLMs to generate code that invokes visual tools, but rely on predefined tool sets, do not support visual prompts or multi-image input, and generalize poorly; (2) Text prompting methods (CoT, ToT) are effective on textual tasks but yield inconsistent results on fine-grained visual perception, as reasoning steps often become decoupled from visual content.

**Key Challenge**: VLMs inherently lack pixel-level visual analysis capability, yet simply invoking external tools sacrifices flexibility and global reasoning. There is a fundamental need to organically combine VLMs' planning and reasoning capabilities with the precise perceptual capabilities of specialized models.

**Goal**: How can VLMs achieve pixel-level fine-grained visual perception while retaining flexible reasoning ability?

**Key Insight**: Inspired by multi-agent collaboration in the LLM agent literature, the paper decomposes complex visual tasks across agents and tool models with distinct roles, with an Orchestrator Agent responsible for integrating evidence and performing final reasoning.

**Core Idea**: Enhance VLM System-2 reasoning and fine-grained visual perception through multi-agent collaboration consisting of an orchestrator, specialized agents, and vision expert models.

## Method

### Overall Architecture
VipAct consists of three tiers: (1) an **Orchestrator Agent** responsible for task analysis, planning, tool selection, evidence aggregation, and final reasoning; (2) **three specialized agents** handling sub-tasks (image captioning, visual prompt interpretation, and image comparison); and (3) **five vision expert models** providing pixel-level perceptual information (depth estimation, object detection, segmentation, visual prompt detection, and embedding similarity). The entire pipeline operates on an iterative Think–Act–Observe loop.

### Key Designs

1. **Orchestrator Agent**:

    - **Function**: Receives images and queries, analyzes task requirements, formulates plans, selects appropriate tools/agents, aggregates all evidence, and derives the final answer.
    - **Mechanism**: Employs a ReAct-style iterative reasoning framework. A key innovation is that the Orchestrator Agent **directly receives image input** (rather than only a textual query), enabling it to formulate more precise plans based on visual content. The orchestrator also incorporates conflict resolution and fact-checking capabilities, rather than blindly trusting tool outputs.
    - **Design Motivation**: Compared to purely text-based planning approaches such as ViperGPT, image input allows the orchestrator to identify salient visual features and set parameters (e.g., *focus*) precisely. Ablation results show a substantial performance drop when image input is removed (73.79 → 64.36).

2. **Specialized Agents**:

    - **Function**: Provide the orchestrator with detailed visual analysis within isolated environments via function calling.
    - Three specialized agents: (a) **Focused Image Captioning**—fine-grained image description with an optional *focus* parameter, e.g., describing only "the red car and background buildings"; (b) **Visual Prompt Description**—dedicated interpretation of visual annotations in images (circles, arrows, bounding boxes, etc.) and the regions or objects they indicate; (c) **Focused Image Comparison**—detailed multi-image comparative analysis with support for specifying elements of interest.
    - **Design Motivation**: A core insight is that multi-agent collaboration elicits more detailed System-2 reasoning—specialized agents focus exclusively on analyzing visual information in isolated environments, free from formatting instructions and other distractions, generating 80% more tokens than single-agent baselines. Ablations confirm consistent performance degradation when multi-agent structure is removed.

3. **Vision Expert Models**:

    - **Function**: Supply pixel-level perceptual information that is typically absent from VLM pretraining data.
    - Five tools are included: Depth-Anything-V2 (depth estimation), SAM (segmentation), YOLOv8 (object detection), Visual Prompt Detector (visual annotation localization), and CLIP (embedding similarity).
    - **Novelty**: Expert models return **dual-modality outputs—text plus processed images**—where the processed images are fed directly back into the orchestrator's subsequent reasoning loop. This makes VipAct one of the earliest frameworks to incorporate visual information into the agent reasoning workflow.
    - **Design Motivation**: VLMs cannot natively perform pixel-level operations such as depth estimation or precise segmentation, necessitating supplementary specialized models.

### Loss & Training
VipAct is a training-free framework that relies entirely on carefully designed prompts and API calls to existing models.

## Key Experimental Results

### Main Results — Blink Benchmark (GPT-4o)

| Method | Similarity | Counting | Depth | Spatial | Average |
|--------|-----------|----------|-------|---------|---------|
| Zero-shot | 65.44 | 50.83 | 64.52 | 79.92 | 63.74 |
| CoT | 63.70 | 65.00 | 73.39 | 82.52 | 65.85 |
| SoM | 63.70 | 43.33 | 68.55 | 76.22 | 60.13 |
| MM-ReAct | - | 30.00 | 0.81 | 63.64 | - |
| ViperGPT | - | 29.17 | 0.00 | 48.95 | - |
| **VipAct** | **81.48** | **70.00** | **90.80** | **86.70** | **73.79** |

### Ablation Study — Blink (GPT-4o, Average across tasks)

| Configuration | Depth | Multi-view | Notes |
|--------------|-------|-----------|-------|
| VipAct (Full) | 90.80 | 62.63 | Complete model |
| w/o Multi-agent | 75.00 | 48.87 | Removing multi-agent causes notable degradation |
| w/o Visual Input | 69.35 | 48.12 | Removing orchestrator image input causes largest drop |
| w/o Spec. Agents | 85.62 | 46.75 | Removing specialized agents |
| w/o Vision Expert | 72.58 | 56.40 | Removing vision expert models |

### Key Findings
- VipAct surpasses all baselines on all 10 Blink sub-tasks, with an average improvement of approximately 10 percentage points (63.74 → 73.79).
- The most significant gain is on depth estimation (64.52 → 90.80), attributable to the Depth-Anything-V2 expert model.
- The SoM visual prompting method actually degrades VLM performance on most tasks (63.74 → 60.13), as the overlaid annotations occlude original visual content.
- Image input to the orchestrator is critical—reverting to text-query-only planning leads to substantial performance drops.
- Error analysis reveals core VLM bottlenecks: fine-grained spatial reasoning (24%), small object perception failure (17%), and difficulty distinguishing visual prompts (15%).

## Highlights & Insights
- **Multi-agent elicits System-2 reasoning**: Multi-agent collaboration increases generated token count by over 80%, yielding more detailed reasoning. This demonstrates that decomposing problems across specialized roles facilitates deeper thinking in VLMs.
- **Visual feedback loop**: Expert model outputs—such as depth maps—are fed as processed images directly back into subsequent reasoning steps, forming a closed loop of "vision → text → (tool call) → vision → text." Most agent frameworks rely solely on textual intermediate results.
- **Focus parameter design**: Allowing the orchestrator to specify a descriptive or comparative focus based on task requirements enables flexible attention control—a design principle transferable to other agent frameworks.

## Limitations & Future Work
- The framework relies entirely on closed-source VLMs (GPT-4o / Gemini / Claude); open-source VLMs cannot run VipAct due to insufficient instruction-following capability.
- Inference costs are high—multiple agent invocations and repeated VLM API calls raise practical deployment concerns.
- The five vision expert models are fixed; new tasks require manual tool addition.
- Fundamental VLM perceptual bottlenecks (spatial reasoning 24%, small objects 17%) cannot be resolved at the framework level.
- Gains on MMVP are limited (68.0 → 70.7), suggesting that tool assistance provides limited benefit for certain perceptual patterns.

## Related Work & Insights
- **vs. MM-ReAct**: Although MM-ReAct also integrates VLMs with tools, it lacks multi-agent collaboration, does not support image-based planning input or multi-image input, and is nearly unusable on Blink.
- **vs. ViperGPT / VisProg**: Code-generation approaches rely solely on textual queries and do not process images, making them unable to handle visual prompts or fine-grained tasks.
- **vs. Direct prompting (CoT / ToT)**: While text prompting can elicit reasoning, it frequently becomes decoupled from visual content; VipAct obtains ground-truth perceptual data through tool use.
- **Insight**: The value of agent frameworks lies not only in tool invocation, but in the collaborative paradigm of "task decomposition + focused analysis + evidence aggregation."

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of multi-agent collaboration and vision experts for VLM perception tasks is a relatively novel attempt, though the architecture itself is not complex.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Rich ablation studies, error analysis, cross-VLM validation, and fairness analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, in-depth analysis, and an extremely detailed appendix.
- **Value**: ⭐⭐⭐⭐ Reveals the enhancement effect of multi-agent collaboration on VLM reasoning, though reliance on closed-source APIs limits practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[AAAI 2026\] Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation](zero-reference_joint_low-light_enhancement_and_deblurring_via_visual_autoregress.md)
- [\[AAAI 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semantic-sensitive_underwater_image_enhancement_with_vlm.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[CVPR 2026\] Proof-of-Perception: Certified Tool-Using Multimodal Reasoning with Compositional Conformal Guarantees](../../CVPR2026/multimodal_vlm/pop_proof_of_perception_conformal_reasoning.md)

</div>

<!-- RELATED:END -->
