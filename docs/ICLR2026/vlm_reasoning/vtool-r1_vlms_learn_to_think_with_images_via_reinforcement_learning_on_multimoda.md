---
title: >-
  [Paper Note] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use
description: >-
  [Multimodal VLM] This paper proposes VTool-R1, the first framework that trains VLMs via reinforcement fine-tuning to generate interleaved textual and visual intermediate reasoning steps…
tags:
  - "Multimodal VLM"
date: 2026-05-08
content_hash: f1183d911b5ed394
---

# VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use

## Paper Information
- **Conference**: ICLR 2026
- **arXiv**: [2505.19255](https://arxiv.org/abs/2505.19255)
- **Code**: [https://github.com/VTOOL-R1/vtool-r1](https://github.com/VTOOL-R1/vtool-r1)
- **Area**: Vision-Language Models / Reinforcement Fine-Tuning / Tool Use / Multimodal Reasoning
- **Keywords**: RFT, VLM, visual reasoning, tool use, GRPO, multimodal chain-of-thought

## TL;DR
This paper proposes VTool-R1, the first framework that trains VLMs via reinforcement fine-tuning to generate interleaved textual and visual intermediate reasoning steps, enabling models to "think with images."

## Background & Motivation

### Core Problem
RFT (Reinforcement Fine-Tuning) has substantially improved the reasoning capabilities of LLMs, yet attempts to replicate this success in VLMs remain confined to **purely textual reasoning**: models process images only during initial encoding, while the reasoning chain is generated entirely in text form, without intermediate visual reasoning steps.

### Why Is Pure-Text Reasoning Insufficient?
Even state-of-the-art VLMs may rely on linguistic shortcuts. For instance, when shown an image of a six-fingered hand and asked "how many fingers are there," the model may answer "five" based on the textual prior that "a hand has five fingers," ignoring the visual evidence.

### Limitations of Prior Work
- **Visual Sketchpad**: Incorporates visual steps at inference time, but lacks a training mechanism and is only effective on strong models such as GPT-4o.
- **Refocus**: Generates visual edits but relies on commercial models to pre-generate them, yielding poor results on weaker open-source models.
- **R1-VL and similar**: Only trains purely textual CoT, without visual reasoning steps.

## Method

### Core Idea

VTool-R1 integrates Python-based visual editing tools into the RFT pipeline, enabling VLMs to autonomously learn—through **outcome-driven rewards**—when and how to generate visual reasoning steps.

### Inference and Rollout Pipeline

Two-round model execution:
1. **Round 1**: The VLM generates Thought 0 (analysis of salient regions) and Action 0 (a tool call or a direct answer) conditioned on the image and question.
2. **Tool Execution**: The generated code is executed in a Python sandbox, producing an edited image $I'$.
3. **Round 2**: The VLM reasons over both the original image and the edited image to produce the final answer.

Formal representation:
$$y \sim \pi_\theta(\cdot | I, x; \texttt{T}) = \pi_\theta(\cdot | I \oplus I', x) = \pi_\theta(\cdot | I \oplus \texttt{T}(y', I), x)$$

where $\oplus$ denotes dual-image concatenated input.

### RFT Training Objective

Only the final reasoning response $y$ is optimized (not the intermediate tool call $y'$):

$$\max_{\pi_\theta} \mathbb{E}_{[I,x] \sim \mathcal{D}, y \sim \pi_\theta(\cdot|I,x;\texttt{T})} [r_\phi(I,x,y)] - \beta \mathbb{D}_{KL}[\pi_\theta(\cdot|I,x;\texttt{T}) \| \pi_{\text{ref}}(\cdot|I,x;\texttt{T})]$$

GRPO-based optimization:

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\min\left(r_{i,t}(\theta)\hat{A}_{i,t}, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\right) - \beta\mathbb{D}_{KL}[\pi_\theta||\pi_{\text{ref}}]\right]$$

### Reward Design

**Pure outcome-driven rewards** are adopted: a lightweight LLM judge evaluates whether the predicted answer matches the ground truth, assigning a reward of 1 for a match.

**Key Finding**: Process rewards (penalizing failed tool calls or rewarding successful ones) lead to reward hacking—the model either avoids tool use entirely or generates spurious "successful" tool calls.

### Visual Editing Tool Set

For table tasks:
- Highlight Column/Row: semi-transparent red overlay
- Mask Column/Row: white mask over irrelevant regions
- Draw Column/Row: red bounding box annotation

For chart tasks: analogous operations applied to individual bars in bar charts.

## Experiments

### Main Results

| Model | Configuration | Chart Split | Table Split |
|-------|---------------|-------------|-------------|
| Qwen2.5-VL 3B | Pure Run | 51.8 | 41.3 |
| Qwen2.5-VL 3B | Tool Use (no training) | 24.6 | 24.3 |
| **Qwen2.5-VL 3B** | **VTool-R1** | **64.0** | **57.9** |
| Qwen2.5-VL 7B | Pure Run | 76.2 | 64.7 |
| **Qwen2.5-VL 7B** | **VTool-R1** | **80.7** | **71.7** |
| GPT-4o | Pure Run | 82.9 | 75.7 |
| GPT-4o | Tool Use | 80.5 | 77.0 |

### Comparison with Other Methods

| Method | Chart Split | Table Split |
|--------|-------------|-------------|
| Deepeyes (7B) | 60.0 | - |
| R1-VL (7B) | 63.8 | 45.4 |
| **VTool-R1 (7B)** | **80.7** | **71.7** |

### Key Findings

1. **RFT enables more effective tool use**: After training, both the 3B and 7B models learn to use tools effectively.
2. **Tool use is non-monotonically increasing**: The frequency and success rate of tool calls fluctuate during training, indicating that the model learns to use tools selectively.
3. **Outcome-driven rewards are most reliable**: Process rewards lead to reward hacking.
4. **VTool-R1 substantially outperforms Deepeyes**: 80.7 vs. 60.0 on the Chart Split.
5. **Convergence is achieved within approximately 50 training steps.**

### Failure Case Analysis
- Correct visual steps are generated but second-round reasoning produces an incorrect answer.
- Visual augmentations contain minor artifacts (e.g., digits occluded by bounding boxes).
- The model incorrectly judges that no tool is needed, yet answers incorrectly without one.
- Tool code execution fails.

## Highlights & Insights

1. **First RFT framework to train VLMs to generate multimodal chain-of-thought.**
2. **Elegant design**: Only the final response is optimized, allowing the model to autonomously decide whether to invoke a tool.
3. **Practically effective**: The 3B model, after training, matches or surpasses GPT-4o's tool-use capability.
4. **In-depth training dynamics analysis**: The evolution of tool-use frequency and success rate reveals adaptive behavior.

## Limitations & Future Work

1. Currently supports only single-round tool calls; multi-round visual reasoning is left for future work.
2. The tool set is limited to selective attention operations and has not yet been extended to more complex visual tools.
3. Multi-image input support from the VLM is required.
4. No precise oracle verifier for tool-call correctness is available.
5. Training is computationally demanding (8×H200 GPUs for the 32B model).

## Related Work & Insights

- **Visual CoT**: ViperGPT (via Python programs), Visual Sketchpad (inference-time sketchpad)
- **LLM/VLM Tool Use**: Search-R1, ReTool — RFT with textual tools
- **VLM RFT**: R1-V, Vision-R1 — text-only reasoning chains
- **Concurrent Work**: Deepeyes, OpenThink-IMG — different tool and task designs

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First successful training of VLMs to generate multimodal reasoning chains
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-scale model comparisons with thorough training dynamics analysis
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-defined formulations
- **Value**: ⭐⭐⭐⭐ — Open-source framework with practical applicability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)
- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](../../AAAI2026/multimodal_vlm/recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)
- [\[ICCV 2025\] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](../../ICCV2025/multimodal_vlm/r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)

</div>

<!-- RELATED:END -->
