---
title: >-
  [Paper Note] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition
description: >-
  [CVPR2026][Multimodal VLM][visual token compression] This paper proposes AdaptVision, which enables VLMs to autonomously determine the minimum number of visual tokens required per sample through a coarse-to-fine active v…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "visual token compression"
  - "adaptive visual acquisition"
  - "reinforcement learning"
  - "tool calling"
  - "efficient VLM"
date: 2026-05-08
content_hash: 22979968e5cb759b
---

# AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition

**Conference**: CVPR2026  
**arXiv**: [2512.03794](https://arxiv.org/abs/2512.03794)  
**Code**: [github.com/adaptvision/adaptvision](https://github.com/adaptvision/adaptvision)  
**Area**: Multimodal VLM  
**Keywords**: visual token compression, adaptive visual acquisition, reinforcement learning, tool calling, efficient VLM

## TL;DR
This paper proposes AdaptVision, which enables VLMs to autonomously determine the minimum number of visual tokens required per sample through a coarse-to-fine active visual mechanism and reinforcement learning training, combined with Decoupled Turn Policy Optimization (DTPO) to achieve an optimal trade-off between efficiency and accuracy.

## Background & Motivation
1. VLMs rely on a large number of visual tokens (e.g., Qwen2.5-VL generates 2,678 tokens for a 2048×1024 image), incurring significant computational and memory overhead.
2. Existing efficient VLM methods compress visual tokens at a fixed ratio (e.g., pruning 50%), lacking adaptive capability for varying task demands.
3. Cognitive neuroscience reveals that the human visual system operates as "active vision"—first capturing coarse low-frequency information, then performing fine-grained analysis on critical regions.
4. Recent "thinking with images" paradigms (e.g., zoom/crop operations in DeepEyes and Mini-o3) demonstrate the potential of active visual reasoning.
5. Applying "thinking with images" to visual token compression—letting the model decide how many visual tokens suffice—remains underexplored.
6. Training such a dual-objective policy with standard GRPO poses challenges of ambiguous credit assignment and optimization imbalance.

## Method

### Overall Architecture
AdaptVision adopts a coarse-to-fine pipeline: it first processes a quarter-resolution image ($I_{low}$) using only 25% of visual tokens. The model autonomously decides whether to answer directly or invoke a cropping tool via `<tool_call>[x1,y1,x2,y2]</tool_call>` to retrieve a critical region $I_{crop}$ from the high-resolution image for subsequent reasoning. The total visual token count is $n_{img} = n_{low} + \mathbf{1}_{tool} \cdot n_{crop}$.

### Key Designs

**Reward Function Design**: Composed of an outcome reward $\mathcal{R}_{oc}$ and a tool reward $\mathcal{R}_{tool}$.

- **Accuracy Reward** $\mathcal{R}_{acc}$: LLM-as-judge evaluates answer correctness (1/0).
- **Format Reward** $\mathcal{R}_{form}$: Enforces compliance of `<think>`, `<answer>`, and `<tool_call>` tags (0.5/0).
- **Balance Reward** $\mathcal{R}_{bal}$: Penalizes correct answers obtained via tool calls by 0.1; also penalizes low-confidence direct correct answers by 0.1 to prevent "lucky guessing."
- **Crop Reward** $\mathcal{R}_{crop}$: GPT-4o evaluates whether the cropped region contains information necessary for answering.
- **Area Penalty** $\mathcal{R}_{area}$: Penalizes excessively large cropped regions to incentivize minimal token usage.

**DTPO (Decoupled Turn Policy Optimization)**:

1. **Decoupled Learning Objectives**: Generated tokens are functionally separated into tool tokens (first turn) and answer tokens (second turn), with gradient signals normalized independently to address the under-optimization of tool tokens in long sequences.
2. **Decoupled Advantage Estimation**: Independent advantage values $A^{tool}$ and $A^{oc}$ are computed for the tool reward and outcome reward respectively, resolving ambiguous credit assignment.

### Loss & Training
A GRPO-based PPO-style clipping loss is employed, with separate normalization factors and advantage values applied to tool tokens and answer tokens:
$$\mathcal{J}_{DTPO} = \mathcal{J}_{tool} + \mathcal{J}_{answer}$$
Each component is normalized independently to prevent gradient signal dilution across the two-turn sequence.

## Key Experimental Results

### Main Results: Comparison with Existing Efficient VLM Methods

| Method | #Token↓ | ChartQA | OCRBench | DocVQA | MME | MathVista | Avg.↑ |
|------|---------|---------|----------|--------|-----|-----------|-------|
| Vanilla (Qwen2.5-VL-7B) | 100% | 79.8 | 81.5 | 95.1 | 2316 | 68.2 | 100% |
| Down-Sample | 25% | 62.9 | 68.8 | 94.3 | 2270 | 62.2 | 92.1% |
| FastV (50%) | 50% | 72.6 | 75.8 | 93.6 | 2308 | 63.7 | 95.8% |
| VisionZip (50%) | 50% | 71.5 | 70.5 | 93.8 | 2209 | 64.1 | 94.8% |
| **AdaptVision** | **<50%** | **Surpasses above** | **Surpasses above** | **Surpasses above** | **Surpasses above** | **Surpasses above** | **Highest** |

### Ablation Study: Contribution of DTPO Components

| Setting | Avg. Performance | Token Usage |
|------|-----------------|-------------|
| GRPO baseline | Lower; unstable training | Tool calls rapidly collapse to overuse |
| + Decoupled normalization | Significant improvement | Stable tool usage rate |
| + Decoupled advantage estimation | Best | Optimal token efficiency |

### Key Findings
- AdaptVision achieves performance superior to state-of-the-art efficient VLM methods while consuming significantly fewer visual tokens.
- Direct training with GRPO leads to an unstable process where tool usage first biases toward direct answering and subsequently collapses into excessive tool invocation.
- Both decoupled designs in DTPO contribute critically to training stability and performance improvement.

## Highlights & Insights
- The coarse-to-fine mechanism of human active vision is elegantly incorporated into VLM efficiency optimization.
- The proposed DTPO algorithm is generalizable and applicable to any multi-turn RL training scenario involving tool calls.
- The multi-layered reward design (accuracy + format + balance + crop + area) reflects a deep understanding of practical training dynamics.

## Limitations & Future Work
- Only a single tool call (1-turn crop) is supported; multi-turn iterative refinement is not explored.
- The crop reward relies on GPT-4o evaluation, resulting in relatively high training costs.
- Experiments are conducted on Qwen2.5-VL-7B; effectiveness on larger-scale models remains to be verified.

## Related Work & Insights
- Compared to VisionThink: VisionThink only coarsely switches between low and high resolution, whereas AdaptVision supports region-level adaptive cropping with greater flexibility.
- Contrasted with fixed-compression methods such as FastV and SparseVLM: a paradigm shift from passive to active compression.
- The decoupling philosophy of DTPO may inspire solutions to gradient conflict issues in multi-objective RL training.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel perspective combining active vision and RL for token compression; DTPO is innovative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (9 VQA benchmarks; comprehensive ablation study)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation; rigorous methodological derivation)
- Value: ⭐⭐⭐⭐ (Practical direction for efficient VLMs; DTPO is transferable)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)
- [\[CVPR 2026\] Phantasia: Context-Adaptive Backdoors in Vision Language Models](phantasia_context-adaptive_backdoors_in_vision_language_models.md)
- [\[CVPR 2026\] SEATrack: Simple, Efficient, and Adaptive Multimodal Tracker](seatrack_multimodal_tracker.md)
- [\[AAAI 2026\] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks](../../AAAI2026/multimodal_vlm/tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](aif_adaptive_information_flow_vlm.md)

</div>

<!-- RELATED:END -->
