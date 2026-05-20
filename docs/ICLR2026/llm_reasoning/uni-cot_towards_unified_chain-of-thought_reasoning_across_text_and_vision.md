---
title: >-
  [Paper Note] Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision
description: >-
  [ICLR2026][LLM Reasoning][Multimodal chain-of-thought] This paper proposes Uni-CoT, a hierarchical macro-micro reasoning framework that decomposes multimodal CoT into macro-level task planning (decomposing complex tasks…
tags:
  - "ICLR2026"
  - "LLM Reasoning"
  - "Multimodal chain-of-thought"
  - "interleaved text-image reasoning"
  - "macro-micro hierarchy"
  - "MDP self-reflection"
  - "unified generation"
date: 2026-05-08
content_hash: 614b5716ae75c9ea
---

# Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision

**Conference**: ICLR2026
**arXiv**: [2508.05606](https://arxiv.org/abs/2508.05606)  
**Code**: [https://github.com/Fr0zenCrane/UniCoT](https://github.com/Fr0zenCrane/UniCoT)  
**Area**: LLM Reasoning
**Keywords**: Multimodal chain-of-thought, interleaved text-image reasoning, macro-micro hierarchy, MDP self-reflection, unified generation

## TL;DR
This paper proposes Uni-CoT, a hierarchical macro-micro reasoning framework that decomposes multimodal CoT into macro-level task planning (decomposing complex tasks into sub-goals) and micro-level sub-task execution (MDP-style self-reflective iterative refinement). Through an attention mask design, the complexity is reduced from $O(T^2)$ to $O(T)$. The method surpasses the BAGEL baseline by +0.02 on GenEval, achieving unified reasoning over interleaved text and images.

## Background & Motivation

**Background**: CoT reasoning has been extensively validated for pure-text LLMs, but multimodal (text + vision) CoT remains in its early stages. Existing multimodal reasoning methods either rely solely on textual CoT while ignoring intermediate visual artifacts, or adopt pipeline-style loosely coupled MLLMs with external image generators.

**Limitations of Prior Work**: (a) Pure-text RL reasoning methods perform poorly on visually grounded tasks (e.g., geometry, navigation); (b) Interleaved text-image sequences are extremely long (~10,000 tokens per step), making naive autoregressive modeling with $O(T^2)$ complexity intractable; (c) Long interleaved sequences lead to training instability.

**Key Challenge**: Multimodal reasoning requires generating intermediate visual states to support inference (e.g., puzzle solving requires observing intermediate results), yet each visual state consumes thousands of tokens, rendering standard CoT computationally and practically infeasible in multimodal settings.

**Goal**: How to efficiently realize interleaved text-vision CoT reasoning?

**Key Insight**: A hierarchical design — the macro level handles task planning (attending only to sub-goal descriptions), while the micro level handles sub-task execution (MDP-style, attending only to adjacent states), with attention masks restricting the visible context to reduce computation.

**Core Idea**: Macro-micro hierarchy + MDP self-reflection + attention masking = multimodal CoT with linear complexity.

## Method

### Overall Architecture
Built upon BAGEL (decoder Transformer + MoE), supporting image understanding (SigLIP2 ViT → 4,900 tokens) and image generation (FLUX VAE → 4,096 tokens). The macro planner decomposes a complex task into $M$ sub-goals → the micro executor performs MDP-style iteration for each sub-goal: initial attempt → self-evaluation → text/image correction → retry.

### Key Designs

1. **Macro Planner/Summarizer**:

    - Function: Decomposes the task into $M$ sub-goals $z_{plan} = \{z_1, ..., z_M\}$, supporting sequential or parallel decomposition; aggregates results upon completion.
    - Mechanism: The macro attention mask attends only to the input, sub-goal descriptions, and the final output of each step, skipping intermediate reasoning details.
    - Design Motivation: Prevents the high-level planner from being distracted by low-level details, while reducing complexity from $O(T^2)$ to $O(T^2/M)$.

2. **Micro MDP Self-Reflective Executor (Micro Operator)**:

    - Function: Performs MDP-style iteration for each sub-goal — initial attempt $h_0$ → evaluation score $eval_t$ → text/image correction → new state $h_{t+1}$.
    - Mechanism: Markov design — the current state $h_t$ depends only on the previous state $h_{t-1}$ and the sub-goal $z_i$, without attending to earlier history.
    - Design Motivation: Further reduces complexity from $O(T^2/M)$ to $O(T)$ (linear), realized via attention masking.

3. **Computational Complexity Analysis**:

    - Naive autoregressive CoT: $O(T^2)$, ~10,000 tokens per step — infeasible.
    - Hierarchical decomposition: $O(T^2/M)$.
    - Hierarchical + MDP (Uni-CoT): $O(T)$ — near-linear, practically trainable.

### Loss & Training
$\mathcal{L}_{joint} = \lambda_{CE} \cdot \mathcal{L}_{CE}^{text} + \mathcal{L}_{MSE}^{image}$. The micro level incorporates four auxiliary tasks (text/image action generation, next-state prediction, reward estimation). Training uses 31K samples (11K macro interleaved pairs + 20K micro examples) on 8×A100 GPUs for approximately one week.

## Key Experimental Results

### Main Results

**GenEval Image Generation Benchmark**:

| Metric | Uni-CoT | BAGEL | FLUX.1-dev | Janus-Pro-7B |
|--------|---------|-------|-----------|-------------|
| Single Object | **0.99** | 0.99 | 0.98 | 0.99 |
| Two Objects | **0.95** | 0.92 | 0.93 | 0.89 |
| Counting | **0.82** | 0.78 | 0.75 | 0.59 |
| Color Attribution | **0.69** | 0.64 | 0.65 | 0.66 |
| Overall | **0.81** | 0.79 | 0.82 | 0.80 |

The most notable gains are in counting (+0.04) and two-object composition (+0.03).

### Ablation Study

| Method | Complexity | Per-step Token Cost |
|--------|-----------|-------------------|
| Naive Autoregressive CoT | $O(T^2)$ | ~10,000 tokens (infeasible) |
| Hierarchical Decomposition | $O(T^2/M)$ | Reduced by factor $M$ |
| **Hierarchical + MDP (Uni-CoT)** | **$O(T)$** | **Linear** |

### Key Findings
- **Intermediate visual artifacts are critical for reasoning**: Pure-text CoT fails on geometry and puzzle tasks that require observing intermediate visual results.
- **MDP Markov assumption is effective**: Attending only to the previous state suffices for high-quality correction without requiring full history.
- **Self-reflective iteration genuinely improves quality**: The model can self-evaluate and correct errors, particularly for counting and spatial relationships.

## Highlights & Insights
- **First practical resolution of computational feasibility for multimodal CoT**: The reduction from $O(T^2)$ to $O(T)$ makes interleaved text-image reasoning tractable on real hardware. The key insight is that "reasoning does not require attending to all history" — the combination of hierarchical structure and Markov attention masking is the central innovation.
- **Unified reasoning framework for understanding and generation**: The MoE architecture built on BAGEL seamlessly switches between understanding and generation pathways, naturally interleaving "perceiving" and "generating" actions during inference.
- **Formal introduction of MDP into multimodal CoT**: Modeling the self-reflection process as an MDP (states, actions, rewards) lays a formal foundation for future RL-based optimization of multimodal reasoning.

## Limitations & Future Work
- **Limited experimental scope**: Results are primarily demonstrated on GenEval for generation; performance on understanding benchmarks (MMBench, MathVista, etc.) is not sufficiently reported in the main paper.
- **Small training data scale**: Only 31K samples, limiting the depth of acquired reasoning capabilities.
- **Insufficient baseline comparison**: Comparisons are largely restricted to the BAGEL baseline, and the +0.02 overall gain is modest. Comparisons with closed-source models such as GPT-4o and Gemini are absent.
- **Future directions**: (1) Replacing SFT with RL (e.g., NRT/GRPO) to train the micro reasoner; (2) Supporting more iterative self-reflection steps; (3) Scaling training data to 100K+ samples.

## Related Work & Insights
- **vs. Pure-text CoT reasoning (o1, R1)**: These methods produce only textual reasoning chains and are ill-suited for visually grounded tasks. Uni-CoT is the first to enable models to "reason and draw simultaneously."
- **vs. Janus-Pro**: Supports both understanding and generation but lacks CoT capability, limiting generation quality to single-pass inference.
- **vs. CogCoM/Visual Sketchpad**: These approaches rely on external tools to produce intermediate visual artifacts, whereas Uni-CoT is fully end-to-end without dependence on external tools.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First practical multimodal CoT framework; the combination of macro-micro hierarchy, MDP, and attention masking is a distinctive contribution.
- Experimental Thoroughness: ⭐⭐⭐ Core concepts are well validated, but evaluation datasets and baseline comparisons are limited.
- Writing Quality: ⭐⭐⭐⭐ Complexity analysis is clearly presented, though the paper structure is somewhat lengthy.
- Value: ⭐⭐⭐⭐⭐ Establishes a computationally feasible framework for multimodal reasoning; open-source code facilitates community follow-up.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ACL2026/llm_reasoning/aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)
- [\[ICLR 2026\] AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)
- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[AAAI 2026\] L2V-CoT: Cross-Modal Transfer of Chain-of-Thought Reasoning via Latent Intervention](../../AAAI2026/llm_reasoning/l2v-cot_cross-modal_transfer_of_chain-of-thought_reasoning_v.md)
- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](../../NeurIPS2025/llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)

</div>

<!-- RELATED:END -->
