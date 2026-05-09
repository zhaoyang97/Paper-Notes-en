---
title: >-
  [Paper Note] From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning
description: >-
  [ICCV 2025][Multimodal VLM][multi-image reasoning] This paper proposes the MIR benchmark, comprising 22,257 multi-image interleaved reasoning QA pairs with five-stage reasoning steps, and introduces a progressive curriculum learning strategy that trains MLLMs from easy to hard samples to improve multi-image interleaved reasoning capability.
tags:
  - ICCV 2025
  - Multimodal VLM
  - multi-image reasoning
  - interleaved data
  - benchmark
  - curriculum learning
  - MLLM
date: 2026-05-08
content_hash: 2a6049b7ecb6fc4d
---

# From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning

**Conference**: ICCV 2025
**arXiv**: [2509.17040](https://arxiv.org/abs/2509.17040)
**Code**: [https://github.com/Shelly-coder239/MIRBench](https://github.com/Shelly-coder239/MIRBench)
**Area**: Multimodal VLM
**Keywords**: multi-image reasoning, interleaved data, benchmark, curriculum learning, MLLM

## TL;DR

This paper proposes the MIR benchmark, comprising 22,257 multi-image interleaved reasoning QA pairs with five-stage reasoning steps, and introduces a progressive curriculum learning strategy that trains MLLMs from easy to hard samples to improve multi-image interleaved reasoning capability.

## Background & Motivation

Multi-image Interleaved Reasoning requires models to jointly understand multiple images alongside their associated textual context, posing unique challenges beyond single-image and non-interleaved multi-image tasks. Existing multi-image benchmarks suffer from three critical limitations:

**Insufficient evaluation of multi-image reasoning**: Existing benchmarks focus primarily on basic multi-image QA without integrating interleaved textual information, overlooking complex reasoning paradigms such as Text2Region and Region2Region.

**Lack of step-by-step reasoning**: Prior work focuses only on final answer accuracy, ignoring intermediate reasoning steps.

**Data leakage risk**: Most datasets are sourced from publicly available data already used in MLLM pretraining.

This paper addresses these issues by constructing a large-scale benchmark containing 138,277 images and introducing a progressive training methodology.

## Method

### Overall Architecture

The MIR benchmark combined with a stage-wise curriculum learning approach. The benchmark covers three major categories (Sequential / Spatial / Analytical) and 12 subtasks. Each instance is annotated with a 5-step structured reasoning process: Summary → Caption → Text2Region → Region2Region → Conclusion. The training strategy first establishes foundational understanding using easy samples, then progressively introduces harder samples.

### Key Designs

1. **Adaptive Difficulty Filter**: Each question is fed to Qwen-VL2.5 for 10 inference passes; samples with accuracy ≥70% are classified as easy, while those below 70% are classified as hard. This probability-based categorization ensures objectivity and reliability.

2. **Stage-wise Curriculum Learning**: Training proceeds in two major phases — first, fine-tuning on easy samples to establish baseline capability; then, multi-stage progressive training on hard samples. Each stage samples 40% from the hard sample pool and gradually reduces the amount of reasoning guidance provided as input:

    - Stage 1: Q + all reasoning steps → A (model only outputs the answer)
    - Stage 2: Q + first 4 steps → A + Step 5 (model generates the conclusion)
    - Stages 3–5: progressively shift more reasoning steps from input to output, until the model autonomously generates the complete reasoning chain and final answer from the raw question alone

3. **Multi-strategy Data Construction Pipeline**: Different strategies are applied for different task types — Spatial tasks use 3D geometric platforms to generate multi-view projections; Sequential tasks rely on video frame extraction and positional change computation; Analytical tasks combine targeted web crawling and text-to-image synthesis. Q&A generation combines predefined templates with LLM assistance, and reasoning steps are produced through human annotation of key steps, LLM-assisted generation, and multi-round human verification.

### Loss & Training

- Based on standard fine-tuning protocols of respective MLLMs (e.g., Qwen2-VL, LLaVA-OneVision)
- Easy samples are trained first to establish foundational understanding; hard samples are introduced progressively across stages
- 40% of hard data is sampled at each stage, with structured adjustment of input-output allocation
- Realizes a paradigm shift from "learning to reason with guidance" to "autonomous reasoning"

## Key Experimental Results

### Main Results

| Method | Spatial | Sequential | Analytical | Avg (in-domain) | MIRBENCH | BLINK | MUIRBench |
|------|---------|------------|------------|-----------------|----------|-------|-----------|
| Qwen2-VL | 29.00% | 56.60% | 45.96% | 40.44% | 50.04% | 50.34% | 45.67% |
| Qwen2-VL (Tuning) | 37.31% | 58.33% | 49.59% | 45.15% | 53.45% | 52.19% | 45.51% |
| Qwen2-VL (Ours) | **40.42%** | **59.65%** | **52.93%** | **51.76%** | **54.47%** | **56.61%** | **46.67%** |
| LLaVA-OneVision | 42.31% | 36.53% | 38.12% | 39.31% | 45.91% | 49.00% | 40.61% |
| LLaVA-OneVision (Tuning) | 45.17% | 39.29% | 41.20% | 42.36% | 47.04% | 50.34% | 42.22% |
| LLaVA-OneVision (Ours) | **50.34%** | **41.27%** | **45.99%** | **47.60%** | **49.23%** | **52.93%** | **45.34%** |

### Ablation Study

Qwen2-VL performance across stages on MIR:

| Stage | Spatial | Sequential | Analytical | Average |
|------|---------|------------|------------|---------|
| Stage 1 (full guidance) | 35.19% | 32.22% | 44.85% | 38.01% |
| Stage 2 | 34.60% | 35.55% | 38.97% | 36.73% |
| Stage 3 | 40.57% | 33.05% | 43.52% | 41.03% |
| Stage 4 | 46.29% | 34.44% | 48.52% | 45.97% |
| Stage 5 (autonomous reasoning) | **50.96%** | **36.66%** | **58.82%** | **51.76%** |

### Key Findings

- The proposed curriculum learning strategy achieves 7%+ improvement in-domain and 1%–5% out-of-domain, substantially outperforming standard fine-tuning (~2%)
- A performance dip occurs at Stage 2 (36.73%) due to the abrupt increase in task complexity; Stages 3–5 show consistent recovery and significant gains
- Instruction-following capability is critical to progressive learning — stronger instruction following correlates with better learning outcomes
- Sequential tasks show the smallest gains, as static frames extracted from video inadequately represent temporal dynamics
- After training, models maintain the structured reasoning pipeline: Summary → Caption → Text2Region → Region2Region → Conclusion

## Highlights & Insights

- **Elegant reasoning step design**: The 5-step structured reasoning process (Summary / Caption / Text2Region / Region2Region / Conclusion) clearly maps to the core cognitive processes underlying multi-image interleaved understanding
- **The "easy-to-hard" curriculum learning strategy** demonstrates substantial effectiveness in multimodal reasoning scenarios, with a clear advantage over direct fine-tuning
- **Rigorous data construction**: Multi-source collection (self-captured content / short-video platforms / educational resources) combined with semi-automatic annotation and multi-round human verification effectively mitigates data leakage
- The MIR dataset is large-scale (22,257 QA pairs, 138,277 images), with an average of 6 images per instance and approximately 3,970 characters per reasoning chain

## Limitations & Future Work

- Evaluation is currently limited to MCQ format; open-ended QA is not addressed
- Images in Sequential tasks are extracted from video frames, potentially losing critical temporal information
- The performance dip at Stage 2 suggests that stage transition design may require finer-grained strategies (e.g., smoother transitions or additional data)
- Curriculum learning training is not evaluated on closed-source models (e.g., GPT-4o, Claude)
- Reasoning step generation relies on LLMs, which may introduce systematic biases

## Related Work & Insights

- Compared to multi-image benchmarks such as MUIRBench and MMIU, MIR is the first to systematically incorporate interleaved reasoning steps
- The "progressive guidance reduction" approach in curriculum learning is generalizable to other tasks requiring step-by-step reasoning
- Unlike Chain-of-Thought prompting, this method internalizes the reasoning process through training rather than relying solely on inference-time prompting

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of interleaved multi-image reasoning and progressive curriculum learning is novel and practically motivated
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 5 MLLMs × 3 training strategies × multiple benchmarks, with sufficient ablation analysis
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, with clear figures and intuitive case studies
- **Value**: ⭐⭐⭐⭐ Provides an important benchmark and an effective training strategy for multi-image interleaved reasoning

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](../../NeurIPS2025/multimodal_vlm/efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[CVPR 2026\] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](../../CVPR2026/multimodal_vlm/wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)

<!-- RELATED:END -->
