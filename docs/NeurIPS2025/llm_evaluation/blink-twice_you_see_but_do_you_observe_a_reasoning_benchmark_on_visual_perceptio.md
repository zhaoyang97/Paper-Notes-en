---
title: >-
  [Paper Note] BLINK-Twice: You See But Do You Observe? A Reasoning Benchmark on Visual Perception
description: >-
  [NeurIPS 2025][LLM Evaluation][visual reasoning] This paper introduces BLINK-Twice, a vision-centric reasoning benchmark comprising 345 visually challenging images, 103 adversarial samples, 896 VQA pairs, and 1…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "visual reasoning"
  - "VLM evaluation"
  - "perception vs. reasoning"
  - "natural adversarial samples"
  - "reasoning chain evaluation"
  - "benchmark"
date: 2026-05-08
content_hash: 1566c74f4889d4b3
---

# BLINK-Twice: You See But Do You Observe? A Reasoning Benchmark on Visual Perception

**Conference**: NeurIPS 2025
**arXiv**: [2510.09361](https://arxiv.org/abs/2510.09361)
**Code**: [GitHub](https://github.com/PicoTrex/BLINK-Twice)
**Area**: LLM Evaluation
**Keywords**: visual reasoning, VLM evaluation, perception vs. reasoning, natural adversarial samples, reasoning chain evaluation, benchmark

## TL;DR
This paper introduces BLINK-Twice, a vision-centric reasoning benchmark comprising 345 visually challenging images, 103 adversarial samples, 896 VQA pairs, and 1,725 annotated reasoning steps. Through seven categories of visual illusion scenarios, it evaluates the "you see but do not observe" reasoning capability of MLLMs. The strongest model, Gemini-2.5 Pro, achieves only 26.9% G-Acc, suggesting that multi-round image observation and active visual interaction are promising directions for improvement.

## Background & Motivation
**Background**: MLLMs have advanced rapidly on VQA and image understanding tasks, while reasoning-augmented models (o1, DeepSeek-R1, QVQ) have introduced chain-of-thought (CoT) mechanisms.

**Limitations of Prior Work**: Existing reasoning benchmarks (MMMU, MathVista, OlympiadBench) are centered on textual knowledge and mathematical reasoning, where visual inputs serve merely as auxiliary context and can often be replaced by text descriptions. These benchmarks evaluate linguistic knowledge and logical reasoning rather than reasoning grounded in visual content.

**Key Challenge**: Models can "see" (perceive) but fail to "observe" (reason)—performance degrades significantly in scenarios requiring inference about true situations from visual details, such as optical illusions, forced perspective, and occlusion-induced misjudgments.

**Goal**: To design a reasoning benchmark where answers must be derived from visual image content rather than prior knowledge.

**Key Insight**: Naturally deceptive images such as visual illusions require models to first identify the misleading appearance and then reason from fine-grained details to the true situation, perfectly instantiating the distinction between "seeing" and "understanding."

**Core Idea**: Construct a visual reasoning evaluation spanning from "See" to "Observe" using seven categories of visually challenging images, natural adversarial samples, and annotated reasoning chains.

## Method

### Overall Architecture
The benchmark consists of three components: (1) seven categories of visually challenging images for evaluating visual reasoning ability; (2) natural adversarial image pairs that are visually similar but semantically opposite, forcing reliance on visual content; and (3) annotated reasoning steps for assessing the quality of the reasoning process beyond final answer correctness.

### Key Designs

1. **Seven Categories of Visual Challenges**:

    - Visual Misleading: coincidental colors or shapes cause misidentification (e.g., a pipe opening resembling a swan's head mistaken for a water splash)
    - Visual Dislocation: spatial coincidences between foreground and background (e.g., a person standing before a tree appearing to have an exaggerated "afro")
    - Art Illusion: flat paintings or landscapes simulating 3D effects
    - Visual Occlusion: partial occlusion causing misidentification of identity or structure
    - Forced Perspective: camera angle distorting apparent size relationships
    - Physical Illusion: visual distortion from physical phenomena such as refraction and reflection
    - Motion Illusion: motion artifacts captured in static images of high-speed movement
    - Design Motivation: each category corresponds to a distinct visual misperception mechanism, requiring models to identify the misleading appearance and reason toward the true situation.

2. **Natural Adversarial Sample Generation**:

    - Function: GPT-4o's image editing capability is used to generate adversarial images that are visually similar to originals but semantically opposite.
    - Mechanism: Images with "No" answers to a given question are edited by GPT-4o to yield "Yes" answers, followed by manual filtering to ensure quality.
    - Design Motivation: Since the same question yields opposite answers for the original and adversarial images, models are forced to genuinely analyze visual content rather than relying on common-sense guessing.
    - A total of 103 high-quality adversarial samples are retained.

3. **Reasoning Chain Annotation and Scoring**:

    - Function: A five-step reasoning chain is annotated (initial perception → identifying the misleading element → detailed visual cues → true situation → final answer).
    - CoT-Score: Evaluates the quality of the reasoning process based on two critical checkpoints—identification of "detailed visual cues" and "true situation."
    - Design Motivation: A correct final answer may result from guessing; only a correct reasoning process indicates genuine understanding.

### Evaluation Metrics
- Q-Acc: accuracy on individual questions
- I-Acc: accuracy requiring both questions on the same image to be correct (stricter)
- G-Acc: accuracy requiring all four questions in a group (including adversarial image) to be correct (strictest)
- CoT-Score: reasoning chain quality score $[0, 1]$

## Key Experimental Results

### Main Results (20 MLLMs Evaluated)

| Model | Q-Acc | I-Acc | G-Acc | CoT-Score |
|------|-------|-------|-------|-----------|
| **Gemini-2.5 Pro** ✩ | **0.667** | **0.470** | **0.269** | 0.584 |
| o1 ✩ | 0.608 | 0.392 | 0.186 | – |
| GPT-4o | 0.571 | 0.351 | 0.198 | **0.601** |
| QVQ-72B ✩ | 0.575 | 0.336 | 0.067 | 0.438 |
| Qwen-2.5-VL-32B ✩ | 0.578 | 0.353 | 0.158 | 0.328 |
| Qwen-2.5-VL-72B | 0.520 | 0.261 | 0.152 | 0.360 |
| InternVL2-8B | 0.478 | 0.194 | 0.083 | 0.194 |

### Ablation Study

| Analysis Dimension | Key Finding | Remarks |
|----------|---------|------|
| CoT Reasoning Augmentation | QVQ > Qwen2-VL-72B (+15%), Claude-3.7-Thinking > Claude-3.7 (+20%) | CoT substantially improves visual reasoning |
| Reasoning Chain Efficiency | QVQ averages 950+ tokens; ground-truth answers require <100 tokens | Substantial redundancy and self-contradiction in generated reasoning |
| Multi-Round Dialogue | Weaker models (Gemini-2.0-flash-thinking) show notable improvement | Repeated image observation improves perception |
| Strong Model Multi-Round | Limited gains for GPT-4o and QwenVL2.5-72B | Already possess stronger visual grounding |
| MM-Eureka | Negligible improvement | Mathematical reasoning augmentation does not transfer to visual reasoning |

### Key Findings
- **Best model G-Acc is only 27%**: Gemini-2.5 Pro leads overall but falls far short under the strictest metric.
- **CoT-Score often falls below answer accuracy**: Indicating that some correct answers result from guessing rather than genuine reasoning.
- **Reasoning models frequently over-reason**: QVQ's self-contradiction mechanism improves accuracy but generates substantial redundant tokens.
- **o3's active visual reasoning represents a new paradigm**: It performs multi-step visual reasoning through dynamic cropping and transformation of image regions, rather than relying solely on text-space reasoning after a single perception pass.

## Highlights & Insights
- **Empirical evidence for "seeing ≠ observing"**: Through carefully designed visual illusions and adversarial samples, the paper quantifies the substantial gap between perception and reasoning in MLLMs, offering methodological significance for MLLM capability evaluation.
- **Process-level reasoning evaluation**: By assessing annotated reasoning chains rather than only final answers, the benchmark reveals the distinction between lucky guessing and genuine understanding.
- **New directions for multimodal reasoning**: The results indicate that CoT in pure text space is insufficient; multi-round image observation and active visual interaction (e.g., o3's dynamic cropping) are needed, pointing toward design principles for next-generation MLLMs.

## Limitations & Future Work
- The dataset is relatively small (345 base images) and may not cover all visual reasoning scenarios.
- Adversarial samples are generated via GPT-4o editing, which may introduce editing artifacts that provide unintended cues to models.
- The seven challenge categories are biased toward optical illusion scenarios; everyday visual reasoning involving causal relationships and physical intuition is not covered.
- CoT-Score relies on GPT-4o for scoring, which may introduce evaluation bias.

## Related Work & Insights
- **vs. BLINK (Fu et al., 2024)**: BLINK focuses on basic perception ("See and Get"), whereas BLINK-Twice additionally requires reasoning ("See then Observe").
- **vs. NaturalBench (2024)**: NaturalBench uses CLIP to filter adversarial samples; this work employs GPT-4o editing for finer-grained controllability.
- **vs. MMMU/MathVista**: Visual inputs in those benchmarks can be replaced by text; this work ensures that answers must be derived from image content.
- **vs. MME-CoT**: MME-CoT evaluates CoT capability without a vision-centric focus; this work specifically targets visually grounded reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ The tripartite design of visual illusions + adversarial samples + reasoning chain evaluation, and the "See vs. Observe" framework, carry methodological value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation of 20 models (including 8 reasoning-augmented models), with multi-round dialogue and efficiency analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ The Sherlock Holmes epigraph is a fitting touch; the narrative is fluent and the visualizations are rich.
- Value: ⭐⭐⭐⭐ Exposes visual reasoning deficiencies in MLLMs and identifies active multimodal reasoning as a future development direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Can You Hear Me Now? A Benchmark for Long-Range Graph Propagation and Beyond](../../ICLR2026/llm_evaluation/can_you_hear_me_now_a_benchmark_for_long-range_graph_propagation_and_beyond.md)
- [\[ICCV 2025\] 3DSRBench: A Comprehensive 3D Spatial Reasoning Benchmark](../../ICCV2025/llm_evaluation/3dsrbench_a_comprehensive_3d_spatial_reasoning_benchmark.md)
- [\[NeurIPS 2025\] PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation](parrot_a_benchmark_for_evaluating_llms_in_cross-system_sql_translation.md)
- [\[NeurIPS 2025\] CLIMB: Class-Imbalanced Learning Benchmark on Tabular Data](climb_class-imbalanced_learning_benchmark_on_tabular_data.md)
- [\[NeurIPS 2025\] RGB-to-Polarization Estimation: A New Task and Benchmark Study](rgb-to-polarization_estimation_a_new_task_and_benchmark_study.md)

</div>

<!-- RELATED:END -->
