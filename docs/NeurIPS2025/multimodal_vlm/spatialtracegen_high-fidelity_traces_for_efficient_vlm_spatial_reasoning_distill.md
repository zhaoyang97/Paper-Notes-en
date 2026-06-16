---
title: >-
  [Paper Note] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation
description: >-
  [NeurIPS 2025 (Workshop on Efficient Reasoning)][Multimodal VLM][VLM] This paper proposes SpatialTraceGen, a framework that distills high-quality multi-step tool-use reasoning traces from large teacher models via automat…
tags:
  - "NeurIPS 2025 (Workshop on Efficient Reasoning)"
  - "Multimodal VLM"
  - "VLM"
  - "spatial reasoning"
  - "knowledge distillation"
  - "reasoning traces"
  - "data generation"
date: 2026-05-08
content_hash: 7987968e1cc6b205
---

# SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation

**Conference**: NeurIPS 2025 (Workshop on Efficient Reasoning)  
**arXiv**: [2511.00054](https://arxiv.org/abs/2511.00054)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: VLM, spatial reasoning, knowledge distillation, reasoning traces, data generation

## TL;DR

This paper proposes SpatialTraceGen, a framework that distills high-quality multi-step tool-use reasoning traces from large teacher models via automated verification, enabling efficient fine-tuning of small VLMs for spatial reasoning.

## Background & Motivation

Vision-language models (VLMs) have achieved strong performance across many domains, yet they continue to struggle with complex spatial reasoning tasks. Such tasks require models to decompose problems and employ tools strategically — for example, inferring relative positions, comparing sizes, or reasoning about spatial relationships between objects.

While large VLMs (e.g., GPT-4V) can handle these tasks effectively, their deployment is costly and inference latency is high. A natural solution is to transfer the reasoning capabilities of large models to smaller, more deployable ones through fine-tuning. However, this approach faces a critical bottleneck: **the lack of high-quality step-by-step reasoning data**. Existing reasoning datasets often contain incomplete steps or erroneous intermediate reasoning, causing fine-tuned smaller models to learn flawed reasoning patterns.

Manual annotation of high-quality reasoning traces is prohibitively expensive and difficult to scale. Automatically generating accurate and complete multi-step reasoning traces thus constitutes the core problem addressed in this paper.

## Method

### Overall Architecture

The core idea of SpatialTraceGen is to distill multi-hop, multi-tool reasoning traces from a large teacher model, verify each reasoning step using an automated verifier, and ultimately produce a high-fidelity training dataset. The full pipeline comprises three stages:

1. **Trace Generation**: The teacher model generates step-by-step reasoning processes — including tool invocations — for spatial reasoning problems.
2. **Trace Verification**: An automated verifier checks the logical correctness of each reasoning step and the consistency of tool call results.
3. **Dataset Construction**: Traces that pass verification are filtered and assembled into a training set for fine-tuning.

### Key Designs

The **Automated Verifier** is the central contribution of this work. Rather than validating only the final answer, the verifier operates at the level of individual reasoning steps:

- **Step-level verification**: Checks whether each reasoning step is logically self-consistent, whether tool call parameters are correct, and whether returned results are interpreted accurately.
- **Cross-step consistency**: Validates that information is propagated coherently across consecutive steps, preventing intermediate reasoning breaks.
- **Quality scoring**: Assigns a quality score to each trace to support downstream data filtering.

The verifier serves as an efficient substitute for human annotation. On the CLEVR-Humans benchmark, the verifier-guided generation process improves the average trace quality score by **17%** while reducing quality variance by more than **40%**.

Regarding the **multi-tool reasoning trace format**, each trace contains:
- Problem decomposition steps
- Visual tool invocations (e.g., object detection, attribute extraction)
- Spatial relationship computation tool calls
- Intermediate reasoning and final conclusions

### Loss & Training

The generated high-quality trace dataset supports two training paradigms:

1. **Supervised Fine-Tuning (SFT)**: Standard sequence-to-sequence fine-tuning directly on the trace data.
2. **Offline Reinforcement Learning (Offline RL)**: Sample-efficient offline RL training that exploits the structured nature and quality scores of the traces.

## Key Experimental Results

### Main Results

Experiments are conducted on the CLEVR-Humans benchmark, primarily comparing different data generation strategies and their impact on downstream fine-tuning performance:

| Data Generation Method | Avg. Quality Score | Quality Variance | Valid Trace Ratio |
|:---|:---:|:---:|:---:|
| Direct generation (no verification) | 0.68 | 0.152 | 61.2% |
| Final-answer verification | 0.73 | 0.124 | 72.5% |
| **SpatialTraceGen (step-level verification)** | **0.80** | **0.089** | **85.3%** |

| Fine-Tuning Strategy | CLEVR-Humans Acc | Reasoning Step Completeness | Tool Call Accuracy |
|:---|:---:|:---:|:---:|
| Base small model (no fine-tuning) | 42.1% | 35.7% | 48.3% |
| SFT (unverified data) | 58.4% | 62.1% | 67.8% |
| SFT (SpatialTraceGen data) | 71.3% | 78.5% | 82.1% |
| Offline RL (SpatialTraceGen data) | 73.8% | 81.2% | 84.6% |

### Ablation Study

| Ablation Setting | Quality Score Δ | Valid Trace Δ |
|:---|:---:|:---:|
| Remove step-level verification | -12% | -14.1% |
| Remove cross-step consistency check | -7% | -8.5% |
| Remove quality score filtering | -5% | -6.2% |

### Key Findings

1. Step-level verification improves trace quality more effectively than final-answer-only verification, since a correct final answer does not guarantee correct intermediate reasoning.
2. The reduction in quality variance (>40%) indicates that the generated data is more stable and consistent, which is critical for downstream fine-tuning.
3. Offline RL offers a modest advantage over standard SFT in exploiting trace data, particularly in reasoning step completeness.

## Highlights & Insights

- **Data quality > data quantity**: This work further corroborates the principle that a small amount of high-quality data is more valuable than a large amount of noisy data.
- **Automated verification as a substitute for human annotation**: The verifier design enables scalable generation of high-quality reasoning data at substantially reduced cost.
- **The importance of step-level granularity**: Verifying only the final answer misses a large proportion of intermediate reasoning errors; step-level verification is the key.

## Limitations & Future Work

1. Evaluation is currently limited to the synthetic CLEVR-Humans setting; validation on real-world scenarios is absent.
2. The verifier relies on rule-based matching, which may not cover all types of reasoning errors.
3. The reasoning capability ceiling of the teacher model imposes an upper bound on the quality of the distilled data.
4. There is no explicit control over trace diversity, which may lead to trace homogenization.

## Related Work & Insights

- This work is complementary to STaR (Self-Taught Reasoner): while STaR generates reasoning chains through self-improvement, SpatialTraceGen ensures quality through an external verifier.
- The framework is generalizable to other VLM tasks requiring multi-step reasoning, such as visual mathematical reasoning and chart understanding.
- The verifier design philosophy is applicable to domains such as code generation where step-level validation is similarly necessary.

## Rating

- **Novelty**: ★★★★☆ — The step-level automated verifier design is the key highlight.
- **Practicality**: ★★★☆☆ — Currently limited to synthetic settings.
- **Experimental Thoroughness**: ★★★☆☆ — The dataset and evaluation scenarios are relatively constrained.
- **Writing Quality**: ★★★★☆ — The framework is described clearly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models](sd-vlm_spatial_measuring_and_understanding_with_depth-encoded_vision-language_mo.md)
- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[NeurIPS 2025\] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM](hawaii_hierarchical_visual_knowledge_transfer_for_efficient_vision-language_mode.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](../../CVPR2026/multimodal_vlm/hificl_highfidelity_incontext_learning_for_multimo.md)
- [\[ICLR 2026\] GLYPH-SR: Can We Achieve Both High-Quality Image Super-Resolution and High-Fidelity Text Recovery via VLM-Guided Latent Diffusion Model?](../../ICLR2026/multimodal_vlm/glyph-sr_can_we_achieve_both_high-quality_image_super-resolution_and_high-fideli.md)

</div>

<!-- RELATED:END -->
