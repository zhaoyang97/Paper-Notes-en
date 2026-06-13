---
title: >-
  [Paper Note] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models
description: >-
  [CVPR 2026][Multimodal VLM][vision foundation model evaluation] This paper proposes AVA-Bench, the first systematic evaluation benchmark that decouples the capabilities of vision foundation models (VFMs) into 14 atomic v…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "vision foundation model evaluation"
  - "atomic visual ability"
  - "benchmark"
  - "VFM"
  - "multimodal assessment"
date: 2026-05-08
content_hash: 444349214442b1c6
---

# AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models

**Conference**: CVPR 2026
**arXiv**: [2506.09082](https://arxiv.org/abs/2506.09082)  
**Code**: [Project Page](https://zheda-mai.github.io/AVA-Bench/)  
**Area**: 3D Vision
**Keywords**: vision foundation model evaluation, atomic visual ability, benchmark, VFM, multimodal assessment

## TL;DR

This paper proposes AVA-Bench, the first systematic evaluation benchmark that decouples the capabilities of vision foundation models (VFMs) into 14 atomic visual abilities (AVAs). By aligning training-test distributions and isolating individual abilities during evaluation, AVA-Bench precisely identifies the strengths and weaknesses of VFMs, and reveals that a 0.5B small model can maintain VFM ranking consistency comparable to a 7B model.

## Background & Motivation

### 1. State of the Field
Vision foundation models (VFMs) such as DINOv2, CLIP, SAM, and SigLIP, pretrained on large-scale data, have become universal feature extraction backbones for diverse downstream visual tasks. The dominant evaluation paradigm combines VFMs with large language models (LLMs) and tests performance on VQA benchmarks.

### 2. Limitations of Prior Work
Existing evaluation protocols suffer from two critical blind spots:
- **Training-test distribution mismatch**: Instruction tuning data and VQA test data follow different distributions, causing erroneous predictions that may stem from data bias rather than actual visual deficiencies in the VFM.
- **Multi-ability entanglement**: VQA questions typically require multiple visual abilities simultaneously, making it impossible to determine whether a model fails due to deficiency across all abilities or a single critical one.

### 3. Root Cause
A principled evaluation methodology is needed that both isolates individual visual abilities for precise diagnosis and ensures training-test distribution consistency, thereby transforming VFM selection from "empirical guesswork" into "engineering-driven decision-making."

### 4. Paper Goals
- Construct an evaluation benchmark that precisely localizes VFM performance across fundamental visual abilities.
- Eliminate evaluation errors introduced by data mismatch and multi-ability entanglement.
- Provide actionable guidance for VFM selection in downstream tasks.

### 5. Starting Point
Inspired by compositional text-to-image benchmarks and VQA question analysis, the paper decomposes complex visual reasoning into 14 "atomic visual abilities" (AVAs), each tested and trained independently, with auxiliary cues such as bounding boxes used to isolate the target ability.

### 6. Core Idea

**Atomic Visual Ability (AVA) Decoupled Evaluation**: The paper defines 14 indivisible fundamental visual abilities, constructs distribution-aligned training/test sets for each, and evaluates VFMs one ability at a time through a LLaVA-style pipeline, producing a "capability fingerprint" for each VFM.

## Method

### Overall Architecture

AVA-Bench comprises three core components:

1. **14 Atomic Visual Ability (AVA) Definitions**: localization, counting, spatial reasoning, orientation recognition, absolute depth estimation, relative depth estimation, color recognition, texture recognition, object recognition, action recognition, emotion recognition, OCR, scene recognition, and fine-grained recognition.
2. **Dataset Construction**: 218K image-question pairs carefully curated from 26 diverse datasets, each pair targeting a single AVA exclusively.
3. **Evaluation Pipeline**: A LLaVA-style two-stage training procedure (connector pretraining + LoRA fine-tuning) with independent training and evaluation per AVA.

### Key Designs

#### Design 1: Atomic Visual Ability Isolation

- **Function**: Ensures each image-question pair tests exactly one AVA.
- **Mechanism**: Auxiliary information in the form of bounding boxes is provided to eliminate interference from other abilities. For example, in depth estimation tasks, bounding boxes of objects are supplied to prevent localization ability from being conflated.
- **Design Motivation**: Addresses multi-ability entanglement in conventional VQA evaluation, enabling clear error attribution. Experiments demonstrate a dramatic difference in spatial reasoning performance with and without bounding boxes—with boxes, all VFMs perform consistently and well; without them, the task degrades into a compound localization + spatial reasoning problem.

#### Design 2: Training-Test Distribution Alignment

- **Function**: Training and test sets for each AVA are split strictly at 80/20, ensuring identical object category and answer distributions.
- **Mechanism**: Distribution consistency across object categories and answer bins is maintained between training and test splits.
- **Design Motivation**: Eliminates evaluation bias caused by data mismatch, ensuring that performance differences genuinely reflect the perceptual capabilities of VFMs.

#### Design 3: Multi-Source Data Aggregation and Quality Control

- **Function**: Samples for each AVA are collected from multiple domain-diverse datasets (e.g., indoor scenes, remote sensing, wildlife), with balanced sample counts and answer distributions.
- **Mechanism**: Cross-domain aggregation improves generalization; fine-grained filtering rules (minimum bounding box area, single-instance constraints, count bin balancing, etc.) ensure data quality.
- **Design Motivation**: Prevents evaluation results from being dominated by single-dataset biases, ensuring robustness of the assessment.

### Loss & Training

- A LLaVA-style two-stage training procedure is adopted: Stage 1 pretrains the connector (VFM frozen, LLM frozen); Stage 2 fine-tunes the connector and LLM with LoRA (VFM remains frozen throughout).
- Each AVA is trained independently on approximately 6K–10K samples; LoRA is used to prevent overfitting.
- Key finding: A 0.5B LLM (Qwen2) can substitute a 7B LLM (Vicuna-1.5) for VFM ranking, reducing GPU cost by 8×.

## Key Experimental Results

### Main Results

**Table 1: Average Ranking of VFMs across 14 AVAs**

| VFM | Pretraining Paradigm | Avg. Rank | Strongest AVA | Weakest AVA |
|-----|---------------------|-----------|--------------|-------------|
| SigLIP-1/2 | Language-supervised (Sigmoid) | Best | Leads on multiple AVAs | — |
| AIMv2 | Multimodal autoregressive | 2nd best | Leads on multiple AVAs | — |
| InternVL-2.5 | Language-supervised | Above average | — | — |
| CLIP | Language-supervised (contrastive) | Average | — | — |
| RADIO | Multi-teacher distillation | Average | Consistently stable | — |
| DINOv2 | Self-supervised contrastive | Below average | Orientation, localization | OCR |
| SAM | Segmentation-supervised | Low | Color recognition | Multiple AVAs |
| MiDaS | Depth-supervised | Low | Depth-related AVAs | Multiple AVAs |

**Table 2: Ranking Consistency between 0.5B and 7B LLM Evaluators**

| Evaluation Config | LLM Scale | GPU Cost | VFM Ranking Consistency |
|------------------|-----------|----------|------------------------|
| Vicuna-1.5 7B | 7B | Baseline (1×) | Reference ranking |
| Qwen2 0.5B | 0.5B | ~0.125× (8× savings) | Highly consistent with 7B |

### Ablation Study

**Effect of Bounding Boxes on Spatial Reasoning**:
- With GT bounding boxes: all VFMs perform nearly perfectly and consistently on spatial reasoning.
- Without bounding boxes: performance diverges substantially, with rankings closely correlated with each model's localization ability (MiDaS and SAM degrade notably).
- Conclusion: Failures on compound tasks are often attributable to deficiency in a single critical AVA rather than a comprehensive lack of visual capability.

**Localization Performance Grouped by Object Size**:
- Large objects (0.3–0.5 normalized area): minimal performance differences across VFMs.
- Small objects: performance gaps increase sharply, with MiDaS and SAM falling notably behind.
- Conclusion: Aggregated metrics can obscure fine-grained performance differences.

### Key Findings

1. **Language supervision is critical for general visual ability**: SigLIP-1/2 and AIMv2 consistently achieve top average rankings, highlighting the central role of language supervision in enhancing general visual competence.
2. **SSL is competitive with language supervision on vision-centric tasks**: DINOv2 matches or outperforms language-supervised models on vision-centric AVAs such as localization, absolute depth estimation, and orientation recognition.
3. **OCR heavily depends on language alignment**: VFMs without language alignment perform significantly worse on OCR.
4. **Low- and mid-level AVAs are generally well-handled**: All VFMs perform well on texture, relative depth, and object recognition, suggesting that VQA failures typically stem from deficiencies in specific critical AVAs rather than pervasive visual incompetence.
5. **Every VFM has at least one specialty**: Even lower-ranked models (e.g., SAM excels at color recognition; DINOv2 excels at orientation recognition) exhibit outstanding performance on at least one individual ability.

## Highlights & Insights

- **Evaluation paradigm innovation**: This is the first work to systematically shift VFM evaluation from "overall VQA scores" to "atomic capability fingerprints," enabling precise diagnosis of VFM capabilities.
- **Practical engineering value**: Capability fingerprints directly guide VFM selection for specific downstream tasks, transforming "empirical guesswork" into "engineering-driven decision-making."
- **Efficiency breakthrough**: A 0.5B LLM can replace a 7B model for VFM ranking, substantially reducing evaluation costs and making large-scale comparative analysis practically feasible.
- **Partial validation of the Platonic Representation Hypothesis**: VFMs trained with different paradigms converge on low- and mid-level AVAs, but significant divergence persists on high-level AVAs.
- **Challenges for non-language-aligned VFMs**: The connector alignment process loses critical visual information (DINOv2's linear probing accuracy drops precipitously from 66.3% to 25.67%), revealing a fundamental challenge in cross-modal alignment.

## Limitations & Future Work

1. **AVA coverage**: The 14 AVAs do not exhaust all fundamental visual abilities; capabilities such as 3D geometric understanding, illumination estimation, and material recognition are not covered.
2. **Absence of combined-ability evaluation**: Only individual AVAs are evaluated; interaction effects and performance degradation patterns under multi-AVA combinations remain unexplored.
3. **Evaluation pipeline limitations**: The LLaVA-style pipeline may inherently disadvantage non-language-aligned VFMs, and the information loss introduced by the connector alignment process remains an open problem.
4. **Static image restriction**: All AVAs are based on static images; dynamic visual abilities such as video understanding and temporal reasoning are not assessed.
5. **Dataset scale and diversity**: Training sets for some AVAs contain only 6–8K samples, which may be insufficient to fully elicit the potential of certain VFMs.

## Related Work & Insights

- **MLLM evaluation** (MMBench, SEED-Bench, etc.): Focuses on end-to-end MLLM performance but cannot disentangle the respective contributions of the VFM and LLM; AVA-Bench addresses this by fixing the LLM and varying the VFM.
- **VFM comparison studies** (vision encoder probing): Some prior work evaluates VFMs via linear probing, but is limited to single tasks; AVA-Bench provides a comprehensive 14-dimensional profile.
- **Compositional T2I evaluation** (T2I-CompBench, DALL-Eval): Defines visual primitives on the generation side, inspiring the ability decomposition approach of AVAs.
- **Inspiration**: The ability decoupling methodology introduced here is generalizable to other domains—for example, analogous "atomic reasoning ability" decomposition benchmarks could be designed for evaluating LLM reasoning capabilities.

## Rating

⭐⭐⭐⭐ A systematic and experimentally rigorous benchmark paper. The definition of 14 AVAs and the dataset construction process are highly meticulous, and the finding that a 0.5B LLM can substitute a 7B model has strong practical value. However, the work lacks combined-ability evaluation and coverage of dynamic visual capabilities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)
- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)

</div>

<!-- RELATED:END -->
