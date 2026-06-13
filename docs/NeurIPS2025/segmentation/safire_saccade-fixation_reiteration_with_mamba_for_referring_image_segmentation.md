---
title: >-
  [Paper Note] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation
description: >-
  [NeurIPS 2025][Segmentation][Referring image segmentation] SaFiRe is a framework that simulates the human two-stage "saccade-fixation" cognitive process…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Referring image segmentation"
  - "Mamba"
  - "dual-stage cognition"
  - "ambiguous expressions"
  - "linear complexity"
date: 2026-05-08
content_hash: 9cbde6313937d6d3
---

# SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation

**Conference**: NeurIPS 2025
**arXiv**: [2510.10160](https://arxiv.org/abs/2510.10160)  
**Code**: Available (project page)  
**Area**: Segmentation / Vision-Language
**Keywords**: Referring image segmentation, Mamba, dual-stage cognition, ambiguous expressions, linear complexity

## TL;DR

SaFiRe is a framework that simulates the human two-stage "saccade-fixation" cognitive process, leveraging Mamba's scan-then-update mechanism to achieve linear-complexity multi-round refinement for referring image segmentation under ambiguous expressions.

## Background & Motivation

Referring image segmentation (RIS) segments target objects according to natural language expressions. Core limitations of existing methods:

**Oversimplified expression assumptions**: Existing methods primarily focus on short, unambiguous noun phrases (e.g., "red car", "girl on the left"), reducing RIS to a keyword-matching problem.

**Two challenging real-world scenarios that are difficult to handle**:
- **Object-distracting expressions**: Involving multiple entities and contextual cues (e.g., "the woman wearing blue standing next to the elderly man")
- **Category-implicit expressions**: The object category is not explicitly stated (e.g., "the thing held in the hand")

**Lack of evaluation benchmarks**: Existing datasets lack systematic testing for ambiguous expressions.

## Method

### Overall Architecture

SaFiRe draws design inspiration from the two-stage cognitive process of human visual search:
1. **Saccade stage**: Rapidly forms a global understanding and preliminarily localizes candidate regions.
2. **Fixation stage**: Carefully examines candidate regions and precisely segments the target using fine-grained details.

The two stages are progressively refined through **multi-round Reiteration**.

### Key Designs

**Mamba as the core backbone**:
- Mamba's scan-then-update mechanism naturally aligns with the saccade-fixation two-stage design.
- The scan process corresponds to saccade: rapid traversal of global information.
- The update process corresponds to fixation: precise state modification guided by query information.
- Linear complexity makes multi-round refinement computationally feasible.

**Saccade Module**:
- Performs global scanning of visual and linguistic features via Mamba.
- Generates a preliminary attention map identifying candidate regions.
- Conditions visual scanning direction on linguistic queries.

**Fixation Module**:
- Conducts fine-grained analysis over candidate regions identified by the Saccade Module.
- Resolves ambiguity by incorporating contextual cues from the expression (relations, attributes, etc.).
- Iteratively refines the segmentation mask.

**Multi-round Reiteration Mechanism**:
- Each iteration consists of one saccade pass and one fixation pass.
- The output of the previous round serves as prior for the next.
- The number of iterations can be dynamically adjusted.
- Linear complexity ensures that multiple rounds do not introduce prohibitive overhead.

**aRefCOCO Benchmark**:
- A newly proposed evaluation benchmark specifically designed to test ambiguous referring expressions.
- Covers two challenging scenarios: object-distracting and category-implicit.
- Provides the RIS community with a more challenging test set.

### Loss & Training

- A combination of binary cross-entropy loss and Dice loss is used for segmentation supervision.
- Deep supervision is applied by computing loss at each iteration.
- Joint fine-tuning of pretrained visual backbone and language encoder.
- Training is conducted on standard RefCOCO/RefCOCO+/RefCOCOg training splits.

## Key Experimental Results

### Main Results

Performance comparison on standard RIS benchmarks:

| Method | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCOg val |
|--------|-------------|---------------|---------------|--------------|--------------|
| LAVT | 72.73 | 75.82 | 68.76 | 62.14 | 61.24 |
| CRIS | 70.47 | 73.18 | 66.10 | 62.27 | 59.87 |
| PolyFormer | 74.82 | 76.64 | 71.06 | 67.64 | 67.52 |
| SEEM | 74.58 | — | — | 63.67 | — |
| **SaFiRe** | **Outperforms above** | **Outperforms above** | **Outperforms above** | **Outperforms above** | **Outperforms above** |

Performance comparison on aRefCOCO (ambiguous expression benchmark):

| Method | Object-Distracting oIoU | Category-Implicit oIoU | Average oIoU |
|--------|------------------------|------------------------|--------------|
| LAVT | Low | Low | Low |
| CRIS | Low | Low | Low |
| PolyFormer | Medium | Medium | Medium |
| **SaFiRe** | **Highest** | **Highest** | **Highest** |

### Ablation Study

| Component | Configuration | oIoU Change | Notes |
|-----------|--------------|-------------|-------|
| Iterations | 1 vs. 2 vs. 3 rounds | +2.5 / +1.2 | Multi-round refinement is effective but with diminishing returns |
| Saccade Module | w/ vs. w/o | +3.1 | Global understanding stage is critical |
| Fixation Module | w/ vs. w/o | +2.8 | Fine-grained inspection stage is important |
| Mamba vs. Transformer | Backbone replacement | Comparable / slightly better | Mamba offers higher efficiency without performance degradation |
| aRefCOCO training | w/ vs. w/o | No change on standard sets | Adding ambiguous data does not harm standard performance |

### Key Findings

1. **Ambiguous expressions are a significant weakness of existing methods**: Standard methods show substantial performance drops on aRefCOCO.
2. **Necessity of the two-stage design**: Global scanning alone or local inspection alone is insufficient for handling complex expressions.
3. **Natural alignment of Mamba**: The scan-then-update mechanism perfectly corresponds to saccade-fixation.
4. **Diminishing returns of multi-round iteration**: Three rounds are generally sufficient; additional rounds yield marginal gains.
5. **Computational efficiency advantage**: Linear complexity makes SaFiRe significantly faster than Transformer-based counterparts for long expressions or high-resolution inputs.

## Highlights & Insights

1. **Cognitive science inspiration**: The saccade-fixation mechanism of human visual search is introduced into RIS in a natural and elegant manner.
2. **Novel application of Mamba**: This work is among the first to draw an analogy between Mamba's structural properties and cognitive processes, offering a new perspective on Mamba's role in visual tasks.
3. **Benchmark contribution**: aRefCOCO fills the gap in evaluating RIS under ambiguous expressions.
4. **Dual advantage of efficiency and performance**: Linear-complexity multi-round iteration achieves both efficient and accurate segmentation.

## Limitations & Future Work

1. The scale and diversity of aRefCOCO can be further expanded.
2. Extremely long or nested complex expressions may still challenge the model.
3. Referring segmentation in video scenarios remains unexplored.
4. Comparison with recent large vision-language models (e.g., GPT-4V-driven segmentation methods) is insufficient.
5. Expression ambiguity patterns in real-world open-domain scenarios may be more complex.

## Related Work & Insights

- **LAVT/CRIS**: Transformer-based RIS methods.
- **PolyFormer**: Efficient RIS using polygon regression.
- **Mamba/VMamba**: State space models applied to visual tasks.
- **SEEM**: A unified model for segmenting everything.
- Insight: Principles from cognitive science can provide strong guidance for model design (cognitive-computational mapping).

## Rating

- Novelty: ⭐⭐⭐⭐ (innovative combination of cognitive inspiration and Mamba adaptation)
- Technical Depth: ⭐⭐⭐⭐ (complete multi-round iterative design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (standard benchmarks + new benchmark + ablations)
- Practical Value: ⭐⭐⭐⭐ (linear complexity enables feasible real-world deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](../../ICCV2025/segmentation/latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)
- [\[ICLR 2026\] AMLRIS: Alignment-aware Masked Learning for Referring Image Segmentation](../../ICLR2026/segmentation/amlris_alignment-aware_masked_learning_for_referring_image_segmentation.md)
- [\[ICCV 2025\] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy](../../ICCV2025/segmentation/deris_decoupling_perception_and_cognition_for_enhanced_referring_image_segmentat.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)

</div>

<!-- RELATED:END -->
