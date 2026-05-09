---
title: >-
  [Paper Note] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions
description: >-
  [NeurIPS 2025][Robotics][CLIP] This paper proposes the CLIP-IN framework, which leverages instruction editing datasets as hard negatives and incorporates long captions to enhance CLIP's fine-grained visual understanding. The approach achieves significant improvements on benchmarks such as MMVP without compromising zero-shot performance, and when integrated into MLLMs, it substantially reduces visual hallucinations.
tags:
  - NeurIPS 2025
  - Robotics
  - CLIP
  - fine-grained visual understanding
  - instruction editing data
  - hard negatives
  - long captions
date: 2026-05-08
content_hash: ed4581e6d020d459
---

# VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions

**Conference**: NeurIPS 2025
**arXiv**: [2508.02329](https://arxiv.org/abs/2508.02329)
**Code**: None
**Area**: Robotics
**Keywords**: CLIP, fine-grained visual understanding, instruction editing data, hard negatives, long captions

## TL;DR

This paper proposes the CLIP-IN framework, which leverages instruction editing datasets as hard negatives and incorporates long captions to enhance CLIP's fine-grained visual understanding. The approach achieves significant improvements on benchmarks such as MMVP without compromising zero-shot performance, and when integrated into MLLMs, it substantially reduces visual hallucinations.

## Background & Motivation

Vision-language models such as CLIP excel at coarse-grained image-text alignment but exhibit notable deficiencies in fine-grained visual understanding:

**Coarse-grained alignment**: CLIP's contrastive learning tends to capture high-level semantic alignment while neglecting subtle differences.

**Short-text limitation**: Standard CLIP relies on short textual descriptions, discarding rich semantic details.

**Lack of hard negatives**: Training data lacks image-text pairs that are highly similar yet semantically distinct.

The paper introduces two core innovations:
- Exploiting **image editing instruction datasets** as a natural source of hard negatives.
- Incorporating **long captions** and **rotary position encodings** to capture richer semantics.

## Method

### Overall Architecture

CLIP-IN comprises two core innovations:
1. Hard-negative contrastive learning based on instruction editing data.
2. Integration of long captions with rotary position encoding.

### Key Designs

1. **Instruction Editing Data as Hard Negatives**:

    - Existing image editing datasets (e.g., InstructPix2Pix) are utilized.
    - Pre- and post-edit image pairs naturally constitute hard negative samples.
    - Example: original image (cat on a red sofa) vs. edited image (cat on a blue sofa).
    - The paired editing instructions provide precise descriptions of semantic differences.

2. **Symmetric Hard-Negative Contrastive Loss**:

    - The model is trained not only to match correct image-text pairs but also to distinguish subtle editing differences.
    - A symmetric design applies hard-negative contrastive learning in both image→text and text→image directions:
    $\mathcal{L}_{\text{HN}} = -\log \frac{e^{s(I, T^+)}}{e^{s(I, T^+)} + \sum_k e^{s(I, T_k^-)}}$

3. **Long Captions + Rotary Position Encoding (RoPE)**:

    - Detailed long-form textual descriptions (typically 100–300 tokens) are introduced.
    - The standard CLIP text encoder is limited to 77 tokens.
    - RoPE is applied to extend context length while preserving positional awareness.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{CLIP}} + \alpha \mathcal{L}_{\text{HN}} + \beta \mathcal{L}_{\text{long}}$$

- $\mathcal{L}_{\text{CLIP}}$: Standard contrastive loss (preserves zero-shot capability).
- $\mathcal{L}_{\text{HN}}$: Hard-negative contrastive loss (improves fine-grained understanding).
- $\mathcal{L}_{\text{long}}$: Long-caption alignment loss.

## Key Experimental Results

### Main Results (Fine-Grained Visual Understanding)

| Method | MMVP ↑ | Winoground ↑ | ARO-Relation ↑ | SugarCrepe ↑ | IN-1K Zero-Shot ↑ |
|--------|--------|-------------|---------------|-------------|------------------|
| CLIP (ViT-L/14) | 28.5 | 35.2 | 62.8 | 75.3 | 75.5 |
| CLIP + NegCLIP | 32.1 | 38.5 | 68.5 | 78.2 | 74.8 |
| CLIP + SigLIP | 30.8 | 37.1 | 66.2 | 77.5 | 76.2 |
| CLIP + DAC | 35.2 | 40.8 | 70.5 | 80.1 | 74.5 |
| **CLIP-IN** | **42.8** | **46.5** | **75.8** | **84.5** | **75.8** |

### MLLM Integration Results

| Visual Encoder | LLaVA-1.5 MMVP ↑ | LLaVA Hallucination Rate ↓ | POPE Acc ↑ | MMBench ↑ |
|---------------|------------------|--------------------------|----------|----------|
| CLIP-ViT-L | 32.5 | 45.2 | 83.5 | 64.8 |
| SigLIP | 35.8 | 42.1 | 85.2 | 66.5 |
| **CLIP-IN** | **45.2** | **32.5** | **88.8** | **68.2** |

### Ablation Study

| Component | MMVP ↑ | Winoground ↑ | IN-1K ↑ |
|-----------|--------|-------------|--------|
| CLIP-IN (full) | 42.8 | 46.5 | 75.8 |
| w/o hard negatives | 33.5 | 39.2 | 76.1 |
| w/o long captions | 38.2 | 43.1 | 75.5 |
| w/o RoPE | 36.5 | 41.8 | 75.2 |
| Random negatives (non-instruction editing) | 35.8 | 40.5 | 75.5 |
| Short captions only | 37.2 | 42.5 | 76.0 |

### Key Findings

1. CLIP-IN achieves a 14.3-point improvement on MMVP (28.5 → 42.8), demonstrating the substantial benefit of the hard-negative strategy.
2. Instruction editing data as hard negatives substantially outperforms random negatives (+7.0 MMVP).
3. Notably, zero-shot ImageNet performance improves rather than degrades (75.5 → 75.8), indicating that fine-grained gains do not compromise general-purpose capability.
4. When integrated into MLLMs, the visual hallucination rate drops from 45.2% to 32.5%, demonstrating significant practical value.

## Highlights & Insights

- **Data source innovation**: Repurposing image editing data as a low-cost source of high-quality hard negatives is a resourceful contribution.
- **Preserved generality**: Simultaneously improving fine-grained understanding while maintaining zero-shot capability is a particularly noteworthy achievement.
- **Downstream value**: The marked reduction in MLLM hallucinations directly enhances practical application quality.
- **RoPE extension**: An elegant solution to CLIP's text length limitation.

## Limitations & Future Work

1. Instruction editing datasets primarily cover visual attribute editing, with insufficient coverage of abstract conceptual differences.
2. Long caption generation relies on external models, which may introduce noise.
3. Although RoPE extends context length, its effectiveness for very long texts (>1000 tokens) remains unverified.
4. Training cost is higher than standard CLIP fine-tuning.

## Related Work & Insights

- **NegCLIP**: A pioneer in enhancing CLIP understanding via negative sample augmentation.
- **InstructPix2Pix**: The image editing dataset used as a data source in this work.
- **SigLIP**: Google's improved contrastive learning approach.
- **DAC**: Caption-augmented contrastive learning.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Practical Value | 5 |
| Overall Recommendation | 4.5 |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[NeurIPS 2025\] FALCON: Fine-grained Activation Manipulation by Contrastive Orthogonal Unalignment for Large Language Model](falcon_fine-grained_activation_manipulation_by_contrastive_orthogonal_unalignmen.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](../../ICLR2026/robotics/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[NeurIPS 2025\] RDD: Retrieval-Based Demonstration Decomposer for Planner Alignment in Long-Horizon Tasks](rdd_retrieval-based_demonstration_decomposer_for_planner_alignment_in_long-horiz.md)
- [\[NeurIPS 2025\] RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation](robocerebra_a_large-scale_benchmark_for_long-horizon_robotic_manipulation_evalua.md)

<!-- RELATED:END -->
