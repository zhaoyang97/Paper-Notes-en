---
title: >-
  [Paper Note] Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning
description: >-
  [ICCV 2025][Image Generation][Concept Unlearning] HUB introduces the first comprehensive benchmark for evaluating concept unlearning methods in text-to-image diffusion models, covering 33 target concepts across 6 evaluation dimensions (faithfulness, alignment, pinpoint-ness, multilingual robustness, adversarial robustness, and efficiency), with 16,000 prompts per concept. The benchmark reveals that no single method achieves superiority across all dimensions.
tags:
  - ICCV 2025
  - Image Generation
  - Concept Unlearning
  - Evaluation Benchmark
  - Text-to-Image Safety
  - Multi-Dimensional Evaluation
  - Diffusion Models
date: 2026-05-08
content_hash: e5cb31f6e1b12bec
---

# Holistic Unlearning Benchmark: A Multi-Faceted Evaluation for Text-to-Image Diffusion Model Unlearning

**Conference**: ICCV 2025
**arXiv**: [2410.05664](https://arxiv.org/abs/2410.05664)
**Code**: [GitHub](https://github.com/ml-postech/HUB)
**Area**: Diffusion Models / Concept Unlearning
**Keywords**: Concept Unlearning, Evaluation Benchmark, Text-to-Image Safety, Multi-Dimensional Evaluation, Diffusion Models

## TL;DR
HUB introduces the first comprehensive benchmark for evaluating concept unlearning methods in text-to-image diffusion models, covering 33 target concepts across 6 evaluation dimensions (faithfulness, alignment, pinpoint-ness, multilingual robustness, adversarial robustness, and efficiency), with 16,000 prompts per concept. The benchmark reveals that no single method achieves superiority across all dimensions.

## Background & Motivation

Large-scale web data used to train text-to-image diffusion models may contain violent, harmful content and copyright-protected intellectual property. Concept unlearning has emerged as a solution to remove specific target concepts from pretrained models.

**Limitations of existing evaluations**:

**Narrow evaluation scope**: Most prior work focuses solely on whether the target concept is removed and whether image quality is preserved, neglecting side effects.

**Insufficient prompt coverage**: Previous methods use at most tens to hundreds of prompts, which is inadequate for thoroughly testing unlearning effectiveness.

**Limited concept coverage**: Different methods evaluate on different concepts, lacking a unified basis for comparison.

**Missing critical dimensions**: Including pinpoint-ness (whether semantically related concepts are over-forgotten), multilingual robustness (whether non-English prompts can still trigger the concept), and practical efficiency.

Root Cause: The absence of a unified, comprehensive evaluation framework for fairly comparing different unlearning methods prevents researchers from obtaining a complete picture of a method's true capabilities.

Core Idea: **Construct a holistic evaluation benchmark covering 33 concepts, 6 dimensions, and 16,000 prompts per concept.**

## Method

### Overall Architecture

The HUB evaluation framework comprises:
- **Concept taxonomy**: 4 major categories (Celebrity ×10, Style ×10, IP ×10, NSFW ×3) = 33 concepts
- **Prompt generation**: An LLM-driven two-step pipeline (attribute extraction → prompt combination generation), yielding ~10,000 prompts per concept
- **Concept detection**: Dedicated classifiers (Q16 for NSFW, GIPHY for Celebrity) + a VLM-based general detection framework
- **Evaluation across 6 dimensions × 15 tasks**

### Key Designs

1. **Six-Dimensional Evaluation System**:

    - **Faithfulness**: Target concept removal ratio + general image quality (FID) + target image quality (Aesthetic Score)
    - **Alignment**: General alignment (PickScore, ImageReward) + selective alignment (QG/A framework detecting whether non-target entities are preserved)
    - **Pinpoint-ness**: Detects whether semantically proximate concepts are over-forgotten (tested using the top-100 WordNet words by CLIP score)
    - **Multilingual Robustness**: Prompts are translated into 5 languages (Spanish/French/German/Italian/Portuguese) to test whether non-English prompts still generate the target concept
    - **Adversarial Robustness**: Three attack methods — Ring-a-Bell, UDA, and UoC
    - **Efficiency**: Training time, GPU memory, and storage requirements
    - Design Motivation: Cover all critical scenarios an unlearning method may encounter in real-world deployment

2. **VLM-Based Concept Detection Framework**:

    - Function: Perform general concept detection using VLMs for concepts lacking dedicated classifiers (IP, Style)
    - Mechanism: A two-step detection process — (1) generate 3 reference images from the original model for in-context learning; (2) apply chain-of-thought reasoning on test images from the unlearned model to determine whether the concept is present
    - Design Motivation: Eliminate the need to train a dedicated detector for each concept, enabling concept-agnostic general detection

3. **LLM-Driven Prompt Generation**:

    - Function: Generate 10,000+ diverse prompts per concept
    - Mechanism: Key attributes are first extracted (e.g., attributes for NSFW-violent include "War," "Murder," etc.), then 1–3 attributes are randomly combined and fed to an LLM to generate diverse prompts
    - Design Motivation: Simple prompts (e.g., "a photo of Mickey Mouse") are insufficient to thoroughly test unlearning; diverse and realistic prompts are required

### Loss & Training

HUB is an evaluation framework and does not involve training. The 7 evaluated unlearning methods are:
- SLD (negative prompt guidance), AC (concept ablation fine-tuning), ESD (erased stable diffusion fine-tuning)
- UCE (unified concept editing via closed-form cross-attention update), SA (selective amnesia / continual learning)
- RECELER (adapter + masking), MACE (masking-guided unlearning)

## Key Experimental Results

### Main Results (Overall Average)

| Method | Target Ratio↓ | General FID↓ | General Alignment↑ | Pinpoint-ness↑ | Multilingual↓ | Adversarial↓ | Training Time (min) |
|--------|------|------|----------|------|------|------|------|
| Original | 0.649 | 13.20 | 0.172 | 0.592 | 0.489 | 0.491 | 0.0 |
| SLD | 0.228 | 16.57 | 0.077 | 0.502 | 0.101 | 0.196 | 0.0 |
| AC | 0.301 | 14.20 | 0.127 | 0.528 | 0.199 | 0.280 | 37.3 |
| ESD | **0.143** | 14.53 | -0.082 | 0.260 | **0.071** | **0.148** | 106.0 |
| UCE | 0.250 | 13.82 | **0.193** | **0.535** | 0.114 | 0.252 | **0.1** |
| SA | 0.173 | 32.57 | -0.322 | 0.131 | 0.079 | 0.166 | 29980.0 |
| RECELER | **0.086** | 15.19 | 0.006 | 0.316 | 0.030 | 0.107 | 100.0 |
| MACE | 0.148 | 15.30 | -0.345 | 0.306 | 0.111 | 0.125 | 140.3 |

### Ablation Study: Performance by Concept Category

| Category | Best Unlearning (Target Ratio) | Best Quality (FID) | Best Pinpoint-ness |
|------|---------|------|------|
| Celebrity | UCE (0.001) | MACE (12.98) | AC (0.429) |
| Style | RECELER (0.038) | MACE (13.09) | UCE (0.696) |
| IP | UCE (0.034) | AC (13.23) | AC (0.552) |
| NSFW | RECELER (0.272) | UCE (13.95) | UCE (0.571) |

### Key Findings

1. **No method dominates across all dimensions**:
    - RECELER achieves the strongest unlearning (lowest target ratio: 0.086) but suffers notable degradation in general image quality
    - UCE is the fastest (0.1 min) and achieves high pinpoint-ness, but unlearning is incomplete
    - SA incurs prohibitive training cost (29,980 min ≈ 20 days) and severely degrades image quality (FID 32.57)
    - ESD achieves the best adversarial robustness, but its general alignment score becomes negative

2. **Pinpoint-ness is a pervasive issue**: All methods exhibit substantially lower pinpoint-ness than the Original (0.592), indicating that semantically related concepts are over-forgotten to varying degrees.

3. **Multilingual prompts represent a critical vulnerability**: Even when English prompts are successfully unlearned, non-English prompts can still trigger the target concept.

4. **Style concepts are the hardest to unlearn**: Even the best-performing method (RECELER) retains a target ratio of 0.038 for Style, whereas Celebrity can be reduced to 0.001.

## Highlights & Insights

1. **Comprehensive and rigorous evaluation framework**: The 6-dimension, 15-task design covers all critical challenges an unlearning method may face in real-world deployment.
2. **Selective alignment via the QG/A framework** is a noteworthy design — it detects whether unlearning "spills over" to non-target entities.
3. **VLM-based concept detection** achieves concept-agnostic general detection and constitutes a methodological contribution in its own right.
4. **The scale of 16,000 prompts per concept** far exceeds prior work and is more effective at exposing method weaknesses.
5. **The "no silver bullet" conclusion** provides important guidance for the research community.

## Limitations & Future Work

- Evaluation is conducted on SD v1.5; newer models (SDXL, SD3.0) may exhibit different behaviors.
- VLM-based concept detection achieves ~83% accuracy, which may introduce detection noise.
- Object-category concepts are excluded (deemed too generic to be practically relevant), although certain scenarios do require forgetting specific objects.
- Efficiency evaluation covers only the training phase, without considering inference efficiency changes post-unlearning.
- Although 33 concepts is a substantial set, coverage remains limited and does not encompass all possible unlearning scenarios.

## Related Work & Insights

- Compared to UnlearnCanvas (Style only) and CPDM (IP/Celebrity/Style only), HUB represents a qualitative leap in comprehensiveness.
- The multilingual robustness dimension exposes a fundamental flaw in current unlearning methods — they apply only local "patches" in the embedding space.
- The pinpoint-ness dimension identifies an important future research direction: achieving "surgical" precise unlearning.

## Rating
- Novelty: ⭐⭐⭐⭐ As a benchmark paper, the 6-dimension evaluation framework is the primary contribution; no new unlearning method is proposed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 33 concepts × 7 methods × 6 dimensions × 15 tasks — an exceptionally large experimental scale.
- Writing Quality: ⭐⭐⭐⭐⭐ The framework is described clearly, tables are well-organized, and conclusions are explicit.
- Value: ⭐⭐⭐⭐⭐ Establishes the first comprehensive standard for concept unlearning, with significant value for advancing the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Meta-Unlearning on Diffusion Models: Preventing Relearning Unlearned Concepts](meta-unlearning_on_diffusion_models_preventing_relearning_unlearned_concepts.md)
- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](munba_machine_unlearning_via_nash_bargaining.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](../../ICLR2026/image_generation/continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[NeurIPS 2025\] OVERT: A Benchmark for Over-Refusal Evaluation on Text-to-Image Models](../../NeurIPS2025/image_generation/overt_a_benchmark_for_over-refusal_evaluation_on_text-to-image_models.md)
- [\[ICCV 2025\] Holistic Tokenizer for Autoregressive Image Generation](holistic_tokenizer_for_autoregressive_image_generation.md)

</div>

<!-- RELATED:END -->
