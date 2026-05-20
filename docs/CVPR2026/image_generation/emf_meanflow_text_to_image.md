---
title: >-
  [Paper Note] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation
description: >-
  [CVPR 2026][Image Generation][MeanFlow] This work is the first to extend the MeanFlow framework from class-label conditioning to text-conditioned image generation. It identifies the semantic discriminability and disentan…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "MeanFlow"
  - "One-Step Generation"
  - "Text-to-Image"
  - "Text Encoder"
  - "Semantic Discriminability"
date: 2026-05-08
content_hash: 4a23c3869d0cf75d
---

# Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation

**Conference**: CVPR 2026
**arXiv**: [2604.18168](https://arxiv.org/abs/2604.18168)  
**Code**: [https://github.com/AMAP-ML/EMF](https://github.com/AMAP-ML/EMF)  
**Area**: Image Generation
**Keywords**: MeanFlow, One-Step Generation, Text-to-Image, Text Encoder, Semantic Discriminability

## TL;DR

This work is the first to extend the MeanFlow framework from class-label conditioning to text-conditioned image generation. It identifies the semantic discriminability and disentanglement of text representations as the key bottlenecks under limited inference steps, and achieves high-quality few-step/one-step T2I generation based on the BLIP3o-NEXT text encoder.

## Background & Motivation

**Background**: MeanFlow is a theoretically grounded flow matching acceleration method that learns the mean velocity field between two time points to enable one-step generation, achieving results on par with standard multi-step models for ImageNet class-conditional generation. Subsequent works improving training strategies and architectures have also focused primarily on the class-conditional setting.

**Limitations of Prior Work**: Extending MeanFlow from discrete class labels to flexible text inputs appears straightforward but is in practice highly non-trivial. Naively plugging an LLM-based text encoder into the MeanFlow framework with standard training strategies yields disappointing results. The numerical stability of the JVP term has repeatedly been identified as the primary bottleneck for scaling consistency-type methods to large-scale T2I generation.

**Key Challenge**: Class labels are discrete and easily distinguishable conditioning signals, whereas text conditions are continuous and semantically complex. Under extremely few-step (e.g., one-step) inference, the model has almost no opportunity to correct semantic errors through iterative denoising, imposing stringent quality requirements on the conditioning signal.

**Goal**: (1) Understand why certain text encoders fail in the few-step setting; (2) Identify key properties that high-quality text representations must possess; (3) Leverage these findings to build the first effective text-conditioned MeanFlow generation model.

**Key Insight**: The authors compare different text encoders under limited inference steps, finding that the BLIP3o-NEXT encoder maintains basic semantic integrity even at one step, whereas the SANA-1.5 encoder exhibits severe semantic degradation in the few-step regime.

**Core Idea**: High-quality text representations require two essential properties — *discriminability* (distinguishing fine-grained semantic differences) and *disentanglement* (preserving the linguistic structure of text). Only encoders with both properties can construct reliable velocity field directions, enabling few-step or even one-step generation.

## Method

### Overall Architecture

The method adapts the pretrained BLIP3o-NEXT diffusion model into the MeanFlow framework. The key modification is the introduction of dual time-embedding layers: $\phi_{interval}(t-r)$ encodes the length of the time interval, and $\phi_{end}(t)$ encodes the current time point. The combined conditional embedding $\phi_{cond}(t,r) = \phi_{interval}(t-r) + \phi_{end}(t)$ jointly controls the velocity network together with the text features.

### Key Designs

1. **Analysis and Validation of Text Representation Discriminability**:

    - **Function**: Validates the cross-modal alignment quality of text encoders.
    - **Mechanism**: On the COCO 2017 training set of 118K samples, query prompts are encoded by the target text encoder to retrieve the most semantically similar image-text pairs; DINOv3 is then used to evaluate the visual feature similarity between the retrieved images and the query images. BLIP3o-NEXT achieves a score of 0.734, CLIP 0.730, Gemma 0.713, and T5 only 0.634.
    - **Design Motivation**: Discriminability implies that the text encoder's output is well aligned with the corresponding image representations, accurately distinguishing semantically similar but distinct texts. In few-step generation, each step's velocity field direction must be sufficiently accurate; encoders with poor discriminability produce ambiguous velocity fields.

2. **Analysis and Validation of Text Representation Disentanglement**:

    - **Function**: Evaluates the text encoder's ability to preserve linguistic structure.
    - **Mechanism**: On the full prompts from DPG-Bench, subsets are created by randomly removing portions of the text; the cosine distance between encodings of the reduced and full versions is then computed. A good encoder should yield small distances between the reduced and full versions (structural preservation). BLIP3o-NEXT scores 0.999, Gemma 0.987, CLIP 0.967, and T5 0.893.
    - **Design Motivation**: Disentanglement ensures that encoded text features retain the original linguistic structure, preventing disproportionate representational drift caused by text variations.

3. **MeanFlow T2I Adaptation**:

    - **Function**: Adapts a pretrained flow matching model for MeanFlow-based one-step/few-step generation.
    - **Mechanism**: The time-embedding layer is duplicated into an interval layer and an endpoint layer. Time step pairs $(t, r)$ are adaptively sampled from uniform or logit-normal distributions, with the proportion of $t \neq r$ samples gradually increasing during training. The standard MeanFlow objective is used: $\mathcal{L}_{MF}(\theta) = \mathbb{E}[\|u_\theta - \text{sg}(u_{tgt})\|^2]$, where the target is computed via JVP.
    - **Design Motivation**: Fine-tuning MeanFlow on top of a pretrained model is considerably easier than training from scratch, since the model already encodes a velocity field. The critical prerequisite, however, is that the text encoder must possess sufficient discriminability and disentanglement.

### Loss & Training

Approximately 170K samples are used (BLIP3o-60k + shareGPT-4o + Echo-4o), with a learning rate of 1e-5, batch size of 128, and training for 150 epochs. The model is fine-tuned from BLIP3o-NEXT.

## Key Experimental Results

### Main Results

| Model | Steps | GenEval↑ | DPG-Bench↑ | HPSv2↑ |
|-------|-------|---------|-----------|--------|
| BLIP3o-NEXT | 30 | 0.91 | 82.05 | 29.42 |
| BLIP3o-NEXT | 4 | 0.86 | 78.15 | 26.96 |
| BLIP3o-NEXT | 1 | 0.46 | 57.05 | 18.54 |
| **EMF (Ours)** | 4 | **0.90** | **81.20** | **29.25** |
| **EMF (Ours)** | 2 | 0.85 | 79.44 | 27.21 |
| **EMF (Ours)** | 1 | 0.74 | 77.36 | 25.77 |
| SANA-Sprint | 4 | 0.77 | - | - |
| rCM | 4 | 0.83 | - | - |

### Ablation Study

| Configuration | GenEval (1-step) | Notes |
|--------------|-----------------|-------|
| BLIP3o-NEXT encoder + MeanFlow | 0.74 | High discriminability + high disentanglement |
| SANA-1.5 encoder + MeanFlow | Fails | Insufficient discriminability |
| SANA-1.5 encoder + SFT fine-tuning + MeanFlow | Still fails | Fine-tuning cannot compensate for encoder deficiencies |

### Key Findings

- EMF with 4 steps nearly matches BLIP3o-NEXT at 30 steps (GenEval 0.90 vs. 0.91), achieving approximately 7.5× acceleration.
- EMF outperforms all distillation-based models (SDXL-Turbo/Lightning/DMD2, etc.) without requiring a teacher model.
- The SANA-1.5 encoder fails to function effectively within MeanFlow even after SFT fine-tuning, demonstrating that the intrinsic properties of the encoder — not the training data — are the bottleneck.
- EMF performance improves consistently with more inference steps (1→2→4→8), unlike conventional consistency models where performance saturates or even degrades as steps increase.

## Highlights & Insights

- The systematic analysis of text encoder *discriminability* and *disentanglement* is highly valuable. Prior work typically evaluates final generation quality only; this paper delves into the properties of the text representation space, providing concrete metrics for selecting or designing text encoders for few-step generation.
- The analysis of "why class labels work in MeanFlow but text does not" is particularly insightful: class labels are inherently discrete and easily distinguishable, naturally possessing high discriminability.
- The comparison with consistency-based methods is also illuminating: consistency methods may degrade with more steps, whereas MeanFlow — as a stable discretization of a continuous flow — continuously benefits from additional steps.

## Limitations & Future Work

- Validation is currently limited to BLIP3o-NEXT, an encoder that happens to exhibit both high discriminability and high disentanglement; generalizability to other encoders meeting these criteria remains uncertain.
- One-step GenEval of 0.74 still lags behind multi-step baselines, leaving room for improvement toward truly high-quality single-step T2I generation.
- The numerical stability issues with JVP computation are mitigated by selecting a suitable encoder but are not fundamentally resolved.
- Future directions include exploring text encoders specifically designed or trained for few-step generation.

## Related Work & Insights

- **vs. Original MeanFlow**: Supports only class-conditional generation; this work is the first extension to text conditioning.
- **vs. SANA-Sprint**: A distillation-based method achieving GenEval 0.77 at 4 steps; this work achieves 0.90, a significant improvement.
- **vs. Consistency Models**: Consistency models may degrade with increasing steps; the proposed MeanFlow approach yields consistent improvement.

## Rating

- Novelty: ⭐⭐⭐⭐ — First extension of MeanFlow to T2I; the text representation analysis is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive multi-benchmark comparisons; systematic encoder analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear logical progression from observations to analysis to methodology.
- Value: ⭐⭐⭐⭐ — Provides actionable guidance for encoder selection in few-step T2I generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)
- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](agentic_retoucher_for_texttoimage_generation.md)
- [\[CVPR 2026\] SOLACE: Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](solace_self_confidence_rewards_t2i.md)
- [\[CVPR 2026\] ChordEdit: One-Step Low-Energy Transport for Image Editing](chordedit_one-step_low-energy_transport_for_image_editing.md)

</div>

<!-- RELATED:END -->
