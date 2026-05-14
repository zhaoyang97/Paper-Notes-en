---
title: >-
  [Paper Note] Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models
description: >-
  [CVPR 2026][Segmentation][Occlusion robustness] This paper proposes OccSAM-Bench, a benchmark that systematically evaluates the occlusion robustness of SAM-family models in endoscopic scenes via synthetically generated s…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Occlusion robustness"
  - "SAM"
  - "endoscopy"
  - "benchmark"
date: 2026-05-08
content_hash: 40c4b75780a17e7a
---

# Seeing Through the Tool: A Controlled Benchmark for Occlusion Robustness in Foundation Segmentation Models

**Conference**: CVPR 2026
**arXiv**: [2604.11711](https://arxiv.org/abs/2604.11711)
**Code**: N/A
**Area**: Image Segmentation
**Keywords**: Occlusion robustness, SAM, endoscopy, benchmark, segmentation

## TL;DR

This paper proposes OccSAM-Bench, a benchmark that systematically evaluates the occlusion robustness of SAM-family models in endoscopic scenes via synthetically generated surgical instrument occlusions. A three-region evaluation protocol is introduced to reveal two distinct behavioral patterns under occlusion: occlusion-aware and occlusion-agnostic.

## Background & Motivation

**Background**: SAM and its successors (SAM 2, SAM 3, MedSAM, etc.) have demonstrated strong zero-shot generalization in medical image segmentation; however, existing evaluations are almost exclusively conducted on clean, curated medical images.

**Limitations of Prior Work**: In clinical endoscopy, target tissues are frequently occluded by surgical instruments, yet no benchmark systematically quantifies the robustness of foundation segmentation models under such occlusions. More critically, standard full-mask evaluation is fundamentally flawed in surgical contexts — a model that erroneously "hallucinates" tissue beneath an instrument may receive a high score simply by overlapping with the hidden ground truth.

**Key Challenge**: Full-mask (amodal) evaluation metrics cannot distinguish between two clinically distinct behaviors: correctly rejecting the occluder versus incorrectly predicting through it.

**Goal**: (1) Establish a controlled surgical occlusion generation framework; (2) propose an evaluation protocol capable of differentiating model behaviors; (3) systematically assess the occlusion robustness of SAM-family models.

**Key Insight**: Occlusions are synthesized under known ground-truth conditions, enabling precise computation of segmentation performance over visible, invisible, and complete regions separately.

**Core Idea**: A three-region evaluation protocol (visible, invisible, complete) is designed to replace conventional single full-mask evaluation, thereby revealing the true behavior of models under occlusion.

## Method

### Overall Architecture

OccSAM-Bench comprises three core components: (1) a controlled occlusion generation pipeline that synthesizes two occlusion types at three severity levels across three public polyp datasets; (2) a three-region evaluation protocol that decomposes segmentation performance into visible, invisible, and complete targets; and (3) a systematic zero-shot evaluation of seven SAM-family models.

### Key Designs

1. **Controlled Occlusion Generation**:

    - **Function**: Simulate two types of occlusion encountered in surgical scenes.
    - **Mechanism**: Surgical instrument pasting (real instrument masks sampled from the Kvasir-Instrument dataset are randomly scaled and rotated before being overlaid onto target images) and CutOut occlusion (rectangular masks placed within the target region to remove image content). Three severity levels are controlled via the occlusion ratio $r = |M_{full} \cap M_{occluder}| / |M_{full}|$: low (0–20%), medium (20–40%), and high (40–60%).
    - **Design Motivation**: Instrument pasting introduces visual confusion, while CutOut removes information without introducing extraneous content; the two methods complementarily isolate the effects of visual confusion and data absence.

2. **Three-Region Evaluation Protocol**:

    - **Function**: Decompose segmentation performance to reveal true model behavior under occlusion.
    - **Mechanism**: Defines $M_{vis} = M_{full} \setminus M_{occ}$ (visible mask) and $M_{inv} = M_{full} \cap M_{occ}$ (invisible mask), and evaluates each independently — the model's ability to segment visible tissue, its tendency to predict into occluded regions, and overall performance. Visible DSC penalizes false positives that extend predictions into the instrument region.
    - **Design Motivation**: Standard full-mask evaluation may reward clinically incorrect predictions (predicting tissue through an instrument); the three-region protocol directly aligns with surgical safety constraints.

3. **Model Behavior Classification**:

    - **Function**: Categorize SAM-family models into two behavioral archetypes.
    - **Mechanism**: Occlusion-aware models (SAM, SAM 2, SAM 3, MedSAM3) prioritize accurate segmentation of visible tissue and reject instrument regions; occlusion-agnostic models (MedSAM, MedSAM2) confidently predict into occluded regions, exhibiting amodal completion behavior.
    - **Design Motivation**: Model selection should be driven by clinical intent — conservative segmentation of visible tissue versus inference of hidden anatomical structures — rather than solely by performance on clean images.

### Loss & Training

This paper is a benchmarking study and does not involve model training. Evaluation employs DSC and 95% Hausdorff distance as metrics, supporting both bounding-box and point prompt modalities.

## Key Experimental Results

### Main Results

| Model | Type | Visible DSC (Tool-High) | Invisible DSC (Tool-High) | Full-mask DSC (Tool-High) |
|-------|------|------------------------|--------------------------|--------------------------|
| SAM 3 | Occlusion-aware | **0.72** | 0.15 | 0.58 |
| MedSAM3 | Occlusion-aware | 0.70 | 0.18 | 0.56 |
| MedSAM2 | Occlusion-agnostic | 0.65 | **0.52** | **0.62** |
| MedSAM | Occlusion-agnostic | 0.58 | 0.45 | 0.55 |
| SAM-Med2D | Neither | 0.42 | 0.22 | 0.35 |

### Ablation Study

| Occlusion Type | Severity | SAM 3 Visible DSC | MedSAM2 Visible DSC |
|----------------|----------|-------------------|---------------------|
| Tool | Low | 0.85 | 0.78 |
| Tool | Medium | 0.78 | 0.72 |
| Tool | High | 0.72 | 0.65 |
| CutOut | Low | 0.83 | 0.80 |
| CutOut | Medium | 0.76 | 0.74 |
| CutOut | High | 0.68 | 0.66 |

### Key Findings

- MedSAM2 is the only model to achieve high invisible DSC while maintaining competitive visible DSC, likely attributable to its video-based fine-tuning strategy.
- SAM-Med2D underperforms across all conditions and does not align with either behavioral pattern.
- Full-mask evaluation is indeed misleading: occlusion-aware and occlusion-agnostic models may attain similar full-mask scores despite exhibiting clinically distinct behaviors.

## Highlights & Insights

- The three-region evaluation protocol is a simple yet profound contribution: visible DSC serves as the primary robustness metric and directly penalizes clinically dangerous over-segmentation — a principle generalizable to any segmentation scenario involving occlusion.
- The direction of medical fine-tuning is found to determine occlusion behavior: general-purpose SAM fine-tuning yields occlusion-aware behavior, whereas treating medical images as video sequences (MedSAM2) induces an amodal completion tendency.

## Limitations & Future Work

- Synthetic occlusions cannot fully replicate the optical physics of real surgery (tissue deformation, specular reflections, etc.).
- Evaluation is limited to polyp segmentation and does not cover other anatomical structures.
- More complex prompting strategies — such as multi-click or positive/negative point pairs — remain unexplored.
- Extension to video settings for evaluating temporal occlusion robustness is a natural next step.

## Related Work & Insights

- **vs. SAMEO**: SAMEO targets amodal segmentation in natural images; this paper argues that directly transferring amodal evaluation to surgical settings is problematic, as instruments are unambiguous non-target occluders.
- **vs. Standard Medical Evaluation**: Existing evaluations such as SAMed are conducted exclusively on clean images, neglecting occlusion as a critical clinical challenge.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-region protocol and behavioral classification constitute a novel evaluation paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 7 models × 3 datasets × 2 occlusion types × 3 severity levels.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated and protocol description is rigorous.
- Value: ⭐⭐⭐⭐ Provides direct practical guidance for the clinical deployment of medical segmentation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mars-Bench: A Benchmark for Evaluating Foundation Models for Mars Science Tasks](../../NeurIPS2025/segmentation/mars-bench_a_benchmark_for_evaluating_foundation_models_for_mars_science_tasks.md)
- [\[CVPR 2026\] GKD: Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation](gkd_generalizable_knowledge_distillation_vfm.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[CVPR 2026\] Prompt-Driven Lightweight Foundation Model for Instance Segmentation-Based Fault Detection in Freight Trains](promptdriven_lightweight_foundation_model_for_inst.md)
- [\[CVPR 2026\] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)

</div>

<!-- RELATED:END -->
