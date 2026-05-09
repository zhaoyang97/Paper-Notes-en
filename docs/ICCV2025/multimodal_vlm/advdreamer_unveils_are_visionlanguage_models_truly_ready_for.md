---
title: >-
  [Paper Note] AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?
description: >-
  [ICCV 2025][Multimodal VLM][VLM robustness] This paper proposes AdvDreamer, a framework that generates physically reproducible adversarial 3D transformation (Adv-3DT) samples from single images via zero-shot monocular pose manipulation, a naturalness reward model, and an inverse semantic probability loss. The framework reveals that current VLMs—including GPT-4o—suffer performance drops of 50–80% under 3D variations, and establishes MM3DTBench, the first VQA benchmark for evaluating VLM robustness to 3D variations.
tags:
  - ICCV 2025
  - Multimodal VLM
  - VLM robustness
  - adversarial 3D transformations
  - monocular pose manipulation
  - naturalness reward model
  - visual grounding
date: 2026-05-08
content_hash: 6ba33c489a6adb20
---

# AdvDreamer Unveils: Are Vision-Language Models Truly Ready for Real-World 3D Variations?

**Conference**: ICCV 2025
**arXiv**: [2412.03002](https://arxiv.org/abs/2412.03002)
**Code**: Not publicly available (includes MM3DTBench benchmark)
**Area**: AI Safety / Multimodal VLM / 3D Robustness
**Keywords**: VLM robustness, adversarial 3D transformations, monocular pose manipulation, naturalness reward model, visual grounding

## TL;DR
This paper proposes AdvDreamer, a framework that generates physically reproducible adversarial 3D transformation (Adv-3DT) samples from single images via zero-shot monocular pose manipulation, a naturalness reward model, and an inverse semantic probability loss. The framework reveals that current VLMs—including GPT-4o—suffer performance drops of 50–80% under 3D variations, and establishes MM3DTBench, the first VQA benchmark for evaluating VLM robustness to 3D variations.

## Background & Motivation
While VLM robustness to 2D adversarial perturbations and style variations has been studied, 3D variations (object rotation, translation, and scaling)—the most prevalent changes in the real world—remain largely unexplored in a systematic manner. Existing 3D robustness evaluations rely on 3D assets or dense multi-view captures, making them inapplicable to single natural images.

## Core Problem
Three key challenges are identified: (1) how to accurately characterize 3D variations under limited priors (single-view only); (2) how to ensure the visual quality of worst-case samples so that performance degradation is attributable to 3D variations rather than image artifacts; and (3) how to make adversarial samples transferable across tasks and architectures.

## Method

### Overall Architecture
Single natural image → foreground-background segmentation (Grounded-SAM) → LRM-based 3D reconstruction → sampling 3D transformation parameters $\Theta$ → rendering transformed foreground → diffusion-model-based re-synthesis onto background → NRM evaluates naturalness + ISP loss evaluates attack strength → CMA-ES optimizes transformation distribution $p^*(\Theta)$

### Key Designs
1. **Zero-Shot Monocular Pose Manipulation (MPM)**: Grounded-SAM segments the foreground → TripoSR performs single-view 3D reconstruction → transformations $\Theta = (\alpha, \beta, \gamma, \Delta x, \Delta y, s)$ (rotation/translation/scaling) are applied → a diffusion model re-synthesizes the transformed foreground onto the original background. No multi-view captures or 3D assets are required.
2. **Naturalness Reward Model (NRM)**: DINOv2 extracts visual features → a dual-stream MLP predicts visual fidelity and physical plausibility scores. Training data: 120K samples automatically annotated by GPT-4o and verified by human annotators. This prevents adversarial optimization from converging to unnatural pseudo-optimal regions.
3. **Inverse Semantic Probability (ISP) Loss**: Operates solely in the image-text alignment space of the visual encoder, minimizing the matching probability of correct semantic attributes. This design is architecture-agnostic (uses only the visual encoder) and task-agnostic (operates in the fundamental alignment space), ensuring cross-VLM transferability.

### Loss & Training
- The distribution $p^*(\Theta)$ is optimized rather than a single point $\Theta$, using CMA-ES (a gradient-free black-box optimizer).
- Overall objective: $\max \mathbb{E}[\mathcal{L}_{\text{ISP}} + \mathcal{L}_{\text{Nat}}]$
- Convergence in 15 iterations; approximately 0.28 GPU-hours per sample (RTX 3090).

## Key Experimental Results

### Zero-Shot Classification (ImageNet, Adv-3DT under $p^*(\Theta)$)

| Model | Clean Accuracy | Adv-3DT Accuracy | Drop |
|-------|---------------|-----------------|------|
| OpenCLIP ViT-B/16 | 98.0% | 54.0% | −44% |
| OpenCLIP ViT-G/14 | 96.4% | 53.5% | −43% |
| BLIP-2 ViT-G/14 | 81.0% | 49.1% | −32% |

### Image Captioning (GPT-Score Drop)
- LLaVA-1.5: GPT-Score drops from 22.9 to 16.9 (−26%)
- GPT-4o: drops from 25.0 to 17.7 (−29%)

### VQA Accuracy
- GPT-4o: 75.4% → 50.7% (−25%)

### Physical-World Experiments

| Scenario | Natural Image Accuracy | Physically Reproduced Adv-3DT Accuracy |
|----------|----------------------|---------------------------------------|
| Zero-shot classification | 100% | 51.3% |
| VQA (Acc.@1) | 83.3% | 33.6% |

### Ablation Study
- NRM improves naturalness scores from 1.6 to 2.52, with only a marginal reduction in attack strength (accuracy: 48.6% → 54.0%).
- The ISP loss achieves a better efficiency-performance trade-off compared to alternatives such as $\text{MF}_{\text{it}}$.
- Adv-3DT samples exhibit strong cross-model transferability: samples optimized on OpenCLIP successfully attack BLIP, SigLIP, and others.
- On MM3DTBench, the majority of 13 evaluated VLMs achieve accuracy below 50%.

## Highlights & Insights
- **3D variations are a blind spot for VLMs**: Even GPT-4o suffers severe performance degradation under simple 3D rotations, indicating a critical 3D bias in current pretraining data.
- **Generative 3D priors replace explicit 3D representations**: LRM is used in place of 3D assets or NeRF, enabling zero-shot single-view 3D manipulation.
- **Naturalness constraints are essential**: Without NRM, optimization finds unnatural shortcuts, making it impossible to attribute performance drops to 3D variations per se.
- **Physical reproducibility**: Digital adversarial samples can be reproduced in the real world, confirming that the threat is genuine.

## Limitations & Future Work
- A gap in attack strength exists between digital and physical reproductions in physical-world experiments.
- Defense mechanisms for improving VLM 3D robustness are not explored.
- LRM reconstruction quality limits 3D manipulation precision for complex objects.
- Only rigid-body transformations are considered; non-rigid deformations are not addressed.

## Related Work & Insights
- **vs. ViewFool/GMVFool**: These methods require NeRF and dense multi-view captures; AdvDreamer requires only a single view and generative priors.
- **vs. Simulator-based methods (e.g., CARLA)**: These require 3D assets; AdvDreamer operates directly from natural images.
- **vs. $L_p$ adversarial attacks**: Pixel-level perturbations are not physically reproducible; AdvDreamer's 3D transformations can be replicated in the real world.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First framework for generating adversarial 3D transformations from single views, revealing a critical blind spot in VLMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers classification, captioning, and VQA across three tasks, plus physical experiments, a benchmark, and ablation studies—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Research-question-driven experimental design with a clear take-away for each finding.
- **Value**: ⭐⭐⭐⭐ The methodology for VLM robustness evaluation is highly referenceable, and the 3D bias findings are significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] STI-Bench: Are MLLMs Ready for Precise Spatial-Temporal World Understanding?](sti-bench_are_mllms_ready_for_precise_spatial-temporal_world_understanding.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](../../NeurIPS2025/multimodal_vlm/learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](../../NeurIPS2025/multimodal_vlm/are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](../../ICLR2026/multimodal_vlm/can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[ICCV 2025\] On Large Multimodal Models as Open-World Image Classifiers](on_large_multimodal_models_as_open-world_image_classifiers.md)

</div>

<!-- RELATED:END -->
