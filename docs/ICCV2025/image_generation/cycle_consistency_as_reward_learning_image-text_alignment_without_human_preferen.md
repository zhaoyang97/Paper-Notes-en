---
title: >-
  [Paper Note] Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences
description: >-
  [ICCV 2025][Image Generation][cycle consistency] Cycle consistency (reconstruction similarity via image→text→image or text→image→text) is employed as a supervision signal in lieu of human preferences to construct the 866K preference dataset CyclePrefDB. The resulting CycleReward model surpasses all existing methods on detailed caption evaluation and can improve both VLMs and diffusion models via DPO.
tags:
  - ICCV 2025
  - Image Generation
  - cycle consistency
  - reward model
  - image-text alignment
  - preference learning
  - DPO
date: 2026-05-08
content_hash: 2c97a06a9a119a77
---

# Cycle Consistency as Reward: Learning Image-Text Alignment without Human Preferences

**Conference**: ICCV 2025
**arXiv**: [2506.02095](https://arxiv.org/abs/2506.02095)
**Code**: [https://cyclereward.github.io/](https://cyclereward.github.io/)
**Area**: Image Generation / Vision-Language Alignment
**Keywords**: cycle consistency, reward model, image-text alignment, preference learning, DPO

## TL;DR

Cycle consistency (reconstruction similarity via image→text→image or text→image→text) is employed as a supervision signal in lieu of human preferences to construct the 866K preference dataset CyclePrefDB. The resulting CycleReward model surpasses all existing methods on detailed caption evaluation and can improve both VLMs and diffusion models via DPO.

## Background & Motivation

Image-text alignment metric $d(x,y)$ is a central problem in multimodal learning, widely used for evaluating VLM/T2I models and improving alignment via RLHF. However, existing approaches face critical bottlenecks:

**Human preference data is expensive and difficult to scale**: ImageReward, HPSv2, PickScore, etc., rely on large-scale human annotation.

**AI feedback (e.g., GPT-4V) is costly and restricted**: closed-source, rate-limited, and long-term availability is not guaranteed.

**Existing metrics are insufficient for evaluating long detailed captions**: most preference datasets contain short text (~20–35 tokens), making them unsuitable for assessing detailed descriptions.

**Direct computation of cycle consistency is feasible but inefficient and non-differentiable**: it requires running full T2I/I2T models.

Core insight: **comparing the reconstructed image with the original after mapping text back to image space** is substantially easier than directly comparing text and image. More accurate captions yield reconstructions closer to the original.

## Method

### Overall Architecture

Given an image-to-text mapping $F: X \to Y$ and a text-to-image mapping $G: Y \to X$, cycle consistency scores are defined as:
- Image-to-text: $s(x \to F(x)) := d_{\text{img}}(x, G(F(x)))$, computed via DreamSim for image similarity.
- Text-to-image: $s(y \to G(y)) := d_{\text{text}}(y, F(G(y)))$, computed via SBERT for text similarity.

Cycle consistency scores are converted into preference pairs: if $s(x \to y_i) > s(x \to y_j)$, then $y_i \succ y_j$.

### Key Designs

1. **CyclePrefDB Preference Dataset Construction**:

    - The DCI dataset (7.6K high-resolution images with dense captions) is used as input.
    - **Image-to-text**: 11 I2T models (from BLIP2 to InternVL2-40B) generate multiple candidate captions per image; SD3 is fixed as the inverse mapping to compute $s(x \to y)$. Older models are deliberately included to produce short/hallucinated captions as negative examples.
    - **Text-to-image**: 4 T2I models (from SD1.5 to FLUX) each generate images with 3 random seeds; LLaVA-1.5-13B is fixed as the inverse mapping to compute $s(y \to x)$.
    - Text is constrained to 77 tokens (T2I model prompt length limit).
    - The final dataset contains 866K preference pairs (398K I2T + 468K T2I).

2. **CycleReward Reward Model Training**:

    - Backbone: BLIP (ViT-L/16 encoder + BERTbase text encoder + 5-layer MLP), totaling 477M parameters.
    - Three variants: CycleReward-I2T / T2I / Combo (jointly trained).
    - I2T loss: $\mathcal{L}_{\text{img}} = -\mathbb{E}[\log \sigma(r_\theta(x,y_i) - r_\theta(x,y_j))]$
    - T2I loss: $\mathcal{L}_{\text{text}} = -\mathbb{E}[\log \sigma(r_\theta(x_i,y) - r_\theta(x_j,y))]$
    - Combo joint loss: $\mathcal{L} = \mathcal{L}_{\text{text}} + \lambda \mathcal{L}_{\text{img}}$ ($\lambda=1$)

3. **DPO Application**:

    - I2T direction: Qwen-VL-Chat is fine-tuned via DPO using CyclePrefDB-I2T.
    - T2I direction: Stable Diffusion 1.5 is trained via Diffusion DPO using CyclePrefDB-T2I.
    - Multiple downstream tasks are improved without any human annotation.

### Loss & Training

- The reward model is trained using the standard Bradley-Terry preference learning loss.
- DPO training directly optimizes the model on preference data without explicit reward modeling.
- Key design decisions: DreamSim (modeling human visual similarity) and SBERT are used to compute cycle consistency scores; ablation studies confirm their superiority over LPIPS, CLIP, and BERTScore.

## Key Experimental Results

### Main Results

Image-text alignment metric evaluation (Pairwise Accuracy %):

| Method | DetailCaps-4870 | GenAI-Bench | Supervision |
|--------|----------------|-------------|-------------|
| CLIPScore | 51.66 | 49.73 | None (pretrained) |
| VQAScore (11B) | 50.24 | **64.13** | None (pretrained) |
| HPSv2 | 54.34 | 56.13 | Human preference |
| PickScore | 51.01 | 57.05 | Human preference |
| ImageReward | 50.70 | 56.70 | Human preference |
| Raw Cycle Consistency | 56.46 | 52.52 | Cycle consistency |
| **CycleReward-Combo** | **60.50** | 55.52 | **Cycle consistency** |

CycleReward surpasses all methods on detailed caption evaluation (including models trained on human preferences), outperforming VQAScore (11B) by 10.26%, while using only 477M parameters (24× smaller).

DPO results (I2T direction, Qwen-VL-Chat):

| Model | DeCapBench | LLaVA-WD | MMHalBench | MMEP |
|-------|-----------|----------|------------|------|
| Baseline | 26.47 | 61.67 | 2.99 | 1460.2 |
| DPO w/ VLFeedback | 28.03 | 69.17 | 3.32 | 1551.5 |
| **DPO w/ CyclePrefDB-I2T** | **30.63** | **70.00** | 3.11 | 1485.7 |

### Ablation Study

Agreement rate between cycle consistency and human preferences (Agreement Rate %):

| Method | RLHF-V | POVID | HPDv2 | PaPv2 | IRDB | Mean |
|--------|--------|-------|-------|-------|------|------|
| GPT-4o | 61.3 | 60.0 | 48.1 | 45.8 | 24.8 | 48.0 |
| Raw Cycle | 58.6 | 61.2 | 60.5 | 59.8 | 54.5 | 58.9 |
| **CycleReward-Combo** | **66.5** | 63.8 | **67.7** | **65.8** | **61.3** | **65.0** |

Decoder ablation: replacing LLaVA-1.5-13B with the stronger InternVL2-26B as the I2T decoder improves the DetailCaps score from 51.74 to 57.21.

### Key Findings

- The cycle consistency signal is more stable than GPT-4o annotations in both I2T and T2I directions (GPT-4o achieves as low as 24.8% agreement on T2I evaluation).
- Training a reward model outperforms directly using raw cycle consistency scores, validating the effectiveness of distillation.
- DPO fine-tuning with CyclePrefDB-I2T improves not only captioning but also perception, reasoning, and hallucination reduction—despite the data containing only captioning instructions.
- CycleReward yields the largest improvement on detailed description tasks in best-of-N sampling and also benefits T2I generation with long-text prompts.
- DreamSim (modeling human visual similarity) and SBERT are the optimal similarity metric choices.

## Highlights & Insights

- **Core idea is concise and elegant**: cycle consistency provides a self-supervised alignment signal requiring no human annotation and is conceptually straightforward.
- **Strong cross-task generalization**: a single signal serves both I2T and T2I directions, and DPO training generalizes to diverse VL tasks beyond captioning.
- **High practical value**: CycleReward is fast (no T2I model required at inference), differentiable, and lightweight at 477M parameters.
- **Data efficiency**: CyclePrefDB is smaller in scale than VLFeedback/Pick-a-Pic yet achieves comparable or superior performance.
- **Comprehensive ablations**: similarity metrics, decoder choice, data scale, and filtering strategies are all analyzed in detail.

## Limitations & Future Work

- Supervision quality depends on the reconstruction fidelity of pretrained decoders; generation errors may introduce misleading preferences.
- Text length is constrained by the 77-token limit of T2I models, precluding evaluation of genuinely long captions.
- VQAScore (11B) still outperforms CycleReward on T2I evaluation—signal quality from cycle consistency in the text-to-image direction warrants further improvement.
- Cycle consistency for other modality pairs (e.g., video-language or audio-text) remains unexplored.
- Whether multi-step cycles (image→text→image→text→...) can provide stronger signals is an open question.

## Related Work & Insights

- Cycle consistency has a long history in unpaired data learning (e.g., CycleGAN); this work extends the concept to preference learning for cross-modal alignment.
- Image2Text2Image directly uses cycle consistency as an evaluation metric; this paper further distills it into a learnable reward model, achieving gains in both speed and performance.
- This approach is complementary to RLHF/DPO pipelines, offering a preference data construction method that depends neither on human nor strong AI annotation.
- A key insight for DPO dataset design: including diverse-quality model outputs is more important than using only the strongest model.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The cycle consistency concept is not new (Image2Text2Image precedes it), but distilling it into a reward model and constructing a large-scale preference dataset constitutes a significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three evaluation dimensions (metric evaluation, BoN sampling, DPO), both I2T and T2I directions, and highly comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logic, compelling motivation, intuitive figures, and rigorous experimental design.
- **Value**: ⭐⭐⭐⭐⭐ Provides a cheap and scalable alignment signal with significant implications for the RLHF/preference learning community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Enhancing Reward Models for High-quality Image Generation: Beyond Text-Image Alignment](enhancing_reward_models_for_high-quality_image_generation_beyond_text-image_alig.md)
- [\[ICCV 2025\] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering](ouroboros_single-step_diffusion_models_for_cycle-consistent_forward_and_inverse_.md)
- [\[ICCV 2025\] ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation](adiee_automatic_dataset_creation_and_scorer_for_instruction_guided_image_editing_evaluation.md)
- [\[ICCV 2025\] Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention](fair_generation_without_unfair_distortions_debiasing_text-to-image_generation_wi.md)
- [\[ICCV 2025\] CompleteMe: Reference-based Human Image Completion](completeme_reference-based_human_image_completion.md)

<!-- RELATED:END -->
