---
title: >-
  [Paper Note] Rethinking VLMs for Image Forgery Detection and Localization
description: >-
  [CVPR 2026][Multimodal VLM][Image forgery detection] This paper proposes IFDL-VLM, a framework that identifies an inherent *semantic plausibility bias* in VLMs — their tendency to favor semantic coherence over authenticity — which impedes forgery detection performance. The framework decouples detection/localization from language explanation into a two-stage optimization pipeline, and leverages localization masks as auxiliary inputs to VLMs to enhance interpretability, achieving comprehensive SOTA results across 9 benchmarks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Image forgery detection
  - vision-language models
  - forgery localization
  - interpretability
  - AIGC security
date: 2026-05-08
content_hash: df2b4acefa92be8e
---

# Rethinking VLMs for Image Forgery Detection and Localization

**Conference**: CVPR 2026
**arXiv**: [2603.12930](https://arxiv.org/abs/2603.12930)
**Code**: [sha0fengGuo/IFDL-VLM](https://github.com/sha0fengGuo/IFDL-VLM)
**Area**: Multimodal VLM
**Keywords**: Image forgery detection, vision-language models, forgery localization, interpretability, AIGC security

## TL;DR

This paper proposes IFDL-VLM, a framework that identifies an inherent *semantic plausibility bias* in VLMs — their tendency to favor semantic coherence over authenticity — which impedes forgery detection performance. The framework decouples detection/localization from language explanation into a two-stage optimization pipeline, and leverages localization masks as auxiliary inputs to VLMs to enhance interpretability, achieving comprehensive SOTA results across 9 benchmarks.

## Background & Motivation

With the rapid advancement of AIGC technologies (diffusion models, GANs, autoregressive Transformers), image manipulation has become increasingly accessible, posing serious challenges for image forgery detection and localization (IFDL). Existing methods have attempted to integrate VLMs (e.g., CLIP + LLM + SAM) into IFDL to improve interpretability, but the authors identify two critical issues:

**Semantic Plausibility vs. Authenticity**: VLMs such as CLIP are pretrained to align high-level semantics with language, causing the visual token representations of manipulated images — even when objects are replaced or inserted — to remain highly similar to those of authentic images (cosine similarity as high as 96–98%), making it nearly impossible for VLMs to distinguish real from forged content.

**Coupling Problem in Existing Pipelines**: Methods such as SIDA and FakeShield jointly optimize detection, localization, and language explanation within a single VLM. However, since VLMs lack forgery-specific priors, this coupling degrades detection and localization performance.

Core insight: Localization masks inherently and explicitly encode forgery concepts, and can therefore serve as additional priors for VLMs, simplifying their training optimization.

## Method

### Overall Architecture

IFDL-VLM decouples the IFDL task into two stages:

- **Stage-1**: Trains a ViT + SAM expert model for forgery detection and localization.
- **Stage-2**: Uses the detection/localization outputs from Stage-1 as auxiliary inputs to fine-tune a VLM for generating language explanations.

### Key Designs

1. **Decoupled Optimization**: Unlike existing methods, IFDL-VLM does not jointly optimize all subtasks within a single VLM. In Stage-1, a trainable ViT (initialized from CLIP-ViT-L/14) extracts a $\langle\text{SEG}\rangle$ token fed into a frozen SAM-H to generate localization masks, while the $\langle\text{CLS}\rangle$ token is used for three-way classification (authentic / fully synthesized / manipulated). **Design Motivation**: The semantic plausibility bias of VLMs interferes with low-level forgery artifact detection; decoupling isolates the detection/localization module from this bias.

2. **Region-Aware Visual Feature Enhancement**: The core innovation of Stage-2. The localization mask $M$ from Stage-1 is element-wise multiplied with the original image $x$ to extract the forged region, which is encoded separately by CLIP and fused via weighted aggregation:

$$T_{vis} = \alpha \cdot \text{CLIP}(x) + (1 - \alpha) \cdot \text{CLIP}(x \odot M)$$

   where $\alpha = 0.5$. This provides two benefits: (a) low-level forgery region cues enrich the visual features, improving the discriminability of representations between authentic and forged images; (b) the localization mask explicitly encodes forgery concepts, relieving the LLM from having to learn them implicitly from data and simplifying training optimization. At inference, the ground-truth $M$ is replaced by the predicted $\hat{M}$ from Stage-1.

3. **Multi-Head Attention Feature Fusion**: Patch-level features from the ViT are projected and aggregated via multi-head attention to produce the $\langle\text{SEG}\rangle$ token used as SAM's prompt embedding, while the global $\langle\text{CLS}\rangle$ token is passed to a linear classifier. This design allows the ViT to simultaneously handle pixel-level localization and image-level detection.

### Loss & Training

**Stage-1 Loss**:

$$\mathcal{L}_{st\text{-}1} = \mathcal{L}_{det} + \mathcal{L}_{loc} = \lambda_{det}\mathcal{L}_{ce}(\hat{D}, D) + \lambda_{bce}\mathcal{L}_{bce}(\hat{M}, M) + \lambda_{dice}\mathcal{L}_{dice}(\hat{M}, M)$$

where $\lambda_{bce} = \lambda_{dice} = \lambda_{det} = 1.0$.

**Stage-2 Loss**:

$$\mathcal{L}_{st\text{-}2} = \mathcal{L}_{ce}(\hat{y}_{des}, y_{des})$$

This is the autoregressive cross-entropy loss over the LLM's generated language explanation. The LLM backbone is Vicuna-13B.

Training details: AdamW optimizer, learning rate 1e-5, linear warmup-decay schedule, batch size 4 with gradient accumulation of 10, FP16/BF16 mixed precision.

## Key Experimental Results

### Main Results

**SID-Set Detection Performance**:

| Method | Overall Acc | Overall F1 | Note |
|--------|-------------|------------|------|
| SIDA-13B | 0.94 | 0.94 | Prev. SOTA |
| UnivFD | 0.65 | 0.80 | Traditional method |
| **IFDL-VLM** | **0.997** | **0.998** | Near-perfect |

**SID-Set Localization Performance**:

| Method | AUC | F1 | IoU | Gain |
|--------|-----|----|-----|------|
| SIDA-7B | 0.87 | 0.74 | 0.44 | - |
| **IFDL-VLM** | **0.99** | **0.87** | **0.65** | +21% IoU |

**Cross-Dataset Generalization (average over 8 datasets)**:

| Method | Avg IoU | Avg F1 | Gain |
|--------|---------|--------|------|
| FakeShield | 0.39 | 0.45 | - |
| SIDA-13B* | 0.38 | 0.45 | - |
| **IFDL-VLM** | **0.47** | **0.58** | +13% IoU, +19% F1 |

### Ablation Study

**Interpretability Evaluation (GPT-5 automated scoring, 0–5)**:

| Dimension | SIDA-13B | IFDL-VLM | Note |
|-----------|----------|----------|------|
| Mask | 1.22 | **2.28** | Localization mask quality |
| Tampered Content | 1.14 | **1.98** | Description of manipulated content |
| Overall | 1.44 | **2.36** | +63.9% improvement |

**CSS Semantic Similarity Evaluation**:

| Dimension | SIDA-13B | IFDL-VLM | Note |
|-----------|----------|----------|------|
| Areas | 0.61 | **0.67** | Tampered regions |
| Tampered Content | 0.44 | **0.49** | Manipulated content |
| CSS (weighted) | 0.57 | **0.62** | +8.8% weighted gain |

### Key Findings

- **VLM priors do not benefit detection/localization**: CLIP visual features exhibit 96–98% cosine similarity between authentic and forged images, providing almost no discriminative signal. Decoupling yields substantial improvements in detection and localization performance.
- **Localization masks augment VLMs**: Providing masks as explicit forgery concept inputs to the VLM significantly improves interpretability (GPT-5 score +63.9%, CSS +8.8%).
- **Human evaluation**: 65.2% of 50 evaluators preferred IFDL-VLM's explanations, compared to only 11.3% preferring SIDA-13B.
- **Cross-dataset generalization**: The framework achieves best performance on 7 out of 8 cross-domain datasets, validating its generalizability.

## Highlights & Insights

- The paper offers a rigorous and counterintuitive analysis of the *semantic plausibility bias* in VLMs and its detrimental effect on forgery detection — a finding of significant practical value.
- The "decouple and feedback" design philosophy is elegant: first train an expert model for detection/localization, then use its outputs to assist the VLM in generating explanations, rather than burdening the VLM with all tasks simultaneously.
- The method is architecturally simple yet highly effective — Stage-1 only adds a ViT and a frozen SAM decoder, and Stage-2 only modifies the visual input, with no complex architectural changes.

## Limitations & Future Work

- Stage-2 depends on the quality of Stage-1 localization masks; localization failures propagate as cascading errors into the explanation stage.
- Validation is limited to Vicuna-13B; it remains unexplored whether stronger LLMs (e.g., larger-scale models) could further improve interpretability.
- In cross-dataset generalization experiments, IFDL-VLM does not surpass FakeShield on IMD2020, indicating room for improvement on specific datasets.
- Computational efficiency is not discussed — the inference latency overhead of the two-stage pipeline is not analyzed.

## Related Work & Insights

- **SIDA / FakeShield**: Representative IFDL + VLM methods that couple CLIP + LLM + SAM in joint training; this paper demonstrates that decoupling is superior.
- **MVSS-Net / CAT-Net**: Traditional IFDL methods relying on handcrafted priors (BayarConv, DCT) to detect low-level artifacts.
- **SAM**: This paper freezes the SAM-H image encoder and fine-tunes only the mask decoder, effectively leveraging its segmentation capability.
- Broader inspiration: For multimodal auxiliary tasks, a paradigm in which expert models first perform reliable base-level inference, whose results are then fed back to large models for high-level understanding, may be generally preferable.

## Rating

- Novelty: ⭐⭐⭐⭐ (Insightful analysis of VLM bias + decoupled feedback design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (9 benchmarks + three-dimensional evaluation of detection/localization/interpretability + human evaluation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear motivation; rigorous logical derivation from observations to solutions)
- Value: ⭐⭐⭐⭐⭐ (Paradigm-level contribution to the IFDL field)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](modix_positional_index_scaling.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](do_vision_language_models_need_to_process_image_tokens.md)
- [\[CVPR 2026\] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models](activation_matters_test-time_activated_negative_labels_for_ood_detection_with_vi.md)
- [\[CVPR 2026\] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)

</div>

<!-- RELATED:END -->
