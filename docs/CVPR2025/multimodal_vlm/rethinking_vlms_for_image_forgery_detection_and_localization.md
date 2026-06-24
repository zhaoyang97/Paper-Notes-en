---
title: >-
  [Paper Note] Rethinking VLMs for Image Forgery Detection and Localization
description: >-
  [CVPR 2025][Multimodal VLM][Image Forgery Detection and Localization] Proposed IFDL-VLM, demonstrating that VLM priors contribute minimally to forgery detection/localization. By decoupling detection/localization from linguistic explanation in a two-stage framework, the method utilizes a ViT+SAM expert model for detection and localization, subsequently employing the generated localization mask as an auxiliary input to enhance VLM training for generating interpretable textual e…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Image Forgery Detection and Localization"
  - "VLM"
  - "SAM"
  - "ViT"
  - "Interpretability"
  - "Tampering Detection"
date: 2026-05-08
content_hash: 49ec6c5846540a12
---

# Rethinking VLMs for Image Forgery Detection and Localization

**Conference**: CVPR 2025  
**arXiv**: [2603.12930](https://arxiv.org/abs/2603.12930)  
**Code**: To be confirmed  
**Area**: Multimodal VLM  
**Keywords**: Image Forgery Detection and Localization, VLM, SAM, ViT, Interpretability, Tampering Detection

## TL;DR

Proposed IFDL-VLM, demonstrating that VLM priors contribute minimally to forgery detection/localization. By decoupling detection/localization from linguistic explanation in a two-stage framework, the method utilizes a ViT+SAM expert model for detection and localization, subsequently employing the generated localization mask as an auxiliary input to enhance VLM training for generating interpretable textual explanations.

## Background & Motivation

**Importance of Image Forgery Detection**: With generative models (GAN, Diffusion) becoming increasingly powerful, the proliferation of high-quality forged images poses a severe threat to information security and social trust, necessitating efficient image forgery detection and localization (IFDL) methods.

**Attempts to Use VLMs in IFDL**: Recent works (e.g., FakeShield, SIDA) attempt to utilize VLMs for IFDL while outputting textual explanations. However, they combine detection, localization, and explanation during joint training.

**Inherent Bias of VLMs**: VLMs are naturally biased toward semantic coherence rather than authenticity judgment, making them insensitive to semantically consistent forgeries (e.g., copy-move, splicing, and other manipulations that do not alter overall semantics).

**Core Finding**: The authors experimentally discovered that VLM priors provide almost no additional gain for detection/localization—relying directly on a ViT-based expert model actually performs better.

**Reverse Finding**: However, the localization mask significantly boosts the interpretability training of the VLM. The mask provides precise spatial information on "where the tampering occurred," guiding the VLM to generate accurate textual explanations.

**Core Idea**: IFDL is decomposed into two stages: first, an expert model is used for detection/localization, and subsequently, the localization results are injected into the VLM to enhance its explanatory capabilities.

## Method

### Overall Architecture

IFDL-VLM adopts a two-stage decoupled architecture:

- **Stage 1**: Train a ViT + SAM expert model for forgery detection and localization, completely independent of the VLM.
- **Stage 2**: Inject the localization mask output from Stage 1 as an auxiliary visual input into the VLM via the Region-Aware Visual Feature Enhancement (RAVFE) module, fine-tuning the VLM to generate interpretable explanations.

### Key Design 1: ViT+SAM Forgery Detection/Localization Expert

- Utilizes a pre-trained ViT as the forgery feature extractor.
- Uses the mask decoder from SAM as the localization head to output pixel-level forgery masks.
- Outputs image-level detection results (real/fake) simultaneously through a binary classification head.
- Avoids introducing any linguistic modules to prevent interference from the semantic bias of VLMs.

### Key Design 2: Region-Aware Visual Feature Enhancement (RAVFE)

- Fuses the predicted forgery mask from Stage 1 with the original visual features.
- Guides the VLM to focus on the tampered regions via attention mechanisms.
- Serves the mask as a soft spatial prior to guide the VLM in understanding "where" and "how" the image was tampered with.
- Utilizes this only during the VLM fine-tuning phase, without affecting the detection/localization model.

### Key Design 3: Core Motivation for Two-Stage Decoupling

- **Why Not End-to-End**: Semantic priors in VLMs interfere with the detection of low-level forgery traces (e.g., JPEG artifacts, boundary inconsistencies), leading to degraded detection/localization performance.
- **Why Adding Mask Helps**: The mask provides explicit spatial evidence, shifting the VLM from "guessing where the anomaly is" to "explaining a known tampered region," significantly reducing the difficulty of the explanation task.

## Key Experimental Results

### Main Results

| Task | Metric | IFDL-VLM | SIDA | FakeShield | Gain |
|------|------|----------|------|-----------|------|
| Detection (SID-Set) | Accuracy | **99.7%** | — | — | — |
| Detection (SID-Set) | F1 | **99.8%** | — | — | — |
| Localization (SID-Set) | IoU | **0.65** | 0.54 | — | +21% |
| Cross-Dataset Localization | Avg IoU (8 benchmarks) | **0.47** | — | 0.42 | +13% |
| Interpretability | GPT-5 Score | **2.44** | 1.67 | — | +0.77 |

### Ablation Study

| Configuration | Detection Acc | Localization IoU | Explanation GPT Score |
|------|---------|---------|---------------|
| Full IFDL-VLM | **99.7%** | **0.65** | **2.44** |
| Using VLM for Detection/Localization | ~95% | ~0.48 | — |
| Stage 2 without Mask Assistance | — | — | ~1.8 |
| End-to-End Joint Training | ~96% | ~0.50 | ~2.0 |

### Key Findings

- The benefit provided by the VLM prior to detection/localization is negligible, and even has a negative impact.
- The localization mask as an auxiliary input significantly improves the accuracy of explanations during VLM training (+0.77 GPT score).
- The ViT+SAM expert model demonstrates excellent generalization across datasets (average of 0.47 IoU across 8 benchmarks).
- The two-stage decoupling outperforms end-to-end joint training across all dimensions.

## Highlights & Insights

1. **Counter-intuitive Finding**: Challenges the "one-size-fits-all VLM" assumption, experimentally demonstrating that VLMs provide minimal help for detection/localization in IFDL tasks, and expert models are superior.
2. **Generality of the Decoupling Concept**: The concept of decoupling "perception" from "understanding/explanation" can be extended to other tasks requiring fine-grained analysis with VLMs.
3. **New Paradigm of Mask as VLM Input**: Utilizing the output of an upstream model as an auxiliary spatial prior for the VLM enhances its capability in fine-grained spatial reasoning.
4. **High Practicality**: High detection accuracy (99.7%) combined with interpretable tampering descriptions meets real-world forensic needs.

## Limitations & Future Work

- Stage 1 and Stage 2 are trained separately, preventing end-to-end gradient optimization.
- The quality of the localization mask directly affects the quality of Stage 2 explanations, causing cascade propagation if Stage 1 fails.
- Validated only on image forgeries; scenarios such as video forgery and deepfakes have not been explored.
- The reliability of using GPT-5 scores as an interpretability evaluation metric remains open to discussion.

## Related Work & Insights

- **FakeShield / SIDA**: Directly employ VLMs end-to-end for IFDL and explanation; this work demonstrates that their detection/localization performance is limited by VLM biases.
- **SAM**: This work leverages the mask decoder architecture of SAM for forgery localization, demonstrating the value of foundation models in downstream security tasks.
- **Insights**: For other security detection tasks (e.g., deepfakes, AI-generated content detection), the decoupled paradigm of expert model + VLM explanation is worth exploring.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Value lies in the counter-intuitive findings, and the decoupling concept is clear.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across 8 cross-datasets + ablation studies + interpretability assessments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and strong argumentation.
- **Value**: ⭐⭐⭐⭐ — High detection accuracy and strong interpretability.
- **Overall Recommendation**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)
- [\[ICLR 2026\] Revisiting Confidence Calibration for Misclassification Detection in VLMs](../../ICLR2026/multimodal_vlm/revisiting_confidence_calibration_for_misclassification_detection_in_vlms.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[ECCV 2024\] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](../../ECCV2024/multimodal_vlm/addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)

</div>

<!-- RELATED:END -->
