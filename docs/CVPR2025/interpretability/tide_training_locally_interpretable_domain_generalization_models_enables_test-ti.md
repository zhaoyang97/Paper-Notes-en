---
title: >-
  [Paper Note] TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction
description: >-
  [CVPR 2025][Interpretability][Single-Source Domain Generalization] This paper proposes TIDE, a novel training scheme for single-source domain generalization. It leverages diffusion models and LLMs to automatically generate class-level concept annotations (e.g., "bird = sharp beak + wings + claws"). By training the model to focus on domain-invariant local concepts rather than global background features via a concept saliency alignment loss, the model can automatically correct…
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Single-Source Domain Generalization"
  - "Local Interpretability"
  - "Concept Alignment"
  - "Test-time Correction"
  - "Diffusion Models"
date: 2026-05-08
content_hash: 4203d4ac9c960650
---

# TIDE: Training Locally Interpretable Domain Generalization Models Enables Test-time Correction

**Conference**: CVPR 2025  
**Code**: None  
**Area**: Domain Generalization  
**Keywords**: Single-Source Domain Generalization, Local Interpretability, Concept Alignment, Test-time Correction, Diffusion Models

## TL;DR

This paper proposes TIDE, a novel training scheme for single-source domain generalization. It leverages diffusion models and LLMs to automatically generate class-level concept annotations (e.g., "bird = sharp beak + wings + claws"). By training the model to focus on domain-invariant local concepts rather than global background features via a concept saliency alignment loss, the model can automatically correct erroneous predictions caused by domain shift during test time using concept saliency maps.

## Background & Motivation

### Background

**Background**: Domain Generalization (DG) aims to train models that can generalize to unseen target domains. Single-source domain generalization (SSDG) is even more challenging—training with data from only a single source domain while requiring robust performance on target domains with vastly different distributions. Existing methods primarily rely on data augmentation (e.g., style transfer, frequency domain transformation) to simulate domain shifts.

**Limitations of Prior Work**: (1) Reliance on global features: Existing methods tend to learn global discriminative features (e.g., the overall appearance of an entire bird). However, these global features are not robust to semantic shifts such as background changes and viewpoint variations. For example, for "birds on water" and "birds on a branch," global features inevitably incorporate background information. (2) Augmentation strategies treat symptoms rather than root causes: Even with extensive augmentations to simulate various domain shifts, models may still learn shortcuts within the augmented data rather than domain-invariant concepts. (3) Lack of concept-level annotations: Training a model to focus on local concepts (e.g., using "beak shape" as a criterion for birds) requires concept-level annotations and localization information, which are absent in existing datasets.

**Key Challenge**: Domain-invariant features are often local (e.g., characteristic parts of an object), but forcing the model to focus on local concepts requires part-level annotations. Without concept annotations, models default to learning global statistical features, leading to failure under domain shift.

**Goal**: (1) How to automatically generate concept-level annotations? (2) How to train the model to focus on domain-invariant local concepts? (3) How to leverage concept information to correct domain shifts at test time?

**Key Insight**: Leverage the cross-domain knowledge of diffusion models and LLMs to automatically generate concept annotations, and design a concept saliency alignment loss to force the classifier's attention to align with the concept regions.

**Core Idea**: Use diffusion models + LLMs to automatically generate category concept annotations and localization maps, train the model to focus on local concepts for interpretable predictions, and detect and correct domain shifts at test time using concept saliency maps.

## Method

### Overall Architecture

TIDE consists of three phases: (1) Concept Annotation Generation: An LLM is utilized to generate a list of discriminative local concepts for each category; then, the cross-attention maps of a diffusion model are leveraged to generate concept localization maps (mapping which pixels correspond to which concept). (2) Concept Alignment Training: A concept saliency alignment loss is designed to align the classifier's gradient saliency maps with the concept localization maps. (3) Test-time Correction: During inference, the classifier's concept saliency map is computed for the test input. If the model relies heavily on non-concept regions (e.g., backgrounds), a domain shift is detected, and the prediction is automatically corrected.

### Key Designs

1. **Automated Concept Annotation Pipeline**:
    - **Function**: Generates category-level concepts and their spatial localization without manual annotation.
    - **Mechanism**: Step 1: Use an LLM (e.g., GPT-4) to generate a list of discriminative concepts for each category—for instance, generating ["large eyes", "flat facial disc", "hooked beak", "feather tufts"] for "owl". Step 2: Leverage the cross-attention maps of a pre-trained diffusion model (e.g., Stable Diffusion) to generate heatmaps localizing each concept in the training images. The diffusion model naturally learns the correspondence between textual concepts and spatial image regions during the denoising process.
    - **Design Motivation**: Manual annotation of concept localization maps is prohibitively expensive. The combination of diffusion models and LLMs leverages the cross-domain knowledge inherent in large-scale pre-trained models. The two components are complementary: the LLM knows *what* concepts are important, while the diffusion model knows *where* those concepts are located.

2. **Concept Saliency Alignment Loss**:
    - **Function**: Forces the classifier to focus on domain-invariant local concept regions.
    - **Mechanism**: (1) Compute gradient saliency maps (such as Grad-CAM or similar techniques) representing where the classifier focuses to make a classification decision. (2) Apply alignment constraints between the gradient saliency map and the concept localization map generated in Step 1—the high-value regions in the saliency map should overlap with the high-value regions in the concept localization map. The loss function can be formulated as a cosine similarity loss or KL divergence between the two maps.
    - **Design Motivation**: Unconstrained classifiers often focus on backgrounds (e.g., water indicating a waterbird, grass indicating a land bird). Concept alignment forces the model to focus on the discriminative parts of the object itself.

3. **Test-time Concept Correction**:
    - **Function**: Detects and corrects errors caused by domain shift during inference.
    - **Mechanism**: For a test sample, the classifier's concept saliency map is computed to inspect whether the model attends to the concept regions. If the saliency map is mostly concentrated on non-concept regions (such as the background), it indicates that the model might be misled by domain shift. At this point, local augmentations or feature modifications are applied to the input—for example, preserving features in the concept regions while masking out non-concept regions, followed by re-classification.
    - **Design Motivation**: Domain-invariant concept saliency maps provide an "anchor" for detecting out-of-distribution shifts during test-time—deviation from the concept regions signals a domain shift.

## Key Experimental Results

### Key Findings

- On single-source domain generalization benchmarks such as PACS, VLCS, and OfficeHome, TIDE significantly outperforms existing SOTA methods (with an accuracy improvement of approximately 2-5%).
- The performance under semantic shifts (e.g., background changes, viewpoint variations) is particularly outstanding.
- Visualization of concept saliency maps demonstrates that the model indeed learns to focus on discriminative local concepts.
- Test-time correction repairs about 30-40% of erroneous predictions without affecting correctly classified samples.
- Ablation studies indicate that the concept alignment loss contributes the most, while test-time correction yields an additional 1-2% improvement.
- The quality of automatically generated concept annotations is close to that of manual annotations.

## Highlights & Insights

- **Automated Concept Annotation**: The combination of LLMs and diffusion models is an innovative approach to generating concept-level annotations.
- **Interpretability Empowering Generalization**: Improving generalizability by enhancing the interpretability of the model (forcing it to focus on the correct concepts).
- **Intuitive Test-Time Correction**: The logical chain of "saliency map deviation $\to$ domain shift detection $\to$ correction" is highly intuitive and complete.
- **Practical Significance**: Single-source domain generalization is the setting closest to real-world deployment scenarios.

## Limitations & Future Work

- Concepts generated by LLMs may be incomplete or inaccurate, and specific domains (such as medical imaging) still require domain-specific expertise.
- The granularity of diffusion model attention maps is limited under high resolutions.
- Test-time correction incurs additional inference overhead (recomputing saliency maps).
- Extension and evaluation in multi-source domain generalization scenarios remain to be investigated.
- The "masking" operation of concept regions in the forward pass might introduce out-of-distribution artifacts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[NeurIPS 2025\] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models](../../NeurIPS2025/interpretability/specialization_after_generalization_towards_understanding_test-time_training_in_.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[CVPR 2025\] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis](prompt-cam_making_vision_transformers_interpretable_for_fine-grained_analysis.md)
- [\[CVPR 2025\] Differentiable Inverse Rendering with Interpretable Basis BRDFs](differentiable_inverse_rendering_with_interpretable_basis_brdfs.md)

</div>

<!-- RELATED:END -->
