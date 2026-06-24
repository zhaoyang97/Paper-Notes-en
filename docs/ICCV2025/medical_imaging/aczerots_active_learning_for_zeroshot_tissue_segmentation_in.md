---
title: >-
  [Paper Note] AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images
description: >-
  [ICCV 2025 (pp. 23508-23518)][Medical Imaging][Zero-shot segmentation] This work proposes AcZeroTS, a framework that integrates active learning with a VLM-based prototype-guided zero-shot segmentation model (ProZS). By simultaneously accounting for uncertainty, diversity, and the ability of selected samples to improve prototype coverage over unseen classes, the framework selects the most informative samples for annotation, achieving high-quality segmentation of both seen and…
tags:
  - "ICCV 2025 (pp. 23508-23518)"
  - "Medical Imaging"
  - "Zero-shot segmentation"
  - "active learning"
  - "pathology images"
  - "vision-language models"
  - "prototype-guided"
date: 2026-05-08
content_hash: a5cb4bc43a1c27cc
---

# AcZeroTS: Active Learning for Zero-shot Tissue Segmentation in Pathology Images

**Conference**: ICCV 2025 (pp. 23508-23518)  
**CVF**: [Link](https://openaccess.thecvf.com/content/ICCV2025/html/Tang_AcZeroTS_Active_Learning_for_Zero-shot_Tissue_Segmentation_in_Pathology_Images_ICCV_2025_paper.html)  
**Authors**: Jiao Tang, Junjie Zhou, Bo Qian, Peng Wan, Yingli Zuo, Wei Shao, Daoqiang Zhang  
**Code**: Unavailable  
**Area**: Semantic Segmentation / Pathology Image Analysis  
**Keywords**: Zero-shot segmentation, active learning, pathology images, vision-language models, prototype-guided  

## TL;DR

This work proposes AcZeroTS, a framework that integrates active learning with a VLM-based prototype-guided zero-shot segmentation model (ProZS). By simultaneously accounting for uncertainty, diversity, and the ability of selected samples to improve prototype coverage over unseen classes, the framework selects the most informative samples for annotation, achieving high-quality segmentation of both seen and unseen tissue types under minimal annotation budgets.

## Background & Motivation

- **Importance of pathological tissue segmentation**: Accurate segmentation of tissue regions in pathology images is critical for computer-aided cancer diagnosis.
- **Limitations of conventional methods**: Traditional segmentation models rely on large-scale annotated datasets, requiring domain experts to provide annotations for every tissue type. Given the complexity of the tumor microenvironment, collecting annotations covering all possible tissue types is exceptionally challenging.
- **Opportunities from VLM-based zero-shot segmentation**: Recent advances in vision-language models (VLMs) have made zero-shot pixel-level segmentation feasible—models trained exclusively on seen classes can generalize to both seen and unseen categories at inference time.
- **Remaining limitations of VLM-based approaches**: Although VLM-based zero-shot segmentation eliminates the need for unseen-class annotations, it still demands substantial annotation effort for seen classes.
- **Core motivation**: Can the annotation burden on seen classes be further reduced? This question motivates the introduction of active learning to jointly optimize seen- and unseen-class segmentation performance under minimal annotation budgets.

## Core Problem

**How to design an effective active learning strategy for zero-shot tissue segmentation, such that a model trained with annotations on only a small subset of seen-class samples can still maintain strong segmentation performance on unseen tissue types?**

Key challenges:
1. Conventional active learning focuses solely on improving performance on the current task (seen classes), without accounting for generalization to unseen classes.
2. Prototype quality directly determines unseen-class recognition in zero-shot segmentation; therefore, sample selection in active learning must be coupled with prototype quality.
3. A limited annotation budget must be allocated to balance seen-class performance and unseen-class generalization simultaneously.

## Method

### Overall Architecture

The AcZeroTS framework comprises two core components:

1. **ProZS (Prototype-guided Zero-shot Segmentation)**: A VLM-based prototype-guided zero-shot segmentation model.
2. **Active Selection Criterion**: A sample selection strategy specifically designed for zero-shot segmentation.

Overall pipeline:
1. **Initialization**: Train ProZS on a small labeled seed set.
2. **Active selection loop**: Select the most informative samples from the unlabeled pool according to the selection criterion.
3. **Expert annotation**: Domain experts annotate the selected samples.
4. **Retraining**: Retrain ProZS on the expanded labeled set.
5. **Iteration**: Repeat steps 2–4 until the annotation budget is exhausted.

### Key Designs

#### 1. ProZS — Prototype-Guided Zero-Shot Segmentation Model

- **Text prompt design**: An LLM is employed to generate descriptive text prompts for each tissue category, rather than relying on bare class names.
- **Visual-semantic alignment**: A VLM (e.g., CLIP or pathology-specific VLMs such as CONCH) is used to align image patch features with text features.
- **Prototype generation**: Visual prototypes for each seen class are learned from training samples.
- **Zero-shot inference**: At inference time, the VLM's text encoder generates semantic prototypes for unseen classes; these are combined with visual prototypes of seen classes for joint classification.
- **Key insight**: Prototypes must accurately represent seen classes while remaining discriminative against unseen classes—establishing a bridge between active learning and zero-shot generalization.

#### 2. Active Selection Criterion for Zero-Shot Segmentation

Conventional active learning criteria (e.g., uncertainty sampling, diversity sampling) optimize only for current-task performance. The proposed criterion jointly considers three dimensions:

- **Uncertainty**: Selects samples for which the model is most uncertain, maximizing informational gain per annotation.
- **Diversity**: Ensures that selected samples are spread across the feature space to avoid redundant annotations.
- **Prototype Coverage for Unseen Classes**: Ensures that prototypes derived from selected samples effectively summarize both seen and unseen classes—constituting the core innovation.

Intuition behind prototype coverage:
- If selected samples yield higher-quality prototypes, those prototypes not only represent seen classes more accurately but also form better classification boundaries with text-based prototypes of unseen classes within the VLM's shared semantic space.
- In essence, the zero-shot generalization objective is explicitly embedded into the active learning selection criterion.

### Loss & Training

- **Training strategy**: Iterative active learning loops that progressively expand the labeled set.
- **Segmentation loss**: A combination of cross-entropy loss and Dice loss is used for supervised segmentation training on seen classes.
- **Prototype update**: Seen-class prototypes are dynamically updated as new labeled samples are incorporated.
- **Inference**: Visual prototypes and text-based semantic prototypes are jointly used to perform inference over all categories (seen + unseen).

## Key Experimental Results

| Dataset | Type | Description |
|---------|------|-------------|
| TNBC | Pathology images | Triple-negative breast cancer tissue segmentation |
| HPBC | Pathology images | Breast cancer pathological tissue segmentation |
| Pascal VOC 2012 | Natural images | Validation of generalizability to natural scene settings |

- AcZeroTS outperforms existing methods on all datasets.
- Specific numerical results are currently unavailable due to inaccessibility of the full PDF; the paper claims to "demonstrate the superiority of our method in comparison with the existing studies."

### Ablation Study

Anticipated ablations include:
1. **Contribution of individual selection criterion components**: Uncertainty only vs. Uncertainty + Diversity vs. full three-component criterion.
2. **Text prompt design in ProZS**: Impact of different prompting strategies (bare class names vs. LLM-generated descriptive prompts) on performance.
3. **Effect of annotation budget**: Performance on seen/unseen classes under varying annotation proportions.
4. **Comparison with alternative active learning strategies**: Random sampling, CoreSet, BADGE, and other baselines.

## Highlights & Insights

1. **Novel problem formulation**: This work is the first to introduce active learning into zero-shot tissue segmentation, addressing the gap whereby VLM-based zero-shot methods still require extensive seen-class annotations.
2. **Zero-shot-aware selection criterion**: Unlike conventional active learning, the proposed criterion explicitly accounts for generalization to unseen classes by incorporating prototype coverage as a selection signal.
3. **Clear theoretical motivation**: The connection between active learning and zero-shot generalization is grounded in prototype quality and semantic space coverage.
4. **Cross-domain validation**: Generalizability is demonstrated not only on pathology datasets but also on natural images (Pascal VOC 2012).
5. **High practical value**: The framework substantially reduces annotation costs for pathological tissue segmentation, with direct implications for clinical deployment.

## Limitations & Future Work

1. **VLM backbone selection**: The paper likely evaluates only specific VLMs; future work could explore pathology-specialized VLMs such as CONCH, UNI, and Virchow.
2. **Reliance on predefined unseen classes**: The zero-shot setting requires unseen class names to be known in advance; extension to open-vocabulary settings is a natural direction.
3. **Single-round vs. iterative human-in-the-loop**: Further exploration of interactive, multi-round annotation refinement warrants investigation.
4. **Extension to finer-grained pathology tasks**: Tasks such as cell segmentation and gland segmentation present additional challenges worth addressing.
5. **Computational efficiency**: The sample selection process in active learning incurs additional computational overhead, which may be non-trivial for large-scale whole-slide image (WSI) scenarios.
6. **Prototype cardinality and update strategy**: A more thorough analysis of using single versus multiple prototypes per class, and of prototype update strategies, is needed.

## Related Work & Insights

| Dimension | Conventional Segmentation | VLM Zero-shot Segmentation | AcZeroTS |
|-----------|--------------------------|---------------------------|----------|
| Seen-class annotation | Large-scale | Large-scale | Minimal (active learning) |
| Unseen-class annotation | Required | Not required | Not required |
| Generalization to novel classes | Not possible | Possible | Possible |
| Annotation efficiency | Low | Medium | High |

- **vs. ZS3Net, SPNet, and related zero-shot segmentation methods**: These methods require full annotations for seen classes; AcZeroTS substantially reduces this requirement through active learning.
- **vs. Direct zero-shot inference with CLIP/CONCH**: Direct zero-shot inference yields relatively low performance; AcZeroTS improves significantly by learning prototypes from a small number of labeled samples.
- **vs. Conventional active learning (CoreSet, BADGE, Entropy, etc.)**: Traditional active learning disregards unseen-class generalization; the selection criterion in AcZeroTS jointly optimizes performance on both seen and unseen classes.

### Broader Implications

1. **A new paradigm of AL + zero-shot generalization**: The idea of coupling active learning sample selection with zero-shot learning objectives is transferable to other domains, such as remote sensing and autonomous driving.
2. **Prototypes as a dual-role bridge**: Prototypes simultaneously connect the semantic spaces of seen and unseen classes and provide a signal for active sample selection—a design principle with broad implications.
3. **Annotation cost reduction in pathology image analysis**: The combination of VLMs and active learning represents a promising direction for reducing medical image annotation costs in future research.
4. **Extension toward open-vocabulary segmentation**: Transitioning from a fixed unseen-class vocabulary to fully open-vocabulary settings is an interesting avenue for future work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to combine active learning with zero-shot tissue segmentation; the zero-shot-aware selection criterion is a genuinely novel design.
- **Technical Depth**: ⭐⭐⭐⭐ — The ProZS prototype-guided model and three-component selection criterion constitute a technically complete solution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Cross-domain validation across two pathology datasets and one natural image dataset.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated and the methodological presentation follows a coherent logical structure.
- **Impact**: ⭐⭐⭐⭐ — Practically valuable for the pathology AI community; annotation efficiency addresses a critical bottleneck.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](../../NeurIPS2025/medical_imaging/few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](../../NeurIPS2025/medical_imaging/zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[ICCV 2025\] SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images](seganypet_universal_promptable_segmentation_from_positron_emission_tomography_im.md)
- [\[CVPR 2025\] Unmasking Biases and Reliability Concerns in Convolutional Neural Networks Analysis of Cancer Pathology Images](../../CVPR2025/medical_imaging/unmasking_biases_and_reliability_concerns_in_convolutional_neural_networks_analy.md)

</div>

<!-- RELATED:END -->
