---
title: >-
  [Paper Note] Do Vision Models Perceive Illusory Motion in Static Images Like Humans?
description: >-
  [CVPR 2026][motion illusion] This paper systematically evaluates a range of optical flow models on static-image motion illusions such as the Rotating Snakes…
tags:
  - "CVPR 2026"
  - "motion illusion"
  - "optical flow models"
  - "human vision"
  - "rotating snakes illusion"
  - "biologically-inspired models"
date: 2026-05-08
content_hash: 0d895bce5266e4d6
---

# Do Vision Models Perceive Illusory Motion in Static Images Like Humans?

**Conference**: CVPR 2026
**arXiv**: [2604.09853](https://arxiv.org/abs/2604.09853)
**Code**: Available
**Area**: Visual Perception / Computational Neuroscience
**Keywords**: motion illusion, optical flow models, human vision, rotating snakes illusion, biologically-inspired models

## TL;DR

This paper systematically evaluates a range of optical flow models on static-image motion illusions such as the Rotating Snakes, finding that only the biologically-inspired Dual-Channel model reproduces the human-perceived rotational motion under simulated saccade conditions.

## Background & Motivation

**Background**: DNNs have surpassed human performance on optical flow benchmarks, yet robustness gaps remain. Visual motion illusions provide a powerful tool for probing human–machine differences, but existing studies have focused primarily on dynamic illusions (e.g., reverse-phi), leaving static-image illusions underexplored.

**Limitations of Prior Work**: The Rotating Snakes illusion—in which humans strongly perceive rotational motion in a completely static image—has not been assessed in terms of whether existing optical flow models can reproduce it. The illusion depends on subtle luminance asymmetries and fixational eye movements.

**Key Challenge**: Standard DNN optical flow models achieve strong benchmark performance, yet it remains unclear whether their computational strategies share fundamental principles with the human visual system.

**Goal**: To evaluate the ability of representative DNN and biologically-inspired motion models to reproduce static-image motion illusions, and to identify the key computational components responsible.

**Key Insight**: An in silico psychophysics approach is adopted, systematically comparing 10 motion estimation models within a unified experimental pipeline.

**Core Idea**: Dual-channel motion processing, transient signals from eye movements, and recurrent integration are the critical mechanisms for reproducing human-like motion perception.

## Method

### Overall Architecture

(1) Generate Rotating Snakes illusion images and control images (three color schemes: grayscale / blue–yellow / red–green); (2) evaluate 10 models under both static and simulated saccade conditions; (3) conduct ablation analyses to identify key components.

### Key Designs

1. **Unified Experimental Pipeline**:

    - **Function**: Enable fair comparison across architectures under controlled conditions.
    - **Mechanism**: All models use official pretrained weights and are evaluated on identical illusion/control images. Simulated saccades are produced by translating images to generate transient retinal slip.
    - **Design Motivation**: Ensure that observed differences are attributable to model architecture rather than training or evaluation discrepancies.

2. **Simulated Saccade Condition**:

    - **Function**: Replicate the physiological conditions under which humans view the Rotating Snakes.
    - **Mechanism**: Human perception of the illusion requires transient signals provided by fixational eye movements such as saccades. Image translation is used to simulate this retinal slip.
    - **Design Motivation**: Psychophysical studies show the illusion is substantially attenuated under fixed gaze; eye movements are the key trigger.

3. **Ablation Analysis**:

    - **Function**: Identify the computational components essential for reproducing the illusion.
    - **Mechanism**: Systematic ablations of the Dual-Channel model examine: (1) the contribution of luminance-based motion signals; (2) the contribution of higher-order color–feature motion signals; (3) the role of the recurrent attention mechanism.
    - **Design Motivation**: Determine which computational principles are necessary for human-like motion perception.

### Loss & Training

This work is purely inference-based; no training is involved. All models are used with their original pretrained weights.

## Key Experimental Results

### Main Results

| Model Type | Static Condition | Saccade Condition | Reproduces Illusion |
|---|---|---|---|
| Multi-scale DNN (FlowNet, etc.) | No rotational flow | No rotational flow | ✗ |
| Recurrent-decoder DNN (RAFT, etc.) | No rotational flow | No rotational flow | ✗ |
| Dual-Channel (biologically-inspired) | Weak signal | Expected rotational motion | ✓ |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| w/o luminance channel | Illusion weakened | Luminance signals contribute significantly |
| w/o color–feature channel | Illusion weakened | Higher-order signals also contribute |
| w/o recurrent attention | Illusion disappears | Critical for integrating local cues |
| Full Dual-Channel | Strongest agreement | All components act synergistically |

### Key Findings

- The majority of DNN optical flow models entirely fail to produce human-consistent motion flow fields on static images.
- The Dual-Channel model exhibits the expected rotational motion only under simulated saccade conditions; the effect is also weak under the static condition.
- The recurrent attention mechanism is the critical component for integrating local cues into global rotational percepts.

## Highlights & Insights

- **Motion illusions as model diagnostic tools**: Leveraging human perceptual biases to distinguish models that "work" from those that "work like humans."
- **Validation of biologically-inspired computational principles**: Dual-channel motion processing, eye-movement transients, and recurrent integration constitute three transferable design principles.
- **Implications for robust visual system design**: Models capable of reproducing human perceptual biases may also exhibit greater robustness in real-world settings.

## Limitations & Future Work

- Only a limited variety of motion illusion types are tested.
- The optical flow estimation performance of the Dual-Channel model is not benchmarked against mainstream DNNs.
- Only zero-shot inference is analyzed; whether fine-tuning could enable DNNs to learn to reproduce the illusion remains unexplored.

## Related Work & Insights

- **vs. standard optical flow benchmarks**: Strong benchmark performance does not imply alignment with human vision; motion illusions provide a complementary evaluation dimension.
- **vs. reverse-phi studies**: Reverse-phi is a dynamic illusion, whereas Rotating Snakes is a static illusion, placing higher demands on the model.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic evaluation of static motion illusions in computational vision.
- Experimental Thoroughness: ⭐⭐⭐⭐ 10 models × multiple conditions × ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Interdisciplinary research well organized.
- Value: ⭐⭐⭐ Informative for optical flow model design, though practical applications are limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](vit3_unlocking_test_time_training_in_vision.md)
- [\[NeurIPS 2025\] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](../../NeurIPS2025/others/brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the.md)
- [\[CVPR 2026\] U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](clipfree_label_free_unsupervised_concept_bottlenec.md)

</div>

<!-- RELATED:END -->
