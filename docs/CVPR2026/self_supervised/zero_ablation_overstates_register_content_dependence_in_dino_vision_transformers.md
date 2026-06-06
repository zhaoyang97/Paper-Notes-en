---
title: >-
  [Paper Note] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers
description: >-
  [CVPR 2026 (HOW Workshop)][Self-Supervised Learning][register tokens] Three substitution control experiments (mean substitution, noise substitution…
tags:
  - "CVPR 2026 (HOW Workshop)"
  - "Self-Supervised Learning"
  - "register tokens"
  - "vision transformers"
  - "zero-ablation"
  - "DINO"
  - "interpretability"
date: 2026-05-08
content_hash: f0c6f4cb81dd4663
---

# Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers

**Conference**: CVPR 2026 (HOW Workshop)
**arXiv**: [2604.14433](https://arxiv.org/abs/2604.14433)  
**Code**: None  
**Area**: Self-Supervised Learning
**Keywords**: register tokens, vision transformers, zero-ablation, DINO, interpretability

## TL;DR

Three substitution control experiments (mean substitution, noise substitution, and cross-image shuffling) demonstrate that zero-ablation overstates the dependence on the precise content of register tokens in DINO-series ViTs — the model requires only "reasonable register-like activations" rather than image-specific values.

## Background & Motivation

Zero-ablation (replacing token activations with zero vectors) is a commonly used method for probing token function in ViTs. In DINOv2+registers and DINOv3, zeroing out register tokens leads to drops of up to 36.6 pp in classification and 30.9 pp in segmentation, superficially suggesting that registers are indispensable. However, zero vectors constitute out-of-distribution inputs relative to native register activations, and may exaggerate true content dependence. This is analogous to the confound in neuroscientific lesion studies, where damage cascades through interconnected circuits and produces an illusion of over-localization.

## Method

### Overall Architecture

Hook-based ablation is applied to three model families — DINOv2, DINOv2+registers, and DINOv3 (ViT-S and ViT-B) — replacing [CLS] or register hidden states after each block's output. Zero-ablation is compared against three substitution controls across four downstream tasks: classification, retrieval, correspondence, and segmentation.

### Key Designs

1. **Three Substitution Control Experiments**: (1) Mean substitution: layer-wise dataset-mean activations calibrated on 5,000 ImageNet images; (2) Noise substitution: layer-wise Gaussian noise matched in mean and variance; (3) Cross-image register shuffling: register activations randomly permuted within a batch, preserving real activation structure while breaking image-specific content.

2. **In-Distribution Validation**: Patch-wise cosine similarity analysis (cosine similarity 0.95–0.999) confirms that all three substitutions genuinely perturb internal representations, ruling out the possibility that substitution leaves features unchanged. JS divergence further quantifies that zero-ablation induces distributional shifts tens to hundreds of times larger than those caused by the substitution controls.

3. **Effective Rank Analysis and Attention Flow**: Registers compress patch geometry (effective rank reduced from 13.5 to 4.0), with DINOv3 exhibiting the most pronounced compression. Attention flow analysis shows that register attention accumulates gradually from intermediate layers, while classification dependence emerges abruptly at layers 10–11.

### Loss & Training

This is an analytical study and involves no training. All evaluations are conducted on frozen features.

## Key Experimental Results

### Main Results

| Condition | DINOv2+R Classification | DINOv3 Classification | DINOv2+R Segmentation | DINOv3 Segmentation |
|-----------|------------------------|----------------------|----------------------|---------------------|
| Full | 67.3% | 62.0% | Baseline | Baseline |
| Zero registers | -18.9 pp | **-36.6 pp** | -9.6 pp | **-30.9 pp** |
| Mean-sub | ≤1 pp change | ≤1 pp change | ≤1 pp change | ≤1 pp change |
| Noise-sub | ≤1 pp change | ≤1 pp change | ≤1 pp change | ≤1 pp change |
| Shuffle | ≤1 pp change | ≤1 pp change | ≤1 pp change | ≤1 pp change |

### Key Findings

- Only zero-ablation produces performance degradation; all three reasonable substitutions preserve performance across all tasks.
- Registers buffer the dependence of dense features on [CLS] (segmentation drop of 37 pp vs. <1 pp).
- Results are fully reproduced at the ViT-B scale.

## Highlights & Insights

- Elegantly exposes the methodological flaw of zero-ablation — injecting out-of-distribution inputs rather than removing function.
- The analogy to lesion studies in neuroscience is apt and pedagogically valuable.
- The conclusion is clear: register function operates as an "contextual channel" as expected; precise content is not required.

## Limitations & Future Work

- Evaluation is conducted only on frozen features; fine-tuned models may behave differently.
- Only DINO-series models are tested; other self-supervised ViTs may exhibit different behavior.
- The workshop format constrains the depth of certain analyses.

## Related Work & Insights

- Provides an important methodological caution for all work that uses zero-ablation for functional probing.
- The concept of "in-distribution controls" for activation substitution generalizes to mechanistic interpretability in NLP.
- The "structural channel" role of register tokens offers guidance for ViT design.

## Rating

7/10 — The methodological contribution is clear and significant, though the scope is limited as a workshop paper.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Vision Transformers Need More Than Registers](vision_transformers_need_more_than_registers.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](../../ICML2026/self_supervised/infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](../../ICML2026/self_supervised/from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)

</div>

<!-- RELATED:END -->
