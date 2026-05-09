---
title: >-
  [Paper Note] OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism
description: >-
  [CVPR 2026][Self-Supervised Learning][generalized category discovery] This paper proposes OmniGCD, the first modality-agnostic generalized category discovery method. A GCDformer trained on synthetic data transforms the GCD latent space of arbitrary modalities at test time into representations more amenable to clustering, achieving zero-shot GCD across 16 datasets spanning four modalities.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - generalized category discovery
  - modality-agnostic
  - zero-shot
  - transformer
  - synthetic training
date: 2026-05-08
content_hash: 71822e3bb2a3b18f
---

# OmniGCD: Abstracting Generalized Category Discovery for Modality Agnosticism

**Conference**: CVPR 2026
**arXiv**: [2604.14762](https://arxiv.org/abs/2604.14762)
**Code**: [github.com/Jordan-HS/OmniGCD](https://github.com/Jordan-HS/OmniGCD)
**Area**: Self-Supervised Learning / Representation Learning
**Keywords**: generalized category discovery, modality-agnostic, zero-shot, transformer, synthetic training

## TL;DR

This paper proposes OmniGCD, the first modality-agnostic generalized category discovery method. A GCDformer trained on synthetic data transforms the GCD latent space of arbitrary modalities at test time into representations more amenable to clustering, achieving zero-shot GCD across 16 datasets spanning four modalities.

## Background & Motivation

- **State of the Field**: Generalized Category Discovery (GCD) emulates human category learning by simultaneously recognizing known classes and discovering novel ones under partial label supervision.
- **Limitations of Prior Work**: Existing GCD methods operate within a single modality and require dataset-specific fine-tuning, ignoring the fundamentally abstract nature of category learning.
- **Root Cause**: Neuroscience suggests that human category formation is an abstract process independent of sensory input, yet current methods conflate modality-specific encoding with category discovery.
- **Paper Goals**: Design a modality-agnostic framework trained once that performs zero-shot GCD across visual, textual, audio, and remote sensing modalities without dataset-specific adaptation.

## Method

### Overall Architecture

OmniGCD maps inputs to a feature space via modality-specific encoders, projects them into a low-dimensional GCD latent space via dimensionality reduction, and concatenates label embeddings or mask tokens before passing them to GCDformer. At test time, GCDformer transforms the latent space—without any gradient updates—to produce representations more suitable for $k$-means clustering.

### Key Designs

1. **GCDformer Transformer**: A non-causal self-attention Transformer based on the GPT-2 architecture. Its input is the concatenation of data tokens (dimensionality-reduced features of dimension $d$) and label tokens (sinusoidal positional encodings or learnable mask tokens of dimension $d_l$). No positional encoding is applied to data tokens, as the GCD input constitutes a set rather than a sequence. The training objective is a contrastive loss that pulls same-class points together and pushes different-class points apart.

2. **Synthetic Data Training**: GCDformer is trained exclusively on synthetic data to preserve modality agnosticism. The synthetic data must satisfy two key properties: (1) sufficient coverage of the GCD latent space, and (2) alignment with the distribution of real data. A low-dimensional latent space is chosen to keep the sampling space tractable, and an appropriate dimensionality reduction method is selected accordingly.

3. **Dimensionality Reduction Method Selection**: PCA, UMAP, and t-SNE are compared. t-SNE achieves the best overall performance in terms of synthetic-to-real KL divergence (1.41) as well as cluster separation, cluster spread, and cluster overlap metrics. The nonlinear nature of t-SNE and its heavy-tailed Student-$t$ distribution better preserve local structure and alleviate the crowding problem.

### Loss & Training

A contrastive loss with a margin parameter is employed: same-class pairs are pulled together via $L_2$ distance, while different-class pairs are pushed apart subject to a margin constraint. GCDformer is trained once and a single model is applied to all 16 datasets.

## Key Experimental Results

### Main Results

Average accuracy gains (in percentage points) across 16 datasets and 4 modalities:

| Modality | Known-Class Gain | Novel-Class Gain |
|----------|-----------------|-----------------|
| Visual | +6.2 pp | +6.2 pp |
| Text | +17.9 pp | +17.9 pp |
| Audio | +1.5 pp | +1.5 pp |
| Remote Sensing | +12.7 pp | +12.7 pp |

GCD on the audio modality is demonstrated for the first time.

### Ablation Study

- t-SNE dimensionality reduction outperforms PCA and UMAP, achieving the lowest KL divergence and best cluster quality.
- GCDformer overfits quickly on synthetic data, necessitating sufficient sampling diversity.
- Encoder quality directly governs final GCD performance.

### Key Findings

- Abstracting GCD as a representation space transformation problem offers a novel perspective.
- Synthetic data training enables genuine modality agnosticism.
- Encoder quality is the primary performance bottleneck—a better "perceptual front-end" directly yields better GCD.

## Highlights & Insights

- The research motivation, inspired by abstract category formation in the prefrontal cortex, is well-grounded in neuroscience.
- Decoupling representation learning from category discovery allows modality encoders and GCD capabilities to advance independently.
- The zero-shot GCD setting addresses a previously uncharted gap in the literature.

## Limitations & Future Work

- The non-parametric nature of t-SNE precludes immediate inference on unseen data points.
- The low-dimensional latent space may discard information critical for fine-grained discrimination in certain tasks.
- The synthetic data generation strategy still requires manual design.

## Related Work & Insights

- The use of Transformers as universal set processors can be generalized to other grouping and clustering tasks.
- The paradigm of synthetic training followed by zero-shot inference offers insights for cross-domain generalization.
- The modality-agnostic GCD benchmark provides an evaluation framework for future work.

## Rating

7/10 — The problem formulation is novel and the cross-modal generalization is impressive; however, the reliance on t-SNE and the constraints imposed by the low-dimensional latent space remain open challenges.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[CVPR 2026\] GeoChemAD: Benchmarking Unsupervised Geochemical Anomaly Detection for Mineral Exploration](geochemad_benchmarking_unsupervised_geochemical_anomaly_detection_for_mineral_ex.md)
- [\[CVPR 2026\] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)

<!-- RELATED:END -->
