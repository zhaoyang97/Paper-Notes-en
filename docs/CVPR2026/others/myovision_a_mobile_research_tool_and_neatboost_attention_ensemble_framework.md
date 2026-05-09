---
title: >-
  [Paper Note] MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection
description: >-
  [CVPR 2026][neuroevolution] This paper proposes MyoVision, a smartphone-based transillumination imaging framework, and the NEATBoost-Attention neuroevolution-optimized ensemble model for low-cost, real-time three-class classification of chicken breast myopathies (Wooden Breast and Spaghetti Meat).
tags:
  - CVPR 2026
  - neuroevolution
  - ensemble learning
  - mobile imaging
  - food quality inspection
  - transillumination
date: 2026-05-08
content_hash: 4420badb5ad77849
---

# MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection

**Conference**: CVPR 2026  
**arXiv**: [2604.13456](https://arxiv.org/abs/2604.13456)  
**Code**: None  
**Area**: Computer Vision Applications  
**Keywords**: neuroevolution, ensemble learning, mobile imaging, food quality inspection, transillumination

## TL;DR

This paper proposes MyoVision, a smartphone-based transillumination imaging framework, and the NEATBoost-Attention neuroevolution-optimized ensemble model for low-cost, real-time three-class classification of chicken breast myopathies (Wooden Breast and Spaghetti Meat).

## Background & Motivation

Wooden Breast (WB) and Spaghetti Meat (SM) are structural myopathies that severely affect poultry meat quality. Current detection relies on subjective manual assessment or expensive laboratory-grade imaging systems. These defects primarily manifest as internal structural abnormalities rather than surface-visible features, posing challenges for automated detection. Existing automated methods depend on specialized hardware such as hyperspectral imaging and near-infrared spectroscopy, which are costly and limited in deployment. The research motivation is to achieve non-destructive, low-cost multi-class myopathy classification using consumer-grade smartphones.

## Method

### Overall Architecture

A three-stage pipeline: (1) smartphone transillumination imaging to acquire 14-bit RAW images; (2) extraction of 16-dimensional structural texture descriptors (gradient statistics, frequency-domain texture responses, and dense tissue features); (3) classification via the NEATBoost-Attention ensemble model.

### Key Designs

1. **Smartphone Transillumination Imaging**: Broadband white light is used to penetrate chicken breast tissue, capturing 2D spatially integrated attenuation patterns that encode aggregate changes in tissue density, fibrotic hardening, and fluid redistribution within muscle structure. 14-bit RAW capture preserves maximum dynamic range.

2. **NEAT Neuroevolution Optimization**: The NEAT algorithm simultaneously evolves network topology and weights to automatically discover 10 hyperparameters for LightGBM and 6 for AttentionMLP, eliminating manual tuning. Each genome encodes a small neural network that generates candidate hyperparameter configurations.

3. **Weighted Probability Fusion Ensemble**: LightGBM handles feature interactions and nonlinear decision boundaries, while AttentionMLP reweights input descriptors via feature attention. Ensemble weights are optimized using the Nelder-Mead simplex method, and final predictions are produced via weighted probability fusion.

### Loss & Training

Fitness evaluation uses weighted F1 scores from stratified cross-validation. Training data is class-balanced using SMOTE. Evolution proceeds through mutation, crossover, and speciation operations.

## Key Experimental Results

### Main Results

| Method | Test Accuracy | F1 Score |
|--------|--------------|----------|
| Traditional ML Baselines | Lower | Lower |
| Deep Learning Baselines | Lower | Lower |
| NEATBoost-Attention | **82.4%** | **0.83** |
| Hyperspectral Imaging Systems | ~Comparable | ~Comparable |

On 336 chicken breast samples, the proposed method achieves 82.4% test accuracy (F1=0.83), matching hyperspectral systems that are orders of magnitude more expensive.

### Key Findings

- NEAT-evolved hyperparameter configurations outperform grid search and random search.
- The feature attention mechanism effectively identifies the most discriminative texture descriptors.
- Transillumination imaging captures internal structural information inaccessible to surface imaging.

## Highlights & Insights

- Achieving detection performance comparable to professional laboratory equipment using consumer-grade devices demonstrates strong practical value.
- NEAT-based automatic architecture search avoids the manual tuning dilemma on small datasets.
- The multimodal research platform design (RAW + LiDAR + SAM + ChatGPT) exhibits strong forward-looking potential.

## Limitations & Future Work

- The dataset of 336 samples is relatively small, and generalization ability requires validation on larger data.
- An accuracy of 82.4% may still be insufficient for industrial deployment.
- Transillumination imaging is susceptible to ambient light, necessitating standardized acquisition conditions.

## Related Work & Insights

- This represents the first exploration of NEAT in food quality assessment, opening a new research direction.
- The transillumination imaging principle is generalizable to internal defect detection in other agricultural products.
- The neuroevolution optimization approach for small-sample tabular data offers a worthwhile reference for future work.

## Rating

5/10 — An interesting cross-domain application, but limited by dataset scale and modest methodological novelty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Boosting Adversarial Transferability via Ensemble Non-Attention](../../AAAI2026/others/boosting_adversarial_transferability_via_ensemble_non-attention.md)
- [\[CVPR 2026\] Crowdsourcing of Real-world Image Annotation via Visual Properties](crowdsourcing_of_real_world_image_annotation_via_visual_properties.md)
- [\[CVPR 2026\] SimRecon: SimReady Compositional Scene Reconstruction from Real Videos](simrecon_simready_compositional_scene_reconstruction_from_real_videos.md)
- [\[AAAI 2026\] Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation](../../AAAI2026/others/lost_in_time_a_meta-learning_framework_for_time-shift-tolerant_physiological_sig.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](neural_collapse_in_test-time_adaptation.md)

</div>

<!-- RELATED:END -->
