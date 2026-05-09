---
title: >-
  [Paper Note] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts
description: >-
  [ICCV 2025][Medical Imaging][long-tailed anomaly detection] This paper proposes a new task and benchmark for Long-Tailed Online Anomaly Detection (LTOAD). The core innovation is replacing class-label dependency with a learnable class-agnostic concept set, combined with a Concept VQ-VAE and a comprehensive prompt learning framework. The proposed method achieves state-of-the-art performance in both offline and online settings without requiring class labels.
tags:
  - ICCV 2025
  - Medical Imaging
  - long-tailed anomaly detection
  - online learning
  - class-agnostic concepts
  - VQ-VAE
  - prompt learning
date: 2026-05-08
content_hash: b8a195c0960e18ce
---

# Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts

**Conference**: ICCV 2025
**arXiv**: [2507.16946](https://arxiv.org/abs/2507.16946)
**Code**: [https://doi.org/10.5281/zenodo.16283852](https://doi.org/10.5281/zenodo.16283852) (benchmark)
**Area**: Medical Imaging / Anomaly Detection
**Keywords**: long-tailed anomaly detection, online learning, class-agnostic concepts, VQ-VAE, prompt learning

## TL;DR
This paper proposes a new task and benchmark for Long-Tailed Online Anomaly Detection (LTOAD). The core innovation is replacing class-label dependency with a learnable class-agnostic concept set, combined with a Concept VQ-VAE and a comprehensive prompt learning framework. The proposed method achieves state-of-the-art performance in both offline and online settings without requiring class labels.

## Background & Motivation

**State of the Field**: Anomaly detection (AD) aims to identify defective regions in images and is widely applied in industrial manufacturing and medical diagnosis. Various settings have been studied in recent years, including unsupervised one-class AD, unified AD (a single model handling all categories), long-tailed AD (LTAD), and online AD.

**Limitations of Prior Work**: Existing LTAD methods are class-aware, requiring class labels of input images to select corresponding class-specific modules. However, class labels are typically unavailable in online streaming data scenarios. Current online AD methods do not account for long-tailed distributions. Furthermore, LTAD architectures have not leveraged recent advances in VQ-VAE, and their prompt learning designs remain incomplete.

**Root Cause**: When long-tailed distribution, online learning, and the absence of class labels co-exist, existing methods break down. Class-aware methods rely on hard switching, which fails entirely when encountering unseen categories.

**Paper Goals**: Define the LTOAD task, design a class-agnostic framework that enables LTAD methods to operate in online settings, and propose an anomaly-adaptive online learning algorithm.

**Starting Point**: A key observation is that the class information of an image can be represented as a combination of multiple "concepts." For example, "transistor" can be composed of the concepts "semiconductor" and "circuit." By replacing explicit class labels $\mathcal{C}$ with a learnable concept set $\widehat{\mathcal{C}}$ and using soft weighting instead of hard switching, the model can remain functional even for unseen categories.

**Core Idea**: Replace class-label dependency with a learnable class-agnostic concept set, combined with a Concept VQ-VAE and anomaly-adaptive online learning, to achieve long-tailed online anomaly detection without class labels.

## Method

### Overall Architecture
The model adopts a two-branch pipeline: (1) Reconstruction branch R — reconstructs visual features via Concept VQ-VAE and detects anomalies by measuring the discrepancy between input and reconstructed features; (2) Semantic branch S — leverages the text-image alignment capability of VLMs, detecting anomalies by comparing visual features against normal/anomaly prompt features. The outputs of both branches are fused via a weighted harmonic mean into the final prediction. All modules use concept scores $\mathbf{p}$ for soft weighting rather than hard class-label switching.

### Key Designs

1. **Class-Agnostic Concept Set**:

    - Function: Replaces explicit class labels with a learned concept set $\widehat{\mathcal{C}}$, assigning a soft label $\mathbf{p} \in [0,1]^{\hat{K}}$ to each image.
    - Mechanism: Given training data and a vocabulary $\mathcal{V}$, CLIP is used to compute the similarity between each image and each word in the vocabulary. The top-$\hat{K}$ words are selected by voting to initialize the concept set, and concept embeddings are set as learnable parameters. At inference, $\mathbf{p} = \text{SoftMax}(\{\langle \mathbf{f}^f, \mathbf{t}_{\hat{c}} \rangle\})$.
    - Design Motivation: The concept set size need not match the actual number of classes, making it flexible and efficient. The soft weighting mechanism remains functional when encountering unseen categories, whereas hard-switching methods fail directly.

2. **Concept VQ-VAE**:

    - Function: The feature reconstruction module in the reconstruction branch, used to detect anomalous regions.
    - Mechanism: Built upon HVQ, with categories replaced by concepts. An independent quantization module $Q_{l,\hat{c}}$ is constructed for each concept at each layer, with its codebook initialized by sampling from the corresponding concept embeddings. During training, the quantization modules are soft-weighted by concept scores. Anomaly prediction is defined as $\hat{Y}^R = \frac{1}{2}(1 - \langle F^i, F^r \rangle)$.
    - Design Motivation: Conventional HVQ requires class labels for hard switching, which is unavailable in online settings. Concept VQ naturally adapts to the label-free scenario through soft weighting.

3. **Comprehensive Prompt Learning Framework**:

    - Function: Constructs normal and anomaly prompt sets for the semantic branch.
    - Mechanism: Normal prompts are initialized with the template "a normal $\hat{c}$"; anomaly prompts are initialized by querying an AI with "What are the 5 most likely anomalies to be observed for $\hat{c}$?" All prompt features are set as learnable. The semantic branch makes predictions by comparing the normal similarity $S^n = \sum_{\hat{c}} p_{\hat{c}} S_{\hat{c}}^n$ against the anomaly similarity $S^a$.
    - Design Motivation: Prior work either lacks category-specific anomaly prompts or does not support concept-level prompts.

4. **Anomaly-Adaptive Online Learning Algorithm $\mathcal{A}^{AA}$**:

    - Function: Updates the model using anomalous sample information in online settings.
    - Mechanism: For each online batch, the current model first predicts a pseudo anomaly map $\hat{M} = \mathcal{T}(\hat{Y})$. Samples likely to be anomalous (i.e., $r(\hat{Y}) \geq \tau$) are assigned a larger gradient weight $\beta$, as anomalous samples are unseen during offline training and thus more informative. Parameters are then updated via EMA to ensure stability: $\theta_t = \gamma \theta_{t-1} + (1-\gamma)\tilde{\theta}_t$.
    - Design Motivation: Naive online learning directly reuses the offline loss function and cannot leverage the information carried by anomalous samples in the online data stream.

### Loss & Training
- The overall loss combines the reconstruction branch loss and the semantic branch loss.
- The final prediction is fused via a weighted harmonic mean: $\hat{Y} = (\alpha(\hat{Y}^R)^{-1} + (1-\alpha)(\hat{Y}^S)^{-1})^{-1}$.
- EMA updates are employed during the online phase to ensure training stability.

## Key Experimental Results

### Main Results (Offline MVTec LTAD)

| Method | Class-Agnostic | exp100-Det | exp100-Seg | step100-Det | step100-Seg |
|--------|---------------|-----------|-----------|------------|------------|
| LTAD | ✗ | 88.86 | 94.46 | 87.36 | 93.83 |
| HVQ | ✗ | 87.43 | 95.25 | 85.39 | 94.17 |
| LTOAD* (label-free) | ✗ | 93.12 | 95.01 | 92.02 | 94.72 |
| **LTOAD** | **✓** | **93.42** | **95.21** | **92.33** | **95.11** |

### Ablation Study

| Configuration | MVTec Det. | MVTec Seg. | Notes |
|--------------|-----------|-----------|-------|
| Full LTOAD | 93.42 | 95.21 | Complete model |
| w/o Concept VQ (original HVQ) | 87.43 | 95.25 | VQ upgrade contributes ~6% in Det. |
| w/o soft switching (hard switching) | ~90 | ~94.2 | Soft switching is more flexible and effective |
| w/o anomaly prompts | ~91 | ~94.5 | Anomaly prompts notably benefit detection |

### Key Findings
- The class-agnostic approach not only avoids performance degradation but surpasses class-label-dependent LTAD methods in most settings (+4.63% image-AUROC on MVTec).
- Concept VQ-VAE contributes the largest improvement over the original HVQ, validating that concept-level quantization is more effective than class-level quantization.
- In the most challenging long-tailed online setting, LTOAD achieves a +0.53% image-AUROC improvement.
- The method demonstrates strong generalization in cross-dataset settings.
- Eight distinct online stream configurations comprehensively evaluate the robustness of the method.

## Highlights & Insights
- **Concept set as a replacement for class labels**: An elegant design — learnable soft concepts replace hard class labels, addressing the absence of class labels in online settings while naturally supporting unseen categories through the compositional nature of concepts. This approach is transferable to any scenario requiring a transition from class-aware to class-agnostic modeling.
- **Comprehensive LTOAD benchmark**: Eight stream configurations (blurry/disjoint × head-first/tail-first/mixed) systematically examine the various challenges posed by the combination of long-tailed distribution and online learning.
- **Anomaly-adaptive sample weighting**: During online learning, anomalous samples are identified and up-weighted using the model's own prediction scores — a concise and effective strategy.

## Limitations & Future Work
- The concept set size $\hat{K}$ must be set manually and may require tuning across different datasets.
- Anomaly prompts are generated via LLM, so their quality depends on the LLM's domain-specific knowledge.
- The EMA decay coefficient $\gamma$ is sensitive and affects online learning stability.
- Each concept requires an independent VQ module, causing parameter count to grow linearly with the number of concepts.
- Results on medical anomaly detection (e.g., Uni-Medical) warrant further in-depth analysis.

## Related Work & Insights
- **vs. LTAD**: LTAD is class-aware and requires class labels for hard switching, making it inapplicable in online settings. LTOAD fully removes class-label dependency via the concept set and achieves superior performance.
- **vs. HVQ**: HVQ employs class-specific codebooks. LTOAD's Concept VQ replaces classes with concepts, yielding greater flexibility.
- **vs. UniAD**: UniAD is a unified AD method but also requires class information. LTOAD surpasses it under a class-agnostic constraint.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — New LTOAD task definition + concept set replacing class labels; highly novel research direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 datasets + 8 online configurations + comprehensive offline/online evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — The LTOAD benchmark provides meaningful contributions to the community.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation via Dictionary Lookup](dictas_a_framework_for_class-generalizable_few-shot_anomaly_segmentation_via_dic.md)
- [\[CVPR 2026\] Bidirectional Multimodal Prompt Learning with Scale-Aware Training for Few-Shot Multi-Class Anomaly Detection](../../CVPR2026/medical_imaging/bidirectional_multimodal_prompt_learning_with_scale-aware_training_for_few-shot_.md)
- [\[ICLR 2026\] Dual Distillation for Few-Shot Anomaly Detection](../../ICLR2026/medical_imaging/dual_distillation_for_few-shot_anomaly_detection.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/medical_imaging/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[CVPR 2026\] MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](../../CVPR2026/medical_imaging/moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)

<!-- RELATED:END -->
