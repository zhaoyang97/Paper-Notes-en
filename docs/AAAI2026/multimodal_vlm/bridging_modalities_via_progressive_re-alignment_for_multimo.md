---
title: >-
  [Paper Note] Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)
description: >-
  [AAAI 2026][Multimodal VLM][Multimodal test-time adaptation] This paper proposes BriMPR, a framework that decomposes multimodal test-time adaptation (MMTTA) into multiple unimodal feature alignment subproblems via a divi…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Multimodal test-time adaptation"
  - "cross-modal alignment"
  - "prompt tuning"
  - "contrastive learning"
  - "distribution calibration"
date: 2026-05-08
content_hash: 6318370c15a7b5dc
---

# Bridging Modalities via Progressive Re-alignment for Multimodal Test-Time Adaptation (BriMPR)

**Conference**: AAAI 2026
**arXiv**: [2511.22862](https://arxiv.org/abs/2511.22862)  
**Code**: [https://github.com/Luchicken/BriMPR](https://github.com/Luchicken/BriMPR)  
**Area**: Multimodal VLM
**Keywords**: Multimodal test-time adaptation, cross-modal alignment, prompt tuning, contrastive learning, distribution calibration

## TL;DR

This paper proposes BriMPR, a framework that decomposes multimodal test-time adaptation (MMTTA) into multiple unimodal feature alignment subproblems via a divide-and-conquer strategy. It first calibrates the global feature distribution of each modality through prompt tuning to achieve initial cross-modal semantic alignment, then refines the alignment via cross-modal masked embedding recombination and instance-level contrastive learning.

## Background & Motivation

- **Test-time adaptation (TTA)** adapts models online using unlabeled test data during inference to bridge the distribution gap between source and target domains. However, existing TTA methods are primarily designed for unimodal tasks.
- **Challenges in multimodal scenarios**: Different modalities may suffer from varying degrees of distribution shift, leading to a **coupled effect** of **shallow unimodal feature shift** and **high-level cross-modal semantic misalignment**.
- Limitations of existing methods:
    - Unimodal TTA methods such as EATA reduce prediction uncertainty by minimizing entropy but cannot effectively bridge the domain gap across modalities.
    - READ dynamically assigns modality weights by updating the self-attention layers of the fusion module but lacks correction of shallow unimodal features.
    - Both types of methods lead to a severe decline in the discriminability of fused multimodal features (confirmed by t-SNE visualization).

## Core Problem

How to effectively decouple and resolve the coupled effect of **unimodal feature shift** and **cross-modal semantic misalignment** in multimodal data during the test phase, so as to re-align the features of each modality?

## Method

### Overall Architecture

BriMPR consists of two progressively enhanced modules:
1. **PMGFA** (Prompt-driven Modality-specific Global Feature Alignment): initial cross-modal alignment
2. **IIAE** (Inter-modal Interaction Enhancement for Alignment Refinement): alignment refinement comprising CMER and IICL

The source model is decomposed into two modality-specific encoders ($\Phi_a$ for audio, $\Phi_v$ for visual), a joint module $\Psi$, and a classifier $h$. **Only the prompts inserted into each modality encoder are updated**; all other parameters are frozen.

### Key Designs

#### 1. PMGFA — Prompt-driven Modality-specific Global Feature Alignment

**Core Idea**: Since each modality is already well-aligned in the source-domain feature space, MMTTA is decomposed into multiple unimodal alignment subproblems. As long as the target features of each modality can be mapped back to the corresponding source feature space, cross-modal semantic alignment is achieved indirectly.

**Key Innovation — Diagonal Covariance as a Substitute for Full Covariance**:
- Conventional methods align distributions by matching first-order and second-order moments (covariance matrix $\Sigma$), but the estimation error of the covariance matrix for high-dimensional data is $O(d^2/n)$.
- The paper proves (Theorem 1) that retaining only the diagonal elements of the covariance matrix (the variance vector) reduces the estimation error to $O(d/n)$, a $d$-fold reduction.
- The universal function approximation capacity of prompt tuning is used to implicitly map the target feature space back to the source feature space.

**Specific Approach**: Learnable prompts are inserted at each layer of each modality encoder, minimizing the difference in mean and standard deviation between source and target feature distributions at each layer:

$$\mathcal{L}_\text{PMGFA} = \sum_{u \in \{a,v\}} \frac{1}{N} \sum_{i=1}^{N} (\|\hat{\mu}_i^{t,u} - \hat{\mu}_i^{s,u}\|_2 + \|\hat{\sigma}_i^{t,u} - \hat{\sigma}_i^{s,u}\|_2)$$

Source-domain statistics are pre-computed offline (requiring only 32 unlabeled source samples) and are not needed during the test phase.

#### 2. CMER — Cross-modal Masked Embedding Recombination

- 50% of the patches of one modality are randomly masked; after encoding, the masked representation is recombined with the complete embeddings of the other modality and fed into the joint module, simulating augmented representations with single-modality corruption.
- Predictions from complete multimodal data serve as pseudo-labels to supervise the learning of the augmented inputs.
- **Adaptive temperature scaling** $\text{AdaTp} = 1 + \tau_0 / (1 + \exp(D_0 - \text{Disc}_J))$: a higher temperature is applied when distributional discrepancy is large to alleviate overconfidence, converging to 1 when discrepancy is small.
- **Adaptive weight** $\lambda_u = 1 - \text{Disc}_u / (\text{Disc}_a + \text{Disc}_v)$: assigns higher weight to the masked augmentation of the modality with smaller distribution shift.

**Intuition**: Deliberately discarding high-quality modality information forces the corrupted modality to independently derive the correct result.

#### 3. IICL — Inter-modal Instance-level Contrastive Learning

- Representations of different modalities from the same instance form positive pairs; those from different instances form negative pairs.
- Standard InfoNCE contrastive loss with temperature hyperparameter $\tau$.
- Further refines cross-modal alignment at the instance level following initial distribution alignment.

### Loss & Training

$$\mathcal{L}_\text{BriMPR} = \mathcal{L}_\text{PMGFA} + \mathcal{L}_\text{CMER} + \mathcal{L}_\text{IICL}$$

- Optimizer: Adam, learning rate 1e-4, batch size 64
- Default number of prompts per layer: 10, randomly initialized
- Masking ratio: 0.5
- AdaTp hyperparameters: $\tau_0=0.2$, $D_0=5$
- Contrastive learning temperature $\tau$: 0.07 for unimodal shift, 0.25 for multimodal shift
- 3 random seeds, RTX-3090

## Key Experimental Results

### Datasets
- **Kinetics50-C / VGGSound-C**: Audio-visual bimodal, based on CAV-MAE, 15 video corruptions + 6 audio corruptions, 5 severity levels
- **CMU-MOSI / CH-SIMS**: Text + video + audio trimodal, real-world domain shift

### Main Results — Unimodal Shift (Severity 5)

| Setting | Source | ABPEM (AAAI'25) | SuMi (ICLR'25) | **BriMPR** |
|---------|--------|-----------------|----------------|-----------|
| K50-C Video Corruption | 60.5 | 64.1 | 63.9 | **65.9** |
| VGGSound-C Video Corruption | 56.2 | 52.4 | 57.3 | **57.7** |
| K50-C Audio Corruption | 69.4 | 71.4 | 71.9 | **72.0** |
| VGGSound-C Audio Corruption | 25.0 | 29.5 | 33.2 | **36.5** |

**The most significant improvements occur when the dominant modality is corrupted**: K50-C video corruption 60.5→65.9 (+5.4); VGGSound-C audio corruption 25.0→36.5 (+11.5).

### Main Results — Multimodal Shift (Severity 5)

| Setting | Source | FOA (ICML'24) | ABPEM (AAAI'25) | **BriMPR** |
|---------|--------|--------------|-----------------|-----------|
| K50-C Both Modalities Corrupted | 31.8 | 39.9 | 39.4 | **40.9** |
| VGGSound-C Both Modalities Corrupted | 9.5 | 13.9 | 15.2 | **20.7** |

Under the VGGSound-C bimodal corruption setting, BriMPR achieves a substantial lead (20.7 vs. 15.2), a gain of 5.5 points.

### Real-world Domain Shift

| Setting | Source | READ | SuMi | **BriMPR** |
|---------|--------|------|------|-----------|
| MOSI→SIMS (ACC) | 46.0 | 32.4 | 44.4 | **58.2** |
| SIMS→MOSI (ACC) | 45.6 | 44.5 | 45.0 | **57.6** |

**Only BriMPR exceeds random chance (>50%) on MOSI→SIMS**; other methods even underperform the Source baseline.

### Ablation Study

1. **Diagonal simplification in PMGFA is effective**: The diagonal form (BriMPR) outperforms KL divergence (1–3% drop) and full moment matching (0.5–2% drop) across all tasks; the non-squared norm formulation also outperforms the squared norm.
2. **CMER weight design is justified**: Swapping the $\lambda_u$ weights leads to a notable performance drop (e.g., K50-C audio corruption 72.0→70.0; VGGSound-C audio corruption 36.5→32.1, a 4.4-point decline), validating the design of assigning higher mask weights to the less-shifted modality.
3. **Progressive module stacking is effective**: (A) PMGFA → (B) +IICL → (C) +CMER yields monotonic improvement; IICL contributes marginally (+0.1–0.3), while CMER contributes more substantially on VGGSound-C audio corruption (+1.2).
4. **Prompts outperform LayerNorm**: Using the same PMGFA loss, optimizing prompts consistently outperforms optimizing LayerNorm parameters while requiring fewer parameters.
5. **Efficiency**: BriMPR inference time is 186.2s (VGGSound-C); learnable parameters total 0.169M, fewer than most baselines (0.218M).

## Highlights & Insights

1. **Clear problem formulation**: The difficulty of MMTTA is attributed to the coupled effect of shallow unimodal shift and cross-modal semantic misalignment; the proposed divide-and-conquer strategy is well-motivated.
2. **Solid theoretical support**: Theorem 1 proves that diagonal covariance estimation error is $d$ times lower than full covariance, providing a theoretical basis for the simplification.
3. **Elegant design**: The adaptive weight and temperature scaling in CMER dynamically adjust based on distributional discrepancy, avoiding manual tuning of additional hyperparameters.
4. **Comprehensive experiments**: Settings span unimodal/multimodal/mixed severity/continual/limited-data/real-world domain shift, with thorough ablation.
5. **Parameter-efficient**: Only 0.169M prompt parameters are optimized; the backbone model remains frozen.

## Limitations & Future Work

1. **Strong Gaussian assumption**: Modeling the feature distribution at each layer as a multivariate Gaussian may fail for complex multimodal distributions; real test data distributions may not approximate a Gaussian.
2. **Dependency on source-domain statistics**: Although only 32 unlabeled source samples are required, source-domain data must be collected prior to testing for pre-computing statistics, limiting applicability to purely source-free scenarios.
3. **Validation limited to audio-visual / text+audio-visual modalities**: Visual-geometric multimodal scenarios such as image–point cloud and RGB–depth have not been explored, raising concerns about generalizability.
4. **Low absolute accuracy under multimodal shift**: VGGSound-C bimodal corruption achieves only 20.7% and Kinetics50-C bimodal corruption only 40.9%, leaving a substantial gap from practical utility.
5. **Marginal contribution of IICL**: Ablation results show IICL contributes only +0.1–0.3, limiting its design value and introducing additional computational overhead from contrastive learning.
6. **Pseudo-label noise**: Although adaptive temperature scaling mitigates overconfidence, error accumulation due to poor pseudo-label quality in early adaptation stages is not sufficiently discussed.
7. **Continual adaptation requires additional domain detection**: BriMPR-continual relies on Z-score-based domain shift detection and prompt resetting, increasing engineering complexity.

## Related Work & Insights

| Method | Strategy | Optimization Objective | Pros & Cons |
|--------|----------|----------------------|-------------|
| **Tent/EATA/SAR** | Update BN/LN parameters | Entropy minimization | Unimodal TTA; cross-modal interaction not considered |
| **READ** | Update self-attention in fusion module | Confidence-aware loss | First MMTTA method; shallow feature correction absent |
| **ABPEM** | Align cross-attention with self-attention | Entropy principal components | Reduces gradient noise; degrades significantly on VGGSound-C video corruption (52.4 vs. Source 56.2) |
| **SuMi** | IQR smoothing + modality mutual information | Selective entropy minimization | Performance on par with Source under multimodal shift |
| **FOA** | CMA-ES learns prompts | Gradient-free optimization | More parameters (1.772M); degrades in some settings |
| **BriMPR** | Divide-and-conquer + prompt + masked recombination + contrastive learning | Distribution alignment + pseudo-label CE + InfoNCE | Best performance with fewest parameters (0.169M); stable across all settings |

**Inspirations**:
1. **Prompts as distribution calibrators**: Prompt tuning can serve not only as a downstream task adapter but also as an implicit mapping tool for feature distributions—a perspective transferable to other domain adaptation scenarios.
2. **Masked augmentation for cross-modal information transfer**: CMER forces weak modalities to learn independently by deliberately masking high-quality modality information, analogous to knowledge distillation under information-constrained conditions.
3. **Theoretical simplification via diagonal moment estimation**: In data-scarce settings such as TTA, simplifying statistical estimation to reduce error is a general technique applicable to other methods requiring online distribution estimation.
4. **Distributional discrepancy as a byproduct for domain detection**: $\text{Disc}_u$ is used simultaneously for loss computation, domain shift detection, and weight assignment—an efficient multi-purpose design.

## Rating ⭐⭐⭐⭐ (4/5)

**Strengths**: Clear problem formulation, solid theory and experiments, intuitive method design, parameter efficiency, and robustness across scenarios. The real-world domain shift experiments (MOSI→SIMS outperforming Source by 12 points) are particularly convincing.

**Deductions**: The IICL module contributes only marginally; the limitations of the Gaussian assumption are not discussed; absolute accuracy under multimodal shift remains low; and applicability is confined to audio-visual classification. Overall, this is a rigorous contribution to MMTTA that advances the research paradigm from "assigning higher weight to high-quality modalities" to "actively calibrating the distribution of each modality."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](panda_test-time_adaptation_with_negative_data_augmentation.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](../../CVPR2026/multimodal_vlm/decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[AAAI 2026\] Aligning the True Semantics: Constrained Decoupling and Distribution Sampling for Cross-Modal Alignment](aligning_the_true_semantics_constrained_decoupling_and_distr.md)
- [\[ICML 2026\] SLQ: Bridging Modalities via Shared Latent Queries for Retrieval with Frozen MLLMs](../../ICML2026/multimodal_vlm/slq_bridging_modalities_via_shared_latent_queries_for_retrieval_with_frozen_mllm.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)

</div>

<!-- RELATED:END -->
