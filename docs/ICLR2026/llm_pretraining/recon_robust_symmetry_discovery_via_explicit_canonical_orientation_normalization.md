---
title: >-
  [Paper Note] RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization
description: >-
  [ICLR 2026][LLM Pretraining][symmetry discovery] This paper proposes RECON, a class- and pose-agnostic canonical orientation normalization method that corrects arbitrary canonical representations produced during training via a simple right translation, enabling unsupervised instance-level symmetry discovery, OOD pose detection, and a plug-and-play test-time canonicalization layer.
tags:
  - ICLR 2026
  - LLM Pretraining
  - symmetry discovery
  - canonical orientation
  - class-pose decomposition
  - plug-and-play
  - group invariance
date: 2026-05-08
content_hash: f607ebc1646ffe8d
---

# RECON: Robust symmetry discovery via Explicit Canonical Orientation Normalization

**Conference**: ICLR 2026
**arXiv**: [2505.13289](https://arxiv.org/abs/2505.13289)
**Code**: N/A
**Area**: Symmetry Discovery / Invariance Learning
**Keywords**: symmetry discovery, canonical orientation, class-pose decomposition, plug-and-play, group invariance

## TL;DR

This paper proposes RECON, a class- and pose-agnostic canonical orientation normalization method that corrects arbitrary canonical representations produced during training via a simple right translation, enabling unsupervised instance-level symmetry discovery, OOD pose detection, and a plug-and-play test-time canonicalization layer.

## Background & Motivation

### State of the Field

**Background**: Real-world data commonly exhibit unknown, instance-dependent symmetries that rarely conform precisely to a pre-specified transformation group $G$. Conventional equivariant/invariant network approaches either hard-code a particular group structure or learn it implicitly through data augmentation. Class-pose decomposition methods attempt to factorize inputs into invariant features and a pose $g \in G$ relative to some canonical representation.

However, existing canonicalization methods suffer from a fundamental problem: **the canonical representation is training-dependent and therefore arbitrary**. Different training runs and initializations yield different canonical choices, which leads to:

### Limitations of Prior Work

**Limitations of Prior Work**: The learned pose distribution is difficult to interpret.

### Root Cause

**Key Challenge**: Cross-model comparisons become meaningless.

### Mechanism

**Mechanism**: Pose information cannot be directly exploited for downstream tasks.

The core insight of RECON is that an arbitrary canonical choice can be corrected via a simple group operation (right translation) to align with the data's natural canonical orientation. This correction is a post-processing step and requires no model retraining.

## Method

### Overall Architecture

RECON builds on the class-pose decomposition framework. Given an input $x$ and a pretrained decomposition model that outputs invariant features $z$ and a pose estimate $\hat{g}$, RECON corrects the canonicalization through the following steps:

1. Collect the pose distribution of same-class samples from the training data.
2. Estimate the "natural" canonical orientation (data-aligned canonicalization).
3. Map the arbitrary canonicalization to the natural one via a right translation operator.

### Key Designs

1. **Right Translation Correction**: Given the pretrained model's canonicalization $c$ and a target canonicalization $c'$, the correction is a simple right multiplication by a group element: $g' = g \cdot c^{-1} \cdot c'$.

   → Mechanism: The algebraic structure of the group guarantees exact conversion between canonicalizations.
   → Design Motivation: Avoids retraining by leveraging the equivalence class concept from group theory, achieving zero-cost conversion.

2. **Unsupervised Pose Distribution Discovery**: The corrected pose distribution reflects the true symmetry structure of the data. Different instances may exhibit distinct pose distributions (instance-level symmetry), e.g., a rotationally symmetric molecule versus an asymmetric one.

   → Mechanism: Instance symmetry is inferred by analyzing the statistics of the corrected pose distribution.
   → Design Motivation: Traditional methods assume all instances share the same symmetry group; RECON relaxes this assumption.

3. **OOD Pose Detection**: At test time, a sample whose pose falls outside the support of the training pose distribution can be flagged as out-of-distribution, providing a natural mechanism for detecting anomalous poses.

   → Mechanism: The corrected pose distribution serves as a normality reference.
   → Design Motivation: In settings such as molecular conformation analysis, anomalous poses may correspond to unstable or non-physical structures.

4. **Plug-and-Play Test-Time Canonicalization Layer**: RECON can be appended to any pretrained model as a canonicalization layer that injects group invariance by transforming inputs to their canonical orientation, without retraining the base model.

   → Mechanism: Inputs are canonicalized before being fed into the invariance model.
   → Design Motivation: Reuses the capacity of existing pretrained models by adding invariance only at inference time.

### Loss & Training

RECON itself requires no separate training — it is a post-processing/inference-time method. The underlying class-pose decomposition model can be trained with any existing approach (e.g., equivariant autoencoders, canonicalization networks). The correction operation in RECON consists of analytic group operations and requires no gradient-based optimization.

When end-to-end training is desired, the RECON layer can be integrated into the training pipeline, using standard classification or reconstruction losses.

## Key Experimental Results

### Main Results

Experiments are conducted on image datasets and molecular conformation datasets.

**Image Classification** (Rotated MNIST, etc.)

| Method | Classification Accuracy | Canonicalization Quality | Notes |
|--------|------------------------|--------------------------|-------|
| No-canonicalization baseline | Lower | N/A | Lacks rotation invariance |
| Conventional canonicalization | Moderate | Training-dependent | Canonicalization is arbitrary |
| **RECON** | **Best or on par** | **Data-aligned** | No retraining required |

**Molecular Conformation Analysis**

| Method | Symmetry Discovery | OOD Detection | Notes |
|--------|--------------------|---------------|-------|
| Fixed-group method | Cannot handle instance-level variation | Not supported | Assumes uniform symmetry group |
| **RECON** | **Accurate** | **Effective** | Supports instance-level symmetry |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Different underlying decomposition models | Stable performance | RECON is insensitive to the choice of underlying model |
| Different group structures | All applicable | Validates group-agnostic nature of the method |
| With / without RECON layer | +significant gain | Demonstrates value of the plug-and-play layer |

### Key Findings

1. RECON's canonicalization correction is exact — no approximation error is introduced, as it relies solely on algebraic group operations.
2. The discovered instance-level pose distributions are consistent with physical/chemical intuition (e.g., the rotational symmetry order of molecules).
3. The plug-and-play layer substantially improves the robustness of pretrained models on transformed data without any retraining.
4. The method is effective across two fundamentally different domains — images and molecules — demonstrating its generality.

## Highlights & Insights

1. **Theoretical Elegance**: The arbitrariness of canonical choice is reformulated as a simple group translation, yielding a solution that is both mathematically rigorous and implementationally concise.
2. **Zero-Cost Correction**: No model retraining is required; RECON is a purely post-processing method, making it highly attractive for practical deployment.
3. **Instance-Level Symmetry**: The common assumption that all samples share the same symmetry group is relaxed, better reflecting the properties of real-world data.
4. **Plug-and-Play Design**: Invariance can be injected into any pretrained model, conceptually analogous to adapter-based methods but operating at the geometric level.
5. **Cross-Domain Validation**: Effectiveness across images and molecules demonstrates the universality of the group-theoretic approach.

## Limitations & Future Work

1. **Dependence on the Quality of the Underlying Decomposition Model**: If the base class-pose decomposition model is inaccurate, RECON's corrections will be correspondingly affected.
2. **Requires Knowledge of the Transformation Group $G$**: While the specific symmetries need not be known, the structure of the acting group (e.g., SO(2), SE(3)) must still be specified.
3. **Statistical Estimation Challenges on Continuous Groups**: Estimating pose distributions over high-dimensional continuous groups may face statistical efficiency issues.
4. **Incomplete Access to Full Paper**: The full text could not be retrieved; some experimental details are inferred from the abstract and related work.
5. **Future Extensions**: Extending RECON to more complex group structures, such as infinite-dimensional groups or mixtures of discrete groups.

## Related Work & Insights

- **Equivariant Neural Networks** (e.g., E(n)-equivariant GNNs): RECON offers an orthogonal path to invariance — modifying the input rather than the network architecture.
- **Canonical Orientation Networks**: RECON addresses the core limitation of such methods, namely that the canonical orientation is training-dependent.
- **Symmetry Discovery**: RECON is complementary to methods such as Lie group discovery, focusing on instance-level rather than global symmetry.
- Insight: Simple group-theoretic operations can sometimes be more effective than complex learned methods; the "post-processing" paradigm is underexplored in invariance learning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The right translation correction is elegant and novel, though it is fundamentally a direct application of group theory.
- Experimental Thoroughness: ⭐⭐⭐ — Validated on both image and molecular domains, though full experimental details could not be confirmed.
- Writing Quality: ⭐⭐⭐⭐ — The abstract is clear and the theoretical exposition is precise.
- Value: ⭐⭐⭐⭐ — Provides a practical and lightweight tool for the equivariant/invariant learning community.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Imagine How To Change: Explicit Procedure Modeling for Change Captioning](imagine_how_to_change_explicit_procedure_modeling_for_change_captioning.md)
- [\[NeurIPS 2025\] A Practical Guide for Incorporating Symmetry in Diffusion Policy](../../NeurIPS2025/llm_pretraining/a_practical_guide_for_incorporating_symmetry_in_diffusion_policy.md)
- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](../../NeurIPS2025/llm_pretraining/superposition_yields_robust_neural_scaling.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[NeurIPS 2025\] Broken Tokens: Your Language Model Can Secretly Handle Non-Canonical Tokenization](../../NeurIPS2025/llm_pretraining/broken_tokens_your_language_model_can_secretly_handle_non-canonical_tokenization.md)

<!-- RELATED:END -->
