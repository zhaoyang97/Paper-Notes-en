---
title: >-
  [Paper Note] On the Value of Cross-Modal Misalignment in Multimodal Representation Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][multimodal contrastive learning] This paper proposes a latent variable model that formalizes cross-modal misalignment into two mechanisms—selection bias and perturbation bias—and theoretically proves that MMCL-learned representations precisely capture the invariant semantic subset unaffected by both biases, thereby unifying the opposing views of misalignment as harmful vs. beneficial.
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "multimodal contrastive learning"
  - "cross-modal misalignment"
  - "latent variable model"
  - "identifiability"
  - "invariant representation learning"
date: 2026-05-08
content_hash: 3f9fb37a16655dcc
---

# On the Value of Cross-Modal Misalignment in Multimodal Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2504.10143](https://arxiv.org/abs/2504.10143)  
**Code**: [Project Page](https://yichaocai.com/misalignment.github.io)  
**Area**: Multimodal VLM
**Keywords**: multimodal contrastive learning, cross-modal misalignment, latent variable model, identifiability, invariant representation learning

## TL;DR

This paper proposes a latent variable model that formalizes cross-modal misalignment into two mechanisms—selection bias and perturbation bias—and theoretically proves that MMCL-learned representations precisely capture the invariant semantic subset unaffected by both biases, thereby unifying the opposing views of misalignment as harmful vs. beneficial.

## Background & Motivation

### 1. State of the Field

Multimodal contrastive learning (MMCL), exemplified by CLIP, learns joint representations through image-text alignment and has achieved remarkable success in zero-shot classification, cross-modal retrieval, and related tasks. Its core assumption is that paired image-text inputs are semantically fully consistent.

### 2. Limitations of Prior Work

Real-world datasets exhibit pervasive **cross-modal misalignment**. Studies show that over 50% of pairs in large-scale video-text datasets contain semantic inconsistencies. Textual descriptions of images are naturally prone to semantic incompleteness or the introduction of erroneous information.

### 3. Root Cause

Two opposing perspectives on misalignment exist in the literature:
- **Mitigation camp**: misalignment is a noise source that induces "hallucinations" in multimodal models and should be eliminated (e.g., filtering strategies in SigLIP, BLIP).
- **Exploitation camp**: deliberately introducing misalignment on style-related information (e.g., random text augmentation) can enhance zero-shot and adversarial robustness.

### 4. Paper Goals

How can these two opposing views be theoretically reconciled? Under what conditions is misalignment harmful, and under what conditions is it beneficial? The paper aims to provide actionable guidance for practical applications.

### 5. Starting Point

The paper constructs a latent variable generative model (LVM) that explicitly models cross-modal misalignment, conducts an identifiability analysis of the MMCL framework, and derives a unified theory from a causal inference perspective.

### 6. Core Idea

The representations learned by MMCL are precisely block-identifiable transformations of the semantic subset that is jointly shared across modalities and unaffected by selection bias and perturbation bias. Misalignment naturally filters out unstable semantics, thereby acting as a regularizer.

## Method

### Overall Architecture

The paper proposes a generative model with three classes of latent variables $\mathcal{Z} = \mathcal{S} \times \mathcal{M}_x \times \mathcal{M}_t$:
- **Semantic variables** $\mathbf{s} \in \mathbb{R}^{n_s}$: describe semantic content shared between image and text (object shape, color, etc.)
- **Image-specific variables** $\mathbf{m}_x$: non-semantic factors such as camera noise and background
- **Text-specific variables** $\mathbf{m}_t$: non-semantic factors such as writing style and tone

A key innovation is that the model permits **arbitrary causal structure** among semantic variables $\mathbf{s}$, unlike prior work that requires independence or a fixed graph structure.

### Key Designs

#### Module 1: Selection Bias $\theta$

**Function**: Determines which semantic information is retained in the text.

**Mechanism**: Selection bias $\theta$ maps to a non-empty semantic subset $\mathbb{I}_\theta \in \mathcal{P}^+(\mathbb{I}_\mathbf{s})$; semantic variables in the complement $\mathbb{I}_\theta^c$ are entirely ignored in the text (e.g., texture information is omitted when describing an object).

**Design Motivation**: Real textual descriptions naturally cover only a portion of an image's semantics; selection bias precisely characterizes this information loss.

#### Module 2: Perturbation Bias $\rho$

**Function**: Introduces erroneous annotations within the selected semantic subset.

**Mechanism**: A perturbable subset $\mathbb{I}_\rho \subseteq \mathbb{I}_\theta$ is defined; a random subset $A \subseteq \mathbb{I}_\rho$ is sampled, and the semantic variables therein are replaced with random values:
$$p_{\tilde{\mathbf{s}}_{\mathbb{I}_\theta} | \mathbf{s}, A} = \delta(\tilde{\mathbf{s}}_{\mathbb{I}_\theta \setminus A} - \mathbf{s}_{\mathbb{I}_\theta \setminus A}) \cdot p_{\tilde{\mathbf{s}}_A | \mathbf{s}_A}$$

**Design Motivation**: This simulates labeling errors (e.g., mislabeling "black" as "red"). A key insight is that such perturbations, unlike causal interventions, do not propagate to downstream variables, because they operate at the observation level rather than the latent variable level.

#### Module 3: Generative Process

Image generation: $\mathbf{x} = g_x(\mathbf{s}, \mathbf{m}_x)$, where $g_x$ is a diffeomorphic mapping.
Text generation: $\mathbf{t}^{(\theta)} = g_{t^{(\theta)}}(\tilde{\mathbf{s}}_{\mathbb{I}_\theta}, \mathbf{m}_t)$, where $g_{t^{(\theta)}}$ is likewise diffeomorphic.

### Core Theoretical Results

**Theorem 4.1 (Identifiability of Semantic Variables under Misalignment)**: Under mild assumptions, minimizing the $\mathcal{L}_{\text{SymAlignMaxEnt}}$ objective causes encoders $f_x, f_t$ to **block-identify** the unbiased semantic sub-variables $\mathbf{s}_{\mathbb{I}_\rho^c}$. That is, the learned representations are an invertible transformation of $\mathbf{s}_{\mathbb{I}_\rho^c}$, with dimensionality $n = |\mathbb{I}_\theta| - |\mathbb{I}_\rho|$.

**Corollary 4.1** (Perfect Alignment → Full Semantic Recovery): When $\theta = 2^{n_s} - 1$ and $\rho = 1$ (no bias), MMCL recovers all $n_s$ semantic variables.

**Corollary 4.2** (Targeted Misalignment → Invariant Representations): When $\mathbb{I}_{var} = \mathbb{I}_\theta^c \cup \mathbb{I}_\rho$ exactly equals the semantic subset sensitive to distribution shift, MMCL automatically recovers the invariant semantics $\mathbf{s}_{\mathbb{I}_{inv}}$ that are robust to distribution shift.

### Loss & Training

The asymptotic MMCL objective is used for theoretical analysis:
$$\mathcal{L}_{\text{SymAlignMaxEnt}} = \mathbb{E}[\|f_x(\mathbf{x}) - f_t(\mathbf{t})\|_2] - \frac{1}{2}(H(f_x(\mathbf{x})) + H(f_t(\mathbf{t})))$$
The first term minimizes the distance between paired samples (alignment); the second term maximizes representation entropy (uniformity).

## Key Experimental Results

### Main Results

**Numerical Simulation** (10-dimensional semantic variables + 5-dimensional modality-specific variables):

| Setting | Retained Semantics $|\mathbb{I}_\theta|$ | Unbiased Semantics R² | Biased Semantics R² | Modality-Specific R² |
|---------|------|------|------|------|
| Independent latent variables | Varies 1→10 | ≈1.0 | ≈0.0 | ≈0.0 |
| Dependent latent variables | Varies 1→10 | ≈1.0 | Partially predictable | ≈0.0 |

**MPI3D-Complex Real Dataset** (7 independent discrete factors):

| Bias Type | Setting | Retained Factor MCC | Missing Factor MCC |
|-----------|---------|---------------------|---------------------|
| Selection | 1 factor → 5 factors | ≥0.8 → ≈1.0 | =0.0 |
| Perturbation | 0 → 4 factors perturbed | ≈1.0 | =0.0 |

**Causal3DIdent** (10-dimensional latent variables with causal graph structure):

| Setting | shape (MCC) | x_pos (MCC) | color (R²) | s_pos (R²) |
|---------|-------------|-------------|------------|------------|
| Full selection + no perturbation | ≈1.0 | ≈1.0 | ≈1.0 | ≈1.0 |
| Shape only | ≈1.0 | =0 | =0 | =0 |

### Ablation Study

**Downstream Tasks (Numerical Simulation)**:

| Encoding Dimension | ID Regression R² | ID Classification Acc | OOD Classification Acc |
|--------------------|------------------|-----------------------|------------------------|
| Retain all 10 semantic dims | ≈1.0 | ≈1.0 | Decreases |
| Remove distribution-sensitive dims | Decreases | Decreases | **Best** |

### OpenCLIP Case Study

Zero-shot evaluation of OpenCLIP pretrained on LAION-400M across 146 visual concepts:

| Concept Group | Coverage | F1 Score |
|---------------|----------|----------|
| Animal | High (>1%) | High |
| Object | High (1.63%) | High |
| Texture | Low (0.07%) | Low |
| Emotion | Low (0.04%) | Low |
| Stereotype | Extremely low (0.0003%) | Extremely low |

### Key Findings

1. Theoretical predictions align closely with experiments: unbiased semantics yield $R^2 \approx 1$, biased semantics yield $R^2 \approx 0$.
2. In settings with causally dependent latent variables, some biased semantics are partially predictable due to statistical dependencies, though to a limited extent.
3. Modality-specific variables are consistently excluded across all settings.
4. Targeted misalignment enhances OOD robustness, validating Corollary 4.2.
5. OpenCLIP systematically fails on low-coverage concepts, validating the selection bias theory.

## Highlights & Insights

1. **Unified perspective**: The paper is the first to theoretically reconcile the opposing views of misalignment as harmful vs. beneficial using identifiability theory, providing a clear characterization of the conditions governing each outcome.
2. **Flexible latent variable model**: By permitting arbitrary causal structure among semantic variables, the framework is more general than prior work requiring independence or fixed graph structures.
3. **Practical insights**:
    - Large-scale pretraining requires comprehensive and consistent annotations to preserve all semantics.
    - In OOD settings, invariant representation learning can be naturally achieved by controlling textual misalignment.
    - Auditing and curating text is more precise and controllable than manipulating latent variables.
4. **Perturbation vs. intervention**: The paper distinguishes observation-level textual perturbations from causal interventions; the latter propagate along the causal graph whereas the former do not.

## Limitations & Future Work

1. The theory is based on the asymptotic MMCL objective (SymAlignMaxEnt); a gap remains between this and the finite-sample InfoNCE objective used in practice.
2. The assumption that generative functions are diffeomorphisms may not be strictly satisfied in real image/text generation.
3. Only block identifiability is analyzed; precise conditions for component-level identifiability are not established.
4. The causal relationship between concept coverage and F1 score in the OpenCLIP experiments requires more rigorous validation.
5. How to precisely control selection/perturbation bias in text for practical applications remains an open question.

## Related Work & Insights

- **Relation to Daunhawer et al. (ICLR 2023)**: That work proves MMCL can identify semantic variables under the assumption of perfect alignment; this paper extends the result to the misalignment setting.
- **Connection to invariant representation learning**: Corollary 4.2 demonstrates that MMCL with targeted misalignment constitutes a novel pathway to achieving invariant learning analogous to IRM/REx.
- **Connection to text augmentation**: Works such as CLAP that improve robustness via random text augmentation can be theoretically explained by this framework as introducing perturbation bias.

## Rating

⭐⭐⭐⭐⭐ (5/5)

The theoretical contributions are rigorous, unifying two opposing views on cross-modal misalignment. Experimental validation spans from synthetic data to real CLIP models, combining theoretical depth with practical significance. This represents an important contribution to the theoretical analysis of multimodal representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)
- [\[ICLR 2026\] DecAlign: Hierarchical Cross-Modal Alignment for Decoupled Multimodal Representation Learning](../../ICLR2026/multimodal_vlm/decalign_hierarchical_cross-modal_alignment_for_decoupled_multimodal_representat.md)
- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](continual_multimodal_contrastive_learning.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)

</div>

<!-- RELATED:END -->
