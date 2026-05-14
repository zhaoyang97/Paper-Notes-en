---
title: >-
  [Paper Note] Angular Steering: Behavior Control via Rotation in Activation Space
description: >-
  [NeurIPS 2025][LLM Safety][activation steering] This paper proposes Angular Steering, which unifies LLM activation steering as rotation operations within a fixed 2D subspace. By parameterizing behavior control through ro…
tags:
  - "NeurIPS 2025"
  - "LLM Safety"
  - "activation steering"
  - "behavior control"
  - "rotation transform"
  - "refusal steering"
  - "norm preservation"
date: 2026-05-08
content_hash: d4b17846806865ee
---

# Angular Steering: Behavior Control via Rotation in Activation Space

**Conference**: NeurIPS 2025
**arXiv**: [2510.26243](https://arxiv.org/abs/2510.26243)
**Code**: [https://github.com/lone17/angular-steering/](https://github.com/lone17/angular-steering/)
**Area**: LLM Safety
**Keywords**: activation steering, behavior control, rotation transform, refusal steering, norm preservation

## TL;DR

This paper proposes Angular Steering, which unifies LLM activation steering as rotation operations within a fixed 2D subspace. By parameterizing behavior control through rotation angle, it provides a continuous, fine-grained, norm-preserving knob spanning 0°–360°, while unifying activation addition and directional ablation as special cases of rotation. The approach achieves robust behavior control on Llama 3, Qwen 2.5, and Gemma 2 (3B–14B).

## Background & Motivation

Activation steering modifies internal representations during LLM inference to control behavior. The core intuition is that features (e.g., "refusal" tendency) correspond to approximately orthogonal directions in activation space. Two dominant methods exist: **activation addition** ($\mathbf{h}' = \mathbf{h} + \alpha \hat{\mathbf{d}}_\text{feat}$) adjusts behavior by adding a scaled feature vector, but tuning the coefficient α is brittle and difficult; **directional ablation** ($\mathbf{h}' = \mathbf{h} - \hat{\mathbf{d}}_\text{feat} \hat{\mathbf{d}}_\text{feat}^\top \mathbf{h}$) removes the feature component entirely via orthogonal projection, but cannot perform partial suppression.

The authors' key insight stems from RMSNorm, which is ubiquitous in modern LLMs (LLaMA 3, Qwen 2.5, Gemma 2). RMSNorm maps activations onto a scaled unit sphere $\bar{\mathbf{h}} = \mathbf{h}/\text{RMS}(\mathbf{h}) \odot \mathbf{g}$, followed by directional scaling via a fixed vector $\mathbf{g}$. This implies that the **direction**, rather than the magnitude, of activations is the fundamental representational unit. **Rotation**—as the unique geometric transformation that both preserves norm and is continuously adjustable—is therefore the natural choice for behavior control.

Furthermore, the authors prove that applying vector addition or orthogonal projection to activations before normalization is mathematically equivalent to rotation after normalization. Specifically, activation addition is a special case of rotation by less than 180°, and directional ablation corresponds to exactly 90° rotation. Angular Steering unifies and generalizes both.

## Method

### Overall Architecture

After the normalization layer of each Transformer block, the activation vector is rotated to a specified angle within a predefined 2D subspace spanned by the feature direction $\hat{\mathbf{d}}_\text{feat}$ and its first principal component $\hat{\mathbf{d}}_\text{PC0}$. The rotation angle θ is the sole control parameter: θ near 0° yields strong refusal; 100° yields indirect response; 200° yields direct compliance; 300° yields redirection.

### Key Designs

1. **Angular Steering Core Operation**:
   - Function: Norm-preserving rotation within a fixed 2D subspace.
   - Mechanism: Given an orthonormal basis $\{\mathbf{b}_1, \mathbf{b}_2\}$ (obtained by orthogonalizing $\hat{\mathbf{d}}_\text{feat}$ and $\hat{\mathbf{d}}_\text{PC0}$), the 2D projection of the activation is rotated to the target angle θ while the remaining dimensions are unchanged: $\mathbf{h}_{\text{steered},\theta} = \mathbf{h} - \text{proj}_P(\mathbf{h}) + |\text{proj}_P(\mathbf{h})| \cdot [\mathbf{b}_1\ \mathbf{b}_2] R_\theta [1\ 0]^\top$
   - Implementation: The projection matrix and $[\mathbf{b}_1\ \mathbf{b}_2] R_\theta [1\ 0]^\top$ can be precomputed; inference requires only a single projection, scaling, and addition.
   - Design Motivation: Constraining rotation to 2D leaves the orthogonal complement entirely unaffected, minimizing interference with other features.

2. **Adaptive Angular Steering**:
   - Function: Applies rotation only to activations positively aligned with the target feature.
   - Mechanism: A conditional mask $\text{mask} = \max(0, \text{sign}(\text{proj}_{\hat{\mathbf{d}}_\text{feat}}(\mathbf{x})))$ is added, so rotation is applied only when the activation has a positive projection along the feature direction.
   - Design Motivation: Ablation experiments show that activations from harmful and harmless samples project in opposite directions along the feature axis. Rotating only positively aligned (harmful) activations further reduces interference with irrelevant features. This is especially important for small models (3B), where the non-adaptive variant produces incoherent outputs.

3. **Automatic Feature Direction Extraction**:
   - Function: Automatically identifies the optimal feature direction without manual layer or direction selection.
   - Mechanism: After each normalization layer, the mean activation difference between contrastive datasets (AdvBench harmful vs. Alpaca harmless) is computed as a candidate direction (totaling $M = 2 \times \text{num\_layers}$ candidates). The candidate with the highest average cosine similarity to all others is selected as the final direction, as high similarity indicates stable recurrence across layers and thus a reliable approximation of the true feature direction.
   - Design Motivation: Eliminates subjective and potentially suboptimal manual selection.

### Loss & Training

No training is required. The rotation operation is inserted directly after normalization layers at inference time. Feature direction extraction requires only a single forward pass over the contrastive dataset.

## Key Experimental Results

### Main Results

**Refusal Steering**: Evaluated on Llama 3 (3B/8B), Qwen 2.5 (3B/7B/14B), and Gemma 2 (9B). Sweeping θ in 10° increments reveals alternating refusal and compliance arcs:

| Angle Range | Behavior | Description |
|-------------|----------|-------------|
| ~0°–60° | Strong refusal | substring matching refusal score ≈ 1.0 |
| ~60°–120° | Indirect response | Beginning to weaken but still evasive |
| ~120°–240° | Direct compliance | refusal score ≈ 0, high harmful score |
| ~240°–360° | Redirection | Neither refuses nor answers directly; offers alternatives |

**TinyBenchmarks Performance Preservation**:

| Configuration | ARC | MMLU | WinoGrande | GSM8k | Overall Trend |
|---------------|-----|------|------------|-------|---------------|
| No steering | baseline | baseline | baseline | baseline | — |
| Angular (full circle) | **Nearly unchanged** | **Nearly unchanged** | **Nearly unchanged** | Slight variation | Baseline maintained at most angles |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| Random plane rotation | Ineffective | 5/6 models show almost no behavioral change |
| Adaptive vs. Non-Adaptive | Adaptive more stable | Largest difference on small models (3B); non-adaptive causes incoherence |
| Perplexity at different angles | Higher PPL in refusal region than baseline | Alignment ≈ surface-level overlay rather than removal of underlying capability |

### Key Findings

- **Safety alignment is primarily superficial**: The perplexity of harmful generations is lower than that of refusal generations, indicating that the model still "knows" harmful content at the base level; alignment shifts only the distribution of early tokens rather than truly removing harmful knowledge.
- Small models (3B) are more susceptible to feature interference—multiple features become entangled within the 2D subspace during rotation, producing incoherent outputs. The adaptive variant effectively mitigates this.
- Gemma-2-9B exhibits the weakest steering effect, possibly due to architectural differences.
- Feature directions show high inter-layer consistency (high cosine similarity), supporting the hypothesis that feature directions are stable across layers.

## Highlights & Insights

- **The unified geometric perspective is exceptionally elegant**: The observation that "addition = partial rotation, ablation = 90° rotation" is formalized as a rigorous mathematical equivalence, unifying disparate activation engineering techniques under a single framework.
- The core insight that **direction > magnitude** exploits the geometric properties of RMSNorm—normalization effectively confines the model's representations to the unit sphere, discarding magnitude information.
- The method provides a continuous spectrum of four behavior types (refusal → indirect → direct → redirect), transforming behavior control from a binary switch into a continuously adjustable knob.
- The Adaptive variant is elegantly minimalist—a single sign mask substantially improves stability on small models.

## Limitations & Future Work

- Feature direction extraction relies on a contrastive dataset; different features (e.g., factuality, creativity) require different contrastive pairs.
- The steering plane is selected heuristically (top-2 PCA components), with no guarantee of optimality across all behaviors and architectures.
- Only refusal and sentiment behaviors are validated; simultaneous steering of multiple features may cause subspace conflicts.
- The 2D subspace assumption may be an oversimplification in high-dimensional space—complex behaviors may require higher-dimensional steering subspaces.
- The method depends on the linear feature assumption (Superposition Hypothesis); it may fail if features are encoded nonlinearly.

## Related Work & Insights

- **vs. Activation Addition (ActAdd)**: ActAdd is equivalent to rotation by less than 180°. The difficulty of tuning α stems from the fact that it simultaneously changes both rotation angle and magnitude—Angular Steering decouples the two by controlling angle alone.
- **vs. Directional Ablation (RepE)**: Directional ablation corresponds to exactly 90° rotation. However, 90° may not be optimal, and ablation cannot leverage information from negative projections.
- **vs. Spectral Editing of Activations**: Directions are constructed in PCA space; Angular Steering further defines a rotation operation within that space.
- **vs. Householder Pseudo-Rotation**: Shares a similar norm-preserving motivation, but Householder is limited to reflections and is less flexible than rotations.
- **Insight**: The geometric properties of RMSNorm suggest that modern LLMs may be inherently suited to rotation-based operations—this warrants further exploration in alignment, safety, and style control settings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The unified geometric framework is highly elegant, subsuming disparate techniques as special cases of rotation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three LLM families × multiple scales + 6 benchmarks + qualitative/quantitative analysis + ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Geometric intuition, theoretical proofs, and experimental validation are seamlessly integrated.
- Value: ⭐⭐⭐⭐⭐ — A significant methodological contribution to the LLM safety and controllability community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Compiling Activation Steering into Weights via Null-Space Constraints for Stealthy Backdoors](../../ACL2026/llm_safety/compiling_activation_steering_into_weights_via_null-space_constraints_for_stealt.md)
- [\[NeurIPS 2025\] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders](saemark_steering_personalized_multilingual_llm_watermarks_with_sparse_autoencode.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](simu_selective_influence_machine_unlearning.md)
- [\[NeurIPS 2025\] MixAT: Combining Continuous and Discrete Adversarial Training for LLMs](mixat_combining_continuous_and_discrete_adversarial_training_for_llms.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)

</div>

<!-- RELATED:END -->
