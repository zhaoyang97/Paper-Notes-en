---
title: >-
  [Paper Note] Angular Steering: Behavior Control via Rotation in Activation Space
description: >-
  [NeurIPS 2025][Signal & Communication][activation steering] This paper proposes Angular Steering, which unifies LLM activation steering as rotation operations within a fixed 2D subspace — providing a continuous…
tags:
  - "NeurIPS 2025"
  - "Signal & Communication"
  - "activation steering"
  - "behavior control"
  - "rotation transformation"
  - "refusal steering"
  - "norm preservation"
date: 2026-05-08
content_hash: 7b7682d553f39577
---

# Angular Steering: Behavior Control via Rotation in Activation Space

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.26243](https://arxiv.org/abs/2510.26243)  
**Code**: [https://github.com/lone17/angular-steering/](https://github.com/lone17/angular-steering/)  
**Area**: Signal Communication
**Keywords**: activation steering, behavior control, rotation transformation, refusal steering, norm preservation

## TL;DR

This paper proposes Angular Steering, which unifies LLM activation steering as rotation operations within a fixed 2D subspace — providing a continuous, fine-grained, norm-preserving behavior control knob spanning 0°–360° via rotation angle. The framework subsumes activation addition and directional ablation as special cases of rotation, and demonstrates robust behavior control on Llama 3 / Qwen 2.5 / Gemma 2 (3B–14B).

## Background & Motivation

Activation steering modifies internal representations during LLM inference to control behavior. The core intuition is that features in language models (e.g., a "refusal" tendency) correspond to approximately orthogonal directions in activation space. Two dominant approaches exist: **activation addition** ($\mathbf{h}' = \mathbf{h} + \alpha \hat{\mathbf{d}}_\text{feat}$) adjusts behavior by adding a scaled feature vector, but tuning the coefficient α is brittle and difficult; **directional ablation** ($\mathbf{h}' = \mathbf{h} - \hat{\mathbf{d}}_\text{feat} \hat{\mathbf{d}}_\text{feat}^\top \mathbf{h}$) removes the component along a feature direction via orthogonal projection, but cannot perform partial suppression.

The authors' key insight stems from RMSNorm, which is widely adopted in modern LLMs (LLaMA 3, Qwen 2.5, Gemma 2). RMSNorm maps activations onto a scaled unit sphere via $\bar{\mathbf{h}} = \mathbf{h}/\text{RMS}(\mathbf{h}) \odot \mathbf{g}$, followed by directional scaling through the fixed vector $\mathbf{g}$. This implies that the **direction**, not the magnitude, of activations constitutes the core representational unit. Consequently, **rotation** — as the only geometric transformation that simultaneously preserves norm and admits continuous adjustment — is the natural choice for behavior control.

Furthermore, the authors prove that applying vector addition or orthogonal projection to activations before normalization is mathematically equivalent to applying rotation after normalization. That is, activation addition is a special case of rotation by less than 180°, and directional ablation is a special case of exactly 90° rotation. Angular Steering unifies and generalizes both.

## Method

### Overall Architecture

After each Transformer block's normalization layer, the activation vector is rotated to a specified angle within a predefined 2D subspace. The 2D subspace is defined by the feature direction $\hat{\mathbf{d}}_\text{feat}$ and its first principal component $\hat{\mathbf{d}}_\text{PC0}$. The rotation angle θ is the sole control parameter: θ near 0° = strong refusal, 100° = indirect response, 200° = direct compliance, 300° = redirection.

### Key Designs

1. **Core Angular Steering Operation**:
    - Function: Norm-preserving rotation within a fixed 2D subspace
    - Mechanism: Given an orthonormal basis $\{\mathbf{b}_1, \mathbf{b}_2\}$ (obtained by orthogonalizing $\hat{\mathbf{d}}_\text{feat}$ and $\hat{\mathbf{d}}_\text{PC0}$), the 2D projection of the activation is rotated to target angle θ while leaving all other dimensions unchanged: $\mathbf{h}_{\text{steered},\theta} = \mathbf{h} - \text{proj}_P(\mathbf{h}) + |\text{proj}_P(\mathbf{h})| \cdot [\mathbf{b}_1\ \mathbf{b}_2] R_\theta [1\ 0]^\top$
    - Implementation: The projection matrix and $[\mathbf{b}_1\ \mathbf{b}_2] R_\theta [1\ 0]^\top$ can be precomputed; inference requires only one projection + scaling + addition
    - Design Motivation: Rotation is confined to the 2D subspace, leaving the orthogonal complement entirely unaffected — minimizing interference with other features

2. **Adaptive Angular Steering**:
    - Function: Apply rotation only to activations positively aligned with the target feature
    - Mechanism: A conditional mask $\text{mask} = \max(0, \text{sign}(\text{proj}_{\hat{\mathbf{d}}_\text{feat}}(\mathbf{x})))$ is introduced, so rotation is applied only when the activation has a positive projection onto the feature direction
    - Design Motivation: Ablation experiments show that activations from harmful and harmless samples project in opposite directions onto the feature axis. Rotating only positively aligned (harmful) activations further reduces interference with unrelated features, and is especially important for small models (3B) — the non-adaptive variant causes incoherent outputs on small models

3. **Automatic Feature Direction Extraction**:
    - Function: Automatically determine the optimal feature direction without manual layer or direction selection
    - Mechanism: Mean-difference activations between contrastive datasets (AdvBench harmful vs. Alpaca harmless) are extracted after each normalization layer as candidate directions (M = 2 × number of layers candidates total). The candidate with the highest average cosine similarity to all others is selected as the final direction — high similarity indicates stable presence across layers and a reliable approximation to the true feature direction
    - Design Motivation: Avoids the subjectivity and potential suboptimality of manual selection

### Loss & Training

No training is required. The rotation operation is inserted after normalization layers directly at inference time. Feature direction extraction requires only a single forward pass over the contrastive dataset.

## Key Experimental Results

### Main Results

**Refusal Steering**:

Evaluated on Llama 3 (3B/8B), Qwen 2.5 (3B/7B/14B), and Gemma 2 (9B). Sweeping θ every 10° over a full circle reveals clear alternating refusal and compliance arcs:

| Angle Range | Behavior | Notes |
|-------------|----------|-------|
| ~0°–60° | Strong refusal | Substring matching refusal score ≈ 1.0 |
| ~60°–120° | Indirect response | Beginning to loosen but still evasive |
| ~120°–240° | Direct compliance | Refusal score ≈ 0, high harmful score |
| ~240°–360° | Redirection | Neither refuses nor answers directly; offers alternatives |

**TinyBenchmarks Performance Retention**:

| Configuration | ARC | MMLU | WinoGrande | GSM8k | Overall Trend |
|---------------|-----|------|------------|-------|---------------|
| No steering | baseline | baseline | baseline | baseline | — |
| Angular (full circle) | **nearly unchanged** | **nearly unchanged** | **nearly unchanged** | slight fluctuation | Baseline maintained at most angles |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| Random plane rotation | Ineffective | 5/6 models show almost no behavior change |
| Adaptive vs. non-Adaptive | Adaptive more stable | Largest difference on small models (3B) — non-Adaptive causes incoherence |
| Perplexity across angles | PPL higher in refusal region than baseline | Alignment ≈ a surface overlay on the distribution, not true removal of underlying capabilities |

### Key Findings

- **Safety alignment is largely superficial**: Perplexity of harmful generations is lower than that of refusal generations, indicating the model retains harmful knowledge at the base level — alignment merely shifts the distribution of the first few tokens rather than truly removing harmful knowledge
- Small models (3B) are more susceptible to feature interference — multiple features become entangled in the 2D subspace during rotation, causing incoherent outputs; the Adaptive variant effectively mitigates this
- Gemma-2-9B exhibits the weakest steering effect, likely due to its architectural differences
- Feature directions show high cross-layer consistency (high cosine similarity), supporting the hypothesis that feature directions are stable across layers

## Highlights & Insights

- **The unified geometric perspective is highly elegant**: The observation that "addition = partial rotation, ablation = 90° rotation" is established as a rigorous mathematical equivalence, unifying scattered activation engineering techniques under a single framework
- The core insight that **direction > magnitude** leverages the geometric properties of RMSNorm — normalization effectively places model computations on a unit sphere, discarding magnitude information
- The framework provides a continuous spectrum across four behavioral modes (refusal → indirect → direct → redirect), transforming behavior control from a binary switch into a continuously adjustable knob
- The Adaptive variant's design is concise yet effective — a single sign mask substantially improves stability on small models

## Limitations & Future Work

- Relies on contrastive datasets to extract feature directions; different features (e.g., factuality, creativity) require different contrastive data
- The steering plane is selected heuristically (top-two PCA components), with no guarantee of optimality across all behaviors and architectures
- Only refusal and sentiment behaviors are validated; simultaneously steering multiple features may produce subspace conflicts
- The 2D subspace assumption may be overly simplistic in high-dimensional space — certain complex behaviors may require higher-dimensional steering spaces
- Depends on the linear feature assumption (Superposition Hypothesis); the method may fail if features are encoded nonlinearly

## Related Work & Insights

- **vs. Activation Addition (ActAdd)**: ActAdd is equivalent to rotation by less than 180°; the difficulty of choosing α stems from the fact that it simultaneously controls both rotation angle and magnitude — Angular Steering decouples the two by adjusting angle alone
- **vs. Directional Ablation (RepE)**: Directional ablation is equivalent to exactly 90° rotation. However, 90° may not be optimal, and ablation cannot exploit information from negative projections
- **vs. Spectral Editing of Activations**: Directions are constructed in PCA space; Angular Steering further defines a rotation operation over this space
- **vs. Householder Pseudo-Rotation**: A similar norm-preserving idea, but Householder transformations are limited to reflections and are less flexible than rotations
- **Insight**: The geometric properties of RMSNorm suggest that modern LLMs may be inherently well-suited to rotation operations — this warrants exploration across alignment, safety, style control, and related settings

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The unified geometric framework is highly elegant, subsuming disparate techniques as special cases of rotation
- Experimental Thoroughness: ⭐⭐⭐⭐ Three LLM families × multiple scales + 6 benchmarks + qualitative/quantitative analysis + ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Geometric intuition, theoretical proofs, and empirical validation are smoothly integrated
- Value: ⭐⭐⭐⭐⭐ A significant methodological contribution to the LLM safety and controllability community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](../../ICLR2026/signal_comm/enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[ICLR 2026\] Mamba-3: Improved Sequence Modeling using State Space Principles](../../ICLR2026/signal_comm/mamba-3_improved_sequence_modeling_using_state_space_principles.md)
- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](../../CVPR2026/signal_comm/actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](../../CVPR2026/signal_comm/clay_conditional_visual_similarity.md)
- [\[NeurIPS 2025\] Memory-Integrated Reconfigurable Adapters: A Unified Framework for Settings with Multiple Tasks](memory-integrated_reconfigurable_adapters_a_unified_framework_for_settings_with_.md)

</div>

<!-- RELATED:END -->
