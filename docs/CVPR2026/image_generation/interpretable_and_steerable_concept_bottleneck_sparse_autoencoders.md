---
title: >-
  [Paper Note] Interpretable and Steerable Concept Bottleneck Sparse Autoencoders
description: >-
  [CVPR 2026][Image Generation][Sparse Autoencoders] This paper identifies that the majority of SAE neurons (~81%) suffer from insufficient interpretability or steerability…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Sparse Autoencoders"
  - "Concept Bottleneck"
  - "Interpretability"
  - "Steerability"
  - "Mechanistic Interpretability"
date: 2026-05-08
content_hash: 78fb115383610186
---

# Interpretable and Steerable Concept Bottleneck Sparse Autoencoders

**Conference**: CVPR 2026
**arXiv**: [2512.10805](https://arxiv.org/abs/2512.10805)  
**Code**: [GitHub](https://github.com/Trustworthy-ML-Lab/CB-SAE)  
**Area**: Image Generation
**Keywords**: Sparse Autoencoders, Concept Bottleneck, Interpretability, Steerability, Mechanistic Interpretability

## TL;DR

This paper identifies that the majority of SAE neurons (~81%) suffer from insufficient interpretability or steerability, and proposes the CB-SAE framework — which prunes low-utility SAE neurons and augments them with a concept bottleneck module — achieving +32.1% interpretability and +14.5% steerability improvements on LVLM and image generation tasks, respectively.

## Background & Motivation

Sparse autoencoders (SAEs) have become a foundational tool for mechanistic interpretability, decomposing dense, polysemantic activations in LLMs/VLMs into sparse, monosemantic latent variables. For practical deployment, SAE features must simultaneously satisfy two conditions: **interpretability** (humans can understand what each neuron encodes) and **steerability** (intervening on neuron activations reliably alters model outputs).

Through empirical analysis, this paper identifies two key limitations of SAEs:
1. **Most neurons are not practically useful**: Among 65,536 SAE neurons, only 18.84% exhibit both high interpretability and high steerability; 36.26% score low on both.
2. **Insufficient coverage of user-desired concepts**: Despite large SAE dictionaries, 27–45% of ImageNet-related concepts remain unrepresented by SAEs.

Concept Bottleneck Models (CBMs) offer explicit concept control but cannot discover novel features. The core idea of this paper is to unify the unsupervised discovery capability of SAEs with the controllability of CBMs into a single framework.

## Method

### Overall Architecture

CB-SAE follows a four-step pipeline:
1. Train a standard SAE → 2. Evaluate the interpretability and steerability of each neuron → 3. Prune low-utility neurons → 4. Train a lightweight concept bottleneck autoencoder alongside the frozen SAE.

### Key Designs

1. **Interpretability and Steerability Metrics**:
    - Function: Systematically evaluate the practical utility of each SAE neuron.
    - Mechanism:
        - **Interpretability**: Uses CLIP-Dissect to associate each SAE neuron with the best-matching concept from a predefined concept set; the highest similarity score serves as the interpretability score.
        - **Steerability**: Evaluated via a forward-pass test through LLaVA/UnCLIP — the target neuron activation is set to a high value $\alpha=50$ while all others are zeroed (or set to a white-image baseline), and the cosine similarity between the output and the sentence embedding of the CLIP-Dissect-assigned concept is computed.
    - Design Motivation: Interpretability $\neq$ steerability. A highly interpretable neuron may have weak causal effect or high entanglement, rendering it unsteerable; a highly steerable neuron may encode abstract or compositional features that are not interpretable. Jointly quantifying both dimensions enables effective pruning.

2. **SAE Neuron Pruning**:
    - Function: Remove low-utility SAE neurons to make room for the concept bottleneck.
    - Mechanism: Neurons are ranked by their combined interpretability + steerability score in ascending order; the bottom $M$ neurons are pruned (defaulting to retaining 30K out of 65K). The corresponding rows/columns of the encoder/decoder matrices are directly removed: $E'_{sae} = E_{sae}[[\omega]\setminus\mathcal{P},:]$, $D'_{sae} = D_{sae}[:,[\omega]\setminus\mathcal{P}]$.
    - Design Motivation: Direct pruning is simpler than weighting or regularization. The concept set $\mathcal{C} = \mathcal{C}_{user} \setminus \mathcal{C}_{rsae}$ contains only concepts missing from the SAE, avoiding redundancy.

3. **Concept Bottleneck Autoencoder (CB-AE) Training**:
    - Function: Augment the retained SAE with encoding and decoding capability for user-specified concepts.
    - Mechanism: A linear encoder $E_{cb} \in \mathbb{R}^{|\mathcal{C}| \times d}$ and decoder $D_{cb} \in \mathbb{R}^{d \times |\mathcal{C}|}$ operate in parallel with the frozen pruned SAE. Reconstruction is given by $\hat{v}' = D'_{sae}z' + b + D_{cb}\sigma_{cb}(c)$, where $\sigma_{cb}$ applies top-$k$ sparsification ($k=5$).
    - Three training objectives are optimized alternately:
        - **Reconstruction loss $\mathcal{L}_r$** (updates $E_{cb}, D_{cb}$): Recovers reconstruction degradation caused by pruning.
        - **Interpretability loss $\mathcal{L}_{int}$** (updates $E_{cb}$): Uses CLIP zero-shot classifiers to generate pseudo-labels; a cosine-cubed similarity loss aligns concept encodings.
        - **Steerability loss $\mathcal{L}_{st}$** (updates $D_{cb}$): Cyclic reconstruction — the reconstructed $\hat{v}'$ is re-encoded through $E_{cb}$ to obtain $\hat{c}$, which is then supervised with the same pseudo-label loss. This ensures that modifying a concept in the decoder is faithfully reflected in the reconstructed features.
    - Design Motivation: The encoder and decoder are updated by separate objectives — the encoder handles "understanding" (interpretability) while the decoder handles "control" (steerability). Cyclic reconstruction provides a task-agnostic steerability objective, enabling the same CB-SAE to control different downstream tasks.

### Loss & Training

Three objectives are optimized alternately without loss-weighting hyperparameters; instead, separate Adam optimizers provide adaptive scaling:
- $\mathcal{L}_r = \|v - \hat{v}'\|_2^2$: Reconstruction fidelity.
- $\mathcal{L}_{int}$: Cosine-cubed similarity loss, updates $E_{cb}$ only.
- $\mathcal{L}_{st}$: Cyclic cosine-cubed similarity loss, updates $D_{cb}$ only.

## Key Experimental Results

### Main Results — Steerable Generation with LLaVA/UnCLIP

| Downstream Model | Method | CLIP-Dissect↑ | Monosemanticity↑ | Unit Vector↑ | White Image↑ |
|---------|------|-------------|--------|---------|--------|
| LLaVA-1.5-7B | SAE | 0.154 | 0.517 | 0.198 | 0.203 |
| LLaVA-1.5-7B | **CB-SAE** | **0.244** | **0.556** | **0.261** | **0.250** |
| LLaVA-MORE | SAE | 0.194 | 0.553 | 0.179 | 0.177 |
| LLaVA-MORE | **CB-SAE** | **0.291** | **0.598** | **0.192** | **0.189** |
| UnCLIP | SAE | 0.058 | 0.540 | 0.642 | 0.654 |
| UnCLIP | **CB-SAE** | **0.092** | **0.594** | **0.659** | **0.664** |

Average interpretability gain: +32.1%; steerability gain: +14.5%.

### Ablation Study — Neuron Type Analysis

| Neuron Type | CLIP-Dissect | Unit Vector | White Image |
|-----------|-------------|---------|--------|
| All SAE neurons | 0.154 | 0.198 | 0.203 |
| Pruned SAE neurons | 0.084 | 0.144 | 0.162 |
| Retained SAE neurons | 0.238 | 0.263 | 0.252 |
| CB neurons | **0.323** | 0.231 | 0.219 |
| All CB-SAE neurons | 0.244 | **0.261** | **0.250** |

### Key Findings

- SAE neurons exhibit a four-quadrant distribution: only 18.84% are high-interpretability + high-steerability, while 36.26% are low on both, indicating severe polarization.
- SAE concept coverage degrades sharply as the concept set grows: 96.3% on Broden → only 28.0% on a 20K English vocabulary.
- CB neurons exhibit substantially higher interpretability than SAE neurons (0.323 vs. 0.154), validating the necessity of concept supervision.
- The steerability loss $\mathcal{L}_{st}$ contributes a +2.9% steerability improvement without degrading interpretability.
- Retaining fewer SAE neurons yields higher scores, but retaining too few harms reconstruction; 30K represents a reasonable trade-off.

## Highlights & Insights

- This paper is the first to systematically expose the trade-off between interpretability and steerability in SAEs, and to quantify the concept coverage gap.
- Unifying SAEs (unsupervised discovery) and CBMs (supervised concept alignment) in a single framework is a natural and effective design choice.
- The cyclic reconstruction steerability loss is an elegant task-agnostic design that enables the same CB-SAE to be applied to both text generation and image generation downstream tasks.
- The concept set selection strategy (adding only concepts absent from the SAE) avoids redundancy.

## Limitations & Future Work

- Relies on CLIP-Dissect for concept assignment, which may itself be inaccurate.
- The steerability of CB neurons remains below that of the retained SAE neurons; better or task-specific steerability losses are needed.
- Validation is limited to CLIP visual encoders; applicability to other visual encoders (e.g., DINOv2) requires further investigation.
- The relationship between SAE "feature splitting" and concept coverage gaps is not thoroughly explored.
- Training depends on pseudo-labels from CLIP zero-shot classifiers, whose accuracy is inherently bounded by CLIP itself.

## Related Work & Insights

- **vs. Standard SAE**: SAEs are purely unsupervised and do not guarantee discovery of user-desired concepts; moreover, most neurons are low-utility. CB-SAE addresses both issues through pruning and augmentation.
- **vs. CBM**: CBMs are confined to predefined concept sets and cannot discover novel features; CB-SAE preserves the discovery capability of SAEs.
- **vs. AlignSAE**: A concurrent work — AlignSAE uses orthogonality losses to separate supervised/unsupervised concepts, while CB-SAE directly prunes low-utility neurons. AlignSAE targets text-based LLMs, whereas CB-SAE targets visual models.

## Rating

- Novelty: ⭐⭐⭐⭐ The unification of SAE and CBM is natural and effective; the interpretability/steerability analysis has independent value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two downstream tasks, detailed ablations and sensitivity analyses, though dataset coverage is limited (ImageNet only).
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear, the methodology is presented in a natural progression, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Meaningfully advances the practical utility of SAEs, particularly for applications requiring control over specific concepts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](../../NeurIPS2025/image_generation/learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)
- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] PureCC: Pure Learning for Text-to-Image Concept Customization](purecc_pure_learning_for_text-to-image_concept_customization.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
