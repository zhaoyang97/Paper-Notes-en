---
title: >-
  [Paper Note] DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions
description: >-
  [CVPR 2026][LLM Safety][machine unlearning] Proposes DAMP (Depth-Aware Modulation via Projection), a one-shot closed-form weight surgery method for class unlearning that achieves selective forgetting by removing forget-c…
tags:
  - "CVPR 2026"
  - "LLM Safety"
  - "machine unlearning"
  - "class forgetting"
  - "weight surgery"
  - "projection"
  - "depth-aware"
date: 2026-05-08
content_hash: ab930b6b7280f773
---

# DAMP: Class Unlearning via Depth-Aware Removal of Forget-Specific Directions

**Conference**: CVPR 2026
**arXiv**: [2604.15166](https://arxiv.org/abs/2604.15166)  
**Code**: None  
**Area**: AI Safety / Machine Unlearning
**Keywords**: machine unlearning, class forgetting, weight surgery, projection, depth-aware

## TL;DR

Proposes DAMP (Depth-Aware Modulation via Projection), a one-shot closed-form weight surgery method for class unlearning that achieves selective forgetting by removing forget-class-specific directions in the editing space of each network stage, with a depth-aware scaling rule enforcing conservative edits in shallow layers and aggressive edits in deep layers.

## Background & Motivation

Machine unlearning aims to remove target knowledge from a trained model. Existing class unlearning methods suffer from three key limitations: (1) weak selectivity — forgetting is often accompanied by degraded performance on retained classes (negative selectivity); (2) residual forget information — even when the output no longer predicts the forget class, its features remain decodable from deep representations; (3) reliance on classifier head suppression — lowering forget-class scores by shifting the last-layer bias rather than truly removing internal representational evidence. These issues indicate that output-level forgetting and representation-level removal are distinct phenomena.

## Method

### Overall Architecture

For each stage $\ell$ of the pretrained network, class prototypes are computed in the input space of the next learnable operator. The forget-class direction is extracted as the residual relative to the span of retained-class prototypes, and the weight matrix of the next operator is updated via projection to reduce sensitivity to the forget direction. A depth-aware coefficient controls the edit strength at each layer.

### Key Designs

1. **Forget Direction Extraction (Retained-Span Residual)**: At each stage's editing space, the forget-class prototype $\boldsymbol{\mu}_f^\ell$ is projected onto the span of retained-class prototypes to obtain an interpretable component; the residual $d_f^\ell = \boldsymbol{\mu}_f^\ell - R^\ell (R^\ell)^\dagger \boldsymbol{\mu}_f^\ell$ constitutes the forget-specific direction. For multi-class forgetting, directions are stacked and orthogonalized via QR decomposition to form a low-rank forget subspace.

2. **Projection Surgery**: A right-projection update is applied to the next operator's weight matrix: $W'^{\ell+1} = W^{\ell+1}(I - \alpha_\ell \widetilde{Q}^\ell (\widetilde{Q}^\ell)^\top)$, removing the weights' sensitivity to forget-subspace directions. This is a closed-form operation requiring no gradient optimization. Biases remain unchanged.

3. **Depth-Aware Scaling**: The coefficient $\alpha_\ell$ is computed from the separability and certainty of a binary linear probe (forget vs. retain) combined with a depth ramp. Shallow layers share more features → small edits; deep layers encode stronger class-specific structure → large edits. Probes are trained only once and require no re-estimation thereafter.

### Loss & Training

No training is required — all statistics (class prototypes, probe accuracies) are computed one-shot from the pretrained model, and weight edits use closed-form projection. The method is architecture-agnostic and supports both CNNs and Transformers.

## Key Experimental Results

### Main Results

Evaluated on MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet with CNNs and Transformers:

| Method | Retain Acc. | Forget Acc. | Selectivity (pp) | Closeness to Retrain |
|--------|-------------|-------------|------------------|----------------------|
| GAU | Drops | Drops | Weak/Negative | Far |
| SalUn | Acceptable | Moderate drop | Weak | Medium |
| **DAMP** | **Well preserved** | **Near zero** | **High positive** | **Closest** |

DAMP significantly outperforms all compared methods on selectivity metrics and most closely approximates the retrain gold standard.

### Ablation Study

- Depth-aware scaling vs. uniform scaling: the former substantially improves retained-class performance.
- Low-rank subspace removal for multi-class forgetting is effective and requires no additional hyperparameters.
- RDM analysis shows that DAMP's representations most closely match those of the retrained network.

### Key Findings

- Much of the "forgetting" in existing methods is essentially classifier head bias suppression.
- DAMP's deep-layer representational geometry most closely matches the retrained network (minimal RDM discrepancy).
- The retained-span residual provides more precise forget-specific information than global directions.

## Highlights & Insights

- The perspective of re-examining forgetting quality through the lens of selectivity is insightful.
- Closed-form projection surgery requires no iterative optimization, offering both efficiency and determinism.
- The depth-aware editing rule has a solid theoretical grounding in hierarchical feature learning theory.

## Limitations & Future Work

- Using the mean as the class prototype may be insufficient for multimodal distributions.
- Projection strength is indirectly controlled by probe separability, with no theoretical guarantee of optimality.
- The method has not been fully validated on large-scale pretrained models (e.g., ImageNet-pretrained ResNet-50+).

## Related Work & Insights

- The subspace projection editing paradigm can be extended to concept unlearning and knowledge editing.
- The depth-aware editing strategy offers reference value for other model modifications requiring layer-wise selection.
- The selectivity metric provides a more comprehensive evaluation dimension for unlearning research.

## Rating

7/10 — The method has a solid theoretical foundation and thorough analysis, though experimental scale could be further extended.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ICML 2026\] Forget to Know, Remember to Use: Context-Aware Unlearning for Large Language Models](../../ICML2026/llm_safety/forget_to_know_remember_to_use_context-aware_unlearning_for_large_language_model.md)
- [\[ICCV 2025\] Forgetting Through Transforming: Enabling Federated Unlearning via Class-Aware Representation Transformation](../../ICCV2025/llm_safety/forgetting_through_transforming_enabling_federated_unlearning_via_class-aware_re.md)
- [\[CVPR 2026\] Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)
- [\[CVPR 2026\] Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)

</div>

<!-- RELATED:END -->
