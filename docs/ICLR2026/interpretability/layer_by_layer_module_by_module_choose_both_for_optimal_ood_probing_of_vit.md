---
title: >-
  [Paper Note] Layer by layer, module by module: Choose both for optimal OOD probing of ViT
description: >-
  [ICLR 2026 (CAO Workshop)][Interpretability][Vision Transformer] This work systematically investigates the intermediate-layer behavior of pretrained ViTs through large-scale linear probing experiments. It finds that dist…
tags:
  - "ICLR 2026 (CAO Workshop)"
  - "Interpretability"
  - "Vision Transformer"
  - "Linear Probing"
  - "Distribution Shift"
  - "Intermediate Representations"
  - "OOD"
date: 2026-05-08
content_hash: 1468f07f85122410
---

# Layer by layer, module by module: Choose both for optimal OOD probing of ViT

**Conference**: ICLR 2026 (CAO Workshop)  
**arXiv**: [2603.05280](https://arxiv.org/abs/2603.05280)  
**Code**: [GitHub](https://github.com/ambroiseodt/vit-probing)  
**Area**: Interpretability  
**Keywords**: Vision Transformer, Linear Probing, Distribution Shift, Intermediate Representations, OOD

## TL;DR
This work systematically investigates the intermediate-layer behavior of pretrained ViTs through large-scale linear probing experiments. It finds that distribution shift is the primary cause of performance degradation in deeper layers, and reveals at the module level that the optimal probing point depends on the degree of shift: probing FFN activations is optimal under significant shift, while probing MHSA post-normalization outputs is optimal under mild shift.

## Background & Motivation
In recent years, a striking phenomenon has been repeatedly observed in foundation models: **intermediate-layer representations often yield more discriminative features than final-layer representations**. This phenomenon was first identified in autoregressively pretrained language models and has since been recognized in vision models trained with supervised and discriminative self-supervised objectives (e.g., DINO, MAE).

However, several critical gaps remain in the existing understanding:

**Unexplained cause**: Why do intermediate layers outperform the final layer? The phenomenon was initially attributed to autoregressive pretraining, yet this fails to explain why models trained with supervised or contrastive objectives exhibit the same behavior.

**Insufficient granularity**: Existing studies typically probe the output of each Transformer block as a whole, overlooking the distinct properties of individual submodules (MHSA, FFN, LayerNorm, etc.) within the block.

**Lack of practical guidance**: For practitioners, the critical question of "which layer and which module to extract features from" lacks a systematic answer.

The central hypothesis is that **distribution shift** — the discrepancy between pretraining and downstream data — is the fundamental cause of deep-layer representation degradation, independent of the pretraining paradigm. In the deeper layers of a ViT, the model becomes increasingly specialized to the distributional characteristics of the pretraining data, reducing its generalization capacity to downstream data.

More specifically, this paper posits that different modules within a Transformer block exhibit varying sensitivity to distribution shift, and therefore the optimal feature extraction point depends not only on layer depth but also on the specific submodule within the block.

## Method

### Overall Architecture
This paper adopts **linear probing** as the core experimental paradigm:
- **Input**: Pretrained ViT models (frozen parameters) + multiple downstream image classification datasets
- **Variables**: Probing layer depth (layer index) × probing module position (module type)
- **Evaluation**: A linear classifier is trained on downstream datasets, and classification accuracy is reported
- **Output**: Systematic findings on the "optimal probing point"

Unlike conventional approaches that probe only the final output of each Transformer block, this paper probes at the **output/input position of every submodule** within the block, including:
- Input, output, and post-normalization output of MHSA (Multi-Head Self-Attention)
- Input, intermediate activations (post-GELU), and output of FFN (Feed-Forward Network)
- Output of LayerNorm
- Representations before and after residual connections

### Key Designs

1. **Distribution Shift Hypothesis Validation (Distribution Shift Analysis)**: Experiments are carefully designed to validate the hypothesis that distribution shift is the primary cause of deep-layer degradation:

    - Downstream datasets spanning a wide range of "distances" from pretraining data are selected: from datasets closely aligned with ImageNet (e.g., ImageNet-V2, ImageNet-Sketch) to domain-specific datasets with large distributional gaps (e.g., EuroSAT remote sensing, Flowers102)
    - ViTs trained with multiple pretraining strategies (supervised, DINO self-supervised, MAE, etc.) are evaluated
    - Key finding: On in-distribution datasets, the final layer (or layers near it) is consistently optimal; on out-of-distribution datasets, intermediate layers substantially outperform the final layer. This pattern holds consistently across pretraining paradigms, providing strong evidence that distribution shift — not the pretraining objective — is the decisive factor

2. **Module-Level Fine-Grained Analysis**: This constitutes the paper's most central contribution. Having established layer-level trends, the analysis is extended to the internals of each Transformer block:

    - Standard practice probes the final output of each block (i.e., post-residual representations), which conflates the contributions of both MHSA and FFN
    - This paper separately probes MHSA output (before and after normalization), FFN input, FFN intermediate activations (after the nonlinear transformation but before the output projection), FFN output, and other positions
    - **Key conclusions**:
        - **Under strong distribution shift**: Probing FFN intermediate activations (post-GELU representations) yields the best performance. A plausible explanation is that the FFN serves a feature transformation role, with its intermediate representations residing at a sweet spot between general and task-specific features
        - **Under weak distribution shift**: Probing the post-normalization output of MHSA yields the best performance. MHSA focuses on spatial relationship modeling, and its global feature aggregation capability remains more valuable when the distributional gap is small

3. **Systematic Cross-Configuration Experiments**: To ensure robustness of the conclusions, experiments cover:

    - Multiple ViT scales (ViT-S, ViT-B, ViT-L)
    - Multiple pretraining methods (supervised ImageNet, DINOv1/v2, MAE, etc.)
    - Multiple patch sizes (16, 14, etc.)
    - 10+ downstream classification datasets spanning varying degrees of distribution shift

### Loss & Training
Linear probing employs standard cross-entropy loss to train the linear classifier. ViT parameters are fully frozen; only linear layer parameters are optimized. All experiments use a unified hyperparameter search strategy to ensure fair comparison. The codebase is built on the OmegaConf configuration system and the accelerate library, supporting flexible experiment configuration.

## Key Experimental Results

### Main Results
Classification accuracy trends across probing depths on datasets with varying degrees of distribution shift:

| Dataset Type | Distribution Shift | Optimal Layer | Optimal Module | Representative Datasets |
|---|---|---|---|---|
| ImageNet variants | Weak | Final or near-final layer | MHSA post-norm output | ImageNet-V2, ImageNet-R |
| General natural images | Moderate | Intermediate layers | FFN activation / MHSA output | CIFAR-10, STL-10 |
| Domain-specific | Strong | Shallow-to-middle layers | FFN intermediate activations | EuroSAT, Flowers102 |

### Ablation Study

| Configuration | Key Finding | Remarks |
|---|---|---|
| Supervised vs. DINO vs. MAE pretraining | Consistent pattern | Distribution shift effect is independent of pretraining paradigm |
| ViT-S vs. ViT-B vs. ViT-L | Consistent pattern | Model scale does not alter core conclusions |
| Block output vs. FFN activation vs. MHSA output | FFN activation optimal under OOD | Demonstrates that standard probing is suboptimal |
| Different patch sizes | Consistent pattern | Patch size has negligible effect on conclusions |

### Key Findings
- **Distribution shift is the sole decisive factor**: Across all pretraining methods and model scales, the degree of distribution shift is highly correlated with the degree of deep-layer degradation, while the type of pretraining objective shows no significant effect
- **Standard block-output probing is consistently suboptimal**: Regardless of the degree of distribution shift, probing points within internal submodules can always be found that outperform probing the block output
- **Final layer is always optimal under in-distribution settings**: When pretraining and downstream data distributions are aligned, the "intermediate layer surpasses final layer" phenomenon does not arise; the complete representation of the final layer is always best
- **FFN activations serve as a universal solution for OOD**: Under all strong-shift scenarios, FFN intermediate activations consistently yield the best representations, suggesting that the FFN nonlinearity is a critical transition point from general to specialized features
- **MHSA is superior under weak shift**: When the distributional gap is small, the global attention patterns learned by MHSA during pretraining remain applicable to downstream data

## Highlights & Insights
- **Precise problem formulation**: A widely observed but insufficiently understood phenomenon (intermediate layers outperforming final layers) is decomposed along two orthogonal dimensions — layer depth and submodule type — for rigorous analysis
- **Counterintuitive findings with strong practical value**: Practitioners typically default to using final-layer or block-output features; this paper provides clear guidelines for selecting the optimal feature extraction point based on the degree of distribution shift
- **FFN as a feature "bottleneck"**: The optimality of FFN intermediate activations under OOD conditions implies that the FFN plays a "feature distillation" role in ViTs — its input is general, its output is specialized, and the intermediate representation resides at the optimal balance point
- **Systematic experimental design**: The study spans multiple dimensions including pretraining method, model scale, downstream datasets, and module type, lending high credibility to the conclusions

## Limitations & Future Work
- As a Workshop paper, the experimental scale and depth of discussion are relatively limited; some findings would benefit from more thorough theoretical analysis
- Only linear probing is considered; conclusions under nonlinear probing (e.g., MLP head) or fine-tuning settings may differ
- The quantitative relationship between the degree of distribution shift and the optimal probing point is not explored — currently only qualitative rules are established ("strong shift → FFN activations, weak shift → MHSA output")
- Dense prediction tasks (e.g., detection, segmentation) are not considered; different modules may exhibit distinct behavior with respect to spatial information preservation
- An in-depth mechanistic analysis of "why FFN intermediate activations are optimal under OOD" is absent, e.g., from the perspective of representational geometry or feature separability
- No automated scheme is proposed for determining distribution shift intensity at inference time to adaptively select a probing strategy in practical deployment

## Related Work & Insights
- **Beyond the final layer (Attentive multilayer fusion for ViTs)**: Focuses on multi-layer fusion strategies, complementary to this paper's approach of selecting a single optimal layer/module
- **ViT-5**: A 2026 ViT improvement work that enhances ViT performance from an architectural perspective, whereas this paper provides insights from a feature utilization standpoint
- **Robust Representation Learning in Masked Autoencoders**: Addresses robust representation learning in MAE; this paper's OOD analysis offers direct reference value
- Directions inspired by this work: Can an adaptive feature fusion strategy be designed that automatically selects the optimal layer–module combination based on the distance between an input sample and the pretraining distribution?

## Rating
- Novelty: ⭐⭐⭐⭐ (Module-level analysis offers a novel perspective, though linear probing methodology itself is not new)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Systematic and comprehensive, though appropriately scoped as a Workshop paper)
- Writing Quality: ⭐⭐⭐⭐ (Clear and concise, with key messages well-highlighted)
- Value: ⭐⭐⭐⭐ (Provides important practical guidance for ViT feature utilization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Exploring Interpretability for Visual Prompt Tuning with Cross-layer Concepts](exploring_interpretability_for_visual_prompt_tuning_with_cross-layer_concepts.md)
- [\[ICML 2026\] Is One Layer Enough? Understanding Inference Dynamics in Tabular Foundation Models](../../ICML2026/interpretability/is_one_layer_enough_understanding_inference_dynamics_in_tabular_foundation_model.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](../../NeurIPS2025/interpretability/towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](../../ICML2026/interpretability/optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](dynamic_reflections_probing_video_representations_with_text_alignment.md)

</div>

<!-- RELATED:END -->
