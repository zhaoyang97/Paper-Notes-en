---
title: >-
  [Paper Note] RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)
description: >-
  [AAAI 2026][Multimodal VLM][CLIP] This paper proposes RMAdapter, a dual-branch adapter architecture that augments the standard adaptation branch with a reconstruction branch (analogous to an AutoEncoder). By sharing the…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "CLIP"
  - "Adapter"
  - "Reconstruction"
  - "few-shot learning"
  - "Vision-Language Model"
date: 2026-05-08
content_hash: e965339817c1f34d
---

# RMAdapter: Reconstruction-based Multi-Modal Adapter for Vision-Language Models (Oral)

**Conference**: AAAI 2026 Oral  
**arXiv**: [2512.06811](https://arxiv.org/abs/2512.06811)  
**Code**: Not mentioned  
**Area**: Vision-Language Model / Parameter-Efficient Fine-Tuning  
**Keywords**: CLIP, Adapter, Reconstruction, few-shot learning, Vision-Language Model

## TL;DR

This paper proposes RMAdapter, a dual-branch adapter architecture that augments the standard adaptation branch with a reconstruction branch (analogous to an AutoEncoder). By sharing the down-projection layer and applying per-layer local reconstruction losses, RMAdapter achieves an optimal balance between task-specific adaptation and general knowledge retention in few-shot CLIP fine-tuning, outperforming state-of-the-art methods (including prompt-based approaches) across three benchmarks: Base-to-Novel generalization, cross-dataset transfer, and domain generalization.

## Background & Motivation

Pre-trained VLMs (e.g., CLIP) face a fundamental tension in few-shot downstream adaptation — the **adaptation-generalization trade-off**:

**Prompt Learning**: The line CoOp → CoCoOp → MaPLe → PromptSRC → CoPrompt has advanced rapidly, yet fundamentally lacks explicit knowledge retention mechanisms. Learned prompts are highly discriminative for seen classes but exhibit strong bias against unseen classes.

**Underexplored Adapter Direction**: Compared to prompt-based methods, adapter approaches remain significantly underexplored. Existing adapters (e.g., MMA) employ only a single branch focused on adaptation, lacking structured designs to govern the balance between discriminability and generalizability.

**Key Observation — Structural Isomorphism Between Adapters and AutoEncoders**: The down-projection → up-projection structure of an adapter is isomorphic to the encoder → decoder structure of an AE, naturally motivating the addition of a reconstruction branch to constrain the feature space from drifting away from the original distribution.

## Method

### Overall Architecture

RMAdapter inserts dual-branch adapters into the upper layers (last $k$ layers) of both the visual and text encoders of CLIP. The entire CLIP backbone is frozen; only adapter parameters are trained. The two branches share the down-projection layer and perform task adaptation and feature reconstruction respectively, with residual connections fusing their outputs with the original CLIP representations.

### Key Designs

1. **Adaptation Branch (RMAdapter_base)**: Standard adapter structure, $x_{down} = \sigma(x W_{down} + b_{down})$, $\text{output} = x_{down} W_{up}^{base} + b_{up}^{base}$, injecting task-specific knowledge.
2. **Reconstruction Branch (RMAdapter_rec)**: A two-layer up-projection structure, $\text{output} = \sigma(x_{down} W_{up1}^{rec} + b_{up1}^{rec}) W_{up2}^{rec} + b_{up2}^{rec}$, reconstructing the latent representation back to the original feature space; an L2 reconstruction loss constrains general knowledge retention.
3. **Shared Down-Projection**: Both branches share $W_{down}$, enabling a Pareto-optimal adaptation-reconstruction trade-off. The shared projection forces both branches to operate in the same low-rank space: the adaptation branch learns task-relevant features while the reconstruction branch ensures this space does not deviate from the original distribution — a natural mutual regularization.
4. **Per-Layer Local Reconstruction Loss**: $\mathcal{L}_{rec}^V = \sum_{i=k}^K \|[c_i, E_i] - \text{RMAdapter}_{rec}([c_i, E_i])\|^2$, computing L2 loss independently at each layer without cross-layer backpropagation, yielding computational efficiency.
5. **Consistency Constraint**: $\mathcal{L}_{con} = \lambda_3 \|x^a - x\|_1 + \lambda_4 \|w^a - w\|_1$, constraining the L1 distance between adapted features and original CLIP features.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{ce} + \mathcal{L}_{con} + \mathcal{L}_{rec}$$

Cross-entropy (classification supervision) + consistency constraint (preventing deviation from original features) + reconstruction loss (preserving general knowledge). L2, L1, and cosine reconstruction objectives are evaluated; L2 proves most stable.

## Key Experimental Results

### Main Results: Base-to-Novel Generalization (Average over 11 Datasets)

| Method | Type | Base Acc | Novel Acc | HM |
|------|------|---------|-----------|-----|
| CLIP (zero-shot) | — | 69.34 | 74.22 | 71.70 |
| CoOp | Prompt | 82.69 | 63.22 | 71.66 |
| CoCoOp | Prompt | 80.47 | 71.69 | 75.83 |
| MaPLe | Prompt | 82.28 | 75.14 | 78.55 |
| PromptSRC | Prompt | 84.26 | 76.10 | 79.97 |
| MMA | Adapter | 83.20 | 76.80 | 79.87 |
| CoPrompt | Prompt | 84.00 | 77.23 | 80.48 |
| **RMAdapter** | **Adapter** | **84.52** | **77.36** | **80.62** |

### Ablation Study: Contribution of Key Designs to HM

| Configuration | HM | Note |
|------|-----|------|
| Single-branch Adapter (MMA) | 79.87 | No reconstruction constraint |
| + Reconstruction branch (non-shared) | ~80.1 | Independent parameters, limited gain |
| + Shared down-projection | ~80.4 | Pareto-optimal trade-off |
| + Consistency constraint | 80.62 | Full model |
| Reconstruction branch: 1-layer up-projection | Slightly lower | Insufficient capacity |
| Reconstruction branch: 2-layer up-projection | Optimal | Sweet spot |
| Reconstruction branch: 3-layer up-projection | Degraded | Few-shot overfitting |

### Key Findings

- **Cross-dataset generalization** (average over 10 datasets): RMAdapter 67.56% vs. CoPrompt 67.00% vs. MMA 66.61%
- **Domain generalization** (average over 4 ImageNet variants): RMAdapter 60.71% vs. PromptSRC 60.65% vs. CoPrompt 60.42%
- **The reconstruction branch adds only ~320K parameters**, +3% GPU memory and +5% training time, yet yields significant performance gains
- **Adapters comprehensively surpass prompt-based methods for the first time**, demonstrating that the adapter direction has been substantially underestimated

## Highlights & Insights

- **Structural Analogy Between Adapters and AutoEncoders**: A particularly elegant observation — the adapter's down-projection → up-projection is isomorphic to an AE's encoder → decoder, making the addition of a reconstruction branch both natural and principled. This approach of "discovering new connections within existing structures" is methodologically instructive.
- **Intuition Behind Shared Down-Projection**: Both branches operate in the same low-rank space; the adaptation branch learns task-specific features while the reconstruction branch ensures the low-rank space does not drift — natural mutual regularization achieving a Pareto-optimal trade-off.
- **No Reliance on Data Augmentation or Complex Prompt Engineering**: The approach is simpler than methods such as CoPrompt.
- **Reconstruction as Regularization**: The reconstruction objective serves as a knowledge retention mechanism, conceptually similar to knowledge distillation but more lightweight — no teacher model forward pass is required.

## Limitations & Future Work

- Experiments are conducted on ViT-B/16 CLIP only; ViT-L, SigLIP, and EVA-CLIP variants are not evaluated.
- Validation is limited to classification tasks; extension to detection, segmentation, and other downstream tasks is not explored.
- The reconstruction branch uses a simple MSE objective; more structured retention strategies (e.g., feature direction preservation, subspace projection) could be explored.
- Integration with other PEFT methods such as LoRA and Prefix Tuning is not discussed.

## Related Work & Insights

| Method Category | Representative Methods | Core Mechanism | Knowledge Retention Strategy |
|----------|---------|---------|-------------|
| Prompt Learning | CoOp, CoCoOp, MaPLe | Learnable prompt tokens | Implicit (no explicit design) |
| Regularized Prompt | PromptSRC, KgCoOp | Prompt + constraints | Self-regularization, text embedding distance |
| Hybrid Prompt+Adapter | CoPrompt | Prompt + adapter | Consistency constraint |
| Single-branch Adapter | CLIP-Adapter, MMA | Adaptation branch only | None |
| **Dual-branch Adapter (Ours)** | **RMAdapter** | **Adaptation + reconstruction branch** | **Explicit reconstruction loss + shared down-projection** |

## Rating

- Novelty: ⭐⭐⭐⭐ Elegant AE-Adapter analogy; dual-branch design is principled and clean
- Experimental Thoroughness: ⭐⭐⭐⭐ 11-dataset + cross-dataset + domain generalization + comprehensive ablation
- Writing Quality: ⭐⭐⭐⭐ Problem motivation and methodological derivation are logically coherent
- Value: ⭐⭐⭐⭐ Demonstrates the underestimation of the adapter direction; provides a generalizable PEFT design paradigm

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)
- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[AAAI 2026\] ImageBindDC: Compressing Multi-modal Data with ImageBind-based Condensation](imagebinddc_compressing_multi-modal_data_with_imagebind-based_condensation.md)
- [\[AAAI 2026\] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)

</div>

<!-- RELATED:END -->
