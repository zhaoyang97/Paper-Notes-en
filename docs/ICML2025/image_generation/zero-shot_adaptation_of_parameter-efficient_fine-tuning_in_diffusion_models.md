---
title: >-
  [Paper Note] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models
description: >-
  [ICML 2025][Image Generation][LoRA Migration] ProLoRA is proposed, a training-free closed-form LoRA cross-model migration method. By decomposing and projecting the source LoRA onto the subspace and null space of the source model weights, and then re-projecting them onto the corresponding spaces of the target model, lossless transfer of style, concept, and acceleration LoRAs across different diffusion models is achieved.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "LoRA Migration"
  - "Zero-Shot Adaptation"
  - "Subspace Projection"
  - "Parameter-Efficient Fine-Tuning"
  - "Diffusion Models"
date: 2026-05-08
content_hash: a0d5bca20e6adfc8
---

# Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models

**Conference**: ICML 2025  
**arXiv**: [2506.04244](https://arxiv.org/abs/2506.04244)  
**Code**: None  
**Area**: Diffusion Models / Model Compression  
**Keywords**: LoRA Migration, Zero-Shot Adaptation, Subspace Projection, Parameter-Efficient Fine-Tuning, Diffusion Models

## TL;DR
ProLoRA is proposed, a training-free closed-form LoRA cross-model migration method. By decomposing and projecting the source LoRA onto the subspace and null space of the source model weights, and then re-projecting them onto the corresponding spaces of the target model, lossless transfer of style, concept, and acceleration LoRAs across different diffusion models is achieved.

## Background & Motivation
**Background**: LoRA is the dominant PEFT method for diffusion models. However, LoRA adapters are tightly coupled with the base model—meaning old LoRAs cannot be directly used when the base model is upgraded (e.g., SDXL to SSD-1B).

**Limitations of Prior Work**: (a) Re-training LoRA requires original data and computational resources; (b) Original training data may be unavailable due to copyright or privacy issues; (c) Existing transfer methods (e.g., LoRA-X) restrict LoRA to only affecting the weight subspace, limiting expressiveness.

**Key Challenge**: The weight update $\Delta W_s$ of standard LoRA simultaneously affects both the subspace and null space of the source model weight $W_s$. How can these two parts be transferred separately?

**Goal**: Transfer any pre-trained LoRA from a source model to a target model without requiring training data.

**Key Insight**: SVD-decompose source/target model weights and separately project the subspace and null space components of LoRA into the corresponding spaces of the target model.

**Core Idea**: $\Delta W_{t\leftarrow s} = U_{t,\parallel}U_{t,\parallel}^\top \Delta W_{s,\parallel} V_{t,\parallel}^\top V_{t,\parallel} + U_{t,\perp}U_{t,\perp}^\top \Delta W_{s,\perp} V_{t,\perp}^\top V_{t,\perp}$

## Method

### Overall Architecture
ProLoRA accomplishes migration in three steps: (1) Identify highly similar module pairs in the source and target models; (2) Decompose the source LoRA into subspace and null space components; (3) Project these two components onto the corresponding spaces of the target model weights.

### Key Designs

1. **Subspace Similarity Metric**:

    - **Function**: Identify corresponding module pairs between source and target models.
    - **Mechanism**: Perform SVD on $W_s$ and $W_t$, and measure the similarity of left/right singular matrices using the Frobenius norm: $\Phi_l(W_s, W_t) = \|U_s^\top U_t\|_F^2 / n$. Highly similar modules are selected using a threshold of 0.8.
    - **Design Motivation**: Source and target models may have different numbers of layers (e.g., SDXL has 70 layers vs. SSD-1B with 40 layers), necessitating the identification of matching pairs.

2. **LoRA Subspace/Null Space Decomposition**:

    - **Function**: Decompose $\Delta W_s$ into two components: one in the subspace of $W_s$ and one in the null space of $W_s$.
    - **Mechanism**: $\Delta W_s \approx \underbrace{U_{s,\parallel}U_{s,\parallel}^\top \Delta W_s V_{s,\parallel}^\top V_{s,\parallel}}_{\Delta W_{s,\parallel}} + \underbrace{U_{s,\perp}U_{s,\perp}^\top \Delta W_s V_{s,\perp}^\top V_{s,\perp}}_{\Delta W_{s,\perp}}$
    - **Design Motivation**: The subspace component adjusts the existing feature directions of the model, whereas the null space component introduces new feature directions. The migration strategies for these two components differ.

3. **Projection into Target Model Space**:

    - **Function**: Map the two decomposed components into the subspace and null space of the target model.
    - **Mechanism**: $\Delta W_{t\leftarrow s,\parallel} = U_{t,\parallel}U_{t,\parallel}^\top \Delta W_{s,\parallel} V_{t,\parallel}^\top V_{t,\parallel}$ (with a similar expression for the null space).
    - **Design Motivation**: Maintain functional equivalence of the LoRA in the target model.

### Computational Complexity
- The initial SVD computation takes $O(mn \cdot \min(m,n))$ time but can be shared across all migrations.
- Subsequent migrations only require matrix multiplications, making them significantly faster than re-training.

## Key Experimental Results

### Main Results

| Dataset | Target Model | Method | HPSv2 ↑ | CSD-MMD ↓ |
|--------|----------|------|---------|-----------|
| BlueFire | SSD-1B | LoRA (Training) | 0.323 | - |
| BlueFire | SSD-1B | **ProLoRA** | 0.318 | **0.021** |
| Paintings | SSD-1B | LoRA (Training) | 0.328 | - |
| Paintings | SSD-1B | **ProLoRA** | 0.318 | **0.013** |
| Origami | SD Eff-v1.0 | LoRA (Training) | 0.253 | - |
| Origami | SD Eff-v1.0 | **ProLoRA** | 0.257 | **0.003** |

### Ablation Study

| Method | CLIP-T ↑ | CLIP-I ↑ | DINOv2 ↑ |
|------|----------|----------|----------|
| No LoRA | 0.251 | 0.521 | 0.352 |
| Copy LoRA | 0.300 | 0.719 | 0.475 |
| **ProLoRA** | **0.287** | **0.737** | **0.501** |
| LoRA (Training) | 0.294 | 0.745 | 0.539 |

### Key Findings
- ProLoRA achieves HPSv2/CSD-MMD scores close to the re-trained LoRA in style transfer.
- In concept LoRA migration, ProLoRA significantly outperforms simple copying and approaches the performance of re-trained LoRA.
- Supports migration between models with different sampling steps (e.g., standard model to LCM 4-step model).

## Highlights & Insights
- **Closed-form, training-free solution**: Only requires SVD decomposition and matrix multiplications; no training data or forward propagation is needed.
- **Dual-space transfer (Subspace + Null Space)**: Compared to LoRA-X, which only transfers subspace components, ProLoRA provides a more complete transfer.
- **Multi-type LoRA support**: Validated across style, concept, and LCM acceleration LoRAs.

## Related Work & Insights
- **vs LoRA-X**: LoRA-X constrains the adapter to affect only the subspace and requires training on the source model. ProLoRA transfers any pre-trained LoRA, including its null space components.
- **vs Knowledge Distillation**: KD requires training data and forward passes. ProLoRA is a pure mathematical operation with a closed-form solution.
- **vs Wang et al. (Synthetic Data Transfer)**: Their method fine-tunes transfer using synthetic data, still requiring computational resources. ProLoRA is entirely training-free.
- This approach can be generalized to the NLP field—enabling training-free migration of LLM LoRAs across different model versions.

## Limitations & Future Work
- The choice of effective rank for SVD (threshold of 0.8) might not be optimal, and different layers may require different thresholds.
- Feasibility of cross-architecture transfer (e.g., UNet to DiT) has not been validated.
- Multi-LoRA composition transfer (e.g., style + concept simultaneously) remains unexplored.
- When the gap between source and target models is excessively large (e.g., major differences in training datasets), subspace similarity might drop and degrade migration quality.
- Transfer of QLoRA (Quantized LoRA) has not been tested.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of decomposing and migrating the subspace and null space is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across three LoRA types and multiple model pairs.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations.
- Value: ⭐⭐⭐⭐⭐ Solves a practical pain point in the LoRA ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](../../NeurIPS2025/image_generation/gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)
- [\[ICML 2025\] Reimagining Parameter Space Exploration with Diffusion Models](reimagining_parameter_space_exploration_with_diffusion_models.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](../../NeurIPS2025/image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)
- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](../../NeurIPS2025/image_generation/semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
