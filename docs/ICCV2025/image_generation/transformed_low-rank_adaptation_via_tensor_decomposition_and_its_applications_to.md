---
title: >-
  [Paper Note] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models
description: >-
  [ICCV 2025][Image Generation][Parameter-Efficient Fine-Tuning] This paper proposes TLoRA, which decomposes the fine-tuning of pretrained weights into a **Transform** adaptation and a **Residual** adaptation…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Parameter-Efficient Fine-Tuning"
  - "LoRA"
  - "Tensor Decomposition"
  - "Text-to-Image"
  - "Subject-Driven Generation"
date: 2026-05-08
content_hash: 16284f8d67a36224
---

# Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models

**Conference**: ICCV 2025  
**arXiv**: [2501.08727](https://arxiv.org/abs/2501.08727)  
**Code**: [https://github.com/taozerui/tlora_diffusion](https://github.com/taozerui/tlora_diffusion)  
**Area**: Image Generation  
**Keywords**: Parameter-Efficient Fine-Tuning, LoRA, Tensor Decomposition, Text-to-Image, Subject-Driven Generation

## TL;DR

This paper proposes TLoRA, which decomposes the fine-tuning of pretrained weights into a **Transform** adaptation and a **Residual** adaptation, parameterized respectively via Tensor Ring Matrix (TRM) and Tensor Ring (TR) decompositions. On SDXL, TLoRA achieves highly parameter-efficient fine-tuning with only 0.4M parameters while outperforming LoRA and other baselines.

## Background & Motivation

Parameter-efficient fine-tuning (PEFT) is a core technique for personalizing large-scale text-to-image models. LoRA and its variants are widely adopted due to their simplicity and efficiency, with the central assumption that the fine-tuning weight update $\Delta = W_* - W_0$ exhibits a low-rank structure. In practice, however, fine-tuning weights of large foundation models often violate this assumption, leading to significant approximation errors for LoRA on challenging tasks.

**Limitations of Prior Work**:
- **LoRA**: The low-rank assumption may not hold in practice, resulting in large approximation errors.
- **OFT (Orthogonal Fine-Tuning)**: Uses block-diagonal matrix parameterization, which is highly sparse and restricts inter-neuron information flow.
- **DoRA**: Although it introduces a magnitude-direction decomposition (implicitly a diagonal transform), the expressiveness of diagonal matrices is limited.
- **Fixed-transform methods (e.g., FouRA)**: Employ predefined fixed transforms (e.g., DFT) that cannot adapt to different models and tasks.

**Core Insight**: The authors observe that if a learnable linear transform $T$ is first applied to the pretrained weight $W_0$ to maximally align it with the target weight $W_*$, the rank of the residual $\Delta'_* = W_* - W_0 T$ is substantially reduced. This enables a more compact structure to approximate the residual, simultaneously achieving higher parameter efficiency and better performance.

**Key Insight**: The paper proposes a unified "Transform + Residual" adaptation framework $y' = (W_0 T + \Delta) x$, and leverages tensor decomposition techniques (TRM and TR) to efficiently parameterize both components.

## Method

### Overall Architecture

The core formulation of TLoRA is:

$$y' = (W_0 T + \Delta) x$$

where $T$ is the transform adaptation (parameterized via TRM) and $\Delta$ is the residual adaptation (parameterized via TR). The two components work in concert: $T$ rotates/projects the pretrained weights toward the target weight space, while $\Delta$ compensates for the remaining low-rank discrepancy.

### Key Designs

1. **Tensor Ring Matrix (TRM) Transform Adaptation**:

    - **Function**: Parameterizes the transform matrix $T$ as a full-rank, dense, yet parameter-efficient structure.
    - **Mechanism**: Tensorizes the $I \times I$ matrix $T$ into $D$ fourth-order core tensors $\mathcal{A}^d \in \mathbb{R}^{I_d \times I_d \times R \times R}$, with matrix entries computed as $T[\overline{i_1 \cdots i_D}, \overline{j_1 \cdots j_D}] = \text{tr}(\mathcal{A}^1[i_1,j_1,:,:] \cdots \mathcal{A}^D[i_D,j_D,:,:])$.
    - **Space Complexity**: $\mathcal{O}(D I^{2/D} R^2)$, far less than the original $\mathcal{O}(I^2)$.
    - **Design Motivation**: TRM naturally represents full-rank dense matrices (when core tensors are dense and full-rank), resolving the sparsity issue of OFT's block-diagonal parameterization. Experiments (Fig. 3) confirm that TRM achieves significantly lower approximation error than BOFT under the same parameter budget.

2. **Tensor Ring (TR) Residual Adaptation**:

    - **Function**: Parameterizes the residual $\Delta$ with an extremely compact structure.
    - **Mechanism**: Decomposes $\Delta$ into $2D$ third-order core tensors $\mathcal{B}^d \in \mathbb{R}^{I_d \times R \times R}$ and $\mathcal{C}^d \in \mathbb{R}^{J_d \times R \times R}$.
    - **Space Complexity**: $\mathcal{O}(D I^{1/D} R^2)$, more compact than TRM.
    - **Design Motivation**: With the assistance of the transform, the rank of the residual is effectively reduced, making the more compact TR structure sufficient. Simulation experiments (Fig. 2c) show that the TRM+TR combination achieves comparable approximation error at extremely low parameter counts (< 10% of LoRA R=1).

3. **Zero Initialization Strategy**:

    - **Function**: Ensures the fine-tuned model is identical to the original model at the start of training.
    - **Mechanism**: The TRM component is initialized to the identity matrix (Proposition 1: set each core tensor slice $\mathcal{A}^d[:,:,r,r'] = I_{I_d}/R$); the first core tensor $\mathcal{B}^1$ of the TR component is initialized to zero, while the remaining cores are initialized with Gaussian noise under the $\mu P$ framework ($\sigma = \Theta(\sqrt{n\_out}/n\_in)$).
    - **Design Motivation**: Consistent with LoRA's zero initialization, this avoids corrupting pretrained model information. Prior TR-PEFT methods used random Gaussian initialization for all factors, leading to training instability.

### Loss & Training

- Optional regularization on the transform component: identity matrix regularization $\mathcal{R}_I$ or orthogonal regularization $\mathcal{R}_O$, both efficiently computable on small-sized core tensors (via Proposition 2) without explicitly constructing the full-size matrix.
- In practice, identity matrix regularization outperforms orthogonal regularization, consistent with findings in the OFT/BOFT literature.
- Training uses the AdamW optimizer with learning rates tuned from {5e-4, 1e-4, 5e-5}.

## Key Experimental Results

### Main Results (Controllable Generation, SD v1.5 + ControlNet)

| Method | Params (M) | L2I Error↓ | S2I mIoU↑ | S2I aACC↑ | C2I IoU↑ | C2I F1↑ |
|------|----------|-----------|----------|----------|---------|--------|
| LoRA r=4 | 0.80 | 5.32 | 27.72 | 64.99 | 0.180 | 0.305 |
| DoRA r=4 | 0.90 | 6.43 | 27.11 | 65.70 | 0.143 | 0.248 |
| OFT r=32 | 1.50 | 7.35 | 28.52 | 66.04 | 0.165 | 0.281 |
| BOFT(2,2) | 2.41 | 7.64 | 28.45 | 66.19 | 0.161 | 0.276 |
| BOFT(4,8) | 20.76 | 5.67 | 28.83 | **67.74** | — | — |
| **TLoRA*(2,4)** | **0.94** | **5.32** | **29.23** | 69.21 | **0.203** | **0.337** |
| TLoRA(2,6) | 0.40 | 5.84 | 27.03 | 65.15 | 0.184 | 0.310 |

### Ablation Study (Simulation — SDXL Inpaint Weight Approximation Error)

| Configuration | Approximation Error Trend | Notes |
|------|------------|------|
| LoRA | Low under low-rank assumption; high for fully fine-tuned weights | Effective only when assumption holds |
| OFT + LoRA | Comparable to LoRA; no improvement | Orthogonal transform does not change column space |
| TRM + LoRA | Significantly lower than LoRA | TRM transform effectively aligns weights |
| TRM + TR | Best under extremely low parameter budget | TR is more flexible than LoRA at very small budgets |
| TR (no transform) | Better than LoRA at very low parameter counts | Improvement diminishes as parameters increase |

### Key Findings
- TLoRA remains competitive with only 0.4M parameters (27% of LoRA R=1).
- The transform component (TRM) consistently improves performance, especially when target weights deviate from the low-rank assumption.
- Zero-initialized TR significantly outperforms randomly initialized TR (LoRETTA achieves an error as high as 118.14 on controllable generation).
- Tensor decomposition methods (LoRETTA, TLoRA) exhibit greater sampling diversity in subject-driven generation.

## Highlights & Insights
- The unified "Transform + Residual" framework subsumes DoRA, OFT, FouRA, and related methods (DoRA ≈ diagonal transform + LoRA; FouRA ≈ fixed DFT transform + LoRA).
- Simulation studies leveraging weight differences between existing pretrained and fine-tuned models (Fig. 2) provide intuitive empirical validation.
- TRM strikes an excellent balance between expressiveness and parameter efficiency as a compact parameterization of full-rank dense matrices.
- Initialization strategy is critical for high-order tensor structures; zero initialization combined with the $\mu P$ framework yields more stable training.

## Limitations & Future Work
- Validation is currently limited to SD v1.5 and SDXL; the method has not been tested on DiT-based architectures (e.g., Flux).
- Manual selection of the tensorization scheme (sub-index dimension partitioning) is required, and hyperparameter choices affect results.
- Theoretical guidance for selecting the ranks of the transform and residual components is lacking.
- Inference requires additional TRM matrix multiplication (though this can be precomputed and merged into the weights).

## Related Work & Insights
- The progression LoRA → DoRA → TLoRA traces a path from low-rank assumption → implicit sparse transform → explicit dense transform.
- Tensor decomposition has a rich foundation in model compression (TT, TR); extending it to PEFT is a promising direction.
- The principle of "finding a better projection space to reduce the rank of the residual" generalizes naturally to other model adaptation scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](../../NeurIPS2025/image_generation/gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](../../NeurIPS2025/image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[ICCV 2025\] Understanding Flatness in Generative Models: Its Role and Benefits](understanding_flatness_in_generative_models_its_role_and_benefits.md)
- [\[ICCV 2025\] AutoPrompt: Automated Red-Teaming of Text-to-Image Models via LLM-Driven Adversarial Prompts](autoprompt_automated_red-teaming_of_text-to-image_models_via_llm-driven_adversar.md)
- [\[ICCV 2025\] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](csd-var_content-style_decomposition_in_visual_autoregressive_models.md)

</div>

<!-- RELATED:END -->
