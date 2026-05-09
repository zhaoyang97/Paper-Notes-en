---
title: >-
  [Paper Note] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold
description: >-
  [NeurIPS 2025][Image Generation][LoRA] This paper proposes StelLA, which decomposes the LoRA adaptation matrix into a three-factor form $USV^\top$ and constrains $U$ and $V$ to the Stiefel manifold for Riemannian optimization, enabling explicit subspace learning during training. StelLA consistently outperforms existing LoRA variants across multiple downstream tasks.
tags:
  - NeurIPS 2025
  - Image Generation
  - LoRA
  - Stiefel Manifold
  - Subspace Learning
  - Riemannian Optimization
  - Three-Factor Decomposition
date: 2026-05-08
content_hash: 586ec745a509fc45
---

# StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold

**Conference**: NeurIPS 2025
**arXiv**: [2510.01938](https://arxiv.org/abs/2510.01938)
**Code**: [GitHub](https://github.com/SonyResearch/stella)
**Area**: Parameter-Efficient Fine-Tuning / Low-Rank Adaptation
**Keywords**: LoRA, Stiefel Manifold, Subspace Learning, Riemannian Optimization, Three-Factor Decomposition

## TL;DR

This paper proposes StelLA, which decomposes the LoRA adaptation matrix into a three-factor form $USV^\top$ and constrains $U$ and $V$ to the Stiefel manifold for Riemannian optimization, enabling explicit subspace learning during training. StelLA consistently outperforms existing LoRA variants across multiple downstream tasks.

## Background & Motivation

LoRA is the dominant method for parameter-efficient fine-tuning of large models, adapting pretrained weights by learning a low-rank matrix $BA^\top$. However, a performance gap remains between LoRA and full fine-tuning. Prior work such as PiSSA and MiLoRA attempts to improve LoRA initialization via SVD decomposition, but these methods only guide the beginning of training and have limited influence on subsequent optimization.

The authors identify a critical issue: **different heuristic subspace selection strategies** — whether to select principal or minor components, and whether to base selection on weights or gradients — yield conflicting conclusions, indicating that manual subspace selection is suboptimal. This naturally raises the question: can the optimal subspace be **learned directly during training**?

Furthermore, the two-factor decomposition $BA^\top$ in LoRA couples the input/output subspaces with the scaling factor, making structured geometric optimization difficult. Inspired by the SVD structure, the authors propose to decouple direction (subspace) from magnitude (scaling), consistent with the idea in DoRA of decomposing weights into magnitude and direction components.

## Method

### Overall Architecture

StelLA represents the low-rank adaptation of each linear layer in a three-factor form:

$$\tilde{W} = W + \frac{\alpha}{r} U S V^\top$$

where $U \in \text{St}(r,m)$ and $V \in \text{St}(r,n)$ are orthonormal bases for the output and input subspaces respectively (constrained to the Stiefel manifold), and $S \in \mathbb{R}^{r \times r}$ learns the mapping between the two subspaces. This design explicitly decouples subspace direction from scaling magnitude, enabling subspace optimization while preserving orthogonality.

### Key Designs

1. **Stiefel Manifold Constraint and Riemannian Optimization**: To ensure $U$ and $V$ remain column-orthogonal throughout training, they are constrained to the Stiefel manifold $\text{St}(k,n) = \{Y \in \mathbb{R}^{n \times k} \mid Y^\top Y = I_k\}$. Optimization proceeds in three steps: (a) converting the Euclidean gradient to the Riemannian gradient $\text{grad}_Y = \nabla_Y - Y(\nabla_Y)^\top Y$; (b) projecting the perturbed gradient produced by the optimizer back to the tangent space $\pi_Y(\Delta) = \Delta - Y \text{symm}(Y^\top \Delta)$; (c) retracting the updated point back to the manifold via polar decomposition $\rho_Y(\Delta) = \text{uf}(Y + \Delta)$. This design allows any existing Euclidean optimizer (e.g., Adam) to be seamlessly adapted into a Riemannian optimizer.

2. **Modular Geometric Optimization Design**: The algorithm is implemented via optimizer hooks: Riemannian gradient conversion as a pre-hook, and projection and retraction as post-hooks. Unlike existing Riemannian optimizers (e.g., Riemannian Adam), StelLA decouples geometric constraints from optimizer logic — it requires no modification of the optimizer's internal momentum or adaptive learning rate mechanisms, treating the optimizer's update direction as a perturbation of the Riemannian gradient and correcting it via projection. A batched SVD strategy (stacking $U$/$V$ matrices of the same shape across layers) achieves a 15–20× speedup.

3. **Gradient Scaling Strategy**: Since the columns of $U$ and $V$ are unit vectors, their element magnitudes are on the order of $1/\sqrt{m}$ and $1/\sqrt{n}$, respectively. When $m \neq n$ (e.g., in LLM FFN layers where the hidden dimension is expanded by a factor of 4), Adam's gradient normalization causes an imbalance in the learning rates of $U$ and $V$. StelLA compensates by multiplying the gradients of $U$ and $V$ by $\sqrt{d/m}$ and $\sqrt{d/n}$ respectively (where $d$ is the hidden dimension of the input token) before the projection step.

### Loss & Training

The training objective is the standard task loss $\mathcal{L}$, but the optimization paths differ: $U$ and $V$ are optimized under constraints on the Stiefel manifold, while $S$ is optimized without constraints in Euclidean space. Initialization uses random column-orthogonal matrices for $U$/$V$ and the identity matrix for $S$, requiring no modification of pretrained weights as in PiSSA.

## Key Experimental Results

### Main Results

**Commonsense Reasoning (LLaMA3-8B, rank=32)**

| Method | Params (%) | BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | Avg. |
|--------|-----------|-------|------|-----------|------------|-------|-------|------|
| LoRA | 0.700 | 75.16 | 88.14 | 95.41 | 86.74 | 90.84 | 78.70 | 85.27 |
| DoRA | 0.710 | 75.38 | 88.01 | 95.35 | 86.29 | 90.54 | 79.69 | 85.16 |
| ScaledAdamW | 0.700 | 75.24 | 88.57 | 95.81 | 85.11 | 91.09 | 80.55 | 85.40 |
| **StelLA** | **0.702** | **75.91** | **89.86** | **96.41** | **87.82** | **91.98** | **82.34** | **86.72** |

StelLA achieves approximately +1.3 percentage points improvement on both LLaMA2-7B and LLaMA3-8B.

**Text-to-Image Generation (SD 1.5, rank=4, FID↓)**

| Dataset | LoRA | DoRA | PiSSA | StelLA |
|---------|------|------|-------|--------|
| BarbieCore | 175.48 | 175.04 | 299.49 | **170.25** |
| Expedition | 156.34 | 155.80 | 291.22 | **146.12** |
| Hornify | 180.48 | 179.58 | 295.15 | **167.53** |

FID reduction reaches up to 12 points.

### Ablation Study

| Configuration | Avg. Accuracy | Notes |
|---------------|--------------|-------|
| StelLA (default) | 86.72 | Non-zero init + gradient scaling |
| Euclidean geometry (no orthogonality constraint) | 84.4 | Demonstrates necessity of orthogonal constraint |
| Quotient space geometry | 85.7 | Inferior to Stiefel product manifold |
| Zero initialization | 86.5 | Small $S$ leads to small $U$/$V$ gradients, slow convergence |
| Pseudo-zero initialization | 84.2 | Corrupts pretrained weights, worst performance |
| SVD-major initialization | 86.7 | Close to random init, showing geometric optimization can learn subspaces automatically |
| Without gradient scaling | 86.4 | Scaling provides +0.3 improvement |
| Polar retraction vs. exponential map | 86.72 vs 86.76 | Comparable performance; polar retraction is more efficient |

### Key Findings

- The Stiefel manifold constraint substantially outperforms unconstrained Euclidean three-factor decomposition (TriLoRA/MoSLoRA): 86.7 vs. 84.4
- StelLA is robust to initialization strategy: random initialization, SVD principal components, and SVD minor components yield comparable performance
- Additional parameter count increases by only $r^2$ (rank squared), which is negligible
- StelLA also achieves consistently superior results on image classification (ViT-Base/Large, 8 datasets)

## Highlights & Insights

- **Elegant design**: The three-factor decomposition combined with Stiefel manifold constraints forms a geometrically natural framework that directly embeds the SVD structure into the training process
- **Modularity**: Implemented via optimizer hooks, StelLA can be combined with any existing optimizer without the need to manually implement special optimizers such as Riemannian Adam
- **Batched SVD acceleration**: The bottleneck operation (polar decomposition) is accelerated 15–20× via cross-layer batching, addressing practical efficiency concerns
- The gradient scaling strategy, while simple, provides effective learning rate balancing for the asymmetric dimensions in FFN layers

## Limitations & Future Work

- The three-factor decomposition and retraction operations introduce additional computational overhead (partially mitigated by batched SVD)
- Integration with adaptive-rank methods such as AdaLoRA (which could constrain $S$ to a diagonal matrix) has not been explored
- Validation on 70B-scale models or other model families such as Mistral/LLaVA is absent
- Joint use with QLoRA (quantization) has not been explored

## Related Work & Insights

- StelLA can be combined with AdaLoRA's rank-adaptive strategy by constraining $S$ to a diagonal matrix, enabling dynamic rank adjustment via singular value pruning
- Orthogonality constraints may also offer potential benefits for adversarial robustness
- The geometric optimization framework opens possibilities for applying other manifold constraints (e.g., Grassmann manifold) in fine-tuning

## Rating

- **Novelty**: ⭐⭐⭐⭐ Introducing Stiefel manifold optimization into LoRA is original; while three-factor decomposition is not novel per se, the geometric constraint design is
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers NLU, NLG, visual classification, and image generation; ablations are comprehensive
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are clear, algorithmic descriptions are rigorous, and ablation design is convincing
- **Value**: ⭐⭐⭐⭐ Provides a plug-and-play improvement over LoRA; code is open-sourced with strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](../../ICCV2025/image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[NeurIPS 2025\] EEGReXferNet: A Lightweight Gen-AI Framework for EEG Subspace Reconstruction via Cross-Subject Transfer Learning and Channel-Aware Embedding](eegrexfernet_a_lightweight_gen-ai_framework_for_eeg_subspace_reconstruction_via_.md)
- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)
- [\[NeurIPS 2025\] Generative Model Inversion Through the Lens of the Manifold Hypothesis](generative_model_inversion_through_the_lens_of_the_manifold_hypothesis.md)

</div>

<!-- RELATED:END -->
