---
title: >-
  [Paper Note] KeepLoRA: Continual Learning with Residual Gradient Adaptation
description: >-
  [ICLR 2026][Multimodal VLM][Continual Learning] By analyzing the SVD decomposition of pretrained model weights, this paper identifies that general knowledge is encoded in the principal subspace while domain-specific know…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Continual Learning"
  - "LoRA"
  - "Gradient Projection"
  - "Subspace Constraint"
  - "Vision-Language Models"
date: 2026-05-08
content_hash: a681fe82ed516550
---

# KeepLoRA: Continual Learning with Residual Gradient Adaptation

**Conference**: ICLR 2026
**arXiv**: [2601.19659](https://arxiv.org/abs/2601.19659)  
**Code**: [GitHub](https://github.com/MaolinLuo/KeepLoRA)  
**Area**: Multimodal VLM
**Keywords**: Continual Learning, LoRA, Gradient Projection, Subspace Constraint, Vision-Language Models

## TL;DR

By analyzing the SVD decomposition of pretrained model weights, this paper identifies that general knowledge is encoded in the principal subspace while domain-specific knowledge resides in the residual subspace. KeepLoRA is proposed to constrain LoRA updates for new tasks within the residual subspace, while using gradient information for initialization to preserve plasticity, achieving an optimal balance among forward stability, backward stability, and plasticity in continual learning.

## Background & Motivation

Pretrained vision-language models (VLMs) face three competing objectives in continual learning:

**Forward Stability**: Preserving pretrained general knowledge and zero-shot transfer capability.

**Backward Stability**: Avoiding forgetting of previously learned tasks.

**Plasticity**: Effectively learning new tasks.

Existing methods each have limitations: reference-data regularization (e.g., ZSCL) relies on external data with high computational overhead; architecture expansion (e.g., Prompt Pool, MoE Adapters) increases inference cost and does not truly integrate knowledge into the model; existing LoRA-based continual learning methods (O-LoRA, InfLoRA, SD-LoRA) lack explicit protection of pretrained general knowledge.

**Key Finding**: By applying SVD decomposition to pretrained weight matrices, the authors find that the principal subspace (directions corresponding to large singular values) primarily encodes general knowledge (robust on general datasets), while the residual subspace (directions of small singular values) encodes domain-specific knowledge (whose removal causes sharp performance drops on specific datasets). This finding directly motivates the method design: new task updates should be constrained to the residual subspace.

## Method

### Overall Architecture

KeepLoRA improves upon standard LoRA by constructing a **unified principal subspace** to protect pretrained and previously learned task knowledge, then restricting LoRA updates for new tasks to the orthogonal complement of this subspace (the residual subspace). Task gradient information is further used to initialize LoRA, preserving plasticity.

### Key Designs

#### 1. Extraction of Pretrained Knowledge Subspace

For each weight matrix $\mathbf{W} \in \mathbb{R}^{d_{in} \times d_{out}}$ subject to updates, SVD decomposition $\mathbf{W} = \mathbf{U}\mathbf{S}\mathbf{V}^\top$ is applied. The top $p$ left singular vectors form the principal subspace $\mathbf{W}_p = \mathbf{U}_{:,1:p}$, satisfying an energy constraint:

$$\|\mathbf{W}_p\|_F^2 \geq \epsilon_w \|\mathbf{W}\|_F^2$$

where $\epsilon_w \in (0,1)$ controls the proportion of retained energy. Experiments show that retaining only a small number of principal components suffices to cover general knowledge.

#### 2. Maintenance of Learned Task Knowledge Subspace

To prevent forgetting of learned tasks, the principal feature directions of each task are extracted. After learning task $t$, the task's feature space (with projections onto the principal subspace and historical task directions removed) is computed as:

$$\hat{\mathbf{X}}_t = \mathbf{X}_t - \mathbf{W}_p \mathbf{W}_p^\top \mathbf{X}_t - \mathbf{M}_{t-1} \mathbf{M}_{t-1}^\top \mathbf{X}_t$$

SVD is applied to $\hat{\mathbf{X}}_t$ to extract the top $m$ principal singular vectors, updating the direction matrix: $\mathbf{M}_t = [\mathbf{M}_{t-1}, \mathbf{V}_{t(:,1:m)}]$, where the number of retained vectors is dynamically determined by an energy threshold $\epsilon_f$.

#### 3. Unified Principal Subspace

The pretrained principal subspace $\mathbf{W}_p$ and the task direction matrix $\mathbf{M}_t$ both operate in the same $d_{in}$-dimensional feature space and are unified as:

$$\mathbf{M}_t' = [\mathbf{W}_p, \mathbf{M}_t]$$

The total number of vectors is bounded above by $d_{in}$, keeping storage overhead manageable. All updates for new tasks must be orthogonal to $\mathbf{M}_{t-1}'$.

#### 4. Gradient-Guided LoRA Initialization (Preserving Plasticity)

The first-step training gradient $\mathbf{G}_t = \nabla_{\mathbf{W}} \mathcal{L}(\mathbf{W}; \mathcal{D}^t)$ is used to initialize LoRA. The gradient is first projected onto the residual subspace:

$$\hat{\mathbf{G}}_t = \underbrace{\mathbf{G}_t}_{\text{Plasticity}} - \underbrace{\mathbf{W}_p \mathbf{W}_p^\top \mathbf{G}_t - \mathbf{M}_{t-1} \mathbf{M}_{t-1}^\top \mathbf{G}_t}_{\text{Forward + Backward Stability}}$$

SVD is applied to $\hat{\mathbf{G}}_t$, and the top $r$ components are used to initialize LoRA:
$$\mathbf{A} = \mathbf{U}_{:,1:r}, \quad \mathbf{B} = \mathbf{S}_{1:r} \mathbf{V}_{:,1:r}^\top$$

$\mathbf{A}$ is frozen during training; only $\mathbf{B}$ is optimized. Since the initial $\frac{\alpha}{r}\mathbf{AB} \neq 0$, the original parameters are adjusted to $\mathbf{W}' = \mathbf{W} - \frac{\alpha}{r}\mathbf{AB}$.

#### 5. Theoretical Guarantee

**Proposition 3.1** proves that freezing $\mathbf{A}$ and optimizing only $\mathbf{B}$ is equivalent to constraining gradient descent within the subspace $\text{span}(\mathbf{A})$:

$$\Delta\mathbf{W} = \frac{\alpha}{r}\mathbf{A}\Delta\mathbf{B} = -c\mathbf{A}\mathbf{A}^\top\mathbf{G}_t$$

$\mathbf{A}\mathbf{A}^\top$ acts as an orthogonal projection operator, automatically confining all updates within $\text{span}(\mathbf{A})$.

### Loss & Training

Training uses only the standard classification loss $\mathcal{L}_{\text{cls}}(\mathbf{B}_t)$, without additional regularization terms or reference data. Upon the arrival of each new task:
1. Compute the first-step gradient, project it onto the residual subspace, and initialize LoRA via SVD.
2. Freeze $\mathbf{A}$ and train $\mathbf{B}$.
3. After training, merge LoRA into the main weights: $\mathbf{W} = \mathbf{W}' + \frac{\alpha}{r}\mathbf{AB}$.
4. Extract the principal feature directions of the task and update the unified principal subspace.

## Key Experimental Results

### Main Results

**MTIL Setting (11-dataset sequence, CLIP ViT-B/16)**:

| Method | Preserves Architecture | No Extra Data | Transfer↑ |
|--------|----------------------|---------------|----------|
| Zero-shot | ✓ | ✓ | 65.4 |
| ZSCL | ✓ | ✗ | 68.1 |
| O-LoRA | ✓ | ✓ | 66.5 |
| InfLoRA | ✓ | ✓ | 67.4 |
| SD-LoRA | ✓ | ✓ | 67.1 |
| **KeepLoRA** | ✓ | ✓ | **69.0** |
| MoE-Adapters | ✗ | ✗ | 68.9 |
| IAP | ✗ | ✓ | 69.2 |
| **KeepLoRA+** | ✗ | ✓ | **69.9** |

KeepLoRA achieves the best Transfer performance while preserving the original architecture and requiring no external data.

**Per-Dataset Transfer Comparison**:

| Method | Aircraft | DTD | EuroSAT | Flowers | OxfordPet | Cars |
|--------|----------|-----|---------|---------|-----------|------|
| O-LoRA | 80.8 | 44.5 | 49.8 | 67.5 | 88.7 | 56.1 |
| InfLoRA | 84.3 | 44.3 | 50.6 | 68.2 | 88.7 | 57.8 |
| **KeepLoRA** | 84.6 | **45.9** | **54.3** | **70.1** | **90.3** | **59.5** |

### Ablation Study

Contribution of individual components:

1. **Removing pretrained principal subspace protection ($\mathbf{W}_p$)**: Forward stability degrades significantly, with loss of general task transfer capability.
2. **Replacing gradient initialization with random initialization**: Plasticity decreases, with degradation in the Last metric.
3. **Removing task direction matrix ($\mathbf{M}_t$)**: Backward stability degrades, with lower accuracy on earlier tasks.
4. **Effect of energy threshold $\epsilon_w$**: Too low leads to insufficient protection; too high shrinks the residual space, impairing plasticity.

### Key Findings

1. **Separation of knowledge encoding between principal and residual subspaces**: Reconstructing weights using only the top $p$ principal singular components leaves performance on general datasets (ImageNet, CIFAR100, etc.) nearly unchanged, while performance on domain-specific datasets (Aircraft, DTD, EuroSAT, etc.) drops sharply.
2. **Equivalence of gradient projection**: Freezing $\mathbf{A}$ and training $\mathbf{B}$ is mathematically equivalent to performing gradient descent within $\text{span}(\mathbf{A})$.
3. **Compactness of the unified subspace**: The total size is bounded above by $d_{in}^2$ and does not grow unboundedly with the number of tasks.
4. **Validation on LLaVA**: KeepLoRA achieves state-of-the-art performance not only on the CLIP dual-encoder model but also on the encoder-decoder architecture of LLaVA.

## Highlights & Insights

- **Elegance of discovery-driven design**: The method is naturally derived from the empirical finding that general knowledge resides in the principal subspace and domain-specific knowledge in the residual subspace, forming a clear and complete logical chain.
- **Unified three-objective framework**: A single formulation simultaneously addresses all three objectives — gradient information preserves plasticity, subtraction of the principal subspace projection ensures forward stability, and subtraction of the task direction projection ensures backward stability.
- **Zero inference overhead**: LoRA can be merged into the original weights after training, adding no extra inference parameters or computation.
- **No external data required**: The method does not rely on reference datasets or generative models, making it more practical for real-world deployment.

## Limitations & Future Work

1. As the task sequence grows, the residual subspace progressively shrinks, potentially degrading plasticity in long-sequence scenarios.
2. The energy thresholds $\epsilon_w$ and $\epsilon_f$ require manual specification; adaptive determination strategies warrant further investigation.
3. The computational cost of applying SVD to each weight matrix may be substantial for large models.
4. Validation is limited to classification tasks; the effectiveness on generative tasks (e.g., VQA, image captioning) remains to be explored.
5. The unified principal subspace assumes that task feature directions are orthogonal, an assumption that may not hold when the number of tasks is very large.

## Related Work & Insights

KeepLoRA shares lineage with GPM (Gradient Projection Memory), with the key distinction being the explicit protection of the pretrained principal subspace. Unlike InfLoRA, which only constrains feature directions to be orthogonal, KeepLoRA constrains both gradient directions and initialization directions to lie within the residual subspace. The idea of gradient-guided initialization is generalizable to other parameter-efficient fine-tuning scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The finding of separated knowledge encoding between principal and residual subspaces is insightful.
- **Technical Quality**: ⭐⭐⭐⭐⭐ — Theoretical proofs are complete and mathematical derivations are rigorous.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple settings and datasets with comprehensive ablation studies.
- **Practicality**: ⭐⭐⭐⭐⭐ — No external data required, zero inference overhead, and deployment-friendly.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, though the dense notation requires careful reading.
- **Overall**: ⭐⭐⭐⭐ (8.5/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Enhanced Continual Learning of Vision-Language Models with Model Fusion](enhanced_continual_learning_of_vision-language_models_with_model_fusion.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](../../NeurIPS2025/multimodal_vlm/continual_multimodal_contrastive_learning.md)
- [\[CVPR 2026\] BriMA: Bridged Modality Adaptation for Multi-Modal Continual Action Quality Assessment](../../CVPR2026/multimodal_vlm/brima_bridged_modality_adaptation_for_multi-modal_continual_action_quality_asses.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2026/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](../../CVPR2026/multimodal_vlm/residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)

</div>

<!-- RELATED:END -->
