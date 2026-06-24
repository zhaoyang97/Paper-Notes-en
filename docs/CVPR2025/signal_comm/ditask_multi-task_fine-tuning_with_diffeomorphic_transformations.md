---
title: >-
  [Paper Note] DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations
description: >-
  [CVPR 2025][Signal & Communication][Diffeomorphic Transformations] Proposes DiTASK, which utilizes continuous piecewise-affine (CPAB) diffeomorphic transformations to smoothly transform the singular values of pretrained weight matrices while keeping the singular vectors unchanged. It achieves full-rank update multi-task fine-tuning with only about 32 parameters per layer, outperforming MTLoRA by 26.27% relative improvement with 75% fewer parameters on PASCAL MTL.
tags:
  - "CVPR 2025"
  - "Signal & Communication"
  - "Diffeomorphic Transformations"
  - "Singular Value Preservation"
  - "Multi-Task Learning"
  - "PEFT"
  - "Vision Transformer"
date: 2026-05-08
content_hash: e2446de884f72f94
---

# DiTASK: Multi-Task Fine-Tuning with Diffeomorphic Transformations

**Conference**: CVPR 2025  
**arXiv**: [2502.06029](https://arxiv.org/abs/2502.06029)  
**Code**: Yes (see paper)  
**Area**: Multi-Task Learning / Parameter-Efficient Fine-Tuning  
**Keywords**: Diffeomorphic Transformations, Singular Value Preservation, Multi-Task Learning, PEFT, Vision Transformer

## TL;DR
Proposes DiTASK, which utilizes continuous piecewise-affine (CPAB) diffeomorphic transformations to smoothly transform the singular values of pretrained weight matrices while keeping the singular vectors unchanged. It achieves full-rank update multi-task fine-tuning with only about 32 parameters per layer, outperforming MTLoRA by 26.27% relative improvement with 75% fewer parameters on PASCAL MTL.

## Background & Motivation

**Background**: Parameter-efficient fine-tuning (PEFT) methods like LoRA perform well in single-task fine-tuning but underperform in multi-task learning (MTL). The low-rank updates of LoRA force multiple tasks to compete within the same restricted subspace, leading to task interference.

**Limitations of Prior Work**: Methods like LoRA directly modify the singular vectors of the weight matrix, disrupting the feature space structure learned by the pretrained model. In MTL, the optimal subspaces for different tasks can be entirely different, and forcing them to share the same low-rank space leads to performance degradation. MTLoRA models each task independently but fails to exploit synergistic effects across tasks.

**Key Challenge**: The trade-off between parameter efficiency and multi-task performance—reducing trainable parameters exacerbates the task interference problem.

**Goal**: How to achieve non-interfering multi-task adaptation with minimal parameters while preserving the structure of the pretrained feature space.

**Key Insight**: Analyzing the weight matrix from the perspective of SVD decomposition—singular vectors encode feature directions, while singular values encode feature intensities. Robust adaptation can be achieved by keeping the singular vectors unchanged (preserving feature directions) and adjusting only the singular values via diffeomorphic transformations (modulating feature intensities).

**Core Idea**: Utilizing diffeomorphic transformations to adjust only singular values while keeping singular vectors unchanged, achieving full-rank multi-task adaptation with only ~32 parameters per layer.

## Method

### Overall Architecture
Based on Swin Transformer, SVD is applied to each pretrained weight matrix $W = U\Sigma V^\top$. $U$ and $V$ are frozen, and a CPAB diffeomorphism $f^\theta$ is used to transform the singular values to obtain $W_A = U \cdot \text{diag}(f^\theta(\sigma_1), ..., f^\theta(\sigma_p)) \cdot V^\top$. Each layer contains two sets of transformation parameters: the joint adaptation $\theta_j$ (learning task synergies) and the task-specific adaptation $\theta_k$ (learning task differences).

### Key Designs

1. **CPAB Diffeomorphic Singular Value Transformation**:

    - Function: Achieving full-rank weight updates with minimal parameters
    - Mechanism: CPAB partitions a closed interval into $\mathcal{N}_P$ segments, defining a piecewise-affine velocity field on each segment, which is integrated to obtain a smooth, invertible, and monotonic diffeomorphic mapping $f^\theta$. Applying this to singular values $\sigma_i \mapsto f^\theta(\sigma_i)$ requires only $\mathcal{N}_P - 1 \approx 32$ parameters. The monotonicity of the diffeomorphism ensures that the transformed singular values preserve their original relative ranking (retaining the relative relationships of feature importance).
    - Design Motivation: LoRA uses $O(r(c_1+c_2))$ parameters for low-rank updates, whereas DiTASK utilizes $O(\mathcal{N}_P)$ parameters for full-rank updates, requiring significantly fewer parameters without disrupting the feature space.

2. **Joint + Task-Specific Dual-Path Adaptation**:

    - Function: Simultaneously learning task synergies and task uniqueness.
    - Mechanism: In each Swin Transformer stage, joint transformation parameters $\theta_j$ (shared across all tasks) are applied to all blocks except the last one, which utilizes task-specific parameters $\theta_k$. The joint parameters capture common adaptations across tasks, while the task-specific parameters capture individual differences.
    - Design Motivation: The purely task-independent scheme of MTLoRA cannot exploit complementary information across tasks, whereas a purely shared scheme would lead to interference.

3. **Feature Space Preservation**:

    - Function: Serving as an implicit regularization to prevent catastrophic forgetting.
    - Mechanism: Singular vectors $U$ and $V$ are entirely frozen, transforming only the singular values. Image denoising experiments visually validate that preserving singular vectors while modulating only the singular values yields significantly better reconstruction quality (PSNR) than LoRA-style low-rank modifications.
    - Design Motivation: Singular vectors encode the feature directions learned during pretraining and are the portions that should least be modified.

### Loss & Training
Standard weighted multi-task loss is used: $\min_\Theta \sum_k \lambda_k \mathcal{L}_k$. Only the CPAB velocity field parameters $\theta$ are optimized, while all other parameters are frozen.

## Key Experimental Results

### Main Results

| Method | SemSeg (mIoU↑) | Human Parts (mIoU↑) | Saliency (mIoU↑) | Normals (rmse↓) | Δm (%) | Params (M) |
|------|---------------|---------------------|------------------|----------------|--------|---------|
| Single-Task Full FT | 67.21 | 61.93 | 62.35 | 17.97 | 0.00 | 112.62 |
| MTLoRA (r=64) | 67.90 | 59.84 | 65.40 | 16.60 | +2.55 | 6.40 |
| **DiTASK (Single-Task)** | **72.20** | **62.33** | **65.70** | **16.55** | **+5.33** | **1.60** |
| **DiTASK (MTL)** | 69.66 | 62.02 | 65.00 | 17.10 | +3.22 | 1.61 |

### Ablation Study

| Configuration | Description |
|------|------|
| Preserving Singular Vectors vs. Not Preserving | PSNR is significantly higher when singular vectors are preserved (validated via image denoising) |
| Joint + Task-Specific vs. Joint-Only | The dual-path design outperforms either single-path variant |
| $\mathcal{N}_P = 32$ | Default setting; increasing this value yields no significant improvement |

### Key Findings
- DiTASK single-task fine-tuning outperforms full fine-tuning (112.62M) by a Δm of 5.33% using only 1.60M parameters, showing extreme parameter efficiency.
- Compared to MTLoRA (r=64, 6.40M), DiTASK MTL (1.61M) achieves a +0.67% Δm improvement with 75% fewer parameters.
- Full-rank updates (achieved via diffeomorphic transformations) are key to outperforming low-rank methods—low-rank constraints are particularly detrimental in MTL.

## Highlights & Insights
- **Mathematical Elegance**: Starting from an SVD perspective, it uses diffeomorphic transformations to preserve singular vectors while transforming singular values, offering clear theory and simple implementation.
- **Extreme Parameter Efficiency**: Full-rank updates can be performed with around 32 parameters per layer, which is an order of magnitude smaller than LoRA (rank=4).
- **Transferable Concept**: The paradigm of preserving feature directions and adjusting feature intensities can be generalized to any scenario requiring efficient adaptation of pretrained models.

## Limitations & Future Work
- Validated only on two dense prediction benchmarks, PASCAL MTL and NYUD, without testing on classification or generation tasks.
- The CPAB transformation applies the same function to all singular values, preventing independent transformations for individual values.
- The architecture design is Swin Transformer-specific; its effectiveness on other backbones such as ViT and DINOv2 is not yet validated.

## Related Work & Insights
- **vs. LoRA/MTLoRA**: LoRA performs low-rank updates which might disrupt singular vectors, whereas DiTASK performs full-rank updates while preserving singular vectors, proving superior in both theory and experiments.
- **vs. SVFT**: SVFT sparsifies singular values, while DiTASK smoothly transforms singular values using diffeomorphisms, maintaining the order of feature importance.
- **vs. Adapter**: Adapters insert extra modules that increase inference overhead, whereas DiTASK directly modifies weights with zero extra inference cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of SVD and diffeomorphic transformations is highly novel and theoretically profound.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thoroughly validated on PASCAL MTL, but the task types are relatively single.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations.
- Value: ⭐⭐⭐⭐ Provides a new paradigm for MTL fine-tuning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](../../ECCV2024/signal_comm/pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[AAAI 2026\] Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion](../../AAAI2026/signal_comm/text-guided_channel_perturbation_and_pretrained_knowledge_integration_for_unifie.md)
- [\[CVPR 2025\] Breaking the Low-Rank Dilemma of Linear Attention](breaking_the_low-rank_dilemma_of_linear_attention.md)
- [\[CVPR 2025\] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention](abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a.md)

</div>

<!-- RELATED:END -->
