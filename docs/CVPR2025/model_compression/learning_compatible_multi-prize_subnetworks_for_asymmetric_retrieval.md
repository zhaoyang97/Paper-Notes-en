---
title: >-
  [Paper Note] PrunNet: Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval
description: >-
  [CVPR 2025][Model Compression][Prunable Network] The authors propose PrunNet (Prunable Network). By learning importance scores for each weight and incorporating conflict-aware gradient integration, PrunNet trains a unified model capable of generating compatible subnetworks at any capacity (20%-100%). It achieves a 46.29 mAP on GLDv2, surpassing the dense network baseline, while ensuring feature compatibility across subnetworks of all capacities.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Prunable Network"
  - "Asymmetric Retrieval"
  - "Subnetwork Compatibility"
  - "Conflict-aware Gradient"
  - "Post-training Pruning"
date: 2026-05-08
content_hash: faa67d4f1edc9a1d
---

# PrunNet: Learning Compatible Multi-Prize Subnetworks for Asymmetric Retrieval

**Conference**: CVPR 2025  
**arXiv**: [2504.11879](https://arxiv.org/abs/2504.11879)  
**Code**: [https://github.com/Bunny-Black/PrunNet](https://github.com/Bunny-Black/PrunNet)  
**Area**: Model Compression / Asymmetric Retrieval  
**Keywords**: Prunable Network, Asymmetric Retrieval, Subnetwork Compatibility, Conflict-aware Gradient, Post-training Pruning

## TL;DR

The authors propose PrunNet (Prunable Network). By learning importance scores for each weight and incorporating conflict-aware gradient integration, PrunNet trains a unified model capable of generating compatible subnetworks at any capacity (20%-100%). It achieves a 46.29 mAP on GLDv2, surpassing the dense network baseline, while ensuring feature compatibility across subnetworks of all capacities.

## Background & Motivation

**Background**: Asymmetric retrieval deploys large models on the server side for offline indexing and small models on the edge side for online querying. The features of both models must be compatible (i.e., they can be matched within the same space).

**Limitations of Prior Work**: Existing methods (such as SFSC) require training a compatible model separately for each capacity level, requiring $N$ training processes for $N$ capacities. Deploying to a new device necessitates retraining.

**Key Challenge**: Subnetworks with different capacities require compatible feature spaces, but capacity differences lead to varying learned representations—where global and local optima may conflict.

**Key Insight**: Learnable scores are used to label the importance of each weight, and a greedy pruning strategy is applied to retain the top-$c$% connections. Gradient conflict projection is introduced to resolve optimization conflicts among subnetworks of different capacities.

**Core Idea**: Learnable importance scores + conflict-aware gradients + compatibility constraints = one-time training for subnetworks of arbitrary capacity.

## Method

### Key Designs

1. **Learnable Weight Importance Scores**: Each weight $w_{ij}^l$ is associated with a score $s_{ij}^l$. During pruning, connections with scores in the top-$c_i$% are retained. These scores are optimized alongside the weights during training.

2. **Conflict-Aware Gradient Integration**: Loss gradients from subnetworks of different capacities may conflict. When $\mathbf{g}_i \cdot \mathbf{g}_j < 0$, $\mathbf{g}_i$ is projected onto the orthogonal direction of $\mathbf{g}_j$: $\hat{\mathbf{g}}_i = \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{|\mathbf{g}_j|^2}\mathbf{g}_j$

3. **Compatibility Constraint**: $\mathcal{L}_{comp} = \|f_{dense}(x) - f_i(x)\|^2$, which ensures that the features of small-capacity subnetworks are aligned with the dense network.

### Loss & Training

$\mathcal{L} = \sum_i \mathcal{L}_{CE}(f_i(x), y) + \lambda \mathcal{L}_{comp}$. Iterative pruning (IP) outperforms one-shot pruning (OSP) because smaller networks inherit weights from larger ones. Adaptive BN post-processing is required.

## Key Experimental Results

### Main Results

GLDv2 landmark retrieval mAP (20%/40%/60%/80%/100% capacity):

| Method | 20% | 60% | 100% |
|------|-----|-----|------|
| SFSC | 42.45 | 43.72 | 44.47 |
| **PrunNet** | **45.61** | **46.05** | **46.29** |

### Ablation Study

| Configuration | Effect |
|------|------|
| Without conflict projection | Performance degradation |
| One-shot pruning (OSP) | Inferior to iterative pruning (IP) |
| Without compatibility constraint | Incompatible subnetwork features |

### Key Findings
- PrunNet outperforms independently trained baselines at all capacity levels, demonstrating that joint training can be superior to independent training.
- Conflict projection is critical; without it, optimization processes of different capacities interfere with each other.
- The 20% capacity variant still achieves a 45.61 mAP (compared to 44.47 for dense SFSC), even surpassing the dense baseline after pruning.

## Highlights & Insights
- **Train once, deploy infinitely**: No need to retrain for new devices; pruning can be performed directly based on target capacity.
- **Generality of conflict projection**: This multi-task gradient conflict resolution strategy can be generalized to any multi-objective optimization problem.

## Limitations & Future Work
- Unstructured pruning does not achieve ideal acceleration on certain hardware.
- The hyperparameter $\alpha$ requires tuning.
- BN layer statistics differ across subnetworks, requiring extra Adaptive BN.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of prunable networks, conflict projection, and compatibility constraints is effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple datasets including GLDv2, In-Shop, and VeRi.
- Writing Quality: ⭐⭐⭐⭐ Clear.
- Value: ⭐⭐⭐⭐ Offers direct utility for multi-device deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] TADFormer: Task-Adaptive Dynamic Transformer for Efficient Multi-Task Learning](tadformer_task-adaptive_dynamic_transformer_for_efficient_multi-task_learning.md)
- [\[NeurIPS 2025\] Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization](../../NeurIPS2025/model_compression/find_your_needle_small_object_image_retrieval_via_multi-object_attention_optimiz.md)
- [\[CVPR 2025\] Understanding Multi-layered Transmission Matrices](understanding_multi-layered_transmission_matrices.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](../../ACL2025/model_compression/more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)
- [\[ICML 2025\] GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration](../../ICML2025/model_compression/gptaq_efficient_finetuning-free_quantization_for_asymmetric_calibration.md)

</div>

<!-- RELATED:END -->
