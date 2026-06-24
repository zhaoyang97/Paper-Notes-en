---
title: >-
  [Paper Note] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers
description: >-
  [CVPR 2025][Model Compression][Vision Transformer] HiAP proposes a multi-granular automatic pruning framework. By deploying learnable Gumbel-Sigmoid gates at both macro (attention heads, FFN blocks) and micro (intra-head dimensions, FFN neurons) levels, the framework automatically discovers the optimal subnetwork within a single-stage end-to-end training process, eliminating the need for manual importance ranking or post-hoc thresholding.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Vision Transformer"
  - "Structured Pruning"
  - "Gumbel-Sigmoid"
  - "Multi-Granularity"
  - "Neural Architecture Search"
date: 2026-05-08
content_hash: 765aee310c0b0abb
---

# HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers

**Conference**: CVPR 2025  
**arXiv**: [2603.12222](https://arxiv.org/abs/2603.12222)  
**Code**: To be confirmed  
**Area**: Model Compression  
**Keywords**: Vision Transformer, Structured Pruning, Gumbel-Sigmoid, Multi-Granularity, Neural Architecture Search

## TL;DR
HiAP proposes a multi-granular automatic pruning framework. By deploying learnable Gumbel-Sigmoid gates at both macro (attention heads, FFN blocks) and micro (intra-head dimensions, FFN neurons) levels, the framework automatically discovers the optimal subnetwork within a single-stage end-to-end training process, eliminating the need for manual importance ranking or post-hoc thresholding.

## Background & Motivation
**Background**: ViT pruning methods have achieved significant progress, but most methods operate at a single granularity. Micro-granularity pruning (e.g., ViT-Slim, which prunes intra-head dimensions and FFN channels) reduces FLOPs, while macro-granularity pruning (e.g., UPDP, which prunes entire blocks) reduces memory bandwidth overhead.

**Limitations of Prior Work**: (1) Single-granularity pruning cannot simultaneously optimize memory bandwidth (requiring macro pruning) and computational workload (requiring micro pruning); (2) existing differentiable search methods rely on post-hoc magnitude thresholding, requiring expert knowledge and multi-stage pipelines.

**Key Challenge**: The real bottleneck of modern hardware is often High Bandwidth Memory (HBM) access rather than pure computation. Even if micro-granularity pruning reduces FLOPs, retaining all layer structures still requires loading weight matrices and attention maps for every layer, leading to limited actual wall-clock acceleration.

**Goal**: How to enable the model to autonomously discover the optimal cross-granularity pruning strategy in a single-stage training process under hardware resource constraints?

**Key Insight**: Formulate pruning as a budget-aware learning problem and replace discrete decisions with Gumbel-Sigmoid continuous relaxations, allowing natural convergence to binary gates via temperature annealing.

**Core Idea**: Dual-level (macro + micro) Gumbel-Sigmoid gating + differentiable MACs cost modeling + structural feasibility constraints = single-stage automatic pruning.

## Method

### Overall Architecture
Given a pre-trained ViT (such as DeiT-Small) as input, a dual-layer gating mechanism is deployed for each Transformer block. The macro-gates control whether to retain entire attention heads ($g_{l,h}$) and FFN blocks ($b_l$), whereas the micro-gates control the intra-head dimensions ($d_{l,h,j}$) and FFN neurons ($c_{l,k}$) within the active structures. During training, the gates use Gumbel-Sigmoid continuous relaxation, with the temperature $\tau$ annealing from 2.0 to 0.5. Consequently, the gates naturally binarize into 0 or 1, allowing the weight matrices to be physically truncated to obtain a directly deployable compact subnetwork.

### Key Designs

1. **Hierarchical Gating**:

    - **Function**: Controls the preservation/discarding of heads/blocks at the macro level, and trims dimensions inside the surviving structures at the micro level.
    - **Mechanism**: $\text{Head}'_{l,h}(X) = g_{l,h} [\text{softmax}(\frac{QK^\top}{\sqrt{D_h}})(V \odot d_{l,h})]$, $\text{FFN}'_l(X) = b_l[(\phi(XW_1) \odot c_l)W_2]$. When the macro gate is 0, the entire head/block is completely bypassed; when active, the micro gates perform channel-wise masking on the surviving structure.
    - **Design Motivation**: Macro pruning saves memory bandwidth (bypassing the memory loading of entire heads or blocks), whereas micro pruning saves computational cost while preserving structural integrity. The two levels complement each other.

2. **Differentiable Cost Modeling**:

    - **Function**: Accurately decomposes MACs into a linear function of the gating variables.
    - **Mechanism**: $\mathbb{E}[C(\mathcal{G})] = \sum_{l,h}(C_1 \cdot \mathbb{E}[g_{l,h}] + C_2 \sum_j \mathbb{E}[g_{l,h} \cdot d_{l,h,j}]) + \sum_{l,k} C_3 \cdot \mathbb{E}[b_l \cdot c_{l,k}]$, decoupling the cost into $\mathcal{L}_{\text{macro}}$ (penalizing $C_1$) and $\mathcal{L}_{\text{micro}}$ (penalizing $C_2, C_3$), which are controlled by independent hyperparameters.
    - **Design Motivation**: Linear decomposition enables the optimizer to clearly attribute hardware penalties to individual structures, explicitly penalizing the structural overhead of empty heads.

3. **Feasibility Penalty**:

    - **Function**: Prevents layer collapse (where an entire layer is greedily pruned, shutting down gradient flow).
    - **Mechanism**: $\mathcal{L}_{f,\text{head}} = \sum_l \text{ReLU}(k_{\min} - \sum_h g_{l,h})^2$, which guarantees that at least $k_{\min}$ heads are preserved in each layer. Similar constraints ensure minimum FFN neurons and attention dimensions.
    - **Design Motivation**: This addresses a common failure mode in differentiable architecture search.

4. **Single-Stage Training + Temperature Annealing**:

    - Gumbel-Sigmoid: $\hat{z} = \sigma((\alpha + \epsilon)/\tau)$. In the early stages when $\tau$ is high, the gates behave like stochastic dropout, helping the network learn robust and distributed representations. As $\tau$ decreases, the gates naturally converge to binary values.
    - Once training is complete, the gates are binarized directly using a threshold of 0.5 to physically truncate the weight matrices, requiring no secondary fine-tuning.

### Loss & Training
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{macro}} \mathcal{L}_{\text{macro}} + \lambda_{\text{micro}} \mathcal{L}_{\text{micro}} + \mathcal{L}_{\text{feasibility}}$$

Where $\mathcal{L}_{\text{task}}$ contains cross-entropy and knowledge distillation losses ($\alpha_{KD}=0.7, T=4.0$).

## Key Experimental Results

### Main Results (ImageNet-1K, DeiT-Small)

| Method | Params (M) | FLOPs (G) | Top-1 Acc (%) | Δ Acc |
|------|-----------|-----------|---------------|-------|
| Dense Baseline | 22.1 | 4.6 | 79.85 | — |
| ViT-Slim | 15.6 | 3.1 | 79.90 | +0.05 |
| GOHSP | 14.4 | 3.0 | 79.98 | +0.13 |
| **HiAP** | 15.0 | 3.1 | 79.10 | -0.75 |
| ViT-Slim | 13.5 | 2.8 | 79.50 | -0.35 |
| **HiAP** | 12.3 | 2.5 | 77.95 | -1.90 |

### Ablation Study (CIFAR-10, ViT-Tiny)

| Method | MACs (M) | Reduction Rate | Acc (%) |
|------|---------|--------|---------|
| Dense | 174.0 | 0% | 90.50 |
| Uniform-Ratio | 116.6 | 33% | 86.63 |
| $\ell_1$-Structured (FFN) | 116.5 | 33% | 87.15 |
| **HiAP (Moderate)** | 116.3 | 33% | **87.56** |
| $\ell_1$-Structured (FFN) | 87.3 | 50% | 86.80 |
| **HiAP (Aggressive)** | 87.1 | 50% | **87.25** |

### Key Findings
- HiAP underperforms GOHSP and ViT-Slim in accuracy (by 0.75-0.88% under the 3.1G FLOPs configuration), but **eliminates the need for multi-stage pipelines**.
- Autonomously discovered pruning strategy: The model boldly prunes macro structures early on (reducing the number of heads from 6 to 2-4 in the first 10 epochs) and fine-tunes micro dimensions later.
- The FFN block of the very last layer (L=12) is identified as completely redundant and is removed.
- Under 33% pruning, the actual latency drops from 5.57ms to 3.86ms ($1.44\times$ speedup); physical truncation ensures that it does not rely on sparse computing engines.
- Decoupled loss validation: The network first shuts down empty heads (eliminating structural overhead $C_1$) before finely pruning micro dimensions (optimizing $C_2, C_3$).

## Highlights & Insights
- **Unified Framework for Dual-Level Gating (Macro + Micro)**: First to integrate block/head-level pruning and dimension/neuron-level pruning into a single differentiable optimization framework, with a theoretical proof (Lemma 1) showing that the dual-level search space strictly subsumes the single-level one.
- **Single-Stage Training without Post-Processing**: Temperature annealing smoothly transitions the gates from soft to hard, eliminating the need for post-hoc fine-tuning or manual threshold tuning.
- **Transferable Concept**: The framework, combining Gumbel-Sigmoid gating with differentiable cost modeling, can be naturally extended to LLM pruning (such as attention heads and FFN experts in MoE).

## Limitations & Future Work
- Accuracy gap compared to SOTA methods: 79.10% vs GOHSP's 79.98% at 3.1G FLOPs, indicating that the simplified pipeline incurs a performance cost.
- The optimization objective is the expected MACs instead of calibrated hardware latency or energy consumption; the actual speedup varies across hardware and custom kernels.
- Validation is limited to DeiT-Small. Experiments on larger models (such as DeiT-Base, Swin) and downstream tasks are currently missing.
- Potential integration with other compression techniques like token pruning or quantization to achieve higher compression ratios.

## Related Work & Insights
- **vs ViT-Slim**: ViT-Slim only performs micro-granularity pruning (intra-head + FFN channel), which requires $\ell_1$ sparsity training followed by ranking-based thresholding. HiAP adds macro-granularity pruning and is fully automatic, though with a slight drop in accuracy.
- **vs GOHSP**: GOHSP utilizes graph ranking combined with an optimization solver. It achieves higher accuracy but involves a convoluted process. HiAP replaces offline solvers with end-to-end gradient-based optimization.
- **vs UPDP**: UPDP only targets block-level pruning (using a genetic algorithm). HiAP's macro-level pruning encompasses both head and block granularities.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The unified framework for multi-granular Gumbel-Sigmoid gating is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐ Only evaluated on DeiT-Small for ImageNet, with limited data points.
- **Writing Quality**: ⭐⭐⭐⭐ Solid theoretical analysis and clear framework diagram.
- **Value**: ⭐⭐⭐ The accuracy lags behind SOTA; the primary contribution lies in simplifying the pruning workflow rather than setting new benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MDP: Multidimensional Vision Model Pruning with Latency Constraint](mdp_multidimensional_vision_model_pruning_with_latency_constraint.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients for Zero-Shot NAS on Vision Transformers](l_swag_zero_shot_nas_vision_transformers.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](../../ECCV2024/model_compression/isomorphic_pruning_for_vision_models.md)
- [\[CVPR 2026\] Collaborative Multi-Mode Pruning for Vision-Language Models](../../CVPR2026/model_compression/collaborative_multi-mode_pruning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
