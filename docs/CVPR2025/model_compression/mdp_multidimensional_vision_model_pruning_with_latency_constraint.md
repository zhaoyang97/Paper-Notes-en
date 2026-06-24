---
title: >-
  [Paper Note] MDP: Multidimensional Vision Model Pruning with Latency Constraint
description: >-
  [CVPR 2025][Model Compression][Structured Pruning] MDP proposes a multidimensional pruning paradigm that models structured pruning at different granularities (such as channels, attention heads, Q/K/V, embedding dimensions, and entire blocks) as a Mixed Integer Non-Linear Programming (MINLP) problem. It jointly solves for the globally optimal pruning structure under strict latency constraints, significantly outperforming existing methods at high pruning ratios.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "Structured Pruning"
  - "Latency Constraint"
  - "Mixed Integer Non-Linear Programming"
  - "Multidimensional Pruning"
  - "Hardware-Aware"
date: 2026-05-08
content_hash: 4b22fe79f55a7e83
---

# MDP: Multidimensional Vision Model Pruning with Latency Constraint

**Conference**: CVPR 2025  
**arXiv**: [2504.02168](https://arxiv.org/abs/2504.02168)  
**Code**: None (Paper mentions it will be open-sourced after acceptance)  
**Area**: Model Compression  
**Keywords**: Structured Pruning, Latency Constraint, Mixed Integer Non-Linear Programming, Multidimensional Pruning, Hardware-Aware

## TL;DR
MDP proposes a multidimensional pruning paradigm that models structured pruning at different granularities (such as channels, attention heads, Q/K/V, embedding dimensions, and entire blocks) as a Mixed Integer Non-Linear Programming (MINLP) problem. It jointly solves for the globally optimal pruning structure under strict latency constraints, significantly outperforming existing methods at high pruning ratios.

## Background & Motivation
**Background**: Structured pruning is a mainstream approach for deploying large models, which reduces computational cost by removing structured parameters such as channels and attention heads. Existing methods usually operate at a single granularity (e.g., channel-level or layer-level).

**Limitations of Prior Work**: (1) Most methods are only effective at moderate pruning ratios (30-50%), because pruning only channels or attention heads makes it difficult to achieve high pruning ratios of 70-90%; (2) Existing latency-aware methods (such as HALP) use overly simplified linear latency models that only consider the number of output channels and ignore the impact of input channel changes on latency, making them completely inapplicable to Transformers (which have 5 interrelated prunable dimensions).

**Key Challenge**: High pruning ratios require simultaneous pruning across multiple granularities (channels + blocks), but the search space for multi-granularity pruning is extremely large. Furthermore, latency non-linearly depends on multiple dimensions, and simple linear models cannot accurately model this.

**Goal** (1) Support joint pruning across various granularities (channels, heads, Q/K/V, embeddings, blocks); (2) Build highly accurate multidimensional latency constraint models; (3) Find the globally optimal pruning structure under a latency budget.

**Key Insight**: The authors realize that the key to the problem lies in "multidimensionality" — whether it is the multi-granularity combination required to achieve high pruning ratios, or the multidimensional interactions required to accurately model latency. By encoding multidimensional pruning variables and block retention/removal decisions jointly into a mathematical optimization problem, the globally optimal solution can be found in a single pass using mature MINLP solvers (Pyomo + MindtPy).

**Core Idea**: Use MINLP to uniformly model multi-granularity pruning and multidimensional latency constraints, directly solving for the globally optimal pruning structure under a latency budget.

## Method

### Overall Architecture
Given a pre-trained model and a target latency budget Ψ, MDP first identifies all prunable dimensions and encodes them as one-hot variables $\omega$, then constructs importance objective functions and latency constraint functions, followed by modeling block removal decisions as binary variables $\kappa$. Finally, the problem is formulated and solved as an MINLP. After solving, the pruned sub-network is extracted and fine-tuned to recover accuracy.

### Key Designs

1. **Multidimensional Variable Encoding and Importance Objective Function**:

    - **Function**: To describe the configurations of all prunable dimensions and their importance in a unified mathematical form.
    - **Mechanism**: For each prunable dimension, a one-hot vector $\omega_b^i$ is used for encoding, where the index of the "hot" bit represents the number of parameters retained in that dimension. In CNNs, there is one prunable dimension per layer (number of output channels), while in Transformers, each block has 5 prunable dimensions (embedding, number of heads, Q/K, V, MLP). The $j$-th element of the importance vector $\vec{I_b^i}$ is the cumulative importance score of retaining the top-$j$ parameters (Taylor score for CNNs, Hessian score for Transformers). The overall importance function is a linear combination of all dimensions: $\mathcal{I}(\omega) = \sum_b \sum_i \omega_b^{i\top} \cdot \vec{I_b^i}$
    - **Design Motivation**: Unlike traditional methods that score and sort single channels, MDP evaluates the "total importance of retaining $j$ channels." This allows the importance to be jointly solved alongside latency constraints within the same optimization framework.

2. **Multidimensional Latency Constraint Modeling**:

    - **Function**: To accurately model the relationship of latency as it varies simultaneously across multiple prunable dimensions.
    - **Mechanism**: Pre-compute a latency Look-Up Table (LUT) for each block, where the dimension of the table equals the number of prunable dimensions. Directly calculating outer product chains is too computationally expensive, so model-specific decomposition is performed: CNNs are decomposed into separate 2D LUTs per layer (input channels $\times$ output channels); Transformers are decomposed into three parts: QK (3D LUT of embedding $\times$ heads $\times$ qk), V-Proj (3D LUT of embedding $\times$ heads $\times$ v), and MLP (2D LUT of embedding $\times$ mlp). The total latency is the sum of the contributions of all blocks.
    - **Design Motivation**: Prior methods like HALP only use a latency model linear with the number of output channels, ignoring the impact of input channels. For Transformers, which have 5 dimensions interactively affecting latency, linear models are completely unviable. The decomposed LUTs ensure both accuracy and computational efficiency.

3. **Block-Level Removal and MINLP Solving**:

    - **Function**: To support the decision of removing entire blocks and optimizing it jointly with fine-grained pruning.
    - **Mechanism**: A binary decision variable $\kappa_b$ is introduced for each block. When $\kappa_b=0$, the block is entirely removed (relying on residual connections to maintain the information flow), making its importance and latency contributions zero. The final optimization problem is to maximize $\mathcal{I}(\omega, \kappa)$ s.t. $\mathcal{C}'(\omega, \kappa) \leq \Psi$, where $\omega$ is a one-hot vector (integer), $\kappa$ is binary (integer), and the importance and latency functions contain polynomial terms (non-linear), constituting an MINLP. It is solved using Pyomo + MindtPy along with Feasibility Pump.
    - **Design Motivation**: At high pruning ratios, fine-grained pruning alone is insufficient; removing entire blocks is necessary to drastically reduce latency. However, block removal and channel pruning must be jointly optimized to find the best balance — which is precisely the strength of MINLP.

### Loss & Training
After solving the MINLP to obtain the optimal structure, the corresponding parameters are retained according to the top-$j$ (importance rank) to extract the pruned sub-network $\hat{\Theta}$. Then, the sub-network is fine-tuned for $E$ epochs on the standard training set to recover accuracy. The entire workflow requires no extra modules or training stages.

## Key Experimental Results

### Main Results

| Model/Dataset | Method | Top-1(%) | Speedup | Description |
|------------|------|---------|--------|------|
| ResNet50/ImageNet | HALP-85% | 68.1 | 3.90× | Prev. SOTA |
| ResNet50/ImageNet | **MDP-85%** | **70.0** | **5.21×** | +1.9% Top-1, +28% speedup |
| ResNet50/ImageNet | HALP-70% | 74.5 | 2.55× | Prev. SOTA |
| ResNet50/ImageNet | **MDP-70%** | **74.8** | **3.06×** | +0.3% Top-1, +20% speedup |
| DeiT-B/ImageNet | Isomorphic-S | 82.41 | 2.56× | Latest Transformer pruning |
| DeiT-B/ImageNet | **MDP-39%** | **82.65** | **2.73×** | +0.24% Top-1, +7% speedup |
| StreamPETR/NuScenes | Dense | 0.449 mAP | 1.0× | Unpruned baseline |
| StreamPETR/NuScenes | **MDP-45%** | **0.451 mAP** | **1.18×** | Pruned mAP surpasses Dense |

### Ablation Study

| Configuration | Description | Advantage |
|------|------|------|
| Channel pruning only | Traditional methods like HALP | Effective at moderate pruning ratios |
| Channel + Block | MDP multi-granularity | Much better accuracy than channel-only at high pruning ratios |
| Linear latency model | HALP's 1D LUT | Inaccurate, inapplicable to Transformers |
| Multidimensional latency model | MDP's decomposed LUT | Accurate modeling, supports both CNNs and Transformers |

### Key Findings
- The advantage of MDP is particularly significant at high pruning ratios: at an 85% pruning rate, it brings a 28% speedup improvement and a 1.9% Top-1 improvement over HALP.
- Block removal is crucial for high pruning ratios — channel pruning alone cannot maintain reasonable accuracy at ratios $>70\%$.
- In 3D detection tasks (StreamPETR), the mAP of MDP-45% actually exceeds that of the original model, indicating that removing redundant structures may have a regularization effect.
- MDP achieves strict latency-constrained pruning for Transformers for the first time, whereas previous methods (e.g., NViT) only used latency as a soft regularization term.

## Highlights & Insights
- **Solving deep learning problems with operations research methods**: Formulating network pruning as an MINLP and solving it using mature numerical optimization tools (Pyomo/MindtPy) rather than greedy search or heuristics. MINLP guarantees global optimality, which greedy methods fail to achieve.
- **Generality of multidimensional latency modeling**: The LUT decomposition concept can be easily extended to new network architectures — simply identify the prunable dimensions and pre-compute the corresponding LUTs. This makes MDP a truly universal pruning framework.
- **Elegant handling of block removal**: By introducing the binary variable $\kappa$, block removal is naturally integrated into the optimization framework without requiring extra linear probes or two-stage processes.

## Limitations & Future Work
- The computational overhead of the MINLP solver is not explicitly reported, and for ultra-large models (e.g., LLaMA-70B), the solving time of MINLP might become a bottleneck.
- LUTs require pre-measuring a large number of latency data points on the target hardware; deploying to new hardware requires reconstructing them.
- Currently only verified on vision models (CNN + ViT), and not yet extended to NLP (e.g., LLM pruning) or multimodal models.
- The fine-tuning strategy is relatively simple (standard training); more efficient recovery methods such as knowledge distillation could be explored.

## Related Work & Insights
- **vs HALP**: HALP also performs latency-constrained pruning but only uses a 1D linear latency model (only considering output channels) and does not support block removal. MDP uses multi-dimensional LUT + MINLP to optimize multiple granularities simultaneously, outperforming HALP in all comparisons.
- **vs NViT**: NViT uses global Hessian importance scoring to prune Transformers, but only treats latency as a soft regularization term, failing to guarantee that the latency constraints are strictly met. MDP strictly satisfies the latency budget through hard constraints.
- **vs Isomorphic Pruning**: Compared to the latest Transformer pruning method, MDP further increases speed by 37% on the same model with higher accuracy.

## Rating
- Novelty: ⭐⭐⭐⭐ Formulating the pruning problem as an MINLP offers a novel perspective, and multidimensional latency modeling fills the gap in latency-constrained Transformer pruning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers multiple architectures and tasks including CNNs, Transformers, and 3D detection, with comprehensive baselines.
- Writing Quality: ⭐⭐⭐⭐ The mathematical derivations are clear and rigorous, though there are many symbols, which creates a slight learning curve.
- Value: ⭐⭐⭐⭐ Highly practical value for high pruning ratio scenarios and latency-constrained Transformer pruning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HiAP: A Multi-Granular Stochastic Auto-Pruning Framework for Vision Transformers](hiap_a_multi-granular_stochastic_auto-pruning_framework_for_vision_transformers.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](../../ECCV2024/model_compression/isomorphic_pruning_for_vision_models.md)
- [\[CVPR 2025\] BHViT: Binarized Hybrid Vision Transformer](bhvit_binarized_hybrid_vision_transformer.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](../../ACL2025/model_compression/beamlora_beam_constraint_lora.md)
- [\[NeurIPS 2025\] Vision-centric Token Compression in Large Language Model](../../NeurIPS2025/model_compression/vision-centric_token_compression_in_large_language_model.md)

</div>

<!-- RELATED:END -->
