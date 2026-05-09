---
title: >-
  [Paper Note] DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration
description: >-
  [NeurIPS 2025][Image Restoration][LLM pruning] This paper proposes DenoiseRotator, a pre-pruning method that applies learnable orthogonal transformations to minimize the information entropy of parameter importance scores, concentrating importance into a small subset of parameters. On LLaMA3-70B under 2:4 semi-structured sparsity, perplexity degradation is reduced by 58% (8.1→3.4). The method is plug-and-play and compatible with Magnitude, Wanda, and SparseGPT.
tags:
  - NeurIPS 2025
  - Image Restoration
  - LLM pruning
  - orthogonal transformation
  - entropy minimization
  - importance concentration
  - semi-structured sparsity
date: 2026-05-08
content_hash: 1ee5559497f28f3c
---

# DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration

**Conference**: NeurIPS 2025
**arXiv**: [2505.23049](https://arxiv.org/abs/2505.23049)
**Code**: [Axel-gu/DenoiseRotator](https://github.com/Axel-gu/DenoiseRotator)
**Area**: Image Restoration
**Keywords**: LLM pruning, orthogonal transformation, entropy minimization, importance concentration, semi-structured sparsity

## TL;DR

This paper proposes DenoiseRotator, a pre-pruning method that applies learnable orthogonal transformations to minimize the information entropy of parameter importance scores, concentrating importance into a small subset of parameters. On LLaMA3-70B under 2:4 semi-structured sparsity, perplexity degradation is reduced by 58% (8.1→3.4). The method is plug-and-play and compatible with Magnitude, Wanda, and SparseGPT.

## Background & Motivation

**Background**: Pruning is a mainstream technique for LLM compression. Methods such as SparseGPT and Wanda use Taylor expansion approximations to estimate per-parameter importance scores and remove the lowest-scoring weights. However, performance degradation under semi-structured sparsity constraints (e.g., the 2:4 pattern) remains severe.

**Limitations of Prior Work**: Existing methods focus exclusively on *which* weights to prune and operate within the fixed parameter space of the pretrained model, without modifying the underlying distribution of importance scores. When importance is dispersed across the entire weight matrix, any pruning selection inevitably removes a non-trivial amount of total importance.

**Key Challenge**: How can parameter importance be redistributed without altering model outputs? If importance can be concentrated into a small number of parameters, removing the remaining large mass of low-importance parameters incurs minimal cost.

**Key Insight**: The computational invariance of Transformers allows orthogonal transformations to be applied to weight matrices without changing model outputs. Orthogonal transformations preserve norms—the total importance of a layer is invariant—but the distribution of importance across parameters can be reshaped.

**Core Idea**: Before pruning, a learnable orthogonal matrix $R$ is trained to minimize the information entropy $\mathcal{H}(P) = -\sum p_{ij} \log p_{ij}$ of normalized importance scores, shifting the importance distribution from uniform to peaked. After training, $R$ is absorbed into the weight matrices, and standard pruning proceeds unchanged.

## Method

### Overall Architecture

The method proceeds in three stages: (1) fuse RMSNorm weights into adjacent linear layers and insert layer-level orthogonal matrices $R_1$ and attention-level orthogonal matrices $R_2$ into each Transformer layer; (2) freeze the original weights and train only $R_1, R_2$ via entropy minimization (2000 steps, lr=0.01); (3) absorb $R$ into the weight matrices and apply any pruning method (Magnitude/Wanda/SparseGPT).

### Key Designs

1. **Entropy-Guided Importance Concentration**:

    - The importance scores $S_{ij}$ of each linear layer are normalized into a discrete probability distribution $p_{ij} = S_{ij} / \sum S_{ij}$.
    - Minimizing the information entropy of this distribution serves as a proxy objective: entropy is permutation-invariant (independent of parameter ordering) and, as a concave function on the probability simplex, its minimization naturally promotes concentration onto a small number of elements.
    - The normalization grouping depends on the position of the orthogonal matrix: right-multiplication yields row-wise normalization, left-multiplication yields column-wise normalization, and two-sided multiplication sums both row and column entropies.

2. **Insertion Positions of Orthogonal Matrices**:

    - **Layer-level rotation $R_1$**: A pair of $(d_{hidden}, d_{hidden})$ orthogonal matrices are inserted at the input and output of each Transformer layer (between residual connections and RMSNorm), affecting Q/K/Up/Gate/Down projections.
    - **Attention-level rotation $R_2$**: Additional rotations are applied to Value and Output projections within self-attention to further concentrate attention-internal importance.
    - Taking the Output projection as an example: the transformed weight is $W' = R_1^\top W R_2$, the transformed input is $X' = R_2^\top X$, and the product $W'X' = R_1^\top WX$ remains unchanged.

3. **Total Importance Invariance**:

    - The orthogonal transformation property $\|Rx\| = \|x\|$ ensures $\sum \mathcal{T}_S(S) = \sum S$, i.e., the total importance of a layer is strictly invariant before and after transformation.
    - Importance is redistributed rather than created or destroyed, ensuring algorithmic stability.

4. **QR Decomposition Parameterization**:

    - Direct gradient descent on orthogonal matrices violates the orthogonality constraint.
    - An unconstrained matrix $A$ is introduced; during the forward pass, QR decomposition $A = QR$ is performed and only the orthogonal component $Q$ is used; gradients are back-propagated through $A$.
    - PyTorch's `torch.qr` natively supports automatic differentiation, eliminating the need for Stiefel manifold optimizers.

5. **Plug-and-Play Compatibility**:

    - DenoiseRotator is fully decoupled from the pruning step and operates as a pre-pruning preprocessing stage.
    - Once $R$ is absorbed into the weights, any downstream pruning method can be applied.
    - The Hessian matrix $H = XX^\top$ is shared between the $R$ training phase and subsequent pruning, requiring no additional calibration.

### Training Details

- Optimizer: Adam, learning rate 0.01, 2000 training steps.
- Precision: bfloat16 (except QR decomposition).
- Initialization: $R$ is initialized as the identity matrix, ensuring the starting point does not alter model behavior.
- Loss function: $\text{Loss}(R_{1,i}, R_{2,i}) = \sum_{\ell \in \mathcal{L}_i} \sum_{\mathcal{G} \in \ell} \mathcal{H}(P_\mathcal{G})$, i.e., the sum of entropies over all normalization groups across all linear layers within a layer.
- LLaMA3-70B + SparseGPT training requires approximately 28 hours on a single A100 with ~30 GB memory.

## Key Experimental Results

### Perplexity Results (WikiText-2)

| Model | Sparsity | Method | Baseline PPL | +DenoiseRotator PPL | Dense PPL |
|------|---------|---------|---------|-------------------|----------|
| LLaMA3-70B | 2:4 | SparseGPT | 10.97 | **6.25** | 2.86 |
| LLaMA3-70B | 50% | SparseGPT | 5.99 | **4.61** | 2.86 |
| LLaMA3-8B | 2:4 | SparseGPT | 17.67 | **10.01** | 6.14 |
| LLaMA3-8B | 50% | SparseGPT | 9.57 | **7.60** | 6.14 |
| Qwen2.5-72B | 2:4 | SparseGPT | 7.19 | **5.85** | 3.88 |
| Qwen2.5-72B | 50% | SparseGPT | 4.94 | **4.78** | 3.88 |
| Mistral-7B | 2:4 | Wanda | 10.18 | **7.80** | 5.95 |
| Mistral-7B | 50% | Wanda | 6.92 | **6.52** | 5.95 |

### Zero-Shot Accuracy (Average over 5 Tasks)

| Model | Sparsity | Method | Baseline Acc | +DenoiseRotator Acc | Dense Acc |
|------|---------|---------|---------|-------------------|----------|
| LLaMA3-70B | 2:4 | SparseGPT | 69.16% | **76.37%** | 80.05% |
| LLaMA3-70B | 50% | SparseGPT | 76.66% | **78.54%** | 80.05% |
| Qwen2.5-72B | 2:4 | SparseGPT | 75.43% | **77.16%** | 78.69% |
| Qwen2.5-72B | 50% | SparseGPT | 78.25% | **78.42%** | 78.69% |

### Entropy Reduction vs. Performance (LLaMA3-8B, SparseGPT 50%)

| Training Steps | Mean Entropy | Perplexity | Zero-Shot Acc |
|---------|-------|--------|-------------|
| 0 (baseline) | 457280 | 9.567 | 66.88% |
| 100 | 396992 (-13%) | 7.701 | 70.54% |
| 400 | 387904 | 7.619 | 70.12% |
| 2000 | 384128 | **7.597** | 69.58% |

### Inference Overhead (LLaMA3-8B, A100)

| Configuration | Per-Layer Latency | Speedup |
|------|---------|--------|
| Dense | 5.80 ms | 1.00× |
| 2:4 Sparse | 4.37 ms | 1.33× |
| 2:4 Sparse + Orthogonal Matrices | 4.69 ms | 1.24× |

Orthogonal matrices add only ~0.32 ms per layer and increase parameter count by approximately 6.7% (~0.5B additional parameters for LLaMA3-8B).

### Key Findings

- **Most striking results on LLaMA3-70B 2:4**: Perplexity degradation reduced from 8.11 to 3.39 (58% reduction); zero-shot accuracy improved from 69.16% to 76.37%.
- **Largest gains for Magnitude pruning**: E.g., Qwen2.5-72B Magnitude 2:4 baseline PPL of 287.70 drops to 8.81 after DenoiseRotator, recovering from unusable to practical.
- **Consistent effectiveness across model families**: Significant improvements across LLaMA3, Qwen2.5, and Mistral, at scales from 7B to 72B.
- **A 13% entropy reduction yields substantial gains**: 100 training steps reduce perplexity from 9.57 to 7.70.
- **Low sensitivity to hyperparameters**: Learning rates in the range 0.01–0.1 and step counts of 400–4000 all yield near-optimal results.

## Highlights & Insights

- **Novelty of the information-theoretic perspective**: Framing pruning robustness as entropy minimization over importance distributions provides a theoretically grounded optimization objective, circumventing the NP-hard combinatorial pruning problem.
- **Elegant exploitation of orthogonal transformations**: Transformer computational invariance ensures transformations do not alter model outputs, while orthogonality preserves total importance—together enabling "cost-free" redistribution of importance.
- **Engineering value as a plug-and-play module**: As a pure preprocessing step that does not modify any downstream pruning procedure, compatibility with Magnitude/Wanda/SparseGPT lowers the barrier to adoption.
- **Turning "unusable" into "usable"**: The largest gains are observed for simple methods such as Magnitude pruning—e.g., Qwen2.5-72B 2:4 sparse PPL reduced from 287.70 to 8.81.

## Limitations & Future Work

- **Non-zero inference overhead**: Layer-level orthogonal matrices cannot be fully absorbed, adding ~0.32 ms per layer and ~6.7% additional parameters; block-diagonal orthogonal matrices are explored to reduce overhead, but this involves an accuracy–efficiency trade-off.
- **No explicit optimization for semi-structured constraints**: The gains under 2:4 sparsity stem from the "random permutation effect" of orthogonal transformations rather than targeted optimization; future work could incorporate structural constraints into the training objective.
- **Non-trivial training cost**: LLaMA3-70B requires approximately 28 hours of single-GPU A100 training, which may be prohibitive for rapid deployment scenarios.
- **Validation limited to post-training settings**: The paper suggests that integrating importance concentration into pretraining or continual training may be more effective, but this remains experimentally unverified.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of entropy minimization and orthogonal transformation is novel in the pruning literature, reframing the problem as "redistributing importance" rather than "selecting which weights to remove."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven models (7B–72B), three pruning methods, two sparsity patterns, with complete ablation studies and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, theoretical derivations are complete, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Directly practical for LLM deployment, particularly for hardware-accelerated 2:4 sparsity scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Skip to the Good Part: Representation Structure & Inference-Time Layer Skipping in Diffusion vs. Autoregressive LLMs](../../ICLR2026/image_restoration/skip_to_the_good_part_representation_structure_inference-time_layer_skipping_in_.md)
- [\[NeurIPS 2025\] SCAN: Self-Denoising Monte Carlo Annotation for Robust Process Reward Learning](scan_self-denoising_monte_carlo_annotation_for_robust_process_reward_learning.md)
- [\[NeurIPS 2025\] GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights](gc4nc_a_benchmark_framework_for_graph_condensation_on_node_classification_with_n.md)
- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)
- [\[NeurIPS 2025\] Improving Diffusion-based Inverse Algorithms under Few-Step Constraint via Learnable Linear Extrapolation](improving_diffusion-based_inverse_algorithms_under_few-step_constraint_via_learn.md)

</div>

<!-- RELATED:END -->
