---
title: >-
  [Paper Note] Function-Space Learning Rates
description: >-
  [ICML2025][Model Compression][function-space learning rate] An efficient Monte Carlo estimation method for **layer-wise function-space learning rates** is proposed. Based on this, **FLeRM** (Function-space Learning Rate Matching) is designed to record function-space learning rates on a small model and automatically adjust the parameter-space learning rates of a large model, enabling hyperparameter transfer across width, depth, initialization scale, and LoRA rank.
tags:
  - "ICML2025"
  - "Model Compression"
  - "function-space learning rate"
  - "hyperparameter transfer"
  - "FLeRM"
  - "model scaling"
  - "LoRA"
date: 2026-05-08
content_hash: 239617f5037bba3a
---

# Function-Space Learning Rates

**Conference**: ICML2025  
**arXiv**: [2502.17405](https://arxiv.org/abs/2502.17405)  
**Code**: [GitHub](https://github.com/edwardmilsom/function-space-learning-rates-paper)  
**Area**: Function-Space Learning Rate / Hyperparameter Transfer  
**Keywords**: function-space learning rate, hyperparameter transfer, FLeRM, model scaling, LoRA

## TL;DR

An efficient Monte Carlo estimation method for **layer-wise function-space learning rates** is proposed. Based on this, **FLeRM** (Function-space Learning Rate Matching) is designed to record function-space learning rates on a small model and automatically adjust the parameter-space learning rates of a large model, enabling hyperparameter transfer across width, depth, initialization scale, and LoRA rank.

## Background & Motivation

### Background

**Background**: The fundamental goal of neural network training is to learn a **function** mapping inputs to outputs. However, traditional learning rates measure changes in the parameter space rather than the function space. This raises a core question: **Can learning in function space be meaningfully quantified and controlled?**

Pre-training large models costs millions of dollars, making hyperparameter search at full scale impractical. Existing hyperparameter transfer methods (such as µP, Modula) typically require:

### Limitations of Prior Work

**Limitations of Prior Work**: Strict architectural assumptions (e.g., sufficiently wide networks, near-random initialization)

### Key Challenge

**Key Challenge**: Complex mathematical tools (Tensor Programs, dynamical mean-field theory)

### Key Insight

**Key Insight**: Rewriting network architecture using specific libraries

These limitations restrict the flexibility of existing methods in practical applications. This paper proposes an **empirical** approach that avoids these constraints by directly measuring the function-space learning rate.

## Method

### Core Concept: Layer-Wise Function-Space Learning Rate

The first-order Taylor approximation of the change in network output $f_{nk}$ caused by the update $\Delta \mathbf{W}^\ell$ of the $\ell$-th layer parameter $\mathbf{W}^\ell$ is:

$$\Delta_\ell f_{nk} = \sum_{ij} \Delta W_{ij}^\ell \frac{df_{nk}}{dW_{ij}^\ell}$$

The function-space learning rate is defined as the RMS norm of the change in output:

$$\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^2 = \frac{1}{NK} \sum_{nk} (\Delta_\ell f_{nk})^2$$

Direct computation requires $NK$ backward passes, which is computationally prohibitive.

### Monte Carlo Estimation

A scalar random projection $\phi = \frac{1}{\sqrt{NK}} \sum_{nk} \omega_{nk} f_{nk}$ ($\omega_{nk} \sim \mathcal{N}(0,1)$) is constructed. Its change with respect to the $\ell$-th layer parameter update satisfies:

$$\Delta_\ell \phi \sim \mathcal{N}(0, \|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^2)$$

Therefore, only **one additional backward pass** is required to obtain a single sample. Estimating the variance through multiple samples yields the function-space learning rate.

### Variance Reduction via Kronecker Factorization

Define $Z_{ij} = \Delta W_{ij}^\ell \cdot \frac{d\phi}{dW_{ij}^\ell}$. Assuming the covariance of $Z_{ij}$ has a Kronecker structure:

$$\text{Cov}[Z_{ij}, Z_{i'j'}] = U_{ii'} V_{jj'}$$

Then, the function-space learning rate can be estimated by **three scalar EMAs**:

$$\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^2 = \frac{\mathbb{E}[\sum_k(\sum_i Z_{ik})^2] \cdot \mathbb{E}[\sum_k(\sum_j Z_{kj})^2]}{\mathbb{E}[\|\mathbf{Z}\|_{\mathcal{F}}^2]}$$

In practice, the EMA is warmed up using 40 batches at the start of training and updated every 100 steps thereafter, making the computational overhead negligible.

### FLeRM: Function-Space Learning Rate Matching

1. **Recording Phase**: Train on a small (inexpensive) base model and record the layer-wise function-space learning rates $\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^{\text{base},t}$.
2. **Transfer Phase**: On the large model, measure the current function-space learning rate $\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^{(t)}$ and automatically dynamically set the layer-wise parameter-space learning rates:

$$\eta^\ell = \eta_0 \cdot \frac{\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^{\text{base},t}}{\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^{(t)}}$$

**Depth Scaling Heuristics**: If the large model has more layers than the base model, the function-space learning rate of the base model is equally distributed among the newly added residual blocks.

## Key Experimental Results

### Experimental Setup


### Main Results

| Model | Architecture | Dataset | Optimizer |
|------|------|--------|--------|
| ResMLP | 4-hidden-layer residual MLP | CIFAR-10 | Adam (constant LR) |
| Transformer (PostNorm/PreNorm/PreNormPostMod) | 2-layer decoder-only | Wikitext-103 subset | Adam (constant LR) |
| GPT-2 / Llama-3.2-1B | LoRA fine-tuning | Cold French Law / Mathpile (~4M tokens) | AdamW |

### FLeRM Width Transfer

- Under standard training, increasing width leads to a **significant shift** in the optimal learning rate (consistent with µP theory).
- With FLeRM, the shift in optimal learning rates is **completely eliminated or dramatically reduced**.
- For Transformers at large widths, FLeRM also **improves training loss**.

### FLeRM Depth Transfer

- ResMLP: Standard training displays a significant shift, while FLeRM substantially brings the optimal LRs closer.
- PostNorm Transformer: Increasing depth in standard training $\rightarrow$ instability threshold shifts to the left $\rightarrow$ deeper models exhibit worse loss; FLeRM aligns the instability thresholds, resulting in a **substantial performance boost for deeper models**.
- PreNorm: Small shifts are corrected by FLeRM.
- PreNormPostMod: Standard training is already depth-invariant; FLeRM preserves this property.

### FLeRM LoRA Rank Transfer

- Under standard training, as the LoRA rank increases from 2 to 32, the optimal LR shifts by **more than an order of magnitude**.
- Using FLeRM (with rank=2 as base), the shift is **eliminated or significantly reduced**, aligning the instability boundaries.

### Dynamic Analysis of Function-Space Learning Rates

- Under a fixed parameter-space LR, the function-space LR **monotonically decreases** during training (except for the input embedding layer) $\rightarrow$ revealing an implicit schedule.
- Different layer types form distinct "bands": the second feed-forward weight matrix (FF2) has the greatest impact on the output (with its function-space LR being an order of magnitude higher than that of the readout layer).

## Highlights & Insights

1. **Architecture-Agnosticism**: The method applies to any PyTorch network without requiring architecture rewrites or dependency on specific packages.
2. **Extremely Low Computational Overhead**: Requires only a small number of additional backward passes (40 passes at the start of training + 1 pass every 100 steps).
3. **Unified Framework**: Solves hyperparameter transfer across four dimensions—width, depth, initialization scale, and LoRA rank—using the same single methodology.
4. **Analysis Tool Value**: The function-space perspective reveals the implicit learning rate schedule of the Adam optimizer—the function-space contributions of different layers dynamically evolve during training.
5. **Counter-Intuitive Discovery**: The second feed-forward layer in the Transformer (rather than the readout layer) has the most significant impact on the output function.

## Limitations & Future Work

1. **Coarse Depth Scaling Heuristics**: When the large model has more layers than the base model, simply dividing the function-space learning rate equally is not necessarily optimal; ablation studies show that more complex matching schemes could further improve performance.
2. **First-Order Taylor Approximation**: The change in function space relies on a linear approximation, which may become inaccurate under large learning rates.
3. **Kronecker Factorization Assumption**: The Kronecker structure of the covariance is an approximation, which may introduce bias.
4. **Limited Experimental Scale**: The largest model tested is around 814M parameters; its effectiveness has not been verified at larger scales (such as tens of billions of parameters).
5. **Only Evaluated on Adam/AdamW**: Other optimizers like SGD, LAMB, etc., are yet to be explored.

## Related Work & Insights

- **µP (Yang & Hu, 2022)**: Analytically derives learning rate scaling rules under the infinite-width assumption; FLeRM avoids this limitation through empirical measurement.
- **Modula (Large et al., 2024)**: A metric-based method based on Lipschitz constants, which requires setting the "mass" of each parameter and rewriting the architecture; FLeRM automatically records this from the base model.
- **Chizat & Netrapalli (2024)**: Quantifies feature learning from the perspective of feature updates and backpropagation; FLeRM directly measures it using automatic differentiation.
- **Everett et al. (2024)**: Empirically discovers that alignment in real-world models is highly dynamic during training, which highlights the difficulty of choosing µP and mean-field assumptions.

## Rating

- Novelty: ⭐⭐⭐⭐ — Empirical estimation of function-space learning rates and cross-scale transfer offer a completely fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers four dimensions (width, depth, initialization, and LoRA) across multiple architectures and ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear mathematical derivation, with a natural flow from motivation to algorithmic logic.
- Value: ⭐⭐⭐⭐ — Directly practically valuable for hyperparameter tuning in large model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](../../CVPR2025/model_compression/lora_subtraction_for_drift-resistant_space_in_exemplar-free_continual_learning.md)
- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](../../CVPR2025/model_compression/dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](../../CVPR2025/model_compression/tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ICLR 2026\] MaskPro: Linear-Space Probabilistic Learning for Strict (N:M)-Sparsity on LLMs](../../ICLR2026/model_compression/maskpro_linear-space_probabilistic_learning_for_strict_nm-sparsity_on_llms.md)

</div>

<!-- RELATED:END -->
