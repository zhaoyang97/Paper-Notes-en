---
title: >-
  [Paper Note] LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation
description: >-
  [ICML 2025][LLM (Other)] LaRoSA proposes a training-free activation sparsification method. By applying layerwise orthogonal rotation matrices, it transforms input activations into a space better suited for sparsification, and combines this with Top-K selection to achieve consistent model-level sparsity and reliable inference acceleration.
tags:
  - "ICML 2025"
  - "LLM (Other)"
date: 2026-05-08
content_hash: c45edf5820ed3c02
---

# LaRoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation

**Conference**: ICML 2025  
**arXiv**: [2507.01299](https://arxiv.org/abs/2507.01299)  
**Area**: LLM/NLP  

## TL;DR

LaRoSA proposes a training-free activation sparsification method. By applying layerwise orthogonal rotation matrices, it transforms input activations into a space better suited for sparsification, and combines this with Top-K selection to achieve consistent model-level sparsity and reliable inference acceleration.

## Background & Motivation

Efficient inference of Large Language Models (LLMs) is currently an important research direction. Leveraging activation sparsity can skip the weight channels corresponding to zero-value activations, thereby reducing memory transfer and computational overhead. However, existing methods suffer from two major limitations:

**ReLU-based methods** (e.g., DejaVu) require extensive recovery training, and modern LLMs (e.g., LLaMA3, Qwen2.5) use non-ReLU activation functions such as SwiGLU, which do not naturally generate sparsity.

**Magnitude-pruning-based methods** (e.g., CATS, TEAL) use offline calibration thresholds, which suffer from three major issues:
   - **Ambiguity and inaccuracy** of threshold definition: Calibration thresholds are difficult to align with the actually required thresholds.
   - **Inability to maintain consistent sparsity**: The actual sparsity deviates significantly from the target value.
   - **Flawed assumption regarding magnitude and channel importance**: Low-magnitude activations may still significantly affect the output if they correspond to weight channels with high norms.

## Method

### Core Idea

The key insight of LaRoSA is that, through orthogonal rotation transformation, activation vectors can be projected into a space where channel importance is more distinguishable, thereby achieving more effective sparsification.

### Layerwise Orthogonal Rotation

For each layer $l$, LaRoSA constructs an orthogonal rotation matrix $\mathbf{Q}_l$ using PCA. The specific steps are:

1. Select a calibration dataset ($M$ sequences) and perform forward propagation to obtain the input activations $\mathbf{X}_l^i$ for each layer.
2. Calculate the covariance matrix and take the average:

$$\text{Cov}(\mathbf{X}_l, \mathbf{X}_l^T) = \frac{1}{M}\sum_{i=0}^{M}\mathbf{X}_l^i(\mathbf{X}_l^i)^T$$

3. Perform eigendecomposition on the covariance matrix and arrange the eigenvectors in descending order of eigenvalues to construct $\mathbf{Q}_l$.

### Residual Adapter

Since residual connections require each layer to use the same rotation matrix, but the optimal rotations for different layers vary significantly, LaRoSA introduces a **residual adapter** $\mathbf{Q}_l^T\mathbf{Q}_{l+1}$ to achieve layerwise independent rotation. The rotation matrices of the first and last layers can be merged into the token embedding and LM head layers, respectively.

### Consistent Activation Sparsity

LaRoSA replaces magnitude pruning with a Top-K function to perform sparsification on the rotated activations:

$$S_k(\tilde{x}_i) = \begin{cases} \tilde{x}_i, & \text{if } |\tilde{x}_i| \in \text{Top}_k(|\tilde{x}_i|) \\ 0, & \text{otherwise} \end{cases}$$

where $k = \alpha \cdot (1-p) \cdot D_{\text{in}}$, $p$ is the target sparsity, and $\alpha$ is a hyperparameter controlling the sparsity coefficients of $h_1$ and $h_2$ within the same block.

### Weight Absorption

The rotation matrix $\mathbf{Q}_l$ can be folded into the weight matrix beforehand to avoid extra computations during inference:

$$\mathbf{Y}_l = S_k(\mathbf{X}_l\mathbf{Q}_l) \cdot (\mathbf{W}_l\mathbf{Q}_l)^T$$

### Hardware-Efficient Custom Kernel

A GEMV kernel is implemented based on Triton: utilizing column-major format to store weights, fusing Top-K into the matrix-vector multiplication, and selectively loading sparse activations and their corresponding weight columns.

## Experiments

### Main Results - Zero-shot Task Accuracy

| Method | LLaMA2-7B Acc7 | LLaMA3-8B Acc7 | Qwen2.5-7B Acc7 |
|---|---|---|---|
| Dense | 66.69 | 70.05 | 70.34 |
| CATS 40% | 49.55 | 55.11 | 61.83 |
| TEAL 40% | 64.92 | 68.14 | 68.61 |
| **LaRoSA 40%** | **66.15** | **68.79** | **69.67** |
| TEAL 50% | 63.22 | 64.92 | 67.76 |
| **LaRoSA 50%** | **64.61** | **67.19** | **69.09** |

### Perplexity Results

Under 40% sparsity on LLaMA2-7B, LaRoSA achieves a perplexity gap of only **0.17** (5.64 vs. 5.47), whereas TEAL is 0.93 and CATS is as high as 39.99.

### Inference Acceleration

LaRoSA achieves a **1.38×** speedup at 50% sparsity and a **1.72×** speedup at 75% sparsity on an A100 GPU. Due to using Top-K to ensure consistent sparsity, the speedup is stable and predictable.

### Inference Model Experiments

On DeepSeek-R1-Distill-Llama3-8B with 25% sparsity, LaRoSA only drops 2.6 points on MATH-500 (85.0 vs. 87.6) and maintains the same performance on AIME-2024 (40.0).

## Highlights & Insights

- **Training-Free**: Requires only 12 minutes of calibration for a 70B model, making it highly deployment-friendly.
- **Consistent Sparsity**: Top-K guarantees constant sparsity for each token, resolving the instability issues of magnitude pruning.
- **Theoretical Support**: The appendix provides a theoretical analysis showing that rotation outperforms magnitude pruning in reducing layerwise empirical error.
- **Cross-Model Robustness**: Demonstrates strong performance across 7B and 70B models of LLaMA2/3, Qwen2.5, and Mistral.
- **Inference Model Compatibility**: Validates the preservation of reasoning capabilities on DeepSeek-R1 distilled models.

## Limitations & Future Work

- Only the input activations of $h_1$ and $h_3$ are rotated; $h_2$ and $h_4$ cannot be rotated due to the constraints of GQA and element-wise multiplication.
- The residual adapter introduces a small amount of extra computation.
- The hyperparameter $\alpha$ needs to be tuned via grid search for each model.
- The Top-K operation itself incurs some overhead, requiring custom GPU kernels to realize actual speedup.
- Under extremely high sparsity (60%+), the performance degradation remains relatively obvious.

## Rating

⭐⭐⭐⭐ (4/5)

The LaRoSA method is elegant and practical, cleverly solving the activation sparsification problem for non-ReLU LLMs through orthogonal rotation. Its training-free nature makes it highly suitable for practical deployment, and the experiments thoroughly cover multiple models and tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning](../../NeurIPS2025/llm_nlp/sparse_mezo_less_parameters_for_better_performance_in_zeroth-order_llm_fine-tuni.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](../../ACL2025/llm_nlp/plangenllms_planning_survey.md)
- [\[ICML 2025\] QuEst: Enhancing Estimates of Quantile-Based Distributional Measures Using Model Predictions](quest_enhancing_estimates_of_quantile-based_distributional_measures_using_model_.md)
- [\[NeurIPS 2025\] Detecting High-Stakes Interactions with Activation Probes](../../NeurIPS2025/llm_nlp/detecting_high-stakes_interactions_with_activation_probes.md)
- [\[ACL 2025\] PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights](../../ACL2025/llm_nlp/praise_enhancing_product_descriptions_with_llm-driven_structured_insights.md)

</div>

<!-- RELATED:END -->
