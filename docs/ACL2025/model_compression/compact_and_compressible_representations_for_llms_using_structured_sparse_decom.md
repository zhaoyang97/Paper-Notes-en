---
title: >-
  [Paper Note] Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition
description: >-
  [ACL 2025][Model Compression][Structured Sparse Decomposition] This paper proposes a structured sparse decomposition method that decomposes the LLM weight matrix into a combination of a low-rank component and a structured sparse component. This achieves high compression ratios while maintaining model performance, enabling efficient deployment of large language models in resource-constrained environments.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Structured Sparse Decomposition"
  - "LLM Compression"
  - "Weight Matrix Decomposition"
  - "Low-Rank Approximation"
  - "Sparse Representation"
date: 2026-05-08
content_hash: 07bcee586899fc81
---

# Compact and Compressible Representations for LLMs Using Structured Sparse Decomposition

**Conference**: ACL 2025  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Structured Sparse Decomposition, LLM Compression, Weight Matrix Decomposition, Low-Rank Approximation, Sparse Representation

## TL;DR
This paper proposes a structured sparse decomposition method that decomposes the LLM weight matrix into a combination of a low-rank component and a structured sparse component. This achieves high compression ratios while maintaining model performance, enabling efficient deployment of large language models in resource-constrained environments.

## Background & Motivation

**Background**: The parameter sizes of large language models (LLMs) have reached tens of billions or even trillions, presenting huge storage and computational overhead during deployment. Existing model compression techniques mainly include quantization, pruning, knowledge distillation, and low-rank approximation.

**Limitations of Prior Work**: Traditional low-rank approximation (such as SVD) effectively compresses models but suffers from severe performance degradation at high compression ratios, because not all information in the weight matrix can be captured by low-rank structures. While unstructured pruning retains accuracy better, its irregular sparsity patterns are difficult to execute efficiently on hardware. Quantization methods also face performance collapse under extremely low bit-widths.

**Key Challenge**: There is a fundamental trade-off between high compression ratios and performance preservation. Low-rank approximation ignores sparse outliers in the weight matrix, which have a disproportionately large impact on model outputs.

**Goal**: Design a hybrid decomposition scheme that simultaneously exploits low-rank structures and structured sparse patterns to achieve better performance preservation at higher compression ratios.

**Key Insight**: The authors observe that LLM weight matrices can naturally decompose into a "smooth" low-rank component and a "sharp" sparse component, where the sparse component is mainly concentrated in a few key channels, exhibiting a structured distribution pattern.

**Core Idea**: By using structured sparse decomposition, the weight matrix is represented as a low-rank matrix plus a column/row-wise structured sparse matrix. This retains the compression advantages of low-rank approximation while accurately capturing performance-critical sparse outliers. Additionally, the structured sparse pattern allows for efficient hardware execution.

## Method

### Overall Architecture
Given an LLM weight matrix $W \in \mathbb{R}^{m \times n}$, this method decomposes it into $W \approx L + S$, where $L$ is a low-rank matrix of rank $r$ ($L = UV^T$, $U \in \mathbb{R}^{m \times r}$, $V \in \mathbb{R}^{n \times r}$), and $S$ is a structured sparse matrix. The overall pipeline is divided into three stages: (1) obtaining residuals from an initial low-rank approximation; (2) applying structured sparse constraints on the residuals to select key elements; (3) jointly optimizing both the low-rank and sparse components to minimize the total reconstruction error.

### Key Designs

1. **Adaptive Rank Selection Mechanism**:

    - **Function**: Automatically determine the optimal low-rank approximation rank $r$ for each weight matrix layer.
    - **Mechanism**: Based on the energy distribution of singular values from SVD, the rank is calculated when the cumulative energy ratio reaches a threshold $\tau$. Since the information distribution of weight matrices across different layers varies, a layer-wise adaptive rank setting is employed instead of a globally uniform one. The threshold $\tau$ is fine-tuned using the output error on a calibration dataset as a feedback signal, ensuring the overall compression ratio meets the target constraint.
    - **Design Motivation**: Fixed-rank decomposition treats all layers identically, but in practice, the rank distribution between attention layers and FFN layers varies significantly. Adaptive rank allocation concentrates the "budget" on layers with high information density.

2. **Structured Sparse Residual Capture**:

    - **Function**: Extract structured sparse components from the residual of the low-rank approximation $R = W - L$.
    - **Mechanism**: The residual matrix is grouped by columns (corresponding to output channels), and the $\ell_2$ norm of each group is computed. The top-$k$ groups with the largest norms are selected for retention, while the rest are set to zero. Within each selected group, elements are further filtered using a threshold, retaining only those whose magnitudes exceed the group mean. The resulting sparse matrix $S$ possesses column-wise structured sparsity, where non-zero elements are concentrated in a few columns.
    - **Design Motivation**: While unstructured sparsity is flexible, it is not hardware-friendly. Column-wise structured sparsity can be efficiently executed using vectorized operations on modern GPUs. Moreover, the column-sparse pattern naturally aligns with the outlier distribution of weight matrices in Transformers, where outliers are typically concentrated in specific output channels.

3. **Joint Iterative Optimization**:

    - **Function**: Alternatingly optimize the low-rank and sparse components to minimize the total reconstruction error.
    - **Mechanism**: An alternating minimization strategy is adopted: optimize $L$ while fixing $S$ (equivalent to performing truncated SVD on $W - S$), and optimize $S$ while fixing $L$ (equivalent to performing structured sparse selection on $W - L$). Once the iterations converge, the two components are adapted to each other, achieving a lower reconstruction error than a single-step decomposition. Instead of simple weight MSE, the optimization objective uses the MSE of the output activations on the calibration dataset.
    - **Design Motivation**: A one-time decomposition cannot handle the coupling relationship between low-rank and sparse components. Iterative optimization allows the two components to adjust to each other, achieving a global optimum.

### Loss & Training
The optimization objective is to minimize the output reconstruction error on calibration data: $\min_{L,S} \|WX - (L+S)X\|_F^2$, where $X$ represents the input activations of the calibration data. The entire process requires no backpropagation or gradient computation and is entirely based on matrix operations, taking only a few minutes per layer. Once the decomposition is complete, a few epochs of fine-tuning can be optionally performed to recover end-to-end performance.

## Key Experimental Results

### Main Results

| Model | Compression Ratio | Method | WikiText PPL | C4 PPL | Average Task Accuracy |
|------|--------|------|-------------|--------|--------------|
| LLaMA-2-7B | 2× | SVD | 8.92 | 11.34 | 61.2% |
| LLaMA-2-7B | 2× | ASVD | 7.84 | 10.21 | 63.8% |
| LLaMA-2-7B | 2× | Ours | **7.12** | **9.53** | **66.4%** |
| LLaMA-2-7B | 4× | SVD | 25.6 | 31.2 | 48.3% |
| LLaMA-2-7B | 4× | GPTQ-4bit | 8.35 | 10.89 | 63.1% |
| LLaMA-2-7B | 4× | Ours | **7.96** | **10.12** | **64.7%** |
| LLaMA-2-13B | 2× | Ours | 6.38 | 8.71 | 69.2% |

### Ablation Study

| Configuration | WikiText PPL | Description |
|------|-------------|------|
| Full model (low-rank + sparse + iterative) | 7.12 | Complete model |
| Low-rank only (no sparse) | 8.92 | Degenerates to standard SVD, PPL +1.80 |
| Low-rank + unstructured sparse | 7.45 | Unstructured sparsity is effective but has low hardware efficiency |
| Low-rank + structured sparse (no iteration) | 7.58 | One-time decomposition is inferior to iterative optimization |
| Fixed rank (globally uniform) | 7.89 | Adaptive rank selection makes a significant contribution |

### Key Findings
- Structured sparse residual capture contributes the most; removing it increases PPL by 1.80 points, indicating that outliers are critical to model performance.
- Iterative optimization further reduces PPL by 0.46 compared to single-step decomposition, with convergence usually achieved in 3-5 iterations.
- Under a high 4× compression ratio, the proposed method achieves performance close to 4-bit GPTQ quantization, and the two methods can be orthogonally combined for further compression.
- Adaptive rank allocation reveals that attention layers require higher ranks, while FFN layers can be approximated with lower ranks and more sparse elements.

## Highlights & Insights
- The hybrid decomposition approach combining low-rank and structured sparsity is highly elegant. It cleverly exploits the intrinsic structural properties of LLM weight matrices, unifying two complementary compression paradigms in a single framework.
- The layer-wise decomposition scheme, which does not require backpropagation, makes the compression process extremely efficient, enabling the compression of a 70B model within a few hours.
- The design of structured sparsity allows the compressed model to utilize existing GPU libraries for sparse matrix acceleration, achieving true inference acceleration rather than merely reducing storage footprint.

## Limitations & Future Work
- The decomposition process relies on a calibration dataset, and the domain distribution of the calibration data may affect performance on different downstream tasks.
- The granularity of structured sparsity (column-wise) may not be optimal for all architectures; row-wise or block-wise sparsity might be more appropriate for certain layers.
- The strategy for combining this approach with quantization methods deserves further exploration; the triple compression of low-rank + sparse + quantization holds great potential.
- Currently validated only on decoder-only models, its effectiveness on encoder-decoder architectures remains to be verified.

## Related Work & Insights
- **vs ASVD**: ASVD also employs adaptive SVD decomposition but does not consider the sparse structure in the residuals. This work significantly improves performance under high compression ratios by introducing structured sparse components.
- **vs SparseGPT**: SparseGPT is a pure sparse pruning method that does not utilize low-rank structures. The proposed hybrid decomposition performs better given the same number of effective parameters.
- **vs GPTQ**: GPTQ is a quantization method that is orthogonal to the proposed decomposition method. The two can be combined to achieve more extreme compression.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The hybrid decomposition idea of low-rank + structured sparsity is innovative, though individual components themselves are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Verified across multiple models and compression ratios, with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The methodology is clearly described, and the motivation is logically derived.
- **Value**: ⭐⭐⭐⭐ Offers practical reference value for efficient LLM deployment, and the method can be orthogonally combined with other compression techniques.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)
- [\[ICLR 2026\] LeSTD: LLM Compression via Learning-based Sparse Tensor Decomposition](../../ICLR2026/model_compression/lestd_llm_compression_via_learning-based_sparse_tensor_decomposition.md)
- [\[ACL 2025\] CFSP: An Efficient Structured Pruning Framework for LLMs with Coarse-to-Fine Activation Information](cfsp_an_efficient_structured_pruning_framework_for_llms_with_coarse-to-fine_acti.md)
- [\[ACL 2025\] Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition](assigning_distinct_roles_to_quantized_and_low-rank_matrices_toward_optimal_weigh.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)

</div>

<!-- RELATED:END -->
