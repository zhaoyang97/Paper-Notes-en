---
title: >-
  [Paper Note] LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions
description: >-
  [NeurIPS 2025][Model Compression][Influence Functions] LayerIF proposes using influence functions (IFs) to quantify the training quality of each layer in LLMs. By aggregating positive influence scores per layer…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Influence Functions"
  - "Layer Quality Estimation"
  - "LoRA-MoE"
  - "Model Pruning"
  - "Layer-wise Sparsity Allocation"
date: 2026-05-08
content_hash: 9834d22cac95ed0b
---

# LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions

**Conference**: NeurIPS 2025
**arXiv**: [2505.23811](https://arxiv.org/abs/2505.23811)
**Code**: None
**Area**: Model Compression
**Keywords**: Influence Functions, Layer Quality Estimation, LoRA-MoE, Model Pruning, Layer-wise Sparsity Allocation

## TL;DR
LayerIF proposes using influence functions (IFs) to quantify the training quality of each layer in LLMs. By aggregating positive influence scores per layer, it derives a data-driven layer importance estimate, which is subsequently applied to two downstream tasks: LoRA-MoE expert allocation and layer-wise sparse pruning. The method achieves accuracy gains of 1.61% and 0.90% on Mistral-7B and Gemma-7B, respectively.

## Background & Motivation

**Background**: Training quality varies significantly across layers of LLMs. Prior studies have shown that intermediate layers provide stronger downstream representations, while the deepest layers can be pruned with minimal loss. Layer quality estimation is critical for downstream tasks such as model compression (pruning) and efficient fine-tuning (LoRA-MoE expert allocation).

**Limitations of Prior Work**: Existing layer quality estimation methods rely primarily on model-side heuristics: AlphaPruning uses the heavy-tailed spectral density of weight matrices (PL_Alpha_Hill), OWL uses outlier activation distributions, and MoLA uses fixed allocation patterns (e.g., 2-4-6-8). These approaches entirely ignore the influence of training data.

**Key Challenge**: Layer-wise specialization patterns differ across tasks and datasets for the same model; methods based solely on weight statistics cannot capture such task-specific variation.

**Goal**: To propose a data-and-model co-driven layer quality estimation framework capable of producing different layer importance scores for the same LLM across different tasks.

**Key Insight**: Influence functions quantify the sensitivity of model loss to perturbations of individual training samples. Well-trained layers are less sensitive to data perturbations (small IF values), while undertrained layers are more sensitive (large IF values)—this sensitivity can serve as a proxy for layer quality.

**Core Idea**: Layer-wise influence function scores are used as data-driven proxies for layer training quality, guiding MoE expert allocation and sparse pruning to achieve task-adaptive, layer-wise resource allocation.

## Method

### Overall Architecture
**Inputs**: a pre-trained LLM, training data $D^{train}$, and validation data $D^{\mathcal{V}}$. For each Transformer block, layer-wise influence function scores are computed → aggregated into a scalar layer quality estimate → mapped via a transfer function to layer-wise resource allocation (number of experts or sparsity ratio) for downstream tasks.

### Key Designs

1. **Layer-wise Influence Function Computation (LayerIF Core)**:

    - **Function**: Quantifies each layer's contribution to the training–validation interaction.
    - **Core formula**: $I^{(l)}(z_i) = -\sum_{j=1}^m \nabla_{\theta^{(l)}} \ell(z_j^{\mathcal{V}}, \theta)^\top (H^{(l)}(\theta))^{-1} \nabla_{\theta^{(l)}} \ell(z_i, \theta)$
    - **Aggregation**: Only positively influential samples (beneficial to the model) are retained and summed per layer: $S^{(l)} = \sum_{i=1}^n \mathbb{I}[I^{(l)}(z_i) > 0] \cdot I^{(l)}(z_i)$
    - **Key insight**: Layers with high $S^{(l)}$ are more sensitive to training data, indicating insufficient training; layers with low $S^{(l)}$ are more stable and better trained.
    - The DataInf method is used to approximate the inverse Hessian, achieving approximately 160× speedup over the Gauss-Newton Hessian.

2. **LoRA-MoE Expert Allocation**:

    - **Function**: Non-uniformly distributes a fixed number of experts across layers according to layer quality.
    - **Mapping steps**: (a) Sign flip $\tilde{v}_i = -v_i$ so that higher-quality layers receive higher scores; (b) Power transformation $\hat{v}_i = \tilde{v}_i^\beta$ to control allocation skewness; (c) Normalization to the target total expert count $T$, ensuring at least one expert per layer.
    - **Final allocation**: $s_i = \lfloor f_i \rfloor + 1$, with remaining experts distributed to the top-$r$ layers by fractional part.
    - **Routing strategy**: Top-K routing + softmax gating + load balancing loss.
    - **Per-layer output**: $h^{i,t} = W_0^{i,t}x + \sum_{j=1}^K S_j^{i,t}(x) A_j^{i,t} B_j^{i,t} x$

3. **Layer-wise Sparse Pruning Allocation**:

    - **Function**: Assigns different pruning ratios to different layers (higher pruning for undertrained layers, lower for well-trained layers).
    - **Preprocessing**: Take absolute values → min-max normalization to [0,1] → Savitzky-Golay filtering to smooth noise.
    - **Mapping function**: $\phi(\tilde{s})_i = \eta \left(\frac{\tilde{s}_i - \tilde{s}_{\min}}{\tilde{s}_{\max} - \tilde{s}_{\min}}(e_2 - e_1) + e_1\right)$
    - **Constraint**: Global sparsity $\sum_{i=1}^L \phi(\tilde{s})_i \cdot d_i = S \cdot \sum_{i=1}^L d_i$
    - Compatible with three mainstream pruning methods: Magnitude, Wanda, and SparseGPT.

### Design Characteristics
- **Model-agnostic**: Applicable to any Transformer architecture.
- **Task-specific**: The same model yields different layer-wise allocations for different datasets (verified via heatmap visualization).
- **Per-block rather than per-matrix**: Ablation studies show that computing IFs at the full Transformer block granularity outperforms computing them separately for Attention and MLP sub-modules (+2.14% accuracy).

## Key Experimental Results

### LoRA-MoE Expert Allocation (Mistral-7B, 160 experts)

| Method | MRPC | CoLA | TextSciQ | CommonQ | OpenBookQ | Avg |
|------|------|------|----------|---------|-----------|-----|
| AlphaLora | 75.30 | 82.07 | 78.52 | 82.06 | 84.60 | 80.51 |
| MoLA-5555 | 82.02 | 82.26 | 64.65 | 80.91 | 64.00 | 74.77 |
| MoLA-8642 | 80.70 | 84.56 | 76.39 | 81.00 | 82.20 | 80.97 |
| **LayerIF (Top 25%)** | **82.63** | **83.80** | **79.59** | 80.75 | 84.60 | **82.27** |

The best LayerIF configuration achieves an average accuracy of 82.27%, representing significant improvements over MoLA-8642 (+1.30%) and AlphaLora (+1.76%).

### Layer-wise Sparse Pruning (Mistral-7B, 50% sparsity)

| Method | Magnitude | Wanda | SparseGPT | Avg |
|------|-----------|-------|-----------|-----|
| Uniform | 56.41 | 58.71 | 60.45 | 58.52 |
| AlphaPruning | 56.74 | 58.48 | 60.07 | 58.43 |
| OWL | 56.16 | 58.75 | 59.99 | 58.30 |
| **LayerIF** | **56.89** | **58.94** | **60.61** | **58.81** |

LayerIF achieves the best performance across all three pruning methods, with an average improvement of 0.90%.

### Ablation: Reversed Allocation Strategy (70% sparsity, Mistral-7B)

| Method | Magnitude | Wanda | SparseGPT | Avg |
|------|-----------|-------|-----------|-----|
| LayerIF (normal) | 33.44 | 32.50 | 41.14 | 35.69 |
| Reversed allocation | 33.10 | 32.02 | 38.45 | 34.52 |

Performance degrades noticeably under reversed allocation, validating the hypothesis that low IF sensitivity implies mature training and thus warrants greater preservation.

### Key Findings
- LayerIF's layer-wise allocations differ substantially across datasets (visualized via heatmaps), confirming task specificity.
- Performance advantages are maintained across three expert-count settings (80, 160, 224), demonstrating robustness.
- Per-block computation outperforms per-matrix computation (+2.14%).
- The Spearman correlation between Hessian-free (TracIn) and Hessian-based (DataInf) methods is low (average ~0.15), indicating that Hessian information is critical for layer quality estimation.
- Aggregating the top 25% of positively influential samples is the optimal strategy, as IF values follow a heavy-tailed distribution.

## Highlights & Insights
- **Dual data-and-model perspective**: This is the first application of influence functions to layer quality estimation, addressing the limitation of purely model-side heuristics that ignore data effects. This paradigm can be generalized to any scenario requiring quantification of the importance of model sub-components.
- **Task-adaptive allocation**: The same model produces different layer-wise allocations for different tasks, which is more principled than fixed patterns such as MoLA's 2-4-6-8. The heatmap comparisons are particularly intuitive.
- **General framework design**: LayerIF is decoupled from specific IF implementations (DataInf can be replaced by future more efficient IF methods) and from specific pruning methods.
- **Practical value of Savitzky-Golay smoothing**: IF scores fluctuate considerably between adjacent layers due to Hessian approximation noise; simple smoothing substantially improves stability.

## Limitations & Future Work
- IF computation incurs non-trivial overhead (requiring Hessian approximation); while the paper argues this is negligible relative to pre-training time, actual wall-clock time warrants further attention.
- Validation is conducted only on 7B-scale models; scalability to larger models (70B+) remains uncertain.
- The hyperparameter $\beta$ (power transformation exponent) requires tuning, and the paper provides no automated selection procedure.
- Layer-wise granularity may be too coarse; finer granularities (e.g., per-head or per-neuron) could yield better results.
- Only NLP tasks are evaluated; applicability to vision and multimodal models remains unexplored.

## Related Work & Insights
- **vs. AlphaPruning**: Estimates layer quality via heavy-tailed spectral density (PL_Alpha_Hill), a purely model-side approach; LayerIF consistently outperforms it by 0.90% on pruning tasks by incorporating a data perspective.
- **vs. OWL**: Performs non-uniform sparsity allocation based on outlier activation distributions; LayerIF surpasses OWL at all tested sparsity levels.
- **vs. MoLA/AlphaLora**: MoLA uses fixed allocation patterns for experts; AlphaLora adaptively uses heavy-tail theory but ignores data; LayerIF's data-driven allocation automatically adjusts across different datasets.
- **vs. Shapley Value methods**: Shapley values can also estimate layer contributions but at exponentially higher computational cost; LayerIF achieves greater efficiency through gradient-based IF approximations.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying influence functions to layer quality estimation is a meaningful new direction, though IFs themselves are an established tool.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two downstream tasks, two models, multiple pruning methods, comprehensive ablation and comparison studies — though limited to 7B-scale models.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear and the pipeline figure is intuitive, though some formulas are verbose.
- **Value**: ⭐⭐⭐⭐ Provides a novel data-driven perspective for MoE expert allocation and pruning, with clearly demonstrated practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Simple Linear Patch Revives Layer-Pruned Large Language Models](a_simple_linear_patch_revives_layerpruned_large_language_mod.md)
- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[NeurIPS 2025\] The Structure of Relation Decoding Linear Operators in Large Language Models](the_structure_of_relation_decoding_linear_operators_in_large_language_models.md)

</div>

<!-- RELATED:END -->
