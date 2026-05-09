---
title: >-
  [Paper Note] ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization
description: >-
  [NeurIPS 2025][Model Compression][Depth Pruning] ReplaceMe is a training-free depth pruning method that uses a small calibration dataset to estimate a linear transformation approximating groups of pruned Transformer blocks. This transformation is fused into adjacent layer weights without introducing additional parameters, achieving 25% pruning on LLaMA-2-7B while retaining approximately 90% of original performance.
tags:
  - NeurIPS 2025
  - Model Compression
  - Depth Pruning
  - Transformer Linearization
  - Training-Free Compression
  - LLM Acceleration
  - Layer Selection
date: 2026-05-08
content_hash: 9b80f4cf856663bc
---

# ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization

**Conference**: NeurIPS 2025
**arXiv**: [2505.02819](https://arxiv.org/abs/2505.02819)
**Code**: [https://github.com/mts-ai/ReplaceMe](https://github.com/mts-ai/ReplaceMe)
**Area**: Model Compression
**Keywords**: Depth Pruning, Transformer Linearization, Training-Free Compression, LLM Acceleration, Layer Selection

## TL;DR
ReplaceMe is a training-free depth pruning method that uses a small calibration dataset to estimate a linear transformation approximating groups of pruned Transformer blocks. This transformation is fused into adjacent layer weights without introducing additional parameters, achieving 25% pruning on LLaMA-2-7B while retaining approximately 90% of original performance.

## Background & Motivation

**Background**: Structured pruning is one of the primary approaches for LLM compression, but most methods require post-pruning retraining (a "healing" phase) that incurs substantial computational overhead.

**Limitations of Prior Work**:
- Directly removing Transformer blocks causes irrecoverable information loss, since deletion implicitly assumes the layer acts as an identity mapping, which does not hold in practice.
- Existing training-free methods (e.g., LaCo, SliceGPT) yield insufficient performance.
- Methods requiring healing (e.g., UIDL, LLMPruner) achieve better results but demand several hours of training and tens of GPU-hours.

**Key Challenge**: Removing a layer is equivalent to assuming it performs an identity mapping, whereas in practice layers perform non-trivial transformations.

**Key Insight**: Approximating the transformation of removed layers with a linear mapping, which is substantially more accurate than the identity assumption.

**Core Idea**: Identify the group of consecutive Transformer blocks with minimal impact, approximate their input–output mapping with a linear transformation $\mathbf{T}$, and fuse $\mathbf{T}$ into the down-projection weights of the adjacent MLP. The resulting model retains the original architecture and introduces no additional parameters.

## Method

### Overall Architecture
ReplaceMe proceeds in four steps: (1) **Layer Selection**—identify the consecutive $n$ layers whose removal minimally affects the output, using cosine distance; (2) **Linear Transformation Estimation**—solve for the optimal linear mapping $\mathbf{T}^*$ on calibration data such that $\mathbf{M}_i \cdot \mathbf{T} + \mathbf{Y}_i \approx \mathbf{L}_{i+n}$; (3) **Weight Fusion**—absorb $\mathbf{T}^*$ into the down-projection matrix of the preceding MLP layer; (4) Remove the $n$ Transformer blocks. The entire process requires **zero training and zero additional parameters**.

### Key Designs

1. **Layer Selection Strategy**:

    - *Function*: Identify the optimal starting position $i^*$ for pruning $n$ consecutive layers.
    - *Mechanism*: $i^* = \arg\min_i h(\mathbf{L}_i, \mathbf{L}_{i+n})$, where cosine distance measures the discrepancy between hidden states at layer $i$ and layer $i+n$, averaged over calibration data.
    - *Design Motivation*: Exhaustive validation confirms that cosine distance reliably identifies optimal or near-optimal pruning positions, outperforming L2 distance.
    - *Comparison with Brute-Force Search*: An exhaustive enumeration of all candidate pruning positions in the Appendix confirms the accuracy of cosine-distance-based localization.

2. **Linear Transformation Estimation — L2 Objective**:

    - *Function*: Solve $\mathbf{T}^* = \arg\min_\mathbf{T} \|\mathbf{M}_i \cdot \mathbf{T} + \mathbf{Y}_i - \mathbf{L}_{i+n}\|_2^2$.
    - *Mechanism*: Classical least-squares solution $\mathbf{T}^* = (\mathbf{M}_i^T \mathbf{M}_i)^{-1} \mathbf{M}_i^T (\mathbf{L}_{i+n} - \mathbf{Y}_i)$, which admits a closed-form solution requiring no iterative optimization.
    - Here $\mathbf{M}_i$ denotes the MLP output at layer $i$, $\mathbf{Y}_i$ the attention sub-block output, and $\mathbf{L}_{i+n}$ the output of layer $i+n$.

3. **Linear Transformation Estimation — Cosine Objective**:

    - *Function*: $\mathbf{T}^* = \arg\min_\mathbf{T} \sum_k \left(1 - \frac{(\mathbf{M}_{i,k} \cdot \mathbf{T} + \mathbf{Y}_{i,k})^\top \mathbf{L}_{i+n,k}}{\|\cdot\| \|\cdot\|}\right)$
    - *Mechanism*: No closed-form solution exists; Adam optimization is applied (lr=1e-4, 10 epochs). In practice, a simplified formulation $\cos(\mathbf{M}_i \mathbf{T},\ \mathbf{L}_{i+n} - \mathbf{Y}_i)$ is used to reduce memory consumption.
    - *Design Motivation*: Cosine distance emphasizes directional alignment over magnitude, which is more appropriate for the residual stream in Transformers.

4. **Weight Fusion**:

    - *Function*: $\mathbf{T}^*$ is multiplied into the down-projection matrix of the MLP at layer $i$, yielding an updated down-projection matrix.
    - *Design Motivation*: MLP output × linear transformation = two consecutive linear operations = a single equivalent linear operation that can be merged into one matrix. After fusion, the model architecture is entirely unchanged and no modification to inference code is required.

5. **Regularization**:

    - L1/L2 regularization improves benchmark accuracy but increases perplexity—a trade-off between accuracy and perplexity exists.
    - An optional Multi-LT configuration applies independent linear transformations to multiple non-overlapping block groups.

### Loss & Training
- **Completely training-free**—only a small calibration dataset is required (e.g., 512 samples from C4/RedPajama).
- The choice of calibration data has limited impact (experiments show less than 1% variation across different sources).
- The analytical approach (L2 objective) completes near-instantaneously; the numerical approach (cosine objective) requires approximately 10 epochs of Adam optimization.

## Key Experimental Results

### Main Results — LLaMA-2-7B (25% Pruning = Removing 8 of 32 Layers)

| Method | Training Required? | C3 | HellaSwag | PIQA | MMLU | Avg. | Retention |
|--------|-------------------|----|-----------|------|------|------|-----------|
| Unpruned | - | 43.8 | 71.3 | 78.1 | 46.8 | 45.3 | 100% |
| LLMPruner | ✓ | 29.7 | 54.6 | 72.0 | 25.3 | 35.4 | 78.2% |
| SliceGPT | ✓ | 31.5 | 47.5 | 68.3 | 28.8 | 35.1 | 77.5% |
| LaCo | ✓ | 39.7 | 55.7 | 69.8 | 26.5 | 37.4 | 82.7% |
| UIDL | ✓ | 40.2 | 59.7 | 69.0 | 44.6 | 40.9 | 90.3% |
| **ReplaceMe (Cosine)** | **✗** | **42.4** | **64.7** | **73.5** | **45.1** | **41.9** | **92.5%** |

### Ablation Study

| Configuration | Avg. Accuracy | Perplexity | Note |
|---------------|--------------|------------|------|
| Direct deletion (identity mapping) | 37.4 | High | No compensation |
| + L2 linear transformation | 40.8 | Medium | Closed-form solution |
| + Cosine linear transformation | **41.9** | Medium | Numerical optimization |
| + Cosine + L1 regularization | 42.1 | Higher | Accuracy↑ Perplexity↑ |

### Cross-Model Validation

| Model | Pruning Rate | ReplaceMe Retention | Best Competitor |
|-------|-------------|---------------------|-----------------|
| LLaMA-2-7B | 25% | **92.5%** | UIDL 90.3% |
| LLaMA-3-8B-Instruct | 25% | **91.8%** | UIDL 89.5% |
| Qwen2.5-7B | 25% | ~90% | - |
| Falcon-11B | 25% | ~89% | - |
| ViT (Vision) | 25% | **93%** | - |

### Key Findings
- **Training-free outperforms training-based (for most methods)**: ReplaceMe without any training surpasses UIDL with LoRA healing (92.5% vs. 90.3%), demonstrating that linear transformation compensation is more effective than post-hoc retraining.
- Cosine objective outperforms L2 objective (+1%), consistent with the finding that directional alignment matters more than magnitude—mirroring the superiority of cosine distance in layer selection.
- Calibration data volume has minimal impact: 512 samples suffice, and increasing to 2K yields less than 0.5% improvement.
- ReplaceMe's CO₂ emissions are only 1/50 of UIDL's—a genuinely green compression approach.

## Highlights & Insights
- The hypothesis that **"consecutive Transformer blocks ≈ a linear transformation"** is thoroughly validated experimentally—this is itself an interesting finding regarding the representational structure of Transformers, suggesting that many intermediate layers in deep Transformers perform only directional refinements.
- **The weight fusion technique is remarkably elegant**: linear transformation × MLP down-projection = a new down-projection, with zero additional parameters and zero inference overhead. This is the most practically significant engineering contribution of the paper.
- Cosine distance is employed **dually**—in both layer selection and transformation estimation—confirming that directional metrics are more suitable than magnitude metrics for the Transformer residual stream.
- The method generalizes effectively to ViTs, extending its applicability to vision Transformers.

## Limitations & Future Work
- Performance degrades rapidly beyond a 25% pruning rate, as the linearity assumption breaks down at higher compression ratios.
- The current approach can only prune **consecutive** blocks; non-consecutive optimal pruning positions may yield better results (the Multi_LT_NC variant partially explores this direction but with limited gains).
- Linear transformations do not capture nonlinear mappings; replacing pruned blocks with low-rank nonlinear mappings or lightweight adapters may be more expressive.
- Performance degrades more on instruction-tuned (Instruct) models than on base models, possibly because instruction-tuned models exhibit stronger inter-layer dependencies.

## Related Work & Insights
- **vs. ShortGPT/LaCo**: These methods directly delete layers (implicitly assuming identity mappings); ReplaceMe's linear mapping compensation is the core improvement.
- **vs. UIDL**: UIDL requires LoRA healing training, whereas ReplaceMe is entirely training-free yet achieves superior results—indicating that the quality of initial compensation matters more than subsequent fine-tuning.
- **vs. SliceGPT**: SliceGPT performs width pruning while ReplaceMe performs depth pruning; the two approaches are orthogonal and can be combined.
- **Insight**: If a linear transformation suffices to approximate intermediate layers well, it may be possible to constrain intermediate layers toward linearity **during training**, thereby designing architectures that are inherently more compressible.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Using linear transformations as layer substitutes is a concise and effective new approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple models, benchmarks, ablation configurations, ViT extension, and CO₂ comparison.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations in the method section are clear; experiments are comprehensive.
- **Value**: ⭐⭐⭐⭐⭐ — Training-free + no additional parameters + strong performance retention = an immediately deployable LLM compression solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Graphon Limit Hypothesis: Understanding Neural Network Pruning via Infinite Width Analysis](the_graphon_limit_hypothesis_understanding_neural_network_pruning_via_infinite_w.md)
- [\[NeurIPS 2025\] Spark Transformer: Reactivating Sparsity in FFN and Attention](spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)
- [\[NeurIPS 2025\] Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)
- [\[NeurIPS 2025\] On the Creation of Narrow AI: Hierarchy and Nonlocality of Neural Network Skills](on_the_creation_of_narrow_ai_hierarchy_and_nonlocality_of_neural_network_skills.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)

</div>

<!-- RELATED:END -->
