---
title: >-
  [Paper Note] IMPACT: Importance-Aware Activation Space Reconstruction
description: >-
  [ACL 2026][Model Compression][Low-rank compression] This paper proposes IMPACT, a framework that shifts LLM low-rank compression from minimizing weight reconstruction error to minimizing importance-weighted activation reconstruction error. By incorporating gradient information into the activation covariance matrix, IMPACT derives a closed-form optimal solution, achieving up to 55.4% model size reduction while preserving accuracy.
tags:
  - ACL 2026
  - Model Compression
  - Low-rank compression
  - activation space reconstruction
  - importance-aware
  - gradient weighting
  - large language models
date: 2026-05-08
content_hash: 83cc54ce66f91297
---

# IMPACT: Importance-Aware Activation Space Reconstruction

**Conference**: ACL 2026
**arXiv**: [2507.03828](https://arxiv.org/abs/2507.03828)
**Code**: Unavailable
**Area**: Model Compression
**Keywords**: Low-rank compression, activation space reconstruction, importance-aware, gradient weighting, large language models

## TL;DR

This paper proposes IMPACT, a framework that shifts LLM low-rank compression from minimizing weight reconstruction error to minimizing importance-weighted activation reconstruction error. By incorporating gradient information into the activation covariance matrix, IMPACT derives a closed-form optimal solution, achieving up to 55.4% model size reduction while preserving accuracy.

## Background & Motivation

**Background**: Large language models (LLMs) achieve strong performance across a wide range of tasks, yet their massive parameter counts make deployment in resource-constrained environments challenging. Low-rank compression is a common technique that reduces parameter counts and computational cost by decomposing weight matrices into low-rank approximations.

**Limitations of Prior Work**: Traditional low-rank compression methods (e.g., SVD-based weight decomposition) assume that weight matrices inherently possess low-rank structure, an assumption that LLM weight matrices frequently violate, leading to suboptimal compression. Some methods instead minimize activation reconstruction error—since LLM activation spaces exhibit more pronounced low-rank structure—but they treat all activation dimensions equally, ignoring the varying contributions of different dimensions to model performance.

**Key Challenge**: Optimizing compression purely from the perspective of reconstruction error minimization is insufficient. The ultimate goal of compression is to preserve model output quality, not to minimize reconstruction error per se. Different activation dimensions vary dramatically in their importance to final predictions, and treating them uniformly leads to accuracy degradation.

**Goal**: To design a compression framework that directly ties the compression objective to model performance—prioritizing the retention of activation dimensions most critical to model outputs rather than minimizing global reconstruction error.

**Key Insight**: The authors observe that LLM activation spaces exhibit substantially more pronounced low-rank structure than weight spaces, and that gradient signals can quantify the sensitivity of each activation dimension to the model's loss. Combining these two observations enables the formulation of a compression optimization problem explicitly oriented toward accuracy preservation.

**Core Idea**: Gradient information is embedded as importance weights into the activation covariance matrix, from which a closed-form optimal compression basis is derived, enabling low-rank compression that explicitly targets accuracy preservation.

## Method

### Overall Architecture

IMPACT takes as input the weight matrices of a pretrained LLM and a small calibration dataset, and outputs low-rank compressed weight matrices. The overall pipeline consists of three steps: (1) collecting activations and gradient information via forward passes on the calibration data; (2) constructing an importance-weighted activation covariance matrix; and (3) solving for the optimal low-rank compression basis via eigendecomposition, from which compressed weights are obtained.

### Key Designs

1. **Activation-Based Compression**:

    - Function: Shifts the compression objective from minimizing weight reconstruction error to minimizing activation reconstruction error.
    - Mechanism: Traditional methods directly decompose the weight matrix $W$ via SVD as $W \approx UV$, which assumes $W$ is inherently low-rank. IMPACT instead minimizes $\|WX - \hat{W}X\|$, i.e., the activation output error, where $X$ denotes the actual activation inputs. This makes the compression basis data-driven, naturally adapting to the model's actual usage patterns.
    - Design Motivation: LLM activation spaces exhibit clearer low-rank structure than weight spaces; using activations as the optimization target better exploits this property.

2. **Gradient-Based Importance Weighting**:

    - Function: Assigns importance weights based on task sensitivity to different activation dimensions.
    - Mechanism: Gradient magnitudes for each activation dimension are computed over the calibration data as importance indicators. The optimization objective is modified from $\|WX - \hat{W}X\|^2$ to $\sum_i \lambda_i \|w_i x - \hat{w}_i x\|^2$, where $\lambda_i$ are importance weights derived from gradients. Activation dimensions with greater influence on model outputs are thus prioritized during compression.
    - Design Motivation: Uniformly minimizing reconstruction error may waste the "rank budget" on unimportant dimensions, whereas gradient information directly reflects each dimension's influence on the loss function.

3. **Closed-Form Solution via Importance-Weighted Covariance**:

    - Function: Reformulates the importance-weighted compression optimization problem as an eigenvalue problem admitting a direct closed-form solution.
    - Mechanism: An importance-weighted activation covariance matrix $C = X \Lambda X^T$ is constructed (where $\Lambda$ is the importance diagonal matrix), and its eigendecomposition yields the top-$k$ eigenvectors as the compression basis. This solution is globally optimal and requires no iterative optimization. The compressed weight is expressed as $\hat{W} = WP_k$, where $P_k$ is the projection matrix formed by the top-$k$ eigenvectors.
    - Design Motivation: The closed-form solution eliminates the computational overhead of iterative optimization while guaranteeing mathematical optimality, making the method both efficient and theoretically well-founded.

## Key Experimental Results

### Main Results

| Model | Compression Rate | IMPACT Perplexity | Baseline SOTA Perplexity | Size Reduction Advantage |
|-------|-----------------|-------------------|--------------------------|--------------------------|
| LLaMA-2-7B | 20% | On par with baseline | ASVD/SliceGPT | Maintains accuracy at higher compression rates |
| LLaMA-2-13B | 25% | Outperforms baseline | Weight SVD methods | 55.4% greater model size reduction |
| OPT-6.7B | 20% | Outperforms baseline | Activation-aware methods | Significantly lower perplexity |
| LLaMA-3-8B | 30% | Comparable to baseline | ASVD | Preserves performance under more aggressive compression |

### Ablation Study

| Configuration | Effect | Description |
|---------------|--------|-------------|
| Full IMPACT | Best | Complete importance weighting + activation reconstruction |
| w/o gradient weighting (uniform weights) | Perplexity increases | Demonstrates the critical contribution of importance weighting |
| Weight-space reconstruction (standard SVD) | Significant degradation | Demonstrates the superiority of activation space over weight space |
| Varying calibration set size | Stable with 256 samples | Method is insensitive to calibration data volume |

### Key Findings

- Gradient importance weighting is the largest single contributor to performance gains—removing it degrades the method to the level of standard activation reconstruction.
- IMPACT's advantage becomes more pronounced at higher compression rates: the lower the retained rank, the more significant the accuracy preservation effect of importance weighting.
- The method generalizes well across different model families (LLaMA, OPT, etc.) and different model scales.
- The closed-form solution makes compression substantially more efficient than methods requiring iterative optimization, with per-layer compression completing in seconds.

## Highlights & Insights

- The strategy of **decoupling and then re-associating compression with performance** is particularly elegant: rather than directly minimizing a proxy loss, gradient signals are used to align the compression objective with final task performance while retaining the elegance of a closed-form solution.
- The **activation space vs. weight space** insight is broadly applicable: the finding that LLM activations are more low-rank than weights can inform the design of other compression and quantization methods.
- The **importance-weighted covariance matrix** design is transferable to other settings requiring low-rank approximation, such as selecting important subspaces during LoRA fine-tuning or feature alignment in knowledge distillation.

## Limitations & Future Work

- Evaluation is primarily conducted on language modeling perplexity; the impact on downstream tasks (question answering, reasoning, etc.) is assessed only to a limited extent.
- Gradient information depends on the calibration data distribution; large discrepancies between calibration data and actual deployment scenarios may affect performance.
- Compression is performed independently per layer without considering inter-layer interactions; jointly optimizing compression bases across layers may yield further improvements.
- Combining this approach with quantization (low-rank + quantization) is a promising direction that the paper does not explore in depth.

## Related Work & Insights

- **vs. ASVD**: ASVD also leverages activation information for SVD but does not incorporate importance weighting, leading to substantial accuracy loss at high compression rates; IMPACT significantly addresses this limitation through gradient weighting.
- **vs. SliceGPT**: SliceGPT removes parameters via orthogonal transformations and is a structured pruning approach; IMPACT retains the low-rank decomposition framework but optimizes basis selection, and the two methods are complementary.
- **vs. GPTQ/AWQ**: These are quantization methods rather than low-rank decomposition; IMPACT can be combined with them to achieve higher overall compression ratios.

## Rating

- Novelty: ⭐⭐⭐⭐ Incorporating gradient importance into the activation covariance matrix for closed-form low-rank compression is a clean and elegant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-compression-rate comparisons are comprehensive; ablation studies validate the contribution of each component.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear; the motivation is presented with a complete logical chain.
- Value: ⭐⭐⭐⭐ Provides a concise and efficient LLM compression method that is practically useful and easy to understand and implement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] KVmix: Gradient-Based Layer Importance-Aware Mixed-Precision Quantization for KV Cache](../../AAAI2026/model_compression/kvmix_gradient-based_layer_importance-aware_mixed-precision_.md)
- [\[ACL 2026\] Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](analytical_ffn-to-moe_restructuring_via_activation_pattern_analysis.md)
- [\[ACL 2026\] Enabling Agents to Communicate Entirely in Latent Space](enabling_agents_to_communicate_entirely_in_latent_space.md)
- [\[CVPR 2026\] GeoFusion-CAD: Structure-Aware Diffusion with Geometric State Space for Parametric 3D Design](../../CVPR2026/model_compression/geofusion-cad_structure-aware_diffusion_with_geometric_state_space_for_parametri.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](../../NeurIPS2025/model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)

</div>

<!-- RELATED:END -->
