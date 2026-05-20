---
title: >-
  [Paper Note] FlattenGPT: Depth Compression for Transformer with Layer Flattening
description: >-
  [ICML 2026][Model Compression][LLM Pruning] This paper proposes FlattenGPT, which first "flattens" and merges adjacent transformer layers in LLMs with highly similar inputs into a single layer with 2× width (retaining al…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "LLM Pruning"
  - "Depth Compression"
  - "Layer Merging"
  - "Channel Pruning"
  - "Nyström Approximation"
date: 2026-05-08
content_hash: 6fd9ba19791c3f0f
---

# FlattenGPT: Depth Compression for Transformer with Layer Flattening

**Conference**: ICML 2026  
**arXiv**: [2602.08858](https://arxiv.org/abs/2602.08858)  
**Code**: Not released  
**Area**: Model Compression / LLM Acceleration / Deep Pruning  
**Keywords**: LLM Pruning, Depth Compression, Layer Merging, Channel Pruning, Nyström Approximation

## TL;DR
This paper proposes FlattenGPT, which first "flattens" and merges adjacent transformer layers in LLMs with highly similar inputs into a single layer with 2× width (retaining all parameter knowledge), then applies channel pruning to the merged layer to restore the original width—thus achieving inference acceleration via depth compression while avoiding the catastrophic performance drop from directly discarding entire layers as in traditional pruning.

## Background & Motivation

**Background**: The high inference cost of LLMs has led to two main pruning approaches. Depth pruning (SLEB, ShortGPT, LaCo) removes entire transformer blocks, yielding high speedup but significant performance loss; channel pruning (LLM-Pruner, SliceGPT) retains all layers but prunes width, preserving performance but with limited acceleration, and inconsistent pruning rates across layers break architectural homogeneity.

**Limitations of Prior Work**: The core issue with depth pruning is "coarse-grained deletion"—some heads/channels in a block may have learned crucial knowledge, but if the block is deemed "redundant," all its knowledge is lost together. Channel pruning's core issue is "architectural inconsistency"—downstream applications like LoRA, CUDA kernels, and inference engines require a uniform architecture for efficient execution. There is a clear gap between these two approaches.

**Key Challenge**: Depth redundancy objectively exists (the authors prove in Lemma 2.1/2.2 that the variance of deep hidden states grows as $\Theta(\ell^2)$, and gradients degenerate into identity mappings dominated by residuals), but there is no intermediate option between "removing whole blocks" and "retaining all blocks." Theoretically, "merging two layers" is possible, but the challenge is how to merge while reducing parameters without sacrificing performance.

**Goal**: (a) Find an operation to "merge adjacent layers" such that the knowledge of both layers is preserved and can continue to collaborate; (b) After merging, compress back to the original width to maintain architectural homogeneity.

**Key Insight**: In Pre-LN Transformers, the cosine similarity between hidden states of adjacent layers $\mathbf{H}^\ell, \mathbf{H}^{\ell+1}$ is typically >0.9. The authors realize that if two layers' inputs are nearly identical, replacing their MHA and MLP with "parallel execution followed by summation" is mathematically close to the original "serial execution," but the parameters are physically combined into a "double-width" layer—thus transforming the depth problem into a width problem, which can be finely handled by channel pruning.

**Core Idea**: First flatten (depth → width), then channel prune (restore width), chaining the two steps to achieve "knowledge retention + architectural homogeneity + inference acceleration."

## Method

### Overall Architecture
A two-stage pipeline: ① **Iterative Layer Flattening**—compute the cross-layer cosine similarity matrix $\mathbf{S}\in\mathbb{R}^{L\times L}$ for all adjacent layers on a calibration set, greedily select the most similar adjacent pair $(B_{\ell-1}, B_\ell)$ to merge into $B_{\ell-1,\ell}$, and iterate until the target compression rate is reached; ② **Channel Pruning**—for each 2× width merged layer, prune half the heads in the MHA part based on head importance, and use Nyström approximation to select top-k channels in the MLP part and compensate for the remaining information.

### Key Designs

1. **Layer Flattening as Parallel-then-Sum**:

    - **Function**: Merge two adjacent layers $B_{\ell-1}, B_\ell$ into a single layer $B_{\ell-1,\ell}$, preserving all parameter knowledge from both layers.
    - **Mechanism**: First, fuse the affine parameters of LayerNorm $\boldsymbol{\alpha}^{\ell-1}, \boldsymbol{\alpha}^\ell$ into the linear projections $\mathbf{W}_Q/W_K/W_V$ (this does not change the output); then horizontally concatenate $\mathbf{W}_Q^{\ell-1}, \mathbf{W}_Q^\ell$ to obtain $\mathbf{W}_Q^{\ell-1,\ell}\in\mathbb{R}^{d\times 2dh}$, and similarly for $W_K, W_V$; vertically concatenate $\mathbf{W}_O$. For the MLP, similarly concatenate $\mathbf{W}_u, \mathbf{W}_g$ (horizontally) and $\mathbf{W}_D$ (vertically). The merged layer's MHA thus becomes the sum of $2H$ parallel attention heads, and the MLP is a double-width $2d_{int}$ intermediate dimension MLP.
    - **Design Motivation**: Since the inputs to the two layers are similar (cos>0.9), approximating the "serial: $\mathbf{H}_\ell=\mathbf{H}_{\ell-1}+B_\ell(\mathbf{H}_{\ell-1}+B_{\ell-1}(\mathbf{H}_{\ell-1}))$" as "parallel: $\mathbf{H}_\ell\approx \mathbf{H}_{\ell-1}+B_{\ell-1}(\mathbf{H}_{\ell-1})+B_\ell(\mathbf{H}_{\ell-1})$" incurs minimal error; this "additive equivalence" is the geometric premise for layer flattening.

2. **Greedy Layer Selection Based on Similarity Matrix**:

    - **Function**: Decide which adjacent layers to merge and when to stop merging.
    - **Mechanism**: Maintain the upper triangular similarity matrix $\mathbf{S}$, and in each round, merge the adjacent pair with the highest $\mathbf{S}_{\ell-1,\ell}$. Key technique: after merging, delete the $\ell-1$-th column and $\ell$-th row from $\mathbf{S}$, so the similarity between the new merged layer $B^{\ell-1,\ell}$ and other layers is indirectly represented by the original $\mathbf{S}_{\ell-1,i}$ and $\mathbf{S}_{j,\ell}$. This "row and column deletion" mechanism ensures that even when merging more than three layers iteratively, the merge span is still constrained by the "distance between the first and last layers."
    - **Design Motivation**: Greedy selection avoids the NP-hard optimal grouping problem; row and column deletion prevents "flattening layers that are too far apart"—if multiple layers are merged consecutively, the semantics of the first and last layers may diverge, and forced merging would disrupt information flow.

3. **MLP Nyström Channel Pruning + Error Compensation**:

    - **Function**: Compress the $2d_{int}$ width merged MLP back to the original $d_{int}$, while "projecting" the information from discarded channels onto the retained channels.
    - **Mechanism**: First, use the ridge leverage score $s_i=[\mathbf{C}_\psi(\mathbf{C}_\psi+\lambda\mathbf{I})]_{ii}^{-1}$ to measure the importance of channel $i$, select the top-k; then adjust the down matrix using the Nyström formula: $\mathbf{W}_D \leftarrow \mathbf{W}_D + (\mathbf{S}_k^\top\mathbf{C}_\psi\mathbf{S}_k+\lambda\mathbf{I})^{-1}\mathbf{S}_k^\top\mathbf{C}_\psi(\mathbf{I}-\mathbf{S}_k\mathbf{S}_k^\top)\mathbf{W}_D$. Lemma 3.1 proves this is the optimal compensation under L2-regularized least squares. For MHA, channel pruning simply removes heads based on importance $f_i=\mathbb{E}[\text{Softmax}(...)\mathbf{X}\mathbf{W}_{V,i}\text{diag}(\mathbf{W}_{O,i}\mathbf{W}_{O,i}^\top)^{1/2}]$.
    - **Design Motivation**: Pure channel selection discards 50% of the information; Nyström compensation "folds" the covariance of the discarded channels onto the down-projection of the retained channels, theoretically guaranteeing optimal MLP output in terms of minimizing L2 error.

### Loss & Training
Completely training-free, using only 128 WikiText-2 sequences for calibration (to estimate $\mathbf{C}_\psi$). Optional RFT: 50K refined Alpaca + LoRA for 2 epochs with lr=1e-4 and lora_r=8 for recovery fine-tuning.

## Key Experimental Results

### Main Results
Experiments on multiple models and sizes (LLaMA-2/3, Qwen-1.5, Baichuan-2), compared with five SOTA depth pruning methods.

| Model/Method | Sparsity | PPL ↓ | Avg Zero-shot Acc |
|--------|------|------|---------|
| LLaMA-2 7B Dense | 0% | 5.47 | 69.00 |
| ShortGPT | 21% | 18.45 | 58.18 |
| BlockPruner | 22% | 11.51 | 60.17 |
| **FlattenGPT** | 21% | **8.68** | **62.49** |
| LLaMA-2 13B Dense | 0% | 4.88 | 71.76 |
| BlockPruner | 25% | 8.16 | 64.53 |
| **FlattenGPT** | 24% | **6.68** | **67.50** |
| Qwen-1.5 7B Dense | 0% | 7.95 | 65.48 |
| **FlattenGPT** | 21% | **16.05** | **57.00** |

For throughput, LLaMA-2 70B with FlattenGPT at 20% sparsity achieves throughput 1.27× and latency 1.26× speedup, matching SLEB (due to identical architecture) but with 5 points higher accuracy.

### Ablation Study

| Configuration | LLaMA-2 7B Avg Acc |
|------|---------|
| Dense | 69.00 |
| FlattenGPT (no RFT) | 63.83 |
| FlattenGPT + RFT | **66.24** |
| LLM-Pruner + RFT | 62.15 |
| Shortened LLaMA + RFT | 61.91 |

### Key Findings
- At the same sparsity, FlattenGPT outperforms ShortGPT by an average of 5 points, and the strongest baseline BlockPruner by 2-3 points, indicating that "retaining knowledge from both layers before compression" preserves information better than "directly deleting a layer."
- With the same final architecture as SLEB (thus identical throughput), FlattenGPT achieves 5 points higher accuracy, showing that the performance gap is entirely due to the training process (flattening + Nyström compensation) rather than inference optimization.
- LLaMA-2 7B retains 90-96% zero-shot performance (20% compression + RFT), making it one of the strongest combinations at this sparsity.

## Highlights & Insights
- **The "depth→width→depth" bridging approach is highly ingenious**: The authors translate the depth compression problem into a width compression problem and then back, enabling the two previously disconnected pruning approaches to "shake hands." This reframing is inspiring for algorithm design—when facing a hard problem, consider whether it can be equivalently transformed into another domain with mature tools.
- **Nyström compensation is a hidden trick for MLP pruning**: Directly selecting top-k channels loses information, but Nyström compensation uses a closed-form solution to fold the covariance of discarded channels back into the retained ones, achieving theoretical optimality; this trick can be independently applied to any MLP compression scenario.
- **Training-free + architecture invariance** is key for industrial deployment—the pruned model can directly use the original CUDA kernel, inference engine, and LoRA hyperparameters, with zero migration cost.

## Limitations & Future Work
- The equivalence of flattening relies on "high similarity of adjacent layer inputs," which holds for Pre-LN, residual-dominated deep networks; for shallow models (<20 layers) or Post-LN architectures with lower similarity, the method may fail.
- Greedy selection does not guarantee global optimality, and the paper does not compare with brute-force search or dynamic programming for upper-bound performance.
- The $\lambda$ (ridge intensity) in Nyström is set to 10× the average singular value empirically; different models may require re-tuning via grid search.
- Experiments on GQA/MoE architectures (except LLaMA-3) are insufficient, so compatibility with future mainstream sparse architectures is unknown.

## Related Work & Insights
- **vs SLEB/ShortGPT**: These methods directly remove entire blocks, while FlattenGPT turns "removal" into "merging + width compression." The final architecture is the same (thus equal inference speed), but FlattenGPT achieves 5 points higher accuracy, indicating that block-level deletion loses crucial information.
- **vs SliceGPT/LLM-Pruner**: These methods perform channel pruning but retain all layers, so throughput is inferior to depth pruning; FlattenGPT applies channel pruning to merged layers, enjoying the acceleration of depth compression.
- **vs LaCo (layer merging by addition)**: LaCo simply adds parameters of two layers, without considering LN fusion or parallel equivalence; FlattenGPT performs LN fusion + parallel architecture equivalence + Nyström compensation, and these three details result in a significant accuracy gap (FlattenGPT 62.49 vs LaCo 54.82).
- **Insights**: Viewing transformer "layers" as "width slices" may generalize to model expansion (splitting a single layer into multiple) or dynamic depth (skipping layers based on input).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "depth to width then prune width" reframing is highly novel, unifying two pruning approaches in a single framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 model families × multiple sizes × multiple pruning methods compared, plus throughput/latency and RFT experiments; however, LLaMA-3 GQA experiments are limited.
- Writing Quality: ⭐⭐⭐⭐ The pruning method comparison in Figure 1 is very clear; Lemma 2.1/2.2 provide theoretical support for redundancy. Algorithms 1-3 are presented as pseudocode, enhancing readability.
- Value: ⭐⭐⭐⭐⭐ Training-free + architectural homogeneity + 5% accuracy improvement + 1.27× acceleration make this highly attractive for industrial LLM deployment teams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization](../../NeurIPS2025/model_compression/replaceme_network_simplification_via_depth_pruning_and_transformer_block_lineari.md)
- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)
- [\[ICML 2025\] Strategic Fusion Optimizes Transformer Compression](../../ICML2025/model_compression/strategic_fusion_optimizes_transformer_compression.md)
- [\[ICML 2026\] Proxy Compression for Language Modeling](proxy_compression_for_language_modeling.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)

</div>

<!-- RELATED:END -->
