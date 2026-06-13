---
title: >-
  [Paper Note] FlattenGPT: Depth Compression for Transformer with Layer Flattening
description: >-
  [ICML 2026][Model Compression][LLM Pruning] This paper proposes FlattenGPT, which first "flattens" and merges adjacent transformer layers with high input similarity into a single layer of $2\times$ width (preserving all…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "LLM Pruning"
  - "Depth Compression"
  - "Layer Merging"
  - "Channel Pruning"
  - "Nyström Approximation"
date: 2026-05-08
content_hash: c893c7801f07a07e
---

# FlattenGPT: Depth Compression for Transformer with Layer Flattening

**Conference**: ICML 2026  
**arXiv**: [2602.08858](https://arxiv.org/abs/2602.08858)  
**Code**: Not released  
**Area**: Model Compression / LLM Acceleration / Depth Pruning  
**Keywords**: LLM Pruning, Depth Compression, Layer Merging, Channel Pruning, Nyström Approximation

## TL;DR
This paper proposes FlattenGPT, which first "flattens" and merges adjacent transformer layers with high input similarity into a single layer of $2\times$ width (preserving all parametric knowledge), and then applies channel pruning to restore the width to its original scale. This approach achieves the inference acceleration of depth compression while avoiding the performance collapse caused by the direct knowledge loss in traditional layer pruning.

## Background & Motivation

**Background**: High LLM inference costs have inspired two pruning paradigms. Depth pruning (SLEB, ShortGPT, LaCo) directly removes entire transformer blocks, offering high speedup but significant performance degradation. Channel pruning (LLM-Pruner, SliceGPT) retains all layers but reduces width, which maintains better performance but yields marginal speedup and disrupts architectural homogeneity due to inconsistent pruning rates across layers.

**Limitations of Prior Work**: The fundamental issue with depth pruning is "coarse-grained deletion"—even if specific heads or channels within a block have learned critical knowledge, all are discarded if the block is deemed "redundant." The fundamental issue with channel pruning is "architectural inconsistency"—downstream applications (like LoRA), CUDA kernels, and inference engines require a uniform architecture for efficient execution. A clear gap exists between these two routes.

**Key Challenge**: Depth redundancy objectively exists (the authors use Lemma 2.1/2.2 to prove that the variance of deep hidden states grows as $\Theta(\ell^2)$ and gradients are dominated by residuals, degrading into identity mappings). However, no intermediate option exists between "deleting the whole block" and "retaining all blocks." Theoretically, one could "merge two layers," but the difficulty lies in merging them such that parameters are reduced without sacrificing performance.

**Goal**: (a) Identify a "merging adjacent layers" operation that allows knowledge from both layers to be preserved and collaborative; (b) compress the merged layer back to the original width to maintain architectural homogeneity.

**Key Insight**: In Pre-LN Transformers, the cosine similarity of hidden states $\mathbf{H}^\ell, \mathbf{H}^{\ell+1}$ between adjacent layers is typically $>0.9$. The authors realize that if inputs to two layers are nearly identical, reconfiguring their MHA and MLP into "parallel execution then summation" is mathematically close to the original "serial execution." However, the parameters physically become "double width" within the same layer—effectively transforming a depth problem into a width problem that can be handled via fine-grained channel pruning.

**Core Idea**: Sequential execution of Layer Flattening (Depth $\rightarrow$ Width) followed by Channel Pruning (Width Recovery) to achieve "Knowledge Preservation + Homogeneous Architecture + Inference Acceleration."

## Method

### Overall Architecture
A two-stage pipeline: ① **Iterative Layer Flattening**—Calculate the cross-layer cosine similarity matrix $\mathbf{S}\in\mathbb{R}^{L\times L}$ across all adjacent layers using a calibration set; greedily select and merge the most similar adjacent pair $(B_{\ell-1}, B_\ell)$ into $B_{\ell-1,\ell}$ iteratively until the target compression rate is reached. ② **Channel Pruning**—For each $2\times$ width merged layer, remove half the heads in the MHA based on head importance, and use Nyström approximation in the MLP to select top-k channels while compensating with residual information.

### Key Designs

1.  **Layer Flattening (Layer Flattening as Parallel-then-Sum)**:
    -   Function: Merges adjacent layers $B_{\ell-1}, B_\ell$ into a single layer $B_{\ell-1,\ell}$ while retaining all parametric knowledge.
    -   Mechanism: First, fuse the LayerNorm affine parameters $\boldsymbol{\alpha}^{\ell-1}, \boldsymbol{\alpha}^\ell$ into linear projections such as $\mathbf{W}_Q/W_K/W_V$ (this step does not change output). Then, horizontally concatenate $\mathbf{W}_Q^{\ell-1}, \mathbf{W}_Q^\ell$ to obtain $\mathbf{W}_Q^{\ell-1,\ell}\in\mathbb{R}^{d\times 2dh}$, with similar treatment for $W_K, W_V$; $\mathbf{W}_O$ is vertically concatenated. For the MLP, $\mathbf{W}_u, \mathbf{W}_g$ (horizontal) and $\mathbf{W}_D$ (vertical) are concatenated. Thus, the MHA of the merged layer becomes the sum of $2H$ parallel attention heads, and the MLP is a double-width MLP with $2d_{int}$ intermediate dimensions.
    -   Design Motivation: Given similar inputs (cos $>0.9$), approximating "Serial: $\mathbf{H}_\ell=\mathbf{H}_{\ell-1}+B_\ell(\mathbf{H}_{\ell-1}+B_{\ell-1}(\mathbf{H}_{\ell-1}))$" with "Parallel: $\mathbf{H}_\ell\approx \mathbf{H}_{\ell-1}+B_{\ell-1}(\mathbf{H}_{\ell-1})+B_\ell(\mathbf{H}_{\ell-1})$" results in minimal error. This "additive equivalence" is the geometric prerequisite for layer flattening.

2.  **Greedy Layer Selection based on Similarity Matrix**:
    -   Function: Decides which adjacent layers to merge and how many iterations to perform.
    -   Mechanism: Maintain an upper triangular similarity matrix $\mathbf{S}$ and merge the pair corresponding to the maximum value $\mathbf{S}_{\ell-1,\ell}$ in each round. Key technique: After merging, delete the $(\ell-1)$-th column and $\ell$-th row of $\mathbf{S}$, so the similarity of the new merged layer $B^{\ell-1,\ell}$ with other layers is indirectly expressed via $\mathbf{S}_{\ell-1,i}$ and $\mathbf{S}_{j,\ell}$. This "row-column deletion" mechanism ensures that even when merging more than 3 layers, the merge span is constrained by the "distance between the first and last layer."
    -   Design Motivation: Greedy selection avoids the NP-hard optimal grouping problem; row-column deletion prevents "flattening layers that are too far apart"—if multiple layers are merged consecutively, the semantic divergence between the first and last layers would break the information flow if merged.

3.  **MLP Nyström Channel Pruning + Error Compensation**:
    -   Function: Compresses the $2d_{int}$ width merged MLP back to the original $d_{int}$ while "projecting" information from discarded channels onto retained ones.
    -   Mechanism: Use ridge leverage score $s_i=[\mathbf{C}_\psi(\mathbf{C}_\psi+\lambda\mathbf{I})]_{ii}^{-1}$ to measure the importance of channel $i$ and select the top-k. Then, adjust the down matrix using the Nyström formula: $\mathbf{W}_D \leftarrow \mathbf{W}_D + (\mathbf{S}_k^\top\mathbf{C}_\psi\mathbf{S}_k+\lambda\mathbf{I})^{-1}\mathbf{S}_k^\top\mathbf{C}_\psi(\mathbf{I}-\mathbf{S}_k\mathbf{S}_k^\top)\mathbf{W}_D$. Lemma 3.1 proves this is the optimal compensation for least squares under L2 regularization. MHA channel pruning simply removes heads based on head importance $f_i=\mathbb{E}[\text{Softmax}(...)\mathbf{X}\mathbf{W}_{V,i}\text{diag}(\mathbf{W}_{O,i}\mathbf{W}_{O,i}^\top)^{1/2}]$.
    -   Design Motivation: Pure channel selection discards 50% of information; Nyström compensation "folds" the covariance of deleted channels into the down-projection of retained channels, theoretically ensuring the MLP output is optimal in the sense of minimized L2 error.

### Loss & Training
Completely training-free, requiring only 128 WikiText-2 sequences for calibration (to estimate $\mathbf{C}_\psi$). Optional RFT (Refined Fine-Tuning): 50K refined Alpaca + LoRA for 2 epochs + lr=1e-4 + lora_r=8 for recovery.

## Key Experimental Results

### Main Results
Evaluated on LLaMA-2/3, Qwen-1.5, and Baichuan-2 across various sizes, compared with 5 SOTA depth pruning methods.

| Model/Method | Sparsity | PPL ↓ | Avg Zero-shot Acc |
|--------------|----------|-------|-------------------|
| LLaMA-2 7B Dense | 0% | 5.47 | 69.00 |
| ShortGPT | 21% | 18.45 | 58.18 |
| BlockPruner | 22% | 11.51 | 60.17 |
| **Ours** | 21% | **8.68** | **62.49** |
| LLaMA-2 13B Dense | 0% | 4.88 | 71.76 |
| BlockPruner | 25% | 8.16 | 64.53 |
| **Ours** | 24% | **6.68** | **67.50** |
| Qwen-1.5 7B Dense | 0% | 7.95 | 65.48 |
| **Ours** | 21% | **16.05** | **57.00** |

In terms of throughput, LLaMA-2 70B with 20% sparsity via FlattenGPT achieves a $1.27\times$ throughput and $1.26\times$ latency speedup, matching SLEB (due to identical architecture) but with a 5-point higher accuracy.

### Ablation Study

| Configuration | LLaMA-2 7B Avg Acc |
|---------------|-------------------|
| Dense | 69.00 |
| Ours (w/o RFT) | 63.83 |
| Ours + RFT | **66.24** |
| LLM-Pruner + RFT | 62.15 |
| Shortened LLaMA + RFT | 61.91 |

### Key Findings
- At the same sparsity level, FlattenGPT outperforms ShortGPT by an average of 5 points and the strongest baseline, BlockPruner, by 2-3 points. This indicates that "merging then compressing" preserves information better than "direct deletion."
- Despite sharing the exact final architecture with SLEB (and thus the same throughput), the 5-point accuracy lead proves the performance gap stems entirely from the process (flattening + Nyström compensation) rather than inference optimization.
- LLaMA-2 7B retains 90-96% of zero-shot performance (at 20% compression + RFT), making it one of the strongest combinations at this sparsity level.

## Highlights & Insights
- **The bridging concept of "Depth $\rightarrow$ Width $\rightarrow$ Depth" is highly ingenious**: The authors translate the depth compression problem into a width compression problem and back, allowing two previously separate pruning paths to converge. This reframe is highly inspiring for algorithm design—when facing a hard problem, check if it can be "equivalently transformed" into a domain with mature tools.
- **Nyström compensation is a hidden trick for MLP pruning**: While direct top-k selection loses information, Nyström compensation uses a closed-form solution to fold the covariance of deleted channels back into the retained ones. This trick is theoretically optimal and can be independently applied to any MLP compression scenario.
- **Training-free + Consistent Architecture** are critical for industrial deployment. Pruned models can directly utilize original CUDA kernels, inference engines, and LoRA hyperparameters with zero migration cost.

## Limitations & Future Work
- The equivalence of flattening depends on the "high similarity of adjacent layer inputs," which holds for residual-dominated deep Pre-LN networks. For shallow models (<20 layers) or Post-LN architectures where similarity is lower, the method may fail.
- Greedy selection does not guarantee a global optimum, and the paper does not compare the performance upper bound against brute-force search or dynamic programming.
- The choice of $\lambda$ (ridge intensity) in Nyström as $10\times$ the average singular value is empirical; different models might require grid searching.
- Lack of extensive experiments on GQA/MoE architectures (except LLaMA-3) makes compatibility with future mainstream sparse architectures unknown.

## Related Work & Insights
- **vs SLEB/ShortGPT**: These methods directly delete entire blocks. FlattenGPT converts "deletion" into "merging + compression," resulting in the same final architecture (equal inference speed) but with a 5-point accuracy lead, suggesting block-level deletion loses critical information.
- **vs SliceGPT/LLM-Pruner**: These perform channel pruning but retain all layers, resulting in lower throughput than depth pruning. FlattenGPT uses channel pruning techniques on merged layers to enjoy the speed of depth compression.
- **vs LaCo (layer merging by addition)**: LaCo simply adds parameters from two layers without considering LN fusion or parallel equivalence. FlattenGPT performs LN fusion + parallel architectural equivalence + Nyström compensation, leading to a massive accuracy gap (FlattenGPT 62.49 vs LaCo 54.82).
- **Insight**: Viewing a Transformer "layer" as a "width slice" might be generalized to model expansion (splitting one layer into many) or dynamic depth (skipping layers based on input).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The reframe of "converting depth to width then pruning width" is highly novel, unifying two pruning routes.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comparison across 4 model families × various sizes × multiple pruning methods + throughput/latency + RFT experiments; however, experiments on LLaMA-3 GQA series are relatively sparse.
- Writing Quality: ⭐⭐⭐⭐ Figure 1's comparison is very clear; Lemma 2.1/2.2 provides a solid theoretical foundation for the existence of redundancy. Algorithms 1-3 are well-presented in pseudocode.
- Value: ⭐⭐⭐⭐⭐ Training-free + homogeneous architecture + 5% accuracy gain + $1.27\times$ speedup makes this highly attractive to industrial teams deploying LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](xkv_cross-layer_kv-cache_compression_via_aligned_singular_vector_extraction.md)
- [\[NeurIPS 2025\] ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization](../../NeurIPS2025/model_compression/replaceme_network_simplification_via_depth_pruning_and_transformer_block_lineari.md)
- [\[ACL 2026\] LEAP: Layer-wise Exit-Aware Pretraining for Efficient Transformer Inference](../../ACL2026/model_compression/leap_layer-wise_exit-aware_pretraining_for_efficient_transformer_inference.md)
- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)
- [\[ICML 2026\] ReSpinQuant: Efficient Layer-Wise LLM Quantization via Subspace Residual Rotation Approximation](respinquant_efficient_layer-wise_llm_quantization_via_subspace_residual_rotation.md)

</div>

<!-- RELATED:END -->
