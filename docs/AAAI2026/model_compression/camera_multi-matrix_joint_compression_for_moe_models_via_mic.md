---
title: >-
  [Paper Note] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis
description: >-
  [AAAI 2026][Model Compression][MoE compression] This paper introduces the concept of "micro-expert" to decompose MoE layer outputs as cross-matrix (up/gate/down_proj) linear combinations, enabling structured pruning (Camera-P) and mixed-precision quantization (Camera-Q) based on energy ranking. On Deepseek-MoE-16B, Qwen2-57B, and Qwen3-30B at 20%–60% sparsity, the method comprehensively outperforms NAEE and D²-MoE; analysis of Qwen2-57B requires less than 5 minutes on a single A100 GPU.
tags:
  - AAAI 2026
  - Model Compression
  - MoE compression
  - micro-expert
  - structured pruning
  - mixed-precision quantization
  - training-free
date: 2026-05-08
content_hash: fa1982542236cbac
---

# CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis

**Conference**: AAAI 2026
**arXiv**: [2508.02322](https://arxiv.org/abs/2508.02322)
**Code**: [https://github.com/xuyuzhuang11/CAMERA](https://github.com/xuyuzhuang11/CAMERA)
**Area**: Model Compression
**Keywords**: MoE compression, micro-expert, structured pruning, mixed-precision quantization, training-free

## TL;DR
This paper introduces the concept of "micro-expert" to decompose MoE layer outputs as cross-matrix (up/gate/down_proj) linear combinations, enabling structured pruning (Camera-P) and mixed-precision quantization (Camera-Q) based on energy ranking. On Deepseek-MoE-16B, Qwen2-57B, and Qwen3-30B at 20%–60% sparsity, the method comprehensively outperforms NAEE and D²-MoE; analysis of Qwen2-57B requires less than 5 minutes on a single A100 GPU.

## Background & Motivation
MoE architectures achieve efficient scaling through sparse activation, yet the increase in parameter count does not yield proportional performance gains, indicating significant structural redundancy. Existing MoE compression methods suffer from two key limitations: (1) whole-expert pruning/merging operates at too coarse a granularity, incurring large information loss or relying on strong assumptions (e.g., functional similarity between experts); (2) partial expert pruning operates independently at the matrix level, neglecting the functional dependencies among the up_proj, gate_proj, and down_proj matrices.

## Core Problem
How can a finer-grained, cross-matrix coordinated compression unit be identified to efficiently compress MoE models while preserving functional integrity? The core challenge is that precisely evaluating the importance of each compression unit is NP-hard (Column Subset Selection Problem), and modern MoE models contain on the order of $10^5$ micro-experts.

## Method

### Overall Architecture
The MoE layer is decomposed as a linear combination of micro-experts: MoE output $\mathbf{y} = \sum_{i}^{N_e} \phi_i \mathbf{w}_i^{down}$, where $\phi_i$ is a scalar combination coefficient and $\mathbf{w}_i^{down}$ is a basis vector. Based on this decomposition, the Camera algorithm efficiently estimates the "decoding-time energy" of each micro-expert for ranking; Camera-P is used for pruning and Camera-Q for mixed-precision quantization.

### Key Designs
1. **Micro-Expert Definition**: The $i$-th micro-expert of each expert is jointly defined by three vectors: $\mathbf{w}_i^{up}$ (the $i$-th row of up_proj), $\mathbf{w}_i^{gate}$ (the $i$-th row of gate_proj), and $\mathbf{w}_i^{down}$ (the $i$-th column of down_proj). The MoE output is a linear combination of all micro-expert outputs, with scalar combination coefficient $\phi_i = A_i \cdot \sigma(\mathbf{w}_i^{gate}\mathbf{x}) \cdot \mathbf{w}_i^{up}\mathbf{x}$. This decomposition reveals the intrinsic structure of MoE layers.

2. **Camera Energy Ranking Algorithm**: Micro-expert energy is defined as $\mathcal{E}_i = [(1-\alpha)\|\mathbf{\Phi}_{:,i}\|_2^2 + \alpha\|\mathbf{\Phi}_{:,i}\|_\infty^2] \cdot \|\mathbf{w}_i\|_2^2$, simultaneously accounting for the $L_2$ norm (overall contribution) and $L_\infty$ norm (peak contribution) of activation coefficients as well as the norm of the basis vector. Theoretical guarantee: the pruning error based on energy ranking differs from the optimal SVD approximation by at most an $O(k)$-delta gap.

3. **Camera-P Structured Pruning**: After energy ranking, the three associated vectors of low-energy micro-experts are simultaneously zeroed out, preserving cross-matrix functional integrity. Processing is performed layer by layer: calibration samples are collected, micro-experts are ranked, pruning is applied, and the updated outputs are propagated to the next layer.

4. **Camera-Q Mixed-Precision Quantization**: Micro-experts are divided into three groups by energy and assigned different bit-widths (e.g., 3/2/1 bit). Crucially, all three parameters of the same micro-expert are assigned the same precision, in contrast to conventional methods that partition along the input dimension of a single matrix.

### Loss & Training
The method is entirely training-free and gradient-free, requiring only 128 calibration sequences of length 2048 (from Wikitext2), processed layer by layer. Camera-P analyzes Qwen2-57B in fewer than 5 minutes on a single A100 GPU.

## Key Experimental Results

| Model | Sparsity | Camera-P Avg | NAEE Avg | D²-MoE Avg |
|--------|------|------|----------|------|
| Deepseek-MoE-16B | 20% | **61.03** | 60.51 | 58.97 |
| Deepseek-MoE-16B | 40% | **58.58** | 54.94 | 54.32 |
| Deepseek-MoE-16B | 60% | **51.62** | 45.28 | 46.72 |
| Qwen2-57B-A14B | 20% | **67.28** | 66.11 | 66.38 |
| Qwen2-57B-A14B | 40% | **66.81** | 63.92 | 64.40 |
| Qwen2-57B-A14B | 60% | **65.17** | 51.40 | 56.32 |
| Qwen3-30B-A3B | 20% | **69.94** | 69.64 | 66.35 |

Camera-Q (2.25-bit average) achieves an average score of 56.56 on Deepseek-MoE-16B, versus 53.45 for GPTQ and 54.45 for MC.

### Ablation Study
- Micro-expert energy distributions are highly non-uniform, validating the effectiveness of energy-based pruning.
- Matching strategy comparison: Camera-Q (cross-matrix consistent precision) 56.56 vs. Camera-Q† (single-matrix partition) 52.69, demonstrating that cross-matrix functional integrity is critical.
- The $\alpha$ parameter (L2 vs. L∞ weighting) has minimal impact on perplexity and average accuracy but affects specific tasks.
- Results are insensitive to calibration data size and source (128–512 samples; Wiki2 vs. C4), indicating strong robustness.
- Camera-P directly reduces parameter count, achieving 1.03–1.06× decoding speedup at 20% sparsity and 1.04–1.42× at 40%.

## Highlights & Insights
- **The "micro-expert" concept is fundamentally novel**—decomposing MoE layers as linear combinations of basis vectors provides a new lens for understanding the internal workings of MoE models.
- **Cross-matrix joint compression preserves functional integrity**—contrasting sharply with conventional per-matrix compression; the Camera-Q vs. Camera-Q† comparison provides clear empirical evidence.
- **Remarkable efficiency**: analyzing a 57B model on a single A100 in under 5 minutes, more than 100× faster than existing methods, making it genuinely practical for deployment.
- Theoretical guarantees: the gap between pruning error and optimal SVD is bounded ($O(k)$-delta).
- Extensible to FFN pruning in dense models and complementary to single-matrix methods such as Wanda.

## Limitations & Future Work
- Advantages are less pronounced on older MoE models with fewer experts (Mixtral-8x7B, Phi3.5-MoE).
- Energy ranking is static and does not account for the possibility that different input samples may require different micro-expert combinations.
- Camera-P does not perform fine-tuning recovery after pruning, leaving room for improvement at high sparsity ratios.
- Integration with parameter-efficient fine-tuning methods such as LoRA has not been explored.
- The mixed-precision quantization component relies partially on GPTQ; combining it with more advanced quantization methods (e.g., SpinQuant) warrants future investigation.

## Related Work & Insights
- **vs. NAEE**: NAEE performs whole-expert search via brute-force enumeration, which does not scale to models with many experts; Camera operates at the micro-expert level with finer granularity and over 100× speedup.
- **vs. D²-MoE**: D²-MoE merges experts prior to low-rank decomposition, assuming inter-expert mergeability and introducing numerical instability; Camera directly identifies important micro-experts and retains them at full precision.
- **vs. MC quantization**: MC assigns bit-widths at the expert level, which is too coarse; Camera-Q assigns precision at the micro-expert level for finer-grained control.

**Broader implications:**
- The micro-expert perspective can be extended to analyze visual expert redundancy in VLMs, complementing token compression approaches such as EM-KD.
- The energy-ranking strategy can be applied to dynamic inference: dynamically selecting a high-energy micro-expert subset conditioned on the input.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The micro-expert concept is a genuinely original contribution that provides a new perspective for understanding MoE models.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3+ MoE models, 20–60% sparsity rates, 9 zero-shot tasks, quantization experiments, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous, motivation is clearly articulated, and experimental analysis is thorough.
- Value: ⭐⭐⭐⭐⭐ Offers substantial practical value for MoE compression; the method is concise, efficient, and deployment-ready.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](../../ICLR2026/model_compression/sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ICLR 2026\] Steering MoE LLMs via Expert (De)Activation](../../ICLR2026/model_compression/steering_moe_llms_via_expert_deactivation.md)
- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)
- [\[AAAI 2026\] Share Your Attention: Transformer Weight Sharing via Matrix-Based Dictionary Learning](share_your_attention_transformer_weight_sharing_via_matrix-based_dictionary_lear.md)
- [\[ICLR 2026\] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](../../ICLR2026/model_compression/kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)

<!-- RELATED:END -->
