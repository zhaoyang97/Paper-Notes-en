---
title: >-
  [Paper Note] SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping
description: >-
  [AAAI 2026][Model Compression][Low-rank compression] SkipCat proposes a rank-maximized low-rank compression framework that introduces two techniques—intra-layer shared projection (Cat) and block skipping (Skip)—to retain…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Low-rank compression"
  - "SVD"
  - "rank maximization"
  - "block skipping"
  - "shared projection"
date: 2026-05-08
content_hash: eaa68f979625f800
---

# SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping

**Conference**: AAAI 2026
**arXiv**: [2512.13494](https://arxiv.org/abs/2512.13494)  
**Code**: None  
**Area**: Model Compression
**Keywords**: Low-rank compression, SVD, rank maximization, block skipping, shared projection

## TL;DR
SkipCat proposes a rank-maximized low-rank compression framework that introduces two techniques—intra-layer shared projection (Cat) and block skipping (Skip)—to retain more effective rank under the same compression ratio. Without any fine-tuning, it achieves up to 7% accuracy improvement on zero-shot tasks over existing low-rank methods.

## Background & Motivation

### State of the Field
Large language models demonstrate strong performance across a wide range of tasks, yet their massive parameter counts pose significant computational and memory challenges for deployment on edge devices. Low-rank compression, which decomposes weight matrices into two low-rank factors to reduce parameters and computation, is a promising direction for model compression.

### Limitations of Prior Work
Naive low-rank compression (e.g., SVD decomposition) has a fundamental limitation: **the retained rank must be reduced to below half of the original rank before any actual computational or memory gains are realized**.

Specifically, for a weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$ decomposed into $B \in \mathbb{R}^{d_{out} \times r}$ and $A \in \mathbb{R}^{r \times d_{in}}$, compression savings only emerge when:
$$r < \frac{d_{in} \cdot d_{out}}{d_{in} + d_{out}}$$
For square matrices ($d_{in} = d_{out}$), this implies $r < R/2$.

### Root Cause
Lower rank yields greater compression but larger performance degradation; higher rank preserves performance but provides no practical compression benefit. The core tension is: how can more effective rank be retained under the same compression ratio?

### Starting Point
The paper exploits structural properties of the model architecture—Q/K/V projections in attention share the same input, and Gate/Up projections in MLP share the same input—to fundamentally alter the relationship between rank and compression ratio via shared projection matrices and block skipping.

## Method

### Overall Architecture
SkipCat consists of two core techniques:
1. **Cat (Intra-layer Shared Low-Rank Projection)**: Matrices that share the same input use a common projection matrix.
2. **Skip (Block Skipping)**: Certain sub-blocks in the low-rank projection are bypassed during computation.

Used jointly, the two techniques significantly increase the number of effective ranks under the same compression budget.

### Key Designs

#### 1. **Cat: Intra-layer Shared Low-Rank Projection (Matrix Concatenation)**
- **Core Idea**: In the attention module, $W_Q, W_K, W_V$ share the same input $x$; similarly, $W_{Gate}$ and $W_{Up}$ in the MLP share the same input. Matrices sharing the same input are concatenated along the output dimension and jointly decomposed via SVD.
- **Formulation**:
  $$W_{QKV} = [W_Q^T, W_K^T, W_V^T]^T \in \mathbb{R}^{3d_{out} \times d_{in}}$$
  decomposed into $B_{QKV} \in \mathbb{R}^{3d_{out} \times r}$ and a shared projection $A_{QKV} = W_{S1} \in \mathbb{R}^{r \times d_{in}}$.
- **Efficiency Analysis**: After amortization, the parameter count per matrix becomes $r(d_{in} + Cd_{out})/C$, where $C$ is the number of concatenated matrices. For the attention module with $C=3$, the cost of the projection matrix is amortized across three matrices.
- **Design Motivation**: The projection matrix $A$ maps input to a lower-dimensional space. Matrices sharing the same input can reuse this mapping, eliminating redundant parameters.

#### 2. **Skip: Block Skipping (via Schur Complement)**
- **Core Idea**: The projection matrix $A$ is partitioned into $[A_1, A_2]$; $A_1$ is absorbed into the reconstruction matrix $B$, bypassing the explicit computation of $A_1$.
- **Key Derivation**:
  $$Wx \approx BAx = B(A_1 x_1 + A_2 x_2) = BA_1(x_1 + A_1^{-1}A_2 x_2) = B'(x_1 + A'x_2)$$
  where $B' = BA_1$ and $A' = A_1^{-1}A_2$.
- **Parameter Count**: Reduced from $r(d_{in} + d_{out})$ to $r(d_{in} + d_{out} - r)$.
- **Numerical Stability Issue**: When $A_1$ is ill-conditioned, $A_1^{-1}$ produces large values that cause FP16 overflow.
- **Solution—Column Pivoting**: Strong Rank-Revealing QR decomposition is used to identify a well-conditioned column subset for $\tilde{A}_1$. After applying column permutation $P$ and re-decomposing, activation magnitudes are reduced by nearly two orders of magnitude, yielding a more uniform distribution.

#### 3. **Joint Use of SkipCat**
- All shared projections are equipped with block skipping; the two techniques are complementary.
- Cat provides limited benefit at low compression ratios, whereas Skip remains effective; at high compression ratios, Cat compensates for Skip's limitations.
- Together, they consistently keep the model within the effective compression region across the full compression range.

### Loss & Training
- **Training-free**: Strong performance is achieved without any fine-tuning.
- **Whitening Preprocessing**: Weight whitening is performed using a mixed calibration set from WikiText-2 and C4 (512 samples).
- **Optional Fine-tuning**: Compatible with LoRA fine-tuning; after fine-tuning at 20% compression, accuracy loss is only 0.39%.
- **Quantization Compatibility**: Hadamard transforms and channel scaling are used to stabilize 8-bit quantization.

## Key Experimental Results

### Main Results

| Model | Compression | Method | WikiText2 PPL | C4 PPL | Zero-shot Avg. Acc. | Acc. Drop |
|-------|-------------|--------|---------------|--------|---------------------|-----------|
| LLaMA2-7B | 0% | Dense | 5.47 | 7.26 | 54.79% | — |
| LLaMA2-7B | 20% | ASVD | 9.06 | 11.66 | 48.81% | 5.98% |
| LLaMA2-7B | 20% | SVD-LLM | 8.82 | 13.42 | 44.84% | 9.95% |
| LLaMA2-7B | 20% | **SkipCat** | **6.29** | **8.95** | **52.59%** | **2.20%** |
| LLaMA2-7B | 30% | SVD-LLM | 11.75 | 19.37 | 41.25% | 13.54% |
| LLaMA2-7B | 30% | **SkipCat** | **7.65** | **11.57** | **48.46%** | **6.34%** |
| Qwen3-8B | 20% | SVD-LLM | 14.33 | 23.21 | 51.66% | 8.62% |
| Qwen3-8B | 20% | **SkipCat** | **11.68** | **19.09** | **56.42%** | **3.86%** |

### Ablation Study

| Cat | Skip | Quant | WikiText2 PPL | C4 PPL | Note |
|-----|------|-------|---------------|--------|------|
| ✗ | ✗ | ✗ | 8.82 | 13.42 | Naive SVD baseline |
| ✓ | ✗ | ✗ | 7.84 | 11.99 | Cat alone |
| ✗ | ✓ | ✗ | 6.71 | 9.32 | Skip contributes more |
| ✓ | ✓ | ✗ | **6.29** | **8.95** | Best combined |
| ✓ | ✓ | ✓ | 6.29 | 8.96 | 8-bit quantization lossless |

### Fine-tuned Results (LLaMA2-7B + LoRA)

| Compression | SVD-LLM | SkipCat | Gap |
|-------------|---------|---------|-----|
| 20% | 51.02% (−3.78) | **54.41%** (−0.39) | SkipCat loses only 0.39% |
| 40% | 46.91% (−7.88) | **48.65%** (−6.15) | |
| 60% | 39.50% (−15.29) | **41.16%** (−13.64) | |

### Key Findings
1. At 30% compression, SkipCat's PPL (7.65) is even lower than the results of competing methods at 20% compression.
2. Zero-shot accuracy improves by approximately **7%** (at 30% compression vs. SVD-LLM).
3. Cat and Skip are complementary—Cat reduces projection redundancy while Skip reduces sub-block computation; their combination yields the best results.
4. Column pivoting is critical for Skip to operate correctly in FP16; without it, BF16 PPL can reach 31,628.
5. Consistent effectiveness is observed on larger models (13B/14B) and different architectures (Qwen3).

## Highlights & Insights
1. **Precise Problem Formulation**: The paper identifies the fundamental bottleneck that rank must fall below half to yield compression benefits and addresses it systematically.
2. **Mathematical Elegance**: The Skip technique is grounded in the Schur complement, with column pivoting ensuring numerical stability.
3. **Strong Training-free Performance**: Only 2.2% accuracy drop at 20% compression in a training-free setting is highly competitive.
4. **Orthogonal Compatibility with Quantization**: Parameter-level compression and precision-level quantization can be stacked without interference.

## Limitations & Future Work
1. Skip requires $A_1$ to be invertible; although column pivoting mitigates this issue, extreme cases may still be problematic.
2. Cat requires matrices to share the same input; applicability to different architectures requires case-by-case analysis.
3. Performance degradation remains significant at high compression ratios (>60%).
4. Evaluation is limited to language models; effectiveness on vision and multimodal models has not been verified.
5. Column permutation preprocessing is required at deployment, slightly increasing deployment complexity.

## Related Work & Insights
- ASVD mitigates the influence of outliers via activation distribution scaling, but the rank limitation of SVD itself remains.
- Basis Sharing reduces storage by sharing basis matrices across layers, but provides no benefit to inference computation or memory transfer.
- The intra-layer sharing proposed in this paper is complementary to cross-layer sharing and can be combined.
- The rank-maximization perspective can be generalized to other low-rank methods (e.g., rank selection in LoRA).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The rank-maximization perspective is novel; the Cat+Skip design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers multiple models, multiple compression ratios, and includes ablations, fine-tuning, and quantization experiments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation is clear; Figure 1 effectively conveys the core idea.
- **Value**: ⭐⭐⭐⭐⭐ — Highly practical; a 7% accuracy improvement represents a significant advance in low-rank compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TLoRA: Task-aware Low Rank Adaptation of Large Language Models](../../ACL2026/model_compression/tlora_task-aware_low_rank_adaptation_of_large_language_models.md)
- [\[ACL 2026\] TalkLoRA: Communication-Aware Mixture of Low-Rank Adaptation for Large Language Models](../../ACL2026/model_compression/talklora_communication-aware_mixture_of_low-rank_adaptation_for_large_language_m.md)
- [\[AAAI 2026\] Group Orthogonal Low-Rank Adaptation for RGB-T Tracking](group_orthogonal_low-rank_adaptation_for_rgb-t_tracking.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[NeurIPS 2025\] C-LoRA: Contextual Low-Rank Adaptation for Uncertainty Estimation in Large Language Models](../../NeurIPS2025/model_compression/c-lora_contextual_low-rank_adaptation_for_uncertainty_estimation_in_large_langua.md)

</div>

<!-- RELATED:END -->
