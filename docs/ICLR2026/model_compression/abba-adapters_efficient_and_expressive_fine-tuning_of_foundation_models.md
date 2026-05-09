---
title: >-
  [Paper Note] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models
description: >-
  [ICLR 2026][Model Compression][Parameter-efficient fine-tuning] This paper proposes ABBA adapters, which parameterize weight updates as the Hadamard product of two independently learnable low-rank matrices, $\Delta W = s(B_1A_1) \odot (B_2A_2)$. Under the same parameter budget, ABBA achieves an effective rank of $r_1 \cdot r_2$ compared to LoRA's $r$, representing a quadratic improvement. Through Khatri-Rao reconstruction, ABBA maintains memory efficiency comparable to LoRA, and significantly outperforms existing PEFT methods on arithmetic and commonsense reasoning tasks.
tags:
  - ICLR 2026
  - Model Compression
  - Parameter-efficient fine-tuning
  - LoRA
  - Hadamard product
  - low-rank adaptation
  - Khatri-Rao decomposition
date: 2026-05-08
content_hash: 6b249d1e2a616c2c
---

# ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models

**Conference**: ICLR 2026
**arXiv**: [2505.14238](https://arxiv.org/abs/2505.14238)
**Code**: [https://github.com/CERT-Lab/abba](https://github.com/CERT-Lab/abba)
**Area**: Model Compression / PEFT
**Keywords**: Parameter-efficient fine-tuning, LoRA, Hadamard product, low-rank adaptation, Khatri-Rao decomposition

## TL;DR
This paper proposes ABBA adapters, which parameterize weight updates as the Hadamard product of two independently learnable low-rank matrices, $\Delta W = s(B_1A_1) \odot (B_2A_2)$. Under the same parameter budget, ABBA achieves an effective rank of $r_1 \cdot r_2$ compared to LoRA's $r$, representing a quadratic improvement. Through Khatri-Rao reconstruction, ABBA maintains memory efficiency comparable to LoRA, and significantly outperforms existing PEFT methods on arithmetic and commonsense reasoning tasks.

## Background & Motivation

**Background**: LoRA is the most widely adopted PEFT method, constraining weight updates to a rank-$r$ subspace via $\Delta W = BA$ ($B \in \mathbb{R}^{m \times r}, A \in \mathbb{R}^{r \times n}$).

**Limitations of Prior Work**: LoRA's updates are strictly confined to a rank-$r$ subspace, inherently limiting expressiveness. HiRA introduces Hadamard products via $\Delta W = W_0 \odot (BA)$ to increase effective rank, but couples updates to the frozen weight $W_0$—when the target update divided element-wise by $W_0$ is not low-rank, HiRA offers no advantage.

**Key Challenge**: High expressiveness (high-rank updates) requires more parameters, yet the fundamental constraint of PEFT is a small parameter count. How can one break the rank barrier under the same parameter budget?

**Goal**: Substantially increase the expressiveness and effective rank of weight updates while maintaining LoRA-level parameter efficiency.

**Key Insight**: Set both factors in the Hadamard product as learnable low-rank matrices, fully decoupling updates from the pretrained weights. Employ Khatri-Rao decomposition to avoid instantiating full-size matrices.

**Core Idea**: The Hadamard product of two rank-$r/2$ matrices can achieve an effective rank of $r^2/4$, a quadratic improvement over LoRA's rank $r$ under the same parameter count.

## Method

### Overall Architecture
In each target layer, LoRA's $\Delta W = BA$ is replaced by $\Delta W = s(B_1A_1) \odot (B_2A_2)$. The four matrices $A_1, B_1, A_2, B_2$ form the "ABBA" structure. For fair comparison, $r_1 = r_2 = r/2$ is set so that the total parameter count matches LoRA at rank $r$.

### Key Designs

1. **Dual Low-Rank Parameterization via Hadamard Product**:

    - Function: Expresses the weight update as the Hadamard (element-wise) product of two independent low-rank matrices.
    - Mechanism: Since $\text{rank}(W_1 \odot W_2) \leq r_1 \cdot r_2$, ABBA's effective rank upper bound is $r_1 \cdot r_2 = r^2/4$, far exceeding LoRA's $r$. Matrix reconstruction experiments confirm that ABBA consistently achieves lower reconstruction error than LoRA under the same parameter budget.
    - Design Motivation: Unlike HiRA, both factors are fully learnable and not tied to $W_0$, freeing the update capacity from the structural constraints of the pretrained weights.

2. **Khatri-Rao Efficient Implementation (Theorem 1)**:

    - Function: Converts ABBA into a LoRA-compatible form via Khatri-Rao decomposition, avoiding the instantiation of full-size matrices.
    - Mechanism: Define $B_{\text{kr}} = B_1 \odot_r B_2 \in \mathbb{R}^{m \times r_1 r_2}$ and $A_{\text{kr}} = (A_1^\top \odot_r A_2^\top)^\top$; then $\Delta W x = B_{\text{kr}}(A_{\text{kr}} x)$, with intermediate activations of dimension only $r_1 r_2$.
    - Design Motivation: A naïve implementation would require constructing two $m \times n$ matrices and computing their Hadamard product, incurring memory costs equivalent to full fine-tuning. Khatri-Rao reconstruction keeps both computation and storage at the low-rank level.

3. **SVD Initialization + Rank Stability**:

    - Function: Initializes $(B_1, A_1)$ via truncated SVD of $W_0$; $(B_2, A_2)$ follows standard LoRA initialization.
    - Mechanism: The Eckart–Young–Mirsky (EYM) theorem guarantees that truncated SVD is the optimal rank-$r_1$ approximation. The scaling factor $s$ must be adjusted with respect to the effective rank $r_1 r_2$ (not $r$); rank stability is formally proved in the paper.
    - Design Motivation: The hybrid initialization anchors the update direction to a meaningful low-rank subspace while preserving the second matrix pair's capacity for task-specific exploration.

### Loss & Training
Standard fine-tuning loss is used. Training hyperparameters are identical to LoRA; only the adapter structure is replaced with ABBA. Code is publicly available.

## Key Experimental Results

### Main Results

**Arithmetic Reasoning (GSM8K, MATH, etc.):**

| Method | Parameters | GSM8K | MATH | Avg ↑ |
|--------|-----------|-------|------|-------|
| LoRA (r=16) | Baseline | Baseline | Baseline | Baseline |
| DoRA | Same | Marginal gain | Marginal gain | Marginal gain |
| HiRA | Same | Better than LoRA | Better than LoRA | Better than LoRA |
| **ABBA (r=8+8)** | **Same** | **Best by significant margin** | **Best by significant margin** | **Best by significant margin** |

**Commonsense Reasoning (Average across multiple datasets):**

| Method | LLaMA-7B | LLaMA-3-8B | Notes |
|--------|---------|-----------|-------|
| LoRA | Baseline | Baseline | |
| **ABBA** | **+2–3 pp** | **+2–3 pp** | Consistently best |

### Ablation Study

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| $r_1 = r_2 = r/2$ | Best | Equal split maximizes $r_1 r_2$ |
| $r_1 \neq r_2$ | Slightly worse | Asymmetric allocation is suboptimal |
| Random init for $(B_1, A_1)$ | Worse | SVD initialization is critical |
| No scaling factor | Training unstable | Rank stability requires appropriate scaling |

### Key Findings
- Matrix reconstruction experiments confirm that ABBA consistently outperforms same-parameter LoRA across various matrix types, validating its higher expressiveness.
- ABBA converges faster in practice than LoRA and HiRA (visualized via an MNIST toy experiment).
- Khatri-Rao reconstruction gives ABBA a memory footprint even smaller than HiRA, which must store the full $W_0$.
- Rank stability analysis shows that $s = 1/(r_1 r_2)$ is the appropriate scaling, consistent with the generalization of rsLoRA's $1/r$ scaling.

## Highlights & Insights
- **Quadratic rank gain at constant parameter count**: The effective rank of $r/2 \times r/2 = r^2/4$ is the central contribution—equivalent to obtaining $r/4\times$ greater expressiveness within the same budget.
- **Engineering elegance of Khatri-Rao**: While Hadamard products cannot be directly "distributed" into matrix-vector multiplication, the Khatri-Rao decomposition elegantly avoids full matrix instantiation—a key technical contribution that makes ABBA practically viable.
- **Fundamental distinction from HiRA**: HiRA fixes one factor as $W_0$ (free but non-learnable); ABBA makes both factors learnable but low-rank (incurring a parameter cost but offering greater flexibility). This raises an interesting trade-off between exploiting pretrained weight structure versus unconstrained learning.

## Limitations & Future Work
- The intermediate activation dimension in Khatri-Rao reconstruction is $r_1 r_2$ (vs. LoRA's $r$), incurring additional FLOPs.
- ABBA does not admit a closed-form optimal solution (the EYM theorem does not apply directly), so optimization relies on gradient descent.
- Initialization requires a truncated SVD of $W_0$ per layer, imposing a one-time upfront cost.
- Validation is limited to LLMs; applicability to vision and multimodal models remains unexplored.

## Related Work & Insights
- **vs. LoRA**: ABBA raises the effective rank from $r$ to $r^2/4$ under the same parameter count, a fundamental gain in expressiveness at the cost of slightly more complex initialization and implementation.
- **vs. HiRA**: HiRA couples updates to the pretrained weights by fixing one Hadamard factor as $W_0$; ABBA is fully learnable and thus more general.
- **vs. DoRA**: DoRA decouples direction and magnitude but the update remains low-rank; ABBA breaks the rank barrier via the Hadamard product.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of dual low-rank Hadamard parameterization and Khatri-Rao efficient implementation is elegant, and the quadratic rank improvement insight is profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four models, arithmetic and commonsense reasoning, matrix reconstruction experiments, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative flows smoothly from motivation to theory to experiments, with clear figures and tables.
- Value: ⭐⭐⭐⭐⭐ As a direct improvement over LoRA, ABBA is simple, practical, and yields significant gains, with open-source code.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[ICLR 2026\] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba](memba_membrane-driven_parameter-efficient_fine-tuning_for_mamba.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](../../NeurIPS2025/model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[ACL 2026\] A Layer-wise Analysis of Supervised Fine-Tuning](../../ACL2026/model_compression/a_layer-wise_analysis_of_supervised_fine-tuning.md)

<!-- RELATED:END -->
