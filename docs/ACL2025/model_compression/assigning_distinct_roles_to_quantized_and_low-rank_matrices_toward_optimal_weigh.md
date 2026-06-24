---
title: >-
  [Paper Note] Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition
description: >-
  [ACL 2025][Model Compression][Weight Quantization] Proposes ODLRI (Outlier-Driven Low-Rank Initialization) to assign an explicit role to the low-rank component in the joint quantization and low-rank optimization (Q+LR) framework—capturing activation outlier-sensitive weights, allowing the quantized component to handle a smoother residual. This consistently reduces perplexity and improves zero-shot accuracy in 2-bit extreme quantization scenarios for Llama2/3 and Mistral.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Weight Quantization"
  - "Low-Rank Decomposition"
  - "Activation Outliers"
  - "KV Cache"
  - "2-bit Quantization"
date: 2026-05-08
content_hash: 2b8633064df665af
---

# Assigning Distinct Roles to Quantized and Low-Rank Matrices Toward Optimal Weight Decomposition

**Conference**: ACL 2025  
**arXiv**: [2506.02077](https://arxiv.org/abs/2506.02077)  
**Code**: None (based on the CALDERA framework)  
**Area**: Model Compression / LLM Quantization  
**Keywords**: Weight Quantization, Low-Rank Decomposition, Activation Outliers, KV Cache, 2-bit Quantization

## TL;DR

Proposes ODLRI (Outlier-Driven Low-Rank Initialization) to assign an explicit role to the low-rank component in the joint quantization and low-rank optimization (Q+LR) framework—capturing activation outlier-sensitive weights, allowing the quantized component to handle a smoother residual. This consistently reduces perplexity and improves zero-shot accuracy in 2-bit extreme quantization scenarios for Llama2/3 and Mistral.

## Background & Motivation

**Background**: Major methods for LLM weight compression include quantization and matrix decomposition. Recently, joint optimization methods occupy a prominent position, decomposing weights as $\mathbf{W} \approx \mathbf{Q} + \mathbf{LR}$ and achieving extreme compression by alternately optimizing the quantized matrix and the low-rank component.

**Limitations of Prior Work**: Existing joint optimization methods (such as CALDERA) adopt "quantization-then-low-rank" or "low-rank-then-quantization" strategies, which are essentially different initialization choices. However, the impact of low-rank component initialization on the final outcome has been overlooked—experiments show that initialization determines the persistent role allocation between the quantized and low-rank components.

**Key Challenge**: Zero initialization reduces LR to an "error correction term", while weight decomposition initialization forces LR to serve as the "primary weight representation"—neither role assignment is optimal.

**Goal**: Find the optimal initialization strategy for the low-rank component in joint Q+LR optimization, letting Q and LR play to their respective strengths.

**Key Insight**: Quantization is highly sensitive to activation outliers (which amplify weight sensitivity), whereas the low-rank component (the product of two low-bit factors) acts equivalent to a higher-bit representation. Thus, LR should be dedicated to capturing outlier-sensitive weights.

## Method

### Overall Architecture

Unified framework Algorithm 1: Initialize $\mathbf{L}_0, \mathbf{R}_0 \leftarrow \text{Initialize}$, then iterate for $T$ rounds: (1) $\mathbf{Q}_t \leftarrow \text{Quantize}(\mathbf{W} - \mathbf{L}_{t-1}\mathbf{R}_{t-1})$; (2) $\mathbf{L}_t, \mathbf{R}_t \leftarrow \text{LRApprox}(\mathbf{W} - \mathbf{Q}_t)$. ODLRI replaces the Initialize step.

### Key Designs

1. **Outlier-Driven Low-Rank Initialization (ODLRI)**:
    - **Function**: Utilizes the Hessian diagonal to identify top-k activation outlier channels, and constructs a restricted covariance matrix $\mathbf{H}_o$ using these channels to initialize the low-rank component.
    - **Mechanism**: $\mathbf{L}_0, \mathbf{R}_0 = \arg\min_{\mathbf{L},\mathbf{R}} \|(\mathbf{W} - \mathbf{LR})\mathbf{H}_o(\mathbf{W} - \mathbf{LR})^\top\|$, where $\mathbf{H}_o = \mathbf{X}_o\mathbf{X}_o^\top$ retains only the top-k outlier channels. Choosing $k < r$ instead of $k = r$ focuses on capturing the most critical outliers.
    - **Design Motivation**: Quantization is highly sensitive to outliers. Passing outlier-sensitive weights to the LR component with stronger representational capacity allows Q to handle more uniform residuals.

2. **Initialization Determines Persistent Roles**:
    - **Function**: Experiments reveal that different initialization strategies lead Q and LR to assume completely different roles.
    - **Mechanism**: Measuring $\|\mathbf{QX}\|/\|\mathbf{WX}\|$ and $\|\mathbf{LRX}\|/\|\mathbf{WX}\|$. Zero initialization $\to$ Q $\approx 0.96$, LR $\approx 0.07$ (Q-dominant); weight decomposition initialization $\to$ Q $\approx 0.40$, LR $\approx 0.66$ (LR-dominant). **Iterative optimization does not alter this role allocation**.
    - **Design Motivation**: This demonstrates that initialization is not just a "starting point," but fundamentally determines the structure of the decomposition.

3. **k-value Selection Strategy**:
    - **Function**: Chooses the number of top-k outlier channels, with $k < r$.
    - **Mechanism**: $k=r$ is equivalent to performing activation-aware low-rank approximation on all weights; $k<r$ more aggressively focuses on outliers. Experiments find the best performance when $k=16$ (much smaller than rank=256).
    - **Design Motivation**: Over-dispersed initialization reduces the concentrated capacity to handle outliers.

### Loss & Training

Post-training quantization (PTQ) method, requiring no training. Q uses the E8 lattice codebook of QuIP# for 2-bit quantization, and LR uses the LPLR iterative algorithm for 4-bit or 16-bit representation. CALDERA defaults to 15 outer iterations and 10 inner iterations.

## Key Experimental Results

### Main Results

Llama2 series (Q=2-bit, LR=4-bit):

| Model | Method | Rank | WikiText-2 PPL↓ | C4 PPL↓ | Zero-shot Avg↑ |
|------|------|------|-----------------|---------|---------------|
| 7B | CALDERA | 256 | 6.47 | 8.47 | 61.1 |
| 7B | +ODLRI | 256 | **6.33** | **8.27** | **62.6** |
| 13B | CALDERA | 256 | 5.56 | 7.39 | 63.8 |
| 13B | +ODLRI | 256 | **5.46** | **7.28** | **63.6** |
| 70B | CALDERA | 256 | 3.99 | 5.78 | 71.3 |
| 70B | +ODLRI | 256 | **3.94** | **5.73** | **71.9** |

Llama3-8B & Mistral-7B (Q=2-bit, LR=4-bit):

| Model | Method | Rank | Wiki2↓ | C4↓ |
|------|------|------|--------|-----|
| Llama3-8B | CALDERA | 256 | 8.70 | 9.77 |
| Llama3-8B | +ODLRI | 256 | **8.12** | **9.33** |
| Mistral-7B | CALDERA | 256 | 5.77 | 6.59 |
| Mistral-7B | +ODLRI | 256 | **5.69** | **6.53** |

### Ablation Study

Impact of k-value (Llama2-7B, rank=256):

| Initialization Strategy | LR 16-bit Wiki2↓ | LR 4-bit Wiki2↓ |
|-----------|------------------|-----------------|
| $\mathbf{H}_o$ (k=r=256) | 6.38 | 6.46 |
| $\mathbf{H}_o$ (k=16<r) | **6.18** | **6.33** |

### Key Findings

1. ODLRI **consistently** reduces perplexity across all models and rank settings, with the only change being the initialization strategy.
2. ODLRI significantly reduces the quantization scale (tighter weight distribution $\to$ more accurate low-bit representation).
3. ODLRI reduces activation-aware error, maintaining its advantage persistently throughout all optimization iterations.
4. $k<r$ performs better than $k=r$: concentrating on outliers is more effective than dispersing capacity.
5. The improvement with 16-bit LR is more pronounced than with 4-bit LR: when quantizing LR, some outlier information is lost due to "secondary" quantization.

## Highlights & Insights

- **Core Insight**: Initialization not only affects convergence speed but also fundamentally determines the "roles" of the respective components in weight decomposition.
- **Novelty**: Modifying only a single line of code (initialization method) achieves consistent improvements on top of the CALDERA framework.
- **Physical Intuition**: Outliers $\to$ high variance $\to$ large quantization error; allowing low-rank components to "absorb" these outliers is highly intuitive.
- **Unified Framework Perspective**: Understanding "quantize-first" or "decompose-first" as initialization choices opens up a new optimization space.

## Limitations & Future Work

- Only focuses on weight-only quantization, without addressing activation quantization or KV cache quantization.
- Only validated within the CALDERA framework; could be generalized to other Q+LR algorithms.
- A noticeable gap still exists between 2-bit quantization and FP16 (7B: 6.33 vs 5.12).
- The interaction effect between ODLRI and model scale remains unexplored (do larger models benefit more?).

## Related Work & Insights

- **CALDERA (NeurIPS 2024)**: The base framework of ODLRI, being the first to perform activation-aware joint Q+LR optimization.
- **SpQR**: Retains outlier weights at high precision; ODLRI shares a similar concept but uses the low-rank component instead of mixed precision.
- **AWQ**: Employs per-channel scaling to protect outlier weights; ODLRI addresses the same issue from the perspective of decomposition.
- Insight: In any scenario involving "decomposition into multiple representations," the role assignment of various components during initialization can have a significant impact.

## Rating

- Novelty: ⭐⭐⭐⭐ Although intuitive, the insight linking initialization to role assignment was previously unrecognized.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 model series, various rank and bit configurations, and multi-dimensional ablation analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear presentation of the unified framework with highly convincing tables and charts.
- Value: ⭐⭐⭐⭐ Simple and highly effective method with practical value for extreme compression scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[NeurIPS 2025\] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression](../../NeurIPS2025/model_compression/qsvd_efficient_low-rank_approximation_for_unified_query-key-value_weight_compres.md)
- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] Disentangling the Roles of Representation and Selection in Data Pruning](disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)

</div>

<!-- RELATED:END -->
