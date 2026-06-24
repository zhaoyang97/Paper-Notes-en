---
title: >-
  [Paper Note] Efficient Knowledge Editing via Minimal Precomputation
description: >-
  [ACL 2025][Knowledge Editing][MEMIT] Demonstrates that the precomputation step (caching 44 million hidden vectors) for knowledge editing methods like MEMIT/ROME/EMMET can be reduced to 2-10 times the theoretical minimum (less than 0.3% of the original size), reducing precomputation time from dozens of hours to minutes with virtually no loss in editing performance.
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "MEMIT"
  - "precomputation"
  - "covariance matrix"
  - "locate-then-edit"
date: 2026-05-08
content_hash: ddc89e3619896334
---

# Efficient Knowledge Editing via Minimal Precomputation

**Conference**: ACL 2025  
**arXiv**: [2506.04226](https://arxiv.org/abs/2506.04226)  
**Code**: [https://github.com/scalable-model-editing/efficient-model-editing](https://github.com/scalable-model-editing/efficient-model-editing)  
**Area**: Knowledge Editing  
**Keywords**: knowledge editing, MEMIT, precomputation, covariance matrix, locate-then-edit

## TL;DR
Demonstrates that the precomputation step (caching 44 million hidden vectors) for knowledge editing methods like MEMIT/ROME/EMMET can be reduced to 2-10 times the theoretical minimum (less than 0.3% of the original size), reducing precomputation time from dozens of hours to minutes with virtually no loss in editing performance.

## Background & Motivation
**Background**: Locate-then-edit methods (MEMIT, ROME, EMMET) can efficiently edit factual knowledge in LLMs without additional training by modifying MLP weight matrices.

**Limitations of Prior Work**:
   - These methods have an overlooked "precomputation step": they require passing approximately 44 million Wikipedia tokens through the model to cache hidden vectors of each layer to construct the covariance matrix $C_0 = K_0 K_0^T$.
   - Precomputation takes 36 hours for GPT-J (6B) and 40 hours for Llama2-7B on a single A6000 GPU.
   - Precomputation time scales linearly with model size, requiring recomputation for every new model.

**Key Challenge**: The editing process itself takes only a few seconds, but precomputation requires dozens of hours, severely limiting the rapid deployment of these methods on new models.

**Key Insight**: From a linear algebra perspective, the invertibility of the $C_{eff}$ matrix only requires $d_k$ linearly independent vectors ($d_k$ being the dimension of the key vector), which is far smaller than 44 million.

**Core Idea**: The number of precomputed vectors only needs to be a few times the key vector dimension $d_k$ to ensure editing performance.

## Method

### Overall Architecture
The closed-form solution of methods such as MEMIT is $\hat{W} = W_0 + (V_E - W_0 K_E) K_E^T (\lambda C_0 + K_E K_E^T)^{-1}$, where $C_0 = K_0 K_0^T$ is the precomputed covariance matrix. The core contribution of this work is to analyze the minimum conditions for constructing $C_0$, demonstrating that an effective $C_0$ can be constructed with a minimal number of tokens.

### Key Designs

1. **Theoretical Minimum Derivation**:

    - Function: Deriving the minimum condition for the invertibility of $C_{eff} = \lambda C_0 + K_E K_E^T$.
    - Mechanism: $C_{eff}$ is the sum of $P+B$ rank-1 matrices ($P$ preservation vectors + $B$ batch editing vectors), with a dimension of $d_k \times d_k$. Invertibility is guaranteed as long as there are $d_k$ linearly independent vectors. Consequently, the theoretical minimum number of precomputed tokens is $d_k - B \approx d_k$.
    - Design Motivation: $d_k = 6400$ for GPT2-XL and $d_k = 16384$ for GPT-J, which are significantly smaller than the original 44 million.

2. **Dynamic Multiplier**:

    - Function: Introducing a hyperparameter $d_m$ (dynamic multiplier), where the actual precomputation uses $P' = d_m \times d_k$ tokens.
    - Mechanism: The theoretical minimum does not guarantee numerical stability (vectors might be approximately linearly dependent), requiring a certain level of redundancy. $d_m = 2$ is sufficient for GPT-J, and $d_m = 10$ is reliable for Llama2-7B.
    - Design Motivation: To find an optimal trade-off between the theoretical minimum and full precomputation.

3. **Regularization Correction (Llama2-7B)**:

    - Function: Adding a regularization term for Llama2-7B during small batch size editing.
    - Mechanism: The hidden vectors of Llama2-7B are highly correlated, potentially causing matrix non-invertibility under low $d_m$. Thus, $\epsilon I$ regularization is added.
    - Design Motivation: To address the numerical instability issues in specific model architectures.

### Precomputation Comparison

| Model | Original Precomputation | $d_m=2$ | $d_m=10$ | Saving Ratio |
|------|-----------|---------|----------|---------|
| GPT2-XL ($d_k$=6400) | 44M tokens | 12.8K | 64K | >99.8% |
| GPT-J ($d_k$=16384) | 44M tokens | 32.8K | 163.8K | >99.6% |
| Llama2-7B ($d_k$=16384) | 44M tokens | - | 163.8K | >99.6% |

## Key Experimental Results

### Main Results
Evaluated on the CounterFact dataset with batch sizes ranging from 1 to 1024:

| Model | Method | $d_m$ | Overall Score | vs Full Precomputation | Precomputation Time |
|------|------|-------|---------------|--------------|-----------|
| GPT-J | FastEMMET | 2 | ≈ Full Precomputation | ≥95% | few seconds |
| GPT-J | FastMEMIT | 2 | ≈ Full Precomputation | ≥95% | few seconds |
| Llama2-7B | FastEMMET | 2 | ≈ Full Precomputation | ≥95% | few seconds |
| Llama2-7B | FastMEMIT | 10 | ≈ Full Precomputation | ≥95% | few minutes |

Original precomputation time: GPT-J ~36h, Llama2-7B ~40h $\to$ FastMEMIT reduces this to the minutes scale.

### Ablation Study

| Configuration ($d_m$) | GPT-J Overall | Llama2-7B Overall | Description |
|-------------|---------------|-------------------|------|
| 1 (Theoretical Min) | Significant performance drop | Matrix not invertible | Insufficient stability |
| 2 | ≥95% | EMMET ≥95%, MEMIT partially unstable | Optimal for GPT series |
| 5 | ≥95% | ≥95% | Reliable |
| 10 | ≈100% | ≈100% | Recommended setting |
| ∞ (Original) | 100% | 100% | 44M tokens |

### Key Findings
- $d_m = 2$ is sufficient for the GPT series models to achieve over 95% editing performance (using only 0.08% of the original precomputation data).
- The hidden vectors of Llama2-7B are more highly correlated, requiring $d_m = 10$ (using 0.25% of the original precomputation data).
- Editing with a small batch size is more sensitive to precomputation, while editing with a larger batch size is more stable.
- It is recommended to use $d_m = 10$ universally to cover all models and batch sizes.

## Highlights & Insights
- **Extremely Elegant Insight**: Beginning with linear algebra invertibility conditions, a simple observation (requiring only $d_k$ independent vectors) leads to a 99.7%+ saving in precomputation. This suggests that the default hyperparameters in many methods (e.g., 44M tokens) lack theoretical grounding.
- **High Practical Value**: Editing can begin minutes after a new model is released, avoiding the need to wait dozens of hours for precomputation. This is highly valuable for rapidly validating knowledge editing methods on new models.
- **Transferable Design of Dynamic Multiplier**: In other scenarios requiring the precomputation of covariance matrices (e.g., Fisher Information matrix approximation, feature covariance estimation), the sample size can be similarly reduced by a significant margin.

## Limitations & Future Work
- The impact of reducing precomputation on sequential editing (editing consecutively multiple times) is not analyzed.
- Performance on downstream tasks (e.g., post-edit QA, reasoning capabilities) is not evaluated, relying only on standard editing metrics.
- The scope of models is limited (only GPT2-XL, GPT-J, Llama2-7B), leaving larger models (70B+) and newer architectures (e.g., Mistral, Qwen) unvalidated.
- The selection of $d_m$ remains empirical, lacking an adaptive adjustment mechanism.

## Related Work & Insights
- **vs Original MEMIT/ROME/EMMET**: Fully compatible; only reduces the precomputation cost with zero modifications to the editing algorithms themselves.
- **vs AlphaEdit**: AlphaEdit uses null-space constraints to solve the forgetting problem in sequential editing, while FastMEMIT addresses precomputation efficiency. The two are orthogonal and complementary.
- **vs In-context Editing (SERAC/ICE)**: These methods do not modify parameters but suffer from low inference efficiency; FastMEMIT retains the efficient inference benefits of parameter modification.

## Rating
- Novelty: ⭐⭐⭐ The insight is elegant, but the technical contribution is relatively limited (essentially analyzing redundant parameters of existing methods).
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong systematic evaluation across three models, two editing methods, and multiple batch sizes.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivation, well-defined motivation, and detailed charts.
- Value: ⭐⭐⭐⭐ Highly practical, directly addressing a key engineering bottleneck in knowledge editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ICLR 2026\] MobiEdit: Resource-efficient Knowledge Editing for Personalized On-device LLMs](../../ICLR2026/knowledge_editing/mobiedit_resource-efficient_knowledge_editing_for_personalized_on-device_llms.md)
- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](../../ACL2026/knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[ICLR 2026\] MoEEdit: Efficient and Routing-Stable Knowledge Editing for Mixture-of-Experts LLMs](../../ICLR2026/knowledge_editing/moeedit_efficient_and_routing-stable_knowledge_editing_for_mixture-of-experts_ll.md)
- [\[ACL 2025\] Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)

</div>

<!-- RELATED:END -->
