---
title: >-
  [Paper Note] FiRA: Can We Achieve Full-Rank Training of LLMs Under Low-Rank Constraint?
description: >-
  [NeurIPS 2025][Model Compression][low-rank training] This paper proposes Fira, the first LLM training framework that achieves full-rank training (full-rank gradients + full-rank weights) under low-rank constraints. By ob…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "low-rank training"
  - "memory-efficient"
  - "full-rank gradient"
  - "Adam optimizer"
  - "gradient projection"
date: 2026-05-08
content_hash: 4a9026915543ff2a
---

# FiRA: Can We Achieve Full-Rank Training of LLMs Under Low-Rank Constraint?

**Conference**: NeurIPS 2025
**arXiv**: [2410.01623](https://arxiv.org/abs/2410.01623)  
**Code**: Available (github.com/xichen-fy/Fira)  
**Area**: Model Compression / LLM Efficiency
**Keywords**: low-rank training, memory-efficient, full-rank gradient, Adam optimizer, gradient projection

## TL;DR

This paper proposes Fira, the first LLM training framework that achieves full-rank training (full-rank gradients + full-rank weights) under low-rank constraints. By observing that the optimizer scaling factors in low-rank and full-rank training are highly similar, Fira approximates the correction of out-of-subspace gradients using low-rank scaling factors, and employs a norm-growth limiter to prevent loss spikes. Fira outperforms LoRA and GaLore in both pretraining and fine-tuning settings.

## Background & Motivation

The primary memory bottleneck in LLM training stems from optimizer states: training LLaMA-7B requires 58 GB of GPU memory, of which the Adam optimizer states alone occupy 28 GB—more than the model parameters themselves. Low-rank training is an effective strategy for reducing memory usage, but existing methods are confined to low-rank subspaces:

**LoRA**: Decomposes weights into low-rank matrices $W = W_0 + BA$, restricting training to a low-rank subspace of the weight space, which limits representational capacity.

**GaLore**: Projects gradients into a low-rank subspace via SVD, i.e., $R_t = P_t^\top G_t$. Although full-rank weights are trained, **gradient information outside the subspace is discarded**.

**ReLoRA**: Attempts to approximate full-rank updates through multiple successive low-rank updates, but still requires a full-rank warm-up phase and cannot achieve fully memory-efficient training.

**Key Challenge**: Low-rank constraints reduce memory ↔ full-rank training preserves performance. Can both be achieved simultaneously?

**Core difficulty**: Out-of-subspace gradients $(G_t - P_t R_t)$ lack corresponding optimizer states for Adam correction. Naively adding them back (as in GaLore-add) is equivalent to applying SGD to this component, which performs poorly and introduces gradient inconsistency.

## Method

### Overall Architecture

The core idea of Fira is to decompose the full-rank gradient into two components and handle each separately:

$$W_{t+1} = W_t - \eta P_t \psi_t(R_t) - \eta \phi_t(R_t)(G_t - P_t R_t)$$

- **$P_t \psi_t(R_t)$**: In-subspace gradient, corrected normally by low-rank Adam.
- **$\phi_t(R_t)(G_t - P_t R_t)$**: Out-of-subspace gradient, approximately corrected via norm-based scaling.
- **$\phi_t(R_t)$**: Scaling factor computed from the low-rank optimizer states.

### Key Designs

#### 1. Core Observation: Similarity of Scaling Factors

The scaling factor is defined as the correction ratio applied by Adam to the gradient norm:

$$\phi_t(R_t) = \frac{\|\psi_t(R_t)\|}{\|R_t\|}$$

**Key finding**: During LLM training, the scaling factors of low-rank and full-rank training are highly similar at the matrix level:

| Model Size | Matrix-level Cosine Sim | Matrix-level MSE | Column-level Cosine Sim | Column-level MSE |
|-----------|------------------------|------------------|------------------------|------------------|
| 60M | 0.9922 | 3e-04 | 0.9273 | 3e-05 |
| 130M | 0.9901 | 2e-04 | 0.9046 | 2e-05 |
| 350M | 0.9893 | 1e-04 | 0.9174 | 1e-05 |
| 1B | 0.9795 | 2e-04 | 0.9229 | 1e-05 |

Cosine similarity exceeds 0.97 and MSE is negligible, indicating that the scaling behavior of the low-rank optimizer can approximate that of the full-rank optimizer.

#### 2. Norm-Based Scaling

**Matrix-level scaling** (Fira-matrix): Applies a uniform scaling factor derived from the low-rank gradient to the entire out-of-subspace gradient matrix.

**Column-level scaling** (Fira, more fine-grained): Computes an independent scaling factor for each column of the weight matrix:

$$\phi_t(R_t)_i = \frac{\|\psi(R_{t,:,i})\|}{\|R_{t,:,i}\|}, \quad i=1,2,\dots,n$$

Column-level similarity is also strong (Cosine Sim > 0.90), enabling more precise approximate correction.

#### 3. Norm-Growth Limiter

**Problem**: Instability in the low-rank optimizer combined with projection matrix switching leads to sudden gradient spikes at early training stages, causing loss spikes.

**Root cause analysis**:
- Switching the projection matrix $P_t$ every $T$ steps creates a mismatch between old optimizer states and the new projection.
- Out-of-subspace gradients retain their original direction but lack Adam's gradient stabilization.

**Solution**: Constrain the relative growth rate of the gradient norm:

$$\text{if } \frac{\|S_t\|}{\|S_{t-1}\|} > \gamma \text{ then } S_t \leftarrow \frac{S_t}{\|S_t\|} \cdot \gamma \|S_{t-1}\|$$

where $\gamma = 1.01$ (universal across all experiments and insensitive to the choice). This converts abrupt spikes into gradual growth, and is more flexible than absolute gradient clipping, which does not account for magnitude differences across different weight matrices.

### Implementation Details

- Compared to GaLore, Fira requires storing only one additional scalar $\|S_{t-1}\|$ per weight matrix, with negligible memory overhead.
- Only 3 extra lines of code are needed; the method is plug-and-play.
- The only additional hyperparameter is $\gamma$, which is fixed at 1.01.

## Key Experimental Results

### Main Results: LLaMA Pretraining (C4 dataset, validation perplexity ↓)

| Method | 60M | 130M | 350M | 1B |
|--------|-----|------|------|-----|
| Full-Rank | 34.06 (0.48G) | 25.08 (1.01G) | 18.80 (2.74G) | 15.56 (10.40G) |
| **Fira** | **31.06 (0.36G)** | **22.73 (0.77G)** | **16.85 (1.90G)** | **14.31 (6.98G)** |
| GaLore | 34.88 (0.36G) | 25.36 (0.77G) | 18.95 (1.90G) | 15.64 (6.98G) |
| LoRA | 34.99 (0.44G) | 33.92 (0.99G) | 25.58 (2.12G) | 19.21 (7.36G) |
| ReLoRA | 37.04 (0.44G) | 29.37 (0.99G) | 29.08 (2.12G) | 18.33 (7.36G) |

Fira **substantially outperforms** GaLore, LoRA, and ReLoRA at all scales, and even surpasses full-rank training (31.06 vs. 34.06), achieving the best performance under identical memory constraints.

### LLaMA 7B Pretraining

Using a rank **8× smaller** (i.e., optimizer state memory reduced to 1/8 of GaLore's), Fira still significantly outperforms GaLore, validating its effectiveness at large scale.

### Fine-tuning (LLaMA-7B, commonsense reasoning, 8 tasks)

| Method | Memory | BoolQ | HellaSwag | WinoGrande | Avg. |
|--------|--------|-------|-----------|------------|------|
| **Fira** | 14.44G | 69.4 | **76.8** | **81.2** | **76.9** |
| GaLore | 14.44G | 69.5 | 32.2 | 18.0 | 62.7 |
| LoRA | 14.53G | 68.9 | 78.1 | 78.8 | 74.7 |
| Full-rank | 56.00G | 64.2 | 42.3 | 66.5 | 58.6 |

GaLore fails severely on HellaSwag and WinoGrande. Fira achieves the best performance on 5 out of 8 tasks and the highest average score of 76.9.

### Ablation Study (LLaMA 60M Pretraining)

| Variant | Perplexity ↓ |
|---------|-------------|
| **Fira (full)** | **31.06** |
| Fira-matrix (matrix-level scaling) | 31.52 |
| Fira-w.o.-limiter (no limiter) | 32.22 |
| Fira-gradient-clipping (clipping instead of limiter) | 31.22 |
| Fira-gradient-shrink | 33.98 |
| Fira-tensor-wise-scaling | 33.81 |
| Fira-w.o.-scaling (no scaling, equiv. to GaLore-add) | 37.06 |

Key conclusions:
- No scaling (37.06) is far worse than with scaling (31.06), confirming the necessity of norm-based scaling.
- Column-level outperforms matrix-level (31.06 vs. 31.52).
- The norm-growth limiter outperforms all alternative stabilization strategies.

### Performance Across Different Ranks

| Rank | Fira | GaLore | Gap |
|------|------|--------|-----|
| 4 | ~35 | ~48 | Large |
| 16 | ~32 | ~37 | Significant |
| 64 | ~31 | ~33 | Clear |
| 128 | ~31 | ~32 | Still present |

Fira approaches full-rank performance even at extremely low ranks, whereas GaLore degrades sharply, demonstrating that Fira effectively leverages out-of-subspace information.

### Key Findings

1. The Adam scaling factors of low-rank and full-rank training are indeed similar—a phenomenon that holds stably across scales (60M → 1B).
2. Fira even surpasses full-rank training, possibly because the stochasticity introduced by norm-based scaling helps escape local optima.
3. Projection matrix switching is the primary source of loss spikes in low-rank training; the norm-growth limiter effectively addresses this.
4. At lower ranks, Fira's advantage over GaLore becomes more pronounced.

## Highlights & Insights

- **Novel theoretical insight**: The observation that "low-rank and full-rank scaling factors are similar" is both interesting and practically useful, providing a theoretical basis for approximating full-rank training with low-rank methods.
- **Surpasses full-rank training**: It is rare for a constrained method to outperform its unconstrained counterpart.
- **Minimal implementation overhead**: Only 3 extra lines of code, plug-and-play, with no architectural modifications.
- **Comprehensive validation**: Effectiveness is verified across scales from 60M to 7B, spanning both pretraining and fine-tuning.

## Limitations & Future Work

1. **SVD computation cost**: Inherited from GaLore; SVD must be performed every $T$ steps (less than 10% overhead, but non-zero).
2. **Insufficient theoretical explanation for scaling factor similarity**: The paper observes the phenomenon but lacks in-depth theoretical analysis.
3. **Only validated with Adam**: Applicability to other optimizers such as AdaFactor and Lion remains unknown.
4. **Final perplexity not reported for 7B pretraining**: Only loss curves are compared.
5. **Future directions**: Integration with quantization-aware training (QLoRA), exploration of adaptive rank scheduling, and extension to diffusion model training.

## Related Work & Insights

- **GaLore**: A pioneer in low-rank gradient projection; Fira builds upon it by recovering out-of-subspace gradients.
- **LoRA/ReLoRA**: Parameter low-rank decomposition approaches; Fira demonstrates that the gradient projection route has a higher performance ceiling.
- **Flora**: A random projection method that performs modestly in fine-tuning settings.
- **Insight**: The scaling behavior of optimizers exhibits stability across subspaces, suggesting that Adam's adaptive mechanism primarily operates on global gradient statistics rather than element-wise details.

## Rating

- Novelty: ★★★★★ (first to achieve full-rank training under low-rank constraints; core observation is highly original)
- Technical Depth: ★★★★☆ (thorough scaling factor analysis; theoretical proof could be more rigorous)
- Experimental Thoroughness: ★★★★★ (60M → 7B, pretraining + fine-tuning, detailed ablations and rank analysis)
- Practical Value: ★★★★★ (plug-and-play, 3 lines of code, significant improvements in low-rank training, easy to deploy)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)
- [\[NeurIPS 2025\] GoRA: Gradient-Driven Adaptive Low Rank Adaptation](gora_gradient-driven_adaptive_low_rank_adaptation.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)

</div>

<!-- RELATED:END -->
