---
title: >-
  [Paper Note] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning
description: >-
  [ICML 2026][Model Compression][LoRA] The authors prove that LoRA cumulative updates are trapped in a fixed low-rank subspace and propose ScaLoRA: after merging the old $AB^\top$ into $W^{pt}$ at each step…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "LoRA"
  - "High-Rank Updates"
  - "Column Scaling"
  - "AdamW Moment Equivariance"
  - "ScaLoRA"
date: 2026-05-08
content_hash: 401f9bdaa88a3f88
---

# ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2510.23818](https://arxiv.org/abs/2510.23818)  
**Code**: Not explicitly stated (None)  
**Area**: Model Compression / Parameter-Efficient Fine-Tuning / LoRA Variants  
**Keywords**: LoRA, High-Rank Updates, Column Scaling, AdamW Moment Equivariance, ScaLoRA

## TL;DR
The authors prove that LoRA cumulative updates are trapped in a fixed low-rank subspace and propose ScaLoRA: after merging the old $AB^\top$ into $W^{pt}$ at each step, the adapter is restarted using an **analytically derived optimal "column scaling"**. This allows AdamW first/second order moment estimators to be transferred via $O((m+n)r)$ equivariant mapping (eliminating the need for resets/warm-up), enabling cumulative updates to naturally achieve high rank. It consistently outperforms LoRA, MoRA, HiRA, ReLoRA, and LoRA-GA on DeBERTaV3, LLaMA2-7B, LLaMA3-8B, and Gemma3-12B.

## Background & Motivation

**Background**: LoRA (Hu et al. 2022) constrains updates in full parameter $W = W^{pt} + AB^\top$ to two thin matrices $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{n \times r}$, where $r \ll m, n$, significantly saving memory and computation. Subsequent variants like DoRA, QLoRA, FourierFT, HiRA, MoRA, and ReLoRA attempt to improve performance or expand applications.

**Limitations of Prior Work**: There remains a **performance gap** between LoRA and full fine-tuning, which intensifies as rank $r$ decreases. Fundamentally, the $T$-step cumulative update $\sum_t \Delta W_t = A_T B_T^\top - A_0 B_0^\top = A_T B_T^\top$ always resides in a fixed rank-$r$ subspace—telescoping effects cancel out cross-step information. Existing "high-rank LoRA" solutions have drawbacks:

- **ReLoRA** periodically merges $AB^\top$ into $W^{pt}$ and reinitializes new $AB^\top$, but each merge requires **restarting the optimizer + re-performing learning-rate warm-up**, leading to slow convergence;
- **MoRA** replaces $A(B^\top X)$ with a non-linear mapping $f_{decompress}(M f_{compress}(X))$ to achieve high rank, but the design of $f_{compress/decompress}$ is labor-intensive;
- **HiRA** uses the Hadamard product $W^{ft} = (AB^\top) \odot W^{pre}$ to achieve high rank, but backpropagation involves $m \times n$ Hadamard products, resulting in **$O(mn)$ memory**, which is not scalable for large LLMs.

**Key Challenge**: To achieve "high-rank cumulative updates using a low-rank adapter," a different subspace is needed at each step. However, changing the subspace invalidates the $(m_t, v_t)$ moment estimators maintained by AdamW, necessitating either a slow restart or an expensive recalculation. These requirements appear incompatible.

**Goal**: Find an analytical expression for the "optimal adapter update" and a transformation form that allows moment estimators to be equivariantly mapped from the old adapter to the new one in $O((m+n)r)$ without restarting. The final objective is to achieve high-rank cumulative updates and fast convergence without increasing memory consumption.

**Key Insight**: Starting from the Lipschitz upper bound of the loss, the authors prove that the optimal adapter at each step is "equivalent to a truncated SVD of the full FT gradient $\nabla \ell(W_t) = U_t \Sigma_t V_t^\top$ taking the top $2r$ directions." Since SVD is too complex, they constrain the "relationship between adapters before and after replacement" to a simple **column scaling** transformation: $\tilde{A} = A \cdot \text{diag}(\alpha), \tilde{B} = B \cdot \text{diag}(\beta)$. This is one of the few transformations that allow AdamW moments to be analytically mapped via equivariance.

**Core Idea**: Search for column scaling factors (global analytical optimal solution) within the LoRA subspace that are "optimal for current loss descent." Every step or every $I$ steps, the $\tilde{A}_t \tilde{B}_t^\top$ after scaling by the optimal $(\alpha^*, \beta^*)$ is merged into $\tilde{W}^{pt}_t$, followed by training new $A_{t+1}, B_{t+1}$. Column scaling allows moments to be transferred equivariantly at almost no cost, thus cumulative updates naturally span multiple directions and increase in rank.

## Method

### Overall Architecture
The LoRA representation $W_t = W^{pt} + A_t B_t^\top$ is retained, but a "virtual merge + replacement" mechanism is introduced: $W_t = \underbrace{(W^{pt} + A_t B_t^\top - \tilde{A}_t \tilde{B}_t^\top)}_{\tilde{W}^{pt}_t,\,\text{merge \& freeze}} + \underbrace{\tilde{A}_t \tilde{B}_t^\top}_{\text{learnable}}$. At each step (or every $I$ steps): (1) Calculate optimal scaling $(\alpha^*_t, \beta^*_t)$ using analytical formulas; (2) Merge the current $A_t B_t^\top$ subspace into $\tilde{W}^{pt}_t$ and replace the learnable part with $\tilde{A}_t = A_t \text{diag}(\alpha^*_t)$, $\tilde{B}_t = B_t \text{diag}(\beta^*_t)$; (3) Map AdamW $m, v$ from old $(A_t, B_t)$ to new $(\tilde{A}_t, \tilde{B}_t)$ equivariantly using Lemma 3.3/3.6; (4) Perform the next GD/AdamW update to get $A_{t+1}, B_{t+1}$. Because the optimal subspace differs each round, the cumulative weight $\sum_{t=0}^{T-1} \Delta \tilde{W}_t = \sum_t A_{t+1} B_{t+1}^\top - \sum_t \tilde{A}_t \tilde{B}_t^\top$ no longer telescopes, and the rank continues to rise.

### Key Designs

1.  **Theoretical Characterization of the Optimal Adapter (Theorem 3.2)**:
    - **Function**: Define the ideal per-step adapter to provide a theoretical target for subsequent approximations.
    - **Mechanism**: From the $L$-smooth assumption, $\ell(W_t + \Delta W_t) \leq \ell(W_t) + \langle \nabla \ell, \Delta W_t \rangle + \frac{L}{2}\|\Delta W_t\|_F^2$. Minimizing the right side yields the optimal full-FT update $\Delta W_t^* = -\frac{1}{L} \nabla \ell(W_t)$. Substituting LoRA's $\Delta \tilde{W}_t = -\eta \nabla \ell \tilde{B}_t \tilde{B}_t^\top - \eta \tilde{A}_t \tilde{A}_t^\top \nabla \ell + O(\eta^2)$ and completing the square results in the equivalent problem "minimize $\|\Delta W_t^* - \Delta \tilde{W}_t\|_F^2$". Theorem 3.2 proves: when $\text{rank}(\nabla \ell(W_t)) \geq 2r$, the optimal $\tilde{A}_t^*, \tilde{B}_t^*$ are equivalent to the rank-$2r$ truncated SVD of $\nabla \ell$, where the first $2r$ singular vectors form the new adapter.
    - **Design Motivation**: This step establishes the correspondence "optimal adapter $\leftrightarrow$ truncated SVD," indicating that the gap between LoRA and full FT is determined by the top-$2r$ singular space of the current gradient. However, SVD complexity $O(Smnr)$ is too expensive and requires optimizer restarts, making it a "theoretical upper bound" requiring cheaper approximations.

2.  **Optimal Column Scaling + AdamW Moment Equivariance (Theorem 3.5 / 3.7 + Lemma 3.6)**:
    - **Function**: Constrain the adapter search space to column scaling $\tilde{A} = A \text{diag}(\alpha)$ and $\tilde{B} = B \text{diag}(\beta)$, and prove (a) global optimal $(\alpha^*, \beta^*)$ can be found analytically in $O((m+n)r^2)$ time; (b) AdamW $m, v$ can be mapped in $O((m+n)r)$ from old $(A, B)$ to new $(\tilde{A}, \tilde{B})$, **avoiding restarts entirely**.
    - **Mechanism**: Under column scaling constraints, the loss upper bound becomes a quadratic problem in $(\alpha, \beta)$ defined by $\|\frac{1}{L}\nabla\ell - \eta \nabla\ell B \text{diag}^2(\beta) B^\top - \eta A \text{diag}^2(\alpha) A^\top \nabla\ell\|_F^2$. Theorem 3.7 proves that if the linear system $[(S_t^{A\top} S_t^A) \odot (S_t^{B\top} S_t^B)] v_t = \lambda_t$ has a non-negative solution (observed in ~80% of LLM layers), the global optimum is $[\alpha^*_t; \beta^*_t] = \pm \frac{1}{\sqrt{L\eta}} v_t^{\circ 1/2}$, where $S_t^A, S_t^B$ are small matrices constructed from gradients and adapters. If non-negativity fails, it degrades to "scalar scaling" (Theorem 3.5), which also has an analytical global optimum. Regarding moments: since $\tilde{A} = A \text{diag}(\alpha)$ is per-column scaling, AdamW moments correspond element-wise to the adapter; multiplying by $\alpha$ scales moments by $\alpha$ (first order) or $\alpha^2$ (second order)—a simple $O((m+n)r)$ operation. Other transformations (row scaling, multiplication by full-rank matrices) do not maintain "moment equivariance."
    - **Design Motivation**: This is the engineering core. Column scaling is chosen specifically because it is **one of the few transformations ensuring analytical moment equivariance**, bypassing all ReLoRA restart/warm-up issues. $L$ is treated as a hyperparameter for grid search.

3.  **ScaLoRA and Amortized Variant ScaLoRA-I**:
    - **Function**: Combine the components into an implementable algorithm and provide an "amortized" variant calculating optimal scaling every $I$ steps.
    - **Mechanism**: At each step, if Theorem 3.7 holds, use column scaling $\tilde{A}_t = A_t \text{diag}(\alpha^*_t), \tilde{B}_t = B_t \text{diag}(\beta^*_t)$ + Lemma 3.6. Otherwise, use scalar scaling from Theorem 3.5 + Lemma 3.3. Merge $A_t B_t^\top - \tilde{A}_t \tilde{B}_t^\top$ in-place into $W^{pt}$, so spatial overhead only increases by $O((m+n+r)r)$. Total time complexity is $O(mnr + (m+n+r)r^2)$. **ScaLoRA-I** performs scaling/merging every $I$ steps, amortizing costs to $1/I$; since $\eta$ is small, optimal scaling is near 1, and frequent updates yield diminishing returns, making $I=10$ nearly lossless.
    - **Design Motivation**: LLMs train layer-by-layer; performing column scaling every step across hundreds of layers adds overhead. The amortized variant makes ScaLoRA scalable to 12B parameter models. Unlike MoRA/HiRA, the design allows for amortization.

### Loss & Training
The LLM loss (CE / language modeling loss) remains unchanged; only the LoRA optimization logic is modified to include scaling-merging before/after AdamW updates. Hyperparameters include $L$ (via grid search), $\eta$, interval $I$, and rank $r$. Evaluations use $r=4$ (GLUE) and $r=8$ (LLaMA/Gemma), where gains are most significant for low ranks. The trade-off is that **storage must save the merged $W_t$ rather than small adapters** (disk is usually not the bottleneck).

## Key Experimental Results

### Main Results
**DeBERTaV3-base on GLUE ($r = 4$)**:

| Method | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI-m | QNLI | RTE | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Full FT | 69.19 | 95.63 | 89.46 | 91.60 | 92.40 | 89.90 | 94.03 | 83.75 | 88.25 |
| LoRA | 68.10 | 95.49 | 89.46 | 91.09 | 91.86 | 90.25 | 94.30 | 84.48 | 88.13 |
| MoRA | 69.67 | 95.45 | 89.62 | 90.90 | 91.83 | 90.05 | 93.81 | 85.44 | 88.35 |
| HiRA | 68.82 | 95.53 | 89.95 | 91.15 | **92.19** | 90.24 | 94.15 | 85.68 | 88.46 |
| **ScaLoRA** | **69.86** | **95.83** | **90.28** | **91.47** | 92.10 | **90.36** | **94.34** | **87.61** | **88.98** |

ScaLoRA achieves the best results in 7 out of 8 tasks, averaging >0.5% higher than HiRA and even outperforming Full FT (which overfits on small datasets).

**LLaMA2-7B / LLaMA3-8B Commonsense Reasoning ($r = 8$)**:

| Model | LoRA | ReLoRA | LoRA-GA | MoRA | HiRA | ScaLoRA | ScaLoRA-I | LoRA $r=32$ |
|---|---|---|---|---|---|---|---|---|
| LLaMA2-7B Avg | 73.63 | 74.40 | 74.34 | 73.82 | 73.95 | 74.51 | **74.75** | 74.52 |
| LLaMA3-8B Avg | 76.83 | 77.26 | 77.22 | 77.27 | 77.46 | **77.85** | 77.57 | 77.54 |

ScaLoRA(-I) at $r=8$ **outperforms LoRA at $r=32$**—achieving better results with 1/4 of the parameters.

### Ablation Study

| Configuration | Observation |
|---|---|
| Full ScaLoRA | Baseline |
| No column scaling (Scalar only, Thm 3.5) | Slight performance drop, but still beats LoRA—scalar scaling provides value |
| Per-step vs. every 10 steps (ScaLoRA-I) | $I=10$ is nearly lossless, confirming optimal scaling is near 1 |
| No moment transfer (Restart every scale) | Severe degradation, equivalent to ReLoRA restart effects |
| Different rank $r$ | ScaLoRA beats LoRA across $r=4, 8, 16, 32$; advantage is largest at low ranks |
| Figure 2(b) on RTE | LoRA cumulative rank = 4 (fixed); ScaLoRA rank reaches 54 |
| Figure 2(c) | $\text{rank}(\nabla \ell(W_t)) \geq 2r$ holds almost everywhere in LLMs |
| Figure 2(d) | ~80% of layers meet non-negativity for column scaling; 20% fallback to scalar |

### Key Findings
- LoRA's cumulative update rank **is really just the nominal $r$**—but by changing subspaces per step, it can rise to 50+ without increasing per-step parameters.
- **Low-rank ScaLoRA > High-rank LoRA**: ScaLoRA is most valuable under tight memory budgets (e.g., $r=8$ beats $r=32$ LoRA).
- Optimal scaling factors $\alpha^*, \beta^*$ are generally close to 1 (due to small $\eta$), making amortized scaling nearly lossless—key for 12B+ models.
- Difference from ReLoRA: ReLoRA is "merge + random restart + warm-up"; ScaLoRA is "merge + analytical optimal scaling + moment equivariance"—the latter is theoretically optimal and computationally cheap.

## Highlights & Insights
- **"Column scaling is a rare moment-equivariant transformation"**: This is a brilliant observation. Most research focuses only on transformation expressivity, ignoring compatibility with AdamW states. ScaLoRA jointly designs the transformation and optimizer state consistency.
- **Theoretical Grounding**: Theorem 3.2's SVD representation provides a "theoretical target." The use of column scaling as a feasible approximation with an analytical global optimum creates a smooth theory-to-engineering transition.
- **Amortization**: Unlike MoRA/HiRA, which enforce high-rank constraints at every step, ScaLoRA-I achieves similar effects with 1/$I$ the overhead, allowing scalability to massive models.
- **Empirical Proof**: The finding that cumulative rank naturally rises to ~50 and plateaus **retroactively validates the LoRA hypothesis**—optimal fine-tuning updates indeed reside on a manifold much higher than $r$ but still limited.

## Limitations & Future Work
- **Extra Storage**: Merged $W_t$ must be saved, meaning one cannot ship just $A_t, B_t$ adapters. This is a limitation for adapter-only deployment scenarios (e.g., sharing a base model across multiple tasks).
- Computational complexity $O(mnr)$ is similar to HiRA and a constant factor higher than vanilla LoRA.
- Assumption $\text{rank}(\nabla \ell(W_t)) \geq 2r$ might not hold for extremely small batches or high $r$.
- Validation is limited to NLU/Commonsense/Math LLM tasks; multimodal, vision, or RL fine-tuning are not tested.
- Combination with QLoRA (quantization) or DoRA (magnitude-direction) is not discussed.

## Related Work & Insights
- **vs LoRA (Hu et al. 2022)**: Base case; ScaLoRA uses the same $A, B$ parameterization but achieves high-rank cumulative updates.
- **vs ReLoRA (Lialin et al. 2024)**: Both merge and restart, but ReLoRA resets the optimizer. ScaLoRA uses optimal scaling + moment equivariance to avoid resets.
- **vs MoRA (Jiang et al. 2024)**: MoRA uses complex non-linear mappings for high rank; ScaLoRA maintains LoRA's simple structure.
- **vs HiRA (Huang et al. 2025)**: HiRA uses Hadamard products with $O(mn)$ memory; ScaLoRA uses $O((m+n+r)r)$.
- **vs LoRA-GA (Wang et al. 2024)**: Theorem 3.2 reveals LoRA-GA is a special case of ScaLoRA's optimal condition at $t=0$.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Column scaling as a moment-equivariant transformation + analytical optimal" is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 model scales (184M to 12B), 3 task types, and 5+ baselines.
- Writing Quality: ⭐⭐⭐⭐ Dense theoretical derivations but clear logic.
- Value: ⭐⭐⭐⭐⭐ Will likely be widely adopted as a high-performance alternative to standard LoRA without significant memory cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ACL 2026\] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](../../ACL2026/model_compression/polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](../../ICLR2026/model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](../../NeurIPS2025/model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)

</div>

<!-- RELATED:END -->
