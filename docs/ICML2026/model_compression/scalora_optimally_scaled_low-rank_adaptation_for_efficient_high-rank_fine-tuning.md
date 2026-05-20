---
title: >-
  [Paper Note] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning
description: >-
  [ICML 2026][Model Compression][LoRA] The authors prove that LoRA's cumulative updates are trapped in a fixed low-rank subspace and propose ScaLoRA: at each step, after merging the old $AB^\top$ into $W^{pt}$…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "LoRA"
  - "high-rank update"
  - "column scaling"
  - "AdamW moment equivariance"
  - "ScaLoRA"
date: 2026-05-08
content_hash: 58f360c4fe1c19a3
---

# ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2510.23818](https://arxiv.org/abs/2510.23818)  
**Code**: Not specified in the paper (none)  
**Area**: Model Compression / Parameter-Efficient Fine-Tuning / LoRA Variants  
**Keywords**: LoRA, high-rank update, column scaling, AdamW moment equivariance, ScaLoRA

## TL;DR
The authors prove that LoRA's cumulative updates are trapped in a fixed low-rank subspace and propose ScaLoRA: at each step, after merging the old $AB^\top$ into $W^{pt}$, the adapter is **restarted with an analytically optimal "column scaling"**, enabling AdamW first/second moments to be transferred equivariantly in $O((m+n)r)$ time (no reset/warm-up needed), and cumulative updates naturally become high-rank. ScaLoRA consistently outperforms LoRA / MoRA / HiRA / ReLoRA / LoRA-GA on DeBERTaV3, LLaMA2-7B, LLaMA3-8B, and Gemma3-12B.

## Background & Motivation

**Background**: LoRA (Hu et al. 2022) constrains updates in the full-parameter $W = W^{pt} + AB^\top$ to two thin matrices $A \in \mathbb{R}^{m \times r}, B \in \mathbb{R}^{n \times r}$, with $r \ll m, n$, greatly saving memory and computation. Subsequent variants like DoRA, QLoRA, FourierFT, HiRA, MoRA, ReLoRA, etc., aim to improve performance or broaden applicability.

**Limitations of Prior Work**: There is **always a performance gap** between LoRA and full fine-tuning, which widens as rank $r$ decreases. The root cause is that the $T$-step cumulative update $\sum_t \Delta W_t = A_T B_T^\top - A_0 B_0^\top = A_T B_T^\top$ always lies in a fixed rank-$r$ subspace—telescoping cancels all cross-step information. Existing "high-rank LoRA" solutions have their own issues:

- **ReLoRA** periodically merges $AB^\top$ into $W^{pt}$ and randomly reinitializes a new $AB^\top$, but each merge requires **optimizer reset + learning-rate warm-up**, leading to slow convergence;
- **MoRA** replaces $A(B^\top X)$ with a nonlinear mapping $f_{decompress}(M f_{compress}(X))$, achieving high rank but requiring laborious design of $f_{compress/decompress}$;
- **HiRA** uses $W^{ft} = (AB^\top) \odot W^{pre}$ (Hadamard product) for high rank, but each step's backward pass involves an $m \times n$ Hadamard product, **$O(mn)$ memory**, making it unscalable for large LLMs.

**Key Challenge**: To "achieve high-rank cumulative updates with low-rank adapters," a different subspace is needed at each step; but changing subspaces invalidates AdamW's $(m_t, v_t)$ moment estimators, requiring either a reset (slow) or recomputation from scratch (expensive). These requirements seem incompatible.

**Goal**: Find an analytic expression for the "optimal adapter update"; identify an adapter transformation that allows the moment estimator to be mapped equivariantly from the old adapter to the new one in $O((m+n)r)$ time, without reset; ultimately, achieve high-rank cumulative updates and fast convergence without increasing memory.

**Key Insight**: Starting from the Lipschitz upper bound of the loss, the authors prove that the optimal adapter at each step is "equivalent to truncating the SVD of the full FT gradient $\nabla \ell(W_t) = U_t \Sigma_t V_t^\top$ to the top $2r$ directions." However, SVD is too expensive, so the relationship between old and new adapters is constrained to **column scaling** $\tilde{A} = A \cdot \text{diag}(\alpha), \tilde{B} = B \cdot \text{diag}(\beta)$—one of the few transformations allowing analytic equivariant transfer of AdamW moments.

**Core Idea**: Within the LoRA subspace, search for the column scaling factors that are "optimal for current loss reduction" (with an analytic global optimum). At each step or every $I$ steps, use the optimal $(\alpha^*, \beta^*)$ to scale and merge $\tilde{A}_t \tilde{B}_t^\top$ into $\tilde{W}^{pt}_t$, then continue training new $A_{t+1}, B_{t+1}$. Column scaling enables almost free equivariant moment transfer, so cumulative updates span more directions and the rank automatically increases.

## Method

### Overall Architecture
Retain LoRA's $W_t = W^{pt} + A_t B_t^\top$ formulation, but introduce a "virtual merge + replace" mechanism: $W_t = \underbrace{(W^{pt} + A_t B_t^\top - \tilde{A}_t \tilde{B}_t^\top)}_{\tilde{W}^{pt}_t,\,\text{merge \& freeze}} + \underbrace{\tilde{A}_t \tilde{B}_t^\top}_{\text{learnable}}$. At each step (or every $I$ steps): (1) Compute the optimal scaling $(\alpha^*_t, \beta^*_t)$ analytically; (2) Merge the current $A_t B_t^\top$ subspace into $\tilde{W}^{pt}_t$, and set the new learnable part as $\tilde{A}_t = A_t \text{diag}(\alpha^*_t)$, $\tilde{B}_t = B_t \text{diag}(\beta^*_t)$; (3) Use Lemma 3.3/3.6 to map AdamW's $m, v$ from old $(A_t, B_t)$ to new $(\tilde{A}_t, \tilde{B}_t)$ equivariantly; (4) Perform the next GD/AdamW update to obtain $A_{t+1}, B_{t+1}$ and repeat. Since each round's optimal subspace differs, the $T$-step cumulative weight $\sum_{t=0}^{T-1} \Delta \tilde{W}_t = \sum_t A_{t+1} B_{t+1}^\top - \sum_t \tilde{A}_t \tilde{B}_t^\top$ no longer telescopes, and the rank keeps increasing.

### Key Designs

1. **Theoretical Characterization of the Optimal Adapter (Theorem 3.2)**:

    - **Function**: Clarifies the "ideal adapter choice at each step," providing a theoretical target for subsequent approximations.
    - **Mechanism**: Under the $L$-smooth assumption, $\ell(W_t + \Delta W_t) \leq \ell(W_t) + \langle \nabla \ell, \Delta W_t \rangle + \frac{L}{2}\|\Delta W_t\|_F^2$. Minimizing the right side yields the full-FT optimal update $\Delta W_t^* = -\frac{1}{L} \nabla \ell(W_t)$. Substituting LoRA's $\Delta \tilde{W}_t = -\eta \nabla \ell \tilde{B}_t \tilde{B}_t^\top - \eta \tilde{A}_t \tilde{A}_t^\top \nabla \ell + O(\eta^2)$ and completing the square, the equivalent problem is "minimize $\|\Delta W_t^* - \Delta \tilde{W}_t\|_F^2$." The theorem proves: when $\text{rank}(\nabla \ell(W_t)) \geq 2r$, the optimal $\tilde{A}_t^*, \tilde{B}_t^*$ is equivalent to taking the rank-$2r$ truncated SVD of $\nabla \ell$, using the top $2r$ left/right singular vectors (partitioned appropriately) to form the new adapter.
    - **Design Motivation**: This step establishes the correspondence "optimal adapter ↔ truncated SVD," showing that the gap between LoRA and full FT is essentially determined by the top-$2r$ singular space of the current gradient. However, SVD's $O(Smnr)$ complexity is too high for per-step computation and would require optimizer reset—so this is a "theoretical upper bound" needing a cheaper approximation.

2. **Optimal Column Scaling + AdamW Moment Equivariance (Theorem 3.5 / 3.7 + Lemma 3.6)**:

    - **Function**: Restricts the "adapter replacement" search space to column scaling transformations $\tilde{A} = A \text{diag}(\alpha)$, $\tilde{B} = B \text{diag}(\beta)$, and proves (a) the global optimum $(\alpha^*, \beta^*)$ can be analytically solved in $O((m+n)r^2)$ time; (b) AdamW's $m, v$ can be mapped from old $(A, B)$ to new $(\tilde{A}, \tilde{B})$ in $O((m+n)r)$, **completely avoiding reset**.
    - **Mechanism**: Under column scaling, the loss upper bound becomes $\|\frac{1}{L}\nabla\ell - \eta \nabla\ell B \text{diag}^2(\beta) B^\top - \eta A \text{diag}^2(\alpha) A^\top \nabla\ell\|_F^2$—a quadratic problem in $(\alpha, \beta)$. Theorem 3.7 proves that when the linear system $[(S_t^{A\top} S_t^A) \odot (S_t^{B\top} S_t^B)] v_t = \lambda_t$ has a non-negative solution (empirically ~80% of LLM layers), the global optimum is $[\alpha^*_t; \beta^*_t] = \pm \frac{1}{\sqrt{L\eta}} v_t^{\circ 1/2}$, where $S_t^A, S_t^B$ are small matrices constructed from the current gradient and adapter. If the non-negativity condition fails, it degenerates to simpler "scalar scaling" (Theorem 3.5), which also has an analytic global optimum. For moments: since $\tilde{A} = A \text{diag}(\alpha)$ is column-wise scaling, AdamW's first/second moments correspond elementwise, so multiplying by $\alpha$ is equivalent to multiplying the moment by $\alpha$ (first moment) or $\alpha^2$ (second moment) per column—a simple $O((m+n)r)$ operation. Other transformations (row scaling, full-rank left/right multiplication) do not allow such "moment equivariance."
    - **Design Motivation**: This is the engineering core of the paper—column scaling is chosen because it is **one of the few transformations allowing analytic moment equivariance**, thus avoiding all the drawbacks of ReLoRA's reset/warm-up. $L$ is treated as a hyperparameter for grid search, not requiring actual Lipschitz estimation.

3. **ScaLoRA and Amortized Variant ScaLoRA-I**:

    - **Function**: Combines the above components into a practical algorithm, and provides a cost-saving variant that computes the optimal scaling every $I$ steps.
    - **Mechanism**: At each step, if Theorem 3.7's non-negativity condition holds, use column scaling $\tilde{A}_t = A_t \text{diag}(\alpha^*_t), \tilde{B}_t = B_t \text{diag}(\beta^*_t)$ + Lemma 3.6 to update moments; otherwise, use scalar scaling from Theorem 3.5: $\tilde{A}_t = \alpha^*_t A_t, \tilde{B}_t = \beta^*_t B_t$ + Lemma 3.3. Merge $A_t B_t^\top - \tilde{A}_t \tilde{B}_t^\top$ in-place into $W^{pt}$, so space overhead is only $O((m+n+r)r)$. Total time complexity is $O(mnr + (m+n+r)r^2)$; for small $r$, the latter is negligible. **ScaLoRA-I** performs scaling and merging every $I$ steps, amortizing the per-step cost by $1/I$; since $\eta$ is small and the optimal scaling is close to 1, frequent scaling yields diminishing returns, and $I = 10$ is nearly lossless.
    - **Design Motivation**: In practice, LLMs are trained layer by layer (hundreds of layers), and per-step column scaling increases overhead; the amortized variant allows ScaLoRA to scale to 12B-parameter models. MoRA/HiRA's high-rank mechanisms impose hard constraints at every step and cannot be amortized; this design choice gives ScaLoRA a particular advantage for large models.

### Loss & Training

The LLM training loss remains unchanged (still task CE / language modeling loss); only the LoRA module's optimization logic is modified: at each step, scaling-merge is performed before/after AdamW update. Main hyperparameters are $L$ (grid search), $\eta$, scaling interval $I$, and LoRA rank $r$. The paper uses $r = 4$ (GLUE) and $r = 8$ (LLaMA/Gemma tasks), showing the most significant improvements at low ranks. The trade-off is that **storage must save the merged $W_t$ instead of just the small adapter** (disk is not a bottleneck, but this differs from standard LoRA).

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

ScaLoRA achieves the best results on 7 out of 8 tasks, with an average over 0.5% higher than HiRA, and even surpasses Full FT (since Full FT overfits on small datasets).

**LLaMA2-7B / LLaMA3-8B Commonsense Reasoning ($r = 8$)**:

| Model | LoRA | ReLoRA | LoRA-GA | MoRA | HiRA | ScaLoRA | ScaLoRA-I | LoRA $r=32$ |
|---|---|---|---|---|---|---|---|---|
| LLaMA2-7B Avg | 73.63 | 74.40 | 74.34 | 73.82 | 73.95 | 74.51 | **74.75** | 74.52 |
| LLaMA3-8B Avg | 76.83 | 77.26 | 77.22 | 77.27 | 77.46 | **77.85** | 77.57 | 77.54 |

ScaLoRA(-I) at $r=8$ **outperforms LoRA $r=32$**—achieving higher performance with only 1/4 the parameters.

On mathematical reasoning (MetaMathQA / GSM8K / MATH) and Gemma3-12B, ScaLoRA also consistently leads (see Section 5+ of the paper for detailed numbers, omitted here for brevity).

### Ablation Study

| Configuration | Observation |
|---|---|
| Full ScaLoRA | baseline |
| Remove column scaling, only scalar scaling (Thm 3.5) | Slight performance drop, but still better than LoRA—indicating scalar scaling alone is beneficial |
| Per-step vs every 10 steps (ScaLoRA-I) | $I=10$ is nearly lossless, confirming that "optimal scaling is close to 1" |
| Disable moment equivariant transfer, re-accumulate moments after each scaling | Severe degradation, equivalent to ReLoRA reset |
| Different ranks $r$ | For $r=4, 8, 16, 32$, ScaLoRA consistently outperforms LoRA; the relative advantage is greater at lower ranks (at high ranks, LoRA is already strong) |
| Figure 2(b) on RTE | LoRA cumulative update rank = 4 (constant); ScaLoRA rank accumulates up to 54 |
| Figure 2(c) | The assumption $\text{rank}(\nabla \ell(W_t)) \geq 2r$ holds almost everywhere in LLMs |
| Figure 2(d) | ~80% of LoRA layers per step satisfy the non-negativity condition and use column scaling; 20% fall back to scalar scaling |

### Key Findings
- The rank of LoRA's cumulative update **is truly just the nominal $r$**—but as long as the subspace changes each step, it can rise to 50+ without increasing per-step parameters.
- **Small rank + ScaLoRA > large rank + LoRA**: ScaLoRA's advantage is most pronounced under tight rank budgets (at $r=8$ it surpasses $r=32$ LoRA), making it especially valuable when memory is extremely constrained.
- The optimal scaling $\alpha^*, \beta^*$ is generally close to 1 (since $\eta$ is small), so amortized scaling is nearly lossless—key for scaling to 12B models.
- Essential difference from ReLoRA: ReLoRA is "merge + random restart + warm-up," while ScaLoRA is "merge + analytic optimal scaling + moment equivariance"—the latter is both theoretically optimal and computationally efficient.

## Highlights & Insights
- The observation that **"column scaling is one of the few moment-equivariant transformations"** is very clever—most research focuses only on the expressive power of transformations, ignoring compatibility with AdamW state. This work treats "transformation + optimizer state consistency" as a joint design goal, yielding an elegant and practical solution.
- **Theoretical guarantees are realized**: Theorem 3.2's SVD representation for the "optimal adapter" is a classic derivation, but the authors treat it as a "theoretical target" rather than a direct implementation, then use column scaling for a computationally feasible approximation + analytic global optimum, smoothly bridging theory and engineering.
- **Amortizability**: Many high-rank LoRA variants (MoRA/HiRA) impose hard high-rank constraints at every step and cannot be amortized; ScaLoRA-I achieves nearly the same effect with periodic optimal scaling, reducing overhead by $1/I$, enabling scaling to 12B-parameter models.
- Empirical finding that "LoRA's cumulative update rank can naturally rise to 50+ and then plateau" **inversely validates the original LoRA hypothesis**—the optimal fine-tuning update indeed lies on a manifold much higher than $r$ but still finite.

## Limitations & Future Work
- **Extra storage**: Must save the merged $W_t$ rather than just $A_t, B_t$; deployment cannot ship only the adapter as in LoRA. The paper claims disk is not a bottleneck, but this is a limitation in strict adapter-only deployment scenarios (e.g., multi-task shared base).
- Computational complexity $O(mnr)$ is similar to HiRA and a constant factor higher than vanilla LoRA; although ScaLoRA-I amortizes this, every $I$ steps still require $O(mnr)$ operations like gradient times $B$/$A$.
- The assumption $\text{rank}(\nabla \ell(W_t)) \geq 2r$ holds empirically almost everywhere, but may not for extremely small batch sizes or very high $r$.
- Only validated on NLU/commonsense/math LLM tasks; not tested on multimodal/vision/RL fine-tuning.
- No discussion of combining with QLoRA (quantization) or DoRA (magnitude-direction decomposition); theoretically, ScaLoRA's scaling-merge mechanism could be applied to these variants, but the paper does not explore this.
- Scaling interval $I$ is a hyperparameter and needs tuning; adaptive strategies could be considered (e.g., only scale when $\alpha^*$ deviates from 1 beyond a threshold).

## Related Work & Insights
- **vs LoRA (Hu et al. 2022)**: Baseline; ScaLoRA uses the same $A, B$ parameterization but changes subspace each step for high-rank accumulation.
- **vs ReLoRA (Lialin et al. 2024)**: Also uses "merge old adapter + learn new adapter," but ReLoRA resets the optimizer; ScaLoRA uses analytic optimal scaling + moment equivariance, avoiding reset.
- **vs MoRA (Jiang et al. 2024)**: MoRA replaces $A B^\top$ with $f_{decompress}(M f_{compress}(\cdot))$ for high rank, requiring careful manual design; ScaLoRA retains LoRA's simple structure.
- **vs HiRA (Huang et al. 2025)**: HiRA uses $(AB^\top) \odot W^{pre}$ (Hadamard product) for high rank but with $O(mn)$ memory; ScaLoRA uses $O((m+n+r)r)$ memory.
- **vs LoRA-GA (Wang et al. 2024)**: Theorem 3.2 reveals that LoRA-GA is actually a special case of this paper's optimal condition at $t=0$ with $P_0 = Q_0 = I_r$ (sufficient but not necessary).
- **vs Flora / FourierFT etc.**: Methods based on structural modification/random projection are orthogonal to this work and may be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Choosing the transformation as AdamW moment-equivariant column scaling + analytic optimum" is a unique and elegant design, naturally combining theory and engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 LLM scales (DeBERTa 184M → Gemma3 12B), 3 task types (GLUE/commonsense/math), 5+ LoRA variant baselines, plus synthetic data visualization and hypothesis validation figures—extremely thorough.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear but dense; tables are highly informative but somewhat crowded.
- Value: ⭐⭐⭐⭐⭐ LoRA is the de facto standard for LLM fine-tuning; achieving stable improvements over LoRA without increasing memory or sacrificing simplicity is a highly practical advance and will be rapidly and widely adopted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](../../ACL2026/model_compression/polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](../../ICLR2026/model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](../../NeurIPS2025/model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[NeurIPS 2025\] Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation](../../NeurIPS2025/model_compression/beyond_higher_rank_token-wise_input-output_projections_for_efficient_low-rank_ad.md)

</div>

<!-- RELATED:END -->
