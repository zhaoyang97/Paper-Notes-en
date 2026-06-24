---
title: >-
  [Paper Note] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning
description: >-
  [ICML 2026][Model Compression][LoRA] The authors prove that LoRA's cumulative updates are trapped in a fixed low-rank subspace and propose ScaLoRA: after merging the old $AB^\top$ into $W^{pt}$ at each step, the adapter is restarted using an **analytically derived optimal "column scaling"**. This allows the first and second moments of AdamW to be transferred equivariantly in $O((m+n)r)$ (eliminating the need for resets or warm-ups), enabling cumulative updates to naturally ac…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "LoRA"
  - "High-Rank Update"
  - "Column Scaling"
  - "AdamW moment equivariance"
  - "ScaLoRA"
date: 2026-05-08
content_hash: 9de821f8b11bd225
---

# ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2510.23818](https://arxiv.org/abs/2510.23818)  
**Code**: Not explicitly stated (None)  
**Area**: Model Compression / Parameter-Efficient Fine-Tuning / LoRA Variants  
**Keywords**: LoRA, High-Rank Update, Column Scaling, AdamW moment equivariance, ScaLoRA

## TL;DR
The authors prove that LoRA's cumulative updates are trapped in a fixed low-rank subspace and propose ScaLoRA: after merging the old $AB^\top$ into $W^{pt}$ at each step, the adapter is restarted using an **analytically derived optimal "column scaling"**. This allows the first and second moments of AdamW to be transferred equivariantly in $O((m+n)r)$ (eliminating the need for resets or warm-ups), enabling cumulative updates to naturally achieve high rank. ScaLoRA consistently outperforms LoRA, MoRA, HiRA, ReLoRA, and LoRA-GA on DeBERTaV3, LLaMA2-7B, LLaMA3-8B, and Gemma3-12B.

## Background & Motivation

**Background**: LoRA (Hu et al. 2022) constrains updates in full parameters $W = W^{pt} + AB^\top$ to two slim matrices $A \in \mathbb{R}^{m \times r}$ and $B \in \mathbb{R}^{n \times r}$, where $r \ll m, n$, significantly saving memory and computation. Subsequent variants like DoRA, QLoRA, FourierFT, HiRA, MoRA, and ReLoRA attempt to improve performance or extend applications.

**Limitations of Prior Work**: There is an **inherent performance gap** between LoRA and full fine-tuning, which worsens as the rank $r$ decreases. Fundamentally, the cumulative update over $T$ steps, $\sum_t \Delta W_t = A_T B_T^\top - A_0 B_0^\top = A_T B_T^\top$, always resides in a fixed rank-$r$ subspace—step-wise information is neutralized by "telescoping." Existing "high-rank LoRA" solutions have various issues:

- **ReLoRA**: Periodically merges $AB^\top$ into $W^{pt}$ and reinitializes a new $AB^\top$, but every merge requires **restarting the optimizer and re-performing learning-rate warm-up**, leading to slow convergence.
- **MoRA**: Replaces $A(B^\top X)$ with a non-linear mapping $f_{decompress}(M f_{compress}(X))$ to achieve high rank, but the design of $f_{compress/decompress}$ is highly labor-intensive.
- **HiRA**: Uses the Hadamard product $W^{ft} = (AB^\top) \odot W^{pre}$ to achieve high rank, but backpropagation involves an $m \times n$ Hadamard product, resulting in **$O(mn)$ memory complexity**, which is not scalable for large LLMs.

**Key Challenge**: To achieve "high-rank cumulative updates using low-rank adapters," a different subspace is needed at each step. However, changing subspaces typically invalidates the $(m_t, v_t)$ moment estimators maintained by AdamW, necessitating either a slow restart or an expensive recalculation from scratch. These two requirements appear incompatible.

**Goal**: Find an analytic expression for the "optimal adapter update" and a transformation form that allows moment estimators to be mapped equivariantly from the old adapter to the new one in $O((m+n)r)$ without restarts, ultimately achieving high-rank cumulative updates and fast convergence without increasing memory.

**Key Insight**: Starting from the Lipschitz upper bound of the loss, the authors prove that the optimal adapter at each step is "equivalent to performing a truncated SVD of the full fine-tuning gradient $\nabla \ell(W_t) = U_t \Sigma_t V_t^\top$ and selecting the top $2r$ directions." Since the complexity of SVD is too high, they further constrain the "relationship between adapters before and after replacement" to a simple **column scaling** transformation: $\tilde{A} = A \cdot \text{diag}(\alpha), \tilde{B} = B \cdot \text{diag}(\beta)$. This is one of the few transformations that allow for the analytic equivariant migration of AdamW moments.

**Core Idea**: Search for column scaling factors (yielding an analytic global optimal solution) that are "optimal for the current loss descent" within the LoRA subspace. Every step or every $I$ steps, after scaling with the optimal $(\alpha^*, \beta^*)$, the product $\tilde{A}_t \tilde{B}_t^\top$ is merged into the frozen $\tilde{W}^{pt}_t$, and training continues with new $A_{t+1}, B_{t+1}$. Column scaling allows moments to migrate equivariantly almost for free, causing the cumulative updates to span an increasing number of different directions and naturally increasing the rank.

## Method

### Overall Architecture
ScaLoRA aims to accumulate high-rank updates using low-rank adapters without restarting the optimizer. It is still formulated as $W_t = W^{pt} + A_t B_t^\top$. However, at each step (or every $I$ steps), the current adapter is "virtually merged and restarted" as $W_t = \underbrace{(W^{pt} + A_t B_t^\top - \tilde{A}_t \tilde{B}_t^\top)}_{\tilde{W}^{pt}_t,\,\text{merged and frozen}} + \underbrace{\tilde{A}_t \tilde{B}_t^\top}_{\text{learnable}}$. First, the optimal "column scaling" $(\alpha^*_t, \beta^*_t)$ is calculated via an analytic formula. The old subspace $A_t B_t^\top$ is merged into the frozen part $\tilde{W}^{pt}_t$, and the new learnable part is replaced with $\tilde{A}_t = A_t \text{diag}(\alpha^*_t)$ and $\tilde{B}_t = B_t \text{diag}(\beta^*_t)$. The inherent equivariance of column scaling is then used to transfer AdamW momentum from the old $(A_t, B_t)$ to the new $(\tilde{A}_t, \tilde{B}_t)$. Finally, a standard GD/AdamW step is taken to obtain $A_{t+1}, B_{t+1}$ for the next round. Since each round falls into a different optimal subspace, the cumulative weights $\sum_t (A_{t+1} B_{t+1}^\top - \tilde{A}_t \tilde{B}_t^\top)$ over $T$ rounds no longer telescope like vanilla LoRA, resulting in a continuous increase in rank.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input: W_t = W^pt + A_t·B_tᵀ<br/>Current low-rank adapter (A_t, B_t)"]
    A --> T["Theoretical Target: Optimal adapter ↔ gradient top-2r singular space<br/>(Thm 3.2, SVD too expensive, used as upper bound)"]
    T --> B["Compute optimal column scaling (α*, β*)<br/>Quadratic global closed-form solution (Thm 3.7)"]
    B -->|Non-negative solution · ~80% of layers| C["Column Scaling<br/>Ã = A·diag(α*), B̃ = B·diag(β*)"]
    B -->|Otherwise · ~20% of layers| C2["Degenerate to Scalar Scaling<br/>Ã = α*·A, B̃ = β*·B (Thm 3.5)"]
    C --> D["Merge old subspace and freeze<br/>W̃^pt = W^pt + A_t·B_tᵀ − Ã_t·B̃_tᵀ"]
    C2 --> D
    D --> E["AdamW Momentum Equivariant Migration<br/>m multiplied by α, v multiplied by α² column-wise, O((m+n)r) (Lemma 3.6/3.3)"]
    E --> F["Perform standard AdamW step → A_{t+1}, B_{t+1}"]
    F -->|Repeat every step (ScaLoRA) or every I steps (ScaLoRA-I)| A
    F --> G["Cumulative updates over T rounds span multiple subspaces<br/>No longer neutralized by telescoping → Rank continuously increases (High-Rank Update)"]
```

### Key Designs

**1. Theoretical Characterization of the Optimal Adapter: Finding the "Theoretical Target"**

What determines the persistent gap between LoRA and full FT? Starting from the $L$-smooth upper bound $\ell(W_t + \Delta W_t) \leq \ell(W_t) + \langle \nabla \ell, \Delta W_t \rangle + \frac{L}{2}\|\Delta W_t\|_F^2$, the authors minimize the right side to find the ideal full-FT update $\Delta W_t^* = -\frac{1}{L} \nabla \ell(W_t)$. Expanding the single-step LoRA update as $\Delta \tilde{W}_t = -\eta \nabla \ell\, \tilde{B}_t \tilde{B}_t^\top - \eta \tilde{A}_t \tilde{A}_t^\top \nabla \ell + O(\eta^2)$, the problem becomes equivalent to approximating $\Delta W_t^*$ with a low-rank $\Delta \tilde{W}_t$, i.e., minimizing $\|\Delta W_t^* - \Delta \tilde{W}_t\|_F^2$. Theorem 3.2 proves that when $\text{rank}(\nabla \ell(W_t)) \geq 2r$, the optimal $\tilde{A}_t^*, \tilde{B}_t^*$ are precisely equivalent to the top-$2r$ left and right singular vectors obtained from a rank-$2r$ truncated SVD of $\nabla \ell$. This links the "optimal adapter ↔ gradient top-$2r$ singular space," showing that the distance between LoRA and full FT is essentially determined by the principal singular directions of the current gradient. However, truncated SVD at every step costs $O(Smnr)$ and requires optimizer restarts, making it too expensive. It serves only as a theoretical upper bound; a much cheaper approximation is needed for implementation.

**2. Optimal Column Scaling + AdamW Momentum Equivariance: Compressing "Adapter Switching" into Free Column Scaling**

The high cost of direct SVD is due to "arbitrary subspace switching" invalidating the momentum $(m_t, v_t)$ of AdamW. The authors narrow the search space to column scaling $\tilde{A} = A \text{diag}(\alpha), \tilde{B} = B \text{diag}(\beta)$—one of the few transformations where moments map analytically. Under this constraint, the loss upper bound becomes a quadratic form of $(\alpha, \beta)$: $\|\frac{1}{L}\nabla\ell - \eta \nabla\ell\, B \text{diag}^2(\beta) B^\top - \eta A \text{diag}^2(\alpha) A^\top \nabla\ell\|_F^2$. Theorem 3.7 proves that if the linear system $[(S_t^{A\top} S_t^A) \odot (S_t^{B\top} S_t^B)]\, v_t = \lambda_t$ has a non-negative solution (where $S_t^A, S_t^B$ are small matrices computed from the gradient and adapter, satisfied by ~80% of layers in LLMs), the global optimum is the closed-form $[\alpha^*_t; \beta^*_t] = \pm \frac{1}{\sqrt{L\eta}} v_t^{\circ 1/2}$ in $O((m+n)r^2)$. If the non-negativity condition is not met, it degenerates to scalar scaling (Theorem 3.5), which also has an analytic global optimum. Momentum equivariance is straightforward: since $\tilde{A} = A \text{diag}(\alpha)$ is a column-wise scaling, and AdamW moments correspond element-wise to the adapter, $m$ is scaled by $\alpha$ and $v$ by $\alpha^2$ column-wise in $O((m+n)r)$, without needing restarts or warm-ups (Lemma 3.6). Row scaling or multiplication by full-rank matrices cannot achieve this—justifying the choice of column scaling. In practice, the Lipschitz constant $L$ is treated as a hyperparameter and found via grid search.

**3. ScaLoRA and the Amortized Variant ScaLoRA-I: A Practical Algorithm for 12B Models**

Combining these elements yields the complete algorithm: at each step, if the non-negativity condition of Theorem 3.7 holds, apply column scaling $\tilde{A}_t = A_t \text{diag}(\alpha^*_t), \tilde{B}_t = B_t \text{diag}(\beta^*_t)$ with Lemma 3.6 for momentum; otherwise, use scalar scaling $\tilde{A}_t = \alpha^*_t A_t, \tilde{B}_t = \beta^*_t B_t$ with Lemma 3.3. Merging $A_t B_t^\top - \tilde{A}_t \tilde{B}_t^\top$ is an in-place write-back to $W^{pt}$. Extra space is only $O((m+n+r)r)$, and total time is $O(mnr + (m+n+r)r^2)$ (the latter is negligible for small $r$). Since the overhead of scaling every matrix in every step across hundreds of layers in an LLM is non-trivial, the amortized version **ScaLoRA-I** is introduced: scaling and merging occur every $I$ steps, reducing per-step overhead to $1/I$. Because the learning rate $\eta$ is small and the optimal scaling is close to 1, frequent scaling has diminishing returns; $I=10$ results in almost no performance loss. This is a key differentiator: MoRA/HiRA enforce high-rank constraints at every step and cannot be amortized, whereas ScaLoRA's "periodic optimal scaling" is amortizable, allowing it to scale to models like Gemma3-12B.

### Loss & Training
The training loss of the LLM remains unchanged (standard task CE / language modeling loss). Only the optimization logic for the LoRA modules is modified: scaling-merging operations are inserted before and after AdamW updates. Primary hyperparameters include $L$ (selected via grid search), $\eta$, the scaling interval $I$, and the LoRA rank $r$. The paper validates $r=4$ (GLUE) and $r=8$ (LLaMA/Gemma tasks), where improvements at low ranks are most significant. The cost involves storing the merged $W_t$ instead of small adapters—not a bottleneck for disk storage, but different from the standard LoRA deployment where only adapters are shipped.

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

ScaLoRA achieves the best performance on 7 out of 8 tasks, with an average ~0.5% higher than HiRA and even exceeding Full FT (likely due to Full FT overestimating on small datasets).

**LLaMA2-7B / LLaMA3-8B Commonsense Reasoning ($r = 8$)**:

| Model | LoRA | ReLoRA | LoRA-GA | MoRA | HiRA | ScaLoRA | ScaLoRA-I | LoRA $r=32$ |
|---|---|---|---|---|---|---|---|---|
| LLaMA2-7B Avg | 73.63 | 74.40 | 74.34 | 73.82 | 73.95 | 74.51 | **74.75** | 74.52 |
| LLaMA3-8B Avg | 76.83 | 77.26 | 77.22 | 77.27 | 77.46 | **77.85** | 77.57 | 77.54 |

At $r=8$, ScaLoRA(-I) **outperforms LoRA $r=32$**—achieving better results with 1/4 of the parameters.

Consistent leadership was also observed in mathematical reasoning (MetaMathQA / GSM8K / MATH) and on Gemma3-12B.

### Ablation Study

| Configuration | Phenomenon |
|---|---|
| Full ScaLoRA | Baseline |
| No column scaling (scalar only, Thm 3.5) | Slight performance drop, but still superior to LoRA—confirming scalar scaling's contribution |
| Every step vs. every 10 steps (ScaLoRA-I) | $I=10$ is nearly lossless, confirming optimal scaling is close to 1 |
| Disabling moment equivariant transfer | Significant degradation, equivalent to ReLoRA's restart effect |
| Different ranks $r$ | ScaLoRA consistently beats LoRA across $r=4, 8, 16, 32$; the relative advantage is larger at lower ranks |
| Figure 2(b) on RTE | LoRA cumulative update rank = 4 (constant); ScaLoRA rank rises to 54 |
| Figure 2(c) | The $\text{rank}(\nabla \ell(W_t)) \geq 2r$ assumption holds almost everywhere in LLMs |
| Figure 2(d) | ~80% of layers satisfy non-negativity at each step for column scaling; 20% fallback to scalar |

### Key Findings
- The cumulative update rank of LoRA is **truly restricted to its nominal $r$**—but by switching subspaces at each step, it can rise to 50+ without increasing per-step parameters.
- **Small Rank + ScaLoRA > Large Rank + LoRA**: ScaLoRA's advantage is most significant under low-rank budgets (outperforming $r=32$ LoRA with $r=8$), making it highly valuable when memory is tight.
- Optimal scaling $\alpha^*, \beta^*$ is generally close to 1 (due to small $\eta$), so amortized scaling is nearly lossless—this is key for scaling to 12B parameters.
- Essential difference from ReLoRA: ReLoRA is "merge + random restart + warm-up," whereas ScaLoRA is "merge + analytic optimal scaling + moment equivariance"—the latter is both theoretically optimal and computationally cheap.

## Highlights & Insights
- **"Column scaling as a moment-equivariant transformation"** is a brilliant observation. Most studies focus only on the expressive power of transformations, ignoring compatibility with AdamW states. By jointly designing the "transformation + optimizer state consistency," the authors achieved an elegant and practical solution.
- **Theoretical grounding for implementation**: Theorem 3.2 provides a classic SVD representation for the "optimal adapter," but the authors treat it as a "theoretical target" rather than a direct implementation goal. They then use column scaling as a computationally feasible approximation with an analytic global optimum, effectively bridging theory and engineering.
- **Amortization**: Many high-rank variants (MoRA/HiRA) enforce high-rank constraints at every step and cannot be amortized. ScaLoRA-I achieves nearly identical effects with 1/$I$ of the overhead, allowing the move to 12B models.
- **Empirical Proof**: The finding that "cumulative update rank naturally rises to 50+ and then plateaus" **inversely proves the rationality of LoRA's original assumption**—optimal fine-tuning updates indeed reside on a manifold higher than $r$ but still finite.

## Limitations & Future Work
- **Extra Storage**: Merged $W_t$ must be saved instead of just $A_t, B_t$. It cannot be deployed like LoRA by shipping only adapters, which is a constraint in strictly adapter-only scenarios (e.g., multi-task shared bases).
- Computational complexity $O(mnr)$ is similar to HiRA and higher than vanilla LoRA by a constant factor. Even with ScaLoRA-I amortization, the $O(mnr)$ gradient-adapter product is required every $I$ steps.
- The $\text{rank}(\nabla \ell(W_t)) \geq 2r$ assumption might fail for extremely small batches or high $r$, though it holds empirically in LLMs.
- Validated only on NLU/Commonsense/Math LLM tasks; not tested on multimodal, vision, or RL fine-tuning.
- No discussion on combinations with QLoRA (quantization) or DoRA (magnitude-direction decomposition). Theoretically, ScaLoRA's mechanism can be applied to these, but it is not explored in the paper.
- The interval $I$ is a hyperparameter requiring tuning; adaptive scaling (e.g., when $\alpha^*$ deviates significantly from 1) could be considered.

## Related Work & Insights
- **vs LoRA (Hu et al. 2022)**: The base case; ScaLoRA uses the same $A, B$ parameterization but switches subspaces every step for high-rank accumulation.
- **vs ReLoRA (Lialin et al. 2024)**: Both share the "merge old + learn new" idea, but ReLoRA restarts the optimizer; ScaLoRA uses analytic optimal scaling and moment equivariance to avoid restarts.
- **vs MoRA (Jiang et al. 2024)**: MoRA achieves high rank via non-linear mapping $f_{decompress}(M f_{compress}(\cdot))$, requiring careful manual design; ScaLoRA maintains LoRA's simple structure.
- **vs HiRA (Huang et al. 2025)**: HiRA uses Hadamard products $(AB^\top) \odot W^{pre}$ for high rank but requires $O(mn)$ memory; ScaLoRA stays at $O((m+n+r)r)$.
- **vs LoRA-GA (Wang et al. 2024)**: Theorem 3.2 reveals that LoRA-GA is effectively a special case of the optimal condition in this paper at $t=0$ with $P_0 = Q_0 = I_r$.
- **vs Flora / FourierFT, e.g.**: These structural modification/random projection approaches are orthogonal and potentially stackable with ScaLoRA.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Selecting column scaling as an AdamW moment-equivariant transformation + analytic optimal solution" is a unique and elegant design, naturally blending theory and engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 LLM scales (184M to 12B), 3 task categories (GLUE/Commonsense/Math), and 5+ LoRA baselines, supplemented by synthetic data and hypothesis-testing figures. Extremely thorough.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear but dense; tables are information-rich but slightly crowded.
- Value: ⭐⭐⭐⭐⭐ Given that LoRA is the de facto standard for LLM fine-tuning, a method that stably outperforms it while maintaining simplicity and memory efficiency is a significant and practical advance likely to be widely adopted.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Energy-Structured Low-Rank Adaptation for Continual Learning](energy-structured_low-rank_adaptation_for_continual_learning.md)
- [\[ACL 2026\] Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions](../../ACL2026/model_compression/polynomial_expansion_rank_adaptation_enhancing_low-rank_fine-tuning_with_high-or.md)
- [\[ICLR 2026\] LoFT: Low-Rank Adaptation That Behaves Like Full Fine-Tuning](../../ICLR2026/model_compression/loft_low-rank_adaptation_that_behaves_like_full_fine-tuning.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[ICML 2026\] Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters](finer_parameter_steps_for_low-rank_peft_a_controlled_study_with_cp_tensor_adapte.md)

</div>

<!-- RELATED:END -->
