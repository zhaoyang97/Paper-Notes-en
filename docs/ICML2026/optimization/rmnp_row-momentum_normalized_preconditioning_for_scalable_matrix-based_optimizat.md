---
title: >-
  [Paper Note] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization
description: >-
  [ICML 2026][Optimization][Preconditioning] Based on the "row block diagonal dominance" structure of the Transformer layer Hessian, this work replaces the expensive Newton-Schulz orthogonalization in the Muon optimizer wi…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Preconditioning"
  - "Muon"
  - "Newton-Schulz"
  - "Row Normalization"
  - "Transformer Hessian"
date: 2026-05-08
content_hash: db06d8ea10b3b3a3
---

# RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization

**Conference**: ICML 2026  
**arXiv**: [2603.20527](https://arxiv.org/abs/2603.20527)  
**Code**: The main text states "Our code is available at this link"  
**Area**: Optimization Algorithms / LLM Pretraining  
**Keywords**: Preconditioning, Muon, Newton-Schulz, Row Normalization, Transformer Hessian

## TL;DR
Based on the "row block diagonal dominance" structure of the Transformer layer Hessian, this work replaces the expensive Newton-Schulz orthogonalization in the Muon optimizer with a single row-wise $\ell_2$ normalization, reducing per-step preconditioning complexity from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$. On GPT-2 / LLaMA pretraining, this yields a 13–44× wall-clock speedup, with perplexity not only maintained but slightly improved.

## Background & Motivation
**Background**: Diagonal preconditioners like Adam/AdamW are cheap but ignore parameter correlations; K-FAC, Shampoo, etc., use Kronecker factorization to capture matrix-level curvature; the recent Muon optimizer uses Newton-Schulz iteration $D_t \approx (V_tV_t^\top)^{-1/2}V_t$ to implicitly realize $H^{-1}$ without explicit inversion, and has become a strong competitor to AdamW in large model pretraining.

**Limitations of Prior Work**: Muon requires five matrix multiplications per step for Newton-Schulz polynomial approximation, with complexity $\mathcal{O}(mn\min(m,n))$. For wide matrices (both $m,n$ large), this quickly becomes a training bottleneck—on GPT-2 1.5B, preconditioning alone takes 36.65 seconds per 100 steps.

**Key Challenge**: Muon is designed for "full-spectrum reconditioning" of $V_tV_t^\top$, but recent works (Zhang et al., Dong et al.) find that the Transformer layer Hessian is actually row block diagonal dominant—only diagonal blocks (intra-row interactions) are significant, while inter-row interactions are negligible. Thus, Muon expends significant computation fitting a structure that is "almost diagonal."

**Goal**: Construct an equivalent approximation to Muon with the same complexity, but retaining only row-level diagonal blocks, thereby reducing complexity to linear without sacrificing optimization quality.

**Key Insight**: Starting from the K-FAC form $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$, the authors hypothesize that only the diagonal elements $\operatorname{diag}(V_tV_t^\top)$ are needed. Empirically, during Transformer training, the "diagonal/non-diagonal magnitude ratio" $r_{\min},r_{\text{avg}},r_{\max}$ of the Gram matrix $V_tV_t^\top$ remains >1 and increases with model size, validating this hypothesis.

**Core Idea**: Replace Newton-Schulz $(V_tV_t^\top)^{-1/2}V_t$ with simple "row vector divided by its row $\ell_2$ norm"—equivalent to using $(\operatorname{diag}(V_tV_t^\top))^{-1/2}\otimes I_n$ as the preconditioner, corresponding exactly to the row block diagonal approximation of the Hessian.

## Method

### Overall Architecture
RMNP and Muon share almost identical algorithmic skeletons: each step (i) compute mini-batch gradient $G_t=\nabla f(W_t;\xi^t)$; (ii) maintain first-order momentum $V_t=\beta V_{t-1}+(1-\beta)G_t$; (iii) precondition to obtain descent direction $D_t$; (iv) update $W_{t+1}=W_t-\eta_t D_t$. The only difference is in step (iii): Muon uses 5-step Newton-Schulz iteration $D_t=\operatorname{NS}_5(V_t)\approx(V_tV_t^\top)^{-1/2}V_t$; RMNP uses $D_t=\operatorname{RN}(V_t)=(\operatorname{diag}(V_tV_t^\top))^{-1/2}V_t$, i.e., directly normalizing each row $V_{t,i:}$ as $V_{t,i:}/\|V_{t,i:}\|_2$. Overall, RMNP follows Muon's hybrid strategy—matrix parameters use RMNP, non-matrix parameters (embedding/biases/norm) continue with AdamW, with separate learning rates $\text{lr}_{\text{AdamW}}$ and $\text{lr}_{\text{Matrix}}$.

### Key Designs

1. **Row-wise $\ell_2$ Normalization Preconditioner**:

    - **Function**: Implements "Hessian diagonal block scaling" preconditioning via a single $\mathcal{O}(mn)$ row-wise normalization operation.
    - **Mechanism**: Starting from $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$, discarding all off-diagonal blocks yields $H_{\text{RMNP}}=(\operatorname{diag}(V_tV_t^\top))^{1/2}\otimes I_n$; its inverse preconditioning on $V_t$ gives $[\ldots]_{i,:}=V_{t,i:}/\sqrt{(V_tV_t^\top)_{ii}}=V_{t,i:}/\|V_{t,i:}\|_2$, i.e., standard row $\ell_2$ normalization. The implementation is simply "row-wise sum of squares, square root, division"—no matrix multiplication.
    - **Design Motivation**: Completely eliminates the $\mathcal{O}(mn\cdot\min(m,n))$ bottleneck in Muon; retains matrix-level adaptivity (still row-wise, not element-wise); aligns with row-normalized optimizers (SRON, SCALE, SWAN, Mano, MOGA) in the LMO framework, but derived from Hessian structure rather than worst-case norm.

2. **Empirical Validation of Hessian Row Block Dominance**:

    - **Function**: Turns the "Newton-Schulz and row-normalization equivalence" from intuition into a measurable, observable empirical phenomenon.
    - **Mechanism**: For the Gram matrix $V_tV_t^\top$, define per-row ratio $r_i \triangleq (V_tV_t^\top)_{ii}/(\frac{1}{m-1}\sum_{j\ne i}|(V_tV_t^\top)_{ij}|)$, aggregate into $r_{\text{avg}},r_{\min},r_{\max}$; track these throughout training on GPT-2 Small/Medium/Large and LLaMA 60M/130M/350M. After warm-up, all three metrics stabilize above 1, and diagonal dominance becomes more pronounced with larger models (on GPT-2 Small, $\bar r_{\text{avg}}\approx 4.9, \bar r_{\max}\approx 60$).
    - **Design Motivation**: Traditional steepest-descent / LMO analysis only provides worst-case guarantees and cannot explain "why this particular norm works well for neural networks"; only by examining the actual loss landscape structure can this be answered, which the authors empirically demonstrate via the Hessian.

3. **Geometric Matching Proof of Nonconvex Convergence**:

    - **Function**: Provides $\mathcal{O}(\epsilon^{-4})$ complexity matching Muon's best existing theory under three different smoothness + convergence criteria combinations.
    - **Mechanism**: Define mixed norms $\|W\|_{1,2}=\sum_i\|W_{i,:}\|_2$ and $\|W\|_{\infty,2}=\max_i \|W_{i,:}\|_2$, which satisfy $|\langle A,B\rangle|\le \|A\|_{1,2}\|B\|_{\infty,2}$. Theorem 5.5 gives $\mathcal{O}(m^2 L_F\sigma^2\Delta\epsilon^{-4})$ complexity under Frobenius-Lipschitz and $\|\nabla f\|_F$ criterion; Theorem 5.7 gives the same $\mathcal{O}(m^2)$ under $\|\nabla f\|_{1,2}$; most crucially, Theorem 5.9 gives $\mathcal{O}(mL_{\infty,2}\sigma^2\Delta\epsilon^{-4})$ (i.e., $\mathcal{O}(m)$ dimension dependence) under $L_{\infty,2}$-smoothness—matching Muon's optimal complexity under nuclear norm smoothness and achieving the minimax lower bound for nonconvex stochastic optimization.
    - **Design Motivation**: To convince the community that "cheap does not mean inferior," it is necessary to prove that RMNP maintains accuracy at the same theoretical scale as Muon; and $\|\cdot\|_{\infty,2}$ smoothness aligns geometrically with RMNP's row normalization, which is the theoretical reason RMNP maintains accuracy.

### Loss & Training
Standard LLM pretraining CE loss. On the optimizer side: cosine annealing schedule + 10% warmup; AdamW part uses $\beta=(0.9,0.95)$, weight decay 0.1; matrix part searches for $\text{lr}_{\text{Matrix}}$ separately. RMNP is applied only to matrix parameters, while embedding / lm-head / biases / layer-norm still use AdamW.

## Key Experimental Results

### Main Results

| Model | Data | Muon ppl | RMNP ppl | RMNP vs AdamW |
|-------|------|----------|----------|--------------|
| GPT-2 Small (125M) | OpenWebText 5B tok | -- | $\Delta$=-0.04 | -1.37 |
| GPT-2 Medium (355M) | OpenWebText 10B tok | -- | -0.07 | -1.49 |
| GPT-2 Large (770M) | OpenWebText 20B tok | -- | -0.24 | -0.84 |
| LLaMA-60M | C4 1B tok | -- | -0.63 | -4.33 |
| LLaMA-130M | C4 2B tok | -- | -0.28 | -1.10 |
| LLaMA-350M | C4 6B tok | -- | -0.02 | -- |

**Preconditioning wall-clock time (100 steps, single RTX Pro 6000, batch 16)**

| Model Size | Muon (s) | RMNP (s) | Speedup |
|------------|----------|----------|---------|
| 60M | 1.480 | 0.115 | 12.9× |
| 125M | 2.975 | 0.201 | 14.8× |
| 355M | 7.380 | 0.401 | 18.4× |
| 770M | 27.070 | 0.611 | 44.3× |
| 1.3B | 30.570 | 0.783 | 39.0× |
| 1.5B | 36.650 | 0.855 | 42.9× |

### Ablation Study

| Configuration | Phenomenon | Description |
|---------------|------------|-------------|
| Full RMNP (row $\ell_2$) | ppl matches or slightly outperforms Muon | Main result |
| Only diagonal dominance metric $r_i$ | $r_{\min}>1$ throughout training | Row block diagonal dominance holds |
| Model scaling (60M→1.5B) | $r_{\text{avg}}, r_{\max}$ keep increasing | Larger models are more diagonal, RMNP more justified |
| 2× training budget | Advantage maintained | RMNP is not just faster in early stages |
| Also applied to LM-head / Embedding | See D.4 | Further efficiency potential |

### Key Findings
- The complexity gap widens with model size: for 60M, Muon preconditioning is 1.48s, RMNP is 12.9× faster; at 1.5B, Muon rises to 36.65s while RMNP remains <1s, a 42.9× speedup. For models ≥1B, Newton-Schulz is already the true end-to-end training bottleneck.
- Perplexity not only does not drop, but is slightly better than Muon at most scales—suggesting that Newton-Schulz's "cross-row" correction may be ineffective or even harmful overfitting for Transformers.
- The three theoretical theorems together show: under $\|\cdot\|_{\infty,2}$ smoothness, which "matches the algorithm's geometry," RMNP achieves $\mathcal{O}(m)$ dimension complexity, dual to Muon's nuclear norm analysis.

## Highlights & Insights
- Uses Hessian structure to guide optimizer design—not relying on worst-case norms, but on "what neural networks actually look like," a key upgrade over LMO-derived row-norm works (SCALE/SWAN/Mano/MOGA).
- Row $\ell_2$ normalization can replace dozens of lines of Newton-Schulz code in Muon with just two lines—almost zero usability cost, direct drop-in.
- Theoretical section provides unified convergence analysis under three norm combinations, especially the geometric match of $\|\cdot\|_{\infty,2}$ smoothness + $\|\cdot\|_{1,2}$ criterion, offering a template for "which norm to choose for matrix-level optimizers."
- The per-row diagonal dominance metrics $r_{\min},r_{\text{avg}},r_{\max}$ can serve as diagnostic tools for "whether to use row-norm optimizers" and be transferred to other architectures.

## Limitations & Future Work
- Experiments focus mainly on GPT-2 and small LLaMA (up to 1.5B); not yet validated on mainstream 70B+ models. Whether the geometric assumptions hold for MoE, Mamba, etc., remains unclear.
- Row block diagonal dominance is a "Transformer phenomenon"; whether the Hessian of CNNs/GNNs also exhibits this, and whether RMNP can drop-in, remains to be answered.
- Experiments are only on pretraining, not covering SFT/RLHF and other post-training stages.
- The best normalization axis for non-square, extremely large-row matrices like embedding/LM-head is not fully addressed; the appendix provides preliminary ablation but no unified recommendation.

## Related Work & Insights
- **vs Muon**: Same philosophy (matrix-level adaptivity), but RMNP explicitly leverages Transformer Hessian structure, reducing preconditioning from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$.
- **vs Shampoo / K-FAC**: Both use Kronecker factorization for diagonal block approximation, but require explicit construction and inversion of $L,R$; RMNP bypasses explicit matrices using only implicit momentum statistics.
- **vs SCALE / SWAN / Mano / MOGA**: Also row/column normalization, but prior works derive from LMO/steepest descent worst-case perspective; RMNP is derived from Hessian structure and provides the first "same order as Muon" nonconvex convergence proof.
- **Insights**: For other optimizers that are "complex to implement but structurally redundant" (e.g., Shampoo), similar empirical checks of Hessian/Gram matrix density can help find cheap equivalents.

## Rating
- Novelty: ⭐⭐⭐⭐ Links row normalization to Transformer Hessian structure, with theoretical complexity matching Muon
- Experimental Thoroughness: ⭐⭐⭐⭐ GPT-2 + LLaMA, multi-scale, preconditioning wall-clock, diagonal dominance metrics, comprehensive coverage
- Writing Quality: ⭐⭐⭐⭐ Clear algorithm diagrams, concise motivation; theory section is dense and requires careful reading
- Value: ⭐⭐⭐⭐⭐ Direct drop-in for large model pretraining, saves 13–44× preconditioning time, extremely high engineering value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[CVPR 2026\] Dynamic Momentum Recalibration in Online Gradient Learning](../../CVPR2026/optimization/dynamic_momentum_recalibration_in_online_gradient_learning.md)
- [\[AAAI 2026\] A Distributed Asynchronous Generalized Momentum Algorithm Without Delay Bounds](../../AAAI2026/optimization/a_distributed_asynchronous_generalized_momentum_algorithm_wi.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)

</div>

<!-- RELATED:END -->
