---
title: >-
  [Paper Note] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization
description: >-
  [ICML 2026][Optimization][Preconditioning] Based on the "row-block diagonally dominant" structure of Transformer layer-wise Hessians…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Preconditioning"
  - "Muon"
  - "Newton-Schulz"
  - "Row normalization"
  - "Transformer Hessian"
date: 2026-05-08
content_hash: ae2d7e67102b583c
---

# RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization

**Conference**: ICML 2026  
**arXiv**: [2603.20527](https://arxiv.org/abs/2603.20527)  
**Code**: The paper mentions "Our code is available at this link"  
**Area**: Optimization Algorithms / LLM Pre-training  
**Keywords**: Preconditioning, Muon, Newton-Schulz, Row normalization, Transformer Hessian

## TL;DR
Based on the "row-block diagonally dominant" structure of Transformer layer-wise Hessians, this paper replaces the expensive Newton-Schulz orthogonalization in the Muon optimizer with a single row-wise $\ell_2$ normalization. This reduces the per-step preconditioning complexity from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$, achieving a 13–44× wall-clock speedup on GPT-2 / LLaMA pre-training, while maintaining or slightly improving perplexity (ppl).

## Background & Motivation
**Background**: Diagonal preconditioners like Adam/AdamW are inexpensive but ignore parameter correlations. Methods like K-FAC and Shampoo use Kronecker-factored structures to capture matrix-level curvature. Recently, Muon has become a strong competitor to AdamW by using Newton-Schulz iterations $D_t \approx (V_tV_t^\top)^{-1/2}V_t$ to implicitly approximate $H^{-1}$ without explicit inversion.

**Limitations of Prior Work**: Muon requires five matrix multiplications for the Newton-Schulz polynomial approximation per step, with a complexity of $\mathcal{O}(mn\min(m,n))$. For wide matrices (where both $m$ and $n$ are large), this overhead becomes a significant training bottleneck—for GPT-2 1.5B, preconditioning alone takes 36.65 seconds every 100 steps.

**Key Challenge**: Muon is designed for "full spectral renormalization of $V_tV_t^\top$," but recent work (Zhang et al., Dong et al.) found that Transformer layer-wise Hessians are actually row-block diagonally dominant—meaning only diagonal blocks (intra-row interactions) are significant, while inter-row interactions are negligible. This implies Muon spends excessive computation fitting a structure that is nearly diagonal.

**Goal**: Construct an equivalent approximation with the same complexity as Muon but retaining only row-level diagonal blocks, thereby reducing the complexity to linear without sacrificing optimization quality.

**Key Insight**: Starting from the K-FAC form $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$, the authors assume that only the diagonal elements $\operatorname{diag}(V_tV_t^\top)$ need to be preserved. They verify this hypothesis by measuring the "diagonal-to-off-diagonal magnitude ratios" $r_{\min}, r_{\text{avg}}, r_{\max}$ of the Gram matrix $V_tV_t^\top$ during Transformer training, which consistently remain $>1$ and increase with model size.

**Core Idea**: Replace Newton-Schulz $(V_tV_t^\top)^{-1/2}V_t$ with a simple "row vector divided by its row $\ell_2$ norm"—equivalent to using $(\operatorname{diag}(V_tV_t^\top))^{-1/2}\otimes I_n$ as a preconditioner, which exactly corresponds to the row-block diagonal approximation of the Hessian.

## Method

### Overall Architecture
RMNP follows an algorithmic skeleton nearly identical to Muon: per step (i) compute mini-batch gradient $G_t=\nabla f(W_t;\xi^t)$; (ii) maintain first-order momentum $V_t=\beta V_{t-1}+(1-\beta)G_t$; (iii) precondition to obtain descent direction $D_t$; (iv) update $W_{t+1}=W_t-\eta_t D_t$. The only difference lies in step (iii): while Muon uses five Newton-Schulz iterations $D_t=\operatorname{NS}_5(V_t)\approx(V_tV_t^\top)^{-1/2}V_t$, RMNP uses $D_t=\operatorname{RN}(V_t)=(\operatorname{diag}(V_tV_t^\top))^{-1/2}V_t$, i.e., $V_{t,i:}/\|V_{t,i:}\|_2$. RMNP adopts Muon’s hybrid strategy—applying RMNP to matrix parameters and AdamW to non-matrix parameters (embeddings, biases, normalization layers), with separate learning rates $\text{lr}_{\text{AdamW}}$ and $\text{lr}_{\text{Matrix}}$.

### Key Designs

1. **Row-level $\ell_2$ Normalization Preconditioner**:

    - **Function**: Achieves the scaling effect of Hessian diagonal blocks through a single row-wise norm division operation with $\mathcal{O}(mn)$ complexity.
    - **Mechanism**: Starting from $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$, discarding all non-diagonal blocks yields $H_{\text{RMNP}}=(\operatorname{diag}(V_tV_t^\top))^{1/2}\otimes I_n$. Its inverse preconditioning on $V_t$ results in $[\ldots]_{i,:}=V_{t,i:}/\sqrt{(V_tV_t^\top)_{ii}}=V_{t,i:}/\|V_{t,i:}\|_2$, which is standard row $\ell_2$ normalization. The implementation consists of only three operations (row-wise sum-of-squares, square root, and division) without matrix multiplications.
    - **Design Motivation**: Completely eliminates the $\mathcal{O}(mn\cdot\min(m,n))$ bottleneck in Muon while retaining matrix-level adaptivity (row-wise rather than element-wise). It aligns with row-normalized optimizers in the LMO framework (SRON, SCALE, SWAN, Mano, MOGA) but is derived from Hessian structure rather than worst-case norm analysis.

2. **Validation of Hessian Row-Block Dominance**:

    - **Function**: Transforms the "equivalence of Newton-Schulz and row-normalization" from an intuition into a measurable empirical phenomenon.
    - **Mechanism**: The authors define the row-wise ratio $r_i \triangleq (V_tV_t^\top)_{ii}/(\frac{1}{m-1}\sum_{j\ne i}|(V_tV_t^\top)_{ij}|)$ for the Gram matrix $V_tV_t^\top$, aggregated as $r_{\text{avg}}, r_{\min}, r_{\max}$. Tracking these during the training of GPT-2 and LLaMA variants reveals that these metrics stabilize in the $>1$ region after warmup, with diagonal dominance becoming more pronounced as the model size increases.
    - **Design Motivation**: Traditional steepest-descent/LMO analyses only provide worst-case guarantees and cannot explain "why this specific norm is beneficial for neural networks." This empirical study of the Hessian provides structural justification from the actual loss landscape.

3. **Geometric Matching Proof for Non-convex Convergence**:

    - **Function**: Provides $\mathcal{O}(\epsilon^{-4})$ complexity (matching Muon's best theory) under three smoothness and convergence criterion combinations.
    - **Mechanism**: Mixed norms $\|W\|_{1,2}=\sum_i\|W_{i,:}\|_2$ and $\|W\|_{\infty,2}=\max_i \|W_{i,:}\|_2$ are used, satisfying $|\langle A,B\rangle|\le \|A\|_{1,2}\|B\|_{\infty,2}$. Theorem 5.5 yields $\mathcal{O}(m^2 L_F\sigma^2\Delta\epsilon^{-4})$ under Frobenius-Lipschitz using $\|\nabla f\|_F$. Theorem 5.7 uses $\|\nabla f\|_{1,2}$ with the same complexity. Theorem 5.9, under $L_{\infty,2}$-smoothness, yields $\mathcal{O}(mL_{\infty,2}\sigma^2\Delta\epsilon^{-4})$ with $\mathcal{O}(m)$ dimension dependence—consistent with Muon's optimal complexity under nuclear norm smoothness, reaching the minimax lower bound for non-convex stochastic optimization.
    - **Design Motivation**: Proves that RMNP maintains accuracy at the same theoretical scale as Muon. The use of $\|\cdot\|_{\infty,2}$ smoothness aligns geometrically with the row-normalization of RMNP.

### Loss & Training
Standard CE loss for LLM pre-training. Optimizer: cosine annealing schedule + 10% warmup; AdamW part uses $\beta=(0.9, 0.95)$ and weight decay of 0.1; $\text{lr}_{\text{Matrix}}$ is searched separately. RMNP is applied only to matrix parameters; embedding, lm-head, biases, and layer-norm use AdamW.

## Key Experimental Results

### Main Results

| Model | Data | Muon ppl | RMNP ppl | RMNP relative to AdamW (Gain) |
|------|------|----------|----------|-----------------|
| GPT-2 Small (125M) | OpenWebText 5B tok | -- | $\Delta$=-0.04 | -1.37 |
| GPT-2 Medium (355M) | OpenWebText 10B tok | -- | -0.07 | -1.49 |
| GPT-2 Large (770M) | OpenWebText 20B tok | -- | -0.24 | -0.84 |
| LLaMA-60M | C4 1B tok | -- | -0.63 | -4.33 |
| LLaMA-130M | C4 2B tok | -- | -0.28 | -1.10 |
| LLaMA-350M | C4 6B tok | -- | -0.02 | -- |

**Preconditioning wall-clock time (100 steps, single RTX Pro 6000, batch 16)**

| Model Scale | Muon (s) | RMNP (s) | Speedup (Gain) |
|----------|----------|----------|------|
| 60M | 1.480 | 0.115 | 12.9× |
| 125M | 2.975 | 0.201 | 14.8× |
| 355M | 7.380 | 0.401 | 18.4× |
| 770M | 27.070 | 0.611 | 44.3× |
| 1.3B | 30.570 | 0.783 | 39.0× |
| 1.5B | 36.650 | 0.855 | 42.9× |

### Ablation Study

| Configuration | Observation | Explanation |
|------|------|------|
| Full RMNP (Row $\ell_2$) | ppl comparable or lower than Muon | Main Result |
| Diagonal Dominance Metric $r_i$ | $r_{\min}>1$ throughout training | Row-block diagonal dominance holds |
| Model Scaling (60M→1.5B) | $r_{\text{avg}}, r_{\max}$ continue to rise | Larger models are more diagonal; RMNP is more justified |
| 2× Training Budget | Advantage maintained | RMNP is not just faster in early stages |
| Applied to LM-head / Embedding | See D.4 | Offers further efficiency potential |

### Key Findings
- The complexity gap widens with model size: at 60M, Muon preconditioning takes 1.48s (RMNP is 12.9× faster); at 1.5B, Muon rises to 36.65s while RMNP remains < 1s (42.9× faster). Newton-Schulz becomes a real bottleneck for end-to-end training in models $\ge 1$B.
- Ppl was maintained and even slightly improved across most scales compared to Muon, suggesting that the "cross-row" corrections in Newton-Schulz might be ineffective or lead to harmful overfitting for Transformers.
- Three Theorems demonstrate that RMNP achieves $\mathcal{O}(m)$ dimensional complexity under $L_{\infty,2}$ smoothness, matching Muon's nuclear norm analysis in a dual setting.

## Highlights & Insights
- Optimizer design is guided by Hessian structure—instead of worst-case norms, it leverages the actual architecture of neural networks.
- Row $\ell_2$ normalization replaces dozens of lines of Newton-Schulz code with only two lines, offering near-zero adoption cost.
- The theoretical section provides a unified convergence analysis across three norm combinations, specifically the geometric matching between $L_{\infty,2}$ smoothness and the $L_{1,2}$ criterion.
- Metrics like $r_{\min}, r_{\text{avg}}, r_{\max}$ for row-wise diagonal dominance can be used as diagnostic tools to determine whether to use row-norm optimizers on other architectures.

## Limitations & Future Work
- Experiments primarily focus on GPT-2 and small LLaMA variants (up to 1.5B); validation on mainstream 70B+ models is pending. Whether the geometric assumptions hold for MoE or Mamba architectures is unknown.
- Row-block diagonal dominance is a "Transformer phenomenon"; whether it applies to CNN/GNN Hessians remains to be seen.
- Experiments are restricted to pre-training and do not cover post-training stages like SFT/RLHF.
- The optimal normalization axis for non-square matrices with extremely large row counts (e.g., embedding / LM-head) is not yet fully determined, though preliminary ablations are provided.

## Related Work & Insights
- **vs Muon**: Shares the philosophy of matrix-level adaptivity but explicitly utilizes Transformer Hessian structure to reduce complexity from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$.
- **vs Shampoo / K-FAC**: Both are diagonal block approximations of Kronecker factorization requiring explicit construct and inversion; RMNP bypasses this via implicit momentum statistics.
- **vs SCALE / SWAN / Mano / MOGA**: These also use row/column normalization but are derived from the worst-case perspective of LMO/steepest descent; RMNP is derived from Hessian structure and offers the first non-convex convergence proof comparable to Muon.
- **Insight**: For other optimizers that are complex but structurally redundant (e.g., Shampoo), empirical tools like measuring Hessian/Gram density can help identify inexpensive equivalents.

## Rating
- **Novelty**: ⭐⭐⭐⭐
- **Experimental Thoroughness**: ⭐⭐⭐⭐
- **Writing Quality**: ⭐⭐⭐⭐
- **Value**: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[ICML 2026\] Taming the Loss Landscape of PINNs with Noisy Feynman-Kac Supervision: Operator Preconditioning and Non-Asymptotic Error Bounds](taming_the_loss_landscape_of_pinns_with_noisy_feynman-kac_supervision_operator_p.md)
- [\[ICML 2026\] Delayed Momentum Aggregation: Communication-efficient Byzantine-robust Federated Learning with Partial Participation](delayed_momentum_aggregation_communication-efficient_byzantine-robust_federated_.md)

</div>

<!-- RELATED:END -->
