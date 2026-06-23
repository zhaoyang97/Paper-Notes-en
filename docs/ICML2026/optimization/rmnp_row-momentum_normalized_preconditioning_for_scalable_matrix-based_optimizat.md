---
title: >-
  [Paper Note] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization
description: >-
  [ICML 2026][Optimization & Theory][Preconditioning] Based on the "row-block diagonal dominant" structure of the Transformer layer-wise Hessian, this paper replaces the expensive Newton-Schulz orthogonalization in the Muon optimizer with a single row-level $\ell_2$ normalization. This reduces the per-step preconditioning complexity from $\mathcal{O}(mn\min(m,n))$ to $\ma
tags:
  - ICML 2026
  - Optimization & Theory
  - Preconditioning
  - Muon
  - Newton-Schulz
  - Transformer Hessian
date: 2026-05-08
content_hash: ccdd42ad4c2d646f
---
# RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization

**Conference**: ICML 2026  
**arXiv**: [2603.20527](https://arxiv.org/abs/2603.20527)  
**Code**: The paper mentions "Our code is available at this link"  
**Area**: Optimization Algorithms / LLM Pre-training  
**Keywords**: Preconditioning, Muon, Newton-Schulz, Row Normalization, Transformer Hessian

## TL;DR
Based on the "row-block diagonal dominant" structure of the Transformer layer-wise Hessian, this paper replaces the expensive Newton-Schulz orthogonalization in the Muon optimizer with a single row-level $\ell_2$ normalization. This reduces the per-step preconditioning complexity from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$, resulting in a 13–44× wall-clock speedup in GPT-2 / LLaMA pre-training with slightly improved perplexity.

## Background & Motivation
**Background**: Diagonal preconditioners like Adam/AdamW are computationally cheap but ignore parameter correlations. K-FAC and Shampoo use Kronecker decomposition to capture matrix-level curvature. Recently, Muon has become a strong competitor to AdamW in LLM pre-training by implicitly implementing $H^{-1}$ via Newton-Schulz iteration $D_t \approx (V_tV_t^\top)^{-1/2}V_t$ without explicit inversion.

**Limitations of Prior Work**: Muon requires 5 matrix multiplications for Newton-Schulz polynomial approximation per step, with a complexity of $\mathcal{O}(mn\min(m,n))$. For wide matrices (large $m, n$), this overhead becomes a training bottleneck—preconditioning alone takes 36.65 seconds every 100 steps for GPT-2 1.5B.

**Key Challenge**: Muon is designed for "full spectral renormalization" of $V_tV_t^\top$. However, recent studies (Zhang et al., Dong et al.) found that Transformer layer-wise Hessians are actually row-block diagonal dominant—interactions are significant only within diagonal blocks (intra-row), while cross-row interactions are negligible. This implies Muon consumes excessive computation to fit a structure that is "nearly diagonal."

**Goal**: Construct an equivalent approximation with the same complexity as Muon but preserving only row-level diagonal blocks, thereby reducing complexity to linear scale without losing optimization quality.

**Key Insight**: Starting from the Muon K-FAC form $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$, the authors assume only the diagonal elements $\operatorname{diag}(V_tV_t^\top)$ need to be preserved. Empirical measurements of the "diagonal-to-off-diagonal magnitude ratio" $r_{\min}, r_{\text{avg}}, r_{\max}$ of the Gram matrix $V_tV_t^\top$ during Transformer training consistently stay $> 1$ and increase with model size, validating this assumption.

**Core Idea**: Replace Newton-Schulz $(V_tV_t^\top)^{-1/2}V_t$ with a simple "row vector divided by row $\ell_2$ norm." This is equivalent to using $(\operatorname{diag}(V_tV_t^\top))^{-1/2}\otimes I_n$ as a preconditioner, corresponding to the row-block diagonal approximation of the Hessian.

## Method

### Overall Architecture
RMNP shares an almost identical algorithmic skeleton with Muon: per step (i) compute mini-batch gradient $G_t=\nabla f(W_t;\xi^t)$; (ii) maintain first-order momentum $V_t=\beta V_{t-1}+(1-\beta)G_t$; (iii) precondition to obtain descent direction $D_t$; (iv) update $W_{t+1}=W_t-\eta_t D_t$. The only difference lies in step (iii): while Muon utilizes 5 Newton-Schulz iterations $D_t=\operatorname{NS}_5(V_t)\approx(V_tV_t^\top)^{-1/2}V_t$, RMNP uses $D_t=\operatorname{RN}(V_t)=(\operatorname{diag}(V_tV_t^\top))^{-1/2}V_t$, which simplifies to $V_{t,i:}/\|V_{t,i:}\|_2$ for each row of the momentum matrix. RMNP follows Muon's hybrid strategy—applying RMNP to matrix parameters and AdamW to others (embeddings/biases/norms) with two sets of learning rates: $\text{lr}_{\text{AdamW}}$ and $\text{lr}_{\text{Matrix}}$.

### Key Designs

**1. Row-level $\ell_2$ Normalization Preconditioner: Equivalent Scaling via Row Norms**

The bottleneck of Muon is its design for "full spectral renormalization," requiring $\mathcal{O}(mn\min(m,n))$ for Newton–Schulz approximation. RMNP stems from the K-FAC form $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$. By discarding off-diagonal blocks, the Hessian becomes $H_{\text{RMNP}}=(\operatorname{diag}(V_tV_t^\top))^{1/2}\otimes I_n$. Its inverse preconditioning effect on momentum $V_t$ is exactly:

$$\big[D_t\big]_{i,:}=\frac{V_{t,i:}}{\sqrt{(V_tV_t^\top)_{ii}}}=\frac{V_{t,i:}}{\|V_{t,i:}\|_2},$$

which is standard row $\ell_2$ normalization. This implementation requires only three operations: row-wise sum of squares, square root, and division. With no matrix multiplications, complexity drops from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$, while maintaining row-wise (rather than element-wise) matrix-level adaptivity. While similar in form to row-normalized optimizers under the LMO framework (SRON, SCALE, SWAN, Mano, MOGA), RMNP is derived from Hessian structures rather than worst-case norms.

**2. Justification via Hessian Row-Block Dominance: Empirical Verification**

To justify discarding off-diagonal blocks, the authors define a row-wise diagonal dominance ratio $r_i\triangleq(V_tV_t^\top)_{ii}/(\frac{1}{m-1}\sum_{j\ne i}|(V_tV_t^\top)_{ij}|)$ for the Gram matrix. Tracking $r_{\text{avg}}, r_{\min}, r_{\max}$ across training for GPT-2 and LLaMA models shows these metrics stabilize at $>1$ after warmup. Furthermore, diagonal dominance becomes more pronounced as model size increases. This provides an explanation that traditional LMO analysis lacks: Transformer Hessians are inherently row-block diagonal dominant, so Muon's computation on "cross-row corrections" is largely redundant.

**3. Geometric Matching Proof for Non-convex Convergence**

The authors prove RMNP's convergence using mixed norms $\|W\|_{1,2}=\sum_i\|W_{i,:}\|_2$ and $\|W\|_{\infty,2}=\max_i\|W_{i,:}\|_2$. Theorem 5.5 establishes $\mathcal{O}(m^2 L_F\sigma^2\Delta\epsilon^{-4})$ convergence under Frobenius-Lipschitz conditions. Crucially, Theorem 5.9 provides a dimension dependency of $\mathcal{O}(mL_{\infty,2}\sigma^2\Delta\epsilon^{-4})$ under $L_{\infty,2}$-smoothness, matching Muon's optimal complexity under nuclear norm smoothness. This alignment between the algorithm's geometry (row normalization) and the smoothness condition explains how RMNP maintains accuracy while reducing cost.

### Loss & Training
Standard LLM pre-training CE loss. Optimizer settings: cosine annealing schedule with 10% warmup; AdamW portion uses $\beta=(0.9, 0.95)$ and weight decay of 0.1; $\text{lr}_{\text{Matrix}}$ is tuned separately. RMNP is applied only to matrix parameters; embeddings, lm-head, biases, and layer-norms use AdamW.

## Key Experimental Results

### Main Results

| Model | Data | Muon ppl | RMNP ppl | RMNP vs AdamW |
|------|------|----------|----------|-----------------|
| GPT-2 Small (125M) | OpenWebText 5B tok | -- | $\Delta$=-0.04 | -1.37 |
| GPT-2 Medium (355M) | OpenWebText 10B tok | -- | -0.07 | -1.49 |
| GPT-2 Large (770M) | OpenWebText 20B tok | -- | -0.24 | -0.84 |
| LLaMA-60M | C4 1B tok | -- | -0.63 | -4.33 |
| LLaMA-130M | C4 2B tok | -- | -0.28 | -1.10 |
| LLaMA-350M | C4 6B tok | -- | -0.02 | -- |

**Preconditioning wall-clock time (100 steps, single RTX Pro 6000, batch 16)**

| Model Scale | Muon (s) | RMNP (s) | Speedup |
|----------|----------|----------|------|
| 60M | 1.480 | 0.115 | 12.9× |
| 125M | 2.975 | 0.201 | 14.8× |
| 355M | 7.380 | 0.401 | 18.4× |
| 770M | 27.070 | 0.611 | 44.3× |
| 1.3B | 30.570 | 0.783 | 39.0× |
| 1.5B | 36.650 | 0.855 | 42.9× |

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Full RMNP (Row $\ell_2$) | ppl comparable to or lower than Muon | Main Result |
| Diagonal dominance $r_i$ | $r_{\min}>1$ throughout training | Row-block dominance holds |
| Model Scaling (60M→1.5B) | $r_{\text{avg}}, r_{\max}$ continue to rise | Larger models are more diagonal-dominant |
| 2× Training budget | Advantage maintained | RMNP is not just faster in early stages |
| Applied to LM-head / Embedding | See D.4 | Potential for further efficiency |

### Key Findings
- The complexity gap widens with model size: speedup grows from 12.9× at 60M to 42.9× at 1.5B. Newton-Schulz becomes an end-to-end bottleneck for models $\ge 1$B.
- Perplexity (ppl) does not degrade and is even slightly better than Muon at most scales, suggesting Newton-Schulz's "cross-row" corrections might be harmful overfitting for Transformers.
- Theoretical theorems show RMNP achieves $\mathcal{O}(m)$ dimension complexity under $\|\cdot\|_{\infty,2}$ smoothness, dual to Muon's nuclear norm analysis.

## Highlights & Insights
- Optimizer design is guided by Hessian structure rather than worst-case norms, providing a more grounded explanation for row-norm effectiveness in neural networks.
- Row $\ell_2$ normalization can replace Newton-Schulz with just two lines of code, offering a zero-cost drop-in replacement.
- The use of $r_{\min}, r_{\text{avg}}, r_{\max}$ metrics serves as a diagnostic tool for determine when to apply row-norm optimizers to other architectures.

## Limitations & Future Work
- Primarily validated on GPT-2 and small LLaMA (up to 1.5B); performance on 70B+ scales and architectures like MoE or Mamba remains to be seen.
- Row-block dominance is a "Transformer phenomenon"; its applicability to CNNs or GNNs is unclear.
- Focus is restricted to pre-training; post-training stages (SFT/RLHF) are not covered.
- Optimal normalization axis for non-square matrices like embeddings or LM-heads lacks a unified recommendation.

## Related Work & Insights
- **vs Muon**: Shares matrix-level adaptivity but explicitly exploits Transformer Hessian structures to reduce complexity from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$.
- **vs Shampoo / K-FAC**: Both use Kronecker block-diagonal approximations but require explicit construction and inversion of $L, R$. RMNP uses implicit momentum statistics.
- **vs SCALE / SWAN / Mano / MOGA**: These use row/column normalization based on LMO/steepest descent worst-case perspectives; RMNP derives this from Hessian structure and provides matching non-convex convergence proofs.
- **Insight**: For complex but redundant optimizers, empirical measurement of Hessian/Gram density can help find cheap, equivalent substitutes.

## Rating
- Novelty: ⭐⭐⭐⭐ Links row normalization to Transformer Hessian structure with theoretical matching.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple scales, wall-clock speedup, and dominance metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear diagrams and motivation; theoretical sections are dense but rigorous.
- Value: ⭐⭐⭐⭐⭐ High engineering value; direct drop-in replacement saving significant preconditioning time.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[ACL 2025\] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)

</div>

<!-- RELATED:END -->
