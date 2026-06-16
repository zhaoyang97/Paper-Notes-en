---
title: >-
  [Paper Note] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization
description: >-
  [ICML 2026][Optimization & Theory][Preconditioning] Based on the "row-block diagonal dominance" structure of Transformer layer-wise Hessians, this paper replaces the expensive Newton-Schulz orthogonalization in the Muon optimizer with a single row-level $\ell_2$ normalization. This reduces the per-step preconditioning complexity from $\mathcal{O}(mn\min(m,n))$ to $\math
tags:
  - ICML 2026
  - Optimization & Theory
  - Preconditioning
  - Muon
  - Newton-Schulz
  - Transformer Hessian
date: 2026-05-08
content_hash: 8245a4b65499dec0
---
# RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization

**Conference**: ICML 2026  
**arXiv**: [2603.20527](https://arxiv.org/abs/2603.20527)  
**Code**: The paper states “Our code is available at this link”  
**Area**: Optimization Algorithms / LLM Pre-training  
**Keywords**: Preconditioning, Muon, Newton-Schulz, Row Normalization, Transformer Hessian

## TL;DR
Based on the "row-block diagonal dominance" structure of Transformer layer-wise Hessians, this paper replaces the expensive Newton-Schulz orthogonalization in the Muon optimizer with a single row-level $\ell_2$ normalization. This reduces the per-step preconditioning complexity from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$, achieving a 13–44× wall-clock speedup in GPT-2 / LLaMA pre-training while maintaining or slightly improving perplexity.

## Background & Motivation
**Background**: Diagonal preconditioners like Adam/AdamW are computationally cheap but ignore correlations between parameters. Methods like K-FAC and Shampoo use Kronecker decomposition to capture matrix-level curvature. Recently, Muon has emerged as a strong competitor to AdamW for large model pre-training by using Newton-Schulz iterations $D_t \approx (V_tV_t^\top)^{-1/2}V_t$ to implicitly implement $H^{-1}$ without explicit inversion.

**Limitations of Prior Work**: Muon requires 5 matrix multiplications for Newton-Schulz polynomial approximation at each step, resulting in a complexity of $\mathcal{O}(mn\min(m,n))$. For wide matrices (where $m,n$ are both large), this overhead becomes a training bottleneck—preconditioning for GPT-2 1.5B alone takes 36.65 seconds every 100 steps.

**Key Challenge**: Muon is designed for full spectral renormalization of $V_tV_t^\top$. However, recent studies (Zhang et al., Dong et al.) found that Transformer layer-wise Hessians are actually row-block diagonal dominant, meaning only interactions within diagonal blocks (same row) are significant, while inter-row interactions are negligible. This implies Muon spends significant computation fitting a structure that is "nearly diagonal."

**Goal**: To construct an equivalent approximation with the same complexity as Muon but retaining only row-level diagonal blocks, thereby reducing complexity to linear scale without sacrificing optimization quality.

**Key Insight**: Starting from the K-FAC form $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$, the authors assume only the diagonal elements $\operatorname{diag}(V_tV_t^\top)$ need to be preserved. Empirical measurements during Transformer training show that the "diagonal-to-off-diagonal magnitude ratios" $r_{\min},r_{\text{avg}},r_{\max}$ of the Gram matrix $V_tV_t^\top$ remain consistently $>1$ and increase with model size, validating this assumption.

**Core Idea**: Replace the Newton-Schulz operation $(V_tV_t^\top)^{-1/2}V_t$ with a simple row vector division by the row $\ell_2$ norm. This is equivalent to using $(\operatorname{diag}(V_tV_t^\top))^{-1/2}\otimes I_n$ as a preconditioner, which corresponds exactly to a row-block diagonal approximation of the Hessian.

## Method

### Overall Architecture
The algorithmic framework of RMNP is nearly identical to Muon: for each step, (i) compute mini-batch gradients $G_t=\nabla f(W_t;\xi^t)$; (ii) maintain first-order momentum $V_t=\beta V_{t-1}+(1-\beta)G_t$; (iii) precondition to obtain the descent direction $D_t$; (iv) update $W_{t+1}=W_t-\eta_t D_t$. The only difference is step (iii): Muon uses 5 Newton-Schulz iterations $D_t=\operatorname{NS}_5(V_t)\approx(V_tV_t^\top)^{-1/2}V_t$, whereas RMNP uses $D_t=\operatorname{RN}(V_t)=(\operatorname{diag}(V_tV_t^\top))^{-1/2}V_t$, which simply normalizes each row $V_{t,i:}$ of the momentum matrix as $V_{t,i:}/\|V_{t,i:}\|_2$. RMNP follows Muon’s hybrid strategy—applying RMNP to matrix parameters and AdamW to non-matrix parameters (embeddings/biases/norms) with separate learning rates $\text{lr}_{\text{AdamW}}$ and $\text{lr}_{\text{Matrix}}$.

### Key Designs

**1. Row-level $\ell_2$ Normalization Preconditioner: Implicit Scaling via Hessian Diagonal Blocks**

The bottleneck of Muon lies in its design for full spectral renormalization of $V_tV_t^\top$, requiring $\mathcal{O}(mn\min(m,n))$ for polynomial approximation. RMNP originates from Muon's K-FAC form $H_{\text{MUON}}=(V_tV_t^\top)^{1/2}\otimes I_n$. By retaining only diagonal blocks, one obtains $H_{\text{RMNP}}=(\operatorname{diag}(V_tV_t^\top))^{1/2}\otimes I_n$. Its inverse preconditioning effect on momentum $V_t$ is:

$$\big[D_t\big]_{i,:}=\frac{V_{t,i:}}{\sqrt{(V_tV_t^\top)_{ii}}}=\frac{V_{t,i:}}{\|V_{t,i:}\|_2},$$

which is standard row $\ell_2$ normalization. This implementation requires only three operations: row-wise sum of squares, square root, and division. It contains no matrix multiplications, reducing complexity from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$ while maintaining row-wise (rather than element-wise) matrix-level adaptivity.

**2. Validation of Hessian Row-Block Dominance: Empirical Evidence**

To justify discarding off-diagonal blocks, the authors define a row-wise diagonal dominance ratio $r_i\triangleq(V_tV_t^\top)_{ii}/(\frac{1}{m-1}\sum_{j\ne i}|(V_tV_t^\top)_{ij}|)$ for the Gram matrix and track $r_{\text{avg}},r_{\min},r_{\max}$ throughout training. Evaluations on GPT-2 and LLaMA variants show these metrics stabilize at $>1$ after warm-up, with dominance becoming more pronounced as model size increases ($r_{\text{avg}}\approx4.9$ for GPT-2 Small). This demonstrates that Transformer layer-wise Hessians are naturally row-block diagonal dominant.

**3. Geometric Matching in Non-convex Convergence Proofs**

The authors prove that RMNP achieves competitive convergence rates. Introducing mixed norms $\|W\|_{1,2}=\sum_i\|W_{i,:}\|_2$ and $\|W\|_{\infty,2}=\max_i\|W_{i,:}\|_2$, Theorem 5.9 establishes a $\mathcal{O}(mL_{\infty,2}\sigma^2\Delta\epsilon^{-4})$ dependence under $L_{\infty,2}$-smoothness. This $\mathcal{O}(m)$ dimension dependence matches the optimal complexity of Muon under nuclear norm smoothness and reaches the minimax lower bound for non-convex stochastic optimization. The choice of $\|\cdot\|_{\infty,2}$ smoothness is crucial as it aligns with the geometry of row normalization.

### Loss & Training
Standard CE loss for LLM pre-training. Optimizer: cosine annealing schedule with 10% warmup; AdamW components use $\beta=(0.9, 0.95)$ and weight decay 0.1. A separate $\text{lr}_{\text{Matrix}}$ is searched for matrix parameters. RMNP is applied only to matrix parameters, while embeddings, lm-heads, biases, and layer-norms use AdamW.

## Key Experimental Results

### Main Results

| Model | Data | Muon ppl | RMNP ppl | Gain vs AdamW |
|------|------|----------|----------|-----------------|
| GPT-2 Small (125M) | OpenWebText 5B tok | -- | $\Delta$=-0.04 | -1.37 |
| GPT-2 Medium (355M) | OpenWebText 10B tok | -- | -0.07 | -1.49 |
| GPT-2 Large (770M) | OpenWebText 20B tok | -- | -0.24 | -0.84 |
| LLaMA-60M | C4 1B tok | -- | -0.63 | -4.33 |
| LLaMA-130M | C4 2B tok | -- | -0.28 | -1.10 |
| LLaMA-350M | C4 6B tok | -- | -0.02 | -- |

**Preconditioning Wall-clock Time (100 steps, single RTX Pro 6000, batch 16)**

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
| Full RMNP (Row $\ell_2$) | PPL comparable/slightly lower than Muon | Main Result |
| Diagonal Dominance $r_i$ | $r_{\min}>1$ throughout training | Row-block diagonal assumption holds |
| Model Scaling (60M→1.5B) | $r_{\text{avg}}, r_{\max}$ increase | Larger models are more diagonal; RMNP becomes more reasonable |
| 2× Training Budget | Advantage maintained | RMNP is not just faster in early stages |
| Applied to LM-head / Embedding | See D.4 | Potential for further efficiency |

### Key Findings
- The complexity gap widens with model size: speedup increases from 12.9× at 60M to ~43× at 1.5B. For models $\ge 1$B, Newton-Schulz becomes a genuine bottleneck for end-to-end training.
- Perplexity does not degrade and is slightly better than Muon at most scales, suggesting that Newton-Schulz's "inter-row" corrections may be ineffective or lead to overfitting in Transformers.
- Convergence theorems prove RMNP achieves $\mathcal{O}(m)$ dimension complexity under geometric alignment, dual to Muon’s nuclear norm analysis.

## Highlights & Insights
- Optimizer design is guided by actual Hessian structure rather than worst-case norms, providing a key evidentiary upgrade over prior row-normalization work (SCALE/SWAN/Mano/MOGA).
- Row $\ell_2$ normalization can replace dozens of lines of Newton-Schulz code with just two lines, offering near-zero cost for drop-in use.
- Unified convergence analysis under mixed norms provides a template for determining which norms are suitable for matrix-level optimizers.
- The $r_{\min},r_{\text{avg}},r_{\max}$ metrics serve as diagnostic tools to determine when row-norm optimizers are applicable to other architectures.

## Limitations & Future Work
- Experiments primarily focus on GPT-2 and small LLaMA (up to 1.5B); validation on 70B+ models is pending. Whether geometric assumptions hold for MoE or Mamba architectures is unknown.
- Row-block diagonal dominance is a "Transformer phenomenon"; it remains to be seen if RMNP is a drop-in solution for CNNs/GNNs.
- Experiments cover pre-training only, excluding post-training stages like SFT/RLHF.
- Optimal normalization axis selection for non-square matrices like embeddings/LM-heads remains an open question.

## Related Work & Insights
- **vs Muon**: Shares the matrix-level adaptivity concept, but RMNP explicitly exploits Transformer Hessian structures to cut complexity from $\mathcal{O}(mn\min(m,n))$ to $\mathcal{O}(mn)$.
- **vs Shampoo / K-FAC**: Both use diagonal block approximations of Kronecker decompositions but require explicit construction and inversion of $L,R$. RMNP bypasses explicit matrices via implicit momentum statistics.
- **vs SCALE / SWAN / Mano / MOGA**: These also use row/column normalization but are derived from the worst-case perspective of LMO/steepest descent. RMNP derives this from Hessian structure and provides the first non-convex convergence proof comparable to Muon.

## Rating
- Novelty: ⭐⭐⭐⭐ Links row-normalization to Transformer Hessian structure with theoretical parity to Muon.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple scales, wall-clock timing, and diagonal dominance metrics.
- Writing Quality: ⭐⭐⭐⭐ Clear diagrams and motivation; theoretical sections are dense but rigorous.
- Value: ⭐⭐⭐⭐⭐ High engineering value; direct drop-in for pre-training with massive efficiency gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Provable Suboptimality of Momentum SGD in Nonstationary Stochastic Optimization](on_the_provable_suboptimality_of_momentum_sgd_in_nonstationary_stochastic_optimi.md)
- [\[ICML 2026\] Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural Dual Warm-Starts](learning-augmented_scalable_linear_assignment_problem_optimization_via_neural_du.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](../../AAAI2026/optimization/ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[CVPR 2026\] FedAdamom: Adaptive Momentum for Improved Generalization in Federated Optimization](../../CVPR2026/optimization/fedadamom_adaptive_momentum_for_improved_generalization_in_federated_optimizatio.md)
- [\[ACL 2025\] ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting](../../ACL2025/optimization/scalebio_bilevel_data_reweighting.md)

</div>

<!-- RELATED:END -->
